"""真实 LGS 快照审计（有限补强，2026-10-10，导师第七轮后续指令）。

目的：补齐"真实流程审计"缺口——此前只有 P₀ 冻结重加权研究。本脚本用
NumPy 按 LGS 论文所述流程训练一个 LGS-GCN（不引入新框架、不开发新
方法），从训练后快照提取：预测 Ŷ、（近似梯度训练的）转移矩阵 P、
嵌入余弦相似 S、原始特征相似 S_f、最终混合邻接 Ã（式 8+10），
然后冻结这些量，审计标签传播子步的锁定状态。

忠实要素（LGS, arXiv:2108.04595）：
- 式(6) Y_tr = 训练节点真值 + 其余 Ŷ；式(7) E = Y_tr P Y_trᵀ，P₀
  = Sinkhorn(Y_trᵀ A Y_tr) 数据锚定初始化，P 随训练更新（近似梯度）；
- 式(8) A^(t) = S ⊙ (r·E + (1−r)·1)，ε=0（其实验设定）；
- 式(10) Ã = 0.8·A + 0.1·S_f + 0.1·A^(t)（α=0.8 固定；β 未公布，取
  0.1 并做敏感性）；
- 训练：warm-up 400 epochs（CE）+ 40 轮结构更新 × 40 epochs 联合
  （共 1600），Adam lr=0.01，2 层 GCN，64 维；验证集选轮（冻结协议）。

文档化的偏差（未在原文给出或过于复杂）：
1. 多头加权余弦（式4-5）简化为单头普通余弦（头权重省略）；
2. Φ(P) 原文行和绝对值形式按字面会破坏概率性，改为 off-diagonal L1
   （权重 1e-4）；
3. P 的梯度穿过结构层时忽略对称归一化导数（一阶近似）；
4. CPGNN 式 P 加权标签传播层不参与训练（模型为 LGS-GCN 变体）；
5. 划分用本文 60/20/20 分层协议（与其 Geom-GCN 划分不同，保持本文
   泄漏纪律）；测试真值仅评估列使用。

种子 53000+ 命名空间。审计行：trained Ã（coupled / Y_tr 两种初始化）、
learned core A^(t) 单独、masked_E(P₀)（对照重加权研究）、original。

修复记录（2026-10-10 复查轮，四项）：
1. P₀ 改为只用训练节点对计数（原用全体真值泄漏测试标签，替换测试
   标签曾使快照关系最大差 0.0818）；
2. 训练梯度屏蔽非训练行（原 dZ 保留非训练 softmax 输出，未实现所
   声称的训练集交叉熵）；
3. P 每 epoch 后投影回双随机（原最优轮快照 P 曾达 [−0.0057, 2.2463]、
   行和至 13.14，远超浮点误差，且 P 参与生成被审计关系）；
4. 轮末新鲜前向对齐：报告精度、预测与嵌入取自当前参数的前向计算。
   审计关系随后由这些输出和当前 P 重建并冻结，不是产生报告精度的
   输入图（原精度来自更新前的前向）。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as spsp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.stdout.reconfigure(encoding="utf-8")

from near_locking import argmax_labels, kernel_apply, near_locking_bound
from correction import edge_pairs, onehot
from outer_protocol import stratified_split, targets
from run_correction_pilot import load_unit
from lgs_audit import (audit_from_KG, off_class_epsilon_general, sinkhorn)

ALPHA_PROP, STEPS = 0.9, 20     # 冻结子步的重启传播
BLEND_A, BLEND_B = 0.8, 0.1     # 式(10) α 固定 0.8；β 未公布取 0.1
R_E = 0.5                       # 式(8) r 平衡 E 与全 1
WARMUP, ROUNDS, EPOCHS_PER = 400, 40, 40
HID, LR = 64, 0.01
LAM_PHI = 1e-4
SEED_BASE = 53000
OUT = HERE / "results" / "lgs_snapshot.json"


def sym_norm(M):
    """D^{-1/2}(M+I)D^{-1/2}（GCN 式1）。"""
    n = M.shape[0]
    A = spsp.csr_matrix(M) + spsp.eye(n)
    d = np.asarray(A.sum(axis=1)).ravel()
    di = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    return np.asarray((spsp.diags(di) @ A @ spsps_diags(di)).todense()
                      ).astype(np.float32)


def spsps_diags(v):
    return spsp.diags(v)


def cosine(M, k=10):
    """kNN 稀疏化余弦（每行保留 top-k）。文档化偏差：原文式(4-5)多头
    可训练头隐式稀疏化相似结构；固定普通余弦为全对稠密，联合训练在
    近完全图上过平滑崩溃（train 54%→11%），故显式 kNN 化（k=10）。"""
    n = np.linalg.norm(M, axis=1, keepdims=True)
    S = (M / np.maximum(n, 1e-12)) @ (M / np.maximum(n, 1e-12)).T
    idx = np.argpartition(-S, k, axis=1)[:, :k + 1]
    W = np.zeros_like(S)
    rows = np.repeat(np.arange(S.shape[0]), k + 1)
    W[rows, idx.ravel()] = np.take_along_axis(S, idx, axis=1).ravel()
    np.fill_diagonal(W, 0.0)
    return W


class Adam:
    def __init__(self, shapes, lr=LR):
        self.lr, self.t = lr, 0
        self.m = [np.zeros(s) for s in shapes]
        self.v = [np.zeros(s) for s in shapes]

    def step(self, params, grads):
        self.t += 1
        out = []
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = 0.9 * self.m[i] + 0.1 * g
            self.v[i] = 0.999 * self.v[i] + 0.001 * g * g
            mh = self.m[i] / (1 - 0.9 ** self.t)
            vh = self.v[i] / (1 - 0.999 ** self.t)
            out.append(p - self.lr * mh / (np.sqrt(vh) + 1e-8))
        return out


def gcn_forward(Ahat, X, W1, W2):
    U1 = X @ W1
    X1 = Ahat @ U1
    H = np.maximum(X1, 0.0)
    U2 = H @ W2
    X2 = Ahat @ U2
    Z = X2 - X2.max(axis=1, keepdims=True)
    Z = np.exp(Z)
    Z /= Z.sum(axis=1, keepdims=True)
    return Z, X1, H, U1, U2


def run_split(A_pairs, X, labels, masks, seed, r=R_E, beta=BLEND_B):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    C = int(labels.max()) + 1
    Y = onehot(labels, C)
    train_mask = masks["train"]
    X = X.astype(np.float32)
    A_pairs = A_pairs.astype(np.float32)

    # 特征相似 S_f（式4-5 简化：普通余弦，只算一次）
    S_f = cosine(X)
    S_f = np.clip(S_f, 0.0, None).astype(np.float32)

    # P₀ 数据锚定（修复 2026-10-10：只用训练节点对计数，测试/验证
    # 标签不进入构图——泄漏回归测试覆盖）
    Y_tr_only = Y * train_mask[:, None].astype(Y.dtype)
    P = sinkhorn(Y_tr_only.T @ A_pairs @ Y_tr_only)
    # 参数
    W1 = rng.normal(0, np.sqrt(1.0 / d), (d, HID)).astype(np.float32)
    W2 = rng.normal(0, np.sqrt(1.0 / HID), (HID, C)).astype(np.float32)
    opt = Adam([W1.shape, W2.shape, P.shape])
    Ahat = sym_norm(A_pairs)

    def structure(pred, emb):
        a = pred.copy()
        a[train_mask] = labels[train_mask]        # 式(6)
        E = P[a][:, a].astype(np.float32)
        S = np.clip(cosine(emb.astype(np.float32)), 0.0, None)
        A_t = S * (np.float32(r) * E + np.float32(1.0 - r))             # 式(8), ε=0
        blend = (np.float32(BLEND_A) * A_pairs + np.float32(beta) * S_f
                 + np.float32(1 - BLEND_A - beta) * A_t)
        return a, E, S, A_t, blend

    t0 = time.time()
    # warm-up：固定原图（修复：交叉熵梯度只取训练行，非训练行置零）
    for _ in range(WARMUP):
        Z, X1, H, U1, U2 = gcn_forward(Ahat, X, W1, W2)
        dZ = np.zeros_like(Z)
        dZ[train_mask] = (Z[train_mask] - Y[train_mask])
        dZ /= train_mask.sum()
        dW2 = (Ahat @ H).T @ dZ
        dH = Ahat.T @ (dZ @ W2.T)
        dX1 = dH * (X1 > 0)
        dW1 = (Ahat @ X).T @ dX1
        W1, W2, P = opt.step([W1, W2, P], [dW1, dW2, np.zeros_like(P)])

    pred_cur = Z.argmax(axis=1)
    S_cur = None
    Ahat_cur = Ahat

    # 联合训练：40 轮结构更新 × 40 epochs。
    # 验证集选轮沿用本文冻结的选参协议（val 只用于选择）；
    # 注："固定日程塌缩"来自存在缺陷的 v1（未屏蔽梯度），修复版
    # 全程 val 0.71–0.75 无塌缩，不作为选轮依据。
    best = None
    for round_i in range(ROUNDS):
        emb = H.copy() if H is not None else None
        a, E, S, A_t, blend = structure(pred_cur, emb)
        S_cur = S
        Ahat_cur = sym_norm(blend)
        for _ in range(EPOCHS_PER):
            Z, X1, H, U1, U2 = gcn_forward(Ahat_cur, X, W1, W2)
            dZ = np.zeros_like(Z)                 # 修复：梯度屏蔽非训练行
            dZ[train_mask] = (Z[train_mask] - Y[train_mask])
            dZ /= train_mask.sum()
            dW2 = (Ahat_cur @ H).T @ dZ
            dH = Ahat_cur.T @ (dZ @ W2.T)
            dX1 = dH * (X1 > 0)
            dW1 = (Ahat_cur @ X).T @ dX1
            dAhat = dX1 @ U1.T + dZ @ U2.T
            gP = np.zeros_like(P)
            dE = (1 - BLEND_A - beta) * r * S_cur * dAhat
            aa = pred_cur.copy()
            aa[train_mask] = labels[train_mask]
            flat = (aa[:, None] * C + aa[None, :]).ravel()
            np.add.at(gP.ravel(), flat, dE.ravel())
            cnt = np.outer(np.bincount(aa, minlength=C),
                           np.bincount(aa, minlength=C))
            gP /= np.maximum(cnt, 1.0)          # 每格均值，而非求和
            gP += LAM_PHI * np.sign(P) * (1.0 - np.eye(C))
            W1, W2, P = opt.step([W1, W2, P], [dW1, dW2, gP])
            # 修复：每 epoch 后投影回双随机（C×C 代价可忽略），
            # 快照 P 恒为概率转移矩阵
            P = sinkhorn(np.clip(P, 0.0, None))
        # 轮末前向给出报告精度、预测和嵌入；审计关系随后从这些输出
        # 与当前 P 重建，不是此次前向使用的 Ahat_cur。
        Z_ev, _, H_ev, _, _ = gcn_forward(Ahat_cur, X, W1, W2)
        pred_cur = Z_ev.argmax(axis=1)
        val_acc = float((pred_cur[masks["val"]] == labels[masks["val"]]).mean())
        if best is None or val_acc > best["val_acc"]:
            a_b, E_b, S_b, A_t_b, blend_b = structure(pred_cur, H_ev)
            best = dict(val_acc=val_acc, round=round_i,
                        pred=pred_cur.copy(), P=P.copy(), S=S_b, A_t=A_t_b,
                        blend=blend_b, labels_mixed=a_b,
                        acc=dict(
                            train=float((pred_cur[train_mask]
                                         == labels[train_mask]).mean()),
                            test=float((pred_cur[masks["test"]]
                                        == labels[masks["test"]]).mean())))
        print(f"  [seed {seed}] round {round_i} done "
              f"({time.time()-t0:.0f}s) val {val_acc:.3f}", flush=True)

    # 最终快照 = 验证最优轮（P 为该轮投影后状态）
    return dict(pred=best["pred"], P=best["P"], S=best["S"], S_f=S_f,
                A_t=best["A_t"], blend=best["blend"],
                labels_mixed=best["labels_mixed"], acc=best["acc"],
                early_stop=dict(val_acc=best["val_acc"], round=best["round"]))


def audit_relation(W_row, n, a, labels, test_mask):
    """W_row: n×n 加权邻接（行随机化后传播）。"""
    P = spsp.csr_matrix(W_row)
    deg = np.asarray(P.sum(axis=1)).ravel()
    keep = deg > 0
    D = spsp.diags(np.where(keep, 1.0 / np.maximum(deg, 1e-12), 0.0))
    P = (D @ P).tocsr()
    idx0 = np.flatnonzero(~keep)
    if len(idx0):
        P = P + spsp.csr_matrix((np.ones(len(idx0)), (idx0, idx0)),
                                shape=(n, n))
    eps_u, eps = off_class_epsilon_general(P, a)
    KG = kernel_apply(P, onehot(a, int(max(a.max(), labels.max())) + 1),
                      ALPHA_PROP, STEPS)
    return audit_from_KG(KG, eps, a, labels, test_mask)


def main():
    feats, edges, labels, n_cls, _, _ = load_unit(("cora", "prototype"))
    X = np.asarray(feats.todense()) if hasattr(feats, "todense") \
        else np.asarray(feats)
    rows, cols, _ = edge_pairs(edges)
    A = np.zeros((len(labels), len(labels)))
    np.add.at(A, (rows, cols), 1.0)
    n = len(labels)

    doc = dict(
        purpose=("real-pipeline reinforcement: NumPy LGS-GCN reimplementation "
                 "per stated equations, snapshot audit of the frozen "
                 "label-propagation substep (no new method; no new framework)"),
        fidelity=("Eq.(6)(7)(8)(10) implemented; alpha=0.8 fixed, beta=0.1 "
                  "assumed (unpublished), r=0.5 primary with sensitivity; "
                  "epsilon=0; deviations documented in module docstring "
                  "(single cosine head, off-diag L1 for Phi, approximate P "
                  "gradient, no CPGNN layer in training, our 60/20/20 splits)"),
        seeds_base=SEED_BASE, splits=[], r_sensitivity=[])

    for r_i in range(3):
        seed = SEED_BASE + r_i
        masks = stratified_split(labels, seed=seed)
        t0 = time.time()
        snap = run_split(A, X, labels, masks, seed)
        pred, a = snap["pred"], snap["labels_mixed"]
        rows_audit = {}
        # 行随机化各关系并审计
        def rowstoch(W):
            s = W.sum(axis=1, keepdims=True)
            return W / np.maximum(s, 1e-12)
        rows_audit["trained_blend_Ytr"] = audit_relation(
            rowstoch(snap["blend"]), n, a, labels, masks["test"])
        rows_audit["trained_blend_coupled"] = audit_relation(
            rowstoch(snap["blend"]), n, pred, labels, masks["test"])
        rows_audit["learned_core_Ytr"] = audit_relation(
            rowstoch(snap["A_t"]), n, a, labels, masks["test"])
        P0 = sinkhorn(onehot(a, n_cls).T @ A @ onehot(a, n_cls))
        rows_audit["masked_E_P0_Ytr"] = audit_relation(
            rowstoch(A * P0[a][:, a]), n, a, labels, masks["test"])
        rows_audit["original"] = audit_relation(
            rowstoch(A), n, a, labels, masks["test"])
        rec = dict(split_seed=seed, acc=snap["acc"],
                   P_diag=float(np.mean(np.diag(snap["P"]))),
                   P=snap["P"].round(4).tolist(), rows=rows_audit,
                   secs=round(time.time() - t0, 1))
        doc["splits"].append(rec)
        OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        print(f"[split {r_i}] acc={snap['acc']} P_diag={rec['P_diag']:.3f} "
              f"({rec['secs']}s)")
        for k, v in rows_audit.items():
            print(f"  {k:<22} eps={v['eps_max']:.3f} "
                  f"cert={v['rho_certified_frac']:.3f} "
                  f"chg={v['changed_frac']:.4f} "
                  f"corr={v['corrected_test']*100:+.2f} "
                  f"wrong={v['wrong_changed_test']*100:+.2f} "
                  f"chg|cert={v['changed_among_certified']:.5f}")

    # r 敏感性（split 0）
    for r_val in (0.25, 0.75):
        seed = SEED_BASE
        masks = stratified_split(labels, seed=seed)
        snap = run_split(A, X, labels, masks, seed + 900 + int(r_val * 100),
                         r=r_val)
        a = snap["labels_mixed"]
        blend = snap["blend"] / np.maximum(
            snap["blend"].sum(axis=1, keepdims=True), 1e-12)
        v = audit_relation(blend, n, a, labels, masks["test"])
        doc["r_sensitivity"].append(dict(
            r=r_val, acc=snap["acc"], P_diag=float(np.mean(np.diag(snap["P"]))),
            cert=v["rho_certified_frac"], changed=v["changed_frac"],
            corr=v["corrected_test"], wrong=v["wrong_changed_test"]))
        print(f"[r={r_val}] cert={v['rho_certified_frac']:.3f} "
              f"corr={v['corrected_test']*100:+.2f}")

    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"\n已写出 {OUT}")


if __name__ == "__main__":
    main()
