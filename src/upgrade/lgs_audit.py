"""LGS 启发的冻结重加权研究（改名自"忠实审计"，2026-10-09 导师第四轮）。

勘误声明（相对上一版）：
1. 上一版 "label_P0" 行只在原图共现对上加权，实际计算 A⊙E(P₀)，
   不是 LGS 式(7) 的完整 E=Y_tr P Y_tr^T。本版补 "full_E" 行（未屏蔽、
   全对类转移关系，经类聚合核实现），A⊙E 行保留并如实命名 "masked_E"。
2. 撤回"省略特征通道只会减少逃逸/更保守"论断：特征关系可主要连同类
   节点，行归一化后跨类质量可升可降（二节点反例 0.50→0.076）。本版
   加 "blend_feature" 行实证测量特征代理通道的影响方向。
3. 本研究是 LGS 启发的重加权变体实验，不是 LGS 原文式(8)(10) 的忠实
   复现（式8 需嵌入相似 S 与超参 r/β/ε、式10 需 β、Φ(P) 与 CPGNN
   传播层细节，原文未全部给出），更不能据此断言完整 LGS 行为。
   忠实"真实流程审计"（从真实 LGS 模型快照提取预测/特征相似/转移
   矩阵/邻接，保留式(8)(10)）登记为后续研究（decision_log）。

行（种子 52000+，描述性）：masked_E（A⊙E(P₀)）、full_E（E(P₀) 全对）、
blend（0.8A+0.2·A⊙E(P₀)，β=0 变体）、blend_feature（0.8A+0.1·(A⊙S̃f)+
0.1·A⊙E(P₀)——特征因子只作用于既有共现对，是"既有边特征重加权"，
**不是新增特征通道**；S̃f=原始特征余弦 kNN 代理，文档化）、original。
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

from near_locking import (argmax_labels, kernel_apply, near_locking_bound,
                          row_stochastic)
from correction import edge_pairs, onehot
from outer_protocol import stratified_split, targets, fit_predictor
from run_correction_pilot import load_unit

ALPHA, STEPS = 0.9, 20
BLEND_ALPHA = 0.8
SEED_BASE = 52000
OUT = HERE / "results" / "lgs_audit.json"


def sinkhorn(M, iters=200, eps=1e-9):
    X = np.asarray(M, dtype=float) + eps
    for _ in range(iters):
        X = X / X.sum(axis=1, keepdims=True)
        X = X / X.sum(axis=0, keepdims=True)
    return X


def off_class_epsilon_general(P, a):
    n, C = len(a), int(a.max()) + 1
    G = onehot(a, C)
    same = P @ G
    eps_u = 1.0 - same[np.arange(n), a]
    return eps_u, float(eps_u.max())


def audit_from_KG(KG, eps, a, labels, test_mask):
    """由传播像 KG = K_T @ onehot(a) 与全局 eps 出诊断（one-hot 专用）。"""
    n = len(a)
    B = near_locking_bound(eps, ALPHA, STEPS)
    rho = 1.0 - KG[np.arange(n), a]
    certified = rho < 0.5
    fin = argmax_labels(KG)
    changed = fin != a
    y = labels[test_mask]
    z0t, ft = a[test_mask], fin[test_mask]
    return dict(
        eps_max=float(eps), global_bound=float(B),
        global_certifies=bool(B < 0.5),
        rho_certified_frac=float(certified.mean()),
        rho_p10_p90=[float(np.percentile(rho, 10)),
                     float(np.percentile(rho, 90))],
        changed_frac=float(changed.mean()),
        changed_among_certified=float(changed[certified].mean())
        if certified.any() else 0.0,
        changed_among_uncertified=float(changed[~certified].mean())
        if (~certified).any() else 0.0,
        corrected_test=float(((z0t != y) & (ft == y)).mean()),
        wrong_changed_test=float(((z0t == y) & (ft != y)).mean()),
        init_acc_test=float((z0t == y).mean()),
        final_acc_test=float((ft == y).mean()))


def audit_row(P, a, labels, test_mask):
    eps_u, eps = off_class_epsilon_general(P, a)
    KG = kernel_apply(P, onehot(a, int(max(a.max(), labels.max())) + 1),
                      ALPHA, STEPS)
    return audit_from_KG(KG, eps, a, labels, test_mask)


def full_label_weights(a, P_cls, n_cls):
    """未屏蔽 E=Y_tr P Y_tr^T 的行随机化权重
    W[u,c] = P[a_u,c] / Σ_{c'} n_{c'} P[a_u,c']。"""
    n = len(a)
    counts = np.bincount(a, minlength=n_cls).astype(float)
    row = P_cls[a]                                  # n × C
    return row / (row @ counts)[:, None]


def apply_full_label(W, a, n_cls, alpha, steps):
    """K_T(E_rowstoch) @ onehot(a)：类聚合核（输入只支持类 one-hot）。
    逐步用当前 Y 的逐类求和 CS[c,c'] = Σ_{v:a_v=c} Y[v,c']，
    (P Y)[u,c'] = Σ_c W[u,c]·CS[c,c'] = (W @ CS)[u,c']。"""
    G = onehot(a, n_cls)
    Y = G.copy()
    for _ in range(steps):
        CS = np.zeros((n_cls, Y.shape[1]))
        np.add.at(CS, a, Y)                      # 当前 Y 的类聚合和
        Y = alpha * (W @ CS) + (1 - alpha) * G
    return Y


def feature_knn_weights(feats, k=10):
    """S̃f 代理：原始特征余弦 kNN（k=10，文档化的简化）。None=无特征。
    稀疏实现（sklearn kneighbors_graph），支持 DBLP 规模。"""
    if feats is None:
        return None
    X = feats if spsp.issparse(feats) else np.asarray(
        feats.todense() if hasattr(feats, "todense") else feats)
    from sklearn.neighbors import kneighbors_graph
    G = kneighbors_graph(X, n_neighbors=k, mode="distance",
                         metric="cosine", include_self=False)
    G.data = 1.0 - G.data                            # 距离 → 余弦相似
    return G.toarray() if G.shape[0] <= 6000 else G.tocsr()


def run_unit(unit, ui, n_splits=3):
    tag, fam = unit
    feats, edges, labels, n_cls, predictor, pfn = load_unit(unit)
    n = len(labels)
    rows, cols, _ = edge_pairs(edges)
    ones = np.ones(len(rows))
    recs = []
    for r in range(n_splits):
        seed = SEED_BASE + ui * 100 + r
        masks = stratified_split(labels, seed=seed)
        tr = targets(labels, masks["train"])
        pred = pfn(masks["train"], tr, seed) if pfn is not None \
            else fit_predictor(feats, masks["train"], tr, seed, predictor)
        a = pred.copy()
        a[masks["train"]] = tr[masks["train"]]
        A = np.zeros((n, n))
        np.add.at(A, (rows, cols), 1.0)
        Y = onehot(a, n_cls)
        P0 = sinkhorn(Y.T @ A @ Y)

        w_masked = P0[a[rows], a[cols]]                       # A ⊙ E(P0)
        w_orig = ones
        W_full = full_label_weights(a, P0, n_cls)             # E(P0) 全对
        eps_full = 1.0 - W_full[np.arange(n), a] * np.bincount(
            a, minlength=n_cls)[a]
        Sfeat = feature_knn_weights(feats)

        rels = {}
        rels["masked_E"] = row_stochastic(n, rows, cols, w_masked)
        rels["blend"] = row_stochastic(
            n, rows, cols, (1 - BLEND_ALPHA) * w_masked + BLEND_ALPHA * w_orig)
        if Sfeat is not None:
            w_feat = np.asarray(Sfeat[rows, cols]).ravel()
            rels["blend_feature"] = row_stochastic(
                n, rows, cols, 0.1 * w_masked + 0.1 * w_feat
                + BLEND_ALPHA * w_orig)
        rels["original"] = row_stochastic(n, rows, cols, w_orig)

        t0 = time.time()
        rec = dict(split_seed=seed, predictor=predictor, rows={})
        for name, P in rels.items():
            rec["rows"][name] = audit_row(P, a, labels, masks["test"])
        KG_full = apply_full_label(W_full, a, n_cls, ALPHA, STEPS)
        rec["rows"]["full_E"] = audit_from_KG(KG_full, float(eps_full.max()),
                                              a, labels, masks["test"])
        recs.append(rec)
        print(f"[{tag}/{fam}] split {r} ({time.time()-t0:.0f}s): "
              + " | ".join(
                  f"{k}: cert={v['rho_certified_frac']:.3f} "
                  f"chg={v['changed_frac']:.4f} "
                  f"corr={v['corrected_test']*100:+.1f}"
                  for k, v in sorted(rec["rows"].items())), flush=True)
    return recs


def main():
    units = [("cora", "prototype"), ("cora", "logreg"),
             ("dblp", "prototype"), ("dblp", "logreg")]
    doc = dict(
        study_name=("LGS-inspired frozen reweighting study (renamed from "
                    "'faithful audit' after review round 4; see "
                    "decision_log). NOT a reproduction of LGS Eqs.(8)/(10): "
                    "rows are reweighting variants of the relation "
                    "construction idea; full LGS behavior is NOT claimed"),
        protocol=("Y_tr per LGS Eq.(6) (train nodes anchored to true "
                  "labels, predictions elsewhere); class-transition "
                  "P0 = Sinkhorn(Y_tr^T A Y_tr); rows: masked_E = A-weighted "
                  "E(P0) on original co-occurrence pairs; full_E = "
                  "unmasked row-stochastic E(P0) over all pairs "
                  "(class-aggregated kernel); blend = 0.8*original + "
                  "0.2*A-weighted E(P0) (beta=0 variant, no monotonicity "
                  "claim w.r.t. the feature channel); blend_feature "
                  "reweights EXISTING pairs by a 0.1 raw-feature cosine "
                  "factor (i.e. A-weighted S~f = A odot S~f on original "
                  "co-occurrence pairs only - feature reweighting of "
                  "existing edges, NOT a new feature channel); "
                  "row-stochastic restart kernel alpha=0.9 T=20; one-hot "
                  "coupled-mixed initialization; test labels only in "
                  "evaluation columns; descriptive; seeds 52000+"),
        follow_up=("real-pipeline audit registered as follow-up: extract "
                   "predictions/feature-similarity/transition matrix/"
                   "adjacency from a real LGS model snapshot, keep Eqs.(8) "
                   "and (10) faithfully, audit the frozen substep"),
        seeds_base=SEED_BASE, units={})
    for ui, unit in enumerate(units):
        doc["units"][f"{unit[0]}/{unit[1]}"] = run_unit(unit, ui)
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"\n已写出 {OUT}")
    for uname, recs in doc["units"].items():
        print(f"--- {uname}")
        for k in sorted(recs[0]["rows"]):
            v = np.mean([[r["rows"][k][m] for m in
                          ("eps_max", "rho_certified_frac", "changed_frac",
                           "corrected_test", "wrong_changed_test",
                           "changed_among_certified",
                           "changed_among_uncertified")] for r in recs],
                        axis=0)
            print(f"  {k:<14} eps={v[0]:.3f} cert={v[1]:.3f} "
                  f"chg={v[2]:.4f} corr={v[3]*100:+.2f} wrong={v[4]*100:+.2f}"
                  f" chg|cert={v[5]:.5f} chg|uncert={v[6]:.3f}")


if __name__ == "__main__":
    main()
