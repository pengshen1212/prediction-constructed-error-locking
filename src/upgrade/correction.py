"""最小干预与同预算基线（计划 §7/§8.2，2026-10-09）。

唯一候选干预（受近锁定分析约束的可靠训练残差校正；本体为 C&S-Correct
在 λ 混合行随机图上的移植，不列为算法创新——见 novelty_matrix.md）：
  P_lam = (1-lam) P_hard + lam P_aux        （两者均非负行随机，零度自环）
  R_train[u] = onehot(y_u) - Z_0[u]，仅在训练节点非零
  C = Z_0 + eta * K_T(P_lam) R_train，argmax(C) 直接评估
  P_aux = 原始候选超图（现有可得关系，非测试真值构造）。

无任何未声明后处理（无 clipping/校准/二轮平滑）。网格与 tie 规则冻结于
protocol.json；候选选择只看验证集 accuracy，平票依次取较小 eta、较小 lam。

基线（同一预测器种子、同一训练标签、同预算上限 16 次拟合评估）：
  no_prop   Z_0 argmax
  prop_original / prop_soft   行随机核传播 Z_0（alpha 网格 4）
  cs        官方式 C&S（对称归一化 S、autoscale 残差 + 平滑，
            alpha1×alpha2 16 格，原始超图——候选同图）
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as spsp

from near_locking import (argmax_labels, kernel_apply, row_stochastic)
from outer_protocol import fit_predictor

ALPHA = 0.9
STEPS = 20
LAMBDA_GRID = (0.0, 0.1, 0.3, 1.0)
ETA_GRID = (0.0, 0.25, 0.5, 1.0)
CS_ALPHA_GRID = (0.5, 0.7, 0.9, 0.95)
PROP_ALPHA_GRID = (0.5, 0.7, 0.9, 0.95)
DEFAULT_GRID = [(lam, eta) for eta in ETA_GRID for lam in LAMBDA_GRID]


# ---------------------------------------------------------------- 结构
def edge_pairs(edges):
    """全部有序对（含对角）——与既有单位权重构造一致。"""
    rows, cols, owner = [], [], []
    for i, e in enumerate(edges):
        k = len(e)
        rows.append(np.repeat(e, k))
        cols.append(np.tile(e, k))
        owner.append(np.full(k * k, i))
    return (np.concatenate(rows), np.concatenate(cols),
            np.concatenate(owner))


def hard_weights(pred, edges):
    return np.array([1.0 if len({int(pred[u]) for u in e}) == 1 else 0.0
                     for e in edges])


def soft_weights(pred, edges, C):
    out = np.zeros(len(edges))
    for i, e in enumerate(edges):
        cnt = np.bincount(pred[list(e)], minlength=C)
        out[i] = cnt.max() / len(e)
    return out


def build_P(edges, weights, n):
    rows, cols, owner = edge_pairs(edges)
    return row_stochastic(n, rows, cols, np.asarray(weights)[owner])


def onehot(z, C):
    Z = np.zeros((len(z), C))
    Z[np.arange(len(z)), z] = 1.0
    return Z


def residual_matrix(train_targets, Z0, train_mask, C):
    """R 仅在训练节点非零；train_targets 在 mask 外为 sentinel -1（不读）。"""
    n = Z0.shape[0]
    R = np.zeros((n, C))
    idx = np.flatnonzero(train_mask)
    R[idx] = onehot(train_targets[idx], C) - Z0[idx]
    return R


def _sym_norm_S(edges, weights, n):
    """官方 C&S 传播算子 S = D^{-1/2}(A+I)D^{-1/2}（加权对，加自环）。"""
    rows, cols, owner = edge_pairs(edges)
    vals = np.asarray(weights)[owner]
    keep = vals != 0.0
    A = spsp.csr_matrix((vals[keep], (rows[keep], cols[keep])),
                        shape=(n, n)) + spsp.eye(n, format="csr")
    d = np.asarray(A.sum(axis=1)).ravel()
    di = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    return (spsp.diags(di) @ A @ spsp.diags(di)).tocsr()


def candidate_scores(P_lam, Z0, R, eta, alpha=ALPHA, steps=STEPS):
    """干预打分（修订 1，见 decision_log）：C = K_T(P_lam) (Z0 + eta R)。

    冻结版 C = Z0 + eta K_T R 的核只作用于稀疏残差、从不作用于基础分数，
    近锁定理论自身的纠错路线（传播像中的跨类质量）无法进入候选。
    修订后 eta=0 精确退化为同图传播臂 K_T Z0；eta=1 时训练行被真值锚定
    （标签扩散形式）。仍单核、无 clipping、无二轮平滑。
    """
    return kernel_apply(P_lam, Z0 + eta * R, alpha, steps)


def _val_acc(C_scores, val_targets, val_mask):
    fin = argmax_labels(C_scores)
    return float((fin[val_mask] == val_targets[val_mask]).mean())


def _shared_setup(features, edges, train_targets, masks, seed, predictor,
                  n_classes, predictor_fn=None):
    if predictor_fn is not None:
        pred = predictor_fn(masks["train"], train_targets, seed)
    else:
        pred = fit_predictor(features, masks["train"], train_targets, seed,
                             predictor)
    C = int(n_classes) if n_classes is not None else \
        int(train_targets[masks["train"]].max()) + 1
    Z0 = onehot(pred, C)
    return pred, C, Z0


def _budget(masks):
    return dict(train_label_nodes=int(masks["train"].sum()),
                val_label_nodes=int(masks["val"].sum()))


# ---------------------------------------------------------------- 候选
def fit_and_select(features, edges, train_targets, val_targets, masks, seed,
                   predictor="prototype", grid=None, n_classes=None,
                   alpha=ALPHA, steps=STEPS, predictor_fn=None):
    """唯一最小干预：验证集选 (lam, eta)；平票取较小 eta 再较小 lam。"""
    pred, C, Z0 = _shared_setup(features, edges, train_targets, masks, seed,
                                predictor, n_classes, predictor_fn)
    n = len(pred)
    rows, cols, owner = edge_pairs(edges)
    P_hard = row_stochastic(n, rows, cols, hard_weights(pred, edges)[owner])
    P_aux = row_stochastic(n, rows, cols, np.ones(len(edges))[owner])
    R = residual_matrix(train_targets, Z0, masks["train"], C)

    grid = grid if grid is not None else DEFAULT_GRID
    trace = []
    best = None
    for lam, eta in sorted(grid, key=lambda t: (t[1], t[0])):  # (eta, lam) 升序
        P_lam = (1.0 - lam) * P_hard + lam * P_aux
        C_sc = candidate_scores(P_lam, Z0, R, eta, alpha, steps)
        va = _val_acc(C_sc, val_targets, masks["val"])
        trace.append(dict(lam=lam, eta=eta, val_acc=va))
        if best is None or va > best[0]:                       # 严格更优才替换
            best = (va, lam, eta, C_sc, P_lam)
    va, lam, eta, C_sc, P_lam = best
    return dict(C=C_sc, Z0=Z0, z0=pred, R=R, lam=lam, eta=eta,
                val_acc=va, P_lam=P_lam, trace=trace,
                budget=_budget(masks))


# ---------------------------------------------------------------- 基线
def no_prop_baseline(features, edges, train_targets, val_targets, masks,
                     seed, predictor="prototype", n_classes=None,
                     predictor_fn=None):
    pred, C, Z0 = _shared_setup(features, edges, train_targets, masks, seed,
                                predictor, n_classes, predictor_fn)
    return dict(C=Z0.copy(), Z0=Z0, z0=pred, params=None,
                val_acc=_val_acc(Z0, val_targets, masks["val"]),
                budget=_budget(masks))


def prop_baseline(features, edges, train_targets, val_targets, masks, seed,
                  predictor="prototype", structure="original",
                  n_classes=None, steps=STEPS, predictor_fn=None):
    pred, C, Z0 = _shared_setup(features, edges, train_targets, masks, seed,
                                predictor, n_classes, predictor_fn)
    n = len(pred)
    rows, cols, owner = edge_pairs(edges)
    if structure == "original":
        w = np.ones(len(edges))
    elif structure == "soft":
        w = soft_weights(pred, edges, C)
    else:
        raise ValueError(structure)
    P = row_stochastic(n, rows, cols, w[owner])
    best = None
    trace = []
    for a in sorted(PROP_ALPHA_GRID):          # 平票取较小 alpha
        ZT = kernel_apply(P, Z0, a, steps)
        va = _val_acc(ZT, val_targets, masks["val"])
        trace.append(dict(alpha=a, val_acc=va))
        if best is None or va > best[0]:
            best = (va, a, ZT)
    va, a, ZT = best
    return dict(C=ZT, Z0=Z0, z0=pred, alpha=a, val_acc=va, trace=trace,
                budget=_budget(masks))


def cs_baseline(features, edges, train_targets, val_targets, masks, seed,
                predictor="prototype", n_classes=None, steps=STEPS,
                predictor_fn=None):
    """官方式 Correct & Smooth（对称归一化、autoscale、原始超图）。"""
    pred, C, Z0 = _shared_setup(features, edges, train_targets, masks, seed,
                                predictor, n_classes, predictor_fn)
    n = len(pred)
    S = _sym_norm_S(edges, np.ones(len(edges)), n)
    tr = masks["train"]
    E0 = np.zeros((n, C))
    E0[tr] = onehot(train_targets[tr], C) - Z0[tr]
    sigma = float(np.abs(E0[tr]).sum(axis=1).mean())

    best = None
    trace = []
    for a1 in sorted(CS_ALPHA_GRID):           # 平票取较小 alpha1 再 alpha2
        E = E0.copy()
        for _ in range(steps):
            E = (1.0 - a1) * E0 + a1 * (S @ E)
        Zr = Z0.copy()
        rows_l1 = np.abs(E).sum(axis=1)
        scale = np.zeros(n)
        nzr = rows_l1 > 0
        scale[nzr] = sigma / rows_l1[nzr]
        Zr[~tr] = Z0[~tr] + E[~tr] * scale[~tr, None]
        for a2 in sorted(CS_ALPHA_GRID):
            G0 = Zr.copy()
            G0[tr] = onehot(train_targets[tr], C)
            G = G0.copy()
            for _ in range(steps):
                G = (1.0 - a2) * G0 + a2 * (S @ G)
            va = _val_acc(G, val_targets, masks["val"])
            trace.append(dict(alpha1=a1, alpha2=a2, val_acc=va))
            if best is None or va > best[0]:
                best = (va, a1, a2, G)
    va, a1, a2, G = best
    return dict(C=G, Z0=Z0, z0=pred, alpha1=a1, alpha2=a2, val_acc=va,
                trace=trace, budget=_budget(masks))


# ---------------------------------------------------------------- 评估
def evaluate(C_scores, test_targets, test_mask, z0=None):
    """最终评估：唯一允许接触 test 真值的函数。主分母 = 全部测试节点。"""
    assert (np.asarray(test_targets)[~test_mask] == -1).all()
    fin = argmax_labels(C_scores)
    y = np.asarray(test_targets)[test_mask]
    f = fin[test_mask]
    out = dict(acc=float((f == y).mean()))
    if z0 is not None:
        z = np.asarray(z0)[test_mask]
        out["corrected"] = float(((z != y) & (f == y)).mean())
        out["wrong_changed"] = float(((z == y) & (f != y)).mean())
        err0 = z != y
        ok0 = ~err0
        out["corrected_given_init_wrong"] = (
            float((f == y)[err0].mean()) if err0.any() else None)
        out["wrong_given_init_right"] = (
            float((f != y)[ok0].mean()) if ok0.any() else None)
    return out
