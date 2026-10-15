"""行随机近锁定理论的核心量（计划 §5，2026-10-09）。

对象：非负行随机矩阵 P（零度节点固定自环），
  Z_{t+1} = alpha P Z_t + (1-alpha) Z_0,  0 <= alpha < 1,
  K_T = (1-alpha) sum_{t=0}^{T-1} alpha^t P^t + alpha^T P^T,  Z_T = K_T Z_0.
K_T 行随机（权重和恰为 1）。对初始 argmax 类 a_u，跨类质量
  M_T(u) = sum_{v: a_v != a_u} K_T[u,v] = 1 - (K_T G)[u, a_u].

理论结论（证明见 notes/upgrade/near_locking_proof.md）：
  eps = max_u sum_{v: a_v != a_u} P[u,v]  ⟹
  M_T(u) <= min(1, eps * alpha * (1 - alpha^T) / (1 - alpha)) =: B(eps, alpha, T)
  （核行递归 M_T <= alpha M_{T-1} + alpha eps (1 - M_{T-1})，M_0 = 0。）
one-hot 初始化：B < 1/2 ⟹ 每节点 argmax 保持（r[u,a_u] = 1 - M_T(u) > 1/2）。
软分数（输入在 [0,1]、统一最小间隔 gamma = min_v margin_v > 0）：
  M_T(u) < gamma / (1 + gamma) ⟹ argmax 保持（逐行差 >= gamma - M_T(1+gamma)）。
eps = 0 恢复硬过滤精确锁定（不是第二套独立贡献）。
对照：对称归一化 S 的有限步总核行和为正时可按行归一化解释最终 argmax，
但 P 的逐步 eps 界不能直接移植（见证明笔记 Remark C）。
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as spsp


def argmax_labels(Z):
    """固定平票协议：numpy argmax（最小下标）。充分条件均用严格不等式。"""
    return np.asarray(Z).argmax(axis=1)


def row_stochastic(n, rows, cols, vals):
    """由加权邻接结构造非负行随机 P；零度行固定为 e_u。

    rows/cols/vals：所有 (u,v) 有序对及权重（超边展开、kNN 等任一来源）。
    """
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    vals = np.asarray(vals, dtype=float)
    A = spsp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    deg = np.asarray(A.sum(axis=1)).ravel()
    empty = deg <= 0
    if empty.any():
        idx = np.flatnonzero(empty)
        A = A + spsp.csr_matrix((np.ones(len(idx)), (idx, idx)), shape=(n, n))
        deg = np.asarray(A.sum(axis=1)).ravel()
    D = spsp.diags(1.0 / deg)
    return (D @ A).tocsr()


def kernel_apply(P, R, alpha, steps):
    """K_T @ R，不显式构造核：Y_0 = R; Y_{t+1} = alpha P Y_t + (1-alpha) R。"""
    Y = np.asarray(R, dtype=float).copy()
    for _ in range(int(steps)):
        Y = alpha * (P @ Y) + (1.0 - alpha) * R
    return Y


def restart_kernel(P, alpha, steps):
    """小矩阵的精确有限步核 K_T（对单位阵做 kernel_apply）。"""
    P = P.toarray() if spsp.issparse(P) else np.asarray(P, dtype=float)
    return kernel_apply(P, np.eye(P.shape[0]), alpha, steps)


def cross_class_mass(K, predicted_class):
    """逐节点跨初始预测类质量 M_T(u) = 1 - (K G)[u, a_u]。"""
    a = np.asarray(predicted_class)
    n, C = K.shape[0], int(a.max()) + 1
    G = np.zeros((n, C))
    G[np.arange(n), a] = 1.0
    mass = K @ G if spsp.issparse(K) else np.asarray(K) @ G
    return 1.0 - mass[np.arange(n), a]


def off_class_epsilon(P, predicted_class):
    """(逐节点 eps(u), 全局 eps = max_u eps(u))。"""
    a = np.asarray(predicted_class)
    n, C = P.shape[0], int(a.max()) + 1
    G = np.zeros((n, C))
    G[np.arange(n), a] = 1.0
    same = P @ G if spsp.issparse(P) else np.asarray(P) @ G
    eps_u = 1.0 - same[np.arange(n), a]
    return eps_u, float(eps_u.max())


def near_locking_bound(eps, alpha, steps):
    """候选上界 B = min(1, eps * alpha * (1 - alpha^T) / (1 - alpha))。"""
    alpha = float(alpha)
    steps = int(steps)
    if alpha <= 0.0:
        return 0.0
    B = float(eps) * alpha * (1.0 - alpha ** steps) / (1.0 - alpha)
    return min(1.0, B)


def no_change_threshold(margin_gamma):
    """软分数不改变充分条件的质量阈值 gamma / (1 + gamma)；one-hot gamma=1。"""
    g = float(margin_gamma)
    return g / (1.0 + g)


def bound_occupancy(P, predicted_class, alpha, steps, margin_gamma=1.0):
    """实证占用统计：界是否非空、逐节点条件满足率（用于原生软图检查）。

    返回 dict：eps、B、经验最大跨类质量、阈值、满足 M_T < 阈值的节点比例、
    以及按 B < 阈值的"可预测锁定"判定（界非空时才有预测力）。
    """
    eps_u, eps = off_class_epsilon(P, predicted_class)
    B = near_locking_bound(eps, alpha, steps)
    n = P.shape[0]
    a = np.asarray(predicted_class)
    G = np.zeros((n, int(a.max()) + 1))
    G[np.arange(n), a] = 1.0
    KG = kernel_apply(P, G, alpha, steps)          # = K_T @ G
    M = 1.0 - KG[np.arange(n), a]                  # 跨类质量 M_T(u)
    thr = no_change_threshold(margin_gamma)
    return dict(eps=eps, bound=B, threshold=thr,
                bound_nonempty=bool(B < thr),
                frac_nodes_below_thr=float((M < thr).mean()),
                max_empirical_mass=float(M.max()),
                predictable_lock=bool(B < thr and (M < thr).all()))
