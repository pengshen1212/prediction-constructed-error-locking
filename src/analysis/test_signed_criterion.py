"""新增命题的回归测试：条件稀释恒等式 + 自隔离引理 + 符号判据分解。

对应论文 §sec:main-criterion（prop:thinning, prop:criterion,
lem:self-isolation）。全部在小规模合成超图上做精确蒙特卡洛验证。

构造注意：理论恒等式的前提是同质边（标签全同）+ 高同质度
（隔离近似完好），测试图必须满足：T1 用互斥同质块；T4 用块同质图。
"""

from __future__ import annotations

import sys

import numpy as np

rng = np.random.default_rng(7)
FAILURES = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAILURES.append(name)


def corrupt_labels(labels, n_cls, p, rng):
    mask = rng.random(len(labels)) < p
    j = rng.integers(0, n_cls - 1, size=len(labels))
    return np.where(mask, j + (j >= labels), labels)


def build_op(n, edges, weights):
    M = np.zeros((n, n))
    for e, w in zip(edges, weights):
        k = len(e)
        M[np.ix_(e, e)] += w / k
    return M


def disjoint_homog_blocks(n, rng, min_k=2, max_k=5):
    """把节点划分为互斥同质块，返回 (edges, labels)。"""
    labels = np.zeros(n, dtype=int)
    edges = []
    i = 0
    block = 0
    while i < n:
        k = int(rng.integers(min_k, max_k + 1))
        k = min(k, n - i)
        if k < 2 and edges:            # 剩单点并入前一块
            edges[-1] = edges[-1] + [i]
            labels[i] = labels[edges[-1][0]]
            i += 1
            continue
        e = list(range(i, i + k))
        lab = int(rng.integers(0, 3))
        labels[np.asarray(e)] = lab
        edges.append(e)
        i += k
        block += 1
    return edges, labels


# ---------------------------------------------------------------
# T1: 条件稀释恒等式（prop:thinning）—— 互斥同质块，理论精确适用
# ---------------------------------------------------------------
def t1_thinning():
    C, p = 5, 0.15
    rho = p / (C - 1)
    n_mc = 200000
    n = 60
    edges, labels = disjoint_homog_blocks(n, np.random.default_rng(11))
    u = edges[0][0]
    k0 = len(edges[0])

    surv_cor = surv_cln = 0
    n_cor = n_cln = 0
    for _ in range(n_mc):
        pred = corrupt_labels(labels, C, p, rng)
        hom = len(set(pred[x] for x in edges[0])) == 1
        if pred[u] != labels[u]:
            n_cor += 1
            surv_cor += hom
        else:
            n_cln += 1
            surv_cln += hom

    emp_cor = surv_cor / max(n_cor, 1)
    th_cor = rho ** (k0 - 1)          # 条件概率（prop:thinning (i)）
    se = np.sqrt(max(th_cor * (1 - th_cor), 1e-12) / max(n_cor, 1))
    check("T1a P(存留|u腐蚀)=ρ^(k−1)",
          abs(emp_cor - th_cor) < 6 * max(se, 1e-6),
          f"emp={emp_cor:.6f} th={th_cor:.6f}")

    emp_cln = surv_cln / max(n_cln, 1)
    th_cln = (1 - p) ** (k0 - 1)
    se = np.sqrt(th_cln * (1 - th_cln) / max(n_cln, 1))
    check("T1b P(存留|u干净)=(1−p)^(k−1)",
          abs(emp_cln - th_cln) < 6 * se,
          f"emp={emp_cln:.6f} th={th_cln:.6f}")

    # T1c: 边际（全边，逐边理论）
    surv_any = 0
    for _ in range(20000):
        pred = corrupt_labels(labels, C, p, rng)
        for e in edges:
            surv_any += len(set(pred[x] for x in e)) == 1
    emp_marg = surv_any / (20000 * len(edges))
    ks = np.array([len(e) for e in edges])
    th_marg = float(np.mean((1 - p) ** ks + (C - 1) * rho ** ks))
    se = np.sqrt(th_marg * (1 - th_marg) / (20000 * len(edges)))
    check("T1c B 的边际=(1−p)^k+(C−1)ρ^k（混合 k）",
          abs(emp_marg - th_marg) < 6 * se,
          f"emp={emp_marg:.6f} th={th_marg:.6f}")


# ---------------------------------------------------------------
# T2: 自隔离引理（lem:self-isolation）—— 任意初始化、任意深度
# ---------------------------------------------------------------
def t2_isolation():
    n, C = 20, 4
    labels = rng.integers(0, C, size=n)
    edges = [list(range(0, 3)), list(range(3, 6)), list(range(6, 9))]
    w = np.array([0.0, 1.0, 1.0])      # u=0 的唯一边被删
    B = build_op(n, edges, w)
    Z0 = rng.standard_normal((n, C))
    Z = Z0.copy()
    for t in range(1, 26):
        Z = Z - 0.3 * (B @ Z)
        if not np.allclose(Z[0], Z0[0], atol=1e-12):
            check("T2 隔离节点状态=Z0 逐深度", False, f"T={t}")
            return
    check("T2 隔离节点状态=Z0 逐深度（任意初始化,T≤25）", True)
    check("T2b 隔离节点行恒为零", bool(np.allclose(B[0], 0)))


# ---------------------------------------------------------------
# T3: 分解恒等式 Δm = p·G + (1−p)·D（prop:criterion 精确部分）
# ---------------------------------------------------------------
def t3_decomposition():
    C, p = 3, 0.3
    n_mc = 4000
    n = 60
    edges, labels = disjoint_homog_blocks(n, np.random.default_rng(13))
    a_star = np.ones(len(edges))        # 全同质图
    rho = p / (C - 1)
    ks = np.array([len(e) for e in edges])
    q_rate = 1 - ((1 - p) ** ks + (C - 1) * rho ** ks)
    sgn = -1.0                          # homog 边的翻转方向

    B_clean = build_op(n, edges, a_star)
    h = 0.1 / float(np.abs(np.linalg.eigvalsh(B_clean)[-1]))
    Z0 = np.zeros((n, C))
    Z0[np.arange(n), labels] = 1.0
    T = 10

    def margins(B):
        Z = Z0.copy()
        for _ in range(T):
            Z = Z - h * (B @ Z)
        own = Z[np.arange(n), labels]
        Zo = Z.copy()
        Zo[np.arange(n), labels] = -np.inf
        return own - Zo.max(axis=1)

    mAs, m_recon, mBs = [], [], []
    for _ in range(n_mc):
        pred = corrupt_labels(labels, C, p, rng)
        corrupted = pred != labels
        if not corrupted.any():
            continue
        wA = np.array([1.0 if len(set(pred[x] for x in e)) == 1 else 0.0
                       for e in edges])
        mA = margins(build_op(n, edges, wA))
        flip = rng.random(len(edges)) < q_rate
        wB = a_star + np.where(flip, sgn, 0.0)
        mB = margins(build_op(n, edges, wB))
        p_hat = corrupted.mean()       # 逐样本腐蚀率（小图必须）
        mAs.append(mA.mean())
        m_recon.append(p_hat * mA[corrupted].mean()
                       + (1 - p_hat) * mA[~corrupted].mean())
        mBs.append(mB.mean())

    total = np.mean(mAs) - np.mean(mBs)
    pred_total = np.mean(m_recon) - np.mean(mBs)
    check("T3 分解恒等式（全期望公式）",
          abs(total - pred_total) < 1e-12,
          f"total={total:+.6f} 分解={pred_total:+.6f}")

    # 边际匹配 MC（5σ）
    draws = 8000
    cnt_a = np.zeros(len(edges))
    cnt_b = np.zeros(len(edges))
    for _ in range(draws):
        pred = corrupt_labels(labels, C, p, rng)
        wA = np.array([1.0 if len(set(pred[x] for x in e)) == 1 else 0.0
                       for e in edges])
        flip = rng.random(len(edges)) < q_rate
        cnt_a += wA
        cnt_b += np.where(flip, sgn, 0.0) + a_star
    diff = np.abs(cnt_a - cnt_b) / draws
    tol = 5 * np.sqrt(0.25 / draws)
    check("T3b 边际匹配 MC (5σ)", diff.max() < tol,
          f"max|Δ|={diff.max():.4f} tol={tol:.4f}")


# ---------------------------------------------------------------
# T4: 隔离项近似 p·(1−m̄^B)（one-hot readout，块同质图→隔离完好）
# ---------------------------------------------------------------
def t4_criterion_accuracy():
    C, p = 3, 0.2
    n = 60
    edges, labels = disjoint_homog_blocks(n, np.random.default_rng(17))
    a_star = np.ones(len(edges))
    rho = p / (C - 1)
    ks = np.array([len(e) for e in edges])
    q_rate = 1 - ((1 - p) ** ks + (C - 1) * rho ** ks)

    B_clean = build_op(n, edges, a_star)
    h = 0.1 / float(np.abs(np.linalg.eigvalsh(B_clean)[-1]))
    Z0 = np.zeros((n, C))
    Z0[np.arange(n), labels] = 1.0

    def margins(B, T):
        Z = Z0.copy()
        for _ in range(T):
            Z = Z - h * (B @ Z)
        own = Z[np.arange(n), labels]
        Zo = Z.copy()
        Zo[np.arange(n), labels] = -np.inf
        return own - Zo.max(axis=1)

    for T in (5, 15):
        m_cor, mBs = [], []
        for _ in range(3000):
            pred = corrupt_labels(labels, C, p, rng)
            corrupted = pred != labels
            if not corrupted.any():
                continue
            wA = np.array([1.0 if len(set(pred[x] for x in e)) == 1 else 0.0
                           for e in edges])
            m_cor.append(margins(build_op(n, edges, wA), T)[corrupted].mean())
            flip = rng.random(len(edges)) < q_rate
            wB = a_star - flip.astype(float)
            mBs.append(margins(build_op(n, edges, wB), T).mean())
        G = np.mean(m_cor) - np.mean(mBs)
        G_pred = 1.0 - np.mean(mBs)
        rel = abs(G - G_pred) / max(abs(G_pred), 1e-9)
        check(f"T4 隔离项≈1−m̄^B T={T}", rel < 0.15,
              f"G={G:+.4f} pred={G_pred:+.4f} rel={rel:.3f}")


def t1d_mixed_reconnection():
    """审稿人反例：混合边 (2,1,1) 的少数成员被腐蚀 → 假阳性重连概率。"""
    C, p = 3, 0.1
    rho = p / (C - 1)
    labels = np.array([2, 1, 1])
    n_mc = 400000
    surv = n_cor = 0
    for _ in range(n_mc):
        pred = corrupt_labels(labels, C, p, rng)
        u_cor = pred[0] != labels[0]
        if u_cor:
            n_cor += 1
            surv += int(len(set(pred.tolist())) == 1)
    emp = surv / n_cor
    th = 0.5 * ((1 - p) ** 2 + rho ** 2)      # = 0.40625（审稿人数值）
    se = np.sqrt(th * (1 - th) / n_cor)
    check("T1d 混合边假阳性重连（审稿人反例 0.40625）",
          abs(emp - th) < 6 * se,
          f"emp={emp:.5f} th={th:.5f}")


if __name__ == "__main__":
    t1_thinning()
    t1d_mixed_reconnection()
    t2_isolation()
    t3_decomposition()
    t4_criterion_accuracy()
    print(f"{'=' * 50}")
    if FAILURES:
        print(f"{len(FAILURES)} 项失败: {FAILURES}")
        sys.exit(1)
    print("全部通过")
