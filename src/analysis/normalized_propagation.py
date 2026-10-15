"""审稿 P1 补充：常用归一化传播算子下的下游比较 + 隔离事件完整界。

1. ε_u^all：条件于 u 被腐蚀，每条 incident 边（含混合边）被预测为
   齐次的概率 p_e^cor；ε_u^all = min{1, Σ p_e^cor}。
   齐次边：ρ^{k−1}；混合边：(1/(C−1)) Σ_{c≠y_u} ∏_{v∈e∖u} r_v(c)。
   （审稿人反例：C=3, p=0.1, 边 (2,1,1), u 为少数 → 0.40625。）
2. 非扩张性检查：λmax(B̂) ≤ 2λmax(B*)（|1−hλ|≤1 的条件）。
3. 归一化 APPNP 传播（对称归一化团展开+自环，扰动图度数归一化——
   非线性归一化的匹配条件不自动传递，此处为经验比较）：
   Z ← αSZ + (1−α)Z0, α=0.9；one-hot 真值（对照）与 couple 初始化；
   margin 与准确率差 + bootstrap CI。
"""

from __future__ import annotations

import json
import pickle
import sys

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

sys.stdout.reconfigure(encoding="utf-8")

CORA = "data/cora-cc"
N_REAL = 100
P_LIST = (0.05, 0.10, 0.20)
DEPTHS = (2, 5, 10, 20)
ALPHA = 0.9
SEED = 2026


def load():
    with open(f"{CORA}/hypergraph.pickle", "rb") as f:
        hg = pickle.load(f)
    with open(f"{CORA}/labels.pickle", "rb") as f:
        labels = np.array(pickle.load(f))
    edges = [sorted(int(u) for u in s) for s in hg.values()]
    return edges, labels


def edge_flat_index(edges):
    rows, cols, owner, invk = [], [], [], []
    for i, e in enumerate(edges):
        k = len(e)
        rows.append(np.repeat(e, k)); cols.append(np.tile(e, k))
        owner.append(np.full(k * k, i)); invk.append(np.full(k * k, 1.0 / k))
    return (np.concatenate(rows), np.concatenate(cols),
            np.concatenate(owner), np.concatenate(invk))


def build_op(n, flat, weights):
    rows, cols, owner, invk = flat
    M = np.zeros((n, n))
    np.add.at(M, (rows, cols), invk * weights[owner])
    return M


def corrupt_labels(labels, n_cls, p, rng):
    mask = rng.random(len(labels)) < p
    j = rng.integers(0, n_cls - 1, size=len(labels))
    return np.where(mask, j + (j >= labels), labels)


def expected_da(edges, labels, n_cls, p):
    rho = p / (n_cls - 1)
    w0 = np.zeros(len(edges))
    for i, e in enumerate(edges):
        labs = labels[np.asarray(e)]
        a = 1.0 if len(set(labs.tolist())) == 1 else 0.0
        ph = sum(float(np.prod(np.where(labs == c, 1 - p, rho)))
                 for c in range(n_cls))
        w0[i] = ph - a
    return w0


def normalized_S(n, edges, w):
    """对称归一化团展开 + 自环（用扰动后权重与度数）。"""
    rows, cols = [], []
    for e, wt in zip(edges, w):
        if wt == 0:
            continue
        for a in e:
            for b in e:
                rows.append(a); cols.append(b)
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    A = A + sp.eye(n, format="csr")
    deg = np.asarray(A.sum(axis=1)).ravel()
    d_is = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    return (sp.diags(d_is) @ A @ sp.diags(d_is)).tocsr()


def appnp_record(S, Z0, checkpoints, labels):
    Z = Z0.copy()
    n = Z.shape[0]
    out = {}
    nxt = sorted(checkpoints)
    for t in range(1, max(checkpoints) + 1):
        Z = ALPHA * (S @ Z) + (1 - ALPHA) * Z0
        if nxt and t == nxt[0]:
            nxt.pop(0)
            own = Z[np.arange(n), labels]
            Zo = Z.copy()
            Zo[np.arange(n), labels] = -np.inf
            margin = own - Zo.max(axis=1)
            acc = float(np.mean(np.argmax(Z, axis=1) == labels))
            out[t] = (float(margin.mean()), acc)
    return out


def p_e_cor(edges, labels, n_cls, p):
    """条件于成员 u 被腐蚀时边被预测为齐次的概率（逐边逐成员）。"""
    rho = p / (n_cls - 1)
    out = [np.zeros(len(e)) for e in edges]
    for i, e in enumerate(edges):
        labs = labels[np.asarray(e)]
        k = len(e)
        for j, y_u in enumerate(labs):
            mates = np.delete(labs, j)
            tot = 0.0
            for c in range(n_cls):
                if c == y_u:
                    continue
                r = np.where(mates == c, 1 - p, rho)
                tot += float(np.prod(r))
            out[i][j] = tot / (n_cls - 1)
    return out


def main():
    edges, labels = load()
    n = len(labels)
    n_cls = len(np.unique(labels))
    m = len(edges)
    flat = edge_flat_index(edges)
    a_star = np.array([1.0 if len(set(labels[u] for u in e)) == 1 else 0.0
                       for e in edges])
    B_clean = build_op(n, flat, a_star)
    lam_clean = float(abs(eigsh(B_clean, k=1, which="LM",
                                return_eigenvectors=False)[0]))
    print(f"n={n}, m={m}, λmax(B*)={lam_clean:.1f}")

    # ---- 审稿人反例核对 ----
    C3, p3 = 3, 0.1
    rho3 = p3 / (C3 - 1)
    ex = 0.5 * ((1 - p3) ** 2 + rho3 ** 2)
    print(f"反例核对 (C=3,p=0.1,边(2,1,1),u 少数): {ex:.5f} (审稿人 0.40625)")

    # ---- 1. ε^all ----
    print("\n=== ε_u^all（完整隔离失败界，含混合边假阳性重连） ===")
    eps_all_stats = {}
    for p in P_LIST:
        pec = p_e_cor(edges, labels, n_cls, p)
        eps = np.zeros(n)
        for e, pec_e in zip(edges, pec):
            for u, val in zip(e, pec_e):
                eps[u] += val
        eps = np.minimum(eps, 1.0)
        rho = p / (n_cls - 1)
        eps_hom = np.zeros(n)
        for e, a in zip(edges, a_star):
            if a == 1:
                for u in e:
                    eps_hom[u] += rho ** (len(e) - 1)
        eps_hom = np.minimum(eps_hom, 1.0)
        print(f"  p={p}: 仅齐次 E[ε]={eps_hom.mean():.4f} "
              f"完整 E[ε^all]={eps.mean():.4f} 中位={np.median(eps):.4f} "
              f"max={eps.max():.3f} (ε>0.5 的节点数 {(eps > 0.5).sum()})")
        eps_all_stats[f"p={p}"] = dict(
            mean=float(eps.mean()), median=float(np.median(eps)),
            max=float(eps.max()), n_gt_half=int((eps > 0.5).sum()),
            hom_only_mean=float(eps_hom.mean()))

    # ---- 2. 非扩张性检查 ----
    print("\n=== 非扩张性：λmax(B̂) ≤ 2λmax(B*)? ===")
    ratios = []
    rngchk = np.random.default_rng(555)
    for p in P_LIST:
        w0 = expected_da(edges, labels, n_cls, p)
        q_rate, sgn = np.abs(w0), np.sign(w0)
        worst = 0.0
        for _ in range(50):
            pred = corrupt_labels(labels, n_cls, p, rngchk)
            wA = np.array([1.0 if len(set(pred[u] for u in e)) == 1 else 0.0
                           for e in edges])
            Bt = build_op(n, flat, wA)
            lam = float(abs(eigsh(Bt, k=1, which="LM",
                                  return_eigenvectors=False,
                                  tol=1e-4, maxiter=3000)[0]))
            worst = max(worst, lam / lam_clean)
        ratios.append(worst)
        print(f"  p={p}: max λmax(B̂)/λmax(B*) = {worst:.3f} "
              f"({'安全(≤2)' if worst <= 2 else '违反'})")

    # ---- 3. 归一化 APPNP 传播 ----
    print(f"\n=== 归一化 APPNP（α={ALPHA}）margin/acc 差（A−B） ===")
    Z0_clean = np.eye(n_cls)[labels]
    results = {"eps_all": eps_all_stats,
               "lam_ratio_max": [float(r) for r in ratios],
               "reviewer_example": float(ex), "appnp": {}}
    for p in P_LIST:
        w0 = expected_da(edges, labels, n_cls, p)
        q_rate, sgn = np.abs(w0), np.sign(w0)
        rng = np.random.default_rng(SEED + int(p * 100))
        for mode in ("clean", "couple"):
            mA = {t: [] for t in DEPTHS}
            mB = {t: [] for t in DEPTHS}
            aA = {t: [] for t in DEPTHS}
            aB = {t: [] for t in DEPTHS}
            for _ in range(N_REAL):
                pred = corrupt_labels(labels, n_cls, p, rng)
                wA = np.array([1.0 if len(set(pred[u] for u in e)) == 1
                               else 0.0 for e in edges])
                flip = rng.random(m) < q_rate
                wB = a_star + np.where(flip, sgn, 0.0)
                Z0 = Z0_clean if mode == "clean" else np.eye(n_cls)[pred]
                rA = appnp_record(normalized_S(n, edges, wA), Z0,
                                  DEPTHS, labels)
                rB = appnp_record(normalized_S(n, edges, wB), Z0,
                                  DEPTHS, labels)
                for t in DEPTHS:
                    mA[t].append(rA[t][0]); mB[t].append(rB[t][0])
                    aA[t].append(rA[t][1]); aB[t].append(rB[t][1])
            rng_bs = np.random.default_rng(99)
            ia = rng_bs.integers(0, N_REAL, (2000, N_REAL))
            ib = rng_bs.integers(0, N_REAL, (2000, N_REAL))
            key = f"p={p}:{mode}"
            results["appnp"][key] = {}
            print(f"\n--- {key} ---")
            for t in DEPTHS:
                x, y = np.array(mA[t]), np.array(mB[t])
                xa = np.array(aA[t])
                ya = np.array(aB[t])
                d = x[ia].mean(1) - y[ib].mean(1)
                lo, hi = np.quantile(d, [0.025, 0.975])
                dac = xa[ia].mean(1) - ya[ib].mean(1)
                loa, hia = np.quantile(dac, [0.025, 0.975])
                r = dict(margin_A=float(x.mean()), margin_B=float(y.mean()),
                         diff=float(x.mean() - y.mean()),
                         ci=[float(lo), float(hi)],
                         acc_diff=float(xa.mean() - ya.mean()),
                         acc_ci=[float(loa), float(hia)])
                results["appnp"][key][f"T={t}"] = r
                print(f"  T={t:>2}: margin diff={r['diff']:+.5f} "
                      f"CI[{lo:+.5f},{hi:+.5f}]; acc diff={r['acc_diff']:+.4f} "
                      f"CI[{r['acc_ci'][0]:+.4f},{r['acc_ci'][1]:+.4f}]")

    with open("results/normalized_propagation.json", "w",
              encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print("\n已写出 results/normalized_propagation.json")


if __name__ == "__main__":
    main()
