"""DBLP-CA coupled 初始化归一化传播比较（向量化快速版）。

与 dblp_normalized.py 相同协议；边索引/同质性全部预计算向量化。
APPNP: Z_t = αS(â)Z_{t−1} + (1−α)Z_0, α=0.9, 对称归一化团展开+自环。
"""

from __future__ import annotations

import json
import pickle
import sys

import numpy as np
import scipy.sparse as sp

sys.stdout.reconfigure(encoding="utf-8")

DBLP = "data/dblp-ca"
N_REAL = 100
P_LIST = (0.05, 0.10, 0.20)
DEPTHS = (2, 5, 10, 20)
ALPHA = 0.9
SEED = 2026


def load():
    with open(f"{DBLP}/hypergraph.pickle", "rb") as f:
        hg = pickle.load(f)
    with open(f"{DBLP}/labels.pickle", "rb") as f:
        labels = np.array(pickle.load(f))
    edges = [sorted(int(u) for u in s) for s in hg.values()]
    return edges, labels


def main():
    edges, labels = load()
    n = len(labels)
    n_cls = len(np.unique(labels))
    m = len(edges)
    k_arr = np.array([len(e) for e in edges])
    k_max = int(k_arr.max())
    edge_arr = np.full((m, k_max), -1, dtype=int)
    for i, e in enumerate(edges):
        edge_arr[i, :len(e)] = e
    # 预计算每条边每类的成员计数矩阵 CNT (m, n_cls)
    CNT = np.zeros((m, n_cls))
    for i, e in enumerate(edges):
        CNT[i] = np.bincount(labels[np.asarray(e)], minlength=n_cls)
    a_star = (CNT.max(axis=1) == k_arr).astype(float)   # 全部成员同类
    # 预计算 Σ_e 1_e 1_e^T 的稀疏结构（flat）
    rows_list, cols_list, owner_list, invk_list = [], [], [], []
    for i, e in enumerate(edges):
        k = len(e)
        rows_list.append(np.repeat(e, k))
        cols_list.append(np.tile(e, k))
        owner_list.append(np.full(k * k, i))
        invk_list.append(np.full(k * k, 1.0 / k))
    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)
    owner = np.concatenate(owner_list)
    invk_flat = np.concatenate(invk_list)
    entries = len(rows)
    print(f"DBLP: n={n}, m={m}, n_cls={n_cls}, 团展开条目 {entries}",
          flush=True)

    def all_hom(pred):
        first = edge_arr[:, 0]
        same = pred[edge_arr] == pred[first][:, None]
        same[edge_arr < 0] = True
        return same.all(axis=1).astype(float)

    def normalized_S(w):
        data = invk_flat * w[owner]
        A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
        A = A + sp.eye(n, format="csr")
        deg = np.asarray(A.sum(axis=1)).ravel()
        d_is = sp.diags(1.0 / np.sqrt(np.maximum(deg, 1e-12)))
        return (d_is @ A @ d_is).tocsr()

    def expected_da(p):
        rho = p / (n_cls - 1)
        # P(全同) = Σ_c ∏_u r_u(c) = Σ_c (1−p)^{n_c} · ρ^{k−n_c}（c 取全部类；
        # c 不在边内时 n_c=0 项恰为 ρ^k）——对齐 attribution 协议的逐边解析式
        terms = np.power(1 - p, CNT) * np.power(rho, k_arr[:, None] - CNT)
        return terms.sum(axis=1) - a_star

    def appnp_record(S, Z0):
        Z = Z0.copy()
        out = {}
        nxt = sorted(DEPTHS)
        for t in range(1, max(DEPTHS) + 1):
            Z = ALPHA * (S @ Z) + (1 - ALPHA) * Z0
            if t == nxt[0]:
                nxt.pop(0)
                own = Z[np.arange(n), labels]
                Zo = Z.copy()
                Zo[np.arange(n), labels] = -np.inf
                margin = own - Zo.max(axis=1)
                acc = float(np.mean(np.argmax(Z, axis=1) == labels))
                out[t] = (float(margin.mean()), acc)
        return out

    Z0_clean = np.eye(n_cls)[labels]
    results = {}
    for p in P_LIST:
        w0 = expected_da(p)
        q_rate, sgn = np.abs(w0), np.sign(w0)
        rng = np.random.default_rng(SEED + int(p * 100))
        for mode in ("clean", "couple"):
            mA = {t: [] for t in DEPTHS}
            mB = {t: [] for t in DEPTHS}
            aA = {t: [] for t in DEPTHS}
            aB = {t: [] for t in DEPTHS}
            for r in range(N_REAL):
                mask = rng.random(n) < p
                j = rng.integers(0, n_cls - 1, size=n)
                pred = np.where(mask, j + (j >= labels), labels)
                wA = all_hom(pred)
                flip = rng.random(m) < q_rate
                wB = a_star + np.where(flip, sgn, 0.0)
                Z0 = Z0_clean if mode == "clean" else np.eye(n_cls)[pred]
                rA = appnp_record(normalized_S(wA), Z0)
                rB = appnp_record(normalized_S(wB), Z0)
                for t in DEPTHS:
                    mA[t].append(rA[t][0]); mB[t].append(rB[t][0])
                    aA[t].append(rA[t][1]); aB[t].append(rB[t][1])
                if (r + 1) % 25 == 0:
                    print(f"  p={p} {mode}: {r + 1}/{N_REAL}", flush=True)
            rng_bs = np.random.default_rng(99)
            ia = rng_bs.integers(0, N_REAL, (2000, N_REAL))
            ib = rng_bs.integers(0, N_REAL, (2000, N_REAL))
            key = f"p={p}:{mode}"
            results[key] = {}
            print(f"--- {key} ---")
            for t in DEPTHS:
                x, y = np.array(mA[t]), np.array(mB[t])
                xa, ya = np.array(aA[t]), np.array(aB[t])
                d = x[ia].mean(1) - y[ib].mean(1)
                lo, hi = np.quantile(d, [0.025, 0.975])
                dac = xa[ia].mean(1) - ya[ib].mean(1)
                loa, hia = np.quantile(dac, [0.025, 0.975])
                rr = dict(margin_A=float(x.mean()), margin_B=float(y.mean()),
                          diff=float(x.mean() - y.mean()),
                          ci=[float(lo), float(hi)],
                          acc_diff=float(xa.mean() - ya.mean()),
                          acc_ci=[float(loa), float(hia)])
                results[key][f"T={t}"] = rr
                print(f"  T={t:>2}: margin diff={rr['diff']:+.5f} "
                      f"CI[{lo:+.5f},{hi:+.5f}]; acc diff "
                      f"{rr['acc_diff']:+.4f} CI[{loa:+.4f},{hia:+.4f}]")

    with open("results/dblp_normalized.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print("\n已写出 results/dblp_normalized.json")


if __name__ == "__main__":
    main()
