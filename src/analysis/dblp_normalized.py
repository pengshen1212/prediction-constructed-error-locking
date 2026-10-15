"""DBLP-CA 上的 coupled 初始化归一化传播比较（审稿人优先建议）。

检验 couple 模式反转是否超出 Cora。APPNP 式传播（对称归一化团展开+
自环，restart 项 (1−α)Z0，α=0.9 为传播权重），扰动度数归一化（经验
比较）。one-hot 真值对照 + couple 初始化，margin/acc 差双区间。
"""

from __future__ import annotations

import json
import pickle
import sys

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

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


def normalized_S(n, edges, w):
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


def appnp_record(S, Z0, checkpoints, labels):
    Z = np.asarray(Z0.todense()) if sp.issparse(Z0) else Z0.copy()
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


def main():
    edges, labels = load()
    n = len(labels)
    n_cls = len(np.unique(labels))
    m = len(edges)
    print(f"DBLP: n={n}, m={m}, n_cls={n_cls}")
    a_star = np.array([1.0 if len(set(labels[u] for u in e)) == 1 else 0.0
                       for e in edges])
    Z0_clean = sp.csr_matrix(np.eye(n_cls)[labels])

    results = {}
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
                Z0 = Z0_clean if mode == "clean" else sp.csr_matrix(
                    np.eye(n_cls)[pred])
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
            results[key] = {}
            print(f"\n--- {key} ---")
            for t in DEPTHS:
                x, y = np.array(mA[t]), np.array(mB[t])
                xa, ya = np.array(aA[t]), np.array(aB[t])
                d = x[ia].mean(1) - y[ib].mean(1)
                lo, hi = np.quantile(d, [0.025, 0.975])
                dac = xa[ia].mean(1) - ya[ib].mean(1)
                loa, hia = np.quantile(dac, [0.025, 0.975])
                r = dict(margin_A=float(x.mean()), margin_B=float(y.mean()),
                         diff=float(x.mean() - y.mean()),
                         ci=[float(lo), float(hi)],
                         acc_diff=float(xa.mean() - ya.mean()),
                         acc_ci=[float(loa), float(hia)])
                results[key][f"T={t}"] = r
                print(f"  T={t:>2}: margin diff={r['diff']:+.5f} "
                      f"CI[{lo:+.5f},{hi:+.5f}]; acc diff "
                      f"{r['acc_diff']:+.4f} CI[{loa:+.4f},{hia:+.4f}]")

    with open("results/dblp_normalized.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print("\n已写出 results/dblp_normalized.json")


if __name__ == "__main__":
    main()
