"""传导检验 v3：修正边际匹配后的下游 margin 传导。

v2 的两个问题（已修）：
  1. labels*(1−2W) 符号翻转只适用于二值标签；7 类下模型 A 的异质边
     永远不会变同质，而模型 B 以 p(1−p) 翻转 —— 边际不匹配，B 系统性
     多出同质边。v3 用多类对称信道（保留 1−p，否则均匀换类）。
  2. q_h=3p(1−p)/q_m=p(1−p) 只对 k=3 边精确。v3 用每边解析
     E[dA_e]=P(所有 pred 相同)−a*_e 作为 B 的精确速率（任意 k）。

输出：margin/acc/信号比 的 A−B 差 + bootstrap CI，线性滤波
Z_{t+1}=(I−hB)Z_t，深度 T∈{1,2,5,10,20}，p∈{0.05,0.10,0.20}。
另加 ReLU 模式（浅层+高噪声处 v2 观察到方向翻转，验证修正后是否仍在）。
"""

from __future__ import annotations

import json
import pickle
import sys

import numpy as np
from scipy.sparse.linalg import eigsh

sys.stdout.reconfigure(encoding="utf-8")

CORA = "data/cora-cc"
N_REALIZATIONS = 100
DEPTH_LIST = (1, 2, 5, 10, 20)
P_LIST = (0.05, 0.10, 0.20)
SEED = 2026


def load():
    with open(f"{CORA}/features.pickle", "rb") as f:
        feats = pickle.load(f)
    with open(f"{CORA}/hypergraph.pickle", "rb") as f:
        hg = pickle.load(f)
    with open(f"{CORA}/labels.pickle", "rb") as f:
        labels = np.array(pickle.load(f))
    edges = [sorted(int(u) for u in s) for s in hg.values()]
    return feats, edges, labels


def edge_flat_index(edges):
    rows, cols, owner, invk = [], [], [], []
    for i, e in enumerate(edges):
        k = len(e)
        rows.append(np.repeat(e, k))
        cols.append(np.tile(e, k))
        owner.append(np.full(k * k, i))
        invk.append(np.full(k * k, 1.0 / k))
    return (np.concatenate(rows), np.concatenate(cols),
            np.concatenate(owner), np.concatenate(invk))


def build_op(n, flat, weights):
    rows, cols, owner, invk = flat
    M = np.zeros((n, n))
    np.add.at(M, (rows, cols), invk * weights[owner])
    return M


def corrupt_labels(labels, n_cls, p, rng):
    n = len(labels)
    mask = rng.random(n) < p
    j = rng.integers(0, n_cls - 1, size=n)
    other = j + (j >= labels)
    return np.where(mask, other, labels)


def expected_da(edges, labels, n_cls, p):
    rho = p / (n_cls - 1)
    w0 = np.zeros(len(edges))
    for i, e in enumerate(edges):
        labs = labels[np.asarray(e)]
        a_star = 1.0 if len(set(labs.tolist())) == 1 else 0.0
        p_hom = 0.0
        for c in range(n_cls):
            r = np.where(labs == c, 1.0 - p, rho)
            p_hom += float(np.prod(r))
        w0[i] = p_hom - a_star
    return w0


def margin_metrics(Z, labels, n_cls):
    n = Z.shape[0]
    Z_true = Z[np.arange(n), labels]
    Z_other = Z.copy()
    Z_other[np.arange(n), labels] = -np.inf
    margin = Z_true - Z_other.max(axis=1)
    pred = np.argmax(Z, axis=1)
    acc = float(np.mean(pred == labels))
    overall = Z.mean(axis=0)
    between = sum(np.sum((labels == c).sum() *
                         (Z[labels == c].mean(0) - overall) ** 2)
                  for c in range(n_cls))
    within = sum(np.sum((Z[labels == c] - Z[labels == c].mean(0)) ** 2)
                 for c in range(n_cls))
    signal = float(between / max(between + within, 1e-12))
    return float(np.mean(margin)), acc, signal


def propagate_record(B, Z0, h, T_max, checkpoints, mode="linear"):
    """传播到 T_max，在 checkpoints 处记录 margin/acc/signal。"""
    Z = Z0.copy()
    out = {}
    nxt = sorted(checkpoints)
    for t in range(1, T_max + 1):
        Z = Z - h * (B @ Z)
        if mode == "relu":
            Z = np.maximum(Z, 0.0)
        if nxt and t == nxt[0]:
            nxt.pop(0)
            out[t] = margin_metrics(Z, LABELS_G, N_CLS_G)
    return out


LABELS_G = None
N_CLS_G = None


def main():
    global LABELS_G, N_CLS_G
    feats, edges, labels = load()
    n = len(labels)
    n_cls = len(np.unique(labels))
    m = len(edges)
    LABELS_G, N_CLS_G = labels, n_cls
    flat = edge_flat_index(edges)

    a_star = np.array([1.0 if len(set(labels[u] for u in e)) == 1 else 0.0
                       for e in edges])
    B_clean = build_op(n, flat, a_star)
    lam = float(abs(eigsh(B_clean, k=1, which="LM",
                          return_eigenvectors=False)[0]))
    h = 1.0 / lam
    Z0 = np.zeros((n, n_cls))
    Z0[np.arange(n), labels] = 1.0
    print(f"n={n}, m={m}, λ_max={lam:.1f}, h={h:.4f}")

    results = {}
    for p in P_LIST:
        w0 = expected_da(edges, labels, n_cls, p)
        q_rate = np.abs(w0)
        sgn = np.sign(w0)
        rng = np.random.default_rng(SEED + int(p * 100))
        print(f"\n--- p={p:.2f} (线性) ---")

        mA = {t: [] for t in DEPTH_LIST}
        mB = {t: [] for t in DEPTH_LIST}
        aA = {t: [] for t in DEPTH_LIST}
        aB = {t: [] for t in DEPTH_LIST}
        sA = {t: [] for t in DEPTH_LIST}
        sB = {t: [] for t in DEPTH_LIST}
        for _ in range(N_REALIZATIONS):
            for model in ("A", "B"):
                if model == "A":
                    pred = corrupt_labels(labels, n_cls, p, rng)
                    w = np.array([1.0 if len(set(pred[u] for u in e)) == 1
                                  else 0.0 for e in edges])
                else:
                    flip = rng.random(m) < q_rate
                    w = a_star + np.where(flip, sgn, 0.0)
                B = build_op(n, flat, w)
                rec = propagate_record(B, Z0, h, max(DEPTH_LIST),
                                       DEPTH_LIST, mode="linear")
                dst = (mA, aA, sA) if model == "A" else (mB, aB, sB)
                for t in DEPTH_LIST:
                    dst[0][t].append(rec[t][0])
                    dst[1][t].append(rec[t][1])
                    dst[2][t].append(rec[t][2])

        rng_bs = np.random.default_rng(99)
        ia = rng_bs.integers(0, N_REALIZATIONS, (2000, N_REALIZATIONS))
        ib = rng_bs.integers(0, N_REALIZATIONS, (2000, N_REALIZATIONS))
        results[f"p={p}"] = {}
        for t in DEPTH_LIST:
            d = np.array(mA[t])[ia].mean(1) - np.array(mB[t])[ib].mean(1)
            lo, hi = np.quantile(d, [0.025, 0.975])
            da = np.array(aA[t])[ia].mean(1) - np.array(aB[t])[ib].mean(1)
            loa, hia = np.quantile(da, [0.025, 0.975])
            r = dict(margin_A=float(np.mean(mA[t])),
                     margin_B=float(np.mean(mB[t])),
                     diff=float(np.mean(mA[t]) - np.mean(mB[t])),
                     ci=[float(lo), float(hi)],
                     acc_A=float(np.mean(aA[t])), acc_B=float(np.mean(aB[t])),
                     acc_diff=float(np.mean(aA[t]) - np.mean(aB[t])),
                     acc_ci=[float(loa), float(hia)],
                     sig_A=float(np.mean(sA[t])), sig_B=float(np.mean(sB[t])))
            results[f"p={p}"][f"T={t}"] = r
            print(f"  T={t:>2}: margin A={r['margin_A']:.4f} "
                  f"B={r['margin_B']:.4f} (diff {r['diff']:+.5f} "
                  f"CI [{lo:+.5f},{hi:+.5f}]); acc diff "
                  f"{r['acc_diff']:+.4f} CI [{loa:+.4f},{hia:+.4f}]")

    # ReLU 模式：v2 在 p=0.20, T≤5 观察到方向翻转
    print("\n--- ReLU 模式（每步截断） ---")
    results["relu"] = {}
    for p in (0.10, 0.20):
        w0 = expected_da(edges, labels, n_cls, p)
        q_rate, sgn = np.abs(w0), np.sign(w0)
        rng = np.random.default_rng(SEED + int(p * 100) + 1)
        depths = (1, 2, 5)
        mA = {t: [] for t in depths}
        mB = {t: [] for t in depths}
        for _ in range(N_REALIZATIONS):
            for model in ("A", "B"):
                if model == "A":
                    pred = corrupt_labels(labels, n_cls, p, rng)
                    w = np.array([1.0 if len(set(pred[u] for u in e)) == 1
                                  else 0.0 for e in edges])
                else:
                    flip = rng.random(m) < q_rate
                    w = a_star + np.where(flip, sgn, 0.0)
                B = build_op(n, flat, w)
                rec = propagate_record(B, Z0, h, max(depths), depths,
                                       mode="relu")
                dst = mA if model == "A" else mB
                for t in depths:
                    dst[t].append(rec[t][0])
        results["relu"][f"p={p}"] = {}
        for t in depths:
            d = np.array(mA[t])[ia].mean(1) - np.array(mB[t])[ib].mean(1)
            lo, hi = np.quantile(d, [0.025, 0.975])
            r = dict(margin_A=float(np.mean(mA[t])),
                     margin_B=float(np.mean(mB[t])),
                     diff=float(np.mean(mA[t]) - np.mean(mB[t])),
                     ci=[float(lo), float(hi)])
            results["relu"][f"p={p}"][f"T={t}"] = r
            print(f"  p={p} T={t}: margin A={r['margin_A']:.4f} "
                  f"B={r['margin_B']:.4f} (diff {r['diff']:+.5f} "
                  f"CI [{lo:+.5f},{hi:+.5f}])")

    with open("results/downstream_v3.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print("\n已写出 results/downstream_v3.json")


if __name__ == "__main__":
    main()
