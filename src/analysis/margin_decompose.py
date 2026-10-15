"""margin 差的机制诊断：为什么范数更大的 A 反而 margin 更高？

分解（p=0.10, T∈{5,20}, 线性滤波, 100 次实现）：
  1. margin 差 = [own 坐标差] − [competitor 坐标差]
     own_i = Z[i, y_i], comp_i = max_{c≠y_i} Z[i,c]
  2. A 的 margin 差按节点类型拆分：被腐蚀节点 vs 未腐蚀节点
  3. 按同质度拆分：节点周围同质边占比（四分位）
"""

from __future__ import annotations

import pickle
import sys

import numpy as np
from scipy.sparse.linalg import eigsh

sys.stdout.reconfigure(encoding="utf-8")

CORA = "data/cora-cc"
N_REAL = 100
P = 0.10
DEPTHS = (5, 20)
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


def own_comp(Z, labels):
    own = Z[np.arange(len(labels)), labels]
    Zo = Z.copy()
    Zo[np.arange(len(labels)), labels] = -np.inf
    return own, Zo.max(axis=1)


def main():
    edges, labels = load()
    n = len(labels)
    n_cls = len(np.unique(labels))
    m = len(edges)
    flat = edge_flat_index(edges)
    a_star = np.array([1.0 if len(set(labels[u] for u in e)) == 1 else 0.0
                       for e in edges])
    B_clean = build_op(n, flat, a_star)
    h = 1.0 / float(abs(eigsh(B_clean, k=1, which="LM",
                              return_eigenvectors=False)[0]))
    Z0 = np.zeros((n, n_cls))
    Z0[np.arange(n), labels] = 1.0

    w0 = expected_da(edges, labels, n_cls, P)
    q_rate, sgn = np.abs(w0), np.sign(w0)
    rng = np.random.default_rng(SEED + int(P * 100))

    # 节点同质度：周围同质边占比
    node_hom = np.zeros(n); node_tot = np.zeros(n)
    for e, a in zip(edges, a_star):
        for u in e:
            node_tot[u] += 1; node_hom[u] += a
    hom_frac = node_hom / np.maximum(node_tot, 1)
    q_hom = np.quantile(hom_frac, [0.25, 0.5, 0.75])

    acc = {t: {"ownA": [], "ownB": [], "cpA": [], "cpB": [],
               "mA_cor": [], "mA_cln": [], "mB_at_cor": [], "mB_rest": [],
               "mA_q": [[] for _ in range(4)], "mB_q": [[] for _ in range(4)]}
           for t in DEPTHS}

    for _ in range(N_REAL):
        pred = corrupt_labels(labels, n_cls, P, rng)
        corrupted = pred != labels
        wA = np.array([1.0 if len(set(pred[u] for u in e)) == 1 else 0.0
                       for e in edges])
        flip = rng.random(m) < q_rate
        wB = a_star + np.where(flip, sgn, 0.0)
        for name, w in (("A", wA), ("B", wB)):
            B = build_op(n, flat, w)
            Z = Z0.copy()
            for t in range(1, max(DEPTHS) + 1):
                Z = Z - h * (B @ Z)
                if t in DEPTHS:
                    own, cp = own_comp(Z, labels)
                    marg = own - cp
                    d = acc[t]
                    if name == "A":
                        d["ownA"].append(own.mean())
                        d["cpA"].append(cp.mean())
                        d["mA_cor"].append(marg[corrupted].mean())
                        d["mA_cln"].append(marg[~corrupted].mean())
                        for qi in range(4):
                            lo = q_hom[max(qi - 1, 0)] if qi else -1
                            sel = (hom_frac > lo) & (hom_frac <= q_hom[qi]) \
                                if qi < 3 else hom_frac > q_hom[2]
                            d["mA_q"][qi].append(marg[sel].mean())
                    else:
                        d["ownB"].append(own.mean())
                        d["cpB"].append(cp.mean())
                        d["mB_at_cor"].append(marg[corrupted].mean())
                        d["mB_rest"].append(marg[~corrupted].mean())
                        for qi in range(4):
                            lo = q_hom[max(qi - 1, 0)] if qi else -1
                            sel = (hom_frac > lo) & (hom_frac <= q_hom[qi]) \
                                if qi < 3 else hom_frac > q_hom[2]
                            d["mB_q"][qi].append(marg[sel].mean())

    print(f"p={P}, {N_REAL} 实现, 同质度四分位切点 {q_hom.round(3)}")
    for t in DEPTHS:
        d = acc[t]
        print(f"\n=== T={t} ===")
        print(f"  own:   A={np.mean(d['ownA']):.4f} B={np.mean(d['ownB']):.4f} "
              f"(A−B={np.mean(d['ownA']) - np.mean(d['ownB']):+.5f})")
        print(f"  comp:  A={np.mean(d['cpA']):.4f} B={np.mean(d['cpB']):.4f} "
              f"(A−B={np.mean(d['cpA']) - np.mean(d['cpB']):+.5f})")
        print(f"  margin(A): 被腐蚀节点={np.mean(d['mA_cor']):.4f} "
              f"未腐蚀={np.mean(d['mA_cln']):.4f} "
              f"(差 {np.mean(d['mA_cln']) - np.mean(d['mA_cor']):+.5f})")
        print(f"  margin(B): 同一批'被腐蚀'节点={np.mean(d['mB_at_cor']):.4f} "
              f"其余={np.mean(d['mB_rest']):.4f} "
              f"(差 {np.mean(d['mB_rest']) - np.mean(d['mB_at_cor']):+.5f})")
        print("  margin 按同质度四分位 (低→高):")
        for qi in range(4):
            a_q, b_q = np.mean(d["mA_q"][qi]), np.mean(d["mB_q"][qi])
            print(f"    Q{qi + 1}: A={a_q:.4f} B={b_q:.4f} (A−B={a_q - b_q:+.5f})")


if __name__ == "__main__":
    main()
