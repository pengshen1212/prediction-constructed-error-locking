"""带符号判据的分解与校准（E1 toggle + E2 分解）。

E1 单边 toggle：homog 边移除 / heterog 边强制加入对 margin 的灵敏度
   c_e(T) = m̄(B* ∓ e) − m̄(B*)，确定符号与非线性形状。
E2 精确分解：E[m^A] − E[m^B] = p·(隔离增益) + (1−p)·(干净节点对比)
   + 每节点判据命中率（腐蚀节点 A 更好 / 干净节点 A 更差的比例）
   + 移除计数的均值/方差对比（稀释 vs 离散的竞争）。
"""

from __future__ import annotations

import json
import pickle
import sys

import numpy as np
from scipy.sparse.linalg import eigsh

sys.stdout.reconfigure(encoding="utf-8")

CORA = "data/cora-cc"
N_REAL = 100
P_LIST = (0.05, 0.10, 0.20)
DEPTHS = (2, 5, 10, 20)
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


def margins_at_checkpoints(B, Z0, h, checkpoints):
    Z = Z0.copy()
    out = {}
    nxt = sorted(checkpoints)
    n = Z.shape[0]
    labels = LABELS_G
    for t in range(1, max(checkpoints) + 1):
        Z = Z - h * (B @ Z)
        if nxt and t == nxt[0]:
            nxt.pop(0)
            own = Z[np.arange(n), labels]
            Zo = Z.copy()
            Zo[np.arange(n), labels] = -np.inf
            out[t] = own - Zo.max(axis=1)
    return out


LABELS_G = None


def e1_toggles(n, flat, edges, a_star, B_clean, Z0, h, rng):
    """单边 toggle 灵敏度 + 多边非线性形状。"""
    hom_idx = np.where(a_star == 1)[0]
    het_idx = np.where(a_star == 0)[0]
    base = margins_at_checkpoints(B_clean, Z0, h, DEPTHS)
    out = {t: {"rem": [], "add": []} for t in DEPTHS}

    sample = rng.choice(hom_idx, size=min(150, len(hom_idx)),
                        replace=False)
    for i in sample:
        w = a_star.copy(); w[i] = 0.0
        m = margins_at_checkpoints(build_op(n, flat, w), Z0, h, DEPTHS)
        for t in DEPTHS:
            out[t]["rem"].append(m[t].mean() - base[t].mean())
    sample = rng.choice(het_idx, size=min(100, len(het_idx)),
                        replace=False)
    for i in sample:
        w = a_star.copy(); w[i] = 1.0
        m = margins_at_checkpoints(build_op(n, flat, w), Z0, h, DEPTHS)
        for t in DEPTHS:
            out[t]["add"].append(m[t].mean() - base[t].mean())

    # 非线性形状：随机移除 r 条 homog 边（平均 over 抽样）
    shape = {t: {} for t in DEPTHS}
    for r in (1, 2, 4, 8, 16, 32):
        vals = {t: [] for t in DEPTHS}
        for _ in range(12):
            w = a_star.copy()
            w[rng.choice(hom_idx, size=r, replace=False)] = 0.0
            m = margins_at_checkpoints(build_op(n, flat, w), Z0, h, DEPTHS)
            for t in DEPTHS:
                vals[t].append(m[t].mean() - base[t].mean())
        for t in DEPTHS:
            shape[t][r] = float(np.mean(vals[t]))
    return out, shape


def e2_decomposition(n, flat, edges, a_star, Z0, h, p, rng):
    """E[m^A]−E[m^B] = p·隔离增益 + (1−p)·干净对比；每节点判据。"""
    m = len(edges)
    hom_mask = a_star == 1
    w0 = expected_da(edges, LABELS_G, NCLS_G, p)
    q_rate, sgn = np.abs(w0), np.sign(w0)

    per_cor = {t: [] for t in DEPTHS}   # E[m_u^A | u 腐蚀]
    per_cln = {t: [] for t in DEPTHS}   # E[m_u^A | u 干净]
    per_B = {t: [] for t in DEPTHS}     # E_EB[m_u]
    rA_cln, rB = [], []                 # 干净节点/A 与 B 的移除边数
    rows = flat[0]
    owner = flat[2]
    node_removed = {}
    for _ in range(N_REAL):
        pred = corrupt_labels(LABELS_G, NCLS_G, p, rng)
        corrupted = pred != LABELS_G
        wA = np.array([1.0 if len(set(pred[u] for u in e)) == 1 else 0.0
                       for e in edges])
        mA = margins_at_checkpoints(build_op(n, flat, wA), Z0, h, DEPTHS)
        flip = rng.random(m) < q_rate
        wB = a_star + np.where(flip, sgn, 0.0)
        mB = margins_at_checkpoints(build_op(n, flat, wB), Z0, h, DEPTHS)
        remA = np.zeros(n)
        remB = np.zeros(n)
        np.add.at(remA, rows, ((wA == 0) & hom_mask)[owner])
        np.add.at(remB, rows, ((wB == 0) & hom_mask)[owner])
        node_removed.clear()
        node_removed["A"], node_removed["B"] = remA, remB
        for t in DEPTHS:
            per_cor[t].append(mA[t][corrupted].mean())
            per_cln[t].append(mA[t][~corrupted].mean())
            per_B[t].append(mB[t].mean())
        rA_cln.append(float(node_removed["A"][~corrupted].mean()))
        rB.append(float(node_removed["B"].mean()))

    res = {}
    for t in DEPTHS:
        ec, el, eb = (np.mean(per_cor[t]), np.mean(per_cln[t]),
                      np.mean(per_B[t]))
        iso = p * (ec - eb)
        cln = (1 - p) * (el - eb)
        res[t] = dict(E_corrupted=float(ec), E_clean=float(el),
                      E_B=float(eb), isolation_gain=float(iso),
                      clean_contrast=float(cln),
                      total_pred=float(iso + cln),
                      measured=float(ec * p + el * (1 - p) - eb))
    disp = dict(rA_cln_mean=float(np.mean(rA_cln)),
                rB_mean=float(np.mean(rB)),
                note="干净节点人均被移除的 homog 边数（A|clean vs B 全体）")
    return res, disp


NCLS_G = None


def main():
    global LABELS_G, NCLS_G
    edges, labels = load()
    n = len(labels)
    n_cls = len(np.unique(labels))
    LABELS_G, NCLS_G = labels, n_cls
    flat = edge_flat_index(edges)
    a_star = np.array([1.0 if len(set(labels[u] for u in e)) == 1 else 0.0
                       for e in edges])
    B_clean = build_op(n, flat, a_star)
    h = 1.0 / float(abs(eigsh(B_clean, k=1, which="LM",
                              return_eigenvectors=False)[0]))
    Z0 = np.zeros((n, n_cls))
    Z0[np.arange(n), labels] = 1.0
    rng = np.random.default_rng(SEED)

    print("=== E1: 单边 toggle 灵敏度（margin 对边增删的响应） ===")
    tog, shape = e1_toggles(n, flat, edges, a_star, B_clean, Z0, h, rng)
    for t in DEPTHS:
        r, a = tog[t]["rem"], tog[t]["add"]
        print(f"  T={t:>2}: 移除 homog 边 Δm̄={np.mean(r):+.5f}"
              f" (sd {np.std(r):.5f}, >0 占比 {np.mean(np.array(r) > 0):.2f});"
              f" 加入 heterog 边 Δm̄={np.mean(a):+.5f}"
              f" (sd {np.std(a):.5f})")
    print("  多边移除形状 Δm̄(r)（检测凹凸/线性叠加）:")
    for t in DEPTHS:
        row = "  ".join(f"r={r}:{v:+.4f}" for r, v in shape[t].items())
        print(f"    T={t:>2}: {row}")

    print("\n=== E2: 精确分解 ===")
    results = {"toggles": {str(t): {"rem_mean": float(np.mean(tog[t]["rem"])),
                                    "add_mean": float(np.mean(tog[t]["add"]))}
                           for t in DEPTHS},
               "shape": {str(t): shape[t] for t in DEPTHS},
               "decomp": {}}
    for p in P_LIST:
        rng = np.random.default_rng(SEED + int(p * 100))
        res, disp = e2_decomposition(n, flat, edges, a_star, Z0, h, p, rng)
        print(f"\n--- p={p} ---  {disp['note']}: "
              f"A|clean {disp['rA_cln_mean']:.2f} vs B {disp['rB_mean']:.2f}")
        for t in DEPTHS:
            r = res[t]
            print(f"  T={t:>2}: 腐蚀节点 m={r['E_corrupted']:.4f} "
                  f"干净 m={r['E_clean']:.4f} B m={r['E_B']:.4f} | "
                  f"隔离项 {r['isolation_gain']:+.5f} "
                  f"干净项 {r['clean_contrast']:+.5f} "
                  f"合计 {r['total_pred']:+.5f} (实测 {r['measured']:+.5f})")
        results["decomp"][f"p={p}"] = {str(t): res[t] for t in DEPTHS}
        results["decomp"][f"p={p}"]["dispersion"] = disp

    with open("results/criterion_decompose.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print("\n已写出 results/criterion_decompose.json")


if __name__ == "__main__":
    main()
