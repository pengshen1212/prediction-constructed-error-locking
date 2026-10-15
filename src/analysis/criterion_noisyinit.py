"""超越 one-hot readout：噪声初始化下的符号翻转预测（E3）。

判据预测：隔离保留的是初始状态。one-hot 初始化（初始即最优）时
隔离增益 ≥ 0 → A margin 更高。当初始化本身有噪声、传播负责去噪时，
被隔离的节点无法被纠正 → 隔离增益变负 → 预测 A margin 更低（翻转）。

三种初始化（p=0.10 图噪声，margin 对真标签）：
  clean : one-hot 真标签（对照，应复现 A > B）
  indep : Z0 = one-hot 标签被独立翻转 20%（去噪区，预测 A ≤ B）
  couple: Z0 = one-hot 生成 A 图的同一批腐蚀预测（半监督现实场景，
          被腐蚀节点既错又隔离，预测 A ≪ B）
B 图与 A 图共用同一 Z0（算子级比较）。
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
P = 0.10
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


def margins_acc(B, Z0, h, checkpoints, labels):
    Z = Z0.copy()
    n = Z.shape[0]
    out = {}
    nxt = sorted(checkpoints)
    for t in range(1, max(checkpoints) + 1):
        Z = Z - h * (B @ Z)
        if nxt and t == nxt[0]:
            nxt.pop(0)
            own = Z[np.arange(n), labels]
            Zo = Z.copy()
            Zo[np.arange(n), labels] = -np.inf
            margin = own - Zo.max(axis=1)
            pred = np.argmax(Z, axis=1)
            acc = float(np.mean(pred == labels))
            # 被腐蚀节点（对 couple 模式）单独统计
            out[t] = (float(margin.mean()), acc)
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
    h = 1.0 / float(abs(eigsh(B_clean, k=1, which="LM",
                              return_eigenvectors=False)[0]))
    Z0_clean = np.zeros((n, n_cls))
    Z0_clean[np.arange(n), labels] = 1.0

    w0 = expected_da(edges, labels, n_cls, P)
    q_rate, sgn = np.abs(w0), np.sign(w0)
    rng = np.random.default_rng(SEED + int(P * 100))

    modes = ("clean", "indep", "couple")
    mA = {mo: {t: [] for t in DEPTHS} for mo in modes}
    mB = {mo: {t: [] for t in DEPTHS} for mo in modes}
    aA = {mo: {t: [] for t in DEPTHS} for mo in modes}
    aB = {mo: {t: [] for t in DEPTHS} for mo in modes}
    mA_cor = {t: [] for t in DEPTHS}   # couple 模式腐蚀节点 margin (A)
    mB_cor = {t: [] for t in DEPTHS}   # couple 模式腐蚀节点 margin (B)
    mA_cln = {t: [] for t in DEPTHS}   # couple 模式干净节点 margin (A)
    mB_cln = {t: [] for t in DEPTHS}   # couple 模式干净节点 margin (B)

    for _ in range(N_REAL):
        pred = corrupt_labels(labels, n_cls, P, rng)
        corrupted = pred != labels
        wA = np.array([1.0 if len(set(pred[u] for u in e)) == 1 else 0.0
                       for e in edges])
        flip = rng.random(m) < q_rate
        wB = a_star + np.where(flip, sgn, 0.0)
        BA, BB = build_op(n, flat, wA), build_op(n, flat, wB)

        inits = {
            "clean": Z0_clean,
            "indep": (lambda: np.eye(n_cls)[corrupt_labels(
                labels, n_cls, 0.20, rng)])(),
            "couple": np.eye(n_cls)[pred],
        }
        for mo, Z0 in inits.items():
            for name, B in (("A", BA), ("B", BB)):
                rec = margins_acc(B, Z0, h, DEPTHS, labels)
                dm = mA[mo] if name == "A" else mB[mo]
                da = aA[mo] if name == "A" else aB[mo]
                for t in DEPTHS:
                    dm[t].append(rec[t][0])
                    da[t].append(rec[t][1])
        # couple 模式：A/B 下腐蚀与干净节点分别的 margin
        for name, B in (("A", BA), ("B", BB)):
            Z = np.eye(n_cls)[pred]
            for t in range(1, max(DEPTHS) + 1):
                Z = Z - h * (B @ Z)
                if t in DEPTHS:
                    own = Z[np.arange(n), labels]
                    Zo = Z.copy(); Zo[np.arange(n), labels] = -np.inf
                    marg = own - Zo.max(axis=1)
                    dst = mA_cor if name == "A" else mB_cor
                    dst2 = mA_cln if name == "A" else mB_cln
                    dst[t].append(float(marg[corrupted].mean()))
                    dst2[t].append(float(marg[~corrupted].mean()))

    rng_bs = np.random.default_rng(99)
    ia = rng_bs.integers(0, N_REAL, (2000, N_REAL))
    ib = rng_bs.integers(0, N_REAL, (2000, N_REAL))
    results = {}
    print(f"p={P}, {N_REAL} 实现；判据预测：clean A>B；indep A≤B；couple A≪B")
    for mo in modes:
        results[mo] = {}
        print(f"\n--- init={mo} ---")
        for t in DEPTHS:
            x, y = np.array(mA[mo][t]), np.array(mB[mo][t])
            xa = np.array(aA[mo][t])
            ya = np.array(aB[mo][t])
            d = x[ia].mean(1) - y[ib].mean(1)
            lo, hi = np.quantile(d, [0.025, 0.975])
            da = xa[ia].mean(1) - ya[ib].mean(1)
            loa, hia = np.quantile(da, [0.025, 0.975])
            r = dict(margin_A=float(x.mean()), margin_B=float(y.mean()),
                     diff=float(x.mean() - y.mean()),
                     ci=[float(lo), float(hi)],
                     acc_A=float(np.mean(aA[mo][t])),
                     acc_B=float(np.mean(aB[mo][t])),
                     acc_diff=float(xa.mean() - ya.mean()),
                     acc_ci=[float(loa), float(hia)])
            results[mo][f"T={t}"] = r
            print(f"  T={t:>2}: margin A={r['margin_A']:.4f} "
                  f"B={r['margin_B']:.4f} diff={r['diff']:+.5f} "
                  f"CI[{lo:+.5f},{hi:+.5f}]; "
                  f"acc A={r['acc_A']:.3f} B={r['acc_B']:.3f}")
    print("\ncouple 模式分解（腐蚀 vs 干净节点，A vs B）:")
    for t in DEPTHS:
        print(f"  T={t:>2}: 腐蚀节点 A={np.mean(mA_cor[t]):+.4f} "
              f"B={np.mean(mB_cor[t]):+.4f}; "
              f"干净节点 A={np.mean(mA_cln[t]):+.4f} "
              f"B={np.mean(mB_cln[t]):+.4f}")
    results["couple_decomp"] = {
        str(t): dict(A_cor=float(np.mean(mA_cor[t])),
                     B_cor=float(np.mean(mB_cor[t])),
                     A_cln=float(np.mean(mA_cln[t])),
                     B_cln=float(np.mean(mB_cln[t])))
        for t in DEPTHS}

    with open("results/criterion_noisyinit.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print("\n已写出 results/criterion_noisyinit.json")


if __name__ == "__main__":
    main()
