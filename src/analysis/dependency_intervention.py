"""依赖强度干预实验（评审第七轮首选）：保持逐边边际不变，随机分组
连续削弱共享节点依赖，检验 (i) 二阶仿射律（可推导）与 (ii) 中心化
谱范数的单调性（纯实验）。

设计：固定合成结构（两类 ±1、三均匀、同配率 0.7）与对称腐蚀信道。
超边随机分到 K 组，每组一套独立节点预测，组内超边复用该组共享预测。
K=1 恢复 A（全部复用）；K=m 时 â_e 为独立 Bernoulli(ph_e)，与 B
（匹配边际独立翻转）在任何 {â_e} 泛函上同律。图、标签、噪声率、
逐边规则、线性算子固定，仅分组与预测抽样变化。

理论（精确，随机分组平均）：
  E‖ΔB − EΔB‖_F² = V0 + C1/K
  V0 = Σ_e ph_e(1−ph_e)（独立通道截距）
  C1 = Σ_{e≠f, e∩f≠∅} Cov₁(δ_e,δ_f)·|e∩f|²/(k_e k_f)
  Cov₁ = P_joint(e,f) − ph_e·ph_f，P_joint = Σ_{c=±1} Π_{v∈e∪f}
  P(pred_v=c)（共享节点去重；不相交边对 Cov₁=0——相关全部经由
  共享节点，这正是主问题的定量形式）。谱范数随 K 无推导，实验检验。

试点门槛（评审）：3 结构种子 × p∈{0.1,0.2} × K∈{1,2,4,8,16,32,m}
× B 参考 × 300 实现；判据 = 仿射 R²、截距/斜率/端点吻合、谱单调。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, ArpackNoConvergence

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")
from rayleigh_influence_bound import instance

SEEDS = (700, 701, 702)
P_LIST = (0.10, 0.20)
K_GRID = (1, 2, 4, 8, 16, 32, "m")
N_REAL = 300


def edge_flat(edges):
    rows, cols, owner, invk = [], [], [], []
    for i, e in enumerate(edges):
        k = len(e)
        rows.append(np.repeat(e, k))
        cols.append(np.tile(e, k))
        owner.append(np.full(k * k, i))
        invk.append(np.full(k * k, 1.0 / k))
    return (np.concatenate(rows), np.concatenate(cols),
            np.concatenate(owner), np.concatenate(invk))


def build_sparse(n, flat, weights):
    """稀疏对称算子（稠密 SVD 在本机 340ms/次是瓶颈；eigsh 毫秒级）。"""
    rows, cols, owner, invk = flat
    vals = invk * weights[owner]
    keep = vals != 0.0
    return sp.csr_matrix((vals[keep], (rows[keep], cols[keep])),
                         shape=(n, n))


def spec_norm(M):
    """对称矩阵谱范数 = 最大 |特征值|。"""
    try:
        lam = eigsh(M, k=1, which="LM", return_eigenvectors=False,
                    tol=1e-6)[0]
        return float(abs(lam))
    except ArpackNoConvergence:
        return float(np.linalg.norm(M.toarray(), 2))


def theory_coeffs(labels, edges, a_star, p, C=None):
    """V0、C1 闭式（任意类数，计数向量形式）；仅共享节点边对进入 C1。"""
    m = len(edges)
    if C is None:
        C = int(labels.max() + 1)
    rho = p / (C - 1)
    labs_e = [labels[np.asarray(e)] for e in edges]
    # ph_e = Σ_c (1−p)^{n_c} ρ^{k−n_c}
    ph = np.zeros(m)
    for i, labs in enumerate(labs_e):
        cnt = np.bincount(labs, minlength=C).astype(float)
        ph[i] = float(np.sum(np.power(1 - p, cnt)
                             * np.power(rho, len(labs) - cnt)))
    V0 = float(np.sum(ph * (1 - ph)))
    k = np.array([len(e) for e in edges])
    from collections import defaultdict
    by_node = defaultdict(list)
    for i, e in enumerate(edges):
        for v in e:
            by_node[v].append(i)
    pairs = set()
    for lst in by_node.values():
        for a in range(len(lst)):
            for b in range(a + 1, len(lst)):
                i, j = lst[a], lst[b]
                if i != j:
                    pairs.add((min(i, j), max(i, j)))
    C1 = 0.0
    for i, j in pairs:
        ei = set(int(v) for v in edges[i])
        inter_nodes = ei & set(int(v) for v in edges[j])
        inter = len(inter_nodes)
        # 并集计数 = n_i + n_j − 共享节点计数
        cnt = (np.bincount(labs_e[i], minlength=C).astype(float)
               + np.bincount(labs_e[j], minlength=C).astype(float)
               - np.bincount(labels[np.asarray(list(inter_nodes))],
                             minlength=C).astype(float))
        U = int(cnt.sum())
        pj = float(np.sum(np.power(1 - p, cnt) * np.power(rho, U - cnt)))
        C1 += 2.0 * (pj - ph[i] * ph[j]) * inter * inter / (k[i] * k[j])
    return V0, C1, ph


def corrupt_C(labels, C, p, rng):
    """对称 C 类腐蚀（与论文信道一致）。"""
    mask = rng.random(len(labels)) < p
    j = rng.integers(0, C - 1, size=len(labels))
    return np.where(mask, j + (j >= labels), labels)


def run(seed, p, data="synthetic", n_real=N_REAL):
    if data == "synthetic":
        labels, edges, a_star, wproto, phat, node_edges = instance(
            seed, 400, 600, 8, 1.0, 0.7)
        C = 2
        flat = edge_flat(edges)
    else:
        import sign_predict as spx
        ds = spx.Dataset(data)
        labels, edges, a_star, C = ds.labels, ds.edges, ds.a_star, ds.C
        flat = (ds.rows, ds.cols, ds.owner, ds.invk)
    n, m = len(labels), len(edges)
    V0, C1, ph = theory_coeffs(labels, edges, a_star, p, C)
    Bbar = build_sparse(n, flat, ph)
    out = dict(V0=V0, C1=C1, curves={})
    rng = np.random.default_rng(seed + 31000)
    edge_arrs = [np.asarray(e) for e in edges]
    def measure(ah):
        Dc = build_sparse(n, flat, ah) - Bbar
        return (float(sp.linalg.norm(Dc) ** 2), spec_norm(Dc))

    def grouped_ah(Keff):
        preds = [corrupt_C(labels, C, p, rng) for _ in range(Keff)]
        grp = rng.integers(0, Keff, m)
        ah = np.zeros(m)
        for g in range(Keff):
            sel = np.flatnonzero(grp == g)
            if len(sel) == 0:
                continue
            prd = preds[g]
            keq = np.array([
                1.0 if len(set(prd[edge_arrs[i]].tolist())) == 1
                else 0.0 for i in sel])
            ah[sel] = keq
        return ah

    for K in K_GRID:
        Keff = m if K == "m" else K
        fc2, fcS = [], []
        for _ in range(n_real):
            ah = ((rng.random(m) < ph).astype(float) if K == "m"
                  else grouped_ah(Keff))
            a, b = measure(ah)
            fc2.append(a)
            fcS.append(b)
        out["curves"][str(K)] = dict(
            fc2=float(np.mean(fc2)),
            fc2_ci=float(1.96 * np.std(fc2) / np.sqrt(n_real)),
            fcS=float(np.mean(fcS)),
            fcS_ci=float(1.96 * np.std(fcS) / np.sqrt(n_real)))
    w0 = ph - a_star
    q, sg = np.abs(w0), np.sign(w0)
    fc2, fcS = [], []
    for _ in range(n_real):
        flip = rng.random(m) < q
        wB = a_star + np.where(flip, sg, 0.0)
        a, b = measure(wB)
        fc2.append(a)
        fcS.append(b)
    out["curves"]["B"] = dict(
        fc2=float(np.mean(fc2)),
        fc2_ci=float(1.96 * np.std(fc2) / np.sqrt(n_real)),
        fcS=float(np.mean(fcS)),
        fcS_ci=float(1.96 * np.std(fcS) / np.sqrt(n_real)))
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="synthetic",
                    choices=["synthetic", "cora", "dblp"])
    ap.add_argument("--n-real", type=int, default=N_REAL)
    ap.add_argument("--seed", type=int, default=710,
                    help="MC 种子；synthetic 模式兼作结构种子基")
    args = ap.parse_args()
    if args.data == "synthetic":
        seeds = SEEDS
        n_real = args.n_real
    else:
        seeds = (args.seed,)
        n_real = args.n_real
    results = {}
    for seed in seeds:
        for p in P_LIST:
            results[f"seed{seed}|p{p}"] = run(seed, p, args.data, n_real)
    print("\n=== 仿射律与端点核对（中心化 F² vs 1/K）===")
    summary = {}
    for key, r in results.items():
        xs = np.array([1.0 / (k if k != "m" else 600) for k in K_GRID])
        ys = np.array([r["curves"][str(k)]["fc2"] for k in K_GRID])
        A = np.vstack([np.ones_like(xs), xs]).T
        coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
        pred = A @ coef
        r2 = 1 - float(np.sum((ys - pred) ** 2)) / float(
            np.sum((ys - ys.mean()) ** 2))
        km = r["curves"]["m"]["fc2"]
        b = r["curves"]["B"]["fc2"]
        summary[key] = dict(V0=r["V0"], C1=r["C1"], intercept=coef[0],
                            slope=coef[1], r2=r2, K1=ys[0], Km=km,
                            B=b)
        print(f"{key}: 截距 {coef[0]:8.2f} (V0 {r['V0']:8.2f})  斜率 "
              f"{coef[1]:9.2f} (C1 {r['C1']:9.2f})  R²={r2:.4f}  "
              f"K=1: {ys[0]:8.2f} vs V0+C1 {r['V0']+r['C1']:8.2f}  "
              f"K=m: {km:6.2f} ≈ B: {b:6.2f} ≈ V0: {r['V0']:6.2f}")
    print("\n=== 中心化谱范数随 K（实验量）===")
    for key, r in results.items():
        v = [r["curves"][str(k)]["fcS"] for k in K_GRID]
        ci = [r["curves"][str(k)]["fcS_ci"] for k in K_GRID]
        mono = all(v[i] >= v[i + 1] - 3 * ci[i + 1]
                   for i in range(len(v) - 1))
        print(f"{key}: " + " ".join(f"{x:5.2f}" for x in v)
              + f" | B {r['curves']['B']['fcS']:5.2f} | "
              f"单调(3CI)={mono}")
    Path("results").mkdir(exist_ok=True)
    suffix = "" if args.data == "synthetic" else f"_{args.data}"
    Path(f"results/depint{suffix}.json").write_text(
        json.dumps({"results": results, "summary": summary},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n→ results/depint{suffix}.json")


if __name__ == "__main__":
    main()
