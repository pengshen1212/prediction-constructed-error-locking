"""方向 2 最后一道门槛：按边大小+标签组成分层校准 + 期望算子检查 + 稀疏化。

用户冻结协议（2026-10-01）：
  - 校准与评估用不重叠划分种子。
  - q_e 按边大小 k × 标签组成（类数比例）分层估计，非全局平均。
  - 报告各层匹配误差。
  - 检查期望误差算子是否匹配（校准组 vs 评估组的 E[ΔB] Frobenius 距离）。
  - 跨评估划分的 A/B/C 置信区间。
  - DBLP 特征计算保持稀疏。
  - 若通过 → 真实预测器入正文；不通过 → 放局限性/附录。
"""

from __future__ import annotations

import json
import pickle
import sys

import numpy as np
import scipy.sparse as sp

sys.stdout.reconfigure(encoding="utf-8")

CORA = "data/cora-cc"
DBLP = "data/dblp-ca"

N_CAL = 5          # 校准种子（估计 q_e）
N_EVAL = 5         # 评估种子（A）
N_BC = 50          # B/C 重抽
BASE_CAL = 100
BASE_EVAL = 200
BASE_BC = 300


def load(data_dir):
    with open(f"{data_dir}/features.pickle", "rb") as f:
        feats = pickle.load(f)
    with open(f"{data_dir}/hypergraph.pickle", "rb") as f:
        hg = pickle.load(f)
    with open(f"{data_dir}/labels.pickle", "rb") as f:
        labels = np.array(pickle.load(f))
    edges = [sorted(s) for s in hg.values()]
    return feats, edges, labels


def build_op(n, edges, weights):
    rows, cols, vals = [], [], []
    for e, w in zip(edges, weights):
        if not w:
            continue
        k = len(e)
        for a in e:
            for b in e:
                rows.append(a); cols.append(b); vals.append(w / k)
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


def snorm(B):
    if B.nnz == 0:
        return 0.0
    from scipy.sparse.linalg import eigsh, ArpackNoConvergence
    try:
        la = eigsh(B, k=1, which='LA', return_eigenvectors=False,
                   tol=1e-5, maxiter=5000)[0]
        li = eigsh(B, k=1, which='SA', return_eigenvectors=False,
                   tol=1e-5, maxiter=5000)[0]
        return float(max(abs(la), abs(li)))
    except ArpackNoConvergence:
        raise RuntimeError(f"Lanczos no convergence on nnz={B.nnz}")


def all_hom(pred, edges):
    return np.array([1.0 if len(set(pred[i] for i in e)) == 1 else 0.0
                     for e in edges])


def crossfit_predict_sparse(feats, labels, seed, n_folds=5):
    """K-fold 交叉拟合，全程稀疏（原型用稀疏均值，距离用分块）。"""
    n = len(labels)
    n_cls = len(np.unique(labels))
    rng = np.random.default_rng(seed)
    fold_idx = rng.permutation(n)
    folds = np.array_split(fold_idx, n_folds)
    pred = np.zeros(n, dtype=int)
    for f in range(n_folds):
        test = folds[f]
        train = np.concatenate([folds[g] for g in range(n_folds) if g != f])
        # 稀疏类均值原型
        protos = np.zeros((n_cls, feats.shape[1]))
        for c in range(n_cls):
            mask = labels[train] == c
            if mask.sum() > 0:
                protos[c] = np.asarray(
                    feats[train][mask].mean(axis=0)).ravel()
        # 分块预测（避免 n×n_cls×d 稠密）
        blk = feats[test]              # 稀疏
        blk_d = blk.toarray() if sp.issparse(blk) else blk
        # 距离 = ‖x‖² - 2⟨x,p⟩ + ‖p‖²
        x_sq = (blk_d ** 2).sum(axis=1)
        p_sq = (protos ** 2).sum(axis=1)
        cross = blk_d @ protos.T       # (|test|, n_cls)
        dists = x_sq[:, None] - 2 * cross + p_sq[None, :]
        pred[test] = np.argmin(dists, axis=1)
    return pred


def edge_stratum(edges, labels, a_star):
    """每条边的分层标签：(k, 类标签元组)。"""
    strata = []
    for i, e in enumerate(edges):
        k = len(e)
        labs = tuple(sorted(labels[j] for j in e))
        strata.append((k, labs))
    return strata


def stratified_q(strata, a_star, a_hat):
    """按 (k, 标签组成) 分层估计边错误率。返回 {stratum: (q, count)}。"""
    d = a_hat - a_star
    q_map = {}
    for i, s in enumerate(strata):
        if s not in q_map:
            q_map[s] = [0.0, 0]    # [sum_err, count]
        q_map[s][1] += 1
        if d[i] != 0:
            q_map[s][0] += 1
    return {s: (v[0] / max(v[1], 1), v[1]) for s, v in q_map.items()}


def run(name, feats, edges_all, labels, tri_only=False):
    n = len(labels)
    edges = [e for e in edges_all if len(e) == 3] if tri_only else edges_all
    a_star = all_hom(labels, edges)
    strata = edge_stratum(edges, labels, a_star)
    tag = "三元" if tri_only else "全边"

    # ---- 校准阶段（估计分层 q）----
    cal_preds = [crossfit_predict_sparse(feats, labels, BASE_CAL + i)
                 for i in range(N_CAL)]
    # 聚合多组校准预测的分层错误率
    q_strat = {}
    for pred in cal_preds:
        a_hat = all_hom(pred, edges)
        qs = stratified_q(strata, a_star, a_hat)
        for s, (q, cnt) in qs.items():
            if s not in q_strat:
                q_strat[s] = [0.0, 0]
            q_strat[s][0] += q * cnt
            q_strat[s][1] += cnt
    q_strat = {s: v[0] / max(v[1], 1) for s, v in q_strat.items()}

    # ---- 评估阶段（不重叠种子）----
    print(f"\n{'='*84}")
    print(f"[{name}-{tag}] 分层校准 + 交叉拟合评估（校准 {N_CAL} 种子 "
          f"100–{BASE_CAL+N_CAL-1}，评估 {N_EVAL} 种子 "
          f"{BASE_EVAL}–{BASE_EVAL+N_EVAL-1}，B/C × {N_BC}）")
    print(f"{'='*84}")
    print(f"  分层数：{len(q_strat)}")

    # 匹配误差报告（前 10 个最大层）
    sorted_strata = sorted(q_strat.items(), key=lambda x: -x[1] if isinstance(x[1], float) else 0)
    eval_preds = [crossfit_predict_sparse(feats, labels, BASE_EVAL + i)
                  for i in range(N_EVAL)]
    print(f"\n  分层匹配质量（评估组实际 vs 校准估计，前 10 大层）：")
    print(f"  {'层 (k, 标签)':>30} | {'n':>5} {'实际':>6} {'估计':>6} {'差':>6}")

    # 期望算子匹配检查
    def expected_error_op(a_star, edges, strata, q_map):
        """E[ΔB] = Σ_e (-1 if hom else +1)·q_e·g_e g_e^T。"""
        weights = []
        for i, e in enumerate(edges):
            s = strata[i]
            q = q_map.get(s, 0.0)
            sgn = -1.0 if a_star[i] == 1.0 else 1.0
            weights.append(sgn * q)
        return build_op(n, edges, weights)

    # A 模型的期望误差算子（跨评估组平均）
    a_hats_eval = [all_hom(pred, edges) for pred in eval_preds]
    d_evals = [ah - a_star for ah in a_hats_eval]
    d_mean = np.mean(d_evals, axis=0)
    E_A = build_op(n, edges, d_mean)

    # B/C 的期望误差算子（用分层 q）
    E_BC = expected_error_op(a_star, edges, strata, q_strat)

    # 期望算子 Frobenius 距离
    E_diff = E_A - E_BC
    fro_diff = float(np.sqrt(E_diff.multiply(E_diff).sum()))
    E_A_norm = float(np.sqrt(E_A.multiply(E_A).sum()))
    print(f"\n  期望误差算子匹配：‖E[ΔB_A] − E[ΔB_BC]‖_F = {fro_diff:.4f}  "
          f"（‖E[ΔB_A]‖_F = {E_A_norm:.4f}，比值 = {fro_diff/E_A_norm:.2%}）")

    # 分层匹配（用第一评估组）
    a_hat0 = a_hats_eval[0]
    d0 = a_hat0 - a_star
    qs_eval = stratified_q(strata, a_star, a_hat0)
    # 按层大小排序取前 10
    top_strata = sorted(qs_eval.items(), key=lambda x: -x[1][1])[:10]
    for s, (q_actual, cnt) in top_strata:
        q_est = q_strat.get(s, 0.0)
        print(f"  {str(s):>30} | {cnt:>5} {q_actual:>6.3f} {q_est:>6.3f} "
              f"{q_actual - q_est:>+6.3f}")

    # ---- A 评估 ----
    bAs = []
    for pred in eval_preds:
        dA = all_hom(pred, edges) - a_star
        bAs.append(snorm(build_op(n, edges, dA)))

    # ---- B/C 用分层 q ----
    rng = np.random.default_rng(BASE_BC)
    bBs, bCs = [], []
    for _ in range(N_BC):
        dB = np.zeros(len(edges))
        for i in range(len(edges)):
            q = q_strat.get(strata[i], 0.0)
            if rng.random() < q:
                dB[i] = -1.0 if a_star[i] == 1.0 else 1.0
        bBs.append(snorm(build_op(n, edges, dB)))
        # C: 分层 q 独立位置
        ent = []
        for i, e in enumerate(edges):
            k = len(e)
            q = q_strat.get(strata[i], 0.0)
            sgn = -1 if a_star[i] == 1.0 else 1
            for a_ in range(k):
                for b_ in range(a_ + 1, k):
                    if rng.random() < q:
                        ent.append((e[a_], e[b_], k, sgn))
        from replication_v2 import clique_op
        bCs.append(snorm(clique_op(n, ent)) if ent else 0.0)

    # ---- 报告 ----
    bAs_a = np.array(bAs); bBs_a = np.array(bBs); bCs_a = np.array(bCs)
    ac = bAs_a.mean() / max(bCs_a.mean(), 1e-9)
    ab = bAs_a.mean() / max(bBs_a.mean(), 1e-9)
    # Bootstrap CI 对 β̄A/β̄C
    boot = np.random.default_rng(999)
    ac_boots = []
    for _ in range(2000):
        ia = boot.choice(len(bAs_a), len(bAs_a), replace=True)
        ic = boot.choice(len(bCs_a), len(bCs_a), replace=True)
        ac_boots.append(bAs_a[ia].mean() / max(bCs_a[ic].mean(), 1e-9))
    ci = np.percentile(ac_boots, [2.5, 97.5])
    print(f"\n  β̄A = {bAs_a.mean():.2f}  β̄B = {bBs_a.mean():.2f}  "
          f"β̄C = {bCs_a.mean():.2f}")
    print(f"  β̄A/β̄C = {ac:.2f}  [{ci[0]:.2f},{ci[1]:.2f}]")
    print(f"  β̄A/β̄B = {ab:.2f}")
    return dict(bA=float(bAs_a.mean()), bB=float(bBs_a.mean()),
                bC=float(bCs_a.mean()), ac=float(ac), ab=float(ab),
                ac_lo=float(ci[0]), ac_hi=float(ci[1]),
                E_match=fro_diff, E_norm=E_A_norm,
                n_strata=len(q_strat))


if __name__ == "__main__":
    results = {}
    for name, path in (("Cora", CORA), ("DBLP", DBLP)):
        feats, edges, labels = load(path)
        for tri in (False, True):
            key = f"{name}-{'三元' if tri else '全边'}"
            results[key] = run(name, feats, edges, labels, tri)
    with open("results/crossfit_v2.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print("\n已写出 results/crossfit_v2.json")
