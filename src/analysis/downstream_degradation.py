"""传导检验：中心化谱放大 → 下游学习退化。

问题：我们知道 ‖ΔB_A − M₀‖₂ > ‖ΔB_B − M₀‖₂（中心化谱分离），
但这不一定意味着 A 的错误导致更差的下游性能。传导链是：

  算子误差 → 谱扰动 → 表示扰动 → 分类退化

实验设计：
  固定超图和真标签。用残差滤波 Z_{t+1} = (I − hB) Z_t 传播。
  比较三种情形的下游性能：
    Clean: 用 B(a*) 传播 → 基线
    A: 用 B(â_A) 传播（节点驱动错误）
    B: 用 B(â_B) 传播（独立错误，匹配边际率）
  
  测量：
    1. 表示误差 ‖Ẑ_T − Z_T‖_F（与理论的传播界直接对应）
    2. 下游分类精度（在传播后的表示上做最近类中心分类）
    3. 类信号 Rayleigh 商（类间分离度）

  如果 A 的退化一致大于 B，传导成立。
"""

from __future__ import annotations

import json
import pickle
import sys

import numpy as np
import scipy.sparse as sp

sys.stdout.reconfigure(encoding="utf-8")

CORA = "data/cora-cc"
N_REALIZATIONS = 50
DEPTH_LIST = (1, 2, 5, 10)


def load():
    with open(f"{CORA}/features.pickle", "rb") as f:
        feats = pickle.load(f)
    with open(f"{CORA}/hypergraph.pickle", "rb") as f:
        hg = pickle.load(f)
    with open(f"{CORA}/labels.pickle", "rb") as f:
        labels = np.array(pickle.load(f))
    edges = [sorted(int(u) for u in s) for s in hg.values()]
    return feats, edges, labels


def build_op(n, edges, weights):
    """Σ w_e J_e / r_e 的稠密矩阵。"""
    M = np.zeros((n, n))
    for e, w in zip(edges, weights):
        if w == 0:
            continue
        k = len(e)
        M[np.ix_(e, e)] += w / k
    return M


def propagate(B, Z0, h, T):
    """残差滤波：Z_{t+1} = (I − hB) Z_t。"""
    n = B.shape[0]
    I = np.eye(n)
    Z = Z0.copy()
    for _ in range(T):
        Z = (I - h * B) @ Z
    return Z


def downstream_metrics(Z, labels, n_cls):
    """在表示上做最近类中心分类，返回精度和类信号 Rayleigh 商。"""
    n = Z.shape[0]
    # 类中心
    centers = np.zeros((n_cls, Z.shape[1]))
    for c in range(n_cls):
        mask = labels == c
        if mask.sum() > 0:
            centers[c] = Z[mask].mean(axis=0)
    # 最近类中心分类
    dists = ((Z[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    pred = np.argmin(dists, axis=1)
    acc = float(np.mean(pred == labels))
    # 类信号 Rayleigh 商（类间方差 / 总方差）
    overall_mean = Z.mean(axis=0)
    between_var = 0.0
    within_var = 0.0
    for c in range(n_cls):
        mask = labels == c
        if mask.sum() > 0:
            diff = centers[c] - overall_mean
            between_var += mask.sum() * np.sum(diff ** 2)
            within_var += np.sum((Z[mask] - centers[c]) ** 2)
    signal_ratio = between_var / max(between_var + within_var, 1e-12)
    return acc, float(signal_ratio)


def experiment():
    feats, edges, labels = load()
    n = len(labels)
    n_cls = len(np.unique(labels))
    m = len(edges)
    a_star = np.array([1.0 if len(set(labels[u] for u in e)) == 1 else 0.0
                       for e in edges])
    sgn = np.where(a_star == 1, -1.0, 1.0)

    # 真实算子和传播参数
    B_clean = build_op(n, edges, a_star)
    eig = np.linalg.eigvalsh(B_clean)
    lam_max = float(abs(eig[-1]))
    h = 1.0 / lam_max  # 保证收缩

    # 特征作为初始表示（PCA 降维加速）
    X = np.asarray(feats.todense()) if sp.issparse(feats) else feats
    # 中心化 + 截断 SVD
    X_c = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
    Z0 = U[:, :50] * S[:50]  # 前 50 个主成分

    # Clean 传播（基线）
    print(f"Clean operator: λ_max = {lam_max:.3f}, h = {h:.4f}")
    results = {}
    for T in DEPTH_LIST:
        Z_clean = propagate(B_clean, Z0, h, T)
        acc_c, sig_c = downstream_metrics(Z_clean, labels, n_cls)
        results.setdefault(T, {})["clean"] = dict(acc=acc_c, signal=sig_c)
        print(f"  T={T}: acc={acc_c:.3f}, signal={sig_c:.3f}")

    # 每种错误率 p 的 A/B 对比
    for p in (0.05, 0.10, 0.20):
        # A 的边际率（精确）
        q_h = 3 * p * (1 - p)     # hom 边
        q_m = p * (1 - p)          # mixed 边
        print(f"\n--- p = {p:.2f} (q_hom={q_h:.4f}, q_mix={q_m:.4f}) ---")
        rng = np.random.default_rng(42 + int(p * 100))
        for T in DEPTH_LIST:
            rep_errs_A, rep_errs_B = [], []
            accs_A, accs_B = [], []
            sigs_A, sigs_B = [], []
            for _ in range(N_REALIZATIONS):
                # A: 节点驱动错误
                W = rng.random(n) < p
                pred_A = labels * (1 - 2 * W.astype(int))
                a_hat_A = np.array(
                    [1.0 if len(set(pred_A[u] for u in e)) == 1 else 0.0
                     for e in edges])
                dA = a_hat_A - a_star
                B_A = build_op(n, edges, a_star + dA)
                Z_A = propagate(B_A, Z0, h, T)
                # B: 独立超边翻转（匹配边际率）
                dB = np.zeros(m)
                for i, e in enumerate(edges):
                    q = q_h if a_star[i] == 1 else q_m
                    if rng.random() < q:
                        dB[i] = sgn[i]
                B_B = build_op(n, edges, a_star + dB)
                Z_B = propagate(B_B, Z0, h, T)
                # 测量
                Z_clean_T = results[T]["clean"]
                Z_clean_T_full = propagate(B_clean, Z0, h, T)
                err_A = float(np.linalg.norm(Z_A - Z_clean_T_full, "fro"))
                err_B = float(np.linalg.norm(Z_B - Z_clean_T_full, "fro"))
                rep_errs_A.append(err_A)
                rep_errs_B.append(err_B)
                acc_A, sig_A = downstream_metrics(Z_A, labels, n_cls)
                acc_B, sig_B = downstream_metrics(Z_B, labels, n_cls)
                accs_A.append(acc_A)
                accs_B.append(acc_B)
                sigs_A.append(sig_A)
                sigs_B.append(sig_B)

            # Bootstrap CI for 差值 A - B
            rng_bs = np.random.default_rng(99)
            ia = rng_bs.integers(0, N_REALIZATIONS, (2000, N_REALIZATIONS))
            ib = rng_bs.integers(0, N_REALIZATIONS, (2000, N_REALIZATIONS))
            acc_diff = np.array(accs_A)[ia].mean(axis=1) - \
                np.array(accs_B)[ib].mean(axis=1)
            err_diff = np.array(rep_errs_A)[ia].mean(axis=1) - \
                np.array(rep_errs_B)[ib].mean(axis=1)
            lo_acc, hi_acc = np.quantile(acc_diff, [0.025, 0.975])
            lo_err, hi_err = np.quantile(err_diff, [0.025, 0.975])

            results[T][f"p={p}"] = dict(
                rep_err_A=float(np.mean(rep_errs_A)),
                rep_err_B=float(np.mean(rep_errs_B)),
                rep_err_ratio=float(np.mean(rep_errs_A) /
                                    max(np.mean(rep_errs_B), 1e-12)),
                rep_err_diff_ci=[float(lo_err), float(hi_err)],
                acc_A=float(np.mean(accs_A)),
                acc_B=float(np.mean(accs_B)),
                acc_diff=float(np.mean(accs_A) - np.mean(accs_B)),
                acc_diff_ci=[float(lo_acc), float(hi_acc)],
                sig_A=float(np.mean(sigs_A)),
                sig_B=float(np.mean(sigs_B)))
            r = results[T][f"p={p}"]
            print(f"  T={T}: rep_err A/B = {r['rep_err_A']:.3f}/"
                  f"{r['rep_err_B']:.3f} (ratio {r['rep_err_ratio']:.2f}), "
                  f"acc A={r['acc_A']:.3f} B={r['acc_B']:.3f} "
                  f"(diff {r['acc_diff']:+.4f} CI "
                  f"[{lo_acc:+.4f},{hi_acc:+.4f}])")

    with open("results/downstream_degradation.json", "w",
              encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print("\n已写出 results/downstream_degradation.json")
    return results


if __name__ == "__main__":
    experiment()
