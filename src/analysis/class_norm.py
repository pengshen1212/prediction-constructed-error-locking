"""类相关谱范数：定义、计算、验证预测能力。

定义：V_class = 中心化 one-hot 标签矩阵的左奇异向量（类子空间正交基）。
  ‖X‖_class = ‖V_class^T X V_class‖₂
  （X 在类子空间上的限制的谱范数）

预测假设：
  H1: 线性/深层时 ‖ΔB_A − M₀‖_class < ‖ΔB_B − M₀‖_class（A 类方向扰动更小
      → A margin 更高 → 下游更好），尽管 ‖·‖₂ 方向相反。
  H2: ReLU + 高噪声 + 浅层时 ‖ΔB_A − M₀‖_class ≥ ‖ΔB_B − M₀‖_class
      （方向翻转的解释）。
  H3: 类对齐度 α = ‖·‖_class / ‖·‖₂ 对 A 和 B 不同——A 的 top 特征向量
      偏离类子空间（局部化），B 的更均匀。

同时计算：传播后的类相关扰动
  ‖(I−hB)^{T−1} (ΔB) V_class‖₂（平滑对 A vs B 的不同衰减）
"""

from __future__ import annotations

import json
import pickle
import sys

import numpy as np
import scipy.sparse as sp

sys.stdout.reconfigure(encoding="utf-8")

CORA = "data/cora-cc"
N_REALIZATIONS = 100


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
    M = np.zeros((n, n))
    for e, w in zip(edges, weights):
        if w == 0:
            continue
        k = len(e)
        M[np.ix_(e, e)] += w / k
    return M


def class_subspace(labels, n_cls):
    """中心化 one-hot 的左奇异向量（类子空间基，n_cls−1 维）。"""
    n = len(labels)
    Y = np.zeros((n, n_cls))
    Y[np.arange(n), labels] = 1.0
    Y_c = Y - Y.mean(axis=0)
    U, S, _ = np.linalg.svd(Y_c, full_matrices=False)
    # 保留非零奇异值对应的向量（≤ n_cls−1）
    rank = int((S > S[0] * 1e-10).sum())
    return U[:, :rank]


def class_norm(X, V):
    """‖X‖_class = ‖V^T X V‖₂。"""
    XV = V.T @ X @ V
    return float(np.linalg.norm(XV, 2))


def spectral_alignment(X, V):
    """X 的 top 特征向量与类子空间的对齐度 = ‖V^T v_top‖² 。"""
    eig, vec = np.linalg.eigh(X)
    v_top = vec[:, -1]
    return float(np.sum((V.T @ v_top) ** 2))


def smoothed_class_perturbation(B_clean, dB, V, h, T):
    """‖(I−hB)^{T−1} · dB · V‖₂——传播 T−1 步后的类方向扰动。"""
    n = B_clean.shape[0]
    I = np.eye(n)
    # dB @ V: (n, k)
    W = dB @ V
    for _ in range(T - 1):
        W = (I - h * B_clean) @ W
    return float(np.linalg.norm(W, 2))


def main():
    feats, edges, labels = load()
    n = len(labels)
    n_cls = len(np.unique(labels))
    a_star = np.array([1.0 if len(set(labels[u] for u in e)) == 1 else 0.0
                       for e in edges])
    sgn = np.where(a_star == 1, -1.0, 1.0)
    m = len(edges)

    B_clean = build_op(n, edges, a_star)
    eig = np.linalg.eigvalsh(B_clean)
    h = 1.0 / float(abs(eig[-1]))
    V = class_subspace(labels, n_cls)
    print(f"n={n}, n_cls={n_cls}, V 维度: {V.shape[1]}")
    print(f"λ_max={float(abs(eig[-1])):.1f}, h={h:.4f}")

    results = {}
    for p in (0.05, 0.10, 0.20):
        q_h = 3 * p * (1 - p)
        q_m = p * (1 - p)
        rng = np.random.default_rng(42 + int(p * 100))
        print(f"\n--- p={p:.2f} ---")

        # 点估计 + 100 次 MC
        cn_A_list, cn_B_list = [], []        # ‖ΔB−M₀‖_class
        sn_A_list, sn_B_list = [], []        # ‖ΔB−M₀‖₂
        al_A_list, al_B_list = [], []        # top-eigvec 对齐度
        sm_A, sm_B = {}, {}                   # smoothed class pert

        for _ in range(N_REALIZATIONS):
            # A
            W = rng.random(n) < p
            pred_A = labels * (1 - 2 * W.astype(int))
            aA = np.array([1.0 if len(set(pred_A[u] for u in e)) == 1
                           else 0.0 for e in edges])
            dA = aA - a_star
            E_A = build_op(n, edges, dA)  # ΔB_A

            # B
            dB = np.zeros(m)
            for i, e in enumerate(edges):
                q = q_h if a_star[i] == 1 else q_m
                if rng.random() < q:
                    dB[i] = sgn[i]
            E_B = build_op(n, edges, dB)  # ΔB_B

            # M₀（共同均值）
            # E_A 和 E_B 期望相同但实现不同；中心化用各自实现减去期望算子
            # 简化：直接用 E（因为 E[E_A]=E[E_B]=M₀，所以 E−M₀ ≈ E 的中心化
            # 但我们没有精确 M₀；用多次平均近似
            # 更好：直接比较 E_A 和 E_B 的类子空间范数（差分效应相同）
            cn_A_list.append(class_norm(E_A, V))
            cn_B_list.append(class_norm(E_B, V))
            sn_A_list.append(float(np.linalg.norm(E_A, 2)))
            sn_B_list.append(float(np.linalg.norm(E_B, 2)))
            al_A_list.append(spectral_alignment(E_A, V))
            al_B_list.append(spectral_alignment(E_B, V))

            # 传播后的类方向扰动（几个深度）
            for T in (1, 5, 10, 20):
                sm_A.setdefault(T, []).append(
                    smoothed_class_perturbation(B_clean, E_A, V, h, T))
                sm_B.setdefault(T, []).append(
                    smoothed_class_perturbation(B_clean, E_B, V, h, T))

        cn_ratio = np.mean(cn_A_list) / max(np.mean(cn_B_list), 1e-12)
        sn_ratio = np.mean(sn_A_list) / max(np.mean(sn_B_list), 1e12) \
            if False else np.mean(sn_A_list) / max(np.mean(sn_B_list), 1e-12)
        al_A, al_B = np.mean(al_A_list), np.mean(al_B_list)

        print(f"  ‖·‖₂:   A={np.mean(sn_A_list):.4f} B={np.mean(sn_B_list):.4f}"
              f"  ratio={sn_ratio:.3f}  (A {'>' if sn_ratio > 1 else '<'} B)")
        print(f"  ‖·‖_cls: A={np.mean(cn_A_list):.4f} "
              f"B={np.mean(cn_B_list):.4f}  ratio={cn_ratio:.3f}  "
              f"(A {'>' if cn_ratio > 1 else '<'} B)")
        print(f"  对齐度:  A={al_A:.4f} B={al_B:.4f}  "
              f"(A {'更集中' if al_A < al_B else '更均匀'})")
        print(f"  类子空间范数/谱范数: A={np.mean(cn_A_list)/np.mean(sn_A_list):.4f}"
              f" B={np.mean(cn_B_list)/np.mean(sn_B_list):.4f}")

        # 传播后的比值
        print(f"  传播后类方向扰动比值 (A/B):")
        for T in sorted(sm_A):
            ra = np.mean(sm_A[T])
            rb = np.mean(sm_B[T])
            ratio = ra / max(rb, 1e-12)
            print(f"    T={T:>2}: A={ra:.4f} B={rb:.4f} ratio={ratio:.3f}"
                  f"  (A {'>' if ratio > 1 else '<'} B)"
                  f"{' ← 传导!' if ratio > 1 else ''}")

        results[f"p={p}"] = dict(
            sn_A=float(np.mean(sn_A_list)), sn_B=float(np.mean(sn_B_list)),
            cn_A=float(np.mean(cn_A_list)), cn_B=float(np.mean(cn_B_list)),
            align_A=float(al_A), align_B=float(al_B),
            smoothed={T: dict(A=float(np.mean(sm_A[T])),
                               B=float(np.mean(sm_B[T])))
                      for T in sm_A})

    with open("results/class_norm.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print("\n已写出 results/class_norm.json")


if __name__ == "__main__":
    main()
