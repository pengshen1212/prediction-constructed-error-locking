"""类相关谱范数（修正 + 快速版）。

修正（相对 downstream_v2/class_norm 旧版）：
  1. 多类对称信道：节点标签以 1−p 保留、以 p/(C−1) 换成其他任一类
     （旧版 labels*(1−2W) 是二值符号翻转，7 类下异质边永远不变同质，
      与模型 B 的 q_m=p(1−p) 边际不匹配）。
  2. 每边解析期望 E[dA_e] = P(所有 pred 相同) − a*_e，
     模型 B 以 |E[dA_e]| 速率、同号翻转 —— 对任意边大小精确匹配边际
     （旧版 q_h=3p(1−p) 只对 k=3 成立）。

快速：
  - ‖X‖_class = ‖V^T X V − C0‖（6×6 小矩阵）
  - ‖X‖_2 用 eigsh k=1（0.07s）而非全谱 eigh
  - build_op 用 np.add.at 向量化

假设：
  H1: 线性传播下 ratio_class = ‖X_A‖_cls/‖X_B‖_cls < 1（A 类方向扰动更小）
      而 ratio_total > 1 —— 解释 margin 方向反转。
  H2: ratio_smoothed(T) 随 T 变化方向（平滑滤掉非类方向的高频扰动）。
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
P_LIST = (0.05, 0.10, 0.20)
SMOOTH_T = (1, 5, 10, 20)
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
    """预计算每条边在稠密矩阵中的 (row, col, owner, 1/k) 条目。"""
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
    """多类对称信道：保留 1−p，否则均匀换成其他类。"""
    n = len(labels)
    mask = rng.random(n) < p
    j = rng.integers(0, n_cls - 1, size=n)          # 均匀于其他类
    other = j + (j >= labels)
    return np.where(mask, other, labels)


def expected_da(edges, labels, n_cls, p):
    """每边解析 E[dA_e] = P(所有 pred 相同) − a*_e。"""
    rho = p / (n_cls - 1)                            # 换到任一特定其他类
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


def class_subspace(labels, n_cls):
    n = len(labels)
    Y = np.zeros((n, n_cls))
    Y[np.arange(n), labels] = 1.0
    Y_c = Y - Y.mean(axis=0)
    U, S, _ = np.linalg.svd(Y_c, full_matrices=False)
    rank = int((S > S[0] * 1e-10).sum())
    return U[:, :rank]


def top_eig(X):
    """对称矩阵谱范数 + top 特征向量（按幅值最大）。"""
    try:
        w, v = eigsh(X, k=1, which="LM", tol=1e-8, maxiter=3000)
        return float(abs(w[0])), v[:, 0]
    except Exception:
        w, v = np.linalg.eigh(X)
        i = int(np.argmax(np.abs(w)))
        return float(abs(w[i])), v[:, i]


def main():
    feats, edges, labels = load()
    n = len(labels)
    n_cls = len(np.unique(labels))
    m = len(edges)
    flat = edge_flat_index(edges)

    a_star = np.array([1.0 if len(set(labels[u] for u in e)) == 1 else 0.0
                       for e in edges])
    sizes = np.array([len(e) for e in edges])
    print(f"n={n}, m={m}, n_cls={n_cls}, "
          f"edge size: min={sizes.min()} med={np.median(sizes):.0f} "
          f"max={sizes.max()}, 同质边占比={a_star.mean():.3f}")

    B_clean = build_op(n, flat, a_star)
    lam = float(abs(eigsh(B_clean, k=1, which="LM",
                          return_eigenvectors=False)[0]))
    h = 1.0 / lam
    V = class_subspace(labels, n_cls)
    print(f"λ_max={lam:.1f}, h={h:.4f}, V: {n}×{V.shape[1]}")

    results = {}
    for p in P_LIST:
        w0 = expected_da(edges, labels, n_cls, p)
        q_rate = np.abs(w0)
        sgn = np.sign(w0)
        M0 = build_op(n, flat, w0)
        C0 = V.T @ M0 @ V
        print(f"\n--- p={p:.2f} ---")
        print(f"  边际匹配: homog |E[dA]| 均值="
              f"{q_rate[a_star == 1].mean():.4f}, "
              f"heterog |E[dA]| 均值={q_rate[a_star == 0].mean():.4f}")

        rng = np.random.default_rng(SEED + int(p * 100))
        cn_A, cn_B = [], []          # ‖X‖_class（谱）
        fn_A, fn_B = [], []          # ‖X‖_class（Frobenius）
        sn_A, sn_B = [], []          # ‖X‖_2
        al_A, al_B = [], []          # top-eigvec 对齐 ‖V^T v‖²
        ce_A, ce_B = [], []          # V^T E V（检验均值≈C0）
        smA, smB = {t: [] for t in SMOOTH_T}, {t: [] for t in SMOOTH_T}

        for _ in range(N_REALIZATIONS):
            for model, store in (("A", None), ("B", None)):
                if model == "A":
                    pred = corrupt_labels(labels, n_cls, p, rng)
                    d = np.array([1.0 if len(set(pred[u] for u in e)) == 1
                                  else 0.0 for e in edges]) - a_star
                else:
                    flip = rng.random(m) < q_rate
                    d = np.where(flip, sgn, 0.0)
                E = build_op(n, flat, d)
                X = E - M0
                C = V.T @ E @ V                       # 6×6
                cn = float(np.linalg.norm(C - C0, 2))
                fn = float(np.linalg.norm(C - C0, "fro"))
                sn, v_top = top_eig(X)
                align = float(np.sum((V.T @ v_top) ** 2))
                if model == "A":
                    cn_A.append(cn); fn_A.append(fn); sn_A.append(sn)
                    al_A.append(align); ce_A.append(C.copy())
                else:
                    cn_B.append(cn); fn_B.append(fn); sn_B.append(sn)
                    al_B.append(align); ce_B.append(C.copy())
                # 平滑后类方向扰动 ‖(I−hB)^{T−1} X V‖
                W = X @ V
                rec = smA if model == "A" else smB
                cur_T = 0
                for t in range(1, max(SMOOTH_T) + 1):
                    if t > 1:
                        W = W - h * (B_clean @ W)
                    if t in rec:
                        rec[t].append(float(np.linalg.norm(W, 2)))

        # 检验 M₀ 的正确性：MC 平均的 V^T E V 应≈C0
        err_A = np.abs(np.mean(ce_A, axis=0) - C0).max()
        err_B = np.abs(np.mean(ce_B, axis=0) - C0).max()
        print(f"  MC 校验 max|mean(V^T E V)−C0|: A={err_A:.2e} B={err_B:.2e}")

        def boot_ratio(x, y, n_bs=2000, seed=7):
            r = np.random.default_rng(seed)
            xi = r.integers(0, len(x), (n_bs, len(x)))
            yi = r.integers(0, len(y), (n_bs, len(y)))
            return np.quantile(np.array(x)[xi].mean(1) /
                               np.array(y)[yi].mean(1), [0.025, 0.975])

        for name, xa, xb in (("class", cn_A, cn_B), ("total", sn_A, sn_B),
                             ("classF", fn_A, fn_B)):
            ra, rb = np.mean(xa), np.mean(xb)
            lo, hi = boot_ratio(xa, xb)
            print(f"  ‖·‖_{name:>5}: A={ra:.4f} B={rb:.4f} "
                  f"ratio={ra / rb:.3f} CI[{lo:.3f},{hi:.3f}]")

        print(f"  对齐度: A={np.mean(al_A):.4f} B={np.mean(al_B):.4f} "
              f"(top-eigvec 落入类子空间的比例)")
        print(f"  类占比 ‖·‖_cls/‖·‖₂: A={np.mean(cn_A) / np.mean(sn_A):.4f}"
              f" B={np.mean(cn_B) / np.mean(sn_B):.4f}")

        print("  平滑后类方向扰动 ‖(I−hB)^{T−1}XV‖:")
        sm_out = {}
        for t in SMOOTH_T:
            ra, rb = np.mean(smA[t]), np.mean(smB[t])
            lo, hi = boot_ratio(smA[t], smB[t])
            sm_out[t] = dict(A=ra, B=rb, ratio=ra / rb, ci=[lo, hi])
            print(f"    T={t:>2}: A={ra:.4f} B={rb:.4f} ratio={ra / rb:.3f} "
                  f"CI[{lo:.3f},{hi:.3f}]")

        results[f"p={p}"] = dict(
            class_A=float(np.mean(cn_A)), class_B=float(np.mean(cn_B)),
            class_ratio=float(np.mean(cn_A) / np.mean(cn_B)),
            class_ci=[float(x) for x in boot_ratio(cn_A, cn_B)],
            total_A=float(np.mean(sn_A)), total_B=float(np.mean(sn_B)),
            total_ratio=float(np.mean(sn_A) / np.mean(sn_B)),
            classF_ratio=float(np.mean(fn_A) / np.mean(fn_B)),
            align_A=float(np.mean(al_A)), align_B=float(np.mean(al_B)),
            mc_check_A=float(err_A), mc_check_B=float(err_B),
            smoothed={t: sm_out[t] for t in SMOOTH_T})

    with open("results/class_norm_fast.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print("\n已写出 results/class_norm_fast.json")


if __name__ == "__main__":
    main()
