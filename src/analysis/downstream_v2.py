"""传导检验 v2：修正 PCA 类信号弱的问题。

v1 的问题：PCA-50 特征的类信号仅 7.7%——A-B 差异被 93% 的类内噪声
淹没，精度差不显著。修正：用 one-hot 标签（100% 类信号），使算子误差
的影响直接作用于类分离。测量 margins（比精度更敏感）。

v2 还修正：
  - 增加传播深度 T∈{1,2,5,10,20}
  - 同时报告精度、margin 均值、类信号比
  - 用 margin 均值差（A−B）的 bootstrap CI 作为主要检验
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
DEPTH_LIST = (1, 2, 5, 10, 20)
P_LIST = (0.05, 0.10, 0.20)


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


def propagate(B, Z0, h, T):
    I = np.eye(B.shape[0])
    Z = Z0.copy()
    for _ in range(T):
        Z = (I - h * B) @ Z
    return Z


def margin_metrics(Z, labels, n_cls):
    """每节点 margin = Z[i,c_true] − max_{c'≠c} Z[i,c']；信号比。"""
    n = Z.shape[0]
    # margin
    Z_true = Z[np.arange(n), labels]
    Z_other = Z.copy()
    Z_other[np.arange(n), labels] = -np.inf
    margin = Z_true - Z_other.max(axis=1)
    pred = np.argmax(Z, axis=1)
    acc = float(np.mean(pred == labels))
    # 信号比
    overall = Z.mean(axis=0)
    between = sum(np.sum((labels == c).sum() * (Z[labels == c].mean(0) - overall)**2)
                  for c in range(n_cls))
    within = sum(np.sum((Z[labels == c] - Z[labels == c].mean(0))**2)
                 for c in range(n_cls))
    signal = float(between / max(between + within, 1e-12))
    return float(np.mean(margin)), float(np.std(margin)), acc, signal


def experiment():
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

    # Z_0 = one-hot 标签（100% 类信号）
    Z0 = np.zeros((n, n_cls))
    Z0[np.arange(n), labels] = 1.0

    print(f"λ_max={float(abs(eig[-1])):.1f}, h={h:.4f}")
    print(f"Z_0 信号: acc=1.000, signal=1.000")
    results = {}
    for T in DEPTH_LIST:
        Z_clean = propagate(B_clean, Z0, h, T)
        m_c, s_c, a_c, sig_c = margin_metrics(Z_clean, labels, n_cls)
        results.setdefault(T, {})["clean"] = dict(
            margin=m_c, acc=a_c, signal=sig_c)
        print(f"  Clean T={T}: margin={m_c:.4f}, acc={a_c:.3f}, "
              f"signal={sig_c:.3f}")

    for p in P_LIST:
        q_h = 3 * p * (1 - p)
        q_m = p * (1 - p)
        print(f"\n--- p={p:.2f} ---")
        rng = np.random.default_rng(42 + int(p * 100))
        for T in DEPTH_LIST:
            margins_A, margins_B = [], []
            accs_A, accs_B = [], []
            sigs_A, sigs_B = [], []
            Z_clean_T = propagate(B_clean, Z0, h, T)
            for _ in range(N_REALIZATIONS):
                # A
                W = rng.random(n) < p
                pred_A = labels * (1 - 2 * W.astype(int))
                aA = np.array([1.0 if len(set(pred_A[u] for u in e)) == 1
                               else 0.0 for e in edges])
                BA = build_op(n, edges, aA)
                ZA = propagate(BA, Z0, h, T)
                # B
                dB = np.zeros(m)
                for i, e in enumerate(edges):
                    q = q_h if a_star[i] == 1 else q_m
                    if rng.random() < q:
                        dB[i] = sgn[i]
                BB = build_op(n, edges, a_star + dB)
                ZB = propagate(BB, Z0, h, T)
                # 测量
                mA, sA, aA_, sigA = margin_metrics(ZA, labels, n_cls)
                mB, sB, aB_, sigB = margin_metrics(ZB, labels, n_cls)
                margins_A.append(mA)
                margins_B.append(mB)
                accs_A.append(aA_)
                accs_B.append(aB_)
                sigs_A.append(sigA)
                sigs_B.append(sigB)

            # Bootstrap CI
            rng_bs = np.random.default_rng(99)
            ia = rng_bs.integers(0, N_REALIZATIONS, (2000, N_REALIZATIONS))
            ib = rng_bs.integers(0, N_REALIZATIONS, (2000, N_REALIZATIONS))
            m_diff = np.array(margins_A)[ia].mean(1) - \
                np.array(margins_B)[ib].mean(1)
            a_diff = np.array(accs_A)[ia].mean(1) - \
                np.array(accs_B)[ib].mean(1)
            lo_m, hi_m = np.quantile(m_diff, [0.025, 0.975])
            lo_a, hi_a = np.quantile(a_diff, [0.025, 0.975])
            r = dict(
                margin_A=float(np.mean(margins_A)),
                margin_B=float(np.mean(margins_B)),
                margin_diff=float(np.mean(margins_A) - np.mean(margins_B)),
                margin_diff_ci=[float(lo_m), float(hi_m)],
                acc_A=float(np.mean(accs_A)), acc_B=float(np.mean(accs_B)),
                acc_diff=float(np.mean(accs_A) - np.mean(accs_B)),
                acc_diff_ci=[float(lo_a), float(hi_a)],
                sig_A=float(np.mean(sigs_A)), sig_B=float(np.mean(sigs_B)))
            results[T][f"p={p}"] = r
            print(f"  T={T}: margin A={r['margin_A']:.4f} "
                  f"B={r['margin_B']:.4f} (diff {r['margin_diff']:+.5f} "
                  f"CI [{lo_m:+.5f},{hi_m:+.5f}]); "
                  f"acc A={r['acc_A']:.3f} B={r['acc_B']:.3f} "
                  f"(diff {r['acc_diff']:+.4f} CI [{lo_a:+.4f},{hi_a:+.4f}])")

    with open("results/downstream_v2.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print("\n已写出 results/downstream_v2.json")


if __name__ == "__main__":
    experiment()
