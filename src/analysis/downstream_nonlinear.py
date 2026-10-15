"""非线性传播下的传导检验：ReLU 是否翻转 A-B 方向？

线性传播中 A 的局部化误差被平滑修复（margin A > B）。
如果加一个 ReLU 非线性层，局部化误差可能被放大而非平滑。

三种传播规则对比：
  Linear:    Z_{t+1} = (I − hB) Z_t
  ReLU:      Z_{t+1} = ReLU((I − hB) Z_t)     （非线性每步）
  OneReLU:   Z_1 = ReLU((I − hB) Z_0);  Z_{t>1} = (I − hB) Z_t  （仅第一步非线性）

用 one-hot 标签（100% 类信号），margin 指标，p ∈ {0.05, 0.10, 20}。
如果 ReLU 后 A margin < B margin，方向翻转——传导条件找到了。
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
DEPTH_LIST = (5, 10, 20)
P_LIST = (0.05, 0.10, 0.20)
MODES = ("linear", "relu", "one_relu")


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


def propagate_nonlinear(B, Z0, h, T, mode):
    I = np.eye(B.shape[0])
    Z = Z0.copy()
    for t in range(T):
        Z_new = (I - h * B) @ Z
        if mode == "relu":
            Z = np.maximum(Z_new, 0)
        elif mode == "one_relu" and t == 0:
            Z = np.maximum(Z_new, 0)
        else:
            Z = Z_new
    return Z


def margin_metrics(Z, labels, n_cls):
    n = Z.shape[0]
    Z_true = Z[np.arange(n), labels]
    Z_other = Z.copy()
    Z_other[np.arange(n), labels] = -np.inf
    margin = Z_true - Z_other.max(axis=1)
    pred = np.argmax(Z, axis=1)
    acc = float(np.mean(pred == labels))
    return float(np.mean(margin)), acc


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
    Z0 = np.zeros((n, n_cls))
    Z0[np.arange(n), labels] = 1.0

    print(f"λ_max={float(abs(eig[-1])):.1f}, h={h:.4f}")
    results = {}

    for mode in MODES:
        print(f"\n{'='*70}")
        print(f"传播模式: {mode}")
        print(f"{'='*70}")
        for T in DEPTH_LIST:
            Z_c = propagate_nonlinear(B_clean, Z0, h, T, mode)
            m_c, a_c = margin_metrics(Z_c, labels, n_cls)
            results.setdefault(mode, {}).setdefault(T, {})["clean"] = dict(
                margin=m_c, acc=a_c)
            print(f"  Clean T={T}: margin={m_c:.4f}, acc={a_c:.3f}")

        for p in P_LIST:
            q_h = 3 * p * (1 - p)
            q_m = p * (1 - p)
            print(f"\n  p={p:.2f}:")
            rng = np.random.default_rng(42 + int(p * 100))
            for T in DEPTH_LIST:
                margins_A, margins_B, accs_A, accs_B = [], [], [], []
                for _ in range(N_REALIZATIONS):
                    # A
                    W = rng.random(n) < p
                    pred_A = labels * (1 - 2 * W.astype(int))
                    aA = np.array(
                        [1.0 if len(set(pred_A[u] for u in e)) == 1
                         else 0.0 for e in edges])
                    BA = build_op(n, edges, aA)
                    ZA = propagate_nonlinear(BA, Z0, h, T, mode)
                    # B
                    dB = np.zeros(m)
                    for i, e in enumerate(edges):
                        q = q_h if a_star[i] == 1 else q_m
                        if rng.random() < q:
                            dB[i] = sgn[i]
                    BB = build_op(n, edges, a_star + dB)
                    ZB = propagate_nonlinear(BB, Z0, h, T, mode)
                    # 测量
                    mA, aA_ = margin_metrics(ZA, labels, n_cls)
                    mB, aB_ = margin_metrics(ZB, labels, n_cls)
                    margins_A.append(mA)
                    margins_B.append(mB)
                    accs_A.append(aA_)
                    accs_B.append(aB_)

                # Bootstrap CI
                rng_bs = np.random.default_rng(99)
                ia = rng_bs.integers(0, N_REALIZATIONS,
                                      (2000, N_REALIZATIONS))
                ib = rng_bs.integers(0, N_REALIZATIONS,
                                     (2000, N_REALIZATIONS))
                m_diff = np.array(margins_A)[ia].mean(1) - \
                    np.array(margins_B)[ib].mean(1)
                a_diff = np.array(accs_A)[ia].mean(1) - \
                    np.array(accs_B)[ib].mean(1)
                lo_m, hi_m = np.quantile(m_diff, [0.025, 0.975])
                lo_a, hi_a = np.quantile(a_diff, [0.025, 0.975])
                r = dict(
                    margin_A=float(np.mean(margins_A)),
                    margin_B=float(np.mean(margins_B)),
                    margin_diff=float(np.mean(margins_A) -
                                     np.mean(margins_B)),
                    margin_ci=[float(lo_m), float(hi_m)],
                    acc_A=float(np.mean(accs_A)),
                    acc_B=float(np.mean(accs_B)),
                    acc_diff=float(np.mean(accs_A) - np.mean(accs_B)),
                    acc_ci=[float(lo_a), float(hi_a)])
                results[mode][T][f"p={p}"] = r
                flipped = "⚠ 翻转!" if r["margin_diff"] < 0 and lo_m < 0 else ""
                print(f"    T={T}: margin A={r['margin_A']:.4f} "
                      f"B={r['margin_B']:.4f} "
                      f"(diff {r['margin_diff']:+.5f} "
                      f"[{lo_m:+.5f},{hi_m:+.5f}]){flipped}; "
                      f"acc A={r['acc_A']:.3f} B={r['acc_B']:.3f}")

    with open("results/downstream_nonlinear.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print("\n已写出 results/downstream_nonlinear.json")


if __name__ == "__main__":
    experiment()
