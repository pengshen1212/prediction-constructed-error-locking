"""
命题二 v2：审计修正后的配对实验（consistency × alignment × depth）。

针对审计记录（prob2-step1-audit-2026-09-26.md）的修正：
  1. 严格互斥划分：探针训练 / 验证(选λ) / 测试 / 估计种子 全部不相交；
     所有扰动模式在同一图、同一特征、同一测试集上配对比较。
  2. XOR 节点 switching（保持符号平衡）作为独立模式；原 OR 集中破坏保留并改名；
     η/ρ=0 时不强制翻转。
  3. 不再声称归一化非扩张；深度曲线用固定扰动单轨迹记录（不再逐深度重采样），
     并记录归一化前最小行范数作为敏感性诊断。
  4. 扰动用二维描述子刻画：三角形受挫率（内部一致性，3-均匀超图可精确计算）
     + 划分错位率（对齐，平衡模式有精确值，受挫模式用谱划分估计）；
     另记录翻转质量与算子范数 ‖A_ŝ−A_s‖₂。
  5. 指标含均值间隔、10% 分位间隔、负间隔比例、错误率；λ 由验证集准确率选取。
  6. 传播本身是带符号图传播（审计第 5 点：可精确归约），本实验用于现象定位，
     超图特有贡献需关联级算子，见报告。

运行：python run_experiment_v2.py [--seeds 10] [--outdir results_v2]
"""

import argparse
import csv
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAMBDA_GRID = [1e-3, 1e-2, 1e-1, 1, 10, 100]
T_GRID = [2, 5, 10, 25, 50, 100]
MODE_ID = {"oracle": 0, "iid": 1, "switch_random": 2, "switch_degree": 3,
           "or_degree": 4, "unsigned": 5}


# ---------------- 数据：图与特征分离生成（便于按 σ 配对） ----------------


def make_graph(rng, n=400, m=600, p_hom=0.7):
    y = np.concatenate([-np.ones(n // 2), np.ones(n - n // 2)]).astype(int)
    y = y[rng.permutation(n)]
    idx = {1: np.where(y == 1)[0], -1: np.where(y == -1)[0]}
    E = np.empty((m, 3), dtype=int)
    for t in range(m):
        if rng.random() < p_hom:
            c = 1 if rng.random() < 0.5 else -1
            E[t] = rng.choice(idx[c], 3, replace=False)
        else:
            maj = 1 if rng.random() < 0.5 else -1
            E[t, :2] = rng.choice(idx[maj], 2, replace=False)
            E[t, 2] = int(rng.choice(idx[-maj], 1)[0])
    return y, E


def make_features(rng, y, sigma_feat, d=8, mu=1.0):
    X = np.empty((len(y), d))
    X[:, 0] = y * mu + sigma_feat * rng.standard_normal(len(y))
    X[:, 1:] = sigma_feat * rng.standard_normal((len(y), d - 1))
    return X


def unordered_pairs(E):
    us = np.concatenate([E[:, 0], E[:, 0], E[:, 1]])
    vs = np.concatenate([E[:, 1], E[:, 2], E[:, 2]])
    return us, vs


def true_relations(y, us, vs):
    return np.where(y[us] == y[vs], 1.0, -1.0)


def normalize_rows(Z):
    nrm = np.linalg.norm(Z, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return Z / nrm


def propagate_record(Z0, us, vs, s_un, T_max, tau=0.5):
    """带检查点与归一化前范数记录的传播（与 v1 propagate 相同的动力学）。"""
    n = Z0.shape[0]
    src = np.concatenate([us, vs])
    dst = np.concatenate([vs, us])
    s = np.concatenate([s_un, s_un])
    deg = np.bincount(src, minlength=n).astype(float)
    deg[deg == 0] = 1.0
    Z = normalize_rows(Z0.copy())
    snaps, minnorms = {}, []
    todo = sorted(T_GRID)
    if 0 in todo:
        snaps[0] = Z.copy()
    for t in range(1, T_max + 1):
        msg = s[:, None] * (Z[dst] - Z[src]) / 3.0
        agg = np.zeros_like(Z)
        np.add.at(agg, src, msg)
        agg /= deg[:, None]
        Z = Z + tau * agg
        minnorms.append(float(np.linalg.norm(Z, axis=1).min()))
        Z = normalize_rows(Z)
        if todo and t == todo[0]:
            snaps[t] = Z.copy()
            todo.pop(0)
    return snaps, min(minnorms), float(np.mean(minnorms[-10:]))


# ---------------- 扰动模式与二维误差描述子 ----------------


def flip_iid(rng, s_true, eta):
    mask = rng.random(s_true.shape[0]) < eta
    return np.where(mask, -s_true, s_true)


def switch_set(rng_free, y, us, vs, s_true, switch_idx):
    """XOR 节点 switching: ŝ_ij = q_i q_j s_ij，q=−1 当且仅当 i∈S。保持平衡。
    诱导的节点划分为 r = q⊙y（cluster id），对齐/上限公式用 r 而非 q。"""
    q = np.ones(len(y))
    q[switch_idx] = -1.0
    s_hat = q[us] * q[vs] * s_true
    return s_hat, q * y


def top_degree_idx(us, vs, n, k):
    deg = np.bincount(us, minlength=n) + np.bincount(vs, minlength=n)
    return np.argsort(deg)[::-1][:k]


def triangle_frustration(s_un):
    """3-均匀超图精确一致性：每个超边 3 对符号乘积为负即受挫。"""
    m = s_un.shape[0] // 3
    s = s_un.reshape(3, m)
    return float(((s[0] * s[1] * s[2]) < 0).mean())


def signed_adjacency(us, vs, s_un, n):
    A = np.zeros((n, n))
    np.add.at(A, (us, vs), s_un)
    np.add.at(A, (vs, us), s_un)
    return A


def spectral_partition(A):
    w, V = np.linalg.eigh(A)
    v = V[:, int(np.argmax(np.abs(w)))]
    q = np.where(v >= 0, 1.0, -1.0)
    return q


def align_err(q, y):
    return float(min((q != y).mean(), (-q != y).mean()))


def cc_disagreement(us, vs, s_un, q):
    pred = np.where(q[us] == q[vs], 1.0, -1.0)
    return float((pred != s_un).mean())


def ceiling_acc(q, y):
    """表征只保留 q 信息时，任何表征分类器的最优准确率（审计式）。"""
    total = 0
    for c in (-1.0, 1.0):
        mask = q == c
        total += max((y[mask] == 1).sum(), (y[mask] == -1).sum())
    return float(total / len(y))


def describe(mode, param, s_hat, q_exact, y, us, vs, s_true, n):
    d = {"mode": mode, "param": param}
    d["flip_mass"] = float((s_hat != s_true).mean())
    d["tri_frust"] = triangle_frustration(s_hat)
    if q_exact is not None:
        d["align_err"] = align_err(q_exact, y)
        d["ceiling"] = ceiling_acc(q_exact, y)
        d["cc"] = cc_disagreement(us, vs, s_hat, q_exact)
    else:
        A = signed_adjacency(us, vs, s_hat, n)
        qs = spectral_partition(A)
        d["align_err"] = align_err(qs, y)
        d["ceiling"] = ceiling_acc(qs, y)
        d["cc"] = cc_disagreement(us, vs, s_hat, qs)
    A_t = signed_adjacency(us, vs, s_true, n)
    A_h = signed_adjacency(us, vs, s_hat, n)
    d["op_norm"] = float(np.linalg.norm(A_h - A_t, 2))
    return d


def build_perturbation(mode, param, rng, y, us, vs, s_true, n):
    """返回 (s_hat, q_exact_or_None)。q_exact 仅平衡模式非 None。"""
    if mode == "oracle":
        return s_true.copy(), y.astype(float)
    if mode == "iid":
        return flip_iid(rng, s_true, param), None
    if mode == "switch_random":
        k = int(round(param * n))
        S = rng.choice(n, k, replace=False)
        return switch_set(rng, y, us, vs, s_true, S)
    if mode in ("switch_degree", "or_degree"):
        k = int(round(param * n))
        S = top_degree_idx(us, vs, n, k)
        if mode == "switch_degree":
            return switch_set(rng, y, us, vs, s_true, S)
        mask = np.isin(us, S) | np.isin(vs, S)  # 原 v2 前的 OR 集中破坏（受挫）
        return np.where(mask, -s_true, s_true), None
    raise ValueError(mode)


# ---------------- 探针：验证集选 λ，测试集报告分位间隔 ----------------


def ridge_w(Z, y, idx, lam):
    A = np.hstack([Z[idx], np.ones((len(idx), 1))])
    return np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y[idx])


def probe_eval(Z, y, train, val, test):
    best_key, best_lam, best_w = None, None, None
    for lam in LAMBDA_GRID:
        w = ridge_w(Z, y, train, lam)
        Av = np.hstack([Z[val], np.ones((len(val), 1))])
        fv = Av @ w
        vacc = float(((fv > 0) == (y[val] > 0)).mean())
        wn = max(np.linalg.norm(w[:-1]), 1e-12)
        vmarg = float((y[val] * fv).mean() / wn)
        key = (round(vacc, 6), round(vmarg, 6))
        if best_key is None or key > best_key:
            best_key, best_lam, best_w = key, lam, w
    wn = max(np.linalg.norm(best_w[:-1]), 1e-12)
    At = np.hstack([Z[test], np.ones((len(test), 1))])
    ft = At @ best_w
    sm = y[test] * ft / wn
    return {"lam": best_lam, "margin": float(sm.mean()),
            "margin_q10": float(np.quantile(sm, 0.10)),
            "neg_frac": float((sm < 0).mean()),
            "acc": float(((ft > 0) == (y[test] > 0)).mean())}


# ---------------- 几何与能量 ----------------


def pair_weights(us, vs, n):
    key = np.minimum(us, vs) * n + np.maximum(us, vs)
    uniq, cnt = np.unique(key, return_counts=True)
    return (uniq // n).astype(int), (uniq % n).astype(int), cnt.astype(float)


def energy_positive(Z, uu, vv, ww):
    diff2 = ((Z[uu] - Z[vv]) ** 2).sum(1)
    return float((ww * diff2).sum() / Z.shape[0])


def class_geometry(Z, y):
    c1, c2 = Z[y == 1].mean(0), Z[y == -1].mean(0)
    intra = 0.5 * ((np.linalg.norm(Z[y == 1] - c1, axis=1) ** 2).mean()
                   + (np.linalg.norm(Z[y == -1] - c2, axis=1) ** 2).mean())
    return float(np.linalg.norm(c1 - c2)), float(intra)


# ---------------- 主流程 ----------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--outdir", default="results_v2")
    args = ap.parse_args()
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.outdir)
    os.makedirs(outdir, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150,
                         "axes.grid": True, "grid.alpha": 0.3, "font.size": 9})

    controlled = ([("oracle", 0.0)]
                  + [("iid", p) for p in (0.1, 0.2, 0.3)]
                  + [("switch_random", p) for p in (0.1, 0.2, 0.3, 0.5)]
                  + [("switch_degree", p) for p in (0.1, 0.2, 0.3)]
                  + [("or_degree", p) for p in (0.1, 0.2, 0.3)]
                  + [("unsigned", 0.0)])
    est_cfg = [(s, l) for s in (0.5, 1.0, 2.0) for l in (8, 32)]

    rows = []
    for seed in range(args.seeds):
        rng_g = np.random.default_rng(7000 + seed)
        y, E = make_graph(rng_g)
        n = len(y)
        us, vs = unordered_pairs(E)
        s_true = true_relations(y, us, vs)
        uu, vv, ww = pair_weights(us, vs, n)

        rng_s = np.random.default_rng(7100 + seed)
        train, val, test = make_splits(rng_s, y)
        X = make_features(np.random.default_rng(7200 + seed), y, 1.0)
        Z0 = normalize_rows(X)

        # oracle 轨迹（作为扰动敏感性的配对参照）
        oracle_snaps, _, _ = propagate_record(Z0, us, vs, s_true, max(T_GRID))

        for mode, param in controlled:
            rng_p = np.random.default_rng(7300 + seed * 100 + MODE_ID[mode] * 100 + int(round(param * 100)))
            if mode == "unsigned":
                s_hat, q_ex = np.ones_like(s_true), None
            else:
                s_hat, q_ex = build_perturbation(mode, param, rng_p, y, us, vs, s_true, n)
            desc = describe(mode, param, s_hat, q_ex, y, us, vs, s_true, n)
            snaps, minpre, lastpre = propagate_record(Z0, us, vs, s_hat, max(T_GRID))
            for T, Z in snaps.items():
                inter, intra = class_geometry(Z, y)
                r = {"seed": seed, "T": T, "sigma_feat": 1.0, "ell": 0,
                     **desc, "E_pos": energy_positive(Z, uu, vv, ww),
                     "inter": inter, "intra": intra,
                     "min_prenorm": minpre, "last_prenorm": lastpre,
                     "d_oracle": float(np.linalg.norm(Z - oracle_snaps[T]) / np.sqrt(n))}
                r.update(probe_eval(Z, y, train, val, test))
                rows.append(r)
        # 基线
        for tag, Z in [("identity", Z0),
                       ("noise", normalize_rows(np.random.default_rng(7400 + seed).standard_normal(X.shape)))]:
            r = {"seed": seed, "T": 0, "sigma_feat": 1.0, "ell": 0, "mode": tag,
                 "param": 0.0, "flip_mass": 0.0, "tri_frust": 0.0, "align_err": 0.0,
                 "ceiling": 1.0, "cc": 0.0, "op_norm": 0.0, "d_oracle": 0.0,
                 "E_pos": energy_positive(Z, uu, vv, ww),
                 "inter": class_geometry(Z, y)[0], "intra": class_geometry(Z, y)[1],
                 "min_prenorm": 1.0, "last_prenorm": 1.0}
            r.update(probe_eval(Z, y, train, val, test))
            rows.append(r)

        # 估计路径：特征按 σ 重生成，估计种子与探针/验证/测试互斥
        oracle_cache = {}
        for sigma, ell in est_cfg:
            rng_f = np.random.default_rng(7500 + seed * 100 + int(sigma * 10))  # 特征只依赖 (seed, σ)
            Xs = make_features(rng_f, y, sigma)
            Z0s = normalize_rows(Xs)
            if sigma not in oracle_cache:
                oracle_cache[sigma] = propagate_record(Z0s, us, vs, s_true, max(T_GRID))[0]
            rng_e = np.random.default_rng(7600 + seed * 100 + int(sigma * 10) * 10 + ell)
            seeds_idx = draw_seeds(rng_e, y, test, val, ell)
            lab = y[seeds_idx]
            p1, p2 = Xs[seeds_idx[lab == 1]].mean(0), Xs[seeds_idx[lab == -1]].mean(0)
            y_hat = np.where((Xs @ (p1 - p2)) >= 0, 1.0, -1.0)
            s_est = np.where(y_hat[us] == y_hat[vs], 1.0, -1.0)
            q = y_hat.astype(float)
            desc = {"mode": "estimated", "param": float(ell),
                    "flip_mass": float((s_est != s_true).mean()),
                    "tri_frust": triangle_frustration(s_est),
                    "align_err": align_err(q, y), "ceiling": ceiling_acc(q, y),
                    "cc": cc_disagreement(us, vs, s_est, q),
                    "op_norm": float(np.linalg.norm(signed_adjacency(us, vs, s_est, n)
                                                    - signed_adjacency(us, vs, s_true, n), 2))}
            snaps, minpre, lastpre = propagate_record(Z0s, us, vs, s_est, max(T_GRID))
            for T, Z in snaps.items():
                inter, intra = class_geometry(Z, y)
                r = {"seed": seed, "T": T, "sigma_feat": sigma, "ell": ell,
                     **desc, "E_pos": energy_positive(Z, uu, vv, ww),
                     "inter": inter, "intra": intra,
                     "min_prenorm": minpre, "last_prenorm": lastpre,
                     "d_oracle": float(np.linalg.norm(Z - oracle_cache[sigma][T]) / np.sqrt(n))}
                r.update(probe_eval(Z, y, train, val, test))
                rows.append(r)
            # 同特征恒等对照
            r = {"seed": seed, "T": 0, "sigma_feat": sigma, "ell": ell,
                 **desc, "E_pos": energy_positive(Z0s, uu, vv, ww),
                 "inter": class_geometry(Z0s, y)[0], "intra": class_geometry(Z0s, y)[1],
                 "min_prenorm": 1.0, "last_prenorm": 1.0, "d_oracle": 0.0,
                 "mode": "estimated_identity", "param": float(ell)}
            r.update(probe_eval(Z0s, y, train, val, test))
            rows.append(r)
        print(f"seed {seed} done")

    keys = sorted({k for r in rows for k in r})
    with open(os.path.join(outdir, "paired.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("rows:", len(rows), "->", os.path.join(outdir, "paired.csv"))
    print("lambda 选择分布:",
          {lam: sum(1 for r in rows if r.get("lam") == lam) for lam in LAMBDA_GRID})


def make_splits(rng, y, probe_n=32, val_n=16):
    pos, neg = np.where(y == 1)[0], np.where(y == -1)[0]
    ptr = rng.choice(pos, probe_n, replace=False)
    ntr = rng.choice(neg, probe_n, replace=False)
    pva = rng.choice(np.setdiff1d(pos, ptr), val_n, replace=False)
    nva = rng.choice(np.setdiff1d(neg, ntr), val_n, replace=False)
    train = np.concatenate([ptr, ntr])
    val = np.concatenate([pva, nva])
    test = np.setdiff1d(np.arange(len(y)), np.concatenate([train, val]))
    return train, val, test


def draw_seeds(rng, y, test, val, ell):
    """从测试集之外的可用节点中抽估计种子（与探针训练/验证/测试均互斥）。"""
    forbidden = np.concatenate([test, val])
    pos = np.setdiff1d(np.where(y == 1)[0], forbidden)
    neg = np.setdiff1d(np.where(y == -1)[0], forbidden)
    # 探针训练集已被排除出 test，此处可用池 = 探针训练集之外的训练侧节点
    return np.concatenate([rng.choice(pos, ell, replace=False), rng.choice(neg, ell, replace=False)])


if __name__ == "__main__":
    main()
