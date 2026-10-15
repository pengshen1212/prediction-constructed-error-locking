"""
命题二第一步实验：关系误判 vs "能量不坍缩 / 分类正确"的分离现象。

设定（对应 research-proposals-2026-09-26.md 第三节"最小可做范围"）：
  - 两类、3-均匀超图 SBM，带噪节点特征（同类吸引 / 异类排斥的真实关系）。
  - 极小传播代理：带符号关系传播 + 每步球面重整化。
    这不是 HAMP/HNRD 的复现，而是它们假设的"类条件相互作用"的最小可分析模型；
    球面重整化扮演深层网络中的归一化层，保证表征有界。
  - 扫描：S1 随机关系翻转率 η × 深度 T；S2 集中于高度数节点的误判（同翻转质量）；
    S3 少标签+带噪特征的关系估计路径；S4 深度效应。
  - 指标：正权重 Dirichlet 能量 E_D（不坍缩指示）、真实/估计关系的带符号能量、
    类内离散度、类间距离、岭探针分类间隔与准确率。
  - 对照：恒等映射、纯噪声、无符号传播（过平滑对照）、原始特征岭探针。

用法：python run_experiment.py [--quick] [--seeds N] [--outdir results]
"""

import argparse
import csv
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- 数据生成：两类 3-均匀 HSBM + 带噪特征 ----------------


def generate_hsbm(rng, n=400, d=8, m=600, mu=1.0, sigma_feat=1.0, p_hom=0.7):
    """节点标签 y∈{−1,+1} 各半并随机排列；特征首维携带类别信号，其余为噪声。
    超边：以 p_hom 概率同类（纯吸引），否则 2+1 混合（含排斥对）。"""
    y = np.concatenate([-np.ones(n // 2), np.ones(n - n // 2)]).astype(int)
    y = y[rng.permutation(n)]
    X = np.empty((n, d))
    X[:, 0] = y * mu + sigma_feat * rng.standard_normal(n)
    X[:, 1:] = sigma_feat * rng.standard_normal((n, d - 1))
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
    return y, X, E


def unordered_pairs(E):
    """每个超边的 3 个无序对 (us, vs)。关系翻转在此层进行，保持相互作用对称。"""
    us = np.concatenate([E[:, 0], E[:, 0], E[:, 1]])
    vs = np.concatenate([E[:, 1], E[:, 2], E[:, 2]])
    return us, vs


def true_relations(y, us, vs):
    return np.where(y[us] == y[vs], 1.0, -1.0)


def flip_random(rng, s_true, eta):
    mask = rng.random(s_true.shape[0]) < eta
    return np.where(mask, -s_true, s_true), mask


def flip_concentrated(y, us, vs, s_true, k):
    """翻转度数最高的 k 个节点的全部关系实例，k 由目标翻转质量反解。"""
    n = y.shape[0]
    deg = np.bincount(us, minlength=n) + np.bincount(vs, minlength=n)
    top = np.argsort(deg)[::-1][:k]
    mask = np.isin(us, top) | np.isin(vs, top)
    return np.where(mask, -s_true, s_true), mask


def concentrated_k_for_eta(y, us, vs, eta):
    """二分求 k 使总翻转质量最接近 eta（单调性：k 越大翻转越多）。"""
    n = y.shape[0]
    deg = np.bincount(us, minlength=n) + np.bincount(vs, minlength=n)
    order = np.argsort(deg)[::-1]
    npairs = us.shape[0]
    lo, hi = 0, n
    best_k = 0
    for _ in range(20):
        mid = (lo + hi) // 2
        top = order[:mid]
        mask = np.isin(us, top) | np.isin(vs, top)
        realized = mask.mean()
        if realized < eta:
            best_k = mid
            lo = mid + 1
        else:
            hi = mid
    return max(best_k, 1)


# ---------------- 传播：带符号关系 + 球面重整化 ----------------


def normalize_rows(Z):
    nrm = np.linalg.norm(Z, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return Z / nrm


def propagate(Z0, us, vs, s_un, T, tau=0.5):
    """a_i = mean over relation instances (i,j): s_ij (z_j - z_i)；
    z <- normalize(z + tau a)。有序实例 = (us,vs)+(vs,us)，关系对称。"""
    n = Z0.shape[0]
    src = np.concatenate([us, vs])
    dst = np.concatenate([vs, us])
    s = np.concatenate([s_un, s_un])
    deg = np.bincount(src, minlength=n).astype(float)
    deg[deg == 0] = 1.0
    Z = normalize_rows(Z0.copy())
    for _ in range(T):
        msg = s[:, None] * (Z[dst] - Z[src]) / 3.0
        agg = np.zeros_like(Z)
        np.add.at(agg, src, msg)
        agg /= deg[:, None]
        Z = normalize_rows(Z + tau * agg)
    return Z


# ---------------- 指标 ----------------


def pair_weights(us, vs, n):
    key = np.minimum(us, vs) * n + np.maximum(us, vs)
    uniq, cnt = np.unique(key, return_counts=True)
    uu = (uniq // n).astype(int)
    vv = (uniq % n).astype(int)
    return uu, vv, cnt.astype(float)  # 每条超边对无序对贡献 1（3 均匀超边内 /3 并入常数）


def energy_positive(Z, uu, vv, ww):
    """正权重 Dirichlet 能量（团展开拉普拉斯二次型 / n，行单位球）。不坍缩指示。"""
    diff2 = ((Z[uu] - Z[vv]) ** 2).sum(1)
    return float((ww * diff2).sum() / Z.shape[0])


def energy_signed(Z, uu_s, vv_s, ww_s, s):
    """带符号能量 Σ w s ||z_i−z_j||² / Σ w。s 用真实关系时，越负 = 类结构对齐越好。"""
    diff2 = ((Z[uu_s] - Z[vv_s]) ** 2).sum(1)
    return float((ww_s * s * diff2).sum() / ww_s.sum())


def class_geometry(Z, y):
    c1, c2 = Z[y == 1].mean(0), Z[y == -1].mean(0)
    intra = 0.5 * (
        (np.linalg.norm(Z[y == 1] - c1, axis=1) ** 2).mean()
        + (np.linalg.norm(Z[y == -1] - c2, axis=1) ** 2).mean()
    )
    return float(np.linalg.norm(c1 - c2)), float(intra)


def ridge_probe(Z, y, train_idx, lam=10.0):
    """线性岭探针（含偏置）。λ=10：深层传播后表征近坍缩（有效秩低），
    弱正则会让最小二乘放大低方差噪声维、失真归一化间隔（诊断见报告）。"""
    A = np.hstack([Z[train_idx], np.ones((len(train_idx), 1))])
    w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y[train_idx])
    wn = np.linalg.norm(w[:-1])

    def evaluate(idx):
        Ai = np.hstack([Z[idx], np.ones((len(idx), 1))])
        f = Ai @ w
        marg = float((y[idx] * f).mean() / wn)
        acc = float(((f > 0) == (y[idx] > 0)).mean())
        return marg, acc

    return evaluate


def all_metrics(Z, y, test_idx, geom):
    uu, vv, ww, uu_s, vv_s, ww_s, s_true_u = geom
    inter, intra = class_geometry(Z, y)
    e_pos = energy_positive(Z, uu, vv, ww)
    e_true = energy_signed(Z, uu_s, vv_s, ww_s, s_true_u)
    return {"E_pos": e_pos, "E_signed_true": e_true, "inter": inter, "intra": intra}


# ---------------- 实验各扫描 ----------------


def run_s1_s4(seeds, etas, Ts, outdir, quick=False):
    """S1：随机翻转 × 深度；S4 是其对 T 的切片。同时记录对照基线。"""
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(1000 + seed)
        y, X, E = generate_hsbm(rng)
        us, vs = unordered_pairs(E)
        s_true = true_relations(y, us, vs)
        n = y.shape[0]
        uu, vv, ww = pair_weights(us, vs, n)
        # 带符号能量用全部无序实例（权重 1）
        geom = (uu, vv, ww, us, vs, np.ones_like(s_true), s_true)

        pos = np.where(y == 1)[0]
        neg = np.where(y == -1)[0]
        train_idx = np.concatenate([rng.choice(pos, 32, replace=False), rng.choice(neg, 32, replace=False)])
        test_idx = np.setdiff1d(np.arange(n), train_idx)

        Z0 = normalize_rows(X)

        def record(tag, Z, eta, T, extra=None):
            ev = ridge_probe(Z, y, train_idx)
            marg, acc = ev(test_idx)
            r = {"exp": tag, "seed": seed, "eta": eta, "T": T,
                 "margin": marg, "acc": acc}
            r.update(all_metrics(Z, y, test_idx, geom))
            if extra:
                r.update(extra)
            rows.append(r)

        # 对照基线（与 T 无关，各记一次）
        record("identity", Z0, 0.0, 0)
        record("noise", normalize_rows(rng.standard_normal(X.shape)), 0.0, 0)
        ev = ridge_probe(X, y, train_idx)  # 无传播：原始特征探针
        marg, acc = ev(test_idx)
        rows.append({"exp": "raw_probe", "seed": seed, "eta": 0.0, "T": 0,
                     "margin": marg, "acc": acc,
                     **all_metrics(Z0, y, test_idx, geom)})

        for T in Ts:
            # 无符号传播（过平滑对照）
            s_unsigned = np.ones_like(s_true)
            record("unsigned", propagate(Z0, us, vs, s_unsigned, T), 0.0, T)
            for eta in etas:
                s_hat, mask = flip_random(rng, s_true, eta)
                realized = float(mask.mean())
                record("random", propagate(Z0, us, vs, s_hat, T), eta, T,
                       {"eta_realized": realized})
        print(f"S1 seed {seed} done")
    _write_csv(os.path.join(outdir, "s1_sweep.csv"), rows)
    return rows


def run_s2(seeds, etas, T, outdir):
    """S2：集中于高度数节点的误判，翻转质量与 S1 网格对齐。"""
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(2000 + seed)
        y, X, E = generate_hsbm(rng)
        us, vs = unordered_pairs(E)
        s_true = true_relations(y, us, vs)
        n = y.shape[0]
        uu, vv, ww = pair_weights(us, vs, n)
        geom = (uu, vv, ww, us, vs, np.ones_like(s_true), s_true)
        pos, neg = np.where(y == 1)[0], np.where(y == -1)[0]
        train_idx = np.concatenate([rng.choice(pos, 32, replace=False), rng.choice(neg, 32, replace=False)])
        test_idx = np.setdiff1d(np.arange(n), train_idx)
        Z0 = normalize_rows(X)
        for eta in etas:
            k = concentrated_k_for_eta(y, us, vs, eta)
            s_hat, mask = flip_concentrated(y, us, vs, s_true, k)
            realized = float(mask.mean())
            Z = propagate(Z0, us, vs, s_hat, T)
            ev = ridge_probe(Z, y, train_idx)
            marg, acc = ev(test_idx)
            r = {"exp": "concentrated", "seed": seed, "eta": eta, "T": T,
                 "margin": marg, "acc": acc, "eta_realized": realized, "k": k}
            r.update(all_metrics(Z, y, test_idx, geom))
            rows.append(r)
        print(f"S2 seed {seed} done")
    _write_csv(os.path.join(outdir, "s2_modes.csv"), rows)
    return rows


def run_s3(seeds, ells, sigmas, T, outdir):
    """S3：真实估计路径——每类 ℓ 个种子标签 + 原型分类器估计节点标签 → 估计关系。
    探针只用种子训练（动力学与评估都不用测试标签设定关系符号）。"""
    rows = []
    for seed in seeds:
        for sigma in sigmas:
            rng = np.random.default_rng(3000 + seed * 10 + int(sigma * 10))
            y, X, E = generate_hsbm(rng, sigma_feat=sigma)
            us, vs = unordered_pairs(E)
            s_true = true_relations(y, us, vs)
            n = y.shape[0]
            uu, vv, ww = pair_weights(us, vs, n)
            geom = (uu, vv, ww, us, vs, np.ones_like(s_true), s_true)
            pos, neg = np.where(y == 1)[0], np.where(y == -1)[0]
            # 固定评估探针（32/类，与种子不相交；只用真标签做评估，不参与关系估计）
            pos_rest = np.setdiff1d(pos, rng.choice(pos, min(max(ells) * 2, len(pos)), replace=False))
            neg_rest = np.setdiff1d(neg, rng.choice(neg, min(max(ells) * 2, len(neg)), replace=False))
            probe64 = np.concatenate([rng.choice(pos_rest, 32, replace=False),
                                      rng.choice(neg_rest, 32, replace=False)])
            for ell in ells:
                train_idx = np.concatenate([rng.choice(pos, ell, replace=False), rng.choice(neg, ell, replace=False)])
                lab_tr = y[train_idx]
                p1 = X[train_idx[lab_tr == 1]].mean(0)
                p2 = X[train_idx[lab_tr == -1]].mean(0)
                y_hat = np.where((X @ (p1 - p2)) >= 0, 1, -1)
                label_err = float((y_hat != y).mean())
                s_est = np.where(y_hat[us] == y_hat[vs], 1.0, -1.0)
                eta_realized = float((s_est != s_true).mean())
                Z0 = normalize_rows(X)
                Z = propagate(Z0, us, vs, s_est, T)
                ev = ridge_probe(Z, y, train_idx)
                test_idx = np.setdiff1d(np.arange(n), train_idx)
                marg, acc = ev(test_idx)
                # 恒等对照（同特征、同种子探针）与固定 64 节点探针（解混淆探针规模 vs 误差相关性）
                ev0 = ridge_probe(Z0, y, train_idx)
                marg0, acc0 = ev0(test_idx)
                evp = ridge_probe(Z, y, probe64)
                marg_p, acc_p = evp(test_idx)
                r = {"exp": "estimated", "seed": seed, "ell": ell, "sigma_feat": sigma,
                     "T": T, "eta_realized": eta_realized, "label_err": label_err,
                     "margin": marg, "acc": acc, "margin_identity": marg0, "acc_identity": acc0,
                     "margin_probe64": marg_p, "acc_probe64": acc_p}
                r.update(all_metrics(Z, y, test_idx, geom))
                rows.append(r)
        print(f"S3 seed {seed} done")
    _write_csv(os.path.join(outdir, "s3_estimation.csv"), rows)
    return rows


# ---------------- 汇总、拟合与作图 ----------------


def _write_csv(path, rows):
    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def agg(rows, key_fields, val_fields=("margin", "acc", "E_pos", "E_signed_true", "inter", "intra", "eta_realized")):
    out = {}
    for r in rows:
        key = tuple(r[k] for k in key_fields)
        out.setdefault(key, {v: [] for v in val_fields if v in r})
        for v in val_fields:
            if v in r:
                out[key][v].append(r[v])
    return {k: {v: (float(np.mean(a)), float(np.std(a))) for v, a in d.items()} for k, d in out.items()}


def fit_linear(eta, gamma, eta_max=0.25):
    m = eta <= eta_max
    if m.sum() < 3:
        return None
    A = np.vstack([eta[m], np.ones(m.sum())]).T
    coef, res, *_ = np.linalg.lstsq(A, gamma[m], rcond=None)
    ss_tot = ((gamma[m] - gamma[m].mean()) ** 2).sum()
    r2 = 1 - (res[0] / ss_tot if len(res) and ss_tot > 0 else 0.0)
    return {"C_rel": float(coef[0]), "gamma0": float(coef[1]), "R2": float(r2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.outdir)
    os.makedirs(outdir, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150,
                         "axes.grid": True, "grid.alpha": 0.3, "font.size": 9})

    if args.quick:
        seeds, etas, Ts = list(range(3)), np.arange(0, 0.51, 0.1), [10, 25]
    else:
        seeds, etas, Ts = list(range(args.seeds)), np.arange(0, 0.51, 0.025), [2, 5, 10, 25, 50]
    T_main = 25

    rows1 = run_s1_s4(seeds, etas, Ts, outdir, args.quick)
    rows2 = run_s2(seeds, etas, T_main, outdir)
    rows3 = run_s3(seeds, ells=[1, 2, 4, 8, 16, 32],
                   sigmas=[0.25, 0.5, 1.0, 2.0], T=T_main, outdir=outdir)

    a1 = agg([r for r in rows1 if r["exp"] == "random"], ["T"])
    baselines = agg([r for r in rows1 if r["exp"] in ("identity", "noise", "unsigned", "raw_probe")], ["exp", "T"])

    summary = {"n_seeds": len(seeds), "T_main": T_main, "fits": {}, "separation": {},
               "baselines": {f"{k[0]}@T{k[1] if len(k) > 1 else 0}":
                             {v: round(m[0], 4) for v, m in d.items()} for k, d in baselines.items()}}

    for T in Ts:
        pts = sorted((eta, d) for (t, eta), d in
                     {(tuple(k)[0], tuple(k)[1]): v for k, v in agg(
                         [r for r in rows1 if r["exp"] == "random"], ["T", "eta"]).items()}.items() if t == T)
        eta = np.array([p[0] for p in pts])
        gam = np.array([p[1]["margin"][0] for p in pts])
        epos = np.array([p[1]["E_pos"][0] for p in pts])
        fit = fit_linear(eta, gam)
        if fit:
            summary["fits"][str(T)] = fit
        # 分离区间：能量仍 ≥ 一半理想值 且 间隔已 ≤ 一半理想值
        if gam[0] > 0 and epos[0] > 0:
            rE, rG = epos / epos[0], gam / gam[0]
            sep = (rE >= 0.5) & (rG <= 0.5)
            if sep.any():
                idx = np.where(sep)[0]
                summary["separation"][str(T)] = [float(eta[idx.min()]), float(eta[idx.max()])]

    with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    _fig1(rows1, a1, baselines, Ts, outdir)
    _fig2_heat(rows1, etas, Ts, outdir)
    _fig3(rows1, rows2, T_main, outdir)
    _fig4(rows3, rows1, outdir)
    print("figures written to", outdir)


def _fig1(rows1, a1, baselines, Ts, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), constrained_layout=True)
    for T in Ts:
        d = a1.get((T,))
        if not d:
            continue
        # 按 eta 排序
        sub = sorted(((r["eta"], r) for r in rows1
                      if r["exp"] == "random" and r["T"] == T), key=lambda t: t[0])
        eta = sorted({r["eta"] for r in rows1 if r["exp"] == "random" and r["T"] == T})
        for ax, field, name in zip(axes, ["E_pos", "margin", "acc"],
                                   ["positive Dirichlet energy $E_D$",
                                    "probe margin $\\gamma_T$", "probe accuracy"]):
            means = {}
            stds = {}
            for r in sub:
                means.setdefault(r[0], []).append(r[1][field])
                stds.setdefault(r[0], []).append(r[1][field])
            xs = sorted(means)
            ys = [np.mean(means[x]) for x in xs]
            es = [np.std(means[x]) for x in xs]
            ax.errorbar(xs, ys, yerr=es, marker="o", ms=2.5, lw=1.2, label=f"T={T}")
            ax.set_xlabel("relation flip rate $\\eta$")
            ax.set_title(name)
    for tag, ls in [("identity", ":"), ("noise", ":")]:
        b = baselines.get((tag, 0)) or baselines.get((tag,))
        if b:
            for ax, field in zip(axes[:2], ["E_pos", "margin"]):
                if field in b:
                    ax.axhline(b[field][0], color="gray", ls=ls, lw=1, label=tag)
    b = baselines.get(("unsigned", 25))
    if b:
        axes[0].axhline(b["E_pos"][0], color="tab:red", ls="--", lw=1, label="unsigned T=25")
    axes[0].legend(fontsize=6, ncol=2)
    axes[1].legend(fontsize=6, ncol=2)
    fig.suptitle("S1: random relation flips vs energy / classification (mean$\\pm$std over seeds)", fontsize=10)
    fig.savefig(os.path.join(outdir, "fig1_eta_sweep.png"))
    plt.close(fig)


def _fig2_heat(rows1, etas, Ts, outdir):
    rnd = [r for r in rows1 if r["exp"] == "random"]
    for field, title in [("E_pos", "$E_D$ (non-collapse)"), ("margin", "margin $\\gamma_T$")]:
        M = np.full((len(Ts), len(etas)), np.nan)
        for i, T in enumerate(Ts):
            for j, eta in enumerate(etas):
                vals = [r[field] for r in rnd if r["T"] == T and abs(r["eta"] - eta) < 1e-9]
                if vals:
                    M[i, j] = np.mean(vals)
        fig, ax = plt.subplots(figsize=(5.5, 3), constrained_layout=True)
        im = ax.imshow(M, aspect="auto", origin="lower", cmap="viridis" if field == "E_pos" else "magma")
        ax.set_xticks(range(len(etas))[::4], [f"{e:.2f}" for e in etas[::4]])
        ax.set_yticks(range(len(Ts)), [str(T) for T in Ts])
        ax.set_xlabel("$\\eta$"), ax.set_ylabel("depth $T$")
        ax.set_title(f"S4 slice: {title} over ($\\eta$, T)")
        fig.colorbar(im)
        fig.savefig(os.path.join(outdir, f"fig2_heat_{field}.png"))
        plt.close(fig)


def _fig3(rows1, rows2, T, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), constrained_layout=True)
    for exp, lab in [("random", "random flips"), ("concentrated", "top-degree flips")]:
        rows = [r for r in (rows1 if exp == "random" else rows2) if r["T"] == T]
        xs, ys, es = [], [], []
        for eta in sorted({r["eta"] for r in rows}):
            vals = [r["margin"] for r in rows if r["eta"] == eta]
            xs.append(eta), ys.append(np.mean(vals)), es.append(np.std(vals))
        for ax, field in zip(axes, ["margin", "acc"]):
            if field == "margin":
                ax.errorbar(xs, ys, yerr=es, marker="o", ms=3, label=lab)
            else:
                xs2, ys2, es2 = [], [], []
                for eta in sorted({r["eta"] for r in rows}):
                    vals = [r[field] for r in rows if r["eta"] == eta]
                    xs2.append(eta), ys2.append(np.mean(vals)), es2.append(np.std(vals))
                ax.errorbar(xs2, ys2, yerr=es2, marker="o", ms=3, label=lab)
    axes[0].set_xlabel("$\\eta$ (target)"), axes[0].set_ylabel("margin"), axes[0].legend()
    axes[1].set_xlabel("$\\eta$ (target)"), axes[1].set_ylabel("accuracy"), axes[1].legend()
    fig.suptitle(f"S2: random vs concentrated misjudgment at matched flip mass (T={T})", fontsize=10)
    fig.savefig(os.path.join(outdir, "fig3_random_vs_concentrated.png"))
    plt.close(fig)


def _fig4(rows3, rows1, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2), constrained_layout=True)
    for ax, xfield in zip(axes, ["label_err", "eta_realized"]):
        for sigma in sorted({r["sigma_feat"] for r in rows3}):
            rows = [r for r in rows3 if r["sigma_feat"] == sigma]
            pts = {}
            for r in rows:
                pts.setdefault((round(r[xfield], 4), r["ell"]), []).append(r["margin"])
            pts = dict(sorted(pts.items()))
            xs = [p[0] for p in pts]
            ys = [np.mean(v) for v in pts.values()]
            ax.plot(xs, ys, "o", ms=3, label=f"$\\sigma$={sigma}")
        ax.set_xlabel("node label error" if xfield == "label_err" else "realized relation flip $\\hat\\eta$")
        ax.set_ylabel("margin (estimated relations)")
    axes[0].legend(fontsize=6)
    # 参照：S1 oracle+random 曲线（T=25）
    ref = [(r["eta"], r["margin"]) for r in rows1 if r["exp"] == "random" and r["T"] == 25]
    if ref:
        et = sorted({e for e, _ in ref})
        gm = [np.mean([g for e, g in ref if e == x]) for x in et]
        axes[1].plot(et, gm, "k--", lw=1, label="S1 controlled flips")
        axes[1].legend(fontsize=6)
    fig.suptitle("S3: estimation path (few labels + noisy features)", fontsize=10)
    fig.savefig(os.path.join(outdir, "fig4_estimation_path.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
