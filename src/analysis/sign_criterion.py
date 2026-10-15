"""可计算双例判据（8 分线尝试，v2）：隔离收益 vs 清洁损失的充分条件。

输入约束（评审要求）：判据只依赖实例可计算量——图结构、信道参数
(p, φ, T, C)、传播算子类（热流 Z←(I−hB)Z, h≤1/λmax(oracle)）。
**任何 A/B 下游差异（实测或校准）都不进入判据**；A-only/B-only
量仅用于逐模型验证松弛，符号结论只在最终对照。

结构（margin）：
  Δ = p·G + (1−p)·D
  G = E[m^A|cor] − E[m^B|cor]
      ├─ E[m^A|cor] ∈ (1−2φ) ± 2R_m·ε̄   [可证：Lemma self-isolation；
      │                                    ε̄ = 并集上界，闭式]
      └─ E[m^B|cor] ≈ (1−φ)β_c^MF + φβ_w^MF
           β_c^MF：Ē_B（期望算子，权重 ph 闭式）上 truth 初始化流
           β_w^MF：Ē_B 上**探针流**——确定性子集（比例 p）置为代表
                    错误类 one-hot（(y+1) mod C），其余 truth；探针
                    成员均值 margin。保持 one-hot 结构（margin 泛函与
                    行混合不对易）；邻居 90% 正确，条件结构正确。
  D = E[m^A−m^B|clean] ≈ D^MF = Ŝ(r_A) − Ŝ(r_B) + T̂(r_A^het) − T̂(r_B^het)
      S(r)：oracle 图随机移除 r 条 hom 边的均值 margin 变化（>0，
      饱和曲线，r 至 256）；T(r)：随机加入 r 条 het 边（<0）。
      r_B = Σ_hom(1−ph)（B 稀释更多），r_A = Σ_hom(1−(1−p)^{k−1})；
      het 侧 r_B^het = Σ_het ph（B 重连），r_A^het ≈ 0。
      竞争 Ŝ(r_B)−Ŝ(r_A) > |T̂(r_B^het)| ⟺ D<0（A 的干净节点受损）。

双例充分条件（松弛逐模型验证：δ_B = max|E[m^B|cor]−β^MF|（B-only）；
δ_cln = 线性/形状模型对 A-only、B-only 干净均值误差之和）：
  (P) p·[(1−2φ) − 2ε̄ − (β^MF+δ_B)] > (1−p)·(−D^MF + δ_cln)  ⟹ Δ>0
  (N) p·[(1−2φ) + 2ε̄ − (β^MF−δ_B)] + (1−p)·(D^MF − δ_cln) < 0 ⟹ Δ<0

输出：results/signcriterion_{ds}.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")
import sign_predict as sp

CHECKPOINTS = (2, 5, 10, 20)
PHI_GRID = [round(x, 2) for x in np.linspace(0, 1, 11)]
SWEEP_P = (0.1, 0.2)
R_FRACS = (0.004, 0.008, 0.016, 0.031, 0.0625, 0.125, 0.25, 0.4, 0.5)


def edge_bins(ds, mask, n_bins=5):
    """按边大小 k 分箱。先验实例规则（不依赖任何结果量）：独立 k 值
    ≤4（窄分布，如 cora k∈{2..5}）→ 单箱（等价全局曲线，避免纯噪声）；
    否则分位分箱（宽分布，如 dblp k∈{2..202}，成分修正是主导项）。"""
    ks = ds.k[mask]
    idx = np.flatnonzero(mask)
    if len(np.unique(ks)) <= 4:
        return [idx]
    qs = np.quantile(ks, np.linspace(0, 1, n_bins + 1)[1:-1])
    bins, lo = [], -np.inf
    for q in np.append(qs, np.inf):
        b = idx[(ks > lo) & (ks <= q)]
        if len(b):
            bins.append(b)
        lo = q
    return bins


def oracle_shapes(ds, seed, n_reps=24):
    """k 分层响应曲线：箱 j 内移除 r 条随机 hom 边（S_j>0）/加入 r 条
    随机 het 边（T_j<0）的 Δ均值 margin。大边(k→202)与小边影响迥异，
    全局曲线在 dblp 上失真（δ_Acln 曾达 0.126）；跨箱叠加误差由逐模型
    δ 与 ν_S 直接度量。算子跟随 sp.OP。"""
    rng = np.random.default_rng(seed)
    Z0 = sp.onehot(ds.labels, ds.C)
    base = sp.flow_stats(sp.op_of(ds, ds.a_star), Z0, ds.h, CHECKPOINTS,
                         ds.labels)
    base_m = {t: base[t][0].mean() for t in CHECKPOINTS}
    hom_bins = edge_bins(ds, ds.a_star == 1.0)
    het_bins = edge_bins(ds, ds.a_star == 0.0)
    FR = (0.008, 0.03, 0.0625, 0.125, 0.25, 0.5)

    def bin_curve(bins, w_val, sign):
        out = {t: {j: {} for j in range(len(bins))}
               for t in CHECKPOINTS}
        for j, bidx in enumerate(bins):
            rgrid = sorted({max(1, int(round(len(bidx) * f)))
                            for f in FR})
            for r in rgrid:
                vals = {t: [] for t in CHECKPOINTS}
                for _ in range(n_reps):
                    w = ds.a_star.copy()
                    w[rng.choice(bidx, size=min(r, len(bidx)),
                                 replace=False)] = w_val
                    st = sp.flow_stats(sp.op_of(ds, w), Z0, ds.h,
                                       CHECKPOINTS, ds.labels)
                    for t in CHECKPOINTS:
                        vals[t].append(st[t][0].mean() - base_m[t])
                for t in CHECKPOINTS:
                    out[t][j][r] = float(np.mean(vals[t]))
        return out

    S = bin_curve(hom_bins, 0.0, +1)
    Tc = bin_curve(het_bins, 1.0, -1)
    return S, Tc, base_m, hom_bins, het_bins


def interp_bin(curve, r):
    rs = np.array(sorted(curve), dtype=float)
    vs = np.array([curve[float(x)] for x in rs])
    return float(np.interp(np.log(max(r, 1.0)), np.log(rs), vs))


def interp_shape(curve, r):
    """log-r 线性插值；r 超出网格端点取端点值。"""
    rs = np.array(sorted(curve), dtype=float)
    vs = np.array([curve[float(x)] for x in rs])
    return float(np.interp(np.log(max(r, 1.0)), np.log(rs), vs))


def retention_table(ds, p):
    """逐边保留率。ph_e = P[â_e=1] = Σ_c (1−p)^{n_c} ρ^{k−n_c}（用实际
    类计数——同质边退化为 (1−p)^k+(C−1)ρ^k；混合边必须按计数算，
    不能套同质公式）。复用已验证的 expected_da = ph − a*。"""
    rho = p / (ds.C - 1)
    ph = ds.expected_da(p) + ds.a_star
    return ph, np.power(1 - p, ds.k - 1), np.power(rho, ds.k - 1)


def eps_bar(ds, p):
    """ε̄：u 腐蚀时 A 下仍有幸存边的概率（并集上界）。条件于
    pred_u=c：边幸存 ⟺ 全部 mate 预测 = c，P = exp(logm[c])/ρ。"""
    rho = p / (ds.C - 1)
    cnt = np.zeros((ds.m, ds.C))
    np.add.at(cnt, (ds.owner, ds.labels[ds.rows]), 1)
    cnt /= ds.k[:, None]
    logm = cnt * np.log(1 - p) + (ds.k[:, None] - cnt) * np.log(rho)
    row = np.exp(logm - np.log(rho))
    eps = np.zeros(ds.n)
    for i, e in enumerate(ds.edges):
        labs = ds.labels[np.asarray(e)]
        s_all = row[i].sum()
        for u, y_u in zip(e, labs):
            eps[u] += (s_all - row[i, y_u]) / (ds.C - 1)
    return float(eps.mean())


def mf_betas(ds, p, T, probe_seed=77):
    """均值场 β。β_c：truth 流；β_w：探针流（确定性子集比例 p）。"""
    ph, _, _ = retention_table(ds, p)
    EB = ds.build_op(ph)
    st = sp.flow_stats(EB, sp.onehot(ds.labels, ds.C), ds.h, [T],
                       ds.labels)
    beta_c = float(st[T][0].mean())
    rng = np.random.default_rng(probe_seed)
    S = rng.choice(ds.n, size=max(1, int(round(ds.n * p))),
                   replace=False)
    P = np.zeros((ds.n, ds.C))
    P[np.arange(ds.n), ds.labels] = 1.0
    Z0 = P.copy()
    Z0[S] = np.roll(P[S], 1, axis=1)          # 代表错误类 (y+1) mod C
    m = sp.flow_stats(EB, Z0, ds.h, [T], ds.labels)[T][0]
    return beta_c, float(m[S].mean())


def a_moments(ds, p, n_sim=50, seed=8100):
    """A-only 位移矩：ed2(p,T) = E[δ² | W=1]（δ = m^A_T(u) − m_u(Z0)，
    couple 初始化最保守），附 SE 与 max disp（报告用）。
    Cauchy–Schwarz：|E[δ·1{¬iso}]| ≤ sqrt(P(¬iso)·E[δ²]) = sqrt(ε̄·ed2)；
    ed2 ≤ ε̄·disp² 时退化回旧界 ε̄·disp，层级自洽。"""
    sums = {t: [] for t in CHECKPOINTS}
    disp = {t: 0.0 for t in CHECKPOINTS}
    for r in range(n_sim):
        rng = np.random.default_rng(seed + r)
        pred = sp.corrupt_labels(ds.labels, ds.C, p, rng)
        corrupted = pred != ds.labels
        Z0 = sp.onehot(pred, ds.C)
        init_m = np.where(corrupted, -1.0, 1.0)
        st = sp.flow_stats(sp.op_of(ds, ds.all_hom(pred)), Z0, ds.h,
                           CHECKPOINTS, ds.labels)
        for t in CHECKPOINTS:
            d2 = (st[t][0] - init_m)[corrupted] ** 2
            sums[t].append(float(d2.mean()))
            disp[t] = max(disp[t], float(np.abs(
                st[t][0] - init_m).max()))
    out = {}
    for t in CHECKPOINTS:
        a = np.array(sums[t])
        out[t] = (float(a.mean()), float(a.std() / np.sqrt(len(a))))
    return out, disp


def a_disp(ds, p, n_sim=30, seed=8100):
    """旧版：仅 max 位移（保留兼容，报告用）。"""
    out, disp = a_moments(ds, p, n_sim=n_sim, seed=seed)
    return disp


def sim_betas(ds, p, T, n_sim=100, seed=5500):
    """B-only 蒙特卡洛 β：直接模拟模型 B 的动力学（不建 A、不做 A−B
    比较）。β_c：truth 初始化的 B 总体均值 margin；β_w：耦合初始化下
    B 在腐蚀节点上的 margin。δ 用 B-only 实测（eval_raw 的 mB_cor）
    验证，与均值场版同一认识论结构。"""
    w0 = ds.expected_da(p)
    q_rate, sgn = np.abs(w0), np.sign(w0)
    P = sp.onehot(ds.labels, ds.C)
    bc, bw = [], []
    for r in range(n_sim):
        rng = np.random.default_rng(seed + r)
        flip = rng.random(ds.m) < q_rate
        wB = ds.a_star + np.where(flip, sgn, 0.0)
        B = sp.op_of(ds, wB)
        bc.append(float(sp.flow_stats(B, P, ds.h, [T], ds.labels)[T][0]
                        .mean()))
        pred = sp.corrupt_labels(ds.labels, ds.C, p, rng)
        Z0 = sp.onehot(pred, ds.C)
        m = sp.flow_stats(B, Z0, ds.h, [T], ds.labels)[T][0]
        bw.append(float(m[pred != ds.labels].mean()))
    return (float(np.mean(bc)), float(np.std(bc) / np.sqrt(n_sim)),
            float(np.mean(bw)), float(np.std(bw) / np.sqrt(n_sim)))


def mc_eps(ds, p, n_sim=40, seed=7300):
    """ε̄ 的条件蒙特卡洛（A-only 信道模拟）：P(u 在 A 下有边 | u 腐蚀)。"""
    rng = np.random.default_rng(seed)
    hit = ncor = 0
    for _ in range(n_sim):
        pred = sp.corrupt_labels(ds.labels, ds.C, p, rng)
        corrupted = pred != ds.labels
        wA = ds.all_hom(pred)
        alive = np.zeros(ds.n, dtype=bool)
        alive[ds.rows[wA[ds.owner] > 0]] = True
        hit += int((alive & corrupted).sum())
        ncor += int(corrupted.sum())
    return hit / ncor


def u_curve(ds, p, T, fracs=(0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2)):
    """Û(f)：把 f 比例节点翻成代表错误类，未翻转节点的均值 margin
    变化。热流：确定性期望算子 Ē_B（权重 ph）；APPNP：逐次采样二元
    边集（每边以 ph 保留）配对测量。D 的 φ 依赖全部来自 B 侧错误
    质量流入（A 侧腐蚀邻居被隔离、自环使 SZ0=Z0 不传播）。"""
    ph, _, _ = retention_table(ds, p)
    rng = np.random.default_rng(9100)
    P = np.zeros((ds.n, ds.C))
    P[np.arange(ds.n), ds.labels] = 1.0
    out = {}
    for f in fracs:
        vals = []
        for rep in range(8):
            if sp.OP["mode"] == "appnp":
                kept = (rng.random(ds.m) < ph).astype(float)
                B = sp.op_of(ds, kept)
                base = sp.flow_stats(B, P, ds.h, [T], ds.labels)[T][0]
            else:
                B = ds.build_op(ph)
                base = sp.flow_stats(B, P, ds.h, [T], ds.labels)[T][0]
            S = rng.choice(ds.n, size=max(1, int(round(ds.n * f))),
                           replace=False)
            Z0 = P.copy()
            # 随机错误类（≠y_u 逐节点独立抽取）：固定平移类会让同类
            # 错误邻居总推同一类、接收端 max-other 质量过度集中
            # （APPNP 下 ~17% 过冲的机制）
            for u in S:
                off = rng.integers(1, ds.C)
                Z0[u] = np.roll(P[u], int(off))
            m = sp.flow_stats(B, Z0, ds.h, [T], ds.labels)[T][0]
            mask = np.ones(ds.n, dtype=bool)
            mask[S] = False
            vals.append(float(m[mask].mean() - base[mask].mean()))
        out[f] = float(np.mean(vals))
    return out


def interp_u(u, f):
    items = sorted((float(k), v) for k, v in u.items())
    fs = np.array([k for k, _ in items])
    vs = np.array([v for _, v in items])
    return float(np.interp(max(f, 0.0), fs, vs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="cora", choices=list(sp.DATA_DIRS))
    ap.add_argument("--operator", default="heat", choices=["heat", "appnp"])
    ap.add_argument("--alpha", type=float, default=0.9)
    args = ap.parse_args()
    sp.OP["mode"] = args.operator
    sp.OP["alpha"] = args.alpha
    ds = sp.Dataset(args.data)
    tag = args.data + (f"_{args.operator}"
                       if args.operator != "heat" else "")
    out = {"dataset": args.data, "operator": args.operator,
           "alpha": sp.OP["alpha"],
           "n": ds.n, "C": ds.C, "m": ds.m, "h": ds.h}
    print(f"数据 {tag}: n={ds.n} C={ds.C} m={ds.m} h={ds.h:.4f} "
          f"算子={args.operator}", flush=True)

    t0 = time.time()
    S, Tcur, base_m, hom_bins, het_bins = oracle_shapes(ds, seed=4200)
    print(f"oracle 分层形状曲线完成 ({time.time()-t0:.0f}s): "
          f"{len(hom_bins)} hom 箱 / {len(het_bins)} het 箱", flush=True)
    out["S_shape"] = {str(t): {str(j): {str(r): v for r, v in c.items()}
                               for j, c in S[t].items()}
                      for t in CHECKPOINTS}
    out["T_shape"] = {str(t): {str(j): {str(r): v for r, v in c.items()}
                               for j, c in Tcur[t].items()}
                      for t in CHECKPOINTS}
    out["hom_bin_sizes"] = [int(len(b)) for b in hom_bins]
    out["het_bin_sizes"] = [int(len(b)) for b in het_bins]
    out["m_oracle"] = base_m

    raw = json.loads(Path(
        f"results/signpred_{tag}_eval_raw.json").read_text(encoding="utf-8"))

    rows = []
    d_Acln = d_Bcln = 0.0
    d_B_phi = {phi: 0.0 for phi in PHI_GRID}      # β 松弛按 φ 分层（B-only）
    nu_S = {}                                      # (p,T) 假设(S)松弛
    slack_by_pT = {}
    for p in SWEEP_P:
        epsb_closed = eps_bar(ds, p)
        epsb = mc_eps(ds, p)
        mom, disp = a_moments(ds, p)
        out.setdefault("eps_bar", {})[f"{p}"] = epsb
        out.setdefault("eps_bar_closed", {})[f"{p}"] = epsb_closed
        out.setdefault("a_disp", {})[f"{p}"] = {str(t): v for t, v
                                                in disp.items()}
        out.setdefault("a_ed2", {})[f"{p}"] = {
            str(t): v for t, v in mom.items()}
        for T in CHECKPOINTS:
            ed2, se2 = mom[T]
            ed2_eff = ed2 + 3 * se2          # MC 保守化
            slack_by_pT[(p, T)] = float(
                np.sqrt(epsb * ed2_eff))
        ph, rA_hom, rA_het = retention_table(ds, p)
        hom = ds.a_star == 1.0
        # 逐箱期望移除/重连数（闭式）
        rB_h = [float((1 - ph)[b].sum()) for b in hom_bins]
        rA_h = [float((1 - rA_hom)[b].sum()) for b in hom_bins]
        rB_e = [float(ph[b].sum()) for b in het_bins]
        rA_e = [float(rA_het[b].sum()) for b in het_bins]
        out.setdefault("r_counts", {})[f"{p}"] = dict(
            rB_hom=sum(rB_h), rA_hom=sum(rA_h), rB_het=sum(rB_e),
            rA_het=sum(rA_e))
        U = {T: u_curve(ds, p, T) for T in CHECKPOINTS}
        out.setdefault("u_curves", {})[f"{p}"] = {
            str(T): v for T, v in U.items()}
        for T in CHECKPOINTS:
            beta_c, se_c, beta_w, se_w = sim_betas(ds, p, T)
            if sp.OP["mode"] == "appnp":
                beta_c_mf, beta_w_mf = beta_c, beta_w   # MF 仅热流有定义
            else:
                beta_c_mf, beta_w_mf = mf_betas(ds, p, T)
            pred_Bcln = base_m[T] \
                + sum(interp_bin(S[T][j], rB_h[j])
                      for j in range(len(hom_bins))) \
                + sum(interp_bin(Tcur[T][j], rB_e[j])
                      for j in range(len(het_bins)))
            pred_Acln = base_m[T] \
                + sum(interp_bin(S[T][j], rA_h[j])
                      for j in range(len(hom_bins))) \
                + sum(interp_bin(Tcur[T][j], rA_e[j])
                      for j in range(len(het_bins)))
            d_thin = pred_Acln - pred_Bcln
            nu_S[(p, T)] = 0.0
            for phi in PHI_GRID:
                # φ 依赖：B 的干净节点在错误邻居背景下衰减更慢（其
                # margin 上升 Û>0），而 A 侧腐蚀邻居被隔离不变 →
                # D(φ) = D_thin − Û(pφ)（注意负号：mB_cln 升 → D 降）
                u_phi = interp_u(U[T], p * phi)
                d_mf = d_thin - u_phi
                pred_Bcln_phi = pred_Bcln + u_phi
                d = raw[str(("sweep", p, "couple", phi, None))][str(T)]
                mA = np.array(d["mA"]).mean()
                mB = np.array(d["mB"]).mean()
                mB_cor = np.array(d["mB_cor"]).mean()
                mA_cln = np.array(d["mA_cln"]).mean()
                mB_cln = np.array(d["mB_cln"]).mean()
                beta_mf = (1 - phi) * beta_c + phi * beta_w
                d_B_phi[phi] = max(d_B_phi[phi], abs(mB_cor - beta_mf))
                d_Acln = max(d_Acln, abs(mA_cln - pred_Acln))
                d_Bcln = max(d_Bcln, abs(mB_cln - pred_Bcln_phi))
                D_meas = mA_cln - mB_cln
                nu_S[(p, T)] = max(nu_S[(p, T)], abs(D_meas - d_mf))
                rows.append(dict(
                    p=p, T=T, phi=phi, meas=float(mA - mB),
                    beta_c_mf=beta_c_mf, beta_w_mf=beta_w_mf,
                    beta_c=beta_c, beta_w=beta_w,
                    beta_se=max(se_c, se_w), beta_mf=beta_mf,
                    d_thin=d_thin, d_mf=d_mf,
                    pred_Acln=pred_Acln, pred_Bcln=pred_Bcln,
                    mA_cln=float(mA_cln), mB_cln=float(mB_cln),
                    mB_cor=float(mB_cor), D_meas=float(D_meas)))
    out["delta_B_by_phi"] = {f"{k}": v for k, v in d_B_phi.items()}
    out["delta_Acln"] = float(d_Acln)
    out["delta_Bcln"] = float(d_Bcln)
    out["nu_S"] = {f"{k[0]}|{k[1]}": v for k, v in nu_S.items()}
    out["rows"] = rows
    print(f"β 松弛(按φ): φ=0 {d_B_phi[0.0]:.4f}, φ=0.5 "
          f"{d_B_phi[0.5]:.4f}, φ=1 {d_B_phi[1.0]:.4f}")
    print(f"严格逐模型: δ_Acln={d_Acln:.4f}, δ_Bcln={d_Bcln:.4f}")
    print(f"假设(S)松弛 ν_S: max={max(nu_S.values()):.4f}, "
          f"mean={np.mean(list(nu_S.values())):.4f}")

    # ---- 双例证书：N 必须用 D 的上界（Δ ≤ p·G_ub + (1−p)(D̂+ν) < 0）；
    # G 侧隔离松弛 = sqrt(ε̄·E[δ²])（Cauchy–Schwarz，A-only 二阶矩）
    d_cln_strict = d_Acln + d_Bcln
    out["slack_iso"] = {f"{k[0]}|{k[1]}": v for k, v
                        in slack_by_pT.items()}
    cert = []
    for r in rows:
        p, phi = r["p"], r["phi"]
        slack_iso = slack_by_pT[(p, r["T"])]
        dB = d_B_phi[phi]
        G_lb = (1 - 2 * phi) - slack_iso - (r["beta_mf"] + dB)
        G_ub = (1 - 2 * phi) + slack_iso - (r["beta_mf"] - dB)
        for kind, dslack in (("S", nu_S[(p, r["T"])]),
                             ("strict", d_cln_strict)):
            D_lb = r["d_mf"] - dslack
            D_ub = r["d_mf"] + dslack
            P_fire = p * G_lb + (1 - p) * D_lb > 0
            N_fire = p * G_ub + (1 - p) * D_ub < 0
            cert.append(dict(kind=kind, p=p, T=r["T"], phi=phi,
                             P=bool(P_fire), N=bool(N_fire),
                             meas=r["meas"], slack_iso=slack_iso))
    out["d_cln_strict"] = d_cln_strict
    out["certificates"] = cert
    for kind in ("strict", "S"):
        sub = [c for c in cert if c["kind"] == kind]
        nP = sum(1 for c in sub if c["P"])
        nN = sum(1 for c in sub if c["N"])
        falseP = sum(1 for c in sub if c["P"] and c["meas"] <= 0)
        falseN = sum(1 for c in sub if c["N"] and c["meas"] >= 0)
        unamb = sum(1 for c in sub if abs(c["meas"]) > 0.001)
        cov = sum(1 for c in sub if (c["P"] or c["N"])
                  and abs(c["meas"]) > 0.001)
        covP = sum(1 for c in sub if c["P"] and abs(c["meas"]) > 0.001)
        covN = sum(1 for c in sub if c["N"] and abs(c["meas"]) > 0.001)
        out[f"summary_{kind}"] = dict(
            nP=nP, nN=nN, falseP=falseP, falseN=falseN,
            n_unambiguous=unamb, certified=cov, certified_P=covP,
            certified_N=covN,
            coverage=(cov / unamb if unamb else None))
        print(f"[{kind}] 证书: P {nP} / N {nN}; 假证书 P {falseP}, "
              f"N {falseN}; 非平凡覆盖 {cov}/{unamb} "
              f"(P {covP}, N {covN})")

    Path(f"results/signcriterion_{tag}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"已写 results/signcriterion_{tag}.json")
    for kind in ("strict", "S"):
        print(f"-- {kind} 版转折带（P 带 | N 带，按 (p,T)）--")
        for p in SWEEP_P:
            for T in CHECKPOINTS:
                sub = [c for c in cert if c["kind"] == kind
                       and c["p"] == p and c["T"] == T]
                Pphis = [c["phi"] for c in sub if c["P"]]
                Nphis = [c["phi"] for c in sub if c["N"]]
                print(f"  p={p} T={T}: P [{min(Pphis) if Pphis else '—'},"
                      f"{max(Pphis) if Pphis else '—'}] | N "
                      f"[{min(Nphis) if Nphis else '—'},"
                      f"{max(Nphis) if Nphis else '—'}]")


if __name__ == "__main__":
    main()
