"""统一冻结确认协议 v3（2026-10-07，评审意见第 2/3 条落实）。

两级阶段，取代 sign_criterion.py 的混合流程：
  build  组件与证书只来自 oracle 图 + A-only/B-only 模拟 + 闭式量，
         **从不读取任何 A−B 差异**（eval_raw 一律不碰）→ 先落盘。
  score  事后读取 eval_raw，只做评价：健全性、覆盖、夹逼、ν_S 观察值
         vs ν_rule（透明性对照，不回写规则）。

统计保证（口径修正，2026-10-07 评审）：
  - MC 构件用单侧 CLT 上界 est + z·SE，z=3.09。**这是渐近置信界**，
    非有限样本上界（Hoeffding 在 n=100、R=1 时 t≈0.37，空泛，故保留
    CLT 实现）；证书相应称为**渐近条件证书**。
  - 覆盖范围：全网格共 26 个独立 MC 估计量（逐 p 的 ε̄×2 + 逐 (p,T)
    的 β_c/β_w/ed₂×24）；z=3.09 单侧名义 0.001/量 → **全网格名义
    （渐近）α ≈ 2.6%**。不主张有限样本总体上界；
  - 数学结构：在「构件误差界成立」前提下 (P)/(N) 蕴含符号结论
    （定理，无概率成分）；
  - MC 层失效概率 ≤ α_total；模型误差层（D 的曲线模型）为**条件假设
    (S')**：ν_rule = NU_FACTOR·(δ_Acln+δ_Bcln)，逐 (p,T) D-free，
    NU_FACTOR=0.25 冻结自三个开发设置（观察比 ≤0.17）——确认集
    不参与该常数选择；
  - δ_Acln/δ_Bcln 的验证用 A-only/B-only 模拟批次：这是**需要
    A/B 模拟的审计**（audit-mode），非纯先验预测，如实标注。

证书档：
  strict：ν = δ_Acln+δ_Bcln（三角不等式，无 (S')，仅 MC 层保证）
  C     ：ν = ν_rule（(S') 条件 + MC 层）

用法：python sign_protocol.py --data cora [--operator appnp] --stage build|score|all
输出：results/signproto_{tag}_{build,score}.json
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
import sign_criterion as scr

CHECKPOINTS = (2, 5, 10, 20)
PHI_GRID = [round(x, 2) for x in np.linspace(0, 1, 11)]
SWEEP_P = (0.1, 0.2)
Z = 3.09          # 单侧 99.9%；Bonferroni 4 构件 → α_total ≤ 0.004
NU_FACTORS = {"heat": 0.25, "appnp": 0.7}  # 按算子族冻结自开发设置：
# heat 两设置观察比 ≤0.159×1.6；appnp 设置 0.538×1.3
N_BETA, N_AUDIT, N_EPS, N_ED2 = 100, 50, 40, 50


def mc_eps_se(ds, p, n_sim=N_EPS, seed=7300):
    rng = np.random.default_rng(seed)
    rates = []
    for _ in range(n_sim):
        pred = sp.corrupt_labels(ds.labels, ds.C, p, rng)
        corrupted = pred != ds.labels
        wA = ds.all_hom(pred)
        alive = np.zeros(ds.n, dtype=bool)
        alive[ds.rows[wA[ds.owner] > 0]] = True
        rates.append(float((alive & corrupted).sum() / corrupted.sum()))
    a = np.array(rates)
    return float(a.mean()), float(a.std() / np.sqrt(len(a)))


def sim_clean(ds, p, seed=7700, n_sim=N_AUDIT):
    """A-only / B-only 审计批次（不含 A−B 比较）：
    mA_cln（A 图，couple 初始化；A 侧与 φ 无关）；mB_cln 在 φ=0/1
    两端点（B 图 + 对应初始化）。一次流扫全 checkpoint。"""
    w0 = ds.expected_da(p)
    q_rate, sgn = np.abs(w0), np.sign(w0)
    P = sp.onehot(ds.labels, ds.C)
    outA = {t: [] for t in CHECKPOINTS}
    outB = {t: {0.0: [], 1.0: []} for t in CHECKPOINTS}
    for r in range(n_sim):
        rng = np.random.default_rng(seed + r)
        pred = sp.corrupt_labels(ds.labels, ds.C, p, rng)
        corrupted = pred != ds.labels
        Z0c = sp.onehot(pred, ds.C)
        stA = sp.flow_stats(sp.op_of(ds, ds.all_hom(pred)), Z0c, ds.h,
                            CHECKPOINTS, ds.labels)
        for t in CHECKPOINTS:
            outA[t].append(float(stA[t][0][~corrupted].mean()))
        flip = rng.random(ds.m) < q_rate
        wB = ds.a_star + np.where(flip, sgn, 0.0)
        B = sp.op_of(ds, wB)
        for phi, Z0 in ((0.0, P), (1.0, Z0c)):
            st = sp.flow_stats(B, Z0, ds.h, CHECKPOINTS, ds.labels)
            for t in CHECKPOINTS:
                outB[t][phi].append(float(st[t][0][~corrupted].mean()))
    def agg(d):
        return {t: (float(np.mean(v)),
                    float(np.std(v) / np.sqrt(len(v)))) for t, v in d.items()}
    return agg(outA), {t: {phi: (float(np.mean(v)),
                                 float(np.std(v) / np.sqrt(len(v))))
                           for phi, v in d.items()}
                       for t, d in outB.items()}


def build(ds, tag):
    out = {"tag": tag, "operator": sp.OP["mode"], "alpha": sp.OP["alpha"],
           "n": ds.n, "C": ds.C, "m": ds.m, "z": Z,
           "nu_factor": NU_FACTORS[sp.OP["mode"]], "built": time.strftime("%Y-%m-%d %H:%M:%S")}
    t0 = time.time()
    S, Tcur, base_m, hom_bins, het_bins = scr.oracle_shapes(ds, seed=4200)
    n_curve_flows = (len(hom_bins) + len(het_bins)) * 6 * 24
    print(f"曲线 {time.time()-t0:.0f}s", flush=True)

    comps = []
    for p in SWEEP_P:
        eps, eps_se = mc_eps_se(ds, p)
        mom, disp = scr.a_moments(ds, p)
        ph, rA_hom, rA_het = scr.retention_table(ds, p)
        rB_h = [float((1 - ph)[b].sum()) for b in hom_bins]
        rA_h = [float((1 - rA_hom)[b].sum()) for b in hom_bins]
        rB_e = [float(ph[b].sum()) for b in het_bins]
        rA_e = [float(rA_het[b].sum()) for b in het_bins]
        U = {T: scr.u_curve(ds, p, T) for T in CHECKPOINTS}
        mA, mB2 = sim_clean(ds, p)
        for T in CHECKPOINTS:
            beta_c, se_c, beta_w, se_w = scr.sim_betas(ds, p, T)
            pred_Bcln0 = base_m[T] \
                + sum(scr.interp_bin(S[T][j], rB_h[j])
                      for j in range(len(hom_bins))) \
                + sum(scr.interp_bin(Tcur[T][j], rB_e[j])
                      for j in range(len(het_bins)))
            pred_Acln = base_m[T] \
                + sum(scr.interp_bin(S[T][j], rA_h[j])
                      for j in range(len(hom_bins))) \
                + sum(scr.interp_bin(Tcur[T][j], rA_e[j])
                      for j in range(len(het_bins)))
            u1 = scr.interp_u(U[T], p)
            pred_Bcln1 = pred_Bcln0 + u1
            ed2, ed2_se = mom[T]
            eps_up = eps + Z * eps_se
            ed2_up = ed2 + Z * ed2_se
            slack_iso = float(np.sqrt(eps_up * ed2_up))
            dA = abs(mA[T][0] - pred_Acln) + Z * mA[T][1]
            dB = max(abs(mB2[T][0.0][0] - pred_Bcln0),
                     abs(mB2[T][1.0][0] - pred_Bcln1)) + Z * mB2[T][0.0][1]
            comps.append(dict(
                p=p, T=T, eps=eps, eps_se=eps_se, ed2=ed2, ed2_se=ed2_se,
                slack_iso=slack_iso, beta_c=beta_c, se_c=se_c,
                beta_w=beta_w, se_w=se_w, dA=dA, dB=dB,
                nu_rule=NU_FACTORS[sp.OP["mode"]] * (dA + dB), nu_strict=dA + dB,
                pred_Acln=pred_Acln, pred_Bcln0=pred_Bcln0,
                pred_Bcln1=pred_Bcln1, U=U[T], mA_cln=mA[T][0],
                mB_cln0=mB2[T][0.0][0], mB_cln1=mB2[T][1.0][0]))
        print(f"  组件 p={p} 完成 ({time.time()-t0:.0f}s)", flush=True)

    # 证书（两档）：D̂(φ) = pred_A − (pred_Bcln0 + Û(pφ))，ν 按档取
    certs = []
    for c in comps:
        for phi in PHI_GRID:
            beta = (1 - phi) * c["beta_c"] + phi * c["beta_w"]
            se = float(np.sqrt(((1 - phi) * c["se_c"]) ** 2
                               + (phi * c["se_w"]) ** 2))
            dBeta = Z * se
            G_lb = (1 - 2 * phi) - c["slack_iso"] - (beta + dBeta)
            G_ub = (1 - 2 * phi) + c["slack_iso"] - (beta - dBeta)
            Dhat = c["pred_Acln"] - (c["pred_Bcln0"] + scr.interp_u(
                c["U"], c["p"] * phi))
            for kind, nu in (("strict", c["nu_strict"]),
                             ("C", c["nu_rule"])):
                P_fire = c["p"] * G_lb + (1 - c["p"]) * (Dhat - nu) > 0
                N_fire = c["p"] * G_ub + (1 - c["p"]) * (Dhat + nu) < 0
                certs.append(dict(kind=kind, p=c["p"], T=c["T"], phi=phi,
                                  P=bool(P_fire), N=bool(N_fire)))
    out["components"] = comps
    out["certificates"] = certs
    out["approx_flow_count"] = int(
        n_curve_flows + 2 * N_BETA * len(comps) + 2 * N_AUDIT * 3 * 2
        + len(comps) * 7 * 8 + N_EPS * 2 + N_ED2 * 2)
    out["build_seconds"] = time.time() - t0
    Path(f"results/signproto_{tag}_build.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"build 完成 {out['approx_flow_count']} 流 "
          f"({out['build_seconds']:.0f}s) → signproto_{tag}_build.json")
    return out


def score(tag):
    b = json.loads(Path(f"results/signproto_{tag}_build.json").read_text(
        encoding="utf-8"))
    raw = json.loads(Path(
        f"results/signpred_{tag}_eval_raw.json").read_text(encoding="utf-8"))
    rows = []
    nu_obs = {}
    for c in b["components"]:
        p, T = c["p"], c["T"]
        # 观察值 ν_S（仅透明性对照；不进规则）
        ds_ = []
        for phi in PHI_GRID:
            d = raw[str(("sweep", p, "couple", phi, None))][str(T)]
            Dm = (np.array(d["mA_cln"]).mean()
                  - np.array(d["mB_cln"]).mean())
            Dhat = (c["pred_Acln"] - (c["pred_Bcln0"] + scr.interp_u(
                c["U"], p * phi)))
            ds_.append(abs(Dm - Dhat))
        nu_obs[f"{p}|{T}"] = float(max(ds_))
        for phi in PHI_GRID:
            d = raw[str(("sweep", p, "couple", phi, None))][str(T)]
            meas = float(np.array(d["mA"]).mean() - np.array(d["mB"]).mean())
            rows.append(dict(p=p, T=T, phi=phi, meas=meas))
    rmap = {(r["p"], r["T"], r["phi"]): r["meas"] for r in rows}
    report = {"tag": tag, "nu_observed": nu_obs,
              "nu_ratio_max": max(
                  nu_obs[f"{c['p']}|{c['T']}"] / (c["dA"] + c["dB"])
                  for c in b["components"])}
    for kind in ("strict", "C"):
        sub = [c for c in b["certificates"] if c["kind"] == kind]
        nP = nN = falseP = falseN = amb_cert = 0
        unamb = cov = 0
        for c in sub:
            m = rmap[(c["p"], c["T"], c["phi"])]
            fired = c["P"] or c["N"]
            if abs(m) > 0.001:
                unamb += 1
                if fired:
                    cov += 1
            else:
                amb_cert += int(fired)
            nP += c["P"]
            nN += c["N"]
            falseP += c["P"] and m <= 0
            falseN += c["N"] and m >= 0
        report[f"summary_{kind}"] = dict(
            nP=nP, nN=nN, falseP=int(falseP), falseN=int(falseN),
            cert_on_ambiguous=amb_cert, n_unambiguous=unamb, covered=cov,
            coverage=(cov / unamb if unamb else None))
        print(f"[{kind}] P{nP}/N{nN}, 假P{int(falseP)}/假N{int(falseN)}, "
              f"ambiguous 上证书 {amb_cert}, 覆盖 {cov}/{unamb}")
    # 夹逼报告
    brackets = []
    for p in SWEEP_P:
        for T in CHECKPOINTS:
            sub = [c for c in b["certificates"] if c["kind"] == "C"
                   and c["p"] == p and c["T"] == T]
            Pphis = [c["phi"] for c in sub if c["P"]]
            Nphis = [c["phi"] for c in sub if c["N"]]
            if Pphis and Nphis:
                width = min(Nphis) - max(Pphis)
            else:
                width = None
            ev = json.loads(Path(
                f"results/signpred_{tag}_eval.json").read_text(
                encoding="utf-8"))
            phi_meas = next(t["phi_meas"] for t in ev["transitions"]
                            if t["p"] == p and t["T"] == T)
            inside = None
            if Pphis and Nphis and phi_meas is not None:
                inside = max(Pphis) <= phi_meas <= min(Nphis)
            brackets.append(dict(p=p, T=T, P_hi=max(Pphis) if Pphis else None,
                                 N_lo=min(Nphis) if Nphis else None,
                                 width=width, phi_meas=phi_meas,
                                 bracket_contains_meas=inside))
    report["brackets"] = brackets
    Path(f"results/signproto_{tag}_score.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("ν 观察比 max =", round(report["nu_ratio_max"], 3),
          f'(规则常数 {b["nu_factor"]})')
    for br in brackets:
        print(f"  p={br['p']} T={br['T']}: P上界{br['P_hi']} "
              f"N下界{br['N_lo']} 带宽{br['width']} "
              f"实测φ*={br['phi_meas'] and round(br['phi_meas'],3)} "
              f"包含={br['bracket_contains_meas']}")
    print(f"→ signproto_{tag}_score.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="cora", choices=list(sp.DATA_DIRS))
    ap.add_argument("--operator", default="heat", choices=["heat", "appnp"])
    ap.add_argument("--stage", default="all", choices=["build", "score",
                                                       "all"])
    args = ap.parse_args()
    sp.OP["mode"] = args.operator
    sp.OP["alpha"] = 0.9
    ds = sp.Dataset(args.data)
    tag = args.data + (f"_{args.operator}" if args.operator != "heat" else "")
    print(f"协议 v3 build/score: {tag}", flush=True)
    if args.stage in ("build", "all"):
        build(ds, tag)
    if args.stage in ("score", "all"):
        score(tag)


if __name__ == "__main__":
    main()
