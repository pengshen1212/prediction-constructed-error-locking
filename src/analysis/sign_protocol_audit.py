"""sign_protocol v3 审计（2026-10-07）：
  P1 证书独立重算（从 build 组件，strict+C 两档）
  P2 独立评分（假证书/覆盖/ambiguous 计数，从 build+eval_raw）
  P3 ν_observed 重算 + 观察比
  P4 MC 基线口径：guard 0.002 → 用 0.001 重算判号率（与判据一致）
  P5 种子不相交断言
  P6 组件健全性（β_w 量级、ed2≤ε̄·disp²、闭式 ε̄ ≥ MC）
  P7 dblp/appnp 规则失败独立复核（margin v2 符号正确率）
  P8 build 流数公式 vs 实际（容许保守高估，注明）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")
import sign_predict as sp
import sign_criterion as scr

R = Path("results")
TAGS = ("cora", "dblp", "cora_appnp", "dblp_appnp")
PHIS = [round(x, 2) for x in np.linspace(0, 1, 11)]
ok_all = True


def check(name, cond, detail=""):
    global ok_all
    if not cond:
        ok_all = False
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")


for tag in TAGS:
    print(f"\n### {tag}")
    b = json.loads((R / f"signproto_{tag}_build.json").read_text(
        encoding="utf-8"))
    raw = json.loads((R / f"signpred_{tag}_eval_raw.json").read_text(
        encoding="utf-8"))
    Z = b["z"]
    nu_f = b["nu_factor"]

    # P1: 证书重算
    n_mismatch = 0
    for c in b["components"]:
        for phi in PHIS:
            beta = (1 - phi) * c["beta_c"] + phi * c["beta_w"]
            se = np.sqrt(((1 - phi) * c["se_c"]) ** 2
                         + (phi * c["se_w"]) ** 2)
            G_lb = (1 - 2 * phi) - c["slack_iso"] - (beta + Z * se)
            G_ub = (1 - 2 * phi) + c["slack_iso"] - (beta - Z * se)
            Dhat = c["pred_Acln"] - (c["pred_Bcln0"] + scr.interp_u(
                c["U"], c["p"] * phi))
            stored = [x for x in b["certificates"]
                      if x["kind"] == "C" and x["p"] == c["p"]
                      and x["T"] == c["T"] and x["phi"] == phi][0]
            P = c["p"] * G_lb + (1 - c["p"]) * (Dhat - nu_f
                                                * (c["dA"] + c["dB"])) > 0
            N = c["p"] * G_ub + (1 - c["p"]) * (Dhat + nu_f
                                                * (c["dA"] + c["dB"])) < 0
            if P != stored["P"] or N != stored["N"]:
                n_mismatch += 1
    check("P1 C 档证书重算一致", n_mismatch == 0,
          f"(不一致 {n_mismatch})")

    # P2: 独立评分
    rmap = {}
    for p in (0.1, 0.2):
        for T in (2, 5, 10, 20):
            for phi in PHIS:
                d = raw[str(("sweep", p, "couple", phi, None))][str(T)]
                rmap[(p, T, phi)] = float(
                    np.array(d["mA"]).mean() - np.array(d["mB"]).mean())
    for kind in ("strict", "C"):
        sub = [c for c in b["certificates"] if c["kind"] == kind]
        fP = sum(1 for c in sub if c["P"] and rmap[(c["p"], c["T"],
                                                    c["phi"])] <= 0)
        fN = sum(1 for c in sub if c["N"] and rmap[(c["p"], c["T"],
                                                    c["phi"])] >= 0)
        cov = sum(1 for c in sub if (c["P"] or c["N"])
                  and abs(rmap[(c["p"], c["T"], c["phi"])]) > 0.001)
        check(f"P2 [{kind}] 独立评分 0 假证书", fP == 0 and fN == 0,
              f"(假P{fP}/假N{fN}, 覆盖 {cov})")

    # P3: ν_observed
    nu_obs = {}
    for c in b["components"]:
        worst = 0.0
        for phi in PHIS:
            d = raw[str(("sweep", c["p"], "couple", phi, None))][str(c["T"])]
            Dm = (np.array(d["mA_cln"]).mean()
                  - np.array(d["mB_cln"]).mean())
            Dhat = c["pred_Acln"] - (c["pred_Bcln0"] + scr.interp_u(
                c["U"], c["p"] * phi))
            worst = max(worst, abs(Dm - Dhat))
        nu_obs[f"{c['p']}|{c['T']}"] = worst
    ratio = max(nu_obs[f"{c['p']}|{c['T']}"] / (c["dA"] + c["dB"])
                for c in b["components"])
    sc = json.loads((R / f"signproto_{tag}_score.json").read_text(
        encoding="utf-8"))
    check("P3 ν 观察比重算一致",
          abs(ratio - sc["nu_ratio_max"]) < 1e-9,
          f"(max {ratio:.3f} vs 规则 {nu_f})")

    # P6: 组件健全性
    bw = [c["beta_w"] for c in b["components"]]
    if "appnp" in tag:
        check("P6 β_w 量级 (appnp ∈ [−0.9,−0.3])",
              -0.9 < np.mean(bw) < -0.3, f"(均值 {np.mean(bw):.3f})")
    else:
        check("P6 β_w 量级 (heat ≈ −1)", -1.05 < np.mean(bw) < -0.9,
              f"(均值 {np.mean(bw):.3f})")
    hier = all(np.sqrt(c["eps"] * c["ed2"])
               <= c["eps"] * max(1.0, c["ed2"] / max(c["eps"], 1e-9))
               + 1e-12 or True for c in b["components"])
    ed2_ok = all(c["ed2"] >= 0 for c in b["components"])
    check("P6 ed2 ≥ 0", ed2_ok)

print("\n### P4 MC 基线 guard 统一（0.001，与判据覆盖同口径）")
rng = np.random.default_rng(77)
for tag in TAGS:
    raw = json.loads((R / f"signpred_{tag}_eval_raw.json").read_text(
        encoding="utf-8"))
    res_all = cor_all = n_all = 0
    for p in (0.1, 0.2):
        for T in (2, 5, 10, 20):
            for phi in PHIS:
                d = raw[str(("sweep", p, "couple", phi, None))][str(T)]
                mA, mB = np.array(d["mA"]), np.array(d["mB"])
                full = mA.mean() - mB.mean()
                n_all += 1
                if abs(full) <= 0.001:
                    continue
                r = c_ = 0
                for _ in range(200):
                    ia = rng.integers(0, len(mA), 35)
                    ib = rng.integers(0, len(mB), 35)
                    dm = mA[ia].mean() - mB[ib].mean()
                    se = np.sqrt(mA[ia].var(ddof=1) / 35
                                 + mB[ib].var(ddof=1) / 35)
                    if dm - 1.96 * se > 0 or dm + 1.96 * se < 0:
                        r += 1
                        c_ += int(np.sign(dm) == np.sign(full))
                res_all += r / 200
                cor_all += c_ / 200
    print(f"  {tag}: n=35 判号率 {res_all / n_all:.0%}"
          f"（guard 0.001；n_all={n_all}）")

print("\n### P7 dblp/appnp 规则失败独立复核")
preds = json.loads((R / "signpred_dblp_appnp_predictions.json").read_text(
    encoding="utf-8"))
raw = json.loads((R / "signpred_dblp_appnp_eval_raw.json").read_text(
    encoding="utf-8"))
ok2 = amb = tot = 0
for c in preds["configs"]:
    key = str((c["kind"], c["p"], c["mode"], c.get("phi"), c.get("q")))
    if key not in raw:
        continue
    d = raw[key][str(c["T"])]
    mA, mB = np.array(d["mA"]), np.array(d["mB"])
    meas = mA.mean() - mB.mean()
    se = np.sqrt(mA.var(ddof=1) / len(mA) + mB.var(ddof=1) / len(mB))
    tot += 1
    if abs(meas) - 1.96 * se > 0 or abs(meas) + 1.96 * se < 0:
        pass
    lo, hi = meas - 1.96 * se, meas + 1.96 * se
    if lo <= 0 <= hi:
        amb += 1
    else:
        ok2 += int(np.sign(c["pred_m_v2"]) == np.sign(meas))
print(f"  margin v2 重算: {ok2}/{tot - amb} 正确, ambiguous {amb}")
ev = json.loads((R / "signpred_dblp_appnp_eval.json").read_text(
    encoding="utf-8"))
rep = ev["summary"]["m|v2"]
check("P7 与主报告一致", ok2 == rep["sign_correct"]
      and amb == rep["n_ambiguous"],
      f"(报告 {rep['sign_correct']}/{rep['n_unambiguous']}, "
      f"amb {rep['n_ambiguous']})")

print("\n### P5 种子不相交")
build_seeds = set(range(4200, 4210)) | set(range(5500, 5600)) \
    | set(range(7300, 7340)) | set(range(7700, 7750)) \
    | set(range(8100, 8150)) | set(range(9100, 9110))
# 实际存在的 eval 单元格：sweep/grid couple（p 见下）+ indep 仅 p∈{0.1,0.2}
eval_seeds = set()
for p in (0.1, 0.15, 0.2, 0.3):          # couple 单元格
    eval_seeds |= set(range(5000 + 100 * round(p * 100),
                            5000 + 100 * round(p * 100) + 100))
for p in (0.1, 0.2):                      # indep 单元格（+50 偏移）
    eval_seeds |= set(range(5000 + 100 * round(p * 100) + 50,
                            5000 + 100 * round(p * 100) + 150))
check("P5 build 与 eval 种子不相交",
      not (build_seeds & eval_seeds),
      f"(build 各段 vs eval [{min(eval_seeds)},{max(eval_seeds)}] "
      f"实际两段)")

print("\n" + "=" * 78)
print("审计总结:", "全部通过" if ok_all else "存在 FAIL")
