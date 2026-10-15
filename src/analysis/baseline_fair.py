"""MC 基线公平性核对 + 三方比较（评审第 2 条，2026-10-07）。

修正三处公平性：
  1. 复用成本模型：直接 MC 覆盖 sweep 网格时，每实现只需 1 对 A/B
     算子构建 + 11 条流（每个 φ 一条，全部深度 T 免费记录）≈ 13 单位；
     网格含 2 个 p → 匹配预算 n = build 流数 / 26。
  2. 判号/覆盖统一在同一配置集（88 个 sweep 配置、|Δ|>0.001 非平凡
     子集）与同一 guard；bootstrap（配置 × 抽样）给覆盖差的区间。
  3. 加入冻结规则 v1/v2 作为基线（校准成本 ≈ 6 格点 × 100 实现 ×
     每实现 A/B 两算子两流 ≈ 2400 单位）；三方各自标注保证类型。

输出：results/baseline_fair.json + 控制台表。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

R = Path("results")
TAGS = ("cora", "dblp", "cora_appnp", "dblp_appnp")
PHIS = [round(x, 2) for x in np.linspace(0, 1, 11)]
TS = (2, 5, 10, 20)
PS = (0.1, 0.2)
N_DRAWS = 200
N_BOOT = 2000


def sweep_measures(raw):
    out = {}
    for p in PS:
        for T in TS:
            for phi in PHIS:
                d = raw[str(("sweep", p, "couple", phi, None))][str(T)]
                out[(p, T, phi)] = dict(
                    mA=np.array(d["mA"]), mB=np.array(d["mB"]),
                    meas=float(np.array(d["mA"]).mean()
                               - np.array(d["mB"]).mean()))
    return out


def main():
    rng = np.random.default_rng(31337)
    result = {}
    for tag in TAGS:
        build = json.loads((R / f"signproto_{tag}_build.json").read_text(
            encoding="utf-8"))
        ev = json.loads((R / f"signpred_{tag}_eval.json").read_text(
            encoding="utf-8"))
        raw = json.loads((R / f"signpred_{tag}_eval_raw.json").read_text(
            encoding="utf-8"))
        preds = json.loads((R / f"signpred_{tag}_predictions.json")
                           .read_text(encoding="utf-8"))
        M = sweep_measures(raw)
        pmap = {(c["kind"], c["p"], c["T"], c["phi"], c["mode"]): c
                for c in preds["configs"]}
        unamb = {k for k, v in M.items() if abs(v["meas"]) > 0.001}

        # 判据（C 档）覆盖：逐配置指示
        cert = {k: False for k in M}
        for c in build["certificates"]:
            if c["kind"] == "C":
                cert[(c["p"], c["T"], c["phi"])] = c["P"] or c["N"]

        # 冻结规则 v1/v2（sweep 配置上的准确率；规则必判号）
        acc_rule = {}
        for v in ("v1", "v2"):
            ok = 0
            for k in unamb:
                c = pmap[("sweep", k[0], k[1], k[2], "couple")]
                ok += int(np.sign(c[f"pred_m_{v}"]) == np.sign(M[k]["meas"]))
            acc_rule[v] = ok / len(unamb)

        # 直接 MC：匹配预算（复用成本 26 单位/实现；数据上限 100）
        crit_flows = build["approx_flow_count"]
        n_fair = crit_flows // 26
        n_use = min(100, max(5, n_fair))
        res = {}
        for k in unamb:
            mA, mB = M[k]["mA"], M[k]["mB"]
            r = 0
            for _ in range(N_DRAWS):
                ia = rng.integers(0, len(mA), n_use)
                ib = rng.integers(0, len(mB), n_use)
                dm = mA[ia].mean() - mB[ib].mean()
                se = np.sqrt(mA[ia].var(ddof=1) / n_use
                             + mB[ib].var(ddof=1) / n_use)
                r += int(dm - 1.96 * se > 0 or dm + 1.96 * se < 0)
            res[k] = r / N_DRAWS

        keys = sorted(unamb)
        cov_c = np.array([cert[k] for k in keys], float)
        cov_m = np.array([res[k] for k in keys], float)
        # bootstrap：配置有放回重采样（判据覆盖与 MC 判号率配对同抽）
        diffs = []
        for _ in range(N_BOOT):
            idx = rng.integers(0, len(keys), len(keys))
            diffs.append(cov_c[idx].mean() - cov_m[idx].mean())
        lo, hi = np.quantile(diffs, [.025, .975])
        row = dict(
            criterion_flows=crit_flows, n_fair_uncapped=n_fair,
            mc_n_used=n_use,
            criterion_C_coverage=float(cov_c.mean()),
            mc_resolution=float(cov_m.mean()),
            diff_ci=[float(lo), float(hi)],
            rule_v1=acc_rule["v1"], rule_v2=acc_rule["v2"],
            rule_flows=2400,
            n_unambiguous=len(keys))
        result[tag] = row
        print(f"== {tag}（非平凡 {len(keys)}/88）")
        print(f"   判据 C 档覆盖 {cov_c.mean():.0%}（build {crit_flows} 流）"
              f" | 直接 MC {cov_m.mean():.0%}（匹配 n={n_fair}"
              f"{'，受数据上限用 n=' + str(n_use) if n_fair > 100 else ''}）")
        print(f"   覆盖差（判据−MC）95% CI [{lo:+.3f}, {hi:+.3f}]")
        print(f"   冻结规则（必判号）: v1 {acc_rule['v1']:.0%}, "
              f"v2 {acc_rule['v2']:.0%}（校准 ~2400 流）")
    Path("results/baseline_fair.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print("→ results/baseline_fair.json")


if __name__ == "__main__":
    main()
