"""直接 Monte Carlo 基线 vs 判据：同预算对比（评审第 5 条）。

问题：既然已经跑 A-only/B-only 模拟，为什么不直接估计 Δ 的均值差
与置信区间？本脚本在相同模拟预算下比较两种方法：

  直接 MC：每配置 n 对实现，mean ± 1.96·SE，CI 不含 0 → 判号。
  判据   ：一次 build（曲线+β+审计+Û，流数计入 signproto build）
           后全网格 (p,T,φ) 证书免费。

口径：sweep 88 配置（8 (p,T) × 11 φ）；预算以"流的次數"计。
直接 MC 的 n 对应预算 n×88；判据预算 = build 的 approx_flow_count。
输出 results/mcbaseline.json + 控制台表。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

SWEEP_P = (0.1, 0.2)
T_GRID = (2, 5, 10, 20)
PHIS = [round(x, 2) for x in np.linspace(0, 1, 11)]
N_CONF = 200


def mc_resolution(raw, n, rng):
    """每配置 n 对实现的判号率/正确率（200 次子采样平均）。"""
    resolved = correct = 0
    n_cfg = 0
    for p in SWEEP_P:
        for T in T_GRID:
            d0 = raw[str(("sweep", p, "couple", 0.0, None))][str(T)]
            ref = None
            for phi in PHIS:
                d = raw[str(("sweep", p, "couple", phi, None))][str(T)]
                mA, mB = np.array(d["mA"]), np.array(d["mB"])
                full = mA.mean() - mB.mean()
                n_cfg += 1
                if abs(full) <= 0.002:
                    continue          # 真值近零：判号无意义，不计数
                res = cor = 0
                for _ in range(N_CONF):
                    ia = rng.integers(0, len(mA), n)
                    ib = rng.integers(0, len(mB), n)
                    dm = mA[ia].mean() - mB[ib].mean()
                    se = np.sqrt(mA[ia].var(ddof=1) / n
                                 + mB[ib].var(ddof=1) / n)
                    lo, hi = dm - 1.96 * se, dm + 1.96 * se
                    if lo > 0 or hi < 0:
                        res += 1
                        cor += int(np.sign(dm) == np.sign(full))
                resolved += res / N_CONF
                correct += cor / N_CONF
    # 返回：所有配置平均判号率；判号时的正确率
    return resolved / n_cfg, (correct / resolved if resolved else None)


def main():
    rng = np.random.default_rng(2026)
    out = {}
    for tag in ("cora", "dblp", "cora_appnp", "dblp_appnp"):
        raw = json.loads(Path(
            f"results/signpred_{tag}_eval_raw.json").read_text(
            encoding="utf-8"))
        build = json.loads(Path(
            f"results/signproto_{tag}_build.json").read_text(
            encoding="utf-8"))
        crit_flows = build["approx_flow_count"]
        rows = {"criterion_flows": crit_flows}
        # 与判据同预算的直接 MC：n = crit_flows / 88
        n_matched = max(3, crit_flows // 88)
        for n in (5, 10, 20, 50, 100, n_matched):
            res, acc = mc_resolution(raw, n, rng)
            rows[f"mc_n{n}"] = dict(resolution=round(res, 3),
                                    accuracy=None if acc is None
                                    else round(acc, 3))
        sc = json.loads(Path(
            f"results/signproto_{tag}_score.json").read_text(
            encoding="utf-8"))
        rows["criterion_C_coverage"] = sc["summary_C"]["coverage"]
        rows["criterion_strict_coverage"] = sc["summary_strict"]["coverage"]
        rows["criterion_falseP_N"] = [sc["summary_C"]["falseP"],
                                      sc["summary_C"]["falseN"]]
        out[tag] = rows
        print(f"== {tag} (判据 build {crit_flows} 流; 同预算直接 MC "
              f"n={n_matched}/配置)")
        for k, v in rows.items():
            if k.startswith("mc_n"):
                print(f"   {k}: 判号率 {v['resolution']:.0%}, "
                      f"正确率 {'—' if v['accuracy'] is None else format(v['accuracy'], '.0%')}")
        print(f"   判据 C 档覆盖 {rows['criterion_C_coverage']:.0%}, "
              f"strict 覆盖 "
              f"{rows['criterion_strict_coverage'] or 0:.0%}, "
              f"假证书 {rows['criterion_falseP_N']}")
    Path("results/mcbaseline.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("→ results/mcbaseline.json")


if __name__ == "__main__":
    main()
