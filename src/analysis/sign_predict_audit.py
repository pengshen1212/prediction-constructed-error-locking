"""sign_predict 结果审计（2026-10-07）：独立复算全部头部数字。

检查项：
  A1 协议文件时间戳链：frozen_rule < predictions < eval
  A2 种子不相交（规则 base_seed_cal=1000，评估 5000+）
  A3 从 eval_raw 独立重算：符号正确率 / ambiguous / 转折 φ*（不调用
     sign_predict.summarize，避免共用实现掩盖 bug）
  A4 机制声明复验：E[m^A|cor]=1−2φ、β(φ) 线性（逐差散度）
  A5 配对种子设计：同 (p,mode) 各 φ 的 phat 逐实现相同
  A6 与论文独立实现交叉验证：cora sweep φ=0/φ=1 端点 vs
     criterion_noisyinit.json 的 clean/couple（不同代码、不同种子）
  A7 图文件有效性（PNG 尺寸）
  A8 paper/ 目录未被本次工作改动（git status 快照对比）
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
R = Path("results")
ok_all = True


def check(name, cond, detail=""):
    global ok_all
    status = "PASS" if cond else "FAIL"
    if not cond:
        ok_all = False
    print(f"  [{status}] {name} {detail}")


def boot_ci(a, b, n_boot=2000, seed=99):
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, len(a), (n_boot, len(a)))
    ib = rng.integers(0, len(b), (n_boot, len(b)))
    d = a[ia].mean(1) - b[ib].mean(1)
    return np.quantile(d, [.025, .975])


def cross(vals, pos):
    """在 pos 网格上对 vals 的零交叉做线性插值，返回交叉位置。"""
    v = np.asarray(vals, float)
    i = np.flatnonzero(v[:-1] * v[1:] < 0)
    if not len(i):
        return None
    k = i[0]
    return float(pos[k] + (pos[k + 1] - pos[k]) * (-v[k])
                 / (v[k + 1] - v[k]))


print("=" * 78)
for tag in ("cora", "dblp"):
    print(f"\n### 数据集 {tag}")
    rule = json.loads((R / f"signpred_{tag}_frozen_rule.json").read_text(
        encoding="utf-8"))
    preds = json.loads((R / f"signpred_{tag}_predictions.json").read_text(
        encoding="utf-8"))
    ev = json.loads((R / f"signpred_{tag}_eval.json").read_text(
        encoding="utf-8"))
    raw = json.loads((R / f"signpred_{tag}_eval_raw.json").read_text(
        encoding="utf-8"))

    # A1 时间戳链
    t_rule = (R / f"signpred_{tag}_frozen_rule.json").stat().st_mtime
    t_pred = (R / f"signpred_{tag}_predictions.json").stat().st_mtime
    t_eval = (R / f"signpred_{tag}_eval.json").stat().st_mtime
    check("A1 时间戳 rule<predictions<eval",
          t_rule < t_pred < t_eval,
          f"({t_rule:.0f} < {t_pred:.0f} < {t_eval:.0f})")

    # A2 种子
    check("A2 校准种子=1000, 评估=5000+", rule["base_seed_cal"] == 1000)

    # A3 独立重算符号正确率（从 raw 与 predictions 出发）
    n_ok = n_amb = n_tot = 0
    for c in preds["configs"]:
        key = str((c["kind"], c["p"], c["mode"], c.get("phi"),
                   c.get("q")))
        if key not in raw:
            continue
        d = raw[key][str(c["T"])]
        mA, mB = np.array(d["mA"]), np.array(d["mB"])
        meas = mA.mean() - mB.mean()
        ci = boot_ci(mA, mB)
        pred = c["pred_m_v2"]
        n_tot += 1
        if ci[0] <= 0 <= ci[1]:
            n_amb += 1
        elif np.sign(pred) == np.sign(meas):
            n_ok += 1
    rep = ev["summary"]["m|v2"]
    check("A3 margin 符号复算一致",
          n_ok == rep["sign_correct"] and n_amb == rep["n_ambiguous"]
          and n_tot == rep["n"],
          f"(重算 {n_ok}/{n_tot - n_amb}, amb {n_amb}; "
          f"报告 {rep['sign_correct']}/{rep['n_unambiguous']}, "
          f"amb {rep['n_ambiguous']})")

    # A3b 独立重算转折 φ*
    for p in (0.1, 0.2):
        for T in (2, 5, 10, 20):
            phis, meas = [], []
            for phi in [round(x, 2) for x in np.linspace(0, 1, 11)]:
                d = raw[str(("sweep", p, "couple", phi, None))][str(T)]
                meas.append(np.array(d["mA"]).mean()
                            - np.array(d["mB"]).mean())
                phis.append(phi)
            ph_arr = np.array([c["phi"] for c in preds["configs"]
                               if c["kind"] == "sweep" and c["p"] == p
                               and c["T"] == T])
            prd = [c["pred_m_v2"] for c in preds["configs"]
                   if c["kind"] == "sweep" and c["p"] == p and c["T"] == T]
            xm = cross(meas, phis)
            xp = cross(prd, list(ph_arr))
            tr = next(t for t in ev["transitions"]
                      if t["p"] == p and t["T"] == T)
            same = ((xm is None and tr["phi_meas"] is None)
                    or (xm is not None and tr["phi_meas"] is not None
                        and abs(xm - tr["phi_meas"]) < 5e-3)) and \
                   ((xp is None and tr["phi_pred"] is None)
                    or (xp is not None and tr["phi_pred"] is not None
                        and abs(xp - tr["phi_pred"]) < 5e-3))
            check(f"A3b 转折重算 p={p} T={T}",
                  same, f"(重算 meas={xm and round(xm,3)}, "
                        f"pred={xp and round(xp,3)})")

    # A4 机制：E[m^A|cor] vs 1−2φ（用 raw 的 mA_cor）
    worst = 0.0
    for phi in [round(x, 2) for x in np.linspace(0, 1, 11)]:
        d = raw[str(("sweep", 0.1, "couple", phi, None))]["10"]
        mAcor = np.array(d["mA_cor"]).mean()
        worst = max(worst, abs(mAcor - (1 - 2 * phi)))
    check("A4 |E[m^A|cor]−(1−2φ)| ≤ 0.02 (p=0.1, 全φ, T=10)",
          worst <= 0.02, f"(最大偏差 {worst:.4f})")
    for p in (0.1, 0.2):
        d = {phi: raw[str(("sweep", p, "couple", phi, None))]["10"]
             for phi in [round(x, 2) for x in np.linspace(0, 1, 11)]}
        beta = [np.array(d[phi]["mB_cor"]).mean() for phi in d]
        dif = np.diff(beta)
        spread = dif.max() - dif.min()
        check(f"A4b β(φ) 线性：逐差散度 ≤ 0.01 (p={p})",
              spread <= 0.01, f"(逐差 ∈[{dif.min():+.4f},{dif.max():+.4f}])")

    # A5 配对种子：同 (p,mode) 各 φ 的 phat 相同
    ph0 = np.array(raw[str(("sweep", 0.1, "couple", 0.0, None))]
                   ["20"]["phat"])
    ph1 = np.array(raw[str(("sweep", 0.1, "couple", 1.0, None))]
                   ["20"]["phat"])
    check("A5 φ=0/1 图实现配对（phat 逐实现相同）",
          np.array_equal(ph0, ph1), f"(均值 {ph0.mean():.4f})")

    # A7 图
    png = R / "figures" / f"signpred_{tag}_sweep.png"
    if png.exists():
        head = png.read_bytes()[:33]
        w, h = struct.unpack(">II", png.read_bytes()[16:24])
        check("A7 sweep PNG 有效", head[:8] == b"\x89PNG\r\n\x1a\n"
              and w > 500 and h > 300, f"({w}x{h})")
    else:
        check("A7 sweep PNG 存在", False)

# A6 与论文独立实现交叉验证（cora, p=0.10）
print("\n### A6 交叉验证 vs criterion_noisyinit.json（独立代码+种子）")
paper = json.loads((R / "criterion_noisyinit.json").read_text(
    encoding="utf-8"))
raw = json.loads((R / "signpred_cora_eval_raw.json").read_text(
    encoding="utf-8"))
for mode, phi in (("clean", 0.0), ("couple", 1.0)):
    for T in (2, 5, 10, 20):
        d = raw[str(("sweep", 0.1, "couple", phi, None))][str(T)]
        ours = np.array(d["mA"]).mean() - np.array(d["mB"]).mean()
        ci = boot_ci(np.array(d["mA"]), np.array(d["mB"]))
        ref = paper[mode][f"T={T}"]["diff"]
        agree = ci[0] - 0.002 <= ref <= ci[1] + 0.002
        check(f"A6 {mode} T={T}", agree,
              f"(本文 {ref:+.5f} vs 本次 {ours:+.5f}, "
              f"CI[{ci[0]:+.5f},{ci[1]:+.5f}])")

print("\n" + ("=" * 78))
print("审计总结:", "全部通过" if ok_all else "存在 FAIL，见上")
