"""干预实验终版分析（评审第八轮口径修正）：

统一表述 V_τ = V0 + τ·C1，τ = 两条不同超边共享预测副本的概率。
实现 = 中间 K∈{1..32} 独立选组（τ=1/K 恰成立）+ 末端点直接独立
Bernoulli（τ=0；"m" 与 "B" 为同一点的两个独立估计）。
拟合只用 6 个分组点；截距（τ→0 外推）与两个 τ=0 估计对照检验。
谱范数只主张：样本均值单调下降、接近独立通道水平（附 MC 误差区
间）——二阶仿射律不移植到谱范数。

输出：修正拟合表 + figures/depint_final.png（双面板：归一化二阶
超额 vs τ 与理论线 y=τ；归一化中心化谱范数 vs τ 及 B 基线）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

R = Path("results")
FILES = {"synthetic": "depint.json", "cora": "depint_cora.json",
         "dblp": "depint_dblp.json"}
K_GROUP = (1, 2, 4, 8, 16, 32)          # 分组点（τ=1/K）


def load():
    out = []
    for name, fn in FILES.items():
        d = json.loads((R / fn).read_text(encoding="utf-8"))["results"]
        for key, r in d.items():
            out.append((name, key, r))
    return out


def main():
    combos = load()
    print("=== 修正拟合（6 个分组点 τ=1/K；外推 τ=0 对照两独立估计）===")
    table = []
    for name, key, r in combos:
        xs = np.array([1.0 / k for k in K_GROUP])
        ys = np.array([r["curves"][str(k)]["fc2"] for k in K_GROUP])
        cis = np.array([r["curves"][str(k)]["fc2_ci"] for k in K_GROUP])
        A = np.vstack([np.ones_like(xs), xs]).T
        coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
        r2 = 1 - float(np.sum((ys - A @ coef) ** 2)) / float(
            np.sum((ys - ys.mean()) ** 2))
        tau0_a = r["curves"]["m"]["fc2"]
        tau0_b = r["curves"]["B"]["fc2"]
        table.append(dict(name=name, key=key, V0=r["V0"], C1=r["C1"],
                          intercept=coef[0], slope=coef[1], r2=r2,
                          tau0_indep=tau0_a, tau0_B=tau0_b,
                          xs=xs.tolist(), ys=ys.tolist(), cis=cis.tolist()))
        print(f"{name:>9} {key}: 截距 {coef[0]:9.2f} (V0 {r['V0']:9.2f}) "
              f"斜率 {coef[1]:9.2f} (C1 {r['C1']:9.2f}) R²={r2:.4f} | "
              f"τ=0: 独立 {tau0_a:9.2f} / B {tau0_b:9.2f} vs 外推截距 "
              f"{coef[0]:9.2f}")
    # 图
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    colors = {"synthetic": "C0", "cora": "C1", "dblp": "C2"}
    for row in table:
        c = colors[row["name"]]
        xs, ys = np.array(row["xs"]), np.array(row["ys"])
        cis = np.array(row["cis"])
        excess = (ys - row["V0"]) / row["C1"]
        excess_ci = cis / row["C1"]
        axes[0].errorbar(xs, excess, yerr=excess_ci, fmt="o", ms=4,
                         color=c, alpha=0.8)
    # 重新画谱范数（需要 curves）—— 从 combos 取
    for (name, key, r), row in zip(combos, table):
        c = colors[name]
        tau = np.array([1.0 / k for k in K_GROUP] + [0.0])
        v = np.array([r["curves"][str(k)]["fcS"] for k in K_GROUP]
                     + [r["curves"]["m"]["fcS"]])
        ci = np.array([r["curves"][str(k)]["fcS_ci"] for k in K_GROUP]
                      + [r["curves"]["m"]["fcS_ci"]])
        b = r["curves"]["B"]["fcS"]
        axes[1].errorbar(tau, v / v[0], yerr=ci / v[0], fmt="s-", ms=4,
                         color=c, alpha=0.75)
        axes[1].plot([0.0], [b / v[0]], "x", color=c, ms=8, mew=2)
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="theory $y=\\tau$")
    axes[0].set_xlabel(r"$\tau$ (P[shared prediction copy])")
    axes[0].set_ylabel(r"$(V_K - V_0)\,/\,C_1$")
    axes[0].set_title("Centered F-energy excess vs theory line")
    axes[1].set_xlabel(r"$\tau$")
    axes[1].set_ylabel("centered spectral norm / value at $\\tau=1$")
    axes[1].set_title("Centered spectral norm (empirical; × = B)")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=colors[k], marker="o", ls="-",
                      label=k) for k in colors]
    handles.append(Line2D([], [], color="k", ls="--", label="theory"))
    axes[1].legend(handles=handles, fontsize=8)
    fig.tight_layout()
    (R / "figures").mkdir(exist_ok=True)
    fig.savefig(R / "figures" / "depint_final.png", dpi=200)
    Path(R / "depint_final_fit.json").write_text(
        json.dumps([{k: v for k, v in row.items()} for row in table],
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print("→ results/depint_final_fit.json, figures/depint_final.png")


if __name__ == "__main__":
    main()
