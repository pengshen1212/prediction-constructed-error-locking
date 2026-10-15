"""P2-B 新种子批次独立复现（2026-10-08）。

补证据清单 P2-B：当前主表是保存轨迹的配对重算，不是独立复现。本脚本
用与既有评估批次（seed 5000 块）完全不相交的新种子块（base 7000），
以同一单位权重协议重复核心对照：
  truth (φ=0) / coupled (φ=1) / independent (q=p)，p ∈ {0.10, 0.20}，
  T=20，Cora 残差流 + Cora/DBLP normalized(α=0.9) 流，每格 100 实现。
配对纪律与原协议一致：同一 (kind,p,mode) 的 φ=0/φ=1 单元格共享图实现
种子（cell_seed 结构照抄），independent 用独立流（+50）。
完成标准：Cora 符号反转、两数据集 coupled-minus-truth 变化与耦合
准确率差分别报告；若 DBLP truth 仍不显著，保留零附近结论。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sign_predict as sp

RESULTS = Path(__file__).resolve().parent / "results"
TAGS = (("cora", "heat"), ("cora_appnp", "appnp"), ("dblp_appnp", "appnp"))
P_LIST = (0.1, 0.2)
T_OUT = 20
N_REAL = 100
BASE_NEW = 7000          # 旧评估块 5000+，完全不相交
N_BOOT = 2000
BOOT_SEED = 99


def cell_seed(base, kind, p, mode):
    """照抄 sign_predict.cell_seed：φ 曲线配对，indep 独立流。"""
    return base + 100 * round(p * 100) + (0 if mode == "couple" else 50)


def summarize(cell_t):
    a = {k: np.asarray(v, dtype=float) for k, v in cell_t.items()}
    margin = a["mA"] - a["mB"]
    accuracy = a["aA"] - a["aB"]
    phat = a["phat"]
    cor = phat * (a["mA_cor"] - a["mB_cor"])
    cln = (1 - phat) * (a["mA_cln"] - a["mB_cln"])
    res = float(np.max(np.abs(margin - cor - cln)))
    assert res < 1e-10, res
    idx = np.random.default_rng(BOOT_SEED).integers(
        0, len(margin), (N_BOOT, len(margin)))

    def ci(v):
        return np.quantile(v[idx].mean(axis=1), [0.025, 0.975]).tolist()
    return dict(n_realizations=len(margin),
                realized_corruption=float(phat.mean()),
                margin_difference=float(margin.mean()), margin_ci=ci(margin),
                accuracy_difference=float(accuracy.mean()),
                accuracy_ci=ci(accuracy),
                corrupted_contribution=float(cor.mean()),
                clean_contribution=float(cln.mean()),
                max_decomposition_residual=res)


def run_tag(tag, mode_op):
    data_key = tag.replace("_appnp", "")
    sp.OP["mode"] = mode_op
    sp.OP["alpha"] = 0.9
    ds = sp.Dataset(data_key)
    print(f"[{tag}] n={ds.n} m={ds.m} C={ds.C} "
          f"op={mode_op}", flush=True)
    out = dict(cells=[], contrasts=[])
    for p in P_LIST:
        cells = {}
        for kind, mode, phi, q in (
                ("sweep", "couple", 0.0, None),
                ("sweep", "couple", 1.0, None),
                ("indep", "indep", None, p)):
            base = cell_seed(BASE_NEW, kind, p, mode)
            t0 = __import__("time").time()
            cell = sp.run_cell(ds, p, mode, phi=phi, q=q,
                               n_real=N_REAL, base_seed=base)
            init = ("truth" if phi == 0 else "coupled") \
                if mode == "couple" else "independent"
            summ = summarize(cell[T_OUT])
            cells[init] = cell[T_OUT]
            out["cells"].append(dict(
                tag=tag, p=p, initialization=init, q=q, depth=T_OUT,
                base_seed=base, **summ))
            print(f"  p={p} {init:<11} margin {summ['margin_difference']:+.5f} "
                  f"{summ['margin_ci']} acc {summ['accuracy_difference']:+.5f} "
                  f"{summ['accuracy_ci']} ({__import__('time').time()-t0:.0f}s)",
                  flush=True)
        # 配对端点对比（truth 与 coupled 共享实现种子）
        np.testing.assert_array_equal(cells["truth"]["phat"],
                                      cells["coupled"]["phat"])
        dt = np.asarray(cells["truth"]["mA"]) - np.asarray(cells["truth"]["mB"])
        dc = np.asarray(cells["coupled"]["mA"]) - np.asarray(cells["coupled"]["mB"])
        at = np.asarray(cells["truth"]["aA"]) - np.asarray(cells["truth"]["aB"])
        ac = np.asarray(cells["coupled"]["aA"]) - np.asarray(cells["coupled"]["aB"])
        contrast = dc - dt
        acc_cost = ac - at
        idx = np.random.default_rng(BOOT_SEED).integers(
            0, len(dt), (N_BOOT, len(dt)))
        out["contrasts"].append(dict(
            tag=tag, p=p, depth=T_OUT,
            coupled_minus_truth_margin=float(contrast.mean()),
            coupled_minus_truth_margin_ci=np.quantile(
                contrast[idx].mean(axis=1), [.025, .975]).tolist(),
            coupled_minus_truth_acc=float(acc_cost.mean()),
            coupled_minus_truth_acc_ci=np.quantile(
                acc_cost[idx].mean(axis=1), [.025, .975]).tolist()))
        c = out["contrasts"][-1]
        print(f"  p={p} coupled−truth margin {c['coupled_minus_truth_margin']:+.5f} "
              f"{c['coupled_minus_truth_margin_ci']} | acc "
              f"{c['coupled_minus_truth_acc']:+.5f} "
              f"{c['coupled_minus_truth_acc_ci']}", flush=True)
    return out


def main():
    doc = dict(
        protocol=("new-seed replication of the core truth/independent/"
                  "coupled contrasts; seeds base 7000 block, disjoint from "
                  "the saved evaluation batch (base 5000); same unit-weight "
                  "operators, alpha=0.9, p in {0.10,0.20}, T=20, 100 "
                  "realizations per cell; paired bootstrap 2000, seed 99"),
        base_seed=BASE_NEW, n_real=N_REAL, n_boot=N_BOOT,
        bootstrap_seed=BOOT_SEED, tags={})
    for tag, mode_op in TAGS:
        doc["tags"][tag] = run_tag(tag, mode_op)
    dest = RESULTS / "seed_batch_replication.json"
    dest.write_text(json.dumps(doc, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    print(f"\n已写出 {dest}")
    print("\n===== 汇总（p=0.10, T=20；与主表口径对照）=====")
    for tag, mode_op in TAGS:
        for c in doc["tags"][tag]["cells"]:
            if c["p"] == 0.1:
                print(f"{tag:<11} {c['initialization']:<11} "
                      f"margin {c['margin_difference']:+.5f} "
                      f"[{c['margin_ci'][0]:+.5f},{c['margin_ci'][1]:+.5f}] "
                      f"acc {c['accuracy_difference']:+.5f}")
        for c in doc["tags"][tag]["contrasts"]:
            if c["p"] == 0.1:
                print(f"{tag:<11} coupled−truth margin "
                      f"{c['coupled_minus_truth_margin']:+.5f} "
                      f"[{c['coupled_minus_truth_margin_ci'][0]:+.5f},"
                      f"{c['coupled_minus_truth_margin_ci'][1]:+.5f}]")


if __name__ == "__main__":
    main()
