"""G4 正式外层验证（计划 §8）。10 个新外层划分 × 5 单元；参数已冻结
（protocol.json），本脚本不再调方法参数。测试真值只在 evaluate() 解封。

用法：python run_correction_validation.py --protocol protocol.json
输出 results/correction_validation.json 与逐块记录。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.stdout.reconfigure(encoding="utf-8")

import correction as cr
from correction import (cs_baseline, evaluate, fit_and_select,
                        no_prop_baseline, prop_baseline)
from near_locking import argmax_labels
from outer_protocol import stratified_split, targets
from run_correction_pilot import load_unit, macro_f1

HISTORICAL_SEED_BLOCKS = [
    (100, 600), (700, 999), (4000, 4999), (5000, 5999), (7000, 7999),
    (50000, 50999),
]
FORMAL_BASE = 60000
METHOD_KEYS = ("candidate", "no_prop", "prop_original", "prop_soft", "cs")


def code_hashes():
    out = {}
    for f in ("correction.py", "near_locking.py", "outer_protocol.py"):
        out[f] = hashlib.sha256(
            (HERE / f).read_bytes()).hexdigest()[:16]
    return out


def check_seed_collisions(units):
    seeds = []
    for ui, unit in enumerate(units):
        seeds += [FORMAL_BASE + ui * 100 + r for r in range(10)]
    for s in seeds:
        for lo, hi in HISTORICAL_SEED_BLOCKS:
            assert not (lo <= s <= hi), f"seed collision: {s}"
    assert not (set(seeds) & set(range(50000, 51000))), "pilot collision"
    assert len(seeds) == len(set(seeds))
    return seeds


def run_block(unit, ui, r, proto):
    tag, fam = unit
    feats, edges, labels, n_cls, predictor, predictor_fn = load_unit(unit)
    n = len(labels)
    seed = FORMAL_BASE + ui * 100 + r
    masks = stratified_split(labels, seed=seed)
    tr = targets(labels, masks["train"])
    va = targets(labels, masks["val"])
    te = targets(labels, masks["test"])
    t0 = time.time()

    kw = dict(predictor=predictor, n_classes=n_cls,
              predictor_fn=predictor_fn)
    cand = fit_and_select(feats, edges, tr, va, masks, seed, **kw)
    baselines = {
        "no_prop": no_prop_baseline(feats, edges, tr, va, masks, seed, **kw),
        "prop_original": prop_baseline(feats, edges, tr, va, masks, seed,
                                       structure="original", **kw),
        "prop_soft": prop_baseline(feats, edges, tr, va, masks, seed,
                                   structure="soft", **kw),
        "cs": cs_baseline(feats, edges, tr, va, masks, seed, **kw),
    }
    best_bl = max(baselines, key=lambda k: baselines[k]["val_acc"])

    rec = dict(unit=f"{tag}/{fam}", split_seed=seed,
               masks={k: v.tolist() for k, v in masks.items()},
               z0=cand["z0"].tolist(),
               params=dict(lam=cand["lam"], eta=cand["eta"],
                           val_acc=cand["val_acc"],
                           strongest_baseline=best_bl,
                           baseline_selection_val={
                               k: baselines[k]["val_acc"]
                               for k in baselines}),
               trace=cand["trace"],
               label_access=dict(train=int(masks["train"].sum()),
                                 val=int(masks["val"].sum()),
                                 test="evaluate() only"),
               secs=round(time.time() - t0, 1))
    mats = {"candidate": cand["C"]}
    mats.update({k: v["C"] for k, v in baselines.items()})
    for name, Cmat in mats.items():
        m = evaluate(Cmat, te, masks["test"], z0=cand["z0"])
        m["macro_f1"] = macro_f1(
            argmax_labels(np.asarray(Cmat))[masks["test"]],
            labels[masks["test"]], n_cls)
        rec[name] = m
    rec["cand_minus_strongest_acc"] = (
        rec["candidate"]["acc"] - rec[best_bl]["acc"])
    print(f"[{tag}/{fam}] split {r}: lam={cand['lam']} "
          f"eta={cand['eta']} cand={rec['candidate']['acc']:.4f} "
          f"best={best_bl}:{rec[best_bl]['acc']:.4f} "
          f"diff={rec['cand_minus_strongest_acc']*100:+.2f}pp "
          f"({rec['secs']}s)", flush=True)
    return rec


def aggregate(blocks):
    """数据集内平均预测器族 → 三数据集等权；外层块级配对 bootstrap。

    勘误（2026-10-09 复查）：各单元（数据集×预测器族）使用互不相交的
    划分种子（seed = base + unit*100 + r），两族并不共享划分；原实现把
    不同族的种子混入同一桶后 min 长度截断，导致总体区间只重采样了部分
    单元。修正为：每族的 10 划分序列独立重采样，族均值平均得数据集
    抽样值，再等权平均三数据集。点估计不受影响。
    """
    by_unit = {}
    for b in blocks:
        by_unit.setdefault(b["unit"], []).append(b)
    per_unit = {}
    for uname, recs in sorted(by_unit.items()):
        d = np.array([r["cand_minus_strongest_acc"] for r in recs])
        per_unit[uname] = dict(
            n_splits=len(recs),
            mean_diff_pp=float(d.mean() * 100),
            cand_acc=float(np.mean([r["candidate"]["acc"] for r in recs])),
            strongest_acc=float(np.mean(
                [r[r["params"]["strongest_baseline"]]["acc"] for r in recs])),
            cand_corrected_pp=float(np.mean(
                [r["candidate"]["corrected"] for r in recs]) * 100),
            cand_wrong_pp=float(np.mean(
                [r["candidate"]["wrong_changed"] for r in recs]) * 100),
            strongest_corrected_pp=float(np.mean(
                [r[r["params"]["strongest_baseline"]]["corrected"]
                 for r in recs]) * 100),
            strongest_wrong_pp=float(np.mean(
                [r[r["params"]["strongest_baseline"]]["wrong_changed"]
                 for r in recs]) * 100),
            method_wins={k: int(sum(
                r["params"]["strongest_baseline"] == k for r in recs))
                for k in METHOD_KEYS[1:]})
    # 族级序列：每个 (dataset, family) 的 10 个划分差值
    fam_series = {}
    for uname, recs in by_unit.items():
        ds, fam = uname.split("/")
        fam_series[(ds, fam)] = np.array([
            r["cand_minus_strongest_acc"]
            for r in sorted(recs, key=lambda r: r["split_seed"])])
    ds_list = sorted({ds for ds, _ in fam_series})
    rng = np.random.default_rng(99)
    B = 2000
    ds_boot, ds_mean, per_dataset = {}, {}, {}
    for ds in ds_list:
        fams = sorted({fam for (d, fam) in fam_series if d == ds})
        fam_draws = []
        for f in fams:
            s = fam_series[(ds, f)]
            idx = rng.integers(0, len(s), (B, len(s)))
            fam_draws.append(s[idx].mean(axis=1))
        ds_boot[ds] = np.mean(fam_draws, axis=0) * 100
        ds_mean[ds] = float(np.mean(
            [fam_series[(ds, f)].mean() for f in fams]) * 100)
        per_dataset[ds] = dict(
            mean_pp=ds_mean[ds],
            ci_pp=[float(np.percentile(ds_boot[ds], 2.5)),
                   float(np.percentile(ds_boot[ds], 97.5))],
            n_splits=int(min(len(fam_series[(ds, f)]) for f in fams)))
    overall_b = np.mean([ds_boot[ds] for ds in ds_list], axis=0)
    overall = float(np.mean([ds_mean[ds] for ds in ds_list]))
    ci = [float(np.percentile(overall_b, 2.5)),
          float(np.percentile(overall_b, 97.5))]
    return dict(per_unit=per_unit, per_dataset=per_dataset,
                overall_equal_weighted_pp=overall,
                overall_ci_pp=ci,
                bootstrap=dict(B=B, seed=99, unit="outer split",
                               scheme=("per-family independent resampling; "
                                       "families averaged within dataset; "
                                       "equal weight across datasets")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default=str(HERE / "protocol.json"))
    ap.add_argument("--out", default=str(HERE / "results"
                                         / "correction_validation.json"))
    args = ap.parse_args()
    proto = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    units = [tuple(u) for u in proto["data"]["units"]]
    n_splits = int(proto["formal"]["outer_splits"])
    check_seed_collisions(units)

    doc = dict(protocol_version=proto["version"],
               code_hashes=code_hashes(),
               intervention_form="see correction.py / decision_log revision",
               splits=n_splits, seed_base=FORMAL_BASE, blocks=[])
    for ui, unit in enumerate(units):
        t0 = time.time()
        for r in range(n_splits):
            doc["blocks"].append(run_block(unit, ui, r, proto))
        print(f"== unit {unit} done {time.time()-t0:.0f}s", flush=True)

    doc["aggregate"] = aggregate(doc["blocks"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(doc, ensure_ascii=False, indent=1),
                              encoding="utf-8")
    a = doc["aggregate"]
    print("\n===== 汇总（candidate − strongest-by-val, pp）=====")
    for d, v in a["per_dataset"].items():
        print(f"  {d:<8} {v['mean_pp']:+.2f}pp "
              f"CI [{v['ci_pp'][0]:+.2f}, {v['ci_pp'][1]:+.2f}] "
              f"(n={v['n_splits']}/family)")
    print(f"  等权总平均 {a['overall_equal_weighted_pp']:+.2f}pp "
          f"95%CI [{a['overall_ci_pp'][0]:+.2f}, {a['overall_ci_pp'][1]:+.2f}]")
    print(f"\n已写出 {args.out}")


if __name__ == "__main__":
    main()
