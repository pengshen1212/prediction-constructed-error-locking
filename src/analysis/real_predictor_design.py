"""从障碍到设计原则（Paper2 续，2026-10-09）。

前置结论：锁定障碍跨预测器族/数据集稳健；解锁途径有二（独立信息、
跨类连接），且跨类连接在低同质性共购图上有害。本脚本把三者接成可
操作的诊断与规则，并完成评审清单中未做的"控制预测质量"轴：

  1 质量退化扫描：把构图预测 pred 用随机腐蚀降到 γ ∈ {0,.1,.2,.3}
    （Cora/DBLP，logreg 族，R=10/5），观察各臂纠错/误改/净收益随质量
    的变化。假设：跨类连接净收益随质量下降收缩乃至转负，独立信息
    解锁保持为正。
  2 无标签同质性代理（构造期可用，不用真标签）：
    P1 平均预测同质性 = mean_e max_c frac(pred=c)（即软权均值）；
    P2 双抽边一致率 = 两次独立预测抽取下"同预测类且类一致"的边比例。
    数据集级：代理应区分 Cora/DBLP（跨类有益）与 Walmart（有害）；
    数据集内：γ 退化使代理下降，同时跨类净收益收缩 —— 代理与收益
    符号同动即内部验证。oracle 边同质性仅作参照（不可用于构造）。
  3 Walmart 深度/α 小网格：跨类有害是否随 α/T 变化（original+coupled
    与 hard+indep 两臂，α∈{.5,.7,.9,.95}×T∈{2,10,20,40}，R=5）。

冻结：logreg 族 seeds 700+（与 v2 相同基础预测）；γ 作用于预测本身
（非真标签）；全部臂同一实现内共享同一（退化后）预测。诚实报告，
若代理与收益符号不同动则如实写。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sign_predict as sp
from real_predictor_locking import structure_weights
from real_predictor_locking_v2 import (load_feats, load_walmart,
                                       propagate, outcome, agg)
from real_predictor_outer_test import (logreg_predict, outer_split)

RESULTS = Path(__file__).resolve().parent / "results"
SEED_LR = 700
SEED_GAMMA = 900
SEED_LP = 800
GAMMAS = (0.0, 0.1, 0.2, 0.3)
N_REAL = {"cora": 10, "dblp": 5}


def proxies(ds, pred_a, pred_b):
    """P1 平均预测同质性（软权均值）与双抽边一致率（无标签）。"""
    w_soft = structure_weights(ds, pred_a, "soft")
    p1_hom = float(w_soft.mean())
    pa = ds.all_hom(pred_a) == 1
    pb = ds.all_hom(pred_b) == 1
    same = np.array([pred_a[e[0]] == pred_b[e[0]] if len(set(pred_a[list(e)])) == 1
                     and len(set(pred_b[list(e)])) == 1 else False
                     for e in ds.edges])
    agree = float((pa & pb & same).mean())
    return p1_hom, agree


def oracle_hom(ds):
    return float((ds.all_hom(ds.labels) == 1).mean())


def part_quality(doc):
    for tag, n_real in (("cora", 10), ("dblp", 5)):
        ds = sp.Dataset(tag)
        feats = load_feats(tag)
        t0 = time.time()
        out = {}
        px = {"p1_hom": [], "agree": [], "oracle": oracle_hom(ds)}
        for r in range(n_real):
            test = outer_split(ds.labels, SEED_LR + 3000 + r)
            pool = np.flatnonzero(~test)
            pred = logreg_predict(feats, ds.labels, pool, SEED_LR + r)
            pred_b = logreg_predict(feats, ds.labels, pool,
                                    SEED_LR + 100 + r)
            for g in GAMMAS:
                pred_g = sp.corrupt_labels(
                    pred, ds.C, g, np.random.default_rng(SEED_GAMMA + 100 * r
                                                         + round(100 * g)))
                err = float((pred_g != ds.labels).mean())
                indep = sp.corrupt_labels(
                    ds.labels, ds.C, err,
                    np.random.default_rng(SEED_GAMMA + 5000 + 100 * r
                                          + round(100 * g)))
                cells = {}
                for name, w, Z0, z0 in (
                        ("hard+coupled", structure_weights(ds, pred_g, "hard"),
                         sp.onehot(pred_g, ds.C), pred_g),
                        ("original+coupled", np.ones(ds.m),
                         sp.onehot(pred_g, ds.C), pred_g),
                        ("soft+coupled", structure_weights(ds, pred_g, "soft"),
                         sp.onehot(pred_g, ds.C), pred_g),
                        ("hard+indep", structure_weights(ds, pred_g, "hard"),
                         sp.onehot(indep, ds.C), indep)):
                    cells[name] = outcome(z0, propagate(ds, w, Z0, 20, 0.9),
                                          ds.labels, test)
                out.setdefault(g, []).append(cells)
                if g == 0.0:
                    h1, ag = proxies(ds, pred_g, pred_b)
                    px["p1_hom"].append(h1)
                    px["agree"].append(ag)
        doc.setdefault("quality", {})[tag] = dict(
            n_realizations=n_real, predictor="cross-fit logreg",
            gammas=list(GAMMAS),
            proxy_p1_hom=float(np.mean(px["p1_hom"])),
            proxy_agree=float(np.mean(px["agree"])),
            proxy_oracle_hom=px["oracle"],
            cells={f"g={g}": {k: agg([c[k] for c in lst])
                              for k in lst[0]}
                   for g, lst in out.items()})
        print(f"[{tag}] 质量扫描完成 ({time.time()-t0:.0f}s)；"
              f"代理: P1同质 {np.mean(px['p1_hom']):.3f} "
              f"双抽一致 {np.mean(px['agree']):.3f} "
              f"oracle {px['oracle']:.3f}")
        for g in GAMMAS:
            c = doc["quality"][tag]["cells"][f"g={g}"]
            print(f"  γ={g}: " + " | ".join(
                f"{k} net={v['net']:+.4f} (corr {v['corrected']:+.4f}"
                f"/wrong {v['wrong_changed']:+.4f})"
                for k, v in c.items()))


def part_walmart_grid(doc):
    edges, labels = load_walmart()
    C = int(labels.max()) + 1

    class DS:
        pass
    ds = DS()
    ds.edges = [tuple(int(u) for u in e) for e in edges]
    ds.labels = labels
    ds.n, ds.C, ds.m = len(labels), C, len(ds.edges)
    ds.k = np.array([len(e) for e in ds.edges])
    rows, cols, owner = [], [], []
    for i, e in enumerate(ds.edges):
        kk = len(e)
        rows.append(np.repeat(e, kk)); cols.append(np.tile(e, kk))
        owner.append(np.full(kk * kk, i))
    ds.rows = np.concatenate(rows); ds.cols = np.concatenate(cols)
    ds.owner = np.concatenate(owner)
    kmax = int(ds.k.max())
    pad = np.full((ds.m, kmax), -1, dtype=np.int64)
    for i, e in enumerate(ds.edges):
        pad[i, :len(e)] = e
    ds.pad = pad
    ds.all_hom = sp.Dataset.__dict__["all_hom"].__get__(ds)

    grid = {}
    px_h, px_a = [], []
    t0 = time.time()
    for r in range(5):
        rng = np.random.default_rng(SEED_LP + r)
        seeds = rng.random(ds.n) < 0.20
        Zseed = sp.onehot(np.where(seeds, labels, 0), C) * seeds[:, None]
        pred = propagate(ds, np.ones(ds.m), Zseed, 50, 0.9).argmax(axis=1)
        pred[seeds] = labels[seeds]
        unseed = ~seeds
        acc_pred = float((pred[unseed] == labels[unseed]).mean())
        # 双抽代理：第二个 LP 种子集
        rng2 = np.random.default_rng(SEED_LP + 100 + r)
        seeds2 = rng2.random(ds.n) < 0.20
        Zs2 = sp.onehot(np.where(seeds2, labels, 0), C) * seeds2[:, None]
        pred2 = propagate(ds, np.ones(ds.m), Zs2, 50, 0.9).argmax(axis=1)
        pred2[seeds2] = labels[seeds2]
        h1, ag = proxies(ds, pred, pred2)
        px_h.append(h1); px_a.append(ag)
        indep = sp.corrupt_labels(labels, C, 1 - acc_pred,
                                  np.random.default_rng(SEED_LP + 5000 + r))
        for a in (0.5, 0.7, 0.9, 0.95):
            for T in (2, 10, 20, 40):
                for name, w, Z0, z0 in (
                        ("hard+coupled", structure_weights(ds, pred, "hard"),
                         sp.onehot(pred, C), pred),
                        ("original+coupled", np.ones(ds.m),
                         sp.onehot(pred, C), pred),
                        ("hard+indep", structure_weights(ds, pred, "hard"),
                         sp.onehot(indep, C), indep)):
                    grid.setdefault((name, a, T), []).append(
                        outcome(z0, propagate(ds, w, Z0, T, a),
                                labels, unseed))
    doc["walmart_grid"] = dict(
        n_realizations=5, alpha=[0.5, 0.7, 0.9, 0.95], T=[2, 10, 20, 40],
        proxy_p1_hom=float(np.mean(px_h)),
        proxy_agree=float(np.mean(px_a)),
        proxy_oracle_hom=oracle_hom(ds),
        cells={f"{n}|a={a}|T={T}": agg(v, boot=False)
               for (n, a, T), v in grid.items()})
    print(f"[walmart] 网格完成 ({time.time()-t0:.0f}s)；"
          f"代理: P1同质 {np.mean(px_h):.3f} 双抽一致 {np.mean(px_a):.3f} "
          f"oracle {oracle_hom(ds):.3f}")
    for a in (0.5, 0.7, 0.9, 0.95):
        oc = [grid[("original+coupled", a, T)][0]["accT"]
              - grid[("original+coupled", a, T)][0]["acc0"] for T in (2, 10, 20, 40)]
        hi = [grid[("hard+indep", a, T)][0]["accT"]
              - grid[("hard+indep", a, T)][0]["acc0"] for T in (2, 10, 20, 40)]
        print(f"  α={a}: original+coupled net "
              + ",".join(f"{x:+.3f}" for x in oc)
              + " | hard+indep net "
              + ",".join(f"{x:+.3f}" for x in hi) + "  (T=2,10,20,40)")


def main():
    doc = dict(protocol=("quality-degradation sweep (gamma in {0,.1,.2,.3} "
                         "applied to the construction predictions, logreg "
                         "family), label-free homophily proxies (mean "
                         "predicted homophility = soft-weight mean; "
                         "two-draw edge agreement), and a Walmart depth/"
                         "alpha grid for the harmful cross-class arm; all "
                         "arms share identical (degraded) predictions "
                         "within a realization"),
               seeds=dict(logreg=SEED_LR, gamma=SEED_GAMMA, lp=SEED_LP))
    part_quality(doc)
    part_walmart_grid(doc)
    dest = RESULTS / "real_predictor_design.json"
    dest.write_text(json.dumps(doc, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    print(f"\n已写出 {dest}")


if __name__ == "__main__":
    main()
