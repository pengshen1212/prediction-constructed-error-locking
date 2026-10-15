"""G3 开发集 pilot（计划 §7）。3 个新外层划分 × 5 单元；只用于开发，
数字不进正式主表。输出 results/correction_pilot.json。

用法：python run_correction_pilot.py --protocol protocol.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))          # prob2-relation-error/
sys.path.insert(0, str(HERE.parent / "analysis"))
sys.stdout.reconfigure(encoding="utf-8")

from near_locking import (bound_occupancy, cross_class_mass, kernel_apply,
                          near_locking_bound, no_change_threshold,
                          row_stochastic)
import correction as cr
from correction import (_sym_norm_S, cs_baseline, edge_pairs, evaluate,
                        fit_and_select, hard_weights, no_prop_baseline,
                        onehot, prop_baseline, residual_matrix,
                        soft_weights)
from outer_protocol import stratified_split, targets

import sign_predict as sp
from real_predictor_locking_v2 import load_feats, load_walmart

DATA = "data"


def load_unit(unit):
    """返回 (features, edges, labels, n_classes, predictor, predictor_fn)。"""
    tag, fam = unit
    if tag in ("cora", "dblp"):
        ds = sp.Dataset(tag)
        feats = load_feats(tag)
        return feats, [tuple(int(u) for u in e) for e in ds.edges], \
            ds.labels, int(ds.C), fam, None
    if tag == "walmart":
        edges, labels = load_walmart()
        edges = [tuple(int(u) for u in e) for e in edges]
        n = len(labels)
        n_cls = int(labels.max()) + 1

        def lp_fn(train_mask, train_targets, seed):
            rng = np.random.default_rng(seed)
            tr_idx = np.flatnonzero(train_mask)
            k = max(1, int(0.25 * len(tr_idx)))
            seeds = rng.choice(tr_idx, size=k, replace=False)
            Zseed = np.zeros((n, n_cls))
            Zseed[seeds, train_targets[seeds]] = 1.0
            S = _sym_norm_S(edges, np.ones(len(edges)), n)
            Z = Zseed.copy()
            for _ in range(50):
                Z = 0.9 * (S @ Z) + 0.1 * Zseed
            pred = Z.argmax(axis=1)
            pred[seeds] = train_targets[seeds]
            return pred

        return None, edges, labels, n_cls, "lp", lp_fn
    raise ValueError(unit)


def macro_f1(fin, y, C):
    from sklearn.metrics import f1_score
    return float(f1_score(y, fin, average="macro", labels=list(range(C))))


def run_unit(unit, n_splits, seed_base, ui, record_bound=True):
    tag, fam = unit
    feats, edges, labels, n_cls, predictor, predictor_fn = load_unit(unit)
    n = len(labels)
    out = []
    for r in range(n_splits):
        seed = seed_base + ui * 100 + r
        masks = stratified_split(labels, seed=seed)
        tr = targets(labels, masks["train"])
        va = targets(labels, masks["val"])
        te = targets(labels, masks["test"])
        t0 = time.time()

        kw = dict(predictor=predictor, n_classes=n_cls,
                  predictor_fn=predictor_fn)
        cand = fit_and_select(feats, edges, tr, va, masks, seed, **kw)
        baselines = {
            "no_prop": no_prop_baseline(feats, edges, tr, va, masks, seed,
                                        **kw),
            "prop_original": prop_baseline(feats, edges, tr, va, masks, seed,
                                           structure="original", **kw),
            "prop_soft": prop_baseline(feats, edges, tr, va, masks, seed,
                                       structure="soft", **kw),
            "cs": cs_baseline(feats, edges, tr, va, masks, seed, **kw),
        }

        # 消融：从 16 格网格复算两个角（去残差 / 去辅助通达）
        Z0, R = cand["Z0"], cand["R"]
        rows, cols, owner = edge_pairs(edges)
        P_hard = row_stochastic(n, rows, cols,
                                hard_weights(cand["z0"], edges)[owner])
        P_aux = row_stochastic(n, rows, cols, np.ones(len(edges))[owner])
        abl = {}
        for name, lam, eta in (("remove_residual", cand["lam"], 0.0),
                               ("remove_aux", 0.0, cand["eta"])):
            P_lam = (1.0 - lam) * P_hard + lam * P_aux
            abl[name] = cr.candidate_scores(P_lam, Z0, R, eta)

        methods = {"candidate": cand["C"],
                   "ablate_remove_residual": abl["remove_residual"],
                   "ablate_remove_aux": abl["remove_aux"]}
        methods.update({k: v["C"] for k, v in baselines.items()})

        rec = dict(split_seed=seed,
                   params=dict(lam=cand["lam"], eta=cand["eta"],
                               val_acc=cand["val_acc"],
                               baseline_params={
                                   k: {kk: vv for kk, vv in v.items()
                                       if kk in ("alpha", "alpha1", "alpha2",
                                                 "val_acc")}
                                   for k, v in baselines.items()}),
                   acc0=float((cand["z0"][masks["test"]]
                               == labels[masks["test"]]).mean()),
                   train_size=int(masks["train"].sum()),
                   test_size=int(masks["test"].sum()),
                   secs=round(time.time() - t0, 1))
        for name, Cmat in methods.items():
            m = evaluate(Cmat, te, masks["test"], z0=cand["z0"])
            m["macro_f1"] = macro_f1(np.asarray(Cmat).argmax(axis=1)[
                masks["test"]], labels[masks["test"]], n_cls)
            rec[name] = m

        # 基线选择（只看验证集）→ 本块"最强标准基线"
        best_bl = max(baselines, key=lambda k: baselines[k]["val_acc"])
        rec["strongest_baseline_by_val"] = best_bl
        rec["cand_minus_strongest_acc"] = (
            rec["candidate"]["acc"] - rec[best_bl]["acc"])

        # 近锁定界占用（one-hot γ=1）：原生结构与选定混合图
        if record_bound:
            occ = {}
            for sname, w in (("hard", hard_weights(cand["z0"], edges)),
                             ("soft", soft_weights(cand["z0"], edges,
                                                   n_cls)),
                             ("original", np.ones(len(edges))),
                             ("mixed_sel", None)):
                if sname == "mixed_sel":
                    P = (1.0 - cand["lam"]) * P_hard + cand["lam"] * P_aux
                else:
                    P = row_stochastic(n, rows, cols, w[owner])
                occ[sname] = bound_occupancy(P, cand["z0"], cr.ALPHA,
                                             cr.STEPS)
            rec["bound_occupancy"] = occ

        # 反向错误瓶颈诊断：错误测试节点可得的纠正核质量（硬图 vs 混合图）
        a = cand["z0"]
        G_tr = onehot(np.where(masks["train"], tr, 0), n_cls) * \
            masks["train"][:, None]
        wrong_te = masks["test"] & (a != labels)
        if wrong_te.any():
            diag = {}
            for dname, P in (("hard", P_hard),
                             ("mixed_sel", (1.0 - cand["lam"]) * P_hard
                              + cand["lam"] * P_aux)):
                KG = kernel_apply(P, G_tr, cr.ALPHA, cr.STEPS)
                rev = KG[np.arange(n)[wrong_te], labels[wrong_te]]
                diag[dname] = dict(
                    mean_corrective_mass=float(np.mean(rev)),
                    frac_positive=float((rev > 1e-9).mean()))
            rec["reverse_mistake_diag"] = diag

        out.append(rec)
        print(f"[{tag}/{fam}] split {r}: lam={cand['lam']} "
              f"eta={cand['eta']} cand.acc={rec['candidate']['acc']:.4f} "
              f"best={best_bl}:{rec[best_bl]['acc']:.4f} "
              f"diff={rec['cand_minus_strongest_acc'] * 100:+.2f}pp "
              f"({rec['secs']}s)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default=str(HERE / "protocol.json"))
    ap.add_argument("--splits", type=int, default=3)
    ap.add_argument("--seed-base", type=int, default=50000)
    ap.add_argument("--out", default=str(HERE / "results"
                                         / "correction_pilot.json"))
    args = ap.parse_args()
    proto = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    assert args.seed_base == 50000, "pilot seed namespace fixed at 50000"

    units = [tuple(u) for u in proto["data"]["units"]]
    doc = dict(protocol_version=proto["version"],
               purpose="G3 development pilot; numbers NOT for main tables",
               splits=args.splits, seed_base=args.seed_base, units={})
    for ui, unit in enumerate(units):
        t0 = time.time()
        doc["units"][f"{unit[0]}/{unit[1]}"] = run_unit(
            unit, args.splits, args.seed_base, ui)
        print(f"== unit {unit} done in {time.time()-t0:.0f}s")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(doc, ensure_ascii=False, indent=1),
                              encoding="utf-8")
    print(f"\n已写出 {args.out}")

    # 汇总：等权三数据集（pilot 内先按单元看）
    for uname, recs in doc["units"].items():
        d = np.mean([r["cand_minus_strongest_acc"] for r in recs]) * 100
        print(f"  {uname:<16} cand-strongest (mean) {d:+.2f}pp")


if __name__ == "__main__":
    main()
