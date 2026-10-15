"""外层测试集协议（评审第十轮修复，2026-10-09）。

问题：交叉拟合下每个节点的预测虽来自未训练它的折，但邻居预测可能来
自训练过它的折 ⟹ 传播存在间接标签泄漏。修复：固定外层测试集 T（20%，
按类分层）；预测器只在 V∖T 上训练（池内 5 折交叉拟合给池节点；测试节
点同样由折模型预测——第 g 组测试节点用排除 pool 折 g 的模型，测试标签
从不进入任何训练）；构图与初始化全部来自这些预测；纠错/误改/净变化
只在 T 节点上计。

臂（同 real_predictor_locking.v1/v2，替换主表口径）：
  结构 original / hard / soft（来自 P1 预测）×
  初始化 coupled(P1) / decoupled(P2 同协议第二种子) / indep(真值标签
  按匹配错误率腐蚀——机制对照，非可部署方案) / truth(oracle 对照)，
  外加 hard+softargmax（锁定演示）。
预测器族：原型（类均值最近）与 softmax 逻辑回归各跑。
Walmart：LP 种子只从 V∖T 抽，评估在 T∩未种子 上。
深度/α 网格（Cora 原型）同协议重跑。
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sign_predict as sp
from real_predictor_locking import structure_weights, soft_init
from real_predictor_locking_v2 import (load_feats, load_walmart,
                                       propagate, outcome, agg)

RESULTS = Path(__file__).resolve().parent / "results"
DATA = "data"
SEED = 4000            # 外层划分与预测种子（与此前所有块不相交）
TEST_FRAC = 0.20


def outer_split(labels, seed):
    """按类分层的外层测试集。"""
    rng = np.random.default_rng(seed)
    test = np.zeros(len(labels), dtype=bool)
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        take = rng.random(len(idx)) < TEST_FRAC
        test[idx[take]] = True
    return test


def proto_predict(feats, labels, pool, seed, n_folds=5):
    """原型预测器：池内 5 折 + 全池模型给全体（T 标签不进训练）。"""
    rng = np.random.default_rng(seed)
    n_cls = int(labels.max()) + 1
    n = len(labels)
    pred = np.zeros(n, dtype=int)

    def protos_of(train):
        protos = np.zeros((n_cls, feats.shape[1]))
        for c in range(n_cls):
            mask = labels[train] == c
            if mask.sum() > 0:
                protos[c] = np.asarray(
                    feats[train][mask].mean(axis=0)).ravel()
        return protos

    def assign(nodes, protos):
        blk = feats[nodes].toarray() if hasattr(feats, "toarray") \
            else feats[nodes]
        x_sq = (blk ** 2).sum(axis=1)
        p_sq = (protos ** 2).sum(axis=1)
        dists = x_sq[:, None] - 2 * blk @ protos.T + p_sq[None, :]
        pred[nodes] = np.argmin(dists, axis=1)

    perm = rng.permutation(pool)
    folds = np.array_split(perm, n_folds)
    for f in range(n_folds):
        train = np.concatenate([folds[g] for g in range(n_folds) if g != f])
        assign(folds[f], protos_of(train))
    # 测试节点也用折模型（第 g 组用 pool∖fold_g 训练），保持种子多样
    # 且测试标签不进任何训练；全池模型对 T 是确定性的，会让解耦臂退化
    others = np.flatnonzero(~np.isin(np.arange(n), pool))
    groups = np.array_split(rng.permutation(others), n_folds)
    for f in range(n_folds):
        train = np.concatenate([folds[g] for g in range(n_folds) if g != f])
        assign(groups[f], protos_of(train))
    return pred


def logreg_predict(feats, labels, pool, seed, n_folds=5):
    """logreg 预测器：同结构。"""
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(seed)
    n = len(labels)
    pred = np.zeros(n, dtype=int)
    perm = rng.permutation(pool)
    folds = np.array_split(perm, n_folds)

    def fit(train):
        clf = LogisticRegression(max_iter=300, C=1.0)
        clf.fit(feats[train], labels[train])
        return clf

    for f in range(n_folds):
        train = np.concatenate([folds[g] for g in range(n_folds) if g != f])
        clf = fit(train)
        pred[folds[f]] = clf.predict(feats[folds[f]])
    others = np.flatnonzero(~np.isin(np.arange(n), pool))
    groups = np.array_split(rng.permutation(others), n_folds)
    for f in range(n_folds):
        train = np.concatenate([folds[g] for g in range(n_folds) if g != f])
        pred[groups[f]] = fit(train).predict(feats[groups[f]])
    return pred


def main():
    doc = dict(protocol=(f"outer test set {int(TEST_FRAC*100)}% "
                         f"stratified; predictors trained only on the "
                         f"pool V∖T (5-fold cross-fit within the pool for "
                         f"pool nodes; test nodes are likewise predicted by "
                         f"fold models, group g using the model trained on "
                         f"pool∖fold_g, so no fold ever sees a test label); "
                         f"construction "
                         f"and initialization from these predictions; "
                         f"correction/wrong-change scored on T only; "
                         f"seeds {SEED}+"),
               families={}, walmart={}, sweep={})
    for tag, n_real, in (("cora", 10), ("dblp", 5)):
        ds = sp.Dataset(tag)
        feats = load_feats(tag)
        for fam, predictor in (("prototype", proto_predict),
                               ("logreg", logreg_predict)):
            t0 = time.time()
            cells = {}
            for r in range(n_real):
                test = outer_split(ds.labels, SEED + r)
                pool = np.flatnonzero(~test)
                pred1 = predictor(feats, ds.labels, pool, SEED + 100 + r)
                pred2 = predictor(feats, ds.labels, pool, SEED + 200 + r)
                err = float((pred1 != ds.labels).mean())
                indep = sp.corrupt_labels(
                    ds.labels, ds.C, err,
                    np.random.default_rng(SEED + 300 + r))
                accT_pool = float((pred1[pool] == ds.labels[pool]).mean())
                accT_test = float((pred1[test] == ds.labels[test]).mean())
                for st in ("hard", "original", "soft"):
                    w = structure_weights(ds, pred1, st)
                    for ini, Z0, z0 in (
                            ("coupled", sp.onehot(pred1, ds.C), pred1),
                            ("decoupled", sp.onehot(pred2, ds.C), pred2),
                            ("indep", sp.onehot(indep, ds.C), indep),
                            ("truth", sp.onehot(ds.labels, ds.C),
                             ds.labels)):
                        cells.setdefault((st, ini), []).append(
                            outcome(z0, propagate(ds, w, Z0, 20, 0.9),
                                    ds.labels, test))
                w_hard = structure_weights(ds, pred1, "hard")
                Zsoft = soft_init(pred1, ds.C,
                                  np.random.default_rng(SEED + 400 + r))
                cells.setdefault(("hard", "softargmax"), []).append(
                    outcome(pred1, propagate(ds, w_hard, Zsoft, 20, 0.9),
                            ds.labels, test))
                if r == 0:
                    print(f"[{tag} {fam}] 外层测试占比 {test.mean():.2f} "
                          f"held-out acc pool={accT_pool:.4f} "
                          f"test={accT_test:.4f}", flush=True)
            doc["families"].setdefault(fam, {})[tag] = dict(
                n_realizations=n_real,
                cells={f"{k[0]}+{k[1]}": agg(v)
                       for k, v in sorted(cells.items())})
            print(f"[{tag} {fam}] 完成 ({time.time()-t0:.0f}s)")
            for k, v in doc["families"][fam][tag]["cells"].items():
                print(f"  {k:<22} acc0={v['acc0']:.4f} net={v['net']:+.4f}"
                      f" corr={v['corrected']:+.4f}"
                      f" wrong={v['wrong_changed']:+.4f}")

    # Walmart：LP 种子只从池抽，评估在 T 上
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
    pad = np.full((ds.m, int(ds.k.max())), -1, dtype=np.int64)
    for i, e in enumerate(ds.edges):
        pad[i, :len(e)] = e
    ds.pad = pad
    ds.all_hom = sp.Dataset.__dict__["all_hom"].__get__(ds)
    cells = {}
    for r in range(5):
        test = outer_split(labels, SEED + 500 + r)
        pool_nodes = np.flatnonzero(~test)
        rng = np.random.default_rng(SEED + 600 + r)
        seeds = np.zeros(ds.n, dtype=bool)
        seeds[rng.choice(pool_nodes, size=int(0.25 * len(pool_nodes)),
                         replace=False)] = True
        Zseed = sp.onehot(np.where(seeds, labels, 0), C) * seeds[:, None]
        pred = propagate(ds, np.ones(ds.m), Zseed, 50, 0.9).argmax(axis=1)
        pred[seeds] = labels[seeds]
        err = float((pred[test] == labels[test]).mean())
        indep = sp.corrupt_labels(labels, C, 1 - err,
                                  np.random.default_rng(SEED + 700 + r))
        for st, w in (("hard", structure_weights(ds, pred, "hard")),
                      ("original", np.ones(ds.m)),
                      ("soft", structure_weights(ds, pred, "soft"))):
            for ini, Z0, z0 in (("coupled", sp.onehot(pred, C), pred),
                                ("indep", sp.onehot(indep, C), indep)):
                cells.setdefault((st, ini), []).append(
                    outcome(z0, propagate(ds, w, Z0, 20, 0.9),
                            labels, test))
        if r == 0:
            print(f"[walmart] LP(池种子25%) 外层测试 acc={err:.4f}",
                  flush=True)
    doc["walmart"] = dict(n_realizations=5,
                          cells={f"{k[0]}+{k[1]}": agg(v)
                                 for k, v in sorted(cells.items())})
    print("[walmart] 完成")
    for k, v in doc["walmart"]["cells"].items():
        print(f"  {k:<20} acc0={v['acc0']:.4f} net={v['net']:+.4f} "
              f"corr={v['corrected']:+.4f} wrong={v['wrong_changed']:+.4f}")

    # 深度/α 网格（Cora 原型，外层测试）
    dsc = sp.Dataset("cora")
    feats_c = load_feats("cora")
    T_LIST = (2, 5, 10, 20, 40)
    A_LIST = (0.5, 0.7, 0.9, 0.95)
    grid = {}
    for r in range(5):
        test = outer_split(dsc.labels, SEED + 800 + r)
        pool = np.flatnonzero(~test)
        pred = proto_predict(feats_c, dsc.labels, pool, SEED + 900 + r)
        indep = sp.corrupt_labels(dsc.labels, dsc.C,
                                  float((pred != dsc.labels).mean()),
                                  np.random.default_rng(SEED + 950 + r))
        w = {s: structure_weights(dsc, pred, s)
             for s in ("hard", "original", "soft")}
        for T in T_LIST:
            for a in A_LIST:
                for name, ww, Z0, z0 in (
                        ("hard+coupled", w["hard"],
                         sp.onehot(pred, dsc.C), pred),
                        ("original+coupled", w["original"],
                         sp.onehot(pred, dsc.C), pred),
                        ("hard+indep", w["hard"],
                         sp.onehot(indep, dsc.C), indep),
                        ("original+truth", w["original"],
                         sp.onehot(dsc.labels, dsc.C), dsc.labels)):
                    grid.setdefault((name, T, a), []).append(
                        outcome(z0, propagate(dsc, ww, Z0, T, a),
                                dsc.labels, test))
    doc["sweep"] = dict(
        dataset="cora", family="prototype", n_realizations=5,
        T=list(T_LIST), alpha=list(A_LIST),
        cells={f"{n}|T={T}|a={a}": agg(v, boot=False)
               for (n, T, a), v in grid.items()})
    mx = max(v[0]["corrected"] for (n, _, _), v in grid.items()
             if n == "hard+coupled" for v in [v])
    print("[cora sweep] hard+coupled 全格纠错最大值 =", mx)

    import os
    dest = Path(os.environ.get(
        "P2_OUTER_TEST_DEST",
        str(RESULTS / "real_predictor_outer_test.json")))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(doc, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    print(f"\n已写出 {dest}")


if __name__ == "__main__":
    main()
