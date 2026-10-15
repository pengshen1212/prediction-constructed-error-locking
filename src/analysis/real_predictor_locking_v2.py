"""真实预测构图补强 v2（评审第九轮后续任务，2026-10-09）。

在 real_predictor_locking.py 之上补四项：
  A 第二预测器族：5 折交叉拟合 softmax 逻辑回归（sklearn，带
    predict_proba），检验锁定/解锁结论不依赖原型分类器。
  B 缓解感知对照（logreg 族，τ=中位置置信度）：
    B1 置信度过滤 + 全体耦合 one-hot 初始化（不弃权）——按命题应
       仍精确锁定（谨慎选边本身不解锁）；
    B2 置信度过滤 + 弃权初始化（低置信节点均匀分数）——只在置信
       节点上计纠错/误改，弃权节点单列。
    与软权重构图（无置信度需求）对比。
  C 深度/α 敏感性（Cora，原型族，5 实现）：
    T ∈ {2,5,10,20,40} × α ∈ {0.5,0.7,0.9,0.95}；硬筛+耦合格应全程
    精确为 0（锁定与深度无关），真值臂展示深度代价。
  D 非引文数据集：Walmart-trips（导出子图，无特征），LP 预测器
    （20% 种子，原图传播）作为真实预测源；只在未种子节点上计分。

冻结：所有臂在同一实现内共享同一预测；τ=全体节点置信度中位数；
报告含负结果与边界，不新增方法主张。
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
from real_predictor_locking import (build_adj_weighted, structure_weights,
                                    SOFT_EPS, soft_init)

from crossfit_v2 import crossfit_predict_sparse

RESULTS = Path(__file__).resolve().parent / "results"
DATA = "data"
WALMART = f"{DATA}/web-hypergraphs/walmart-trips"
SEED_LR = 700          # logreg 交叉拟合种子（与 v1 的 100/300 块不相交）
SEED_LP = 800          # Walmart LP 种子
N_BOOT = 2000


def crossfit_logreg(feats, labels, seed, n_folds=5, conf_out=False):
    """5 折交叉拟合 softmax 逻辑回归（held-out 预测 + 置信度）。"""
    from sklearn.linear_model import LogisticRegression
    n = len(labels)
    rng = np.random.default_rng(seed)
    fold_idx = rng.permutation(n)
    folds = np.array_split(fold_idx, n_folds)
    pred = np.zeros(n, dtype=int)
    conf = np.zeros(n)
    for f in range(n_folds):
        test = folds[f]
        train = np.concatenate([folds[g] for g in range(n_folds) if g != f])
        clf = LogisticRegression(max_iter=300, C=1.0)
        clf.fit(feats[train], labels[train])
        pr = clf.predict_proba(feats[test])
        pred[test] = clf.classes_[pr.argmax(axis=1)]
        conf[test] = pr.max(axis=1)
    if conf_out:
        return pred, conf
    return pred


def propagate(ds, weights, Z0, T, alpha):
    import scipy.sparse as spsp
    vals = weights[ds.owner]
    keep = vals != 0.0
    A = spsp.csr_matrix((vals[keep], (ds.rows[keep], ds.cols[keep])),
                        shape=(ds.n, ds.n)) + spsp.eye(ds.n, format="csr")
    deg = np.asarray(A.sum(axis=1)).ravel()
    d_is = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    S = (spsp.diags(d_is) @ A @ spsp.diags(d_is)).tocsr()
    Z = Z0.copy()
    for _ in range(T):
        Z = alpha * (S @ Z) + (1 - alpha) * Z0
    return Z


def outcome(z0, ZT, labels, mask=None):
    if mask is None:
        mask = np.ones(len(labels), dtype=bool)
    fin = np.argmax(ZT, axis=1)
    return dict(
        acc0=float((z0[mask] == labels[mask]).mean()),
        accT=float((fin[mask] == labels[mask]).mean()),
        corrected=float(((z0[mask] != labels[mask]) &
                         (fin[mask] == labels[mask])).mean()),
        wrong_changed=float(((z0[mask] == labels[mask]) &
                             (fin[mask] != labels[mask])).mean()))


def agg(lst, boot=True):
    out = {}
    for k in ("acc0", "accT", "corrected", "wrong_changed"):
        v = np.array([c[k] for c in lst])
        out[k] = float(v.mean())
        if boot:
            idx = np.random.default_rng(99).integers(0, len(v),
                                                     (N_BOOT, len(v)))
            out[f"{k}_ci"] = np.quantile(
                v[idx].mean(axis=1), [0.025, 0.975]).tolist()
    out["net"] = out["accT"] - out["acc0"]
    return out


def load_feats(tag):
    with open(f"{sp.DATA_DIRS[tag]}/features.pickle", "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------- A+B
def part_AB(doc):
    for tag, n_real in (("cora", 10), ("dblp", 5)):
        ds = sp.Dataset(tag)
        feats = load_feats(tag)
        t0 = time.time()
        cells = {}
        for r in range(n_real):
            pred, conf = crossfit_logreg(feats, ds.labels, SEED_LR + r,
                                         conf_out=True)
            acc_pred = float((pred == ds.labels).mean())
            tau = float(np.median(conf))
            confident = conf >= tau
            err_rate = 1.0 - acc_pred
            indep = sp.corrupt_labels(
                ds.labels, ds.C, err_rate,
                np.random.default_rng(SEED_LR + 5000 + r))

            def add(name, weights, Z0, mask=None, z0=None):
                z0 = pred if z0 is None else z0
                cells.setdefault(name, []).append(
                    outcome(z0, propagate(ds, weights, Z0, 20, 0.9),
                            ds.labels, mask))

            # A：logreg 族主对照
            add("hard+coupled", structure_weights(ds, pred, "hard"),
                sp.onehot(pred, ds.C))
            add("hard+indep", structure_weights(ds, pred, "hard"),
                sp.onehot(indep, ds.C), z0=indep)
            add("original+coupled", structure_weights(ds, pred, "original"),
                sp.onehot(pred, ds.C))
            add("soft+coupled", structure_weights(ds, pred, "soft"),
                sp.onehot(pred, ds.C))
            # B1：置信过滤（同类且全体成员 conf≥τ）+ 不弃权 one-hot
            conf_hard = np.array([
                1.0 if (len(set(pred[u] for u in e)) == 1 and
                        confident[list(e)].all()) else 0.0
                for e in ds.edges])
            add("conffilter+coupled", conf_hard, sp.onehot(pred, ds.C))
            # B2：同结构 + 弃权初始化（低置信节点均匀分数），只在置信节点计分
            Zab = sp.onehot(pred, ds.C).astype(float)
            Zab[~confident] = 1.0 / ds.C
            add("conffilter+abstain", conf_hard, Zab, mask=confident)
            if r == 0:
                kept = float((conf_hard == 1.0).mean())
                print(f"[{tag}] logreg acc={acc_pred:.4f} τ={tau:.3f} "
                      f"confident={confident.mean():.2f} "
                      f"conf-filter 保留边={kept:.3f}", flush=True)
        doc["tags"][tag] = dict(n_realizations=n_real,
                                predictor="cross-fit softmax logistic",
                                cells={k: agg(v) for k, v in cells.items()})
        print(f"[{tag}] A+B 完成 ({time.time()-t0:.0f}s)")
        for k, v in doc["tags"][tag]["cells"].items():
            print(f"  {k:<24} acc0={v['acc0']:.4f} net={v['net']:+.4f} "
                  f"corr={v['corrected']:+.4f} wrong={v['wrong_changed']:+.4f}")


# ---------------------------------------------------------------- C
def part_C(doc):
    tag = "cora"
    ds = sp.Dataset(tag)
    feats = load_feats(tag)
    T_LIST = (2, 5, 10, 20, 40)
    A_LIST = (0.5, 0.7, 0.9, 0.95)
    grid = {}
    for r in range(5):
        pred = crossfit_predict_sparse(feats, ds.labels, SEED_LR + 200 + r)
        acc_pred = float((pred == ds.labels).mean())
        indep = sp.corrupt_labels(ds.labels, ds.C, 1 - acc_pred,
                                  np.random.default_rng(SEED_LR + 6000 + r))
        w = {s: structure_weights(ds, pred, s) for s in ("hard", "original",
                                                         "soft")}
        for T in T_LIST:
            for a in A_LIST:
                for name, ww, Z0, z0 in (
                        ("hard+coupled", w["hard"], sp.onehot(pred, ds.C), pred),
                        ("original+coupled", w["original"],
                         sp.onehot(pred, ds.C), pred),
                        ("soft+coupled", w["soft"], sp.onehot(pred, ds.C), pred),
                        ("hard+indep", w["hard"], sp.onehot(indep, ds.C), indep),
                        ("original+truth", w["original"],
                         sp.onehot(ds.labels, ds.C), ds.labels)):
                    grid.setdefault((name, T, a), []).append(
                        outcome(z0, propagate(ds, ww, Z0, T, a), ds.labels))
    doc["sweep"] = dict(
        dataset=tag, predictor="cross-fit prototype", n_realizations=5,
        T=list(T_LIST), alpha=list(A_LIST),
        cells={f"{name}|T={T}|a={a}": agg(v, boot=False)
               for (name, T, a), v in grid.items()})
    print("[cora] C 完成；hard+coupled 全格纠错最大值 =",
          max(c["corrected"] for (n, _, _), v in grid.items()
              if n == "hard+coupled" for c in v))
    for name in ("original+coupled", "original+truth"):
        row = [f"a={a}:" + ",".join(
            f"{grid[(name, T, a)][0]['accT'] - grid[(name, T, a)][0]['acc0']:+.3f}"
            for T in T_LIST)
            for a in A_LIST]
        print(f"  {name}: " + " | ".join(row))


# ---------------------------------------------------------------- D
def load_walmart(max_nodes=20000):
    edges_raw = []
    with open(f"{WALMART}/hyperedges-walmart-trips.txt") as f:
        for line in f:
            e = sorted(set(int(x) - 1 for x in line.strip().split(",")))
            if all(u < max_nodes for u in e):
                edges_raw.append(e)
    labels = []
    with open(f"{WALMART}/node-labels-walmart-trips.txt") as f:
        for i, line in enumerate(f):
            if i >= max_nodes:
                break
            labels.append(int(line.strip()) - 1)
    labels = np.array(labels)
    used = sorted(set(u for e in edges_raw for u in e))
    remap = {u: i for i, u in enumerate(used)}
    edges = [tuple(remap[u] for u in e) for e in edges_raw]
    return edges, labels[used]


def part_D(doc):
    edges, labels = load_walmart()
    C = int(labels.max()) + 1
    # 组装 Dataset 兼容对象（复用 sign_predict 的向量化结构）
    class DS:
        pass
    ds = DS()
    ds.edges = [tuple(int(u) for u in e) for e in edges]
    ds.labels = labels
    ds.n = len(labels)
    ds.C = C
    ds.m = len(ds.edges)
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

    t0 = time.time()
    cells = {}
    for r in range(5):
        rng = np.random.default_rng(SEED_LP + r)
        seeds = rng.random(ds.n) < 0.20
        Zseed = sp.onehot(np.where(seeds, labels, 0), C) * seeds[:, None]
        Zlp = propagate(ds, np.ones(ds.m), Zseed, 50, 0.9)
        pred = Zlp.argmax(axis=1)
        pred[seeds] = labels[seeds]
        unseed = ~seeds
        acc_pred = float((pred[unseed] == labels[unseed]).mean())
        indep = sp.corrupt_labels(labels, C, 1 - acc_pred,
                                  np.random.default_rng(SEED_LP + 5000 + r))
        mask = unseed

        def add(name, weights, Z0, z0):
            cells.setdefault(name, []).append(
                outcome(z0, propagate(ds, weights, Z0, 20, 0.9),
                        labels, mask))

        add("hard+coupled", structure_weights(ds, pred, "hard"),
            sp.onehot(pred, C), pred)
        add("original+coupled", np.ones(ds.m), sp.onehot(pred, C), pred)
        add("soft+coupled", structure_weights(ds, pred, "soft"),
            sp.onehot(pred, C), pred)
        add("hard+indep", structure_weights(ds, pred, "hard"),
            sp.onehot(indep, C), indep)
        if r == 0:
            kept = float((structure_weights(ds, pred, "hard") == 1).mean())
            print(f"[walmart] n={ds.n} m={ds.m} C={C} "
                  f"LP acc(unseeded)={acc_pred:.4f} 种子比例={seeds.mean():.2f} "
                  f"硬筛保留边={kept:.3f}", flush=True)
    doc["walmart"] = dict(
        predictor="label propagation, 20% seeds, original structure, T=50",
        evaluation="unseeded nodes only", n_realizations=5,
        cells={k: agg(v) for k, v in cells.items()})
    print(f"[walmart] D 完成 ({time.time()-t0:.0f}s)")
    for k, v in doc["walmart"]["cells"].items():
        print(f"  {k:<20} acc0={v['acc0']:.4f} net={v['net']:+.4f} "
              f"corr={v['corrected']:+.4f} wrong={v['wrong_changed']:+.4f}")


def main():
    doc = dict(
        protocol=("v2 reinforcement: second predictor family (cross-fit "
                  "softmax logistic regression with probabilities); "
                  "mitigation-aware arms at median confidence threshold "
                  "tau (confident-filter+full one-hot init; abstain init "
                  "scored on confident nodes only); depth/alpha grid "
                  "(T in 2..40, alpha in .5.. .95, Cora prototype); "
                  "Walmart-trips with a 20%-seed label-propagation "
                  "predictor, scored on unseeded nodes. All arms share "
                  "identical predictions within a realization."),
        seeds=dict(logreg=SEED_LR, lp=SEED_LP), tags={}, )
    part_AB(doc)
    part_C(doc)
    part_D(doc)
    dest = RESULTS / "real_predictor_locking_v2.json"
    dest.write_text(json.dumps(doc, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    print(f"\n已写出 {dest}")


if __name__ == "__main__":
    main()
