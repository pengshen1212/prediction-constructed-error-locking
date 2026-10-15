"""真实训练预测构图的自确认与纠错实验（评审第九轮，2026-10-09）。

评审定位：Paper2 的关键补充——不用注入噪声，而用真实训练得到的预测
（5 折交叉拟合原型分类器，每个节点的预测来自未训练该节点的模型，
即天然 held-out）同时构图与初始化，检验：

  1) 类别锁定（结构命题）：硬同类筛边 + 非负归一化传播 + argmax=预测
     的任意初始化（含软分数）⟹ argmax 永远不变，纠错率=误改率=0。
  2) 解锁途径（对照臂）：
     - 原始候选结构（不做预测筛边，全部超边保留）；
     - 软关系（边权 = 预测同类比例，允许跨预测类连接）；
     - 解耦初始化（图来自预测 P1，初始化来自同质量独立预测 P2）；
     - 真值 / 独立初始化（率匹配）。
  3) 代价：解锁臂同时引入误改（init 对 → 最终错），报告净变化。

指标（全部节点，逐实现配对）：acc(0)、acc(T=20)、纠错率（0→1 份额）、
误改率（1→0 份额）、净变化。协议与主文一致：单位权重、α=0.9、T=20。
预测质量在同一实现内对所有臂完全相同（同一 P1 构图），跨臂可比；
预测器的 held-out 精度逐实现记录。
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

from crossfit_v2 import crossfit_predict_sparse

RESULTS = Path(__file__).resolve().parent / "results"
T_OUT = 20
N_REAL = {"cora": 20, "dblp": 10}
SEED_GRAPH = 100      # P1（构图+耦合初始化）
SEED_INIT = 300       # P2（解耦初始化，同质量独立抽取）
SEED_INDEP = 500      # 率匹配独立腐蚀
SOFT_EPS = 0.3        # 软初始化扰动幅度（argmax 保持不变）


def soft_init(pred, C, rng):
    """argmax=pred 的软分数初始化：one-hot + 均匀+随机扰动后归一。"""
    Z = np.full((len(pred), C), SOFT_EPS / C)
    Z[np.arange(len(pred)), pred] += 1.0 - SOFT_EPS / 2
    Z += rng.random(Z.shape) * SOFT_EPS / C
    assert (np.argmax(Z, axis=1) == pred).all()
    return Z


def structure_weights(ds, pred, kind):
    if kind == "original":
        return np.ones(ds.m)
    if kind == "hard":
        return ds.all_hom(pred).astype(float)
    if kind == "soft":
        cnt = np.zeros((ds.m, ds.C))
        np.add.at(cnt, (ds.owner, pred[ds.rows]), 1)
        # owner 每边 k² 项、每成员重复 k 次 ⟹ cnt_e,c = k_e·n_c；
        # 软权 = 最大预测类占比 max_c n_c/k_e
        return (cnt.max(axis=1) / ds.k.astype(float) ** 2).clip(0.0, 1.0)
    raise ValueError(kind)


def build_adj_weighted(ds, weights):
    """加权 APPNP 算子：S = D^{-1/2}(A(a)+I)D^{-1/2}，A(a)=Σ a_e 𝟙𝟙ᵀ。

    Dataset.build_adj 会把权重二值化（主协议 a∈{0,1} 下等价），
    软结构需要真正的连续权重版本。
    """
    import scipy.sparse as spsp
    vals = weights[ds.owner]
    keep = vals != 0.0
    A = spsp.csr_matrix((vals[keep], (ds.rows[keep], ds.cols[keep])),
                        shape=(ds.n, ds.n))
    A = A + spsp.eye(ds.n, format="csr")
    deg = np.asarray(A.sum(axis=1)).ravel()
    d_is = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    return (spsp.diags(d_is) @ A @ spsp.diags(d_is)).tocsr()


def run_arm(ds, weights, Z0):
    S = build_adj_weighted(ds, weights)
    sp.OP["mode"] = "appnp"
    sp.OP["alpha"] = 0.9
    _, acc = sp.flow_stats(S, Z0, None, (T_OUT,), ds.labels)[T_OUT]
    return acc


def main():
    doc = dict(
        protocol=("real cross-fit prototype predictions construct relations "
                  "and/or initialization; every node's prediction comes from "
                  "a fold that excluded it (held-out); unit-weight APPNP "
                  "alpha=0.9, T=20; structures: original (all edges), hard "
                  "same-predicted-class filter, soft cross-class weights "
                  "(max predicted-class fraction); initializations: coupled "
                  "(P1), decoupled (independent same-quality P2), truth, "
                  "independent rate-matched, soft-argmax (hard structure "
                  "only, demonstrates locking beyond one-hot)"),
        seeds=dict(graph=SEED_GRAPH, init=SEED_INIT, indep=SEED_INDEP),
        tags={})
    for tag in ("cora", "dblp"):
        ds = sp.Dataset(tag)
        feats = None
        import pickle
        with open(f"{sp.DATA_DIRS[tag]}/features.pickle", "rb") as f:
            feats = pickle.load(f)
        n_real = N_REAL[tag]
        arms = {}
        for r in range(n_real):
            t0 = time.time()
            pred1 = crossfit_predict_sparse(feats, ds.labels, SEED_GRAPH + r)
            pred2 = crossfit_predict_sparse(feats, ds.labels, SEED_INIT + r)
            err_rate = float((pred1 != ds.labels).mean())
            rng = np.random.default_rng(SEED_INDEP + r)
            indep = sp.corrupt_labels(ds.labels, ds.C, err_rate, rng)
            acc0_p1 = float((pred1 == ds.labels).mean())

            cells = []
            for struct in ("original", "hard", "soft"):
                w = structure_weights(ds, pred1, struct)
                for init_name, init in (
                        ("coupled", sp.onehot(pred1, ds.C)),
                        ("decoupled", sp.onehot(pred2, ds.C)),
                        ("truth", sp.onehot(ds.labels, ds.C)),
                        ("indep", sp.onehot(indep, ds.C))):
                    acc = run_arm(ds, w, init)
                    z0 = np.argmax(init, axis=1)
                    corr = float(((z0 != ds.labels) & (acc == 1)).mean())
                    wrong = float(((z0 == ds.labels) & (acc == 0)).mean())
                    cells.append(dict(
                        structure=struct, init=init_name,
                        acc0=float((z0 == ds.labels).mean()),
                        accT=float(acc.mean()), corrected=corr,
                        wrong_changed=wrong, net=float(acc.mean()
                                                       - (z0 == ds.labels).mean())))
            # 软初始化演示：硬筛边 + argmax 保持 pred1 的软分数
            w_hard = structure_weights(ds, pred1, "hard")
            Zsoft = soft_init(pred1, ds.C,
                              np.random.default_rng(SEED_INDEP + 9000 + r))
            acc = run_arm(ds, w_hard, Zsoft)
            cells.append(dict(
                structure="hard", init="softargmax",
                acc0=acc0_p1, accT=float(acc.mean()),
                corrected=float(((pred1 != ds.labels) & (acc == 1)).mean()),
                wrong_changed=float(((pred1 == ds.labels) & (acc == 0)).mean()),
                net=float(acc.mean() - acc0_p1)))
            for c in cells:
                arms.setdefault((c["structure"], c["init"]), []).append(c)
            if r == 0 or r == n_real - 1:
                kept = float((w_hard == 1.0).mean())
                print(f"[{tag}] r={r} pred acc={acc0_p1:.4f} "
                      f"硬筛保留边比例={kept:.3f} ({time.time()-t0:.0f}s)",
                      flush=True)

        tagdoc = dict(n_realizations=n_real, cells=[])
        idx = np.random.default_rng(99).integers(0, n_real, (2000, n_real))
        for (struct, init_name), lst in sorted(arms.items()):
            base = dict(structure=struct, init=init_name,
                        acc0=float(np.mean([c["acc0"] for c in lst])),
                        accT=float(np.mean([c["accT"] for c in lst])),
                        corrected=float(np.mean([c["corrected"] for c in lst])),
                        wrong_changed=float(np.mean(
                            [c["wrong_changed"] for c in lst])))
            base["net"] = base["accT"] - base["acc0"]
            for q in ("corrected", "wrong_changed"):
                v = np.array([c[q] for c in lst])
                base[f"{q}_ci"] = np.quantile(
                    v[idx].mean(axis=1), [0.025, 0.975]).tolist()
            tagdoc["cells"].append(base)
        doc["tags"][tag] = tagdoc
        print(f"[{tag}] 结构×初始化结果：")
        print(f"{'结构':<9} {'初始化':<11} {'acc0':>7} {'accT':>7} "
              f"{'纠错':>8} {'误改':>8}")
        for c in tagdoc["cells"]:
            print(f"{c['structure']:<9} {c['init']:<11} {c['acc0']:>7.4f} "
                  f"{c['accT']:>7.4f} {c['corrected']:>+8.4f} "
                  f"{c['wrong_changed']:>+8.4f}")
    dest = RESULTS / "real_predictor_locking.json"
    dest.write_text(json.dumps(doc, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    print(f"\n已写出 {dest}")


if __name__ == "__main__":
    main()
