"""P2-A 隔离机制定向干预实验（2026-10-08）。

补证据清单 P2-A：初始化对照只建立耦合依赖，条件核算只说明哪组节点
贡献差异，都不能唯一分离"隔离本身"。本脚本在同一实现内做配对干预：

固定预测标签 pred、耦合初始化(φ=1)、B 边翻转 ξ、传播参数
（normalized/APPNP α=0.9, T=20, p=0.10），四臂比较：
  AR0 基线：原图。
  AR1 定向重连：把结构性隔离的错误节点 u（无任何保留边）换入一条
      "全体成员预测同为 pred(u)" 的保留边（替换成员 x）——u 解除隔离，
      边仍保留。
  AR2 匹配随机重连（对照）：同一条边 e、同一个被替换成员 x，但换入
      匹配的非隔离错误节点 u'：y_{u'}=y_u、pred(u')=pred(u)（错误节点、
      同真类同预测类，只是不隔离）。修改边的真标签多元集在 AR1/AR2
      间完全一致，B 臂对同一条边复用同一 ξ 抽样 ⟹ 配对差中 x 的移除
      效应与 B 响应完全抵消，只剩"加入节点是否隔离"。
  AR3 clean 节点关系移除（附加对照）：同样数量的换边，只作用于 clean
      节点：把保留边中的 clean 节点 z 换成同真类的 clean 节点 z''
      （边仍保留、真标签类型不变），z 失去一条保留关系（稀释）。

报告（按实现配对，arm−AR0 与 AR1−AR2）：
  整体/corrupted/clean 的 margin 与 accuracy 的 A−B 对比变化，
  C_cor = phat·(mA_cor−mB_cor) 与 C_cln 的变化，定向目标节点与对照
  节点自身的 margin 变化。预先记录匹配后仍存在的差异（AR1 加入节点
  保留度为 0、AR2 ≥1；各臂修改边数占比），不宣称控制完美。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as spsp

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sign_predict as sp

RESULTS = Path(__file__).resolve().parent / "results"
TAGS = ("cora_appnp", "dblp_appnp")
P_RATE = 0.10
T_OUT = 20
N_REAL = 100
BASE_SEED = 8000        # 实现种子；与 P2-B 的 7000 块不相交
CHOICE_SEED = 8500      # 臂构造选择种子（边/成员/对照节点抽取）
MAX_SWAPS = 250         # 每实现最多换边数（|I| 超出时随机子集）
N_BOOT = 2000
BOOT_SEED = 99


def edge_retention_P(edge, labels, p, C):
    """P_e = Σ_c Π_{u∈e} r_u(c)，对称腐蚀 r_u(y_u)=1-p, 其余 ρ。"""
    rho = p / (C - 1)
    logs = np.empty((len(edge), C))
    for i, u in enumerate(edge):
        logs[i, :] = np.log(rho)
        logs[i, labels[u]] = np.log(1 - p)
    # 用 logsumexp 防下溢
    mx = logs.max(axis=0)
    return float(np.sum(np.exp(mx + np.log(np.exp(logs - mx).prod(axis=0)))))


def edge_hom(edge, labels):
    return 1.0 if len(set(int(labels[u]) for u in edge)) == 1 else 0.0


def build_S(rows, cols, owner, weights, n):
    """APPNP 归一化算子，按（可能已换边的）关联结构构建。"""
    keep = weights[owner] != 0.0
    ones = np.ones(int(keep.sum()))
    A = spsp.csr_matrix((ones, (rows[keep], cols[keep])), shape=(n, n))
    A = A + spsp.eye(n, format="csr")
    deg = np.asarray(A.sum(axis=1)).ravel()
    d_is = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    return (spsp.diags(d_is) @ A @ spsp.diags(d_is)).tocsr()


def patched_incidence(ds, arm, base):
    """把 arm 换过的边的新成员写进关联数组（owner/invk 分块不变）。"""
    rows = base["rows"].copy()
    cols = base["cols"].copy()
    for eid in arm.modified:
        e = arm.edges[eid]
        k = len(e)
        sl = slice(base["start"][eid], base["start"][eid] + k * k)
        rows[sl] = np.repeat(e, k)
        cols[sl] = np.tile(e, k)
    return rows, cols


def propagate_pair(ds, wA, wB, Z0, T, base, arm):
    """同初始化下 A/B 两算子的 per-node margin 与 accuracy。

    算子按 arm 换边后的成员构建（原 Dataset.build_adj 用缓存的原关联
    数组，会忽略换边，导致干预从未进入传播——已修正）。
    """
    sp.OP["mode"] = "appnp"
    sp.OP["alpha"] = 0.9
    rows, cols = patched_incidence(ds, arm, base)
    SA = build_S(rows, cols, base["owner"], wA, ds.n)
    SB = build_S(rows, cols, base["owner"], wB, ds.n)
    mA, aA = sp.flow_stats(SA, Z0, None, (T,), ds.labels)[T]
    mB, aB = sp.flow_stats(SB, Z0, None, (T,), ds.labels)[T]
    return mA, aA, mB, aB


class Arm:
    """一副修改后的边表：记录 (edge_id, new_member_set) 增量。"""

    def __init__(self, edges):
        self.edges = [list(e) for e in edges]
        self.modified = set()

    def swap(self, eid, x, v):
        e = self.edges[eid]
        e[e.index(x)] = v
        self.modified.add(eid)


def rebuild_weights(ds, arm, wA0, wB0, flip, p, pred):
    """在修改边上重算 wA、a*、P、q、s；B 复用同一 flip。"""
    wA = wA0.copy()
    wB = wB0.copy()
    for eid in arm.modified:
        e = arm.edges[eid]
        hom = edge_hom(e, ds.labels)
        wA[eid] = 1.0 if all(pred[u] == pred[e[0]] for u in e) else 0.0
        P_e = edge_retention_P(e, ds.labels, p, ds.C)
        s_e = 1.0 - 2.0 * hom
        wB[eid] = hom + (s_e if flip[eid] else 0.0)
    if not arm.modified:
        np.testing.assert_array_equal(wA, wA0)
        np.testing.assert_array_equal(wB, wB0)
    return wA, wB


def stats_of(mA, aA, mB, aB, mask_groups, phat):
    """margin/acc 的分组统计与 A−B 对比。"""
    out = {}
    for gname, idx in mask_groups.items():
        if len(idx) == 0:
            out[gname] = None
            continue
        out[gname] = dict(
            margin_AminusB=float(mA[idx].mean() - mB[idx].mean()),
            mA=float(mA[idx].mean()), mB=float(mB[idx].mean()),
            acc_AminusB=float(aA[idx].mean() - aB[idx].mean()),
            aA=float(aA[idx].mean()), aB=float(aB[idx].mean()))
    cor = mask_groups["corrupted"]
    cln = mask_groups["clean"]
    out["C_cor"] = float(phat * (mA[cor].mean() - mB[cor].mean()))
    out["C_cln"] = float((1 - phat) * (mA[cln].mean() - mB[cln].mean()))
    out["acc_A"] = float(aA.mean())
    out["acc_B"] = float(aB.mean())
    return out


def diff_stats(s1, s0):
    """arm − baseline 的逐实现差。"""
    out = {}
    for k, v in s1.items():
        if isinstance(v, dict):
            out[k] = {kk: v[kk] - s0[k][kk] for kk in v}
        elif isinstance(v, float):
            out[k] = v - s0[k]
    return out


def sub_stats(a, b):
    out = {}
    for k, v in a.items():
        if isinstance(v, dict):
            out[k] = {kk: v[kk] - b[k][kk] for kk in v}
        elif isinstance(v, float):
            out[k] = v - b[k]
    return out


def run_realization(ds, r):
    labels, C, m = ds.labels, ds.C, ds.m
    rng = np.random.default_rng(BASE_SEED + r)
    choice = np.random.default_rng(CHOICE_SEED + r)

    # 与 sign_predict.run_cell 相同的抽样顺序（couple, φ=1）
    pred = sp.corrupt_labels(labels, C, P_RATE, rng)
    corrupted = pred != labels
    wA0 = ds.all_hom(pred)
    w0 = ds.expected_da(P_RATE)
    q_rate, sgn = np.abs(w0), np.sign(w0)
    flip = rng.random(m) < q_rate
    wB0 = ds.a_star + np.where(flip, sgn, 0.0)
    rng.random(len(labels))                      # φ=1 的 coupled 掩码抽样
    Z0 = sp.onehot(pred, C)                      # 耦合初始化
    phat = float(corrupted.mean())

    # 邻接与保留度
    base = dict(rows=ds.rows, cols=ds.cols, owner=ds.owner,
                start=np.concatenate(
                    ([0], np.cumsum(ds.k.astype(np.int64) ** 2)))[:-1])
    node_edges = [[] for _ in range(ds.n)]
    for eid, e in enumerate(ds.edges):
        for u in e:
            node_edges[u].append(eid)
    deg_ret = np.zeros(ds.n)
    for u in range(ds.n):
        deg_ret[u] = sum(wA0[eid] for eid in node_edges[u])

    # 保留边按公共预测类分组
    retained_by_class = {c: [] for c in range(C)}
    common_pred = np.full(m, -1)
    for eid, e in enumerate(ds.edges):
        if wA0[eid] == 1.0:
            c = int(pred[e[0]])
            common_pred[eid] = c
            retained_by_class[c].append(eid)

    raw_deg = np.zeros(ds.n, dtype=int)
    for eid, e in enumerate(ds.edges):
        for u in e:
            raw_deg[u] += 1
    # 原始度为 0 的节点不属于"错误导致的隔离"，排除并记录
    n_deg0 = int(((raw_deg == 0) & corrupted).sum())
    isolated = [u for u in range(ds.n)
                if corrupted[u] and deg_ret[u] == 0 and raw_deg[u] > 0]
    # AR2 一级对照池：非隔离错误节点按 (y, pred) 分组（精确匹配）
    pool = {}
    # AR2 二级回退池：非隔离、pred=c（保持边保留；多为 clean 类-c 节点）
    pool2 = {}
    for u in range(ds.n):
        if deg_ret[u] > 0:
            if corrupted[u]:
                pool.setdefault((int(labels[u]), int(pred[u])), []).append(u)
            pool2.setdefault(int(pred[u]), []).append(u)

    # ---- 选择目标与配对 ----
    targets = list(isolated)
    if len(targets) > MAX_SWAPS:
        targets = list(choice.choice(targets, MAX_SWAPS, replace=False))
    used_added = set()
    used_removed = set()
    used_edge = set()
    plan = []          # (u_i, u'_i, eid, x_i)
    skipped = dict(no_edge=0, no_member=0, no_control=0)
    tier1 = 0          # 对照为同(y,pred)非隔离错误节点
    tier2 = 0          # 回退：同pred非隔离（多为 clean），记录差异
    for u in targets:
        c_u, y_u = int(pred[u]), int(labels[u])
        cand_e = [eid for eid in retained_by_class[c_u]
                  if u not in ds.edges[eid] and eid not in used_edge]
        if not cand_e:
            skipped["no_edge"] += 1
            continue
        eid = int(choice.choice(cand_e))
        e = ds.edges[eid]
        cand_x = [x for x in e
                  if deg_ret[x] >= 2 and x not in used_removed
                  and x not in used_added]
        if not cand_x:
            skipped["no_member"] += 1
            continue
        x = int(choice.choice(cand_x))
        cpool = [w for w in pool.get((y_u, c_u), [])
                 if w not in used_added and w not in used_removed
                 and w not in e and deg_ret[w] > 0]
        if cpool:
            tier1 += 1
        else:
            cpool = [w for w in pool2.get(c_u, [])
                     if w not in used_added and w not in used_removed
                     and w not in e and w != u]
            if not cpool:
                skipped["no_control"] += 1
                continue
            tier2 += 1
        up = int(choice.choice(cpool))
        plan.append((u, up, eid, x))
        used_added.update((u, up))
        used_added.add(x)  # x 不再作为加入节点
        used_removed.add(x)
        used_edge.add(eid)

    # ---- AR3：clean 节点关系移除（同样换边数）----
    clean_degre2 = [z for z in range(ds.n)
                    if not corrupted[z] and deg_ret[z] >= 2]
    plan3 = []
    used3_added, used3_removed, used3_edge = set(), set(), set()
    if len(clean_degre2) > 0:
        n3 = min(len(plan), len(clean_degre2))
        zs = list(choice.choice(clean_degre2, n3, replace=False))
        clean_by_label = {}
        for z in range(ds.n):
            if not corrupted[z]:
                clean_by_label.setdefault(int(labels[z]), []).append(z)
        for z in zs:
            y_z = int(labels[z])
            cand_e = [eid for eid in node_edges[z]
                      if wA0[eid] == 1.0 and eid not in used3_edge]
            if not cand_e:
                continue
            eid = int(choice.choice(cand_e))
            cand_x2 = [w for w in clean_by_label.get(y_z, [])
                       if w not in ds.edges[eid] and w not in used3_added
                       and w not in used3_removed and w != z]
            if not cand_x2:
                continue
            z2 = int(choice.choice(cand_x2))
            plan3.append((z, z2, eid))
            used3_added.add(z2)
            used3_removed.add(z)
            used3_edge.add(eid)

    # ---- 传播 ----
    groups = dict(
        all=np.arange(ds.n),
        corrupted=np.flatnonzero(corrupted),
        clean=np.flatnonzero(~corrupted),
        targets=np.array([p_[0] for p_ in plan], dtype=int),
        controls=np.array([p_[1] for p_ in plan], dtype=int),
        removed=np.array([p_[3] for p_ in plan], dtype=int),
    )
    if len(groups["targets"]) == 0:
        groups["targets"] = np.array([], dtype=int)
        groups["controls"] = np.array([], dtype=int)
        groups["removed"] = np.array([], dtype=int)

    res = {}
    arm0 = Arm(ds.edges)
    mA, aA, mB, aB = propagate_pair(ds, wA0, wB0, Z0, T_OUT, base, arm0)
    res["AR0"] = stats_of(mA, aA, mB, aB, groups, phat)
    mB_ar0_at_targets = mB[groups["targets"]] if len(groups["targets"]) else None

    for name, swaps in (("AR1", [(u, up, eid, x) for u, up, eid, x in plan]),
                        ("AR2", [(up, u, eid, x) for u, up, eid, x in plan])):
        arm = Arm(ds.edges)
        for add, _rm, eid, x in swaps:
            arm.swap(eid, x, add)
        wA1, wB1 = rebuild_weights(ds, arm, wA0, wB0, flip, P_RATE, pred)
        # 结构自检：AR1 中每个目标节点确实换入了一条保留边
        if name == "AR1":
            for u_i, _up, eid, x_i in plan:
                assert u_i in arm.edges[eid] and x_i not in arm.edges[eid]
                assert wA1[eid] == 1.0
        mA1, aA1, mB1, aB1 = propagate_pair(ds, wA1, wB1, Z0, T_OUT,
                                            base, arm)
        res[name] = stats_of(mA1, aA1, mB1, aB1, groups, phat)
        res[name]["mB_at_AR0_targets"] = (
            float(mB1[groups["targets"]].mean())
            if len(groups["targets"]) else None)

    arm3 = Arm(ds.edges)
    for z, z2, eid in plan3:
        arm3.swap(eid, z, z2)
    wA3, wB3 = rebuild_weights(ds, arm3, wA0, wB0, flip, P_RATE, pred)
    mA3, aA3, mB3, aB3 = propagate_pair(ds, wA3, wB3, Z0, T_OUT, base, arm3)
    res["AR3"] = stats_of(mA3, aA3, mB3, aB3, groups, phat)

    diag = dict(
        n_isolated=len(isolated), n_deg0_excluded=n_deg0,
        n_targets=len(plan), control_tier1=tier1, control_tier2=tier2,
        control_corrupted_frac=float(np.mean(
            [corrupted[p_[1]] for p_ in plan])) if plan else None,
        skipped=skipped,
        n_clean_swaps=len(plan3),
        mod_edge_frac=len(plan) / m,
        target_deg_all=np.array([deg_ret[u] for u in
                                  [p_[0] for p_ in plan]]),
        control_deg_ret_pre=np.array([deg_ret[p_[1]] for p_ in plan]),
        edge_sizes=np.array([len(ds.edges[p_[2]]) for p_ in plan]),
        mB_ar0_at_targets=(float(mB_ar0_at_targets.mean())
                           if mB_ar0_at_targets is not None
                           and len(mB_ar0_at_targets) else None),
    )
    return res, diag


def agg(name, per_real, key_path):
    vals = np.array([r[name] for r in per_real], dtype=float)
    return vals


def boot_ci(vals, n_boot=N_BOOT, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), (n_boot, len(vals)))
    means = vals[idx].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def collect(rows_stats, field):
    """field 如 ('AR1','targets','margin_AminusB')；空组计 0。"""
    out = []
    for r in rows_stats:
        v = r
        for k in field:
            if not isinstance(v, dict) or k not in v:
                v = None
                break
            v = v[k]
        out.append(v if isinstance(v, (int, float)) else 0.0)
    return np.array(out, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=int, default=0)
    args = ap.parse_args()
    doc = dict(
        protocol=("targeted rewiring intervention on isolated corrupted "
                  "nodes vs matched random rewiring; fixed pred labels, "
                  "coupled init phi=1, fixed B flips, APPNP alpha=0.9, "
                  "T=20, p=0.10; same edge and same removed member shared "
                  "by AR1/AR2; matched added node: corrupted, non-isolated, "
                  "same true label and same predicted class; AR3: clean-"
                  "node relation-removal reshuffle"),
        seeds=dict(realization_base=BASE_SEED, choice_base=CHOICE_SEED,
                   max_swaps=MAX_SWAPS, n_real=N_REAL),
        tags={})
    for tag in TAGS:
        data_key = tag.replace("_appnp", "")
        sp.OP["mode"] = "appnp"
        sp.OP["alpha"] = 0.9
        ds = sp.Dataset(data_key)
        print(f"[{tag}] n={ds.n} m={ds.m} C={ds.C}", flush=True)
        n_real = args.pilot or N_REAL
        rows_stats, rows_diag = [], []
        for r in range(n_real):
            t0 = time.time()
            res, diag = run_realization(ds, r)
            rows_stats.append(res)
            rows_diag.append(diag)
            if args.pilot or r % 20 == 19:
                print(f"  r={r} |I|={diag['n_isolated']} K={diag['n_targets']} "
                      f"tier1={diag['control_tier1']} tier2={diag['control_tier2']} "
                      f"ctr_cor={diag['control_corrupted_frac']} "
                      f"skip={sum(diag['skipped'].values())} "
                      f"clean_sw={diag['n_clean_swaps']} "
                      f"({time.time()-t0:.1f}s)", flush=True)
        if args.pilot:
            degs = np.concatenate([d["control_deg_ret_pre"]
                                   for d in rows_diag if len(
                                       d["control_deg_ret_pre"])])
            print("  对照节点换前保留度: min/med/max",
                  degs.min(), np.median(degs), degs.max())
            es = np.concatenate([d["edge_sizes"] for d in rows_diag
                                 if len(d["edge_sizes"])])
            print("  修改边大小: min/med/max", es.min(),
                  int(np.median(es)), es.max())
            continue

        tagdoc = dict(
            n_realizations=n_real,
            mean_isolated=float(np.mean(
                [d["n_isolated"] for d in rows_diag])),
            mean_targets=float(np.mean(
                [d["n_targets"] for d in rows_diag])),
            mean_mod_edge_frac=float(np.mean(
                [d["mod_edge_frac"] for d in rows_diag])),
            total_skipped={k: int(sum(d["skipped"][k]
                                      for d in rows_diag))
                           for k in rows_diag[0]["skipped"]},
            mean_deg0_excluded=float(np.mean(
                [d["n_deg0_excluded"] for d in rows_diag])),
            total_control_tier1=int(sum(d["control_tier1"]
                                        for d in rows_diag)),
            total_control_tier2=int(sum(d["control_tier2"]
                                        for d in rows_diag)),
            mean_control_corrupted_frac=float(np.mean(
                [d["control_corrupted_frac"] for d in rows_diag
                 if d["control_corrupted_frac"] is not None])),
            control_deg_ret_pre_median=float(np.median(np.concatenate(
                [d["control_deg_ret_pre"] for d in rows_diag
                 if len(d["control_deg_ret_pre"])]))),
            edge_size_median=float(np.median(np.concatenate(
                [d["edge_sizes"] for d in rows_diag
                 if len(d["edge_sizes"])]))),
            contrasts=[])
        # 逐实现配对差
        for arm in ("AR1", "AR2", "AR3"):
            for grp in ("all", "corrupted", "clean", "targets",
                        "controls", "removed"):
                d = collect(rows_stats, (arm, grp, "margin_AminusB")) - \
                    collect(rows_stats, ("AR0", grp, "margin_AminusB"))
                da = collect(rows_stats, (arm, grp, "acc_AminusB")) - \
                    collect(rows_stats, ("AR0", grp, "acc_AminusB"))
                tagdoc["contrasts"].append(dict(
                    arm=arm, group=grp, quantity="margin_AminusB",
                    mean=float(d.mean()), ci=boot_ci(d)))
                tagdoc["contrasts"].append(dict(
                    arm=arm, group=grp, quantity="acc_AminusB",
                    mean=float(da.mean()), ci=boot_ci(da)))
            for q in ("C_cor", "C_cln", "acc_A", "acc_B"):
                d = collect(rows_stats, (arm, q)) - \
                    collect(rows_stats, ("AR0", q))
                tagdoc["contrasts"].append(dict(
                    arm=arm, group="summary", quantity=q,
                    mean=float(d.mean()), ci=boot_ci(d)))
        # AR1−AR2 直接配对差
        for grp in ("all", "corrupted", "clean", "targets", "controls"):
            d = collect(rows_stats, ("AR1", grp, "margin_AminusB")) - \
                collect(rows_stats, ("AR2", grp, "margin_AminusB"))
            da = collect(rows_stats, ("AR1", grp, "acc_AminusB")) - \
                collect(rows_stats, ("AR2", grp, "acc_AminusB"))
            tagdoc["contrasts"].append(dict(
                arm="AR1-AR2", group=grp, quantity="margin_AminusB",
                mean=float(d.mean()), ci=boot_ci(d)))
            tagdoc["contrasts"].append(dict(
                arm="AR1-AR2", group=grp, quantity="acc_AminusB",
                mean=float(da.mean()), ci=boot_ci(da)))
        for q in ("C_cor", "C_cln"):
            d = collect(rows_stats, ("AR1", q)) - collect(rows_stats, ("AR2", q))
            tagdoc["contrasts"].append(dict(
                arm="AR1-AR2", group="summary", quantity=q,
                mean=float(d.mean()), ci=boot_ci(d)))
        # 基线水平（供解释）
        base = {}
        for grp in ("corrupted", "clean"):
            base[f"margin_AB_{grp}"] = float(np.mean(collect(
                rows_stats, ("AR0", grp, "margin_AminusB"))))
            base[f"mA_{grp}"] = float(np.mean(collect(
                rows_stats, ("AR0", grp, "mA"))))
            base[f"mB_{grp}"] = float(np.mean(collect(
                rows_stats, ("AR0", grp, "mB"))))
        base["C_cor"] = float(np.mean(collect(rows_stats, ("AR0", "C_cor"))))
        base["C_cln"] = float(np.mean(collect(rows_stats, ("AR0", "C_cln"))))
        base["mA_targets"] = float(np.mean(collect(
            rows_stats, ("AR0", "targets", "mA"))))
        base["mB_targets"] = float(np.mean(collect(
            rows_stats, ("AR0", "targets", "mB"))))
        tagdoc["baseline"] = base
        doc["tags"][tag] = tagdoc
        print(f"[{tag}] 基线 C_cor={base['C_cor']:+.5f} "
              f"C_cln={base['C_cln']:+.5f} "
              f"目标节点 mA={base['mA_targets']:.4f} "
              f"mB={base['mB_targets']:.4f}")
        for c in tagdoc["contrasts"]:
            if c["arm"] in ("AR1", "AR1-AR2") and c["quantity"] in (
                    "margin_AminusB", "C_cor", "C_cln"):
                print(f"  {c['arm']:<7} {c['group']:<10} {c['quantity']:<14} "
                      f"{c['mean']:+.6f} [{c['ci'][0]:+.6f},{c['ci'][1]:+.6f}]")
    if not args.pilot:
        dest = RESULTS / "isolation_intervention.json"
        dest.write_text(json.dumps(doc, ensure_ascii=False, indent=1),
                        encoding="utf-8")
        print(f"\n已写出 {dest}")


if __name__ == "__main__":
    main()
