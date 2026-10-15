"""符号预测冻结协议：把 Prop criterion 的事后分解升级为事前符号预测。

评审意见（2026-10-07）要求的协议，逐条落实：
  1. 校准设置（p∈{0.05,0.10}, T∈{2,5}）中估计隔离效应与干净节点传播响应；
  2. 冻结计算规则 —— 拟合只用校准配置，评估配置的实测 A/B 差异永不回写；
  3. 在未参与校准的噪声率(p∈{0.15,0.20,0.30})、深度(T∈{10,20})与
     初始化条件（耦合 φ 连续扫描 + indep 族）中提前预测 A/B 谁更好；
  4. 报告符号正确率、预测误差、接近零差异时的不确定性（bootstrap CI，
     CI 覆盖 0 记为 ambiguous 单列）。

冻结规则（margin 版；acc 版同构）：
  Δ̂(p,T,φ) = p·[ (1−2φ) − β̂(p,T,φ) ] + (1−p)·[ (1−φ)D̂₀(p,T) + φ D̂₁(p,T) ]
  β̂ = (1−φ)β̂_c + φ β̂_w
  - 1−2φ / 1−φ：被隔离腐蚀节点保留初始 margin/acc（闭式，无拟合；
    Prop criterion (ii) 的推广，ε^all 松弛同论文）
  - β_c：truth 初始化（φ=0）下 B 在腐蚀节点上的 margin（B 与腐蚀独立，
    等于总体均值）
  - β_w：couple 初始化（φ=1）下 B 在腐蚀节点上的 margin
  - D₀/D₁：φ=0 / φ=1 校准配置的干净节点对比 E[m^A−m^B | clean]
  - β_c, β_w, D₀, D₁ 各自对 (p, logT) 做 4 参数精确插值（4 校准格点）
  消融 v1：D 不随 φ 混合（只用 D₀）——检验 D 的 φ 依赖是否重要。
  机制对照（只报告不用于判号）：D̂_mech = 单边 toggle 响应 × Prop thinning
  闭式稀释率差。

种子纪律（配对设计）：同一 (kind,p,mode) 的所有 φ / q 单元格共享图实现
种子（第 r 个实现的图与翻转完全相同，仅初始条件不同），φ 曲线为配对比较。
校准种子 1000+，评估种子 5000+，冒烟 9000+（冒烟只验机制、结果丢弃）。

阶段与文件（按顺序执行，评估阶段只读预测文件）：
  cal     → results/signpred_cal.json, results/signpred_frozen_rule.json
  predict → results/signpred_predictions.json（纯规则代入，无任何实测）
  eval    → results/signpred_eval.json, signpred_report.md,
            figures/signpred_sweep.png
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

sys.stdout.reconfigure(encoding="utf-8")

DATA_DIRS = {
    "cora": "data/cora-cc",
    "dblp": "data/dblp-ca",
}

CAL_P = (0.05, 0.10, 0.15)
CAL_T = (2, 5, 10)
EVAL_GRID_P = (0.15, 0.20, 0.30)
EVAL_GRID_T = (10, 20)
EVAL_GRID_PHI = (0.25, 0.50, 0.75)
SWEEP_P = (0.10, 0.20)
SWEEP_T = (2, 5, 10, 20)
SWEEP_PHI = tuple(round(x, 2) for x in np.linspace(0, 1, 11))
INDEP_PQ = ((0.10, 0.10), (0.10, 0.30), (0.20, 0.10), (0.20, 0.30))
INDEP_T = (10, 20)
CHECKPOINTS = (2, 5, 10, 20)

BASE_CAL = 1000
BASE_EVAL = 5000
BASE_SMOKE = 9000
N_BOOT = 2000


# ----------------------------------------------------------------------
# 数据与算子
# ----------------------------------------------------------------------

class Dataset:
    def __init__(self, key):
        d = DATA_DIRS[key]
        with open(f"{d}/hypergraph.pickle", "rb") as f:
            hg = pickle.load(f)
        with open(f"{d}/labels.pickle", "rb") as f:
            labels = np.array(pickle.load(f))
        self.edges = [sorted(int(u) for u in s) for s in hg.values()]
        self.labels = labels
        self.n = len(labels)
        self.C = int(len(np.unique(labels)))
        self.m = len(self.edges)
        self.k = np.array([len(e) for e in self.edges])
        self.a_star = np.array(
            [1.0 if len(set(labels[u] for u in e)) == 1 else 0.0
             for e in self.edges])
        rows, cols, owner, invk = [], [], [], []
        for i, e in enumerate(self.edges):
            kk = len(e)
            rows.append(np.repeat(e, kk))
            cols.append(np.tile(e, kk))
            owner.append(np.full(kk * kk, i))
            invk.append(np.full(kk * kk, 1.0 / kk))
        self.rows = np.concatenate(rows)
        self.cols = np.concatenate(cols)
        self.owner = np.concatenate(owner)
        self.invk = np.concatenate(invk)
        kmax = int(self.k.max())
        pad = np.full((self.m, kmax), -1, dtype=np.int64)
        for i, e in enumerate(self.edges):
            pad[i, :len(e)] = e
        self.pad = pad
        self.B_clean = self.build_op(self.a_star)
        lam = eigsh(self.B_clean, k=1, which="LM",
                    return_eigenvectors=False, tol=1e-6)[0]
        self.h = 1.0 / float(abs(lam))

    def build_op(self, weights):
        vals = self.invk * weights[self.owner]
        keep = vals != 0.0
        return sp.csr_matrix(
            (vals[keep], (self.rows[keep], self.cols[keep])),
            shape=(self.n, self.n))

    def build_adj(self, weights):
        """APPNP 用：保留边的二元邻接（无 invk 权重）。"""
        keep = weights[self.owner] != 0.0
        ones = np.ones(int(keep.sum()))
        r, c = self.rows[keep], self.cols[keep]
        A = sp.csr_matrix((ones, (r, c)), shape=(self.n, self.n))
        A = A + sp.eye(self.n, format="csr")
        deg = np.asarray(A.sum(axis=1)).ravel()
        d_is = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
        return (sp.diags(d_is) @ A @ sp.diags(d_is)).tocsr()

    def all_hom(self, pred):
        """向量化：边内预测全一致 → 权重 1（与逐边 set() 版完全等价）。"""
        pv = pred[self.pad]
        valid = self.pad >= 0
        pmn = np.min(np.where(valid, pv, 1 << 30), axis=1)
        pmx = np.max(np.where(valid, pv, -1), axis=1)
        return (pmn == pmx).astype(float)

    def expected_da(self, p):
        """匹配边际的期望 D_A（论文 Prop thinning (iii) 的 ph − a*），
        向量化：ph_e = Σ_c (1−p)^{n_{e,c}} ρ^{k_e−n_{e,c}}。"""
        rho = p / (self.C - 1)
        cnt = np.zeros((self.m, self.C))
        np.add.at(cnt, (self.owner, self.labels[self.rows]), 1)
        cnt /= self.k[:, None]     # rows 在每边 k×k 块中重复 k 次
        ph = np.exp(cnt * np.log(1 - p)
                    + (self.k[:, None] - cnt) * np.log(rho)).sum(axis=1)
        return ph - self.a_star


def corrupt_labels(labels, n_cls, p, rng):
    mask = rng.random(len(labels)) < p
    j = rng.integers(0, n_cls - 1, size=len(labels))
    return np.where(mask, j + (j >= labels), labels)


def onehot(labels, n_cls):
    Z = np.zeros((len(labels), n_cls))
    Z[np.arange(len(labels)), labels] = 1.0
    return Z


def flow_stats(B, Z0, h, checkpoints, labels):
    """一次扫描记录所有 checkpoint 的 margin/acc。
    B 为热流算子（h ≤ 1/λmax）或 APPNP 的 S（OP['mode']=='appnp' 时
    Z←αSZ+(1−α)Z0，h 不用）。"""
    n = len(labels)
    Z = Z0.copy()
    out = {}
    nxt = sorted(checkpoints)
    for t in range(1, max(checkpoints) + 1):
        if OP["mode"] == "appnp":
            Z = OP["alpha"] * (B @ Z) + (1 - OP["alpha"]) * Z0
        else:
            Z = Z - h * (B @ Z)
        if nxt and t == nxt[0]:
            nxt.pop(0)
            own = Z[np.arange(n), labels]
            Zo = Z.copy()
            Zo[np.arange(n), labels] = -np.inf
            margin = own - Zo.max(axis=1)
            out[t] = (margin, np.argmax(Z, axis=1) == labels)
    return out


def op_of(ds, weights):
    return ds.build_adj(weights) if OP["mode"] == "appnp" \
        else ds.build_op(weights)


# ----------------------------------------------------------------------
# 单元格测量
# ----------------------------------------------------------------------

KEYS = ("mA", "mB", "mA_cor", "mB_cor", "mA_cln", "mB_cln",
        "aA", "aB", "aA_cor", "aB_cor", "aA_cln", "aB_cln", "phat")


def run_cell(ds, p, mode, phi=None, q=None, *, n_real, base_seed):
    """mode='couple'：腐蚀节点以概率 φ 携带耦合错误初始标签，干净节点
    恒为真标签（φ=0 即 clean init，φ=1 即论文 couple 模式）；
    mode='indep'：全体节点独立翻转率 q。逐实现统计。"""
    n, C, m = ds.n, ds.C, ds.m
    labels = ds.labels
    w0 = ds.expected_da(p)
    q_rate, sgn = np.abs(w0), np.sign(w0)
    rec = {t: {k: [] for k in KEYS} for t in CHECKPOINTS}
    for r in range(n_real):
        rng = np.random.default_rng(base_seed + r)
        pred = corrupt_labels(labels, C, p, rng)
        corrupted = pred != labels
        wA = ds.all_hom(pred)
        flip = rng.random(m) < q_rate
        wB = ds.a_star + np.where(flip, sgn, 0.0)
        if mode == "couple":
            coupled = corrupted & (rng.random(n) < phi)
            init = np.where(coupled, pred, labels)
        elif mode == "indep":
            init = corrupt_labels(labels, C, q, rng)
        else:
            raise ValueError(mode)
        Z0 = onehot(init, C)
        stA = flow_stats(op_of(ds, wA), Z0, ds.h, CHECKPOINTS, labels)
        stB = flow_stats(op_of(ds, wB), Z0, ds.h, CHECKPOINTS, labels)
        for t in CHECKPOINTS:
            mA, aA = stA[t]
            mB, aB = stB[t]
            d = rec[t]
            d["mA"].append(float(mA.mean()))
            d["mB"].append(float(mB.mean()))
            d["mA_cor"].append(float(mA[corrupted].mean()))
            d["mB_cor"].append(float(mB[corrupted].mean()))
            d["mA_cln"].append(float(mA[~corrupted].mean()))
            d["mB_cln"].append(float(mB[~corrupted].mean()))
            d["aA"].append(float(aA.mean()))
            d["aB"].append(float(aB.mean()))
            d["aA_cor"].append(float(aA[corrupted].mean()))
            d["aB_cor"].append(float(aB[corrupted].mean()))
            d["aA_cln"].append(float(aA[~corrupted].mean()))
            d["aB_cln"].append(float(aB[~corrupted].mean()))
            d["phat"].append(float(corrupted.mean()))
    return {t: {k: np.array(v) for k, v in d.items()}
            for t, d in rec.items()}


def cell_summary(cell):
    """均值 + A−B 差异的 bootstrap CI。"""
    rng = np.random.default_rng(99)
    n = len(next(iter(cell[CHECKPOINTS[0]].values())))
    ia = rng.integers(0, n, (N_BOOT, n))
    ib = rng.integers(0, n, (N_BOOT, n))
    out = {}
    for t, d in cell.items():
        dm = d["mA"][ia].mean(1) - d["mB"][ib].mean(1)
        da = d["aA"][ia].mean(1) - d["aB"][ib].mean(1)
        out[t] = dict(
            d_margin=float(d["mA"].mean() - d["mB"].mean()),
            d_margin_ci=[float(x) for x in np.quantile(dm, [.025, .975])],
            d_acc=float(d["aA"].mean() - d["aB"].mean()),
            d_acc_ci=[float(x) for x in np.quantile(da, [.025, .975])],
            D_clean_margin=float(d["mA_cln"].mean() - d["mB_cln"].mean()),
            D_clean_acc=float(d["aA_cln"].mean() - d["aB_cln"].mean()),
            G_cor_margin=float(d["mA_cor"].mean() - d["mB_cor"].mean()),
            mB_cor=float(d["mB_cor"].mean()),
            aB_cor=float(d["aB_cor"].mean()),
            mB_pop=float(d["mB"].mean()),
            m_A_cor=float(d["mA_cor"].mean()),
            phat=float(d["phat"].mean()),
        )
    return out


# ----------------------------------------------------------------------
# 校准
# ----------------------------------------------------------------------

def alpha_toggles(ds, depths, n_toggle, seed):
    """oracle 图上随机移除单条 hom 边的均值 margin 响应（折算到每条
    受影响节点）。用于机制对照 D̂_mech，不进入判号规则。"""
    rng = np.random.default_rng(seed)
    hom_idx = np.flatnonzero(ds.a_star == 1.0)
    Z0 = onehot(ds.labels, ds.C)
    base = flow_stats(op_of(ds, ds.a_star), Z0, ds.h, depths, ds.labels)
    per_node_resp = {t: [] for t in depths}
    sample = rng.choice(hom_idx, size=min(n_toggle, len(hom_idx)),
                        replace=False)
    for i in sample:
        w = ds.a_star.copy()
        w[i] = 0.0
        st = flow_stats(op_of(ds, w), Z0, ds.h, depths, ds.labels)
        for t in depths:
            per_node_resp[t].append(
                (st[t][0].mean() - base[t][0].mean()) * ds.n / ds.k[i])
    return {t: float(np.mean(v)) for t, v in per_node_resp.items()}


def basis(p, T):
    # v4 基函数：在 Cora 开发集上从 {logT, linT, T+logT, full} 中选出
    # （held-out 符号 99%、T=20 转折误差 0.028），dblp 上预注册使用
    return np.array([1.0, p, float(T), np.log(T), p * T, p * np.log(T)])


def fit_map(cells):
    """校准格点上的 OLS（v3：3p×3T=9 格点 > 4 参数，带残差自由度）。"""
    keys = sorted(cells)
    X = np.array([basis(p, T) for p, T in keys])
    y = np.array([cells[k] for k in keys])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def eval_map(coef, p, T):
    return float(basis(p, T) @ coef)


def mech_loss_diff(ds, p):
    """clean 节点人均 (loss_A − loss_B) 的闭式均值（Prop thinning）。"""
    rho = p / (ds.C - 1)
    keepA = np.power(1 - p, ds.k - 1)
    keepB = np.power(1 - p, ds.k) + (ds.C - 1) * np.power(rho, ds.k)
    per_edge = (1 - keepA) - (1 - keepB)   # loss_A − loss_B
    hom = ds.a_star == 1.0
    node_diff = np.zeros(ds.n)
    members = np.concatenate([np.asarray(e) for e, a in
                              zip(ds.edges, ds.a_star) if a == 1])
    np.add.at(node_diff, members, np.repeat(per_edge[hom], ds.k[hom]))
    return float(node_diff.mean())


def stage_cal(ds, n_real, base):
    cal = {"cells": {}, "n_real": n_real, "base_seed": base}
    for p in CAL_P:
        for phi, tag in ((0.0, "phi0"), (1.0, "phi1")):
            t0 = time.time()
            cell = run_cell(ds, p, "couple", phi=phi, n_real=n_real,
                            base_seed=base)
            cal["cells"][f"p={p}|{tag}"] = cell_summary(cell)
            print(f"  cal p={p} {tag} 用时 {time.time()-t0:.1f}s",
                  flush=True)
    cal["alpha_toggle"] = alpha_toggles(ds, CAL_T, n_toggle=120,
                                        seed=base + 900)

    def grab(tag, key):
        return {(p, t): cal["cells"][f"p={p}|{tag}"][t][key]
                for p in CAL_P for t in CAL_T}

    x = np.log(np.array(CAL_T, dtype=float))
    yv = np.array([cal["alpha_toggle"][t] for t in CAL_T])
    rule = {
        "protocol": "frozen sign-prediction v4 (cal p<=0.15, T<=10, OLS, "
                    "basis [1,p,T,logT,pT,plogT], D phi-mixed)",
        "cal_P": list(CAL_P), "cal_T": list(CAL_T), "n_real": n_real,
        "base_seed_cal": base, "dataset": DATASET_NAME,
        "beta_c_m": fit_map(grab("phi0", "mB_cor")).tolist(),
        "beta_w_m": fit_map(grab("phi1", "mB_cor")).tolist(),
        "D0_m": fit_map(grab("phi0", "D_clean_margin")).tolist(),
        "D1_m": fit_map(grab("phi1", "D_clean_margin")).tolist(),
        "beta_c_a": fit_map(grab("phi0", "aB_cor")).tolist(),
        "beta_w_a": fit_map(grab("phi1", "aB_cor")).tolist(),
        "D0_a": fit_map(grab("phi0", "D_clean_acc")).tolist(),
        "D1_a": fit_map(grab("phi1", "D_clean_acc")).tolist(),
        "alpha_toggle_per_node": cal["alpha_toggle"],
        "alpha_loglin": np.polyfit(x, yv, 1).tolist(),
        "mech_loss_diff": {f"{p}": mech_loss_diff(ds, p)
                           for p in sorted(set(CAL_P) | set(EVAL_GRID_P)
                                           | set(SWEEP_P))},
    }
    return cal, rule


def predict_rule(rule, p, T, phi, metric="m", variant="v2"):
    """冻结规则。variant: v2(D φ 混合) / v1(D=D0) / mech(D=toggle×闭式)。"""
    beta_c = eval_map(np.array(rule[f"beta_c_{metric}"]), p, T)
    beta_w = eval_map(np.array(rule[f"beta_w_{metric}"]), p, T)
    beta = (1 - phi) * beta_c + phi * beta_w
    init = (1 - 2 * phi) if metric == "m" else (1 - phi)
    if variant == "v1":
        D = eval_map(np.array(rule[f"D0_{metric}"]), p, T)
    elif variant == "mech":
        a0, a1 = rule["alpha_loglin"]
        alpha = a0 + a1 * np.log(T)
        D = alpha * rule["mech_loss_diff"][f"{p}"]
    else:
        D0 = eval_map(np.array(rule[f"D0_{metric}"]), p, T)
        D1 = eval_map(np.array(rule[f"D1_{metric}"]), p, T)
        D = (1 - phi) * D0 + phi * D1
    return p * (init - beta) + (1 - p) * D


def crossing_phi(fn, lo=0.0, hi=1.0, step=1e-3):
    grid = np.arange(lo, hi + step / 2, step)
    v = np.array([fn(x) for x in grid])
    idx = np.flatnonzero(v[:-1] * v[1:] < 0)
    if len(idx) == 0:
        return None
    i = int(idx[0])
    x0, x1, v0, v1 = grid[i], grid[i + 1], v[i], v[i + 1]
    return float(x0 + (x1 - x0) * (-v0) / (v1 - v0))


def stage_predict(rule, path):
    configs = []
    for p in EVAL_GRID_P:
        for T in EVAL_GRID_T:
            for phi in EVAL_GRID_PHI:
                configs.append(dict(kind="grid", p=p, T=T, phi=phi,
                                    mode="couple"))
    for p in SWEEP_P:
        for T in SWEEP_T:
            for phi in SWEEP_PHI:
                configs.append(dict(kind="sweep", p=p, T=T, phi=phi,
                                    mode="couple"))
    for p, q in INDEP_PQ:
        for T in INDEP_T:
            configs.append(dict(kind="indep", p=p, T=T, phi=q,
                                mode="indep", q=q))
    preds = []
    for c in configs:
        row = dict(c)
        for metric in ("m", "a"):
            for variant in ("v2", "v1", "mech"):
                row[f"pred_{metric}_{variant}"] = predict_rule(
                    rule, c["p"], c["T"], c["phi"], metric, variant)
        row["phi_cross_v2_m"] = crossing_phi(
            lambda x, p=c["p"], T=c["T"]: predict_rule(
                rule, p, T, x, "m", "v2"))
        row["phi_cross_v2_a"] = crossing_phi(
            lambda x, p=c["p"], T=c["T"]: predict_rule(
                rule, p, T, x, "a", "v2"))
        preds.append(row)
    doc = {"rule_file": f"signpred_{DATASET_NAME}"
                        f"{'' if OP['mode'] == 'heat' else '_' + OP['mode']}"
                        f"_frozen_rule.json",
           "configs": preds,
           "note": "纯规则代入生成于评估之前；评估阶段只读不写"}
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    return doc


# ----------------------------------------------------------------------
# 评估
# ----------------------------------------------------------------------

def cell_seed(base, kind, p, mode):
    """同一 (kind,p,mode) 共享种子 → φ 曲线配对。"""
    return base + 100 * round(p * 100) + (0 if mode == "couple" else 50)


def stage_eval(ds, preds, n_real, base):
    groups = {}
    for c in preds["configs"]:
        groups.setdefault((c["kind"], c["p"], c["mode"],
                           c.get("phi"), c.get("q")), set()).add(c["T"])
    results = {}
    raw = {}
    for i, (gk, Ts) in enumerate(sorted(groups.items(), key=str)):
        kind, p, mode, phi, q = gk
        t0 = time.time()
        cell = run_cell(ds, p, mode, phi=phi, q=q, n_real=n_real,
                        base_seed=cell_seed(base, kind, p, mode))
        summ = cell_summary(cell)
        print(f"  eval {gk} 用时 {time.time()-t0:.1f}s", flush=True)
        raw[str(gk)] = {str(t): {k: v.tolist() for k, v in d.items()}
                        for t, d in cell.items()}
        for T in Ts:
            results[str((kind, p, T, mode, phi, q))] = summ[T]
    return results, raw


def boot_crossing(raw, p, T, phis, n_boot=1000):
    """配对 bootstrap：同一 (p,mode) 的各 φ 单元格共享实现种子，
    同一重采样索引作用于全部 φ，得到实测转折 φ* 的 CI。"""
    curves = []
    for phi in phis:
        d = raw[str(("sweep", p, "couple", phi, None))][str(T)]
        curves.append(np.array(d["mA"]) - np.array(d["mB"]))
    M = np.array(curves)                    # (n_phi, n_real)
    n = M.shape[1]
    rng = np.random.default_rng(123)
    ia = rng.integers(0, n, (n_boot, n))
    crosses = []
    for b in range(n_boot):
        m = M[:, ia[b]].mean(axis=1)
        ph = np.linspace(0, 1, 201)
        v = np.interp(ph, phis, m)
        c = crossing_phi(lambda x, ph=ph, v=v: np.interp(x, ph, v),
                         step=0.005)
        if c is not None:
            crosses.append(c)
    if not crosses:
        return None, [None, None], 0.0
    lo, hi = np.quantile(crosses, [.025, .975])
    crosses = np.array(crosses)
    center = float(np.median(crosses))
    return center, [float(lo), float(hi)], float(len(crosses) / n_boot)


def summarize(rows, rule, raw=None):
    def table(metric, variant):
        sub = [r for r in rows
               if r.get(f"sign_ok_{metric}_{variant}") is not None]
        amb = sum(1 for r in rows if r[f"ambiguous_{metric}"])
        ok = sum(r[f"sign_ok_{metric}_{variant}"] for r in sub)
        errs = np.array([r[f"err_{metric}_{variant}"] for r in rows])
        meass = np.array([r[f"meas_{metric}"] for r in rows])
        return dict(
            n=len(rows), n_unambiguous=len(sub), n_ambiguous=amb,
            sign_correct=int(ok),
            sign_rate=(ok / len(sub)) if sub else None,
            mae=float(np.abs(errs).mean()),
            mae_over_range=float(np.abs(errs).mean()
                                 / max(np.ptp(meass), 1e-12)),
            corr=float(np.corrcoef(errs + meass, meass)[0, 1])
            if len(rows) > 2 and np.ptp(errs + meass) > 0 else None)

    out = {f"{metric}|{variant}": table(metric, variant)
           for metric in ("m", "a") for variant in ("v2", "v1", "mech")}
    trans = []
    for p in SWEEP_P:
        for T in SWEEP_T:
            pts = sorted((r for r in rows if r["kind"] == "sweep"
                          and r["p"] == p and r["T"] == T),
                         key=lambda r: r["phi"])
            phis = np.array([r["phi"] for r in pts])
            meas = np.array([r["meas_m"] for r in pts])
            los = np.array([r["ci_m"][0] for r in pts])
            his = np.array([r["ci_m"][1] for r in pts])
            pred = np.array([predict_rule(rule, p, T, x, "m", "v2")
                             for x in phis])
            phi_meas = crossing_phi(lambda x: np.interp(x, phis, meas))
            phi_boot, phi_ci, boot_rate = (None, [None, None], 0.0)
            if raw is not None:
                phi_boot, phi_ci, boot_rate = boot_crossing(
                    raw, p, T, phis)
            trans.append(dict(
                p=p, T=T, phi_meas=phi_meas,
                phi_boot=phi_boot, phi_boot_ci=phi_ci,
                boot_cross_rate=boot_rate,
                phi_pred=crossing_phi(lambda x: np.interp(x, phis, pred)),
                meas=meas.tolist(), lo=los.tolist(), hi=his.tolist(),
                pred=pred.tolist(), phis=phis.tolist()))
    return out, trans


def make_figure(trans, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(
        len(SWEEP_P), len(SWEEP_T),
        figsize=(3.6 * len(SWEEP_T), 3.0 * len(SWEEP_P)), squeeze=False)
    for i, p in enumerate(SWEEP_P):
        for j, T in enumerate(SWEEP_T):
            tr = next(t for t in trans if t["p"] == p and t["T"] == T)
            ax = axes[i][j]
            x = tr["phis"]
            ax.fill_between(x, tr["lo"], tr["hi"], alpha=0.25,
                            label="measured 95% CI")
            ax.plot(x, tr["meas"], "o-", ms=4, label="measured Δ")
            ax.plot(x, tr["pred"], "s--", ms=4, label="frozen Δ̂")
            ax.axhline(0, color="k", lw=0.6)
            if tr["phi_meas"] is not None:
                ax.axvline(tr["phi_meas"], color="C0", ls=":")
            if tr["phi_pred"] is not None:
                ax.axvline(tr["phi_pred"], color="C1", ls=":")
            ax.set_title(f"p={p}, T={T}")
            ax.set_xlabel("coupling φ")
            if j == 0:
                ax.set_ylabel("Δ margin (A−B)")
    h1, l1 = axes[0][0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    h1 += [Line2D([], [], color="C0", ls=":"), Line2D([], [], color="C1",
                                                      ls=":")]
    l1 += ["measured φ*", "predicted φ̂*"]
    fig.legend(h1, l1, loc="lower center", ncol=4, fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def write_report(outdir, rows, summary, trans, rule, n_real, results):
    L = ["# 符号预测冻结协议结果", ""]
    L.append(f"数据集: {rule['dataset']} (n={DS.n}, C={DS.C}, m={DS.m})；"
             f"校准 p∈{CAL_P}, T∈{CAL_T}（{n_real} 实现，种子 "
             f"{rule['base_seed_cal']}+，φ=0/1 两端点配对同图）；评估种子 "
             f"{BASE_EVAL}+；bootstrap {N_BOOT}。")
    L.append("")
    L.append("## 冻结规则系数（basis [1, p, logT, p·logT]）")
    for k in ("beta_c_m", "beta_w_m", "D0_m", "D1_m", "beta_c_a",
              "beta_w_a", "D0_a", "D1_a"):
        L.append(f"- {k}: {[round(float(x), 5) for x in rule[k]]}")
    L.append(f"- alpha_loglin (logT 线性): "
             f"{[round(float(x), 6) for x in rule['alpha_loglin']]}")
    L.append("")
    L.append("## 符号正确率 / 预测误差")
    for key, v in summary.items():
        if v["sign_rate"] is None:
            continue
        L.append(f"- **{key}**: {v['sign_correct']}/{v['n_unambiguous']} "
                 f"({v['sign_rate']:.0%})，ambiguous {v['n_ambiguous']}，"
                 f"MAE={v['mae']:.5f}（量程的 {v['mae_over_range']:.1%}），"
                 f"corr={v['corr'] if v['corr'] is None else round(v['corr'], 3)}")
    L.append("")
    L.append("## 转折定位（φ 扫描，margin）")
    L.append("| p | T | 实测 φ* | bootstrap CI | 预测 φ̂* | |差| |")
    L.append("|---|---|---------|--------------|---------|------|")
    for t in trans:
        ci = ("—" if t["phi_boot_ci"][0] is None
              else f"[{t['phi_boot_ci'][0]:.3f},{t['phi_boot_ci'][1]:.3f}]"
                   f" ({t['boot_cross_rate']:.0%})")
        if t["phi_meas"] is not None and t["phi_pred"] is not None:
            L.append(f"| {t['p']} | {t['T']} | {t['phi_meas']:.3f} | "
                     f"{ci} | {t['phi_pred']:.3f} | "
                     f"{abs(t['phi_meas'] - t['phi_pred']):.3f} |")
        else:
            L.append(f"| {t['p']} | {t['T']} | {t['phi_meas']} | "
                     f"{ci} | {t['phi_pred']} | — |")
    L.append("")
    L.append("## 每配置明细（v2 规则）")
    L.append("| kind | p | T | φ/q | meas Δm | CI | pred Δ̂m | 判号 | "
             "meas Δacc | pred Δ̂acc | 判号acc |")
    L.append("|------|---|---|-----|---------|----|---------|------|"
             "-----------|-------------|---------|")
    for r in rows:
        s = "AMB" if r["ambiguous_m"] else (
            "✓" if r.get("sign_ok_m_v2") else "✗")
        sa = "AMB" if r["ambiguous_a"] else (
            "✓" if r.get("sign_ok_a_v2") else "✗")
        L.append(
            f"| {r['kind']} | {r['p']} | {r['T']} | {r['phi']} | "
            f"{r['meas_m']:+.5f} | [{r['ci_m'][0]:+.5f},{r['ci_m'][1]:+.5f}]"
            f" | {r['pred_m_v2']:+.5f} | {s} | {r['meas_a']:+.5f} | "
            f"{r['pred_a_v2']:+.5f} | {sa} |")
    (outdir / f"signpred_{rule['dataset']}_report.md").write_text(
        "\n".join(L), encoding="utf-8")


# ----------------------------------------------------------------------

DATASET_NAME = None
DS = None
OP = {"mode": "heat", "alpha": 0.9}     # appnp: S=D^{-1/2}(A+I)D^{-1/2},
                                        # Z←αSZ+(1−α)Z0（论文 §dblp 约定）


def main():
    global DATASET_NAME, DS
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="cora", choices=list(DATA_DIRS))
    ap.add_argument("--stage", default="all",
                    choices=["cal", "predict", "eval", "all"])
    ap.add_argument("--n-real", type=int, default=100)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--operator", default="heat", choices=["heat", "appnp"])
    ap.add_argument("--alpha", type=float, default=0.9)
    args = ap.parse_args()
    OP["mode"] = args.operator
    OP["alpha"] = args.alpha
    DATASET_NAME = args.data
    DS = Dataset(DATASET_NAME)
    print(f"数据 {DATASET_NAME} [{OP['mode']}"
          f"{' α=' + str(OP['alpha']) if OP['mode'] == 'appnp' else ''}]: "
          f"n={DS.n} C={DS.C} m={DS.m} h={DS.h:.4f} "
          f"hom={int(DS.a_star.sum())}", flush=True)
    outdir = Path("results_smoke" if args.smoke else "results")
    outdir.mkdir(exist_ok=True)
    if not args.smoke:
        (outdir / "figures").mkdir(exist_ok=True)
    tag = DATASET_NAME + (f"_{OP['mode']}" if OP["mode"] != "heat" else "")
    base = BASE_SMOKE if args.smoke else (
        BASE_CAL if args.stage in ("cal", "all") else BASE_EVAL)
    rule_path = outdir / f"signpred_{tag}_frozen_rule.json"
    pred_path = outdir / f"signpred_{tag}_predictions.json"

    if args.stage in ("cal", "all"):
        print("=== 阶段 1：校准 ===", flush=True)
        cal, rule = stage_cal(DS, args.n_real, base)
        (outdir / f"signpred_{tag}_cal.json").write_text(
            json.dumps(cal, ensure_ascii=False, indent=1), encoding="utf-8")
        rule_path.write_text(
            json.dumps(rule, ensure_ascii=False, indent=1),
            encoding="utf-8")
        print(f"已写 {rule_path}")
    if args.stage in ("predict", "all"):
        rule = json.loads(rule_path.read_text(encoding="utf-8"))
        print("=== 阶段 2：预测落盘（先于任何评估运行） ===", flush=True)
        preds = stage_predict(rule, pred_path)
        print(f"已写 {pred_path}（{len(preds['configs'])} 配置）")
    if args.stage in ("eval", "all"):
        rule = json.loads(rule_path.read_text(encoding="utf-8"))
        preds = json.loads(pred_path.read_text(encoding="utf-8"))
        assert pred_path.stat().st_mtime >= rule_path.stat().st_mtime, \
            "预测文件必须不早于规则文件"
        ev_base = BASE_SMOKE if args.smoke else BASE_EVAL
        print(f"=== 阶段 3：评估（种子 {ev_base}+，只读预测） ===",
              flush=True)
        results, raw = stage_eval(DS, preds, args.n_real, ev_base)
        (outdir / f"signpred_{tag}_eval_raw.json").write_text(
            json.dumps(raw, ensure_ascii=False), encoding="utf-8")
        rows = []
        for c in preds["configs"]:
            key = str((c["kind"], c["p"], c["T"], c["mode"],
                       c.get("phi"), c.get("q")))
            if key not in results:
                continue
            r = results[key]
            row = dict(c)
            for metric, dk, ck in (("m", "d_margin", "d_margin_ci"),
                                   ("a", "d_acc", "d_acc_ci")):
                meas = r[dk]
                lo, hi = r[ck]
                row[f"meas_{metric}"] = meas
                row[f"ci_{metric}"] = [lo, hi]
                row[f"ambiguous_{metric}"] = bool(lo <= 0.0 <= hi)
                for variant in ("v2", "v1", "mech"):
                    pv = c[f"pred_{metric}_{variant}"]
                    row[f"err_{metric}_{variant}"] = float(pv - meas)
                    if not row[f"ambiguous_{metric}"]:
                        row[f"sign_ok_{metric}_{variant}"] = bool(
                            np.sign(pv) == np.sign(meas))
            row["G_meas"] = r["G_cor_margin"]
            row["D_meas"] = r["D_clean_margin"]
            row["beta_w_meas"] = r["mB_cor"]
            rows.append(row)
        summary, trans = summarize(rows, rule, raw)
        doc = dict(summary=summary, transitions=trans, rows=rows,
                   rule=rule)
        (outdir / f"signpred_{tag}_eval.json").write_text(
            json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
        if not args.smoke:
            make_figure(trans,
                        outdir / "figures" / f"signpred_{tag}_sweep.png")
        write_report(outdir, rows, summary, trans, rule, args.n_real,
                     results)
        print(f"已写 {outdir}/signpred_{tag}_eval.json, "
              f"signpred_{tag}_report.md")
        for key, v in summary.items():
            if v["sign_rate"] is not None:
                print(f"  {key}: {v['sign_correct']}/{v['n_unambiguous']}"
                      f" ({v['sign_rate']:.0%}), MAE={v['mae']:.5f}")
        for t in trans:
            boot = ("" if t["phi_boot"] is None
                    else f" boot[{t['phi_boot_ci'][0]:.3f},"
                         f"{t['phi_boot_ci'][1]:.3f}]"
                         f"({t['boot_cross_rate']:.0%})")
            print(f"  转折 p={t['p']} T={t['T']}: 实测 "
                  f"{None if t['phi_meas'] is None else round(t['phi_meas'], 3)}"
                  f"{boot} 预测 "
                  f"{None if t['phi_pred'] is None else round(t['phi_pred'], 3)}")


if __name__ == "__main__":
    main()
