"""
§27 节点影响的 Rayleigh 夹逼证书 + 加权重叠界（S 严格化第一步）。

理论（证明见笔记 §27；有限维线性代数，无谱隙、无渐近条件）：

  L1 夹逼证书（已证）：对称 M、扰动 δM，f=‖M‖₂；w、w' 为前后
     |zᵀMz| 的单位极大化向量；s = wᵀδMw，s' = w'ᵀδMw'。则

       Δ := |‖M+δM‖₂ − ‖M‖₂| ≤ max( |s|, |s'| )。

     证明：f' = |w'ᵀM'w'| ≤ |w'ᵀMw'| + |s'| ≤ f + |s'|，反向同理。∎

  L1' 亏损界（诊断，已证）：令 ρ = f − |w'ᵀMw'| ≥ 0，
     ρ' = f' − |wᵀM'w| ≥ 0，则 f'−f ≤ |s'|−ρ、f−f' ≤ |s|−ρ'。
     若两极值 Rayleigh 值及两个交叉 Rayleigh 值全同号，
     max(|s|−ρ', |s'|−ρ) = Δ 恒等；仅两极值同为底/顶还不够。
     它解释了部分 L1 松动（一阶值被亏损 ρ 吸收，
     即 level repulsion 阻尼），但因此不能单独充当先验界。

  L2 加权重叠界（非零范数分支已证）：翻转节点 i 只改变 (a) 含 i 的受影响边 aff(i)
     的系数（±1/3 秩一项）与 (b) i 的对齐位（P_A 第 i 对角元翻转）。
     对选定的非零特征值极值向量 z，记 c_e(z) = zᵀP_A1_e（旧对齐投影），

     |zᵀδM_i z| ≤ (1/3)X_i(z) + (2|z_i|/3)Ĝ_i(z) + (z_i²/3)h_i,
       X_i(z) = Σ_{e∈aff(i)} c_e(z)²（受影响边重叠质量，偏斜感知）,
     Ĝ_i(z) = Σ_{e∋i, Δa'_e≠0} |c_e(z)|,   h_i = |{e∋i : Δa'_e≠0}|。
     当 f 或 f' 数值上接近零时支撑论证不可靠，改用 f+f' 上界；
     真正零范数时该界等于 Δ，且总能界住 L1 Rayleigh 量。

  推论（实例级 S 证书，已证）：
     S(X) = Σq_iΔ_i² ≤ Σ q_i · max(RHS_i(w), RHS_i(w'))² =: S_ov(X)，
     中间经 L1 的 max(|s|,|s'|)。解析侧连接：x_e = c_e(w)² 的 L4 泛函
     Σx²、恒等式 ΣΔa_e·x_e = 3·(wᵀMw) 与 Markov 计数 #{x_e≥t} ≤ Σx/t。

经验部分（本脚本测量，不构成证明）：
  E1 L1 紧度（旋转损失）与 L1' 恒等性诊断。
  E2 L2 重叠界紧度与 S 证书（对照 §26 平均近似在 σ=1 偏低 25×且非界）。
  E3 偏斜结构：x_e 分布、Σx²、Markov 计数、恒等式。
  E4 多基准：S_ov 的跨基准浓度。
"""

import sys

import numpy as np
from scipy.special import ndtr

sys.stdout.reconfigure(encoding="utf-8")
from propagation_chain_checked import sample_edges


def build_operator(n, edges, delta_a, aligned):
    """M = P_A ΔB P_A = Σ_e (Δa_e/3)(P_A1_e)(P_A1_e)ᵀ，稠密 n×n。"""
    M = np.zeros((n, n))
    for e_i in np.flatnonzero(delta_a):
        e = [j for j in edges[e_i] if aligned[j]]
        if not e:
            continue
        c = delta_a[e_i] / 3.0
        for a in e:
            for b in e:
                M[a, b] += c
    return M


def extremizer(M):
    """(f, w)：f=‖M‖₂，w = |zᵀMz| 的单位极大化向量。"""
    lam, V = np.linalg.eigh(M)
    j = int(np.argmax(np.abs(lam)))
    return abs(float(lam[j])), V[:, j]


def instance(seed, n, m, ell, sigma, rho_hom, d=8):
    rng = np.random.default_rng(seed)
    labels = np.tile([1, -1], n // 2)
    rng.shuffle(labels)
    edges = sample_edges(rng, labels, m, rho_hom)
    el = labels[edges]
    a_star = np.all(el == el[:, :1], axis=1).astype(float)

    seed_rng = np.random.default_rng(seed + 10000)
    sp = np.zeros((ell, d)); sp[:, 0] = 1.0
    sn = np.zeros((ell, d)); sn[:, 0] = -1.0
    sp += sigma * seed_rng.standard_normal((ell, d))
    sn += sigma * seed_rng.standard_normal((ell, d))
    wproto = sp.mean(axis=0) - sn.mean(axis=0)
    p = float(ndtr(-wproto[0] / (sigma * np.linalg.norm(wproto))))

    node_edges = [[] for _ in range(n)]
    for e_i, e in enumerate(edges):
        for v in e:
            node_edges[v].append(e_i)
    return labels, edges, a_star, wproto, p, node_edges


def base_state(X, wproto, labels, edges):
    pred = np.where(X @ wproto >= 0, 1, -1)
    aligned = pred == labels
    ep = pred[edges]
    a_hat = np.all(ep == ep[:, :1], axis=1).astype(float)
    return pred, aligned, a_hat


def node_analysis(n, edges, labels, a_star, pred, aligned, a_hat, node_edges,
                  w, M, f):
    """逐节点：exact Δ_i、L1 夹逼、L2 精化重叠界。

    精化要点（推导见笔记 §27.3）：w 支撑于 A（M 的行结构），w' 支撑于 A'。
    - i∉A：w_i=0，s = (1/3)Σσc_e² 纯系数项，|s| ≤ X_i(w)/3（无交叉项）。
    - i∈A：本征方程 (Mw)_i=λw_i 使交叉项为二次型：
      |s| ≤ X/3 + 2|λ|w_i² + (2|w_i|/3)Σ|c| + w_i²(d_chg+k)/3。
    - w' 侧对称（基换为 M'，扰动 −δM）。
    """
    deltas = np.zeros(n)
    cert_raw = np.zeros(n)    # L1: max(|s|, |s'|)
    cert_ov = np.zeros(n)     # L2 精化: max(bnd_s, bnd_sp)
    cert_ov0 = np.zeros(n)    # L2 朴素版（线性交叉记账，对照）
    Xw = np.zeros(n)
    k_aff = np.zeros(n, dtype=int)
    rot = np.zeros(n)
    delta_a = a_hat - a_star
    lam = float(w @ M @ w)  # = ±f

    for i in range(n):
        pred_new = -pred[i]
        a_new_full = a_hat.copy()
        for e_i in node_edges[i]:
            e = edges[e_i]
            pn = [pred_new if v == i else pred[v] for v in e]
            a_new_full[e_i] = 1.0 if (pn[0] == pn[1] == pn[2]) else 0.0
        aff = [e_i for e_i in node_edges[i] if a_new_full[e_i] != a_hat[e_i]]
        k_aff[i] = len(aff)
        delta_a_new = a_new_full - a_star
        chg_old = [e_i for e_i in node_edges[i] if delta_a[e_i] != 0.0]
        chg_new = [e_i for e_i in node_edges[i] if delta_a_new[e_i] != 0.0]
        if not aff and not chg_old and not chg_new:
            continue  # δM = 0

        aligned_new = aligned.copy()
        aligned_new[i] = pred_new == labels[i]
        M2 = build_operator(n, edges, delta_a_new, aligned_new)
        f2, wp = extremizer(M2)
        dM = M2 - M
        deltas[i] = abs(f2 - f)
        s = float(w @ dM @ w)
        sp = float(wp @ dM @ wp)
        cert_raw[i] = max(abs(s), abs(sp))
        rot[i] = abs(float(w @ wp))
        lam2 = float(wp @ M2 @ wp)
        if f <= 1e-12 or f2 <= 1e-12:
            # A zero matrix has arbitrary extremizers, so support-based R2 is
            # invalid. f+f2 also bounds cert_raw for near-zero matrices.
            cert_ov[i] = f + f2
            cert_ov0[i] = cert_ov[i]
            continue

        # c_e(w)（旧对齐）、c'_e(w')（新对齐）
        c_aff = [sum(w[j] for j in edges[e_i] if aligned[j]) for e_i in aff]
        X = float(np.sum(np.square(c_aff)))
        cp_aff = [sum(wp[j] for j in edges[e_i] if aligned_new[j]) for e_i in aff]
        Xp = float(np.sum(np.square(cp_aff)))

        # w 侧精化界
        if aligned[i]:
            G = float(np.sum(np.abs(c_aff)))
            bnd_s = (X / 3.0 + 2.0 * abs(lam) * w[i] ** 2
                     + 2.0 * abs(w[i]) * G / 3.0
                     + w[i] ** 2 * (len(chg_old) + k_aff[i]) / 3.0)
        else:
            bnd_s = X / 3.0  # w_i = 0：纯系数项
        # w' 侧精化界（对称：基 M'，扰动 −δM）
        if aligned_new[i]:
            Gp = float(np.sum(np.abs(cp_aff)))
            bnd_sp = (Xp / 3.0 + 2.0 * abs(lam2) * wp[i] ** 2
                      + 2.0 * abs(wp[i]) * Gp / 3.0
                      + wp[i] ** 2 * (len(chg_new) + k_aff[i]) / 3.0)
        else:
            bnd_sp = Xp / 3.0
        cert_ov[i] = max(bnd_s, bnd_sp)
        Xw[i] = X

        # 朴素 L2（旧版线性记账，对照）
        c_cn = [sum(w[j] for j in edges[e_i] if aligned[j]) for e_i in chg_new]
        cp_co = [sum(wp[j] for j in edges[e_i] if aligned_new[j]) for e_i in chg_new]
        v0 = (X / 3.0 + 2.0 * abs(w[i]) * (float(np.sum(np.abs(c_aff))) + float(np.sum(np.abs(c_cn)))) / 3.0
              + w[i] ** 2 * len(chg_new) / 3.0)
        v0p = (Xp / 3.0 + 2.0 * abs(wp[i]) * float(np.sum(np.abs(cp_aff + cp_co))) / 3.0
               + wp[i] ** 2 * len(chg_new) / 3.0)
        cert_ov0[i] = max(v0, v0p)

    return {"delta": deltas, "cert_raw": cert_raw, "cert_ov": cert_ov,
            "cert_ov0": cert_ov0, "Xw": Xw, "k_aff": k_aff, "rot": rot}


def projections(n, edges, delta_a, aligned, w):
    """changed 边的 x_e = (wᵀP_A1_e)² 与符号。"""
    x, da = [], []
    for e_i in np.flatnonzero(delta_a):
        c = sum(w[j] for j in edges[e_i] if aligned[j])
        x.append(c * c)
        da.append(delta_a[e_i])
    return np.array(x), np.array(da)


def run(seed=700, n=400, m=600, ell=8, sigma=1.0, rho_hom=0.7, base_count=1,
        verbose=True):
    labels, edges, a_star, wproto, p, node_edges = instance(
        seed, n, m, ell, sigma, rho_hom)
    bases = []
    feat_rng = np.random.default_rng(seed + 20000)
    for b in range(base_count):
        X = sigma * feat_rng.standard_normal((n, 8))
        X[:, 0] += labels
        pred, aligned, a_hat = base_state(X, wproto, labels, edges)
        M = build_operator(n, edges, a_hat - a_star, aligned)
        f, w = extremizer(M)
        res = node_analysis(n, edges, labels, a_star, pred, aligned, a_hat,
                            node_edges, w, M, f)
        q = np.where(aligned, p, 1 - p)
        S = float(np.sum(q * res["delta"] ** 2))
        S_L1 = float(np.sum(q * res["cert_raw"] ** 2))
        S_ov = float(np.sum(q * res["cert_ov"] ** 2))
        S_ov0 = float(np.sum(q * res["cert_ov0"] ** 2))
        x, da = projections(n, edges, a_hat - a_star, aligned, w)
        lam_w = float(w @ M @ w)
        bases.append({
            "f": f, "S": S, "S_L1": S_L1, "S_ov": S_ov, "S_ov0": S_ov0, **res,
            "x": x, "da": da, "p": p, "lam_w": lam_w,
            "aligned_frac": float(aligned.mean()),
        })
        if verbose:
            print(f"    base {b}: f={f:.3f}  S={S:.4f}  S_L1={S_L1:.4f} "
                  f"({S_L1/max(S,1e-12):.2f})  S_ov={S_ov:.4f} "
                  f"({S_ov/max(S,1e-12):.2f}；朴素 {S_ov0/max(S,1e-12):.2f})",
                  flush=True)
    return bases


def summarize(bs):
    b = bs[0]
    d = b["delta"]
    nz = d > 1e-12
    x = b["x"]; da = b["da"]

    print(f"  [E1] L1 证书紧度（已证；紧度为经验）:")
    print(f"    非零 Δ 节点 {int(nz.sum())}/{len(d)}")
    r = b["cert_raw"][nz] / d[nz]
    v = int(np.sum(d > b["cert_raw"] + 1e-10))
    print(f"    L1 max(|s|,|s'|)/Δ: 中位 {np.median(r):.2f}  q90 {np.quantile(r,0.9):.2f}  "
          f"max {r.max():.2f}  违反 {v}")
    print(f"    旋转 |w·w'|<0.5 的节点: {int(np.sum((b['rot']>1e-12)&(b['rot']<0.5)))}，"
          f"其 Δ 合计占比 {float(d[(b['rot']<0.5)&(d>0)].sum()/max(d.sum(),1e-12)):.1%}")

    print(f"  [E2] L2 重叠界与 S 证书:")
    r = b["cert_ov"][nz] / b["cert_raw"][nz]
    v = int(np.sum(b["cert_raw"] > b["cert_ov"] + 1e-10))
    print(f"    精化重叠/L1: 中位 {np.median(r):.2f}  max {r.max():.2f}  违反 {v}")
    print(f"    S={b['S']:.4f}  S_L1={b['S_L1']:.4f} ({b['S_L1']/max(b['S'],1e-12):.2f})  "
          f"S_ov={b['S_ov']:.4f} ({b['S_ov']/max(b['S'],1e-12):.2f}；"
          f"朴素 {b['S_ov0']/max(b['S'],1e-12):.2f})")
    print(f"    （§26 对照：平均近似偏低 25×@σ1 / 1.7×@σ0.5，且非界）")

    print(f"  [E3] 偏斜 + 恒等式 + Markov:")
    print(f"    wᵀMw = {b['lam_w']:.4f}（范数分支: {'底' if b['lam_w']<0 else '顶'}）")
    print(f"    C={len(x)}  恒等式 ΣΔa·x/3 = {float(np.sum(da*x))/3:.4f}")
    print(f"    x: mean={x.mean():.4f} max={x.max():.4f} max/mean={x.max()/max(x.mean(),1e-12):.0f}×  "
          f"零占比 {float((x<1e-12).mean()):.0%}")
    print(f"    Σx²={float((x**2).sum()):.4f}  Σx={x.sum():.4f}")
    for t in (0.05, 0.1, 0.25):
        cnt = int((x >= t).sum())
        mk = float(x.sum()) / t
        print(f"    t={t}: #{{x≥t}}={cnt}  Markov Σx/t={mk:.1f}"
              f"{'  ✓' if cnt <= mk + 1e-9 else '  ✗ 违反!'}")


if __name__ == "__main__":
    print("=" * 70)
    print("§27 Rayleigh 夹逼证书 + 精化加权重叠界：S 严格化第一步")
    print("=" * 70)
    for sigma in (1.0, 0.75, 0.5):
        print(f"\n{'='*26} σ = {sigma} {'='*26}")
        summarize(run(seed=700, sigma=sigma))

    print(f"\n{'='*26} σ = 1.0 多基准（证书浓度）{'='*10}")
    bases = run(seed=700, sigma=1.0, base_count=4)
    Ss = np.array([b["S"] for b in bases])
    SL = np.array([b["S_L1"] for b in bases])
    So = np.array([b["S_ov"] for b in bases])
    X2 = np.array([float((b["x"] ** 2).sum()) for b in bases])
    print(f"  S:    mean={Ss.mean():.4f} cv={Ss.std()/max(Ss.mean(),1e-12):.2f}")
    print(f"  S_L1: mean={SL.mean():.4f} cv={SL.std()/max(SL.mean(),1e-12):.2f}")
    print(f"  S_ov: mean={So.mean():.4f} cv={So.std()/max(So.mean(),1e-12):.2f}")
    print(f"  Σx²:  mean={X2.mean():.4f} cv={X2.std()/max(X2.mean(),1e-12):.2f}")
    print(f"  全部基准 S ≤ S_L1 ≤ S_ov: "
          f"{bool(np.all(Ss <= SL + 1e-12)) and bool(np.all(SL <= So + 1e-12))}")
