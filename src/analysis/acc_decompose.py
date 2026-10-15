"""§58 准确率口径修正（用户论文审计 1–2 项的执行）。

问题：
  (a) 定理 4 的卷积分位数控制的是种子 × 测试特征**联合**随机性下的经验错分率
      （K6 脚本 L90–95 重采样了 W 与 X），稿中"仅对种子抽样"不准确。
  (b) K7 的 acc = 1 − flips.mean() 是扰动读出与理想读出的一致率，非对真标签
      的准确率。真实准确率需加理想错误项：
          acc_true(est) = 1 − [ε_ideal + flips_among_ideal_correct]
                        = 1 − ε_ideal − flips_among_correct
      其中 ε_ideal = 理想算子传播读出的错分率（经验上 ≈ align，v2 簇继承，
      但这是实证转移，非定理）。

本脚本在 K7 同型实例上测量三量分解并给出修正后的准确率下界组装。
"""

import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
MU, D = 1.0, 8


def run_seed(seed_i, sigma, ell, n=400, m=600, T=25):
    r = np.random.default_rng(5000 + seed_i)
    y = np.array([1] * (n // 2) + [-1] * (n - n // 2))
    y = y[r.permutation(n)]
    E = np.empty((m, 3), dtype=int)
    for t in range(m):
        if r.random() < 0.7:
            c = 1 if r.random() < 0.5 else -1
            E[t] = r.choice(np.where(y == c)[0], 3, replace=False)
        else:
            maj = 1 if r.random() < 0.5 else -1
            E[t, :2] = r.choice(np.where(y == maj)[0], 2, replace=False)
            E[t, 2] = int(r.choice(np.where(y == -maj)[0], 1)[0])
    us = np.concatenate([E[:, 0], E[:, 0], E[:, 1]])
    vs_ = np.concatenate([E[:, 1], E[:, 2], E[:, 2]])
    s_true = np.where(y[us] == y[vs_], 1.0, -1.0)

    def nrm(Z):
        q = np.linalg.norm(Z, axis=1, keepdims=True); q[q == 0] = 1
        return Z / q

    def prop(Z0, s):
        src = np.concatenate([us, vs_]); dst = np.concatenate([vs_, us])
        sv = np.concatenate([s, s])
        deg = np.bincount(src, minlength=n).astype(float); deg[deg == 0] = 1
        Z = nrm(Z0.copy())
        for _ in range(T):
            msg = sv[:, None] * (Z[dst] - Z[src]) / 3.0
            agg = np.zeros_like(Z); np.add.at(agg, src, msg)
            Z = nrm(Z + 0.5 * agg / deg[:, None])
        return Z

    X = np.zeros((n, D))
    X[:, 0] = y * MU + sigma * r.standard_normal(n)
    X[:, 1:] = sigma * r.standard_normal((n, D - 1))
    pos, neg = np.where(y == 1)[0][:ell], np.where(y == -1)[0][:ell]
    p1 = sigma / np.sqrt(ell) * r.standard_normal(D); p1[0] += MU
    p2 = sigma / np.sqrt(ell) * r.standard_normal(D); p2[0] -= MU
    wproto = p1 - p2
    yhat = np.where(X @ wproto >= 0, 1, -1)
    s_est = np.where(yhat[us] == yhat[vs_], 1.0, -1.0)

    Z_id = prop(nrm(X), s_true)
    Z_es = prop(nrm(X), s_est)
    w = Z_id[y == 1].mean(0) - Z_id[y == -1].mean(0)
    w /= np.linalg.norm(w)
    tst = np.setdiff1d(np.arange(n), np.concatenate([pos, neg]))

    gamma = y[tst] * (Z_id[tst] @ w)     # 理想边距（>0 = 理想正确）
    ghat = y[tst] * (Z_es[tst] @ w)      # 估计边距
    e_row = np.linalg.norm(Z_es[tst] - Z_id[tst], axis=1)
    aligned = (yhat[tst] == y[tst])

    ideal_wrong = gamma < 0                       # 理想读出错分
    flips_correct = (gamma > 0) & (ghat < 0)      # 理想对但翻错（新增错分）
    recovered = (gamma < 0) & (ghat > 0)          # 理想错但恢复对
    est_wrong = ghat < 0                          # 估计读出总错分
    at_risk = aligned & (np.abs(gamma) <= e_row)

    return dict(
        n_tst=len(tst),
        align=float((yhat != y).mean()),
        eps_node=float((yhat != y)[tst].mean()),
        eps_ideal=float(ideal_wrong.mean()),        # ε_ideal
        flip_among_correct=float(flips_correct.mean()),
        recovered=float(recovered.mean()),
        acc_true=float(1 - est_wrong.mean()),        # 对真标签的准确率
        acc_formula=float(1 - ideal_wrong.mean() - flips_correct.mean() + recovered.mean()),
        acc_k7_style=float(1 - (np.sign(gamma) != np.sign(ghat)).mean()),
        bound=float(1 - ideal_wrong.mean() - flips_correct.mean()),
        mis=int((~aligned).sum()), risk=int(at_risk.sum()),
        eps_node_vs_ideal=float(np.abs((yhat != y)[tst].mean()
                                       - ideal_wrong.mean())),
    )


if __name__ == "__main__":
    print("=" * 74)
    print("§58 准确率三量分解：ε_ideal + 理想正确群新翻转 vs 估计读出真实准确率")
    print("=" * 74)
    for sigma, ell in ((1.0, 32), (1.0, 8), (0.5, 8)):
        rows = [run_seed(sd, sigma, ell) for sd in range(5)]
        g = lambda k: float(np.mean([r[k] for r in rows]))
        print(f"\nσ={sigma}, ℓ={ell}（5 种子，测试集 |tst|={rows[0]['n_tst']}）")
        print(f"  ε_node(测试集) = {g('eps_node'):.4f}   "
              f"ε_ideal(理想传播读出错分) = {g('eps_ideal'):.4f}   "
              f"|差| = {g('eps_node_vs_ideal'):.4f}")
        print(f"  理想正确群新翻转 = {g('flip_among_correct'):.4f}   "
              f"真实准确率 = {g('acc_true'):.4f}")
        print(f"  恢复正确（理想错→扰动对）= {g('recovered'):.4f}")
        print(f"  分解式 1−ε_ideal−新翻转+恢复 = {g('acc_formula'):.4f} "
              f"（与真实准确率逐点相等："
              f"{all(abs(r['acc_formula'] - r['acc_true']) < 1e-12 for r in rows)}）")
        print(f"  [对照] K7 旧式 acc=1−flips = {g('acc_k7_style'):.4f}"
              f"（高估 {g('acc_k7_style') - g('acc_true'):.4f}"
              f" = ε_ideal − ε_ideal群内翻回 ≈ ε_ideal）")
        print(f"  装配下界(测试集分母 |tst|): 1 − ε_ideal − (|M|+风险集)/|tst| = "
              f"{1 - g('eps_ideal') - (g('mis') + g('risk')) / rows[0]['n_tst']:.4f}"
              f"   （κ 转移项 = ε_ideal − ε_node = "
              f"{g('eps_ideal') - g('eps_node'):+.4f}，经验）")
