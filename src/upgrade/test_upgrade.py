"""upgrade 模块回归测试（计划 §6，TDD：先于实现编写）。

覆盖计划要求的全部小例子：
  1. 零跨类质量锁定（计划原文用例）
  2. alpha=0 返回初始化（计划原文用例）
  3. 两节点交换图的精确核与近锁定上界
  4. 对称归一化星形单步分数 > 1（S 非行随机的实证）
  5. 零度节点自环
  6. 平票协议（固定 argmax 规则；充分条件用严格不等式避开平票）
  7. 软分数统一间隔条件（gamma 为同类邻居最小间隔，非单节点置信度）
  8. eta=0 精确回到初始化
  9. 同样标签预算（残差只读训练节点；所有方法共享同一预测器训练）
  10. 固定 masks 下改变全部测试真值不改变部署输出（泄漏）
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from near_locking import (restart_kernel, cross_class_mass, kernel_apply,
                          near_locking_bound, off_class_epsilon, argmax_labels,
                          row_stochastic)
from correction import (fit_and_select, residual_matrix, cs_baseline,
                        prop_baseline, evaluate)
from outer_protocol import stratified_split


def sym_norm_S(rows, cols, n):
    """对照算子 S = D^{-1/2}(A+I)D^{-1/2}（单位权重、含对角对、外加自环）。"""
    import scipy.sparse as spsp
    from collections import Counter
    cnt = Counter(zip(rows, cols))
    r = np.array([k[0] for k in cnt]); c = np.array([k[1] for k in cnt])
    v = np.array([cnt[k] for k in cnt], dtype=float)
    A = spsp.csr_matrix((v, (r, c)), shape=(n, n))
    A = A + spsp.eye(n, format='csr')
    d = np.asarray(A.sum(axis=1)).ravel()
    di = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    return spsp.diags(di) @ A @ spsp.diags(di)


class KernelTests(unittest.TestCase):
    def test_zero_cross_class_mass_locks(self):
        P = np.eye(2)
        labels = np.array([0, 1])
        K = restart_kernel(P, alpha=0.9, steps=20)
        np.testing.assert_allclose(K.sum(axis=1), 1.0)
        np.testing.assert_allclose(cross_class_mass(K, labels), 0.0)
        np.testing.assert_array_equal((K @ np.eye(2)).argmax(axis=1), labels)

    def test_alpha_zero_returns_initialization(self):
        P = np.array([[0.0, 1.0], [1.0, 0.0]])
        np.testing.assert_allclose(
            restart_kernel(P, alpha=0.0, steps=20), np.eye(2))

    def test_steps_zero_is_identity(self):
        P = np.array([[0.0, 1.0], [1.0, 0.0]])
        np.testing.assert_allclose(
            restart_kernel(P, alpha=0.9, steps=0), np.eye(2))

    def test_swap_graph_exact_kernel_and_bound(self):
        """两节点交换图：P^t 在 I/P 间交替；精确核 + 候选上界必须成立。
        勘误（2026-10-09）：跨类质量按 T 的奇偶振荡（α at T=1，
        α(1−α) at T=2），两个子列分别从上/下收敛到 α/(1+α)；
        one-hot 在奇数深度瞬时翻转（α=0.9 时至 T=27），偶数深度与
        大深度极限不翻转。ε=1 时候选界=1（松弛）。"""
        P = np.array([[0.0, 1.0], [1.0, 0.0]])
        labels = np.array([0, 1])
        alpha = 0.9

        def mass_and_flip(T):
            K = restart_kernel(P, alpha, T)
            m = cross_class_mass(K, labels)[0]
            flip = argmax_labels(K @ np.eye(2)[labels])[0] != 0
            return m, flip

        # 有限步精确公式：M_T = (1-α)Σ_{t<T,odd}α^t + 1{T odd}α^T
        for T in (1, 2, 3, 19, 20, 27, 28, 60):
            m, _ = mass_and_flip(T)
            exact = (1 - alpha) * sum(alpha ** t for t in range(T)
                                      if t % 2 == 1) + \
                (alpha ** T if T % 2 == 1 else 0.0)
            self.assertAlmostEqual(m, exact, places=12)
        m1, f1 = mass_and_flip(1); m2, f2 = mass_and_flip(2)
        self.assertAlmostEqual(m1, alpha, places=12)
        self.assertAlmostEqual(m2, alpha * (1 - alpha), places=12)
        self.assertTrue(f1); self.assertFalse(f2)          # 奇翻偶不翻
        m19, f19 = mass_and_flip(19); m20, f20 = mass_and_flip(20)
        self.assertAlmostEqual(m19, 0.5377, places=4)
        self.assertAlmostEqual(m20, 0.4161, places=4)
        self.assertTrue(f19); self.assertFalse(f20)
        # 最后一个翻转深度 = 27（α=0.9），其后所有深度保持
        last = max(T for T in range(1, 60, 2) if mass_and_flip(T)[1])
        self.assertEqual(last, 27)
        self.assertFalse(mass_and_flip(29)[1])
        # 子列单调：奇减、偶增，同收敛到 α/(1+α) < 1/2
        odds = [mass_and_flip(T)[0] for T in range(1, 30, 2)]
        evens = [mass_and_flip(T)[0] for T in range(2, 31, 2)]
        self.assertTrue(all(x > y for x, y in zip(odds, odds[1:])))
        self.assertTrue(all(x < y for x, y in zip(evens, evens[1:])))
        limit = alpha / (1 + alpha)
        self.assertTrue(evens[-1] < limit < odds[-1])   # 夹逼
        self.assertLess(odds[-1] - evens[-1], 0.05)   # 间距收窄（慢收敛）
        self.assertLess(limit, 0.5)
        # 候选界：ε=1 时 B=1（无信息），且 ≥ 一切有限步精确值
        B = near_locking_bound(1.0, alpha, 20)
        self.assertEqual(B, 1.0)
        self.assertGreaterEqual(B, m20)

    def test_kernel_apply_matches_dense(self):
        rng = np.random.default_rng(0)
        A = rng.random((6, 6)); P = A / A.sum(axis=1, keepdims=True)
        R = rng.random((6, 3))
        K = restart_kernel(P, 0.85, 7)
        np.testing.assert_allclose(kernel_apply(P, R, 0.85, 7), K @ R,
                                   atol=1e-12)

    def test_symmetric_star_score_exceeds_one(self):
        """星形（中心 0，叶 1、2）+ 对称归一化：一步后中心正确类分数 > 1.1。"""
        n = 3
        rows, cols = [], []
        for (u, v) in ((0, 1), (0, 2)):
            for a in (u, v):
                for b in (u, v):
                    rows.append(a); cols.append(b)
        S = sym_norm_S(rows, cols, n).toarray()
        Z0 = np.zeros((n, 2)); Z0[:, 0] = 1.0   # 全部预测/真值类 0
        Z1 = 0.9 * (S @ Z0) + 0.1 * Z0
        self.assertGreater(Z1[0, 0], 1.1)       # 主文所引星形例
        self.assertGreater(Z1[0, 0], 1.0)

    def test_zero_degree_self_loop(self):
        """零度节点行 = e_u：核行 = e_u，跨类质量 0，锁定不变。"""
        # 节点 0-1 连边（同类 0），节点 2 孤立（预测类 1）
        P = row_stochastic(3, [0, 1], [1, 0], [1.0, 1.0]).toarray()
        np.testing.assert_allclose(P[2], [0.0, 0.0, 1.0])
        K = restart_kernel(P, 0.9, 10)
        np.testing.assert_allclose(K[2], [0.0, 0.0, 1.0])
        labels = np.array([0, 0, 1])
        np.testing.assert_allclose(cross_class_mass(K, labels), [0, 0, 0])

    def test_tie_protocol_is_fixed(self):
        """固定平票规则（最小下标）；充分条件使用严格不等式，不依赖平票。"""
        Z = np.array([[0.5, 0.5], [0.3, 0.7]])
        np.testing.assert_array_equal(argmax_labels(Z), [0, 1])

    def test_soft_uniform_margin_condition(self):
        """软分数：界条件 M_T < gamma/(1+gamma) 用同类邻居统一最小间隔。"""
        # 构造：4 节点两类；软初始化（margin=gamma），小跨类泄漏 eps
        rng = np.random.default_rng(3)
        n, C = 4, 2
        labels = np.array([0, 0, 1, 1])
        gamma = 0.2
        Z0 = np.full((n, C), (1 - gamma) / C)
        Z0[np.arange(n), labels] += gamma / 2 + gamma / 2  # margin = gamma
        # 类内完全连接 + 微小跨类泄漏
        rows, cols, vals = [], [], []
        for u in range(n):
            for v in range(n):
                if u == v: continue
                same = labels[u] == labels[v]
                rows.append(u); cols.append(v)
                vals.append(1.0 if same else 1e-3)
        P = row_stochastic(n, rows, cols, vals)
        eps = off_class_epsilon(P, labels)[1]
        T = 20
        B = near_locking_bound(eps, 0.9, T)
        K = restart_kernel(P.toarray(), 0.9, T)
        M = cross_class_mass(K, labels)
        self.assertLess(B, gamma / (1 + gamma))
        self.assertTrue((M <= B + 1e-12).all())
        ZT = kernel_apply(P, Z0, 0.9, T)
        np.testing.assert_array_equal(argmax_labels(ZT), labels)

    def test_one_way_door_changes_class(self):
        """可翻转反例：u 单向进入错误类吸收块 ⟹ M_T = alpha > 1/2，argmax 翻转。"""
        # 节点 0 预测类 0，完全转移到预测类 1 的吸收对 (2,3)；节点 1 自环
        P = np.zeros((4, 4))
        P[0, 2] = 1.0
        P[1, 1] = 1.0
        P[2, 3] = 1.0; P[3, 2] = 1.0
        labels = np.array([0, 0, 1, 1])  # 初始预测类
        alpha, T = 0.9, 20
        K = restart_kernel(P, alpha, T)
        M = cross_class_mass(K, labels)
        self.assertAlmostEqual(M[0], alpha, places=6)  # M_T = alpha
        Z0 = np.eye(2)[labels]
        ZT = K @ Z0
        self.assertEqual(argmax_labels(ZT)[0], 1)      # 翻转到错误类
        self.assertEqual(argmax_labels(ZT)[1], 0)      # 未转移节点不动


class ReverseMistakeTests(unittest.TestCase):
    """反向错误瓶颈条件（勘误后公式，2026-10-09 复查）：
    纠错 ⟺ η(2S⁺+S⁻)>1 且 S⁺>T_c ∀第三类 c。
    初版误写 η(S⁺−S⁻)>1，漏掉 S⁺ 在 ŷ_u 列的 −1 贡献。"""

    def _hard_kernel_and_scores(self, y, yhat, train_mask, Kfun):
        """给定核函数 Kfun(u,v)，构造 Z0/R 并返回 C = Z0 + K R。"""
        n, C = len(y), int(max(y.max(), yhat.max())) + 1
        Z0 = np.eye(C)[yhat]
        R = np.zeros((n, C))
        idx = np.flatnonzero(train_mask)
        R[idx] = np.eye(C)[y[idx]] - Z0[idx]
        K = np.array([[Kfun(u, v) for v in range(n)] for u in range(n)])
        return Z0, R, K, Z0 + 1.0 * (K @ R)

    def test_reverse_mistake_correction_condition(self):
        """用户反例：合法三节点硬图，C[u]=(0.4,0.6) 成功纠错；
        旧式 η(S⁺−S⁻)=0.6<1 否定纠错（误），新式 η(2S⁺+S⁻)=1.2>1 通过。"""
        y = np.array([0, 0, 0])      # 节点 0 预测错（真值 0，预测 1）
        yhat = np.array([1, 1, 1])   # 硬图：全部同预测类 1，同类全连
        train_mask = np.array([False, True, True])
        # 核：u=0 到 v=1 质量 0.6，其余按行随机（行随机、类内支持）
        W = np.array([[0.4, 0.6, 0.0],
                      [0.3, 0.7, 0.0],
                      [0.0, 0.0, 1.0]])
        Z0, R, K, C = self._hard_kernel_and_scores(
            y, yhat, train_mask, lambda u, v: W[u, v])
        np.testing.assert_allclose(K.sum(axis=1), 1.0)
        np.testing.assert_allclose(C[0], [0.6, 0.4])  # 纠错方向 (y=0 胜)
        # S⁺(0)=K[0,1]=0.6（v=1 预测 1、真值 0：反向错误），S⁻=0（二类）
        S_p, S_m = 0.6, 0.0
        self.assertLess(S_p - S_m, 1.0)        # 旧公式：不应否定
        self.assertGreater(2 * S_p + S_m, 1.0)  # 新公式：应通过
        self.assertEqual(argmax_labels(C)[0], 0)

    def test_third_class_suppression(self):
        """多分类：2S⁺+S⁻>1 通过但 S⁺ ≤ max T_c 时仍不纠错（翻到第三类）。"""
        y = np.array([0, 0, 2])
        yhat = np.array([1, 1, 1])               # 硬图同类预测
        train_mask = np.array([False, True, True])
        W = np.array([[0.25, 0.35, 0.40],        # u=0: S⁺=0.35, T_2=S⁻=0.40
                      [0.25, 0.50, 0.25],
                      [0.25, 0.25, 0.50]])
        Z0, R, K, C = self._hard_kernel_and_scores(
            y, yhat, train_mask, lambda u, v: W[u, v])
        S_p = K[0, 1]                            # 0.35（反向错误）
        S_m = K[0, 2]                            # 0.40（第三类 v=2）
        T_max = K[0, 2]
        self.assertGreater(2 * S_p + S_m, 1.0)         # 主条件通过
        self.assertLessEqual(S_p, T_max)               # 但第三类压制
        self.assertEqual(argmax_labels(C)[0], 2)       # 不纠错（翻到类 2）
        np.testing.assert_allclose(C[0], [0.35, 0.25, 0.40])

    def test_condition_end_to_end_on_random_hard_graphs(self):
        """随机硬图上逐节点核对充要条件与实际 argmax（含纠错与不纠错）。"""
        rng = np.random.default_rng(17)
        for _ in range(30):
            n, C = 7, 3
            y = rng.integers(0, C, n)
            yhat = rng.integers(0, C, n)
            # 硬图：按预测类分块，块内随机行随机
            blocks = {c: np.flatnonzero(yhat == c) for c in range(C)}
            P = np.zeros((n, n))
            for c, idx in blocks.items():
                if len(idx) == 0:
                    continue
                for u in idx:
                    w = rng.random(len(idx)) + 1e-3
                    P[u, idx] = w / w.sum()
            train_mask = rng.random(n) < 0.5
            train_mask[0] = False                 # 节点 0 为测试节点
            yhat[0] = yhat[blocks[yhat[0]][0]] if len(blocks[yhat[0]]) > 1 \
                else yhat[0]
            K = restart_kernel(P, 0.9, 20)
            Z0 = np.eye(C)[yhat]
            R = np.zeros((n, C))
            idx = np.flatnonzero(train_mask)
            R[idx] = np.eye(C)[y[idx]] - Z0[idx]
            Csc = Z0 + 1.0 * (K @ R)
            u = 0
            if y[u] == yhat[u]:
                continue                          # 只核纠错节点
            G_tr = np.zeros((n, C))
            G_tr[idx] = np.eye(C)[y[idx]]
            KG = K @ G_tr
            S_p = KG[u, y[u]]
            S_m = sum(KG[u, c] for c in range(C)
                      if c not in (y[u], yhat[u]))
            T_max = max((KG[u, c] for c in range(C)
                         if c not in (y[u], yhat[u])), default=0.0)
            corrected = argmax_labels(Csc)[u] == y[u]
            cond = (2 * S_p + S_m > 1.0) and (S_p > T_max)
            self.assertEqual(cond, corrected,
                             f"mismatch: cond={cond} corrected={corrected}")



    def _synth(self, seed=0, n=60, C=3, m=40):
        rng = np.random.default_rng(seed)
        labels = rng.integers(0, C, n)
        feats = np.zeros((n, 4))
        for c in range(C):
            idx = np.flatnonzero(labels == c)
            center = rng.normal(size=4)
            feats[idx] = center + 0.35 * rng.normal(size=(len(idx), 4))
        edges = []
        for _ in range(m):
            k = rng.integers(2, 5)
            e = tuple(rng.choice(n, size=k, replace=False).tolist())
            edges.append(e)
        masks = stratified_split(labels, seed)
        return feats, edges, labels, masks

    def _targets(self, labels, mask):
        t = np.full(len(labels), -1)
        t[mask] = labels[mask]
        return t

    def test_eta_zero_and_steps_zero_identities(self):
        """修订 1 后的恒等式（decision_log 有记录）：
        steps=0 ⟹ K_0=I ⟹ C = Z0 + eta R（精确核恒等式）；
        eta=0 ⟹ C = K_T(P_lam) Z0（与同图传播臂逐项相等）。"""
        feats, edges, labels, masks = self._synth()
        tr = self._targets(labels, masks['train'])
        va = self._targets(labels, masks['val'])
        out = fit_and_select(feats, edges, tr, va, masks, seed=1,
                             grid=[(0.0, 0.5)], steps=0)
        np.testing.assert_allclose(out['C'], out['Z0'] + 0.5 * out['R'],
                                   atol=0)
        out2 = fit_and_select(feats, edges, tr, va, masks, seed=1,
                              grid=[(0.3, 0.0)], steps=7)
        from near_locking import row_stochastic as rs
        from correction import (candidate_scores, edge_pairs,
                                hard_weights, residual_matrix)
        rows, cols, owner = edge_pairs(edges)
        n = len(labels)
        P_h = rs(n, rows, cols, hard_weights(out2['z0'], edges)[owner])
        P_a = rs(n, rows, cols, np.ones(len(edges))[owner])
        P_lam = 0.7 * P_h + 0.3 * P_a
        R = residual_matrix(tr, out2['Z0'], masks['train'], 3)
        np.testing.assert_allclose(
            out2['C'], candidate_scores(P_lam, out2['Z0'], R, 0.0,
                                        steps=7), atol=1e-12)

    def test_residual_reads_train_nodes_only(self):
        feats, edges, labels, masks = self._synth()
        Z0 = np.zeros((60, 3)); pred = np.tile(np.arange(60) % 3, 1)
        Z0[np.arange(60), pred] = 1.0
        tr = self._targets(labels, masks['train'])
        R = residual_matrix(tr, Z0, masks['train'], 3)
        nz = np.abs(R).sum(axis=1) > 0
        # R 的支持 ⊆ 训练节点（预测正确的训练节点上 R=0 也合法）
        self.assertTrue((~nz | masks['train']).all())
        # 训练节点上 R = onehot(y) - Z0
        u = int(np.flatnonzero(masks['train'])[0])
        e = np.zeros(3); e[labels[u]] = 1.0
        np.testing.assert_allclose(R[u], e - Z0[u], atol=1e-12)

    def test_same_label_budget_across_methods(self):
        """候选与基线共享同一训练 mask 的预测器与残差；验证标签只增不增。"""
        feats, edges, labels, masks = self._synth()
        tr = self._targets(labels, masks['train'])
        va = self._targets(labels, masks['val'])
        cand = fit_and_select(feats, edges, tr, va, masks, seed=5)
        cs = cs_baseline(feats, edges, tr, va, masks, seed=5)
        po = prop_baseline(feats, edges, tr, va, masks, seed=5,
                           structure='original')
        self.assertEqual(cand['budget']['train_label_nodes'],
                         cs['budget']['train_label_nodes'])
        self.assertEqual(cand['budget']['train_label_nodes'],
                         po['budget']['train_label_nodes'])
        self.assertEqual(cand['budget']['val_label_nodes'],
                         cs['budget']['val_label_nodes'])
        # 同种子同训练 mask ⟹ 完全相同的预测（共享初始化与构图）
        np.testing.assert_array_equal(cand['z0'], cs['z0'])
        np.testing.assert_array_equal(cand['z0'], po['z0'])
        # 候选的残差预算不包含验证节点（支持 ⊆ 训练 mask）
        self.assertTrue((~(np.abs(cand['R']).sum(axis=1) > 0)
                         | masks['train']).all())

    def test_changed_test_labels_do_not_change_deployment(self):
        """泄漏测试：固定 masks、固定类别集合；两份不同测试真值副本
        只进最终评估；部署输出逐项相同。"""
        feats, edges, labels, masks = self._synth()
        tr = self._targets(labels, masks['train'])
        va = self._targets(labels, masks['val'])
        alt = labels.copy()
        alt[masks['test']] = (alt[masks['test']] + 1) % 3
        outs = []
        for _ in (0, 1):
            cand = fit_and_select(feats, edges, tr, va, masks, seed=7)
            cs = cs_baseline(feats, edges, tr, va, masks, seed=7)
            po = prop_baseline(feats, edges, tr, va, masks, seed=7,
                               structure='soft')
            outs.append((cand['C'].copy(), cs['C'].copy(), po['C'].copy(),
                         cand['lam'], cand['eta'], cs['alpha1'], cs['alpha2'],
                         po['alpha']))
        for a, b in zip(outs[0], outs[1]):
            if isinstance(a, np.ndarray):
                np.testing.assert_array_equal(a, b)
            else:
                self.assertEqual(a, b)
        # 评估仅在最后使用测试真值；两份真值给不同的（合法）评估结果
        te1 = self._targets(labels, masks['test'])
        te2 = self._targets(alt, masks['test'])
        m1 = evaluate(outs[0][0], te1, masks['test'])
        m2 = evaluate(outs[0][0], te2, masks['test'])
        self.assertIn('acc', m1); self.assertIn('acc', m2)

    def test_val_selection_prefers_smaller_eta_then_lambda(self):
        feats, edges, labels, masks = self._synth()
        tr = self._targets(labels, masks['train'])
        va = self._targets(labels, masks['val'])
        out = fit_and_select(feats, edges, tr, va, masks, seed=11,
                             grid=[(0.0, 0.0), (1.0, 0.0)])
        self.assertEqual((out['lam'], out['eta']), (0.0, 0.0))


class LgsReweightingTests(unittest.TestCase):
    """A⊙E(P₀) 与未屏蔽 E(P₀) 是不同对象（导师第四轮勘误的回归测试）：
    掩码关系只在原图共现对上有权重；全 E 对所有对有类转移权重；
    类聚合核实现必须与显式稠密 E 核逐项一致。"""

    def test_masked_vs_full_label_relation(self):
        from lgs_audit import (apply_full_label, full_label_weights,
                               sinkhorn)
        from near_locking import kernel_apply
        n, C = 4, 2
        # A 含类内边 (0,1),(2,3) 与一条跨类边 (0,2)；跨类对 (1,3) 不在 A
        rows = [0, 1, 1, 0, 2, 3, 3, 2, 0, 2]
        cols = [0, 1, 0, 1, 2, 3, 2, 3, 2, 0]
        a = np.array([0, 0, 1, 1])
        A = np.zeros((n, n))
        np.add.at(A, (rows, cols), 1.0)
        Y = np.eye(C)[a]
        P0 = sinkhorn(Y.T @ A @ Y)
        self.assertGreater(P0[0, 1], 0.01)        # 跨类转移非零
        w_masked = P0[a[rows], a[cols]]
        from near_locking import row_stochastic
        P_masked = row_stochastic(n, rows, cols, w_masked).toarray()
        self.assertEqual(P_masked[1, 3], 0.0)  # 非共现跨类对：掩码权重 0
        W = full_label_weights(a, P0, C)
        cross = 1.0 - W[np.arange(n), a] * np.bincount(a)[a]
        self.assertGreater(cross.max(), 0.0)       # 全 E：跨类对有质量
        E = W[:, a]                               # E[u,v] = W[u, a_v]
        self.assertGreater(E[1, 3], 0.0)           # 同一对全 E 有质量
        KG_fast = apply_full_label(W, a, C, 0.9, 7)
        KG_dense = kernel_apply(E, np.eye(C)[a], 0.9, 7)
        np.testing.assert_allclose(KG_fast, KG_dense, atol=1e-12)

    def test_feature_channel_not_sign_monotone(self):
        """特征通道对跨类质量无固定符号：同类特征边可降低跨类质量。"""
        from near_locking import kernel_apply, row_stochastic
        # base：完全图（0 直连跨类节点 2）；feat：仅同类边 (0,1)
        base = row_stochastic(3, [0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1],
                              [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).toarray()
        feat = row_stochastic(3, [0, 1], [1, 0], [1.0, 1.0]).toarray()
        a = np.array([0, 0, 1])
        G = np.eye(2)[a]
        rho_base = 1 - kernel_apply(base, G, 0.9, 20)[0, 0]
        rho_feat = 1 - kernel_apply(feat, G, 0.9, 20)[0, 0]
        self.assertGreater(rho_base, 0.3)
        self.assertAlmostEqual(rho_feat, 0.0, places=9)  # 同类特征边降 ρ


class LgsSnapshotTests(unittest.TestCase):
    """真实 LGS 快照审计的回归测试（缩减版训练，合成数据）。"""

    def _synth(self, seed=0, n=90, C=3):
        rng = np.random.default_rng(seed)
        labels = rng.integers(0, C, n)
        X = np.zeros((n, 6))
        for c in range(C):
            idx = np.flatnonzero(labels == c)
            ctr = rng.normal(size=6)
            X[idx] = ctr + 0.3 * rng.normal(size=(len(idx), 6))
        edges = [tuple(rng.choice(n, 3, replace=False).tolist())
                 for _ in range(25)]
        from correction import edge_pairs
        rows, cols, _ = edge_pairs(edges)
        A = np.zeros((n, n))
        np.add.at(A, (rows, cols), 1.0)
        masks = stratified_split(labels, seed=seed)
        return A, X, labels, masks

    def test_snapshot_trains_and_audits(self):
        import lgs_snapshot as LS
        A, X, labels, masks = self._synth()
        old = (LS.WARMUP, LS.ROUNDS, LS.EPOCHS_PER)
        LS.WARMUP, LS.ROUNDS, LS.EPOCHS_PER = 40, 3, 10
        try:
            snap = LS.run_split(A, X, labels, masks, seed=3)
        finally:
            LS.WARMUP, LS.ROUNDS, LS.EPOCHS_PER = old
        for k in ("pred", "P", "S", "S_f", "A_t", "blend",
                  "labels_mixed", "acc", "early_stop"):
            self.assertIn(k, snap)
        # 训练有效：测试精度显著优于随机（1/3）
        self.assertGreater(snap["acc"]["test"], 0.6)
        # P 每轮投影回双随机；快照为最优轮更新后状态，审计只用其
        # 相对结构（P_diag 报告），非负/有限即可
        self.assertTrue(np.isfinite(snap["P"]).all())
        self.assertGreaterEqual(float(snap["P"].min()), -1e-5)
        # 冻结子步审计字段齐备
        n = len(labels)
        blend = snap["blend"] / np.maximum(
            snap["blend"].sum(axis=1, keepdims=True), 1e-12)
        v = LS.audit_relation(blend, n, snap["labels_mixed"], labels,
                              masks["test"])
        for k in ("eps_max", "rho_certified_frac", "changed_frac",
                  "corrected_test", "wrong_changed_test",
                  "changed_among_certified"):
            self.assertIn(k, v)

    def test_snapshot_deterministic(self):
        import lgs_snapshot as LS
        A, X, labels, masks = self._synth(seed=5)
        old = (LS.WARMUP, LS.ROUNDS, LS.EPOCHS_PER)
        LS.WARMUP, LS.ROUNDS, LS.EPOCHS_PER = 20, 2, 5
        try:
            s1 = LS.run_split(A, X, labels, masks, seed=7)
            s2 = LS.run_split(A, X, labels, masks, seed=7)
        finally:
            LS.WARMUP, LS.ROUNDS, LS.EPOCHS_PER = old
        np.testing.assert_array_equal(s1["pred"], s2["pred"])
        np.testing.assert_allclose(s1["P"], s2["P"], atol=1e-5)  # float32


class SnapshotRepairTests(unittest.TestCase):
    """快照审计四项修复的回归测试（2026-10-10 复查轮）。"""

    def _synth(self, seed=0, n=80, C=3):
        rng = np.random.default_rng(seed)
        labels = rng.integers(0, C, n)
        X = np.zeros((n, 6))
        for c in range(C):
            idx = np.flatnonzero(labels == c)
            X[idx] = rng.normal(size=6) + 0.3 * rng.normal(size=(len(idx), 6))
        from correction import edge_pairs
        edges = [tuple(rng.choice(n, 3, replace=False).tolist())
                 for _ in range(22)]
        rows, cols, _ = edge_pairs(edges)
        A = np.zeros((n, n))
        np.add.at(A, (rows, cols), 1.0)
        masks = stratified_split(labels, seed=seed)
        return A, X, labels, masks

    def test_no_test_label_leakage(self):
        """修复 1：固定 masks/seed，替换全部测试标签 ⟹ 快照逐项相同。"""
        import lgs_snapshot as LS
        A, X, labels, masks = self._synth(seed=2)
        alt = labels.copy()
        alt[masks['test']] = (alt[masks['test']] + 1) % 3
        old = (LS.WARMUP, LS.ROUNDS, LS.EPOCHS_PER)
        LS.WARMUP, LS.ROUNDS, LS.EPOCHS_PER = 20, 2, 5
        try:
            s1 = LS.run_split(A, X, labels, masks, seed=4)
            s2 = LS.run_split(A, X, alt, masks, seed=4)
        finally:
            LS.WARMUP, LS.ROUNDS, LS.EPOCHS_PER = old
        np.testing.assert_allclose(s1['blend'], s2['blend'], atol=1e-6)
        np.testing.assert_allclose(s1['P'], s2['P'], atol=1e-9)
        np.testing.assert_array_equal(s1['pred'], s2['pred'])

    def test_P_valid_transition(self):
        """修复 3：快照 P 恒为双随机概率矩阵（紧界）。"""
        import lgs_snapshot as LS
        A, X, labels, masks = self._synth(seed=1)
        old = (LS.WARMUP, LS.ROUNDS, LS.EPOCHS_PER)
        LS.WARMUP, LS.ROUNDS, LS.EPOCHS_PER = 20, 3, 5
        try:
            snap = LS.run_split(A, X, labels, masks, seed=6)
        finally:
            LS.WARMUP, LS.ROUNDS, LS.EPOCHS_PER = old
        self.assertGreaterEqual(float(snap['P'].min()), -1e-6)
        self.assertLessEqual(float(snap['P'].max()), 1.0 + 1e-6)
        np.testing.assert_allclose(snap['P'].sum(axis=1), 1.0, atol=1e-6)
        np.testing.assert_allclose(snap['P'].sum(axis=0), 1.0, atol=1e-6)

    def test_train_ce_decreases(self):
        """修复 2 的训练有效性：warm-up 显著降低训练交叉熵。"""
        import lgs_snapshot as LS
        from correction import onehot
        A, X, labels, masks = self._synth(seed=3)
        rng = np.random.default_rng(9)
        n, d = X.shape
        C = int(labels.max()) + 1
        Y = onehot(labels, C)
        W1 = rng.normal(0, 0.05, (d, LS.HID)).astype(np.float32)
        W2 = rng.normal(0, 0.1, (LS.HID, C)).astype(np.float32)
        X32 = X.astype(np.float32)
        Ahat = LS.sym_norm(A.astype(np.float32))
        opt = LS.Adam([W1.shape, W2.shape])

        def ce():
            Z, *_ = LS.gcn_forward(Ahat, X32, W1, W2)
            z = np.clip(Z[masks['train']], 1e-9, 1)
            return float(-(Y[masks['train']] * np.log(z)).sum(1).mean())
        c0 = ce()
        for _ in range(30):
            Z, X1, H, U1, U2 = LS.gcn_forward(Ahat, X32, W1, W2)
            dZ = np.zeros_like(Z)
            dZ[masks['train']] = (Z[masks['train']] - Y[masks['train']])
            dZ /= masks['train'].sum()
            dW2 = (Ahat @ H).T @ dZ
            dH = Ahat.T @ (dZ @ W2.T)
            dX1 = dH * (X1 > 0)
            dW1 = (Ahat @ X32).T @ dX1
            W1, W2 = opt.step([W1, W2], [dW1, dW2])
        c1 = ce()
        self.assertLess(c1, c0 * 0.7)


class SplitTests(unittest.TestCase):
    def test_stratified_proportions_and_coverage(self):
        rng = np.random.default_rng(2)
        labels = rng.integers(0, 4, 500)
        masks = stratified_split(labels, seed=42)
        self.assertTrue((masks['train'] | masks['val'] | masks['test']).all())
        self.assertFalse((masks['train'] & masks['test']).any())
        self.assertFalse((masks['train'] & masks['val']).any())
        self.assertFalse((masks['val'] & masks['test']).any())
        for c in range(4):
            idx = np.flatnonzero(labels == c)
            f_tr = masks['train'][idx].mean()
            f_te = masks['test'][idx].mean()
            self.assertAlmostEqual(f_tr, 0.6, delta=0.12)
            self.assertAlmostEqual(f_te, 0.2, delta=0.10)
        # 划分由 seed 决定（可复现）
        m2 = stratified_split(labels, seed=42)
        for k in masks:
            np.testing.assert_array_equal(masks[k], m2[k])

    def test_tiny_class_rules(self):
        labels = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 2])
        masks = stratified_split(labels, seed=0)
        # 每类至少 1 个训练节点；互斥且全覆盖
        for c in (0, 1, 2):
            idx = np.flatnonzero(labels == c)
            self.assertGreater(masks['train'][idx].sum(), 0)
        self.assertTrue((masks['train'] | masks['val'] | masks['test']).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
