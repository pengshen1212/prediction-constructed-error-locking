"""外层信息协议（计划 §6/§8.1，2026-10-09）。

职责：
  - 按类分层 60/20/20 train/val/test 划分（舍入与最小训练数规则固定）；
  - 预测器只在 train 上拟合，transductive 地给出全体预测（真值仅经
    masked 副本流动：train_targets / val_targets / test_targets 各自
    含 sentinel -1，方法侧永远看不到自己 mask 外的真值）；
  - 类别集合来自公开任务定义（labels 的取值域），不由测试标签决定。

泄漏边界：本模块的任何函数都不把 test 标签交给方法层；evaluate()
（correction.py）是唯一最终接收 test 真值的地方。
"""

from __future__ import annotations

import numpy as np


TRAIN_FRAC = 0.60
VAL_FRAC = 0.20
# test = 其余（约 20%）。小类规则：n_c=1 → 全 train；n_c=2 → 1 train 1 test；
# n_c>=3 → test=max(1, round(0.2 n_c)), val=max(1, round(0.2 n_c))，
# 且训练节点 = 其余并保证 >= 1（不够时从 test 回收）。


def stratified_split(labels, seed, fracs=(TRAIN_FRAC, VAL_FRAC)):
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    train = np.zeros(len(labels), dtype=bool)
    val = np.zeros(len(labels), dtype=bool)
    test = np.zeros(len(labels), dtype=bool)
    for c in np.unique(labels):          # 类别集合来自全体声明域
        idx = np.flatnonzero(labels == c)
        rng.shuffle(idx)
        n_c = len(idx)
        if n_c == 1:
            train[idx[0]] = True
            continue
        if n_c == 2:
            train[idx[0]] = True
            test[idx[1]] = True
            continue
        n_test = max(1, int(round(0.2 * n_c)))
        n_val = max(1, int(round(0.2 * n_c)))
        if n_c - n_test - n_val < 1:     # 保证 >= 1 个训练节点
            n_test = min(n_test, n_c - 1)
            n_val = min(n_val, n_c - n_test - 1)
        test[idx[:n_test]] = True
        val[idx[n_test:n_test + n_val]] = True
        train[idx[n_test + n_val:]] = True
    return dict(train=train, val=val, test=test)


def targets(labels, mask):
    """masked 标签副本：mask 外一律 sentinel -1（方法层断言不读）。"""
    t = np.full(len(labels), -1, dtype=int)
    t[mask] = np.asarray(labels)[mask]
    return t


def _as_dense(features):
    if hasattr(features, "toarray"):        # scipy sparse
        return np.asarray(features.todense())
    return np.asarray(features)


def fit_predictor(features, train_mask, train_targets, seed,
                  predictor="prototype"):
    """在 train 上拟合预测器，返回全体预测（transductive 特征可用）。"""
    if predictor == "prototype":
        X = _as_dense(features)
        y = train_targets[train_mask]
        n_cls = int(y.max()) + 1
        protos = np.zeros((n_cls, X.shape[1]))
        for c in range(n_cls):
            m = y == c
            if m.sum() > 0:
                protos[c] = X[train_mask][m].mean(axis=0)
        x_sq = (X ** 2).sum(axis=1)
        p_sq = (protos ** 2).sum(axis=1)
        d = x_sq[:, None] - 2 * X @ protos.T + p_sq[None, :]
        return np.argmin(d, axis=1)
    if predictor == "logreg":
        from sklearn.linear_model import LogisticRegression
        X = _as_dense(features)
        clf = LogisticRegression(max_iter=300, C=1.0, random_state=int(seed))
        clf.fit(X[train_mask], train_targets[train_mask])
        return clf.predict(X)
    if predictor == "lp":
        # Walmart 族：无特征；由调用方提供 edges/传播器（见 run 脚本）。
        raise ValueError("lp predictor needs the block builder in "
                         "run_correction_*.py (edges + S propagation)")
    raise ValueError(predictor)
