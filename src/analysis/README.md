# 命题二第一步实验：关系误判 vs 能量/分类分离

对应 `library/hgnn-theory/research-proposals-2026-09-26.md` 第三节（命题二"最小可做范围"）。
主报告（含定理条件对照表与结果解读）：`library/hgnn-theory/prob2-step1-rel-error-2026-09-26.md`。

## 设计一句话

两类、3-均匀超图 SBM + 带噪特征；带符号关系传播（同类吸引/异类排斥 + 每步球面重整化，作为 HAMP 式类条件相互作用的**最小代理模型**，非 HAMP/HNRD 复现）；扫描关系误判率 η、深度 T、误判结构（iid / top-degree 集中 / 少标签+带噪特征的真实估计），测量正权重 Dirichlet 能量、真实关系带符号能量、类内/类间几何、岭探针分类间隔与准确率。

## 运行

```bash
python run_experiment.py            # 完整：10 seeds, eta 0:0.025:0.5, T∈{2,5,10,25,50}
python run_experiment.py --quick    # 冒烟：3 seeds, 粗网格
```

The revised spectral-bound grid is written separately as
`results/gate2_spectral_grid_revised.csv`; the previous
`gate2_spectral_grid.csv` is retained as the legacy run. To recompute only its
probability thresholds and derived coverage metrics without repeating spectral
norm calculations, run `python recompute_gate2_grid.py`.

输出到 `results/`：`s1_sweep.csv`（受控翻转×深度）、`s2_modes.csv`（集中误判）、
`s3_estimation.csv`（估计路径，含 probe64 解混淆列）、`summary.json`（线性拟合、
分离区间、基线）、`fig1..fig4*.png`。

## 关键建模选择（防止误读结果）

- **球面重整化**扮演深层网络的归一化层，是表征有界性的来源；也因此 C_rel(T) 呈饱和而非爆炸——这是本模型的性质，不是 HAMP 的结论。
- **岭探针 λ=10**：深层传播后表征有效秩极低，弱正则（1e-3）会让最小二乘放大低方差噪声维，margin/‖w‖ 严重失真（诊断见主报告附录 A）。所有探针统一 λ=10。
- **S1/S2 的"oracle 关系"来自真标签**，是受控算子扰动实验（上界性质）；S3 才是诚实估计路径（动力学只用 ℓ 个种子标签）。
- 探针训练标签只用于评估读出，不参与 S1/S2 的关系设定。

## 指标定义

- `E_pos`：团展开正权重拉普拉斯二次型 / n（行在单位球上）。过平滑坍缩指示器。
- `E_signed_true`：Σ w·s_true·‖z_i−z_j‖²/Σw。真实类结构对齐度，越负越好；过零 = 失配。
- `margin`：岭探针归一化间隔 y·(w·z+b)/‖w‖（测试节点均值）；`acc`：探针准确率。
- `inter/intra`：类心距 / 类内平均离散度。
