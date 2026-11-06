# 实验方法学审查 / Methodology review

> 2026-11-06。起因：用户质疑「为什么搞得这么不对，有没有参考其他人的项目」。
> 结论：**没有系统参考**。本仓库是按 PennyLane API 从零拼出的「能跑通」的 demo，
> 把接口正确当成了实验设计正确。本文列出查明的问题、实测证据与修正状态。
> 审查过程中补跑了 3 组诊断实验，原始数据在 `results/diag_*.json`（该目录被 gitignore）。
> **同日二轮修正**：已用 `experiments/train_multiseed.py` 完成 5 seed 重跑，原始指标
> 归档在 `docs/multiseed_metrics.json`；编码对照确认为 −0.10 ± 8.68 pt（不显著），
> 最佳量子配置 0.6640 ± 0.0216。见第 5 节。

## 一句话结论

准确率低有一部分确实是 toy 配置与任务难度，但更严重的是：
**一个种子 bug 让最重要的那组「受控对照」既不可复现、也不受控，而 README 却把它当成了主要结论。**
把初始权重真正固定后，原声称的 +15.5 个百分点变成 −3.0 / +12.5（两个 seed 符号相反），该结论撤回。

## 问题清单

| # | 严重度 | 问题 | 状态 |
| --- | --- | --- | --- |
| 1 | P0 | `train_amplitude.py` 模型构造早于 `set_seed`，结果不可复现 | 已修 |
| 2 | P0 | 同脚本 4 个模型初始权重各不相同，受控对照不成立 | 已修 |
| 3 | P0 | 全部结论都是 `seed=0` 单次、无方差 | **已修**：5 seed + 均值/标准差 + 配对 CI（第 5 节） |
| 4 | P1 | 量子输出 `[-1,1]` 直接接线性头接 CE，无 scale / 无非线性 | **已修**：`models.py` 支持 MLP 头 / 固定与可学习缩放，5 seed 实测见第 5 节 |
| 5 | P1 | 单次编码的表达力上限（degree-1 傅里叶 / 二次型） | 设计缺陷，需换架构 |
| 6 | P1 | 无经典 pre_net、无官方推荐的 `tanh(·)*π/2` 输入缩放 | 未修 |
| 7 | P2 | 任务口径与官方 demo 不对标（10 类端到端 vs 二分类 / 量子核） | 文档已说明 |
| 8 | P2 | 上一轮把低准确率归因于 PCA / 参数量 / barren plateau | 部分推翻，见下 |

## 1. P0：可复现性与「受控对照」

`qml.qnn.TorchLayer` 用 torch 全局 RNG 初始化权重（默认 `[0, 2π]` 均匀，见参考 [1]）。
因此**必须在构造模型之前**设种子。三个实验脚本的情况：

| 脚本 | `set_seed` 位置 | 是否可复现 |
| --- | --- | --- |
| `train_mnist.py` | `main()` 开头，模型之前 | 是 |
| `scan_capacity.py` | `run_one()` 开头，模型之前 | 是，且各组从同一起点，受控成立 |
| `train_amplitude.py` | 只在 `fit()` 内，模型已构造完 | **否** |
| `quantum-llm/train_llm.py` | `main()` 开头，模型之前 | 是 |

两次独立进程构造同一个 `VQCClassifier(4,10,2)`，实测初始权重前 4 个值：

```
run 1  [0.3820, 3.7360, 1.5519, 5.0900]
run 2  [4.1794, 4.8146, 2.1964, 2.3293]
```

同一进程内按 `train_amplitude.py` 顺序构造 4 个模型，`Angle-10` 与 `Amplitude-10` 的
权重 shape 都是 `(2, 10, 3)`，本可以完全相同，实测首值分别为
`[2.151, 4.034, 1.609]` 与 `[2.814, 3.610, 2.279]` —— 即 README 所称
「唯一差别是编码方式」并不成立，连初始权重都不同。

**修复**：`train_amplitude.py` 改为在每次构造模型前 `set_seed`，并把模型改为
lazy builder（`record(name, model_fn, ...)` 内先设种子再调用 `model_fn()`）。
同 shape 的两组现在拿到完全相同的初始权重与分类头。

## 2. P0：把初始权重固定后，+15.5pt 结论不成立

诊断脚本 `results/diag_controlled.py`：固定同一组 `(2,10,3)` 初始权重、同一分类头、
同一训练 shuffle，只换编码与输入（PCA-10 vs 完整 784 维），600/200 样本、8 epochs、
lr 0.01，两组各跑两个不同的初始权重：

| 初始权重 | Angle-10 | Amplitude-10 | 差值 |
| --- | --- | --- | --- |
| seed0 | 0.4400 | 0.4100 | **−3.0 pt** |
| seed1 | 0.3750 | 0.5000 | **+12.5 pt** |

原 README 的 0.3750 → 0.5300（**+15.5 pt**）来自单次运行 + 未固定的初始化，
波动幅度与「结论」同量级，**不能作为编码方式优劣的证据**。同一 Angle-4 配置在三个脚本里
分别得到 0.330 / 0.350 / 0.420，本身也说明单次结果噪声约 9 个百分点。

> 方法论结论：QML 小样本实验不带多 seed 均值与方差，任何个位数百分点的差异都不足以支撑结论。

## 3. P1：实测出来的真正可修复因素

诊断脚本 `results/diag_bottleneck.py`，同数据同优化器，只改初始化 / 输出缩放 / 分类头：

| 配置 | Angle-4 acc / loss | Amplitude-10 acc / loss |
| --- | --- | --- |
| 原版：默认 `[0,2π]` 初始化 + `Linear` 头 | 0.3300 / 1.774 | 0.5050 / 1.998 |
| 改小初始化 `N(0, 0.01²)` | 0.2800 / 2.096 | 0.5050 / 2.086 |
| 小初始化 + 输出 ×10 | 0.4100 / 1.653 | 0.4700 / 1.617 |
| 小初始化 + MLP 头 | 0.4350 / 1.665 | — |
| 小初始化 + MLP 头 + 输出 ×5 | **0.4400** / 1.677 | **0.5850 / 1.050** |

- **最有效的是分类头，不是量子电路**：Amplitude-10 的 loss 从 1.998 降到 1.050，
  准确率 +8 pt；Angle-4 +11 pt。原因是 `PauliZ` 期望恒在 `[-1,1]`，裸接
  `Linear(n,10)` 再喂 `CrossEntropyLoss` 时 logits 幅度太小、softmax 对差异不敏感，
  梯度被系统性压制 —— 即文献 [8] 的 *measurement-induced logit contraction*，
  官方/论文的对策是可学习 scale（QMT）或经典非线性 post_net。
- **「默认 `[0,2π]` 初始化是元凶」这个假设被实测推翻**：换成官方 demo 常用的
  `0.01` 小初始化，Angle-4 反而从 0.330 掉到 0.280；Amplitude-10 准确率持平但
  loss 更差（1.998→2.086）。小权重让电路近似恒等，输出对所有样本几乎一样，
  同样缺乏区分度。所以不能盲抄「小初始化防 barren plateau」。
- 因此上一轮把 `Angle-10` 比 `Angle-4` 差归因于 barren plateau，也证据不足：
  在 P0 的初始化噪声下，这个差值完全可能只是运气。

> **⚠ 二轮修正（2026-11-06）**：本节所有数字都是**单次运行**，且
> 「小初始化 + MLP 头 + 输出缩放」一次混改了三个变量，只能当线索、不能当单项效应引用。
> 5 seed 重跑（第 5 节）把变量拆开后，单项收益远小于这里：MLP 头
> +6.60 ± 4.97 pt（Angle-4，CI 勉强不跨 0）/ +6.20 ± 5.44 pt（Amplitude-10，CI 跨 0），
> 固定 ×5 缩放 +3.20 ± 3.78 pt（不显著），可学习缩放与固定 ×5 等价（−0.20 pt）。
> 早期的 +8 / +11 pt 是多重改动叠加与单次噪声共同放大的结果。

## 4. P1：与官方 / 文献做法的结构性偏差

| 项 | 本仓库 | 参考做法 |
| --- | --- | --- |
| 权重初始化 | 默认 `[0,2π]` 均匀（TorchLayer 默认） | 官方 VQC demo 用 `0.01*np.random.randn` [2]；迁移学习 demo `q_delta=0.01` [3]；Qiskit `0.1*(2*rand-1)` [4] |
| 输入缩放 | min-max → `[-π,π]`（实测 29% 值落在 `|x|<0.5`，1.15% 落在 `|x|>2.5` 的 cos 饱和区） | `q_in = tanh(pre_out)*π/2`，刻意避开 cos 平坦区 [3][9] |
| 经典前端 | 无；PCA 或 784 原始像素直接进量子层 | ResNet18 冻结 → `Linear(512,4)` [3]；CNN → `Linear(64,2)` [4]；PCA/CAE → 10 维 [7] |
| 量子后端 | 原为裸 `Linear(n_qubits, 10)`，无非线性、无 scale；**二轮已支持** `head="mlp"` 与固定 / 可学习 scale（默认仍为旧行为） | `post_net` / `Dense(10, softmax)` [5][6]；或 QMT 可学习缩放 [8] |
| 特征映射 | 单次 `AngleEmbedding(RY)` | Qiskit 默认 `ZZFeatureMap`（含二次相互作用）[10]；data re-uploading 扩展频谱 [11][12] |
| 任务口径 | 10 类端到端 VQC | Qiskit 官方 MNIST 是 0/1 二分类、200 样本 [4]；PennyLane quanvolution 是 10 类但量子电路**固定不可训练**、只当核用、50 样本 30 epochs [5] |

两条硬性的表达力上界，说明剩余差距不是调参能补的：

- 单次角度编码的模型是关于输入的 **degree-1 多元三角多项式**，频谱
  `Ω={-1,0,1}^D`；`RY(x)` + `PauliZ` 读数是 `cos(x)` 型偶函数，无额外偏置时
  `x` 与 `−x` 不可分 [11][13]。
- `AmplitudeEmbedding` + 酉 + `PauliZ` 期望等价于
  `f(x)=tr(ρ(x)M)=Σ M′_ij x_i x_j / ‖x‖²`，即**输入的二次型**，对应二次多项式核
  `(xᵀx′)²`，可被经典高效计算；变分电路只改变测量基，不提升关于 x 的代数次数 [14][15]。

所以「把 qubits / 参数堆大」无法绕过这两条上界，这与 `scan_capacity.py` 的实测
（深度 8→16 层准确率 0.470→0.480）一致。真正的出路是 data re-uploading、
多可观测量、或混合 CNN 前端。

## 5. 多 seed 重跑：结论与撤回（2026-11-06 二轮）

`experiments/train_multiseed.py`：真实 MNIST IDX、600 训练 / 200 测试、10 类、24 epochs、
batch 32、lr 0.01、**5 个 seed**。数据切分固定 `seed=0`（方差只来自初始化与 shuffle），
每次构造模型前重置种子；脚本断言同 qubits 各组的量子初始权重逐元素相同、
`Angle-10` 与 `Amplitude-10` 的分类头也相同 —— **5/5 seed 全部 PASS**。
原始数据归档在 `docs/multiseed_metrics.json`，图在 `docs/figures/multiseed_*.png`。

| 模型 | 参数 | acc@8 | 最终 acc@24 |
| --- | --- | --- | --- |
| Angle-4 / Linear（原基线） | 74 | 0.3470 ± 0.0431 | 0.4320 ± 0.0375 |
| Angle-4 / MLP | 514 | 0.4770 ± 0.0517 | 0.4980 ± 0.0289 |
| Angle-4 / MLP ×5 | 514 | 0.4920 ± 0.0340 | 0.4790 ± 0.0108 |
| Angle-10 / Linear | 170 | 0.5060 ± 0.0263 | 0.5710 ± 0.0321 |
| Amplitude-10 / Linear | 170 | 0.4800 ± 0.0398 | 0.5700 ± 0.0583 |
| Amplitude-10 / MLP | 742 | 0.5110 ± 0.0397 | 0.6320 ± 0.0305 |
| **Amplitude-10 / MLP ×5** | 742 | 0.6180 ± 0.0256 | **0.6640 ± 0.0216** |
| Amplitude-10 / MLP + 可学习缩放 | 743 | 0.5750 ± 0.0352 | 0.6620 ± 0.0399 |
| MLP-4（经典） | 490 | 0.5800 ± 0.0221 | 0.5880 ± 0.0168 |
| MLP-784（经典） | 25450 | 0.8550 ± 0.0100 | 0.8640 ± 0.0065 |

**关键配对差值**（百分点，均值 ± 标准差，t 分布 95% CI，n=5）：

| 对比 | 差值 | 95% CI | 判定 |
| --- | --- | --- | --- |
| `Amplitude-10 / Linear − Angle-10 / Linear`（严格受控编码对照） | −0.10 ± 8.68 | [−10.88, +10.68] | **不显著，旧 +15.5 pt 正式撤回** |
| `Angle-4` MLP 头 − Linear 头 | +6.60 ± 4.97 | [+0.43, +12.77] | 勉强显著 |
| `Angle-4` 固定 ×5 − 仅 MLP 头 | −1.90 ± 2.33 | [−4.79, +0.99] | 不显著 |
| `Amplitude-10` MLP 头 − Linear 头 | +6.20 ± 5.44 | [−0.55, +12.95] | 不显著 |
| `Amplitude-10` 固定 ×5 − 仅 MLP 头 | +3.20 ± 3.78 | [−1.50, +7.90] | 不显著 |
| 可学习缩放 − 固定 ×5 | −0.20 ± 3.13 | [−4.09, +3.69] | 不显著（两者等价） |
| 最佳量子配置 − 原基线 | **+23.20 ± 5.40** | [+16.49, +29.91] | **显著，5/5 seed 为正** |

1. **编码方式不是准确率瓶颈**：严格受控下差值几乎为零，逐 seed
   `[+12.5, +1.5, +2.0, −7.0, −9.5]` 符号随初始化翻转。旧 README 的 +15.5 pt 与第 2 节
   的 −3.0 / +12.5 pt 都只是初始化运气；「不压缩」解除了工程上的限制，但**不带来可测精度收益**。
2. **0.432 → 0.664 的提升是配置组合的联合效果**：输入表示（PCA-4 → 784 维）、分类头、
   输出缩放、训练预算（8 → 24 epochs）、量子参数量（24 → 60）一起变了。单项可分离的
   收益只有几 pt 且多数不显著，不能归给「量子表达力提升」。
3. **经典基线仍全面领先**：`MLP-4` 0.588 > 最佳 Angle-4 0.498（同输入），
   `MLP-784` 0.864 > 最佳量子 0.664。与第 4 节两条表达力上界一致，堆 qubits 补不上。
4. **本节仍有的限制**：参数预算不对等（742 vs 25450）、未做多重比较校正、固定数据切分、
   状态向量模拟（振幅编码在真机需 O(2ⁿ) 状态制备）、`n=5` 的 CI 偏宽。
5. **工程验证**：两次独立 `--quick` 运行除 `seconds` 字段外逐字段一致（可复现）；
   `tests/smoke_test.py` 新增三种分类头的前向/反向、`eval_train=False` 路径、
   以及「构造前设种子必须得到相同初始权重」的断言。

## 6. 已修 / 待修

**已修（二轮：多 seed 重跑）**
- 新增 `experiments/train_multiseed.py`：10 组 × 5 seed、受控性断言、均值 ± 标准差、
  配对差值与 95% CI；归档 `docs/multiseed_metrics.json` 与两张 `docs/figures/multiseed_*.png`。
- `qml_lab/models.py`：`head="linear"|"mlp"`、`scale=<float>|"learnable"`，默认行为向后兼容。
- `qml_lab/train.py`：`eval_train=False` 跳过训练集全量前向（2ⁿ 模拟下省时），未测量的
  `train_acc` 记为 `None` 而不是填占位数字。
- `tests/smoke_test.py`：覆盖新分类头、缩放、轻量训练路径与初始化受控性。
- `README.md`：全部结论性数字替换为 5 seed 统计，+15.5 pt 正式撤回，旧单次结果降级为存档。

**已修（一轮）**
- `experiments/train_amplitude.py`：模型构造前重置种子、改 lazy builder、
  `main()` 开头加进程级 `set_seed`。四个模型现在共享同一初始权重（同 shape 时完全相同）。
- README 中「严格受控」「+15.5 pt 归因」等错误表述已加撤回说明。

**待修（按优先级）**
1. 等参数预算对照 + 量子核 QSVC 第三条基线；`scan_capacity.py` 的 16 qubits 那组（762 s）不必再跑满。
2. 加经典 pre_net + `tanh(·)*π/2` 缩放，或改混合架构（CNN 提特征 → 小量子头）——
   目前最有希望缩小与 `MLP-784` 差距的方向。
3. 换编码：data re-uploading / 多可观测量读出，突破 degree-1 与二次型上界。
4. 更多 seed（≥20）与多重比较校正，给第 5 节那些「不显著」项更紧的置信区间。
5. 噪声模型、有限 `--shots`、`lightning.gpu` 后端。

## 参考（均为 2026-11-06 实检索）

1. PennyLane `qml.qnn.TorchLayer` 文档（默认 `[0,2π]`、`init_method`）：https://docs.pennylane.ai/en/stable/code/api/pennylane.qnn.TorchLayer.html
2. PennyLane 官方 Variational classifier demo（`0.01*np.random.randn`）：https://pennylane.ai/qml/demos/tutorial_variational_classifier
3. PennyLane Quantum transfer learning（`q_delta=0.01`、`tanh*π/2`、ResNet18 + DressedQuantumNet）：https://pennylane.ai/qml/demos/tutorial_quantum_transfer_learning/
4. Qiskit `TorchConnector` MNIST tutorial（0/1 二分类、200 样本、显式初始权重）：https://qcrepository.org/2606.24566.1
5. PennyLane Quanvolution（10 类、固定不可训练量子核、50 样本 / 30 epochs）：https://pennylane.ai/qml/demos/tutorial_quanvolution
6. PennyLane 多分类 one-vs-all + margin loss：https://pennylane.ai/demos/tutorial_multiclass_classification
7. ICCAD 2021，PCA/CAE + 10-qubit QNN，4000 样本 / 20 epochs / Adagrad lr=0.5：https://ar5iv.arxiv.org/html/2109.02862
8. *Mitigating measurement-induced training instability in hybrid QNN*（EAAI 2026，QMT / logit contraction）：https://arxiv.org/abs/2606.22551
9. Springer 2026，`tanh(feat)*π/2` 与 cos 饱和区梯度分析：https://link.springer.com/content/pdf/10.1007/s42484-026-00431-3.pdf
10. Qiskit `ZZFeatureMap` 官方教程：https://qiskit.org/ecosystem/machine-learning/locale/hi_IN/tutorials/02a_training_a_quantum_model_on_a_real_dataset.html
11. Schuld, Sweke, Meyer, *Effect of data encoding on expressive power*, PRA 103 032430：https://arxiv.org/abs/2008.08605
12. Pérez-Salinas et al., *Data re-uploading for a universal quantum classifier*, Quantum 4 226：https://arxiv.org/abs/1907.02085
13. PennyLane data-reuploading classifier demo：https://www.pennylane.ai/demos/tutorial_data_reuploading_classifier
14. Schuld, *Supervised quantum machine learning models are kernel methods*：https://arxiv.org/abs/2101.11020
15. McClean et al., *Barren plateaus*, Nat. Commun. 9 4812：https://arxiv.org/abs/1803.11173 ；Grant et al., *An initialization strategy*, Quantum 3 214：https://arxiv.org/abs/1903.05076
