# 实验方法学审查 / Methodology review

> 2026-11-06。起因：用户质疑「为什么搞得这么不对，有没有参考其他人的项目」。
> 结论：**没有系统参考**。本仓库是按 PennyLane API 从零拼出的「能跑通」的 demo，
> 把接口正确当成了实验设计正确。本文列出查明的问题、实测证据与修正状态。
> 审查过程中补跑了 3 组诊断实验，原始数据在 `results/diag_*.json`（该目录被 gitignore）。
> **同日二轮修正**：已用 `experiments/train_multiseed.py` 完成 5 seed 重跑，原始指标
> 归档在 `docs/multiseed_metrics.json`；编码对照确认为 −0.10 ± 8.68 pt（不显著），
> 二轮最佳 0.6640 ± 0.0216。见第 5 节。
> **同日三 / 四轮**：`docs/ceiling_metrics.json`（第 8 节）把 66% 定位到量子读出的
> 信息损失；`docs/readout_metrics.json`（第 9 节）用多可观测量把读出从 10 维扩到 40 维，
> 当前最佳 0.7720 ± 0.0115，冻结探针从 0.679 升到 0.811。

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
| 6 | P1 | 无经典 pre_net、无官方推荐的 `tanh(·)*π/2` 输入缩放 | **已实测**：`HybridCNNClassifier` 补上两种 CNN 前端 + `tanh*π/2`，5 seed 见第 8 节；但量子层在混合结构里无正贡献、轻量版训练崩溃，单独 lr 也未能解决（第 9 节），**仍未解决** |
| 7 | P2 | 任务口径与官方 demo 不对标（10 类端到端 vs 二分类 / 量子核） | 文档已说明 |
| 8 | P2 | 上一轮把低准确率归因于 PCA / 参数量 / barren plateau | 部分推翻，见下 |
| 9 | P1 | 读出只取 n 个有界 `PauliZ` 期望，是硬信息瓶颈 | **已修且已验证**：`make_rich_layer` 支持 X/Y/ZZ，10→40 维读出使 0.664→0.772、探针 0.679→0.811（第 9 节）；真机测量分组代价未测 |

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

**待修（2026-11-06 四轮后按优先级，详见第 9.5 节）**
1. 量子层在混合架构里的初始化 / 深度 / lr 系统调参 —— 单独 lr 这一项已试且排除。
2. 按 [12][13] 原始设计重做 data re-uploading（本轮朴素实现为负结果）。
3. 量子核 QSVC 第三条基线与严格等总参数预算对照；`scan_capacity.py` 的 16 qubits 那组（762 s）不必再跑满。
4. 更多 seed（≥20）与多重比较校正，给「不显著」项更紧的置信区间。
5. 噪声模型、有限 `--shots`（多可观测量需分组测量）、`lightning.gpu` 后端。

**三轮已完成**：经典 pre_net + `tanh(·)*π/2` 缩放、两种混合 CNN 前端、
训练 / 数据预算翻倍、冻结量子读出信息探针 —— 全部 5 seed 结果见第 8 节。

**四轮已完成（`experiments/break_readout.py`，见第 9 节）**
- `qml_lab/quantum.py`：`make_rich_layer` / `make_observables` / `n_readout`，
  支持 `z` / `z+x` / `z+x+y+zz` 等规格与 `reupload=True`；在 `"z"` 规格下与旧层逐位一致（回归测试）。
- `qml_lab/models.py`：`RichVQCClassifier`、`HybridRichClassifier`。
- `qml_lab/train.py`：`quantum_lr` 参数分组，把 `.q` 子模块单独设 lr。
- `tests/smoke_test.py`：新增 `check_readout_layers()`（维度 / 梯度 / 逐位一致性 / 单独 lr）。
- 11 组 × 5 seed 原始数据、7 组冻结读出探针、配对 CI、逐 seed 受控断言归档
  `docs/readout_metrics.json`，三张图 `docs/figures/readout_*.png`。
- 结论：加宽读出 +10.80 ± 1.35 pt、探针回收 13.2 pt；单独 lr 与朴素重上传**未达预期**。

## 7. 外部精度基准：参考实现到底能到多少

> 起因：用户问「你参考的只有 60% 多的准确率？」。**0.6640 是本仓库自己跑出的数字，不是抄自任何参考实现。**
> 第 4 节只对照了架构与超参，没有建立外部精度基准，这是审查的遗漏。本节补上（2026-11-06 实检索）。

**先把可比性说清楚**：下表**不能横向排名**。类别数（2 vs 10）、训练样本（50 vs 60000）、
是否有 ImageNet 预训练前端、量子电路是否可训练、经典后处理规模全都不同；同一篇论文里
10 类与 2 类能差 20–30 pt。很多标题写「MNIST」的 QML 论文其实只做 0/1 二分类。

| 参考实现 | 任务口径 | 报告准确率 | 与本仓库的关键差异 | 来源 |
| --- | --- | --- | --- | --- |
| PennyLane *Quantum transfer learning* demo | **非 MNIST**：ants/bees 2 类，3 epochs | `0.8497`（页面打印 `Best test accuracy`，实为验证集最佳） | ResNet18(ImageNet) 冻结提 512 维特征 → `Linear(512,4)` → `tanh*π/2` → 4-qubit QNN → `Linear(4,2)`；**pre_net 与 post_net 都有** | [3] |
| 同论文 Mari et al. 2019，30 epochs | 同上 2 类 | `0.967`（simulator）；真机 ibmqx4 `0.95` / Aspen-4-4 `0.80` | 同上 + 每 10 epoch 降 lr | [16] |
| QCNN, Hur et al. 2021 | MNIST **0/1 二分类**，12665 / 2115 | 约 `99%`（表内最高 98.7%，5 次重复） | PCA / autoencoder 预处理 + 多种编码 + 卷积式量子滤波；**不是 10 类** | [17] |
| Quanvolution 小样本对照（Sooksatra 2021；综述 Table 5） | MNIST **10 类**，50 训练 / 30 测试，30 epochs | CNN `0.6667` / QNN **`0.7000`** | 固定不可训练量子核 + 经典线性/CNN 后处理；**唯一与本仓库同难度档** | [18][19] |
| VSQL, AAAI 2021 | MNIST **10 类**，全量 60000 / 10000 | `91.13%`（**二级来源，原文结果表未抓到，待核**） | 局部 shadow circuits 滑动提取特征 + 经典 FCNN(softmax)，约 928 参数 | [20] |
| MPS 混合变分分类器, ICASSP 2025 | MNIST 10 类 | `95.96%`（二分类 99.95%） | MPS 表示的硬件高效线路 + 最小残差 | [21] |
| 并行量子线路混合模型, Senokosov 2023 | MNIST 10 类，full dataset | `99.21%` | 并行量子线路 + 经典主干，参数量为对应经典模型 1/8 | [22] |
| 经典 CNN | MNIST 10 类 | `>99%` | 公认基线 | — |
| **本仓库（5 seed）** | MNIST 10 类，**600 / 200**，24 epochs | **`0.6640 ± 0.0216`** | **无预训练前端、单次编码、仅 n 个 `PauliZ` 读出、纯端到端 VQC** | `docs/multiseed_metrics.json` |

**三条解读**

1. **「60% 多」是一个真实存在的档位，但只在最难的角落**：极小样本 + 10 类 + 无预训练前端 +
   量子侧承担几乎全部判别。综述里 50 张训练图的 quanvolution 是 `0.6667 / 0.7000`，
   与我们用 600 张图得到的 `0.664` 几乎重合 —— 说明我们**卡在同一档**，而不是「参考项目就这水平」。
2. **别人的 90–99% 不是靠堆 qubits / 参数**：靠的是第 4 节列出的、我们缺的三件东西 ——
   **经典特征前端**（ResNet18 / CAE / CNN）、**经典非线性 post_net**、**多局部读出**
   （VSQL shadow circuits / quanvolution 逐区域测量）。这与第 5 节「深度 8→16 层几乎不动」
   和两条表达力上界一致：瓶颈在信息进出量子层的方式，不在量子层规模。
3. 因此 `0.664` 应读作「**未采用参考实现架构的基线结果**」，不是「QML 在 MNIST 上的水平」。
   把它推到 90%+ 的最小改动是加 CNN/CAE 前端 + `tanh(·)*π/2` 输入缩放 + 已有的 MLP 头，
   即第 6 节待修 #2；这也是下一步实验最值得做的方向。

**未取到 / 待核（不编数字）**

- Qiskit 官方 `TorchConnector` MNIST 教程原链接已 404，本轮未拿到官方确切准确率（第三方同类
  二分类项目 README 写 85–95%，非官方数据，不纳入上表）。参考 [4][10] 的 URL 需换有效地址。
- ICCAD 2021 [7] 的 Table III 在 HTML 中只剩标题与相对提升（`+14.1%` train / `+25.3%` val
  accuracy），绝对数值在表格图片里，未取到。可确认的是它**有经典 CAE/PCA 前端 + 经典全连接后处理**，
  不是纯 QNN。
- VSQL 的 `91.13%` 来自二级摘要；arXiv HTML 的 MNIST 结果表未包含在抓取内容中，
  原文只确认了 10 类用全量数据集、量子 shadow features 后接经典 FCNN。**核对前不作为结论引用。**

## 8. 为什么只有 66%：瓶颈分解（2026-11-06 三轮）

> 起因：用户追问「为什么才达到 66% 的分数」。第 7 节只说明「别人能到 90–99%」，
> 没回答**我们这个配置的分数具体卡在哪一段**。本节用
> `experiments/diagnose_ceiling.py` 把候选瓶颈分别单独加回去，并直接测量
> 量子读出向量里还剩多少可分信息。原始数据归档 `docs/ceiling_metrics.json`，
> 图 `docs/figures/ceiling_ladder.png` 与 `ceiling_curves.png`。

**协议**：真实 MNIST IDX、10 类、**5 seed** `[0..4]`、数据切分固定 `seed=0`、
每个模型构造前 `set_seed`；`base` 与第 5 节主结果同配置（600/200、24 epochs、
`Amplitude-10 / MLP ×5`、742 参数），本次重跑得到 `0.6640 ± 0.0216`，与归档逐 seed 一致，
说明跨脚本可复现。`load_mnist_images` 与 `load_mnist_raw` 的抽样一致性在脚本里
用**像素逐元素**断言（不只是标签顺序）。

九个组：`long` 只把 epochs 24→48；`data` 只把训练样本 600→1200（测试 200→300）；
`{hyb,nq,cnn}_{light,heavy}` 换经典 CNN 前端，`light` 用自适应池化（约 3.3k 主干参数），
`heavy` 用摊平 `Linear(16*7*7,64)`（约 52k）；`hyb` 在前端后接 8-qubit VQC，
`nq` 是**同一前端但跳过量子层**（等架构对照），`cnn` 是纯经典 CNN 直接出 10 类。
另外对 `base / long / hyb_light / hyb_heavy` 冻结量子层，把读出 z 交给 sklearn 的
logreg / RBF-SVC / MLP 探针；原始 784 维像素做同样探针作为输入信息上限。

### 8.1 九组结果（5 seed 均值 ± 标准差）

| 组 | 改动 | 参数 | 最终 acc | 相对 base |
| --- | --- | --- | --- | --- |
| `base` | — | 742 | `0.6640 ± 0.0216` | — |
| `long` | epochs ×2 | 742 | `0.6770 ± 0.0160` | **+1.30 ± 1.89 pt**（CI 跨 0） |
| `data` | 训练样本 ×2 | 742 | `0.7040 ± 0.0104` | **+4.00 ± 1.77 pt**（CI [+1.80,+6.20]） |
| `hyb_light` | 轻量 CNN → 8q VQC → MLP | 4258 | `0.5400 ± 0.2693` | 比 `base` 更低且极不稳定 |
| `nq_light` | 同前端，无 VQC | 4210 | `0.7580 ± 0.0828` | — |
| `cnn_light` | 轻量 CNN 直接分类 | 3658 | **`0.8630 ± 0.0428`** | **+19.90 ± 6.17 pt**（CI [+12.24,+27.56]） |
| `hyb_heavy` | 强 CNN → 8q VQC → MLP | 52674 | `0.9030 ± 0.0205` | **+23.90 ± 3.99 pt**（CI [+18.94,+28.86]） |
| `nq_heavy` | 同前端，无 VQC | 52626 | `0.9210 ± 0.0195` | — |
| `cnn_heavy` | 强 CNN 直接分类 | 52138 | **`0.9340 ± 0.0108`** | — |

配对差值均为同 seed 相减、t 分布 95% CI（n=5，未做多重比较校正）：

| 对比 | 差值 | 95% CI | 判定 |
| --- | --- | --- | --- |
| `long − base` | +1.30 ± 1.89 | [−1.05, +3.65] | 不显著 |
| `data − base` | +4.00 ± 1.77 | [+1.80, +6.20] | 显著但量级很小 |
| `cnn_light − base` | +19.90 ± 6.17 | [+12.24, +27.56] | 显著 |
| `hyb_heavy − base` | +23.90 ± 3.99 | [+18.94, +28.86] | 显著，5/5 为正 |
| `hyb_light − nq_light`（量子层贡献） | −21.80 ± 21.00 | [−47.87, +4.27] | 不显著（方差过大） |
| `hyb_heavy − nq_heavy`（量子层贡献） | −1.80 ± 2.31 | [−4.67, +1.07] | 不显著，4/5 seed 为负或零 |

`hyb_light` 逐 seed `[0.505, 0.595, 0.705, 0.795, 0.100]`：seed 4 的 loss 停在 2.3046
（≈ln10），完全没训动。弱特征 + 8-qubit VQC 的优化不稳定性是本轮新发现的失败模式，
也说明「量子层负贡献」不能简单归因为量子表达力 —— 它有一部分是没有单独调参。

### 8.2 冻结读出探针：66% 就是 z 的上限

| 特征 | 维度 | logreg | RBF-SVC | MLP |
| --- | --- | --- | --- | --- |
| `base` 量子读出 z | 10 | 0.6590 | **0.6790** | 0.6630 |
| `long` 量子读出 z（训练翻倍） | 10 | 0.6700 | 0.6830 | 0.6590 |
| `hyb_light` 量子读出 z | 8 | 0.5830 | 0.5850 | 0.5800 |
| `hyb_heavy` 量子读出 z | 8 | 0.9060 | 0.9150 | 0.9080 |
| **原始 784 维像素**（输入信息上限） | 784 | 0.8350 | **0.8550** | 0.8200 |

这是本节最关键的证据：端到端 `base` 是 `0.6640`，而把它的 10 维读出 z 交给
**任意**经典强分类器最高只有 `0.6790` —— 分类头、优化器、训练预算已经把 z 里的
判别信息榨干了，后面的 `Linear` / MLP 头不是瓶颈。原始像素探针 `0.8550`
与量子 z 探针 `0.6790` 相差 **17.6 pt**，这 17.6 pt 是在
`AmplitudeEmbedding → 酉 → n 个 PauliZ 期望` 这一步丢掉的，与第 4 节的两条
表达力上界（degree-1 傅里叶 / 输入二次型）和有界低维读出一致。
`long` 的 z 探针只从 0.6790 变到 0.6830，从信息角度再次排除「没训够」。
反过来 `hyb_heavy` 的 z 探针 0.9150 说明：**只要经典前端先把信息浓缩成好特征，
量子读出也能保住高分** —— 瓶颈在信息的进出方式，不在「量子」二字。

### 8.3 结论与限制

1. **主因是表示与读出，不是预算**：训练翻倍 +1.3 pt、数据翻倍 +4.0 pt；
   换经典 CNN 特征前端 +19.9 pt（轻量，3.7k 参数）到 +23.9 pt（强前端）。
   `0.664` 应读作「让量子层直接从原始像素端到端判别」这条设计路线的天花板。
2. **量子层在混合结构里没带来收益**：同前端去掉量子层总是更高
   （0.758 > 0.540；0.921 > 0.903；纯 CNN 0.863 / 0.934）。但两个配对差值 CI 都跨 0，
   **不能宣称量子层显著有害**；轻量版还出现 1/5 完全失败，说明有未解决的优化问题。
3. 这解释了第 7 节的外部基准：别人的 90–99% 靠经典前端 + 非线性 post_net +
   多局部读出；我们原实现三件都没有。第 5 节「深度 8→16 层几乎不动」与此处
   「z 探针封顶」是同一个现象的两个测法。

**限制（不夸大）**：`data` 组测试集 300 与 `base` 的 200 不同，绝对数值不完全可比；
`heavy` 前端 52k 参数多于 `MLP-784` 的 25k，`light` 3.7k 也多于量子的 742，
参数预算仍不对等；探针是冻结特征的可分性上界而非端到端训练；量子层未单独调
lr / 初始化 / 深度；`n=5`、未做多重比较校正；仍是状态向量模拟。

**下一步优先级（已在第 9 节执行，结果有正有负）**：① 给量子层单独调参与初始化策略
→ 单独 lr 已试，**未解决** `hyb_light` 的训练崩溃；② 多可观测量 / data re-uploading
→ 已实跑，**前者有效、后者在当前实现下为负**；③ 等参数预算对照与量子核第三条基线
→ 只做到「等分类头预算」，量子核仍未做。

## 9. 加宽读出：把 17.6 pt 的信息损失收回来（2026-11-06 四轮）

> 第 8 节把 `0.664` 定位为「量子读出 z 只有 n 个有界数」。本节直接针对这一条，
> 跑 `experiments/break_readout.py`（11 组 × 5 seed × 24 epochs）。新增两个可复用
> 组件：`qml_lab/quantum.py::make_rich_layer`（可配置 observables / reupload）与
> `qml_lab/train.py::train_classifier(quantum_lr=...)`（量子层参数分组）。
> 原始数据 `docs/readout_metrics.json`，图 `docs/figures/readout_{ladder,info,curves}.png`。

### 9.1 协议与受控性

* 真实 IDX、600/200、10 类；三条数据路径（`load_mnist_raw` 784 维 L2 /
  `load_mnist` PCA-10 或 PCA-40 / `load_mnist_images` 28×28）用同一 seed 与按类配额。
  脚本断言标签序列一致、raw 与 img 的像素逐样本一致（`same_split_asserted: true`）。
* 每个模型**构造前** `set_seed(seed)`；按「seed + 模型族 + qubits」分桶，断言同族
  同 seed 的量子初始权重逐元素相同，**不一致立即抛异常**。结果 **5/5 seed × 8/8 组
  PASS**。混合组先构造 CNN 主干会消耗 RNG，故单独分桶、不与纯 VQC 组比。
* **回归保证**：`make_rich_layer(observables="z")` 与旧 `make_amplitude_layer` 在相同
  seed 下初始权重和前向输出**逐位相同**（已写进 `tests/smoke_test.py`）。本轮
  `z_base` 实测 `0.6640 ± 0.0216`，与第 5、8 节的归档逐 seed 一致 —— 跨脚本可复现。
* 两次独立 `--quick` 运行除耗时外逐字段一致。
* 关键对照设计：`mlp40` 用 PCA-40 接**与 `zfull` 完全相同结构的 MLP 头**（1642 参数），
  用来排除「涨分只是因为分类头第一层从 10 维变 40 维、参数变多」。

### 9.2 十一组结果（5 seed 均值 ± 标准差）

| 组 | 观测量 / 改动 | 读出维 | 参数 | 最终 acc | 冻结读出探针上界 |
| --- | --- | --- | --- | --- | --- |
| `z_base` | Amplitude-10，仅 Z | 10 | 742 | `0.6640 ± 0.0216` | `0.6790` |
| `zx` | + X | 20 | 1062 | `0.7320 ± 0.0244` | `0.7970` |
| `zfull` | + X + Y + 相邻 ZZ | 40 | 1702 | **`0.7720 ± 0.0115`** | **`0.8110`** |
| `a10` | Angle-10，仅 Z | 10 | 742 | `0.6320 ± 0.0385` | `0.6410` |
| `a10_reup` | + 每层重嵌入输入 | 10 | 742 | `0.5220 ± 0.0652` | `0.5570` |
| `a10_full` | 重上传 + 40 维读出 | 40 | 1702 | `0.7330 ± 0.0251` | `0.7380` |
| `hyb_l` | 轻量 CNN → 8q Z，量子层 lr=0.001 | 8 | 4258 | `0.5150 ± 0.1701` | — |
| `hyb_l_rich` | 轻量 CNN → 8q Z+X 重上传 | 16 | 4514 | `0.6930 ± 0.1345` | `0.7340` |
| `hyb_l_nq` | 同前端，跳过量子层 | 8 | 4210 | `0.7580 ± 0.0828` | — |
| `cnn_l` | 纯轻量 CNN 直接分类 | — | 3658 | `0.8630 ± 0.0428` | — |
| `mlp40` | PCA-40 + 同结构 MLP 头 | — | 1642 | `0.8520 ± 0.0104` | — |

原始 784 维像素探针上限 `0.8550`（RBF-SVC）。配对差值（pt，95% CI）：
`zx−z_base` **+6.80 ± 0.84** [+5.76,+7.84]；`zfull−z_base` **+10.80 ± 1.35**
[+9.12,+12.48]；`zfull−zx` **+4.00 ± 1.73** [+1.85,+6.15]；
`a10_reup−a10` **−11.00 ± 8.09** [−21.04,−0.96]；
`a10_full−a10_reup` **+21.10 ± 5.03** [+14.85,+27.35]；
`hyb_l_rich−hyb_l` +17.80 ± 19.64 [−6.58,+42.18]（**CI 跨 0**）；
`hyb_l_rich−hyb_l_nq` −6.50 ± 18.51 [−29.48,+16.48]（**CI 跨 0**）；
`cnn_l−hyb_l_rich` **+17.00 ± 11.53** [+2.69,+31.31]。

### 9.3 结论

1. **第 8 节的诊断被正向验证**：读出 10→20→40 维，端到端 acc 0.664→0.732→0.772，
   冻结探针 0.679→0.797→0.811。上一轮「量子链路丢掉约 17.6 pt 可分信息」中，
   **回收 13.2 pt**，剩余到像素探针（0.855）的 gap 仅 4.4 pt。两条曲线几乎平行、
   端到端始终比探针低 0.02–0.04，说明优化器一直把 z 里的信息榨干 —— 瓶颈确实在
   读出而不是优化，这也再次解释了第 5 节「8→16 层几乎不动」。
2. **不是头参数变多的假象**：三组量子权重都是 60 个，额外参数全在分类头第一层；
   但等分类头预算的 `mlp40` 只有 0.8520，而单把头从 10 维换 40 维不可能凭空产生
   0.679→0.811 的**冻结特征**可分性提升。训练 loss 同步从 0.6785 降到 0.1496。
3. **一次纠正上一轮的优先级判断**：第 8 节把「给量子层单独调 lr」列为最高优先级、
   预期能消除 `hyb_light` 的 1/5 seed 崩溃。实测 `lr=0.001` 后仍是
   `0.5150 ± 0.1701`（逐 seed 0.305/0.610/0.570/0.715/0.375），**没有解决**。
   加宽读出把均值抬到 0.6930 但方差仍 0.1345，配对差值 CI 跨 0、不显著；
   同前端跳过量子层 0.7580、纯 CNN 0.8630。**量子层在混合结构里仍无可证实的正贡献**。
4. **朴素重上传是负结果，但不能外推到文献 [12]**：`a10_reup` 比 `a10` 低
   11.00 ± 8.09 pt，且冻结探针 0.6410→0.5570（信息本身变少，不只是优化变差）。
   本实现是「`StronglyEntanglingLayers` 每层切片 + 每层全量 `AngleEmbedding` 重置」，
   与 Pérez-Salinas 等用单 qubit 通用 `Rot` 逐层嵌入的原始设计不同，也没为它单独
   调 lr / 层数 / 输入缩放（参考实现常用 `tanh(·)*π/2`）。**结论只覆盖本实现**。
   给它换成 40 维读出后回到 0.7330，说明收益仍主要来自加宽读出。

### 9.4 限制（不夸大）

* **真机代价被隐藏**：解析模拟器里多观测量几乎免费（batch fwd+bwd 0.087→0.115 s），
  但 X/Y/Z/ZZ 互不対易，真机必须分测量组多次采样，shots 成本近似随观测量线性上升；
  振幅编码另有 O(2ⁿ) 状态制备问题。本结果**不能**当作硬件可行性证据。
* `n=5`、未做多重比较校正；8 个配对检验里出现 3 个 CI 跨 0 的不确定项。
* 探针是冻结特征的可分性上界，不是端到端训练结果；sklearn MLP 探针有
  `ConvergenceWarning`（500 iter 未收敛），其数值宜读作保守下界。
* `Angle10`（PCA-10）与 `Amp10`（全 784 维）输入表示不同，只做组内配对；
  `mlp40` 是「等分类头」而非严格等总参数预算（1702 vs 1642）。
* 混合组方差极大（std 0.13–0.17），所有涉及它的配对结论都只是不确定观察，
  不能宣称「量子层有害」。

### 9.5 剩余待办（按性价比）

① 量子层在混合架构里的初始化 / 深度 / lr 系统调参（单独 lr 这一项已排除）；
② 按 [12][13] 原始设计重做 data re-uploading；
③ 量子核 QSVC 第三条基线与严格等总参数预算对照；
④ ≥20 seed + 多重比较校正，给第 5 节那些「不显著」项更紧的区间；
⑤ 有限 `--shots` / 噪声模型下重测多可观测量（这是真机代价的直接检验）。

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
16. Mari et al., *Transfer learning in hybrid classical-quantum neural networks*, Quantum 4 340（`0.967` / QPU `0.95`、`0.80`）：https://arxiv.org/abs/1912.08278
17. Hur et al., *Quantum convolutional neural network for classical data classification*（MNIST 0/1，约 99%）：https://arxiv.org/abs/2108.00661
18. Sooksatra et al., *Evaluating Accuracy and Adversarial Robustness of Quanvolutional Neural Networks*（50 训练图）：https://www.rivas.ai/pdfs/sooksatra2021evaluating.pdf
19. *Quantum Machine Learning: A Review and Case Studies*, Table 5（CNN 0.6667 / QNN 0.7000）：https://pmc.ncbi.nlm.nih.gov/articles/PMC9955545/
20. Li, Song, Wang, *VSQL: Variational Shadow Quantum Learning for Classification*, AAAI 2021：https://arxiv.org/abs/2012.08288
21. Sun, Xu, Duan, *An Efficient Hybrid Quantum Variational Classifier With Matrix Product State*, ICASSP 2025（10 类 95.96%）：https://doi.org/10.1109/ICASSP49660.2025.10888238
22. Senokosov et al., *Quantum machine learning for image classification*（10 类 99.21%）：https://arxiv.org/abs/2304.09224

> 注：参考 [4] 与 [10] 的 Qiskit 教程链接在 2026-11-06 复核时已 404，第 7 节据此未引用其准确率数字。
