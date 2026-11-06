# 量子手写数字识别 · Quantum MNIST

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![PennyLane](https://img.shields.io/badge/PennyLane-0.45-9b59b6)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![License](https://img.shields.io/badge/license-MIT-green)
![Hardware](https://img.shields.io/badge/quantum%20hardware-not%20required-success)

> 用参数化量子电路（VQC）做 MNIST 手写数字分类，并与同输入维度的经典 MLP 放在同一套训练配置下对照。**可复现、纯模拟、不接真机。**

本仓库只做一件事：量子机器学习版本的 MNIST 分类。姊妹仓库 [quantum-llm](https://github.com/quantum-machine-learning-code/quantum-llm) 做玩具级量子增强语言模型，两者代码各自自包含，可独立 clone 与运行。

## 快速结论（2026-11-06 多 seed 重跑）

真实 MNIST IDX，600 训练 / 200 测试、10 类，**5 个 seed × 24 epochs**，报告均值 ± 标准差。Python 3.13 / PennyLane 0.45.1 / torch 2.15 CPU。原始数据在 [`docs/multiseed_metrics.json`](docs/multiseed_metrics.json)，复现命令见[快速开始](#快速开始)。

| 模型 | 参数量 | 测试准确率（5 seed） | 单 seed 耗时 |
| --- | --- | --- | --- |
| 原基线：Angle-4 + Linear 头 | 74 | 0.4320 ± 0.0375 | 16.1 s |
| **最佳量子配置：Amplitude-10 + MLP 头 ×5** | 742 | **0.6640 ± 0.0216** | 48.2 s |
| 经典 MLP（同样 784 维输入） | 25450 | **0.8640 ± 0.0065** | 0.6 s |

- 改良后的量子模型相对原基线 **+23.20 ± 5.40 个百分点**（95% CI [+16.49, +29.91]，5 个 seed 全部为正）。这是输入表示、分类头、输出缩放、训练预算与量子参数量的**联合效果**，不是单一改动的功劳。
- **经典 MLP 仍领先约 20 pt、快约 80 倍**；参数预算也不对等（742 vs 25450），这不是公平的能力比较，更不是量子优势。
- 旧的「振幅编码比角度编码好 +15.5 pt」**已证伪**：严格受控（同 10 qubits / 同 170 参数 / 同分类头 / 初始权重逐元素相同）下，5 seed 配对差值 **−0.10 ± 8.68 pt**，95% CI [−10.88, +10.68] 跨 0，逐 seed 符号相反。
- **0.664 不是该领域的水平，也不是抄来的数字**：外部参考实现里，同样「小样本 10 类」的 quanvolution 对照报 0.6667 / 0.7000（50 张训练图），而带经典预训练前端 / 多局部读出的混合模型在 MNIST 10 类上报 91–99%，0/1 二分类普遍 95–99%。本仓库是**无预训练前端、单次编码、仅 n 个 `PauliZ` 读出**的纯端到端 VQC，落在这个子类别的常见区间，但明显低于参考实现的最好结果。完整对照与可比性说明见 [REVIEW.md 第 7 节](docs/REVIEW.md#7-外部精度基准参考实现到底能到多少)。
- **66% 的原因已实测定位**：冻结量子层后，10 维 `PauliZ` 读出的可分性上限只有 0.679（原始像素 0.855）；训练 / 数据预算翻倍只值 +1.3 / +4.0 pt，而换经典 CNN 特征前端能到 0.863–0.934。见下方「为什么只有 66%：瓶颈分解」一节。

> **⚠ 方法学审查（2026-11-06）**：本仓库最初未对照任何参考实现，且 `train_amplitude.py` 有种子 bug（模型构造早于 `set_seed`），使早期单次运行的结论不可用。现已修复并用 `train_multiseed.py` 重跑；下文[历史结果](#历史结果单次运行仅存档)只作存档，不用于下结论。完整问题清单、实测证据与 15 条参考链接见 [docs/REVIEW.md](docs/REVIEW.md)。

## 能做什么，不能做什么

- **默认配置是模拟预算，不是物理限制**：`default.qubit` 状态维度 2ⁿ。本机实测单 batch 前向+反向：4/8/12/16 qubits 约 0.06–1.2 s，20 qubits 就要 **63 s**。默认 `qubits=4` 只是为了让笔记本几分钟跑完一个 epoch。`--qubits` 与 `--layers` 都能往上调，量子参数量为 `layers × qubits × 3`（4 qubits × 2 层 = 24 只是这一个配置的数字，不存在「最多 24 个」的限制）。
- **压到 4 维是编码方式造成的，不是量子机器学习的要求**：`AngleEmbedding` 用一个 qubit 装一个旋转角，所以特征维必须等于 qubits 数。换成 `AmplitudeEmbedding`，n 个 qubits 的态空间可容纳 2ⁿ 个振幅 —— **10 个 qubits（1024 ≥ 784）就能无损装下完整图像，完全不用 PCA**。可跑对照见 [`experiments/train_amplitude.py`](experiments/train_amplitude.py)；`experiments/scan_capacity.py` 则把「输入维度」与「电路深度」两个因素分开扫。
- **但放大不是免费的**：qubits 与层数增加会放大 barren plateau（随机初始化下梯度方差随 n 指数衰减）、状态制备成本与 2ⁿ 模拟内存；读出端只取 n 个 `PauliZ` 期望，本身也是信息瓶颈。而且**纠缠越强，经典模拟越难** —— 张量网络只在低纠缠时高效。所以「假设有很多强关联 qubits」恰恰是量子优势最可能出现、也最无法在笔记本上验证的区域。
- **不宣称量子优势**：在这个规模下量子模型打平甚至落后于经典基线是常态。当前主结果已做 5 seed 重复与配对统计，但仍缺等参数预算对照；任何「更好」的结论还需要第三条基线与更严格的统计检验。
- **不需要真机凭证**：纯模拟无需 IBM Quantum 等账号；只有切到真实量子硬件时才需要 token。

## 目录结构

```
quantum-mnist/
├── README.md
├── LICENSE                      # MIT
├── requirements.txt
├── qml_lab/
│   ├── utils.py                 # 随机种子、路径、JSON、参数量、终端表格
│   ├── data.py                  # load_mnist（PCA）/ load_mnist_raw（784 维）/ load_mnist_images（28×28 图像）
│   ├── quantum.py               # AngleEmbedding / AmplitudeEmbedding 两种可微分层
│   ├── models.py                # VQC / AmplitudeVQC / MLP / HybridCNN / CNN 五类模型
│   ├── train.py                 # 分类训练循环与分批准确率评估
│   └── viz.py                   # 指标图绘制
├── experiments/train_mnist.py   # 主实验：VQC vs MLP（历史单次运行）
├── experiments/train_amplitude.py  # 编码对照：振幅编码无需 PCA（历史单次运行）
├── experiments/scan_capacity.py    # 容量扫描：输入维度 vs 电路深度（历史单次运行）
├── experiments/train_multiseed.py  # **当前主结果**：5 seed 受控对照 + 统计
├── experiments/diagnose_ceiling.py # 瓶颈分解：预算 / 架构 / 量子读出信息探针
├── tests/smoke_test.py          # 离线冒烟测试（不联网、不下载 MNIST）
├── docs/DEV_NOTES.md            # 本机环境与离线数据准备备注
├── docs/REVIEW.md               # 方法学审查：问题清单、实测证据、参考实现对照
├── docs/multiseed_metrics.json  # 归档的 5 seed 原始指标（可核对 README 数字）
├── docs/ceiling_metrics.json    # 归档的瓶颈分解原始指标（9 组 × 5 seed + 探针）
├── docs/figures/                # 实测对比图（含 multiseed_* / ceiling_*）
├── data/                        # 本地数据缓存（gitignore）
└── results/                     # 运行产物（gitignore）
```

## 安装

要求 Python 3.11+，纯 CPU 即可运行默认配置。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Windows 下把 `.venv/bin/python` 换成 `.venv\Scripts\python.exe`、路径改用反斜杠。

`torchvision` 用于自动下载真实 MNIST，与 `torch` 版本强绑定（已验证 `torch 2.15.0 + torchvision 0.29.2`）。**更稳的做法**是把 MNIST 的 4 个 IDX 文件放进 `data/MNIST/raw/`，用 `--data-source idx` 完全离线运行 —— 详见 [`docs/DEV_NOTES.md`](docs/DEV_NOTES.md)。

## 快速开始

```bash
.venv/bin/python tests/smoke_test.py
.venv/bin/python experiments/train_mnist.py --quick
```

`--quick` 把样本量、迭代数压到最小并强制使用 `synthetic` 数据（不联网），仅用于验证流程跑通，**不用于报告结论**；合成数据会在输出 JSON 的 `meta.note` 中明确标注。

**当前主结果**用下面这条重跑（5 seed × 10 组 × 24 epochs，纯 CPU 约 25 分钟）：

```bash
.venv/bin/python experiments/train_multiseed.py --data-source idx \
    --seeds 0,1,2,3,4 --epochs 24 --train-samples 600 --test-samples 200 \
    --out results --fig-dir docs/figures
```

产物：`results/multiseed_metrics.json`（自动归档到 `docs/multiseed_metrics.json`）、`docs/figures/multiseed_accuracy.png` 与 `multiseed_bars.png`。加 `--quick` 是离线极小自检，不用于结论。

**瓶颈分解实验**（9 组 × 5 seed，纯 CPU 约 30 分钟，回答「为什么只有 66%」）：

```bash
.venv/bin/python experiments/diagnose_ceiling.py --data-source idx \
    --out results --archive docs/ceiling_metrics.json --fig-dir docs/figures
```

产物：`docs/ceiling_metrics.json`（逐 seed 曲线、分组统计、读出探针）、`docs/figures/ceiling_ladder.png` 与 `ceiling_curves.png`。`--quick` 为离线自检。

历史主实验（单次运行，仅存档）的复现命令：

```bash
.venv/bin/python experiments/train_mnist.py --data-source idx \
    --qubits 4 --layers 2 --classes 10 --train-samples 1200 --test-samples 300 \
    --epochs 12 --batch-size 32 --lr 0.01 --seed 0
```

需要 `data/MNIST/raw/` 下放好 4 个 IDX 文件；没有就去掉 `--data-source idx`，让 `auto` 走在线下载回退链。全部产物写入 `results/`。

两个补充实验同样用离线 IDX 数据：

```bash
# 编码对照：10 qubits 的振幅编码吃下完整 784 维，不压 PCA
.venv/bin/python experiments/train_amplitude.py --data-source idx --out results

# 容量扫描：把输入维度与电路深度两个因素分开扫（默认含 16 qubits，耗时较长）
.venv/bin/python experiments/scan_capacity.py --data-source idx --out results
```

两者都支持 `--quick`（极小规模离线自检）。`--qubits 16` 以上的扫描在纯 CPU 上会明显变慢，可自行缩减 `--dims` / `--layer-sweep` 列表。

## 实验设计

- **主实验数据流**：MNIST 展平 → `PCA` 降到 `n_qubits` 维 → 缩放到 `[-π, π]` → 量子电路 → 测量 → `Linear` 分类头。
- **编码（两条路径）**：`AngleEmbedding(RY)` 一个 qubit 装一个特征，故 `特征维 = n_qubits`，必须降维；`AmplitudeEmbedding` 把输入当作 2ⁿ 维态的振幅，`2^n ≥ 784` 即可（n=10），**不降维**。后者见 `train_amplitude.py`。
- **Ansatz**：`StronglyEntanglingLayers`，每层每 qubit 3 个旋转角，量子参数量 = `layers × qubits × 3`。4 qubits × 2 层 = 24，10 qubits × 2 层 = 60，16 qubits × 16 层 = 768 —— 参数量完全可以放大，只是要付 2ⁿ 模拟成本。
- **测量**：对每个 qubit 取 `PauliZ` 期望，得到长度为 `n_qubits` 的实向量。这是另一处信息瓶颈：无论输入是 4 维还是 784 维，读出都只有 n 个数（可改用多可观测量或数据重上传缓解）。
- **分类头与输出缩放（审查后新增）**：量子读数恒在 `[-1,1]`，裸接 `Linear` + 交叉熵会压制梯度。`VQCClassifier(..., head="mlp", scale=5.0)` 可补上经典非线性与固定缩放，`scale="learnable"` 把系数本身变成可训练参数。默认 `head="linear", scale=1.0` 保持旧行为；5 seed 实测见[当前主结果](#当前主结果5-seed-受控对照2026-11-06-重跑)。
- **数据源回退**：`idx`（直接读本地 4 个 IDX 文件，完全离线）→ `torchvision` → `sklearn.fetch_openml` → `synthetic`。
- **Baseline**：同输入维度、同训练配置的经典 MLP；`--skip-baseline` 可跳过。
- **指标**：准确率曲线、参数量、训练耗时，写入 `results/mnist_metrics.json` 并出三张图。
- **采样与求导**：`--shots 0` 为解析模拟（默认）；正整数启用有限采样并自动切换到 `parameter-shift`；`--diff-method` 可显式指定。

## 参数速查

> 下表是**历史主实验 `train_mnist.py`** 的参数；当前主结果用的 `train_multiseed.py`
> 另有 `--seeds` / `--report-epoch` / `--archive` 等，默认值以其 `argparse` 为准。
> 若与脚本实现不一致，以脚本内 `argparse` 默认值为准。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--qubits` | `4` | qubit 数，也等于 PCA 降维后的特征维 |
| `--layers` | `2` | `StronglyEntanglingLayers` 层数 |
| `--classes` | `10` | 分类类别数 |
| `--train-samples` | `2000` | 训练样本数 |
| `--test-samples` | `500` | 测试样本数 |
| `--epochs` | `20` | 训练轮数 |
| `--batch-size` | `32` | 批大小 |
| `--lr` | `0.01` | 学习率 |
| `--seed` | `0` | 随机种子 |
| `--data-source` | `auto` | `idx → torchvision → openml → synthetic` 回退 |
| `--data-dir` | `data` | 本地数据 / 缓存目录 |
| `--diff-method` | `backprop` | PennyLane 求导方式（解析模拟下） |
| `--shots` | `0` | `0` 解析模拟；正整数为有限采样 |
| `--out` | `results` | 指标与图片输出目录 |
| `--quick` / `--skip-baseline` | 关闭 | 极小规模验证 / 跳过经典基线 |

## 实测结果与复现

### 当前主结果：5 seed 受控对照（2026-11-06 重跑）

配置：本地 IDX 真实 MNIST，600 训练 / 200 测试、10 类、24 epochs、batch 32、lr 0.01、解析模拟 `backprop`。**数据切分固定 `seed=0`**，5 个 seed 只改变模型初始化与 mini-batch shuffle；每次构造模型前重置种子，脚本断言同 qubits 各组初始权重逐元素相同（5/5 PASS），并额外断言 `Angle-10` 与 `Amplitude-10` 的分类头相同。完整原始数据（每个 seed 的曲线、受控断言、配对差值与 CI）：[`docs/multiseed_metrics.json`](docs/multiseed_metrics.json)。

| 模型 | 输入 / 编码 | 参数 | acc@8 | 最终 acc@24 | 单 seed 耗时 |
| --- | --- | --- | --- | --- | --- |
| Angle-4 / Linear（原基线） | PCA→4 | 74 | 0.3470 ± 0.0431 | 0.4320 ± 0.0375 | 16.1 s |
| Angle-4 / MLP | PCA→4 | 514 | 0.4770 ± 0.0517 | 0.4980 ± 0.0289 | 14.3 s |
| Angle-4 / MLP ×5 | PCA→4 | 514 | 0.4920 ± 0.0340 | 0.4790 ± 0.0108 | 14.4 s |
| Angle-10 / Linear | PCA→10 | 170 | 0.5060 ± 0.0263 | 0.5710 ± 0.0321 | 55.7 s |
| Amplitude-10 / Linear | 784（无压缩） | 170 | 0.4800 ± 0.0398 | 0.5700 ± 0.0583 | 48.0 s |
| Amplitude-10 / MLP | 784 | 742 | 0.5110 ± 0.0397 | 0.6320 ± 0.0305 | 49.0 s |
| **Amplitude-10 / MLP ×5** | 784 | 742 | **0.6180 ± 0.0256** | **0.6640 ± 0.0216** | 48.2 s |
| Amplitude-10 / MLP + 可学习缩放 | 784 | 743 | 0.5750 ± 0.0352 | 0.6620 ± 0.0399 | 48.0 s |
| MLP-4（经典，同 PCA-4 输入） | 4 | 490 | 0.5800 ± 0.0221 | 0.5880 ± 0.0168 | 0.5 s |
| MLP-784（经典，全维） | 784 | 25450 | 0.8550 ± 0.0100 | 0.8640 ± 0.0065 | 0.6 s |

![多 seed 测试准确率曲线（均值 ± 1 标准差）](docs/figures/multiseed_accuracy.png)

![最终准确率柱状图（含误差棒）](docs/figures/multiseed_bars.png)

**怎么读这张表**：

1. **提升是真的，但不能归因给「量子变强」**：最佳配置相对原基线 +23.20 ± 5.40 pt（95% CI [+16.49, +29.91]，5 seed 全为正，逐 seed `+21 / +18 / +19 / +29 / +29`）。这是换输入表示、换分类头、加输出缩放、把训练从 8 拉到 24 epochs、量子参数从 24 增到 60 的**联合效果**。
2. **「不压缩」本身没有可测收益**：`Angle-10 / Linear` 与 `Amplitude-10 / Linear` 是严格受控对照（同 10 qubits、同 170 参数、同 `Linear` 头、初始权重逐元素相同、同一批图），5 seed 配对差值 **−0.10 ± 8.68 pt，CI [−10.88, +10.68]**，逐 seed `+12.5 / +1.5 / +2.0 / −7.0 / −9.5`。旧 README 的 +15.5 pt 是初始化运气。
3. **可分离的单项收益比早期诊断小得多**：MLP 头相对 Linear 头在 Angle-4 上 +6.60 pt（CI [+0.43, +12.77]，勉强显著）、在 Amplitude-10 上 +6.20 pt（CI 跨 0）；固定 ×5 缩放相对各自「仅 MLP 头」组为 −1.90 pt（Angle-4）/ +3.20 pt（Amplitude-10），可学习缩放相对固定 ×5 为 −0.20 pt，**均不显著**，可学习缩放与固定 ×5 实际等价。
4. **仍然没追上经典**：同 PCA-4 输入下 `MLP-4` 0.588 > 最佳 Angle-4 0.498；全维 `MLP-784` 0.864 > 最佳量子 0.664。这与「单次角度编码是 degree-1 傅里叶 / 振幅编码是输入二次型」的表达力上界一致，堆 qubits 补不上。
5. **口径限制**：固定数据集让方差只反映初始化与 shuffle；`n=5` 的 95% CI 用 t 分布、未做多重比较校正；参数预算不对等；仍是状态向量模拟，振幅编码在真机上需要 O(2ⁿ) 状态制备或 QRAM。

### 为什么只有 66%：瓶颈分解（2026-11-06 三轮实验）

`experiments/diagnose_ceiling.py`：真实 MNIST IDX、10 类、**5 seed**、数据切分固定 `seed=0`、每个模型构造前设种子。把训练预算、数据预算、架构三个候选瓶颈**分别单独加回去**，再加一个「冻结量子读出」的信息探针。原始数据（9 组逐 seed 曲线、探针、配对差值）：[`docs/ceiling_metrics.json`](docs/ceiling_metrics.json)。

| 组 | 相对 base 的改动 | 参数 | 最终准确率 |
| --- | --- | --- | --- |
| base（Amp10 / MLP ×5） | — | 742 | 0.6640 ± 0.0216 |
| 训练预算 ×2 | 只把 24 → 48 epochs | 742 | 0.6770 ± 0.0160 |
| 数据预算 ×2 | 只把训练样本 600 → 1200 | 742 | 0.7040 ± 0.0104 |
| 轻量 CNN，直接分类 | 换架构（自适应池化前端） | 3658 | **0.8630 ± 0.0428** |
| 轻量 CNN + VQC | 同前端，但接 8-qubit 量子头 | 4258 | 0.5400 ± 0.2693 |
| 强 CNN 前端 + VQC | 52k 参数前端 + 8-qubit 量子头 | 52674 | 0.9030 ± 0.0205 |
| 强 CNN，直接分类 | 同前端，无量子层 | 52138 | **0.9340 ± 0.0108** |

![九组配置的准确率阶梯（均值 ± 1 标准差）](docs/figures/ceiling_ladder.png)

![训练与数据预算的收敛曲线](docs/figures/ceiling_curves.png)

**结论：66% 几乎就是「量子读出里还剩多少信息」，不是没训够。**

1. **预算不是主因**：训练翻倍 **+1.30 ± 1.89 pt**（CI 跨 0）、数据翻倍 **+4.00 ± 1.77 pt**（CI [+1.80, +6.20]）。曲线约 10 epoch 就进平台；48 epoch 的 loss 从 0.68 降到 0.50，准确率却几乎不动——模型是在对已经丢失的信息继续拟合。
2. **读出是硬瓶颈**：冻结训练好的量子层，把 10 维 `PauliZ` 向量 z 交给经典强分类器（logreg / RBF-SVC / MLP），最高只有 **0.6790**，而端到端模型是 **0.6640** —— 分类头与优化器已经把 z 里的信息榨干，换更强的后处理也上不去。同样探针直接读原始 784 维像素可得 **0.8550**，量子链路丢掉了约 **17.6 pt** 的可分信息。`long` 组的 z 探针 0.6830 几乎没变，再次排除「训练不足」。
3. **换表示能补回来**：同样 600 张图 / 24 epochs，让经典 CNN 先提特征，轻量前端（3.7k 参数）到 **0.8630**（配对 +19.90 ± 6.17 pt，CI [+12.24, +27.56]），强前端到 0.9030（+23.90 ± 3.99 pt）。所以 **66% 是「让量子层直接端到端判别」这条设计路线的天花板，不是 QML 的上限**。
4. **但量子层在混合结构里也没有正贡献**：同一前端下，去掉量子层反而更高（轻量 0.758 vs 0.540；强前端 0.921 vs 0.903）。强前端配对差值 −1.80 ± 2.31 pt，4/5 seed 为负但 **CI 跨 0，不能宣称显著有害**；轻量前端有 1/5 seed 完全训练失败（acc 0.10、loss 2.30 ≈ ln10），标准差 0.269，说明「弱特征 + 8-qubit VQC」优化极不稳定。

**限制**：`data` 组测试集为 300（base 为 200），绝对数值不完全可比；强前端 52k 参数多于 `MLP-784` 的 25k，参数预算仍不对等；量子层未单独调 lr / 初始化 / 深度，负贡献可能部分来自优化；探针是冻结特征的可分性上界、不是端到端训练；n=5、未做多重比较校正。完整协议与配对 CI 见 [docs/REVIEW.md](docs/REVIEW.md) 第 8 节。

### 历史结果（单次运行，仅存档）

> 以下三节是 2026-11-06 审查**之前**的 `seed=0` 单次实验，其中编码对照还受种子 bug 影响。数字保留供追溯，**不得用于下结论**；所有结论性表述请看上一节与 [REVIEW.md](docs/REVIEW.md)。

配置：本地 IDX 真实 MNIST、1200 训练 / 300 测试、PCA→4 维、4 qubits / 2 层、12 epochs、batch 32、lr 0.01、解析模拟 `backprop`。

![VQC 与 MLP 的测试准确率曲线](docs/figures/mnist_accuracy.png)

参数量与耗时对比见 [`docs/figures/mnist_params.png`](docs/figures/mnist_params.png) 与 [`docs/figures/mnist_time.png`](docs/figures/mnist_time.png)。

### 怎么读这些数字（历史单次实验）

- 量子版本**没赢过经典版本**，且慢约 42 倍。这反映的是**当前规模下量子模拟的性价比**，不能推广成「量子机器学习不行」。
- 低准确率来自几个**可以分别调整**的因素：角度编码强制把 784 维压到 4 维、电路只有 24 个量子参数、读出只有 4 个 `PauliZ` 期望、样本与 epoch 预算都很小。**审查后补充的一条**：量子输出恒在 `[-1,1]`，却裸接 `Linear` 头再喂交叉熵，logits 幅度太小导致梯度被压制（[EAAI 2026, arXiv:2606.22551](https://arxiv.org/abs/2606.22551)）。早期单次诊断里换成 MLP 头 + 缩放让 Amplitude-10 的 loss 从 1.998 降到 1.050、准确率 +8 pt，但那次**同时改了初始化、头、缩放三个变量**；5 seed 重跑后可分离的单项收益只有 +6.20 ± 5.44 pt（CI 跨 0）与 +3.20 ± 3.78 pt（不显著），见[当前主结果](#当前主结果5-seed-受控对照2026-11-06-重跑)。
- 但**表达力上界是真实存在的**：单次角度编码是 degree-1 三角多项式，振幅编码 + 酉 + PauliZ 等价于输入的二次型（Schuld 2021）。这部分差距堆参数补不上，需要 data re-uploading / 多可观测量 / 混合 CNN 前端。
- 多 `--seed` 重复与配对统计**已完成**（`train_multiseed.py`）。要公平比较还缺：相同参数预算、更好的编码、量子核方法等第三条基线。
- 准确的表述是：**在这组配置下，量子电路可被纳入 PyTorch 训练循环并正常收敛，但在这台机器上的精度与耗时都不占优。**

### 编码方式与「4 维限制」是不是必需

复现：`python experiments/train_amplitude.py --data-source idx`。真实 MNIST、600 训练 / 200 测试、8 epochs、seed=0、单次运行。

| 模型 | 输入维 | qubits | 量子参数 | 总参数 | 测试准确率 | 耗时 |
| --- | --- | --- | --- | --- | --- | --- |
| Angle-4 | 4（PCA） | 4 | 24 | 74 | 0.4200 | 17.6 s |
| Angle-10 | 10（PCA） | 10 | 60 | 170 | 0.3750 | 71.8 s |
| **Amplitude-10** | **784（无压缩）** | **10** | **60** | **170** | **0.5300** | 58.1 s |
| MLP-784（经典参照） | 784 | — | 0 | 25450 | 0.8450 | 0.8 s |

![不同编码方式的准确率曲线](docs/figures/amplitude_accuracy.png)

> **⚠ 该结论已撤回（2026-11-06）**：这一组实验的初始权重没有被控制 —— `train_amplitude.py` 原先在模型构造之后才 `set_seed`，四个模型的量子权重各不相同，且独立进程之间结果不可复现。把 `Angle-10` 与 `Amplitude-10` 的初始权重与分类头完全固定后（`results/diag_controlled.py`），两组不同初始权重给出 **−3.0 pt** 与 **+12.5 pt**，符号相反。因此上表的 0.3750 → 0.5300（+15.5 pt）**不能作为编码方式优劣的证据**，它主要反映初始化与单次运行的噪声。代码已修；**5 seed 重跑确认编码方式无显著收益**（−0.10 ± 8.68 pt，CI [−10.88, +10.68]），见[当前主结果](#当前主结果5-seed-受控对照2026-11-06-重跑)。详见 [REVIEW.md](docs/REVIEW.md)。

**原本声称的对照**：`Angle-10` 与 `Amplitude-10` 的 qubits 数、ansatz、量子参数量（60）和训练配置相同，编码方式与是否需要压到 10 维不同 —— 但实际上初始权重也不同，所以并非受控对照。「压维是编码方式造成的、可以解除」这一点仍然成立（振幅编码确实用 2^10 ≥ 784 装下完整图像），但**5 seed 受控重跑显示它不带来显著的准确率收益**（−0.10 ± 8.68 pt，CI 跨 0）。

经典 MLP 在同样的 784 维上到 0.8450，量子侧在电路设计、读出与优化上仍有明显差距，这一差距量级不太可能被噪声解释。但「`Angle-10` 比 `Angle-4` 差 = barren plateau」这个归因**证据不足**：两组初始权重未受控，差值完全可能来自初始化运气。`scan_capacity.py` 的深度饱和趋势（8→16 层几乎不动）才是更可靠的证据，因为该脚本在模型构造前就重置了种子。

**必须说清的代价**：状态向量模拟器里写入振幅初态是 O(1) 的，真实硬件上需要 O(2ⁿ) 深度的状态制备电路或尚不存在的 QRAM；振幅编码是线性映射，非线性只来自变分电路，读出仍只有 n 个 `PauliZ` 期望。这个实验证明的是「编码方式能解除压缩瓶颈」，**不是**「真机上能便宜地跑大图像」。耗时也与主实验不同配置、并发运行，仅供量级参考。

### 容量扫描：放大 qubits 和参数量到底有没有用

复现：`python experiments/scan_capacity.py --data-source idx`。真实 MNIST、600 训练 / 200 测试、8 epochs、seed=0、单次运行；固定其余超参，一次只扫一个因素。

**A. 输入维度（= qubits 数，固定 2 层）**

| n_features / qubits | 量子参数 | VQC 准确率 | MLP 准确率 | VQC 耗时 |
| --- | --- | --- | --- | --- |
| 4 | 24 | 0.3500 | 0.6050 | 11.3 s |
| 8 | 48 | 0.4000 | 0.6850 | 23.2 s |
| 12 | 72 | 0.3950 | 0.7250 | 73.7 s |
| 16 | 96 | 0.4400 | 0.7700 | 761.7 s |

**B. 电路深度（固定 4 qubits）**

| layers | 量子参数 | VQC 准确率 | MLP 准确率 | VQC 耗时 |
| --- | --- | --- | --- | --- |
| 2 | 24 | 0.3500 | 0.6050 | 10.6 s |
| 4 | 48 | 0.4550 | 0.5650 | 17.1 s |
| 8 | 96 | 0.4700 | 0.5650 | 30.1 s |
| 16 | 192 | 0.4800 | 0.5900 | 60.0 s |

![容量扫描：准确率随输入维度和电路深度的变化](docs/figures/capacity_scan.png)

- **参数量确实可以放大**：24 → 192 没有任何代码或物理限制，只是 `--qubits` / `--layers` 的事；量子参数量恒为 `layers × qubits × 3`。
- **但收益递减且很快饱和**：深度 2→16 层（参数 ×8）只带来 +13 个百分点，8→16 层几乎不动。
- **输入信息确实在增加，VQC 却没吃满**：MLP 随维度从 0.605 涨到 0.770，VQC 只从 0.350 涨到 0.440，差距反而拉大。所以限制不只在「压维」，还在量子侧的读出（只有 n 个 `PauliZ` 期望）与优化。这也说明原先把低准确率单一归因于 PCA 是不完整的。
- **2ⁿ 是真墙**：16 qubits 比 4 qubits 慢约 **67 倍**（11 s → 762 s），20 qubits 在本机单 batch 就要 63 s。
- ~~「最有效的是换表示，+15.5 pt」~~ **该结论已撤回**：5 seed 受控对照显示编码方式差值为 −0.10 ± 8.68 pt、CI 跨 0（见[当前主结果](#当前主结果5-seed-受控对照2026-11-06-重跑)）。「堆参数收益递减」这一趋势本身仍由本表支持，但不能再拿编码对照当作「有效改动」的例子。

**数据口径提醒**：以上两表均为 `seed=0` 单次运行、无方差；`train_amplitude.py` 当时还有种子 bug，**绝对数值与结论均不可采用**。`scan_capacity.py` 在模型构造前设种子，组间趋势受控但同样没有方差。同一 Angle-4 配置在三个脚本里分别得到 0.330 / 0.350 / 0.420，波动约 9 pt，与多数「结论级」差异同量级 —— 这正是后来改用 5 seed 的原因。不同脚本配置不同，绝对数值不可跨表逐位比较；**结论性数字已被 `train_multiseed.py` 的重跑结果替换**。详见 [REVIEW.md](docs/REVIEW.md)。

### 运行产物

`results/` 与 `data/` 被 `.gitignore` 忽略，以下均为运行后本地生成；`docs/figures/` 归档了实测对比图。为让 GitHub 上的数字可核对，**5 seed 的原始指标已归档入库**：[`docs/multiseed_metrics.json`](docs/multiseed_metrics.json)。

| 文件 | 内容 |
| --- | --- |
| `docs/multiseed_metrics.json` | **归档的 5 seed 原始数据**：每个 seed 的曲线 / loss / 耗时、受控性断言、汇总统计、配对差值与 95% CI |
| `results/multiseed_metrics.json` | 同上（运行产物，被忽略） |
| `docs/figures/multiseed_accuracy.png` / `multiseed_bars.png` | 均值 ± 1 标准差曲线 / 带误差棒的最终准确率 |
| `docs/ceiling_metrics.json` | **归档的瓶颈分解原始数据**：9 组 × 5 seed 曲线、分组统计、量子读出探针、像素探针 |
| `docs/figures/ceiling_ladder.png` / `ceiling_curves.png` | 九组配置准确率阶梯 / 预算翻倍的收敛曲线 |
| `results/mnist_metrics.json` | 历史主实验：准确率、参数量、耗时、配置与 `meta`（含数据源标注） |
| `results/mnist_accuracy.png` | 量子 vs 经典的测试准确率曲线 |
| `results/mnist_params.png` / `mnist_time.png` | 参数量 / 耗时对比柱状图 |
| `results/amplitude_metrics.json` | 编码对照：四组指标、配置与真机代价 caveat |
| `results/amplitude_accuracy.png` / `amplitude_bars.png` | 编码方式对照曲线 / 柱状图 |
| `results/capacity_scan.json` | 容量扫描：输入维度与电路深度两组原始指标 |
| `results/capacity_scan.png` | 两条容量曲线（准确率随维度 / 随层数变化） |

## 技术栈与下一步

**PennyLane**（可微分量子电路）+ **PyTorch**（混合训练），辅以 scikit-learn（PCA）/ NumPy / Matplotlib；默认 `default.qubit` 状态向量模拟。[Qiskit Machine Learning](https://qiskit.github.io/qiskit-machine-learning/) 是接 IBM 生态时的另一条路线。

**已实现**：amplitude encoding 无压缩对照（`train_amplitude.py`）、输入维度与电路深度分离的容量扫描（`scan_capacity.py`）、**5 seed 受控对照与配对差值置信区间**（`train_multiseed.py`）、带非线性的 MLP 分类头与固定 / 可学习输出缩放、**9 组 × 5 seed 的瓶颈分解与量子读出信息探针**（`diagnose_ceiling.py`）、轻量 / 强特征两种混合 CNN 前端。方法学审查见 [`docs/REVIEW.md`](docs/REVIEW.md)。

下一步：等参数预算对照、给量子层单独调 lr / 初始化（轻量前端有 1/5 seed 训练失败）、数据重上传（data re-uploading）、多可观测量读出以突破「只有 n 个 `PauliZ` 数」的瓶颈、量子核 QSVC 第三条基线、噪声模型与有限 `--shots`、`lightning.gpu` 后端推到 20+ qubits。

## 相关仓库与许可

- 姊妹仓库：[quantum-llm](https://github.com/quantum-machine-learning-code/quantum-llm) · 玩具级量子增强语言模型
- 组织主页：[quantum-machine-learning-code](https://github.com/quantum-machine-learning-code)

本项目采用 [MIT License](LICENSE)。
