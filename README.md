# 量子手写数字识别 · Quantum MNIST

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![PennyLane](https://img.shields.io/badge/PennyLane-0.45-9b59b6)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![License](https://img.shields.io/badge/license-MIT-green)
![Hardware](https://img.shields.io/badge/quantum%20hardware-not%20required-success)

> 用参数化量子电路（VQC）做 MNIST 手写数字分类，并与同输入维度的经典 MLP 放在同一套训练配置下对照。**可复现、纯模拟、不接真机。**

本仓库只做一件事：量子机器学习版本的 MNIST 分类。姊妹仓库 [quantum-llm](https://github.com/quantum-machine-learning-code/quantum-llm) 做玩具级量子增强语言模型，两者代码各自自包含，可独立 clone 与运行。

## 快速结论（实跑结果）

Python 3.13 / PennyLane 0.45.1 / torch 2.15 CPU，`seed=0`，单次运行，真实 MNIST。复现命令见[快速开始](#快速开始)。

| 模型 | 参数量 | 测试准确率 | 训练耗时 |
| --- | --- | --- | --- |
| VQC（量子） | 74（量子线路参数 24） | **0.3733** | 26.46 s |
| 经典 MLP | 250 | **0.5733** | 0.63 s |

**量子模型没赢，而且慢约 42 倍。** 这证明的是「量子电路能被纳入 PyTorch 训练循环并正常收敛」，不是量子机器学习的能力上限 —— 配置刻意保持最小、未调优。完整解读见[怎么读这些数字](#怎么读这些数字)。

## 能做什么，不能做什么

- **默认配置是模拟预算，不是物理限制**：`default.qubit` 状态维度 2ⁿ。本机实测单 batch 前向+反向：4/8/12/16 qubits 约 0.06–1.2 s，20 qubits 就要 **63 s**。默认 `qubits=4` 只是为了让笔记本几分钟跑完一个 epoch。`--qubits` 与 `--layers` 都能往上调，量子参数量为 `layers × qubits × 3`（4 qubits × 2 层 = 24 只是这一个配置的数字，不存在「最多 24 个」的限制）。
- **压到 4 维是编码方式造成的，不是量子机器学习的要求**：`AngleEmbedding` 用一个 qubit 装一个旋转角，所以特征维必须等于 qubits 数。换成 `AmplitudeEmbedding`，n 个 qubits 的态空间可容纳 2ⁿ 个振幅 —— **10 个 qubits（1024 ≥ 784）就能无损装下完整图像，完全不用 PCA**。可跑对照见 [`experiments/train_amplitude.py`](experiments/train_amplitude.py)；`experiments/scan_capacity.py` 则把「输入维度」与「电路深度」两个因素分开扫。
- **但放大不是免费的**：qubits 与层数增加会放大 barren plateau（随机初始化下梯度方差随 n 指数衰减）、状态制备成本与 2ⁿ 模拟内存；读出端只取 n 个 `PauliZ` 期望，本身也是信息瓶颈。而且**纠缠越强，经典模拟越难** —— 张量网络只在低纠缠时高效。所以「假设有很多强关联 qubits」恰恰是量子优势最可能出现、也最无法在笔记本上验证的区域。
- **不宣称量子优势**：在这个规模下量子模型打平甚至落后于经典基线是常态。任何「更好」的结论都需要等参数预算、多次重复实验和统计检验。
- **不需要真机凭证**：纯模拟无需 IBM Quantum 等账号；只有切到真实量子硬件时才需要 token。

## 目录结构

```
quantum-mnist/
├── README.md
├── LICENSE                      # MIT
├── requirements.txt
├── qml_lab/
│   ├── utils.py                 # 随机种子、路径、JSON、参数量、终端表格
│   ├── data.py                  # MNIST 加载：load_mnist（PCA）/ load_mnist_raw（全 784 维）
│   ├── quantum.py               # AngleEmbedding / AmplitudeEmbedding 两种可微分层
│   ├── models.py                # VQCClassifier / AmplitudeVQCClassifier / MLPClassifier
│   ├── train.py                 # 分类训练循环与分批准确率评估
│   └── viz.py                   # 指标图绘制
├── experiments/train_mnist.py   # 主实验：VQC vs MLP
├── experiments/train_amplitude.py  # 编码对照：振幅编码无需 PCA
├── experiments/scan_capacity.py    # 容量扫描：输入维度 vs 电路深度
├── tests/smoke_test.py          # 离线冒烟测试（不联网、不下载 MNIST）
├── docs/DEV_NOTES.md            # 本机环境与离线数据准备备注
├── docs/figures/                # 本次实测的对比图（已归档）
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

下面这条正是本 README 实测结果的复现命令：

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
- **数据源回退**：`idx`（直接读本地 4 个 IDX 文件，完全离线）→ `torchvision` → `sklearn.fetch_openml` → `synthetic`。
- **Baseline**：同输入维度、同训练配置的经典 MLP；`--skip-baseline` 可跳过。
- **指标**：准确率曲线、参数量、训练耗时，写入 `results/mnist_metrics.json` 并出三张图。
- **采样与求导**：`--shots 0` 为解析模拟（默认）；正整数启用有限采样并自动切换到 `parameter-shift`；`--diff-method` 可显式指定。

## 参数速查

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

配置：本地 IDX 真实 MNIST、1200 训练 / 300 测试、PCA→4 维、4 qubits / 2 层、12 epochs、batch 32、lr 0.01、解析模拟 `backprop`。

![VQC 与 MLP 的测试准确率曲线](docs/figures/mnist_accuracy.png)

参数量与耗时对比见 [`docs/figures/mnist_params.png`](docs/figures/mnist_params.png) 与 [`docs/figures/mnist_time.png`](docs/figures/mnist_time.png)。

### 怎么读这些数字

- 量子版本**没赢过经典版本**，且慢约 42 倍。这反映的是**当前规模下量子模拟的性价比**，不能推广成「量子机器学习不行」。
- 低准确率来自几个**可以分别调整**的因素：角度编码强制把 784 维压到 4 维、电路只有 24 个量子参数、读出只有 4 个 `PauliZ` 期望、样本与 epoch 预算都很小。它们都是配置，不是量子机器学习的上限 —— 下一节用受控实验逐项拆开验证。
- 要公平比较至少需要：相同参数预算、更好的编码、多 `--seed` 重复取均值与方差，以及量子核方法等第三条基线。
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

**受控对照**：`Angle-10` 与 `Amplitude-10` 的 qubits 数、ansatz、量子参数量（60）和训练配置完全相同，唯一差别是编码方式以及因此需不需要把输入压到 10 维。准确率 0.3750 → 0.5300（+15.5 个百分点）。这说明「压到几维」确实是当时的准确率瓶颈，但它是**可以解除的配置选择**，不是量子机器学习的固有限制。

两个反向证据同样重要：`Angle-10` 参数比 `Angle-4` 多，准确率反而更低 —— qubits / 层数增加会放大 barren plateau 与优化难度，「把参数量放大」不是无条件有效；经典 MLP 在同样的 784 维上到 0.8450，量子侧在电路设计、读出与优化上仍有明显差距。

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
- **最有效的不是堆参数，是换表示**：同 qubits、同 60 个量子参数下把角度编码换成振幅编码 +15.5 个百分点（上一节），比任何单纯「放大」都明显。

**数据口径提醒**：以上均为 `seed=0` 单次运行，未做统计重复；每张表内部是受控对照，但不同脚本之间模型构造与 `set_seed` 的先后顺序不同，绝对数值**不可跨表逐位比较**，只看趋势。经典 MLP 列存在小幅波动（同一配置 0.565–0.605），因为它与 VQC 共享随机状态、未单独固定初始化；VQC 列每组都从同一 `set_seed` 开始，是严格受控的。要发结论仍需多 `--seed` 重复与方差。

### 运行产物

`results/` 与 `data/` 被 `.gitignore` 忽略，以下均为运行后本地生成；`docs/figures/` 归档了本次实测的对比图。

| 文件 | 内容 |
| --- | --- |
| `results/mnist_metrics.json` | 主实验：准确率、参数量、耗时、配置与 `meta`（含数据源标注） |
| `results/mnist_accuracy.png` | 量子 vs 经典的测试准确率曲线 |
| `results/mnist_params.png` / `mnist_time.png` | 参数量 / 耗时对比柱状图 |
| `results/amplitude_metrics.json` | 编码对照：四组指标、配置与真机代价 caveat |
| `results/amplitude_accuracy.png` / `amplitude_bars.png` | 编码方式对照曲线 / 柱状图 |
| `results/capacity_scan.json` | 容量扫描：输入维度与电路深度两组原始指标 |
| `results/capacity_scan.png` | 两条容量曲线（准确率随维度 / 随层数变化） |

## 技术栈与下一步

**PennyLane**（可微分量子电路）+ **PyTorch**（混合训练），辅以 scikit-learn（PCA）/ NumPy / Matplotlib；默认 `default.qubit` 状态向量模拟。[Qiskit Machine Learning](https://qiskit.github.io/qiskit-machine-learning/) 是接 IBM 生态时的另一条路线。

**已实现**：amplitude encoding 无压缩对照（`train_amplitude.py`）、输入维度与电路深度分离的容量扫描（`scan_capacity.py`）。

下一步：多个 `--seed` 重复实验与统计检验、等参数预算对照、数据重上传（data re-uploading）、增加读出可观测量、量子核 QSVC 第三条基线、噪声模型与有限 `--shots`、`lightning.gpu` 后端推到 20+ qubits。

## 相关仓库与许可

- 姊妹仓库：[quantum-llm](https://github.com/quantum-machine-learning-code/quantum-llm) · 玩具级量子增强语言模型
- 组织主页：[quantum-machine-learning-code](https://github.com/quantum-machine-learning-code)

本项目采用 [MIT License](LICENSE)。
