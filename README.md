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

- **模拟器的内存墙**：`default.qubit` 是状态向量模拟器，状态维度 2ⁿ，每多一个 qubit 内存翻倍。4–12 qubits 在笔记本上是舒适区；20+ 需要 GPU 模拟器（Lightning / cuQuantum）。默认 `qubits=4` 就是为这个边界留余量。
- **PCA 是有损压缩**：把 784 维图像压到 4 维会丢掉绝大部分信息，这是当前准确率的主要瓶颈，不是量子电路本身的缺陷。
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
│   ├── data.py                  # MNIST 加载：idx / torchvision / openml / synthetic，PCA 与缩放
│   ├── quantum.py               # PennyLane 设备、QNode、AngleEmbedding + ansatz
│   ├── models.py                # VQCClassifier / MLPClassifier
│   ├── train.py                 # 分类训练循环与分批准确率评估
│   └── viz.py                   # 指标图绘制
├── experiments/train_mnist.py   # 实验入口
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

## 实验设计

- **数据流**：MNIST 图像展平 → `PCA` 降到 `n_qubits` 维 → 线性缩放到 `[-π, π]` → 量子电路 → 测量得到经典向量 → `Linear` 分类头输出 logits。
- **编码**：`AngleEmbedding` 的 `RY` 旋转，每个降维特征作为一个 qubit 的旋转角，因此特征维必须等于 `n_qubits`。
- **Ansatz**：`StronglyEntanglingLayers`，层数由 `--layers` 控制（4 qubits × 2 层 = 24 个可训练量子参数）。
- **测量**：对每个 qubit 取 `PauliZ` 期望，得到长度为 `n_qubits` 的实数向量。
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
- 主要原因：4 qubits + 2 层只有 24 个可训练量子参数；PCA 把 784 维压到 4 维丢掉绝大部分信息；10 分类头容量有限；经典 MLP 参数更多、优化更成熟；量子层还要为每个样本跑一遍状态向量模拟。
- 要公平比较至少需要：相同参数预算、更好的编码（amplitude encoding、数据重上传）、多 `--seed` 重复取均值与方差，以及量子核方法等第三条基线。
- 准确的表述是：**在这组配置下，量子电路可被纳入 PyTorch 训练循环并正常收敛，但在这台机器上的精度与耗时都不占优。**

### 运行产物

`results/` 与 `data/` 被 `.gitignore` 忽略，以下均为运行后本地生成；`docs/figures/` 归档了本次实测的对比图。

| 文件 | 内容 |
| --- | --- |
| `results/mnist_metrics.json` | 准确率、参数量、耗时、配置与 `meta`（含数据源标注） |
| `results/mnist_accuracy.png` | 量子 vs 经典的测试准确率曲线 |
| `results/mnist_params.png` / `mnist_time.png` | 参数量 / 耗时对比柱状图 |

## 技术栈与下一步

**PennyLane**（可微分量子电路）+ **PyTorch**（混合训练），辅以 scikit-learn（PCA）/ NumPy / Matplotlib；默认 `default.qubit` 状态向量模拟。[Qiskit Machine Learning](https://qiskit.github.io/qiskit-machine-learning/) 是接 IBM 生态时的另一条路线。

下一步：多个 `--seed` 重复实验与统计检验、等参数预算对照、amplitude encoding / 数据重上传、量子核 QSVC 第三条基线、噪声模型与有限 `--shots`、`lightning.gpu` 后端推到 20+ qubits。

## 相关仓库与许可

- 姊妹仓库：[quantum-llm](https://github.com/quantum-machine-learning-code/quantum-llm) · 玩具级量子增强语言模型
- 组织主页：[quantum-machine-learning-code](https://github.com/quantum-machine-learning-code)

本项目采用 [MIT License](LICENSE)。
