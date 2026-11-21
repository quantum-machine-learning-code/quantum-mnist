# Quantum MNIST · 三范式受控量子机器学习基准

[English](README_EN.md) | 简体中文

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![PennyLane](https://img.shields.io/badge/PennyLane-0.45-9b59b6)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![License](https://img.shields.io/badge/license-MIT-green)
![Hardware](https://img.shields.io/badge/quantum%20hardware-not%20required-success)

> 在同一套严格受控框架下，清晰区分**（I）经典算法在经典上**、**（II）经典算法移植到量子线路**与**（III）量子原生算法在量子线路上**，覆盖经典图像（MNIST）与原生纠缠量子态（1D Cluster-Ising SPT 相）双赛道。**纯模拟、零测试集泄露、全量参数拆解、含真实有限 shots 评估。**

---

## 一屏结论：研究问题 ↔ 受控实验 ↔ 判定

| 核心研究问题 | 受控实验（[`docs/three_regimes_metrics.json`](docs/three_regimes_metrics.json)） | 实测结论（5 独立分层 seed，Holm-Bonferroni 校正） |
| --- | --- | --- |
| 1. 经典模型套量子层（Regime II）在 MNIST 上有量子优势吗？ | `vqc_amp` vs 固定随机二次型 & `poly2`；`patch_vqc` vs 48 参经典滤波；`hybrid_vqc` vs 双 BN 等宽 `hybrid_noq` | **无优势**。`vqc_amp`（`0.7760`）虽高于未训练随机二次型（`0.6515`，`p_holm=0.024`），但远低于经典 `poly2`（`0.9155`）且 200 shots 下暴跌 `-36.70 pt`；等参 `patch_vqc`（`0.8705`）与 48 参经典滤波（`0.8715`）持平；修复 BN 后 `hybrid_noq`（`0.8310`）高于 `hybrid_vqc`（`0.7600`）。 |
| 2. 去掉经典 MLP 封装后（Regime III），量子原生模型在 MNIST 上表现如何？ | 4q 深度重上传纯 Born 读出（96q / 0c）vs 96 参经典三角网；6q 纠缠 IQP 核（Val 调带宽 $\alpha$）vs 同预处理 RBF/Poly-2 | **能独立收敛，但不超越等参/同预处理经典基线**。纯量子重上传 `0.6175` 与 96 参 `classical_trig_net` `0.6165` 完全持平（`+0.10 pt`）；纠缠 IQP 核 `0.7380` 与 `poly2`（`0.7415`）/ `rbf`（`0.7545`）持平，并实测验证高 $\alpha$ 下的指数核集中现象。 |
| 3. 量子原生算法（Regime III）在**原生量子态数据**上何时真正有效？ | 1D Cluster-Ising $n=8$ PBC 基态 SPT vs 非 SPT（顺磁+反铁磁）：Cong-et-al. QCNN（36q / 0c / 1 QWC 基）vs 局域观测量 & 全态 Oracle | **显著有效且抗采样噪声**。QCNN 解析 **`0.9967`**、1000 shots **`0.9950`**、200 shots **`0.9633`**（无显著衰减），Holm 显著超越单比特局域期望（`0.6917` / `0.9167`，`p_holm ≤ 0.0068`），仅用 36 参数与 1 个测量基即逼近指数层析成本的全态向量 Oracle（`1.0000`）。 |

---

## 三范式受控基准（Track A + Track B）

**统一协议**：每个 seed 独立构造严格互不相交的分层 `Train / Val / Test` 切分（所有神经网络、VQC、核方法与 QCNN 共用同一套切分语义）；所有超参（如 IQP 带宽 $\alpha$、SVM $C$、checkpoint 选择）严格在 `Val` 上决定，`Test` 仅评估一次；所有模型显式拆解 `量子参数 / 经典封装参数 / 离线预处理拟合参数（如 PCA）` 并以 `qml.pauli.group_observables` 程序化统计 QWC 测量基数。

### Track A：经典数据 — MNIST 10 类（600 Train / 200 Val / 400 Test，5 seeds）

| 范式 | 模型 | 量子参 | 经典封装参 | PCA 拟合参 | QWC 基 | Test Acc（解析） | 有限 Shots / 备注 |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| **I · 经典** | `classical_poly2_unscaled` | 0 | 0 | 0 | 0 | **`0.9155 ± 0.0124`** | 与 `qkernel_amp` 逐样本恒等（精确去量子化） |
| **I · 经典** | `classical_patch_lowrank` | 0 | 20,364（滤波 48） | 0 | 0 | `0.8715 ± 0.0157` | 与 `patch_vqc_matched` 同总参 / `batch=64` / BN / WD |
| **II · 移植** | `patch_vqc_matched` | 48 | 20,316 | 0 | 3 | `0.8705 ± 0.0262` | 配对差 `−0.10 ± 1.82 pt`（`p_holm=1.0`，量子占 0.24%） |
| **I · 经典** | `hybrid_noq_matched` | 0 | 5,162 | 0 | 0 | `0.8310 ± 0.0569` | `stabilize=True`（双 BN + 32d 等宽桥接） |
| **II · 移植** | `hybrid_vqc_matched` | 48 | 5,114 | 0 | 3 | `0.7600 ± 0.0904` | 配对差 `−7.10 ± 10.34 pt`（`p_holm=0.798`，量子占 0.93%） |
| **II · 移植** | `vqc_amp_matched`（10q 40d） | 60 | 1,642 | 0 | 3 | `0.7760 ± 0.0314` | **200 shots: `0.4090`（`−36.70 pt`）｜1000 shots: `0.6360`（`−14.00 pt`）** |
| **I · 经典** | `classical_random_quad`（40d） | 0 | 1,642 | 0 | 0 | `0.6515 ± 0.0516` | 0 学习投影参数；`vqc_amp` 差 `+12.45 pt`（`p_holm=0.024`） |
| **I · 经典** | `classical_rbf_pca6` | 0 | 0 | 5,488 | 0 | `0.7545 ± 0.0287` | 同逐 PC 标准化 PCA-6，Val 调 `gamma/C` |
| **I · 经典** | `classical_poly2_pca6` | 0 | 0 | 5,488 | 0 | `0.7415 ± 0.0212` | 同逐 PC 标准化 PCA-6，Val 调 `C` |
| **III · 原生** | `qnative_iqp_kernel`（6q 2-rep） | 0 | 0 | 5,488 | 1 | `0.7380 ± 0.0339` | 真实纠缠 IQP 线路；`offdiag` 随 $\alpha\in[0.15,0.75]$ 由 `0.875→0.019` |
| **III · 原生** | `qnative_reupload_fidelity`（4q L=4） | **96** | **0** | 10,192 | 1 | `0.6175 ± 0.0307` | 零经典 MLP 封装，纯计算基 Born 概率读出（100% 量子参数） |
| **I · 经典** | `classical_trig_net`（PCA-12） | 0 | 96 | 10,192 | 0 | `0.6165 ± 0.0347` | 等参三角网；配对差 `+0.10 ± 4.39 pt`（`p_holm=1.0`） |

### Track B：原生量子态数据 — 1D Cluster-Ising SPT 相分类（$n=8$ PBC，120/60/120，5 seeds）

哈密顿量 $H(h_1, h_2) = -\sum_{i=0}^{7} Z_{i-1}X_i Z_{i+1} - h_1\sum_i X_i - h_2\sum_i X_i X_{i+1}$（精确稀疏对角化生成 SPT 相 vs 非 SPT [顺磁 + 反铁磁] 基态 $|\psi\rangle$）：

| 范式 | 模型 | 量子参 | 经典参 | QWC 基 | Test Acc（解析） | 相对 `qcnn_spt` 配对差（Holm 校正） / 有限 Shots |
| --- | --- | ---: | ---: | ---: | --- | --- |
| **III · 原生** | **`qcnn_spt`（Cong-et-al. QCNN）** | **36** | **0** | **1** | **`0.9967 ± 0.0045`** | **200 shots: `0.9633 ± 0.0356`（`p_holm=0.198`）｜1000 shots: `0.9950 ± 0.0045`** |
| **I · 经典** | `classical_1body_logreg`（24d） | 0 | 25 | — | `0.6917 ± 0.0899` | QCNN **`+30.50 ± 9.10 pt`**（CI `[+19.20, +41.80]`，`p_holm=0.00678` ✓） |
| **I · 经典** | `classical_1body_mlp`（24d） | 0 | 833 | — | `0.9167 ± 0.0118` | QCNN **`+8.00 ± 0.95 pt`**（CI `[+6.82, +9.18]`，`p_holm=0.000235` ✓） |
| **I · 经典** | `classical_2body_mlp`（48d） | 0 | 1,601 | — | `0.9517 ± 0.0379` | QCNN `+4.50 ± 3.47 pt`（CI `[+0.20, +8.80]`，`p_holm=0.132`） |
| **I · 经典** | `classical_full_state_mlp`（$2^8$ Oracle） | 0 | 8,257 | — | `1.0000 ± 0.0000` | `−0.33 ± 0.45 pt`（`p_holm=0.198`；需 $O(2^n)$ 指数级态层析） |

**物理边界说明**：在经典图像上，单拷贝 `AmplitudeEmbedding` + 任何 CPTP 量子通道（包括带中间测量的 QCNN）期望值都严格受限于归一化像素的齐次二次型，且真机需 $O(2^n)$ 态制备；相反，在原生量子态 $|\psi\rangle$ 上，无经典装载瓶颈，Cong-et-al. QCNN 经 $8\to 4\to 2\to 1$ 相干量子池化直接提取非局域弦序，以 36 参数和 1 个测量基即在有限 shots 下保持高准确率。

---

## 历史实验索引（Rounds 1–8，完整审计见 [`docs/REVIEW.md`](docs/REVIEW.md)）

- **全量 MNIST（Round 7，60,000/10,000，3 seeds，[`docs/full_metrics.json`](docs/full_metrics.json)）**：165 维振幅 VQC（`60` 量子 + `5,642` 经典头，3 QWC 基）达 `0.9421 ± 0.0008`（单 seed ~42 min），经典 `PCA-165 + MLP`（`5,642` MLP + **`130,144` PCA 拟合参数**）达 `0.9554 ± 0.0028`（~2.3 s）。
- **Round 8 外部审计复算（[`docs/audit_metrics.json`](docs/audit_metrics.json)，已修正 `batch_size=64` 与权重衰减对齐）**：`qkernel_amp`（`0.9210`）与未标准化经典 `poly2(coef0=0)` 逐 seed 恒等；`PatchLinearClassifier` 消融得 `0.8910 ± 0.0143` vs `PatchVQC` `0.8800 ± 0.0112`（`+1.10 pt`，CI 跨 0，快 ~80 倍）；20 项配对检验 Bonferroni 后仅 2/20 显著。
- **全部历史原始归档**：`docs/{multiseed,ceiling,readout,info,push,budget,why_low,full,open_issues,audit,three_regimes}_metrics.json` 与 [`docs/REVIEW.md`](docs/REVIEW.md)（§1–§16）。

---

## 安装与复现

要求 **Python 3.11+**，纯 CPU 离线可运行（本地 MNIST 4 个 IDX 文件放置说明见 [`docs/DEV_NOTES.md`](docs/DEV_NOTES.md)）。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e ".[dev]"

# 1. 离线单元测试与回归契约（8 项测试：角度核精确性 / 逐 PC 标准化 / QWC / QCNN / --quick）
.venv/bin/python -m pytest -v tests/smoke_test.py

# 2. 三范式严格受控基准（Track A MNIST + Track B Cluster-Ising SPT，~5.5 min CPU / 5 workers）
.venv/bin/python experiments/benchmark_three_regimes.py --data-source idx \
    --workers 5 --torch-threads 2 --archive docs/three_regimes_metrics.json

# 3. 离线快速冒烟检查（合成数据，写入 results/ 不覆盖 docs/）
.venv/bin/python experiments/benchmark_three_regimes.py --quick
.venv/bin/python experiments/audit_round8.py --data-source idx --archive docs/audit_metrics.json
```

### 目录概览

- `qml_lab/`：`quantum.py`（VQC / QWC 分组计数 / 有限 shots / 纠缠 IQP 核）、`models.py`（分类器 / 去量子化对照 / `DataReuploadingFidelityClassifier` / `parameter_breakdown` / `assert_matched_config`）、`quantum_phases.py`（1D Cluster-Ising 稀疏对角化 + Cong-et-al. QCNN）、`data.py`（逐 PC 标准化 + 不重叠 TVT 切分）、`train.py`、`utils.py`、`viz.py`。
- `experiments/`：`benchmark_three_regimes.py`（当前主基准）、`audit_round8.py` 及 Rounds 1–8 历史脚本。
- `tests/smoke_test.py`、`docs/REVIEW.md`、`docs/DEV_NOTES.md`、`docs/*_metrics.json`。

---

## 相关仓库与许可

- 姊妹仓库：[quantum-llm](https://github.com/quantum-machine-learning-code/quantum-llm) · 组织主页：[quantum-machine-learning-code](https://github.com/quantum-machine-learning-code)
- 本项目采用 [MIT License](LICENSE)。
