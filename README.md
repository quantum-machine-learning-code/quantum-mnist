# Quantum MNIST · 三范式受控量子机器学习基准

[English](README_EN.md) | 简体中文

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![PennyLane](https://img.shields.io/badge/PennyLane-0.45-9b59b6)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![License](https://img.shields.io/badge/license-MIT-green)
![Hardware](https://img.shields.io/badge/quantum%20hardware-not%20required-success)

> 在同一套严格受控框架下，清晰区分**（I）经典算法在经典上**、**（II）经典算法移植到量子线路**与**（III）量子原生算法在量子线路上**，覆盖经典图像（MNIST）、原生纠缠量子态（1D Cluster-Ising SPT 相）与**真实硬件耦合拓扑 SWAP 路由及 `default.mixed` 去极化噪声**。**纯模拟、零测试集泄露、全量参数拆解、含真实有限 shots 与含噪密度矩阵评估。**

---

## 正式实验报告索引（IMRaD + QML 可复现核查表）

| 报告 | 主题与范围 | 归档数据与配图 | 核心结论 |
| --- | --- | --- | --- |
| [**报告 01**：MNIST 三范式基准](docs/reports/01_mnist_three_regimes_report.md) | Track A 三范式对照、有限 shots 衰减与 Rounds 7–8 审计综合 | [`three_regimes_metrics.json`](docs/three_regimes_metrics.json) · [`audit_metrics.json`](docs/audit_metrics.json) · [`full_metrics.json`](docs/full_metrics.json) | 经典图像上范式 II 无量子优势，200 shots 暴跌 `−36.70 pt`；范式 III 零封装能收敛但不超越等参经典基线。 |
| [**报告 02**：Cluster-Ising QCNN](docs/reports/02_cluster_ising_qcnn_report.md) | Track B 原生 1D Cluster-Ising $n=8$ PBC 拓扑相分类 | [`three_regimes_metrics.json`](docs/three_regimes_metrics.json) · [`qcnn_spt_phases.png`](docs/figures/qcnn_spt_phases.png) | Cong-et-al. QCNN（36q/0c/1 QWC）达 **`0.9967`**（200 shots `0.9633`），Holm 显著超越单比特局域期望（`p_holm ≤ 0.0068`）。 |
| [**报告 03**：硬件拓扑与去极化噪声](docs/reports/03_hardware_topology_noise_report.md) | 4 拓扑 SABRE SWAP 路由 × 5 线路 + `default.mixed` 去极化仿真 | [`hardware_topology_metrics.json`](docs/hardware_topology_metrics.json) · [`topology_noise_degradation.png`](docs/figures/topology_noise_degradation.png) | `heavy_hex` / `linear_chain` 令全连接 IQP/QFT 门数膨胀 `2.9×–3.6×`；IQP 核方差塌缩至 **`3.4%`**（`−11.40 pt`, `p_holm=0.0342`），QCNN 决策边界保持 `≥0.9950`。 |

---

## 一屏结论：研究问题 ↔ 受控实验 ↔ 判定

| 核心研究问题 | 受控实验 | 实测结论（5 独立分层 seed，Holm-Bonferroni 校正） |
| --- | --- | --- |
| 1. 经典模型套量子层（Regime II）在 MNIST 上有量子优势吗？ | `vqc_amp` vs 随机二次型 & `poly2`；`patch_vqc` vs 48 参滤波；`hybrid_vqc` vs 双 BN 等宽 `hybrid_noq` | **无优势**。`vqc_amp`（`0.7760`）远低于经典 `poly2`（`0.9155`）且 200 shots 暴跌 `−36.70 pt`；等参 `patch_vqc`（`0.8705`）与 `classical_patch_lowrank`（`0.8715`）持平；`hybrid_noq`（`0.8310`）高于 `hybrid_vqc`（`0.7600`）。 |
| 2. 去掉经典 MLP 封装后（Regime III），量子原生模型在 MNIST 上表现如何？ | 4q 深度重上传 Born 读出（96q/0c）vs 96 参三角网；6q 纠缠 IQP 核 vs 同预处理 RBF/Poly-2 | **能独立收敛，与等参/同预处理经典基线持平**。纯量子重上传 `0.6175` 持平 `classical_trig_net` `0.6165`（`p_holm=1.0`）；IQP 核 `0.7380` 持平 `poly2`（`0.7415`）/ `rbf`（`0.7545`），高 $\alpha$ 下触发指数核集中。 |
| 3. 量子原生算法（Regime III）在**原生量子态**上何时真正有效？ | 1D Cluster-Ising $n=8$ PBC 基态 SPT vs 非 SPT：Cong-et-al. QCNN（36q/0c/1 QWC）vs 局域期望 & Oracle | **显著有效且抗采样噪声**。QCNN 解析 **`0.9967`**、1000 shots **`0.9950`**、200 shots **`0.9633`**（`p_holm=0.198`），显著超越单比特期望（`0.6917` / `0.9167`，`p_holm ≤ 0.0068`），逼近指数层析全态 Oracle（`1.0000`）。 |
| 4. 真实硬件耦合拓扑与去极化噪声如何改变上述结论？ | 4 拓扑（`all_to_all / grid_2d / heavy_hex / linear_chain`）SABRE SWAP 路由 + `default.mixed`（$p_{2q}\le 0.01$） | **稀疏拓扑按指数放大全连接线路核集中**。`heavy_hex`/`linear_chain` 令 IQP 物理 2Q 门增至 `2.90×`/`3.40×`（QFT 达 `3.025×`/`3.625×`），$p_{2q}=0.01$ 下 IQP 核方差暴跌至 `5.2%`/`3.4%`、准确率跌至 `0.5560`/`0.5420`（`−11.40 pt`, `p_holm=0.0342`）；QCNN 解析决策保持 `0.9950–0.9967`（对称信道仅缩窄极化幅度）。 |

---

## 三范式受控基准（Track A + Track B）

### Track A：经典数据 — MNIST 10 类（600 Train / 200 Val / 400 Test，5 seeds，[`docs/three_regimes_metrics.json`](docs/three_regimes_metrics.json)）

| 范式 | 模型 | 量子参 | 经典封装参 | PCA 拟合参 | QWC 基 | Test Acc（解析） | 有限 Shots / 备注 |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| **I · 经典** | `classical_poly2_unscaled` | 0 | 0 | 0 | 0 | **`0.9155 ± 0.0124`** | 与 `qkernel_amp` 逐样本恒等（精确去量子化） |
| **I · 经典** | `classical_patch_lowrank` | 0 | 20,364（滤波 48） | 0 | 0 | `0.8715 ± 0.0157` | 与 `patch_vqc_matched` 同总参 / `batch=64` / BN / WD |
| **II · 移植** | `patch_vqc_matched` | 48 | 20,316 | 0 | 3 | `0.8705 ± 0.0262` | 配对差 `−0.10 ± 1.82 pt`（`p_holm=1.0`，量子占 0.24%） |
| **I · 经典** | `hybrid_noq_matched` | 0 | 5,162 | 0 | 0 | `0.8310 ± 0.0569` | `stabilize=True`（双 BN + 32d 等宽桥接） |
| **II · 移植** | `hybrid_vqc_matched` | 48 | 5,114 | 0 | 3 | `0.7600 ± 0.0904` | 配对差 `−7.10 ± 10.34 pt`（`p_holm=0.798`，量子占 0.93%） |
| **II · 移植** | `vqc_amp_matched`（10q 40d） | 60 | 1,642 | 0 | 3 | `0.7760 ± 0.0314` | **200 shots: `0.4090`（`−36.70 pt`）｜1000 shots: `0.6360`（`−14.00 pt`）** |
| **I · 经典** | `classical_random_quad`（40d） | 0 | 1,642 | 0 | 0 | `0.6515 ± 0.0516` | 0 学习投影参数；`vqc_amp` 差 `+12.45 pt`（`p_holm=0.024`） |
| **I · 经典** | `classical_rbf_pca6` / `poly2_pca6` | 0 | 0 | 5,488 | 0 | `0.7545` / `0.7415` | 同逐 PC 标准化 PCA-6，Val 调 `gamma/C` |
| **III · 原生** | `qnative_iqp_kernel`（6q 2-rep） | 0 | 0 | 5,488 | 1 | `0.7380 ± 0.0339` | 真实纠缠 IQP 核；`offdiag` 随 $\alpha\in[0.15,0.75]$ 由 `0.875→0.019` |
| **III · 原生** | `qnative_reupload_fidelity`（4q L=4） | **96** | **0** | 10,192 | 1 | `0.6175 ± 0.0307` | 零经典 MLP 封装，纯 Born 概率读出（100% 量子参数） |
| **I · 经典** | `classical_trig_net`（PCA-12） | 0 | 96 | 10,192 | 0 | `0.6165 ± 0.0347` | 等参三角网；配对差 `+0.10 ± 4.39 pt`（`p_holm=1.0`） |

### Track B：原生量子态 — 1D Cluster-Ising SPT 相分类（$n=8$ PBC，120/60/120，5 seeds）

| 范式 | 模型 | 量子参 | 经典参 | QWC 基 | Test Acc（解析） | 相对 `qcnn_spt` 配对差（Holm 校正） / 有限 Shots |
| --- | --- | ---: | ---: | ---: | --- | --- |
| **III · 原生** | **`qcnn_spt`（Cong-et-al. QCNN）** | **36** | **0** | **1** | **`0.9967 ± 0.0045`** | **200 shots: `0.9633 ± 0.0356`（`p_holm=0.198`）｜1000 shots: `0.9950 ± 0.0045`** |
| **I · 经典** | `classical_1body_logreg`（24d） | 0 | 25 | — | `0.6917 ± 0.0899` | QCNN **`+30.50 ± 9.10 pt`**（CI `[+19.20, +41.80]`，`p_holm=0.00678` ✓） |
| **I · 经典** | `classical_1body_mlp`（24d） | 0 | 833 | — | `0.9167 ± 0.0118` | QCNN **`+8.00 ± 0.95 pt`**（CI `[+6.82, +9.18]`，`p_holm=0.000235` ✓） |
| **I · 经典** | `classical_2body_mlp`（48d） | 0 | 1,601 | — | `0.9517 ± 0.0379` | QCNN `+4.50 ± 3.47 pt`（CI `[+0.20, +8.80]`，`p_holm=0.132`） |
| **I · 经典** | `classical_full_state_mlp`（$2^8$ Oracle） | 0 | 8,257 | — | `1.0000 ± 0.0000` | `−0.33 ± 0.45 pt`（`p_holm=0.198`；需 $O(2^n)$ 指数级态层析） |

---

## 硬件耦合拓扑 SWAP 路由与 `default.mixed` 去极化噪声基准（[`docs/hardware_topology_metrics.json`](docs/hardware_topology_metrics.json)）

基于双向 SABRE 启发式 SWAP 路由（1 SWAP = 3 物理 CNOT，全部 20 组无噪态向量误差 `≤ 2.776e-16`）与 PennyLane `default.mixed` 含噪密度矩阵仿真（5 seeds，$p_{2q} = 0.010$）：

| 耦合拓扑（最大度） | `qcnn_spt` 8q（物理 2Q / 深度） | `qcnn_spt` Acc ($p_{2q}=0.01$) | `iqp_kernel` 6q（物理 2Q / 深度） | `iqp_kernel` Acc ($p_{2q}=0 \to 0.01$) | IQP 核方差保留比 `std_ratio` | `qft_8` 8q（物理 2Q / 深度） |
| --- | --- | --- | --- | --- | ---: | --- |
| `all_to_all`（$n-1$） | `47 (1.00×) / 19` | `0.9967 ± 0.0045` | `30 (1.00×) / 15` | `0.6560 → 0.6100` | `0.3338` | `40 (1.00×) / 16` |
| `grid_2d`（3） | `56 (1.19×) / 25` | `0.9967 ± 0.0045` | `66 (2.20×) / 41` | `0.6560 → 0.5700` | `0.0993` | `91 (2.28×) / 59` |
| `heavy_hex`（≤3） | `80 (1.70×) / 51` | `0.9950 ± 0.0045` | `87 (2.90×) / 63` | `0.6560 → 0.5560` | `0.0525` | `121 (3.03×) / 93` |
| `linear_chain`（2） | `86 (1.83×) / 37` | `0.9950 ± 0.0045` | `102 (3.40×) / 59` | `0.6560 → 0.5420`（`p_holm=0.0342` ✓） | **`0.0345`** | `145 (3.63×) / 79` |

---

## 历史归档与复现

- **Rounds 1–8 历史归档**：全量 MNIST（60k/10k, [`docs/full_metrics.json`](docs/full_metrics.json)）165d 振幅 VQC 达 `0.9421 ± 0.0008`（vs 经典 `pca165` `0.9554` / 冻结随机酉 `0.9440`）；Round-8 审计（[`docs/audit_metrics.json`](docs/audit_metrics.json)）证伪 `qkernel_amp == plain_poly2_c0`（`0.9210`）且 `PatchLinearClassifier`（`0.8910`）持平 `PatchVQC`（`0.8800`）；全过程见 [`docs/REVIEW.md`](docs/REVIEW.md)。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e ".[dev]"

# 1. 离线单元测试与回归契约（9 项测试：含拓扑/SABRE/default.mixed 与全部 --quick 契约）
.venv/bin/python -m pytest -v tests/smoke_test.py

# 2. 三范式主基准 & 硬件耦合拓扑/去极化噪声基准（真实 MNIST IDX，纯 CPU 可复现）
.venv/bin/python experiments/benchmark_three_regimes.py --data-source idx --workers 5 --archive docs/three_regimes_metrics.json
.venv/bin/python experiments/benchmark_hardware_topology.py --data-source idx --workers 20 --archive docs/hardware_topology_metrics.json

# 3. 离线快速冒烟检查（写入 results/ 不覆盖 docs/）
.venv/bin/python experiments/benchmark_three_regimes.py --quick
.venv/bin/python experiments/benchmark_hardware_topology.py --quick
```

### 目录概览与相关仓库

- `qml_lab/`：`quantum.py`、`models.py`、`quantum_phases.py`（1D Cluster-Ising + QCNN）、`hardware_topology.py`（4 拓扑 + 双向 SABRE 路由 + `default.mixed` 噪声仿真）、`data.py`、`train.py`、`utils.py`、`viz.py`。
- `docs/reports/`（3 篇正式中文 IMRaD 实验报告）、`docs/figures/`（4 张 300 dpi 英文标注论文配图）、`docs/*_metrics.json`、`tests/smoke_test.py`。
- 姊妹仓库：[quantum-llm](https://github.com/quantum-machine-learning-code/quantum-llm) · 组织主页：[quantum-machine-learning-code](https://github.com/quantum-machine-learning-code) · [MIT License](LICENSE)。
