# Quantum MNIST · Controlled Three-Regime QML Benchmark

English | [简体中文](README.md)

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![PennyLane](https://img.shields.io/badge/PennyLane-0.45-9b59b6)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![License](https://img.shields.io/badge/license-MIT-green)
![Hardware](https://img.shields.io/badge/quantum%20hardware-not%20required-success)

> A strictly controlled benchmark separating **(I) Classical algorithms on classical compute**, **(II) Classical templates ported onto quantum circuits**, and **(III) Quantum-native algorithms on quantum circuits** across classical images (MNIST), native entangled quantum states (1D Cluster-Ising SPT phases), and **hardware coupling topology SWAP-routing with `default.mixed` depolarizing noise**. **Pure simulation, zero test peeking, full parameter accounting, real finite-shot and noisy density-matrix evaluation.**

---

## Formal Experimental Reports (IMRaD + QML Reproducibility Checklists)

| Report | Scope & Experimental Coverage | Archived JSON & Figures | Verified Headline Finding |
| --- | --- | --- | --- |
| [**Report 01**: MNIST Three-Regime Benchmark](docs/reports/01_mnist_three_regimes_report.md) | Track A three-regime benchmark, finite-shot collapse, Rounds 7–8 audit synthesis | [`three_regimes_metrics.json`](docs/three_regimes_metrics.json) · [`audit_metrics.json`](docs/audit_metrics.json) · [`full_metrics.json`](docs/full_metrics.json) | No Regime II advantage on MNIST (`vqc_amp` drops `−36.70 pt` at 200 shots); zero-wrapper Regime III models converge but tie matched classical surrogates. |
| [**Report 02**: Cluster-Ising SPT QCNN](docs/reports/02_cluster_ising_qcnn_report.md) | Track B native 1D Cluster-Ising $n=8$ PBC SPT phase classification | [`three_regimes_metrics.json`](docs/three_regimes_metrics.json) · [`qcnn_spt_phases.png`](docs/figures/qcnn_spt_phases.png) | Cong-et-al. QCNN (`36q/0c/1 QWC`) reaches **`0.9967`** (`0.9633` at 200 shots), Holm-significantly beating 1-body observables (`p_holm ≤ 0.0068`). |
| [**Report 03**: Hardware Topology & Noise](docs/reports/03_hardware_topology_noise_report.md) | 4-topology SABRE SWAP routing × 5 circuits + `default.mixed` depolarizing noise | [`hardware_topology_metrics.json`](docs/hardware_topology_metrics.json) · [`topology_noise_degradation.png`](docs/figures/topology_noise_degradation.png) | `heavy_hex` / `linear_chain` inflate all-to-all IQP/QFT 2Q gates `2.9×–3.6×`; IQP spread collapses to **`3.4%`** (`−11.40 pt`, `p_holm=0.0342`), while QCNN analytic sign holds `≥0.9950`. |

---

## At a Glance: Research Question ↔ Experiment ↔ Verdict

| Research Question | Controlled Benchmark | Verified Verdict (5 independent splits, Holm-Bonferroni) |
| --- | --- | --- |
| 1. Do classical-wrapper VQCs (Regime II) offer a verifiable advantage on MNIST? | `vqc_amp` vs. random quadratic & `poly2`; `patch_vqc` vs. 48-param filter; `hybrid_vqc` vs. dual-BN `hybrid_noq` | **No.** `vqc_amp` (`0.7760`) trails classical `poly2` (`0.9155`) and collapses `−36.70 pt` at 200 shots; matched `patch_vqc` (`0.8705`) ties `classical_patch_lowrank` (`0.8715`); `hybrid_noq` (`0.8310`) beats `hybrid_vqc` (`0.7600`). |
| 2. How do zero-wrapper quantum-native models (Regime III) behave on MNIST? | 4q deep re-uploading Born readout (`96q/0c`) vs. 96-param trig net; 6q entangling IQP kernel vs. RBF/Poly-2 | **Trainable without MLP crutches, parity with matched classical surrogates.** Pure re-uploading (`0.6175`) ties `classical_trig_net` (`0.6165`, `p_holm=1.0`); IQP (`0.7380`) ties `poly2` (`0.7415`) / `rbf` (`0.7545`) and exhibits exponential concentration at high $\alpha$. |
| 3. Where does a quantum-native algorithm (Regime III) genuinely excel? | 1D Cluster-Ising $n=8$ PBC SPT vs. non-SPT ground states: Cong-et-al. QCNN (`36q/0c/1 QWC`) vs. local observables & oracle | **Decisive, shot-resilient extraction of nonlocal order.** QCNN hits **`0.9967`** analytic, **`0.9950`** at 1000 shots, and **`0.9633`** at 200 shots (`p_holm=0.198`), beating 1-body observables (`0.6917` / `0.9167`, `p_holm ≤ 0.0068`) and matching the $O(2^n)$ full-state oracle (`1.0000`). |
| 4. How do real hardware coupling topologies and depolarizing noise change these findings? | 4 topologies (`all_to_all / grid_2d / heavy_hex / linear_chain`) SABRE SWAP routing + `default.mixed` ($p_{2q}\le 0.01$) | **Sparse topologies exponentially amplify kernel concentration on all-to-all circuits.** `heavy_hex`/`linear_chain` inflate IQP 2Q gates `2.90×`/`3.40×` (QFT `3.025×`/`3.625×`), collapsing IQP off-diag spread to `5.2%`/`3.4%` and accuracy to `0.5560`/`0.5420` (`−11.40 pt`, `p_holm=0.0342`); QCNN analytic boundary stays `0.9950–0.9967` (symmetric channels shrink margin without sign flip). |

---

## Three-Regime Controlled Benchmark (Track A + Track B)

### Track A: Classical Data — MNIST 10-Class (600 Train / 200 Val / 400 Test, 5 Seeds, [`docs/three_regimes_metrics.json`](docs/three_regimes_metrics.json))

| Regime | Model | Quantum Params | Classical Wrapper | PCA Fitted | QWC Bases | Test Acc (Analytic) | Finite Shots / Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| **I · Classical** | `classical_poly2_unscaled` | 0 | 0 | 0 | 0 | **`0.9155 ± 0.0124`** | Exact dequantization of `qkernel_amp` (bitwise identical) |
| **I · Classical** | `classical_patch_lowrank` | 0 | 20,364 (48 filter) | 0 | 0 | `0.8715 ± 0.0157` | Exact parity with `patch_vqc_matched` (`batch=64`, BN, WD) |
| **II · Ported** | `patch_vqc_matched` | 48 | 20,316 | 0 | 3 | `0.8705 ± 0.0262` | Paired `−0.10 ± 1.82 pt` (`p_holm=1.0`; 0.24% quantum) |
| **I · Classical** | `hybrid_noq_matched` | 0 | 5,162 | 0 | 0 | `0.8310 ± 0.0569` | `stabilize=True` (dual BN + matched 32d bridge) |
| **II · Ported** | `hybrid_vqc_matched` | 48 | 5,114 | 0 | 3 | `0.7600 ± 0.0904` | Paired `−7.10 ± 10.34 pt` (`p_holm=0.798`; 0.93% quantum) |
| **II · Ported** | `vqc_amp_matched` (10q 40d) | 60 | 1,642 | 0 | 3 | `0.7760 ± 0.0314` | **200 shots: `0.4090` (`−36.70 pt`) · 1000 shots: `0.6360` (`−14.00 pt`)** |
| **I · Classical** | `classical_random_quad` (40d) | 0 | 1,642 | 0 | 0 | `0.6515 ± 0.0516` | 0 learned projection params; `vqc_amp` `+12.45 pt` (`p_holm=0.024`) |
| **I · Classical** | `classical_rbf_pca6` / `poly2_pca6` | 0 | 0 | 5,488 | 0 | `0.7545` / `0.7415` | Identical per-PC standardized PCA-6, Val-tuned `gamma/C` |
| **III · Native** | `qnative_iqp_kernel` (6q 2-rep) | 0 | 0 | 5,488 | 1 | `0.7380 ± 0.0339` | Real entangling IQP kernel; `offdiag` drops `0.875→0.019` over $\alpha\in[0.15,0.75]$ |
| **III · Native** | `qnative_reupload_fidelity` (4q L=4) | **96** | **0** | 10,192 | 1 | `0.6175 ± 0.0307` | Zero classical MLP wrapper; pure Born readout (100% quantum) |
| **I · Classical** | `classical_trig_net` (PCA-12) | 0 | 96 | 10,192 | 0 | `0.6165 ± 0.0347` | Matched 96-param trig net; paired `+0.10 ± 4.39 pt` (`p_holm=1.0`) |

### Track B: Native Quantum States — 1D Cluster-Ising SPT Phases ($n=8$ PBC, 120/60/120, 5 Seeds)

| Regime | Model | Quantum Params | Classical Params | QWC Bases | Test Acc (Analytic) | Paired vs. `qcnn_spt` (Holm) / Finite Shots |
| --- | --- | ---: | ---: | ---: | --- | --- |
| **III · Native** | **`qcnn_spt` (Cong-et-al. QCNN)** | **36** | **0** | **1** | **`0.9967 ± 0.0045`** | **200 shots: `0.9633 ± 0.0356` (`p_holm=0.198`) · 1000 shots: `0.9950 ± 0.0045`** |
| **I · Classical** | `classical_1body_logreg` (24d) | 0 | 25 | — | `0.6917 ± 0.0899` | QCNN **`+30.50 ± 9.10 pt`** (CI `[+19.20, +41.80]`, `p_holm=0.00678` ✓) |
| **I · Classical** | `classical_1body_mlp` (24d) | 0 | 833 | — | `0.9167 ± 0.0118` | QCNN **`+8.00 ± 0.95 pt`** (CI `[+6.82, +9.18]`, `p_holm=0.000235` ✓) |
| **I · Classical** | `classical_2body_mlp` (48d) | 0 | 1,601 | — | `0.9517 ± 0.0379` | QCNN `+4.50 ± 3.47 pt` (CI `[+0.20, +8.80]`, `p_holm=0.132`) |
| **I · Classical** | `classical_full_state_mlp` ($2^8$ oracle) | 0 | 8,257 | — | `1.0000 ± 0.0000` | `−0.33 ± 0.45 pt` (`p_holm=0.198`; requires $O(2^n)$ state tomography) |

---

## Hardware Coupling Topology SWAP Routing & `default.mixed` Noise ([`docs/hardware_topology_metrics.json`](docs/hardware_topology_metrics.json))

Bidirectional SABRE-style SWAP routing (1 SWAP = 3 physical CNOTs, max noiseless statevector error `≤ 2.776e-16` across all 20 circuit×topology pairs) and PennyLane `default.mixed` density-matrix simulation (5 seeds, $p_{2q} = 0.010$):

| Topology (Max Degree) | `qcnn_spt` 8q (Phys 2Q / Depth) | `qcnn_spt` Acc ($p_{2q}=0.01$) | `iqp_kernel` 6q (Phys 2Q / Depth) | `iqp_kernel` Acc ($p_{2q}=0 \to 0.01$) | IQP Spread `std_ratio` | `qft_8` 8q (Phys 2Q / Depth) |
| --- | --- | --- | --- | --- | ---: | --- |
| `all_to_all` ($n-1$) | `47 (1.00×) / 19` | `0.9967 ± 0.0045` | `30 (1.00×) / 15` | `0.6560 → 0.6100` | `0.3338` | `40 (1.00×) / 16` |
| `grid_2d` (3) | `56 (1.19×) / 25` | `0.9967 ± 0.0045` | `66 (2.20×) / 41` | `0.6560 → 0.5700` | `0.0993` | `91 (2.28×) / 59` |
| `heavy_hex` (≤3) | `80 (1.70×) / 51` | `0.9950 ± 0.0045` | `87 (2.90×) / 63` | `0.6560 → 0.5560` | `0.0525` | `121 (3.03×) / 93` |
| `linear_chain` (2) | `86 (1.83×) / 37` | `0.9950 ± 0.0045` | `102 (3.40×) / 59` | `0.6560 → 0.5420` (`p_holm=0.0342` ✓) | **`0.0345`** | `145 (3.63×) / 79` |

---

## Historical Archive & Reproduction

- **Rounds 1–8 Archives**: Full MNIST (60k/10k, [`docs/full_metrics.json`](docs/full_metrics.json)) 165d amplitude VQC hit `0.9421 ± 0.0008` (vs. classical `pca165` `0.9554` / frozen random unitary `0.9440`); Round-8 audit ([`docs/audit_metrics.json`](docs/audit_metrics.json)) verified `qkernel_amp == plain_poly2_c0` (`0.9210`) and `PatchLinearClassifier` (`0.8910`) tying `PatchVQC` (`0.8800`); full log in [`docs/REVIEW.md`](docs/REVIEW.md).

```bash
python3 -m venv .venv
.venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e ".[dev]"

# 1. Offline unit & regression test suite (9 tests: topology/SABRE/default.mixed & all --quick contracts)
.venv/bin/python -m pytest -v tests/smoke_test.py

# 2. Three-regime benchmark & hardware coupling topology/noise benchmark (real MNIST IDX, CPU-reproducible)
.venv/bin/python experiments/benchmark_three_regimes.py --data-source idx --workers 5 --archive docs/three_regimes_metrics.json
.venv/bin/python experiments/benchmark_hardware_topology.py --data-source idx --workers 20 --archive docs/hardware_topology_metrics.json

# 3. Quick offline smoke checks (write to results/ only, never overwrite docs/)
.venv/bin/python experiments/benchmark_three_regimes.py --quick
.venv/bin/python experiments/benchmark_hardware_topology.py --quick
```

### Repository Layout & Related Links

- `qml_lab/`: `quantum.py`, `models.py`, `quantum_phases.py` (1D Cluster-Ising + QCNN), `hardware_topology.py` (4 topologies + bidirectional SABRE router + `default.mixed` noise), `data.py`, `train.py`, `utils.py`, `viz.py`.
- `docs/reports/` (3 standalone Chinese IMRaD experimental reports), `docs/figures/` (4 publication 300-dpi English-labeled figures), `docs/*_metrics.json`, `tests/smoke_test.py`.
- Sister benchmark: [quantum-llm](https://github.com/quantum-machine-learning-code/quantum-llm) · Organization: [quantum-machine-learning-code](https://github.com/quantum-machine-learning-code) · [MIT License](LICENSE).
