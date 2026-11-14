# Quantum MNIST

English | [简体中文](README.md)

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![PennyLane](https://img.shields.io/badge/PennyLane-0.45-9b59b6)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![License](https://img.shields.io/badge/license-MIT-green)
![Hardware](https://img.shields.io/badge/quantum%20hardware-not%20required-success)

> Variational quantum circuits (VQCs) vs. input-matched / parameter-matched / data-budget-matched classical MLPs, CNNs, and classical quadratic kernels on MNIST. **Reproducible, pure state-vector simulation, no quantum hardware required.**

This repository focuses solely on quantum machine learning benchmarks for MNIST handwritten digit classification. The sister repository [quantum-llm](https://github.com/quantum-machine-learning-code/quantum-llm) studies modular addition and toy character-level language models; both codebases are self-contained.

---

## Executive Summary (2026-11-07 · Rounds 1–8 Including Full 60k/10k MNIST)

> **External Audit Note (2026-11-07 · [REVIEW.md §15](docs/REVIEW.md), [`audit_round8.py`](experiments/audit_round8.py), [`docs/audit_metrics.json`](docs/audit_metrics.json))**: Round-8 numbers are bitwise reproducible, with three tightened interpretations:
> 1. `qkernel_amp` (`0.9210`) is **seed-for-seed identical to a classical quadratic kernel without `StandardScaler`** (`qkernel_patch` `0.9310` likewise has an exact classical quadratic computation) and must not be conflated with end-to-end quantum models.
> 2. Replacing the 8-qubit shared circuit in `PatchVQC` (`0.8800 ± 0.0112`) with a classical `Linear(196, 32) + Tanh` block while keeping everything else unchanged gives **`0.8920 ± 0.0067`** (paired `+1.20 pt`, CI crosses zero, ~50× faster) — confirming the gain comes from the 2D local-patch inductive bias rather than a verifiable quantum contribution.
> 3. After Bonferroni correction across the 20 Round-8 paired tests, **2/20** remain significant (down from 13/20 uncorrected). Moreover, kernel seeds reflect dataset resampling whereas VQC seeds reflect model initialization/batch shuffle, so the two families cannot be paired cross-group.

All small-sample tables below use real MNIST IDX images (10 classes, 200 fixed test images, **5 seeds**, Python 3.13 / PennyLane 0.45.1 / PyTorch 2.15 CPU), archived in `docs/*_metrics.json`. **Round 7** evaluates on the full **60,000 train / 10,000 test** split (3 epochs, batch 128, 3 seeds, `default.qubit + backprop`), which is not directly comparable digit-for-digit to the 200-image test splits.

### Full MNIST (Round 7, 2026-11-06 · 60,000 Train / 10,000 Test, 3 Seeds)

| Model | Parameters | Full Test Accuracy (3 seeds) | Time / seed |
| --- | --- | --- | --- |
| **Amp10 / 165-dim readout (Quantum)** | 5,702 | **`0.9421 ± 0.0008`** | 2,462–2,654 s |
| Amp10 / 40-dim readout (Quantum) | 1,702 | `0.9014 ± 0.0020` | 633–637 s |
| Classical PCA-165 + Matched MLP Head | 5,642 | **`0.9554 ± 0.0028`** | 2.2–2.4 s |
| Classical MLP-784 | 25,450 | `0.9546 ± 0.0012` | 2.8–2.9 s |
| Frozen random unitary 165-dim readout + sklearn probe | — | `0.9440` (`logreg 0.9027` / `MLP 0.9440`) | 634 s |

- Full-scale data raises 165-dim amplitude VQC from `0.8450` (600 train, 200 test) to **`0.9421 ± 0.0008`**, narrowing the gap with matched classical PCA-165 (`0.9554`) to **`1.33 pt`** (paired 95% CI `[+0.45, +2.20] pt`, 3/3 seeds in favor of classical) at ~**1,000×** higher runtime. **No quantum advantage is claimed.**
- `lightning.qubit` did not support `backprop` on this circuit and `adjoint` ran at 11.9 samples/s vs. 98.8 samples/s on `default.qubit`.

### Small-Sample Benchmarks (Rounds 1–5 & 8 · 200 Test Images, 5 Seeds)

| Model | Train Images | Parameters | Test Accuracy (5 seeds) | Time / seed |
| --- | --- | --- | --- | --- |
| Legacy baseline: Angle-4 + Linear head | 600 | 74 | `0.4320 ± 0.0375` | 16.1 s |
| Round 2: Amp10 + MLP ×5 (`PauliZ` only) | 600 | 742 | `0.6640 ± 0.0216` | 48.2 s |
| Round 4: Amp10 + 40-dim readout (`z+x+y+zz`) | 600 | 1,702 | `0.7720 ± 0.0115` | 96.4 s |
| Round 5: Amp10 + 75-dim readout (all `ZZ`) | 600 | 2,822 | `0.8250 ± 0.0127` | 155 s |
| **Round 5: Amp10 + 165-dim (`ZZ+XX+YY`)** | 600 | 5,702 | **`0.8450 ± 0.0100`** | 556 s |
| **Round 5: 165-dim + 1,800 train / 8 epochs** | 1,800 | 5,702 | **`0.8660 ± 0.0185`** | 530 s |
| **Round 8: Local `PatchVQC` (shared 8q, 3×3 14×14 patches)** | 600 | 20,364 (48 quantum) | **`0.8800 ± 0.0112`** | **117 s** |
| Round 8: Analytic fidelity QSVC (10q global / 8q 9-patch; classically computable) | 600 | — | `0.9210 ± 0.0074` / `0.9310 ± 0.0164` | 0.01 s |
| Round 8: Stabilized serial hybrid `lightCNN + 8q VQC` (no skip) | 600 | 5,162 | `0.7650 ± 0.0464` | 116 s |
| Round 8: Angle-10 + Pérez-Salinas trainable re-upload (`Z` / 40-dim) | 600 | 782 / 1,742 | `0.6580 ± 0.0246` / `0.7470 ± 0.0214` | 115 / 207 s |
| Classical MLP-784 (same batch) | 600 | 25,450 | `0.8580 ± 0.0130` | 0.8 s |
| Classical PCA-165 + matched head | 600 | 5,642 | `0.8440 ± 0.0089` | 0.7 s |
| **Classical PCA-165 + matched head (budget-matched)** | 1,800 | 5,642 | **`0.9030 ± 0.0097`** | 0.7 s |

---

## Core Diagnostic Findings (Rounds 3–8)

1. **Readout bottleneck, not barren plateaus (Rounds 3–5, [`probe_information.py`](experiments/probe_information.py), [`break_readout.py`](experiments/break_readout.py), [`push_ceiling.py`](experiments/push_ceiling.py))**:
   - Measuring only $n=10$ `PauliZ` observables caps frozen readout separability at `0.679` (vs. `0.855` on raw pixels). Expanding to 40 and 165 observables keeps quantum circuit parameters fixed at **60** while recovering frozen separability (`0.679 → 0.811 → 0.890`) and lifting 5-seed accuracy (`0.6640 → 0.7720 → 0.8450`).
   - Quantum gradient norms actually grow (`0.048 → 0.897`) while weights drift only `6.9%`, ruling out barren plateaus; even an untrained random unitary with 165 observables yields a `0.860` frozen probe score.
   - At equal data + near-equal parameter budget (`1,800` images, 8 epochs), classical `PCA-165 + MLP` reaches **`0.9030 ± 0.0097`**, still `3.7 pt` ahead of `b165_d1800` (`0.8660`) and ~750× faster (`0.7 s` vs. `530 s`).
2. **Homogeneous quadratic ceiling & lack of translation invariance (Round 6, [`why_low_acc.py`](experiments/why_low_acc.py))**:
   - `⟨x|U† P_j U|x⟩` under amplitude encoding computes homogeneous quadratic polynomials of normalized pixels. Under unified probing (`StandardScaler + C=10 + coef0=1`), 600-sample 165-dim quantum features (`0.875`) match full classical `poly-2` across all 307,720 monomials (`0.880`).
   - ±1px shifts preserve pixel inner product `0.673`, yet perturb 165-dim quantum features by `1.20×` intra-class dispersion — explaining why ±1px shift augmentation hurts amplitude VQC (`-4.40 pt`) while helping local receptive fields.
   - Global L2 norm `‖x‖₂` is separable alone at `0.215`, yet concatenating it back into 165-dim end-to-end training changes accuracy by only `+0.10 ± 1.47 pt`.
3. **Resolving four open issues in Round 8 ([`resolve_open_issues.py`](experiments/resolve_open_issues.py))**:
   - **Local PatchVQC**: Splitting 28×28 images into 3×3 overlapping 14×14 windows (`stride=7`) processed by a shared 8-qubit circuit (48 quantum parameters) reaches **`0.8800 ± 0.0112`** and runs **5.7× faster** than global 10q (`117 s` vs. `665 s`), though the classical ablation (`Linear(196, 32) + Tanh`) achieves `0.8920 ± 0.0067`.
   - **Serial Hybrid CNN+VQC Stabilization**: Adaptive pooling collapsed inter-sample std to `~0.003`, drowned out by `Linear` biases (`~0.15`) into constant angles (`0/5` collapse -> `0.5980 ± 0.1062`). Strictly serial `BatchNorm1d → Linear(bias=False) → BatchNorm1d → Tanh*(π/2) → 8q VQC(N(0,0.1²), 32d) → BatchNorm1d` (`HybridRichClassifier(..., stabilize=True)`, zero skip connections) lifts accuracy to **`0.7650 ± 0.0464`** (`0/5` collapses), on par with skipping the quantum block (`nq_light`: `0.7440 ± 0.0651`).
   - **Pérez-Salinas Trainable Data Re-Uploading**: Adding bounded inputs and per-layer trainable scale/shift $\text{RY}(\omega_{l,i}\tilde{x}_i + \theta_{l,i})$ reverses the `-11.00 pt` naive re-uploading drop (`0.5220 → 0.6580 ± 0.0246` with `Z`-only, and `0.7470 ± 0.0214` with 40-dim readouts).
   - **Analytic Quantum Kernels (QSVC)**: Achieves `0.9210` (10q global amplitude) and `0.9310` (8q 9-patch), equivalent to unscaled classical quadratic kernels (not end-to-end quantum circuits).

---

## Repository Structure

- `qml_lab/`: Reusable library modules (`data.py`, `quantum.py`, `models.py`, `train.py`, `utils.py`, `viz.py`).
- `experiments/`: Scripts for all 8 rounds (`train_mnist.py`, `train_amplitude.py`, `scan_capacity.py`, `train_multiseed.py`, `diagnose_ceiling.py`, `break_readout.py`, `probe_information.py`, `push_ceiling.py`, `classical_budget.py`, `why_low_acc.py`, `train_full_mnist.py`, `resolve_open_issues.py`, `audit_round8.py`).
- `tests/smoke_test.py`: Offline smoke & contract tests (uses synthetic fallback; no network downloads).
- `docs/`: Full Chinese methodological review ([`docs/REVIEW.md`](docs/REVIEW.md)), offline setup notes ([`docs/DEV_NOTES.md`](docs/DEV_NOTES.md)), archived JSON metrics (`*_metrics.json`), and figures (`docs/figures/`).

---

## Installation & Quickstart

Requires **Python 3.11+**. Default configurations run offline on CPU.

```bash
python3 -m venv .venv
# Install CPU-only PyTorch + torchvision first to avoid large CUDA wheels on Linux/CI:
.venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e ".[dev]"
```

Or install runtime dependencies with `pip install -r requirements.txt`. For offline real-MNIST runs, place the 4 MNIST IDX `.gz`/raw files in `data/MNIST/raw/` and pass `--data-source idx` (see [`docs/DEV_NOTES.md`](docs/DEV_NOTES.md)). Data fallback order is `idx → torchvision → openml → synthetic`.

### Offline Tests & Reproduction Commands

```bash
# Offline test suite (no internet required)
.venv/bin/python -m pytest -v tests/smoke_test.py
.venv/bin/python tests/smoke_test.py

# Quick end-to-end smoke checks (synthetic data, writes to results/ or --out)
.venv/bin/python experiments/train_mnist.py --quick
.venv/bin/python experiments/probe_information.py --quick
.venv/bin/python experiments/classical_budget.py --quick
.venv/bin/python experiments/resolve_open_issues.py --quick --no-figures

# Round 8 full reproduction (~19 min CPU, 12 workers) & audit (~20 s classical)
.venv/bin/python experiments/resolve_open_issues.py --data-source idx \
    --workers 12 --torch-threads 2 --archive docs/open_issues_metrics.json \
    --fig-dir docs/figures
.venv/bin/python experiments/audit_round8.py --data-source idx \
    --archive docs/audit_metrics.json

# Round 7 full MNIST (60k/10k, ~45 min CPU, 6 workers)
.venv/bin/python experiments/train_full_mnist.py --data-source idx \
    --workers 6 --torch-threads 4 --epochs 3 --seeds 0,1,2 \
    --batch-size 128 --archive docs/full_metrics.json --fig-dir docs/figures
```

---

## Related Repositories & License

- Sister benchmark: [quantum-llm](https://github.com/quantum-machine-learning-code/quantum-llm)
- Organization: [quantum-machine-learning-code](https://github.com/quantum-machine-learning-code)

Licensed under the [MIT License](LICENSE).
