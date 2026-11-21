# Quantum MNIST · Controlled Three-Regime QML Benchmark

English | [简体中文](README.md)

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![PennyLane](https://img.shields.io/badge/PennyLane-0.45-9b59b6)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![License](https://img.shields.io/badge/license-MIT-green)
![Hardware](https://img.shields.io/badge/quantum%20hardware-not%20required-success)

> A strictly controlled benchmark separating **(I) Classical algorithms on classical compute**, **(II) Classical templates ported onto quantum circuits**, and **(III) Quantum-native algorithms on quantum circuits** across both classical images (MNIST) and native entangled quantum states (1D Cluster-Ising SPT phases). **Pure simulation, zero test peeking, full parameter accounting, real finite-shot evaluation.**

---

## At a Glance: Research Question ↔ Experiment ↔ Verdict

| Research Question | Controlled Benchmark ([`docs/three_regimes_metrics.json`](docs/three_regimes_metrics.json)) | Verified Verdict (5 independent splits, Holm-Bonferroni) |
| --- | --- | --- |
| 1. Do classical-wrapper VQCs (Regime II) offer a verifiable advantage on MNIST? | `vqc_amp` vs. fixed random quadratic & `poly2`; `patch_vqc` vs. 48-param classical filter; `hybrid_vqc` vs. BN- & width-matched `hybrid_noq` | **No.** `vqc_amp` (`0.7760`) beats untrained random quadratic features (`0.6515`, `p_holm=0.024`), but trails classical `poly2` (`0.9155`) and collapses `-36.70 pt` at 200 shots; matched `patch_vqc` (`0.8705`) ties 48-param classical patch filtering (`0.8715`); with matched BN, `hybrid_noq` (`0.8310`) outperforms `hybrid_vqc` (`0.7600`). |
| 2. How do zero-wrapper quantum-native models (Regime III) behave on MNIST? | 4q deep data re-uploading Born-rule classifier (`96q / 0c`) vs. 96-param `classical_trig_net`; 6q entangling IQP kernel (Val-tuned $\alpha$) vs. matched RBF/Poly-2 | **Trainable without MLP crutches, parity with matched classical surrogates.** Pure-quantum re-uploading (`0.6175`) ties 96-param `classical_trig_net` (`0.6165`, `+0.10 pt`); entangling IQP (`0.7380`) ties matched `poly2` (`0.7415`) / `rbf` (`0.7545`) while exposing exponential kernel concentration at high $\alpha$. |
| 3. Where does a quantum-native algorithm (Regime III) genuinely excel? | 1D Cluster-Ising $n=8$ PBC SPT vs. non-SPT (PM + AFM) ground states: Cong-et-al. QCNN (`36q / 0c / 1 QWC basis`) vs. local observables & full-state oracle | **Decisive, shot-resilient extraction of nonlocal order.** QCNN reaches **`0.9967`** analytic, **`0.9950`** at 1000 shots, and **`0.9633`** at 200 shots (no significant drop), Holm-significantly beating 1-body observables (`0.6917` / `0.9167`, `p_holm ≤ 0.0068`) and matching the exponential-tomography full-state oracle (`1.0000`) with 36 parameters and 1 measurement basis. |

---

## Three-Regime Controlled Benchmark (Track A + Track B)

**Protocol**: Every seed draws a strictly disjoint stratified `Train / Val / Test` split shared identically across all neural, VQC, kernel, and QCNN models (unified seed semantics). All model selection (IQP bandwidth $\alpha$, SVM $C$, best checkpoint) uses `Val` only; `Test` is evaluated once. Every row reports explicit `Quantum / Classical Wrapper / Preprocessing (PCA)` parameter counts and programmatic QWC basis counts via `qml.pauli.group_observables`.

### Track A: Classical Data — MNIST 10-Class (600 Train / 200 Val / 400 Test, 5 Seeds)

| Regime | Model | Quantum Params | Classical Wrapper | PCA Fitted | QWC Bases | Test Acc (Analytic) | Finite Shots / Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| **I · Classical** | `classical_poly2_unscaled` | 0 | 0 | 0 | 0 | **`0.9155 ± 0.0124`** | Exact dequantization of `qkernel_amp` (bitwise identical) |
| **I · Classical** | `classical_patch_lowrank` | 0 | 20,364 (48 filter) | 0 | 0 | `0.8715 ± 0.0157` | Exact parity with `patch_vqc_matched` (`batch=64`, BN, WD) |
| **II · Ported** | `patch_vqc_matched` | 48 | 20,316 | 0 | 3 | `0.8705 ± 0.0262` | Paired `−0.10 ± 1.82 pt` (`p_holm=1.0`; 0.24% quantum) |
| **I · Classical** | `hybrid_noq_matched` | 0 | 5,162 | 0 | 0 | `0.8310 ± 0.0569` | `stabilize=True` (dual BN + matched 32d bridge) |
| **II · Ported** | `hybrid_vqc_matched` | 48 | 5,114 | 0 | 3 | `0.7600 ± 0.0904` | Paired `−7.10 ± 10.34 pt` (`p_holm=0.798`; 0.93% quantum) |
| **II · Ported** | `vqc_amp_matched` (10q 40d) | 60 | 1,642 | 0 | 3 | `0.7760 ± 0.0314` | **200 shots: `0.4090` (`−36.70 pt`) · 1000 shots: `0.6360` (`−14.00 pt`)** |
| **I · Classical** | `classical_random_quad` (40d) | 0 | 1,642 | 0 | 0 | `0.6515 ± 0.0516` | 0 learned projection params; `vqc_amp` `+12.45 pt` (`p_holm=0.024`) |
| **I · Classical** | `classical_rbf_pca6` | 0 | 0 | 5,488 | 0 | `0.7545 ± 0.0287` | Identical per-PC standardized PCA-6, Val-tuned `gamma/C` |
| **I · Classical** | `classical_poly2_pca6` | 0 | 0 | 5,488 | 0 | `0.7415 ± 0.0212` | Identical per-PC standardized PCA-6, Val-tuned `C` |
| **III · Native** | `qnative_iqp_kernel` (6q 2-rep) | 0 | 0 | 5,488 | 1 | `0.7380 ± 0.0339` | Real entangling IQP circuit; `offdiag` drops `0.875→0.019` over $\alpha\in[0.15,0.75]$ |
| **III · Native** | `qnative_reupload_fidelity` (4q L=4) | **96** | **0** | 10,192 | 1 | `0.6175 ± 0.0307` | Zero classical MLP wrapper; pure Born-rule readout (100% quantum) |
| **I · Classical** | `classical_trig_net` (PCA-12) | 0 | 96 | 10,192 | 0 | `0.6165 ± 0.0347` | Matched 96-param trig net; paired `+0.10 ± 4.39 pt` (`p_holm=1.0`) |

### Track B: Native Quantum-State Data — 1D Cluster-Ising SPT Phases ($n=8$ PBC, 120/60/120, 5 Seeds)

Hamiltonian $H(h_1, h_2) = -\sum_{i=0}^{7} Z_{i-1}X_i Z_{i+1} - h_1\sum_i X_i - h_2\sum_i X_i X_{i+1}$ (exact sparse diagonalization for SPT vs. non-SPT [paramagnetic + antiferromagnetic] ground states $|\psi\rangle$):

| Regime | Model | Quantum Params | Classical Params | QWC Bases | Test Acc (Analytic) | Paired vs. `qcnn_spt` (Holm) / Finite Shots |
| --- | --- | ---: | ---: | ---: | --- | --- |
| **III · Native** | **`qcnn_spt` (Cong-et-al. QCNN)** | **36** | **0** | **1** | **`0.9967 ± 0.0045`** | **200 shots: `0.9633 ± 0.0356` (`p_holm=0.198`) · 1000 shots: `0.9950 ± 0.0045`** |
| **I · Classical** | `classical_1body_logreg` (24d) | 0 | 25 | — | `0.6917 ± 0.0899` | QCNN **`+30.50 ± 9.10 pt`** (CI `[+19.20, +41.80]`, `p_holm=0.00678` ✓) |
| **I · Classical** | `classical_1body_mlp` (24d) | 0 | 833 | — | `0.9167 ± 0.0118` | QCNN **`+8.00 ± 0.95 pt`** (CI `[+6.82, +9.18]`, `p_holm=0.000235` ✓) |
| **I · Classical** | `classical_2body_mlp` (48d) | 0 | 1,601 | — | `0.9517 ± 0.0379` | QCNN `+4.50 ± 3.47 pt` (CI `[+0.20, +8.80]`, `p_holm=0.132`) |
| **I · Classical** | `classical_full_state_mlp` ($2^8$ oracle) | 0 | 8,257 | — | `1.0000 ± 0.0000` | `−0.33 ± 0.45 pt` (`p_holm=0.198`; requires $O(2^n)$ state tomography) |

**Physics Boundary**: On classical images, single-copy `AmplitudeEmbedding` followed by any CPTP channel (including QCNNs with mid-circuit pooling) remains strictly a homogeneous quadratic form in $\hat{x}$ and hides $O(2^n)$ state preparation. On native SPT quantum states $|\psi\rangle$, no classical data-loading bottleneck exists, and coherent $8\to 4\to 2\to 1$ quantum pooling extracts nonlocal string order directly with 36 parameters and 1 measurement basis.

---

## Historical Archive Pointer (Rounds 1–8 · Full Methodology in [`docs/REVIEW.md`](docs/REVIEW.md))

- **Full MNIST (Round 7, 60k/10k, 3 seeds, [`docs/full_metrics.json`](docs/full_metrics.json))**: 165-dim amplitude VQC (`60` quantum + `5,642` classical head, 3 QWC bases) reached `0.9421 ± 0.0008` (~42 min/seed) vs. classical `PCA-165 + MLP` (`5,642` MLP + **`130,144` fitted PCA weights**) at `0.9554 ± 0.0028` (~2.3 s/seed).
- **Round 8 Audit Recheck ([`docs/audit_metrics.json`](docs/audit_metrics.json), fixed `batch_size=64` & weight-decay parity)**: `qkernel_amp` (`0.9210`) equals unscaled classical `poly2(coef0=0)` seed-for-seed; `PatchLinearClassifier` scores `0.8910 ± 0.0143` vs. `PatchVQC` `0.8800 ± 0.0112` (`+1.10 pt`, CI crosses 0, ~80× faster); Bonferroni reduces Round-8 significance from 13/20 to 2/20.
- **All raw JSON archives**: `docs/{multiseed,ceiling,readout,info,push,budget,why_low,full,open_issues,audit,three_regimes}_metrics.json` and [`docs/REVIEW.md`](docs/REVIEW.md) (§1–§16).

---

## Installation & Reproduction

Requires **Python 3.11+**; runs offline on CPU (place the 4 MNIST IDX files in `data/MNIST/raw/`, see [`docs/DEV_NOTES.md`](docs/DEV_NOTES.md)).

```bash
python3 -m venv .venv
.venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e ".[dev]"

# 1. Offline unit & regression test suite (8 tests: angle kernel / per-PC scaling / QWC / QCNN / --quick)
.venv/bin/python -m pytest -v tests/smoke_test.py

# 2. Controlled three-regime benchmark (Track A MNIST + Track B Cluster-Ising SPT, ~5.5 min CPU / 5 workers)
.venv/bin/python experiments/benchmark_three_regimes.py --data-source idx \
    --workers 5 --torch-threads 2 --archive docs/three_regimes_metrics.json

# 3. Quick offline smoke check & refreshed Round-8 audit
.venv/bin/python experiments/benchmark_three_regimes.py --quick
.venv/bin/python experiments/audit_round8.py --data-source idx --archive docs/audit_metrics.json
```

### Repository Layout

- `qml_lab/`: `quantum.py` (VQC / `count_qwc_groups` / finite-shot evaluator / entangling IQP kernel), `models.py` (classifiers / dequantized surrogates / `DataReuploadingFidelityClassifier` / `parameter_breakdown` / `assert_matched_config`), `quantum_phases.py` (1D Cluster-Ising sparse diagonalization + Cong-et-al. QCNN), `data.py` (per-PC standardization + disjoint TVT splits), `train.py`, `utils.py`, `viz.py`.
- `experiments/`: `benchmark_three_regimes.py` (primary benchmark), `audit_round8.py`, plus historical Rounds 1–8 scripts.
- `tests/smoke_test.py`, `docs/REVIEW.md`, `docs/DEV_NOTES.md`, `docs/*_metrics.json`.

---

## Related Repositories & License

- Sister benchmark: [quantum-llm](https://github.com/quantum-machine-learning-code/quantum-llm) · Organization: [quantum-machine-learning-code](https://github.com/quantum-machine-learning-code)
- Licensed under the [MIT License](LICENSE).
