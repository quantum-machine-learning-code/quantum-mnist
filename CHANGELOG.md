# Changelog

All notable changes to `quantum-mnist` will be documented in this file.

## [Unreleased]

### Fixed
- Fixed 2× angle error in `experiments/resolve_open_issues.py` (`qkernel_angle`: `* 0.25 -> * 0.5`, exact `RY` fidelity $\prod_i \cos^2((x_i-x'_i)/2)$, verified `< 1e-12` against `default.qubit`).
- Added `standardize_per_pc=True` (`tanh(z)*(pi/2)`) to `qml_lab/data.py::load_mnist` eliminating the global scalar min-max `-0.815 rad` shift across PCA components, plus strictly disjoint `load_mnist_tvt`.
- Fixed `HybridRichClassifier` (`qml_lab/models.py`) so `quantum=False, stabilize=True` retains `post_norm = BatchNorm1d(dim)` and supports parameter/width-matched `match_readout_dim=True`; guarded against double-`tanh` squashing with `trainable_reupload=True`; renamed misnamed `RichVQCClassifier.post_ln` to `post_bn` (with backward-compatible alias).
- Fixed `experiments/audit_round8.py` `PatchLinearClassifier` parity with `PatchVQC` (`batch_size=64` and `no_decay_modules=("proj",)` via `qml_lab/train.py`) and refreshed `docs/audit_metrics.json`.
- Corrected 5 mismatched per-seed rows and QSVC runtimes in `docs/REVIEW.md` §14 to match `docs/open_issues_metrics.json` verbatim, and added an explicit archival note for `docs/full_metrics.json`.

### Added
- Hardware coupling topology & noise module `qml_lab/hardware_topology.py` (`networkx>=3.0` added to `pyproject.toml` and `requirements.txt`): 4 canonical coupling graphs (`all_to_all`, `linear_chain`, `grid_2d`, and connected bipartite `heavy_hex` with max degree ≤ 3), deterministic bidirectional SABRE-style SWAP router (1 SWAP = 3 physical CNOTs, max noiseless statevector error `2.776e-16`), 5 real circuit extractors (`vqc_amp`, `patch_vqc`, `qnative_iqp_kernel`, `qcnn_spt`, `qft_8`), and PennyLane `default.mixed` depolarizing + readout-error density-matrix simulation.
- Hardware topology benchmark script `experiments/benchmark_hardware_topology.py`, archived 5-seed metrics [`docs/hardware_topology_metrics.json`](docs/hardware_topology_metrics.json), 4 publication-quality 300-dpi figures in `docs/figures/` (`three_regimes_mnist.png`, `qcnn_spt_phases.png`, `topology_routing_overhead.png`, `topology_noise_degradation.png`), and hardware-topology regression tests in `tests/smoke_test.py`.
- Three standalone formal Chinese IMRaD experimental reports with QML reproducibility checklists in `docs/reports/` (`01_mnist_three_regimes_report.md`, `02_cluster_ising_qcnn_report.md`, `03_hardware_topology_noise_report.md`), linked from synced [`README.md`](README.md) and [`README_EN.md`](README_EN.md).
- Programmatic `count_qwc_groups`, finite-shot QWC evaluation (`evaluate_rich_layer_shots`), and real PennyLane entangling IQP feature-map kernel (`iqp_feature_states` / `iqp_kernel_matrix`) in `qml_lab/quantum.py`.
- `parameter_breakdown`, `assert_matched_config`, dequantized classical surrogates (`RandomQuadraticClassifier`, `PatchLowRankClassifier`, `ClassicalTrigNet`), and zero-wrapper `DataReuploadingFidelityClassifier` in `qml_lab/models.py`.
- Native quantum-state benchmark module `qml_lab/quantum_phases.py` implementing 1D Cluster-Ising ($n=8$ PBC) exact sparse diagonalization and a true Cong-et-al. `QCNNPhaseClassifier` (36 quantum params, 0 classical MLP params, 1 QWC basis).
- Controlled Three-Regime benchmark `experiments/benchmark_three_regimes.py` (`docs/three_regimes_metrics.json`) with strict Train/Val/Test splits, 5 seeds, finite-shot checks, and Holm-Bonferroni correction, plus regression tests in `tests/smoke_test.py`.
- Concise, 100%-synced bilingual [`README.md`](README.md) and [`README_EN.md`](README_EN.md).

## [0.2.0] - 2026-11-14

### Added
- Packaging metadata via [`pyproject.toml`](pyproject.toml) (`qml_lab` package, Python `>=3.11`, optional `[dev]` dependencies, pytest and ruff configuration).
- GitHub Actions CI workflow ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) running `ruff check .` and `pytest -v tests/smoke_test.py` on Python 3.11 and 3.12 with CPU-only PyTorch + `torchvision`.
- Monthly Dependabot automation ([`.github/dependabot.yml`](.github/dependabot.yml)) for `pip` and `github-actions`.
- English documentation [`README_EN.md`](README_EN.md) with bidirectional language switcher in [`README.md`](README.md), plus [`CONTRIBUTING.md`](CONTRIBUTING.md).
- Rounds 5–8 benchmark & audit scripts (`probe_information.py`, `push_ceiling.py`, `classical_budget.py`, `why_low_acc.py`, `train_full_mnist.py`, `resolve_open_issues.py`, `audit_round8.py`) and archived JSON/figures in `docs/`.

### Fixed & Changed
- Relaxed `torchvision>=0.29` to `torchvision>=0.19` in [`requirements.txt`](requirements.txt) for broader compatibility with `torch>=2.4` and Python 3.11+.
- Wrapped temporary `self.pre_net.eval()` toggling during single-sample `BatchNorm1d` forward passes in `try ... finally: self.pre_net.train(was_training)` inside [`qml_lab/models.py`](qml_lab/models.py).
- Respected CLI `--out` under `--quick` in `experiments/probe_information.py`, `experiments/classical_budget.py`, and `experiments/resolve_open_issues.py`, and updated `check_quick_scripts()` in [`tests/smoke_test.py`](tests/smoke_test.py) to use a temporary directory so tests never pollute `results/`.
- Cleaned up ruff lint findings (`F401`, `E702`, `E741`) across `qml_lab/`, `experiments/`, and `tests/`.

## [0.1.0] - 2026-11-06

### Added
- Initial MNIST quantum vs. classical benchmark harness (`qml_lab/`), PCA/amplitude encoding comparisons, capacity scans, and multi-seed readout diagnostics.
