# Changelog

All notable changes to `quantum-mnist` will be documented in this file.

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
