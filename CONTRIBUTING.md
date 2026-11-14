# Contributing to quantum-mnist

Thank you for helping keep `quantum-mnist` reproducible and methodologically rigorous!

## Development Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e ".[dev]"
```

## Verification Before Opening a Pull Request

Run both lint checks and the offline smoke/contract test suite:

```bash
.venv/bin/python -m ruff check .
.venv/bin/python -m pytest -v tests/smoke_test.py
```

## Methodological & Coding Standards

1. **Controlled baselines**: Compare quantum models against classical baselines using identical train/test splits, random seeds (weight initialization verified where applicable), training budgets, and preprocessing pipelines.
2. **Preprocessing consistency**: Keep kernel/probe feature scaling (`StandardScaler`, `coef0`, regularization) identical across comparisons and report multiple-testing corrections (e.g., Bonferroni) alongside paired confidence intervals.
3. **Offline tests**: Everything in `tests/` must run offline without downloading MNIST; `synthetic` fallback data is strictly for smoke checks (`--quick`) and must never be used to report accuracy claims.
4. **Exception-safe state**: Restore PyTorch module `.training` states inside `try ... finally` blocks whenever temporarily switching modes (`eval()` / `train()`).
