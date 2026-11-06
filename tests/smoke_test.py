"""离线冒烟测试：直接 python tests/smoke_test.py 运行，不依赖 pytest / 不联网。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from qml_lab.data import load_mnist
from qml_lab.models import MLPClassifier, VQCClassifier
from qml_lab.train import train_classifier
from qml_lab.utils import set_seed


def _finite_all(values: object) -> bool:
    """判断一组 loss / 指标是否全部为有限数。"""
    return bool(np.isfinite(np.asarray(values, dtype=float)).all())


def check_classifier() -> None:
    X_train, y_train, X_test, y_test, _ = load_mnist(
        n_train=64, n_test=32, n_features=2, classes=2,
        source="synthetic", seed=0,
    )
    X = np.asarray(X_train, dtype=float)
    assert X.shape == (64, 2), f"X_train 形状应为 (64,2)，实际 {X.shape}"
    assert _finite_all(X), "X_train 含 NaN/Inf"
    assert bool((X >= -np.pi - 1e-6).all() and (X <= np.pi + 1e-6).all()), \
        "X_train 取值应落在 [-pi, pi]"
    y = np.asarray(y_train, dtype=float)
    assert bool(((y >= 0) & (y <= 1)).all()), "标签应落在 0..1"

    for name, model in [("VQC", VQCClassifier(2, 2, 1)), ("MLP", MLPClassifier(2, 2, 4))]:
        hist = train_classifier(
            model=model,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            epochs=1, batch_size=8, lr=0.05, seed=0,
            eval_batch_size=8, verbose=False,
        )
        assert _finite_all(hist["train_loss"]), f"{name} train_loss 不是有限数: {hist['train_loss']}"
        accs = np.asarray(hist["test_acc"], dtype=float)
        assert _finite_all(accs), f"{name} test_acc 不是有限数"
        assert bool(((accs >= 0) & (accs <= 1)).all()), f"{name} test_acc 应落在 [0,1]"
        acc = float(hist["final_test_acc"])
        assert 0.0 <= acc <= 1.0, f"{name} final_test_acc 应落在 [0,1]，实际 {acc}"


def check_heads_and_controls() -> None:
    """分类头 / 输出缩放 / 轻量训练路径 / 构造前设种子的受控性。"""
    X_train, y_train, X_test, y_test, _ = load_mnist(
        n_train=32, n_test=16, n_features=2, classes=2,
        source="synthetic", seed=0,
    )
    for head, scale in [("linear", 1.0), ("mlp", 5.0), ("mlp", "learnable")]:
        model = VQCClassifier(2, 2, 1, head=head, scale=scale, hidden=8)
        out = model(torch.as_tensor(X_train[:4], dtype=torch.float32))
        assert tuple(out.shape) == (4, 2), f"{head}/{scale} logits 形状错误"
        out.sum().backward()
        for name, p in model.named_parameters():
            assert p.grad is not None and bool(torch.isfinite(p.grad).all()), \
                f"{head}/{scale} 的 {name} 梯度缺失或非有限"

    model = VQCClassifier(2, 2, 1)
    hist = train_classifier(
        model=model, X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        epochs=1, batch_size=8, lr=0.05, seed=0,
        eval_batch_size=8, verbose=False, eval_train=False,
    )
    assert all(v is None for v in hist["train_acc"]), \
        "eval_train=False 时 train_acc 应标记为未测量"
    assert _finite_all(hist["train_loss"]) and _finite_all(hist["test_acc"])

    # TorchLayer 权重在构造时消耗全局 RNG：构造前重置种子必须得到完全相同的初值
    set_seed(0)
    a = VQCClassifier(2, 2, 1).q.layer.weights.detach().clone()
    set_seed(0)
    b = VQCClassifier(2, 2, 1).q.layer.weights.detach().clone()
    assert torch.equal(a, b), "构造前 set_seed 未能固定 TorchLayer 初始权重"


def main() -> None:
    check_classifier()
    check_heads_and_controls()
    print("SMOKE OK")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"SMOKE FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
