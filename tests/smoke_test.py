"""离线冒烟测试：直接 python tests/smoke_test.py 运行，不依赖 pytest / 不联网。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from qml_lab.data import load_mnist, load_mnist_images, load_mnist_raw
from qml_lab.models import (
    CNNClassifier, HybridCNNClassifier, HybridRichClassifier, MLPClassifier,
    RichVQCClassifier, VQCClassifier,
)
from qml_lab.quantum import (
    make_amplitude_layer, make_rich_layer, n_readout,
)
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


def check_hybrid_and_images() -> None:
    """图像数据路径与混合 CNN 前端：形状、抽样一致性、前向与梯度。"""
    X, y, Xt, yt, meta = load_mnist_images(
        n_train=16, n_test=8, classes=2, source="synthetic", seed=0)
    Xr, yr, Xrt, yrt, _ = load_mnist_raw(
        n_train=16, n_test=8, classes=2, source="synthetic", seed=0)
    assert X.shape == (16, 1, 28, 28), f"图像形状错误 {X.shape}"
    assert meta["shape"] == "NCHW" and not meta["pca"]
    assert np.array_equal(y, yr) and np.array_equal(yt, yrt), "两条数据路径抽样不一致"
    flat = X.reshape(len(X), -1) * 255.0
    expected = flat / np.linalg.norm(flat, axis=1, keepdims=True)
    assert np.allclose(Xr, expected, atol=1e-4), "raw 与 images 的像素不一致"

    xb = torch.as_tensor(X[:4], dtype=torch.float32)
    for frontend in ("light", "heavy"):
        for quantum in (True, False):
            set_seed(0)
            model = HybridCNNClassifier(
                n_qubits=2, n_classes=2, n_layers=1, hidden=8,
                quantum=quantum, head="mlp", scale=1.0, frontend=frontend)
            z = model.quantum_features(xb)
            assert tuple(z.shape) == (4, 2), f"读出形状错误 {z.shape}"
            out = model(xb)
            assert tuple(out.shape) == (4, 2), f"logits 形状错误 {out.shape}"
            out.sum().backward()
            for name, p in model.named_parameters():
                assert p.grad is not None and bool(torch.isfinite(p.grad).all()), \
                    f"{frontend}/quantum={quantum} 的 {name} 梯度缺失或非有限"
        cnn = CNNClassifier(2, frontend=frontend)
        out = cnn(xb)
        assert tuple(out.shape) == (4, 2)
        out.sum().backward()


def check_readout_layers() -> None:
    """多可观测量 / 数据重上传：维度、梯度、与旧基线的逐位一致性。"""
    B, nq = 8, 4
    x = torch.randn(B, 16)
    x = x / x.norm(dim=1, keepdim=True)  # 2**4 = 16，振幅编码可装下
    xa = torch.randn(B, nq)

    # 回归：新层在 "z" 规格下必须与旧振幅层逐位相同
    set_seed(0)
    old = make_amplitude_layer(nq, 2, n_features=16)
    set_seed(0)
    new = make_rich_layer(nq, 2, "amplitude", "z", n_features=16)
    assert torch.equal(old.layer.weights, new.layer.weights), \
        "重构后量子初始权重与旧层不一致"
    with torch.no_grad():
        assert torch.allclose(old(x), new(x), atol=0, rtol=0), \
            "重构后前向输出与旧层不一致"

    for spec in ("z", "z+x", "z+x+y", "z+x+zz", "z+x+y+zz"):
        dim = n_readout(nq, spec)
        set_seed(0)
        layer = make_rich_layer(nq, 2, "amplitude", spec, n_features=16)
        out = layer(x)
        assert tuple(out.shape) == (B, dim), \
            f"{spec} 读出维度应为 {dim}，实际 {tuple(out.shape)}"
        out.pow(2).mean().backward()
        g = layer.layer.weights.grad
        assert g is not None and bool(torch.isfinite(g).all()) and float(g.norm()) > 0, \
            f"{spec} 梯度缺失 / 非有限 / 全零"

    # 重上传：参数 shape 与初始权重不变，但输入被每层重新嵌入
    set_seed(0)
    plain = make_rich_layer(nq, 2, "angle", "z+x")
    set_seed(0)
    reup = make_rich_layer(nq, 2, "angle", "z+x", reupload=True)
    assert torch.equal(plain.layer.weights, reup.layer.weights), \
        "重上传不应改变参数量或初始权重"
    out = reup(xa)
    assert tuple(out.shape) == (B, 2 * nq)
    out.pow(2).mean().backward()
    assert float(reup.layer.weights.grad.norm()) > 0, "重上传层梯度全零"

    # 分类器 + 量子层单独学习率
    y = torch.randint(0, 2, (B,))
    set_seed(0)
    model = RichVQCClassifier(nq, 2, 2, "amplitude", "z+x+zz",
                              n_features=16, head="mlp", scale=5.0)
    torch.nn.functional.cross_entropy(model(x), y).backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and bool(torch.isfinite(p.grad).all()), \
            f"RichVQC 的 {name} 梯度缺失或非有限"

    img = torch.rand(B, 1, 28, 28)
    set_seed(0)
    hyb = HybridRichClassifier(n_qubits=nq, n_classes=2, n_layers=2,
                               hidden=8, quantum=True, observables="z+x",
                               reupload=True, frontend="light")
    assert hyb.readout_dim == 2 * nq
    torch.nn.functional.cross_entropy(hyb(img), y).backward()
    for name, p in hyb.named_parameters():
        assert p.grad is not None and bool(torch.isfinite(p.grad).all()), \
            f"HybridRich 的 {name} 梯度缺失或非有限"
    hist = train_classifier(
        model=hyb, X_train=img, y_train=y, X_test=img, y_test=y,
        epochs=1, batch_size=4, lr=0.01, quantum_lr=0.001,
        verbose=False, eval_train=False,
    )
    assert _finite_all(hist["train_loss"]) and _finite_all(hist["test_acc"]), \
        "quantum_lr 参数分组导致训练指标非有限"


def main() -> None:
    check_classifier()
    check_heads_and_controls()
    check_hybrid_and_images()
    check_readout_layers()
    print("SMOKE OK")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"SMOKE FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
