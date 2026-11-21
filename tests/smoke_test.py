"""离线冒烟测试：支持直接 python tests/smoke_test.py 或 pytest 运行，不联网。"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from qml_lab.data import (
    fetch_mnist_raw, l2_normalize, load_mnist,
    load_mnist_images, load_mnist_raw, load_mnist_tvt, stratified_subset,
)
from qml_lab.models import (
    AmplitudeVQCClassifier, ClassicalTrigNet, CNNClassifier,
    DataReuploadingFidelityClassifier, HybridCNNClassifier,
    HybridRichClassifier, MLPClassifier, PatchLowRankClassifier,
    PatchVQCClassifier, RandomQuadraticClassifier, RichVQCClassifier,
    VQCClassifier, assert_matched_config, parameter_breakdown,
)
from qml_lab.quantum import (
    count_qwc_groups, evaluate_rich_layer_shots, iqp_feature_states,
    iqp_kernel_matrix, make_amplitude_layer, make_rich_layer,
    n_quantum_params, n_readout,
)
from qml_lab.quantum_phases import (
    QCNNPhaseClassifier, cluster_ising_ground_state,
    generate_cluster_ising_dataset,
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

    vqc = VQCClassifier(2, 2, 1)
    qf = vqc.quantum_features(torch.as_tensor(X_train[:4], dtype=torch.float32))
    assert tuple(qf.shape) == (4, 2) and qf.dtype == torch.float32

    for name, model in [("VQC", vqc), ("MLP", MLPClassifier(2, 2, 4))]:
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
        weight_decay=1e-4, cosine_lr=True,
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
    """图像数据路径与混合 CNN 前端（含 stabilize=True）：形状、抽样一致性、前向与梯度。"""
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

    # 测试稳定化混合架构 (stabilize=True)
    set_seed(0)
    hyb_stab = HybridRichClassifier(
        n_qubits=3, n_classes=2, n_layers=1, hidden=8,
        quantum=True, head="mlp", scale=1.0, frontend="light",
        observables="z+x+y+zzall", stabilize=True,
    )
    assert hyb_stab.post_norm is not None, "stabilize=True 应启用读出 LayerNorm"
    assert float(hyb_stab.q.layer.weights.detach().std().item()) < 0.2, "stabilize=True 应使用近零初始化"
    z_stab = hyb_stab.quantum_features(xb)
    assert tuple(z_stab.shape) == (4, n_readout(3, "z+x+y+zzall"))
    logits_stab = hyb_stab(xb)
    logits_stab.sum().backward()
    for name, p in hyb_stab.named_parameters():
        assert p.grad is not None and bool(torch.isfinite(p.grad).all()), \
            f"HybridRich(stabilize=True) 的 {name} 梯度缺失或非有限"


def check_readout_layers() -> None:
    """多可观测量 / 数据重上传（含可训练重上传）/ AmplitudeVQC / 范数保留。"""
    B, nq = 8, 4
    raw_x = torch.randn(B, 16) * 2.5 + 1.0
    x = raw_x / raw_x.norm(dim=1, keepdim=True)  # 2**4 = 16，振幅编码可装下
    xa = torch.randn(B, nq)

    # AmplitudeVQCClassifier 与 .quantum_features()
    set_seed(0)
    amp_clf = AmplitudeVQCClassifier(nq, 2, 2, n_features=16, head="mlp", scale=5.0, hidden=8)
    zf = amp_clf.quantum_features(x)
    assert tuple(zf.shape) == (B, nq) and zf.dtype == torch.float32
    out_amp = amp_clf(x)
    assert tuple(out_amp.shape) == (B, 2)
    out_amp.sum().backward()

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

    # 全两体关联读出：维度 = 3n 单比特 + k * n(n-1)/2 个两体项
    pairs = nq * (nq - 1) // 2
    for spec, expected in (("z+x+y+zzall", 3 * nq + pairs),
                           ("z+x+y+zzall+xxall+yyall", 3 * nq + 3 * pairs)):
        assert n_readout(nq, spec) == expected, f"{spec} 维度错误"
        set_seed(0)
        wide = make_rich_layer(nq, 2, "amplitude", spec, n_features=16)
        out = wide(x)
        assert tuple(out.shape) == (B, expected), f"{spec} 前向形状错误"
        out.pow(2).mean().backward()
        g = wide.layer.weights.grad
        assert g is not None and bool(torch.isfinite(g).all()) and float(g.norm()) > 0, \
            f"{spec} 梯度缺失 / 非有限 / 全零"

    # 朴素重上传：参数 shape 与初始权重不变，但输入被每层重新嵌入
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

    # Pérez-Salinas 可训练重上传 (reupload="trainable" / trainable_reupload=True)
    set_seed(0)
    treup = make_rich_layer(nq, 2, "angle", "z+x", reupload="trainable")
    assert hasattr(treup.layer, "in_scale") and hasattr(treup.layer, "in_shift")
    assert n_quantum_params(nq, 2, trainable_reupload=True) == 2 * nq * 3 + 4 * nq
    out_tr = treup(xa)
    assert tuple(out_tr.shape) == (B, 2 * nq)
    out_tr.pow(2).mean().backward()
    assert float(treup.layer.weights.grad.norm()) > 0
    assert float(treup.layer.in_scale.grad.norm()) > 0
    assert float(treup.layer.in_shift.grad.norm()) > 0

    # 分类器 + 范数保留 (preserve_norm=True) + .quantum_features()
    y = torch.randint(0, 2, (B,))
    set_seed(0)
    model_norm = RichVQCClassifier(
        nq, 2, 2, "amplitude", "z+x+zz",
        n_features=16, head="mlp", scale=5.0,
        preserve_norm=True, post_norm=True,
    )
    assert model_norm.readout_dim == n_readout(nq, "z+x+zz") + 1
    qf_norm = model_norm.quantum_features(raw_x)
    assert tuple(qf_norm.shape) == (B, model_norm.readout_dim)
    torch.nn.functional.cross_entropy(model_norm(raw_x), y).backward()
    for name, p in model_norm.named_parameters():
        assert p.grad is not None and bool(torch.isfinite(p.grad).all()), \
            f"RichVQC(preserve_norm=True) 的 {name} 梯度缺失或非有限"

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


def check_patch_vqc_and_sampling() -> None:
    """测试 PatchVQCClassifier 局部多窗口量子模型与无放回均衡抽样。"""
    B = 4
    # 构造 8x8 迷你图像，其中第 0 张图的左上角 4x4 patch 全为 0（验证全零背景 patch 安全性）
    imgs = torch.rand(B, 1, 8, 8)
    imgs[0, 0, :4, :4] = 0.0
    y = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    set_seed(0)
    pmodel = PatchVQCClassifier(
        n_qubits=4, n_classes=2, n_layers=1,
        patch_size=4, stride=4, image_size=8,
        observables="z+x+zz", preserve_patch_norm=True,
        hidden=8, scale=1.0, dropout=0.0,
    )
    assert pmodel.n_patches == 4
    zf = pmodel.quantum_features(imgs)
    assert tuple(zf.shape) == (B, 4 * (n_readout(4, "z+x+zz") + 1))
    assert bool(torch.isfinite(zf).all()), "全零背景 patch 导致量子特征出现 NaN/Inf"
    logits = pmodel(imgs)
    assert tuple(logits.shape) == (B, 2)
    torch.nn.functional.cross_entropy(logits, y).backward()
    for name, p in pmodel.named_parameters():
        assert p.grad is not None and bool(torch.isfinite(p.grad).all()), \
            f"PatchVQC 的 {name} 梯度缺失或非有限"

    # 测试 _select 与 stratified_subset：当某类样本数 < n // C 但总样本数 >= n 时绝不重复抽样
    X_imbal = np.arange(30, dtype=np.float32).reshape(30, 1)
    # 类别 0 只有 6 个样本，类别 1 有 24 个样本（模拟 MNIST 数字 5 样本偏少）
    y_imbal = np.array([0] * 6 + [1] * 24, dtype=np.int64)
    Xs, ys = stratified_subset(X_imbal, y_imbal, n=20, classes=2, seed=0)
    assert len(Xs) == 20 and len(ys) == 20
    assert len(np.unique(Xs[:, 0])) == 20, "当 n <= 总样本数时不应出现有放回重复样本"
    assert int((ys == 0).sum()) == 6 and int((ys == 1).sum()) == 14

    Xs_all, ys_all = stratified_subset(X_imbal, y_imbal, n=30, classes=2, seed=0)
    assert len(np.unique(Xs_all[:, 0])) == 30, "全量抽样必须无遗漏、无重复覆盖全部 30 个样本"

    # 测试公开辅助函数 fetch_mnist_raw 与 l2_normalize
    Xr, yr, Xe, ye = fetch_mnist_raw("data", source="synthetic", seed=0)
    Xn = l2_normalize(Xr[:5])
    assert np.allclose(np.linalg.norm(Xn, axis=1), 1.0, atol=1e-5)


def check_p0_fixes_and_contracts() -> None:
    """P0 数学/控制变量/QWC 回归断言：
    1. AngleEmbedding(RY) 解析保真度核 prod cos^2((x-x')/2) 与 PennyLane default.qubit 误差 < 1e-12；
    2. resolve_open_issues.py 使用修正后的 * 0.5（非旧版错误的 * 0.25）；
    3. load_mnist(standardize_per_pc=True) 消除全局标量 min-max 偏移，逐 PC 均值近似 0 且落在 [-pi/2, pi/2]；
    4. HybridRichClassifier(quantum=False, stabilize=True) 保留 post_norm 且支持 match_readout_dim=True；
    5. HybridRichClassifier(..., trainable_reupload=True) 禁用内层 bound_inputs 避免双重 tanh；
    6. RichVQCClassifier.post_bn 与向后兼容别名 post_ln 一致；
    7. count_qwc_groups 与 parameter_breakdown / assert_matched_config 契约。
    """
    import pennylane as qml

    # 1. Exact AngleEmbedding(RY) fidelity kernel verification (< 1e-12)
    dev = qml.device("default.qubit", wires=4)

    @qml.qnode(dev)
    def _angle_state(x):
        qml.AngleEmbedding(x, wires=range(4), rotation="Y")
        return qml.state()

    rng = np.random.default_rng(7)
    x1 = rng.uniform(-np.pi, np.pi, size=4)
    x2 = rng.uniform(-np.pi, np.pi, size=4)
    exact_fid = float(np.abs(np.vdot(_angle_state(x1), _angle_state(x2))) ** 2)
    analytic_fid = float(np.prod(np.cos((x1 - x2) * 0.5) ** 2))
    assert abs(exact_fid - analytic_fid) < 1e-12, f"Angle kernel mismatch: {exact_fid} vs {analytic_fid}"

    # Verify resolve_open_issues.py uses * 0.5 (not * 0.25)
    roi_src = (ROOT / "experiments" / "resolve_open_issues.py").read_text(encoding="utf-8")
    assert "(Xps[:, None, :] - Xps[None, :, :]) * 0.5" in roi_src
    assert "(Xpst[:, None, :] - Xps[None, :, :]) * 0.5" in roi_src
    assert "* 0.25" not in roi_src

    # Verify audit_round8.py uses batch_size=64 and no_decay_modules parity
    aud_src = (ROOT / "experiments" / "audit_round8.py").read_text(encoding="utf-8")
    assert "batch_size=64" in aud_src and 'no_decay_modules=("proj",)' in aud_src

    # 2. Per-PC standardization removes global min-max phase shift
    X_std, _, Xt_std, _, meta_std = load_mnist(
        n_train=60, n_test=20, n_features=6, classes=3,
        source="synthetic", seed=0, standardize_per_pc=True,
    )
    assert meta_std["standardize_per_pc"] is True
    assert bool((X_std >= -np.pi / 2.0 - 1e-5).all() and (X_std <= np.pi / 2.0 + 1e-5).all())
    assert bool((Xt_std >= -np.pi / 2.0 - 1e-5).all() and (Xt_std <= np.pi / 2.0 + 1e-5).all())
    assert float(np.max(np.abs(X_std.mean(axis=0)))) < 0.25, "逐主成分标准化后均值应接近 0"

    # 3. HybridRichClassifier quantum=False + stabilize=True + double-tanh guard
    set_seed(0)
    hyb_noq = HybridRichClassifier(
        n_qubits=4, n_classes=2, n_layers=2, hidden=8,
        quantum=False, head="mlp", scale=1.0, frontend="light",
        observables="z+x+y+zz", stabilize=True, match_readout_dim=True,
    )
    assert hyb_noq.post_norm is not None, "quantum=False + stabilize=True 必须保留 post_norm"
    assert hyb_noq.readout_dim == n_readout(4, "z+x+y+zz")
    xb_img = torch.rand(4, 1, 28, 28)
    out_noq = hyb_noq(xb_img)
    assert tuple(out_noq.shape) == (4, 2)
    out_noq.sum().backward()

    set_seed(0)
    hyb_reup = HybridRichClassifier(
        n_qubits=4, n_classes=2, n_layers=2, hidden=8,
        quantum=True, head="mlp", scale=1.0, frontend="light",
        observables="z", trainable_reupload=True, stabilize=True,
    )
    assert hyb_reup.q.bound_inputs is False, "HybridRichClassifier 已做 tanh*pi/2，不得双重 tanh 压缩"

    # 4. RichVQCClassifier.post_bn and backward-compat alias post_ln
    rv = RichVQCClassifier(4, 2, 1, "amplitude", "z", n_features=16, post_norm=True)
    assert rv.post_bn is not None and rv.post_ln is rv.post_bn

    # 5. QWC grouping contract & parameter_breakdown / assert_matched_config
    assert count_qwc_groups(10, "z") == 1
    assert count_qwc_groups(10, "z+x") == 2
    assert count_qwc_groups(10, "z+x+y+zz") == 3
    assert count_qwc_groups(10, "z+x+y+zzall+xxall+yyall") == 3

    pb_q = parameter_breakdown(rv, preprocessing_params=0)
    assert pb_q["quantum_params"] == n_quantum_params(4, 1)
    assert pb_q["quantum_params"] + pb_q["classical_wrapper_params"] == pb_q["total_model_params"]
    pb_pca = parameter_breakdown(MLPClassifier(10, 2, 8), preprocessing_params=784 + 10 * 784)
    assert pb_pca["preprocessing_fitted_params"] == 8624
    assert pb_pca["effective_total_params"] == pb_pca["total_model_params"] + 8624

    assert_matched_config({"bs": 64, "lr": 0.01}, {"bs": 64, "lr": 0.01}, ("bs", "lr"))
    raised = False
    try:
        assert_matched_config({"bs": 64}, {"bs": 32}, ("bs",))
    except AssertionError:
        raised = True
    assert raised, "assert_matched_config 应对不匹配配置抛出 AssertionError"


def check_quantum_native_and_three_regimes() -> None:
    """验证 Regime I 去量子化对照、有限 shots 采样、IQP 核、深度重上传与 Track B Cluster-Ising QCNN。"""
    # 1. Disjoint TVT loader
    tvt = load_mnist_tvt(24, 12, 12, classes=3, source="synthetic", seed=0, pca_dims=(4, 6))
    assert tvt["raw"]["X_train"].shape == (24, 784)
    assert tvt["pca"][4]["X_val"].shape == (12, 4)
    assert tvt["pca"][6]["preprocessing_fitted_params"] == 784 + 6 * 784

    # 2. RandomQuadratic & PatchLowRank parameter parity
    set_seed(0)
    rq = RandomQuadraticClassifier(16, 16, 3, head="mlp", scale=5.0, hidden=8)
    xr = torch.randn(4, 16)
    xr = xr / xr.norm(dim=1, keepdim=True)
    out_rq = rq(xr)
    assert tuple(out_rq.shape) == (4, 3)
    out_rq.sum().backward()

    set_seed(0)
    pl = PatchLowRankClassifier(14, 7, 28, 10, 32, 48, True, "mlp", 1.0, 64, 0.1)
    set_seed(0)
    pv = PatchVQCClassifier(8, 10, 2, 14, 7, 28, "z+x+y+zz", True, head="mlp", scale=1.0, hidden=64, dropout=0.1)
    assert parameter_breakdown(pl)["total_model_params"] == parameter_breakdown(pv)["total_model_params"] == 20364

    # 3. Finite-shot QWC evaluation
    rv4 = RichVQCClassifier(4, 2, 1, "amplitude", "z+x+y+zz", n_features=16, head="mlp", hidden=8)
    z_sh = evaluate_rich_layer_shots(rv4.q.layer.weights, xr, n_qubits=4, observables="z+x+y+zz", shots=50, seed=0)
    assert tuple(z_sh.shape) == (4, 16) and bool(torch.isfinite(z_sh).all())

    # 4. Regime III on classical data: DataReuploadingFidelityClassifier (0 classical wrapper params) + ClassicalTrigNet
    reup = DataReuploadingFidelityClassifier(n_qubits=2, n_layers=2, n_classes=3)
    pb_reup = parameter_breakdown(reup)
    assert pb_reup["classical_wrapper_params"] == 0 and pb_reup["quantum_param_share"] == 1.0
    x6 = torch.randn(4, 6)
    p_born = reup.born_probs(x6)
    assert torch.allclose(p_born.sum(dim=-1), torch.ones(4), atol=1e-5)
    reup(x6).sum().backward()
    trig = ClassicalTrigNet(6, 3, n_freqs=2, bias=False)
    trig(x6).sum().backward()

    # 5. Regime III entangling IQP kernel (real PennyLane states)
    s_tr = iqp_feature_states(tvt["pca"][4]["X_train"][:8], n_qubits=4, alpha=0.3, reps=2)
    K = iqp_kernel_matrix(s_tr, s_tr)
    assert K.shape == (8, 8) and np.allclose(np.diag(K), 1.0, atol=1e-10)
    assert np.allclose(K, K.T, atol=1e-10)

    # 6. Track B: 1D Cluster-Ising exact ground state + Cong-et-al. QCNN (0 classical params)
    psi_spt, gap_spt, s_spt = cluster_ising_ground_state(0.0, 0.0, n_qubits=4)
    assert np.isclose(np.linalg.norm(psi_spt), 1.0, atol=1e-5)
    assert gap_spt > 1.5 and s_spt > 0.99
    ds = generate_cluster_ising_dataset(n_train=12, n_val=6, n_test=6, n_qubits=4, seed=0)
    qcnn = QCNNPhaseClassifier(n_qubits=4)
    pb_qcnn = parameter_breakdown(qcnn)
    assert pb_qcnn["classical_wrapper_params"] == 0 and pb_qcnn["quantum_param_share"] == 1.0
    st = torch.as_tensor(ds["states"]["X_train"][:4], dtype=torch.float32)
    pb_out = qcnn.born_probs(st)
    assert torch.allclose(pb_out.sum(dim=-1), torch.ones(4), atol=1e-5)
    qcnn(st).sum().backward()
    sh_acc = qcnn.evaluate_finite_shots(ds["states"]["X_test"], ds["y_test"], shots=50, seed=0)
    assert 0.0 <= sh_acc <= 1.0


def check_quick_scripts() -> None:
    """验证 probe_information.py、classical_budget.py、resolve_open_issues.py 与 benchmark_three_regimes.py 的 --quick 契约。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        for script, expected_name in [
            ("experiments/probe_information.py", "quick_info_metrics.json"),
            ("experiments/classical_budget.py", "quick_budget_metrics.json"),
            ("experiments/resolve_open_issues.py", "quick_open_issues_metrics.json"),
            ("experiments/benchmark_three_regimes.py", "quick_three_regimes_metrics.json"),
        ]:
            cmd = [sys.executable, str(ROOT / script), "--quick", "--out", tmpdir]
            if "resolve_open_issues" in script:
                cmd.append("--no-figures")
            proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
            assert proc.returncode == 0, f"{script} --quick 失败:\n{proc.stderr}\n{proc.stdout}"
            out_file = tmp_path / expected_name
            assert out_file.exists(), f"{script} --quick 未在临时目录生成 {expected_name}"


# 供 pytest 自动发现的 test_* 入口
test_classifier = check_classifier
test_heads_and_controls = check_heads_and_controls
test_hybrid_and_images = check_hybrid_and_images
test_readout_layers = check_readout_layers
test_patch_vqc_and_sampling = check_patch_vqc_and_sampling
test_p0_fixes_and_contracts = check_p0_fixes_and_contracts
test_quantum_native_and_three_regimes = check_quantum_native_and_three_regimes
test_quick_scripts = check_quick_scripts


def main() -> None:
    check_classifier()
    check_heads_and_controls()
    check_hybrid_and_images()
    check_readout_layers()
    check_patch_vqc_and_sampling()
    check_p0_fixes_and_contracts()
    check_quantum_native_and_three_regimes()
    check_quick_scripts()
    print("SMOKE OK")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"SMOKE FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
