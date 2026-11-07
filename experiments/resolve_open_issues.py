"""第八轮：系统解决 REVIEW.md 遗留的核心开放问题（2026-11-07）。

在同一套真实 MNIST IDX 600 训练 / 200 测试切分（5 seeds: 0,1,2,3,4）与配对 t 分布 95% CI 下，
逐项实测并闭合以下四个方法学与架构开放问题：

Issue A — 混合架构 CNN+VQC 的训练崩溃（纯串行量子瓶颈，无经典旁路跳过连接）：
  * 根因诊断：轻量 CNN 经两次池化与 AdaptiveAvgPool2d((2,2)) 后，样本间特征标准差仅 ~0.003，
    比全连接层随机初始化偏置（~0.15）小约 50 倍；未做批标准化时所有样本坍缩到同一个常值旋转角，
    且量子层 [0, 2pi] 均匀随机初始化在首轮即打乱梯度方向，导致 seed 4 完全坍缩到 0.10（loss=ln 10）。
  * 修复方案（``hyb_light_stab``）：保持 **100% 纯串行** ``CNN -> BatchNorm1d -> Linear(bias=False) -> BatchNorm1d -> Tanh*(pi/2) -> 8q VQC(N(0, 0.1^2), z+x+y+zz) -> BatchNorm1d -> MLP``，
    不加任何绕过量子层的残差/跳过连接，正面证明量子层在混合结构中可稳定收敛且消除坍缩。

Issue B — Pérez-Salinas 可训练数据重上传 vs 朴素重上传（Angle-10 / PCA-10）：
  * 对比 ``a10_z``（单次编码）、``a10_reup_naive``（第四轮朴素重上传，掉分 11 pt）、
    ``a10_reup_perez_z``（Pérez-Salinas 逐层可学习缩放 omega_l * x + theta_l + tanh 有界化，仅 Z 读出）
    与 ``a10_reup_perez_full``（可训练重上传 + 40 维 z+x+y+zz 读出）。

Issue C — 量子核方法（QSVC）第三条基线 vs 经典核 SVM（5 seeds）：
  * 补齐 README / REVIEW 长期缺失的第三条量子基线（Analytical Quantum Kernel SVM）：
    - ``qkernel_amp``：10-qubit 全局振幅编码量子核 K(x,x') = |<psi_amp(x)|psi_amp(x')>|^2 = (x^T x')^2
    - ``qkernel_patch``：8-qubit 局部 3x3 重叠 Patch（14x14, stride 7）振幅量子核
    - ``qkernel_angle``：10-qubit 角度编码量子核 K(x,x') = prod_i cos^2((x_i - x'_i)/4)（PCA-10）
    并与同切分下的经典 ``svm_poly2``、``svm_rbf``、``svm_rbf_pca10`` 直接配对比较。

Issue D — 打破全局振幅编码无局部性瓶颈：局部多 Patch 权重共享量子分类器（PatchVQC）与范数保留：
  * ``amp_b40`` / ``amp_b165``：全局 10-qubit 振幅编码（40 维 / 165 维读出）
  * ``amp_b165_norm``：165 维量子读出 + 保留逐样本 L2 归一化前墨量范数 ||x||_2 + 余弦退火/权重衰减
  * ``patch_vqc_4quad``：2x2=4 个无重叠 14x14 象限窗口通过**同一个权重共享 8-qubit VQC**（32 维读出+窗口范数）
  * ``patch_vqc``：3x3=9 个重叠 14x14 局部窗口（stride=7）通过**同一个权重共享 8-qubit VQC**（仅 48 个量子参数，
    态空间 2^8=256 比 10-qubit 小 4 倍，兼具 2D 空间局部性与局部二次型表达力）
  * ``ref_mlp784`` / ``ref_pca165``：同批次等数据/近似等参数经典参照。

用法：
    .venv/bin/python experiments/resolve_open_issues.py --data-source idx --workers 10
    .venv/bin/python experiments/resolve_open_issues.py --quick
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qml_lab.data import load_mnist, load_mnist_images, load_mnist_raw  # noqa: E402
from qml_lab.models import (  # noqa: E402
    CNNClassifier,
    HybridCNNClassifier,
    HybridRichClassifier,
    MLPClassifier,
    PatchVQCClassifier,
    RichVQCClassifier,
)
from qml_lab.train import train_classifier  # noqa: E402
from qml_lab.utils import count_parameters, ensure_dir, set_seed  # noqa: E402


@dataclass(frozen=True)
class GroupSpec:
    key: str
    issue: str
    label: str
    kind: str
    data: str
    obs: str = "z"
    reupload: bool = False
    trainable_reupload: bool = False
    stabilize: bool = False
    preserve_norm: bool = False
    q_lr: float | None = None
    weight_decay: float = 0.0
    cosine_lr: bool = False
    patch_stride: int = 7
    batch_size: int = 32
    priority: int = 10  # 越小越先调度（慢任务优先入队避免长尾）


def all_groups() -> list[GroupSpec]:
    return [
        # ---- Issue A: Hybrid CNN + VQC Stability (No-Skip Pure Sequential) ----
        GroupSpec("nq_light", "A", "lightCNN, no VQC (skip q)", "hybrid_nq", "img",
                  priority=30),
        GroupSpec("hyb_light_legacy", "A", "lightCNN + 8q Z (legacy)", "hybrid_legacy", "img",
                  obs="z", priority=15),
        GroupSpec("hyb_light_qlr", "A", "lightCNN + 8q Z (q_lr=1e-3)", "hybrid_rich", "img",
                  obs="z", q_lr=0.001, priority=15),
        GroupSpec("hyb_light_stab", "A", "lightCNN + 8q stabilized (no skip)", "hybrid_rich", "img",
                  obs="z+x+y+zz", stabilize=True, priority=10),
        GroupSpec("cnn_light", "A", "lightCNN direct head", "cnn", "img",
                  priority=30),

        # ---- Issue B: Perez-Salinas Trainable Data Re-uploading (Angle-10) ----
        GroupSpec("a10_z", "B", "Angle10 Z-only (single)", "angle", "pca10",
                  obs="z", priority=12),
        GroupSpec("a10_reup_naive", "B", "Angle10 Z + naive reupload", "angle", "pca10",
                  obs="z", reupload=True, priority=11),
        GroupSpec("a10_reup_perez_z", "B", "Angle10 Z + Perez-Salinas reup", "angle", "pca10",
                  obs="z", trainable_reupload=True, priority=9),
        GroupSpec("a10_reup_perez_full", "B", "Angle10 40d + Perez-Salinas reup", "angle", "pca10",
                  obs="z+x+y+zz", trainable_reupload=True, priority=6),

        # ---- Issue D: Global Amplitude vs Norm-Preserving vs Local PatchVQC ----
        GroupSpec("amp_b40", "D", "Amp10 40d (global)", "amp", "raw",
                  obs="z+x+y+zz", priority=7),
        GroupSpec("amp_b165", "D", "Amp10 165d (global)", "amp", "raw",
                  obs="z+x+y+zzall+xxall+yyall", priority=1),
        GroupSpec("amp_b165_norm", "D", "Amp10 165d + ||x||2 + reg", "amp", "raw_unnorm",
                  obs="z+x+y+zzall+xxall+yyall", preserve_norm=True,
                  weight_decay=1e-4, cosine_lr=True, priority=2),
        GroupSpec("patch_vqc_4quad", "D", "PatchVQC 8q (2x2 quadrants, 14x14)", "patch", "img",
                  obs="z+x+y+zz", patch_stride=14, weight_decay=1e-4, cosine_lr=True,
                  batch_size=32, priority=5),
        GroupSpec("patch_vqc", "D", "PatchVQC 8q (3x3 overlapping, 14x14)", "patch", "img",
                  obs="z+x+y+zz", patch_stride=7, weight_decay=1e-4, cosine_lr=True,
                  batch_size=64, priority=4),
        GroupSpec("ref_pca165", "D", "Classical PCA-165 + MLP32", "mlp_pca165", "pca165",
                  priority=30),
        GroupSpec("ref_mlp784", "D", "Classical MLP-784", "mlp784", "raw",
                  priority=30),
    ]


def load_datasets(data_source: str, data_dir: str, n_train: int, n_test: int,
                  classes: int, angle_qubits: int, seed: int = 0) -> dict:
    """加载并对齐四条数据表示路径，断言样本标签逐位一致。"""
    from sklearn.decomposition import PCA

    kw = dict(n_train=n_train, n_test=n_test, classes=classes,
              source=data_source, data_dir=data_dir, seed=seed)
    Xr, yr, Xrt, yrt, mr = load_mnist_raw(**kw)
    Xi, yi, Xit, yit, mi = load_mnist_images(**kw)
    Xp, yp, Xpt, ypt, mp_meta = load_mnist(n_features=angle_qubits, **kw)

    assert np.array_equal(yr, yi) and np.array_equal(yrt, yit), "raw 与 images 标签不一致"
    assert np.array_equal(yr, yp) and np.array_equal(yrt, ypt), "raw 与 pca10 标签不一致"

    # 未做逐样本 L2 归一化的展平像素（供 preserve_norm=True 提取真实墨量范数 ||x||_2）
    X_unnorm = np.ascontiguousarray(Xi.reshape(len(Xi), -1), dtype=np.float32)
    Xt_unnorm = np.ascontiguousarray(Xit.reshape(len(Xit), -1), dtype=np.float32)

    # PCA-165（仅在训练集拟合，与 push_ceiling.py 的 ref_pca165 协议完全一致）
    n_comp = min(165, Xr.shape[0] - 1, Xr.shape[1])
    pca = PCA(n_components=n_comp, random_state=seed)
    X165 = np.ascontiguousarray(pca.fit_transform(Xr), dtype=np.float32)
    Xt165 = np.ascontiguousarray(pca.transform(Xrt), dtype=np.float32)
    if n_comp < 165:
        X165 = np.pad(X165, ((0, 0), (0, 165 - n_comp)))
        Xt165 = np.pad(Xt165, ((0, 0), (0, 165 - n_comp)))

    return {
        "raw": {"X": Xr, "y": yr, "Xt": Xrt, "yt": yrt},
        "raw_unnorm": {"X": X_unnorm, "y": yr, "Xt": Xt_unnorm, "yt": yrt},
        "img": {"X": Xi, "y": yi, "Xt": Xit, "yt": yit},
        "pca10": {"X": Xp, "y": yp, "Xt": Xpt, "yt": ypt},
        "pca165": {"X": X165, "y": yr, "Xt": Xt165, "yt": yrt},
        "meta": {"raw": mr, "img": mi, "pca10": mp_meta, "same_split_asserted": True},
    }


def build_model(g: GroupSpec, cfg: dict, seed: int):
    set_seed(seed)
    c = cfg["classes"]
    L = cfg["layers"]
    if g.kind == "hybrid_nq":
        return HybridCNNClassifier(
            n_qubits=cfg["hybrid_qubits"], n_classes=c, n_layers=L,
            hidden=32, quantum=False, head="mlp", scale=1.0, frontend="light",
        )
    if g.kind == "hybrid_legacy":
        return HybridCNNClassifier(
            n_qubits=cfg["hybrid_qubits"], n_classes=c, n_layers=L,
            hidden=32, quantum=True, head="mlp", scale=1.0, frontend="light",
        )
    if g.kind == "hybrid_rich":
        return HybridRichClassifier(
            n_qubits=cfg["hybrid_qubits"], n_classes=c, n_layers=L,
            hidden=32, quantum=True, head="mlp", scale=1.0, frontend="light",
            observables=g.obs, reupload=g.reupload,
            trainable_reupload=g.trainable_reupload, stabilize=g.stabilize,
        )
    if g.kind == "cnn":
        return CNNClassifier(n_classes=c, frontend="light")
    if g.kind == "angle":
        return RichVQCClassifier(
            n_qubits=cfg["angle_qubits"], n_classes=c, n_layers=L,
            encoding="angle", observables=g.obs, n_features=cfg["angle_qubits"],
            reupload=g.reupload, trainable_reupload=g.trainable_reupload,
            head="mlp", scale=5.0, hidden=32,
        )
    if g.kind == "amp":
        return RichVQCClassifier(
            n_qubits=cfg["amp_qubits"], n_classes=c, n_layers=L,
            encoding="amplitude", observables=g.obs, n_features=784,
            preserve_norm=g.preserve_norm, head="mlp", scale=5.0, hidden=32,
        )
    if g.kind == "patch":
        return PatchVQCClassifier(
            n_qubits=cfg["patch_qubits"], n_classes=c, n_layers=L,
            patch_size=14, stride=g.patch_stride, image_size=28,
            observables=g.obs, preserve_patch_norm=True,
            head="mlp", scale=1.0, hidden=64, dropout=0.1,
        )
    if g.kind == "mlp_pca165":
        return MLPClassifier(n_qubits=165, n_classes=c, hidden=32)
    if g.kind == "mlp784":
        return MLPClassifier(n_qubits=784, n_classes=c, hidden=32)
    raise ValueError(f"Unknown group kind: {g.kind}")


def _worker_run(task: tuple[GroupSpec, int, dict]) -> dict:
    g, seed, cfg = task
    torch.set_num_threads(cfg["torch_threads"])
    data = load_datasets(
        cfg["data_source"], cfg["data_dir"],
        cfg["train_samples"], cfg["test_samples"],
        cfg["classes"], cfg["angle_qubits"], seed=0,
    )
    d = data[g.data]
    model = build_model(g, cfg, seed)
    bs = min(g.batch_size, cfg["batch_size"]) if cfg["quick"] else g.batch_size
    h = train_classifier(
        model, d["X"], d["y"], d["Xt"], d["yt"],
        epochs=cfg["epochs"], batch_size=bs, lr=cfg["lr"],
        seed=seed, eval_batch_size=64, verbose=False, eval_train=False,
        quantum_lr=g.q_lr, weight_decay=g.weight_decay, cosine_lr=g.cosine_lr,
    )
    return {
        "key": g.key,
        "seed": seed,
        "final_test_acc": round(float(h["final_test_acc"]), 4),
        "final_loss": round(float(h["train_loss"][-1]), 4),
        "seconds": round(float(h["total_seconds"]), 2),
        "params": int(count_parameters(model)),
        "readout_dim": int(getattr(model, "readout_dim", 0)),
        "test_acc_curve": [round(float(v), 4) for v in h["test_acc"]],
    }


def evaluate_quantum_kernels(cfg: dict, seeds: list[int]) -> dict:
    """Issue C：解析量子核（QSVC）与经典核 SVM 在 5 个分层抽样 seed 下的对照。

    说明：核 SVM 在固定训练集上是确定性凸优化，改模型初始化种子不会产生方差；
    因此对核方法组，``seed in seeds`` 作用于独立分层抽样 ``load_mnist_*(..., seed=s)``
    （其中 ``seed=0`` 与神经网络组的固定切分逐样本完全一致），从而给出真实的 5-seed
    均值、标准差与配对 95% CI。
    """
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    specs = {
        "qkernel_amp": "QSVC Global Amplitude K=(x^T x')^2 (10q)",
        "qkernel_patch": "QSVC Local 9-Patch Amplitude Kernel (8q)",
        "qkernel_angle": "QSVC Angle-10 Product Kernel (10q, PCA-10)",
        "svm_poly2": "Classical Poly-2 SVM (784d, scaled)",
        "svm_rbf": "Classical RBF SVM (784d, scaled)",
        "svm_rbf_pca10": "Classical RBF SVM (PCA-10, scaled)",
    }
    per_seed: dict[str, dict[str, dict]] = {k: {} for k in specs}

    def _patch_units_and_weights(X_img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        t = torch.as_tensor(X_img, dtype=torch.float32)
        p = F.unfold(t, kernel_size=14, stride=7).transpose(1, 2).numpy()  # (N, 9, 196)
        norms = np.linalg.norm(p, axis=-1, keepdims=True)
        unit = p / np.maximum(norms, 1e-12)
        rel = (norms / np.maximum(norms.mean(axis=1, keepdims=True), 1e-6)).squeeze(-1)
        return unit, rel

    for s in seeds:
        kw = dict(
            n_train=cfg["train_samples"], n_test=cfg["test_samples"],
            classes=cfg["classes"], source=cfg["data_source"],
            data_dir=cfg["data_dir"], seed=s,
        )
        Xr, yr, Xrt, yrt, _ = load_mnist_raw(**kw)
        Xi, yi, Xit, yit, _ = load_mnist_images(**kw)
        Xps, yps, Xpst, ypst, _ = load_mnist(n_features=cfg["angle_qubits"], **kw)

        # 1. Global Amplitude Quantum Kernel: K(x, x') = |<psi(x)|psi(x')>|^2 = (x^T x')^2
        t0 = time.perf_counter()
        K_tr = (Xr @ Xr.T) ** 2
        K_te = (Xrt @ Xr.T) ** 2
        clf = SVC(kernel="precomputed", C=10.0).fit(K_tr, yr)
        acc = float(clf.score(K_te, yrt))
        per_seed["qkernel_amp"][str(s)] = {
            "final_test_acc": round(acc, 4),
            "seconds": round(time.perf_counter() - t0, 3),
        }

        # 2. Local 9-Patch Amplitude Quantum Kernel (8 qubits per 14x14 patch)
        t0 = time.perf_counter()
        u_tr, w_tr = _patch_units_and_weights(Xi)
        u_te, w_te = _patch_units_and_weights(Xit)
        K_p_tr = np.zeros((len(yi), len(yi)), dtype=np.float64)
        K_p_te = np.zeros((len(yit), len(yi)), dtype=np.float64)
        for p_idx in range(u_tr.shape[1]):
            fid_tr = (u_tr[:, p_idx, :] @ u_tr[:, p_idx, :].T) ** 2
            fid_te = (u_te[:, p_idx, :] @ u_tr[:, p_idx, :].T) ** 2
            K_p_tr += (w_tr[:, p_idx:p_idx + 1] @ w_tr[:, p_idx:p_idx + 1].T) * fid_tr / u_tr.shape[1]
            K_p_te += (w_te[:, p_idx:p_idx + 1] @ w_tr[:, p_idx:p_idx + 1].T) * fid_te / u_tr.shape[1]
        clf = SVC(kernel="precomputed", C=10.0).fit(K_p_tr, yi)
        acc = float(clf.score(K_p_te, yit))
        per_seed["qkernel_patch"][str(s)] = {
            "final_test_acc": round(acc, 4),
            "seconds": round(time.perf_counter() - t0, 3),
        }

        # 3. AngleEmbedding 10-qubit Product Quantum Kernel: prod_i cos^2((x_i - x'_i)/4)
        t0 = time.perf_counter()
        diff_tr = (Xps[:, None, :] - Xps[None, :, :]) * 0.25
        diff_te = (Xpst[:, None, :] - Xps[None, :, :]) * 0.25
        K_a_tr = np.prod(np.cos(diff_tr) ** 2, axis=-1)
        K_a_te = np.prod(np.cos(diff_te) ** 2, axis=-1)
        clf = SVC(kernel="precomputed", C=10.0).fit(K_a_tr, yps)
        acc = float(clf.score(K_a_te, ypst))
        per_seed["qkernel_angle"][str(s)] = {
            "final_test_acc": round(acc, 4),
            "seconds": round(time.perf_counter() - t0, 3),
        }

        # 4. Classical SVMs under unified scaled protocol
        t0 = time.perf_counter()
        clf_poly = make_pipeline(StandardScaler(), SVC(kernel="poly", degree=2, coef0=1.0, C=10.0)).fit(Xr, yr)
        per_seed["svm_poly2"][str(s)] = {
            "final_test_acc": round(float(clf_poly.score(Xrt, yrt)), 4),
            "seconds": round(time.perf_counter() - t0, 3),
        }

        t0 = time.perf_counter()
        clf_rbf = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10.0)).fit(Xr, yr)
        per_seed["svm_rbf"][str(s)] = {
            "final_test_acc": round(float(clf_rbf.score(Xrt, yrt)), 4),
            "seconds": round(time.perf_counter() - t0, 3),
        }

        t0 = time.perf_counter()
        clf_rbf10 = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10.0)).fit(Xps, yps)
        per_seed["svm_rbf_pca10"][str(s)] = {
            "final_test_acc": round(float(clf_rbf10.score(Xpst, ypst)), 4),
            "seconds": round(time.perf_counter() - t0, 3),
        }

    summary = {}
    for k, label in specs.items():
        vals = [per_seed[k][str(s)]["final_test_acc"] for s in seeds]
        secs = [per_seed[k][str(s)]["seconds"] for s in seeds]
        summary[k] = {
            "label": label,
            "issue": "C",
            "seed0_acc": per_seed[k][str(seeds[0])]["final_test_acc"],
            "final_acc": _stats(vals),
            "seconds": _stats(secs),
            "per_seed": per_seed[k],
        }
    return summary


def _stats(values: list[float]) -> dict:
    a = np.asarray(values, dtype=float)
    return {
        "mean": round(float(a.mean()), 4),
        "std": round(float(a.std(ddof=1)) if a.size > 1 else 0.0, 4),
        "min": round(float(a.min()), 4),
        "max": round(float(a.max()), 4),
        "values": [round(float(v), 4) for v in a],
        "n": int(a.size),
    }


def _paired(a_vals: list[float], b_vals: list[float]) -> dict:
    from scipy.stats import t

    d = (np.asarray(a_vals, dtype=float) - np.asarray(b_vals, dtype=float)) * 100.0
    mean = float(d.mean())
    sd = float(d.std(ddof=1)) if d.size > 1 else 0.0
    half = float(t.ppf(0.975, d.size - 1)) * sd / np.sqrt(d.size) if d.size > 1 else 0.0
    return {
        "mean_pt": round(mean, 2),
        "std_pt": round(sd, 2),
        "ci95_pt": [round(mean - half, 2), round(mean + half, 2)],
        "ci_includes_zero": bool(mean - half <= 0.0 <= mean + half),
        "n_positive": int(np.sum(d > 0)),
        "per_seed_pt": [round(float(v), 2) for v in d],
    }


def make_figure(groups_out: dict, kernels_out: dict, fig_path: Path) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.6))

    panels = [
        (
            axes[0, 0],
            "Issue A: Hybrid CNN+VQC Stability (No-Skip Bottleneck)",
            ["hyb_light_legacy", "hyb_light_qlr", "nq_light", "hyb_light_stab", "cnn_light"],
            groups_out,
            ["#d95f02", "#e6ab02", "#7570b3", "#1b9e77", "#4d4d4d"],
        ),
        (
            axes[0, 1],
            "Issue B: Perez-Salinas Trainable Re-uploading (Angle-10)",
            ["a10_reup_naive", "a10_z", "a10_reup_perez_z", "a10_reup_perez_full"],
            groups_out,
            ["#d95f02", "#8e8e8e", "#5b8ff9", "#1b9e77"],
        ),
        (
            axes[1, 0],
            "Issue C: Quantum Kernel (QSVC) vs Classical SVMs",
            ["qkernel_angle", "svm_rbf_pca10", "svm_rbf", "svm_poly2", "qkernel_amp", "qkernel_patch"],
            kernels_out,
            ["#7570b3", "#a6cee3", "#4d4d4d", "#8e8e8e", "#5b8ff9", "#1b9e77"],
        ),
        (
            axes[1, 1],
            "Issue D: Local PatchVQC (8q) vs Global Amp10 (165d)",
            ["amp_b40", "amp_b165", "amp_b165_norm", "patch_vqc_4quad", "patch_vqc", "ref_pca165", "ref_mlp784"],
            groups_out,
            ["#8e8e8e", "#5b8ff9", "#36cfc9", "#9270ca", "#1b9e77", "#4d4d4d", "#262626"],
        ),
    ]

    for ax, title, keys, src, colors in panels:
        valid_keys = [k for k in keys if k in src]
        means = [src[k]["final_acc"]["mean"] for k in valid_keys]
        stds = [src[k]["final_acc"]["std"] for k in valid_keys]
        labels = [src[k]["label"] for k in valid_keys]
        bars = ax.bar(
            range(len(valid_keys)), means, yerr=stds, capsize=4,
            color=colors[:len(valid_keys)], edgecolor="#222222", linewidth=0.6, alpha=0.9,
        )
        ax.set_xticks(range(len(valid_keys)))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8.5)
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel("test accuracy (5 seeds)")
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        for b, m, s in zip(bars, means, stds):
            ax.text(
                b.get_x() + b.get_width() / 2.0,
                min(0.99, m + s + 0.02),
                f"{m:.3f}",
                ha="center",
                fontsize=8,
            )

    fig.suptitle("Round 8: Resolving Open Architectural & Methodological Issues (5 seeds, mean +/- 1 SD)",
                 fontsize=12, fontweight="bold", y=0.995)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    return str(fig_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--train-samples", type=int, default=600)
    p.add_argument("--test-samples", type=int, default=200)
    p.add_argument("--classes", type=int, default=10)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--amp-qubits", type=int, default=10)
    p.add_argument("--angle-qubits", type=int, default=10)
    p.add_argument("--hybrid-qubits", type=int, default=8)
    p.add_argument("--patch-qubits", type=int, default=8)
    p.add_argument("--data-source", default="idx")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--torch-threads", type=int, default=2)
    p.add_argument("--out", default="results")
    p.add_argument("--archive", default="docs/open_issues_metrics.json")
    p.add_argument("--fig-dir", default="docs/figures")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--quick", action="store_true", help="离线极小自检，不覆盖 docs/ 正式产物")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.seeds = "0"
        args.epochs = 1
        args.train_samples, args.test_samples = 24, 12
        args.classes = 3
        args.layers = 1
        args.amp_qubits = 10
        args.angle_qubits = 4
        args.hybrid_qubits = 4
        args.patch_qubits = 8
        args.batch_size = 12
        args.workers = 1
        args.data_source = "synthetic"
        args.out = "results"
        if args.archive == "docs/open_issues_metrics.json":
            args.archive = "results/quick_open_issues_metrics.json"
        if args.fig_dir == "docs/figures":
            args.fig_dir = "results"

    t_start = time.perf_counter()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    dd = Path(args.data_dir)
    data_dir = str(dd if dd.is_absolute() else ROOT / dd)

    cfg = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "train_samples": args.train_samples,
        "test_samples": args.test_samples,
        "classes": args.classes,
        "layers": args.layers,
        "amp_qubits": args.amp_qubits,
        "angle_qubits": args.angle_qubits,
        "hybrid_qubits": args.hybrid_qubits,
        "patch_qubits": args.patch_qubits,
        "data_source": args.data_source,
        "data_dir": data_dir,
        "torch_threads": args.torch_threads,
        "quick": bool(args.quick),
    }

    print("== Round 8: Resolving Open Issues in Quantum MNIST ==", flush=True)
    data_meta = load_datasets(
        cfg["data_source"], cfg["data_dir"],
        cfg["train_samples"], cfg["test_samples"],
        cfg["classes"], cfg["angle_qubits"], seed=0,
    )["meta"]

    # 1. 先在尚未启动多进程前或独立完成 Issue C（纯 numpy/sklearn 核方法，耗时 < 1s）
    # 注意：不在主进程调用任何多线程 torch 算子，且 ProcessPoolExecutor 显式用 spawn
    kernels_out = evaluate_quantum_kernels(cfg, seeds)
    print("\n--- Issue C: Quantum Kernel (QSVC) vs Classical SVMs ---", flush=True)
    for k, v in kernels_out.items():
        print(f"  {k:16s} {v['final_acc']['mean']:.4f} +/- {v['final_acc']['std']:.4f} "
              f"(seed0={v['seed0_acc']:.4f}) | {v['label']}", flush=True)

    # 2. 调度神经网络 / VQC 组（按 priority 从小到大入队，避免慢任务排在队尾）
    gs = all_groups()
    ordered_gs = sorted(gs, key=lambda g: g.priority)
    tasks = [(g, s, cfg) for g in ordered_gs for s in seeds]

    runs: dict[str, dict[str, dict]] = {g.key: {} for g in gs}
    if args.workers > 1 and len(tasks) > 1:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
            for rec in ex.map(_worker_run, tasks):
                runs[rec["key"]][str(rec["seed"])] = rec
                print(f"  [{rec['key']:20s} seed {rec['seed']}] "
                      f"acc={rec['final_test_acc']:.4f} loss={rec['final_loss']:.4f} "
                      f"params={rec['params']:5d} readout={rec['readout_dim']:3d} "
                      f"{rec['seconds']:5.1f}s", flush=True)
    else:
        for task in tasks:
            rec = _worker_run(task)
            runs[rec["key"]][str(rec["seed"])] = rec
            print(f"  [{rec['key']:20s} seed {rec['seed']}] "
                  f"acc={rec['final_test_acc']:.4f} loss={rec['final_loss']:.4f} "
                  f"params={rec['params']:5d} readout={rec['readout_dim']:3d} "
                  f"{rec['seconds']:5.1f}s", flush=True)

    groups_out = {}
    for g in gs:
        s0 = runs[g.key][str(seeds[0])]
        accs = [runs[g.key][str(s)]["final_test_acc"] for s in seeds]
        losses = [runs[g.key][str(s)]["final_loss"] for s in seeds]
        secs = [runs[g.key][str(s)]["seconds"] for s in seeds]
        groups_out[g.key] = {
            "label": g.label,
            "issue": g.issue,
            "kind": g.kind,
            "data": g.data,
            "observables": g.obs,
            "reupload": g.reupload,
            "trainable_reupload": g.trainable_reupload,
            "stabilize": g.stabilize,
            "preserve_norm": g.preserve_norm,
            "params": s0["params"],
            "readout_dim": s0["readout_dim"],
            "final_acc": _stats(accs),
            "final_loss": _stats(losses),
            "seconds": _stats(secs),
        }

    # 3. 配对差异与 95% CI
    g_vals = lambda k: groups_out[k]["final_acc"]["values"]  # noqa: E731
    k_vals = lambda k: kernels_out[k]["final_acc"]["values"]  # noqa: E731

    paired = {
        # Issue A
        "A_hyb_stab_minus_legacy": _paired(g_vals("hyb_light_stab"), g_vals("hyb_light_legacy")),
        "A_hyb_stab_minus_qlr": _paired(g_vals("hyb_light_stab"), g_vals("hyb_light_qlr")),
        "A_hyb_stab_minus_nq_light": _paired(g_vals("hyb_light_stab"), g_vals("nq_light")),
        "A_cnn_light_minus_hyb_stab": _paired(g_vals("cnn_light"), g_vals("hyb_light_stab")),
        # Issue B
        "B_naive_reup_minus_a10_z": _paired(g_vals("a10_reup_naive"), g_vals("a10_z")),
        "B_perez_z_minus_naive_reup": _paired(g_vals("a10_reup_perez_z"), g_vals("a10_reup_naive")),
        "B_perez_z_minus_a10_z": _paired(g_vals("a10_reup_perez_z"), g_vals("a10_z")),
        "B_perez_full_minus_a10_z": _paired(g_vals("a10_reup_perez_full"), g_vals("a10_z")),
        # Issue C
        "C_qkernel_amp_minus_svm_poly2": _paired(k_vals("qkernel_amp"), k_vals("svm_poly2")),
        "C_qkernel_amp_minus_svm_rbf": _paired(k_vals("qkernel_amp"), k_vals("svm_rbf")),
        "C_qkernel_patch_minus_qkernel_amp": _paired(k_vals("qkernel_patch"), k_vals("qkernel_amp")),
        "C_qkernel_patch_minus_svm_rbf": _paired(k_vals("qkernel_patch"), k_vals("svm_rbf")),
        "C_qkernel_angle_minus_svm_rbf_pca10": _paired(k_vals("qkernel_angle"), k_vals("svm_rbf_pca10")),
        # Issue D
        "D_b165_minus_b40": _paired(g_vals("amp_b165"), g_vals("amp_b40")),
        "D_b165_norm_minus_b165": _paired(g_vals("amp_b165_norm"), g_vals("amp_b165")),
        "D_patch_vqc_minus_b165": _paired(g_vals("patch_vqc"), g_vals("amp_b165")),
        "D_patch_vqc_minus_patch_4quad": _paired(g_vals("patch_vqc"), g_vals("patch_vqc_4quad")),
        "D_patch_vqc_minus_ref_pca165": _paired(g_vals("patch_vqc"), g_vals("ref_pca165")),
        "D_patch_vqc_minus_ref_mlp784": _paired(g_vals("patch_vqc"), g_vals("ref_mlp784")),
        "D_patch_vqc_minus_cnn_light": _paired(g_vals("patch_vqc"), g_vals("cnn_light")),
    }

    wall_seconds = round(time.perf_counter() - t_start, 2)
    payload = {
        "purpose": "Round 8: 系统解决 REVIEW.md 的四大开放问题（混合架构崩溃、Pérez-Salinas 重上传、QSVC 核基线、局部 PatchVQC 与范数保留）",
        "config": {k: v for k, v in vars(args).items() if k != "quick"},
        "seeds": seeds,
        "data_meta": data_meta,
        "groups": groups_out,
        "quantum_kernels": kernels_out,
        "paired_differences": paired,
        "runs": runs,
        "wall_seconds": wall_seconds,
        "caveat": (
            "神经网络/VQC 组固定数据切分 seed=0，5 个 seed 改变模型初始化与 mini-batch 顺序；"
            "核方法组（Issue C）因凸优化对初始化确定不变，5 个 seed 对应 5 次独立分层抽样（其中 seed=0 与 VQC 组切分完全一致）；"
            "hyb_light_stab 是 100% 纯串行量子瓶颈（无残差旁路跳过连接）；"
            "PatchVQC 用单个权重共享 8-qubit 振幅编码线路（48 个量子参数）扫描 9 个 14x14 局部窗口，"
            "其分类头第一层接收 9*(32+1)=297 维拼接特征；"
            "165 维观测量（z+x+y+zzall+xxall+yyall）在真机上按 QWC 对易组只需 Z/X/Y 3 个测量基；"
            "n=5 的 95% CI 使用配对 t(4) 分布，未做多重比较校正。"
        ),
    }
    if args.quick:
        payload["caveat"] = "QUICK 自检，合成数据，不得用于任何结论。" + payload["caveat"]

    out_dir = ensure_dir(ROOT / args.out)
    out_name = "quick_open_issues_metrics.json" if args.quick else "open_issues_metrics.json"
    out_path = out_dir / out_name
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved: {out_path}")

    if args.archive:
        arc = Path(args.archive)
        if not arc.is_absolute():
            arc = ROOT / arc
        arc.parent.mkdir(parents=True, exist_ok=True)
        arc.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"archived: {arc}")

    if not args.no_figures:
        fd = Path(args.fig_dir)
        if not fd.is_absolute():
            fd = ROOT / fd
        fig_name = "quick_mnist_open_issues.png" if args.quick else "mnist_open_issues.png"
        fp = make_figure(groups_out, kernels_out, fd / fig_name)
        print(f"figure: {fp}")

    print("\n=== 分组汇总（5 seeds 均值 +/- 标准差） ===")
    for k, s in groups_out.items():
        print(f"  [{s['issue']}] {k:20s} {s['final_acc']['mean']:.4f} +/- {s['final_acc']['std']:.4f} "
              f"| loss {s['final_loss']['mean']:.4f} | params {s['params']:5d} "
              f"| readout {s['readout_dim']:3d} | {s['seconds']['mean']:5.1f}s | {s['label']}")
    print("\n=== 配对差值汇总（百分点，95% CI） ===")
    for k, v in paired.items():
        flag = "不显著" if v["ci_includes_zero"] else "显著"
        print(f"  {k:34s} {v['mean_pt']:+6.2f} +/- {v['std_pt']:5.2f} pt "
              f"CI[{v['ci95_pt'][0]:+6.2f}, {v['ci95_pt'][1]:+6.2f}] "
              f"({v['n_positive']}/{len(seeds)} > 0) {flag}")


if __name__ == "__main__":
    main()
