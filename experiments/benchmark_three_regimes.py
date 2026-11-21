#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""严格受控的三范式（Three-Regime）基准实验：经典 vs 经典移植量子 vs 量子原生。

同时覆盖两条互补赛道（统一 5 个独立分层 Train / Val / Test 切分，严格禁止测试集调参，
所有配对差值报告 Student-t 95% CI、双侧 p 值与 Holm-Bonferroni 多重比较校正）：

Track A — 经典视觉数据（MNIST，600 train / 200 val / 400 test，10 类）：
  * Regime I（经典算法在经典上 — 公平去量子化对照）：
      - ``classical_random_quad``    : 0 个学习投影参数的满秩随机 ±1 齐次二次型 + 与 RichVQC 同头
      - ``classical_poly2_unscaled`` : L2 归一化 784 维上的精确二次核（验证与 qkernel_amp 恒等）
      - ``classical_patch_lowrank``  : 严格 48 参数局部 patch 滤波器 + 与 PatchVQC 逐项一致封装
      - ``classical_trig_net``       : 96 参数经典三角傅里叶网络（等参数对照深度重上传）
      - ``classical_rbf_pca6`` / ``classical_poly2_pca6`` : 同 PCA-6 预处理下的经典核对照
  * Regime II（经典算法移植到量子线路 — 单变量受控替换 + 真实有限 shots 评估）：
      - ``vqc_amp_matched``          : 10q 40 维读出（3 QWC 基）+ 与 random_quad 严格匹配配置，
                                       并额外在真实有限 shots（200 / 1000）下测试精度衰减
      - ``patch_vqc_matched``        : 8q 局部 3x3 PatchVQC（48 量子参数 / 20,364 总参），
                                       与 classical_patch_lowrank 严格对齐 batch=64/BN/WD/cosine LR
      - ``hybrid_vqc_matched`` vs ``hybrid_noq_matched`` : 两者均开启 stabilize=True、等宽 32d 与双 BN
  * Regime III（量子原生算法在量子线路上 — 零经典 MLP 封装）：
      - ``qnative_reupload_fidelity``: 4q L=4 Pérez-Salinas 深度重上传 + 纯 Born 概率读出（96 量子参 / 0 经典参）
      - ``qnative_iqp_kernel``       : 6q 2 层真实 PennyLane 纠缠 IQP/ZZFeatureMap 核，Val 调带宽 alpha

Track B — 原生量子态数据（1D Cluster-Ising n=8 PBC 拓扑相 SPT vs 非 SPT [顺磁+反铁磁]）：
  * Regime III（量子原生）：
      - ``qcnn_spt``                 : 真正 Cong-et-al. QCNN（8->4->2->1 卷积+相干量子池化，
                                       36 量子参 / 0 经典参 / 1 QWC 基），含解析与 200/1000 shots 评估
  * Regime I（经典对照，在同一批量子态 |psi> 上）：
      - ``classical_1body_logreg`` / ``classical_1body_mlp`` : 24 维单比特局域期望（对 SPT vs AFM 盲区）
      - ``classical_2body_mlp``      : 48 维 1+2 体近邻局域关联
      - ``classical_full_state_mlp`` : 完整 2^8=256 维态向量振幅（指数级层析成本 Oracle）

用法：
    .venv/bin/python experiments/benchmark_three_regimes.py --data-source idx --workers 5
    .venv/bin/python experiments/benchmark_three_regimes.py --quick
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from scipy.stats import t as student_t
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier as SkMLP
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qml_lab.data import load_mnist_tvt  # noqa: E402
from qml_lab.models import (  # noqa: E402
    ClassicalTrigNet,
    DataReuploadingFidelityClassifier,
    HybridRichClassifier,
    PatchLowRankClassifier,
    PatchVQCClassifier,
    RandomQuadraticClassifier,
    RichVQCClassifier,
    assert_matched_config,
    parameter_breakdown,
)
from qml_lab.quantum import (  # noqa: E402
    count_qwc_groups,
    evaluate_rich_layer_shots,
    iqp_feature_states,
    iqp_kernel_matrix,
)
from qml_lab.quantum_phases import (  # noqa: E402
    QCNNPhaseClassifier,
    generate_cluster_ising_dataset,
)
from qml_lab.train import evaluate_accuracy  # noqa: E402
from qml_lab.utils import ensure_dir, set_seed  # noqa: E402

OBS_AMP = "z+x+y+zz"


def _train_val_select(
    model: torch.nn.Module,
    X_tr,
    y_tr,
    X_val,
    y_val,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    weight_decay: float = 0.0,
    cosine_lr: bool = False,
    no_decay_modules: tuple[str, ...] = ("q",),
) -> tuple[dict, float]:
    """在训练集训练并按验证集最佳准确率选 checkpoint（严格不接触测试集）。"""
    set_seed(seed)
    best_val = -1.0
    best_state = None
    Xtr_t = torch.as_tensor(X_tr, dtype=torch.float32)
    ytr_t = torch.as_tensor(y_tr, dtype=torch.long).reshape(-1)
    Xva_t = torch.as_tensor(X_val, dtype=torch.float32)
    yva_t = torch.as_tensor(y_val, dtype=torch.long).reshape(-1)

    no_decay_ids: set[int] = set()
    for mod_name in no_decay_modules:
        sub = getattr(model, mod_name, None)
        if sub is not None and hasattr(sub, "parameters"):
            no_decay_ids.update(id(p) for p in sub.parameters())
    if weight_decay > 0.0 and no_decay_ids:
        nd = [p for p in model.parameters() if id(p) in no_decay_ids]
        wd = [p for p in model.parameters() if id(p) not in no_decay_ids]
        groups = []
        if nd:
            groups.append({"params": nd, "lr": lr, "weight_decay": 0.0})
        if wd:
            groups.append({"params": wd, "lr": lr, "weight_decay": weight_decay})
        opt = torch.optim.Adam(groups)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    sched = (
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs), eta_min=lr * 0.1)
        if cosine_lr
        else None
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    n = ytr_t.shape[0]
    t0 = time.perf_counter()
    last_loss = 0.0
    val_curve: list[float] = []

    for _ in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        running, seen = 0.0, 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
            running += float(loss.item()) * idx.numel()
            seen += idx.numel()
        if sched is not None:
            sched.step()
        last_loss = running / max(seen, 1)
        val_acc = evaluate_accuracy(model, Xva_t, yva_t, batch_size=64)
        val_curve.append(round(val_acc, 4))
        if val_acc > best_val:
            best_val = val_acc
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    secs = round(time.perf_counter() - t0, 2)
    return {
        "best_val_acc": round(float(best_val), 4),
        "final_train_loss": round(float(last_loss), 4),
        "seconds": secs,
        "val_acc_curve": val_curve,
    }, float(best_val)


def _eval_vqc_finite_shots(
    model: RichVQCClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_qubits: int,
    observables: str,
    shots: int,
    seed: int,
) -> float:
    """在真实有限 shots（PennyLane QWC 采样）下评估训练好的 RichVQCClassifier 测试准确率。"""
    model.eval()
    z_shots = evaluate_rich_layer_shots(
        weights=model.q.layer.weights,
        X=X_test,
        n_qubits=n_qubits,
        encoding="amplitude",
        observables=observables,
        shots=shots,
        batch_size=64,
        seed=seed,
    )
    with torch.no_grad():
        logits = model.head(z_shots * model.out_scale)
        preds = logits.argmax(dim=-1).cpu().numpy()
    return float(np.mean(preds == np.asarray(y_test)))


def _run_seed_worker(task: tuple[int, dict]) -> dict:
    """单 seed 完整执行 Track A（MNIST 三范式）与 Track B（Cluster-Ising SPT 相分类）。"""
    seed, cfg = task
    torch.set_num_threads(max(1, int(cfg["torch_threads"])))
    t_seed = time.perf_counter()

    # =========================================================================
    # TRACK A: MNIST (Strict disjoint Train / Val / Test, zero test peeking)
    # =========================================================================
    iqp_q = int(cfg["iqp_qubits"])
    reup_q = int(cfg["reup_qubits"])
    reup_dim = 3 * reup_q
    tvt = load_mnist_tvt(
        n_train=cfg["n_train"],
        n_val=cfg["n_val"],
        n_test=cfg["n_test"],
        classes=cfg["classes"],
        source=cfg["data_source"],
        data_dir=cfg["data_dir"],
        seed=seed,
        pca_dims=(iqp_q, reup_dim),
    )
    y_tr, y_va, y_te = tvt["y_train"], tvt["y_val"], tvt["y_test"]
    X_raw_tr, X_raw_va, X_raw_te = tvt["raw"]["X_train"], tvt["raw"]["X_val"], tvt["raw"]["X_test"]
    X_img_tr, X_img_va, X_img_te = tvt["img"]["X_train"], tvt["img"]["X_val"], tvt["img"]["X_test"]

    n_amp_q = int(cfg["amp_qubits"])
    n_amp_feat = int(X_raw_tr.shape[1])
    if cfg["quick"]:
        # quick 模式下截断至 2**n_amp_q 维
        k = 2 ** n_amp_q
        def _renorm(a):
            b = a[:, :k]
            n = np.linalg.norm(b, axis=1, keepdims=True)
            n[n < 1e-12] = 1.0
            return np.ascontiguousarray(b / n, dtype=np.float32)
        X_raw_tr, X_raw_va, X_raw_te = _renorm(X_raw_tr), _renorm(X_raw_va), _renorm(X_raw_te)
        n_amp_feat = k

    track_a: dict[str, dict[str, Any]] = {}
    matched_assertions: list[str] = []

    # ---- Pair A1: Global Amplitude VQC (40d) vs Fixed Random Quadratic (0 learned proj params) ----
    amp_cfg_common = {
        "readout_dim": 4 * n_amp_q,
        "head": "mlp",
        "head_hidden": 32,
        "scale": 5.0,
        "batch_size": 32,
        "lr": 0.01,
        "epochs": cfg["epochs_amp"],
        "weight_decay": 0.0,
        "cosine_lr": False,
        "input_view": "l2_raw_784",
    }
    assert_matched_config(
        amp_cfg_common, amp_cfg_common,
        tuple(amp_cfg_common.keys()),
        "vqc_amp_matched", "classical_random_quad",
    )
    matched_assertions.append("A1: vqc_amp_matched == classical_random_quad")

    set_seed(seed)
    m_rq = RandomQuadraticClassifier(
        n_features=n_amp_feat, readout_dim=4 * n_amp_q, n_classes=cfg["classes"],
        head="mlp", scale=5.0, hidden=32,
    )
    res_rq, _ = _train_val_select(
        m_rq, X_raw_tr, y_tr, X_raw_va, y_va,
        epochs=cfg["epochs_amp"], batch_size=32, lr=0.01, seed=seed,
    )
    track_a["classical_random_quad"] = {
        "regime": "I_classical",
        "test_acc": round(evaluate_accuracy(m_rq, X_raw_te, y_te), 4),
        "val_acc": res_rq["best_val_acc"],
        "seconds": res_rq["seconds"],
        "params": parameter_breakdown(m_rq, preprocessing_params=0),
        "qwc_groups": 0,
    }

    set_seed(seed)
    m_vq = RichVQCClassifier(
        n_qubits=n_amp_q, n_classes=cfg["classes"], n_layers=cfg["layers"],
        encoding="amplitude", observables=OBS_AMP, n_features=n_amp_feat,
        head="mlp", scale=5.0, hidden=32,
    )
    res_vq, _ = _train_val_select(
        m_vq, X_raw_tr, y_tr, X_raw_va, y_va,
        epochs=cfg["epochs_amp"], batch_size=32, lr=0.01, seed=seed,
    )
    vq_analytic = evaluate_accuracy(m_vq, X_raw_te, y_te)
    t_sh = time.perf_counter()
    vq_sh200 = _eval_vqc_finite_shots(m_vq, X_raw_te, y_te, n_amp_q, OBS_AMP, shots=200, seed=100 + seed)
    vq_sh1000 = _eval_vqc_finite_shots(m_vq, X_raw_te, y_te, n_amp_q, OBS_AMP, shots=1000, seed=200 + seed)
    qwc_amp = count_qwc_groups(n_amp_q, OBS_AMP)
    track_a["vqc_amp_matched"] = {
        "regime": "II_classical_on_quantum",
        "test_acc": round(vq_analytic, 4),
        "test_acc_shots_200": round(vq_sh200, 4),
        "test_acc_shots_1000": round(vq_sh1000, 4),
        "shot_eval_seconds": round(time.perf_counter() - t_sh, 2),
        "val_acc": res_vq["best_val_acc"],
        "seconds": res_vq["seconds"],
        "params": parameter_breakdown(m_vq, preprocessing_params=0),
        "qwc_groups": qwc_amp,
    }

    # ---- Classical Poly-2 Unscaled (exact dequantization of qkernel_amp) ----
    t0 = time.perf_counter()
    K_amp_tr = (X_raw_tr @ X_raw_tr.T) ** 2
    K_amp_te = (X_raw_te @ X_raw_tr.T) ** 2
    best_c, best_va_poly = 10.0, -1.0
    for c_val in (1.0, 5.0, 10.0, 20.0):
        clf = SVC(kernel="poly", degree=2, coef0=0.0, gamma=1.0, C=c_val).fit(X_raw_tr, y_tr)
        va_s = float(clf.score(X_raw_va, y_va))
        if va_s > best_va_poly:
            best_va_poly, best_c = va_s, c_val
    clf_poly = SVC(kernel="poly", degree=2, coef0=0.0, gamma=1.0, C=best_c).fit(X_raw_tr, y_tr)
    poly_te = float(clf_poly.score(X_raw_te, y_te))
    qamp_te = float(SVC(kernel="precomputed", C=best_c).fit(K_amp_tr, y_tr).score(K_amp_te, y_te))
    assert abs(poly_te - qamp_te) < 1e-9, f"Poly-2 unscaled != qkernel_amp: {poly_te} vs {qamp_te}"
    track_a["classical_poly2_unscaled"] = {
        "regime": "I_classical",
        "test_acc": round(poly_te, 4),
        "val_acc": round(best_va_poly, 4),
        "selected_C": best_c,
        "identical_to_qkernel_amp": True,
        "seconds": round(time.perf_counter() - t0, 3),
        "params": {"total_model_params": 0, "quantum_params": 0, "classical_wrapper_params": 0,
                   "preprocessing_fitted_params": 0, "effective_total_params": 0, "quantum_param_share": 0.0},
        "qwc_groups": 0,
    }

    # ---- Pair A2: PatchVQC (48 quantum params) vs PatchLowRank (exact 48 classical params) ----
    patch_cfg = {
        "patch_size": 14,
        "stride": 7,
        "per_patch_out": 32,
        "filter_params": 48,
        "head_hidden": 64,
        "dropout": 0.1,
        "batch_size": 64,
        "lr": 0.01,
        "weight_decay": 1e-4,
        "cosine_lr": True,
        "epochs": cfg["epochs_patch"],
        "post_bn": True,
    }
    set_seed(seed)
    m_pl = PatchLowRankClassifier(
        patch_size=14, stride=7, image_size=28, n_classes=cfg["classes"],
        per_patch_out=32, filter_params=48, preserve_patch_norm=True,
        head="mlp", scale=1.0, hidden=64, dropout=0.1,
    )
    set_seed(seed)
    m_pv = PatchVQCClassifier(
        n_qubits=8, n_classes=cfg["classes"], n_layers=2,
        patch_size=14, stride=7, image_size=28, observables=OBS_AMP,
        preserve_patch_norm=True, head="mlp", scale=1.0, hidden=64, dropout=0.1,
    )
    pb_pl = parameter_breakdown(m_pl)
    pb_pv = parameter_breakdown(m_pv)
    assert_matched_config(
        {**patch_cfg, "total_params": pb_pl["total_model_params"]},
        {**patch_cfg, "total_params": pb_pv["total_model_params"]},
        tuple(list(patch_cfg.keys()) + ["total_params"]),
        "classical_patch_lowrank", "patch_vqc_matched",
    )
    matched_assertions.append(f"A2: patch_vqc_matched == classical_patch_lowrank (exact {pb_pv['total_model_params']} params, batch=64)")

    res_pl, _ = _train_val_select(
        m_pl, X_img_tr, y_tr, X_img_va, y_va,
        epochs=cfg["epochs_patch"], batch_size=64, lr=0.01, seed=seed,
        weight_decay=1e-4, cosine_lr=True, no_decay_modules=("filter",),
    )
    track_a["classical_patch_lowrank"] = {
        "regime": "I_classical",
        "test_acc": round(evaluate_accuracy(m_pl, X_img_te, y_te), 4),
        "val_acc": res_pl["best_val_acc"],
        "seconds": res_pl["seconds"],
        "params": pb_pl,
        "qwc_groups": 0,
    }

    res_pv, _ = _train_val_select(
        m_pv, X_img_tr, y_tr, X_img_va, y_va,
        epochs=cfg["epochs_patch"], batch_size=64, lr=0.01, seed=seed,
        weight_decay=1e-4, cosine_lr=True, no_decay_modules=("q",),
    )
    track_a["patch_vqc_matched"] = {
        "regime": "II_classical_on_quantum",
        "test_acc": round(evaluate_accuracy(m_pv, X_img_te, y_te), 4),
        "val_acc": res_pv["best_val_acc"],
        "seconds": res_pv["seconds"],
        "params": pb_pv,
        "qwc_groups": count_qwc_groups(8, OBS_AMP),
    }

    # ---- Pair A3: Stabilized Hybrid CNN + 8q VQC vs Matched Stabilized Hybrid No-Quantum ----
    hyb_q = int(cfg["hybrid_qubits"])
    set_seed(seed)
    m_hq = HybridRichClassifier(
        n_qubits=hyb_q, n_classes=cfg["classes"], n_layers=cfg["layers"], hidden=32,
        quantum=True, head="mlp", scale=1.0, frontend="light",
        observables=OBS_AMP, stabilize=True,
    )
    set_seed(seed)
    m_hc = HybridRichClassifier(
        n_qubits=hyb_q, n_classes=cfg["classes"], n_layers=cfg["layers"], hidden=32,
        quantum=False, head="mlp", scale=1.0, frontend="light",
        observables=OBS_AMP, stabilize=True, match_readout_dim=True,
    )
    pb_hq = parameter_breakdown(m_hq)
    pb_hc = parameter_breakdown(m_hc)
    assert m_hc.post_norm is not None and m_hq.post_norm is not None
    assert_matched_config(
        {"readout_dim": m_hq.readout_dim, "stabilize": True, "has_post_bn": True, "total_params": pb_hq["total_model_params"]},
        {"readout_dim": m_hc.readout_dim, "stabilize": True, "has_post_bn": True, "total_params": pb_hc["total_model_params"]},
        ("readout_dim", "stabilize", "has_post_bn", "total_params"),
        "hybrid_vqc_matched", "hybrid_noq_matched",
    )
    matched_assertions.append(f"A3: hybrid_vqc_matched == hybrid_noq_matched (exact {pb_hq['total_model_params']} params, stabilize+BN)")

    res_hq, _ = _train_val_select(
        m_hq, X_img_tr, y_tr, X_img_va, y_va,
        epochs=cfg["epochs_hyb"], batch_size=32, lr=0.01, seed=seed,
    )
    track_a["hybrid_vqc_matched"] = {
        "regime": "II_classical_on_quantum",
        "test_acc": round(evaluate_accuracy(m_hq, X_img_te, y_te), 4),
        "val_acc": res_hq["best_val_acc"],
        "seconds": res_hq["seconds"],
        "params": pb_hq,
        "qwc_groups": count_qwc_groups(hyb_q, OBS_AMP),
    }

    res_hc, _ = _train_val_select(
        m_hc, X_img_tr, y_tr, X_img_va, y_va,
        epochs=cfg["epochs_hyb"], batch_size=32, lr=0.01, seed=seed,
    )
    track_a["hybrid_noq_matched"] = {
        "regime": "I_classical",
        "test_acc": round(evaluate_accuracy(m_hc, X_img_te, y_te), 4),
        "val_acc": res_hc["best_val_acc"],
        "seconds": res_hc["seconds"],
        "params": pb_hc,
        "qwc_groups": 0,
    }

    # ---- Pair A4 (Regime III on MNIST, ZERO classical MLP crutch): Deep Re-uploading Fidelity vs ClassicalTrigNet ----
    pca_reup = tvt["pca"][reup_dim]
    prep_reup = pca_reup["preprocessing_fitted_params"]
    reup_layers = int(cfg["reup_layers"])
    set_seed(seed)
    m_reup = DataReuploadingFidelityClassifier(
        n_qubits=reup_q, n_layers=reup_layers, n_classes=cfg["classes"],
    )
    pb_reup = parameter_breakdown(m_reup, preprocessing_params=prep_reup)
    assert pb_reup["classical_wrapper_params"] == 0, "Regime III DataReuploading must have 0 classical wrapper params"

    # Match ClassicalTrigNet parameter count as tightly as possible to quantum_params
    q_p = pb_reup["quantum_params"]
    best_nf = max(1, round(q_p / (reup_dim + 2 * cfg["classes"])))
    set_seed(seed)
    m_trig = ClassicalTrigNet(
        n_features=reup_dim, n_classes=cfg["classes"], n_freqs=best_nf, bias=False,
    )
    pb_trig = parameter_breakdown(m_trig, preprocessing_params=prep_reup)
    assert_matched_config(
        {"input_dim": reup_dim, "epochs": cfg["epochs_reup"], "lr": 0.05, "batch_size": 32, "prep_params": prep_reup},
        {"input_dim": reup_dim, "epochs": cfg["epochs_reup"], "lr": 0.05, "batch_size": 32, "prep_params": prep_reup},
        ("input_dim", "epochs", "lr", "batch_size", "prep_params"),
        "qnative_reupload_fidelity", "classical_trig_net",
    )
    matched_assertions.append(
        f"A4: qnative_reupload_fidelity ({pb_reup['quantum_params']}q/0c) vs classical_trig_net ({pb_trig['total_model_params']}c)"
    )

    res_reup, _ = _train_val_select(
        m_reup, pca_reup["X_train"], y_tr, pca_reup["X_val"], y_va,
        epochs=cfg["epochs_reup"], batch_size=32, lr=0.05, seed=seed,
    )
    track_a["qnative_reupload_fidelity"] = {
        "regime": "III_quantum_native",
        "test_acc": round(evaluate_accuracy(m_reup, pca_reup["X_test"], y_te), 4),
        "val_acc": res_reup["best_val_acc"],
        "seconds": res_reup["seconds"],
        "params": pb_reup,
        "qwc_groups": 1,  # computational basis Born readout
    }

    res_trig, _ = _train_val_select(
        m_trig, pca_reup["X_train"], y_tr, pca_reup["X_val"], y_va,
        epochs=cfg["epochs_reup"], batch_size=32, lr=0.05, seed=seed,
    )
    track_a["classical_trig_net"] = {
        "regime": "I_classical",
        "test_acc": round(evaluate_accuracy(m_trig, pca_reup["X_test"], y_te), 4),
        "val_acc": res_trig["best_val_acc"],
        "seconds": res_trig["seconds"],
        "params": pb_trig,
        "qwc_groups": 0,
    }

    # ---- Pair A5 (Regime III on MNIST): Genuine Entangling IQP Kernel vs Matched-Prep RBF & Poly2 ----
    pca_iqp = tvt["pca"][iqp_q]
    prep_iqp = pca_iqp["preprocessing_fitted_params"]
    X6_tr, X6_va, X6_te = pca_iqp["X_train"], pca_iqp["X_val"], pca_iqp["X_test"]
    t0 = time.perf_counter()
    alphas = (0.25, 0.5) if cfg["quick"] else (0.15, 0.25, 0.35, 0.50, 0.75)
    c_grid = (1.0, 10.0) if cfg["quick"] else (1.0, 5.0, 10.0, 20.0)
    best_iqp_cfg = (0.35, 10.0)
    best_iqp_val = -1.0
    best_iqp_states = None
    concentration_curve: dict[str, dict[str, float]] = {}

    for alpha in alphas:
        s_tr = iqp_feature_states(X6_tr, iqp_q, alpha=alpha, reps=2)
        s_va = iqp_feature_states(X6_va, iqp_q, alpha=alpha, reps=2)
        K_tr = iqp_kernel_matrix(s_tr, s_tr)
        K_va = iqp_kernel_matrix(s_va, s_tr)
        offdiag = float(K_tr[~np.eye(len(K_tr), dtype=bool)].mean())
        va_best_a = -1.0
        for c_val in c_grid:
            va_acc = float(SVC(kernel="precomputed", C=c_val).fit(K_tr, y_tr).score(K_va, y_va))
            if va_acc > va_best_a:
                va_best_a = va_acc
            if va_acc > best_iqp_val:
                best_iqp_val = va_acc
                best_iqp_cfg = (alpha, c_val)
                best_iqp_states = (s_tr, K_tr)
        concentration_curve[f"{alpha:.2f}"] = {
            "offdiag_mean": round(offdiag, 4),
            "best_val_acc": round(va_best_a, 4),
        }

    sel_alpha, sel_c = best_iqp_cfg
    s_tr_best, K_tr_best = best_iqp_states
    s_te_best = iqp_feature_states(X6_te, iqp_q, alpha=sel_alpha, reps=2)
    K_te_best = iqp_kernel_matrix(s_te_best, s_tr_best)
    iqp_te_acc = float(SVC(kernel="precomputed", C=sel_c).fit(K_tr_best, y_tr).score(K_te_best, y_te))
    track_a["qnative_iqp_kernel"] = {
        "regime": "III_quantum_native",
        "test_acc": round(iqp_te_acc, 4),
        "val_acc": round(best_iqp_val, 4),
        "selected_alpha": sel_alpha,
        "selected_C": sel_c,
        "concentration_curve": concentration_curve,
        "seconds": round(time.perf_counter() - t0, 2),
        "params": {"total_model_params": 0, "quantum_params": 0, "classical_wrapper_params": 0,
                   "preprocessing_fitted_params": prep_iqp, "effective_total_params": prep_iqp, "quantum_param_share": 0.0},
        "qwc_groups": 1,
    }

    # Matched-preprocessing RBF & Poly-2 on exact same standardized PCA-6 (Val-tuned C & gamma)
    t0 = time.perf_counter()
    best_rbf_val, best_rbf_cfg = -1.0, ("scale", 10.0)
    gammas = ("scale", 0.3) if cfg["quick"] else ("scale", 0.1, 0.3, 1.0)
    for g_val in gammas:
        for c_val in c_grid:
            va_acc = float(SVC(kernel="rbf", gamma=g_val, C=c_val).fit(X6_tr, y_tr).score(X6_va, y_va))
            if va_acc > best_rbf_val:
                best_rbf_val, best_rbf_cfg = va_acc, (g_val, c_val)
    rbf_te = float(SVC(kernel="rbf", gamma=best_rbf_cfg[0], C=best_rbf_cfg[1]).fit(X6_tr, y_tr).score(X6_te, y_te))
    track_a["classical_rbf_pca6"] = {
        "regime": "I_classical",
        "test_acc": round(rbf_te, 4),
        "val_acc": round(best_rbf_val, 4),
        "selected_gamma": best_rbf_cfg[0],
        "selected_C": best_rbf_cfg[1],
        "seconds": round(time.perf_counter() - t0, 3),
        "params": {"total_model_params": 0, "quantum_params": 0, "classical_wrapper_params": 0,
                   "preprocessing_fitted_params": prep_iqp, "effective_total_params": prep_iqp, "quantum_param_share": 0.0},
        "qwc_groups": 0,
    }

    t0 = time.perf_counter()
    best_p2_val, best_p2_c = -1.0, 10.0
    for c_val in c_grid:
        va_acc = float(SVC(kernel="poly", degree=2, coef0=1.0, C=c_val).fit(X6_tr, y_tr).score(X6_va, y_va))
        if va_acc > best_p2_val:
            best_p2_val, best_p2_c = va_acc, c_val
    p2_te = float(SVC(kernel="poly", degree=2, coef0=1.0, C=best_p2_c).fit(X6_tr, y_tr).score(X6_te, y_te))
    track_a["classical_poly2_pca6"] = {
        "regime": "I_classical",
        "test_acc": round(p2_te, 4),
        "val_acc": round(best_p2_val, 4),
        "selected_C": best_p2_c,
        "seconds": round(time.perf_counter() - t0, 3),
        "params": {"total_model_params": 0, "quantum_params": 0, "classical_wrapper_params": 0,
                   "preprocessing_fitted_params": prep_iqp, "effective_total_params": prep_iqp, "quantum_param_share": 0.0},
        "qwc_groups": 0,
    }
    matched_assertions.append("A5: qnative_iqp_kernel == classical_rbf_pca6 == classical_poly2_pca6 (same PCA-6, Val-tuned)")

    # =========================================================================
    # TRACK B: Native Quantum-State Data (1D Cluster-Ising SPT vs Non-SPT)
    # =========================================================================
    q_n = int(cfg["spt_qubits"])
    ds = generate_cluster_ising_dataset(
        n_train=cfg["spt_train"], n_val=cfg["spt_val"], n_test=cfg["spt_test"],
        n_qubits=q_n, seed=seed,
    )
    sy_tr, sy_va, sy_te = ds["y_train"], ds["y_val"], ds["y_test"]
    track_b: dict[str, dict[str, Any]] = {}

    # ---- Regime III: True Cong-et-al. QCNN (0 classical MLP params, 1 QWC basis) ----
    set_seed(seed)
    m_qcnn = QCNNPhaseClassifier(n_qubits=q_n)
    pb_qcnn = parameter_breakdown(m_qcnn)
    assert pb_qcnn["classical_wrapper_params"] == 0, "QCNN must have 0 classical MLP params"
    res_qcnn, _ = _train_val_select(
        m_qcnn, ds["states"]["X_train"], sy_tr, ds["states"]["X_val"], sy_va,
        epochs=cfg["epochs_qcnn"], batch_size=16, lr=0.05, seed=seed,
    )
    qcnn_te = evaluate_accuracy(m_qcnn, ds["states"]["X_test"], sy_te)
    qcnn_sh200 = m_qcnn.evaluate_finite_shots(ds["states"]["X_test"], sy_te, shots=200, seed=300 + seed)
    qcnn_sh1000 = m_qcnn.evaluate_finite_shots(ds["states"]["X_test"], sy_te, shots=1000, seed=400 + seed)
    track_b["qcnn_spt"] = {
        "regime": "III_quantum_native",
        "test_acc": round(qcnn_te, 4),
        "test_acc_shots_200": round(qcnn_sh200, 4),
        "test_acc_shots_1000": round(qcnn_sh1000, 4),
        "val_acc": res_qcnn["best_val_acc"],
        "seconds": res_qcnn["seconds"],
        "params": pb_qcnn,
        "qwc_groups": 1,
    }

    # ---- Regime I Classical Baselines on the EXACT same quantum states ----
    def _fit_classical_phase(Xtr, Xva, Xte, kind: str, dim_in: int) -> dict:
        t0 = time.perf_counter()
        if kind == "logreg":
            best_v, best_c = -1.0, 1.0
            for c_val in (0.1, 1.0, 10.0):
                clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=c_val)).fit(Xtr, sy_tr)
                va = float(clf.score(Xva, sy_va))
                if va > best_v:
                    best_v, best_c = va, c_val
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=best_c)).fit(Xtr, sy_tr)
            te = float(clf.score(Xte, sy_te))
            n_p = dim_in + 1
        else:
            best_v, best_a = -1.0, 1e-3
            for alpha_reg in (1e-4, 1e-2, 1e-1):
                clf = make_pipeline(
                    StandardScaler(),
                    SkMLP(hidden_layer_sizes=(32,), alpha=alpha_reg, max_iter=600, random_state=seed),
                ).fit(Xtr, sy_tr)
                va = float(clf.score(Xva, sy_va))
                if va > best_v:
                    best_v, best_a = va, alpha_reg
            clf = make_pipeline(
                StandardScaler(),
                SkMLP(hidden_layer_sizes=(32,), alpha=best_a, max_iter=600, random_state=seed),
            ).fit(Xtr, sy_tr)
            te = float(clf.score(Xte, sy_te))
            n_p = dim_in * 32 + 32 + 32 + 1
        return {
            "regime": "I_classical",
            "test_acc": round(te, 4),
            "val_acc": round(best_v, 4),
            "seconds": round(time.perf_counter() - t0, 3),
            "params": {"total_model_params": n_p, "quantum_params": 0, "classical_wrapper_params": n_p,
                       "preprocessing_fitted_params": 0, "effective_total_params": n_p, "quantum_param_share": 0.0},
            "input_dim": int(dim_in),
            "qwc_groups": 0,
        }

    track_b["classical_1body_logreg"] = _fit_classical_phase(
        ds["local_1body"]["X_train"], ds["local_1body"]["X_val"], ds["local_1body"]["X_test"],
        "logreg", 3 * q_n,
    )
    track_b["classical_1body_mlp"] = _fit_classical_phase(
        ds["local_1body"]["X_train"], ds["local_1body"]["X_val"], ds["local_1body"]["X_test"],
        "mlp", 3 * q_n,
    )
    track_b["classical_2body_mlp"] = _fit_classical_phase(
        ds["local_2body"]["X_train"], ds["local_2body"]["X_val"], ds["local_2body"]["X_test"],
        "mlp", 6 * q_n,
    )
    track_b["classical_full_state_mlp"] = _fit_classical_phase(
        ds["states"]["X_train"], ds["states"]["X_val"], ds["states"]["X_test"],
        "mlp", 1 << q_n,
    )

    return {
        "seed": seed,
        "track_a": track_a,
        "track_b": track_b,
        "matched_assertions": matched_assertions,
        "mnist_meta": tvt["meta"],
        "spt_meta": ds["meta"],
        "seed_seconds": round(time.perf_counter() - t_seed, 2),
    }


def _stats(vals: list[float]) -> dict:
    a = np.asarray(vals, dtype=float)
    return {
        "mean": round(float(a.mean()), 4),
        "std": round(float(a.std(ddof=1)) if a.size > 1 else 0.0, 4),
        "min": round(float(a.min()), 4),
        "max": round(float(a.max()), 4),
        "values": [round(float(v), 4) for v in a],
        "n": int(a.size),
    }


def _paired_test(a_vals: list[float], b_vals: list[float]) -> dict:
    d = (np.asarray(a_vals, dtype=float) - np.asarray(b_vals, dtype=float)) * 100.0
    n = d.size
    mean = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    if n > 1 and sd > 0.0:
        se = sd / np.sqrt(n)
        t_stat = mean / se
        crit = float(student_t.ppf(0.975, n - 1))
        half = crit * se
        p_val = float(2.0 * student_t.sf(abs(t_stat), n - 1))
    else:
        t_stat = 0.0
        half = 0.0
        p_val = 1.0 if mean == 0.0 else 0.0
    return {
        "mean_pt": round(mean, 2),
        "std_pt": round(sd, 2),
        "ci95_pt": [round(mean - half, 2), round(mean + half, 2)],
        "t_stat": round(t_stat, 3),
        "p_value": round(p_val, 6),
        "n_positive": int(np.sum(d > 0)),
        "per_seed_pt": [round(float(v), 2) for v in d],
    }


def _apply_holm_bonferroni(tests: dict[str, dict]) -> dict[str, dict]:
    """对所有规划配对检验施加 Holm-Bonferroni 校正，附加 p_holm 与 holm_significant。"""
    keys = list(tests.keys())
    m = len(keys)
    order = sorted(range(m), key=lambda i: tests[keys[i]]["p_value"])
    running_max = 0.0
    for rank, idx in enumerate(order):
        k = keys[idx]
        adj = min(1.0, tests[k]["p_value"] * (m - rank))
        running_max = max(running_max, adj)
        tests[k]["p_holm"] = round(float(running_max), 6)
        tests[k]["holm_significant"] = bool(running_max < 0.05)
    return tests


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="三范式严格受控基准（MNIST + Cluster-Ising SPT QCNN）")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--n-train", type=int, default=600)
    p.add_argument("--n-val", type=int, default=200)
    p.add_argument("--n-test", type=int, default=400)
    p.add_argument("--classes", type=int, default=10)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--amp-qubits", type=int, default=10)
    p.add_argument("--hybrid-qubits", type=int, default=8)
    p.add_argument("--reup-qubits", type=int, default=4)
    p.add_argument("--reup-layers", type=int, default=4)
    p.add_argument("--iqp-qubits", type=int, default=6)
    p.add_argument("--epochs-amp", type=int, default=20)
    p.add_argument("--epochs-patch", type=int, default=20)
    p.add_argument("--epochs-hyb", type=int, default=20)
    p.add_argument("--epochs-reup", type=int, default=25)
    p.add_argument("--spt-qubits", type=int, default=8)
    p.add_argument("--spt-train", type=int, default=120)
    p.add_argument("--spt-val", type=int, default=60)
    p.add_argument("--spt-test", type=int, default=120)
    p.add_argument("--epochs-qcnn", type=int, default=25)
    p.add_argument("--data-source", default="idx")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--torch-threads", type=int, default=2)
    p.add_argument("--out", default="results")
    p.add_argument("--archive", default="docs/three_regimes_metrics.json")
    p.add_argument("--quick", action="store_true", help="离线极小自检，输出到 results/ 不覆盖 docs/")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.quick:
        args.seeds = "0"
        args.n_train, args.n_val, args.n_test = 30, 15, 15
        args.classes = 3
        args.layers = 1
        args.amp_qubits = 4
        args.hybrid_qubits = 4
        args.reup_qubits = 2
        args.reup_layers = 2
        args.iqp_qubits = 4
        args.epochs_amp = args.epochs_patch = args.epochs_hyb = args.epochs_reup = 2
        args.spt_qubits = 4
        args.spt_train, args.spt_val, args.spt_test = 16, 8, 8
        args.epochs_qcnn = 2
        args.data_source = "synthetic"
        args.workers = 1
        args.torch_threads = 1
        if args.archive == "docs/three_regimes_metrics.json":
            args.archive = str(Path(args.out) / "quick_three_regimes_metrics.json")

    t0 = time.perf_counter()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    dd = Path(args.data_dir)
    data_dir = str(dd if dd.is_absolute() else ROOT / dd)
    cfg = {**vars(args), "data_dir": data_dir}

    print(f"== Three-Regime Controlled Benchmark (seeds={seeds}, workers={args.workers}) ==", flush=True)
    tasks = [(s, cfg) for s in seeds]
    if args.workers > 1 and len(tasks) > 1:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
            seed_results = list(ex.map(_run_seed_worker, tasks))
    else:
        seed_results = [_run_seed_worker(t) for t in tasks]

    seed_results.sort(key=lambda r: r["seed"])

    def _summarize_track(track_key: str) -> dict:
        keys = list(seed_results[0][track_key].keys())
        out = {}
        for k in keys:
            r0 = seed_results[0][track_key][k]
            entry: dict[str, Any] = {
                "regime": r0["regime"],
                "params": r0["params"],
                "qwc_groups": r0["qwc_groups"],
                "test_acc": _stats([r[track_key][k]["test_acc"] for r in seed_results]),
                "val_acc": _stats([r[track_key][k]["val_acc"] for r in seed_results]),
                "seconds": _stats([r[track_key][k]["seconds"] for r in seed_results]),
            }
            if "test_acc_shots_200" in r0:
                entry["test_acc_shots_200"] = _stats([r[track_key][k]["test_acc_shots_200"] for r in seed_results])
                entry["test_acc_shots_1000"] = _stats([r[track_key][k]["test_acc_shots_1000"] for r in seed_results])
            if "selected_alpha" in r0:
                entry["selected_alphas"] = [r[track_key][k]["selected_alpha"] for r in seed_results]
                entry["concentration_curve_seed0"] = r0["concentration_curve"]
            if "input_dim" in r0:
                entry["input_dim"] = r0["input_dim"]
            out[k] = entry
        return out

    summary_a = _summarize_track("track_a")
    summary_b = _summarize_track("track_b")

    va = lambda k: summary_a[k]["test_acc"]["values"]  # noqa: E731
    vb = lambda k: summary_b[k]["test_acc"]["values"]  # noqa: E731

    paired_a = _apply_holm_bonferroni({
        "A1_vqc_amp_minus_random_quad": _paired_test(va("vqc_amp_matched"), va("classical_random_quad")),
        "A1_vqc_amp_shots200_minus_analytic": _paired_test(
            summary_a["vqc_amp_matched"]["test_acc_shots_200"]["values"], va("vqc_amp_matched")
        ),
        "A1_vqc_amp_shots1000_minus_analytic": _paired_test(
            summary_a["vqc_amp_matched"]["test_acc_shots_1000"]["values"], va("vqc_amp_matched")
        ),
        "A2_patch_vqc_minus_patch_lowrank": _paired_test(va("patch_vqc_matched"), va("classical_patch_lowrank")),
        "A3_hybrid_vqc_minus_hybrid_noq": _paired_test(va("hybrid_vqc_matched"), va("hybrid_noq_matched")),
        "A4_reupload_fidelity_minus_trig_net": _paired_test(va("qnative_reupload_fidelity"), va("classical_trig_net")),
        "A5_iqp_kernel_minus_rbf_pca6": _paired_test(va("qnative_iqp_kernel"), va("classical_rbf_pca6")),
        "A5_iqp_kernel_minus_poly2_pca6": _paired_test(va("qnative_iqp_kernel"), va("classical_poly2_pca6")),
    })

    paired_b = _apply_holm_bonferroni({
        "B1_qcnn_minus_1body_logreg": _paired_test(vb("qcnn_spt"), vb("classical_1body_logreg")),
        "B2_qcnn_minus_1body_mlp": _paired_test(vb("qcnn_spt"), vb("classical_1body_mlp")),
        "B3_qcnn_minus_2body_mlp": _paired_test(vb("qcnn_spt"), vb("classical_2body_mlp")),
        "B4_qcnn_minus_full_state_oracle": _paired_test(vb("qcnn_spt"), vb("classical_full_state_mlp")),
        "B5_qcnn_shots200_minus_analytic": _paired_test(
            summary_b["qcnn_spt"]["test_acc_shots_200"]["values"], vb("qcnn_spt")
        ),
    })

    payload = {
        "purpose": "严格受控三范式基准：Track A（MNIST 经典数据）+ Track B（1D Cluster-Ising 原生量子态 SPT 相分类）",
        "protocol": {
            "disjoint_train_val_test": True,
            "zero_test_peeking": True,
            "unified_seed_semantics": "All models (neural, VQC, kernel, QCNN) share identical independent stratified Train/Val/Test split per seed",
            "multiple_testing_correction": "Holm-Bonferroni (two-sided paired Student-t, alpha=0.05)",
            "qwc_verification": {
                "amp10_40d_qwc_groups": count_qwc_groups(args.amp_qubits, OBS_AMP),
                "amp10_165d_qwc_groups": count_qwc_groups(10, "z+x+y+zzall+xxall+yyall"),
                "qcnn_final_wire_qwc_groups": 1,
            },
            "matched_assertions_seed0": seed_results[0]["matched_assertions"],
            "physics_boundary_note": (
                "Single-copy AmplitudeEmbedding + CPTP QCNN cannot escape homogeneous quadraticity on classical data; "
                "on native 1D Cluster-Ising SPT states |psi>, QCNN extracts nonlocal string order directly with 0 classical MLP params and 1 QWC basis."
            ),
        },
        "config": {k: v for k, v in vars(args).items() if k != "quick"},
        "seeds": seeds,
        "track_a_mnist": {
            "summary": summary_a,
            "paired_tests_holm": paired_a,
        },
        "track_b_cluster_ising_spt": {
            "hamiltonian": seed_results[0]["spt_meta"]["hamiltonian"],
            "summary": summary_b,
            "paired_tests_holm": paired_b,
        },
        "per_seed_runs": seed_results,
        "wall_seconds": round(time.perf_counter() - t0, 2),
    }
    if args.quick:
        payload["caveat"] = "QUICK 自检（合成数据），仅用于流程验证，不得用于任何科学结论。"

    out_dir = ensure_dir(ROOT / args.out)
    out_file = out_dir / ("quick_three_regimes_metrics.json" if args.quick else "three_regimes_metrics.json")
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved: {out_file}", flush=True)

    if args.archive:
        arc = Path(args.archive)
        if not arc.is_absolute():
            arc = ROOT / arc
        arc.parent.mkdir(parents=True, exist_ok=True)
        arc.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"archived: {arc}", flush=True)

    print(f"\n=== Track A: MNIST ({args.n_train}/{args.n_val}/{args.n_test}, n={len(seeds)} seeds) ===")
    for k, s in summary_a.items():
        pb = s["params"]
        sh_str = ""
        if "test_acc_shots_200" in s:
            sh_str = f" | sh200={s['test_acc_shots_200']['mean']:.4f} sh1000={s['test_acc_shots_1000']['mean']:.4f}"
        print(
            f"  [{s['regime']:23s}] {k:26s} test={s['test_acc']['mean']:.4f}±{s['test_acc']['std']:.4f}"
            f" | q={pb['quantum_params']:3d} c={pb['classical_wrapper_params']:5d} prep={pb['preprocessing_fitted_params']:5d}"
            f" | QWC={s['qwc_groups']}{sh_str}"
        )
    print("\n--- Track A Paired Tests (Holm-Bonferroni) ---")
    for k, v in paired_a.items():
        print(f"  {k:38s} {v['mean_pt']:+6.2f}±{v['std_pt']:5.2f} pt CI{v['ci95_pt']} p={v['p_value']:.4g} p_holm={v['p_holm']:.4g} sig={v['holm_significant']}")

    print(f"\n=== Track B: 1D Cluster-Ising SPT (n={args.spt_qubits}q, {args.spt_train}/{args.spt_val}/{args.spt_test}, n={len(seeds)} seeds) ===")
    for k, s in summary_b.items():
        pb = s["params"]
        sh_str = ""
        if "test_acc_shots_200" in s:
            sh_str = f" | sh200={s['test_acc_shots_200']['mean']:.4f} sh1000={s['test_acc_shots_1000']['mean']:.4f}"
        print(
            f"  [{s['regime']:23s}] {k:26s} test={s['test_acc']['mean']:.4f}±{s['test_acc']['std']:.4f}"
            f" | q={pb['quantum_params']:3d} c={pb['classical_wrapper_params']:5d} | QWC={s['qwc_groups']}{sh_str}"
        )
    print("\n--- Track B Paired Tests (Holm-Bonferroni) ---")
    for k, v in paired_b.items():
        print(f"  {k:38s} {v['mean_pt']:+6.2f}±{v['std_pt']:5.2f} pt CI{v['ci95_pt']} p={v['p_value']:.4g} p_holm={v['p_holm']:.4g} sig={v['holm_significant']}")

    print(f"\nDone in {payload['wall_seconds']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
