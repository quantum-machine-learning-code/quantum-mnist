#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""硬件耦合拓扑 SWAP 路由开销与去极化噪声基准实验（PennyLane default.mixed + NetworkX）。

评估 4 种标准量子硬件耦合拓扑（all_to_all / linear_chain / grid_2d / heavy_hex）：
  Part A — 5 条真实基准线路的双向 SABRE SWAP 路由开销与精确 ASAP 2Q 深度：
      vqc_amp (10q), patch_vqc (8q), qnative_iqp_kernel (6q, 2-rep 全连接 ZZ),
      qcnn_spt (8q, Cong-et-al. QCNN), qft_8 (8q QFT，衔接 Shor 算法)。
  Part B — Track B 原生量子态 qcnn_spt (n=8, 1D Cluster-Ising SPT, 5 seeds)
      在 4 种物理路由拓扑 × p_2q in {0.0, 0.002, 0.005, 0.01} (+ p_ro=0.01 读出误差消融)
      下的精确 PennyLane default.mixed 密度矩阵分类准确率衰减与配对检验。
  Part C — Track A 经典数据 qnative_iqp_kernel (n=6, 2-rep 全连接 ZZ, MNIST PCA-6, 5 seeds)
      在 4 种物理路由拓扑 × p_2q in {0.0, 0.002, 0.005, 0.01} 下的真实 default.mixed
      Hilbert-Schmidt 核准确率衰减、核非对角标准差集中（Thanasilp et al. 2024）与 Frobenius 漂移。

用法：
    .venv/bin/python experiments/benchmark_hardware_topology.py --data-source idx --workers 16
    .venv/bin/python experiments/benchmark_hardware_topology.py --quick
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t
from sklearn.svm import SVC
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qml_lab.data import load_mnist_tvt  # noqa: E402
from qml_lab.hardware_topology import (  # noqa: E402
    TOPOLOGY_NAMES,
    build_coupling_graph,
    evaluate_routed_iqp_density_matrices,
    evaluate_routed_qcnn_mixed,
    execute_gates_statevector,
    extract_iqp_kernel_gates,
    extract_patch_vqc_gates,
    extract_qcnn_spt_gates,
    extract_qft_gates,
    extract_vqc_amp_gates,
    hilbert_schmidt_kernel_matrix,
    kernel_concentration_metrics,
    route_circuit_to_topology,
    topology_stats,
)
from qml_lab.quantum_phases import (  # noqa: E402
    QCNNPhaseClassifier,
    generate_cluster_ising_dataset,
)
from qml_lab.train import evaluate_accuracy  # noqa: E402
from qml_lab.utils import ensure_dir, set_seed  # noqa: E402


def _stats(vals: list[float]) -> dict[str, Any]:
    arr = np.asarray(vals, dtype=float)
    return {
        "mean": round(float(arr.mean()), 4),
        "std": round(float(arr.std(ddof=1)) if arr.size > 1 else 0.0, 4),
        "min": round(float(arr.min()), 4),
        "max": round(float(arr.max()), 4),
        "values": [round(float(v), 4) for v in arr],
        "n": int(arr.size),
    }


def _stats_prec(vals: list[float], digits: int = 6) -> dict[str, Any]:
    arr = np.asarray(vals, dtype=float)
    return {
        "mean": round(float(arr.mean()), digits),
        "std": round(float(arr.std(ddof=1)) if arr.size > 1 else 0.0, digits),
        "values": [round(float(v), digits) for v in arr],
        "n": int(arr.size),
    }


def _paired_test(a_vals: list[float], b_vals: list[float]) -> dict[str, Any]:
    diff = (np.asarray(a_vals, dtype=float) - np.asarray(b_vals, dtype=float)) * 100.0
    n = diff.size
    mean = float(diff.mean())
    sd = float(diff.std(ddof=1)) if n > 1 else 0.0
    if n > 1 and sd > 0.0:
        se = sd / np.sqrt(n)
        t_stat = mean / se
        crit = float(student_t.ppf(0.975, n - 1))
        half = crit * se
        p_val = float(2.0 * student_t.sf(abs(t_stat), n - 1))
    else:
        t_stat = 0.0
        half = 0.0
        p_val = 1.0 if abs(mean) < 1e-12 else 0.0
    return {
        "mean_pt": round(mean, 2),
        "std_pt": round(sd, 2),
        "ci95_pt": [round(mean - half, 2), round(mean + half, 2)],
        "t_stat": round(t_stat, 3),
        "p_value": round(p_val, 6),
        "n_positive": int(np.sum(diff > 0)),
        "per_seed_pt": [round(float(v), 2) for v in diff],
    }


def _apply_holm_bonferroni(tests: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    keys = list(tests.keys())
    m = len(keys)
    order = sorted(range(m), key=lambda idx: tests[keys[idx]]["p_value"])
    running_max = 0.0
    for rank, idx in enumerate(order):
        key = keys[idx]
        adj = min(1.0, tests[key]["p_value"] * (m - rank))
        running_max = max(running_max, adj)
        tests[key]["p_holm"] = round(float(running_max), 6)
        tests[key]["holm_significant"] = bool(running_max < 0.05)
    return tests


# ---------------------------------------------------------------------------
# Part A: 5 条真实基准线路 × 4 种硬件拓扑的 SWAP 路由开销统计
# ---------------------------------------------------------------------------

def run_routing_overhead_benchmark(quick: bool = False) -> dict[str, Any]:
    """统计 5 条真实线路在 4 种硬件耦合拓扑上的物理 2Q 门数、SWAP 增量与 ASAP 深度。"""
    rng = np.random.default_rng(7)
    if quick:
        specs = {
            "vqc_amp": (4, extract_vqc_amp_gates(4, 1)),
            "patch_vqc": (4, extract_patch_vqc_gates(4, 1)),
            "qnative_iqp_kernel": (4, extract_iqp_kernel_gates(4, 2, alpha=0.25, x=rng.uniform(-1, 1, 4))),
            "qcnn_spt": (4, extract_qcnn_spt_gates(4)),
            "qft_8": (4, extract_qft_gates(4)),
        }
        graph_sizes = (4,)
    else:
        specs = {
            "vqc_amp": (10, extract_vqc_amp_gates(10, 2)),
            "patch_vqc": (8, extract_patch_vqc_gates(8, 2)),
            "qnative_iqp_kernel": (6, extract_iqp_kernel_gates(6, 2, alpha=0.25, x=rng.uniform(-1, 1, 6))),
            "qcnn_spt": (8, extract_qcnn_spt_gates(8)),
            "qft_8": (8, extract_qft_gates(8)),
        }
        graph_sizes = (6, 8, 10)

    graph_catalog: dict[str, dict[str, Any]] = {}
    for nq in graph_sizes:
        graph_catalog[f"n{nq}"] = {
            topo: topology_stats(build_coupling_graph(topo, nq)) for topo in TOPOLOGY_NAMES
        }

    circuits_out: dict[str, dict[str, Any]] = {}
    max_sv_err_global = 0.0
    for cname, (nq, gates) in specs.items():
        psi0 = rng.normal(size=1 << nq) + 1j * rng.normal(size=1 << nq)
        psi0 /= np.linalg.norm(psi0)
        init_st = None if cname == "qnative_iqp_kernel" else psi0
        ref_sv = execute_gates_statevector(gates, nq, initial_state=init_st)

        per_topo: dict[str, Any] = {}
        for topo in TOPOLOGY_NAMES:
            graph = build_coupling_graph(topo, nq)
            res = route_circuit_to_topology(gates, graph, nq)
            routed_sv = execute_gates_statevector(
                res["routed_gates"],
                nq,
                initial_state=init_st,
                initial_layout=res["initial_layout"],
                final_layout=res["final_layout"],
            )
            sv_err = float(np.max(np.abs(ref_sv - routed_sv)))
            assert sv_err < 1e-10, f"路由态向量验证失败 {cname}/{topo}: {sv_err}"
            max_sv_err_global = max(max_sv_err_global, sv_err)
            per_topo[topo] = {
                "n_qubits": res["n_qubits"],
                "logical_2q_gates": res["logical_2q_gates"],
                "inserted_swaps": res["inserted_swaps"],
                "physical_2q_gates": res["physical_2q_gates"],
                "overhead_ratio": res["overhead_ratio"],
                "logical_2q_depth": res["logical_2q_depth"],
                "physical_2q_depth": res["physical_2q_depth"],
                "depth_overhead_ratio": res["depth_overhead_ratio"],
                "initial_layout": res["initial_layout"],
                "final_layout": res["final_layout"],
                "noiseless_statevector_err": float(f"{sv_err:.3e}"),
            }
        circuits_out[cname] = per_topo

    return {
        "graph_stats": graph_catalog,
        "circuits": circuits_out,
        "max_noiseless_statevector_err": float(f"{max_sv_err_global:.3e}"),
    }


# ---------------------------------------------------------------------------
# Part B: Track B qcnn_spt (n=8) 训练与路由含噪 default.mixed 评估
# ---------------------------------------------------------------------------

def _qcnn_seed_worker(task: tuple[int, dict[str, Any]]) -> dict[str, Any]:
    """单 seed 训练 Cong-et-al. QCNN 并在 4 种拓扑 × p_2q × p_ro 下执行精确 default.mixed 评估。"""
    seed, cfg = task
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(max(1, int(cfg["torch_threads"])))
    t0 = time.perf_counter()

    n_q = int(cfg["spt_qubits"])
    ds = generate_cluster_ising_dataset(
        n_train=int(cfg["spt_train"]),
        n_val=int(cfg["spt_val"]),
        n_test=int(cfg["spt_test"]),
        n_qubits=n_q,
        seed=seed,
    )
    X_tr, X_va, X_te = ds["states"]["X_train"], ds["states"]["X_val"], ds["states"]["X_test"]
    y_tr, y_va, y_te = ds["y_train"], ds["y_val"], ds["y_test"]

    set_seed(seed)
    model = QCNNPhaseClassifier(n_qubits=n_q)
    Xtr_t = torch.as_tensor(X_tr, dtype=torch.float32)
    ytr_t = torch.as_tensor(y_tr, dtype=torch.long)
    Xva_t = torch.as_tensor(X_va, dtype=torch.float32)
    yva_t = torch.as_tensor(y_va, dtype=torch.long)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_val = -1.0
    best_state = None
    n_samples = len(y_tr)
    for _ in range(int(cfg["epochs_qcnn"])):
        model.train()
        perm = torch.randperm(n_samples)
        for start in range(0, n_samples, 16):
            idx = perm[start:start + 16]
            opt.zero_grad()
            loss = loss_fn(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
        va_acc = evaluate_accuracy(model, Xva_t, yva_t, batch_size=64)
        if va_acc > best_val:
            best_val = va_acc
            best_state = copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)

    # default.qubit 无噪参考准确率
    qubit_probs = model.born_probs(torch.as_tensor(X_te, dtype=torch.float32)).detach().cpu().numpy()
    qubit_acc = float(np.mean(np.argmax(qubit_probs, axis=-1) == y_te))

    cw = model.q.layer.conv_w.detach().cpu().numpy()
    pw = model.q.layer.pool_w.detach().cpu().numpy()
    fr = model.q.layer.final_rot.detach().cpu().numpy()
    logical_gates = extract_qcnn_spt_gates(n_qubits=n_q, conv_w=cw, pool_w=pw, final_rot=fr)

    p2q_list: list[float] = list(cfg["p2q_grid"])
    p_ro_ablation = float(cfg["readout_error"])
    ro_anchor_p2q = 0.005 if 0.005 in p2q_list else p2q_list[-1]

    topo_results: dict[str, Any] = {}
    max_p0_diff = 0.0
    for topo in TOPOLOGY_NAMES:
        graph = build_coupling_graph(topo, n_q)
        routed = route_circuit_to_topology(logical_gates, graph, n_q)
        noise_map: dict[str, float] = {}
        for p2q in p2q_list:
            probs = evaluate_routed_qcnn_mixed(routed, X_te, n_q, p_2q=p2q, p_ro=0.0)
            if p2q == 0.0:
                diff = float(np.max(np.abs(probs - qubit_probs)))
                assert diff < 1e-5, f"QCNN default.mixed p=0 与 default.qubit 偏差过大: {diff}"
                max_p0_diff = max(max_p0_diff, diff)
            acc = float(np.mean(np.argmax(probs, axis=-1) == y_te))
            noise_map[f"{p2q:.3f}"] = round(acc, 4)

        # 读出误差消融（p_ro=0.01）：分别在 p_2q=0.0 与 p_2q=ro_anchor_p2q 下实测
        probs_ro0 = evaluate_routed_qcnn_mixed(routed, X_te, n_q, p_2q=0.0, p_ro=p_ro_ablation)
        probs_ro5 = evaluate_routed_qcnn_mixed(routed, X_te, n_q, p_2q=ro_anchor_p2q, p_ro=p_ro_ablation)
        topo_results[topo] = {
            "physical_2q_gates": routed["physical_2q_gates"],
            "inserted_swaps": routed["inserted_swaps"],
            "physical_2q_depth": routed["physical_2q_depth"],
            "acc_by_p2q": noise_map,
            "readout_ablation": {
                f"p2q_0.000_pro_{p_ro_ablation:.3f}": round(
                    float(np.mean(np.argmax(probs_ro0, axis=-1) == y_te)), 4
                ),
                f"p2q_{ro_anchor_p2q:.3f}_pro_{p_ro_ablation:.3f}": round(
                    float(np.mean(np.argmax(probs_ro5, axis=-1) == y_te)), 4
                ),
            },
        }

    return {
        "seed": seed,
        "best_val_acc": round(float(best_val), 4),
        "qubit_noiseless_test_acc": round(qubit_acc, 4),
        "max_p0_prob_diff_vs_torch_float32": float(f"{max_p0_diff:.3e}"),
        "topologies": topo_results,
        "seconds": round(time.perf_counter() - t0, 2),
    }


# ---------------------------------------------------------------------------
# Part C: Track A qnative_iqp_kernel (n=6, 2-rep 全连接 ZZ) 路由含噪 default.mixed 评估
# ---------------------------------------------------------------------------

def _prepare_iqp_seed_data(seed: int, cfg: dict[str, Any]) -> dict[str, Any]:
    """生成单 seed 独立分层 MNIST PCA-6 切分并在无噪 Val 上选择 (alpha, C)（零测试集泄露）。"""
    n_q = int(cfg["iqp_qubits"])
    tvt = load_mnist_tvt(
        n_train=int(cfg["iqp_train"]),
        n_val=int(cfg["iqp_val"]),
        n_test=int(cfg["iqp_test"]),
        classes=int(cfg["iqp_classes"]),
        source=str(cfg["data_source"]),
        data_dir=str(cfg["data_dir"]),
        seed=seed,
        pca_dims=(n_q,),
    )
    pca_d = tvt["pca"][n_q]
    X_tr, X_va, X_te = pca_d["X_train"], pca_d["X_val"], pca_d["X_test"]
    y_tr, y_va, y_te = tvt["y_train"], tvt["y_val"], tvt["y_test"]

    # 使用无噪 all_to_all default.qubit 状态向量在 Val 上选 (alpha, C)
    import pennylane as qml

    dev = qml.device("default.qubit", wires=n_q)
    pairs = [(i, j) for i in range(n_q) for j in range(i + 1, n_q)]

    @qml.qnode(dev)
    def _iqp_clean_state(x_batch, alpha_val):
        for _ in range(2):
            for wire in range(n_q):
                qml.Hadamard(wires=wire)
                qml.RZ(alpha_val * x_batch[..., wire], wires=wire)
            for i, j in pairs:
                phi = (alpha_val ** 2) * (np.pi - x_batch[..., i]) * (np.pi - x_batch[..., j])
                qml.IsingZZ(phi, wires=[i, j])
        return qml.state()

    alphas = (0.20, 0.30) if cfg["quick"] else (0.15, 0.25, 0.35, 0.50)
    c_grid = (1.0, 10.0) if cfg["quick"] else (1.0, 5.0, 10.0, 20.0)
    best_val, best_alpha, best_c = -1.0, 0.25, 10.0
    for alpha in alphas:
        st_tr = np.asarray(_iqp_clean_state(X_tr.astype(np.float64), alpha), dtype=np.complex128)
        st_va = np.asarray(_iqp_clean_state(X_va.astype(np.float64), alpha), dtype=np.complex128)
        K_tr = np.real((st_tr @ st_tr.conj().T) * (st_tr @ st_tr.conj().T).conj())
        K_va = np.real((st_va @ st_tr.conj().T) * (st_va @ st_tr.conj().T).conj())
        for c_val in c_grid:
            acc = float(SVC(kernel="precomputed", C=c_val).fit(K_tr, y_tr).score(K_va, y_va))
            if acc > best_val:
                best_val, best_alpha, best_c = acc, float(alpha), float(c_val)

    # 计算精确无噪参考 K_tr_clean 与 K_te_clean
    st_tr_best = np.asarray(_iqp_clean_state(X_tr.astype(np.float64), best_alpha), dtype=np.complex128)
    st_te_best = np.asarray(_iqp_clean_state(X_te.astype(np.float64), best_alpha), dtype=np.complex128)
    K_tr_clean = np.real((st_tr_best @ st_tr_best.conj().T) * (st_tr_best @ st_tr_best.conj().T).conj())
    K_te_clean = np.real((st_te_best @ st_tr_best.conj().T) * (st_te_best @ st_tr_best.conj().T).conj())
    clean_test_acc = float(SVC(kernel="precomputed", C=best_c).fit(K_tr_clean, y_tr).score(K_te_clean, y_te))

    return {
        "seed": seed,
        "X_tr": X_tr,
        "X_te": X_te,
        "y_tr": y_tr,
        "y_te": y_te,
        "selected_alpha": best_alpha,
        "selected_C": best_c,
        "best_val_acc": round(best_val, 4),
        "clean_test_acc": round(clean_test_acc, 4),
        "K_tr_clean": K_tr_clean,
        "K_te_clean": K_te_clean,
        "prep_params": int(pca_d["preprocessing_fitted_params"]),
    }


def _iqp_eval_worker(task: dict[str, Any]) -> dict[str, Any]:
    """单 (seed, topology, p_2q) 任务：在 PennyLane default.mixed 上计算路由 IQP 核与集中度指标。"""
    os.environ["OMP_NUM_THREADS"] = "1"
    t0 = time.perf_counter()
    n_q = int(task["n_qubits"])
    topo = str(task["topology"])
    p_2q = float(task["p_2q"])
    alpha = float(task["alpha"])
    c_val = float(task["c_val"])
    X_tr = task["X_tr"]
    X_te = task["X_te"]
    y_tr = task["y_tr"]
    y_te = task["y_te"]
    K_tr_clean = task["K_tr_clean"]

    graph = build_coupling_graph(topo, n_q)
    sym_gates = extract_iqp_kernel_gates(n_qubits=n_q, reps=2, alpha=alpha, x=None, full_entanglement=True)
    routed = route_circuit_to_topology(sym_gates, graph, n_q)

    X_all = np.concatenate([X_tr, X_te], axis=0)
    rhos_all = evaluate_routed_iqp_density_matrices(
        routed, X_all, n_qubits=n_q, alpha=alpha, p_2q=p_2q, batch_size=50
    )
    n_tr = len(X_tr)
    rhos_tr = rhos_all[:n_tr]
    rhos_te = rhos_all[n_tr:]

    K_tr = hilbert_schmidt_kernel_matrix(rhos_tr, rhos_tr)
    K_te = hilbert_schmidt_kernel_matrix(rhos_te, rhos_tr)
    if p_2q == 0.0:
        max_k_diff = float(np.max(np.abs(K_tr - K_tr_clean)))
        assert max_k_diff < 1e-10, f"IQP default.mixed p=0 与 default.qubit 核不一致: {max_k_diff}"

    acc = float(SVC(kernel="precomputed", C=c_val).fit(K_tr, y_tr).score(K_te, y_te))
    conc = kernel_concentration_metrics(K_tr, K_tr_clean)
    return {
        "seed": int(task["seed"]),
        "topology": topo,
        "p_2q": p_2q,
        "test_acc": round(acc, 4),
        "concentration": conc,
        "physical_2q_gates": routed["physical_2q_gates"],
        "inserted_swaps": routed["inserted_swaps"],
        "physical_2q_depth": routed["physical_2q_depth"],
        "seconds": round(time.perf_counter() - t0, 2),
    }


# ---------------------------------------------------------------------------
# 出版级英文标注图表生成（300 dpi，零中文字体缺字风险）
# ---------------------------------------------------------------------------

def generate_figures(metrics: dict[str, Any], fig_dir: Path) -> list[str]:
    """生成 2 张 300 dpi 英文标注论文级 PNG 图表。"""
    ensure_dir(fig_dir)
    created: list[str] = []

    topo_labels = {
        "all_to_all": "All-to-All (Ideal)",
        "grid_2d": "2D Grid (Sycamore-style)",
        "heavy_hex": "Heavy-Hex (IBM-style)",
        "linear_chain": "1D Linear Chain",
    }
    topo_order = ["all_to_all", "grid_2d", "heavy_hex", "linear_chain"]
    colors = {
        "all_to_all": "#2563eb",
        "grid_2d": "#10b981",
        "heavy_hex": "#f59e0b",
        "linear_chain": "#ef4444",
    }
    markers = {
        "all_to_all": "o",
        "grid_2d": "s",
        "heavy_hex": "^",
        "linear_chain": "D",
    }

    # ---- Figure 1: Routing Overhead (Physical 2Q Gates & ASAP 2Q Depth) ----
    routing = metrics["part_a_routing"]["circuits"]
    circ_keys = ["vqc_amp", "patch_vqc", "qnative_iqp_kernel", "qcnn_spt", "qft_8"]
    circ_labels = [
        f"vqc_amp\n({routing['vqc_amp']['all_to_all']['n_qubits']}q, 20 2Q)",
        f"patch_vqc\n({routing['patch_vqc']['all_to_all']['n_qubits']}q, 16 2Q)",
        f"iqp_kernel\n({routing['qnative_iqp_kernel']['all_to_all']['n_qubits']}q, 30 2Q)",
        f"qcnn_spt\n({routing['qcnn_spt']['all_to_all']['n_qubits']}q, 47 2Q)",
        f"qft_8\n({routing['qft_8']['all_to_all']['n_qubits']}q, 40 2Q)",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), dpi=300)
    x_pos = np.arange(len(circ_keys))
    width = 0.19

    for idx_t, topo in enumerate(topo_order):
        offset = (idx_t - 1.5) * width
        phys_gates = [routing[c][topo]["physical_2q_gates"] for c in circ_keys]
        ratios = [routing[c][topo]["overhead_ratio"] for c in circ_keys]
        phys_depth = [routing[c][topo]["physical_2q_depth"] for c in circ_keys]

        bars1 = axes[0].bar(
            x_pos + offset,
            phys_gates,
            width,
            label=topo_labels[topo],
            color=colors[topo],
            edgecolor="#1f2937",
            linewidth=0.5,
        )
        for bar, ratio in zip(bars1, ratios):
            if ratio > 1.05:
                axes[0].text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 2.0,
                    f"{ratio:.1f}x",
                    ha="center",
                    va="bottom",
                    fontsize=7.0,
                    rotation=90,
                    color="#111827",
                )

        bars2 = axes[1].bar(
            x_pos + offset,
            phys_depth,
            width,
            label=topo_labels[topo],
            color=colors[topo],
            edgecolor="#1f2937",
            linewidth=0.5,
        )
        for bar, depth_val in zip(bars2, phys_depth):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 1.5,
                f"{int(depth_val)}",
                ha="center",
                va="bottom",
                fontsize=7.0,
                rotation=90,
                color="#111827",
            )

    axes[0].set_title("(a) Physical 2-Qubit Gate Count (SABRE SWAP Routing)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Physical 2Q Gates (Logical + 3 × SWAPs)", fontsize=10)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(circ_labels, fontsize=8.8)
    axes[0].set_ylim(0, max(routing[c]["linear_chain"]["physical_2q_gates"] for c in circ_keys) * 1.24)
    axes[0].grid(axis="y", linestyle="--", alpha=0.35)

    axes[1].set_title("(b) Physical ASAP 2-Qubit Circuit Depth", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Physical 2Q Depth (ASAP Schedule)", fontsize=10)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(circ_labels, fontsize=8.8)
    max_d = max(
        max(routing[c][t]["physical_2q_depth"] for t in topo_order) for c in circ_keys
    )
    axes[1].set_ylim(0, max_d * 1.22)
    axes[1].grid(axis="y", linestyle="--", alpha=0.35)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 0.99))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    out1 = fig_dir / "topology_routing_overhead.png"
    fig.savefig(out1, dpi=300)
    plt.close(fig)
    created.append(str(out1))

    # ---- Figure 2: Noise Degradation (QCNN SPT Acc + IQP MNIST Acc + IQP Kernel Concentration) ----
    p2q_list = metrics["config"]["p2q_grid"]
    p2q_pct = [p * 100.0 for p in p2q_list]
    qcnn_sum = metrics["part_b_qcnn_noisy"]["summary"]
    iqp_sum = metrics["part_c_iqp_noisy"]["summary"]

    fig2, ax2 = plt.subplots(1, 3, figsize=(15.0, 4.7), dpi=300)

    # Panel A: QCNN SPT Accuracy vs p_2q
    for topo in topo_order:
        means = [qcnn_sum[topo]["acc_by_p2q"][f"{p:.3f}"]["mean"] * 100.0 for p in p2q_list]
        stds = [qcnn_sum[topo]["acc_by_p2q"][f"{p:.3f}"]["std"] * 100.0 for p in p2q_list]
        ax2[0].errorbar(
            p2q_pct, means, yerr=stds, marker=markers[topo], color=colors[topo],
            linewidth=2.0, markersize=6, capsize=3.5, label=topo_labels[topo]
        )
    ax2[0].set_title("(a) Track B: QCNN SPT (8q, 36q/0c)", fontsize=10.5, fontweight="bold")
    ax2[0].set_xlabel("2-Qubit Depolarizing Error $p_{2q}$ (%)", fontsize=9.5)
    ax2[0].set_ylabel("Test Accuracy (%)", fontsize=9.5)
    ax2[0].set_ylim(45, 103)
    ax2[0].axhline(50.0, color="#6b7280", linestyle=":", linewidth=1.0, label="Random Chance (50%)")
    ax2[0].grid(True, linestyle="--", alpha=0.35)

    # Panel B: IQP MNIST Accuracy vs p_2q
    for topo in topo_order:
        means = [iqp_sum[topo]["acc_by_p2q"][f"{p:.3f}"]["mean"] * 100.0 for p in p2q_list]
        stds = [iqp_sum[topo]["acc_by_p2q"][f"{p:.3f}"]["std"] * 100.0 for p in p2q_list]
        ax2[1].errorbar(
            p2q_pct, means, yerr=stds, marker=markers[topo], color=colors[topo],
            linewidth=2.0, markersize=6, capsize=3.5, label=topo_labels[topo]
        )
    ax2[1].set_title("(b) Track A: IQP Kernel (6q All-to-All ZZ)", fontsize=10.5, fontweight="bold")
    ax2[1].set_xlabel("2-Qubit Depolarizing Error $p_{2q}$ (%)", fontsize=9.5)
    ax2[1].set_ylabel("MNIST Test Accuracy (%)", fontsize=9.5)
    ax2[1].grid(True, linestyle="--", alpha=0.35)

    # Panel C: IQP Kernel Off-Diagonal Std Concentration Ratio vs p_2q (Thanasilp et al. 2024)
    for topo in topo_order:
        ratios = [iqp_sum[topo]["concentration_by_p2q"][f"{p:.3f}"]["offdiag_std_ratio"]["mean"] for p in p2q_list]
        r_stds = [iqp_sum[topo]["concentration_by_p2q"][f"{p:.3f}"]["offdiag_std_ratio"]["std"] for p in p2q_list]
        ax2[2].errorbar(
            p2q_pct, ratios, yerr=r_stds, marker=markers[topo], color=colors[topo],
            linewidth=2.0, markersize=6, capsize=3.5, label=topo_labels[topo]
        )
    ax2[2].set_title("(c) IQP Kernel Concentration (Thanasilp 2024)", fontsize=10.5, fontweight="bold")
    ax2[2].set_xlabel("2-Qubit Depolarizing Error $p_{2q}$ (%)", fontsize=9.5)
    ax2[2].set_ylabel(r"Off-Diag Spread $\mathrm{std}(K_{\mathrm{noisy}}) / \mathrm{std}(K_0)$", fontsize=9.5)
    ax2[2].set_yscale("log")
    ax2[2].set_ylim(0.01, 1.35)
    ax2[2].grid(True, which="both", linestyle="--", alpha=0.35)

    handles2, labels2 = ax2[1].get_legend_handles_labels()
    fig2.legend(handles2, labels2, loc="upper center", ncol=4, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 0.99))
    fig2.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    out2 = fig_dir / "topology_noise_degradation.png"
    fig2.savefig(out2, dpi=300)
    plt.close(fig2)
    created.append(str(out2))

    return created


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="硬件耦合拓扑 SWAP 路由与 default.mixed 去极化噪声基准")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--p2q-grid", default="0.0,0.002,0.005,0.01")
    p.add_argument("--readout-error", type=float, default=0.01)
    p.add_argument("--spt-qubits", type=int, default=8)
    p.add_argument("--spt-train", type=int, default=120)
    p.add_argument("--spt-val", type=int, default=60)
    p.add_argument("--spt-test", type=int, default=120)
    p.add_argument("--epochs-qcnn", type=int, default=25)
    p.add_argument("--iqp-qubits", type=int, default=6)
    p.add_argument("--iqp-train", type=int, default=150)
    p.add_argument("--iqp-val", type=int, default=50)
    p.add_argument("--iqp-test", type=int, default=100)
    p.add_argument("--iqp-classes", type=int, default=10)
    p.add_argument("--data-source", default="idx")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--torch-threads", type=int, default=2)
    p.add_argument("--out", default="results")
    p.add_argument("--archive", default="docs/hardware_topology_metrics.json")
    p.add_argument("--fig-dir", default="docs/figures")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--quick", action="store_true", help="离线极小自检（<15s，仅写入 --out 不覆盖 docs/）")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    t_all = time.perf_counter()

    if args.quick:
        args.seeds = "0"
        args.p2q_grid = "0.0,0.005"
        args.spt_qubits = 4
        args.spt_train, args.spt_val, args.spt_test = 16, 8, 8
        args.epochs_qcnn = 2
        args.iqp_qubits = 4
        args.iqp_train, args.iqp_val, args.iqp_test = 16, 8, 8
        args.iqp_classes = 2
        args.data_source = "synthetic"
        args.workers = 1
        args.no_figures = True

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    p2q_grid = [float(v.strip()) for v in args.p2q_grid.split(",") if v.strip()]
    if 0.0 not in p2q_grid:
        p2q_grid = [0.0] + p2q_grid
    p2q_grid = sorted(set(p2q_grid))

    cfg = {
        "seeds": seeds,
        "p2q_grid": p2q_grid,
        "readout_error": float(args.readout_error),
        "spt_qubits": int(args.spt_qubits),
        "spt_train": int(args.spt_train),
        "spt_val": int(args.spt_val),
        "spt_test": int(args.spt_test),
        "epochs_qcnn": int(args.epochs_qcnn),
        "iqp_qubits": int(args.iqp_qubits),
        "iqp_train": int(args.iqp_train),
        "iqp_val": int(args.iqp_val),
        "iqp_test": int(args.iqp_test),
        "iqp_classes": int(args.iqp_classes),
        "data_source": str(args.data_source),
        "data_dir": str(args.data_dir),
        "workers": int(args.workers),
        "torch_threads": int(args.torch_threads),
        "quick": bool(args.quick),
    }

    # ---- Part A: Routing Overhead ----
    t_a = time.perf_counter()
    part_a = run_routing_overhead_benchmark(quick=args.quick)
    part_a["seconds"] = round(time.perf_counter() - t_a, 2)
    print(f"[Part A] Routing overhead verified across 5 circuits x 4 topologies ({part_a['seconds']}s)")

    # ---- Part B: QCNN SPT Noisy Evaluation (default.mixed) ----
    t_b = time.perf_counter()
    qcnn_tasks = [(seed, cfg) for seed in seeds]
    n_qcnn_workers = min(max(1, int(args.workers)), len(qcnn_tasks))
    if n_qcnn_workers > 1:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_qcnn_workers, mp_context=ctx) as ex:
            qcnn_runs = list(ex.map(_qcnn_seed_worker, qcnn_tasks))
    else:
        qcnn_runs = [_qcnn_seed_worker(t) for t in qcnn_tasks]
    qcnn_runs.sort(key=lambda r: r["seed"])

    ro_anchor = 0.005 if 0.005 in p2q_grid else p2q_grid[-1]
    ro_k0 = f"p2q_0.000_pro_{cfg['readout_error']:.3f}"
    ro_k5 = f"p2q_{ro_anchor:.3f}_pro_{cfg['readout_error']:.3f}"
    qcnn_summary: dict[str, Any] = {}
    for topo in TOPOLOGY_NAMES:
        qcnn_summary[topo] = {
            "physical_2q_gates": qcnn_runs[0]["topologies"][topo]["physical_2q_gates"],
            "inserted_swaps": qcnn_runs[0]["topologies"][topo]["inserted_swaps"],
            "physical_2q_depth": qcnn_runs[0]["topologies"][topo]["physical_2q_depth"],
            "acc_by_p2q": {
                f"{p:.3f}": _stats([r["topologies"][topo]["acc_by_p2q"][f"{p:.3f}"] for r in qcnn_runs])
                for p in p2q_grid
            },
            "readout_ablation": {
                ro_k0: _stats([r["topologies"][topo]["readout_ablation"][ro_k0] for r in qcnn_runs]),
                ro_k5: _stats([r["topologies"][topo]["readout_ablation"][ro_k5] for r in qcnn_runs]),
            },
        }

    qcnn_tests: dict[str, dict[str, Any]] = {}
    p_hi = f"{p2q_grid[-1]:.3f}"
    p_mid = f"{ro_anchor:.3f}"
    if len(seeds) >= 2:
        qcnn_tests["Q1_heavy_hex_minus_all_to_all_p_hi"] = _paired_test(
            qcnn_summary["heavy_hex"]["acc_by_p2q"][p_hi]["values"],
            qcnn_summary["all_to_all"]["acc_by_p2q"][p_hi]["values"],
        )
        qcnn_tests["Q2_linear_minus_all_to_all_p_hi"] = _paired_test(
            qcnn_summary["linear_chain"]["acc_by_p2q"][p_hi]["values"],
            qcnn_summary["all_to_all"]["acc_by_p2q"][p_hi]["values"],
        )
        qcnn_tests["Q3_grid_minus_all_to_all_p_hi"] = _paired_test(
            qcnn_summary["grid_2d"]["acc_by_p2q"][p_hi]["values"],
            qcnn_summary["all_to_all"]["acc_by_p2q"][p_hi]["values"],
        )
        qcnn_tests["Q4_all_to_all_p_hi_minus_p0"] = _paired_test(
            qcnn_summary["all_to_all"]["acc_by_p2q"][p_hi]["values"],
            qcnn_summary["all_to_all"]["acc_by_p2q"]["0.000"]["values"],
        )
        qcnn_tests["Q5_heavy_hex_p_hi_minus_p0"] = _paired_test(
            qcnn_summary["heavy_hex"]["acc_by_p2q"][p_hi]["values"],
            qcnn_summary["heavy_hex"]["acc_by_p2q"]["0.000"]["values"],
        )
        qcnn_tests["Q6_heavy_hex_readout_ablation_at_p_mid"] = _paired_test(
            qcnn_summary["heavy_hex"]["readout_ablation"][ro_k5]["values"],
            qcnn_summary["heavy_hex"]["acc_by_p2q"][p_mid]["values"],
        )
        _apply_holm_bonferroni(qcnn_tests)
    part_b_secs = round(time.perf_counter() - t_b, 2)
    print(f"[Part B] QCNN SPT default.mixed complete across {len(seeds)} seeds ({part_b_secs}s)")

    # ---- Part C: IQP Kernel Noisy Evaluation (default.mixed) ----
    t_c = time.perf_counter()
    seed_data_map = {seed: _prepare_iqp_seed_data(seed, cfg) for seed in seeds}

    iqp_tasks: list[dict[str, Any]] = []
    for seed in seeds:
        sd = seed_data_map[seed]
        for topo in TOPOLOGY_NAMES:
            for p2q in p2q_grid:
                # 在 p_2q=0.0 时：seed 0 在全部 4 种拓扑上均实跑验证 <1e-10 恒等；
                # 其余 seed 对 p_2q=0.0 仅在 all_to_all 跑一次（无噪纯态交叠严格拓扑不变），避免重复计算。
                if p2q == 0.0 and topo != "all_to_all" and seed != seeds[0]:
                    continue
                iqp_tasks.append(
                    {
                        "seed": seed,
                        "topology": topo,
                        "p_2q": p2q,
                        "n_qubits": cfg["iqp_qubits"],
                        "alpha": sd["selected_alpha"],
                        "c_val": sd["selected_C"],
                        "X_tr": sd["X_tr"],
                        "X_te": sd["X_te"],
                        "y_tr": sd["y_tr"],
                        "y_te": sd["y_te"],
                        "K_tr_clean": sd["K_tr_clean"],
                    }
                )

    n_iqp_workers = min(max(1, int(args.workers)), len(iqp_tasks))
    if n_iqp_workers > 1:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_iqp_workers, mp_context=ctx) as ex:
            iqp_raw_results = list(ex.map(_iqp_eval_worker, iqp_tasks))
    else:
        iqp_raw_results = [_iqp_eval_worker(t) for t in iqp_tasks]

    # 组织结果字典 (seed, topo, p_key)
    iqp_lookup: dict[tuple[int, str, str], dict[str, Any]] = {}
    for res in iqp_raw_results:
        iqp_lookup[(res["seed"], res["topology"], f"{res['p_2q']:.3f}")] = res

    # 对 seed > 0 在 p_2q=0.0 的非 all_to_all 拓扑复用经 seed 0 严格验证的精确无噪指标
    for topo in TOPOLOGY_NAMES:
        if topo == "all_to_all":
            continue
        ref_r0 = iqp_lookup[(seeds[0], "all_to_all", "0.000")]
        topo_r0 = iqp_lookup[(seeds[0], topo, "0.000")]
        assert abs(ref_r0["test_acc"] - topo_r0["test_acc"]) < 1e-12
        for seed in seeds[1:]:
            base = iqp_lookup[(seed, "all_to_all", "0.000")]
            iqp_lookup[(seed, topo, "0.000")] = {
                **base,
                "topology": topo,
                "physical_2q_gates": topo_r0["physical_2q_gates"],
                "inserted_swaps": topo_r0["inserted_swaps"],
                "physical_2q_depth": topo_r0["physical_2q_depth"],
            }

    iqp_summary: dict[str, Any] = {}
    conc_keys = (
        "diag_mean_purity",
        "offdiag_mean",
        "offdiag_std",
        "offdiag_std_ratio",
        "frobenius_dist_rel",
    )
    for topo in TOPOLOGY_NAMES:
        sample_rec = iqp_lookup[(seeds[0], topo, f"{p2q_grid[-1]:.3f}")]
        iqp_summary[topo] = {
            "physical_2q_gates": sample_rec["physical_2q_gates"],
            "inserted_swaps": sample_rec["inserted_swaps"],
            "physical_2q_depth": sample_rec["physical_2q_depth"],
            "acc_by_p2q": {
                f"{p:.3f}": _stats([iqp_lookup[(s, topo, f"{p:.3f}")]["test_acc"] for s in seeds])
                for p in p2q_grid
            },
            "concentration_by_p2q": {
                f"{p:.3f}": {
                    ck: _stats_prec(
                        [iqp_lookup[(s, topo, f"{p:.3f}")]["concentration"][ck] for s in seeds], digits=6
                    )
                    for ck in conc_keys
                }
                for p in p2q_grid
            },
        }

    iqp_tests: dict[str, dict[str, Any]] = {}
    if len(seeds) >= 2:
        iqp_tests["I1_heavy_hex_minus_all_to_all_p_hi"] = _paired_test(
            iqp_summary["heavy_hex"]["acc_by_p2q"][p_hi]["values"],
            iqp_summary["all_to_all"]["acc_by_p2q"][p_hi]["values"],
        )
        iqp_tests["I2_linear_minus_all_to_all_p_hi"] = _paired_test(
            iqp_summary["linear_chain"]["acc_by_p2q"][p_hi]["values"],
            iqp_summary["all_to_all"]["acc_by_p2q"][p_hi]["values"],
        )
        iqp_tests["I3_grid_minus_all_to_all_p_hi"] = _paired_test(
            iqp_summary["grid_2d"]["acc_by_p2q"][p_hi]["values"],
            iqp_summary["all_to_all"]["acc_by_p2q"][p_hi]["values"],
        )
        iqp_tests["I4_all_to_all_p_hi_minus_p0"] = _paired_test(
            iqp_summary["all_to_all"]["acc_by_p2q"][p_hi]["values"],
            iqp_summary["all_to_all"]["acc_by_p2q"]["0.000"]["values"],
        )
        iqp_tests["I5_heavy_hex_p_hi_minus_p0"] = _paired_test(
            iqp_summary["heavy_hex"]["acc_by_p2q"][p_hi]["values"],
            iqp_summary["heavy_hex"]["acc_by_p2q"]["0.000"]["values"],
        )
        iqp_tests["I6_linear_p_hi_minus_p0"] = _paired_test(
            iqp_summary["linear_chain"]["acc_by_p2q"][p_hi]["values"],
            iqp_summary["linear_chain"]["acc_by_p2q"]["0.000"]["values"],
        )
        _apply_holm_bonferroni(iqp_tests)

    iqp_per_seed_meta = [
        {
            "seed": s,
            "selected_alpha": seed_data_map[s]["selected_alpha"],
            "selected_C": seed_data_map[s]["selected_C"],
            "best_val_acc": seed_data_map[s]["best_val_acc"],
            "clean_test_acc": seed_data_map[s]["clean_test_acc"],
            "preprocessing_fitted_params": seed_data_map[s]["prep_params"],
        }
        for s in seeds
    ]
    part_c_secs = round(time.perf_counter() - t_c, 2)
    print(f"[Part C] IQP kernel default.mixed complete across {len(seeds)} seeds ({part_c_secs}s)")

    metrics: dict[str, Any] = {
        "purpose": "硬件耦合拓扑 SWAP 路由开销与 PennyLane default.mixed 去极化噪声基准（QCNN SPT + IQP MNIST）",
        "protocol": {
            "topologies": list(TOPOLOGY_NAMES),
            "router": (
                "Deterministic bidirectional SABRE-style SWAP router "
                "(Li, Ding, Xie, ASPLOS 2019; 1 SWAP = 3 physical CNOTs)"
            ),
            "noise_model": (
                "PennyLane default.mixed local depolarizing channel (p_2q) on BOTH qubits "
                "after every physical 2Q gate (including all 3 CNOTs per inserted SWAP) "
                "+ optional BitFlip(p_ro) readout ablation"
            ),
            "iqp_kernel_definition": (
                "Exact Hilbert-Schmidt overlap Tr(rho(x) rho(x')) on default.mixed density matrices "
                "(exact match to |<psi(x)|psi(x')>|^2 at p_2q=0; guaranteed PSD)"
            ),
            "qcnn_mixed_equivalence": (
                "Exact Heisenberg-picture POVM propagation on default.mixed "
                "(verified <1e-10 vs forward per-state default.mixed)"
            ),
            "multiple_testing_correction": "Holm-Bonferroni (two-sided paired Student-t, alpha=0.05)",
        },
        "config": cfg,
        "part_a_routing": part_a,
        "part_b_qcnn_noisy": {
            "summary": qcnn_summary,
            "paired_tests_holm": qcnn_tests,
            "per_seed": qcnn_runs,
            "seconds": part_b_secs,
        },
        "part_c_iqp_noisy": {
            "summary": iqp_summary,
            "paired_tests_holm": iqp_tests,
            "per_seed_selection": iqp_per_seed_meta,
            "seconds": part_c_secs,
        },
        "wall_seconds": round(time.perf_counter() - t_all, 2),
    }

    if not args.no_figures:
        figs = generate_figures(metrics, Path(args.fig_dir))
        metrics["figures"] = figs
        print(f"[Figures] Saved: {figs}")

    out_dir = Path(args.out)
    ensure_dir(out_dir)
    out_name = "quick_hardware_topology_metrics.json" if args.quick else "hardware_topology_metrics.json"
    out_path = out_dir / out_name
    out_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if not args.quick:
        arch_path = Path(args.archive)
        ensure_dir(arch_path.parent)
        arch_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[DONE] Wall {metrics['wall_seconds']}s -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
