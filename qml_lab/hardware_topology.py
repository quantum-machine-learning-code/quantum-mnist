# -*- coding: utf-8 -*-
"""硬件耦合拓扑 SWAP 路由与去极化噪声仿真（PennyLane default.mixed + NetworkX）。

实现 4 种标准量子硬件耦合图、基于双向 SABRE 启发式（Li, Ding, Xie, ASPLOS 2019）的
确定性 SWAP 路由编译器，以及 5 条真实基准线路（vqc_amp / patch_vqc / qnative_iqp_kernel /
qcnn_spt / qft_8）的物理 2Q 门开销与密度矩阵含噪执行。

4 种耦合拓扑：
  1. ``all_to_all``  : 完全图 K_n（理想 / 单区离子阱，零 SWAP 开销）
  2. ``linear_chain``: 一维近邻链 P_n（最大度 2）
  3. ``grid_2d``     : 2D 方形网格（Google Sycamore/Willow 风格，最大度 <= 4）
  4. ``heavy_hex``   : IBM Eagle/Heron 重六边形晶格连通子图（显式 data + bridge 比特，
                       严格满足连通且最大度 <= 3，Chamberland et al. PRX 2020）
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Any

import networkx as nx
import numpy as np
import pennylane as qml

__all__ = [
    "TOPOLOGY_NAMES",
    "GateOp",
    "build_coupling_graph",
    "topology_stats",
    "compute_2q_depth",
    "extract_vqc_amp_gates",
    "extract_patch_vqc_gates",
    "extract_iqp_kernel_gates",
    "extract_qcnn_spt_gates",
    "extract_qft_gates",
    "route_circuit_to_topology",
    "permute_statevector",
    "execute_gates_statevector",
    "simulate_routed_qcnn_forward_mixed",
    "evaluate_routed_qcnn_mixed",
    "evaluate_routed_iqp_density_matrices",
    "hilbert_schmidt_kernel_matrix",
    "kernel_concentration_metrics",
]

TOPOLOGY_NAMES: tuple[str, ...] = ("all_to_all", "linear_chain", "grid_2d", "heavy_hex")


@dataclass(frozen=True)
class GateOp:
    """不可变逻辑/物理量子门描述。"""

    name: str
    wires: tuple[int, ...]
    params: tuple[float, ...] = ()
    logical_wires: tuple[int, ...] = ()
    is_swap_cnot: bool = False


def _build_heavy_hex_lattice(n_rows: int = 4, n_cols: int = 4) -> nx.Graph:
    """构造标准 IBM Heavy-Hex 晶格（砖墙表示，Chamberland et al., PRX 2020）。

    每个水平行交替排列 data（度 <= 3）与 horizontal bridge（度 <= 2）；
    每两行相邻的偶/奇列 data 顶点之间再通过 vertical bridge（度 2）相连，
    确保任意边均连接 data 与 bridge（严格二分、任意节点度 <= 3、围长 12）。
    """
    lat = nx.Graph()
    for row in range(n_rows):
        for col in range(n_cols):
            d_node = ("d", row, col)
            lat.add_node(d_node, role="data")
            if col + 1 < n_cols:
                b_node = ("bh", row, col)
                lat.add_node(b_node, role="bridge")
                lat.add_edge(d_node, b_node)
                lat.add_edge(b_node, ("d", row, col + 1))
        if row + 1 < n_rows:
            for col in range(n_cols):
                # 砖墙六边形奇偶交替垂直桥：每条六边形边恰有 1 个 bridge 比特
                if (row + col) % 2 == 0:
                    bv_node = ("bv", row, col)
                    lat.add_node(bv_node, role="bridge")
                    lat.add_edge(("d", row, col), bv_node)
                    lat.add_edge(bv_node, ("d", row + 1, col))
    return lat


def build_coupling_graph(topology: str, n_qubits: int) -> nx.Graph:
    """构造指定拓扑的 n 比特连通耦合图（节点编号 0..n_qubits-1）。"""
    if n_qubits < 2:
        raise ValueError(f"n_qubits 至少为 2，实际 {n_qubits}")
    topo = topology.strip().lower()
    if topo == "all_to_all":
        graph = nx.complete_graph(n_qubits)
        for node in graph.nodes:
            graph.nodes[node]["role"] = "all"
    elif topo == "linear_chain":
        graph = nx.path_graph(n_qubits)
        for node in graph.nodes:
            graph.nodes[node]["role"] = "chain"
    elif topo == "grid_2d":
        if n_qubits == 6:
            rows, cols = 2, 3
        elif n_qubits == 8:
            rows, cols = 2, 4
        elif n_qubits == 10:
            rows, cols = 2, 5
        elif n_qubits == 4:
            rows, cols = 2, 2
        else:
            rows = int(math.floor(math.sqrt(n_qubits)))
            while rows > 1 and n_qubits % rows != 0:
                rows -= 1
            if rows == 1 and n_qubits >= 4:
                rows = int(math.ceil(math.sqrt(n_qubits)))
            cols = int(math.ceil(n_qubits / rows))
        full_grid = nx.grid_2d_graph(rows, cols)
        ordered = [(r, c) for r in range(rows) for c in range(cols)][:n_qubits]
        sub = full_grid.subgraph(ordered).copy()
        mapping = {coord: idx for idx, coord in enumerate(ordered)}
        graph = nx.relabel_nodes(sub, mapping, copy=True)
        for node in graph.nodes:
            graph.nodes[node]["role"] = "grid"
        assert max(dict(graph.degree()).values()) <= 4
    elif topo == "heavy_hex":
        side = max(4, int(math.ceil(math.sqrt(n_qubits))) + 2)
        lat = _build_heavy_hex_lattice(n_rows=side, n_cols=side)
        # 从内部中心 data 节点做确定性 BFS 扩展，取前 n_qubits 个节点构成真实连通子图
        center = ("d", side // 2, side // 2)
        visited = [center]
        seen = {center}
        queue: deque[Any] = deque([center])
        while len(visited) < n_qubits and queue:
            cur = queue.popleft()
            neighbors = sorted(lat.neighbors(cur), key=str)
            for nb in neighbors:
                if nb not in seen:
                    seen.add(nb)
                    visited.append(nb)
                    queue.append(nb)
                    if len(visited) >= n_qubits:
                        break
        sub = lat.subgraph(visited).copy()
        mapping = {node: idx for idx, node in enumerate(visited)}
        graph = nx.relabel_nodes(sub, mapping, copy=True)
        # 严格校验重六边形耦合图不变量：连通、最大度 <= 3、bridge 度 <= 2、二分无同角色邻边
        assert nx.is_connected(graph), "heavy_hex 子图必须连通"
        max_deg = max(dict(graph.degree()).values())
        assert max_deg <= 3, f"heavy_hex 最大度必须 <= 3，实际 {max_deg}"
        for node, deg in graph.degree():
            role = graph.nodes[node]["role"]
            if role == "bridge":
                assert deg <= 2, f"heavy_hex bridge 比特 {node} 度必须 <= 2，实际 {deg}"
            else:
                assert deg <= 3, f"heavy_hex data 比特 {node} 度必须 <= 3，实际 {deg}"
        for u, v in graph.edges():
            assert graph.nodes[u]["role"] != graph.nodes[v]["role"], (
                f"heavy_hex 边 ({u},{v}) 违反 data-bridge 二分结构"
            )
    else:
        raise ValueError(f"未知拓扑 {topology!r}，可选 {TOPOLOGY_NAMES}")

    assert nx.is_connected(graph), f"拓扑 {topo} (n={n_qubits}) 非连通图"
    return graph


def topology_stats(graph: nx.Graph) -> dict[str, Any]:
    """提取硬件耦合图结构统计量。"""
    degs = [deg for _, deg in graph.degree()]
    return {
        "n_nodes": int(graph.number_of_nodes()),
        "n_edges": int(graph.number_of_edges()),
        "max_degree": int(max(degs)),
        "avg_degree": round(float(np.mean(degs)), 3),
        "diameter": int(nx.diameter(graph)),
        "avg_shortest_path": round(float(nx.average_shortest_path_length(graph)), 3),
        "connected": bool(nx.is_connected(graph)),
    }


def compute_2q_depth(gates: list[GateOp], n_qubits: int) -> int:
    """精确 ASAP（As-Soon-As-Possible）2-qubit 门深度调度。"""
    wire_time = [0] * n_qubits
    for gate in gates:
        if len(gate.wires) == 2:
            u, v = gate.wires
            scheduled = max(wire_time[u], wire_time[v]) + 1
            wire_time[u] = scheduled
            wire_time[v] = scheduled
    return max(wire_time) if wire_time else 0


# ---------------------------------------------------------------------------
# 5 条真实基准线路的门序列提取器
# ---------------------------------------------------------------------------

def _extract_sel_gates(n_qubits: int, n_layers: int, weights: np.ndarray | None = None) -> list[GateOp]:
    """与 PennyLane StronglyEntanglingLayers 逐位一致的门序列提取。"""
    if weights is None:
        rng = np.random.default_rng(0)
        w = rng.uniform(-math.pi, math.pi, size=(n_layers, n_qubits, 3))
    else:
        w = np.asarray(weights, dtype=np.float64)
    gates: list[GateOp] = []
    for layer_idx in range(n_layers):
        step = (layer_idx % (n_qubits - 1)) + 1
        for wire in range(n_qubits):
            gates.append(
                GateOp(
                    "Rot",
                    (wire,),
                    (float(w[layer_idx, wire, 0]), float(w[layer_idx, wire, 1]), float(w[layer_idx, wire, 2])),
                    logical_wires=(wire,),
                )
            )
        for wire in range(n_qubits):
            target = (wire + step) % n_qubits
            gates.append(GateOp("CNOT", (wire, target), (), logical_wires=(wire, target)))
    return gates


def extract_vqc_amp_gates(
    n_qubits: int = 10, n_layers: int = 2, weights: np.ndarray | None = None
) -> list[GateOp]:
    """提取 Track A ``vqc_amp`` (10q, 2L StronglyEntanglingLayers) 真实门序列。"""
    return _extract_sel_gates(n_qubits=n_qubits, n_layers=n_layers, weights=weights)


def extract_patch_vqc_gates(
    n_qubits: int = 8, n_layers: int = 2, weights: np.ndarray | None = None
) -> list[GateOp]:
    """提取 Track A ``patch_vqc`` (8q, 2L StronglyEntanglingLayers) 真实门序列。"""
    return _extract_sel_gates(n_qubits=n_qubits, n_layers=n_layers, weights=weights)


def extract_iqp_kernel_gates(
    n_qubits: int = 6,
    reps: int = 2,
    alpha: float = 0.25,
    x: np.ndarray | None = None,
    full_entanglement: bool = True,
) -> list[GateOp]:
    """提取 ``qnative_iqp_kernel``（默认 n=6, 2-rep 全连接 ZZFeatureMap）门序列。

    当 ``x=None`` 时，参数留空并以 ``logical_wires`` 标记数据依赖，便于一次路由后对任意批量样本
    在 PennyLane ``default.mixed`` 上直接向量化执行。
    """
    x_arr = None if x is None else np.asarray(x, dtype=np.float64).reshape(-1)
    if full_entanglement:
        pairs = [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]
    else:
        seen = set()
        pairs = []
        raw = [(i, (i + 1) % n_qubits) for i in range(n_qubits)]
        if n_qubits >= 4:
            raw.extend((i, (i + 2) % n_qubits) for i in range(n_qubits))
        for a, b in raw:
            edge = (min(a, b), max(a, b))
            if edge[0] != edge[1] and edge not in seen:
                seen.add(edge)
                pairs.append(edge)

    gates: list[GateOp] = []
    for _ in range(reps):
        for wire in range(n_qubits):
            gates.append(GateOp("Hadamard", (wire,), (), logical_wires=(wire,)))
            if x_arr is None:
                gates.append(GateOp("IQP_RZ", (wire,), (float(alpha),), logical_wires=(wire,)))
            else:
                gates.append(GateOp("RZ", (wire,), (float(alpha * x_arr[wire]),), logical_wires=(wire,)))
        for i, j in pairs:
            if x_arr is None:
                gates.append(GateOp("IQP_ZZ", (i, j), (float(alpha),), logical_wires=(i, j)))
            else:
                phi = float((alpha ** 2) * (math.pi - x_arr[i]) * (math.pi - x_arr[j]))
                gates.append(GateOp("IsingZZ", (i, j), (phi,), logical_wires=(i, j)))
    return gates


def extract_qcnn_spt_gates(
    n_qubits: int = 8,
    conv_w: np.ndarray | None = None,
    pool_w: np.ndarray | None = None,
    final_rot: np.ndarray | None = None,
) -> list[GateOp]:
    """提取 Track B 真正 Cong-et-al. ``qcnn_spt`` (8->4->2->1) 完整门序列。"""
    if n_qubits < 4 or (n_qubits & (n_qubits - 1)) != 0:
        raise ValueError(f"QCNN 要求 n_qubits 为 >=4 的 2 的幂，实际 {n_qubits}")
    n_scales = int(round(math.log2(n_qubits)))
    rng = np.random.default_rng(0)
    cw = rng.normal(0.0, 0.4, size=(n_scales, 8)) if conv_w is None else np.asarray(conv_w, dtype=np.float64)
    pw = rng.normal(0.0, 0.4, size=(n_scales, 3)) if pool_w is None else np.asarray(pool_w, dtype=np.float64)
    fr = rng.normal(0.0, 0.2, size=(3,)) if final_rot is None else np.asarray(final_rot, dtype=np.float64)

    gates: list[GateOp] = []

    def _emit_conv(par: np.ndarray, w0: int, w1: int) -> None:
        gates.append(GateOp("RY", (w0,), (float(par[0]),), logical_wires=(w0,)))
        gates.append(GateOp("RY", (w1,), (float(par[1]),), logical_wires=(w1,)))
        gates.append(GateOp("RZ", (w0,), (float(par[2]),), logical_wires=(w0,)))
        gates.append(GateOp("RZ", (w1,), (float(par[3]),), logical_wires=(w1,)))
        gates.append(GateOp("IsingZZ", (w0, w1), (float(par[4]),), logical_wires=(w0, w1)))
        gates.append(GateOp("IsingXX", (w0, w1), (float(par[5]),), logical_wires=(w0, w1)))
        gates.append(GateOp("RY", (w0,), (float(par[6]),), logical_wires=(w0,)))
        gates.append(GateOp("RY", (w1,), (float(par[7]),), logical_wires=(w1,)))

    def _emit_pool(par: np.ndarray, kept: int, dropped: int) -> None:
        gates.append(GateOp("CRZ", (dropped, kept), (float(par[0]),), logical_wires=(dropped, kept)))
        gates.append(GateOp("CRY", (dropped, kept), (float(par[1]),), logical_wires=(dropped, kept)))
        gates.append(GateOp("CRX", (dropped, kept), (float(par[2]),), logical_wires=(dropped, kept)))

    active = list(range(n_qubits))
    for scale_idx in range(n_scales):
        m = len(active)
        for idx in range(0, m, 2):
            _emit_conv(cw[scale_idx], active[idx], active[(idx + 1) % m])
        if m > 2:
            for idx in range(1, m, 2):
                _emit_conv(cw[scale_idx], active[idx], active[(idx + 1) % m])
        next_active = []
        for idx in range(0, m, 2):
            kept, dropped = active[idx], active[idx + 1]
            _emit_pool(pw[scale_idx], kept=kept, dropped=dropped)
            next_active.append(kept)
        active = next_active

    final_wire = active[0]
    gates.append(
        GateOp(
            "Rot",
            (final_wire,),
            (float(fr[0]), float(fr[1]), float(fr[2])),
            logical_wires=(final_wire,),
        )
    )
    return gates


def extract_qft_gates(n_qubits: int = 8) -> list[GateOp]:
    """提取精确 n 比特量子傅里叶变换（QFT，含位反转 3-CNOT SWAP，衔接 Shor 算法）。"""
    gates: list[GateOp] = []
    for i in range(n_qubits):
        gates.append(GateOp("Hadamard", (i,), (), logical_wires=(i,)))
        for j in range(i + 1, n_qubits):
            angle = math.pi / float(1 << (j - i))
            gates.append(GateOp("ControlledPhaseShift", (j, i), (angle,), logical_wires=(j, i)))
    for i in range(n_qubits // 2):
        a, b = i, n_qubits - 1 - i
        gates.append(GateOp("CNOT", (a, b), (), logical_wires=(a, b)))
        gates.append(GateOp("CNOT", (b, a), (), logical_wires=(b, a)))
        gates.append(GateOp("CNOT", (a, b), (), logical_wires=(a, b)))
    return gates


# ---------------------------------------------------------------------------
# 确定性双向 SABRE 风格 SWAP 路由器（Li, Ding, Xie, ASPLOS 2019）
# ---------------------------------------------------------------------------

def _interaction_graph(gates: list[GateOp], n_qubits: int) -> nx.Graph:
    """统计逻辑线路的两比特交互加权图，用于公平初始布局。"""
    ig = nx.Graph()
    ig.add_nodes_from(range(n_qubits))
    for gate in gates:
        if len(gate.wires) == 2:
            u, v = gate.wires
            if ig.has_edge(u, v):
                ig[u][v]["weight"] += 1.0
            else:
                ig.add_edge(u, v, weight=1.0)
    return ig


def _bfs_order(graph: nx.Graph, n_qubits: int, use_weight: bool = False) -> list[int]:
    """按度/加权度降序确定性 BFS 遍历节点顺序。"""
    def _key_fn(node: int) -> tuple[float, int]:
        if use_weight:
            deg_w = sum(d.get("weight", 1.0) for _, _, d in graph.edges(node, data=True))
            return (-float(deg_w), node)
        return (-float(graph.degree(node)), node)

    order: list[int] = []
    seen: set[int] = set()
    for start in sorted(range(n_qubits), key=_key_fn):
        if start in seen:
            continue
        queue = deque([start])
        seen.add(start)
        while queue:
            cur = queue.popleft()
            order.append(cur)
            for nb in sorted(graph.neighbors(cur), key=_key_fn):
                if nb not in seen:
                    seen.add(nb)
                    queue.append(nb)
    return order


def _spectral_order(graph: nx.Graph, n_qubits: int, weight: str | None = None) -> list[int]:
    """利用拉普拉斯 Fiedler 向量给出确定性一维谱排序。"""
    lap = nx.laplacian_matrix(graph, nodelist=list(range(n_qubits)), weight=weight).toarray().astype(np.float64)
    _, evecs = np.linalg.eigh(lap)
    fiedler = evecs[:, 1] if n_qubits > 1 else np.zeros(n_qubits)
    # 固定符号以保证确定性
    max_idx = int(np.argmax(np.abs(fiedler)))
    if fiedler[max_idx] < 0.0:
        fiedler = -fiedler
    return sorted(range(n_qubits), key=lambda node: (round(float(fiedler[node]), 10), node))


def _candidate_initial_layouts(gates: list[GateOp], graph: nx.Graph, n_qubits: int) -> list[list[int]]:
    """构造一组公平、确定性的初始逻辑->物理布局候选（Identity + BFS + Spectral）。"""
    ig = _interaction_graph(gates, n_qubits)
    layouts: list[list[int]] = [list(range(n_qubits))]

    def _pair_orders(log_ord: list[int], phys_ord: list[int]) -> list[int]:
        layout = [0] * n_qubits
        for log_q, phys_q in zip(log_ord, phys_ord):
            layout[log_q] = phys_q
        return layout

    log_bfs = _bfs_order(ig, n_qubits, use_weight=True)
    phys_bfs = _bfs_order(graph, n_qubits, use_weight=False)
    layouts.append(_pair_orders(log_bfs, phys_bfs))

    log_spec = _spectral_order(ig, n_qubits, weight="weight")
    phys_spec = _spectral_order(graph, n_qubits, weight=None)
    layouts.append(_pair_orders(log_spec, phys_spec))
    layouts.append(_pair_orders(log_spec, list(reversed(phys_spec))))

    # 针对环状 / 1D 链路的居中映射（按物理图中心度排序匹配交互度）
    ecc = nx.eccentricity(graph)
    phys_center = sorted(range(n_qubits), key=lambda node: (ecc[node], -graph.degree(node), node))
    log_deg = sorted(
        range(n_qubits),
        key=lambda node: (-sum(d.get("weight", 1.0) for _, _, d in ig.edges(node, data=True)), node),
    )
    layouts.append(_pair_orders(log_deg, phys_center))

    # 去重并保序
    uniq: list[list[int]] = []
    seen_set: set[tuple[int, ...]] = set()
    for lay in layouts:
        key = tuple(lay)
        if key not in seen_set:
            seen_set.add(key)
            uniq.append(lay)
    return uniq


def _build_gate_dag(gates: list[GateOp], n_qubits: int) -> tuple[list[list[int]], list[int]]:
    """按每个逻辑比特的先后依赖构造线路 DAG 邻接表与入度。"""
    n_gates = len(gates)
    succ: list[list[int]] = [[] for _ in range(n_gates)]
    indeg = [0] * n_gates
    last_on_wire = [-1] * n_qubits
    for idx, gate in enumerate(gates):
        preds_added: set[int] = set()
        for wire in gate.wires:
            prev = last_on_wire[wire]
            if prev != -1 and prev not in preds_added:
                succ[prev].append(idx)
                indeg[idx] += 1
                preds_added.add(prev)
            last_on_wire[wire] = idx
    return succ, indeg


def _sabre_route_pass(
    gates: list[GateOp],
    graph: nx.Graph,
    dist: dict[int, dict[int, int]],
    init_layout: list[int],
    n_qubits: int,
    ext_size: int = 20,
    ext_weight: float = 0.5,
    decay_step: float = 0.001,
) -> tuple[list[GateOp], list[int], int]:
    """执行一次确定性 SABRE 启发式前向 SWAP 路由（Li et al., ASPLOS 2019）。"""
    layout = list(init_layout)  # logical -> physical
    inv_layout = [0] * n_qubits
    for log_q, phys_q in enumerate(layout):
        inv_layout[phys_q] = log_q

    succ, indeg = _build_gate_dag(gates, n_qubits)
    ready_1q: deque[int] = deque()
    front_2q: list[int] = []
    for idx, deg in enumerate(indeg):
        if deg == 0:
            if len(gates[idx].wires) == 1:
                ready_1q.append(idx)
            else:
                front_2q.append(idx)

    routed: list[GateOp] = []
    inserted_swaps = 0
    decay = [1.0] * n_qubits
    edges_sorted = sorted((min(u, v), max(u, v)) for u, v in graph.edges())
    edge_set = set(edges_sorted)
    last_swap: tuple[int, int] | None = None
    stagnant_swaps = 0

    def _release_successors(gate_idx: int) -> None:
        for nxt in succ[gate_idx]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                if len(gates[nxt].wires) == 1:
                    ready_1q.append(nxt)
                else:
                    front_2q.append(nxt)

    def _apply_swap(p_u: int, p_v: int) -> None:
        nonlocal inserted_swaps, last_swap
        log_u, log_v = inv_layout[p_u], inv_layout[p_v]
        layout[log_u], layout[log_v] = p_v, p_u
        inv_layout[p_u], inv_layout[p_v] = log_v, log_u
        # 将 1 个物理 SWAP 精确分解为耦合边 (p_u, p_v) 上的 3 个物理 CNOT
        routed.append(GateOp("CNOT", (p_u, p_v), (), is_swap_cnot=True))
        routed.append(GateOp("CNOT", (p_v, p_u), (), is_swap_cnot=True))
        routed.append(GateOp("CNOT", (p_u, p_v), (), is_swap_cnot=True))
        inserted_swaps += 1
        decay[p_u] += decay_step
        decay[p_v] += decay_step
        last_swap = (min(p_u, p_v), max(p_u, p_v))

    def _collect_extended_set() -> list[int]:
        if not front_2q:
            return []
        ext: list[int] = []
        visited = set(front_2q)
        bfs_q: deque[int] = deque(front_2q)
        while bfs_q and len(ext) < ext_size:
            cur = bfs_q.popleft()
            for nxt in succ[cur]:
                if nxt not in visited:
                    visited.add(nxt)
                    bfs_q.append(nxt)
                    if len(gates[nxt].wires) == 2:
                        ext.append(nxt)
                        if len(ext) >= ext_size:
                            break
        return ext

    while ready_1q or front_2q:
        progressed = False
        # 1. 立即执行所有就绪的单比特门
        while ready_1q:
            g_idx = ready_1q.popleft()
            gate = gates[g_idx]
            phys_w = layout[gate.wires[0]]
            routed.append(
                GateOp(gate.name, (phys_w,), gate.params, logical_wires=gate.logical_wires)
            )
            _release_successors(g_idx)
            progressed = True

        # 2. 执行前沿中物理相邻的所有两比特门
        remaining_front: list[int] = []
        executed_2q: list[int] = []
        for g_idx in sorted(front_2q):
            gate = gates[g_idx]
            pu, pv = layout[gate.wires[0]], layout[gate.wires[1]]
            if (min(pu, pv), max(pu, pv)) in edge_set:
                routed.append(
                    GateOp(gate.name, (pu, pv), gate.params, logical_wires=gate.logical_wires)
                )
                executed_2q.append(g_idx)
                progressed = True
            else:
                remaining_front.append(g_idx)
        front_2q = remaining_front
        for g_idx in executed_2q:
            _release_successors(g_idx)

        if progressed:
            stagnant_swaps = 0
            last_swap = None
            continue

        # 3. 前沿阻塞：选择最小化 SABRE 启发代价的候选物理边插入 SWAP
        if not front_2q:
            continue

        # 死锁保护：若连续无进展 SWAP 过多，直接沿最短路推进最近前沿门
        if stagnant_swaps >= 2 * n_qubits:
            best_g = min(
                front_2q,
                key=lambda gi: (dist[layout[gates[gi].wires[0]]][layout[gates[gi].wires[1]]], gi),
            )
            pu = layout[gates[best_g].wires[0]]
            pv = layout[gates[best_g].wires[1]]
            path = nx.shortest_path(graph, pu, pv)
            _apply_swap(path[0], path[1])
            stagnant_swaps += 1
            continue

        active_phys = set()
        for g_idx in front_2q:
            active_phys.add(layout[gates[g_idx].wires[0]])
            active_phys.add(layout[gates[g_idx].wires[1]])

        cand_edges = [e for e in edges_sorted if e[0] in active_phys or e[1] in active_phys]
        if last_swap is not None and len(cand_edges) > 1:
            filtered = [e for e in cand_edges if e != last_swap]
            if filtered:
                cand_edges = filtered

        ext_set = _collect_extended_set()
        n_front = float(len(front_2q))
        n_ext = float(len(ext_set))

        best_edge = cand_edges[0]
        best_key = (float("inf"), float("inf"), best_edge[0], best_edge[1])

        for p_u, p_v in cand_edges:
            log_u, log_v = inv_layout[p_u], inv_layout[p_v]
            # 模拟交换 (p_u, p_v)
            layout[log_u], layout[log_v] = p_v, p_u

            front_sum = 0.0
            min_f = float("inf")
            for g_idx in front_2q:
                d_val = float(dist[layout[gates[g_idx].wires[0]]][layout[gates[g_idx].wires[1]]])
                front_sum += d_val
                if d_val < min_f:
                    min_f = d_val
            score = front_sum / n_front
            if ext_set:
                ext_sum = sum(
                    float(dist[layout[gates[gi].wires[0]]][layout[gates[gi].wires[1]]]) for gi in ext_set
                )
                score += ext_weight * (ext_sum / n_ext)
            score *= max(decay[p_u], decay[p_v])

            # 还原
            layout[log_u], layout[log_v] = p_u, p_v

            key = (round(score, 12), min_f, p_u, p_v)
            if key < best_key:
                best_key = key
                best_edge = (p_u, p_v)

        _apply_swap(best_edge[0], best_edge[1])
        stagnant_swaps += 1

    return routed, layout, inserted_swaps


def route_circuit_to_topology(
    gates: list[GateOp],
    graph: nx.Graph,
    n_qubits: int,
) -> dict[str, Any]:
    """将逻辑门序列通过双向 SABRE SWAP 路由映射到给定硬件耦合图。

    流程：
      1. 构造公平初始布局候选（Identity / 加权 BFS / 谱 Fiedler / 图中心对齐）；
      2. 对每个候选执行 2 轮双向 SABRE 反向遍历优化（forward -> reverse -> forward），
         自动搜索最优逻辑->物理初始映射；
      3. 严格断言输出的每一条物理两比特门均位于耦合图边集内，并计算精确 ASAP 2Q 深度。
    """
    if graph.number_of_nodes() != n_qubits:
        raise ValueError(f"耦合图节点数 {graph.number_of_nodes()} != n_qubits={n_qubits}")

    logical_2q = sum(1 for g in gates if len(g.wires) == 2)
    logical_depth = compute_2q_depth(gates, n_qubits)
    dist = {u: dict(d) for u, d in nx.all_pairs_shortest_path_length(graph)}

    # 零 SWAP 快速路径：若恒等布局下全部 2Q 门均已在耦合边上（例如 all_to_all），直接映射
    if all(graph.has_edge(g.wires[0], g.wires[1]) for g in gates if len(g.wires) == 2):
        best_routed = [
            GateOp(g.name, g.wires, g.params, logical_wires=g.logical_wires, is_swap_cnot=g.is_swap_cnot)
            for g in gates
        ]
        best_init = list(range(n_qubits))
        best_final = list(range(n_qubits))
        best_swaps = 0
        best_depth = logical_depth
    else:
        rev_gates = list(reversed(gates))
        best_result: tuple[int, int, tuple[int, ...], list[GateOp], list[int], list[int]] | None = None
        for seed_layout in _candidate_initial_layouts(gates, graph, n_qubits):
            cur_layout = list(seed_layout)
            for _ in range(2):
                routed_fwd, final_fwd, swaps_fwd = _sabre_route_pass(
                    gates, graph, dist, cur_layout, n_qubits
                )
                depth_fwd = compute_2q_depth(routed_fwd, n_qubits)
                cand_key = (swaps_fwd, depth_fwd, tuple(cur_layout))
                if best_result is None or cand_key < best_result[:3]:
                    best_result = (
                        swaps_fwd,
                        depth_fwd,
                        tuple(cur_layout),
                        routed_fwd,
                        list(cur_layout),
                        final_fwd,
                    )
                if swaps_fwd == 0:
                    break
                # SABRE 双向反向遍历：用前向终布局路由逆序线路，得到改进后的初始布局
                _, rev_final, _ = _sabre_route_pass(rev_gates, graph, dist, final_fwd, n_qubits)
                cur_layout = rev_final

        assert best_result is not None
        best_swaps, best_depth, _, best_routed, best_init, best_final = best_result

    # 严格物理边与门数守恒校验
    physical_2q = 0
    for gate in best_routed:
        if len(gate.wires) == 2:
            physical_2q += 1
            assert graph.has_edge(gate.wires[0], gate.wires[1]), (
                f"路由后的物理 2Q 门 {gate.name}{gate.wires} 不在耦合图边集内"
            )
    assert physical_2q == logical_2q + 3 * best_swaps, (
        f"2Q 门计数不一致: physical={physical_2q} != logical={logical_2q} + 3*{best_swaps}"
    )

    overhead = float(physical_2q) / max(logical_2q, 1)
    depth_overhead = float(best_depth) / max(logical_depth, 1)
    return {
        "n_qubits": int(n_qubits),
        "logical_2q_gates": int(logical_2q),
        "inserted_swaps": int(best_swaps),
        "physical_2q_gates": int(physical_2q),
        "overhead_ratio": round(overhead, 4),
        "logical_2q_depth": int(logical_depth),
        "physical_2q_depth": int(best_depth),
        "depth_overhead_ratio": round(depth_overhead, 4),
        "initial_layout": best_init,
        "final_layout": best_final,
        "routed_gates": best_routed,
    }


# ---------------------------------------------------------------------------
# 精确态向量检验 & PennyLane default.mixed 含噪密度矩阵仿真
# ---------------------------------------------------------------------------

def permute_statevector(
    psi: np.ndarray, layout: list[int], inverse: bool = False
) -> np.ndarray:
    """在逻辑量子比特序与物理量子比特序之间做精确张量轴置换。

    ``layout[log_q] = phys_q``：
      * ``inverse=False``（逻辑 -> 物理）：物理轴 ``p`` 对应逻辑比特 ``inv_layout[p]``；
      * ``inverse=True`` （物理 -> 逻辑）：逻辑轴 ``q`` 对应物理比特 ``layout[q]``。
    支持单态 ``(2**n,)`` 或批量 ``(B, 2**n)``。
    """
    arr = np.asarray(psi, dtype=np.complex128)
    n_qubits = len(layout)
    if inverse:
        perm = list(layout)
    else:
        perm = [0] * n_qubits
        for log_q, phys_q in enumerate(layout):
            perm[phys_q] = log_q

    if arr.ndim == 1:
        reshaped = arr.reshape((2,) * n_qubits)
        return np.transpose(reshaped, axes=perm).reshape(1 << n_qubits)
    batch = arr.shape[0]
    reshaped = arr.reshape((batch,) + (2,) * n_qubits)
    axes = [0] + [p + 1 for p in perm]
    return np.transpose(reshaped, axes=axes).reshape(batch, 1 << n_qubits)


def _apply_gate_op(gate: GateOp, adjoint: bool = False) -> None:
    """将单条 GateOp 应用到当前 PennyLane 线路上下文。"""
    sign = -1.0 if adjoint else 1.0
    name = gate.name
    wires = gate.wires
    params = gate.params
    if name == "Hadamard":
        qml.Hadamard(wires=wires[0])
    elif name == "RY":
        qml.RY(sign * params[0], wires=wires[0])
    elif name == "RZ":
        qml.RZ(sign * params[0], wires=wires[0])
    elif name == "Rot":
        if adjoint:
            qml.Rot(-params[2], -params[1], -params[0], wires=wires[0])
        else:
            qml.Rot(params[0], params[1], params[2], wires=wires[0])
    elif name == "CNOT":
        qml.CNOT(wires=list(wires))
    elif name == "IsingZZ":
        qml.IsingZZ(sign * params[0], wires=list(wires))
    elif name == "IsingXX":
        qml.IsingXX(sign * params[0], wires=list(wires))
    elif name == "CRZ":
        qml.CRZ(sign * params[0], wires=list(wires))
    elif name == "CRY":
        qml.CRY(sign * params[0], wires=list(wires))
    elif name == "CRX":
        qml.CRX(sign * params[0], wires=list(wires))
    elif name == "ControlledPhaseShift":
        qml.ControlledPhaseShift(sign * params[0], wires=list(wires))
    else:
        raise ValueError(f"不支持的量子门 {name!r}")


def execute_gates_statevector(
    gates: list[GateOp],
    n_qubits: int,
    initial_state: np.ndarray | None = None,
    initial_layout: list[int] | None = None,
    final_layout: list[int] | None = None,
) -> np.ndarray:
    """在 PennyLane ``default.qubit`` 上精确执行门序列并返回逻辑比特序态向量。"""
    dev = qml.device("default.qubit", wires=n_qubits)
    init_phys = None
    if initial_state is not None:
        init_phys = (
            permute_statevector(initial_state, initial_layout, inverse=False)
            if initial_layout is not None
            else np.asarray(initial_state, dtype=np.complex128)
        )

    @qml.qnode(dev)
    def _circ():
        if init_phys is not None:
            qml.StatePrep(init_phys, wires=range(n_qubits), normalize=True)
        for gate in gates:
            _apply_gate_op(gate, adjoint=False)
        return qml.state()

    out_state = np.asarray(_circ(), dtype=np.complex128)
    if final_layout is not None:
        out_state = permute_statevector(out_state, final_layout, inverse=True)
    return out_state


def simulate_routed_qcnn_forward_mixed(
    routed_result: dict[str, Any],
    states: np.ndarray,
    n_qubits: int,
    p_2q: float = 0.0,
    p_ro: float = 0.0,
) -> np.ndarray:
    """前向 PennyLane ``default.mixed`` 逐态模拟路由后的 QCNN（用于无噪/海森堡精确性验证）。"""
    dev = qml.device("default.mixed", wires=n_qubits)
    routed_gates: list[GateOp] = routed_result["routed_gates"]
    init_layout: list[int] = routed_result["initial_layout"]
    final_wire = int(routed_result["final_layout"][0])
    states_phys = permute_statevector(np.asarray(states, dtype=np.complex128), init_layout, inverse=False)

    @qml.qnode(dev)
    def _fwd(psi_phys):
        qml.StatePrep(psi_phys, wires=range(n_qubits), normalize=True)
        for gate in routed_gates:
            _apply_gate_op(gate, adjoint=False)
            if len(gate.wires) == 2 and p_2q > 0.0:
                qml.DepolarizingChannel(p_2q, wires=gate.wires[0])
                qml.DepolarizingChannel(p_2q, wires=gate.wires[1])
        if p_ro > 0.0:
            qml.BitFlip(p_ro, wires=final_wire)
        return qml.probs(wires=final_wire)

    probs = np.asarray(_fwd(states_phys), dtype=np.float64)
    if probs.ndim == 1:
        probs = probs[None, :]
    return probs


def evaluate_routed_qcnn_mixed(
    routed_result: dict[str, Any],
    states: np.ndarray,
    n_qubits: int,
    p_2q: float = 0.0,
    p_ro: float = 0.0,
) -> np.ndarray:
    """海森堡对偶图像上的精确 PennyLane ``default.mixed`` 路由 QCNN 评估。

    由于所有酉门与去极化/比特翻转 Pauli 信道均为线性 CPTP 映射且噪声信道严格自伴
    （``D_p^dagger = D_p``, ``B_p^dagger = B_p``），对固定路由线路在 ``default.mixed`` 上
    以合法密度矩阵 ``rho_seed = |0><0|_{final_wire} x I / 2^(n-1)`` 反向传播一次即可获得
    精确含噪 POVM 算符 ``E_0 = E^dagger(|0><0| x I)``，进而一次性计算全部输入态的精确
    Born 概率 ``P(0|psi) = <psi|E_0|psi>``，与逐态前向 ``default.mixed`` 误差 ``< 1e-10``。
    """
    dev = qml.device("default.mixed", wires=n_qubits)
    routed_gates: list[GateOp] = routed_result["routed_gates"]
    init_layout: list[int] = routed_result["initial_layout"]
    final_wire = int(routed_result["final_layout"][0])

    dim = 1 << n_qubits
    scale = float(1 << (n_qubits - 1))
    mask_bit = 1 << (n_qubits - 1 - final_wire)
    diag = np.zeros(dim, dtype=np.float64)
    for basis_idx in range(dim):
        if (basis_idx & mask_bit) == 0:
            diag[basis_idx] = 1.0 / scale
    rho_seed = np.diag(diag)

    @qml.qnode(dev)
    def _heisenberg_povm():
        qml.QubitDensityMatrix(rho_seed, wires=range(n_qubits))
        if p_ro > 0.0:
            qml.BitFlip(p_ro, wires=final_wire)
        for gate in reversed(routed_gates):
            if len(gate.wires) == 2 and p_2q > 0.0:
                qml.DepolarizingChannel(p_2q, wires=gate.wires[0])
                qml.DepolarizingChannel(p_2q, wires=gate.wires[1])
            _apply_gate_op(gate, adjoint=True)
        return qml.density_matrix(wires=range(n_qubits))

    eff_m0 = scale * np.asarray(_heisenberg_povm(), dtype=np.complex128)
    states_phys = permute_statevector(np.asarray(states, dtype=np.complex128), init_layout, inverse=False)
    if states_phys.ndim == 1:
        states_phys = states_phys[None, :]
    p0 = np.real(np.einsum("bi,ij,bj->b", states_phys.conj(), eff_m0, states_phys))
    p0 = np.clip(p0, 0.0, 1.0)
    return np.stack([p0, 1.0 - p0], axis=1)


def evaluate_routed_iqp_density_matrices(
    routed_result: dict[str, Any],
    X: np.ndarray,
    n_qubits: int,
    alpha: float,
    p_2q: float = 0.0,
    batch_size: int = 50,
) -> np.ndarray:
    """在 PennyLane ``default.mixed`` 上执行路由后的 IQP 核线路，返回 ``(N, 2**n, 2**n)`` 密度矩阵。"""
    routed_gates: list[GateOp] = routed_result["routed_gates"]
    X_np = np.asarray(X, dtype=np.float64)
    if X_np.ndim == 1:
        X_np = X_np[None, :]
    dev = qml.device("default.mixed", wires=n_qubits)

    @qml.qnode(dev)
    def _iqp_mixed(x_batch):
        for gate in routed_gates:
            if gate.name == "Hadamard":
                qml.Hadamard(wires=gate.wires[0])
            elif gate.name == "IQP_RZ":
                lq = gate.logical_wires[0]
                qml.RZ(alpha * x_batch[..., lq], wires=gate.wires[0])
            elif gate.name == "IQP_ZZ":
                la, lb = gate.logical_wires
                phi = (alpha ** 2) * (math.pi - x_batch[..., la]) * (math.pi - x_batch[..., lb])
                qml.IsingZZ(phi, wires=list(gate.wires))
                if p_2q > 0.0:
                    qml.DepolarizingChannel(p_2q, wires=gate.wires[0])
                    qml.DepolarizingChannel(p_2q, wires=gate.wires[1])
            elif gate.name == "CNOT":
                qml.CNOT(wires=list(gate.wires))
                if p_2q > 0.0:
                    qml.DepolarizingChannel(p_2q, wires=gate.wires[0])
                    qml.DepolarizingChannel(p_2q, wires=gate.wires[1])
            else:
                _apply_gate_op(gate, adjoint=False)
                if len(gate.wires) == 2 and p_2q > 0.0:
                    qml.DepolarizingChannel(p_2q, wires=gate.wires[0])
                    qml.DepolarizingChannel(p_2q, wires=gate.wires[1])
        return qml.density_matrix(wires=range(n_qubits))

    chunks = []
    for start in range(0, len(X_np), batch_size):
        xb = X_np[start:start + batch_size]
        dm = np.asarray(_iqp_mixed(xb), dtype=np.complex128)
        if dm.ndim == 2:
            dm = dm[None, :, :]
        chunks.append(dm)
    return np.concatenate(chunks, axis=0)


def hilbert_schmidt_kernel_matrix(rhos_a: np.ndarray, rhos_b: np.ndarray) -> np.ndarray:
    """由含噪密度矩阵计算精确 Hilbert-Schmidt 保真度核 ``K_ij = Tr(rho_a[i] rho_b[j])``。

    无噪纯态下严格等于 ``|<psi_a[i]|psi_b[j]>|^2``；在对称自伴去极化信道下等价于
    两路含噪态制备的 SWAP-test / compute-uncompute 交叠且保证 Gram 矩阵半正定（PSD）。
    """
    a_flat = np.asarray(rhos_a, dtype=np.complex128).reshape(len(rhos_a), -1)
    b_flat = np.asarray(rhos_b, dtype=np.complex128).reshape(len(rhos_b), -1)
    return np.real(a_flat @ b_flat.conj().T)


def kernel_concentration_metrics(K_noisy: np.ndarray, K_clean: np.ndarray) -> dict[str, float]:
    """统计核矩阵的去极化集中指标（检验 Thanasilp et al. 2024 噪声诱导核集中）。"""
    n = K_noisy.shape[0]
    off_mask = ~np.eye(n, dtype=bool)
    off_noisy = K_noisy[off_mask]
    off_clean = K_clean[off_mask]
    clean_std = float(np.std(off_clean))
    noisy_std = float(np.std(off_noisy))
    clean_norm = float(np.linalg.norm(K_clean, ord="fro"))
    frob_abs = float(np.linalg.norm(K_noisy - K_clean, ord="fro"))
    return {
        "diag_mean_purity": round(float(np.mean(np.diag(K_noisy))), 6),
        "offdiag_mean": round(float(np.mean(off_noisy)), 6),
        "offdiag_std": round(noisy_std, 6),
        "offdiag_std_ratio": round(noisy_std / max(clean_std, 1e-12), 6),
        "frobenius_dist_rel": round(frob_abs / max(clean_norm, 1e-12), 6),
        "frobenius_dist_abs": round(frob_abs, 6),
    }
