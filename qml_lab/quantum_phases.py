# -*- coding: utf-8 -*-
"""原生量子态基准（Track B）：1D Cluster-Ising SPT 相分类与真正 Cong-et-al. QCNN。

哈密顿量（周期边界条件 PBC，默认 n=8 比特）：
    H(h1, h2) = -sum_i Z_{i-1} X_i Z_{i+1} - h1 sum_i X_i - h2 sum_i X_i X_{i+1}

物理背景（Cong, Choi, Lukin, Nature Physics 2019）：
  * SPT（Cluster）相（|h1|, |h2| 较小）：具有非局域弦序（string order）与 Z2 x Z2 保护拓扑序；
    在精确 cluster 态上所有单比特与两体局域期望均严格为零。
  * 非 SPT 相包含顺磁相（|h1| > 1，单比特 <X_i> 非零）与反铁磁 Ising 相（h2 < -1，
    在弱均匀场 h1 下单比特 <X_i> 仍近似为零，但两体 <X_i X_{i+1}> 接近 -1）。
  * 与经典像素振幅编码不同，这里的输入本身就是量子多体基态 |psi(h1, h2)>（无 O(2^n) 经典装载作弊）：
    Cong-et-al. QCNN 通过平移不变的两比特卷积酉 + 量子池化（8 -> 4 -> 2 -> 1，按延迟测量原理等价于
    中间测量+经典前馈），在 0 个经典 MLP 参数、仅 1 个最终量子比特测量基下直接提取非局域拓扑相序。
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pennylane as qml
from scipy import sparse
from scipy.sparse.linalg import eigsh
import torch
from torch import nn

__all__ = [
    "cluster_ising_operators",
    "cluster_ising_ground_state",
    "generate_cluster_ising_dataset",
    "local_1body_features",
    "local_2body_features",
    "QCNNPhaseClassifier",
]

_I2_SP = sparse.eye(2, format="csr", dtype=np.float64)
_X2_SP = sparse.csr_matrix([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
_Z2_SP = sparse.csr_matrix([[1.0, 0.0], [0.0, -1.0]], dtype=np.float64)


def _kron_sparse(mats: list[sparse.csr_matrix]) -> sparse.csr_matrix:
    out = mats[0]
    for m in mats[1:]:
        out = sparse.kron(out, m, format="csr")
    return out


@lru_cache(maxsize=4)
def cluster_ising_operators(n_qubits: int = 8) -> dict:
    """构造 n 比特 PBC 1D Cluster-Ising 稀疏哈密顿量分量与弦序算符（缓存复用）。"""
    if n_qubits < 4 or (n_qubits & (n_qubits - 1)) != 0:
        raise ValueError(f"QCNN 要求 n_qubits 为 >=4 的 2 的幂（如 4 或 8），实际 {n_qubits}")
    dim = 1 << n_qubits
    h_cl = sparse.csr_matrix((dim, dim), dtype=np.float64)
    h_x = sparse.csr_matrix((dim, dim), dtype=np.float64)
    h_xx = sparse.csr_matrix((dim, dim), dtype=np.float64)

    for i in range(n_qubits):
        m_cl = [_I2_SP] * n_qubits
        m_cl[(i - 1) % n_qubits] = _Z2_SP
        m_cl[i] = _X2_SP
        m_cl[(i + 1) % n_qubits] = _Z2_SP
        h_cl = h_cl - _kron_sparse(m_cl)

        m_x = [_I2_SP] * n_qubits
        m_x[i] = _X2_SP
        h_x = h_x - _kron_sparse(m_x)

        m_xx = [_I2_SP] * n_qubits
        m_xx[i] = _X2_SP
        m_xx[(i + 1) % n_qubits] = _X2_SP
        h_xx = h_xx - _kron_sparse(m_xx)

    # 弦序算符：n=8 时 K_1 K_3 = Z_0 X_1 I_2 X_3 Z_4；n=4 时 K_0 = Z_3 X_0 Z_1
    m_str = [_I2_SP] * n_qubits
    if n_qubits >= 8:
        m_str[0] = _Z2_SP
        m_str[1] = _X2_SP
        m_str[3] = _X2_SP
        m_str[4] = _Z2_SP
    else:
        m_str[n_qubits - 1] = _Z2_SP
        m_str[0] = _X2_SP
        m_str[1] = _Z2_SP
    string_op = _kron_sparse(m_str)

    return {
        "H_cluster": h_cl.tocsr(),
        "H_x": h_x.tocsr(),
        "H_xx": h_xx.tocsr(),
        "string_op": string_op.tocsr(),
    }


def cluster_ising_ground_state(
    h1: float, h2: float, n_qubits: int = 8
) -> tuple[np.ndarray, float, float]:
    """精确稀疏对角化求 ``H(h1, h2)`` 基态 ``|psi>``、能隙与弦序期望值。

    固定最大幅值分量为正以统一全局符号规范（对量子线路严格无关，对经典振幅基准公平）。
    """
    ops = cluster_ising_operators(n_qubits)
    H = ops["H_cluster"] + float(h1) * ops["H_x"] + float(h2) * ops["H_xx"]
    if H.shape[0] <= 32:
        evals, evecs = np.linalg.eigh(H.toarray())
    else:
        evals, evecs = eigsh(H, k=2, which="SA", tol=1e-10)
        order = np.argsort(evals)
        evals, evecs = evals[order], evecs[:, order]
    psi = evecs[:, 0].astype(np.float64)
    idx_max = int(np.argmax(np.abs(psi)))
    if psi[idx_max] < 0.0:
        psi = -psi
    psi = psi / max(float(np.linalg.norm(psi)), 1e-12)
    gap = float(evals[1] - evals[0])
    s_order = float(psi @ (ops["string_op"] @ psi))
    return psi.astype(np.float32), gap, s_order


def _apply_pauli_1q(T: np.ndarray, wire: int, pauli: str) -> np.ndarray:
    """在形状 ``(B, 2, ..., 2)`` 的态张量上对单个比特施加 X/Y/Z。"""
    ax = wire + 1
    out = np.empty_like(T)
    sl0 = [slice(None)] * T.ndim
    sl1 = [slice(None)] * T.ndim
    sl0[ax] = 0
    sl1[ax] = 1
    t0, t1 = tuple(sl0), tuple(sl1)
    if pauli == "X":
        out[t0] = T[t1]
        out[t1] = T[t0]
    elif pauli == "Y":
        out[t0] = -1.0j * T[t1]
        out[t1] = 1.0j * T[t0]
    else:  # Z
        out[t0] = T[t0]
        out[t1] = -T[t1]
    return out


def local_1body_features(states: np.ndarray, n_qubits: int = 8) -> np.ndarray:
    """提取所有单比特局域 Pauli 期望值 ``<X_i>, <Y_i>, <Z_i>``，维度 ``3 * n_qubits``。"""
    S = np.asarray(states, dtype=np.complex128)
    B = S.shape[0]
    T = S.reshape((B,) + (2,) * n_qubits)
    feats = []
    for i in range(n_qubits):
        for p in ("X", "Y", "Z"):
            pt = _apply_pauli_1q(T, i, p).reshape(B, -1)
            feats.append(np.real(np.sum(S.conj() * pt, axis=1)))
    return np.ascontiguousarray(np.stack(feats, axis=1), dtype=np.float32)


def local_2body_features(states: np.ndarray, n_qubits: int = 8) -> np.ndarray:
    """提取 1-body + 近邻 2-body 局域 Pauli 期望值，维度 ``6 * n_qubits``。"""
    S = np.asarray(states, dtype=np.complex128)
    B = S.shape[0]
    T = S.reshape((B,) + (2,) * n_qubits)
    f1 = local_1body_features(states, n_qubits=n_qubits)
    feats2 = []
    for i in range(n_qubits):
        j = (i + 1) % n_qubits
        for p in ("X", "Y", "Z"):
            pt = _apply_pauli_1q(_apply_pauli_1q(T, i, p), j, p).reshape(B, -1)
            feats2.append(np.real(np.sum(S.conj() * pt, axis=1)))
    f2 = np.stack(feats2, axis=1).astype(np.float32)
    return np.ascontiguousarray(np.concatenate([f1, f2], axis=1), dtype=np.float32)


def _sample_phase_split(
    n_samples: int, n_qubits: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, dict]:
    """在 SPT（label=1）与非 SPT（50% 顺磁 + 50% 反铁磁，label=0）深相区抽样。"""
    n_spt = n_samples // 2
    n_non = n_samples - n_spt
    n_pm = n_non // 2
    n_afm = n_non - n_pm

    states, labels, subphases, gaps, strings = [], [], [], [], []
    # SPT 相：|h1| <= 0.35, |h2| <= 0.35
    for _ in range(n_spt):
        h1 = float(rng.uniform(-0.35, 0.35))
        h2 = float(rng.uniform(-0.35, 0.35))
        psi, gap, s_val = cluster_ising_ground_state(h1, h2, n_qubits=n_qubits)
        states.append(psi)
        labels.append(1)
        subphases.append("spt")
        gaps.append(gap)
        strings.append(s_val)

    # 顺磁相（非 SPT）：|h1| in [1.35, 2.00], |h2| <= 0.35
    for _ in range(n_pm):
        sgn = float(rng.choice([-1.0, 1.0]))
        h1 = sgn * float(rng.uniform(1.35, 2.00))
        h2 = float(rng.uniform(-0.35, 0.35))
        psi, gap, s_val = cluster_ising_ground_state(h1, h2, n_qubits=n_qubits)
        states.append(psi)
        labels.append(0)
        subphases.append("paramagnetic")
        gaps.append(gap)
        strings.append(s_val)

    # 反铁磁 Ising 相（非 SPT）：|h1| <= 0.35, h2 in [-1.85, -1.25]
    for _ in range(n_afm):
        h1 = float(rng.uniform(-0.35, 0.35))
        h2 = float(rng.uniform(-1.85, -1.25))
        psi, gap, s_val = cluster_ising_ground_state(h1, h2, n_qubits=n_qubits)
        states.append(psi)
        labels.append(0)
        subphases.append("antiferromagnetic")
        gaps.append(gap)
        strings.append(s_val)

    S = np.ascontiguousarray(np.stack(states, axis=0), dtype=np.float32)
    y = np.ascontiguousarray(labels, dtype=np.int64)
    spt_mask = y == 1
    non_mask = y == 0
    stats = {
        "n_samples": int(len(y)),
        "min_gap": round(float(np.min(gaps)), 4),
        "spt_string_order_mean": round(float(np.mean(np.asarray(strings)[spt_mask])), 4),
        "non_spt_string_order_mean": round(float(np.mean(np.asarray(strings)[non_mask])), 4),
    }
    return S, y, stats


def generate_cluster_ising_dataset(
    n_train: int = 120,
    n_val: int = 60,
    n_test: int = 120,
    n_qubits: int = 8,
    seed: int = 0,
) -> dict:
    """生成严格独立的 Train / Val / Test 1D Cluster-Ising 基态数据集与局域特征视图。"""
    rng = np.random.default_rng(seed)
    S_tr, y_tr, st_tr = _sample_phase_split(n_train, n_qubits, rng)
    S_val, y_val, st_val = _sample_phase_split(n_val, n_qubits, rng)
    S_te, y_te, st_te = _sample_phase_split(n_test, n_qubits, rng)

    return {
        "states": {"X_train": S_tr, "X_val": S_val, "X_test": S_te},
        "local_1body": {
            "X_train": local_1body_features(S_tr, n_qubits),
            "X_val": local_1body_features(S_val, n_qubits),
            "X_test": local_1body_features(S_te, n_qubits),
        },
        "local_2body": {
            "X_train": local_2body_features(S_tr, n_qubits),
            "X_val": local_2body_features(S_val, n_qubits),
            "X_test": local_2body_features(S_te, n_qubits),
        },
        "y_train": y_tr,
        "y_val": y_val,
        "y_test": y_te,
        "meta": {
            "hamiltonian": "H = -sum Z_{i-1}X_iZ_{i+1} - h1 sum X_i - h2 sum X_iX_{i+1} (PBC)",
            "n_qubits": int(n_qubits),
            "seed": int(seed),
            "train_stats": st_tr,
            "val_stats": st_val,
            "test_stats": st_te,
        },
    }


def _conv_2q(params, w0: int, w1: int) -> None:
    """平移不变的两比特卷积块（8 参数，含 IsingZZ/IsingXX 纠缠与单比特旋转）。"""
    qml.RY(params[0], wires=w0)
    qml.RY(params[1], wires=w1)
    qml.RZ(params[2], wires=w0)
    qml.RZ(params[3], wires=w1)
    qml.IsingZZ(params[4], wires=[w0, w1])
    qml.IsingXX(params[5], wires=[w0, w1])
    qml.RY(params[6], wires=w0)
    qml.RY(params[7], wires=w1)


def _pool_2q(params, kept: int, dropped: int) -> None:
    """相干受控量子池化（按延迟测量原理严格等价于测量 dropped 后对 kept 施加前馈旋转）。"""
    qml.CRZ(params[0], wires=[dropped, kept])
    qml.CRY(params[1], wires=[dropped, kept])
    qml.CRX(params[2], wires=[dropped, kept])


class _QCNNCore(nn.Module):
    """Cong-et-al. QCNN 线路核心（零经典参数，仅保留最终 1 比特 Born 概率读出）。"""

    _is_quantum_module = True

    def __init__(self, n_qubits: int = 8):
        super().__init__()
        if n_qubits < 4 or (n_qubits & (n_qubits - 1)) != 0:
            raise ValueError(f"QCNN 要求 n_qubits 为 >=4 的 2 的幂，实际 {n_qubits}")
        self.n_qubits = int(n_qubits)
        self.n_scales = int(round(np.log2(self.n_qubits)))
        dev = qml.device("default.qubit", wires=self.n_qubits)

        def circuit(inputs, conv_w, pool_w, final_rot):
            qml.StatePrep(inputs, wires=range(self.n_qubits), normalize=True)
            active = list(range(self.n_qubits))
            for scale_i in range(self.n_scales):
                m = len(active)
                # 偶键卷积（共享参数，严格平移不变）
                for idx in range(0, m, 2):
                    _conv_2q(conv_w[scale_i], active[idx], active[(idx + 1) % m])
                # 奇键卷积（当活跃比特数 > 2 时应用 PBC 奇键）
                if m > 2:
                    for idx in range(1, m, 2):
                        _conv_2q(conv_w[scale_i], active[idx], active[(idx + 1) % m])
                # 量子池化：将奇数位信息相干汇聚至偶数位并丢弃奇数位（m -> m // 2）
                next_active = []
                for idx in range(0, m, 2):
                    kept, dropped = active[idx], active[idx + 1]
                    _pool_2q(pool_w[scale_i], kept=kept, dropped=dropped)
                    next_active.append(kept)
                active = next_active

            final_wire = active[0]
            qml.Rot(final_rot[0], final_rot[1], final_rot[2], wires=final_wire)
            return qml.probs(wires=final_wire)

        weight_shapes = {
            "conv_w": (self.n_scales, 8),
            "pool_w": (self.n_scales, 3),
            "final_rot": (3,),
        }
        qnode = qml.qnode(dev, diff_method="backprop")(circuit)
        self.layer = qml.qnn.TorchLayer(qnode, weight_shapes).to(torch.float32)
        with torch.no_grad():
            nn.init.normal_(self.layer.conv_w, mean=0.0, std=0.4)
            nn.init.normal_(self.layer.pool_w, mean=0.0, std=0.4)
            nn.init.normal_(self.layer.final_rot, mean=0.0, std=0.2)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        probs = self.layer(states.to(torch.float32)).to(torch.float32)
        if probs.dim() == 2 and probs.shape[0] != states.shape[0] and probs.shape[1] == states.shape[0]:
            probs = probs.transpose(0, 1)
        return probs


class QCNNPhaseClassifier(nn.Module):
    """Regime III 真正 Cong-et-al. QCNN（0 经典 MLP 参数，100% 量子参数）。

    对 ``n_qubits=8`` 态向量执行 ``8 -> 4 -> 2 -> 1`` 平移不变卷积+量子池化，
    直接输出单比特 Born 规则对数概率 ``log(P(y | psi))``（总参数 36，全部为量子门角度，
    测量仅需 1 个 QWC 测量基）。
    """

    def __init__(self, n_qubits: int = 8):
        super().__init__()
        self.n_qubits = int(n_qubits)
        self.q = _QCNNCore(n_qubits=n_qubits)
        self.readout_dim = 2
        self.n_qwc_groups = 1

    def born_probs(self, states: torch.Tensor) -> torch.Tensor:
        return self.q(states)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        probs = self.born_probs(states)
        return torch.log(probs.clamp_min(1e-7))

    def evaluate_finite_shots(self, states, labels, shots: int = 200, seed: int = 0) -> float:
        """在真实有限 shots（单比特 1 个 QWC 测量基）下评估准确率。"""
        dev = qml.device("default.qubit", wires=self.n_qubits, seed=seed)
        n_scales = self.q.n_scales

        @qml.qnode(dev)
        def _circ(inputs, conv_w, pool_w, final_rot):
            qml.StatePrep(inputs, wires=range(self.n_qubits), normalize=True)
            active = list(range(self.n_qubits))
            for scale_i in range(n_scales):
                m = len(active)
                for idx in range(0, m, 2):
                    _conv_2q(conv_w[scale_i], active[idx], active[(idx + 1) % m])
                if m > 2:
                    for idx in range(1, m, 2):
                        _conv_2q(conv_w[scale_i], active[idx], active[(idx + 1) % m])
                next_active = []
                for idx in range(0, m, 2):
                    kept, dropped = active[idx], active[idx + 1]
                    _pool_2q(pool_w[scale_i], kept=kept, dropped=dropped)
                    next_active.append(kept)
                active = next_active
            final_wire = active[0]
            qml.Rot(final_rot[0], final_rot[1], final_rot[2], wires=final_wire)
            return qml.probs(wires=final_wire)

        qnode = qml.set_shots(_circ, shots=int(shots))
        S_np = np.asarray(states, dtype=np.float64)
        y_np = np.asarray(labels, dtype=np.int64)
        cw = self.q.layer.conv_w.detach().cpu().numpy()
        pw = self.q.layer.pool_w.detach().cpu().numpy()
        fr = self.q.layer.final_rot.detach().cpu().numpy()
        probs = np.asarray(qnode(S_np, cw, pw, fr), dtype=np.float64)
        if probs.ndim == 2 and probs.shape[0] != len(y_np) and probs.shape[1] == len(y_np):
            probs = probs.T
        preds = np.argmax(probs, axis=-1)
        return float(np.mean(preds == y_np))
