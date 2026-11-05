# -*- coding: utf-8 -*-
"""可微分的量子变分线路层（PennyLane + PyTorch）。"""
from __future__ import annotations

import pennylane as qml
import torch
from torch import nn


def n_quantum_params(n_qubits: int, n_layers: int) -> int:
    """StronglyEntanglingLayers 每个量子比特、每层有 3 个旋转角。"""
    return n_layers * n_qubits * 3


class _Float32VQC(nn.Module):
    """薄封装：统一输入/输出为 float32；cast 是可微算子，不破坏梯度图。"""

    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer  # 注册为子模块，参数可被 model.parameters() 收集

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        out = self.layer(x)
        out = out.to(torch.float32)  # TorchLayer 常返回 float64
        # 少数 PennyLane 版本广播时返回 (n_qubits, B)，纠正为契约要求的 (B, n_qubits)
        if out.dim() == 2 and out.shape[0] != x.shape[0] and out.shape[1] == x.shape[0]:
            out = out.transpose(0, 1)
        return out


def make_vqc_layer(
    n_qubits: int = 4,
    n_layers: int = 2,
    diff_method: str = "backprop",
    shots: int | None = None,
) -> nn.Module:
    """构造线路层：输入 (B, n_qubits)，输出 (B, n_qubits) 的 PauliZ 期望值。"""
    analytic = shots is None or shots == 0
    # 有限 shots 是采样估计，不支持解析 backprop，改用 parameter-shift
    if not analytic and diff_method == "backprop":
        diff_method = "parameter-shift"
        print("[quantum] 有限 shots 不支持 backprop，diff_method 已自动切换为 parameter-shift")

    # 解析模拟不传 shots 参数；有限采样才把 shots 交给设备
    if analytic:
        dev = qml.device("default.qubit", wires=n_qubits)
    else:
        dev = qml.device("default.qubit", wires=n_qubits, shots=shots)

    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shapes = {"weights": (n_layers, n_qubits, 3)}
    # 新版 PennyLane 默认 interface="auto"，TorchLayer 会按 torch 张量自动推断接口；
    # 旧版若强制要求 interface="torch"，回退显式声明以兼容。
    try:
        qnode = qml.qnode(dev, diff_method=diff_method)(circuit)
        layer = qml.qnn.TorchLayer(qnode, weight_shapes)
    except Exception:
        qnode = qml.qnode(dev, interface="torch", diff_method=diff_method)(circuit)
        layer = qml.qnn.TorchLayer(qnode, weight_shapes)

    layer = layer.to(torch.float32)  # 统一权重精度，避免与 float32 输入混用
    return _Float32VQC(layer)


def make_amplitude_layer(
    n_qubits: int = 10,
    n_layers: int = 2,
    diff_method: str = "backprop",
    shots: int | None = None,
    n_features: int = 784,
) -> nn.Module:
    """振幅编码层：输入 (B, n_features) 的 L2 归一化向量，输出 (B, n_qubits)。

    2**n_qubits >= n_features 才能装下完整特征；不足的振幅用 0 补齐。
    与 AngleEmbedding 的关键区别：**不需要把 784 维压到 4 维**，
    n 个 qubits 的态空间天然容纳 2^n 个振幅。

    重要边界：在状态向量模拟器里，制备这个初态是「免费」的（直接写入态向量）；
    真实量子硬件上需要 O(2^n) 深度的状态制备电路或尚不存在的 QRAM，
    因此这个实验的耗时不代表真机成本。
    """
    if 2 ** n_qubits < n_features:
        raise ValueError(
            f"2**{n_qubits}={2 ** n_qubits} < n_features={n_features}；"
            f"至少需要 n_qubits={int(n_features - 1).bit_length()} 才能装下"
        )

    analytic = shots is None or shots == 0
    if not analytic and diff_method == "backprop":
        diff_method = "parameter-shift"
        print("[quantum] 有限 shots 不支持 backprop，diff_method 已自动切换为 parameter-shift")

    dev = (qml.device("default.qubit", wires=n_qubits) if analytic
           else qml.device("default.qubit", wires=n_qubits, shots=shots))

    def circuit(inputs, weights):
        qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True, pad_with=0.0)
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shapes = {"weights": (n_layers, n_qubits, 3)}
    try:
        qnode = qml.qnode(dev, diff_method=diff_method)(circuit)
        layer = qml.qnn.TorchLayer(qnode, weight_shapes)
    except Exception:
        qnode = qml.qnode(dev, interface="torch", diff_method=diff_method)(circuit)
        layer = qml.qnn.TorchLayer(qnode, weight_shapes)

    layer = layer.to(torch.float32)
    return _Float32VQC(layer)
