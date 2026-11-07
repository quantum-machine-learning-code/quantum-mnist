# -*- coding: utf-8 -*-
"""可微分的量子变分线路层（PennyLane + PyTorch）。"""
from __future__ import annotations

import math

import pennylane as qml
import torch
from torch import nn


def n_quantum_params(n_qubits: int, n_layers: int, trainable_reupload: bool = False) -> int:
    """StronglyEntanglingLayers 每个量子比特、每层有 3 个旋转角；可训练重上传另加每层 2n 个输入缩放与偏置。"""
    base = n_layers * n_qubits * 3
    return base + (2 * n_layers * n_qubits if trainable_reupload else 0)


class _Float32VQC(nn.Module):
    """薄封装：统一输入/输出为 float32；cast 是可微算子，不破坏梯度图。"""

    def __init__(self, layer: nn.Module, bound_inputs: bool = False):
        super().__init__()
        self.layer = layer  # 注册为子模块，参数可被 model.parameters() 收集
        self.bound_inputs = bound_inputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if self.bound_inputs:
            # 将任意实数输入平滑压缩到 [-pi/2, pi/2] 的单调区间，避免周期旋转相位混叠
            x = torch.tanh(x / (math.pi / 2.0)) * (math.pi / 2.0)
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


# ---------------------------------------------------------------------------
# 可配置读出 + 数据重上传
#
# 动机（见 docs/REVIEW.md 第 8 节）：5 seed 实测把训练好的量子层冻结后，只取
# n 个 PauliZ 期望交给任意经典强分类器，最高只有 0.679；而原始像素是 0.855。
# 约 17.6 pt 的可分信息丢在「单次编码 → 酉 → 只测 n 个有界数」这条链路上。
# 这里提供两条互相独立的解锁手段：
#   * observables: 把读出从 n 个 Z 扩到 X/Y/ZZ 关联等多个期望
#   * reupload:    每一层重新嵌入输入，把单次编码的 degree-1 傅里叶谱提高
# ---------------------------------------------------------------------------

_READOUT_PARTS = ("z", "x", "y", "zz", "zzall", "xxall", "yyall")


def make_observables(n_qubits: int, spec: str):
    """按 ``+`` 分隔的规格构造可观测测量表。

    ``z`` / ``x`` / ``y`` 各贡献 n 个单比特期望；``zz`` 贡献 n 个相邻环
    ``Z_i Z_{i+1}`` 关联；``zzall`` / ``xxall`` / ``yyall`` 贡献
    **所有** ``n(n-1)/2`` 个两体关联（如 ``Z_i Z_j``，i<j）。
    两体关联比单比特期望携带更多输入信息：振幅编码 + 酉 + Pauli 期望
    整体是输入的二次型，扩充观测量等于扩充可访问的二次统计量集合。

    ``default.qubit`` 在解析模式下允许互不対易的可观测量同时测量；
    有限 shots 时 PennyLane 会自动按 QWC 对易组拆分（例如 165 维规格只需 Z/X/Y 3 个测量基）。

    规格按固定顺序展开，与 ``+`` 里的书写顺序无关；旧规格 ``"z"`` /
    ``"z+x+y+zz"`` 的展开顺序保持不变，以保证回归测试逐位一致。
    """
    parts = [p.strip().lower() for p in spec.split("+")]
    unknown = [p for p in parts if p not in _READOUT_PARTS]
    if unknown:
        raise ValueError(f"未知可观测量 {unknown}，可选 {_READOUT_PARTS}")
    ops = []
    if "z" in parts:
        ops.extend(qml.PauliZ(i) for i in range(n_qubits))
    if "x" in parts:
        ops.extend(qml.PauliX(i) for i in range(n_qubits))
    if "y" in parts:
        ops.extend(qml.PauliY(i) for i in range(n_qubits))
    if "zz" in parts:
        ops.extend(qml.PauliZ(i) @ qml.PauliZ((i + 1) % n_qubits)
                   for i in range(n_qubits))
    if "zzall" in parts:
        ops.extend(qml.PauliZ(i) @ qml.PauliZ(j)
                   for i in range(n_qubits) for j in range(i + 1, n_qubits))
    if "xxall" in parts:
        ops.extend(qml.PauliX(i) @ qml.PauliX(j)
                   for i in range(n_qubits) for j in range(i + 1, n_qubits))
    if "yyall" in parts:
        ops.extend(qml.PauliY(i) @ qml.PauliY(j)
                   for i in range(n_qubits) for j in range(i + 1, n_qubits))
    return ops


def n_readout(n_qubits: int, spec: str) -> int:
    """与 :func:`make_observables` 对应的读出维度，供分类头构造使用。"""
    return len(make_observables(n_qubits, spec))


def make_rich_layer(
    n_qubits: int = 10,
    n_layers: int = 2,
    encoding: str = "amplitude",
    observables: str = "z",
    n_features: int = 784,
    reupload: bool | str = False,
    trainable_reupload: bool = False,
    init_std: float | None = None,
    diff_method: str = "backprop",
    shots: int | None = None,
) -> nn.Module:
    """通用可变量子线路层：可配置编码、读出与数据重上传（含 Pérez-Salinas 可训练重上传）。

    参数
    ----
    encoding : ``"amplitude"`` 用 ``AmplitudeEmbedding``（输入 784 维，
        要求 ``2**n_qubits >= n_features``）；``"angle"`` 用
        ``AngleEmbedding(RY)``（输入维度必须等于 ``n_qubits``）。
    observables : 传给 :func:`make_observables` 的规格，例如 ``"z+x"``。
    reupload : 若为 ``True``，每层原样重嵌入输入（朴素重上传，与第四轮归档逐位兼容）；
        若为 ``"trainable"``（或 ``trainable_reupload=True``），启用 Pérez-Salinas et al.
        (2019) 可训练数据重上传：对有界输入 ``tanh(x/(pi/2))*(pi/2)`` 在每层施加可学习输入缩放
        ``omega_{l,i} ~ N(1.0, 0.1^2)`` 与旋转偏置 ``theta_{l,i} ~ N(0, 0.05^2)``。
    init_std : 若指定（如 ``0.05``），将变分旋转权重初始化为 ``N(0, init_std^2)`` 近恒等初值。
    """
    is_trainable_reup = bool(trainable_reupload) or (
        isinstance(reupload, str) and reupload.strip().lower() == "trainable"
    )
    is_naive_reup = bool(reupload) and not is_trainable_reup

    if encoding not in ("amplitude", "angle"):
        raise ValueError(f"未知编码 {encoding!r}，可选 'amplitude' / 'angle'")
    if encoding == "amplitude":
        if is_naive_reup or is_trainable_reup:
            raise ValueError("AmplitudeEmbedding 会重置量子态，不支持数据重上传")
        if 2 ** n_qubits < n_features:
            raise ValueError(
                f"2**{n_qubits}={2 ** n_qubits} < n_features={n_features}"
            )
    ops = make_observables(n_qubits, observables)

    analytic = shots is None or shots == 0
    if not analytic and diff_method == "backprop":
        diff_method = "parameter-shift"
        print("[quantum] 有限 shots 不支持 backprop，diff_method 已自动切换为 parameter-shift")
    dev = (qml.device("default.qubit", wires=n_qubits) if analytic
           else qml.device("default.qubit", wires=n_qubits, shots=shots))

    def _embed(inputs):
        if encoding == "amplitude":
            qml.AmplitudeEmbedding(inputs, wires=range(n_qubits),
                                   normalize=True, pad_with=0.0)
        else:
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")

    if is_trainable_reup:
        def circuit(inputs, weights, in_scale, in_shift):
            for layer_i in range(n_layers):
                angles = inputs * in_scale[layer_i] + in_shift[layer_i]
                qml.AngleEmbedding(angles, wires=range(n_qubits), rotation="Y")
                qml.StronglyEntanglingLayers(weights[layer_i:layer_i + 1],
                                             wires=range(n_qubits))
            return [qml.expval(op) for op in ops]

        weight_shapes = {
            "weights": (n_layers, n_qubits, 3),
            "in_scale": (n_layers, n_qubits),
            "in_shift": (n_layers, n_qubits),
        }
    else:
        def circuit(inputs, weights):
            if is_naive_reup:
                for layer_i in range(n_layers):
                    _embed(inputs)
                    qml.StronglyEntanglingLayers(weights[layer_i:layer_i + 1],
                                                 wires=range(n_qubits))
            else:
                _embed(inputs)
                qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(op) for op in ops]

        weight_shapes = {"weights": (n_layers, n_qubits, 3)}

    try:
        qnode = qml.qnode(dev, diff_method=diff_method)(circuit)
        layer = qml.qnn.TorchLayer(qnode, weight_shapes)
    except Exception:
        qnode = qml.qnode(dev, interface="torch", diff_method=diff_method)(circuit)
        layer = qml.qnn.TorchLayer(qnode, weight_shapes)

    layer = layer.to(torch.float32)
    with torch.no_grad():
        if is_trainable_reup:
            std_w = 0.05 if init_std is None else float(init_std)
            nn.init.normal_(layer.weights, mean=0.0, std=std_w)
            nn.init.normal_(layer.in_scale, mean=1.0, std=0.1)
            nn.init.normal_(layer.in_shift, mean=0.0, std=0.05)
        elif init_std is not None:
            nn.init.normal_(layer.weights, mean=0.0, std=float(init_std))

    return _Float32VQC(layer, bound_inputs=is_trainable_reup)
