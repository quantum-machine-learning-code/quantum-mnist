"""VQC 量子分类器、振幅编码分类器与经典 MLP 基线。"""
from __future__ import annotations

import torch
from torch import nn

try:
    from .quantum import make_amplitude_layer, make_vqc_layer
except ImportError:
    from quantum import make_amplitude_layer, make_vqc_layer


def _build_head(n_in: int, n_classes: int, head: str, hidden: int) -> nn.Module:
    """分类头。``linear`` 为原始基线；``mlp`` 给量子读出补上经典非线性。"""
    if head == "linear":
        return nn.Linear(n_in, n_classes)
    if head == "mlp":
        return nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, n_classes),
        )
    raise ValueError(f"未知分类头 {head!r}，可选 'linear' / 'mlp'")


def _set_scale(module: nn.Module, scale) -> None:
    """量子读数恒在 [-1,1]，放大后再进分类头可缓解 logit contraction。

    ``learnable`` 时缩放系数作为可训练参数（QMT 思路，见 docs/REVIEW.md [8]）；
    否则注册为固定 buffer，不参与优化也不改变原有行为。
    """
    if isinstance(scale, str) and scale.lower() == "learnable":
        module.out_scale = nn.Parameter(torch.tensor(1.0))
    else:
        module.register_buffer("out_scale", torch.tensor(float(scale)))


class VQCClassifier(nn.Module):
    """角度编码：输入维度必须等于 n_qubits（一个特征一个旋转角）。"""

    def __init__(
        self,
        n_qubits: int,
        n_classes: int,
        n_layers: int = 2,
        diff_method: str = "backprop",
        shots: int | None = None,
        head: str = "linear",
        scale=1.0,
        hidden: int = 32,
    ):
        super().__init__()
        self.q = make_vqc_layer(
            n_qubits=n_qubits,
            n_layers=n_layers,
            diff_method=diff_method,
            shots=shots,
        )
        self.head = _build_head(n_qubits, n_classes, head, hidden)
        _set_scale(self, scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        z = self.q(x).to(torch.float32)  # 量子层输出可能是 float64
        return self.head(z * self.out_scale)  # (B, n_classes) logits


class AmplitudeVQCClassifier(nn.Module):
    """振幅编码：输入可为完整 784 维，n_qubits 只需满足 2**n >= 784。

    与 VQCClassifier 的区别是**输入不需要 PCA 压缩**；代价见 make_amplitude_layer
    文档里关于真机状态制备的说明。
    """

    def __init__(
        self,
        n_qubits: int,
        n_classes: int,
        n_layers: int = 2,
        n_features: int = 784,
        diff_method: str = "backprop",
        shots: int | None = None,
        head: str = "linear",
        scale=1.0,
        hidden: int = 32,
    ):
        super().__init__()
        self.q = make_amplitude_layer(
            n_qubits=n_qubits,
            n_layers=n_layers,
            diff_method=diff_method,
            shots=shots,
            n_features=n_features,
        )
        self.head = _build_head(n_qubits, n_classes, head, hidden)
        _set_scale(self, scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        z = self.q(x).to(torch.float32)
        return self.head(z * self.out_scale)


class MLPClassifier(nn.Module):
    """与量子模型对比的经典 MLP 基线；首个参数实际是输入维度，不限 qubits 数。"""

    def __init__(self, n_qubits: int, n_classes: int, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_qubits, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.to(torch.float32))
