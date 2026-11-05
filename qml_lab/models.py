"""VQC 量子分类器、振幅编码分类器与经典 MLP 基线。"""
from __future__ import annotations

import torch
from torch import nn

try:
    from .quantum import make_amplitude_layer, make_vqc_layer
except ImportError:
    from quantum import make_amplitude_layer, make_vqc_layer


class VQCClassifier(nn.Module):
    """角度编码：输入维度必须等于 n_qubits（一个特征一个旋转角）。"""

    def __init__(
        self,
        n_qubits: int,
        n_classes: int,
        n_layers: int = 2,
        diff_method: str = "backprop",
        shots: int | None = None,
    ):
        super().__init__()
        self.q = make_vqc_layer(
            n_qubits=n_qubits,
            n_layers=n_layers,
            diff_method=diff_method,
            shots=shots,
        )
        self.head = nn.Linear(n_qubits, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        z = self.q(x).to(torch.float32)  # 量子层输出可能是 float64
        return self.head(z)  # (B, n_classes) logits


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
    ):
        super().__init__()
        self.q = make_amplitude_layer(
            n_qubits=n_qubits,
            n_layers=n_layers,
            diff_method=diff_method,
            shots=shots,
            n_features=n_features,
        )
        self.head = nn.Linear(n_qubits, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        z = self.q(x).to(torch.float32)
        return self.head(z)


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
