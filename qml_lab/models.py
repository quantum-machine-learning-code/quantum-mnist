"""VQC 量子分类器与经典 MLP 基线。"""
from __future__ import annotations

import torch
from torch import nn

try:
    from .quantum import make_vqc_layer
except ImportError:
    from quantum import make_vqc_layer

class VQCClassifier(nn.Module):
    """量子线路提取 n_qubits 维特征，再接线性分类头。"""

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


class MLPClassifier(nn.Module):
    """与 VQC 对比的经典 MLP 基线。"""

    def __init__(self, n_qubits: int, n_classes: int, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_qubits, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.to(torch.float32))


