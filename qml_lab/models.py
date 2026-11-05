# -*- coding: utf-8 -*-
"""VQC / MLP 分类器，以及量子增强的玩具级自回归字符语言模型。"""
from __future__ import annotations

import torch
from torch import nn

try:
    from .quantum import make_vqc_layer
except ImportError:  # 兼容把 qml_lab/ 直接加入 sys.path 后 import models 的方式
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


class TinyLM(nn.Module):
    """字符级自回归模型：embedding -> 可微变换层(量子/经典) -> 词表分类头。"""

    def __init__(self, vocab_size: int, n_qubits: int, transform: nn.Module):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, n_qubits)
        self.transform = transform
        self.head = nn.Linear(n_qubits, vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        h = self.embed(idx)  # (B, T, Q) float32
        B, T, Q = h.shape
        z = self.transform(h.reshape(B * T, Q))  # 展平后逐位置过变换层
        z = z.to(torch.float32)  # 量子 transform 可能返回 float64；cast 可微，保留梯度
        return self.head(z.reshape(B, T, Q))  # (B, T, V) logits
