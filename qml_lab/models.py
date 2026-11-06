"""VQC 量子分类器、振幅编码分类器、混合 CNN 前端与经典基线。"""
from __future__ import annotations

import math

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


def _cnn_backbone(frontend: str = "light") -> tuple[nn.Sequential, int]:
    """共享卷积主干，返回 ``(module, 输出维度)``。

    ``light``：自适应池化 → 32 维，约 3.3k 参数；参数量与量子模型同量级，
    用来隔离「表示方式」而不是「参数更多」。
    ``heavy``：摊平 7x7x16 → 64 维，约 52k 参数；近似参考实现里
    「强经典特征前端」的角色（ResNet18 / CAE）。两者的差别本身就是诊断变量。
    """
    conv = [
        nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    ]
    if frontend == "heavy":
        return (nn.Sequential(*conv, nn.Flatten(),
                              nn.Linear(16 * 7 * 7, 64), nn.ReLU()), 64)
    if frontend == "light":
        return (nn.Sequential(*conv, nn.AdaptiveAvgPool2d((2, 2)),
                              nn.Flatten(), nn.Linear(16 * 2 * 2, 32),
                              nn.ReLU()), 32)
    raise ValueError(f"未知前端 {frontend!r}，可选 'light' / 'heavy'")


def _as_images(x: torch.Tensor) -> torch.Tensor:
    """把 (B, 784) 或 (B, 1, 28, 28) 统一成 CNN 输入；像素需已缩放至 [0,1]。"""
    x = x.to(torch.float32)
    if x.dim() == 2 and x.shape[1] == 784:
        return x.view(-1, 1, 28, 28)
    if x.dim() == 4 and x.shape[1] == 1:
        return x
    raise ValueError(f"期望 (B,784) 或 (B,1,28,28)，实际 {tuple(x.shape)}")


class CNNClassifier(nn.Module):
    """纯经典 CNN 基线：与混合模型完全同一套卷积主干，直接输出 logits。"""

    def __init__(self, n_classes: int = 10, frontend: str = "light"):
        super().__init__()
        self.features, width = _cnn_backbone(frontend)
        self.head = nn.Linear(width, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(_as_images(x)))


class HybridCNNClassifier(nn.Module):
    """经典 CNN 前端 → 可变量子线路 → 经典非线性分类头。

    对齐参考实现的 "dressed quantum net" 结构（PennyLane quantum transfer
    learning 的 ``pre_net → tanh(·)*π/2 → VQC → post_net`` [3]、ICCAD 2021
    的 CAE+QNN [7]）：让经典网络承担高维图像的特征提取，量子层只处理少量
    已被压缩、且幅度被 tanh 限制在 cos 非饱和区的特征。

    ``quantum=False`` 时跳过量子层，用于**等架构**的经典对照 —— 这样能测出
    量子层在这一混合结构里到底贡献了多少，而不是把 CNN 的功劳算给量子。
    """

    def __init__(
        self,
        n_qubits: int = 8,
        n_classes: int = 10,
        n_layers: int = 2,
        hidden: int = 32,
        quantum: bool = True,
        head: str = "mlp",
        scale=1.0,
        diff_method: str = "backprop",
        frontend: str = "light",
    ):
        super().__init__()
        self.backbone, width = _cnn_backbone(frontend)
        self.pre_net = nn.Sequential(nn.Linear(width, n_qubits), nn.Tanh())
        self.q = (make_vqc_layer(n_qubits=n_qubits, n_layers=n_layers,
                                 diff_method=diff_method) if quantum else None)
        self.head = _build_head(n_qubits, n_classes, head, hidden)
        _set_scale(self, scale)

    def pre_features(self, x: torch.Tensor) -> torch.Tensor:
        """前端输出，乘 π/2 后落在 [-π/2, π/2]（避开 RY 的 cos 饱和区）。"""
        f = self.pre_net(self.backbone(_as_images(x)))
        return f * math.pi / 2.0

    def quantum_features(self, x: torch.Tensor) -> torch.Tensor:
        """量子读出向量 z；``quantum=False`` 时返回前端特征本身。"""
        f = self.pre_features(x)
        return self.q(f).to(torch.float32) if self.q is not None else f

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.quantum_features(x)
        return self.head(z * self.out_scale)
