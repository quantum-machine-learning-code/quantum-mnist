"""VQC 量子分类器、振幅编码分类器、局部多 Patch 量子分类器、混合 CNN 前端与经典基线。"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

try:
    from .quantum import (make_amplitude_layer, make_rich_layer,
                          make_vqc_layer, n_readout)
except ImportError:
    from quantum import (make_amplitude_layer, make_rich_layer,
                         make_vqc_layer, n_readout)


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
        self.readout_dim = int(n_qubits)
        self.head = _build_head(n_qubits, n_classes, head, hidden)
        _set_scale(self, scale)

    def quantum_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取量子层读出特征向量（float32）。"""
        return self.q(x.to(torch.float32)).to(torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.quantum_features(x)
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
        self.readout_dim = int(n_qubits)
        self.head = _build_head(n_qubits, n_classes, head, hidden)
        _set_scale(self, scale)

    def quantum_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取量子层读出特征向量（float32）。"""
        return self.q(x.to(torch.float32)).to(torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.quantum_features(x)
        return self.head(z * self.out_scale)


class RichVQCClassifier(nn.Module):
    """可配置读出 / 数据重上传 / 范数保留的 VQC。

    与 :class:`AmplitudeVQCClassifier` 的差别只在量子层：``observables`` 决定
    读出维度，``reupload``（或 ``trainable_reupload``）决定输入是否被每层重新嵌入，
    ``preserve_norm=True`` 在振幅编码前提取样本 L2 范数并拼接到量子读出末维，
    补回因逐样本 L2 归一化而丢弃的笔画墨量标量信息。默认参数下量子权重 shape 与
    随机数消耗顺序不变，可与「只测 Z」基线做初始权重逐元素相同的受控对照。
    """

    def __init__(
        self,
        n_qubits: int,
        n_classes: int,
        n_layers: int = 2,
        encoding: str = "amplitude",
        observables: str = "z",
        n_features: int = 784,
        reupload: bool | str = False,
        head: str = "mlp",
        scale=1.0,
        hidden: int = 32,
        diff_method: str = "backprop",
        shots: int | None = None,
        trainable_reupload: bool = False,
        preserve_norm: bool = False,
        post_norm: bool = False,
        init_std: float | None = None,
    ):
        super().__init__()
        self.n_features = int(n_features)
        self.encoding = encoding
        self.preserve_norm = bool(preserve_norm)
        self.q = make_rich_layer(
            n_qubits=n_qubits, n_layers=n_layers, encoding=encoding,
            observables=observables, n_features=n_features,
            reupload=reupload, trainable_reupload=trainable_reupload,
            init_std=init_std, diff_method=diff_method, shots=shots,
        )
        self.q_readout_dim = n_readout(n_qubits, observables)
        self.readout_dim = self.q_readout_dim + (1 if self.preserve_norm else 0)
        self.head = _build_head(self.readout_dim, n_classes, head, hidden)
        _set_scale(self, scale)
        self.norm_bn = nn.BatchNorm1d(1) if self.preserve_norm else None
        self.post_ln = nn.BatchNorm1d(self.readout_dim) if post_norm else None

    def quantum_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取量子读出向量（若 ``preserve_norm=True`` 则含标准化后的样本范数列）。"""
        x = x.to(torch.float32)
        if x.dim() > 2:
            x = x.reshape(x.shape[0], -1)
        if self.preserve_norm:
            if x.shape[1] == self.n_features + 1:
                x_vec = x[:, :self.n_features]
                norms = x[:, self.n_features:]
            else:
                norms = x.norm(dim=1, keepdim=True)
                x_vec = x / norms.clamp_min(1e-12)
            z_q = self.q(x_vec).to(torch.float32)
            if self.norm_bn is not None and (norms.shape[0] > 1 or not self.training):
                n_feat = self.norm_bn(norms) * 0.2
            else:
                n_feat = norms * 0.2
            z = torch.cat([z_q, n_feat], dim=-1)
        else:
            z = self.q(x).to(torch.float32)
        if self.post_ln is not None and (z.shape[0] > 1 or not self.training):
            z = self.post_ln(z)
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.quantum_features(x)
        return self.head(z * self.out_scale)


class PatchVQCClassifier(nn.Module):
    """局部多 Patch 权重共享量子分类器（PatchVQC / Quanvolutional VQC）。

    解决全局 784 维振幅编码完全没有 2D 空间局部性与平移鲁棒性的结构性缺陷：
    将 28x28 图像按 ``patch_size`` 与 ``stride`` 切分为局部空间窗口（默认 14x14、
    步长 7，共 3x3=9 个局部重叠视野，每个窗口 196 维 <= 2^8=256），全部通过**同一个
    共享权重**的 8-qubit 振幅编码量子层提取局部二次型期望，并结合局部窗口相对范数加权
    与范数特征拼接，经 BatchNorm1d 标准化后送入分类头。
    """

    def __init__(
        self,
        n_qubits: int = 8,
        n_classes: int = 10,
        n_layers: int = 2,
        patch_size: int = 14,
        stride: int = 7,
        image_size: int = 28,
        observables: str = "z+x+y+zz",
        preserve_patch_norm: bool = True,
        init_std: float | None = None,
        head: str = "mlp",
        scale=1.0,
        hidden: int = 64,
        dropout: float = 0.1,
        diff_method: str = "backprop",
        shots: int | None = None,
    ):
        super().__init__()
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.stride = int(stride)
        self.patch_dim = self.patch_size * self.patch_size
        self.grid_size = (self.image_size - self.patch_size) // self.stride + 1
        self.n_patches = self.grid_size * self.grid_size
        self.preserve_patch_norm = bool(preserve_patch_norm)

        self.q = make_rich_layer(
            n_qubits=n_qubits,
            n_layers=n_layers,
            encoding="amplitude",
            observables=observables,
            n_features=self.patch_dim,
            init_std=init_std,
            diff_method=diff_method,
            shots=shots,
        )
        self.patch_readout_dim = n_readout(n_qubits, observables)
        per_patch_out = self.patch_readout_dim + (1 if self.preserve_patch_norm else 0)
        self.readout_dim = self.n_patches * per_patch_out
        self.post_norm = nn.BatchNorm1d(self.readout_dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = _build_head(self.readout_dim, n_classes, head, hidden)
        _set_scale(self, scale)

    def _to_image(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        side = self.image_size
        if x.dim() == 2 and x.shape[1] == side * side:
            return x.view(-1, 1, side, side)
        if x.dim() == 4 and x.shape[1] == 1 and x.shape[2] == side and x.shape[3] == side:
            return x
        raise ValueError(f"期望 (B,{side*side}) 或 (B,1,{side},{side})，实际 {tuple(x.shape)}")

    def quantum_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取所有局部空间 Patch 的量子读出拼接向量（经 BatchNorm1d 标准化）。"""
        img = self._to_image(x)
        B = img.shape[0]
        # unfold -> (B, patch_dim, n_patches) -> (B, n_patches, patch_dim)
        patches = F.unfold(img, kernel_size=self.patch_size, stride=self.stride).transpose(1, 2)
        p_norm = patches.norm(dim=-1, keepdim=True)  # (B, n_patches, 1)

        # 避免全零背景 patch 在 AmplitudeEmbedding(normalize=True) 时除零
        safe_patches = patches.clone()
        zero_mask = p_norm.squeeze(-1) < 1e-6
        if zero_mask.any():
            safe_patches[zero_mask, 0] = 1.0
        unit_patches = safe_patches / safe_patches.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        q_flat = self.q(unit_patches.reshape(B * self.n_patches, self.patch_dim)).to(torch.float32)
        q_patches = q_flat.reshape(B, self.n_patches, self.patch_readout_dim)

        # 按样本内各 patch 的相对能量加权量子期望，抑制空白角落背景噪声被归一化放大
        rel_norm = p_norm / p_norm.mean(dim=1, keepdim=True).clamp_min(1e-6)
        q_weighted = q_patches * rel_norm
        if self.preserve_patch_norm:
            feat = torch.cat([q_weighted, p_norm], dim=-1)
        else:
            feat = q_weighted
        flat_feat = feat.reshape(B, self.readout_dim)
        if B > 1 or not self.training:
            flat_feat = self.post_norm(flat_feat)
        return flat_feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.quantum_features(x)
        return self.head(self.drop(z * self.out_scale))


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
        self.readout_dim = int(n_qubits)
        self.post_norm = None
        self.head = _build_head(n_qubits, n_classes, head, hidden)
        _set_scale(self, scale)

    def pre_features(self, x: torch.Tensor) -> torch.Tensor:
        """前端输出，乘 π/2 后落在 [-π/2, π/2]（避开 RY 的 cos 饱和区）。"""
        f = self.pre_net(self.backbone(_as_images(x)))
        return f * math.pi / 2.0

    def quantum_features(self, x: torch.Tensor) -> torch.Tensor:
        """量子读出向量 z；``quantum=False`` 时返回前端特征本身。"""
        f = self.pre_features(x)
        if self.q is None:
            return f
        z = self.q(f).to(torch.float32)
        if getattr(self, "post_norm", None) is not None and (z.shape[0] > 1 or not self.training):
            z = self.post_norm(z)
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.quantum_features(x)
        return self.head(z * self.out_scale)


class HybridRichClassifier(HybridCNNClassifier):
    """CNN 前端 → **多可观测量 / 重上传 / 稳定化**量子头 → 经典非线性头。

    在 :class:`HybridCNNClassifier` 之上换掉量子层，用来检验「解开 n 个有界
    读数」在混合架构里是否同样有效。``quantum=False`` 时跳过量子层，head
    自动改接前端的 n_qubits 维特征，仍是等架构的经典对照。

    ``stabilize=True`` 时启用纯串行稳定化量子瓶颈（无经典旁路跳过连接）：
      1. ``pre_net`` 采用 ``BatchNorm1d(width) -> Linear(width, n_qubits, bias=False) -> BatchNorm1d(n_qubits) -> Tanh() * (pi/2)``，
         消除轻量 CNN 自适应池化后样本间标准差仅 ~0.003 而被随机偏置掩盖导致的常值坍缩；
      2. 量子层采用受控小方差初始化 ``N(0, 0.1^2)``，避免第 0 轮均匀随机 ``[0, 2pi]`` 旋转彻底打乱 CNN 特征；
      3. 量子读出后接 ``BatchNorm1d(readout_dim)`` 逐通道按批标准化各 Pauli 观测量尺度。
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
        frontend: str = "light",
        observables: str = "z",
        reupload: bool | str = False,
        diff_method: str = "backprop",
        trainable_reupload: bool = False,
        stabilize: bool = False,
        init_std: float | None = None,
    ):
        # 不调用 super().__init__()：量子层与 head 维度都要按新规格重建
        nn.Module.__init__(self)
        self.stabilize = bool(stabilize)
        self.backbone, width = _cnn_backbone(frontend)
        if self.stabilize:
            self.pre_net = nn.Sequential(
                nn.BatchNorm1d(width),
                nn.Linear(width, n_qubits, bias=False),
                nn.BatchNorm1d(n_qubits),
                nn.Tanh(),
            )
            eff_init_std = 0.1 if init_std is None else float(init_std)
        else:
            self.pre_net = nn.Sequential(nn.Linear(width, n_qubits), nn.Tanh())
            eff_init_std = init_std

        self.q = (make_rich_layer(
            n_qubits=n_qubits, n_layers=n_layers, encoding="angle",
            observables=observables, reupload=reupload,
            trainable_reupload=trainable_reupload, init_std=eff_init_std,
            diff_method=diff_method) if quantum else None)
        dim = (n_readout(n_qubits, observables) if quantum else n_qubits)
        self.readout_dim = dim
        self.post_norm = nn.BatchNorm1d(dim) if (self.stabilize and quantum) else None
        self.head = _build_head(dim, n_classes, head, hidden)
        _set_scale(self, scale)

    def pre_features(self, x: torch.Tensor) -> torch.Tensor:
        """前端输出，乘 π/2 后落在 [-π/2, π/2]；单样本训练步自动跳过 BatchNorm1d。"""
        h = self.backbone(_as_images(x))
        if self.stabilize and h.shape[0] == 1 and self.training:
            # 单样本 batch 下临时以 eval 统计量通过 pre_net 中的 BatchNorm1d
            was_training = self.pre_net.training
            self.pre_net.eval()
            f = self.pre_net(h)
            self.pre_net.train(was_training)
        else:
            f = self.pre_net(h)
        return f * math.pi / 2.0
