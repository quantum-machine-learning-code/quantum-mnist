"""VQC 量子分类器、振幅编码分类器、局部多 Patch 量子分类器、混合 CNN 前端与经典基线。"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

try:
    from .quantum import (make_amplitude_layer, make_rich_layer,
                          make_vqc_layer, n_quantum_params, n_readout)
except ImportError:
    from quantum import (make_amplitude_layer, make_rich_layer,
                         make_vqc_layer, n_quantum_params, n_readout)


def parameter_breakdown(model: nn.Module, preprocessing_params: int = 0) -> dict:
    """显式拆解模型的量子参数、经典封装参数与离线预处理拟合参数（如 PCA）。

    任何标记 ``_is_quantum_module = True`` 的子模块（或 ``model.q``）内的参数均计入
    ``quantum_params``；其余模型参数计入 ``classical_wrapper_params``；
    ``preprocessing_params`` 显式记录 PCA（``784 + k*784``）等未计入 PyTorch 模块的拟合参数。
    """
    q_ids: set[int] = set()
    for mod in model.modules():
        if getattr(mod, "_is_quantum_module", False):
            for p in mod.parameters():
                q_ids.add(id(p))
    q_sub = getattr(model, "q", None)
    if q_sub is not None and isinstance(q_sub, nn.Module):
        for p in q_sub.parameters():
            q_ids.add(id(p))

    seen: set[int] = set()
    q_params = 0
    total_params = 0
    for p in model.parameters():
        pid = id(p)
        if pid in seen:
            continue
        seen.add(pid)
        n = int(p.numel())
        total_params += n
        if pid in q_ids:
            q_params += n

    c_params = total_params - q_params
    prep = int(preprocessing_params)
    return {
        "total_model_params": total_params,
        "quantum_params": q_params,
        "classical_wrapper_params": c_params,
        "preprocessing_fitted_params": prep,
        "effective_total_params": total_params + prep,
        "quantum_param_share": round(q_params / total_params, 6) if total_params else 0.0,
    }


def assert_matched_config(
    cfg_a: dict,
    cfg_b: dict,
    keys: tuple[str, ...],
    label_a: str = "A",
    label_b: str = "B",
) -> None:
    """受控实验契约断言：确认两个对照臂在全部指定控制变量上完全一致。"""
    mismatches = []
    for k in keys:
        if k not in cfg_a or k not in cfg_b:
            mismatches.append(f"{k}: missing ({k in cfg_a} vs {k in cfg_b})")
        elif cfg_a[k] != cfg_b[k]:
            mismatches.append(f"{k}: {label_a}={cfg_a[k]!r} != {label_b}={cfg_b[k]!r}")
    if mismatches:
        raise AssertionError(
            f"受控变量不匹配（{label_a} vs {label_b}）：" + "; ".join(mismatches)
        )


class _ButterflyGivens(nn.Module):
    """固定维度上的纯正交 Givens 旋转混合层，对任意正整数 ``n_params`` 精确控制参数量。"""

    def __init__(self, dim: int, n_params: int):
        super().__init__()
        if dim < 2:
            raise ValueError(f"dim 必须 >= 2，实际 {dim}")
        pairs_per_layer = dim // 2
        self.dim = int(dim)
        self.n_params = int(n_params)
        self.angles = nn.Parameter(torch.randn(self.n_params) * 0.1)
        layer_specs: list[tuple[int, int]] = []
        idx_a_list: list[list[int]] = []
        idx_b_list: list[list[int]] = []
        rem = self.n_params
        layer_i = 0
        offset = 0
        while rem > 0:
            take = min(pairs_per_layer, rem)
            shift = layer_i % dim
            order = [(i + shift) % dim for i in range(dim)]
            idx_a_list.append(order[0:2 * take:2])
            idx_b_list.append(order[1:2 * take:2])
            layer_specs.append((offset, offset + take))
            offset += take
            rem -= take
            layer_i += 1
        self._layer_specs = layer_specs
        for i, (ia, ib) in enumerate(zip(idx_a_list, idx_b_list)):
            self.register_buffer(f"idx_a_{i}", torch.tensor(ia, dtype=torch.long))
            self.register_buffer(f"idx_b_{i}", torch.tensor(ib, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for i, (lo, hi) in enumerate(self._layer_specs):
            ia = getattr(self, f"idx_a_{i}")
            ib = getattr(self, f"idx_b_{i}")
            ang = self.angles[lo:hi]
            c = torch.cos(ang)
            s = torch.sin(ang)
            xa = h[..., ia]
            xb = h[..., ib]
            out = h.clone()
            out[..., ia] = c * xa - s * xb
            out[..., ib] = s * xa + c * xb
            h = out
        return h


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
        self.post_bn = nn.BatchNorm1d(self.readout_dim) if post_norm else None

    @property
    def post_ln(self) -> nn.Module | None:
        """向后兼容别名；实际模块为 ``BatchNorm1d``（详见 ``post_bn``）。"""
        return self.post_bn

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
        if self.post_bn is not None and (z.shape[0] > 1 or not self.training):
            z = self.post_bn(z)
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
        match_readout_dim: bool = False,
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

        # pre_features 已做 Tanh() * (pi/2) 有界映射，因此显式传 bound_inputs=False
        # 避免与 trainable_reupload 叠加产生 tanh(tanh(u)) 双重压缩
        self.q = (make_rich_layer(
            n_qubits=n_qubits, n_layers=n_layers, encoding="angle",
            observables=observables, reupload=reupload,
            trainable_reupload=trainable_reupload, init_std=eff_init_std,
            diff_method=diff_method, bound_inputs=False) if quantum else None)

        target_dim = n_readout(n_qubits, observables)
        self.classical_bridge: nn.Module | None = None
        if quantum:
            dim = target_dim
        elif match_readout_dim and target_dim != n_qubits:
            dim = target_dim
            # 等宽且近似等参的经典桥接层：固定随机正交展开(0参数) + 等参数量 Givens 旋转
            q_equiv = n_quantum_params(
                n_qubits, n_layers,
                trainable_reupload=bool(trainable_reupload) or (
                    isinstance(reupload, str) and reupload.strip().lower() == "trainable"
                ),
            )
            proj = torch.randn(n_qubits, dim) / math.sqrt(n_qubits)
            self.register_buffer("_bridge_proj", proj)
            self.classical_bridge = _ButterflyGivens(dim, q_equiv)
        else:
            dim = n_qubits
            self._bridge_proj = None

        self.readout_dim = dim
        # 修复：quantum=False 且 stabilize=True 时必须保留 post_norm，否则经典对照缺失 BN
        self.post_norm = nn.BatchNorm1d(dim) if self.stabilize else None
        self.head = _build_head(dim, n_classes, head, hidden)
        _set_scale(self, scale)

    def quantum_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取读出特征向量；``quantum=False`` 时走等宽/等 BN 的经典对照路径。"""
        f = self.pre_features(x)
        if self.q is not None:
            z = self.q(f).to(torch.float32)
        elif self.classical_bridge is not None:
            if getattr(self, "_bridge_proj", None) is not None:
                z = torch.tanh(self.classical_bridge(f @ self._bridge_proj))
            else:
                z = torch.tanh(self.classical_bridge(f))
        else:
            z = f
        if self.post_norm is not None and (z.shape[0] > 1 or not self.training):
            z = self.post_norm(z)
        return z

    def pre_features(self, x: torch.Tensor) -> torch.Tensor:
        """前端输出，乘 π/2 后落在 [-π/2, π/2]；单样本训练步自动跳过 BatchNorm1d。"""
        h = self.backbone(_as_images(x))
        if self.stabilize and h.shape[0] == 1 and self.training:
            # 单样本 batch 下临时以 eval 统计量通过 pre_net 中的 BatchNorm1d
            was_training = self.pre_net.training
            try:
                self.pre_net.eval()
                f = self.pre_net(h)
            finally:
                self.pre_net.train(was_training)
        else:
            f = self.pre_net(h)
        return f * math.pi / 2.0


class RandomQuadraticClassifier(nn.Module):
    """Regime I 去量子化对照：0 个学习投影参数的满秩随机齐次二次型 + 与 RichVQC 完全相同的头。

    使用 3 组固定随机正交基 ``Q_b in O(d)``（对应 Z/X/Y 3 个 QWC 测量基）与固定随机 ``±1`` 符号向量
    ``s_j in {-1, +1}^d`` 计算 ``z_j(x) = sum_i s_{j,i} (Q_{b(j)} x)_i^2 = x^T (Q^T diag(s_j) Q) x``：
    与振幅编码 VQC 的 ``<x|U† P_j U|x>`` 严格同属特征值 ``±1``、取值 ``[-1, 1]`` 的满秩齐次二次型函数类，
    且无任何数据拟合投影参数。
    """

    def __init__(
        self,
        n_features: int = 784,
        readout_dim: int = 40,
        n_classes: int = 10,
        n_bases: int = 3,
        head: str = "mlp",
        scale=5.0,
        hidden: int = 32,
    ):
        super().__init__()
        self.n_features = int(n_features)
        self.readout_dim = int(readout_dim)
        self.n_bases = int(n_bases)
        # 3 组固定随机正交变换 + 固定随机 ±1 符号掩码（buffer，0 个可训练参数）
        bases = []
        for _ in range(self.n_bases):
            q_mat, _ = torch.linalg.qr(torch.randn(self.n_features, self.n_features))
            bases.append(q_mat.to(torch.float32))
        self.register_buffer("orth_bases", torch.stack(bases, dim=0))  # (B_g, d, d)
        signs = torch.where(
            torch.rand(self.readout_dim, self.n_features) < 0.5,
            torch.ones(self.readout_dim, self.n_features),
            -torch.ones(self.readout_dim, self.n_features),
        )
        self.register_buffer("signs", signs.to(torch.float32))
        self.register_buffer(
            "basis_idx",
            torch.arange(self.readout_dim, dtype=torch.long) % self.n_bases,
        )
        self.head = _build_head(self.readout_dim, n_classes, head, hidden)
        _set_scale(self, scale)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.dim() > 2:
            x = x.reshape(x.shape[0], -1)
        # 按 3 个正交基分别计算 (Q_b x)^2，再用 ±1 掩码收缩成 [-1, 1] 内的二次型期望
        outs = []
        for b_i in range(self.n_bases):
            mask = self.basis_idx == b_i
            if not mask.any():
                continue
            rot_sq = (x @ self.orth_bases[b_i]).pow(2)  # (N, d)
            outs.append((mask, rot_sq @ self.signs[mask].T))
        z = torch.empty(x.shape[0], self.readout_dim, dtype=torch.float32, device=x.device)
        for mask, val in outs:
            z[:, mask] = val
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x) * self.out_scale)


class PatchLowRankClassifier(nn.Module):
    """Regime I 严格等参对照：与 PatchVQCClassifier 封装逐项一致，仅将 48 参数 8q 线路
    替换为 48 个可训练参数的经典局部 patch 滤波器（固定零参随机展开 + 48 参数 Givens 正交混合）。
    """

    def __init__(
        self,
        patch_size: int = 14,
        stride: int = 7,
        image_size: int = 28,
        n_classes: int = 10,
        per_patch_out: int = 32,
        filter_params: int = 48,
        preserve_patch_norm: bool = True,
        head: str = "mlp",
        scale=1.0,
        hidden: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.stride = int(stride)
        self.patch_dim = self.patch_size * self.patch_size
        self.grid_size = (self.image_size - self.patch_size) // self.stride + 1
        self.n_patches = self.grid_size * self.grid_size
        self.per_patch_out = int(per_patch_out)
        self.preserve_patch_norm = bool(preserve_patch_norm)

        # 固定零参正交基（对应振幅编码 0 参数展开）
        q_mat, _ = torch.linalg.qr(torch.randn(self.patch_dim, self.per_patch_out))
        self.register_buffer("fixed_basis", q_mat.to(torch.float32))
        self.filter = _ButterflyGivens(self.per_patch_out, filter_params)

        per_patch = self.per_patch_out + (1 if self.preserve_patch_norm else 0)
        self.readout_dim = self.n_patches * per_patch
        self.post_norm = nn.BatchNorm1d(self.readout_dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = _build_head(self.readout_dim, n_classes, head, hidden)
        _set_scale(self, scale)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        img = x.to(torch.float32)
        side = self.image_size
        if img.dim() == 2 and img.shape[1] == side * side:
            img = img.view(-1, 1, side, side)
        B = img.shape[0]
        patches = F.unfold(img, kernel_size=self.patch_size, stride=self.stride).transpose(1, 2)
        p_norm = patches.norm(dim=-1, keepdim=True)
        safe = patches.clone()
        zero_mask = p_norm.squeeze(-1) < 1e-6
        if zero_mask.any():
            safe[zero_mask, 0] = 1.0
        unit = safe / safe.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        flat_u = unit.reshape(B * self.n_patches, self.patch_dim)
        h = torch.tanh(self.filter(flat_u @ self.fixed_basis) * math.sqrt(self.patch_dim / self.per_patch_out))
        h = h.reshape(B, self.n_patches, self.per_patch_out)
        rel_norm = p_norm / p_norm.mean(dim=1, keepdim=True).clamp_min(1e-6)
        hw = h * rel_norm
        feat = torch.cat([hw, p_norm], dim=-1) if self.preserve_patch_norm else hw
        flat_feat = feat.reshape(B, self.readout_dim)
        if B > 1 or not self.training:
            flat_feat = self.post_norm(flat_feat)
        return flat_feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.drop(self.features(x) * self.out_scale))


class _QuantumReuploadCore(nn.Module):
    """纯量子 Pérez-Salinas 深度重上传线路内核（零经典 MLP 封装，纯 Born 概率读出）。"""

    _is_quantum_module = True

    def __init__(self, n_qubits: int = 4, n_layers: int = 6):
        super().__init__()
        import pennylane as qml

        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        self.in_features = 3 * self.n_qubits
        dev = qml.device("default.qubit", wires=self.n_qubits)

        def circuit(inputs, in_scale, in_shift):
            for layer_i in range(self.n_layers):
                for q in range(self.n_qubits):
                    phi = inputs[..., 3 * q] * in_scale[layer_i, q, 0] + in_shift[layer_i, q, 0]
                    theta = inputs[..., 3 * q + 1] * in_scale[layer_i, q, 1] + in_shift[layer_i, q, 1]
                    omega = inputs[..., 3 * q + 2] * in_scale[layer_i, q, 2] + in_shift[layer_i, q, 2]
                    qml.Rot(phi, theta, omega, wires=q)
                for q in range(self.n_qubits):
                    qml.CNOT(wires=[q, (q + 1) % self.n_qubits])
            return qml.probs(wires=range(self.n_qubits))

        weight_shapes = {
            "in_scale": (self.n_layers, self.n_qubits, 3),
            "in_shift": (self.n_layers, self.n_qubits, 3),
        }
        qnode = qml.qnode(dev, diff_method="backprop")(circuit)
        self.layer = qml.qnn.TorchLayer(qnode, weight_shapes).to(torch.float32)
        with torch.no_grad():
            nn.init.normal_(self.layer.in_scale, mean=0.0, std=0.25)
            nn.init.normal_(self.layer.in_shift, mean=0.0, std=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = self.layer(x.to(torch.float32)).to(torch.float32)
        if probs.dim() == 2 and probs.shape[0] != x.shape[0] and probs.shape[1] == x.shape[0]:
            probs = probs.transpose(0, 1)
        return probs


class DataReuploadingFidelityClassifier(nn.Module):
    """Regime III 量子原生分类器（无经典 MLP 封装）：深度 Pérez-Salinas 重上传 + 纯 Born 规则读出。

    在 ``n_qubits=4``（``2^4=16 >= 10`` 类）上每层用通用 SU(2) 门 ``Rot(w*x + b)``
    重上传 ``3*n_qubits=12`` 维特征并施加环状 CNOT 纠缠；直接取前 ``n_classes`` 个正交计算基态
    的 Born 概率 ``p_c(x) = |<c|psi_theta(x)>|^2`` 计算对数概率 logits（0 个经典参数）。
    """

    def __init__(self, n_qubits: int = 4, n_layers: int = 4, n_classes: int = 10):
        super().__init__()
        if 2 ** n_qubits < n_classes:
            raise ValueError(f"2**{n_qubits} < n_classes={n_classes}")
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        self.n_classes = int(n_classes)
        self.q = _QuantumReuploadCore(n_qubits=n_qubits, n_layers=n_layers)
        self.readout_dim = int(n_classes)

    def born_probs(self, x: torch.Tensor) -> torch.Tensor:
        """返回完整 ``2^n`` 维 Born 概率分布（归一化，和为 1）。"""
        return self.q(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = self.born_probs(x)[:, :self.n_classes]
        return torch.log(probs.clamp_min(1e-7))


class ClassicalTrigNet(nn.Module):
    """Regime I 匹配对照：与深度数据重上传等参数量、等三角傅里叶函数类的经典网络。"""

    def __init__(
        self,
        n_features: int = 12,
        n_classes: int = 10,
        n_freqs: int = 3,
        bias: bool = False,
    ):
        super().__init__()
        self.freq = nn.Linear(n_features, n_freqs, bias=bias)
        self.head = nn.Linear(2 * n_freqs, n_classes, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ang = self.freq(x.to(torch.float32))
        feat = torch.cat([torch.cos(ang), torch.sin(ang)], dim=-1)
        return self.head(feat)
