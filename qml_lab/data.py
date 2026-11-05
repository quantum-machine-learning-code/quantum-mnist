"""qml_lab 数据：MNIST（真实 / 合成）与字符级语言模型语料。

真实 MNIST 依次尝试 torchvision -> sklearn OpenML，任一步失败都打印原因并回退，
最终兜底为高斯合成数据（仅用于离线冒烟测试）。路径一律用 pathlib。
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import torch

__all__ = ["load_mnist", "load_corpus", "build_char_data", "iter_batches"]

# 数据源别名，方便其它模块用不同名字指定
_SOURCE_ALIASES = {
    "auto": "auto",
    "idx": "idx", "local": "idx", "raw": "idx", "mnist-raw": "idx",
    "torchvision": "torchvision", "torch": "torchvision", "mnist": "torchvision",
    "openml": "openml", "sklearn": "openml", "fetch_openml": "openml",
    "synthetic": "synthetic", "synth": "synthetic", "fake": "synthetic",
}

# 自写英文语料：技术短句重复若干遍，无版权问题，含重复结构且长度 >= 800
_CORPUS_BLOCK = (
    "A qubit can hold a superposition of zero and one until we measure it. "
    "We rotate the state with a small angle and then read the result again. "
    "The circuit learns a pattern by tuning each gate one step at a time. "
    "Gradients flow back through the layer and update the weights slowly. "
    "Noise makes the signal weak but the model still finds a simple rule. "
    "We repeat the same short sentence to give the tokenizer enough data. "
)
_BUILTIN_CORPUS = _CORPUS_BLOCK * 4


def _normalize_source(source) -> str:
    return _SOURCE_ALIASES.get(str(source).strip().lower(), "auto")


def _split_counts(total: int, n_classes: int) -> list[int]:
    """把总样本数按类平均分配，余数摊到前几类，总和严格等于 total。"""
    base, rem = divmod(int(total), n_classes)
    return [base + (1 if i < rem else 0) for i in range(n_classes)]


def _to_labels(y) -> np.ndarray:
    a = np.asarray(y)
    if a.dtype.kind in "iu":
        return a.astype(np.int64)
    return a.astype(str).astype(np.int64)


def _read_idx_images(path) -> np.ndarray:
    """直接解析 IDX3 图像文件，返回 (N, 784) 的 uint8 数组。"""
    with open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"非法 IDX3 文件 {path}: magic={magic}")
        return np.fromfile(f, dtype=np.uint8, count=n * rows * cols).reshape(n, rows * cols)


def _read_idx_labels(path) -> np.ndarray:
    """直接解析 IDX1 标签文件，返回 (N,) 的 uint8 数组。"""
    with open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"非法 IDX1 文件 {path}: magic={magic}")
        return np.fromfile(f, dtype=np.uint8, count=n)


def _fetch_raw(src: str, data_dir) -> tuple:
    """取真实数据源原始 (Xtr, ytr, Xte, yte)，失败时抛异常由上层回退。"""
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    if src == "idx":
        # 直接读本地 MNIST IDX 文件，完全离线；避免 torchvision 在线下载卡住
        raw = root / "MNIST" / "raw"
        Xtr = _read_idx_images(raw / "train-images-idx3-ubyte")
        ytr = _read_idx_labels(raw / "train-labels-idx1-ubyte")
        Xte = _read_idx_images(raw / "t10k-images-idx3-ubyte")
        yte = _read_idx_labels(raw / "t10k-labels-idx1-ubyte")
    elif src == "torchvision":
        # 延迟导入：ImportError / 网络失败都会被上层捕获
        from torchvision import datasets
        tr = datasets.MNIST(root=str(root), train=True, download=True)
        te = datasets.MNIST(root=str(root), train=False, download=True)
        Xtr = np.asarray(tr.data.numpy()).reshape(len(tr.data), -1)
        Xte = np.asarray(te.data.numpy()).reshape(len(te.data), -1)
        ytr = np.asarray(tr.targets.numpy())
        yte = np.asarray(te.targets.numpy())
    else:  # openml
        from sklearn.datasets import fetch_openml
        b = fetch_openml("mnist_784", version=1, as_frame=False, data_home=str(root))
        X = np.asarray(b.data)
        y = np.asarray(b.target)
        # OpenML 官方顺序：前 60000 训练，后 10000 测试
        Xtr, ytr = X[:60000], y[:60000]
        Xte, yte = X[60000:], y[60000:]
    return (
        np.asarray(Xtr, dtype=np.float32),
        _to_labels(ytr),
        np.asarray(Xte, dtype=np.float32),
        _to_labels(yte),
    )


def _select(X, y, counts, class_ids, rng) -> tuple[np.ndarray, np.ndarray]:
    """按类均衡抽样，并把原始类别标签重映射为连续的 0..C-1。"""
    Xs, ys = [], []
    X = np.asarray(X)
    for new_lab, (orig, cnt) in enumerate(zip(class_ids, counts)):
        pool = np.flatnonzero(y == orig)
        if pool.size == 0:
            # 指定类别不存在时从全集取样兜底，避免抛异常
            pool = np.arange(len(y))
        take = rng.choice(pool, size=cnt, replace=pool.size < cnt)
        Xs.append(X[take])
        ys.append(np.full(cnt, new_lab, dtype=np.int64))
    return (
        np.concatenate(Xs, axis=0).astype(np.float32),
        np.concatenate(ys).astype(np.int64),
    )


def _apply_pca(Xtr, Xte, n_features, seed) -> tuple[np.ndarray, np.ndarray]:
    """PCA 仅在训练集上 fit，再同时 transform 训练 / 测试。"""
    from sklearn.decomposition import PCA
    k = min(int(n_features), Xtr.shape[1], Xtr.shape[0])
    pca = PCA(n_components=k, random_state=seed)
    A = pca.fit_transform(Xtr).astype(np.float32)
    B = pca.transform(Xte).astype(np.float32)
    if k < n_features:
        # 维度不足时补零，保证输出严格为 n_features 维
        A = np.pad(A, ((0, 0), (0, n_features - k)))
        B = np.pad(B, ((0, 0), (0, n_features - k)))
    return A, B


def _scale_to_pi(Xtr, Xte) -> tuple[np.ndarray, np.ndarray]:
    """用训练集全局 min/max 把训练与测试一起缩放到 [-pi, pi] 并 clip。"""
    lo = float(np.asarray(Xtr).min())
    hi = float(np.asarray(Xtr).max())
    if hi - lo < 1e-12:
        hi = lo + 1.0
    span = hi - lo

    def _s(X):
        Z = (X.astype(np.float32) - lo) / span * (2.0 * np.pi) - np.pi
        return np.clip(Z, -np.pi, np.pi).astype(np.float32)

    return _s(Xtr), _s(Xte)


def _make_synthetic(counts_tr, counts_te, n_features, n_classes, seed) -> tuple:
    """每类一个高斯中心 + 噪声的可分数据，维度直接是 n_features（无需 PCA）。"""
    rng = np.random.default_rng(seed)
    centers = rng.normal(0.0, 1.0, size=(n_classes, n_features)).astype(np.float32)
    # 沿第 0 维拉开中心，保证类别线性可分
    centers[:, 0] = np.arange(n_classes, dtype=np.float32) * 4.0 - (n_classes - 1) * 2.0

    def _gen(counts):
        Xs, ys = [], []
        for lab, cnt in enumerate(counts):
            X = centers[lab] + rng.normal(0.0, 0.35, size=(cnt, n_features))
            Xs.append(X.astype(np.float32))
            ys.append(np.full(cnt, lab, dtype=np.int64))
        return np.concatenate(Xs, axis=0), np.concatenate(ys)

    Xtr, ytr = _gen(counts_tr)
    Xte, yte = _gen(counts_te)
    return Xtr, ytr, Xte, yte


def load_mnist(n_train=2000, n_test=500, n_features=4, classes=10,
               source="auto", data_dir="data", seed=0) -> tuple:
    """加载 MNIST 子集，返回 (X_train, y_train, X_test, y_test, meta)。

    X_* 为 float32、shape (N, n_features)、取值 [-pi, pi]；y_* 为 int64、标签 0..C-1。
    source='auto' 依次尝试 torchvision -> openml -> synthetic，绝不抛异常。
    """
    n_features = max(1, int(n_features))
    if isinstance(classes, (int, np.integer)):
        class_ids = list(range(int(classes)))
    else:
        class_ids = [int(c) for c in classes]
    if len(class_ids) == 0:
        class_ids = [0]
    C = len(class_ids)
    counts_tr = _split_counts(n_train, C)
    counts_te = _split_counts(n_test, C)

    src = _normalize_source(source)
    if src == "auto":
        # 本地 IDX 优先：完全离线、最快，也不会卡在在线下载
        chain = ["idx", "torchvision", "openml"]
    elif src == "synthetic":
        chain = []
    else:
        chain = [src]

    Xtr = ytr = Xte = yte = None
    used = "synthetic"
    reasons: list[str] = []
    for candidate in chain:
        try:
            Xr, yr, Xe, ye = _fetch_raw(candidate, data_dir)
            r = np.random.default_rng(seed)
            Xtr, ytr = _select(Xr, yr, counts_tr, class_ids, r)
            Xte, yte = _select(Xe, ye, counts_te, class_ids, r)
            Xtr, Xte = _apply_pca(Xtr, Xte, n_features, seed)
            used = candidate
            break
        except Exception as e:  # 含 ImportError / 网络失败 / 维度问题
            msg = f"{candidate}: {type(e).__name__}: {e}"
            reasons.append(msg)
            print(f"[data] source failed: {msg}; falling back")

    if used == "synthetic":
        print("[data] using synthetic data (offline smoke test only)")
        Xtr, ytr, Xte, yte = _make_synthetic(counts_tr, counts_te, n_features, C, seed)

    Xtr, Xte = _scale_to_pi(Xtr, Xte)
    Xtr = np.ascontiguousarray(Xtr, dtype=np.float32)
    Xte = np.ascontiguousarray(Xte, dtype=np.float32)
    ytr = np.ascontiguousarray(ytr, dtype=np.int64)
    yte = np.ascontiguousarray(yte, dtype=np.int64)

    if used == "synthetic":
        note = "SYNTHETIC - not real MNIST，仅用于离线冒烟测试"
    else:
        note = (
            f"real MNIST via {used}; balanced sampling; "
            f"PCA({n_features}) fit on train only; scaled to [-pi, pi]"
        )
    meta = {
        "source": used,
        "n_classes": C,
        "n_features": n_features,
        "class_ids": list(class_ids),
        "note": note,
        "n_train": int(Xtr.shape[0]),
        "n_test": int(Xte.shape[0]),
        "seed": int(seed),
        "pca": used != "synthetic",
        "fallback_reasons": reasons,
    }
    return Xtr, ytr, Xte, yte, meta


def load_corpus(path=None) -> str:
    """有 path 就读文件；否则返回内置自写英文语料（长度 >= 800）。"""
    if path is not None:
        return Path(path).read_text(encoding="utf-8")
    return _BUILTIN_CORPUS


def build_char_data(text) -> tuple[torch.Tensor, int, dict, dict]:
    """字符级编码：返回 (long 一维张量, vocab_size, stoi, itos)。

    字符表用 sorted(set(text))；itos 为 {int: str} 映射。
    """
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    return data, len(chars), stoi, itos


def iter_batches(data_tensor, seq_len, batch_size, seed):
    """无限生成器：每次 yield (x, y)，shape 均为 (batch_size, seq_len)。

    y 为 x 右移一位；用 torch.Generator().manual_seed(seed) 保证可复现。
    """
    data = data_tensor.reshape(-1).long()
    if data.numel() < seq_len + 2:
        # 语料过短时循环拼接，保证能取出等长片段
        data = data.repeat((seq_len + 2) // data.numel() + 2)
    g = torch.Generator().manual_seed(int(seed))
    n = data.numel()
    max_start = n - seq_len - 1  # y 需要多取一位
    while True:
        starts = torch.randint(0, max_start, (batch_size,), generator=g).tolist()
        x = torch.stack([data[s:s + seq_len] for s in starts])
        y = torch.stack([data[s + 1:s + 1 + seq_len] for s in starts])
        yield x, y
