"""分类器训练循环：分批评估准确率与 mini-batch 训练。"""
from __future__ import annotations

import math
import time

import torch

try:
    from .utils import set_seed
except ImportError:
    from utils import set_seed


def _as_float_tensor(X):
    """输入转浮点张量；已是浮点张量则保留 dtype，避免与量子层权重精度冲突。"""
    t = X if isinstance(X, torch.Tensor) else torch.as_tensor(X)
    if not torch.is_floating_point(t):
        t = t.float()
    return t


def _as_long_tensor(y):
    """标签转一维 long 张量，供 CrossEntropyLoss 使用。"""
    t = y if isinstance(y, torch.Tensor) else torch.as_tensor(y)
    return t.long().reshape(-1)


def _finite(value, label: str) -> float:
    """保证记录值是有限 Python float；非有限值打印警告并回退为 0.0。"""
    f = float(value)
    if not math.isfinite(f):
        print(f"[警告] {label} 出现非有限值 {f}，已回退为 0.0")
        return 0.0
    return f


def evaluate_accuracy(model, X, y, batch_size: int = 64) -> float:
    """分批前向计算准确率，避免一次性把大 batch 塞给量子电路爆内存。"""
    X = _as_float_tensor(X)
    y = _as_long_tensor(y)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for start in range(0, y.shape[0], batch_size):
            logits = model(X[start:start + batch_size])
            preds = logits.argmax(dim=-1)
            yb = y[start:start + batch_size]
            correct += (preds == yb).sum().item()
            total += yb.numel()
    return correct / total if total else 0.0


def train_classifier(model, X_train, y_train, X_test, y_test, epochs: int = 20,
                     batch_size: int = 32, lr: float = 0.01, seed: int = 0,
                     eval_batch_size: int = 64, verbose: bool = True,
                     eval_train: bool = True, quantum_lr: float | None = None,
                     weight_decay: float = 0.0, cosine_lr: bool = False) -> dict:
    """训练分类器。

    ``eval_train=False`` 时跳过每 epoch 在**训练集**上的全量 loss / acc 前向：
    状态向量模拟下 2^n 的前向很贵，多 seed 实验只需测试集曲线。此时
    ``train_loss`` 改用训练时 mini-batch loss 的加权平均，``train_acc`` 记为
    ``None``（不测量就不填数字），``test_acc`` 仍每 epoch 计算。默认 ``True``
    与原行为完全一致，主实验脚本不受影响。

    ``quantum_lr`` 不为 None 时，把模型 ``.q`` 子模块的参数单独分一组用该
    学习率，其余（CNN 前端 / pre_net / head / scale）用 ``lr``。
    """
    set_seed(seed)
    Xtr, ytr = _as_float_tensor(X_train), _as_long_tensor(y_train)
    Xte, yte = _as_float_tensor(X_test), _as_long_tensor(y_test)

    if quantum_lr is not None and getattr(model, "q", None) is not None:
        qids = {id(p) for p in model.q.parameters()}
        qparams = [p for p in model.parameters() if id(p) in qids]
        cparams = [p for p in model.parameters() if id(p) not in qids]
        opt = torch.optim.Adam(
            [{"params": qparams, "lr": quantum_lr, "weight_decay": 0.0},
             {"params": cparams, "lr": lr, "weight_decay": weight_decay}],
        )
    elif weight_decay > 0.0 and getattr(model, "q", None) is not None:
        qids = {id(p) for p in model.q.parameters()}
        qparams = [p for p in model.parameters() if id(p) in qids]
        cparams = [p for p in model.parameters() if id(p) not in qids]
        opt = torch.optim.Adam(
            [{"params": qparams, "lr": lr, "weight_decay": 0.0},
             {"params": cparams, "lr": lr, "weight_decay": weight_decay}],
        )
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs), eta_min=lr * 0.1)
             if cosine_lr else None)
    loss_fn = torch.nn.CrossEntropyLoss()
    history = {"train_loss": [], "train_acc": [], "test_acc": [], "epoch_seconds": []}
    n = ytr.shape[0]
    t0 = time.perf_counter()

    for ep in range(1, epochs + 1):
        ep_t0 = time.perf_counter()
        model.train()
        perm = torch.randperm(n)  # 每 epoch 打乱做 mini-batch
        running_loss, seen = 0.0, 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
            running_loss += loss.item() * idx.numel()
            seen += idx.numel()
        if sched is not None:
            sched.step()

        model.eval()
        if eval_train:
            # 全量训练集分批前向，按样本数加权求平均 loss
            loss_sum, cnt = 0.0, 0
            with torch.no_grad():
                for start in range(0, n, eval_batch_size):
                    yb = ytr[start:start + eval_batch_size]
                    l = loss_fn(model(Xtr[start:start + eval_batch_size]), yb)
                    b = yb.numel()
                    loss_sum += l.item() * b
                    cnt += b
            train_loss = loss_sum / cnt if cnt else 0.0
            train_acc = evaluate_accuracy(model, Xtr, ytr, batch_size=eval_batch_size)
            tl = _finite(train_loss, "train_loss")
            ta = _finite(train_acc, "train_acc")
        else:
            tl = _finite(running_loss / seen if seen else 0.0, "train_loss")
            ta = None  # 未测量，不填占位数字

        test_acc = evaluate_accuracy(model, Xte, yte, batch_size=eval_batch_size)
        ep_seconds = time.perf_counter() - ep_t0

        tea = _finite(test_acc, "test_acc")
        es = _finite(ep_seconds, "epoch_seconds")
        history["train_loss"].append(tl)
        history["train_acc"].append(ta)
        history["test_acc"].append(tea)
        history["epoch_seconds"].append(es)
        if verbose:
            ta_str = "n/a   " if ta is None else f"{ta:.4f}"
            print(f"epoch {ep:>3}/{epochs} | loss {tl:.4f} | "
                  f"train_acc {ta_str} | test_acc {tea:.4f} | {es:.2f}s")

    history["final_test_acc"] = history["test_acc"][-1] if history["test_acc"] else 0.0
    history["total_seconds"] = _finite(time.perf_counter() - t0, "total_seconds")
    return history
