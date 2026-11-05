# -*- coding: utf-8 -*-
"""训练循环：VQC / MLP 分类器的评估与训练，以及玩具级字符语言模型训练。"""
from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F

try:
    from .utils import set_seed
except ImportError:  # 兼容把 qml_lab/ 直接加入 sys.path 后按顶层模块导入
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
                     eval_batch_size: int = 64, verbose: bool = True) -> dict:
    set_seed(seed)
    Xtr, ytr = _as_float_tensor(X_train), _as_long_tensor(y_train)
    Xte, yte = _as_float_tensor(X_test), _as_long_tensor(y_test)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    history = {"train_loss": [], "train_acc": [], "test_acc": [], "epoch_seconds": []}
    n = ytr.shape[0]
    t0 = time.perf_counter()

    for ep in range(1, epochs + 1):
        ep_t0 = time.perf_counter()
        model.train()
        perm = torch.randperm(n)  # 每 epoch 打乱做 mini-batch
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()

        # 全量训练集分批前向，按样本数加权求平均 loss
        model.eval()
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
        test_acc = evaluate_accuracy(model, Xte, yte, batch_size=eval_batch_size)
        ep_seconds = time.perf_counter() - ep_t0

        tl = _finite(train_loss, "train_loss")
        ta = _finite(train_acc, "train_acc")
        tea = _finite(test_acc, "test_acc")
        es = _finite(ep_seconds, "epoch_seconds")
        history["train_loss"].append(tl)
        history["train_acc"].append(ta)
        history["test_acc"].append(tea)
        history["epoch_seconds"].append(es)
        if verbose:
            print(f"epoch {ep:>3}/{epochs} | loss {tl:.4f} | "
                  f"train_acc {ta:.4f} | test_acc {tea:.4f} | {es:.2f}s")

    history["final_test_acc"] = history["test_acc"][-1] if history["test_acc"] else 0.0
    history["total_seconds"] = _finite(time.perf_counter() - t0, "total_seconds")
    return history


def train_lm(model, batch_gen, steps: int = 200, lr: float = 0.01, seed: int = 0,
             verbose: bool = True, log_every: int = 20) -> dict:
    set_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    history = {"step": [], "loss": [], "perplexity": [], "seconds": []}
    t0 = time.perf_counter()

    for step, (x, y) in zip(range(steps), batch_gen):
        s = step + 1
        opt.zero_grad()
        logits = model(x)                       # (B, T, V)
        V = logits.shape[-1]
        loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
        loss.backward()
        opt.step()

        lf = _finite(loss.item(), "loss")
        pf = _finite(math.exp(min(lf, 20.0)), "perplexity")  # clamp 后再 exp，防溢出
        sec = time.perf_counter() - t0         # 累计耗时，便于画 loss~时间曲线

        history["step"].append(s)
        history["loss"].append(lf)
        history["perplexity"].append(pf)
        history["seconds"].append(_finite(sec, "seconds"))
        if verbose and log_every > 0 and (s % log_every == 0 or s == steps):
            print(f"step {s:>4}/{steps} | loss {lf:.4f} | ppl {pf:.2f} | {sec:.1f}s")

    history["final_loss"] = history["loss"][-1] if history["loss"] else 0.0
    history["final_perplexity"] = history["perplexity"][-1] if history["perplexity"] else 1.0
    history["total_seconds"] = _finite(time.perf_counter() - t0, "total_seconds")
    return history
