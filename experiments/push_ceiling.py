#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第五轮正式对照：把诊断定位到的两个可修复瓶颈分别解除。

`probe_information.py`（第五轮诊断）给出的证据：

  * 冻结读出上界随宽度 10/20/40/75/165 维 = 0.630/0.730/0.815/0.870/0.890；
    165 维甚至高于原始像素 600 张图的 0.855。当前端到端只用了 40 维。
  * 量子权重训练前后只变 6.9%，梯度范数 0.048 → 0.897（不是 barren plateau）；
    165 维**随机不训练**的量子层上界已有 0.860，训练只值 +3 pt。
  * 像素 RBF 探针随训练样本 600 → 3000 → 12000 = 0.850 → 0.925 → 0.950；
    但同样 600 张图的 CNN 是 0.934，说明数据不足可以用归纳偏置补偿。
  * 40 维端到端 train loss 已降到 0.167 而 test acc 0.775 —— 已经过拟合，
    继续加 epoch 无用。

所以本脚本只动两件事：**读出宽度**与**训练数据的数量/多样性**，并配
等结构经典对照：

    b40         Amp10 / 40 维读出 / 600 张 / 24 ep          （复现 0.772 基线）
    b75         Amp10 / 75 维（全部 ZZ 两体关联）
    b165        Amp10 / 165 维（全部 ZZ+XX+YY 两体关联）
    b165_h128   同 b165，但分类头 hidden 32 → 128
    b165_d1800  同 b165，训练样本 600 → 1800，并把 epoch 24 → 8
                （固定「总样本呈现次数」= 14400，隔离数据多样性 vs 训练预算）
    b165_aug    同 b165，600 张原图 + 2 个 ±1px 平移变体 = 1800 张 / 8 ep
                （与 d1800 等样本数，区分「更多真图」与「增强」）
    ref_mlp784  经典 MLP，同 784 维输入 / 同训练配置
    ref_pca165  PCA-165 + 与 b165 完全同结构的 MLP 头（近似等参数预算）
    ref_cnn     轻量 CNN，同样 600 张图（归纳偏置参照）

协议：5 seed、数据切分固定 seed=0 且**所有组共用同一固定测试集**、
每个模型构造前 set_seed、受控断言同族同 seed 的量子初始权重逐元素相同
（用 sha256 比对）、配对差值 95% CI。为让 165 维的 45 个任务在合理时间内
跑完，支持 `--workers` 多进程并行；每个任务独立构造与训练，结果与串行等价。

运行：
  python experiments/push_ceiling.py --data-source idx --workers 8 \
      --out results --archive docs/push_metrics.json --fig-dir docs/figures
  python experiments/push_ceiling.py --quick     # 离线流程自检，不用于结论
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qml_lab.data import load_mnist_images, load_mnist_raw  # noqa: E402
from qml_lab.models import (  # noqa: E402
    CNNClassifier, MLPClassifier, RichVQCClassifier,
)
from qml_lab.quantum import n_readout  # noqa: E402
from qml_lab.train import train_classifier  # noqa: E402
from qml_lab.utils import count_parameters, ensure_dir, set_seed  # noqa: E402

OBS40 = "z+x+y+zz"
OBS75 = "z+x+y+zzall"
OBS165 = "z+x+y+zzall+xxall+yyall"
NQ = 10


@dataclass(frozen=True)
class Group:
    key: str
    label: str
    kind: str          # amp / mlp / pca / cnn
    obs: str = OBS40
    n_train: int = 600
    epochs: int = 24
    hidden: int = 32
    augment: int = 0   # 每个原图额外生成的平移变体数


def build_groups(args) -> dict[str, Group]:
    """主进程与 worker 用同一份组定义（多进程 spawn 下必须可重建）。"""
    if args.quick:
        return {
            "b40": Group("b40", "40-dim", "amp", OBS40, 40, 2),
            "b165": Group("b165", "165-dim", "amp", OBS165, 40, 2),
            "ref_mlp784": Group("ref_mlp784", "MLP", "mlp", OBS40, 40, 2),
            "ref_pca165": Group("ref_pca165", "PCA", "pca", OBS165, 40, 2),
            "ref_cnn": Group("ref_cnn", "CNN", "cnn", OBS40, 40, 2),
        }
    nt = args.train_samples
    base_ep = args.epochs
    big = min(args.pool_train, nt * 3)
    data_ep = max(1, int(round(base_ep * nt / big)))
    return {
        "b40": Group("b40", "Amp10 / 40-dim readout (baseline)", "amp", OBS40,
                     n_train=nt, epochs=base_ep),
        "b75": Group("b75", "Amp10 / 75-dim (all ZZ pairs)", "amp", OBS75,
                     n_train=nt, epochs=base_ep),
        "b165": Group("b165", "Amp10 / 165-dim (all ZZ+XX+YY)", "amp", OBS165,
                      n_train=nt, epochs=base_ep),
        "b165_h128": Group("b165_h128", "165-dim + wider MLP head", "amp",
                           OBS165, n_train=nt, epochs=base_ep, hidden=128),
        "b165_d1800": Group("b165_d1800", f"165-dim + {big} real images / {data_ep}ep",
                            "amp", OBS165, n_train=big, epochs=data_ep),
        "b165_aug": Group("b165_aug", f"165-dim + {nt} imgs x3 shift aug / {data_ep}ep",
                          "amp", OBS165, n_train=nt, epochs=data_ep, augment=2),
        "ref_mlp784": Group("ref_mlp784", "classical MLP-784", "mlp",
                            n_train=nt, epochs=base_ep),
        "ref_pca165": Group("ref_pca165", "PCA-165 + same MLP head", "pca",
                            n_train=nt, epochs=base_ep),
        "ref_cnn": Group("ref_cnn", f"light CNN (same {nt} images)", "cnn",
                         n_train=nt, epochs=base_ep),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="解除读出宽度与数据规模瓶颈（5 seed）")
    p.add_argument("--seeds", type=str, default="0,1,2,3,4")
    p.add_argument("--epochs", type=int, default=24, help="基准组 epoch；数据组用 Group.epochs")
    p.add_argument("--train-samples", type=int, default=600)
    p.add_argument("--test-samples", type=int, default=200)
    p.add_argument("--pool-train", type=int, default=3000,
                   help="统一大训练池，所有组从中取嵌套子集，测试集固定")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--classes", type=int, default=10)
    p.add_argument("--data-source", type=str, default="idx")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--out", type=str, default="results")
    p.add_argument("--archive", type=str, default="docs/push_metrics.json")
    p.add_argument("--fig-dir", type=str, default="docs/figures")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--torch-threads", type=int, default=2, help="每个 worker 的 torch 线程数")
    p.add_argument("--quick", action="store_true", help="离线极小自检，不用于结论")
    return p.parse_args()


# ---------------------------------------------------------------------------
# 数据：一次性加载统一大池，按类嵌套取子集；每个 worker 进程内部缓存。
# ---------------------------------------------------------------------------
_CACHE: dict[tuple, dict] = {}


def subset(X: np.ndarray, y: np.ndarray, n_train: int, classes: int):
    """_select 按类别顺序拼接，所以「每类取前 k 个」得到严格嵌套的均衡子集。"""
    per = X.shape[0] // classes
    k = n_train // classes
    assert k <= per, f"子集 {n_train} 超过池容量 {X.shape[0]}"
    idx = np.concatenate([np.arange(c * per, c * per + k) for c in range(classes)])
    return (np.ascontiguousarray(X[idx]),
            np.ascontiguousarray(np.asarray(y)[idx]))


def _shift2d(A: np.ndarray, dr: int, dc: int) -> np.ndarray:
    """整批 28x28 图像平移，边界补零（不做环绕，避免伪影）。"""
    B = np.zeros_like(A)
    h, w = A.shape[1], A.shape[2]
    rs = slice(max(0, -dr), min(h, h - dr))
    rd = slice(max(0, dr), min(h, h + dr))
    cs = slice(max(0, -dc), min(w, w - dc))
    cd = slice(max(0, dc), min(w, w + dc))
    B[:, rd, cd] = A[:, rs, cs]
    return B


def shift_augment(X: np.ndarray, variants: int) -> np.ndarray:
    """对 L2 归一化后的 784 维向量做 ±1px 平移，再重新归一化。

    平移在线性缩放前后等价，因此 reshape→平移→重新 L2 归一化即可复现
    「平移后的图片走振幅编码」的输入，无需回到 uint8 像素。
    """
    A = X.reshape(-1, 28, 28)
    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)][:variants]
    chunks = [X]
    for dr, dc in dirs:
        B = _shift2d(A, dr, dc).reshape(-1, 784)
        n = np.linalg.norm(B, axis=1, keepdims=True)
        n[n < 1e-12] = 1.0
        chunks.append(np.ascontiguousarray(B / n, dtype=np.float32))
    return np.ascontiguousarray(np.concatenate(chunks, axis=0), dtype=np.float32)


def _l2(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n < 1e-12] = 1.0
    return np.ascontiguousarray(X / n, dtype=np.float32)


def get_data(args) -> dict:
    key = (args.pool_train, args.test_samples, args.data_source,
           args.data_dir, args.classes, args.quick)
    if key in _CACHE:
        return _CACHE[key]
    src = "synthetic" if args.quick else args.data_source
    # 多进程 worker 的 cwd 不保证与主进程一致，数据目录统一解析成绝对路径
    dd = Path(args.data_dir)
    data_dir = str(dd if dd.is_absolute() else ROOT / dd)
    kw = dict(classes=args.classes, source=src, data_dir=data_dir, seed=0)
    Xr, yr, Xte, yte, m_raw = load_mnist_raw(
        n_train=args.pool_train, n_test=args.test_samples, **kw)
    Xi, yi, Xite, yite, m_img = load_mnist_images(
        n_train=args.pool_train, n_test=args.test_samples, **kw)
    assert np.array_equal(yr, yi) and np.array_equal(yte, yite), \
        "raw 与 images 两条路径抽样不一致"
    if src != "synthetic":
        flat = Xi.reshape(len(Xi), -1) * 255.0
        assert np.allclose(Xr, _l2(flat), atol=1e-4), "raw 与图像像素不一致"
    d = {"raw": (Xr, yr), "img": (Xi, yi),
         "Xtest": Xte, "ytest": yte, "imgtest": Xite,
         "meta": {"raw": m_raw, "img": m_img, "same_split_asserted": True}}
    _CACHE[key] = d
    return d


def pca_features(Xtr, Xte, dim: int, seed: int):
    """PCA 只在该组实际训练集上 fit；输入与量子组同为 L2 归一化 784 维。"""
    from sklearn.decomposition import PCA
    k = min(dim, Xtr.shape[0], Xtr.shape[1])
    p = PCA(n_components=k, random_state=seed)
    A = p.fit_transform(Xtr).astype(np.float32)
    B = p.transform(Xte).astype(np.float32)
    if k < dim:
        A = np.pad(A, ((0, 0), (0, dim - k)))
        B = np.pad(B, ((0, 0), (0, dim - k)))
    return A, B


def group_tensors(g: Group, args, d: dict, seed: int):
    """返回该组的 (X_train, y_train, X_test, y_test)。"""
    if g.kind == "cnn":
        Xp, yp = d["img"]
        Xt, yt = subset(Xp, yp, g.n_train, args.classes)
        return Xt, yt, d["imgtest"], d["ytest"]
    Xp, yp = d["raw"]
    Xtr, ytr = subset(Xp, yp, g.n_train, args.classes)
    Xte, yte = d["Xtest"], d["ytest"]
    if g.augment:
        Xtr = shift_augment(Xtr, g.augment)
        ytr = np.ascontiguousarray(np.tile(ytr, 1 + g.augment))
    if g.kind == "pca":
        dim = n_readout(NQ, OBS165)
        Xtr, Xte = pca_features(Xtr, Xte, dim, seed=0)
    return Xtr, ytr, Xte, yte


def build_model(g: Group, dim_in: int, seed: int, args):
    """构造模型**之前**重置种子（TorchLayer 在构造时消耗 torch 全局 RNG）。"""
    set_seed(seed)
    if g.kind == "amp":
        return RichVQCClassifier(
            NQ, args.classes, args.layers, encoding="amplitude",
            observables=g.obs, n_features=dim_in, head="mlp",
            scale=5.0, hidden=g.hidden)
    if g.kind in ("mlp", "pca"):
        return MLPClassifier(dim_in, args.classes, hidden=g.hidden)
    return CNNClassifier(args.classes, frontend="light")


def run_task(task: tuple) -> dict:
    g_key, seed, argv = task
    args = argparse.Namespace(**argv)
    torch.set_num_threads(max(1, int(args.torch_threads)))
    g = build_groups(args)[g_key]
    d = get_data(args)
    X, y, Xt, yt = group_tensors(g, args, d, seed)
    dim_in = int(np.asarray(X).reshape(len(X), -1).shape[1])
    model = build_model(g, dim_in, seed, args)

    q = getattr(model, "q", None)
    init_hash = None
    if q is not None:
        init_hash = hashlib.sha256(
            q.layer.weights.detach().cpu().numpy().tobytes()).hexdigest()

    t0 = time.perf_counter()
    h = train_classifier(
        model, X, y, Xt, yt, epochs=g.epochs, batch_size=args.batch_size,
        lr=args.lr, seed=seed, eval_batch_size=64, verbose=False,
        eval_train=False)
    seconds = round(time.perf_counter() - t0, 2)
    return {
        "group": g_key, "seed": seed,
        "final_test_acc": round(float(h["final_test_acc"]), 4),
        "final_loss": round(float(h["train_loss"][-1]), 4),
        "seconds": seconds,
        "params": int(count_parameters(model)),
        "readout_dim": int(getattr(model, "readout_dim", 0)),
        "n_train": int(len(y)),
        "dim_in": dim_in,
        "init_weight_sha256": init_hash,
        "test_acc_curve": [round(float(v), 4) for v in h["test_acc"]],
    }


def _stats(values) -> dict:
    a = np.asarray(values, dtype=float)
    return {"mean": round(float(a.mean()), 4),
            "std": round(float(a.std(ddof=1)) if a.size > 1 else 0.0, 4),
            "min": round(float(a.min()), 4), "max": round(float(a.max()), 4),
            "values": [round(float(v), 4) for v in a], "n": int(a.size)}


def _paired(a: dict, b: dict, seeds: list[int]) -> dict:
    from scipy.stats import t
    d = np.asarray([a["final_acc"]["values"][i] - b["final_acc"]["values"][i]
                    for i in range(len(seeds))], dtype=float) * 100.0
    mean = float(d.mean())
    sd = float(d.std(ddof=1)) if len(d) > 1 else 0.0
    half = (float(t.ppf(0.975, len(d) - 1)) * sd / np.sqrt(len(d))
            if len(d) > 1 else 0.0)
    return {"mean_pt": round(mean, 2), "std_pt": round(sd, 2),
            "ci95_pt": [round(mean - half, 2), round(mean + half, 2)],
            "ci_includes_zero": bool(mean - half <= 0 <= mean + half),
            "per_seed_pt": [round(float(v), 2) for v in d]}


def make_figures(groups_out: dict, curves: dict, out_dir: Path,
                 info_path: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    keys = list(groups_out)
    means = [groups_out[k]["final_acc"]["mean"] for k in keys]
    stds = [groups_out[k]["final_acc"]["std"] for k in keys]
    colors = ["#8e8e8e", "#5b8ff9", "#5ad8a6", "#5ad8a6", "#e8684a",
              "#e8684a", "#6dc8ec", "#6dc8ec", "#6dc8ec"]

    fig, ax = plt.subplots(figsize=(13, 5.8))
    bars = ax.bar(range(len(keys)), means, yerr=stds, capsize=5,
                  color=colors[:len(keys)], edgecolor="#333333",
                  linewidth=0.6, alpha=0.9)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([groups_out[k]["label"] for k in keys],
                       rotation=18, ha="right", fontsize=8)
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title("Pushing the ceiling: wider readout + more data (5 seeds, mean +/- 1 sd)")
    ax.grid(axis="y", alpha=0.3)
    for b, m, s in zip(bars, means, stds):
        ax.text(b.get_x() + b.get_width() / 2, m + s + 0.025, f"{m:.3f}",
                ha="center", fontsize=8.5)
    fig.tight_layout()
    p = out_dir / "push_ladder.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made.append(str(p))

    # 端到端 vs 诊断里的冻结读出上界（后者是单 seed 示意，单独标注）
    fig, ax = plt.subplots(figsize=(9, 4.8))
    order = [k for k in ("b40", "b75", "b165") if k in groups_out]
    dims = [groups_out[k]["readout_dim"] for k in order]
    end2end = [groups_out[k]["final_acc"]["mean"] for k in order]
    ax.plot(dims, end2end, "o-", linewidth=2, label="end-to-end test acc (5 seeds)")
    ceilings = []
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        for k, spec in (("b40", OBS40), ("b75", OBS75), ("b165", OBS165)):
            if k in order and spec in info.get("readout_probes", {}):
                e = info["readout_probes"][spec]["states"]["trained"]
                ceilings.append(max(e[n] for n in ("logreg", "rbf_svc", "mlp")))
    if len(ceilings) == len(order):
        ax.plot(dims, ceilings, "s--", linewidth=2,
                label="frozen-readout ceiling (diagnostic, 1 seed)")
    ax.set_xlabel("number of readout observables (amplitude-10, 60 quantum weights)")
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0.3, 1.0)
    ax.set_title("Wider readout: end-to-end accuracy tracks the information ceiling")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    for x, y in zip(dims, end2end):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=8)
    fig.tight_layout()
    p = out_dir / "push_width.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made.append(str(p))

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for k in ("b40", "b75", "b165", "b165_d1800", "b165_aug"):
        if k not in curves or not curves[k]:
            continue
        a = np.asarray(curves[k], dtype=float)
        xs = np.linspace(0, 100, a.shape[1])
        ax.plot(xs, a.mean(axis=0), label=groups_out[k]["label"], linewidth=2)
        if a.shape[0] > 1:
            ax.fill_between(xs, a.mean(axis=0) - a.std(axis=0, ddof=1),
                            a.mean(axis=0) + a.std(axis=0, ddof=1), alpha=0.12)
    ax.set_xlabel("training progress (% of that group's epochs)")
    ax.set_ylabel("test accuracy")
    ax.set_title("Convergence: wider readout and more data (5 seeds)")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = out_dir / "push_curves.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made.append(str(p))
    return made


def main() -> None:
    args = parse_args()
    if args.quick:
        args.seeds = "0"
        args.pool_train, args.train_samples, args.test_samples = 60, 40, 20
        args.data_source, args.torch_threads = "synthetic", 1
        args.out, args.fig_dir, args.archive = "results", "results", ""
        args.no_figures = True
        args.classes = 4
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    gs = build_groups(args)

    argv = vars(args).copy()
    tasks = [(k, s, argv) for k in gs for s in seeds]
    print(f"tasks={len(tasks)} workers={args.workers} "
          f"torch_threads/worker={args.torch_threads}", flush=True)

    results: list[dict] = []
    t0 = time.perf_counter()
    if args.workers <= 1:
        for i, task in enumerate(tasks, 1):
            r = run_task(task)
            results.append(r)
            print(f"  [{i:2d}/{len(tasks)}] {r['group']:12s} seed {r['seed']} "
                  f"acc={r['final_test_acc']:.4f} {r['seconds']}s", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run_task, t): t for t in tasks}
            for i, fut in enumerate(as_completed(futs), 1):
                r = fut.result()
                results.append(r)
                print(f"  [{i:2d}/{len(tasks)}] {r['group']:12s} seed {r['seed']} "
                      f"acc={r['final_test_acc']:.4f} {r['seconds']}s", flush=True)
    wall = round(time.perf_counter() - t0, 2)

    runs: dict[str, dict] = {k: {} for k in gs}
    curves: dict[str, list] = {k: [] for k in gs}
    for r in results:
        runs[r["group"]][str(r["seed"])] = {
            k: v for k, v in r.items() if k not in ("group", "seed")}
        curves[r["group"]].append([float(v) for v in r["test_acc_curve"]])

    groups_out = {}
    for k, g in gs.items():
        recs = [runs[k][str(s)] for s in seeds]
        groups_out[k] = {
            "label": g.label, "kind": g.kind, "observables":
                g.obs if g.kind == "amp" else None,
            "n_train": recs[0]["n_train"], "epochs": g.epochs,
            "hidden": g.hidden, "augment": g.augment,
            "params": recs[0]["params"], "readout_dim": recs[0]["readout_dim"],
            "final_acc": _stats([r["final_test_acc"] for r in recs]),
            "final_loss": _stats([r["final_loss"] for r in recs]),
            "seconds": _stats([r["seconds"] for r in recs]),
        }

    # 受控断言：同 (seed, kind, qubits) 的量子初始权重必须逐元素相同
    hash_by_bucket: dict[tuple, str] = {}
    control_checks: dict[str, dict] = {}
    for r in results:
        h = r["init_weight_sha256"]
        if h is None:
            continue
        # 所有 amp 组都是 10 qubits / 2 层 / 同一次 set_seed 后构造，
        # 只改观测量与分类头不应改变量子初始权重；因此全部归到同一个桶。
        bucket = (str(r["seed"]), "amp", NQ)
        ok = bucket not in hash_by_bucket or hash_by_bucket[bucket] == h
        hash_by_bucket.setdefault(bucket, h)
        control_checks.setdefault(str(r["seed"]), {})[r["group"]] = ok

    pairs = [("b75", "b40"), ("b165", "b40"), ("b165", "b75"),
             ("b165_h128", "b165"), ("b165_d1800", "b165"),
             ("b165_aug", "b165"), ("b165_aug", "b165_d1800"),
             ("ref_mlp784", "b165"), ("ref_pca165", "b165"),
             ("ref_cnn", "b165")]
    paired = {f"{a} - {b}": _paired(groups_out[a], groups_out[b], seeds)
              for a, b in pairs if a in groups_out and b in groups_out}

    payload = {
        "config": {k: v for k, v in vars(args).items() if k != "quick"},
        "seeds": seeds,
        "wall_seconds": wall,
        "data_meta": {
            "pool_train": args.pool_train,
            "test_samples": args.test_samples,
            "same_fixed_test_set_for_all_groups": True,
            "nested_balanced_subsets": True,
            "note": ("所有组从同一 3000 张训练池按类取嵌套前缀，共用同一 200 张测试集；"
                     "与第四轮 600/200 抽样的绝对数值不逐位可比，组内配对有效"),
        },
        "groups": groups_out,
        "controlled_checks": control_checks,
        "paired_differences": paired,
        "runs": runs,
        "caveat": ("5 seed 均值 ± 标准差，配对 95% CI 用 t 分布、未做多重比较校正；"
                   "数据组固定总样本呈现次数 14400（1800x8）以隔离数据多样性，"
                   "与 600x24 组的 epoch 数不同；PCA 参照在各自训练集上 fit；"
                   "165 个观测量在真机上需分组测量、shots 成本近似线性上升；"
                   "振幅编码在真机上需 O(2^n) 状态制备。"),
    }
    if args.quick:
        payload["caveat"] = "QUICK 自检，合成数据，不得用于任何结论。" + payload["caveat"]

    out_dir = ensure_dir(ROOT / args.out)
    p = out_dir / "push_metrics.json"
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=float),
                 encoding="utf-8")
    print(f"\nsaved: {p}", flush=True)
    figs = []
    if not args.no_figures:
        figs = make_figures(groups_out, curves, ROOT / args.fig_dir,
                            ROOT / "docs" / "info_metrics.json")
    if args.archive and not args.quick:
        arc = ROOT / args.archive
        arc.parent.mkdir(parents=True, exist_ok=True)
        arc.write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                                  default=float), encoding="utf-8")
        print(f"saved: {arc}", flush=True)
    for f in figs:
        print(f"saved: {f}", flush=True)

    print(f"\nwall time {wall}s", flush=True)
    print("\n=== 受控性断言（量子初始权重哈希逐元素等价） ===", flush=True)
    for s, checks in control_checks.items():
        print(f"  seed {s}: {sum(checks.values())}/{len(checks)} PASS", flush=True)
    print("\n=== 分组汇总（均值 ± 标准差） ===", flush=True)
    for k in gs:
        g = groups_out[k]
        print(f"  {k:12s} {g['final_acc']['mean']:.4f} ± {g['final_acc']['std']:.4f} "
              f"| loss {g['final_loss']['mean']:.4f} | params {g['params']:6d} "
              f"| readout {g['readout_dim']:3d} | n_train {g['n_train']:4d} "
              f"| {g['label']}", flush=True)
    print("\n=== 配对差值（百分点，95% CI） ===", flush=True)
    for k, v in paired.items():
        flag = "不显著" if v["ci_includes_zero"] else "显著"
        print(f"  {k:28s} {v['mean_pt']:+6.2f} ± {v['std_pt']:5.2f} "
              f"CI[{v['ci95_pt'][0]:+6.2f},{v['ci95_pt'][1]:+6.2f}] {flag}",
              flush=True)


if __name__ == "__main__":
    main()
