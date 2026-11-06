#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""「为什么只有 66%」：把准确率瓶颈拆成可量化的几段。

起因：5 seed 重跑后最佳量子配置只有 0.6640。本脚本不猜原因，而是把三个
候选瓶颈**分别单独加回去**，看每一项值多少分：

    base    : Amplitude-10 / MLP x5，600/200，24 ep   （复现主结果，作为同批次基准）
    long    : 同模型、同数据，epochs 加倍             （训练预算够不够 / 收没收敛）
    data    : 同模型、同 epochs，训练样本加倍          （数据预算够不够）
    hybrid  : CNN 前端 -> 8-qubit VQC -> MLP 头        （照参考实现补上经典特征前端）
    cnn_nq  : 同一 CNN 前端，但**跳过量子层**          （等架构经典对照：量子层贡献多少）
    cnn     : 纯经典 CNN 主干直接出 10 类              （同预算下的经典上限参照）

另外做**量子读出探针**：冻结训练好的量子层，把 8/10 维读出 z 交给 sklearn 的
线性 / RBF / MLP 分类器，测 z 里还剩多少可分信息；并用原始 784 维像素做同样
探针，作为「输入信息上限」的对照。探针只反映特征的判别信息，不是端到端训练。

方法学约定（见 docs/REVIEW.md）：
  * 默认 5 个 seed，报均值 ± 标准差；数据切分固定 seed=0；
  * 每个模型**构造前** set_seed（TorchLayer 用 torch 全局 RNG 初始化）；
  * quick 模式只做离线流程自检，结果不得用于任何结论。

运行：
  python experiments/diagnose_ceiling.py --data-source idx --out results \
      --archive docs/ceiling_metrics.json --fig-dir docs/figures
  python experiments/diagnose_ceiling.py --quick
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qml_lab.data import load_mnist_images, load_mnist_raw  # noqa: E402
from qml_lab.models import (  # noqa: E402
    AmplitudeVQCClassifier, CNNClassifier, HybridCNNClassifier,
)
from qml_lab.train import train_classifier  # noqa: E402
from qml_lab.utils import count_parameters, ensure_dir, set_seed  # noqa: E402


@dataclass(frozen=True)
class Group:
    key: str
    label: str
    kind: str                  # amp / hybrid / cnn_nq / cnn
    data: str                  # raw600 / raw_more / img600
    epochs: int
    frontend: str = "light"    # 仅混合 / CNN 组使用


def groups(args) -> list[Group]:
    return [
        Group("base", "Amp10 MLPx5 (baseline)", "amp", "raw600", args.epochs),
        Group("long", "base + epochs x2", "amp", "raw600", args.long_epochs),
        Group("data", "base + data x2", "amp", "raw_more", args.epochs),
        Group("hyb_light", "lightCNN + VQC", "hybrid", "img600", args.epochs, "light"),
        Group("nq_light", "lightCNN, no VQC", "cnn_nq", "img600", args.epochs, "light"),
        Group("cnn_light", "lightCNN direct", "cnn", "img600", args.epochs, "light"),
        Group("hyb_heavy", "heavyCNN + VQC", "hybrid", "img600", args.epochs, "heavy"),
        Group("nq_heavy", "heavyCNN, no VQC", "cnn_nq", "img600", args.epochs, "heavy"),
        Group("cnn_heavy", "heavyCNN direct", "cnn", "img600", args.epochs, "heavy"),
    ]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="分解 66% 准确率的瓶颈来源")
    p.add_argument("--seeds", type=str, default="0,1,2,3,4")
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--long-epochs", type=int, default=48)
    p.add_argument("--train-samples", type=int, default=600)
    p.add_argument("--test-samples", type=int, default=200)
    p.add_argument("--more-train", type=int, default=1200)
    p.add_argument("--more-test", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--classes", type=int, default=10)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--amp-qubits", type=int, default=10, help="振幅编码 qubits（需 2**n >= 特征维）")
    p.add_argument("--hybrid-qubits", type=int, default=8, help="混合模型量子头 qubits")
    p.add_argument("--data-source", type=str, default="idx")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--out", type=str, default="results")
    p.add_argument("--archive", type=str, default="docs/ceiling_metrics.json")
    p.add_argument("--fig-dir", type=str, default="docs/figures")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--quick", action="store_true", help="离线极小自检，不用于结论")
    return p.parse_args()


def _l2(X: np.ndarray) -> np.ndarray:
    A = np.asarray(X, dtype=np.float32)
    n = np.linalg.norm(A, axis=1, keepdims=True)
    n[n < 1e-12] = 1.0
    return np.ascontiguousarray(A / n, dtype=np.float32)


def load_all(args) -> dict:
    """数据切分固定 seed=0；img 与 raw 用同一抽样顺序，比较的是同一批图。"""
    kw = dict(classes=args.classes, source=args.data_source,
              data_dir=args.data_dir, seed=0)
    raw, yr, rawt, yt, m1 = load_mnist_raw(
        n_train=args.train_samples, n_test=args.test_samples, **kw)
    more, ym, moret, ymt, m2 = load_mnist_raw(
        n_train=args.more_train, n_test=args.more_test, **kw)
    img, yi, imgt, yit, m3 = load_mnist_images(
        n_train=args.train_samples, n_test=args.test_samples, **kw)

    # 关键受控性断言：两条数据路径抽到的是同一批图（不只是标签顺序相同）
    flat = img.reshape(len(img), -1) * 255.0
    assert np.array_equal(yr, yi), "raw 与 img 的标签不一致，抽样未受控"
    assert np.array_equal(yt, yit), "raw 与 img 的测试标签不一致，抽样未受控"
    assert np.allclose(raw, _l2(flat), atol=1e-4), "raw 与 img 的像素不一致，抽样未受控"

    if args.quick:
        # 振幅编码需要 2**n >= 特征维；quick 下把 784 维截到 2**amp_qubits 维
        k = 2 ** args.amp_qubits
        raw, rawt = _l2(raw[:, :k]), _l2(rawt[:, :k])
        more, moret = _l2(more[:, :k]), _l2(moret[:, :k])

    return {
        "raw600": dict(X=raw, y=yr, Xt=rawt, yt=yt),
        "raw_more": dict(X=more, y=ym, Xt=moret, yt=ymt),
        "img600": dict(X=img, y=yi, Xt=imgt, yt=yit),
        "meta": {"raw600": m1, "raw_more": m2, "img600": m3,
                 "same_split_asserted": True},
    }


def make_model(g: Group, args, data: dict, seed: int):
    """构造模型前重置种子 —— TorchLayer 的权重在构造时消耗 torch 全局 RNG。"""
    set_seed(seed)
    d = data[g.data]
    if g.kind == "amp":
        return AmplitudeVQCClassifier(
            args.amp_qubits, args.classes, args.layers,
            n_features=int(d["X"].shape[1]),
            head="mlp", scale=5.0, hidden=args.hidden,
        )
    if g.kind == "hybrid":
        return HybridCNNClassifier(
            n_qubits=args.hybrid_qubits, n_classes=args.classes,
            n_layers=args.layers, hidden=args.hidden,
            quantum=True, head="mlp", scale=5.0, frontend=g.frontend,
        )
    if g.kind == "cnn_nq":
        return HybridCNNClassifier(
            n_qubits=args.hybrid_qubits, n_classes=args.classes,
            n_layers=args.layers, hidden=args.hidden,
            quantum=False, head="mlp", scale=1.0, frontend=g.frontend,
        )
    return CNNClassifier(args.classes, frontend=g.frontend)


def _readout(model, X, batch: int = 64) -> np.ndarray:
    """提取量子层读出向量 z（hybrid 走 quantum_features，amp 直接走 q）。"""
    model.eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.as_tensor(X[i:i + batch], dtype=torch.float32)
            if hasattr(model, "quantum_features"):
                z = model.quantum_features(xb)
            else:
                z = model.q(xb).to(torch.float32)
            chunks.append(z.cpu().numpy())
    return np.concatenate(chunks, axis=0)


def _probes():
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier as SkMLP
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    return {
        "logreg": lambda: make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)),
        "rbf_svc": lambda: make_pipeline(
            StandardScaler(), SVC(kernel="rbf", C=10.0)),
        "mlp": lambda: make_pipeline(
            StandardScaler(), SkMLP(hidden_layer_sizes=(64, 32),
                                    max_iter=500, random_state=0)),
    }


def run_probes(model, d: dict, probe_cache: dict) -> dict:
    """冻结特征，用经典强分类器测「读出里还剩多少可分信息」。"""
    Ztr, Zte = _readout(model, d["X"]), _readout(model, d["Xt"])
    out = {"readout_dim": int(Ztr.shape[1]),
           "readout_std": round(float(Ztr.std()), 4)}
    for name, factory in _probes().items():
        clf = factory()
        clf.fit(Ztr, d["y"])
        out[name] = round(float(clf.score(Zte, d["yt"])), 4)
    return out


def pixel_probe(d: dict, args) -> dict:
    """输入信息上限：直接对原始像素做同样的冻结特征探针。"""
    X = np.asarray(d["X"])
    Xt = np.asarray(d["Xt"])
    if X.ndim > 2:
        X, Xt = X.reshape(len(X), -1), Xt.reshape(len(Xt), -1)
    out = {"dim": int(X.shape[1])}
    for name, factory in _probes().items():
        clf = factory()
        clf.fit(X, d["y"])
        out[name] = round(float(clf.score(Xt, d["yt"])), 4)
    return out


def _stats(values) -> dict:
    a = np.asarray(values, dtype=float)
    return {"mean": round(float(a.mean()), 4),
            "std": round(float(a.std(ddof=1)) if a.size > 1 else 0.0, 4),
            "min": round(float(a.min()), 4), "max": round(float(a.max()), 4),
            "values": [round(float(v), 4) for v in a], "n": int(a.size)}


def make_figures(groups_out: dict, curves: dict, out_dir: Path, args) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    keys = list(groups_out.keys())
    labels = [groups_out[k]["label"] for k in keys]
    means = [groups_out[k]["final_acc"]["mean"] for k in keys]
    stds = [groups_out[k]["final_acc"]["std"] for k in keys]

    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    colors = ["#8e8e8e", "#5b8ff9", "#5ad8a6", "#e8684a", "#9270ca", "#6dc8ec"]
    bars = ax.bar(range(len(keys)), means, yerr=stds, capsize=5,
                  color=[colors[i % len(colors)] for i in range(len(keys))],
                  edgecolor="#333333", linewidth=0.6, alpha=0.9)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title("Where the 66% ceiling comes from (5 seeds, mean +/- 1 sd)")
    ax.grid(axis="y", alpha=0.3)
    for b, m, s in zip(bars, means, stds):
        y = m + s + 0.025
        ax.text(b.get_x() + b.get_width() / 2, y, f"{m:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    p = out_dir / "ceiling_ladder.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made.append(str(p))

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for k in ("base", "long", "data"):
        a = np.asarray(curves[k], dtype=float)
        xs = np.arange(1, a.shape[1] + 1)
        ax.plot(xs, a.mean(axis=0), label=groups_out[k]["label"], linewidth=2)
        if a.shape[0] > 1:
            ax.fill_between(xs, a.mean(axis=0) - a.std(axis=0, ddof=1),
                            a.mean(axis=0) + a.std(axis=0, ddof=1), alpha=0.15)
    ax.axvline(args.epochs, color="#888888", linestyle="--", linewidth=1,
               label=f"original budget = {args.epochs} ep")
    ax.set_xlabel("epoch")
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title("Training / data budget: does more help?")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = out_dir / "ceiling_curves.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made.append(str(p))
    return made


def main() -> None:
    args = parse_args()
    if args.quick:
        args.seeds = "0"
        args.epochs, args.long_epochs = 2, 3
        args.train_samples, args.test_samples = 40, 20
        args.more_train, args.more_test = 60, 20
        args.data_source = "synthetic"
        args.amp_qubits, args.hybrid_qubits = 4, 4
        args.out, args.fig_dir = "results", "results"
        args.archive, args.no_figures = "", True
    set_seed(0)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    data = load_all(args)
    gs = groups(args)
    out_dir = ensure_dir(ROOT / args.out)

    runs: dict[str, dict] = {}
    curves: dict[str, list] = {}
    probe_runs: dict[str, dict] = {}
    probe_keys = {"base", "long", "hyb_light", "hyb_heavy"}

    for g in gs:
        d = data[g.data]
        runs[g.key] = {}
        curves[g.key] = []
        probe_runs[g.key] = {}
        for seed in seeds:
            model = make_model(g, args, data, seed)
            h = train_classifier(
                model, d["X"], d["y"], d["Xt"], d["yt"],
                epochs=g.epochs, batch_size=args.batch_size, lr=args.lr,
                seed=seed, eval_batch_size=64, verbose=False, eval_train=False,
            )
            rec = {
                "final_test_acc": round(float(h["final_test_acc"]), 4),
                "final_loss": round(float(h["train_loss"][-1]), 4),
                "seconds": round(float(h["total_seconds"]), 2),
                "params": int(count_parameters(model)),
                "test_acc_curve": [round(float(v), 4) for v in h["test_acc"]],
            }
            runs[g.key][str(seed)] = rec
            curves[g.key].append(h["test_acc"])
            if g.key in probe_keys:
                probe_runs[g.key][str(seed)] = run_probes(model, d, {})
            print(f"  [{g.key:8s} seed {seed}] acc={rec['final_test_acc']:.4f} "
                  f"loss={rec['final_loss']:.4f} params={rec['params']} "
                  f"{rec['seconds']}s", flush=True)

    groups_out = {}
    for g in gs:
        accs = [runs[g.key][str(s)]["final_test_acc"] for s in seeds]
        groups_out[g.key] = {
            "label": g.label, "kind": g.kind, "data": g.data,
            "epochs": g.epochs, "n_train": int(data[g.data]["X"].shape[0]),
            "params": runs[g.key][str(seeds[0])]["params"],
            "final_acc": _stats(accs),
            "final_loss": _stats([runs[g.key][str(s)]["final_loss"] for s in seeds]),
        }

    probes_out = {}
    for k, per_seed in probe_runs.items():
        if not per_seed:
            continue
        probes_out[k] = {name: _stats([per_seed[str(s)][name] for s in seeds])
                         for name in ("logreg", "rbf_svc", "mlp")}
        probes_out[k]["readout_dim"] = per_seed[str(seeds[0])]["readout_dim"]
        probes_out[k]["readout_std"] = per_seed[str(seeds[0])]["readout_std"]

    payload = {
        "config": {k: v for k, v in vars(args).items()
                   if k not in ("quick",)},
        "seeds": seeds,
        "data_meta": data["meta"],
        "groups": groups_out,
        "readout_probes": probes_out,
        "pixel_probe": pixel_probe(data["raw600"], args),
        "runs": runs,
        "caveat": ("探针是冻结特征 + 经典分类器的可分性上界，不是端到端训练结果；"
                   "不同组的样本数 / 特征维 / 参数量不同，绝对数值不可直接排名。"
                   "结论性数字均为 5 seed 均值 ± 标准差。"),
    }
    if args.quick:
        payload["caveat"] = "QUICK 自检，合成数据，不得用于任何结论。" + payload["caveat"]

    out_path = out_dir / "ceiling_metrics.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=float),
                        encoding="utf-8")
    print(f"\nsaved: {out_path}")

    figs = []
    if not args.no_figures:
        figs = make_figures(groups_out, curves, ROOT / args.fig_dir, args)
    if args.archive and not args.quick:
        arc = ROOT / args.archive
        arc.parent.mkdir(parents=True, exist_ok=True)
        arc.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=float),
                       encoding="utf-8")
        print(f"saved: {arc}")
    for f in figs:
        print(f"saved: {f}")

    print("\n=== 分组汇总（均值 ± 标准差） ===")
    for g in gs:
        s = groups_out[g.key]
        print(f"  {g.key:8s} {s['final_acc']['mean']:.4f} ± {s['final_acc']['std']:.4f} "
              f"| loss {s['final_loss']['mean']:.4f} | params {s['params']} "
              f"| n_train {s['n_train']} | {s['label']}")
    print("\n=== 量子读出探针（冻结特征，测试集 acc） ===")
    for k, v in probes_out.items():
        print(f"  {k:8s} dim={v['readout_dim']} "
              f"logreg {v['logreg']['mean']:.4f} | rbf {v['rbf_svc']['mean']:.4f} "
              f"| mlp {v['mlp']['mean']:.4f}")
    pp = payload["pixel_probe"]
    print(f"  {'pixel784':8s} dim={pp['dim']} "
          f"logreg {pp['logreg']:.4f} | rbf {pp['rbf_svc']:.4f} | mlp {pp['mlp']:.4f}")


if __name__ == "__main__":
    main()
