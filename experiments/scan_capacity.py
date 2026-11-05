"""容量扫描：把「输入维度（PCA 压缩）」与「量子参数量（电路深度）」分开检验。

回答一个具体问题：4 qubits / 24 个量子参数下的低准确率，到底是
「把 784 维图像压到 4 维丢了信息」，还是「量子电路容量不够」？

做法：固定其余超参，只扫一个因素，VQC 与经典 MLP 在完全相同的输入上对照。
  A. 维度扫描  n_features = 4, 8, 12, 16（layers 固定）
  B. 深度扫描  layers = 2, 4, 8, 16（n_features 固定）
经典 MLP 作为「这块输入里还剩多少可分信息」的参照系：
若只有 MLP 随维度上升，说明瓶颈在量子侧；若两者同步上升，瓶颈在输入压缩。

运行（真实 MNIST，离线 IDX）：
  python experiments/scan_capacity.py --out results
只做快速自检：
  python experiments/scan_capacity.py --quick --out results
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qml_lab.data import load_mnist
from qml_lab.models import MLPClassifier, VQCClassifier
from qml_lab.quantum import n_quantum_params
from qml_lab.train import train_classifier
from qml_lab.utils import count_parameters, ensure_dir, fmt_table, save_json, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="容量扫描：输入维度 vs 量子参数量")
    p.add_argument("--dims", type=int, nargs="+", default=[4, 8, 12, 16],
                   help="维度扫描的特征维 / qubit 数列表")
    p.add_argument("--layer-sweep", type=int, nargs="+", default=[2, 4, 8, 16],
                   help="深度扫描的 StronglyEntanglingLayers 层数")
    p.add_argument("--fixed-layers", type=int, default=2, help="维度扫描时固定的层数")
    p.add_argument("--fixed-qubits", type=int, default=4, help="深度扫描时固定的 qubit 数")
    p.add_argument("--train-samples", type=int, default=600)
    p.add_argument("--test-samples", type=int, default=200)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--classes", type=int, default=10)
    p.add_argument("--hidden", type=int, default=16, help="经典 MLP 隐层宽度")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-source", type=str, default="idx")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--out", type=str, default="results")
    p.add_argument("--quick", action="store_true", help="极小规模流程自检")
    return p.parse_args()


def run_one(n_features: int, layers: int, cfg: dict[str, Any]) -> dict[str, Any]:
    """在给定 (特征维, 层数) 下分别训练 VQC 与 MLP，返回汇总。"""
    set_seed(cfg["seed"])
    X, y, Xt, yt, meta = load_mnist(
        n_train=cfg["train_samples"],
        n_test=cfg["test_samples"],
        n_features=n_features,
        classes=cfg["classes"],
        source=cfg["data_source"],
        data_dir=cfg["data_dir"],
        seed=cfg["seed"],
    )

    records: dict[str, Any] = {
        "n_features": n_features,
        "layers": layers,
        "data_source": meta["source"],
        "quantum_params": n_quantum_params(n_features, layers),
    }

    models = {
        "VQC": VQCClassifier(n_features, cfg["classes"], layers),
        "MLP": MLPClassifier(n_features, cfg["classes"], hidden=cfg["hidden"]),
    }
    for name, model in models.items():
        hist = train_classifier(
            model, X, y, Xt, yt,
            epochs=cfg["epochs"], batch_size=cfg["batch_size"], lr=cfg["lr"],
            seed=cfg["seed"], eval_batch_size=cfg["batch_size"], verbose=False,
        )
        records[name] = {
            "params": int(count_parameters(model)),
            "test_acc": float(hist["final_test_acc"]),
            "train_acc": float(hist["train_acc"][-1]),
            "seconds": float(hist["total_seconds"]),
            "test_acc_curve": [float(v) for v in hist["test_acc"]],
        }

    print(
        f"  n_features={n_features:2d} layers={layers:2d} "
        f"qparams={records['quantum_params']:3d} | "
        f"VQC {records['VQC']['test_acc']:.4f} ({records['VQC']['seconds']:6.1f}s) | "
        f"MLP {records['MLP']['test_acc']:.4f} ({records['MLP']['seconds']:5.1f}s)",
        flush=True,
    )
    return records


def plot_scan(dim_rows: list[dict], layer_rows: list[dict], path: Path) -> None:
    """双面板图：左为输入维度扫描，右为电路深度扫描。"""
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    for ax, rows, xkey, xlabel, title in [
        (axes[0], dim_rows, "n_features", "features / qubits (PCA dim)", "A. Input dimension (layers fixed)"),
        (axes[1], layer_rows, "layers", "StronglyEntanglingLayers depth", "B. Circuit depth (qubits fixed)"),
    ]:
        xs = [r[xkey] for r in rows]
        for name, marker in [("VQC", "o"), ("MLP", "s")]:
            ax.plot(xs, [r[name]["test_acc"] for r in rows], marker=marker, label=name, linewidth=1.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("test accuracy")
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend()

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.quick:
        args.dims = [2, 4]
        args.layer_sweep = [1, 2]
        args.train_samples, args.test_samples, args.epochs = 64, 32, 1
        args.batch_size, args.classes, args.data_source = 8, 2, "synthetic"
        args.fixed_layers, args.fixed_qubits = 1, 2

    cfg = vars(args).copy()
    out_dir = ensure_dir(args.out)
    print(f"data source: {args.data_source} | train/test: {args.train_samples}/{args.test_samples} "
          f"| epochs: {args.epochs} | seed: {args.seed}", flush=True)

    print("--- A. input dimension sweep ---", flush=True)
    dim_rows = [run_one(d, args.fixed_layers, cfg) for d in args.dims]
    print("--- B. circuit depth sweep ---", flush=True)
    layer_rows = [run_one(args.fixed_qubits, L, cfg) for L in args.layer_sweep]

    payload = {
        "config": {k: v for k, v in cfg.items() if k != "quick"},
        "dimension_sweep": dim_rows,
        "depth_sweep": layer_rows,
        "caveat": "single seed, single run, no statistical repetition; "
                  "reduced sample/epoch budget vs the main experiment, "
                  "so absolute numbers are NOT comparable to the headline baseline",
    }
    save_json(payload, str(out_dir / "capacity_scan.json"))
    plot_scan(dim_rows, layer_rows, out_dir / "capacity_scan.png")

    rows = []
    for r in dim_rows + layer_rows:
        rows.append([
            f"dim={r['n_features']:2d} L={r['layers']:2d}",
            r["quantum_params"],
            f"{r['VQC']['test_acc']:.4f}",
            f"{r['MLP']['test_acc']:.4f}",
            f"{r['VQC']['seconds']:.1f}s",
        ])
    print(fmt_table(rows, ["配置", "量子参数", "VQC 准确率", "MLP 准确率", "VQC 耗时"]))
    print(f"saved: {out_dir / 'capacity_scan.json'} , {out_dir / 'capacity_scan.png'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[scan_capacity] 扫描失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        raise
