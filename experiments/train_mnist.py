"""实验1：VQC 量子变分分类器做 MNIST，与经典 MLP 对比准确率/参数量/耗时。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

# 保证从任意 cwd 运行都能 import qml_lab（须在导入 qml_lab 之前）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from qml_lab.data import load_mnist
from qml_lab.models import MLPClassifier, VQCClassifier
from qml_lab.quantum import n_quantum_params
from qml_lab.train import train_classifier
from qml_lab.utils import (
    count_parameters,
    ensure_dir,
    fmt_table,
    pick_device,
    save_json,
    set_seed,
)
from qml_lab.viz import plot_bars, plot_curves


def _to_builtin(obj: Any) -> Any:
    """递归把 numpy 标量/数组转成 Python 内置类型，保证 JSON 里没有 numpy 原生类型。"""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    return obj


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="实验1：VQC 做 MNIST，与经典 MLP 对照")
    p.add_argument("--qubits", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--classes", type=int, default=10)
    p.add_argument("--train-samples", type=int, default=2000)
    p.add_argument("--test-samples", type=int, default=500)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-source", type=str, default="auto")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--diff-method", type=str, default="backprop")
    p.add_argument("--shots", type=int, default=0, help="0 表示解析模拟")
    p.add_argument("--out", type=str, default="results")
    p.add_argument("--quick", action="store_true", help="极小规模冒烟运行")
    p.add_argument("--skip-baseline", action="store_true", help="跳过经典 MLP 基线")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # --quick：压到最小规模，并强制合成数据（不联网、不下载 MNIST）
    if args.quick:
        args.classes = 2
        args.train_samples = 64
        args.test_samples = 32
        args.epochs = 1
        args.batch_size = 8
        args.data_source = "synthetic"

    set_seed(args.seed)
    out_dir = ensure_dir(args.out)
    print(f"device: {pick_device()}")

    X_train, y_train, X_test, y_test, meta = load_mnist(
        n_train=args.train_samples,
        n_test=args.test_samples,
        n_features=args.qubits,
        classes=args.classes,
        source=args.data_source,
        data_dir=str(Path(args.data_dir)),
        seed=args.seed,
    )
    print(f"数据源: {meta['source']}")
    print(f"说明: {meta['note']}")

    # 量子模型：shots=0 -> None 走解析模拟
    vqc = VQCClassifier(
        n_qubits=args.qubits,
        n_classes=args.classes,
        n_layers=args.layers,
        diff_method=args.diff_method,
        shots=args.shots or None,
    )
    hist_vqc = train_classifier(
        model=vqc,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        eval_batch_size=args.batch_size,
        verbose=True,
    )

    # 每个模型一条记录，便于统一生成汇总 / 图表 / 对比表
    records: dict[str, dict[str, Any]] = {
        "VQC": {
            "model": vqc,
            "history": hist_vqc,
            "quantum_params": int(
                n_quantum_params(n_qubits=args.qubits, n_layers=args.layers)
            ),
        },
    }
    if not args.skip_baseline:
        mlp = MLPClassifier(n_qubits=args.qubits, n_classes=args.classes, hidden=16)
        hist_mlp = train_classifier(
            model=mlp,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            eval_batch_size=args.batch_size,
            verbose=True,
        )
        records["MLP"] = {"model": mlp, "history": hist_mlp, "quantum_params": 0}

    summary: dict[str, Any] = {}
    histories: dict[str, Any] = {}
    params_values: dict[str, int] = {}
    time_values: dict[str, float] = {}
    rows: list[list[Any]] = []
    for name, rec in records.items():
        h = rec["history"]
        params = int(count_parameters(rec["model"]))
        acc = float(h["final_test_acc"])
        secs = float(h["total_seconds"])
        summary[name] = {
            "params": params,
            "final_test_acc": acc,
            "total_seconds": secs,
            "n_quantum_params": rec["quantum_params"],
        }
        histories[name] = h
        params_values[name] = params
        time_values[name] = secs
        rows.append([name, params, f"{acc:.4f}", f"{secs:.2f}"])

    # 保存指标：整体转内置类型，避免 numpy 标量进 JSON
    payload = {
        "config": vars(args),
        "meta": meta,
        "histories": histories,
        "summary": summary,
    }
    save_json(_to_builtin(payload), str(out_dir / "mnist_metrics.json"))

    # 图表统一输出到 --out
    plot_curves(
        histories=histories,
        metric="test_acc",
        ylabel="Test Accuracy",
        path=str(out_dir / "mnist_accuracy.png"),
        title="Exp1: VQC vs MLP test accuracy",
    )
    plot_bars(
        values=params_values,
        ylabel="Parameter Count",
        path=str(out_dir / "mnist_params.png"),
        title="Exp1: parameter count",
    )
    plot_bars(
        values=time_values,
        ylabel="Train Time (s)",
        path=str(out_dir / "mnist_time.png"),
        title="Exp1: training time",
    )

    print(fmt_table(rows, ["模型", "参数量", "测试准确率", "训练耗时(s)"]))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # 实验脚本：给出清晰中文提示后保留堆栈
        print(f"[train_mnist] 实验失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        raise
