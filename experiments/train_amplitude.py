"""编码方式对照：角度编码必须压维，振幅编码不用。

核心问题：`AngleEmbedding` 一个 qubit 装一个特征，于是 784 维图像被 PCA 压到 4 维；
`AmplitudeEmbedding` 用 n 个 qubits 的 2^n 维态空间装 2^n 个振幅，
10 个 qubits（2^10 = 1024）就能吃下完整 784 维，**无需任何有损压缩**。

受控对照（同 qubits、同 ansatz、同量子参数量、同训练配置，只换编码）：
  Angle-10   PCA -> 10 维 + 10 qubits 角度编码
  Amplitude-10    完整 784 维 + 10 qubits 振幅编码
另加两个参照：
  Angle-4    PCA -> 4 维 + 4 qubits（仓库主实验的基线配置）
  MLP-784    完整 784 维的经典 MLP（衡量这份输入里还剩多少可分信息）

重要边界：模拟器里写入振幅初态是 O(1) 的，真机上需要 O(2^n) 深度的
状态制备电路或尚不存在的 QRAM。本实验的耗时**不能外推到真机**。

运行：
  python experiments/train_amplitude.py --data-source idx --out results
  python experiments/train_amplitude.py --quick --out results   # 离线自检
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qml_lab.data import load_mnist, load_mnist_raw
from qml_lab.models import AmplitudeVQCClassifier, MLPClassifier, VQCClassifier
from qml_lab.quantum import n_quantum_params
from qml_lab.train import train_classifier
from qml_lab.utils import count_parameters, ensure_dir, fmt_table, save_json, set_seed
from qml_lab.viz import plot_bars, plot_curves


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="振幅编码 vs 角度编码：是否需要 PCA 压缩")
    p.add_argument("--qubits", type=int, default=10, help="振幅/大角度编码的 qubit 数，需 2**n >= 784")
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--classes", type=int, default=10)
    p.add_argument("--train-samples", type=int, default=600)
    p.add_argument("--test-samples", type=int, default=200)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-source", type=str, default="idx")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--hidden", type=int, default=32, help="经典 MLP 隐层宽度")
    p.add_argument("--out", type=str, default="results")
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def fit(model, X, y, Xt, yt, cfg: dict[str, Any]) -> dict[str, Any]:
    """统一训练配置，返回与 train_mnist.py 兼容的 history 结构。"""
    set_seed(cfg["seed"])
    return train_classifier(
        model, X, y, Xt, yt,
        epochs=cfg["epochs"], batch_size=cfg["batch_size"], lr=cfg["lr"],
        seed=cfg["seed"], eval_batch_size=cfg["batch_size"], verbose=False,
    )


def main() -> None:
    args = parse_args()
    if args.quick:
        args.qubits, args.layers = 4, 1
        args.train_samples, args.test_samples, args.epochs = 64, 32, 1
        args.batch_size, args.classes, args.data_source = 8, 2, "synthetic"

    # 自检模式下 2^4=16 < 784，用一个小特征数走通流程即可
    n_raw_features = 16 if args.quick else 784
    cfg = vars(args).copy()
    set_seed(args.seed)  # 进程级可复现；每组模型构造前还会在 record() 内再次重置
    out_dir = ensure_dir(args.out)

    print("== 编码方式对照：角度编码（需 PCA） vs 振幅编码（无需 PCA）==", flush=True)
    print(f"qubits={args.qubits} (2^{args.qubits}={2 ** args.qubits}), "
          f"layers={args.layers}, classes={args.classes}, seed={args.seed}", flush=True)

    summary: dict[str, Any] = {}
    histories: dict[str, Any] = {}
    metas: dict[str, Any] = {}

    def record(name: str, model_fn, X, y, Xt, yt, n_q: int, meta: dict) -> None:
        # 关键：TorchLayer 的权重与分类头都在构造时消耗全局 RNG。
        # 必须在**构造模型之前**重置种子，否则各组初始权重不同、受控对照失效，
        # 且独立进程之间结果不可复现（fit/ 与 train_classifier 里的 set_seed 都太晚）。
        set_seed(cfg["seed"])
        model = model_fn()
        print(f"  training {name} ...", flush=True)
        h = fit(model, X, y, Xt, yt, cfg)
        acc = float(h["final_test_acc"])
        secs = float(h["total_seconds"])
        summary[name] = {
            "params": int(count_parameters(model)),
            "quantum_params": int(n_quantum_params(n_q, args.layers)),
            "final_test_acc": acc,
            "total_seconds": secs,
            "input_dim": int(X.shape[1]),
            "n_qubits": n_q,
        }
        histories[name] = h
        metas[name] = meta
        print(f"    acc={acc:.4f}  params={summary[name]['params']}  time={secs:.1f}s", flush=True)

    # ---- 角度编码组：特征维必须等于 qubits 数，因此必须 PCA 压缩 ----
    angle_dims = [2, 4] if args.quick else sorted({4, args.qubits})
    for n_q in angle_dims:
        label = f"Angle-{n_q} (PCA->{n_q})"
        X, y, Xt, yt, meta = load_mnist(
            n_train=args.train_samples, n_test=args.test_samples,
            n_features=n_q, classes=args.classes, source=args.data_source,
            data_dir=args.data_dir, seed=args.seed,
        )
        record(label, lambda q=n_q: VQCClassifier(q, args.classes, args.layers),
               X, y, Xt, yt, n_q, meta)

    # ---- 振幅编码组：完整特征，不压缩 ----
    Xr, yr, Xtr_, ytr_, meta_raw = load_mnist_raw(
        n_train=args.train_samples, n_test=args.test_samples,
        classes=args.classes, source=args.data_source,
        data_dir=args.data_dir, seed=args.seed,
    )
    if args.quick:
        # 自检：截取前 n_raw_features 维，保持流程一致
        Xr, Xtr_ = Xr[:, :n_raw_features], Xtr_[:, :n_raw_features]
    record(
        f"Amplitude-{args.qubits} (full {Xr.shape[1]}-d)",
        lambda: AmplitudeVQCClassifier(args.qubits, args.classes, args.layers,
                                       n_features=int(Xr.shape[1])),
        Xr, yr, Xtr_, ytr_, args.qubits, meta_raw,
    )

    # ---- 经典参照：同样的完整输入 ----
    record(
        f"MLP-{Xr.shape[1]} (classical)",
        lambda: MLPClassifier(int(Xr.shape[1]), args.classes, hidden=args.hidden),
        Xr, yr, Xtr_, ytr_, 0, meta_raw,
    )

    payload = {
        "config": {k: v for k, v in cfg.items() if k != "quick"},
        "metas": metas,
        "histories": histories,
        "summary": summary,
        "caveat": (
            "Single seed, single run, reduced sample/epoch budget - not statistically robust. "
            "Amplitude state preparation is O(1) in a statevector simulator but needs O(2^n) "
            "gates or QRAM on real hardware; timings here do NOT extrapolate to hardware. "
            "Amplitude encoding is a linear map; the only nonlinearity comes from the variational "
            "ansatz, and the readout is still just n_qubits PauliZ expectation values."
        ),
    }
    save_json(payload, str(out_dir / "amplitude_metrics.json"))

    plot_curves(
        histories=histories, metric="test_acc", ylabel="Test Accuracy",
        path=str(out_dir / "amplitude_accuracy.png"),
        title="Encoding comparison: angle (PCA) vs amplitude (no PCA)",
    )
    plot_bars(
        values={k: v["final_test_acc"] for k, v in summary.items()},
        ylabel="Test Accuracy", path=str(out_dir / "amplitude_bars.png"),
        title="Test accuracy by encoding",
    )

    rows = [
        [name, s["input_dim"], s["n_qubits"], s["quantum_params"],
         s["params"], f"{s['final_test_acc']:.4f}", f"{s['total_seconds']:.1f}s"]
        for name, s in summary.items()
    ]
    print(fmt_table(rows, ["模型", "输入维", "qubits", "量子参数", "总参数", "测试准确率", "耗时"]))

    print("\n注意：模拟器里振幅初态是「免费」写入的；真机需要 O(2^n) 状态制备或 QRAM，"
          "本实验耗时不能外推到真实量子硬件。", flush=True)
    print(f"saved: {out_dir / 'amplitude_metrics.json'} , "
          f"{out_dir / 'amplitude_accuracy.png'} , {out_dir / 'amplitude_bars.png'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[train_amplitude] 实验失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        raise
