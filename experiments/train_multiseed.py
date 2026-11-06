"""多 seed 受控对照：手写数字识别到底能到多少，以及哪些改动真的有效。

2026-11-06 方法学审查后的重跑协议（见 docs/REVIEW.md）：

* **数据固定**：所有组、所有 seed 用同一批真实 MNIST 样本（数据切分 seed 恒为 0），
  方差只来自模型初始化与训练 shuffle，不混入重采样噪声。
  `load_mnist` 与 `load_mnist_raw` 在同一 seed 下抽到相同样本索引，
  所以「角度编码 + PCA」与「振幅编码 + 全 784 维」比较的是**同一批图**。
* **初始权重受控**：`qml.qnn.TorchLayer` 用 torch 全局 RNG 初始化，因此每次构造模型前
  都 `set_seed(seed)`；脚本会断言同 qubits 的所有量子组初始权重逐元素相同，
  并额外断言 `Angle-10 / Linear` 与 `Amplitude-10 / Linear` 的分类头也相同。
  这两条是上一轮「受控对照」缺失的关键检查。
* **统计重复**：默认 5 个 seed，报告 acc@8（对齐旧实验）与 acc@24 的均值 ± 标准差，
  对关键配对差值给 95% 置信区间；不再用单次运行下结论。
* **逐步分离变量**：`linear → mlp → mlp+scale5 → mlp+learnable scale`，
  把「分类头非线性」与「输出缩放」的贡献分开，不再像早期诊断那样一次性混改三个因素。

运行：
  python experiments/train_multiseed.py --data-source idx --out results
  python experiments/train_multiseed.py --quick          # 离线极小自检
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qml_lab.data import load_mnist, load_mnist_raw  # noqa: E402
from qml_lab.models import (  # noqa: E402
    AmplitudeVQCClassifier, MLPClassifier, VQCClassifier,
)
from qml_lab.quantum import n_quantum_params  # noqa: E402
from qml_lab.train import train_classifier  # noqa: E402
from qml_lab.utils import count_parameters, ensure_dir, fmt_table, set_seed  # noqa: E402


@dataclass(frozen=True)
class Spec:
    key: str
    label: str
    kind: str            # angle / amplitude / mlp
    data: str            # pca4 / pca10 / raw
    qubits: int = 0
    head: str = "linear"
    scale: Any = 1.0
    input_dim: int = 0


def specs() -> list[Spec]:
    return [
        Spec("A4_linear", "Angle-4 / Linear", "angle", "pca4", qubits=4),
        Spec("A4_mlp", "Angle-4 / MLP", "angle", "pca4", qubits=4, head="mlp"),
        Spec("A4_mlp_s5", "Angle-4 / MLP x5", "angle", "pca4", qubits=4, head="mlp", scale=5.0),
        Spec("A10_linear", "Angle-10 / Linear", "angle", "pca10", qubits=10),
        Spec("Amp10_linear", "Amplitude-10 / Linear", "amplitude", "raw", qubits=10),
        Spec("Amp10_mlp", "Amplitude-10 / MLP", "amplitude", "raw", qubits=10, head="mlp"),
        Spec("Amp10_mlp_s5", "Amplitude-10 / MLP x5", "amplitude", "raw",
             qubits=10, head="mlp", scale=5.0),
        Spec("Amp10_mlp_learn", "Amplitude-10 / MLP + learnable scale", "amplitude", "raw",
             qubits=10, head="mlp", scale="learnable"),
        Spec("MLP4", "MLP-4 (classical, PCA-4)", "mlp", "pca4", input_dim=4),
        Spec("MLP784", "MLP-784 (classical, full)", "mlp", "raw", input_dim=784),
    ]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="多 seed 受控对照：VQC / 振幅编码 / 经典基线")
    p.add_argument("--seeds", type=str, default="0,1,2,3,4")
    p.add_argument("--train-samples", type=int, default=600)
    p.add_argument("--test-samples", type=int, default=200)
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--report-epoch", type=int, default=8, help="对齐旧实验的中间时点")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--classes", type=int, default=10)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--data-source", type=str, default="idx")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--out", type=str, default="results")
    p.add_argument("--fig-dir", type=str, default="docs/figures")
    p.add_argument("--archive", type=str, default="docs/multiseed_metrics.json",
                   help="把完整原始指标归档到仓库（results/ 被 gitignore）")
    p.add_argument("--quick", action="store_true", help="极小规模离线自检，不用于结论")
    return p.parse_args()


def load_all(args) -> dict[str, dict]:
    """数据切分固定 seed=0；返回每个 data key 的 (Xtr, ytr, Xte, yte, meta)。"""
    kw = dict(n_train=args.train_samples, n_test=args.test_samples,
              classes=args.classes, source=args.data_source,
              data_dir=args.data_dir, seed=0)
    raw_kw = dict(kw)
    raw_kw.pop("n_features", None)
    out: dict[str, dict] = {}
    for name, feats in (("pca4", 4), ("pca10", 10)):
        X, y, Xt, yt, meta = load_mnist(n_features=feats, **kw)
        out[name] = dict(X=X, y=y, Xt=Xt, yt=yt, meta=meta)
    X, y, Xt, yt, meta = load_mnist_raw(**raw_kw)
    out["raw"] = dict(X=X, y=y, Xt=Xt, yt=yt, meta=meta)
    return out


def make_model(spec: Spec, args, n_raw_features: int, data: dict):
    """构造模型前重置种子 —— TorchLayer 的权重在构造时消耗 torch 全局 RNG。"""
    set_seed(args._seed)
    if spec.kind == "angle":
        return VQCClassifier(
            spec.qubits, args.classes, args.layers,
            head=spec.head, scale=spec.scale, hidden=args.hidden,
        )
    if spec.kind == "amplitude":
        return AmplitudeVQCClassifier(
            spec.qubits, args.classes, args.layers,
            n_features=n_raw_features, head=spec.head, scale=spec.scale,
            hidden=args.hidden,
        )
    # 经典 MLP 的输入维直接取实际数据（quick 模式下 raw 会被截到 16 维）
    in_dim = int(data[spec.data]["X"].shape[1])
    return MLPClassifier(in_dim, args.classes, hidden=args.hidden)


def init_signature(model) -> dict[str, torch.Tensor]:
    """取量子权重与分类头首层权重，用于跨组受控性断言。"""
    sig: dict[str, torch.Tensor] = {}
    if hasattr(model, "q"):
        sig["q"] = model.q.layer.weights.detach().clone()
    first = next(model.head.parameters()) if hasattr(model, "head") else None
    if first is None and hasattr(model, "net"):
        first = next(model.net.parameters())
    if first is not None:
        sig["head0"] = first.detach().clone()
    return sig


def stats(values: list[float]) -> dict[str, Any]:
    a = np.asarray(values, dtype=float)
    return {
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "min": float(a.min()), "max": float(a.max()),
        "values": [round(float(v), 4) for v in a],
    }


def paired(left: list[float], right: list[float]) -> dict[str, Any]:
    """right - left 的逐 seed 配对差值与 95% 置信区间（t 分布，缺失则正态近似）。"""
    d = np.asarray(right, dtype=float) - np.asarray(left, dtype=float)
    n = d.size
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    try:
        from scipy.stats import t as student_t
        crit = float(student_t.ppf(0.975, n - 1))
    except Exception:
        crit = 1.96
    mean = float(d.mean())
    enough = n >= 2 and sd > 0.0
    half = crit * sd / np.sqrt(n) if enough else 0.0
    return {
        "mean_pt": round(mean * 100, 2),
        "std_pt": round(sd * 100, 2),
        "ci95_pt": ([round((mean - half) * 100, 2), round((mean + half) * 100, 2)]
                    if enough else None),
        "ci_includes_zero": (bool(mean - half <= 0.0 <= mean + half) if enough else None),
        "per_seed_pt": [round(float(v) * 100, 2) for v in d],
    }


def summarize(runs: dict, all_specs: list[Spec], report_epoch: int) -> tuple[dict, dict]:
    summary, curves = {}, {}
    for sp in all_specs:
        rs = runs[sp.key]
        acc8 = [r["acc_at_report"] for r in rs.values()]
        accf = [r["final_test_acc"] for r in rs.values()]
        curves[sp.key] = np.asarray([r["test_acc_curve"] for r in rs.values()], dtype=float)
        summary[sp.key] = {
            "label": sp.label,
            "params": rs[next(iter(rs))]["params"],
            "acc_at_report": stats(acc8),
            "final_acc": stats(accf),
            "seconds": stats([r["seconds"] for r in rs.values()]),
            "final_loss": stats([r["final_loss"] for r in rs.values()]),
        }
    get = lambda key, field: [r[field] for r in runs[key].values()]  # noqa: E731
    pairs = {
        "encoding_Amp10_minus_A10 (linear heads)":
            paired(get("A10_linear", "final_test_acc"), get("Amp10_linear", "final_test_acc")),
        "head_A4_mlp_minus_linear":
            paired(get("A4_linear", "final_test_acc"), get("A4_mlp", "final_test_acc")),
        "scale_A4_s5_minus_mlp":
            paired(get("A4_mlp", "final_test_acc"), get("A4_mlp_s5", "final_test_acc")),
        "head_Amp10_mlp_minus_linear":
            paired(get("Amp10_linear", "final_test_acc"), get("Amp10_mlp", "final_test_acc")),
        "scale_Amp10_s5_minus_mlp":
            paired(get("Amp10_mlp", "final_test_acc"), get("Amp10_mlp_s5", "final_test_acc")),
        "learnable_minus_fixed5_Amp10":
            paired(get("Amp10_mlp_s5", "final_test_acc"), get("Amp10_mlp_learn", "final_test_acc")),
        "best_quantum_minus_A4_linear":
            paired(get("A4_linear", "final_test_acc"), get("Amp10_mlp_s5", "final_test_acc")),
    }
    return summary, {"curves": curves, "pairs": pairs}


def make_figures(curves: dict, summary: dict, fig_dir: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = ensure_dir(fig_dir)
    epochs = next(iter(curves.values())).shape[1]
    n_seeds = next(iter(curves.values())).shape[0]
    ddof = 1 if n_seeds > 1 else 0
    xs = np.arange(1, epochs + 1)
    made: list[str] = []

    # 曲线：均值 ± 1 标准差
    show = ["A4_linear", "A4_mlp_s5", "Amp10_linear", "Amp10_mlp_s5", "MLP4", "MLP784"]
    colors = ["#6b7280", "#2563eb", "#9ca3af", "#dc2626", "#0d9488", "#111827"]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for key, color in zip(show, colors):
        a = curves[key]
        m, s = a.mean(axis=0), a.std(axis=0, ddof=ddof)
        ax.plot(xs, m, label=summary[key]["label"], color=color, lw=1.8)
        ax.fill_between(xs, m - s, m + s, color=color, alpha=0.13)
    ax.axhline(0.1, ls="--", lw=1, color="#9ca3af", label="chance (10 classes)")
    ax.set_xlabel("epoch"); ax.set_ylabel("test accuracy")
    ax.set_title("Multi-seed test accuracy (mean +/- 1 SD, n=%d)" % next(iter(curves.values())).shape[0])
    ax.set_ylim(0, 1.0); ax.grid(alpha=0.25); ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    p1 = fig_dir / "multiseed_accuracy.png"
    fig.savefig(p1, dpi=150); plt.close(fig); made.append(str(p1))

    # 柱状：最终准确率 + 误差棒
    keys = list(summary.keys())
    means = [summary[k]["final_acc"]["mean"] for k in keys]
    stds = [summary[k]["final_acc"]["std"] for k in keys]
    bar_colors = ["#93c5fd" if k.startswith("A4") else
                  ("#fca5a5" if k.startswith("Amp") else
                   ("#bfdbfe" if k.startswith("A10") else "#a7f3d0")) for k in keys]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    bars = ax.bar(range(len(keys)), means, yerr=stds, capsize=4,
                  color=bar_colors, edgecolor="#374151", lw=0.7)
    for b, m, s in zip(bars, means, stds):
        ax.text(b.get_x() + b.get_width() / 2, m + s + 0.018, f"{m:.3f}",
                ha="center", fontsize=8)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("final test accuracy"); ax.set_ylim(0, 1.0)
    ax.set_title("Final accuracy after %d epochs (mean +/- 1 SD, error bars)" % epochs)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    p2 = fig_dir / "multiseed_bars.png"
    fig.savefig(p2, dpi=150); plt.close(fig); made.append(str(p2))
    return made


def main() -> None:
    args = parse_args()
    if args.quick:
        args.seeds = "0"
        args.train_samples, args.test_samples = 32, 16
        args.epochs, args.report_epoch = 2, 1
        args.batch_size, args.classes, args.data_source = 8, 2, "synthetic"
        args.fig_dir, args.archive = "results", ""
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    all_specs = specs()
    out_dir = ensure_dir(args.out)
    out_path = out_dir / "multiseed_metrics.json"

    print("== 多 seed 受控对照：量子手写数字识别 ==", flush=True)
    print(f"seeds={seeds} epochs={args.epochs} train/test={args.train_samples}/{args.test_samples} "
          f"classes={args.classes} source={args.data_source}", flush=True)

    data = load_all(args)
    n_raw = int(data["raw"]["X"].shape[1])
    if args.quick:
        n_raw = 16  # 4 qubits -> 2^4 = 16，自检只需走通流程
        for key in ("raw",):
            d = data[key]
            d["X"], d["Xt"] = d["X"][:, :n_raw], d["Xt"][:, :n_raw]
    print(f"data: pca4={data['pca4']['X'].shape} pca10={data['pca10']['X'].shape} "
          f"raw={n_raw}d, source={data['raw']['meta']['source']}", flush=True)

    runs: dict[str, dict] = {sp.key: {} for sp in all_specs}
    payload: dict[str, Any] = {
        "config": {k: v for k, v in vars(args).items() if not k.startswith("_")},
        "data_meta": {k: v["meta"] for k, v in data.items()},
        "specs": [asdict(s) for s in all_specs],
        "runs": runs,
        "controlled_checks": {},
        "caveat": (
            "Fixed data split (seed=0) across all groups and seeds: variance reflects "
            "initialization and mini-batch shuffling only. Quantum weights are reset with "
            "set_seed BEFORE model construction and verified element-wise. "
            "Statevector simulation only; amplitude state preparation is not free on real hardware. "
            "Not an equal-parameter-budget comparison; classical MLP-784 has far more parameters."
        ),
    }

    for seed in seeds:
        sigs: dict[str, dict[str, torch.Tensor]] = {}
        for sp in all_specs:
            args._seed = seed
            model = make_model(sp, args, n_raw, data)
            sigs[sp.key] = init_signature(model)
            d = data[sp.data]
            print(f"  seed={seed} {sp.key:20s} ...", end=" ", flush=True)
            h = train_classifier(
                model, d["X"], d["y"], d["Xt"], d["yt"],
                epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                seed=seed, eval_batch_size=args.batch_size,
                verbose=False, eval_train=False,
            )
            curve = [round(v, 4) for v in h["test_acc"]]
            rec = {
                "test_acc_curve": curve,
                "acc_at_report": curve[min(args.report_epoch, len(curve)) - 1],
                "final_test_acc": curve[-1],
                "final_loss": round(float(h["train_loss"][-1]), 4),
                "seconds": round(float(h["total_seconds"]), 2),
                "params": int(count_parameters(model)),
                "quantum_params": int(n_quantum_params(sp.qubits, args.layers)) if sp.qubits else 0,
            }
            runs[sp.key][str(seed)] = rec
            print(f"acc@8={rec['acc_at_report']:.4f} final={rec['final_test_acc']:.4f} "
                  f"loss={rec['final_loss']:.4f} {rec['seconds']}s", flush=True)
            out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=float),
                                encoding="utf-8")

        # 受控性断言：同 qubits 的量子组必须拿到完全相同的初始量子权重
        checks: dict[str, bool] = {}
        for n_q in (4, 10):
            group = [k for k, s in sigs.items() if "q" in s and s["q"].shape[1] == n_q]
            ref = sigs[group[0]]["q"]
            checks[f"q{n_q}_weights_identical"] = all(
                torch.equal(ref, sigs[k]["q"]) for k in group)
        checks["A10_vs_Amp10_head_identical"] = torch.equal(
            sigs["A10_linear"]["head0"], sigs["Amp10_linear"]["head0"])
        payload["controlled_checks"][str(seed)] = checks
        bad = [k for k, ok in checks.items() if not ok]
        print(f"  controlled checks seed={seed}: {'PASS' if not bad else 'FAIL ' + ','.join(bad)}",
              flush=True)
        if bad:
            raise RuntimeError(f"受控性断言失败 seed={seed}: {bad}")

    summary, extra = summarize(runs, all_specs, args.report_epoch)
    payload["summary"] = summary
    payload["paired_differences"] = extra["pairs"]
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=float),
                        encoding="utf-8")
    if args.archive:
        arc = ROOT / args.archive
        arc.parent.mkdir(parents=True, exist_ok=True)
        arc.write_text(out_path.read_text(encoding="utf-8"), encoding="utf-8")

    figs = make_figures(extra["curves"], summary, ROOT / args.fig_dir)
    print("\n== 汇总（均值 ± 标准差，n=%d）==" % len(seeds), flush=True)
    rows = []
    for sp in all_specs:
        s = summary[sp.key]
        rows.append([sp.label, s["params"],
                     f"{s['acc_at_report']['mean']:.4f} ± {s['acc_at_report']['std']:.4f}",
                     f"{s['final_acc']['mean']:.4f} ± {s['final_acc']['std']:.4f}",
                     f"{s['final_loss']['mean']:.4f}",
                     f"{s['seconds']['mean']:.1f}s"])
    print(fmt_table(rows, ["模型", "参数", f"acc@{args.report_epoch}", "最终 acc", "loss", "耗时"]))
    print("\n== 关键配对差值（百分点，95% CI）==", flush=True)
    for name, p in extra["pairs"].items():
        if p["ci95_pt"] is None:
            ci = "CI[样本不足/无方差]"
            flag = "不适用"
        else:
            ci = f"CI[{p['ci95_pt'][0]:+.2f},{p['ci95_pt'][1]:+.2f}]"
            flag = "不显著" if p["ci_includes_zero"] else "显著"
        print(f"  {name:44s} {p['mean_pt']:+6.2f} ± {p['std_pt']:5.2f} "
              f"{ci} {flag} per-seed={p['per_seed_pt']}", flush=True)
    print(f"\nsaved: {out_path}")
    if args.archive:
        print(f"saved: {ROOT / args.archive} (归档到仓库，可核对)")
    for f in figs:
        print(f"saved: {f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[train_multiseed] 实验失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        raise
