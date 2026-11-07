"""全量 MNIST（60000 训练 / 10000 测试）上的 VQC 与经典对照。

前面五轮都只用 600 / 1800 张训练图；本脚本回答「用满全量数据能到多少」。
真实 IDX 全量读取，逐样本 L2 归一化（与 load_mnist_raw 一致），
训练时用全量训练集，每 epoch 在 1000 张均衡监控子集上报进度，
**最终准确率一律在全量 10000 张测试集上评估一次**。

组：
  q165    Amp10 / 165 维读出（ZZ+XX+YY 全两体）  全量
  q40     Amp10 / 40 维读出                      全量
  pca165  经典 PCA-165 + 同结构 MLP 头           全量（近似等参数）
  mlp784  经典 MLP-784                          全量
  probe   冻结随机酉 165 维读出的可分性上界（不训练量子层）
  s*      40 维读出、1 epoch 的规模曲线 600 / 3000 / 12000 / 60000

工程要点：默认 ``default.qubit + backprop + batch 128``；实测
``lightning.qubit`` 不支持该线路的 backprop，adjoint 仅 11.9 samples/s
（default.qubit 为 98.8），故不使用。多进程每 worker 独立加载数据、
构造模型前 set_seed，保证初始权重受控。

用法：
  python experiments/train_full_mnist.py --data-source idx --workers 6 \
      --archive docs/full_metrics.json --fig-dir docs/figures
  python experiments/train_full_mnist.py --quick   # 合成数据自检，不用于结论
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qml_lab.data import fetch_mnist_raw, l2_normalize, stratified_subset  # noqa: E402
from qml_lab.models import MLPClassifier, RichVQCClassifier  # noqa: E402
from qml_lab.quantum import n_readout  # noqa: E402
from qml_lab.train import evaluate_accuracy, train_classifier  # noqa: E402
from qml_lab.utils import count_parameters, set_seed  # noqa: E402

NQ = 10
OBS40 = "z+x+y+zz"
OBS165 = "z+x+y+zzall+xxall+yyall"
_CACHE: dict = {}


@dataclass(frozen=True)
class Group:
    key: str
    kind: str          # amp | pca | mlp | probe
    obs: str
    n_train: int
    epochs: int
    hidden: int
    label: str


def build_groups(args) -> dict[str, Group]:
    ep = args.epochs
    gs = {
        "q165": Group("q165", "amp", OBS165, args.train_samples, ep, 32,
                      "Amp10 / 165-dim readout / full MNIST"),
        "q40": Group("q40", "amp", OBS40, args.train_samples, ep, 32,
                     "Amp10 / 40-dim readout / full MNIST"),
        "pca165": Group("pca165", "pca", "", args.train_samples, ep, 32,
                        "classical PCA-165 + same MLP head / full MNIST"),
        "mlp784": Group("mlp784", "mlp", "", args.train_samples, ep, 32,
                        "classical MLP-784 / full MNIST"),
        "probe": Group("probe", "probe", OBS165, args.train_samples, 0, 32,
                       "frozen random-unitary 165-dim readout upper bound"),
    }
    for n in [int(s) for s in args.scale_points.split(",") if s.strip()]:
        if n <= args.train_samples:
            gs[f"s{n}"] = Group(f"s{n}", "amp", OBS40, n, 1, 32,
                                f"40-dim readout / {n} images / 1 epoch")
    return gs


def l2(X: np.ndarray) -> np.ndarray:
    return l2_normalize(X)


def get_data(args) -> dict:
    """真实 IDX 全量；每进程独立缓存，避免 Windows spawn 传大数组。"""
    key = (args.data_source, args.data_dir, args.monitor_samples, args.quick)
    if key in _CACHE:
        return _CACHE[key]
    dd = Path(args.data_dir)
    data_dir = str(dd if dd.is_absolute() else ROOT / dd)
    if args.quick:
        rng = np.random.default_rng(0)
        n, nt = 400, 100
        X = rng.normal(0, 1, (n, 784)).astype(np.float32)
        y = rng.integers(0, 4, n).astype(np.int64)
        Xt = rng.normal(0, 1, (nt, 784)).astype(np.float32)
        yt = rng.integers(0, 4, nt).astype(np.int64)
        Xr, yr, Xe, ye = X, y, Xt, yt
        source = "synthetic"
    else:
        Xr, yr, Xe, ye = fetch_mnist_raw(data_dir, source=args.data_source, seed=0)
        source = args.data_source
    Xr, Xe = l2(Xr), l2(Xe)

    # 每 epoch 用的均衡监控子集（从全量测试集按类取前 k 个）
    k = max(1, args.monitor_samples // args.classes)
    idx = []
    for c in range(args.classes):
        pos = np.flatnonzero(np.asarray(ye) == c)[:k]
        idx.append(pos)
    mon = np.concatenate(idx)
    if args.train_samples < len(Xr):
        X_sub, y_sub = stratified_subset(Xr, yr, args.train_samples,
                                         classes=args.classes, seed=0)
    else:
        X_sub = np.ascontiguousarray(Xr[:args.train_samples])
        y_sub = np.ascontiguousarray(np.asarray(yr)[:args.train_samples])
    d = {
        "X": X_sub,
        "y": y_sub,
        "Xtest_full": Xe, "ytest_full": ye,
        "Xmon": np.ascontiguousarray(Xe[mon]),
        "ymon": np.ascontiguousarray(np.asarray(ye)[mon]),
        "source": source,
        "pool_train": int(len(Xr)), "pool_test": int(len(Xe)),
    }
    _CACHE[key] = d
    return d


def build_model(g: Group, dim_in: int, seed: int, args):
    """构造模型之前重置种子：TorchLayer 在构造时消耗 torch 全局 RNG。"""
    set_seed(seed)
    if g.kind == "amp":
        return RichVQCClassifier(
            NQ, args.classes, args.layers, encoding="amplitude",
            observables=g.obs, n_features=dim_in, head="mlp",
            scale=5.0, hidden=g.hidden)
    return MLPClassifier(dim_in, args.classes, hidden=g.hidden)


def run_probe(g: Group, d: dict, args) -> dict:
    """随机酉下 165 维读出，全量特征上的冻结可分性上界（统一 StandardScaler 口径）。"""
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier as SkMLP
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from qml_lab.quantum import make_rich_layer

    torch.set_num_threads(max(1, args.torch_threads))
    set_seed(0)
    layer = make_rich_layer(NQ, args.layers, encoding="amplitude",
                            observables=OBS165, n_features=784).eval()

    def feat(X):
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 256):
                out.append(layer(torch.as_tensor(np.asarray(X[i:i + 256]))).numpy())
        return np.concatenate(out, axis=0).astype(np.float64)

    t0 = time.perf_counter()
    Ftr, Fte = feat(d["X"]), feat(d["Xtest_full"])
    out = {}
    for name, clf in [
        ("logreg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=1.0))),
        ("mlp", make_pipeline(StandardScaler(), SkMLP(hidden_layer_sizes=(64, 32), max_iter=300, random_state=0))),
    ]:
        clf.fit(Ftr, d["y"])
        out[name] = round(float(np.mean(clf.predict(Fte) == d["ytest_full"])), 4)
    return {"group": "probe", "seed": 0, "final_test_acc": max(out.values()),
            "probes": out, "seconds": round(time.perf_counter() - t0, 2),
            "note": "冻结随机酉（非训练后权重，StandardScaler 统一口径）；全量 10000 张测试集"}


def run_task(task: tuple) -> dict:
    g_key, seed, argv = task
    args = argparse.Namespace(**argv)
    import torch
    torch.set_num_threads(max(1, args.torch_threads))
    gs = build_groups(args)
    g = gs[g_key]
    d = get_data(args)
    if g.kind == "probe":
        return run_probe(g, d, args)

    n = min(g.n_train, len(d["X"]))
    if n < len(d["X"]):
        Xtr, ytr = stratified_subset(d["X"], d["y"], n, classes=args.classes, seed=0)
    else:
        Xtr = np.ascontiguousarray(d["X"][:n])
        ytr = np.ascontiguousarray(d["y"][:n])
    Xmon, ymon = d["Xmon"], d["ymon"]
    dim_in = int(Xtr.reshape(len(Xtr), -1).shape[1])
    if g.kind == "pca":
        # PCA 只在训练集 fit，监控集与全量测试集共用同一次变换
        from sklearn.decomposition import PCA
        dim_in = min(n_readout(NQ, OBS165), n - 1, Xtr.shape[1])
        pca = PCA(n_components=dim_in, random_state=0).fit(Xtr)
        Xtr = pca.transform(Xtr).astype(np.float32)
        Xmon = pca.transform(Xmon).astype(np.float32)
        Xtest_full = pca.transform(d["Xtest_full"]).astype(np.float32)
    else:
        Xtest_full = d["Xtest_full"]

    model = build_model(g, dim_in, seed, args)
    q = getattr(model, "q", None)
    init_hash = None
    if q is not None:
        init_hash = hashlib.sha256(
            q.layer.weights.detach().cpu().numpy().tobytes()).hexdigest()

    t0 = time.perf_counter()
    h = train_classifier(
        model, Xtr, ytr, Xmon, ymon, epochs=g.epochs,
        batch_size=args.batch_size, lr=args.lr, seed=seed,
        eval_batch_size=args.batch_size, verbose=False, eval_train=False)
    train_seconds = round(time.perf_counter() - t0, 2)
    full_acc = float(evaluate_accuracy(model, Xtest_full, d["ytest_full"],
                                       batch_size=args.batch_size))
    result = {
        "group": g_key, "seed": seed,
        "final_test_acc": round(full_acc, 4),
        "monitor_acc_curve": [round(float(v), 4) for v in h["test_acc"]],
        "final_monitor_acc": round(float(h["final_test_acc"]), 4),
        "final_loss": round(float(h["train_loss"][-1]), 4),
        "seconds": round(time.perf_counter() - t0, 2),
        "train_seconds": train_seconds,
        "params": int(count_parameters(model)),
        "readout_dim": int(getattr(model, "readout_dim", 0)),
        "n_train": n, "dim_in": dim_in,
        "init_weight_sha256": init_hash,
    }
    print(f"[done] {g_key:10s} seed={seed} full_acc={full_acc:.4f} "
          f"monitor={result['final_monitor_acc']:.4f} "
          f"{result['seconds']}s", flush=True)
    return result


def summarize(runs: list[dict]) -> dict:
    from statistics import mean, stdev
    grouped: dict[str, list[float]] = {}
    for r in runs:
        grouped.setdefault(r["group"], []).append(r["final_test_acc"])
    out = {}
    for k, vals in grouped.items():
        out[k] = {
            "mean": round(mean(vals), 4),
            "std": round(stdev(vals), 4) if len(vals) > 1 else 0.0,
            "values": [round(v, 4) for v in vals], "n": len(vals),
        }
    return out


def make_figure(result: dict, fig_dir: str, fig_name: str = "full_mnist.png") -> int:
    """从归档结果画图；可用 --redraw-from 单独调用，不必重跑 40 分钟实验。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stats = result["summary"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    names = ["q165", "q40", "pca165", "mlp784", "probe"]
    names = [n for n in names if n in stats]
    vals = [stats[n]["mean"] for n in names]
    errs = [stats[n]["std"] for n in names]
    colors = ["#7F77DD" if n.startswith("q") else
              "#1D9E75" if n == "probe" else "#378ADD" for n in names]
    axes[0].bar(names, vals, yerr=errs, color=colors, capsize=4)
    for i, (v, e) in enumerate(zip(vals, errs)):
        axes[0].text(i, v + e + 0.012, f"{v:.4f}", ha="center", fontsize=9)
    axes[0].set_ylim(0, 1.08)
    axes[0].set_ylabel("test acc (10000 images)")
    axes[0].set_title("Full MNIST: quantum vs classical")
    axes[0].grid(axis="y", alpha=0.3)

    sp = sorted([k for k in stats if k.startswith("s")],
                key=lambda k: int(k[1:]))
    if sp:
        xs = [int(k[1:]) for k in sp]
        ys = [stats[k]["mean"] for k in sp]
        es = [stats[k]["std"] for k in sp]
        axes[1].errorbar(xs, ys, yerr=es, fmt="o-", color="#7F77DD", capsize=3)
        for x, y, e in zip(xs, ys, es):
            axes[1].annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                             xytext=(6, 8), fontsize=9)
        axes[1].set_xscale("log")
        axes[1].set_ylim(0.15, 1.0)
        axes[1].set_xlabel("training images (40-dim readout, 1 epoch)")
        axes[1].set_ylabel("test acc")
        axes[1].set_title("Data scaling: 600 -> 60000")
        axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fd = Path(fig_dir)
    if not fd.is_absolute():
        fd = ROOT / fd
    fd.mkdir(parents=True, exist_ok=True)
    fig.savefig(fd / fig_name, dpi=130)
    plt.close(fig)
    print(f"figure -> {fd / fig_name}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-source", default="idx")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--train-samples", type=int, default=60000)
    ap.add_argument("--monitor-samples", type=int, default=1000)
    ap.add_argument("--classes", type=int, default=10)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--scale-points", default="600,3000,12000,60000")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--torch-threads", type=int, default=4)
    ap.add_argument("--archive", default="docs/full_metrics.json")
    ap.add_argument("--fig-dir", default="docs/figures")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--redraw-from", default="",
                    help="只从已有归档 JSON 重画图，不重跑实验")
    args = ap.parse_args()

    if args.redraw_from:
        src = Path(args.redraw_from)
        if not src.is_absolute():
            src = ROOT / src
        return make_figure(json.loads(src.read_text(encoding="utf-8")),
                           args.fig_dir)

    if args.quick:
        args.data_source = "synthetic"
        args.train_samples, args.monitor_samples, args.classes = 120, 40, 4
        args.epochs, args.batch_size = 1, 32
        args.seeds, args.scale_points, args.workers = "0", "60", 1
        args.torch_threads = 1
        if args.archive == "docs/full_metrics.json":
            args.archive = "results/quick_full_metrics.json"
        if args.fig_dir == "docs/figures":
            args.fig_dir = "results"

    t0 = time.time()
    gs = build_groups(args)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    argv = {k: v for k, v in vars(args).items()}
    tasks = []
    for key, g in gs.items():
        ss = [0] if g.kind == "probe" else seeds
        for s in ss:
            tasks.append((key, s, argv))
    print(f"full MNIST: {len(tasks)} tasks, workers={args.workers}, "
          f"train={args.train_samples}, epochs={args.epochs}, "
          f"batch={args.batch_size}", flush=True)

    runs = []
    if args.workers <= 1:
        for t in tasks:
            runs.append(run_task(t))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run_task, t): t for t in tasks}
            for f in as_completed(futs):
                try:
                    runs.append(f.result())
                except Exception as e:
                    t = futs[f]
                    print(f"[FAIL] {t[0]} seed={t[1]}: "
                          f"{type(e).__name__}: {e}", flush=True)

    if not runs:
        print("no runs finished", flush=True)
        return 1

    stats = summarize(runs)
    print("\n=== summary (full 10000-image test set) ===", flush=True)
    for k in sorted(stats):
        s = stats[k]
        print(f"  {k:10s} {s['mean']:.4f} ± {s['std']:.4f}  "
              f"(n={s['n']}, {s['values']})", flush=True)

    result = {
        "purpose": "全量 MNIST（60000 训练 / 10000 测试）上的 VQC 与等预算经典对照",
        "config": {k: v for k, v in vars(args).items()},
        "data": {"source": args.data_source,
                 "note": "真实 IDX 全量；每 epoch 用 1000 张均衡监控子集，"
                         "最终准确率一律在全量 10000 张测试集评估"},
        "groups": {k: dict(g.__dict__) for k, g in gs.items()},
        "summary": stats,
        "runs": sorted(runs, key=lambda r: (r["group"], r["seed"])),
        "wall_seconds": round(time.time() - t0, 2),
        "caveat": (
            "n=3 seed，未做多重比较校正；监控子集与全量测试集数值会有差异，"
            "以 final_test_acc 为准；probe 是冻结随机酉的可分性上界，不是端到端结果；"
            "全量用 default.qubit + backprop（lightning 不支持该线路 backprop，"
            "adjoint 实测慢 8 倍）；真机上振幅编码 O(2^n) 状态制备与 165 个观测量"
            "的 shots 成本不在此模拟结果内。"
        ),
    }
    if args.archive:
        out = Path(args.archive)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        print(f"archived -> {out.relative_to(ROOT)}", flush=True)

    if not args.no_figures:
        try:
            fname = "quick_mnist_full.png" if args.quick else "full_mnist.png"
            make_figure(result, args.fig_dir, fig_name=fname)
        except Exception as e:
            print(f"[warn] figure failed: {e}", flush=True)

    print(f"done in {result['wall_seconds']} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
