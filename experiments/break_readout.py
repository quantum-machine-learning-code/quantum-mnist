#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""解开「只有 n 个有界读数」这个瓶颈：多可观测量 / 数据重上传 / 单独 lr。

起因（docs/REVIEW.md 第 8 节）：5 seed 实测把训练好的量子层冻结，只取 n 个
PauliZ 期望交给任意经典强分类器，最高只有 0.6790，而原始像素是 0.8550。
约 17.6 pt 的可分信息丢在「单次编码 → 酉变换 → 只测 n 个有界数」这条链路上。

本脚本不猜原因，把三条候选解锁手段**分别单独加回去**，看各值多少分：

    z_base   : Amplitude-10 / Z-only / MLP x5        （复现主结果 0.664，同批次基准）
    zx       : 同上，读出改为 Z+X（20 维）
    zfull    : 同上，读出改为 Z+X+Y+ZZ（40 维，单比特 Bloch + 两体关联）
    a10      : Angle-10 / Z-only（PCA-10 输入）
    a10_reup : 同上但每层重新嵌入输入（data re-uploading，参数量不变）
    a10_full : Angle-10 / 重上传 + 40 维读出
    hyb_l    : lightCNN -> 8q Z-only，量子层用单独 lr
    hyb_l_rich : lightCNN -> 8q 重上传 + Z+X，量子层用单独 lr
    hyb_l_nq : 同一 CNN 前端跳过量子层（等架构经典对照）
    cnn_l    : 纯经典 CNN 直接分类
    mlp40    : PCA-40 + 与 zfull 完全相同结构的 MLP 头（等分类头参数预算）

另做**冻结读出探针**：训练完后把量子读出向量 z 交给 sklearn 的线性 / RBF /
MLP 分类器，测 z 里还剩多少可分信息；并用原始 784 维像素做同样探针作为
「输入信息上限」。探针是冻结特征的可分性上界，不是端到端训练结果。

方法学约定（docs/REVIEW.md）：
  * 5 个 seed，报均值 ± 标准差与配对差值 95% CI；数据切分固定 seed=0；
  * 每个模型**构造前** set_seed（TorchLayer 用 torch 全局 RNG 初始化权重）；
  * 断言同 qubits 各组量子初始权重逐元素相同；
  * quick 模式仅离线流程自检，结果不得用于任何结论。

运行：
  python experiments/break_readout.py --data-source idx --out results \
      --archive docs/readout_metrics.json --fig-dir docs/figures
  python experiments/break_readout.py --quick
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qml_lab.data import load_mnist, load_mnist_images, load_mnist_raw  # noqa: E402
from qml_lab.models import (  # noqa: E402
    CNNClassifier, HybridRichClassifier, MLPClassifier, RichVQCClassifier,
)
from qml_lab.quantum import n_readout  # noqa: E402
from qml_lab.train import train_classifier  # noqa: E402
from qml_lab.utils import count_parameters, ensure_dir, set_seed  # noqa: E402

FULL_OBS = "z+x+y+zz"


@dataclass(frozen=True)
class Group:
    key: str
    label: str
    kind: str            # amp / angle / hybrid / hybrid_nq / cnn / mlp
    data: str            # raw / pca / pca_full / img
    obs: str = "z"
    reupload: bool = False
    q_lr: float | None = None


def groups() -> list[Group]:
    return [
        Group("z_base", "Amp10 Z-only (baseline)", "amp", "raw"),
        Group("zx", "Amp10 Z+X", "amp", "raw", "z+x"),
        Group("zfull", "Amp10 Z+X+Y+ZZ", "amp", "raw", FULL_OBS),
        Group("a10", "Angle10 Z-only", "angle", "pca"),
        Group("a10_reup", "Angle10 Z + reupload", "angle", "pca", "z", True),
        Group("a10_full", "Angle10 full + reupload", "angle", "pca", FULL_OBS, True),
        Group("hyb_l", "lightCNN + 8q Z (q_lr)", "hybrid", "img", "z", False, 0.001),
        Group("hyb_l_rich", "lightCNN + 8q rich (q_lr)", "hybrid", "img", "z+x", True, 0.001),
        Group("hyb_l_nq", "lightCNN, no VQC", "hybrid_nq", "img"),
        Group("cnn_l", "lightCNN direct", "cnn", "img"),
        Group("mlp40", "PCA-full + same MLP head", "mlp", "pca_full"),
    ]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="分解量子读出瓶颈")
    p.add_argument("--seeds", type=str, default="0,1,2,3,4")
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--train-samples", type=int, default=600)
    p.add_argument("--test-samples", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--quantum-lr", type=float, default=0.001)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--classes", type=int, default=10)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--amp-qubits", type=int, default=10)
    p.add_argument("--angle-qubits", type=int, default=10)
    p.add_argument("--hybrid-qubits", type=int, default=8)
    p.add_argument("--data-source", type=str, default="idx")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--out", type=str, default="results")
    p.add_argument("--archive", type=str, default="docs/readout_metrics.json")
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
    """切分固定 seed=0；raw / pca / img 抽到的必须是同一批图。"""
    kw = dict(classes=args.classes, source=args.data_source,
              data_dir=args.data_dir, seed=0)
    raw, yr, rawt, yt, m1 = load_mnist_raw(
        n_train=args.train_samples, n_test=args.test_samples, **kw)
    pca, yp, pcat, ypt, m2 = load_mnist(
        n_train=args.train_samples, n_test=args.test_samples,
        n_features=args.angle_qubits, **kw)
    n_full = n_readout(args.amp_qubits, FULL_OBS)
    full, yf, fullt, yft, m3 = load_mnist(
        n_train=args.train_samples, n_test=args.test_samples,
        n_features=n_full, **kw)
    img, yi, imgt, yit, m4 = load_mnist_images(
        n_train=args.train_samples, n_test=args.test_samples, **kw)

    assert np.array_equal(yr, yp) and np.array_equal(yr, yi), "三条路径标签不一致"
    assert np.array_equal(yt, ypt) and np.array_equal(yt, yit), "测试标签不一致"
    assert np.array_equal(yr, yf), "PCA-full 标签不一致"
    if args.data_source != "synthetic":
        flat = img.reshape(len(img), -1) * 255.0
        assert np.allclose(raw, _l2(flat), atol=1e-4), "raw 与 img 像素不一致"

    if args.quick:
        k = 2 ** args.amp_qubits
        raw, rawt = _l2(raw[:, :k]), _l2(rawt[:, :k])

    return {
        "raw": dict(X=raw, y=yr, Xt=rawt, yt=yt),
        "pca": dict(X=pca, y=yp, Xt=pcat, yt=ypt),
        "pca_full": dict(X=full, y=yf, Xt=fullt, yt=yft),
        "img": dict(X=img, y=yi, Xt=imgt, yt=yit),
        "meta": {"raw": m1, "pca": m2, "pca_full": m3, "img": m4,
                 "same_split_asserted": True,
                 "readout_dims": {s: n_readout(args.amp_qubits, s)
                                  for s in ("z", "z+x", FULL_OBS)}},
    }


def make_model(g: Group, args, data: dict, seed: int):
    """构造模型前重置种子 —— TorchLayer 的权重在构造时消耗 torch 全局 RNG。"""
    set_seed(seed)
    if g.kind in ("amp", "angle"):
        nq = args.amp_qubits if g.kind == "amp" else args.angle_qubits
        return RichVQCClassifier(
            nq, args.classes, args.layers,
            encoding="amplitude" if g.kind == "amp" else "angle",
            observables=g.obs, n_features=int(data[g.data]["X"].shape[1]),
            reupload=g.reupload, head="mlp", scale=5.0, hidden=args.hidden,
        )
    if g.kind == "hybrid":
        return HybridRichClassifier(
            n_qubits=args.hybrid_qubits, n_classes=args.classes,
            n_layers=args.layers, hidden=args.hidden, quantum=True,
            head="mlp", scale=5.0, frontend="light",
            observables=g.obs, reupload=g.reupload,
        )
    if g.kind == "hybrid_nq":
        return HybridRichClassifier(
            n_qubits=args.hybrid_qubits, n_classes=args.classes,
            n_layers=args.layers, hidden=args.hidden, quantum=False,
            head="mlp", scale=1.0, frontend="light",
        )
    if g.kind == "cnn":
        return CNNClassifier(args.classes, frontend="light")
    return MLPClassifier(int(data[g.data]["X"].shape[1]), args.classes,
                         hidden=args.hidden)


def quantum_weights(model):
    q = getattr(model, "q", None)
    return None if q is None else q.layer.weights.detach().clone()


def _readout(model, X, batch: int = 64) -> np.ndarray:
    """提取量子层读出向量 z。"""
    model.eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.as_tensor(X[i:i + batch], dtype=torch.float32)
            if hasattr(model, "quantum_features"):
                z = model.quantum_features(xb)
            else:
                z = model.q(xb)
            chunks.append(z.to(torch.float32).cpu().numpy())
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


def run_probes(model, d: dict) -> dict:
    Ztr, Zte = _readout(model, d["X"]), _readout(model, d["Xt"])
    out = {"readout_dim": int(Ztr.shape[1]),
           "readout_std": round(float(Ztr.std()), 4)}
    for name, factory in _probes().items():
        clf = factory()
        clf.fit(Ztr, d["y"])
        out[name] = round(float(clf.score(Zte, d["yt"])), 4)
    return out


def pixel_probe(d: dict) -> dict:
    X, Xt = np.asarray(d["X"]), np.asarray(d["Xt"])
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


def _paired(a: dict, b: dict, seeds: list[int]) -> dict:
    """配对差值（单位：百分点）与 t 分布 95% CI。"""
    from scipy.stats import t
    d = np.asarray([a["final_acc"]["values"][i] - b["final_acc"]["values"][i]
                    for i in range(len(seeds))], dtype=float) * 100.0
    mean, sd = float(d.mean()), float(d.std(ddof=1)) if len(d) > 1 else 0.0
    half = (float(t.ppf(0.975, len(d) - 1)) * sd / np.sqrt(len(d))
            if len(d) > 1 else 0.0)
    return {"mean_pt": round(mean, 2), "std_pt": round(sd, 2),
            "ci95_pt": [round(mean - half, 2), round(mean + half, 2)],
            "ci_includes_zero": bool(mean - half <= 0 <= mean + half),
            "per_seed_pt": [round(float(v), 2) for v in d]}


def make_figures(groups_out: dict, curves: dict, probes: dict,
                 pixel: dict, out_dir: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    keys = list(groups_out)
    means = [groups_out[k]["final_acc"]["mean"] for k in keys]
    stds = [groups_out[k]["final_acc"]["std"] for k in keys]
    colors = ["#8e8e8e", "#5b8ff9", "#5ad8a6", "#e8684a", "#e8684a",
              "#e8684a", "#9270ca", "#9270ca", "#6dc8ec", "#6dc8ec", "#6dc8ec"]

    fig, ax = plt.subplots(figsize=(13, 5.8))
    bars = ax.bar(range(len(keys)), means, yerr=stds, capsize=5,
                  color=colors[:len(keys)], edgecolor="#333333",
                  linewidth=0.6, alpha=0.9)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([groups_out[k]["label"] for k in keys],
                       rotation=20, ha="right", fontsize=8.5)
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title("Breaking the n-readout bottleneck (5 seeds, mean +/- 1 sd)")
    ax.grid(axis="y", alpha=0.3)
    for b, m, s in zip(bars, means, stds):
        ax.text(b.get_x() + b.get_width() / 2, m + s + 0.025, f"{m:.3f}",
                ha="center", fontsize=8.5)
    fig.tight_layout()
    p = out_dir / "readout_ladder.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made.append(str(p))

    # 第二张：读出维度增加时，「端到端准确率」与「冻结读出的可分性上限」怎么变
    order = [k for k in ("z_base", "zx", "zfull") if k in probes]
    if order:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        dims = [probes[k]["readout_dim"] for k in order]
        end2end = [groups_out[k]["final_acc"]["mean"] for k in order]
        best_probe = [max(probes[k][n]["mean"] for n in ("logreg", "rbf_svc", "mlp"))
                      for k in order]
        ax.plot(dims, end2end, "o-", linewidth=2, label="end-to-end test acc")
        ax.plot(dims, best_probe, "s--", linewidth=2,
                label="frozen-readout info ceiling (best probe)")
        ax.axhline(pixel["rbf_svc"], color="#888888", linestyle=":",
                   label=f"raw-pixel probe (RBF) = {pixel['rbf_svc']:.3f}")
        ax.set_xlabel("number of readout observables (amplitude-10)")
        ax.set_ylabel("test accuracy")
        ax.set_ylim(0.3, 1.0)
        ax.set_title("Does a wider readout actually keep more information?")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        for x, y in zip(dims, end2end):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                        xytext=(0, -14), ha="center", fontsize=8)
        fig.tight_layout()
        p = out_dir / "readout_info.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        made.append(str(p))

    # 第三张：振幅三组曲线
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for k in ("z_base", "zx", "zfull"):
        a = np.asarray(curves[k], dtype=float)
        xs = np.arange(1, a.shape[1] + 1)
        ax.plot(xs, a.mean(axis=0), label=groups_out[k]["label"], linewidth=2)
        if a.shape[0] > 1:
            ax.fill_between(xs, a.mean(axis=0) - a.std(axis=0, ddof=1),
                            a.mean(axis=0) + a.std(axis=0, ddof=1), alpha=0.15)
    ax.set_xlabel("epoch")
    ax.set_ylabel("test accuracy")
    ax.set_title("Amplitude-10: Z-only vs wider readouts (5 seeds)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = out_dir / "readout_curves.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made.append(str(p))
    return made


def main() -> None:
    args = parse_args()
    if args.quick:
        args.seeds, args.epochs = "0", 2
        args.train_samples, args.test_samples = 40, 20
        args.data_source = "synthetic"
        args.amp_qubits = args.angle_qubits = args.hybrid_qubits = 4
        args.out, args.fig_dir = "results", "results"
        args.archive, args.no_figures = "", True
    set_seed(0)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    data = load_all(args)
    gs = groups()
    out_dir = ensure_dir(ROOT / args.out)
    probe_keys = {"z_base", "zx", "zfull", "a10", "a10_reup", "a10_full",
                  "hyb_l_rich"}

    runs: dict[str, dict] = {}
    curves: dict[str, list] = {}
    probe_runs: dict[str, dict] = {}
    control_checks: dict[str, dict] = {}
    # 按「模型族 + qubits」分桶：混合组先构造 CNN 主干会消耗 torch RNG，
    # 它的量子初始权重本就不应与纯 VQC 组相同，但组内必须逐元素相同。
    ref_weights: dict[tuple[str, int], torch.Tensor] = {}

    for g in gs:
        d = data[g.data]
        runs[g.key], curves[g.key], probe_runs[g.key] = {}, [], {}
        for seed in seeds:
            model = make_model(g, args, data, seed)
            w = quantum_weights(model)
            if w is not None:
                # seed 必须进 key：不同 seed 的初始权重本来就应当不同
                key = (str(seed), g.kind, int(w.shape[1]))
                ok = (key not in ref_weights or
                      bool(torch.equal(w, ref_weights[key])))
                if key not in ref_weights:
                    ref_weights[key] = w
                control_checks.setdefault(str(seed), {})[g.key] = ok
                if not ok:
                    raise AssertionError(
                        f"量子初始权重未受控：{g.key} seed={seed}；"
                        "受控对照要求同族模型在同一 seed 下初始权重逐元素相同")
            h = train_classifier(
                model, d["X"], d["y"], d["Xt"], d["yt"],
                epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                seed=seed, eval_batch_size=64, verbose=False, eval_train=False,
                quantum_lr=(args.quantum_lr if g.q_lr is not None else None),
            )
            rec = {
                "final_test_acc": round(float(h["final_test_acc"]), 4),
                "final_loss": round(float(h["train_loss"][-1]), 4),
                "seconds": round(float(h["total_seconds"]), 2),
                "params": int(count_parameters(model)),
                "readout_dim": int(getattr(model, "readout_dim", 0)),
                "test_acc_curve": [round(float(v), 4) for v in h["test_acc"]],
            }
            runs[g.key][str(seed)] = rec
            curves[g.key].append(h["test_acc"])
            if g.key in probe_keys:
                probe_runs[g.key][str(seed)] = run_probes(model, d)
            print(f"  [{g.key:12s} seed {seed}] acc={rec['final_test_acc']:.4f} "
                  f"loss={rec['final_loss']:.4f} params={rec['params']} "
                  f"readout={rec['readout_dim']} {rec['seconds']}s", flush=True)

    groups_out = {}
    for g in gs:
        groups_out[g.key] = {
            "label": g.label, "kind": g.kind, "data": g.data,
            "observables": g.obs if g.kind in ("amp", "angle", "hybrid") else None,
            "reupload": g.reupload, "quantum_lr": g.q_lr,
            "epochs": args.epochs, "n_train": int(data[g.data]["X"].shape[0]),
            "params": runs[g.key][str(seeds[0])]["params"],
            "readout_dim": runs[g.key][str(seeds[0])]["readout_dim"],
            "final_acc": _stats([runs[g.key][str(s)]["final_test_acc"]
                                 for s in seeds]),
            "final_loss": _stats([runs[g.key][str(s)]["final_loss"]
                                  for s in seeds]),
        }

    probes_out = {}
    for k, per_seed in probe_runs.items():
        if not per_seed:
            continue
        probes_out[k] = {name: _stats([per_seed[str(s)][name] for s in seeds])
                         for name in ("logreg", "rbf_svc", "mlp")}
        probes_out[k]["readout_dim"] = per_seed[str(seeds[0])]["readout_dim"]

    pairs = [("zx", "z_base"), ("zfull", "z_base"), ("zfull", "zx"),
             ("a10_reup", "a10"), ("a10_full", "a10_reup"),
             ("hyb_l_rich", "hyb_l"), ("hyb_l_rich", "hyb_l_nq"),
             ("cnn_l", "hyb_l_rich")]
    paired = {f"{a} - {b}": _paired(groups_out[a], groups_out[b], seeds)
              for a, b in pairs}

    payload = {
        "config": {k: v for k, v in vars(args).items() if k not in ("quick",)},
        "seeds": seeds,
        "data_meta": data["meta"],
        "groups": groups_out,
        "readout_probes": probes_out,
        "pixel_probe": pixel_probe(data["raw"]),
        "controlled_checks": control_checks,
        "paired_differences": paired,
        "runs": runs,
        "caveat": ("探针是冻结特征 + 经典分类器的可分性上界，不是端到端训练结果；"
                   "多可观测量会增加分类头参数量，mlp40 为等分类头预算的经典参照；"
                   "振幅编码在真机上需要 O(2^n) 状态制备；n=5 未做多重比较校正。"
                   "结论性数字均为 5 seed 均值 ± 标准差。"),
    }
    if args.quick:
        payload["caveat"] = "QUICK 自检，合成数据，不得用于任何结论。" + payload["caveat"]

    out_path = out_dir / "readout_metrics.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=float),
                        encoding="utf-8")
    print(f"\nsaved: {out_path}")

    figs = []
    if not args.no_figures:
        figs = make_figures(groups_out, curves, probes_out,
                            payload["pixel_probe"], ROOT / args.fig_dir)
    if args.archive and not args.quick:
        arc = ROOT / args.archive
        arc.parent.mkdir(parents=True, exist_ok=True)
        arc.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=float),
                       encoding="utf-8")
        print(f"saved: {arc}")
    for f in figs:
        print(f"saved: {f}")

    print("\n=== 受控性断言（量子初始权重逐元素相同） ===")
    for s, checks in control_checks.items():
        print(f"  seed {s}: {sum(checks.values())}/{len(checks)} PASS"
              + ("" if all(checks.values()) else f"  FAILED={[k for k, v in checks.items() if not v]}"))

    print("\n=== 分组汇总（均值 ± 标准差） ===")
    for g in gs:
        s = groups_out[g.key]
        print(f"  {g.key:12s} {s['final_acc']['mean']:.4f} ± {s['final_acc']['std']:.4f} "
              f"| loss {s['final_loss']['mean']:.4f} | params {s['params']:6d} "
              f"| readout {s['readout_dim']:2d} | {s['label']}")
    print("\n=== 冻结读出探针（测试集 acc，5 seed 均值） ===")
    for k, v in probes_out.items():
        print(f"  {k:12s} dim={v['readout_dim']:2d} "
              f"logreg {v['logreg']['mean']:.4f} | rbf {v['rbf_svc']['mean']:.4f} "
              f"| mlp {v['mlp']['mean']:.4f}")
    pp = payload["pixel_probe"]
    print(f"  {'pixel784':12s} dim={pp['dim']} "
          f"logreg {pp['logreg']:.4f} | rbf {pp['rbf_svc']:.4f} | mlp {pp['mlp']:.4f}")
    print("\n=== 配对差值（百分点，95% CI） ===")
    for k, v in paired.items():
        flag = "不显著" if v["ci_includes_zero"] else "显著"
        print(f"  {k:28s} {v['mean_pt']:+6.2f} ± {v['std_pt']:5.2f} "
              f"CI[{v['ci95_pt'][0]:+6.2f},{v['ci95_pt'][1]:+6.2f}] {flag}")


if __name__ == "__main__":
    main()
