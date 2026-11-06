#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第五轮诊断：0.772 之后剩下的分，到底丢在哪一层？

前四轮已经证明：10 个 PauliZ 读出的可分性上限只有 0.679，扩到 40 个
观测量后上界 0.811、端到端 0.772，而原始像素探针 0.855、经典 MLP 0.864、
同样 600 张训练图的轻量 / 强 CNN 是 0.863 / 0.934。本脚本不再猜，直接用
**廉价的信息探针**把剩余 gap 拆成四个候选解释：

  1. 训练不足 —— 比较「随机初始化量子权重 / 零权重 / 训练后权重」的读出上界，
     并记录量子参数梯度范数与权重漂移（barren plateau 诊断）。
  2. 读出还不够宽 —— 把观测量从 40 扩到 75 / 165（所有两体 ZZ / XX / YY 关联），
     看冻结读出的可分性上界是否继续上升。
  3. 二次型表达力是硬上限 —— 对同一批读出加 poly2 / poly3 核探针；若高次核
     明显更好，说明「单次振幅编码 + 酉 + Pauli 期望 = 输入的二次型」本身封顶。
  4. 数据 / 归纳偏置 —— 像素探针随训练样本数 600 → 3000 → 12000 的变化，
     以及同维度 PCA 经典特征跑同一套探针；CNN 用同样 600 张图能到 0.934，
     所以若 PCA 探针也明显高于量子读出，差距应归因于表示/归纳偏置而非数据量。

口径（重要）：探针是**冻结特征 + 经典分类器**的可分性上界，不是端到端训练
结果；除「训练后权重」外，本脚本的量子权重对比是单 seed 的**示意性诊断**，
结论性数字仍以 5 seed 的 `push_ceiling.py` 为准。

运行：
  python experiments/probe_information.py --data-source idx \
      --out results --archive docs/info_metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qml_lab.data import load_mnist, load_mnist_raw  # noqa: E402
from qml_lab.models import RichVQCClassifier  # noqa: E402
from qml_lab.quantum import make_rich_layer, n_readout  # noqa: E402
from qml_lab.train import evaluate_accuracy  # noqa: E402
from qml_lab.utils import ensure_dir, set_seed  # noqa: E402

NQ = 10
N_LAYERS = 2
TRAIN_OBS = "z+x+y+zz"
SPECS = [
    "z",                                  # 10
    "z+x",                                # 20
    "z+x+y+zz",                           # 40  当前主结果
    "z+x+y+zzall",                        # 75  全部 ZZ 两体关联
    "z+x+y+zzall+xxall+yyall",            # 165 全部 ZZ/XX/YY 两体关联
]
CLASSICAL_DIMS = [10, 20, 40, 75, 165]
WEIGHT_STATES = ("random", "zero", "trained")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="量子读出的信息上界分解（诊断）")
    p.add_argument("--seeds", type=str, default="0", help="诊断用 seed；训练后权重只取第一个")
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--train-samples", type=int, default=600)
    p.add_argument("--test-samples", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--classes", type=int, default=10)
    p.add_argument("--data-source", type=str, default="idx")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--out", type=str, default="results")
    p.add_argument("--archive", type=str, default="docs/info_metrics.json")
    p.add_argument("--big-train", type=int, default=12000,
                   help="像素探针的数据规模扫描上限")
    p.add_argument("--no-classical", action="store_true", help="跳过同维度 PCA 参照")
    return p.parse_args()


def _probes(full: bool = False) -> dict:
    """冻结特征探针。full=True 时加入多项式核（检验更高次非线性是否还有信息）。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier as SkMLP
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    out = {
        "logreg": lambda: make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)),
        "rbf_svc": lambda: make_pipeline(
            StandardScaler(), SVC(kernel="rbf", C=10.0)),
        "mlp": lambda: make_pipeline(
            StandardScaler(), SkMLP(hidden_layer_sizes=(64, 32),
                                    max_iter=300, random_state=0)),
    }
    if full:
        out["poly2_svc"] = lambda: make_pipeline(
            StandardScaler(), SVC(kernel="poly", degree=2, C=10.0, coef0=1.0))
        out["poly3_svc"] = lambda: make_pipeline(
            StandardScaler(), SVC(kernel="poly", degree=3, C=10.0, coef0=1.0))
    return out


def fit_probes(Ztr: np.ndarray, ytr: np.ndarray, Zte: np.ndarray,
               yte: np.ndarray, full: bool = False) -> dict:
    out = {}
    for name, factory in _probes(full).items():
        clf = factory()
        clf.fit(Ztr, ytr)
        out[name] = round(float(clf.score(Zte, yte)), 4)
    return out


def readout_of(layer, X: np.ndarray, batch: int = 64) -> np.ndarray:
    layer.eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            z = layer(torch.as_tensor(X[i:i + batch], dtype=torch.float32))
            chunks.append(z.to(torch.float32).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def set_weights(layer, w: torch.Tensor) -> None:
    with torch.no_grad():
        layer.layer.weights.copy_(w.to(torch.float32))


def geometry(Z: np.ndarray, y: np.ndarray) -> dict:
    """读出的几何诊断：有效秩、奇异值谱、Fisher 类间/类内散度比。"""
    A = np.asarray(Z, dtype=float)
    A = A - A.mean(axis=0, keepdims=True)
    s = np.linalg.svd(A, compute_uv=False)
    var = s ** 2
    total = float(var.sum())
    cum = np.cumsum(var) / max(total, 1e-12)
    eff99 = int(np.searchsorted(cum, 0.99) + 1)
    # Fisher 判别比：类间散度 / 类内散度（都按自由度归一）
    y = np.asarray(y)
    labels = np.unique(y)
    mu = A.mean(axis=0, keepdims=True)
    between, within = 0.0, 0.0
    for c in labels:
        Ac = A[y == c]
        mc = Ac.mean(axis=0, keepdims=True)
        between += Ac.shape[0] * float(((mc - mu) ** 2).sum())
        within += float(((Ac - mc) ** 2).sum())
    between /= max(len(labels) - 1, 1)
    within /= max(A.shape[0] - len(labels), 1)
    keep = min(12, len(s))
    return {
        "effective_rank_99": eff99,
        "singular_values_head": [round(float(v), 4) for v in s[:keep]],
        "variance_first_component": round(float(var[0] / max(total, 1e-12)), 4),
        "fisher_ratio": round(float(between / max(within, 1e-12)), 4),
    }


def train_and_track(model, d: dict, args, seed: int) -> tuple[torch.Tensor, dict]:
    """训练 40 维读出模型，并额外记录量子参数的梯度范数与权重漂移。

    梯度范数是 barren plateau 的直接证据；权重漂移说明量子层到底有没有在学。
    """
    set_seed(seed)
    X = torch.as_tensor(d["X"], dtype=torch.float32)
    y = torch.as_tensor(d["y"], dtype=torch.long)
    Xt = torch.as_tensor(d["Xt"], dtype=torch.float32)
    yt = torch.as_tensor(d["yt"], dtype=torch.long)
    w0 = model.q.layer.weights.detach().clone()
    qparams = [p for p in model.q.parameters()]
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    hist = {"test_acc": [], "train_loss": [], "q_grad_norm": [], "q_drift": []}
    n = len(y)
    t0 = time.perf_counter()

    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n)
        running, seen, first_gn = 0.0, 0, 0.0
        for start in range(0, n, args.batch_size):
            idx = perm[start:start + args.batch_size]
            opt.zero_grad()
            loss = loss_fn(model(X[idx]), y[idx])
            loss.backward()
            if start == 0:
                first_gn = float(torch.sqrt(sum(
                    (p.grad.detach() ** 2).sum() for p in qparams
                    if p.grad is not None)))
            opt.step()
            running += loss.item() * idx.numel()
            seen += idx.numel()
        model.eval()
        with torch.no_grad():
            drift = float((model.q.layer.weights - w0).norm())
        acc = evaluate_accuracy(model, Xt, yt, batch_size=64)
        hist["test_acc"].append(round(acc, 4))
        hist["train_loss"].append(round(running / max(seen, 1), 4))
        hist["q_grad_norm"].append(round(first_gn, 6))
        hist["q_drift"].append(round(drift, 6))
        print(f"  [train] ep {ep:>2}/{args.epochs} loss {hist['train_loss'][-1]:.4f} "
              f"acc {acc:.4f} |q_grad| {first_gn:.5f} drift {drift:.4f}", flush=True)

    hist["final_test_acc"] = hist["test_acc"][-1]
    hist["seconds"] = round(time.perf_counter() - t0, 2)
    hist["quantum_param_count"] = int(sum(p.numel() for p in qparams))
    return model.q.layer.weights.detach().clone(), hist


def classical_reference(args, dims: list[int], d_test: tuple) -> dict:
    """同维度 PCA 经典特征跑同一套探针：判断量子链路是保留还是损坏了信息。"""
    out = {}
    for dim in dims:
        Xp, yp, Xpt, ypt, meta = load_mnist(
            n_train=args.train_samples, n_test=args.test_samples,
            n_features=dim, classes=args.classes, source=args.data_source,
            data_dir=args.data_dir, seed=0)
        # _select 的 rng 消耗只取决于样本数/类别/seed，与 n_features 无关：
        # 不同 PCA 维度抽到的是同一批图，因此直接断言标签与主数据逐位相同。
        _, y_fixed = d_test
        assert np.array_equal(ypt, y_fixed), "PCA 路径测试标签与主数据不一致"
        out[str(dim)] = {
            "dim": int(Xp.shape[1]),
            "source": meta["source"],
            **fit_probes(Xp, yp, Xpt, ypt, full=False),
        }
        print(f"  [classical] PCA-{dim:3d} rbf {out[str(dim)]['rbf_svc']:.4f} "
              f"logreg {out[str(dim)]['logreg']:.4f}", flush=True)
    return out


def pixel_scaling(args, Xt: np.ndarray, yt: np.ndarray) -> dict:
    """像素探针随训练样本数的变化：数据规模到底是不是瓶颈。"""
    Xbig, ybig, _, _, meta = load_mnist_raw(
        n_train=args.big_train, n_test=args.test_samples,
        classes=args.classes, source=args.data_source,
        data_dir=args.data_dir, seed=0)
    rng = np.random.default_rng(123)
    perm = rng.permutation(len(Xbig))
    Xbig, ybig = Xbig[perm], ybig[perm]
    out = {"source": meta["source"], "test_fixed": True, "points": {}}
    for n in (600, 3000, args.big_train):
        n = min(n, len(Xbig))
        if n in out["points"]:
            continue
        out["points"][str(n)] = fit_probes(Xbig[:n], ybig[:n], Xt, yt, full=False)
        print(f"  [pixels] n_train {n:5d} rbf "
              f"{out['points'][str(n)]['rbf_svc']:.4f} "
              f"logreg {out['points'][str(n)]['logreg']:.4f}", flush=True)
    return out


def main() -> None:
    args = parse_args()
    seed = int(args.seeds.split(",")[0])
    kw = dict(classes=args.classes, source=args.data_source,
              data_dir=args.data_dir, seed=0)
    Xtr, ytr, Xte, yte, meta = load_mnist_raw(
        n_train=args.train_samples, n_test=args.test_samples, **kw)
    d = {"X": Xtr, "y": ytr, "Xt": Xte, "yt": yte}

    print("=== 1. 像素探针（固定 600/200 测试集） ===", flush=True)
    pixel = {"dim": int(Xtr.shape[1]),
             **fit_probes(Xtr, ytr, Xte, yte, full=False)}
    print(f"  pixel784 logreg {pixel['logreg']:.4f} rbf {pixel['rbf_svc']:.4f} "
          f"mlp {pixel['mlp']:.4f}", flush=True)

    print("=== 2. 数据规模扫描（同一固定测试集） ===", flush=True)
    scaling = pixel_scaling(args, Xte, yte)

    print("=== 3. 训练 40 维读出模型，记录梯度 / 漂移 ===", flush=True)
    set_seed(seed)
    model = RichVQCClassifier(
        NQ, args.classes, N_LAYERS, encoding="amplitude",
        observables=TRAIN_OBS, n_features=int(Xtr.shape[1]),
        head="mlp", scale=5.0, hidden=args.hidden)
    random_w = model.q.layer.weights.detach().clone()
    trained_w, history = train_and_track(model, d, args, seed)
    zero_w = torch.zeros_like(random_w)
    weights = {"random": random_w, "zero": zero_w, "trained": trained_w}
    history["weight_change_norm"] = round(
        float((trained_w - random_w).norm()), 6)
    history["weight_relative_change"] = round(
        float((trained_w - random_w).norm() / max(float(random_w.norm()), 1e-12)), 6)

    print("=== 4. 各读出宽度 × 三种权重的冻结探针 ===", flush=True)
    readout = {}
    for spec in SPECS:
        dim = n_readout(NQ, spec)
        set_seed(0)
        layer = make_rich_layer(
            n_qubits=NQ, n_layers=N_LAYERS, encoding="amplitude",
            observables=spec, n_features=int(Xtr.shape[1]))
        rec = {"spec": spec, "dim": dim, "states": {}}
        for state in WEIGHT_STATES:
            set_weights(layer, weights[state])
            Ztr = readout_of(layer, Xtr)
            Zte = readout_of(layer, Xte)
            full = (state == "trained")
            entry = fit_probes(Ztr, ytr, Zte, yte, full=full)
            if state == "trained":
                entry["geometry"] = geometry(Ztr, ytr)
                entry["readout_std"] = round(float(Ztr.std()), 4)
            rec["states"][state] = entry
            best = max(entry[k] for k in ("logreg", "rbf_svc", "mlp"))
            print(f"  [{spec:30s} dim={dim:3d} {state:7s}] "
                  f"logreg {entry['logreg']:.4f} rbf {entry['rbf_svc']:.4f} "
                  f"best3 {best:.4f}", flush=True)
        readout[spec] = rec

    print("=== 5. 同维度 PCA 经典参照 ===", flush=True)
    classical = {} if args.no_classical else classical_reference(
        args, CLASSICAL_DIMS, (Xte, yte))

    payload = {
        "config": {k: v for k, v in vars(args).items()},
        "data_meta": meta,
        "purpose": ("把 0.772 的剩余 gap 分解为训练 / 读出宽度 / 二次型表达力 / "
                    "数据与归纳偏置；冻结特征探针是可分性上界，不是端到端训练结果"),
        "trained_run": history,
        "pixel_probe": pixel,
        "pixel_data_scaling": scaling,
        "readout_probes": readout,
        "classical_pca_reference": classical,
        "weight_change": {
            "norm": history["weight_change_norm"],
            "relative": history["weight_relative_change"],
            "grad_norm_first_epoch": history["q_grad_norm"][0],
            "grad_norm_last_epoch": history["q_grad_norm"][-1],
        },
        "caveat": ("除 40 维训练组外均为单 seed 示意性诊断；结论性数字见 "
                   "push_ceiling.py 的 5 seed 正式对照。探针为冻结特征 + 经典分类器"
                   "的可分性上界；poly 核在 165 维上可能过拟合；更多观测量在真机上"
                   "需要更多测量分组与 shots。"),
    }
    out_dir = ensure_dir(ROOT / args.out)
    p = out_dir / "info_metrics.json"
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=float),
                 encoding="utf-8")
    print(f"\nsaved: {p}", flush=True)
    if args.archive:
        arc = ROOT / args.archive
        arc.parent.mkdir(parents=True, exist_ok=True)
        arc.write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                                  default=float), encoding="utf-8")
        print(f"saved: {arc}", flush=True)

    print("\n=== 诊断摘要 ===", flush=True)
    print(f"  量子权重变化 ||Δw||/||w|| = {history['weight_relative_change']:.4f}，"
          f"梯度范数 {history['q_grad_norm'][0]:.5f} → {history['q_grad_norm'][-1]:.5f}")
    for spec in SPECS:
        r = readout[spec]
        b_rand = max(r["states"]["random"][k] for k in ("logreg", "rbf_svc", "mlp"))
        b_train = max(r["states"]["trained"][k] for k in ("logreg", "rbf_svc", "mlp"))
        print(f"  dim {r['dim']:3d} | 随机权重上界 {b_rand:.4f} | "
              f"训练后上界 {b_train:.4f} | 训练带来 {100*(b_train-b_rand):+.2f} pt")
    biggest = str(min(args.big_train, len(scaling["points"])))
    big_key = sorted(scaling["points"], key=int)[-1]
    print(f"  原始像素探针（{args.train_samples} 张）rbf {pixel['rbf_svc']:.4f}；"
          f"{big_key} 张 rbf {scaling['points'][big_key]['rbf_svc']:.4f} "
          f"(scanned up to {biggest})")


if __name__ == "__main__":
    main()
