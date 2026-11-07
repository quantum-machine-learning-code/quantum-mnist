#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""等数据预算的经典对照：补上 `push_ceiling.py` 里缺失的公平比较。

`push_ceiling.py` 的最佳量子组 `b165_d1800` 用 1800 张真图 / 8 epochs，
而内置经典对照 `ref_mlp784` / `ref_pca165` / `ref_cnn` 都只有 600 张 / 24
epochs。若直接比，会把「多看 1200 张图」的红利算到量子头上。本脚本用
**完全相同的数据切分与训练配置**重跑三条经典基线，使比较公平：

    mlp784_1800   : 784 维 L2 归一化像素 → MLP(hidden=32)，1800 张 / 8 ep
    pca165_1800   : PCA-165（在 1800 张上 fit）→ 同结构 MLP 头
    cnn_1800      : 轻量 CNN（同一批 1800 张 28×28 图）

另外补一组 `mlp784_1800_e24`：数据、epoch 都拉满，用来给出「这台机器上
这个经典模型在 1800 张图上大致能到多少」的宽松参照。

协议与 `push_ceiling.py` 一致：3000 张训练池按类取前 180 张/类 = 1800，
200 张测试集固定，5 seed，构造模型前 set_seed，Adam lr=0.01 / batch 32，
`eval_train=False`。产物 `docs/budget_metrics.json`。

运行：
  python experiments/classical_budget.py --data-source idx \
      --archive docs/budget_metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qml_lab.data import load_mnist_images, load_mnist_raw  # noqa: E402
from qml_lab.models import CNNClassifier, MLPClassifier  # noqa: E402
from qml_lab.train import train_classifier  # noqa: E402
from qml_lab.utils import count_parameters, set_seed  # noqa: E402

GROUPS = {
    "mlp784_1800": ("raw", 32, 8),
    "pca165_1800": ("pca", 32, 8),
    "cnn_1800": ("img", 32, 8),
    "mlp784_1800_e24": ("raw", 32, 24),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="等数据预算经典对照")
    p.add_argument("--seeds", type=str, default="0,1,2,3,4")
    p.add_argument("--pool-train", type=int, default=3000)
    p.add_argument("--n-train", type=int, default=1800)
    p.add_argument("--test-samples", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--classes", type=int, default=10)
    p.add_argument("--data-source", type=str, default="idx")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--out", type=str, default="results")
    p.add_argument("--archive", type=str, default="docs/budget_metrics.json")
    p.add_argument("--quick", action="store_true", help="离线极小自检，不覆盖 docs/ 正式归档")
    return p.parse_args()


def _stats(values) -> dict:
    a = np.asarray(values, dtype=float)
    return {"mean": round(float(a.mean()), 4),
            "std": round(float(a.std(ddof=1)) if a.size > 1 else 0.0, 4),
            "min": round(float(a.min()), 4), "max": round(float(a.max()), 4),
            "values": [round(float(v), 4) for v in a], "n": int(a.size)}


def main() -> None:
    args = parse_args()
    if args.quick:
        args.seeds = "0"
        args.pool_train, args.n_train, args.test_samples = 60, 40, 20
        args.classes = 4
        args.data_source = "synthetic"
        if args.archive == "docs/budget_metrics.json":
            args.archive = "results/quick_budget_metrics.json"
    dd = Path(args.data_dir)
    data_dir = str(dd if dd.is_absolute() else ROOT / dd)
    kw = dict(classes=args.classes, source=args.data_source,
              data_dir=data_dir, seed=0)
    Xr, yr, Xte, yte, m_raw = load_mnist_raw(
        n_train=args.pool_train, n_test=args.test_samples, **kw)
    Xi, yi, Xite, yite, m_img = load_mnist_images(
        n_train=args.pool_train, n_test=args.test_samples, **kw)
    assert np.array_equal(yr, yi) and np.array_equal(yte, yite), "两条路径抽样不一致"

    # 与 push_ceiling.py 完全相同的嵌套均衡子集（每类取前 k 个）
    per = args.pool_train // args.classes
    k = args.n_train // args.classes
    idx = np.concatenate([np.arange(c * per, c * per + k)
                          for c in range(args.classes)])
    Xr, yr = np.ascontiguousarray(Xr[idx]), np.ascontiguousarray(yr[idx])
    Xi, yi = np.ascontiguousarray(Xi[idx]), np.ascontiguousarray(yi[idx])

    from sklearn.decomposition import PCA
    pca_dim = min(165, Xr.shape[0], Xr.shape[1])
    pca = PCA(n_components=pca_dim, random_state=0)
    Xp = pca.fit_transform(Xr).astype(np.float32)
    Xpt = pca.transform(Xte).astype(np.float32)
    if pca_dim < 165:
        Xp = np.pad(Xp, ((0, 0), (0, 165 - pca_dim)))
        Xpt = np.pad(Xpt, ((0, 0), (0, 165 - pca_dim)))

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    runs = {g: {} for g in GROUPS}
    for g, (kind, hidden, epochs) in GROUPS.items():
        eff_epochs = 2 if args.quick else epochs
        for seed in seeds:
            set_seed(seed)
            if kind == "img":
                X, y, Xt, yt, model = Xi, yi, Xite, yte, CNNClassifier(
                    args.classes, frontend="light")
            elif kind == "pca":
                X, y, Xt, yt = Xp, yr, Xpt, yte
                model = MLPClassifier(X.shape[1], args.classes, hidden=hidden)
            else:
                X, y, Xt, yt = Xr, yr, Xte, yte
                model = MLPClassifier(X.shape[1], args.classes, hidden=hidden)
            h = train_classifier(model, X, y, Xt, yt, epochs=eff_epochs,
                                 batch_size=args.batch_size, lr=args.lr,
                                 seed=seed, verbose=False, eval_train=False,
                                 eval_batch_size=64)
            runs[g][str(seed)] = {
                "final_test_acc": round(float(h["final_test_acc"]), 4),
                "final_loss": round(float(h["train_loss"][-1]), 4),
                "seconds": round(float(h["total_seconds"]), 2),
                "params": int(count_parameters(model)),
                "n_train": int(len(y)), "epochs": eff_epochs,
                "test_acc_curve": [round(float(v), 4) for v in h["test_acc"]],
            }
            print(f"  [{g:18s} seed {seed}] acc={runs[g][str(seed)]['final_test_acc']:.4f} "
                  f"params={runs[g][str(seed)]['params']} "
                  f"{runs[g][str(seed)]['seconds']}s", flush=True)

    out = {
        "config": vars(args),
        "seeds": seeds,
        "data_meta": {"source": m_raw["source"],
                      "pool_train": args.pool_train,
                      "n_train": args.n_train,
                      "test_samples": args.test_samples,
                      "note": "与 push_ceiling.py 的 b165_d1800 完全相同的训练子集与测试集"},
        "groups": {g: {
            "kind": GROUPS[g][0], "epochs": GROUPS[g][2],
            "params": runs[g][str(seeds[0])]["params"],
            "final_acc": _stats([runs[g][str(s)]["final_test_acc"] for s in seeds]),
        } for g in GROUPS},
        "runs": runs,
        "caveat": ("经典对照与量子组同数据切分 / 同训练配置 / 同 5 seed；"
                   "参数预算仍不对等（量子组 5702 参数 vs MLP-784 25450）；"
                   "n=5 未做多重比较校正。"),
    }
    p = ROOT / args.archive
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float),
                 encoding="utf-8")
    print(f"\nsaved: {p}", flush=True)
    for g in GROUPS:
        s = out["groups"][g]["final_acc"]
        print(f"  {g:18s} {s['mean']:.4f} ± {s['std']:.4f}", flush=True)


if __name__ == "__main__":
    main()
