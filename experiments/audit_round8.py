"""第八轮外部审查（2026-11-07）：三项可核查的复算与消融。

背景：`df62712` 把 QSVC / PatchVQC 作为第八轮主要成果。本脚本不改原实验，
只补上原表里缺失的公平对照，并复核统计结论。

A. **QSVC 恒等性复算**：`qkernel_amp = (x_hat^T x_hat')^2` 与经典二次多项式核
   `SVC(kernel="poly", degree=2, coef0=0, gamma=1)`（同样不做 StandardScaler）
   在数学上等价。逐 seed 复算，并列出「加了 StandardScaler / coef0=1」的对照，
   量化第八轮表格里 `qkernel_amp - svm_poly2 = +2.30 pt` 有多少来自预处理差异。

B. **PatchVQC 量子层消融**：把共享 8-qubit VQC 换成共享的
   `Linear(196, 32) + Tanh`，其余（3x3 重叠 14x14 窗口、逐 patch L2 归一化、
   样本内相对范数加权、拼接 9 个窗口范数、BatchNorm1d(297)、dropout 0.1、
   hidden=64 的 MLP 头、scale=1.0、24 epochs / lr 0.01 / weight_decay 1e-4 /
   余弦退火）逐项保持一致。固定 seed=0 数据切分，5 个 seed 只改初始化，
   与第八轮 VQC 组口径一致，因此可做逐 seed 配对检验。
   经典替身参数更多（约 2.6 万 vs 2.0 万），是给经典组的 handicap：
   若打平或更好，说明 0.880 主要来自经典局部窗口归纳偏置。

C. **多重比较复核**：第八轮归档了 20 个配对检验、声明「未做多重比较校正」。
   这里按 Bonferroni 重算临界 t 值，统计校正后仍显著的检验数。

用法：
    .venv/bin/python experiments/audit_round8.py --data-source idx \
        --archive docs/audit_metrics.json
    .venv/bin/python experiments/audit_round8.py --quick     # 离线合成数据自检
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qml_lab.data import load_mnist_images, load_mnist_raw  # noqa: E402
from qml_lab.train import train_classifier  # noqa: E402
from qml_lab.utils import count_parameters, set_seed  # noqa: E402


class PatchLinearClassifier(nn.Module):
    """PatchVQCClassifier 的经典替身：仅把量子线路换成 196->32 线性投影。"""

    def __init__(self, patch_size: int = 14, stride: int = 7, image_size: int = 28,
                 n_classes: int = 10, hidden: int = 64, dropout: float = 0.1,
                 per_patch_out: int = 32):
        super().__init__()
        self.patch_size, self.stride, self.image_size = patch_size, stride, image_size
        self.n_patches = ((image_size - patch_size) // stride + 1) ** 2
        self.per_patch_out = per_patch_out
        self.proj = nn.Sequential(
            nn.Linear(patch_size * patch_size, per_patch_out), nn.Tanh()
        )
        self.readout_dim = self.n_patches * (per_patch_out + 1)
        self.post_norm = nn.BatchNorm1d(self.readout_dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(self.readout_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )
        self.register_buffer("out_scale", torch.tensor(1.0))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        img = x.to(torch.float32)
        if img.dim() == 2 and img.shape[1] == self.image_size ** 2:
            img = img.view(-1, 1, self.image_size, self.image_size)
        B = img.shape[0]
        patches = F.unfold(img, kernel_size=self.patch_size,
                           stride=self.stride).transpose(1, 2)  # (B, P, D)
        p_norm = patches.norm(dim=-1, keepdim=True)
        safe = patches.clone()
        zero = p_norm.squeeze(-1) < 1e-6
        if zero.any():
            safe[zero, 0] = 1.0
        unit = safe / safe.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        f = self.proj(unit.reshape(B * self.n_patches, -1))
        f = f.reshape(B, self.n_patches, self.per_patch_out).to(torch.float32)
        rel = p_norm / p_norm.mean(dim=1, keepdim=True).clamp_min(1e-6)
        feat = torch.cat([f * rel, p_norm], dim=-1).reshape(B, self.readout_dim)
        if B > 1 or not self.training:
            feat = self.post_norm(feat)
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.drop(self.features(x) * self.out_scale))


def kernel_recheck(args, seeds: list[int]) -> dict:
    """A：量子振幅核 vs 各种经典二次核（逐 seed 同一切分）。"""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    out: dict[str, list[float]] = {k: [] for k in
                                   ["qkernel_amp", "plain_poly2_c0", "plain_poly2_c1",
                                    "scaled_poly2_c0", "scaled_poly2_c1"]}
    for s in seeds:
        Xr, yr, Xt, yt, _ = load_mnist_raw(
            n_train=args.train_samples, n_test=args.test_samples, classes=args.classes,
            source=args.data_source, data_dir=args.data_dir, seed=s)
        K_tr, K_te = (Xr @ Xr.T) ** 2, (Xt @ Xr.T) ** 2
        out["qkernel_amp"].append(float(
            SVC(kernel="precomputed", C=10.0).fit(K_tr, yr).score(K_te, yt)))
        out["plain_poly2_c0"].append(float(
            SVC(kernel="poly", degree=2, coef0=0, gamma=1.0, C=10.0)
            .fit(Xr, yr).score(Xt, yt)))
        out["plain_poly2_c1"].append(float(
            SVC(kernel="poly", degree=2, coef0=1, gamma="scale", C=10.0)
            .fit(Xr, yr).score(Xt, yt)))
        out["scaled_poly2_c0"].append(float(
            make_pipeline(StandardScaler(),
                          SVC(kernel="poly", degree=2, coef0=0, C=10.0))
            .fit(Xr, yr).score(Xt, yt)))
        out["scaled_poly2_c1"].append(float(
            make_pipeline(StandardScaler(),
                          SVC(kernel="poly", degree=2, coef0=1, C=10.0))
            .fit(Xr, yr).score(Xt, yt)))
    summary = {}
    for k, vals in out.items():
        summary[k] = {"mean": round(float(np.mean(vals)), 4),
                      "std": round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 4),
                      "values": [round(v, 4) for v in vals]}
    summary["qkernel_amp_equals_plain_poly2_c0"] = all(
        abs(a - b) < 1e-9 for a, b in
        zip(out["qkernel_amp"], out["plain_poly2_c0"]))
    return summary


def ablation(args, seeds: list[int]) -> dict:
    """B：量子层消融，固定 seed=0 数据切分，5 seed 只改初始化。"""
    torch.set_num_threads(args.torch_threads)
    Xi, yi, Xit, yit, _ = load_mnist_images(
        n_train=args.train_samples, n_test=args.test_samples, classes=args.classes,
        source=args.data_source, data_dir=args.data_dir, seed=0)
    runs = []
    for s in seeds:
        set_seed(s)  # 必须在构造模型之前（TorchLayer / Linear 初始化用全局 RNG）
        model = PatchLinearClassifier()
        t0 = time.perf_counter()
        h = train_classifier(
            model, Xi, yi, Xit, yit,
            epochs=args.epochs, batch_size=32, lr=0.01, seed=s,
            eval_batch_size=64, verbose=False, eval_train=False,
            weight_decay=1e-4, cosine_lr=True,
        )
        runs.append({"seed": s, "acc": round(float(h["final_test_acc"]), 4),
                     "loss": round(float(h["train_loss"][-1]), 4),
                     "params": int(count_parameters(model)),
                     "seconds": round(time.perf_counter() - t0, 2)})
        print(f"[B] seed={s} acc={runs[-1]['acc']:.4f} "
              f"params={runs[-1]['params']} secs={runs[-1]['seconds']}", flush=True)

    # 与归档的 PatchVQC 逐 seed 配对
    arch = ROOT / "docs" / "open_issues_metrics.json"
    paired = None
    if arch.exists():
        d = json.loads(arch.read_text(encoding="utf-8"))
        q_runs = d.get("runs", {}).get("patch_vqc", {})
        qvals = None
        if all(str(s) in q_runs for s in seeds):
            qvals = [float(q_runs[str(s)]["final_test_acc"]) for s in seeds]
        if qvals:
            from scipy import stats
            diff = [100 * (r["acc"] - q) for r, q in zip(runs, qvals)]
            arr = np.asarray(diff, dtype=float)
            se = arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
            tcrit = stats.t.ppf(0.975, len(arr) - 1) if len(arr) > 1 else 0.0
            paired = {"patch_vqc_per_seed": qvals,
                      "ablation_per_seed": [r["acc"] for r in runs],
                      "diff_pt_linear_minus_quantum": [round(v, 3) for v in diff],
                      "mean_pt": round(float(arr.mean()), 3),
                      "std_pt": round(float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, 3),
                      "ci95_pt": [round(float(arr.mean() - tcrit * se), 3),
                                  round(float(arr.mean() + tcrit * se), 3)]}
    return {"runs": runs,
            "mean": round(float(np.mean([r["acc"] for r in runs])), 4),
            "std": round(float(np.std([r["acc"] for r in runs], ddof=1))
                         if len(runs) > 1 else 0.0, 4),
            "paired_with_patch_vqc": paired}


def multiplicity_recheck() -> dict:
    """C：按 Bonferroni 复核第八轮 20 个配对检验。"""
    from scipy import stats
    path = ROOT / "docs" / "open_issues_metrics.json"
    if not path.exists():
        return {"note": "open_issues_metrics.json 不存在，跳过"}
    d = json.loads(path.read_text(encoding="utf-8"))
    p = d.get("paired_differences", {})
    n = len(p)
    t975 = float(stats.t.ppf(0.975, 4))
    tbonf = float(stats.t.ppf(1 - 0.05 / (2 * n), 4)) if n else 0.0
    rows = {}
    naive = bonf = 0
    for k, v in p.items():
        lo, hi = v["ci95_pt"]
        sig = not (lo <= 0 <= hi)
        se = (hi - lo) / (2 * t975) if t975 else 0.0
        t = abs(v["mean_pt"]) / se if se > 0 else 0.0
        bsig = t > tbonf
        naive += int(sig); bonf += int(bsig)
        rows[k] = {"mean_pt": v["mean_pt"], "t": round(t, 3),
                   "naive_significant": sig, "bonferroni_significant": bsig}
    return {"n_tests": n, "t_threshold_naive": round(t975, 3),
            "t_threshold_bonferroni": round(tbonf, 3),
            "naive_significant": naive, "bonferroni_significant": bonf,
            "per_test": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-source", default="idx")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--train-samples", type=int, default=600)
    ap.add_argument("--test-samples", type=int, default=200)
    ap.add_argument("--classes", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--torch-threads", type=int, default=2)
    ap.add_argument("--archive", default="docs/audit_metrics.json")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.data_source, args.train_samples, args.test_samples = "synthetic", 40, 20
        args.epochs, args.seeds, args.archive = 1, "0", ""
        args.classes = 4
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    result = {"purpose": "第八轮外部审查：QSVC 恒等性 / PatchVQC 量子层消融 / 多重比较复核",
              "config": {"data_source": args.data_source, "seeds": args.seeds,
                         "epochs": args.epochs, "train_samples": args.train_samples,
                         "test_samples": args.test_samples, "classes": args.classes},
              "quick": bool(args.quick)}
    t0 = time.perf_counter()
    print("=== A: QSVC 恒等性 ===", flush=True)
    result["A_kernel_recheck"] = kernel_recheck(args, seeds)
    print(json.dumps(result["A_kernel_recheck"], ensure_ascii=False), flush=True)
    print("=== B: PatchVQC 量子层消融 ===", flush=True)
    result["B_patch_ablation"] = ablation(args, seeds)
    print(json.dumps(result["B_patch_ablation"], ensure_ascii=False)[:1200], flush=True)
    if not args.quick:
        print("=== C: 多重比较复核 ===", flush=True)
        result["C_multiplicity"] = multiplicity_recheck()
        c = result["C_multiplicity"]
        print(f"  {c['naive_significant']}/{c['n_tests']} -> "
              f"{c['bonferroni_significant']}/{c['n_tests']}", flush=True)
    result["wall_seconds"] = round(time.perf_counter() - t0, 2)
    if args.archive:
        out = ROOT / args.archive
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
        print(f"archive -> {out}", flush=True)
    print("AUDIT OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
