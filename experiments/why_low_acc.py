"""为什么准确率「相对低」：把三类结构性上限量化出来（2026-11-06 六轮诊断）。

不训练量子模型，只做便宜的经典探针与几何测量，回答三个问题：

A. 振幅编码必须逐样本 L2 归一化，丢掉的 ||x|| 到底携带多少分类信息？
B. 量子读出 f_j(x) = <x|U†P_jU|x> 是 x 的齐次二次型。那么「全部二次单项式」
   （poly-2 核，等价于把 165 个观测量放宽到 307720 个）能到多少？
   如果量子 165 维已经接近这个天花板，就说明继续加观测量没有空间。
C. 无局部性 / 平移不变性：1px 平移下，量子态内积与读出向量变化有多大？
   这解释了为什么平移增强反而掉分（push_ceiling.py 的 b165_aug）。

用法：
    python experiments/why_low_acc.py --data-source idx --archive docs/why_low_metrics.json
    python experiments/why_low_acc.py --quick          # 合成数据极小自检，不用于结论
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qml_lab.data import load_mnist_images, load_mnist_raw  # noqa: E402

OBS165 = "z+x+y+zzall+xxall+yyall"


def l2(X: np.ndarray) -> np.ndarray:
    A = np.asarray(X, dtype=np.float64)
    n = np.linalg.norm(A, axis=1, keepdims=True)
    n[n < 1e-12] = 1.0
    return A / n


def shift_images(X: np.ndarray, dy: int = 1, dx: int = 1) -> np.ndarray:
    """28x28 图像零填充平移，返回同样 shape 的 (n, 784)。"""
    n = X.shape[0]
    imgs = X.reshape(n, 28, 28)
    out = np.zeros_like(imgs)
    ys = slice(max(dy, 0), 28 + min(dy, 0))
    xs = slice(max(dx, 0), 28 + min(dx, 0))
    yd = slice(max(-dy, 0), 28 - max(dy, 0))
    xd = slice(max(-dx, 0), 28 - max(dx, 0))
    out[:, yd, xd] = imgs[:, ys, xs]
    return out.reshape(n, -1)


def acc_of(clf, Xtr, ytr, Xte, yte) -> float:
    clf.fit(Xtr, ytr)
    return float(np.mean(clf.predict(Xte) == yte))


def classic_probe(Xtr, ytr, Xte, yte) -> dict:
    """同一切分下同时跑两套口径。

    ``scaled_*`` 与 probe_information.py 的历史探针完全一致
    （StandardScaler + C=10 + coef0=1），``plain_*`` 是未标准化的对照。
    两者数字**不可混比**，这里并列是为了防止把预处理差异误读成结论。
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    models = {
        "logreg": lambda: LogisticRegression(max_iter=2000, C=1.0),
        "rbf": lambda: SVC(kernel="rbf", C=10.0),
        "poly2_c0": lambda: SVC(kernel="poly", degree=2, coef0=0.0, C=10.0),
        "poly2_c1": lambda: SVC(kernel="poly", degree=2, coef0=1.0, C=10.0),
    }
    out = {}
    for tag, wrap in (("plain", lambda c: c),
                      ("scaled", lambda c: make_pipeline(StandardScaler(), c))):
        for name, make in models.items():
            out[f"{tag}_{name}"] = round(acc_of(wrap(make()), Xtr, ytr, Xte, yte), 4)
    return out


def nested(X: np.ndarray, y: np.ndarray, n: int, classes: int):
    per = len(X) // classes
    k = n // classes
    assert k <= per, f"子集 {n} 超过池容量 {len(X)}"
    idx = np.concatenate([np.arange(c * per, c * per + k) for c in range(classes)])
    return np.ascontiguousarray(X[idx]), np.ascontiguousarray(y[idx])


def quantum_readout(X: np.ndarray, observables: str, torch_threads: int = 2) -> np.ndarray:
    """随机酉下的 165 维读出（诊断已证明随机酉上界 0.860，不需要训练）。"""
    import torch
    from qml_lab.quantum import make_rich_layer

    torch.set_num_threads(torch_threads)
    torch.manual_seed(0)
    layer = make_rich_layer(n_qubits=10, n_layers=2, encoding="amplitude",
                            observables=observables, n_features=784).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 32):
            xb = torch.as_tensor(np.asarray(X[i:i + 32], dtype=np.float32))
            out.append(layer(xb).numpy())
    return np.concatenate(out, axis=0).astype(np.float64)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-source", default="idx")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--pool-train", type=int, default=3000)
    ap.add_argument("--test-samples", type=int, default=200)
    ap.add_argument("--classes", type=int, default=10)
    ap.add_argument("--train-samples", type=int, default=600,
                    help="A 部分用的小样本规模，与正式实验一致")
    ap.add_argument("--scaling-points", default="600,1800,3000")
    ap.add_argument("--torch-threads", type=int, default=2)
    ap.add_argument("--archive", default="docs/why_low_metrics.json")
    ap.add_argument("--fig-dir", default="docs/figures")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--quick", action="store_true", help="合成数据自检，数字不可用于结论")
    args = ap.parse_args()

    if args.quick:
        args.data_source = "synthetic"
        args.pool_train, args.train_samples, args.test_samples = 80, 40, 20
        args.classes = 4
        args.scaling_points = "40,80"
        args.archive, args.fig_dir = "", "results"
        args.no_figures = True

    t0 = time.time()
    dd = Path(args.data_dir)
    data_dir = str(dd if dd.is_absolute() else ROOT / dd)
    kw = dict(classes=args.classes, source=args.data_source,
              data_dir=data_dir, seed=0)
    Xi, yi, Xite, yite, m_img = load_mnist_images(
        n_train=args.pool_train, n_test=args.test_samples, **kw)
    Xr, yr, Xte, yte, m_raw = load_mnist_raw(
        n_train=args.pool_train, n_test=args.test_samples, **kw)
    assert np.array_equal(yi, yr) and np.array_equal(yite, yte), "两条数据路径抽样不一致"

    flat = np.asarray(Xi, dtype=np.float64).reshape(len(Xi), -1)
    flat_test = np.asarray(Xite, dtype=np.float64).reshape(len(Xite), -1)
    if args.data_source != "synthetic":
        assert np.allclose(Xr, l2(flat), atol=1e-5), "raw 路径应与图像 L2 归一化一致"
    norm_all = np.linalg.norm(flat, axis=1, keepdims=True)
    norm_all[norm_all < 1e-12] = 1.0
    norm_test = np.linalg.norm(flat_test, axis=1, keepdims=True)
    norms, _ = nested(norm_all, yi, args.train_samples, args.classes)
    Xt, yt = nested(flat, yi, args.train_samples, args.classes)
    Xn, _ = nested(Xr, yi, args.train_samples, args.classes)

    result: dict = {
        "purpose": "解释 0.866 的结构性上限：归一化信息损失 / 二次型天花板 / 无平移不变性",
        "config": {k: v for k, v in vars(args).items() if k != "quick"},
        "data_meta": {"source": m_raw.get("source"), "note": m_raw.get("note"),
                      "pool_train": args.pool_train, "test_samples": args.test_samples,
                      "same_split_asserted": True},
        "theory": {
            "quantum_feature": "f_j(x) = <x|U† P_j U|x>，振幅编码下是归一化像素的齐次二次型",
            "n_qubits": 10,
            "hilbert_dim": 2 ** 10,
            "independent_homogeneous_quadratic_monomials_784d": 784 * 785 // 2,
            "traceless_hermitian_operator_dim": 4 ** 10 - 1,
            "readout_dim_165": 165,
            "note": "165 个观测量是 4^10-1 维无迹厄米算符空间里的 165 个方向；"
                    "poly-2 核 (coef0=0) 给出全部齐次二次单项式，是量子特征类的超集",
        },
    }

    # ---- A. L2 归一化丢了什么 -------------------------------------------------
    from sklearn.linear_model import LogisticRegression
    norm_clf = LogisticRegression(max_iter=2000, C=1.0)
    norm_only = acc_of(norm_clf, norms, yt, norm_test, yte)
    result["normalization_loss"] = {
        "chance_level": round(1.0 / args.classes, 4),
        "norm_only_logreg_acc": round(norm_only, 4),
        "raw_pixels_probe": classic_probe(Xt, yt, flat_test, yte),
        "l2_normalized_pixels_probe": classic_probe(Xn, yt, Xte, yte),
        "note": "振幅编码必须逐样本除以 ||x||；若仅用 ||x|| 一维就能超过 chance，"
                "说明被丢弃的墨量/亮度标量本身携带类别信息",
    }
    print(f"[A] norm-only logreg = {norm_only:.4f} (chance {1/args.classes:.2f})")
    print("    raw   ", result["normalization_loss"]["raw_pixels_probe"])
    print("    L2    ", result["normalization_loss"]["l2_normalized_pixels_probe"])

    # ---- B. 二次型天花板 ------------------------------------------------------
    points = {}
    for n in [int(s) for s in args.scaling_points.split(",") if s.strip()]:
        if n > args.pool_train:
            continue
        Xs, ys = nested(Xr, yi, n, args.classes)
        # 在 L2 归一化像素上：poly2 核 = 全部二次单项式（量子特征类的超集）
        p = classic_probe(Xs, ys, Xte, yte)
        points[str(n)] = {"n_train": n, **p}
        print(f"[B] n={n:5d} scaled rbf={p['scaled_rbf']:.4f} "
              f"poly2={p['scaled_poly2_c1']:.4f} | plain rbf={p['plain_rbf']:.4f} "
              f"poly2={p['plain_poly2_c0']:.4f}")

    # 同口径下，直接量一次「165 维量子读出」本身的可分性，避免拿不同预处理的数字比较
    quantum_probe = None
    if not args.quick:
        Xq, yq = nested(Xr, yi, min(args.train_samples, args.pool_train), args.classes)
        Ftr = quantum_readout(Xq, OBS165, args.torch_threads)
        Fte = quantum_readout(np.asarray(Xte, dtype=np.float64), OBS165,
                              args.torch_threads)
        quantum_probe = classic_probe(Ftr, yq, Fte, yte)
        print(f"[D] quantum 165-dim scaled rbf={quantum_probe['scaled_rbf']:.4f} "
              f"poly2={quantum_probe['scaled_poly2_c1']:.4f} "
              f"(历史归档 0.890 / 0.900)")

    result["quadratic_ceiling"] = {
        "points": points,
        "quantum_165dim_readout_probe_same_protocol": quantum_probe,
        "reference_from_archive": {
            "quantum_165dim_frozen_upper_bound": 0.890,
            "quantum_165dim_frozen_poly2": 0.900,
            "quantum_165dim_e2e_600": 0.845,
            "quantum_165dim_e2e_1800": 0.866,
            "classical_pca165_equal_budget_1800": 0.903,
            "strong_cnn_600": 0.934,
        },
        "note": "scaled_* 口径与 probe_information.py 一致（StandardScaler + C=10 + "
                "coef0=1）；plain_* 是未标准化对照，两套数字不可混比。"
                "poly-2 核给出全部 307720 个齐次二次单项式，是量子 165 维读出的超集；"
                "两者在同口径下的差值 = 读出投影数太少造成的可分性缺口，"
                "而端到端 0.845/0.866 与冻结上界的差值 = 训练/泛化缺口",
    }

    # ---- C. 平移敏感度 --------------------------------------------------------
    n_shift = min(args.test_samples, 200)
    imgs = np.asarray(Xite, dtype=np.float64).reshape(len(Xite), -1)[:n_shift]
    moved = shift_images(imgs, 1, 1)
    A, B = l2(imgs), l2(moved)
    cos_shift = np.sum(A * B, axis=1)

    y_arr = np.asarray(yite)[:n_shift]
    same_class_cos = []
    for c in np.unique(y_arr):
        idx = np.flatnonzero(y_arr == c)
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                same_class_cos.append(float(np.dot(A[idx[a]], A[idx[b]])))
    same_class_cos = np.asarray(same_class_cos) if same_class_cos else np.array([0.0])

    geom = {
        "state_cosine_after_1px_shift_mean": round(float(cos_shift.mean()), 4),
        "state_cosine_after_1px_shift_median": round(float(np.median(cos_shift)), 4),
        "same_class_state_cosine_mean": round(float(same_class_cos.mean()), 4),
        "same_class_state_cosine_median": round(float(np.median(same_class_cos)), 4),
        "n_same_class_pairs": int(len(same_class_cos)),
    }
    if not args.quick:
        F = quantum_readout(A, OBS165, args.torch_threads)
        Fs = quantum_readout(B, OBS165, args.torch_threads)
        qshift = np.linalg.norm(F - Fs, axis=1)
        spread = np.linalg.norm(F - F.mean(axis=0, keepdims=True), axis=1)
        wc = np.zeros_like(F)
        for c in np.unique(y_arr):
            m = y_arr == c
            if m.sum() > 1:
                wc[m] = F[m] - F[m].mean(axis=0, keepdims=True)
        within = float(np.linalg.norm(wc, axis=1).mean())
        geom.update({
            "readout_165_relative_change_by_shift": round(float(qshift.mean() / max(spread.mean(), 1e-9)), 4),
            "readout_165_shift_vs_within_class_spread": round(float(qshift.mean() / max(within, 1e-9)), 4),
            "readout_165_shift_l2_mean": round(float(qshift.mean()), 4),
            "readout_165_spread_l2_mean": round(float(spread.mean()), 4),
            "readout_165_within_class_spread_l2_mean": round(within, 4),
        })
    result["translation_sensitivity"] = {
        **geom,
        "reference": {"b165_d1800_real_images": 0.866, "b165_aug_shift_x3": 0.822,
                      "paired_delta_pt": -4.4},
        "note": "振幅编码把整图当作一个全局态向量，没有局部感受野 / 池化。"
                "若 1px 平移使态内积接近甚至低于同类内平均，增强样本就不是有效的同类扩充",
    }
    print(f"[C] 1px shift cos = {cos_shift.mean():.4f} | "
          f"same-class cos = {same_class_cos.mean():.4f}")
    if not args.quick:
        print(f"    readout relative change = "
              f"{geom['readout_165_relative_change_by_shift']:.4f} | "
              f"vs within-class spread = "
              f"{geom['readout_165_shift_vs_within_class_spread']:.4f}")

    result["wall_seconds"] = round(time.time() - t0, 2)
    result["caveat"] = (
        "A/B 用 sklearn 探针（非端到端训练），C/gamma 沿用历史归档设置未另调参；"
        "scaled_* 与 plain_* 是两套预处理口径，禁止混比——本脚本并列两者正是为了"
        "避免把 StandardScaler / coef0 的差异误读成信息量差异；"
        "poly2 核在 784 维上等价于 307720 个二次单项式，是量子 165 维读出的超集，"
        "两者不是等预算对照，只能读成「特征类上限」；"
        "C 部分量子读出用随机酉（诊断已证明随机酉冻结上界 0.860 与训练后 0.890 接近），"
        "不是训练后权重；数据规模扫描与正式实验共用同一嵌套切分，"
        "但探针与端到端模型不同，不做逐位比较。"
    )

    if args.archive:
        out = Path(args.archive)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"archived -> {out.relative_to(ROOT)}")

    if not args.no_figures:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
            ax = axes[0]
            keys = ["scaled_logreg", "scaled_rbf", "scaled_poly2_c1"]
            raw = [result["normalization_loss"]["raw_pixels_probe"][k] for k in keys]
            nor = [result["normalization_loss"]["l2_normalized_pixels_probe"][k] for k in keys]
            x = np.arange(len(keys))
            ax.bar(x - 0.18, raw, 0.36, label="raw pixels", color="#85B7EB")
            ax.bar(x + 0.18, nor, 0.36, label="L2 normalized (amplitude)", color="#F0997B")
            ax.set_xticks(x, ["logreg", "RBF", "poly-2"])
            ax.set_ylim(0, 1.0)
            ax.set_ylabel("test acc (600 train, scaled protocol)")
            ax.set_title("A: L2 normalization costs almost nothing")
            ax.legend(fontsize=9)
            ax.grid(axis="y", alpha=0.3)

            ax = axes[1]
            ns = sorted(int(k) for k in points)
            ax.plot(ns, [points[str(n)]["scaled_poly2_c1"] for n in ns], "o-",
                    color="#7F77DD", label="poly-2 kernel (all quadratic monomials)")
            ax.plot(ns, [points[str(n)]["scaled_rbf"] for n in ns], "s-",
                    color="#1D9E75", label="RBF kernel")
            if result["quadratic_ceiling"]["quantum_165dim_readout_probe_same_protocol"]:
                q = result["quadratic_ceiling"]["quantum_165dim_readout_probe_same_protocol"]
                ax.plot([min(ns)], [q["scaled_rbf"]], "D", color="#D85A30", ms=8,
                        label="quantum 165-dim readout (same protocol)")
            ax.axhline(0.866, ls=":", color="#888780", lw=1.2,
                       label="quantum e2e best 0.866")
            ax.set_xlabel("training images")
            ax.set_ylabel("test acc (scaled protocol)")
            ax.set_ylim(0.5, 1.0)
            ax.set_title("B: 165 Pauli expectations vs all quadratic monomials")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fd = Path(args.fig_dir)
            if not fd.is_absolute():
                fd = ROOT / fd
            fd.mkdir(parents=True, exist_ok=True)
            fig.savefig(fd / "why_low.png", dpi=130)
            plt.close(fig)
            print(f"figure -> {fd / 'why_low.png'}")
        except Exception as e:  # pragma: no cover
            print(f"[warn] figure failed: {e}")

    print(f"done in {result['wall_seconds']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
