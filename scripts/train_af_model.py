"""房颤样心律筛查：逻辑回归分类器（在规则版特征上训练）。

要点：
  · 特征取 results/af_mitdb.csv（由 scripts/eval_af_mitdb.py 生成），
    全部是 RR 域特征——可穿戴单导联场景不依赖 P 波识别；
  · 交叉验证按"记录"分组（GroupKFold）：同一记录内相邻窗口高度相关，
    不分组会严重高估性能——这是这类评价最常见的泄漏来源；
  · 同时报告规则版与模型版的对照，并保存模型供智能体管线调用。

用法：
    python scripts/train_af_model.py [--csv results/af_mitdb.csv] [--save models/af_logreg.joblib]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

FEATURES = ["hr", "cv_rr", "rmssd_norm", "pnn50", "shannon_rr", "sample_entropy",
            "shannon_drr", "sd1_sd2"]


def load(csv_path: str):
    import csv
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    X = np.array([[float(r[f]) for f in FEATURES] for r in rows])
    y = np.array([1 if r["label"] == "af" else 0 for r in rows])
    groups = np.array([r["record"] for r in rows])
    rule_score = np.array([float(r["score"]) for r in rows])
    return X, y, groups, rule_score


def metrics(y, p):
    tp = int(((y == 1) & (p == 1)).sum()); fp = int(((y == 0) & (p == 1)).sum())
    tn = int(((y == 0) & (p == 0)).sum()); fn = int(((y == 1) & (p == 0)).sum())
    se = tp / (tp + fn) if tp + fn else 0.0
    sp = tn / (tn + fp) if tn + fp else 0.0
    ppv = tp / (tp + fp) if tp + fp else 0.0
    acc = (tp + tn) / max(1, len(y))
    return dict(tp=tp, fp=fp, tn=tn, fn=fn, se=se, sp=sp, ppv=ppv, acc=acc)


def sweep(y, score, thresholds=None):
    out = []
    for t in (thresholds if thresholds is not None else np.arange(0.1, 0.95, 0.05)):
        m = metrics(y, (score >= t).astype(int))
        m["thr"] = float(t)
        out.append(m)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/af_mitdb.csv")
    ap.add_argument("--save", default="models/af_logreg.joblib")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y, groups, rule_score = load(args.csv)
    print(f"样本 {len(y)}（AF {int(y.sum())} / 窦律 {int((1-y).sum())}），"
          f"记录 {len(set(groups))} 条，特征 {len(FEATURES)} 维")

    gkf = GroupKFold(n_splits=args.folds)
    oof = np.zeros(len(y))
    for k, (tr, te) in enumerate(gkf.split(X, y, groups)):
        pipe = make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000, class_weight="balanced"))
        pipe.fit(X[tr], y[tr])
        oof[te] = pipe.predict_proba(X[te])[:, 1]

    print("\n== 交叉验证（按记录分组，out-of-fold）==")
    print(f"{'阈值':>6} {'Se%':>7} {'Sp%':>7} {'PPV%':>7} {'Acc%':>7}")
    best = None
    for m in sweep(y, oof):
        if m["se"] >= 0.90 and (best is None or m["acc"] > best["acc"]):
            best = m
        print(f"{m['thr']:>6.2f} {m['se']*100:>7.1f} {m['sp']*100:>7.1f} "
              f"{m['ppv']*100:>7.1f} {m['acc']*100:>7.1f}")
    if best:
        print(f"\n拟定工作点 thr={best['thr']:.2f}（Se≥90% 下 Acc 最高）："
              f"Se={best['se']*100:.1f}% Sp={best['sp']*100:.1f}% "
              f"PPV={best['ppv']*100:.1f}% Acc={best['acc']*100:.1f}%")

    # 规则版对照（同一批样本）
    print("\n== 规则版对照 ==")
    rb = None
    for m in sweep(y, rule_score):
        if m["se"] >= 0.90 and (rb is None or m["acc"] > rb["acc"]):
            rb = m
    if rb:
        print(f"规则版最优工作点 thr={rb['thr']:.2f}：Se={rb['se']*100:.1f}% "
              f"Sp={rb['sp']*100:.1f}% PPV={rb['ppv']*100:.1f}% Acc={rb['acc']*100:.1f}%")

    # 全量拟合 + 保存
    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, class_weight="balanced"))
    pipe.fit(X, y)
    coefs = pipe.named_steps["logisticregression"].coef_[0]
    print("\n全量模型系数（标准化后）：")
    for f, c in sorted(zip(FEATURES, coefs), key=lambda kv: -abs(kv[1])):
        print(f"  {f:>16s} {c:+.3f}")

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    import joblib
    joblib.dump({"pipeline": pipe, "features": FEATURES, "cv_thr": best["thr"] if best else 0.5}, args.save)
    print("已保存模型：", args.save)


if __name__ == "__main__":
    main()
