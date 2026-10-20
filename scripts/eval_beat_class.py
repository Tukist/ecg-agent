"""心拍分类评价（AAMI 五类：N/S/V/F/Q）——室性早搏检出的指标来源。

口径（对齐 ANSI/AAMI EC57 的通行做法）：
  · 只评价"本检测器检出并与标注配对成功（±150 ms）"的心拍；
  · 排除 102/104/107/217 四条含起搏节律的记录；
  · 特征 = RR 域（前/后 RR、相对局部主导 RR 的提前量）+ 形态域
    （QRS 宽度、幅度、与记录内中位模板的相关系数、降采样模板）；
  · 分类器 = RandomForest，按"记录"分组 5 折交叉验证（避免记录内泄漏）；
  · 输出：各类 Se/PPV/F1 与混淆矩阵。

用法：
    python scripts/eval_beat_class.py [--out results/beat_class.csv]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import wfdb  # noqa: E402

from ecg_agent.algorithms.metrics import AAMI_MAP, match_beats  # noqa: E402
from ecg_agent.algorithms.preprocess import preprocess  # noqa: E402
from ecg_agent.algorithms.qrs import pan_tompkins  # noqa: E402
from ecg_agent.io import mitdb  # noqa: E402

CLASSES = ["N", "S", "V", "F", "Q"]
TEMPLATE_POINTS = 24


def beat_features(x: np.ndarray, fs: float, r: int, rr_prev: float, rr_next: float,
                  rr_local: float, template: np.ndarray, amp_med: float = 1.0) -> list[float]:
    w = int(round(0.12 * fs))
    a, b = r - w, r + w
    seg = x[a:b] if a >= 0 and b <= len(x) else None
    if seg is None or len(seg) != 2 * w:
        return None
    seg_c = seg - seg.mean()
    n = np.linalg.norm(seg_c) + 1e-12
    corr = float(np.dot(seg_c, template) / (n * (np.linalg.norm(template) + 1e-12)))
    amp = float(np.abs(seg_c).max())
    # QRS 宽度：包络降到峰值 30% 的持续时长
    env = np.abs(seg_c)
    thr = 0.3 * amp
    above = env >= thr
    width = float(above.sum()) / fs
    # 降采样模板
    idx = np.linspace(0, len(seg_c) - 1, TEMPLATE_POINTS).astype(int)
    tmpl = (seg_c[idx] / (amp + 1e-12)).tolist()
    return [rr_prev, rr_next,
            rr_prev / rr_local if rr_local > 0 else 1.0,   # 提前量
            rr_next / rr_local if rr_local > 0 else 1.0,   # 代偿间歇
            rr_prev / rr_next if rr_next > 0 else 1.0,     # 前后不对称
            corr,                                           # 形态相关（与记录中位模板）
            amp, amp / (amp_med + 1e-12),                   # 绝对/相对幅度
            width,                                          # QRS 宽度
            float(np.abs(np.diff(seg_c)).max()),            # 最大斜率
            *tmpl]


def collect_record(rec: str):
    x, fs, ann = mitdb.read(rec, channel=0)
    xc, fs = preprocess(x, fs, target_fs=250.0)
    rp = pan_tompkins(xc, fs)
    ref_s, ref_syms = mitdb.beat_annotations(ann)
    # 把参考标注换算到 250 Hz 时间轴
    ref_t = ref_s / 360.0 * fs
    # 配对：每个参考拍找最近的检出拍
    rows = []
    used = set()
    rr = np.diff(rp) / fs
    # 记录内中位模板
    w = int(round(0.12 * fs))
    segs = []
    for r in rp:
        a, b = r - w, r + w
        if a >= 0 and b <= len(xc):
            s = xc[a:b] - xc[a:b].mean()
            if np.linalg.norm(s) > 0:
                segs.append(s / np.linalg.norm(s))
    if len(segs) < 10:
        return []
    med = np.median(np.vstack(segs), axis=0)
    med /= np.linalg.norm(med) + 1e-12

    # 记录内幅度中位数（后加的相对特征用）
    amps_all = []
    for r in rp:
        a, b = r - w, r + w
        if a >= 0 and b <= len(xc):
            amps_all.append(float(np.abs(xc[a:b] - xc[a:b].mean()).max()))
    amp_med = float(np.median(amps_all)) if amps_all else 1.0

    for rs, sym in zip(ref_t, ref_syms):
        cls = AAMI_MAP.get(sym)
        if cls is None:
            continue
        k = int(np.argmin(np.abs(rp - rs)))
        if k in used or abs(rp[k] - rs) > 0.15 * fs:
            continue
        used.add(k)
        if k == 0 or k >= len(rp) - 1:
            continue
        rr_prev = (rp[k] - rp[k - 1]) / fs
        rr_next = (rp[k + 1] - rp[k]) / fs
        lo, hi = max(0, k - 10), min(len(rr), k + 10)
        rr_local = float(np.median(rr[lo:hi])) if hi > lo else rr_prev
        f = beat_features(xc, fs, int(rp[k]), rr_prev, rr_next, rr_local, med, amp_med)
        if f:
            rows.append((rec, cls, f))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/beat_class.csv")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    recs = [r for r in mitdb.record_ids() if r not in mitdb.AAMI_EXCLUDE]
    if args.limit:
        recs = recs[: args.limit]

    data = []
    for rec in recs:
        rows = collect_record(rec)
        data += rows
        print(f"  {rec}: {len(rows)} 拍")
    X = np.array([d[2] for d in data], dtype=np.float64)
    y = np.array([d[1] for d in data])
    groups = np.array([d[0] for d in data])
    print(f"\n合计 {len(y)} 拍（检出且配对成功），类别分布 "
          f"{ {c: int((y == c).sum()) for c in CLASSES} }")

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold

    gkf = GroupKFold(n_splits=5)
    pred = np.empty(len(y), dtype=object)
    for tr, te in gkf.split(X, y, groups):
        clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                     class_weight="balanced_subsample", n_jobs=-1, random_state=0)
        clf.fit(X[tr], y[tr])
        pred[te] = clf.predict(X[te])

    print(f"\n{'类别':>4} {'支持数':>7} {'Se%':>7} {'PPV%':>7} {'F1%':>7}")
    out_rows = []
    for c in CLASSES:
        tp = int(((y == c) & (pred == c)).sum())
        fn = int(((y == c) & (pred != c)).sum())
        fp = int(((y != c) & (pred == c)).sum())
        se = tp / (tp + fn) if tp + fn else 0.0
        ppv = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * se * ppv / (se + ppv) if se + ppv else 0.0
        print(f"{c:>4} {int((y == c).sum()):>7} {se*100:>7.1f} {ppv*100:>7.1f} {f1*100:>7.1f}")
        out_rows.append({"class": c, "support": int((y == c).sum()), "se": round(se, 4),
                         "ppv": round(ppv, 4), "f1": round(f1, 4)})
    acc = float((pred == y).mean())
    print(f"\n总体准确率（检出拍口径）：{acc*100:.1f}%")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["class", "support", "se", "ppv", "f1"])
        w.writeheader(); w.writerows(out_rows)
    print("已写出", args.out)


if __name__ == "__main__":
    main()
