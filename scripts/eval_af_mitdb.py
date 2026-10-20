"""房颤样心律筛查评价（MIT-BIH 节律标注）。

评价口径：
  · 金标准：mitdb 逐窗节律标注（全库扫描）——
      '(AFIB'/'(AFL' 段内的窗口 → 阳性；(N' 段内的窗口 → 阴性；跨段窗剔除
  · 窗口：30 s 不重叠；窗内 R 波由本项目的 Pan-Tompkins 检测器给出
  · 输出：阈值扫描表（Se/Sp/PPV/Acc）+ 特征单变量对照
  · 说明：筛查阶段刻意做成高敏感/中等特异，特异度的补足靠五层策略的
    L1 质量门控、L3 二次确认与 L4 人审（见 scripts/ablation_layers.py）

用法：
    python scripts/eval_af_mitdb.py [--out results/af_mitdb.csv]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ecg_agent.algorithms import rhythm as R  # noqa: E402
from ecg_agent.algorithms.preprocess import preprocess  # noqa: E402
from ecg_agent.algorithms.qrs import pan_tompkins  # noqa: E402
from ecg_agent.io import mitdb  # noqa: E402

WIN = 30.0


def prf(tp, fp, tn, fn):
    se = tp / (tp + fn) if tp + fn else 0.0
    sp = tn / (tn + fp) if tn + fp else 0.0
    ppv = tp / (tp + fp) if tp + fp else 0.0
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "se": se, "sp": sp, "ppv": ppv, "acc": acc}


def collect_windows():
    rows = []
    for rec in mitdb.record_ids():
        x, fs, ann = mitdb.read(rec, channel=0)
        segs = mitdb.rhythm_timeline(ann, fs)
        if not any(k in ("af", "normal") for _, _, k in segs):
            continue
        xc, _ = preprocess(x, fs, target_fs=None)
        rpk = pan_tompkins(xc, fs)
        r_t = np.asarray(rpk) / fs
        t_end = float(r_t[-1]) if len(r_t) else 0.0
        n_af = n_nm = 0
        t = 0.0
        while t + WIN <= t_end:
            lab = mitdb.label_window(segs, t, t + WIN)
            if lab in ("af", "normal"):
                seg = r_t[(r_t >= t) & (r_t < t + WIN)]
                rr = np.diff(seg)
                if len(rr) >= 6:
                    f = R._af_features(rr)
                    if f:
                        rows.append({"record": rec, "t_start": round(t, 1), "label": lab,
                                     "score": R.af_score_rules(f), **f})
                        n_af += lab == "af"
                        n_nm += lab == "normal"
            t += WIN
        if n_af or n_nm:
            print(f"  {rec}: AF 窗 {n_af}, 窦律窗 {n_nm}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/af_mitdb.csv")
    args = ap.parse_args()

    print("逐记录收集窗口：")
    rows = collect_windows()
    if not rows:
        print("没有可用窗口——检查节律标注解析")
        return
    y = np.array([1 if r["label"] == "af" else 0 for r in rows])
    s = np.array([r["score"] for r in rows])
    print(f"\n窗口总数 {len(rows)}（AF {int(y.sum())} / 窦律 {int((1-y).sum())}），"
          f"来自 {len(set(r['record'] for r in rows))} 条记录")

    print(f"\n{'阈值':>6} {'Se%':>7} {'Sp%':>7} {'PPV%':>7} {'Acc%':>7}")
    best = None
    for thr in np.arange(0.2, 0.95, 0.05):
        p = (s >= thr).astype(int)
        tp = int(((y == 1) & (p == 1)).sum()); fp = int(((y == 0) & (p == 1)).sum())
        tn = int(((y == 0) & (p == 0)).sum()); fn = int(((y == 1) & (p == 0)).sum())
        m = prf(tp, fp, tn, fn)
        star = ""
        # 选工作点：Se ≥ 90% 前提下 Acc 最高
        if m["se"] >= 0.90 and (best is None or m["acc"] > best[1]["acc"]):
            best = (thr, m); star = ""
        print(f"{thr:>6.2f} {m['se']*100:>7.1f} {m['sp']*100:>7.1f} {m['ppv']*100:>7.1f} {m['acc']*100:>7.1f}")

    if best:
        thr, m = best
        print(f"\n拟定工作点 thr={thr:.2f}（约束 Se≥90% 下 Acc 最高）："
              f"Se={m['se']*100:.1f}%  Sp={m['sp']*100:.1f}%  PPV={m['ppv']*100:.1f}%  Acc={m['acc']*100:.1f}%")

    print("\n单变量对照（窗口级）：")
    for key, thr in [("cv_rr", 0.12), ("shannon_drr", 0.6), ("pnn50", 0.4), ("sample_entropy", 0.6)]:
        v = np.array([r[key] for r in rows])
        pp = (v >= thr).astype(int)
        tp = int(((y == 1) & (pp == 1)).sum()); fp = int(((y == 0) & (pp == 1)).sum())
        tn = int(((y == 0) & (pp == 0)).sum()); fn = int(((y == 1) & (pp == 0)).sum())
        mm = prf(tp, fp, tn, fn)
        print(f"  {key:>16s} ≥{thr}: Se={mm['se']*100:5.1f}% Sp={mm['sp']*100:5.1f}% Acc={mm['acc']*100:5.1f}%")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    import csv
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("\n已写出", args.out)


if __name__ == "__main__":
    main()
