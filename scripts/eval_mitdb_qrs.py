"""在 MIT-BIH 心律失常数据库（mitdb）上评价 QRS 检测。

用法：
    python scripts/eval_mitdb_qrs.py [--limit N] [--out results/mitdb_qrs.csv]
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import wfdb  # noqa: E402

from ecg_agent.algorithms.metrics import beat_metrics  # noqa: E402
from ecg_agent.algorithms.preprocess import preprocess  # noqa: E402
from ecg_agent.algorithms.qrs import pan_tompkins, xqrs  # noqa: E402

MITDB = "data/mitdb/mit-bih-arrhythmia-database-1.0.0"
BEAT_SYMS = set("NLRBAaJSVrFejnE/fQ?")
DETECTORS = {"pt": pan_tompkins, "xqrs": xqrs}


def ref_beats(rec: str):
    ann = wfdb.rdann(os.path.join(MITDB, rec), "atr")
    mask = np.array([s in BEAT_SYMS for s in ann.symbol])
    return ann.sample[mask]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条记录（调试用）")
    ap.add_argument("--out", default="results/mitdb_qrs.csv")
    ap.add_argument("--detector", default="pt", choices=["pt", "xqrs"], help="用哪个检测器")
    args = ap.parse_args()

    recs = sorted({os.path.basename(p)[:-4] for p in glob.glob(os.path.join(MITDB, "*.hea"))
                   if os.path.exists(p[:-4] + ".atr")})
    if args.limit:
        recs = recs[: args.limit]

    rows = []
    print(f"{'记录':>6} {'参考':>7} {'检出':>7} {'TP':>7} {'FP':>6} {'FN':>6} {'Se%':>7} {'PPV%':>7} {'耗时s':>6}")
    for rec in recs:
        r = wfdb.rdrecord(os.path.join(MITDB, rec), physical=True)
        fs = r.fs
        ref = ref_beats(rec)
        x = r.p_signal[:, 0]
        xc, _ = preprocess(x, fs, target_fs=None)  # 评测保持各自原生采样率
        t0 = time.time()
        det = DETECTORS[args.detector](xc, fs)
        dt = time.time() - t0
        m = beat_metrics(det, ref, fs)
        rows.append({"record": rec, "fs": fs, **m, "seconds": dt})
        print(f"{rec:>6} {m['n_ref']:>7} {m['n_det']:>7} {m['tp']:>7} {m['fp']:>6} {m['fn']:>6} "
              f"{m['se']*100:>7.2f} {m['ppv']*100:>7.2f} {dt:>6.1f}")

    tp = sum(r["tp"] for r in rows); fp = sum(r["fp"] for r in rows); fn = sum(r["fn"] for r in rows)
    se = tp / (tp + fn) if tp + fn else 0
    ppv = tp / (tp + fp) if tp + fp else 0
    print(f"\n总计：Se={se*100:.2f}%  PPV={ppv*100:.2f}%  "
          f"(TP={tp} FP={fp} FN={fn}, {len(rows)} 条记录)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    import csv
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("已写出", args.out)


if __name__ == "__main__":
    main()
