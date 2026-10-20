"""从 PPG-DaLiA 的 RespiBAN h5 提取 ECG 并缓存（700 Hz → 250 Hz）。

产出：
  data/cache/ppg_dalia/SX_ecg.npy       （float32，250 Hz，已去 32768 直流偏置）
  data/cache/ppg_dalia/manifest.csv     （时长、采样率、均值/标准差等）

说明：本数据集无逐拍心电标注，只能用于"真实长时程"演示与报警率统计，
      不用于报告敏感度/特异度（那用 MIT-BIH）。

用法：
    python scripts/extract_ppg_dalia.py [--overwrite]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ecg_agent.algorithms.preprocess import resample_to  # noqa: E402
from ecg_agent.io import respiban  # noqa: E402

TARGET_FS = 250.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_dir = respiban.CACHE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for sid in respiban.subject_ids():
        cp = out_dir / f"{sid}_ecg.npy"
        if cp.exists() and not args.overwrite:
            ecg = np.load(cp)
            fs = TARGET_FS
        else:
            raw, fs_in = respiban.read_ecg(sid)
            ecg = resample_to(raw, fs_in, TARGET_FS).astype(np.float32)
            fs = TARGET_FS
            np.save(cp, ecg)
        dur_h = len(ecg) / fs / 3600
        rows.append({"subject": sid, "fs": fs, "n_samples": len(ecg), "hours": round(dur_h, 3),
                     "mean": round(float(ecg.mean()), 1), "std": round(float(ecg.std()), 1),
                     "min": int(ecg.min()), "max": int(ecg.max())})
        print(f"{sid}: {dur_h:.2f} 小时已缓存（{len(ecg)} 采样 @ {fs:.0f} Hz）")

    import csv
    with open(out_dir / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    total_h = sum(r["hours"] for r in rows)
    print(f"\n共 {len(rows)} 例，累计 {total_h:.1f} 小时；清单见 {out_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
