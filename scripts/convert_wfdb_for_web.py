"""把本地 WFDB 格式心电记录转成网页端可上传的 .npy / .csv。

支持格式：
  · .hea + .dat（标准 WFDB）：European ST-T、PTB-XL、BUT-QDB 的 ECG
  · .hea + .mat（MATLAB v4）：中文 12 导联心律失常库（CPSC2018）

用法示例：
  .venv/Scripts/python.exe scripts/convert_wfdb_for_web.py \
      "D:/data box/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/records100/00000/00001_lr" \
      --channel II --out data/_upload_ptbxl.npy

  .venv/Scripts/python.exe scripts/convert_wfdb_for_web.py \
      "D:/data box/european-st-t-database-1.0.0/e0103" \
      --channel MLIII --out data/_upload_est.npy

转换完后，在网页端「上传文件」里选择生成的 .npy，
采样率按 .hea 里的实际值填（PTB-XL records100 是 100 Hz，records500 是 500 Hz，
European ST-T 是 250 Hz，BUT-QDB 是 1000 Hz，中文 12 导联是 500 Hz）。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def find_record_path(input_path: str) -> str:
    """如果用户给的是目录或带扩展名的路径，统一找到 WFDB record 前缀。"""
    p = Path(input_path)
    if p.is_dir():
        # BUT-QDB：文件夹名就是记录名
        heas = list(p.glob("*.hea"))
        if len(heas) == 1:
            return str(heas[0].with_suffix(""))
        # 优先取 *_ECG.hea
        for h in heas:
            if h.stem.endswith("_ECG"):
                return str(h.with_suffix(""))
        raise ValueError(f"目录 {input_path} 里没有唯一 .hea 文件：{[h.name for h in heas]}")
    if p.suffix in (".hea", ".dat", ".mat", ".atr"):
        return str(p.with_suffix(""))
    return input_path


def main():
    parser = argparse.ArgumentParser(description="WFDB 记录 → 网页端可上传的 .npy/.csv")
    parser.add_argument("record", help="WFDB 记录路径（可带或不带 .hea/.dat 后缀，也可是 BUT-QDB 记录目录）")
    parser.add_argument("--channel", default=None, help="导联名（如 II、MLIII、ECG）或通道索引 0/1/2；不写则默认第 0 通道")
    parser.add_argument("--out", default="data/_upload_converted.npy", help="输出文件路径，默认 data/_upload_converted.npy")
    parser.add_argument("--csv", action="store_true", help="同时输出一份 .csv（与 .npy 同路径改后缀）")
    args = parser.parse_args()

    import wfdb

    rec = find_record_path(args.record)
    r = wfdb.rdrecord(rec, physical=True)

    # 通道选择
    sig_names = [str(n) for n in r.sig_name]
    if args.channel is None:
        ch = 0
        ch_name = sig_names[ch]
        print(f"未指定通道，使用第 0 通道：{ch_name}")
    elif args.channel.isdigit():
        ch = int(args.channel)
        ch_name = sig_names[ch]
    else:
        if args.channel not in sig_names:
            raise ValueError(f"记录里没有导联 '{args.channel}'，可用：{sig_names}")
        ch = sig_names.index(args.channel)
        ch_name = args.channel

    x = np.asarray(r.p_signal[:, ch], dtype=np.float64)
    fs = float(r.fs)
    duration_s = len(x) / fs

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, x)
    print(f"✅ 已保存 {out_path}")
    print(f"   导联：{ch_name}  |  采样率：{fs:.0f} Hz  |  时长：{duration_s/60:.1f} 分钟  |  点数：{len(x)}")

    if args.csv:
        csv_path = out_path.with_suffix(".csv")
        np.savetxt(csv_path, x, delimiter=",", fmt="%.6f")
        print(f"✅ 已保存 {csv_path}")

    print(f"\n下一步：在网页端「上传文件」里选择 {out_path}，采样率填 {fs:.0f} Hz。")


if __name__ == "__main__":
    main()
