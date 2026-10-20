"""从 PPG-DaLiA 的 RespiBAN h5 文件里提取 ECG 通道。

数据说明（已核实）：
  - 目录：PPG_FieldStudy/S1..S15/S*_RespiBAN.h5（默认取项目同级目录，可用 PPG_DALIA_ROOT 覆盖）
  - 顶层 group 以设备 MAC 命名，内含 raw/channel_1..8
  - channel_1 的 sensor 属性为 "ECG"，采样率 700 Hz，16 bit 无符号、以 32768 为零点
  - 每个受试者约 2.5 小时

注意：本数据集没有逐拍心电标注，只能用于"真实长时程"演示与报警率统计，
      不能用于报告敏感度/特异度（那要用 MIT-BIH 等有标注数据）。
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# 项目根目录（src/ecg_agent/io/respiban.py → 上 3 级）
PROJECT_ROOT = Path(__file__).resolve().parents[3]
# 原始 h5 数据目录：可用环境变量 PPG_DALIA_ROOT 覆盖（默认为项目同级的 PPG_FieldStudy）
DATASET_ROOT = Path(os.environ.get("PPG_DALIA_ROOT", str(PROJECT_ROOT.parent / "PPG_FieldStudy")))
CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "ppg_dalia"
_ECG_OFFSET = 32768.0  # 16bit 无符号数据的零点


def subject_ids() -> list[str]:
    return [f"S{i}" for i in range(1, 16)]


def h5_path(subject: str) -> Path:
    return DATASET_ROOT / subject / f"{subject}_RespiBAN.h5"


def cache_path(subject: str) -> Path:
    return CACHE_DIR / f"{subject}_ecg.npy"


def read_ecg(subject_or_path: str | Path, channel: str = "channel_1"):
    """读取 ECG。

    返回 (ecg, fs)：
      ecg: float64，已去掉 32768 直流偏置的 ADC 计数（任意单位 a.u.，未校准成 mV）
      fs : 采样率 Hz（原始 700）
    """
    import h5py

    path = h5_path(subject_or_path) if str(subject_or_path).startswith("S") and "_" not in str(subject_or_path) else Path(subject_or_path)
    with h5py.File(path, "r") as f:
        grp = f[list(f.keys())[0]]
        fs = int(grp.attrs["sampling rate"])
        ecg = grp["raw"][channel][:, 0].astype(np.float64) - _ECG_OFFSET
    return ecg, fs


def load_or_build_cache(subject: str, overwrite: bool = False):
    """优先读缓存 npy，没有就现读 h5 并落盘（约 50 MB/例）。"""
    cp = cache_path(subject)
    if cp.exists() and not overwrite:
        ecg = np.load(cp)
        return ecg, 700
    ecg, fs = read_ecg(subject)
    cp.parent.mkdir(parents=True, exist_ok=True)
    np.save(cp, ecg.astype(np.float32))
    return ecg, fs
