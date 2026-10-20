"""心电预处理：重采样、去基线漂移、带通滤波、工频陷波。

设计取舍：
  - 去基线用两级中值滤波（200 ms + 600 ms），这是动态心电文献里的经典做法，
    优于单纯高通滤波（后者会引起 ST 段失真——调研报告里专门提到过
    1.0 Hz 高通会把 V1/V2 抬高造成假阳性）。
  - 中值滤波放在重采样之后做，省一半计算量。
"""

from __future__ import annotations

from math import gcd

import numpy as np
from scipy import ndimage
from scipy import signal as sps


def resample_to(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """多相重采样（自带抗混叠滤波）。"""
    if int(fs_in) == int(fs_out):
        return x
    g = gcd(int(round(fs_in)), int(round(fs_out)))
    return sps.resample_poly(x, int(round(fs_out)) // g, int(round(fs_in)) // g)


def remove_baseline_median(x: np.ndarray, fs: float, win1: float = 0.2, win2: float = 0.6) -> np.ndarray:
    """两级中值滤波估计基线漂移并扣除。"""
    w1 = int(round(win1 * fs)) | 1  # 强制奇数窗
    w2 = int(round(win2 * fs)) | 1
    b1 = ndimage.median_filter(x, size=w1, mode="nearest")
    b2 = ndimage.median_filter(b1, size=w2, mode="nearest")
    return x - b2


def bandpass(x: np.ndarray, fs: float, lo: float = 0.5, hi: float = 40.0, order: int = 3) -> np.ndarray:
    nyq = fs / 2.0
    hi = min(hi, nyq * 0.95)
    b, a = sps.butter(order, [lo / nyq, hi / nyq], btype="band")
    return sps.filtfilt(b, a, x)


def notch(x: np.ndarray, fs: float, f0: float = 50.0, q: float = 30.0) -> np.ndarray:
    if f0 >= fs / 2.0 * 0.95:
        return x
    b, a = sps.iirnotch(f0 / (fs / 2.0), q)
    return sps.filtfilt(b, a, x)


def preprocess(
    x: np.ndarray,
    fs: float,
    target_fs: float | None = 250.0,
    use_median_baseline: bool = True,
    mains: float | None = 50.0,
    band: tuple[float, float] = (0.5, 40.0),
) -> tuple[np.ndarray, float]:
    """完整预处理链。

    参数
      target_fs: 统一到的分析采样率；None 表示保持原采样率（评测用）。
      use_median_baseline: False 时改用 0.5 Hz 高通（用于消融对比）。
    返回 (x_clean, fs_out)
    """
    x = np.asarray(x, dtype=np.float64)

    if target_fs is not None and int(round(target_fs)) != int(round(fs)):
        x = resample_to(x, fs, target_fs)
        fs = float(target_fs)

    if use_median_baseline:
        x = remove_baseline_median(x, fs)
    else:
        b, a = sps.butter(2, 0.5 / (fs / 2.0), btype="high")
        x = sps.filtfilt(b, a, x)

    x = bandpass(x, fs, band[0], band[1])
    if mains is not None:
        x = notch(x, fs, f0=mains)
    return x, fs


def prepare_for_quality(x: np.ndarray, fs_in: float, target_fs: float = 250.0) -> tuple[np.ndarray, float]:
    """质量评估专用预处理：只重采样 + 去直流，不做带通/陷波。

    为什么：SQI 要评价"采集到的信号干不干净"，如果在带通滤波之后再算，
    基线漂移和工频干扰已经被滤掉，质量门控就形同虚设（实测注入 10 倍
    基线漂移的噪声，带通后质量分仍有 0.86）。所以质量评估必须在原始
    （仅去直流）信号上做。
    """
    x = np.asarray(x, dtype=np.float64)
    if target_fs is not None and int(round(target_fs)) != int(round(fs_in)):
        x = resample_to(x, fs_in, target_fs)
        fs_in = float(target_fs)
    x = x - np.median(x)
    return x, float(fs_in)
