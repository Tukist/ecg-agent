"""心率与心率变异性（HRV）指标。

全部基于 R 波序列（RR 间期），单位统一用秒 / 毫秒。
"""

from __future__ import annotations

import numpy as np


def rr_intervals(r_peaks: np.ndarray, fs: float) -> np.ndarray:
    """RR 间期（秒）。"""
    r = np.asarray(r_peaks, dtype=np.float64)
    if len(r) < 2:
        return np.array([])
    return np.diff(r) / fs


def hr_series(r_peaks: np.ndarray, fs: float, win: float = 10.0):
    """按滑动窗给出心率曲线（bpm）。

    返回 (times, hr)：times 为窗口中心时刻（秒），hr 为该窗内 RR 换算的心率。
    """
    r = np.asarray(r_peaks, dtype=np.float64) / fs
    if len(r) < 2:
        return np.array([]), np.array([])
    total = r[-1]
    times, hrs = [], []
    t = win / 2
    while t < total:
        lo, hi = t - win / 2, t + win / 2
        seg = r[(r >= lo) & (r < hi)]
        if len(seg) >= 2:
            rr = np.diff(seg).mean()
            times.append(t)
            hrs.append(60.0 / rr)
        t += win
    return np.array(times), np.array(hrs)


def rmssd(rr: np.ndarray) -> float:
    if len(rr) < 2:
        return 0.0
    return float(np.sqrt(np.mean(np.diff(rr) ** 2)))


def sdnn(rr: np.ndarray) -> float:
    return float(np.std(rr)) if len(rr) > 1 else 0.0


def pnn50(rr: np.ndarray) -> float:
    """相邻 RR 差值超过 50 ms 的比例。"""
    if len(rr) < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(rr)) > 0.05))


def sample_entropy(x: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    """样本熵（Richman & Moorman 2000）。

    m 维与 m+1 维模板的匹配概率之比取负对数；数值越小越"规整"。
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 20:
        return 0.0
    r = r_factor * np.std(x)
    if r <= 0:
        return 0.0

    def _count(mm: int) -> int:
        tmpl = np.lib.stride_tricks.sliding_window_view(x, mm)
        cnt = 0
        for i in range(len(tmpl) - 1):
            d = np.max(np.abs(tmpl[i + 1:] - tmpl[i]), axis=1)
            cnt += int(np.sum(d <= r))
        return cnt

    b = _count(m)
    a = _count(m + 1)
    if b == 0 or a == 0:
        return 0.0
    return float(-np.log(a / b))


def shannon_entropy_rr(rr: np.ndarray, bins: int = 20) -> float:
    """RR 序列的概率分布香农熵（归一化到 0–1）。

    房颤时 RR 分布被"打散"，熵显著升高——Tateno 等人经典房颤判据的核心量。
    """
    rr = np.asarray(rr)
    if len(rr) < 4:
        return 0.0
    hist, _ = np.histogram(rr, bins=bins)
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)) / np.log(bins))


def poincare_features(rr: np.ndarray) -> dict:
    """Poincaré 图特征：SD1（短时变异）与 SD2（长时变异）。"""
    if len(rr) < 2:
        return {"sd1": 0.0, "sd2": 0.0, "sd1_sd2": 0.0}
    x, y = rr[:-1], rr[1:]
    sd1 = float(np.sqrt(0.5) * np.std(y - x))
    sd2 = float(np.sqrt(0.5) * np.std(y + x))
    return {"sd1": sd1, "sd2": sd2, "sd1_sd2": sd1 / sd2 if sd2 > 0 else 0.0}


def hrv_features(rr: np.ndarray) -> dict:
    """汇总 HRV 特征，供异常筛查与报告使用。"""
    rr = np.asarray(rr, dtype=np.float64)
    rr = rr[(rr > 0.2) & (rr < 4.0)]  # 去掉明显异常的间期
    if len(rr) < 4:
        return {"n": int(len(rr)), "mean_rr": 0.0, "hr": 0.0, "sdnn": 0.0, "rmssd": 0.0,
                "pnn50": 0.0, "cv_rr": 0.0, "sample_entropy": 0.0, "shannon_rr": 0.0,
                "sd1": 0.0, "sd2": 0.0, "sd1_sd2": 0.0}
    po = poincare_features(rr)
    mean_rr = float(np.mean(rr))
    return {
        "n": int(len(rr)),
        "mean_rr": mean_rr,
        "hr": 60.0 / mean_rr,
        "sdnn": sdnn(rr),
        "rmssd": rmssd(rr),
        "pnn50": pnn50(rr),
        "cv_rr": float(np.std(rr) / mean_rr),
        "sample_entropy": sample_entropy(rr),
        "shannon_rr": shannon_entropy_rr(rr),
        **po,
    }
