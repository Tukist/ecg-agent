"""短记录分析（< 30 s）——补充长记录 rhythm 筛查覆盖不到的场景。

主要面向 PTB-XL、中文 12 导联等 10 秒静态心电记录：
  · 给出逐拍摘要（总数、心率统计、整齐度）
  · 基于 RR 模式 + 形态宽度提示「疑似早搏」
  · 提示长间歇（>3 s）

注意：
  · 本模块只做提示级输出，不替代 rhythm.py 的 30 s 窗筛查；
  · 长记录（>= 30 s）仍走原有流程，本模块结果仅作附加参考；
  · 疑似早搏的判定是轻量规则，不是 AAMI 五类分类器，PPV 有限。
"""

from __future__ import annotations

import numpy as np


def _qrs_width(x: np.ndarray, fs: float, r: int, win_s: float = 0.12) -> float:
    """返回单拍 QRS 宽度（秒）：以 R 峰为中心，包络降到峰值 30% 的持续时间。"""
    w = int(round(win_s * fs))
    a, b = r - w, r + w
    if a < 0 or b > len(x):
        return 0.0
    seg = x[a:b] - x[a:b].mean()
    amp = float(np.abs(seg).max())
    if amp < 1e-9:
        return 0.0
    env = np.abs(seg)
    above = env >= 0.3 * amp
    return float(above.sum()) / fs


def _build_template(x: np.ndarray, fs: float, r_peaks: np.ndarray,
                    win_s: float = 0.12) -> tuple[np.ndarray, float]:
    """用记录内所有拍的中位形态作为正常窦律模板，返回（模板, 幅度中位数）。"""
    w = int(round(win_s * fs))
    segs, amps = [], []
    for r in r_peaks:
        a, b = r - w, r + w
        if a < 0 or b > len(x):
            continue
        seg = x[a:b] - x[a:b].mean()
        amp = float(np.abs(seg).max())
        if amp < 1e-9:
            continue
        amps.append(amp)
        segs.append(seg / (amp + 1e-12))
    if len(segs) < 3:
        return np.zeros(2 * w), 1.0
    med = np.median(np.vstack(segs), axis=0)
    norm = np.linalg.norm(med) + 1e-12
    return med / norm, float(np.median(amps)) if amps else 1.0


def analyze(x: np.ndarray, fs: float, r_peaks: np.ndarray) -> dict:
    """分析短记录，返回结构化摘要。

    返回字典示例：
      {
        "n_beats": 15,
        "hr_mean": 78.0,
        "hr_min": 70.0,
        "hr_max": 85.0,
        "regular": True,
        "rr_cv": 0.05,
        "suspected_ectopy": 2,
        "suspected_wide_qrs": 1,
        "pause_events": [(t_start, t_end, rr_s), ...],
        "beat_details": [{"idx": 0, "time_s": 0.5, "rr_prev_s": 0.8,
                          "rr_next_s": 0.9, "width_s": 0.09,
                          "morph_corr": 0.98, "flag": "normal"}, ...]
      }
    """
    if len(r_peaks) < 3:
        return {
            "n_beats": len(r_peaks),
            "hr_mean": 0.0, "hr_min": 0.0, "hr_max": 0.0,
            "regular": False, "rr_cv": 0.0,
            "suspected_ectopy": 0, "suspected_wide_qrs": 0,
            "pause_events": [], "beat_details": [],
            "too_few_beats": True,
        }

    rr = np.diff(r_peaks) / fs
    rr_times = r_peaks[1:] / fs  # rr[i] 对应第 i+1 拍之后的时间
    hr = 60.0 / rr

    template, amp_med = _build_template(x, fs, r_peaks)

    # 逐拍分析（跳过首拍，因为它没有 rr_prev）
    beat_details = []
    n_ectopy = 0
    n_wide = 0
    pause_events = []

    for i, r in enumerate(r_peaks):
        t = r / fs
        rr_prev = float(rr[i - 1]) if i > 0 else None
        rr_next = float(rr[i]) if i < len(rr) else None

        # 局部主导 RR（中位）
        lo, hi = max(0, i - 1), min(len(rr), i + 1)
        local_rrs = rr[lo:hi]
        rr_local = float(np.median(local_rrs)) if len(local_rrs) else (rr_prev or 0.8)

        width = _qrs_width(x, fs, r)
        morph_corr = 0.0
        if np.linalg.norm(template) > 1e-6:
            w = len(template) // 2
            a, b = r - w, r + w
            if a >= 0 and b <= len(x):
                seg = x[a:b] - x[a:b].mean()
                amp = float(np.abs(seg).max())
                if amp > 1e-9:
                    seg_n = seg / (amp + 1e-12)
                    morph_corr = float(np.dot(seg_n, template))

        flag = "normal"
        # 早搏判定：提前出现 + 代偿间歇，或形态明显不同
        if rr_prev is not None and rr_next is not None:
            premature = rr_prev < 0.78 * rr_local and rr_next > 1.15 * rr_local
            morph_abnormal = width > 0.12 and morph_corr < 0.70
            if premature or morph_abnormal:
                flag = "suspected_ectopy"
                n_ectopy += 1

        # 宽 QRS 单独计数（即使不被判早搏）
        if width > 0.12:
            n_wide += 1

        # 长间歇：rr_next 表示该拍之后到下一拍的间隔
        if rr_next is not None and rr_next >= 3.0:
            pause_events.append((t, t + rr_next, rr_next))

        beat_details.append({
            "idx": i,
            "time_s": round(t, 3),
            "rr_prev_s": round(rr_prev, 3) if rr_prev is not None else None,
            "rr_next_s": round(rr_next, 3) if rr_next is not None else None,
            "width_s": round(width, 3),
            "morph_corr": round(morph_corr, 3),
            "flag": flag,
        })

    rr_cv = float(np.std(rr) / np.mean(rr)) if np.mean(rr) > 0 else 0.0
    regular = rr_cv < 0.15 and len(pause_events) == 0

    return {
        "n_beats": len(r_peaks),
        "duration_s": float(len(x) / fs),
        "hr_mean": float(np.mean(hr)),
        "hr_min": float(np.min(hr)),
        "hr_max": float(np.max(hr)),
        "regular": regular,
        "rr_cv": rr_cv,
        "suspected_ectopy": n_ectopy,
        "suspected_wide_qrs": n_wide,
        "pause_events": pause_events,
        "beat_details": beat_details,
        "too_few_beats": False,
    }
