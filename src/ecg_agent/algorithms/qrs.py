"""QRS 检测。

两个独立检测器：
  1. `pan_tompkins` —— 自实现 Pan-Tompkins（1985），作为主检测器；
  2. `xqrs`          —— wfdb 库的 XQRS，作为"第二意见"，用于五层策略里的
                       事件二次确认（两个独立算法都同意才升级预警）。

Pan-Tompkins 原流程：5–15 Hz 带通 → 五点微分 → 逐点平方 → 150 ms 滑窗积分
→ 双阈值自适应判决（SPKI/NPKI）+ 回溯搜索（THR2）。
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sps

__all__ = ["pan_tompkins", "xqrs"]


def _pt_stages(x: np.ndarray, fs: float):
    """步骤 1–4：带通 → 微分 → 平方 → 积分。返回 (带通信号, 积分信号)。"""
    nyq = fs / 2.0
    hi = min(15.0, nyq * 0.9)
    b, a = sps.butter(3, [5.0 / nyq, hi / nyq], btype="band")
    bp = sps.filtfilt(b, a, x)

    # 五点微分 y[n] = (-x[n-2] -2x[n-1] + 2x[n+1] + x[n+2]) / (8T)
    kernel = np.array([-1.0, -2.0, 0.0, 2.0, 1.0]) * (fs / 8.0)
    d = np.convolve(bp, kernel, mode="same")
    sq = d * d

    w = max(1, int(round(0.150 * fs)))
    integ = np.convolve(sq, np.ones(w) / w, mode="same")
    return bp, integ


def _localize_r(bp: np.ndarray, idx: int, fs: float) -> int:
    """把积分峰映射回 R 波位置。

    注意：这里的滑动平均积分是"居中"实现（convolve 'same'），积分峰与 R 波
    基本对齐（偏差在一个积分窗之内）。因此用 ±100 ms 双侧窗，按 |幅值| 取极值
    ——单向左侧搜索会把一半心拍定位到偏左 150 ms 的位置。
    """
    w = max(1, int(round(0.10 * fs)))
    lo = max(0, idx - w)
    hi = min(len(bp), idx + w + 1)
    if hi <= lo:
        return idx
    seg = bp[lo:hi]
    return lo + int(np.argmax(np.abs(seg)))


def pan_tompkins(x: np.ndarray, fs: float, init_sec: float = 2.0) -> np.ndarray:
    """返回 R 波样本下标（升序、int 数组）。"""
    x = np.asarray(x, dtype=np.float64)
    if len(x) < fs:  # 太短
        return np.array([], dtype=int)

    bp, integ = _pt_stages(x, fs)
    n = len(integ)
    learn = min(n, int(round(init_sec * fs)))

    # —— 学习阶段初始化阈值 ——
    # 积分信号阈值（SPKI/NPKI，决定"是否可能是心拍"）
    spki = float(np.max(integ[:learn])) * 0.25
    npki = float(np.mean(integ[:learn])) * 0.125
    if spki <= npki:
        spki = float(np.max(integ[:learn])) * 0.5
    # 带通信号幅度阈值（SPKF/NPKF，决定"是心拍还是 T 波/噪声"）
    absbp = np.abs(bp)
    spkf = float(np.max(absbp[:learn])) * 0.25
    npkf = float(np.mean(absbp[:learn])) * 0.5
    if spkf <= npkf:
        spkf = float(np.max(absbp[:learn])) * 0.5

    cand, _ = sps.find_peaks(integ)
    r_peaks: list[int] = []
    r_amp: list[float] = []  # 每个检测拍在带通信号上的幅值
    r_val: list[float] = []  # 每个检测拍的积分信号值（近邻竞争的评判依据）
    rr_avg1: list[float] = []  # 常规 RR（含轻度异常）
    rr_avg2: list[float] = []  # 严格 RR（用于回溯限幅）
    last_r: int | None = None
    min_rr = 0.25  # s，物理下限

    def _rr_means():
        m1 = float(np.mean(rr_avg1[-8:])) if rr_avg1 else None
        m2 = float(np.mean(rr_avg2[-8:])) if rr_avg2 else None
        return m1, m2

    def _accept(idx_integ: int, from_searchback: bool = False) -> bool:
        nonlocal last_r, spki, spkf, npkf
        r = _localize_r(bp, idx_integ, fs)
        amp = float(absbp[r])
        val = float(integ[idx_integ])

        comp_win = min_rr
        if last_r is not None and (r - last_r) / fs < comp_win:
            # —— 近邻竞争：0.25 s 内的两个候选是"同一次 QRS 的双峰"，
            # 留带通幅值大的那个。108 号记录是"小 R 波 + 大 S 波"形态，
            # 靠这条才能锁到大 S。窗口不能放宽到 T 波位置（0.3 s 左右），
            # 否则快节律和早搏会被误杀（202/203/222 的教训）。
            if amp > r_amp[-1]:
                r_peaks[-1] = r
                r_val[-1] = val
                r_amp[-1] = amp
                last_r = r
                return True
            return False
        # —— 第二级幅度判决（PT 论文的 SPKF/NPKF 机制）——
        # 积分峰过了阈值，还要在带通信号上足够"像心拍"，否则按噪声处理。
        # 这一步专门挡 T 波/S 波级别的低幅峰（实测误检的幅值只有正常拍的 ~0.3）。
        thrf = npkf + 0.25 * (spkf - npkf)
        if from_searchback:
            thrf *= 0.5
        # —— T 波判别（论文口径）——：候选落在 T 波半径内（RR < 0.36*RR_AVG），
        # 且带通幅值明显弱于 QRS（< 0.5 × SPKF），判为 T 波。
        # 平均 RR < 0.6 s（心率 > 100）时不做此判断——快节律下不存在
        # "落在 T 波位置"的余量，做了会误杀真拍（203 号记录的教训）。
        # 遗留问题：114/119 号记录的 T 波在带通幅值上不比 QRS 弱，
        # 这条判据挡不住（详见 docs/算法进展.md 的误差分析）。
        _, m2 = _rr_means()
        if m2 is not None and last_r is not None and m2 > 0.6:
            rr_cur = (r - last_r) / fs
            if rr_cur < 0.36 * m2 and amp < 0.5 * spkf:
                npkf = 0.125 * amp + 0.875 * npkf
                return False
        if amp < thrf:
            npkf = 0.125 * amp + 0.875 * npkf
            return False

        p = float(integ[idx_integ])
        if from_searchback:
            spki = 0.25 * p + 0.75 * spki
        else:
            spki = 0.125 * p + 0.875 * spki
        spkf = 0.125 * amp + 0.875 * spkf
        r_peaks.append(r)
        r_amp.append(amp)
        r_val.append(val)
        if last_r is not None:
            rr = (r - last_r) / fs
            m1, m2 = _rr_means()
            if m2 is None:
                rr_avg1.append(rr)
                rr_avg2.append(rr)
            elif 0.92 * m2 <= rr <= 1.16 * m2:
                rr_avg1.append(rr)
                rr_avg2.append(rr)
            elif 1.16 * m2 < rr <= 1.66 * m2 or 0.36 * m2 <= rr < 0.92 * m2:
                rr_avg1.append(rr)
            # rr < 0.36*m2 → 疑似 T 波，不计入统计（仍保留检测结果，
            # 因为真房颤/早搏本来就会短 RR，保守起见不丢）
            elif rr < 0.36 * m2:
                rr_avg1.append(rr)
        last_r = r
        return True

    for i in cand:
        p = float(integ[i])

        # —— 回溯搜索：距上一个 R 波超过 1.66*RR_AVG2 就回头看 ——
        _, m2 = _rr_means()
        if last_r is not None and m2 is not None:
            rr_missed_limit = 1.66 * m2
            if (i - last_r) / fs > rr_missed_limit:
                lo = last_r + int(round(0.20 * fs))
                hi = min(n, last_r + int(round(rr_missed_limit * fs)))
                if hi > lo:
                    j = lo + int(np.argmax(integ[lo:hi]))
                    thr2 = 0.5 * (npki + 0.25 * (spki - npki))
                    if integ[j] > thr2:
                        _accept(j, from_searchback=True)

        thr1 = npki + 0.25 * (spki - npki)
        if p > thr1:
            _accept(i)
        else:
            npki = 0.125 * p + 0.875 * npki

    return np.asarray(r_peaks, dtype=int)


def xqrs(x: np.ndarray, fs: float) -> np.ndarray:
    """wfdb 的 XQRS 检测器（独立第二意见）。"""
    import warnings

    import wfdb.processing as wp

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            peaks = wp.xqrs_detect(np.asarray(x, dtype=np.float64), fs=fs, verbose=False)
        except Exception:
            return np.array([], dtype=int)
    return np.asarray(peaks, dtype=int)


def compare_rr(r1: np.ndarray, r2: np.ndarray, fs: float, tol: float = 0.15) -> float:
    """两个检测器的一致性（±tol 秒内互认为同拍的比例，F1）。"""
    if len(r1) == 0 or len(r2) == 0:
        return 0.0
    import bisect

    a = np.asarray(r1) / fs
    b = np.asarray(r2) / fs
    tp = 0
    for t in a:
        k = bisect.bisect_left(b, t - tol)
        if k < len(b) and abs(b[k] - t) <= tol:
            tp += 1
            continue
        if k > 0 and abs(b[k - 1] - t) <= tol:
            tp += 1
    prec = tp / len(a)
    rec = tp / len(b)
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
