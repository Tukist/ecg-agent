"""信号质量评估（SQI）——五层防误报策略的第 1 层。

按调研报告确定的规格实现：
  · 6 个子频段：0–1 / 1–5 / 5–15 / 15–50 / 50–100 / 100–250 Hz
  · 12 维特征：6 个能量比 + 6 个能量熵
  · 另加工程上惯用的 3 个 SQI 指标（峰度、QRS 频带功率比、基线功率比）
    —— 它们便宜且可解释，用来把 12 维谱特征"锚"到生理意义上

输出：每个窗口 0–1 的质量分 + 判定优劣。阈值是先用规则定的，后续可用
合成噪声标定（见 `calibrate_synthetic`）。
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sps

# 报告口径的 6 个子频段（下限 0.5 是因为预处理已去掉更低频的基线）
BANDS = [(0.5, 1.0), (1.0, 5.0), (5.0, 15.0), (15.0, 50.0), (50.0, 100.0), (100.0, 250.0)]


def _band_energy(f, pxx, lo, hi):
    m = (f >= lo) & (f < hi)
    return float(np.trapezoid(pxx[m], f[m])) if m.any() else 0.0


def _band_entropy(f, pxx, lo, hi, bins=8):
    """子带内的谱熵（归一化 0–1）：噪声填满子带时熵高，单一频率成分时熵低。"""
    m = (f >= lo) & (f < hi)
    if not m.any():
        return 0.0
    p = pxx[m]
    if p.sum() <= 0:
        return 0.0
    # 降采样成固定 bin 数再算熵
    edges = np.linspace(0, len(p), bins + 1).astype(int)
    pb = np.array([p[edges[i]:edges[i + 1]].sum() for i in range(bins)])
    if pb.sum() <= 0:
        return 0.0
    pb = pb / pb.sum()
    pb = pb[pb > 0]
    return float(-np.sum(pb * np.log(pb)) / np.log(bins))


def spectral_features(x: np.ndarray, fs: float) -> dict:
    """12 维谱特征 + 3 个 SQI 指标。"""
    f, pxx = sps.welch(x, fs=fs, nperseg=min(len(x), int(4 * fs)), detrend="linear")
    total = _band_energy(f, pxx, 0.5, fs / 2.0) + 1e-12
    out = {}
    ratios = []
    for i, (lo, hi) in enumerate(BANDS):
        e = _band_energy(f, pxx, lo, min(hi, fs / 2.0))
        ratios.append(e)
        out[f"band_ratio_{i}"] = e / total
    for i, (lo, hi) in enumerate(BANDS):
        out[f"band_entropy_{i}"] = _band_entropy(f, pxx, lo, min(hi, fs / 2.0))
    # 工程 SQI
    e_qrs = _band_energy(f, pxx, 5.0, 15.0)
    e_all = _band_energy(f, pxx, 0.5, 40.0) + 1e-12
    e_base = _band_energy(f, pxx, 0.5, 1.0)
    out["pSQI"] = e_qrs / e_all            # QRS 频带占比，典型 0.5–0.8
    out["basSQI"] = 1.0 - e_base / e_all   # 基线漂移占比越低越好
    out["kSQI"] = float(((x - x.mean()) ** 4).mean() / (x.var() ** 2 + 1e-12))  # 峰度
    return out


def _clip01(v, lo, hi):
    return float(np.clip((v - lo) / (hi - lo) if hi > lo else 0.0, 0.0, 1.0))


def score_window(x: np.ndarray, fs: float, r_peaks: np.ndarray | None = None) -> dict:
    """单窗质量分（0–1）。

    设计原则：每个子分都与"信号干不干净"单调对应，可解释、可单独核查。
    子分构成（等权平均）：
      s_p    带内 QRS 占比：5–15 Hz 占 0.5–40 Hz 的比例，落在 0.25–0.55 得分高
             ——太低说明窄带噪声盖住 QRS，太高说明信号尖刺化
      s_b    基线漂移占比：0–0.5 Hz 占比，越低越好（对应呼吸/运动引起的漂移）
      s_m    工频占比：48–52 Hz 占比，越低越好
      s_h    高频噪声占比：40–100 Hz 占比，越低越好（肌电/热噪声）
      s_k    峰度：在 0.5–40 Hz 带通信号上算，2–12 之间最好
             （太平=无 QRS 结构；太尖=脉冲噪声）
      s_c    形态一致性：各拍与中位模板的平均相关（需要 r_peaks），越高越好

    注意：本函数在"仅去直流"的信号上运行——带通之后基线/工频都被滤掉了，
    质量门控会形同虚设（这是实测踩过的坑）。
    """
    from .preprocess import bandpass

    xf = bandpass(np.asarray(x, dtype=np.float64), fs, 0.5, 40.0)

    f, pxx = sps.welch(x, fs=fs, nperseg=min(len(x), int(4 * fs)), detrend="linear")
    e_total = float(np.trapezoid(pxx, f)) + 1e-12
    e_band = _band_energy(f, pxx, 0.5, min(40.0, fs / 2.0)) + 1e-12
    p_qrs = _band_energy(f, pxx, 5.0, 15.0) / e_band
    r_base = _band_energy(f, pxx, 0.0, 0.5) / e_band
    r_mains = _band_energy(f, pxx, 48.0, 52.0) / e_total if fs > 110 else 0.0
    r_hf = _band_energy(f, pxx, 40.0, 100.0) / e_total if fs > 90 else 0.0

    k = float(((xf - xf.mean()) ** 4).mean() / (xf.var() ** 2 + 1e-12))
    if k < 12.0:
        s_k = _clip01(k, 2.0, 5.0) if k < 5 else 1.0
    else:
        s_k = 1.0 - _clip01(k, 12.0, 60.0)

    s_p = _clip01(p_qrs, 0.25, 0.50)
    s_b = 1.0 - _clip01(r_base, 0.05, 0.35)
    s_m = 1.0 - _clip01(r_mains, 0.02, 0.20)
    s_h = 1.0 - _clip01(r_hf, 0.05, 0.35)

    s_c = 0.5
    if r_peaks is not None and len(r_peaks) >= 3:
        w = int(round(0.12 * fs))
        segs = []
        for r in np.asarray(r_peaks):
            a, b = int(r) - w, int(r) + w
            if a < 0 or b > len(xf):
                continue
            seg = xf[a:b]
            seg = seg - seg.mean()
            n = np.linalg.norm(seg)
            if n > 0:
                segs.append(seg / n)
        if len(segs) >= 3:
            S = np.vstack(segs)
            med = np.median(S, axis=0)
            med /= np.linalg.norm(med) + 1e-12
            corr = S @ med
            s_c = _clip01(float(np.mean(corr)), 0.5, 0.9)

    score = float(np.clip(np.mean([s_p, s_b, s_m, s_h, s_k, s_c]), 0.0, 1.0))
    return {"quality": score, "kSQI": k, "pSQI": p_qrs, "basSQI": 1.0 - r_base,
            "mains_ratio": r_mains, "hf_ratio": r_hf, "template_corr": s_c,
            "s_p": s_p, "s_b": s_b, "s_m": s_m, "s_h": s_h, "s_k": s_k, "s_c": s_c,
            **{k2: v for k2, v in spectral_features(x, fs).items() if k2.startswith("band_")}}


def assess(x: np.ndarray, fs: float, win: float = 10.0, step: float | None = None,
           r_peaks: np.ndarray | None = None, thr: float = 0.6):
    """滑窗质量评估。

    返回 list[dict]，每项含 t_start / t_end / quality / 各子分数。
    """
    step = step or win
    w = int(round(win * fs))
    s = int(round(step * fs))
    out = []
    for a in range(0, max(0, len(x) - w + 1), s):
        seg = x[a:a + w]
        rp = None
        if r_peaks is not None:
            rp = np.asarray(r_peaks) - a
            rp = rp[(rp >= 0) & (rp < w)]
        row = score_window(seg, fs, rp)
        row["t_start"] = a / fs
        row["t_end"] = (a + w) / fs
        row["pass"] = row["quality"] >= thr
        out.append(row)
    return out


def synth_noise_bank(fs: float, seconds: float, seed: int = 0):
    """合成 4 类典型干扰，用于质量门控的标定与压力测试。

    对齐调研报告 3.4 节的"误报技术本源"：基线漂移、工频、肌电、运动伪差。
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * fs)
    t = np.arange(n) / fs

    baseline = 0.6 * np.sin(2 * np.pi * 0.25 * t + rng.uniform(0, 6.28))  # 呼吸性基线漂移
    mains = 0.15 * np.sin(2 * np.pi * 50 * t)                             # 50 Hz 工频
    emg = 0.10 * sps.butter(3, [20 / (fs / 2), 90 / (fs / 2)], btype="band")[0] * 0  # 占位
    # 肌电：20–90 Hz 带限白噪
    white = rng.normal(0, 1, n)
    b, a = sps.butter(3, [20 / (fs / 2), min(90, fs / 2 * 0.95) / (fs / 2)], btype="band")
    emg = 0.12 * sps.filtfilt(b, a, white)
    # 运动伪差：突发大幅低频扰动
    motion = np.zeros(n)
    for _ in range(max(1, int(seconds / 10))):
        i0 = rng.integers(0, max(1, n - int(2 * fs)))
        L = int(rng.uniform(0.3, 1.5) * fs)
        motion[i0:i0 + L] += rng.uniform(0.5, 2.0) * np.sin(
            2 * np.pi * rng.uniform(0.5, 3) * t[:L] + rng.uniform(0, 6.28))
    return {"baseline": baseline, "mains": mains, "emg": emg, "motion": motion}
