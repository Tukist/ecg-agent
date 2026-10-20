"""分析管线：数据接入 → 预处理 → 检测 → 质量 → 筛查 → 五层策略。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..algorithms import quality as Q
from ..algorithms.hrv import hr_series, hrv_features, rr_intervals
from ..algorithms.preprocess import prepare_for_quality, preprocess
from ..algorithms.qrs import pan_tompkins, xqrs
from ..layers import strategy as S


@dataclass
class AnalysisResult:
    source: str
    fs: float
    duration_s: float
    x: np.ndarray                      # 预处理后的信号（250 Hz）
    r_peaks: np.ndarray
    r_peaks_second: np.ndarray
    quality_windows: list
    hrv: dict
    hr: tuple                          # (times, hr)
    decision: S.Decision
    log: list = field(default_factory=list)   # 全流程决策日志
    timings: dict = field(default_factory=dict)


def load_source(source: str, minutes: float | None = None, start_s: float = 0.0):
    """支持三类输入：

      ppg-dalia:S1       —— PPG-DaLiA 缓存（250 Hz 长时程真实数据）
      mitdb:203          —— MIT-BIH 记录（带标注，用于演示/评测）
      file:/path/x.npy   —— 现成 npy（配 --fs）
    """
    if source.startswith("ppg-dalia:"):
        from ..io import respiban

        sid = source.split(":", 1)[1].upper()
        x = np.load(respiban.cache_path(sid))
        fs = 250.0
        x = x.astype(np.float64)
    elif source.startswith("mitdb:"):
        from ..io import mitdb

        rec = source.split(":", 1)[1]
        x, fs, _ = mitdb.read(rec, channel=0)
        x = np.asarray(x, dtype=np.float64)
    elif source.startswith("file:"):
        path = source.split(":", 1)[1]
        x = np.load(path).astype(np.float64)
        fs = 250.0
    else:
        raise ValueError(f"不认识的来源：{source}（支持 ppg-dalia:S1 / mitdb:100 / file:xx.npy）")

    if start_s > 0:
        a = int(start_s * fs)
        x = x[a:] if minutes is None else x[a:a + int(minutes * 60 * fs)]
    elif minutes is not None:
        x = x[: int(minutes * 60 * fs)]
    return x, float(fs)


def analyze(source: str, minutes: float | None = None, start_s: float = 0.0,
            target_fs: float = 250.0, layers: str = "完整五层",
            verbose: bool = True) -> AnalysisResult:
    log: list[str] = []
    timings: dict[str, float] = {}

    def step(name: str, fn):
        t0 = time.time()
        out = fn()
        dt = time.time() - t0
        timings[name] = dt
        msg = f"[工具:{name}] {dt:.2f}s"
        log.append(msg)
        if verbose:
            print(msg)
        return out

    x_raw, fs_in = load_source(source, minutes=minutes, start_s=start_s)
    log.append(f"[工具:load] 来源 {source}，原始 {len(x_raw)/fs_in/60:.1f} 分钟 @ {fs_in:.0f} Hz")

    x, fs = step("preprocess", lambda: preprocess(x_raw, fs_in, target_fs=target_fs))
    # 质量评估在"仅去直流"的原始信号上做（带通会把基线/工频噪声滤掉，
    # 在滤波后评估等于没评估——见 preprocess.prepare_for_quality 的说明）
    x_q, fs_q = prepare_for_quality(x_raw, fs_in, target_fs=target_fs)
    r1 = step("detect_qrs(pan_tompkins)", lambda: pan_tompkins(x, fs))
    r2 = step("detect_qrs(xqrs, 第二意见)", lambda: xqrs(x, fs))
    log.append(f"[工具:detect_qrs] 主检测器 {len(r1)} 拍，第二检测器 {len(r2)} 拍")

    qw = step("assess_quality", lambda: Q.assess(x_q, fs_q, r_peaks=r1))
    pass_rate = float(np.mean([w["pass"] for w in qw])) if qw else 0.0
    log.append(f"[工具:assess_quality] {len(qw)} 个 10 s 窗，合格率 {pass_rate*100:.1f}%")

    rr = rr_intervals(r1, fs)
    feats = step("hrv_features", lambda: hrv_features(rr))
    log.append(f"[工具:hrv_features] 平均心率 {feats['hr']:.0f} bpm，"
               f"SDNN {feats['sdnn']*1000:.0f} ms，RMSSD {feats['rmssd']*1000:.0f} ms")

    hrs = step("hr_series", lambda: hr_series(r1, fs, win=10.0))

    dec = step("apply_layers", lambda: S.run(x, fs, r1, r2, cfg=S.preset(layers), quality_windows=qw))
    log += dec.log

    return AnalysisResult(source=source, fs=fs, duration_s=len(x) / fs, x=x, r_peaks=r1,
                          r_peaks_second=r2, quality_windows=qw, hrv=feats, hr=hrs,
                          decision=dec, log=log, timings=timings)
