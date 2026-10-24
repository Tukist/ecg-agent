"""分析管线：数据接入 → 预处理 → 检测 → 质量 → 筛查 → 五层策略。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..algorithms import quality as Q
from ..algorithms import rhythm as R
from ..algorithms import short_record as SR
from ..algorithms.hrv import hr_series, hrv_features, rr_intervals
from ..algorithms.preprocess import prepare_for_quality, preprocess
from ..algorithms.qrs import pan_tompkins, xqrs
from ..io import db_labels
from ..layers import strategy as S

# 短记录阈值：小于该时长时，额外输出逐拍摘要（不影响 rhythm 筛查流程）
SHORT_RECORD_THRESHOLD_S = 30.0


def _short_record_events(short_info: dict, fs: float, af_enabled: bool = True,
                         tachy: float = 110.0, brady: float = 50.0) -> list[R.Event]:
    """把短记录逐拍摘要转成与原策略层一致的事件对象，参与总体分级。

    生成的事件类型与 rhythm.py 一致：
      · pause  ：长间歇
      · tachy  ：平均心率过快
      · brady  ：平均心率过慢
      · ectopy ：频发早搏提示
      · af_suspect：节律不齐（af_enabled=False 时不出，用于儿科）
    """
    if not short_info or short_info.get("too_few_beats"):
        return []

    events: list[R.Event] = []
    n_beats = max(1, short_info.get("n_beats", 1))
    duration_s = short_info.get("duration_s", 0.0) or short_info.get("n_beats", 0)

    # 长间歇
    for t_start, t_end, rr_s in short_info.get("pause_events", []):
        events.append(R.Event(
            kind="pause",
            start=float(t_start),
            end=float(t_end),
            detail=f"短记录内 RR 间期 {rr_s:.1f} s（≥3 s）",
            confidence=min(1.0, rr_s / 6.0),
            severity="refer" if rr_s >= R.PAUSE_S_STRICT else "caution",
            evidence={"rr_s": round(float(rr_s), 2)}
        ))

    # 心动过速/过缓：用整段平均心率（短记录持续时间短，按整体报）
    # 阈值跟随人群：成人沿用短记录口径 110/50，儿童用年龄段阈值
    hr_mean = short_info.get("hr_mean", 0.0)
    if hr_mean >= tachy:
        events.append(R.Event(
            kind="tachy",
            start=0.0,
            end=float(duration_s),
            detail=f"短记录平均心率 {hr_mean:.0f} bpm（≥{tachy:.0f} bpm）",
            confidence=min(1.0, (hr_mean - tachy * 0.8) / 60.0),
            severity="refer" if hr_mean >= max(150.0, tachy) else "caution",
            evidence={"mean_bpm": round(float(hr_mean), 1)}
        ))
    elif 0 < hr_mean < brady:
        events.append(R.Event(
            kind="brady",
            start=0.0,
            end=float(duration_s),
            detail=f"短记录平均心率 {hr_mean:.0f} bpm（<{brady:.0f} bpm）",
            confidence=min(1.0, (brady - hr_mean) / 20.0),
            severity="caution",
            evidence={"mean_bpm": round(float(hr_mean), 1)}
        ))

    # 节律不齐：合并成房颤样心律提示（儿科未验证房颤筛查，af_enabled=False 时不出）
    if af_enabled and not short_info.get("regular", True):
        rr_cv = short_info.get("rr_cv", 0.0)
        events.append(R.Event(
            kind="af_suspect",
            start=0.0,
            end=float(duration_s),
            detail=f"短记录节律不齐（RR 变异系数 {rr_cv:.3f}）",
            confidence=min(1.0, rr_cv / 0.25),
            severity="caution",
            evidence={"cv_rr": round(float(rr_cv), 3)}
        ))

    return events


def _recompute_level(events: list[R.Event]) -> str:
    """沿用 strategy.apply_l2 的最大严重度规则。"""
    order = {"info": 0, "caution": 1, "refer": 2}
    lvl = "normal"
    for e in events:
        if order.get(e.severity, 0) > order.get(lvl, 0):
            lvl = e.severity
    return lvl


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
    short_record: dict | None = None   # 短记录逐拍摘要（仅 < 30 s 时填充）
    db_label: str | None = None        # 数据库参考标注（仅 wfdb 来源）
    subject: dict = field(default_factory=dict)  # 人群先验（年龄/性别/BMI → 阈值）


def load_source(source: str, minutes: float | None = None, start_s: float = 0.0):
    """支持四类输入：

      ppg-dalia:S1                —— PPG-DaLiA 缓存（250 Hz 长时程真实数据）
      mitdb:203                   —— MIT-BIH 记录（带标注，用于演示/评测）
      file:/path/x.npy            —— 现成 npy（配 --fs）
      wfdb:/path/to/rec|channel   —— 本地 WFDB 记录（.hea+.dat/.mat），channel 可为导联名或索引
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
    elif source.startswith("wfdb:"):
        import wfdb

        body = source[len("wfdb:"):]
        path, _, channel = body.rpartition("|")
        if not path:
            raise ValueError(f"WFDB 来源格式错误：{source}（应为 wfdb:/path/to/rec|channel）")
        r = wfdb.rdrecord(path, physical=True)
        if channel.isdigit():
            ch = int(channel)
        else:
            if channel not in r.sig_name:
                raise ValueError(f"记录 {path} 没有导联 '{channel}'，可用：{list(r.sig_name)}")
            ch = r.sig_name.index(channel)
        x = np.asarray(r.p_signal[:, ch], dtype=np.float64)
        fs = float(r.fs)
    else:
        raise ValueError(f"不认识的来源：{source}（支持 ppg-dalia:S1 / mitdb:100 / file:xx.npy / wfdb:/path|channel）")

    if start_s > 0:
        a = int(start_s * fs)
        x = x[a:] if minutes is None else x[a:a + int(minutes * 60 * fs)]
    elif minutes is not None:
        x = x[: int(minutes * 60 * fs)]
    return x, float(fs)


def analyze(source: str, minutes: float | None = None, start_s: float = 0.0,
            target_fs: float = 250.0, layers: str = "完整五层",
            verbose: bool = True, profile=None,
            screen_overrides: dict | None = None) -> AnalysisResult:
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

    # 短记录附加分析（不影响长记录流程）
    short_info = None
    duration_s = len(x) / fs
    if duration_s < SHORT_RECORD_THRESHOLD_S and len(r1) >= 3:
        short_info = step("short_record_summary", lambda: SR.analyze(x, fs, r1))
        if short_info:
            log.append(f"[工具:short_record] 记录较短（{duration_s:.1f}s），"
                       f"检出 {short_info['n_beats']} 拍，"
                       f"疑似早搏 {short_info['suspected_ectopy']} 次")

    dec = step("apply_layers", lambda: S.run(x, fs, r1, r2, cfg=S.preset(layers),
                                             quality_windows=qw, profile=profile,
                                             screen_overrides=screen_overrides))
    log += dec.log

    # 短记录：把逐拍摘要转成事件并入总体分级（沿用原 strategy 的事件/分级形式）
    if short_info:
        short_events = _short_record_events(
            short_info, fs,
            af_enabled=dec.population.get("af_enabled", True),
            # 短记录口径成人沿用 110；儿童用年龄段过速阈值
            tachy=max(110.0, float(dec.population.get("tachy", 110.0))),
            brady=float(dec.population.get("brady", 50.0)),
        )
        if short_events:
            dec.events.extend(short_events)
            dec.level = _recompute_level(dec.events)
            log.append(f"[短记录事件] 补充 {len(short_events)} 个逐拍摘要事件，"
                       f"总体分级更新为 {dec.level}")

    # 数据库参考标注（仅展示，不参与判断）
    db_label = db_labels.lookup(source)
    if db_label:
        log.append(f"[工具:db_label] 数据库参考标注：{db_label}")

    return AnalysisResult(source=source, fs=fs, duration_s=duration_s, x=x, r_peaks=r1,
                          r_peaks_second=r2, quality_windows=qw, hrv=feats, hr=hrs,
                          decision=dec, log=log, timings=timings,
                          short_record=short_info, db_label=db_label,
                          subject=dec.population)
