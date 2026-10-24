"""心律异常筛查：把 RR 序列/心拍序列变成"疑似事件"。

设计口径（对齐课程要求与调研报告）：
  · 输出一律是"疑似/关注/建议就医"，不是诊断；
  · 每个事件都带置信度与依据，供五层策略分级与二次确认；
  · 事件类型：停搏/长间歇、心动过速、心动过缓、房颤样心律（RR 不规则）、
    早搏（形态+提前量）。

房颤筛查特征集刻意选成"RR 域"的经典组合（不依赖 P 波识别）——
可穿戴单导联场景下 P 波检测不可靠，RR 不规则度是主判据。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .hrv import hrv_features, rr_intervals, sample_entropy, shannon_entropy_rr

# —— 阈值（默认值；个体定标层会覆盖部分阈值）——
PAUSE_S = 3.0          # 停搏/长间歇：RR 超过该值
PAUSE_S_STRICT = 4.0   # 严重档
TACHY_BPM = 100.0      # 心动过速
BRADY_BPM = 50.0       # 心动过缓
RATE_MIN_DUR = 30.0    # 心率异常持续时间门槛
AF_WIN = 30.0          # 房颤筛查滑窗长度
ECTOPY_CORR = 0.85     # 形态相关阈值（低于此值且提前 → 疑似早搏）
ECTOPY_PREMATURE = 0.85  # 提前量：RR / 局部主导 RR


@dataclass
class Event:
    kind: str
    start: float
    end: float
    detail: str
    confidence: float = 0.5
    severity: str = "info"
    evidence: dict = field(default_factory=dict)

    def as_dict(self):
        return {"kind": self.kind, "start": round(self.start, 1), "end": round(self.end, 1),
                "detail": self.detail, "confidence": round(self.confidence, 2),
                "severity": self.severity, "evidence": self.evidence}


def detect_pauses(r_peaks, fs, thr: float = PAUSE_S) -> list[Event]:
    rr = rr_intervals(r_peaks, fs)
    r = np.asarray(r_peaks) / fs
    out = []
    for i, v in enumerate(rr):
        if v >= thr:
            out.append(Event("pause", float(r[i]), float(r[i + 1]),
                             f"RR 间期 {v:.1f} s（≥{thr:.0f} s）",
                             confidence=min(1.0, v / (thr * 1.5)),
                             severity="refer" if v >= PAUSE_S_STRICT else "caution",
                             evidence={"rr_s": round(float(v), 2)}))
    return out


def detect_rate_events(r_peaks, fs, tachy: float = TACHY_BPM, brady: float = BRADY_BPM,
                       min_dur: float = RATE_MIN_DUR) -> list[Event]:
    """持续 ≥min_dur 秒的心动过速/过缓。"""
    rr = rr_intervals(r_peaks, fs)
    r = np.asarray(r_peaks) / fs
    if len(rr) < 3:
        return []
    bpm = 60.0 / rr
    out = []
    for want_high in (True, False):
        flag = bpm > tachy if want_high else bpm < brady
        kind = "tachy" if want_high else "brady"
        i = 0
        while i < len(flag):
            if flag[i]:
                j = i
                while j + 1 < len(flag) and flag[j + 1]:
                    j += 1
                dur = float(r[j] - r[i]) if j > i else 0.0
                if dur >= min_dur:
                    mean_bpm = float(np.mean(bpm[i:j + 1]))
                    out.append(Event(kind, float(r[i]), float(r[j + 1] if j + 1 < len(r) else r[j]),
                                     f"持续 {dur:.0f} s，平均 {mean_bpm:.0f} bpm",
                                     confidence=min(1.0, dur / (min_dur * 2)),
                                     severity="caution",
                                     evidence={"mean_bpm": round(mean_bpm, 1), "duration_s": round(dur, 1)}))
                i = j + 1
            else:
                i += 1
    return out


def _af_features(rr: np.ndarray) -> dict:
    """30 s 窗口的房颤筛查特征（RR 域）。"""
    if len(rr) < 6:
        return {}
    f = hrv_features(rr)
    drr = np.diff(rr)
    return {
        "hr": f["hr"],
        "cv_rr": f["cv_rr"],
        "rmssd_norm": f["rmssd"] / f["mean_rr"] if f["mean_rr"] > 0 else 0.0,
        "pnn50": f["pnn50"],
        "shannon_rr": shannon_entropy_rr(rr, bins=20),
        "sample_entropy": sample_entropy(rr),
        # Tateno 式：ΔRR 归一化后的香农熵（房颤时 ΔRR 分布弥散）
        "shannon_drr": shannon_entropy_rr(np.abs(drr), bins=15),
        "sd1_sd2": f["sd1_sd2"],
    }


AF_RULES = {
    # 规则组合（保守版）：需要同时满足"RR 明显不规则"与"熵升高"
    # 阈值为经典文献区间，最终以本地数据标定为准（见 docs/算法进展.md）
    "cv_rr": 0.10,
    "shannon_drr": 0.55,
    "pnn50": 0.30,
}


def af_score_rules(f: dict, rules: dict | None = None) -> float:
    """把特征映射成 0–1 的"房颤可能性"（规则版，可解释）。

    rules 可覆盖默认阈值（供五层策略 L5 个体定标使用）。
    """
    if not f:
        return 0.0
    r = {**AF_RULES, **(rules or {})}
    s_cv = float(np.clip((f["cv_rr"] - r["cv_rr"]) / 0.15, 0, 1))
    s_ent = float(np.clip((f["shannon_drr"] - r["shannon_drr"]) / 0.25, 0, 1))
    s_pnn = float(np.clip((f["pnn50"] - r["pnn50"]) / 0.35, 0, 1))
    # 心率过快/过慢都会干扰 RR 变异指标，做一点抑制
    hr = f["hr"]
    s_hr = 1.0 if 50 <= hr <= 130 else 0.5
    return float(np.mean([s_cv, s_ent, s_pnn]) * s_hr)


def screen_af_windows(r_peaks, fs, win: float = AF_WIN, step: float | None = None,
                      thr: float = 0.5, use_model: bool = True,
                      rules: dict | None = None) -> list[dict]:
    """滑窗房颤筛查。返回每个窗口的特征与得分。

    use_model=True 时优先用训练好的逻辑回归（RR 域特征，按记录分组交叉验证
    训练，见 scripts/train_af_model.py）；模型缺失时自动回退规则版。
    """
    step = step or win
    scorer = None
    if use_model:
        from . import afmodel

        if afmodel.load() is not None:
            scorer = afmodel.score
    r = np.asarray(r_peaks) / fs
    if len(r) < 10:
        return []
    t_end = float(r[-1])
    rows = []
    t = 0.0
    while t + win <= t_end:
        seg = r[(r >= t) & (r < t + win)]
        rr = np.diff(seg)
        f = _af_features(rr)
        if f:
            f["t_start"], f["t_end"] = t, t + win
            s_model = scorer(f) if scorer is not None else None
            f["af_score_rules"] = af_score_rules(f, rules)
            f["af_score"] = s_model if s_model is not None else f["af_score_rules"]
            f["scorer"] = "logreg" if s_model is not None else "rules"
            f["af_suspect"] = f["af_score"] >= thr
            rows.append(f)
        t += step
    return rows


def af_events(windows: list[dict], min_windows: int = 2) -> list[Event]:
    """把连续的疑似窗口合并成事件（天然带"持续确认"的语义）。"""
    out = []
    run = []
    for w in windows:
        if w["af_suspect"]:
            run.append(w)
        else:
            if len(run) >= min_windows:
                out.append(_merge(run))
            run = []
    if len(run) >= min_windows:
        out.append(_merge(run))
    return out


def _merge(run: list[dict]) -> Event:
    score = float(np.mean([w["af_score"] for w in run]))
    return Event("af_suspect", run[0]["t_start"], run[-1]["t_end"],
                 f"RR 不规则持续 {len(run) * 30} s（窗口评分均值 {score:.2f}）",
                 confidence=score,
                 severity="refer" if score >= 0.7 else "caution",
                 evidence={"n_windows": len(run), "mean_score": round(score, 2)})


def detect_ectopy(x: np.ndarray, r_peaks, fs: float, corr_thr: float = ECTOPY_CORR,
                  premature: float = ECTOPY_PREMATURE, win: float = 0.12) -> list[Event]:
    """早搏筛查：形态相关 + 提前量。

    对每个 R 波截取 ±win 窗口，算它与"本段中位模板"的相关系数；
    相关系数低 且 RR 明显短于局部主导 RR → 疑似早搏。
    """
    r = np.asarray(r_peaks)
    if len(r) < 5:
        return []
    w = int(round(win * fs))
    segs = []
    idx = []
    for i, ri in enumerate(r):
        a, b = ri - w, ri + w
        if a < 0 or b > len(x):
            continue
        s = x[a:b].astype(np.float64)
        s = s - s.mean()
        n = np.linalg.norm(s)
        if n <= 0:
            continue
        segs.append(s / n)
        idx.append(i)
    if len(segs) < 5:
        return []
    S = np.vstack(segs)
    med = np.median(S, axis=0)
    med /= np.linalg.norm(med) + 1e-12
    corr = S @ med

    rr = np.diff(r) / fs
    events = []
    for k, i in enumerate(idx):
        if i == 0 or i >= len(rr) + 1:
            continue
        rr_prev = rr[i - 1]
        lo = max(0, i - 10)
        local = np.median(rr[lo:min(len(rr), i + 10)])
        if corr[k] < corr_thr and local > 0 and rr_prev < premature * local:
            t = float(r[i] / fs)
            events.append(Event("ectopy", t, t + 0.2,
                                f"形态相关 {corr[k]:.2f}（<{corr_thr}），提前量 RR={rr_prev:.2f} s"
                                f"（局部主导 {local:.2f} s）",
                                confidence=float(np.clip((corr_thr - corr[k]) / corr_thr + 0.2, 0, 1)),
                                severity="info",
                                evidence={"corr": round(float(corr[k]), 2),
                                          "rr_prev": round(float(rr_prev), 2)}))
    return events


def screen_all(x: np.ndarray, r_peaks, fs: float, cfg: dict | None = None) -> list[Event]:
    """跑完所有筛查规则，返回事件列表（未经过五层策略过滤）。"""
    cfg = cfg or {}
    ev = []
    ev += detect_pauses(r_peaks, fs, cfg.get("pause_s", PAUSE_S))
    ev += detect_rate_events(r_peaks, fs, cfg.get("tachy", TACHY_BPM), cfg.get("brady", BRADY_BPM),
                             cfg.get("rate_min_dur", RATE_MIN_DUR))
    ev += detect_ectopy(x, r_peaks, fs, cfg.get("ectopy_corr", ECTOPY_CORR),
                        cfg.get("ectopy_premature", ECTOPY_PREMATURE))
    # 房颤筛查可按人群关闭（儿科未验证，见 subject.py）；
    # 关闭后仍保留早搏/心率/停搏筛查，只是不出"疑似房颤"结论
    if cfg.get("af_enabled", True):
        wins = screen_af_windows(r_peaks, fs, cfg.get("af_win", AF_WIN), thr=cfg.get("af_thr", 0.6),
                                 use_model=cfg.get("af_use_model", True),
                                 rules=cfg.get("af_rules"))
        ev += af_events(wins, cfg.get("af_min_windows", 2))
    ev.sort(key=lambda e: e.start)
    return ev
