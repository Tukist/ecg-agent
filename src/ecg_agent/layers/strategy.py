"""五层防误报策略（课程重点：多重策略保证精确性、少误报）。

五层与依据（对齐调研报告，避免"笼统说五层都有文献支撑"）：

  L1 质量门控   —— 先判信号质量再判病；来源：信号质量评估文献（SQI 谱特征）
  L2 分级预警   —— 不做"报警/不报警"的二值输出；来源：2026 报告规范共识的
                   三层分流（危急值/请会诊/常规签发），五级为本项目扩展
  L3 事件二次确认 —— 两个独立检测器/连续多窗一致才升级；来源：Fitbit 45 万人队列
                   （加了"再次触发确认"后 PPV 32.2% → 98.2%）
  L4 人审兜底   —— 输出限"提示 + 建议就医复核"；来源：监管口径（QDA/QYE）
  L5 个体定标   —— 全局固定阈值→个人基线；来源：浙大博士论文（四种心拍形态
                   峰度分布重叠，全局阈值区分不开）

重要口径：本模块只做"疑似事件提示"，不做诊断；输出文案由 `agent/report.py` 统一
落实为提示级措辞。每层可单独开关，用于消融实验（results/ablation_*.csv）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..algorithms import quality as Q
from ..algorithms import rhythm as R
from ..algorithms.hrv import rr_intervals


@dataclass
class LayerConfig:
    l1_quality_gate: bool = True
    l2_grading: bool = True
    l3_confirmation: bool = True
    l4_human_review: bool = True
    l5_individual: bool = True
    quality_thr: float = 0.6
    confirm_min_windows: int = 2


@dataclass
class Decision:
    level: str = "normal"                 # normal / attention / refer
    events: list = field(default_factory=list)          # 通过全部层的疑似事件
    suppressed: list = field(default_factory=list)      # 被压掉的事件（含原因）
    log: list = field(default_factory=list)             # 决策日志（逐层）
    coverage: float = 0.0                 # 有效监测时长占比（过质量门控的窗口比例）
    calibrated: dict = field(default_factory=dict)      # L5 个体化后的阈值

    @property
    def n_alerts(self) -> int:
        """落到用户侧的可行动报警数（attention 及以上）。"""
        return sum(1 for e in self.events if e.severity in ("caution", "refer"))


# ---------------------------------------------------------------- L5 个体定标
def calibrate(r_peaks, fs: float, baseline_min: float = 5.0) -> dict:
    """用本人数据的前若干分钟建立基线，返回个体化阈值。

    依据：报告里"形态分组/个体定标"一节——全局固定阈值在个体间区分不开，
    阈值必须用自己的数据标定（这也是文献阈值不能直接抄的原因）。
    """
    r = np.asarray(r_peaks) / fs
    if len(r) < 10:
        return {}
    base = r[r <= r[0] + baseline_min * 60]
    rr = np.diff(base)
    rr = rr[(rr > 0.2) & (rr < 3.0)]
    if len(rr) < 5:
        rr = np.diff(r)
        rr = rr[(rr > 0.2) & (rr < 3.0)]
    if len(rr) < 5:
        return {}
    med = float(np.median(rr))
    mad = float(np.median(np.abs(rr - med))) or 0.02
    cv = float(np.std(rr) / np.mean(rr))
    return {
        "median_rr": med,
        "hr_baseline": 60.0 / med,
        "cv_baseline": cv,
        # 停搏阈值：不低于 3.0 s，且随个人基础 RR 抬高（生理上长间歇是相对的）
        "pause_s": max(3.0, 3.5 * med),
        # 心动过速/过缓：以个人基线 ±20 bpm 为界（比全局固定值更贴个体）
        "tachy": max(100.0, 60.0 / med + 25.0),
        "brady": min(50.0, 60.0 / med - 20.0),
        # 房颤样不规则：以个人基线 CV 的 2.5 倍为下限，但不低于 0.10
        "af_cv": max(0.10, 2.5 * cv),
    }


# ---------------------------------------------------------------- L1 质量门控
def _window_quality(quality_windows: list[dict], t0: float, t1: float) -> float:
    """事件时段内的平均质量分（无覆盖窗口时返回 1.0，即不因此压掉）。"""
    qs = [w["quality"] for w in quality_windows if w["t_end"] > t0 and w["t_start"] < t1]
    return float(np.mean(qs)) if qs else 1.0


def apply_l1(events: list, quality_windows: list[dict], thr: float, log: list):
    kept, dropped = [], []
    for e in events:
        q = _window_quality(quality_windows, e.start, e.end)
        e.evidence["quality"] = round(q, 2)
        if q < thr:
            dropped.append((e, f"质量分 {q:.2f} < {thr}（L1 质量门控压掉）"))
        else:
            kept.append(e)
    log.append(f"[L1 质量门控] {len(events)} 个候选事件，压掉 {len(dropped)} 个（阈 {thr}）")
    return kept, dropped


# ---------------------------------------------------------------- L3 二次确认
def apply_l3(events: list, x: np.ndarray, fs: float, r_peaks_second, log: list,
             min_windows: int = 2):
    """两个独立路径确认：

      · 检测器级：用第二检测器（XQRS）的 R 波序列重跑同类筛查，事件是否复现；
      · 时间级：AF 类事件必须跨 ≥min_windows 个连续窗口（筛查时已保证），
        停搏类要求第二检测器也看到同样的长 RR。
    """
    if r_peaks_second is None or len(np.asarray(r_peaks_second)) < 5:
        log.append("[L3 二次确认] 无第二检测器结果，跳过（事件保留但标记未复核）")
        return events, []
    rr2 = rr_intervals(r_peaks_second, fs)
    r2 = np.asarray(r_peaks_second) / fs
    kept, downgraded = [], []
    for e in events:
        ok = False
        if e.kind == "pause":
            seg = rr2[(r2[:-1] >= e.start - 1.0) & (r2[:-1] <= e.end + 1.0)]
            ok = bool(len(seg) and seg.max() >= 0.7 * e.evidence.get("rr_s", R.PAUSE_S))
        elif e.kind in ("tachy", "brady", "af_suspect", "ectopy"):
            # 在事件时段内，第二检测器序列应当支持同样的节律异常。
            # 瞬时事件（早搏约 0.2 s）够不到 4 拍，用 ±3 s 扩展窗再取拍。
            pad = 3.0 if (e.end - e.start) < 5.0 else 0.0
            mask = (r2 >= e.start - pad) & (r2 <= e.end + pad)
            seg = r2[mask]
            if len(seg) >= 4:
                rr_seg = np.diff(seg)
                if e.kind == "tachy":
                    ok = bool(np.mean(60.0 / rr_seg) > 90)
                elif e.kind == "brady":
                    ok = bool(np.mean(60.0 / rr_seg) < 60)
                elif e.kind == "af_suspect":
                    ok = bool(np.std(rr_seg) / np.mean(rr_seg) > 0.06)
                else:  # ectopy
                    ok = bool(len(rr_seg) and rr_seg.min() < 0.8 * np.median(rr_seg))
        else:
            ok = True
        e.evidence["confirmed"] = bool(ok)
        if ok:
            kept.append(e)
        elif e.kind == "ectopy":
            # 早搏本来就是 info 级（从不上报为报警），二次确认不通过时
            # 降级保留——报告里仍可统计"疑似早搏次数"，措辞为提示级
            e.severity = "info"
            e.detail += "（第二检测器未复现，仅计数提示）"
            kept.append(e)
        else:
            downgraded.append((e, "第二检测器未能复现（L3 二次确认未通过）"))
    log.append(f"[L3 二次确认] {len(events)} 个事件，{len(kept)} 个复现，{len(downgraded)} 个降级")
    return kept, downgraded


# ---------------------------------------------------------------- L2 分级
SEVERITY_ORDER = {"info": 0, "caution": 1, "refer": 2}


def apply_l2(events: list, log: list):
    """把事件映射成三级分流（对齐 2026 报告规范共识的三层分流框架）：
        refer   ：请会诊/尽快就医级（长停搏、高置信房颤样事件、持续极端心率）
        caution ：需要关注（一般持续心率异常、普通停搏）
        info    ：仅记录（单发早搏等）
    """
    for e in events:
        s = e.severity
        if e.kind == "pause":
            s = "refer" if e.evidence.get("rr_s", 0) >= R.PAUSE_S_STRICT else "caution"
        elif e.kind == "af_suspect":
            s = "refer" if e.confidence >= 0.7 else "caution"
        elif e.kind in ("tachy", "brady"):
            s = "caution"
            bpm = e.evidence.get("mean_bpm", 0)
            if e.kind == "tachy" and bpm >= 150:
                s = "refer"
            if e.kind == "brady" and 0 < bpm <= 40:
                s = "refer"
        elif e.kind == "ectopy":
            s = "info"
        e.severity = s
    lvl = "normal"
    for e in events:
        if SEVERITY_ORDER[e.severity] > SEVERITY_ORDER.get(lvl, 0):
            lvl = e.severity
    log.append(f"[L2 分级] 事件分级："
               f"{ {k: sum(1 for e in events if e.severity == k) for k in ('info','caution','refer')} }")
    return lvl


# ---------------------------------------------------------------- L4 人审兜底
def apply_l4(decision: Decision, log: list):
    """给结论配上"提示级"措辞与人审要求（监管口径：不提供诊断/预警）。"""
    decision.log.append("[L4 人审兜底] 输出限'提示 + 建议就医复核'口径；"
                        "refer 级事件必须由医生复核后才能形成结论")
    return decision


# ---------------------------------------------------------------- 总装
def run(x: np.ndarray, fs: float, r_peaks, r_peaks_second=None,
        cfg: LayerConfig | None = None, quality_windows: list[dict] | None = None) -> Decision:
    """按五层顺序执行，返回 Decision（含决策日志，可打印给演示用）。"""
    cfg = cfg or LayerConfig()
    log: list[str] = []
    decision = Decision()

    # —— L5 先跑：个体定标改的是筛查阈值，必须在筛查之前 ——
    calib = {}
    if cfg.l5_individual:
        calib = calibrate(r_peaks, fs)
    decision.calibrated = calib
    log.append(f"[L5 个体定标] {'启用：' + str({k: round(v, 2) for k, v in calib.items()}) if calib else '未启用或无有效基线'}")

    scr_cfg = {}
    if calib:
        scr_cfg = {"pause_s": calib["pause_s"], "tachy": calib["tachy"], "brady": calib["brady"]}
        # L5 个体定标同样作用于房颤判定：
        #   ① 规则版——把 cv_rr 阈值换成个人基线推导值；
        #   ② 模型版——个人基线 RR 变异越大，判定"房颤样"的证据门槛越高
        #      （af_thr = 0.6 + max(0, cv基线 - 0.08)，上限 0.9）
        cv = calib.get("cv_baseline", 0.0)
        scr_cfg["af_rules"] = {"cv_rr": calib.get("af_cv", 0.10)}
        scr_cfg["af_thr"] = float(min(0.9, 0.6 + max(0.0, cv - 0.08)))
    events = R.screen_all(x, r_peaks, fs, scr_cfg)
    log.append(f"[筛查] 原始候选事件 {len(events)} 个")

    # —— L1 质量门控 ——
    suppressed = []
    if cfg.l1_quality_gate:
        if quality_windows is None:
            quality_windows = Q.assess(x, fs, r_peaks=r_peaks)
        events, dropped = apply_l1(events, quality_windows, cfg.quality_thr, log)
        suppressed += dropped
        decision.coverage = float(np.mean([w["pass"] for w in quality_windows])) if quality_windows else 0.0
    else:
        log.append("[L1 质量门控] 关闭")

    # —— L3 二次确认 ——
    if cfg.l3_confirmation:
        events, down = apply_l3(events, x, fs, r_peaks_second, log, cfg.confirm_min_windows)
        suppressed += down
    else:
        log.append("[L3 二次确认] 关闭")

    # —— L2 分级 ——
    if cfg.l2_grading:
        decision.level = apply_l2(events, log)
    else:
        log.append("[L2 分级] 关闭")
        decision.level = "attention" if events else "normal"

    decision.events = events
    decision.suppressed = suppressed

    # —— L4 人审兜底 ——
    if cfg.l4_human_review:
        apply_l4(decision, log)
    else:
        log.append("[L4 人审兜底] 关闭")

    decision.log = log
    return decision


PRESETS = {
    "裸阈值": dict(l1_quality_gate=False, l3_confirmation=False, l5_individual=False),
    "+质量门控": dict(l1_quality_gate=True, l3_confirmation=False, l5_individual=False),
    "+二次确认": dict(l1_quality_gate=True, l3_confirmation=True, l5_individual=False),
    "+个体定标": dict(l1_quality_gate=True, l3_confirmation=True, l5_individual=True),
    "完整五层": dict(l1_quality_gate=True, l2_grading=True, l3_confirmation=True,
                     l4_human_review=True, l5_individual=True),
}


def preset(name: str) -> LayerConfig:
    return LayerConfig(**PRESETS[name])
