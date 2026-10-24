"""报告渲染：图（PNG）+ 文字报告（Markdown）。

口径（监管红线）：只出"提示/建议就医"，不出现"诊断/预警"字样；
所有结论附"非诊断用途、需医生复核"说明。
"""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np

KIND_CN = {
    "pause": "长间歇/疑似停搏",
    "tachy": "心动过速",
    "brady": "心动过缓",
    "af_suspect": "疑似房颤样心律",
    "ectopy": "疑似早搏",
}
LEVEL_CN = {"normal": "未见需即刻关注的异常", "caution": "关注", "refer": "建议尽快就医复核"}
LEVEL_ORDER = {"info": 0, "caution": 1, "refer": 2}


def _setup_font():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def render_figure(res, path: str):
    """根据记录类型自动选择长记录或短记录图。"""
    if res.short_record and res.duration_s < 30:
        return render_short_figure(res, path)
    return render_long_figure(res, path)


def render_short_figure(res, path: str):
    """短记录（< 30 s）专用图：波形+逐拍标注、RR 间期、统计摘要。"""
    plt = _setup_font()
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1.2, 1])

    ax_wave = fig.add_subplot(gs[0, :])
    ax_rr = fig.add_subplot(gs[1, :])
    ax_text = fig.add_subplot(gs[2, :])

    fs = res.fs
    x = res.x
    t = np.arange(len(x)) / fs
    s = res.short_record
    beats = s.get("beat_details", [])

    # ① 完整波形 + R 峰 + 疑似早搏标注
    ax_wave.plot(t, x, lw=0.8, color="#374151", label="ECG")
    normal_r = [b["time_s"] for b in beats if b["flag"] == "normal"]
    normal_y = [x[int(b["time_s"] * fs)] for b in beats if b["flag"] == "normal"]
    ectopy_r = [b["time_s"] for b in beats if b["flag"] == "suspected_ectopy"]
    ectopy_y = [x[int(b["time_s"] * fs)] for b in beats if b["flag"] == "suspected_ectopy"]
    ax_wave.plot(normal_r, normal_y, "v", ms=5, color="#059669", label="正常心拍")
    if ectopy_r:
        ax_wave.plot(ectopy_r, ectopy_y, "v", ms=6, color="#dc2626", label="疑似早搏")
    ax_wave.set_xlabel("时间 (秒)")
    ax_wave.set_ylabel("幅值 (a.u.)")
    ax_wave.set_title("① 短记录全波形与逐拍标注（红心=疑似早搏）")
    ax_wave.legend(loc="upper right", fontsize=9)

    # ② RR 间期 / 心率 逐拍
    if beats:
        idx = [b["idx"] for b in beats[1:]]
        rr_prev = [b["rr_prev_s"] * 1000 for b in beats[1:] if b["rr_prev_s"] is not None]
        hr = [60.0 / b["rr_prev_s"] for b in beats[1:] if b["rr_prev_s"] is not None]
        ax_rr.bar(idx, rr_prev, color="#3b82f6", alpha=0.7, label="RR 间期 (ms)")
        ax_rr.axhline(1000, ls="--", lw=0.8, color="#6b7280")
        ax_rr.set_xlabel("心拍序号")
        ax_rr.set_ylabel("RR 间期 (ms)", color="#3b82f6")
        ax_rr.tick_params(axis="y", labelcolor="#3b82f6")
        ax_rr.set_title("② 逐拍 RR 间期与心率")

        ax_hr = ax_rr.twinx()
        ax_hr.plot(idx, hr, "o-", ms=3, lw=1.0, color="#dc2626", label="心率 (bpm)")
        ax_hr.set_ylabel("心率 (bpm)", color="#dc2626")
        ax_hr.tick_params(axis="y", labelcolor="#dc2626")

    # ③ 文字摘要表格
    ax_text.axis("off")
    if s.get("too_few_beats"):
        summary = "心拍过少，无法做逐拍摘要。"
    else:
        summary = (
            f"总心拍：{s['n_beats']} 拍\n"
            f"心率：{s['hr_min']:.0f}–{s['hr_max']:.0f} bpm，平均 {s['hr_mean']:.0f} bpm\n"
            f"节律：{'整齐' if s['regular'] else '不齐'}（RR 变异系数 {s['rr_cv']:.3f}）\n"
            f"疑似早搏：{s['suspected_ectopy']} 次；宽 QRS：{s['suspected_wide_qrs']} 次\n"
            f"长间歇（≥3 s）：{len(s['pause_events'])} 次"
        )
    ax_text.text(0.05, 0.95, summary, transform=ax_text.transAxes, fontsize=11,
                 verticalalignment="top",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="#f3f4f6", edgecolor="#d1d5db"))

    db_label = res.db_label or "无"
    ax_text.text(0.05, 0.45, f"数据库参考标注：{db_label}", transform=ax_text.transAxes,
                 fontsize=10, verticalalignment="top", wrap=True,
                 color="#4b5563")

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def render_long_figure(res, path: str):
    plt = _setup_font()
    fig, axes = plt.subplots(4, 1, figsize=(12, 13))

    # ① 心率曲线
    ax = axes[0]
    t, hr = res.hr
    if len(t):
        ax.plot(t / 60, hr, lw=1.0, color="#2563eb")
        ax.axhline(100, ls="--", lw=0.8, color="#f59e0b")
        ax.axhline(50, ls="--", lw=0.8, color="#f59e0b")
    for e in res.decision.events:
        if e.severity in ("caution", "refer"):
            ax.axvspan(e.start / 60, max(e.end, e.start + 1) / 60,
                       color="#dc2626" if e.severity == "refer" else "#f59e0b", alpha=0.18)
    ax.set_ylabel("心率 (bpm)")
    ax.set_xlabel("时间 (分钟)")
    ax.set_title("① 心率曲线与疑似事件（色带=上报事件）")

    # ② 信号质量
    ax = axes[1]
    if res.quality_windows:
        qt = [w["t_start"] / 60 for w in res.quality_windows]
        qs = [w["quality"] for w in res.quality_windows]
        ax.plot(qt, qs, lw=1.0, color="#059669")
        ax.axhline(0.6, ls="--", lw=0.8, color="#dc2626")
        ax.set_ylim(0, 1.05)
    ax.set_ylabel("质量分")
    ax.set_xlabel("时间 (分钟)")
    ax.set_title("② 信号质量（SQI 0–1，虚线=门控阈值 0.6）")

    # ③ Poincaré
    ax = axes[2]
    from ..algorithms.hrv import rr_intervals

    rr = rr_intervals(res.r_peaks, res.fs)
    if len(rr) > 2:
        ax.scatter(rr[:-1] * 1000, rr[1:] * 1000, s=4, alpha=0.4, color="#7c3aed")
        ax.set_xlabel("RR$_n$ (ms)")
        ax.set_ylabel("RR$_{n+1}$ (ms)")
    ax.set_title("③ RR 间期 Poincaré 图（散得越开=节律越不规则）")

    # ④ 波形示例（取 30 秒，优先含事件）
    ax = axes[3]
    fs = res.fs
    if res.decision.events:
        e = max(res.decision.events, key=lambda e: LEVEL_ORDER[e.severity])
        t0 = max(0.0, e.start - 8)
    else:
        t0 = 0.0
    a, b = int(t0 * fs), int((t0 + 30) * fs)
    seg = res.x[a:b]
    tt = np.arange(len(seg)) / fs + t0
    ax.plot(tt, seg, lw=0.7, color="#111827")
    rp = res.r_peaks[(res.r_peaks >= a) & (res.r_peaks < b)]
    ax.plot(rp / fs, res.x[rp], "v", ms=4, color="#dc2626")
    ax.set_xlabel("时间 (秒)")
    ax.set_ylabel("幅值 (a.u.)")
    ax.set_title("④ 波形示例（红点=R 波检测；a.u.=设备原始计数，未校准 mV）")

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _event_rows(events, collapse_over: int = 5):
    """把事件整理成展示行；同类型事件超过 collapse_over 个时折叠为一行。"""
    from collections import defaultdict

    groups = defaultdict(list)
    for e in events:
        groups[(e.kind, e.severity)].append(e)
    rows = []
    for (kind, sev), es in groups.items():
        es = sorted(es, key=lambda x: x.start)
        if len(es) > collapse_over:
            rows.append({"kind": kind, "severity": sev, "start": es[0].start,
                         "end": es[-1].end, "count": len(es),
                         "detail": f"{es[0].start/60:.1f}–{es[-1].end/60:.1f} min 区间内共 {len(es)} 次",
                         "confidence": max(e.confidence for e in es)})
        else:
            for e in es:
                rows.append({"kind": kind, "severity": e.severity, "start": e.start,
                             "end": e.end, "count": 1, "detail": e.detail,
                             "confidence": e.confidence})
    rows.sort(key=lambda r: r["start"])
    return rows


def render_markdown(res, path: str, fig_path: str | None = None, llm_text: str | None = None):
    d = res.decision
    lines = []
    lines.append(f"# 心电分析报告（提示级）\n")
    lines.append(f"- 数据来源：`{res.source}`")
    p = getattr(res, "subject", None) or d.population or {}
    if p:
        lines.append(f"- 受检者：{p.get('summary', '未提供')}")
        lines.append(f"- 人群参考阈值：{p.get('threshold_text', '-')}")
    lines.append(f"- 记录时长：{res.duration_s/60:.1f} 分钟（分析采样率 {res.fs:.0f} Hz）")
    lines.append(f"- 有效监测时长占比：{d.coverage*100:.1f}%（通过质量门控的窗口比例）")
    lines.append(f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"- 生成工具：ecg_agent 心电分析智能体（规则策略版，非诊断用途）\n")

    lines.append("## 一、结论（提示级）\n")
    n_alert = d.n_alerts
    if d.level == "normal" and n_alert == 0:
        lines.append("本次记录**未见需要即刻关注的疑似异常心律**。")
    elif d.level == "refer":
        lines.append(f"本次记录检出 **{n_alert}** 处**建议尽快就医复核**的疑似心律变化（见下表）。")
    else:
        lines.append(f"本次记录检出 **{n_alert}** 处**需要关注**的疑似心律变化（见下表）。")
    lines.append("\n> 本结果由算法自动生成，仅作健康提示，**不构成诊断**；"
                 "如有不适或需要结论，请携带完整记录由医生复核。\n")

    CN = "一二三四五六七八九十"
    sec = 3

    lines.append("## 二、心率概览\n")
    h = res.hrv
    lines.append(f"- 平均心率：{h['hr']:.0f} bpm")
    lines.append(f"- SDNN：{h['sdnn']*1000:.0f} ms；RMSSD：{h['rmssd']*1000:.0f} ms；pNN50：{h['pnn50']*100:.0f}%")
    lines.append(f"- RR 变异系数：{h['cv_rr']:.3f}；样本熵：{h['sample_entropy']:.2f}\n")

    if res.short_record:
        lines.append(f"## {CN[sec-1]}、短记录逐拍摘要\n")
        sec += 1
        s = res.short_record
        if s.get("too_few_beats"):
            lines.append("心拍过少，无法做逐拍摘要。\n")
        else:
            lines.append(f"- 总心拍数：{s['n_beats']} 拍")
            lines.append(f"- 心率范围：{s['hr_min']:.0f}–{s['hr_max']:.0f} bpm，平均 {s['hr_mean']:.0f} bpm")
            lines.append(f"- 节律整齐度：{'整齐' if s['regular'] else '不齐'}（RR 变异系数 {s['rr_cv']:.3f}）")
            lines.append(f"- 疑似早搏：{s['suspected_ectopy']} 次")
            lines.append(f"- 宽 QRS（>120 ms）：{s['suspected_wide_qrs']} 次")
            if s['pause_events']:
                lines.append(f"- 长间歇（≥3 s）：{len(s['pause_events'])} 次")
            else:
                lines.append("- 长间歇（≥3 s）：无")
            lines.append("")
            if s['beat_details']:
                lines.append("| 序号 | 时间(s) | 前 RR(s) | 后 RR(s) | QRS宽(s) | 形态相关 | 标记 |")
                lines.append("|------|---------|----------|----------|----------|----------|------|")
                for b in s['beat_details']:
                    flag_cn = {"normal": "正常", "suspected_ectopy": "疑似早搏"}.get(b['flag'], b['flag'])
                    lines.append(f"| {b['idx']} | {b['time_s']} | {b['rr_prev_s'] or '-'} | "
                                 f"{b['rr_next_s'] or '-'} | {b['width_s']} | {b['morph_corr']} | {flag_cn} |")
                lines.append("")

    if res.db_label:
        lines.append(f"## {CN[sec-1]}、数据库参考标注（仅展示，不参与判断）\n")
        sec += 1
        lines.append(f"{res.db_label}\n")

    lines.append(f"## {CN[sec-1]}、疑似事件（经五层策略过滤后上报）\n")
    sec += 1
    if d.events:
        lines.append("| # | 类型 | 时间 | 分级 | 置信度 | 依据 |")
        lines.append("|---|------|------|------|--------|------|")
        for i, r in enumerate(_event_rows(d.events), 1):
            lines.append(f"| {i} | {KIND_CN.get(r['kind'], r['kind'])} | {r['start']/60:.1f} min | "
                         f"{LEVEL_CN.get(r['severity'], r['severity'])} | "
                         f"{r['confidence']:.2f} | {r['detail']} |")
    else:
        lines.append("（无）")
    lines.append("")

    if d.suppressed:
        lines.append(f"## {CN[sec-1]}、被防误报策略压掉的候选（附录）\n")
        sec += 1
        lines.append("这些候选曾触发筛查阈值，但被后续层判定为不可报告（这正是'少误报'的来源）：\n")
        lines.append("| # | 类型 | 起始 | 压掉原因 |")
        lines.append("|---|------|------|----------|")
        for i, (e, why) in enumerate(d.suppressed[:30], 1):
            lines.append(f"| {i} | {KIND_CN.get(e.kind, e.kind)} | {e.start/60:.1f} min | {why} |")
        if len(d.suppressed) > 30:
            lines.append(f"| … | | | 其余 {len(d.suppressed)-30} 条略 |")
        lines.append("")

    if fig_path:
        lines.append(f"## {CN[sec-1]}、图\n")
        sec += 1
        lines.append(f"![分析图]({os.path.basename(fig_path)})\n")

    lines.append(f"## {CN[sec-1]}、决策日志（白盒记录）\n")
    sec += 1
    lines.append("```")
    lines.extend(res.log)
    lines.append("```\n")

    if llm_text:
        lines.append(f"## {CN[sec-1]}、AI 解读（大模型生成，仅供参考）\n")
        sec += 1
        lines.append(llm_text.strip() + "\n")
        lines.append("> 本段由大语言模型根据上面的算法结果改写生成，**不参与任何算法判断**，"
                     "措辞与结论以算法结果为准。\n")

    lines.append(f"## {CN[sec-1]}、方法与环境\n")
    lines.append("- 预处理：两级中值滤波去基线 + 0.5–40 Hz 带通 + 50 Hz 陷波")
    lines.append("- QRS 检测：自实现 Pan-Tompkins（SPKI/NPKI 双阈值 + 回溯搜索），"
                 "第二检测器 XQRS 用于事件二次确认")
    lines.append("- 质量评估（在仅去直流的原始信号上）：QRS 带比、基线比、工频比、高频比、"
                 "带通峰度、形态一致性 → 0–1 质量分；另计算 6 子频段能量比/熵（12 维）备查")
    lines.append("- 房颤样筛查：RR 域 8 维特征 + 逻辑回归（按记录分组交叉验证训练）")
    lines.append("- 五层策略：质量门控 → 分级 → 二次确认 → 人审兜底 → 个体定标")
    if p:
        lines.append(f"- 人群参数：{p.get('summary', '未提供')}；"
                     f"人群参考阈值 {p.get('threshold_text', '-')}")
        if d.calibrated:
            lines.append(f"- 个体定标后阈值：过速 >{d.calibrated['tachy']:.0f} bpm、"
                         f"过缓 <{d.calibrated['brady']:.0f} bpm、"
                         f"停搏 >{d.calibrated['pause_s']:.1f} s")
        if not p.get("af_enabled", True):
            lines.append("- 儿科说明：房颤筛查在 18 岁以下未验证，本次未启用；"
                         "年龄段心率阈值来自临床参考范围，未经本项目数据重新验证")
        elif not p.get("age_known", True):
            lines.append("- 人群说明：未提供年龄/性别/BMI，按成人默认阈值处理")
        for note in p.get("notes", []):
            lines.append(f"- 人群提示：{note}")
    lines.append("- 已知限制：本报告的臂带/腕带形态未定标；阈值来自公开数据集，"
                 "部署到自研设备前必须用自己的数据重新标定\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def render_all(res, out_dir: str, stem: str = "report", llm_text: str | None = None):
    os.makedirs(out_dir, exist_ok=True)
    fig = render_figure(res, os.path.join(out_dir, f"{stem}_fig.png"))
    md = render_markdown(res, os.path.join(out_dir, f"{stem}.md"), fig, llm_text=llm_text)
    return md, fig
