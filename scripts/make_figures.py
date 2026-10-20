"""生成汇报用图表（读 results/ 下的 CSV，输出 PNG）。

产出：
  results/fig_ablation.png     五层消融：正常组报警率 vs 房颤检出率
  results/fig_af.png           房颤筛查：评分分布 + 阈值扫描曲线
  results/fig_qrs.png          QRS 检测：两种检测器对比 + 逐记录散点
  results/fig_quality.png      质量门控：干净/干扰段质量分分布
"""

from __future__ import annotations

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def fig_ablation():
    path = "results/ablation_summary.csv"
    if not os.path.exists(path):
        print("跳过消融图（缺", path, "）")
        return
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    plt = _plt()
    names = [r["preset"] for r in rows]
    alerts = [float(r["alerts_per_hour"]) for r in rows]
    det = [float(r["af_detect_rate"]) * 100 for r in rows]

    fig, ax1 = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(names))
    b = ax1.bar(x - 0.2, alerts, width=0.4, color="#dc2626", label="正常组报警率（次/小时，越低越好）")
    ax1.set_ylabel("报警率（次/小时）", color="#dc2626")
    ax1.tick_params(axis="y", labelcolor="#dc2626")
    ax1.set_ylim(0, max(alerts + [0.5]) * 1.3)
    ax2 = ax1.twinx()
    ax2.plot(x + 0.2, det, "o-", color="#2563eb", lw=2, label="房颤检出率（%，越高越好）")
    ax2.set_ylabel("房颤检出率（%）", color="#2563eb")
    ax2.tick_params(axis="y", labelcolor="#2563eb")
    ax2.set_ylim(0, 110)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=10)
    ax1.set_title("五层防误报策略消融（串联各层后：误报↓、检出保持）")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig("results/fig_ablation.png", dpi=140)
    plt.close(fig)
    print("已写出 results/fig_ablation.png")


def fig_af():
    path = "results/af_mitdb.csv"
    if not os.path.exists(path):
        print("跳过房颤图（缺", path, "）")
        return
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    y = np.array([1 if r["label"] == "af" else 0 for r in rows])
    s = np.array([float(r["score"]) for r in rows])
    plt = _plt()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))

    bins = np.linspace(0, 1, 21)
    ax1.hist(s[y == 0], bins=bins, alpha=0.6, label=f"窦律窗 (n={int((1-y).sum())})", color="#059669")
    ax1.hist(s[y == 1], bins=bins, alpha=0.6, label=f"房颤窗 (n={int(y.sum())})", color="#dc2626")
    ax1.set_xlabel("房颤评分（规则版）")
    ax1.set_ylabel("窗口数")
    ax1.set_title("评分分布（两类的可分性）")
    ax1.legend(fontsize=9)

    ths = np.arange(0.2, 0.95, 0.05)
    ses, sps = [], []
    for t in ths:
        p = (s >= t).astype(int)
        tp = int(((y == 1) & (p == 1)).sum()); fn = int(((y == 1) & (p == 0)).sum())
        tn = int(((y == 0) & (p == 0)).sum()); fp = int(((y == 0) & (p == 1)).sum())
        ses.append(tp / max(1, tp + fn) * 100)
        sps.append(tn / max(1, tn + fp) * 100)
    ax2.plot(ths, ses, "o-", label="敏感度 Se", color="#dc2626")
    ax2.plot(ths, sps, "s-", label="特异度 Sp", color="#2563eb")
    ax2.set_xlabel("判定阈值")
    ax2.set_ylabel("%")
    ax2.set_title("阈值扫描（30 s 窗，mitdb 节律标注）")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig("results/fig_af.png", dpi=140)
    plt.close(fig)
    print("已写出 results/fig_af.png")


def fig_qrs():
    p1, p2 = "results/mitdb_qrs.csv", "results/mitdb_qrs_xqrs.csv"
    if not (os.path.exists(p1) and os.path.exists(p2)):
        print("跳过 QRS 图（缺对比 CSV）")
        return
    a = list(csv.DictReader(open(p1, encoding="utf-8")))
    b = list(csv.DictReader(open(p2, encoding="utf-8")))
    plt = _plt()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))

    se_a = [float(r["se"]) * 100 for r in a]
    ppv_a = [float(r["ppv"]) * 100 for r in a]
    se_b = [float(r["se"]) * 100 for r in b]
    ppv_b = [float(r["ppv"]) * 100 for r in b]
    ax1.scatter(se_a, ppv_a, s=26, alpha=0.7, label="自实现 Pan-Tompkins", color="#2563eb")
    ax1.scatter(se_b, ppv_b, s=26, alpha=0.7, label="wfdb XQRS", color="#f59e0b", marker="^")
    ax1.axhline(99, ls="--", lw=0.8, color="#9ca3af")
    ax1.axvline(99, ls="--", lw=0.8, color="#9ca3af")
    ax1.set_xlabel("Se (%)")
    ax1.set_ylabel("PPV (%)")
    ax1.set_title("逐记录 QRS 检测（mitdb 48 条，±150 ms）")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # 汇总条形
    def agg(rows):
        tp = sum(int(r["tp"]) for r in rows); fp = sum(int(r["fp"]) for r in rows)
        fn = sum(int(r["fn"]) for r in rows)
        return tp / (tp + fn) * 100, tp / (tp + fp) * 100
    se1, ppv1 = agg(a); se2, ppv2 = agg(b)
    x = np.arange(2)
    ax2.bar(x - 0.18, [se1, se2], width=0.34, label="Se", color="#dc2626")
    ax2.bar(x + 0.18, [ppv1, ppv2], width=0.34, label="PPV", color="#059669")
    for xi, (s_, p_) in enumerate([(se1, ppv1), (se2, ppv2)]):
        ax2.text(xi - 0.18, s_ + 0.1, f"{s_:.2f}", ha="center", fontsize=9)
        ax2.text(xi + 0.18, p_ + 0.1, f"{p_:.2f}", ha="center", fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(["自实现 Pan-Tompkins", "wfdb XQRS"])
    ax2.set_ylim(95, 100.5)
    ax2.set_ylabel("%")
    ax2.set_title("全库汇总")
    ax2.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig("results/fig_qrs.png", dpi=140)
    plt.close(fig)
    print("已写出 results/fig_qrs.png")


def fig_quality():
    """干净记录 vs 注入干扰的质量分分布。"""
    try:
        from ecg_agent.algorithms import quality as Q
        from ecg_agent.algorithms.preprocess import prepare_for_quality
        from ecg_agent.io import mitdb
    except Exception as e:
        print("跳过质量图（依赖导入失败：%s）" % e)
        return
    x, fs, _ = mitdb.read("100", channel=0)
    x = x[:300 * int(fs)]
    xq, fsq = prepare_for_quality(x, fs, 250.0)
    clean = [w["quality"] for w in Q.assess(xq, fsq)]
    bank = Q.synth_noise_bank(fsq, 300, seed=1)
    series = {"干净": clean}
    for k, sc in [("基线漂移", ("baseline", 4.0)), ("工频", ("mains", 8.0)),
                  ("肌电", ("emg", 6.0)), ("运动伪差", ("motion", 4.0))]:
        y = xq + bank[sc[0]][:len(xq)] * sc[1]
        series[k] = [w["quality"] for w in Q.assess(y, fsq)]

    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 4.3))
    data = [series[k] for k in series]
    bp = ax.boxplot(data, tick_labels=list(series.keys()), patch_artist=True)
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor("#bbf7d0" if i == 0 else "#fecaca")
    ax.axhline(0.6, ls="--", color="#dc2626", lw=1.2, label="门控阈值 0.6")
    ax.set_ylabel("质量分")
    ax.set_title("质量门控对 4 类干扰的响应（record 100 注入）")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig("results/fig_quality.png", dpi=140)
    plt.close(fig)
    print("已写出 results/fig_quality.png")


def fig_beat():
    """AAMI 五类心拍分类：各类 Se/PPV/F1。"""
    path = "results/beat_class.csv"
    if not os.path.exists(path):
        print("跳过心拍分类图（缺", path, "）")
        return
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 4.3))
    names = [f"{r['class']} (n={r['support']})" for r in rows]
    se = [float(r["se"]) * 100 for r in rows]
    ppv = [float(r["ppv"]) * 100 for r in rows]
    f1 = [float(r["f1"]) * 100 for r in rows]
    x = np.arange(len(rows))
    ax.bar(x - 0.27, se, width=0.26, label="Se", color="#dc2626")
    ax.bar(x, ppv, width=0.26, label="PPV", color="#059669")
    ax.bar(x + 0.27, f1, width=0.26, label="F1", color="#2563eb")
    for xi in x:
        for dx, vals in ((-0.27, se), (0.0, ppv), (0.27, f1)):
            ax.text(xi + dx, vals[xi] + 1.2, f"{vals[xi]:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(0, 108)
    ax.set_ylabel("%")
    ax.set_title("AAMI 五类心拍分类（mitdb，RandomForest，按记录分组 CV）")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig("results/fig_beat.png", dpi=140)
    plt.close(fig)
    print("已写出 results/fig_beat.png")


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    fig_qrs()
    fig_af()
    fig_quality()
    fig_beat()
    fig_ablation()
