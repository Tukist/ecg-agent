"""一键演示（纯 Python 版，不依赖 bash）——Windows 上直接双击/命令行都能跑。

用法（在 ecg_agent 目录下）：
    .venv\\Scripts\\python.exe scripts\\demo.py

与 scripts/demo.sh 的四段演示一致：
  ① 含房颤段记录（mitdb:203）      ② 窦律记录（mitdb:100，期望无报警）
  ③ 同一记录裸阈值 vs 完整五层     ④ 真实穿戴长时程（PPG-DaLiA S1）
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from ecg_agent.agent import pipeline, report as R  # noqa: E402


def run(title: str, source: str, minutes=None, start=0.0, layers="完整五层", out=None, stem="report"):
    print()
    print("-" * 62)
    print(f"▶ {title}")
    print("-" * 62)
    res = pipeline.analyze(source, minutes=minutes, start_s=start, layers=layers, verbose=False)
    for line in res.log:
        print("  " + line)
    d = res.decision
    print(f"\n  结论：{R.LEVEL_CN.get(d.level, d.level)} | 上报事件 {d.n_alerts} 个 | "
          f"被压掉候选 {len(d.suppressed)} 个 | 有效监测占比 {d.coverage*100:.1f}%")
    for e in d.events:
        print(f"    · [{e.severity}] {R.KIND_CN.get(e.kind, e.kind)} @ {e.start/60:.1f} min（{e.detail}）")
    if out:
        md, fig = R.render_all(res, out, stem=stem)
        print(f"  报告已写出：{md}")
    return res


def main():
    print("=" * 62)
    print(" 心电分析智能体演示（ecg_agent）")
    print(" 口径：只出提示级结论（疑似/建议就医），不做诊断")
    print("=" * 62)

    run("① 含房颤段记录（mitdb:203，第 10–40 分钟）",
        "mitdb:203", minutes=30, start=600.0, out="reports/demo_203", stem="mitdb203")

    run("② 窦律记录（mitdb:100，前 20 分钟）——期望：无上报事件",
        "mitdb:100", minutes=20, out="reports/demo_100", stem="mitdb100")

    print("\n【③ 的看点】同一条记录（114，低幅+噪声），裸阈值会大量误报，串联五层后被压掉。")
    run("③ 同一段记录 · 裸阈值配置（只有筛查，没有防误报层）",
        "mitdb:114", minutes=30, layers="裸阈值")
    run("③ 同一段记录 · 完整五层（串联防误报策略）",
        "mitdb:114", minutes=30, layers="完整五层",
        out="reports/demo_ablation_114", stem="mitdb114_五层")

    if os.path.exists("data/cache/ppg_dalia/S1_ecg.npy"):
        run("④ 真实穿戴长时程（PPG-DaLiA S1，前 30 分钟）",
            "ppg-dalia:S1", minutes=30, out="reports/demo_S1", stem="ppg_dalia_S1")
    else:
        print("\n（跳过 PPG-DaLiA 演示：先运行 scripts/extract_ppg_dalia.py 生成缓存）")

    print()
    print("=" * 62)
    print(" 报告在 reports/ 下（Markdown + 图 PNG）")
    print(" 指标与误差分析见 docs/算法进展.md，汇报图见 results/fig_*.png")
    print("=" * 62)


if __name__ == "__main__":
    main()
