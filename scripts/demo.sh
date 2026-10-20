#!/usr/bin/env bash
# 一键演示：心电分析智能体（提示级，非诊断）
#
# 用法：bash scripts/demo.sh
# 需要先建好 .venv（见 README）。

set -u
cd "$(dirname "$0")/.."
PY=".venv/Scripts/python.exe"
export PYTHONPATH=src

echo "=============================================================="
echo " 心电分析智能体演示（ecg_agent）"
echo " 口径：只出提示级结论（疑似/建议就医），不做诊断"
echo "=============================================================="

run() {  # 参数：标题 来源 附加参数...
  local title="$1"; shift
  echo
  echo "--------------------------------------------------------------"
  echo "▶ $title"
  echo "--------------------------------------------------------------"
  "$PY" -m ecg_agent.cli analyze "$@" 2>&1 | sed 's/^/  /'
}

# ① 有异常的真实记录：MIT-BIH 203（含房颤段）
run "① 含房颤段记录（mitdb:203，第 10–40 分钟）" \
    --source mitdb:203 --start 600 --minutes 30 --out reports/demo_203 --stem mitdb203

# ② 干净记录：不应该报警（防误报的反面验证）
run "② 窦律记录（mitdb:100，前 20 分钟）——期望：无上报事件" \
    --source mitdb:100 --minutes 20 --out reports/demo_100 --stem mitdb100

# ③ 同一段记录、不同策略配置（消融演示：裸阈值 vs 完整五层）
#    用 114 号记录（低幅+噪声，裸阈值在这里会误报 37 条，串联五层后只剩 6 条）
run "③ 同一段记录 · 裸阈值配置（只有筛查，没有防误报层）" \
    --source mitdb:114 --minutes 30 --layers 裸阈值
run "③ 同一段记录 · 完整五层（串联防误报策略）" \
    --source mitdb:114 --minutes 30 --layers 完整五层 \
    --out reports/demo_ablation_114 --stem mitdb114_五层

# ④ 真实穿戴长时程数据：PPG-DaLiA S1（前 30 分钟）
if [ -f "data/cache/ppg_dalia/S1_ecg.npy" ]; then
  run "④ 真实穿戴长时程（PPG-DaLiA S1，前 30 分钟）" \
      --source ppg-dalia:S1 --minutes 30 --out reports/demo_S1 --stem ppg_dalia_S1
else
  echo
  echo "（跳过 PPG-DaLiA 演示：先运行 scripts/extract_ppg_dalia.py 生成缓存）"
fi

echo
echo "=============================================================="
echo " 报告已生成在 reports/ 下（Markdown + 图 PNG）"
echo " 指标与误差分析见 docs/算法进展.md，汇报图见 results/fig_*.png"
echo "=============================================================="
