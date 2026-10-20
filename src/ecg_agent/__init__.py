"""ecg_agent —— 可穿戴心电分析智能体（算法基线 + 五层防误报策略）。

课程项目「便携式多导联心脏检测仪」算法模块：
  - io          : 数据接入（PPG-DaLiA / MIT-BIH / PhysioNet2017）
  - algorithms  : 预处理、信号质量评估、QRS 检测、HR/HRV、心律异常筛查
  - layers      : 五层防误报策略（质量门控 / 分级预警 / 二次确认 / 人审话术 / 个体定标）
  - agent       : 工具封装 + 决策流程 + 报告渲染
"""

__version__ = "0.1.0"
