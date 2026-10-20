# ecg_agent —— 可穿戴心电分析智能体（算法主线）

课程项目「便携式多导联心脏检测仪」的算法交付物：**心电分析管线 + 五层防误报策略 + 最小智能体外壳**。

课程要求点名两件事——"用 AI 智能体简化调研及算法基本编写，完成基本算法落地"、
"重点防误报错报，用多重策略保证精确性"——本目录就是这两条的落地：

1. **基本算法落地**：预处理 → 信号质量评估 → QRS 检测 → 心率/HRV → 心律异常筛查，
   全部自实现（Python + numpy/scipy），在公开标注数据上给出可复算的指标；
2. **多重策略防误报**：五层策略（质量门控 → 分级 → 二次确认 → 人审兜底 → 个体定标），
   每层可单独开关，配有消融实验脚本。

> 口径红线（同调研报告）：所有输出都是**提示级**（"疑似""建议就医复核"），
> **不做诊断、不做预警**；报告里也不出现"诊断/预警率"这类表述。

---

## 快速开始

```bash
cd ecg_agent            # 换成你的实际路径

# 1) 隔离环境（已建好则可跳过；--system-site-packages 复用 numpy/scipy/sklearn/matplotlib）
python -m venv --system-site-packages .venv
.venv/Scripts/python.exe -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple wfdb modelscope kaggle joblib

# 2) 一键演示：分析一段记录 → 打印决策日志 → 生成图文报告
#    推荐用 Python 版（Windows 上不依赖 bash；直接双击也行）
.venv/Scripts/python.exe scripts/demo.py
#    Git Bash 用户也可以用：bash scripts/demo.sh
#    注意：Windows 命令行里的 `bash` 会指向 WSL 的 bash（System32\bash.exe），
#    如果 WSL 没配好会报 0x80072747，那不是本项目的问题——改用上面这条 Python 命令，
#    或显式调 Git Bash："D:/Git/bin/bash.exe" scripts/demo.sh

# 3) 手动分析
PYTHONPATH=src .venv/Scripts/python.exe -m ecg_agent.cli analyze \
    --source mitdb:203 --start 600 --minutes 30 --out reports/demo
#    --source 支持 ppg-dalia:S1（真实长时程）/ mitdb:100（带标注）/ file:xx.npy
#    --layers 支持 裸阈值 / +质量门控 / +二次确认 / +个体定标 / 完整五层（消融演示）
```

## 网页端（Streamlit，本地运行）

双击项目根目录的 **`start_web.bat`**，或命令行：

```cmd
cd /d D:\pythoncode\frb\ecg_agent
set PYTHONPATH=src
.venv\Scripts\python.exe -m streamlit run app.py --server.port 8600
```

然后浏览器打开 **http://localhost:8600**（局域网内手机也能开：启动日志里的 Network URL）。

网页上能做：选数据源（mitdb 记录 / 穿戴长时程 / 上传 npy、csv）→ 选五层策略 →
「开始分析」→ 看结论、事件表、被压掉候选、图文报告、决策日志 → 一键生成 AI 解读、追问。

> **端口为什么是 8600**：Windows/Hyper-V 把 **8457–8556** 保留成排除端口段，
> Streamlit 默认的 8501 正好落在里面，绑定会报 `WinError 10013`。
> 查自己机器的保留段：`netsh int ipv4 show excludedportrange protocol=tcp`。

## 📦 换一台电脑怎么跑（发给同学看这段）

包里已经带了 **5 条 MIT-BIH 记录 + 1 例穿戴数据**，解压后可以直接跑全部演示；
要复算全库指标，再补全数据（约 77 MB，国内 20 MB/s）。

```cmd
:: 1) 建环境（Python 3.10+，推荐 3.13）
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

:: 2) 检查/补齐数据
.venv\Scripts\python.exe scripts\setup_data.py            :: 看现状
.venv\Scripts\python.exe scripts\setup_data.py --mitdb    :: 补全 48 条记录

:: 3) 跑起来
双击 start_web.bat                                       :: 网页端 → http://localhost:8600
set PYTHONPATH=src
.venv\Scripts\python.exe -m ecg_agent.cli analyze --source mitdb:114 --minutes 30 --llm
```

> 包内**没有** API Key（`.ds_key` 已被排除）。要测大模型解读，自己配一个：
> 项目根目录新建 `.ds_key` 文件、写一行 key；或 `set DEEPSEEK_API_KEY=sk-xxx`。

## 大模型解读（可选，DeepSeek）

算法本身**完全离线**；大模型只做"把结果改写成给人看的解读"，**不参与任何判断**。

```bash
# 配 key（二选一）：
#   cmd:        set DEEPSEEK_API_KEY=sk-xxxx        （临时，当前窗口有效）
#   cmd:        setx DEEPSEEK_API_KEY "sk-xxxx"     （永久，重开窗口生效）
#   或：把 key 写成一行存在项目根目录的 .ds_key 文件

# 带 AI 解读出报告
PYTHONPATH=src .venv/Scripts/python.exe -m ecg_agent.cli analyze --source mitdb:114 --minutes 30 --llm

# 追问模式（基于本次分析结果问答；不带 -q 则进入交互）
PYTHONPATH=src .venv/Scripts/python.exe -m ecg_agent.cli ask --source mitdb:114 --minutes 30 -q "为什么报了6个事件？"
```

`.ds_key` 已加进 `.gitignore`，**不要提交、不要外发**。

## 目录结构

```
ecg_agent/
├── src/ecg_agent/
│   ├── io/            数据接入：respiban.py（PPG-DaLiA h5 提 ECG）、mitdb.py（标注解析）
│   ├── algorithms/    preprocess / quality(质量门控) / qrs / hrv / rhythm / metrics / afmodel
│   ├── layers/        strategy.py —— 五层防误报策略（可逐层开关）
│   ├── agent/         pipeline.py（工具编排+决策日志）、report.py（图文报告）
│   └── cli.py         命令行入口
├── scripts/           提取缓存 / 评测 / 训练 / 消融 / 出图 / 演示
├── data/              公开数据与缓存（大文件，不入库）
├── results/           指标 CSV 与汇报图 PNG
├── models/            af_logreg.joblib（房颤筛查模型）
├── reports/           演示生成的报告样例
└── docs/算法进展.md    组会/汇报用的一页纸（指标、误差分析、下一步）
```

## 数据来源与用途

| 数据 | 位置 | 用途 | 许可/来源 |
|------|------|------|-----------|
| **MIT-BIH 心律失常库**（mitdb，48 条 30 min，360 Hz，逐拍+节律标注） | `data/mitdb/` | **唯一指标来源**：QRS 检测 Se/PPV、房颤时段筛查、五层消融 | PhysioNet → ModelScope 镜像（`OmniData/MIT-BIH_Arrhythmia_Database`）|
| **PPG-DaLiA**（RespiBAN `channel_1` = ECG，700 Hz，15 例 × 2.5 h） | 原始 h5 放 `PPG_FieldStudy/`（或设 `PPG_DALIA_ROOT`），缓存到 `data/cache/ppg_dalia/` | 真实长时程演示（穿戴场景的报警率/质量覆盖率），**不出 Se/Sp**（无心律标注） | UCI / ModelScope |
| PhysioNet/CinC 2017（Kaggle 版） | ~~已弃用~~ | 该镜像每条仅 2000 采样点（≈6.7 s），不能做 30 s 窗房颤评估 | Kaggle `luigisaetta/physionet2017ecg` |

> 注意：`data/` 下数据体积较大（mitdb ≈ 100 MB，缓存 ≈ 400 MB），不要提交到公开仓库。

## 一键复现全部指标

```bash
.venv/Scripts/python.exe scripts/extract_ppg_dalia.py            # ① 提取 PPG-DaLiA 的 ECG 并缓存
.venv/Scripts/python.exe scripts/eval_mitdb_qrs.py               # ② QRS 检测（自实现 PT）
.venv/Scripts/python.exe scripts/eval_mitdb_qrs.py --detector xqrs   #    对照：wfdb XQRS
.venv/Scripts/python.exe scripts/eval_af_mitdb.py                # ③ 房颤时段筛查（规则版）
.venv/Scripts/python.exe scripts/train_af_model.py               # ④ 训练/交叉验证房颤模型
.venv/Scripts/python.exe scripts/ablation_layers.py              # ⑤ 五层消融（真实数据+噪声压力测试）
.venv/Scripts/python.exe scripts/make_figures.py                 # ⑥ 生成汇报图
```

## 主要结果（详见 `docs/算法进展.md`）

| 任务 | 数据 | 指标 |
|------|------|------|
| QRS 检测（自实现 Pan-Tompkins） | mitdb 48 条全库，±150 ms | **Se 99.22% / PPV 98.81%** |
| QRS 检测（wfdb XQRS 对照） | 同上 | Se 99.09% / PPV 99.36% |
| 房颤样心律筛查（规则版，30 s 窗） | mitdb 节律标注，2044 窗 | Se 90.4% / Sp 89.7% / Acc 89.8% |
| 房颤样心律筛查（逻辑回归，按记录分组 CV） | 同上 | **Se 91.7% / Sp 93.6% / Acc 93.4%** |
| 室性早搏（AAMI V 类，RandomForest） | mitdb 99,680 拍 | **Se 94.8% / PPV 85.5%** |
| 五层防误报消融 | mitdb 窦律时段 4.0 h | **报警率 5.95 → 1.84 次/小时（-69%）** |

> 汇报图见 `results/fig_*.png`（QRS 对比 / 房颤评分与阈值扫描 / 质量门控响应 / 心拍分类 / 五层消融）。

## 已知限制（写在前面，汇报时主动交代）

- 房颤筛查的 PPV 受人群患病率影响大（本评价集 AF 占比约 10.7%，PPV 仅 60% 上下）——
  这恰是"多重策略"存在的理由，也是上线前必须用目标人群数据重新标定的原因；
- 阈值全部来自公开数据（多为欧美人群、院内采集），**换到自研设备/自采数据必须重新标定**；
- 质量门控对运动伪差偏宽松（实测注入 ×4 运动伪差仍有约 7% 窗口评分为"合格"）；
- PPG-DaLiA 只用于"长时程真实数据"演示，不用于报告敏感度/特异度（它没有心律标注）；
- 本目录不涉及 7 导联/12 导联的临床覆盖讨论（那是调研报告的范围），也不做心梗定位。
