"""心电分析智能体 · 网页端（Streamlit，本地运行）

启动：
    .venv\\Scripts\\python.exe -m streamlit run app.py
或双击项目根目录的「启动网页端.bat」。

功能：选数据源 → 跑五层策略分析 → 看结论/事件/图/决策日志 → 可选 DeepSeek 解读与追问。
口径：所有输出为提示级（疑似 / 建议就医复核），不做诊断。
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st  # noqa: E402

from ecg_agent.agent import pipeline  # noqa: E402
from ecg_agent.agent import report as R  # noqa: E402
from ecg_agent.io import mitdb, respiban  # noqa: E402
from ecg_agent.layers import strategy as S  # noqa: E402

st.set_page_config(page_title="心电分析智能体", page_icon="❤️", layout="wide")

# ------------------------------------------------------------------ 侧边栏
with st.sidebar:
    st.title("❤️ 心电分析智能体")
    st.caption("提示级结果 · 非诊断用途 · 需医生复核")

    src_kind = st.radio("数据源", ["MIT-BIH 记录（带标注）", "穿戴长时程（PPG-DaLiA）", "上传文件"],
                        help="MIT-BIH 用于演示与验证；PPG-DaLiA 是真实穿戴 37 小时数据")

    source = None
    if src_kind.startswith("MIT-BIH"):
        recs = mitdb.record_ids()
        rec = st.selectbox("记录号", recs, index=recs.index("114") if "114" in recs else 0)
        st.caption("114 号是低幅+噪声记录，适合演示防误报；100 号是干净窦律记录")
        source = f"mitdb:{rec}"
    elif src_kind.startswith("穿戴"):
        sid = st.selectbox("受试者", respiban.subject_ids(), index=0)
        st.caption("每例约 2.5 小时（700 Hz 提取后按 250 Hz 缓存）")
        source = f"ppg-dalia:{sid}"
    else:
        up = st.file_uploader("上传 .npy（一维数组）或 .csv（单列）", type=["npy", "csv"])
        fs_up = st.number_input("采样率 Hz", 50, 1000, 250, 10)
        if up is not None:
            import numpy as np

            tmp = os.path.join("data", "_upload.npy")
            os.makedirs("data", exist_ok=True)
            if up.name.endswith(".npy"):
                arr = np.load(up, allow_pickle=False)
            else:
                arr = np.loadtxt(up, delimiter=",", ndmin=1)
            np.save(tmp, np.asarray(arr, dtype=np.float64).ravel())
            source = f"file:{tmp}"
            st.caption(f"已载入 {len(arr)} 点，按 {fs_up:.0f} Hz 处理")
            st.session_state["_upload_fs"] = fs_up

    st.divider()
    minutes = st.number_input("分析时长（分钟，0=全部）", 0.0, 600.0, 30.0, 5.0)
    start = st.number_input("起始位置（秒）", 0.0, 100000.0, 0.0, 60.0)
    layers = st.selectbox("五层策略", list(S.PRESETS.keys()),
                          index=list(S.PRESETS.keys()).index("完整五层"))
    st.caption("选「裸阈值」对比：同一条记录会出现大量误报")
    run = st.button("▶ 开始分析", type="primary", width="stretch")

    st.divider()
    from ecg_agent.agent import llm

    ok = bool(llm.api_key())
    st.caption(f"DeepSeek：{'已配置 ✅' if ok else '未配置（AI 解读不可用）'}")

# ------------------------------------------------------------------ 执行分析
if run and source:
    if source.startswith("file:"):
        # 上传文件用指定采样率：临时改写 load_source 的默认 250
        minutes_v = minutes or None
        start_v = start
        with st.spinner("读数据 → 预处理 → 双检测器 → 质量评估 → 五层策略…"):
            t0 = time.time()
            res = pipeline.analyze(source, minutes=minutes_v, start_s=start_v, layers=layers,
                                   verbose=False)
            # 上传文件若采样率不是 250，重采样一次
            fs_up = float(st.session_state.get("_upload_fs", 250))
            if abs(fs_up - 250.0) > 1e-6:
                from ecg_agent.algorithms.preprocess import preprocess
                res.x, res.fs = preprocess(res.x, res.fs, target_fs=250.0)
                res.log.append(f"[工具:preprocess] 上传文件按 {fs_up:.0f} Hz → 250 Hz 重采样")
            elapsed = time.time() - t0
    else:
        with st.spinner("读数据 → 预处理 → 双检测器 → 质量评估 → 五层策略…"):
            t0 = time.time()
            res = pipeline.analyze(source, minutes=(minutes or None), start_s=start,
                                   layers=layers, verbose=False)
            elapsed = time.time() - t0
    st.session_state["res"] = res
    st.session_state["elapsed"] = elapsed
    st.session_state["chat"] = []

res = st.session_state.get("res")

if res is None:
    st.info("👈 在左侧选好数据源和参数，点「开始分析」。\n\n"
            "**建议第一次这样试**：数据源选 MIT-BIH、记录号 **114**、时长 30 分钟、"
            "五层策略选「裸阈值」跑一次，再选「完整五层」跑一次——对比上报事件数。")
    st.stop()

d = res.decision
st.caption(f"来源 `{res.source}` ｜ 时长 {res.duration_s/60:.1f} 分钟 ｜ "
           f"分析耗时 {st.session_state.get('elapsed', 0):.1f} 秒 ｜ 策略：{layers}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("总体分级", {"normal": "未见需即刻关注", "caution": "关注", "refer": "建议就医复核"}.get(d.level, d.level))
c2.metric("上报事件", d.n_alerts)
c3.metric("被压掉候选", len(d.suppressed))
c4.metric("有效监测占比", f"{d.coverage*100:.0f}%")

tab1, tab2, tab3, tab4 = st.tabs(["📋 结论与事件", "📈 图", "🧠 决策日志", "💬 AI 解读 & 追问"])

with tab1:
    if d.n_alerts == 0:
        st.success("本次记录未见需要即刻关注的疑似异常心律。")
    else:
        st.warning(f"检出 {d.n_alerts} 处需要提示的疑似心律变化，建议携带完整记录就医复核。")
    st.markdown("**上报事件**")
    alerts = [e for e in d.events if e.severity in ("caution", "refer")]
    infos = [e for e in d.events if e.severity not in ("caution", "refer")]
    if alerts:
        st.dataframe([{"类型": R.KIND_CN.get(e.kind, e.kind), "时间(分)": round(e.start / 60, 1),
                       "分级": R.LEVEL_CN.get(e.severity, e.severity),
                       "置信度": round(e.confidence, 2), "依据": e.detail} for e in alerts],
                     width="stretch", hide_index=True)
    else:
        st.write("（无）")
    if infos:
        from collections import Counter

        cnt = Counter(e.kind for e in infos)
        st.markdown("**仅计数提示（不上报为报警）**：" +
                    "、".join(f"{R.KIND_CN.get(k, k)} {v} 次" for k, v in cnt.items()))
    if d.suppressed:
        with st.expander(f"被防误报策略压掉的候选（{len(d.suppressed)} 个，可审计）"):
            st.dataframe([{"类型": R.KIND_CN.get(e.kind, e.kind), "时间(分)": round(e.start / 60, 1),
                           "压掉原因": why} for e, why in d.suppressed[:200]],
                         width="stretch", hide_index=True)
    st.caption("本结果由算法自动生成，仅供健康提示、**不构成诊断**，需医生复核。")

with tab2:
    os.makedirs("reports/_web", exist_ok=True)
    fig_path = os.path.join("reports", "_web", "fig.png")
    R.render_figure(res, fig_path)
    st.image(fig_path, width="stretch")

with tab3:
    st.code("\n".join(res.log), language="text")
    st.caption("这是智能体的决策过程：每一步调用哪个工具、每个候选事件为什么被压掉，全部可追溯。")

with tab4:
    if not ok:
        st.warning("未配置 DeepSeek API Key，AI 解读不可用。\n\n"
                   "配置方法：项目根目录放一个 `.ds_key` 文件（内容一行 key），"
                   "或设置环境变量 `DEEPSEEK_API_KEY`。")
    else:
        from ecg_agent.agent import llm

        if st.button("生成 AI 解读（把结果改写成给受检者看的人话）"):
            with st.spinner("调用 DeepSeek…"):
                try:
                    st.session_state["narr"] = llm.narrate(res)
                except Exception as e:
                    st.error(f"调用失败：{type(e).__name__}: {e}")
        if st.session_state.get("narr"):
            st.markdown(st.session_state["narr"])
            st.caption("本段由大语言模型改写，不参与算法判断；措辞与结论以算法结果为准。")

        st.divider()
        st.markdown("**追问**（基于本次分析结果回答）")
        for q, a in st.session_state.get("chat", []):
            with st.chat_message("user"):
                st.write(q)
            with st.chat_message("assistant"):
                st.write(a)
        q = st.chat_input("例如：为什么报了 6 个事件、另外 55 个被压掉？会不会漏诊？")
        if q:
            with st.chat_message("user"):
                st.write(q)
            with st.chat_message("assistant"):
                with st.spinner("思考中…"):
                    try:
                        a = llm.ask(res, q)
                    except Exception as e:
                        a = f"[调用失败] {type(e).__name__}: {e}"
                st.write(a)
            st.session_state.setdefault("chat", []).append((q, a))
