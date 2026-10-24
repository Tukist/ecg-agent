"""心电分析智能体 · 网页端（Streamlit，本地运行）

启动：
    .venv\\Scripts\\python.exe -m streamlit run app.py
或双击项目根目录的「启动网页端.bat」。

功能：选数据源 → 跑五层策略分析 → 看结论/事件/图/决策日志 → 可选 DeepSeek 解读与追问。
口径：所有输出为提示级（疑似 / 建议就医复核），不做诊断。
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from ecg_agent.agent import pipeline  # noqa: E402
from ecg_agent.agent import report as R  # noqa: E402
from ecg_agent.io import db_labels, mitdb, respiban  # noqa: E402
from ecg_agent.layers import strategy as S  # noqa: E402
from ecg_agent.subject import SubjectProfile  # noqa: E402

st.set_page_config(page_title="心电分析智能体", page_icon="❤️", layout="wide")

# ------------------------------------------------------------------ 侧边栏
with st.sidebar:
    st.title("❤️ 心电分析智能体")
    st.caption("提示级结果 · 非诊断用途 · 需医生复核")

    src_kind = st.radio("数据源", ["MIT-BIH 记录（带标注）", "穿戴长时程（PPG-DaLiA）", "上传文件"],
                        help="MIT-BIH 用于演示与验证；PPG-DaLiA 是真实穿戴 37 小时数据；"
                             "上传文件支持 .npy / .csv 单文件，或一次上传 WFDB 记录的 .hea+.dat/.mat 多文件")

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
        up = st.file_uploader(
            "上传数据文件",
            type=["npy", "csv", "hea", "dat", "mat"],
            accept_multiple_files=True,
            help="· .npy / .csv：单文件，需手动填采样率\n"
                 "· WFDB：一次上传同一记录的 .hea + .dat/.mat（可含标注 .atr），自动识别导联和采样率"
        )

        if up:
            import numpy as np
            import wfdb

            # 清理之前的临时目录
            old_tmp = st.session_state.get("_wfdb_tmp_dir")
            if old_tmp and os.path.isdir(old_tmp):
                try:
                    shutil.rmtree(old_tmp)
                except Exception:
                    pass
                st.session_state.pop("_wfdb_tmp_dir", None)

            if len(up) == 1 and up[0].name.lower().endswith((".npy", ".csv")):
                # 单文件模式：.npy / .csv
                f = up[0]
                fs_up = st.number_input("采样率 Hz", 50, 1000, 250, 10)
                tmp = os.path.join("data", "_upload.npy")
                os.makedirs("data", exist_ok=True)
                if f.name.endswith(".npy"):
                    arr = np.load(f, allow_pickle=False)
                else:
                    arr = np.loadtxt(f, delimiter=",", ndmin=1)
                np.save(tmp, np.asarray(arr, dtype=np.float64).ravel())
                source = f"file:{tmp}"
                st.caption(f"已载入 {len(arr)} 点，按 {fs_up:.0f} Hz 处理")
                st.session_state["_upload_fs"] = fs_up
                st.session_state.pop("_wfdb_channels", None)
                st.session_state.pop("_wfdb_fs", None)

            else:
                # 多文件模式：WFDB 记录
                # 必须有 .hea，其余 .dat/.mat/.atr 等作为配套文件
                hea_files = [f for f in up if f.name.lower().endswith(".hea")]
                if not hea_files:
                    st.error("⚠️ 缺少 .hea 头文件。"
                             "请在文件对话框中按住 **Ctrl** 同时选中该记录的 `.hea` 和 `.dat`/`.mat` 文件，再一起打开。"
                             "例如 European ST-T 的 `e0103` 需要同时上传 `e0103.hea` + `e0103.dat`。")
                else:
                    tmp_dir = tempfile.mkdtemp(prefix="wfdb_upload_")
                    st.session_state["_wfdb_tmp_dir"] = tmp_dir

                    for f in up:
                        dest = os.path.join(tmp_dir, f.name)
                        with open(dest, "wb") as out:
                            out.write(f.getbuffer())

                    # 以第一个 .hea 作为主记录
                    hea_name = hea_files[0].name[:-4]
                    rec_path = os.path.join(tmp_dir, hea_name)

                    try:
                        rec = wfdb.rdrecord(rec_path, physical=True)
                        channels = list(rec.sig_name)
                        fs_detected = float(rec.fs)
                        st.session_state["_wfdb_channels"] = channels
                        st.session_state["_wfdb_fs"] = fs_detected
                        st.session_state["_wfdb_rec_path"] = rec_path
                        st.caption(f"检测到 {len(channels)} 个导联：{', '.join(channels)} ｜ 采样率 {fs_detected:.0f} Hz")
                    except Exception as e:
                        st.error(f"读取 WFDB 记录失败：{e}")
                        channels = []
                        fs_detected = None

                    if channels:
                        # 推荐默认导联
                        default_candidates = ["II", "MLIII", "ECG", "V1", "V2", "V5"]
                        default_ch = next((c for c in default_candidates if c in channels), channels[0])
                        channel = st.selectbox(
                            "导联/通道",
                            channels,
                            index=channels.index(default_ch),
                            help="12 导联建议选 II；European ST-T 建议选 MLIII；BUT-QDB 单导联选 ECG"
                        )
                        source = f"wfdb:{rec_path}|{channel}"
                        st.caption(f"已选导联：{channel}")

                        # 尝试读取数据库参考标注（如果上传的是已知数据库中的记录）
                        db_label = db_labels.lookup(source)
                        if db_label:
                            st.info(f"数据库参考标注：{db_label}")
        else:
            st.info("请选择文件上传：\n"
                    "· `.npy` / `.csv`：选单个文件即可\n"
                    "· WFDB 记录：在文件对话框里按住 **Ctrl** 同时选中 `.hea` + `.dat`/`.mat`（可含 `.atr`）再打开")
            # 清理可能残留的状态
            for k in ["_wfdb_channels", "_wfdb_fs", "_wfdb_rec_path"]:
                st.session_state.pop(k, None)

    st.divider()
    st.subheader("受检者信息（可选）")
    age_in = st.number_input("年龄（岁，0=未填写）", 0.0, 120.0, 0.0, 1.0,
                             help="儿童填实际年龄；1 个月以下可填 0.1。0 表示未填写，按成人默认")
    sex_cn = st.selectbox("性别", ["未填写", "男", "女"])
    bmi_in = st.number_input("BMI（0=未填写）", 0.0, 60.0, 0.0, 0.1,
                             help="用于过速上限的弱修正；儿童 BMI 应按年龄性别百分位判读")
    profile = SubjectProfile(age=age_in or None,
                             sex=None if sex_cn == "未填写" else sex_cn,
                             bmi=bmi_in or None)
    with st.expander("人群参数说明"):
        st.caption("阈值顺序：人群先验 → 本人前 5 分钟基线微调。"
                   "年龄段心率区间来自临床参考范围；性别/BMI 只做过速上限的弱修正。"
                   "儿科档位未在本项目数据上验证，房颤筛查在 18 岁以下不启用。")

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
                                   verbose=False, profile=profile)
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
                                   layers=layers, verbose=False, profile=profile)
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
_p = d.population or {}
if _p:
    st.caption(f"受检者：{_p.get('summary', '未提供')} ｜ "
               f"人群参考阈值：{_p.get('threshold_text', '-')}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("总体分级", {"normal": "未见需即刻关注", "caution": "关注", "refer": "建议就医复核"}.get(d.level, d.level))
c2.metric("上报事件", d.n_alerts)
c3.metric("被压掉候选", len(d.suppressed))
c4.metric("有效监测占比", f"{d.coverage*100:.0f}%")

tab1, tab2, tab3, tab4 = st.tabs(["📋 结论与事件", "📈 图", "🧠 决策日志", "💬 AI 解读 & 追问"])

with tab1:
    if d.level == "normal" and d.n_alerts == 0:
        st.success("本次记录未见需要即刻关注的疑似异常心律。")
    elif d.level == "refer":
        st.warning(f"总体分级：建议就医复核。检出 {d.n_alerts} 处需要提示的疑似心律变化，建议携带完整记录就医复核。")
    else:
        st.warning(f"总体分级：关注。检出 {d.n_alerts} 处需要提示的疑似心律变化，建议携带完整记录就医复核。")

    if res.short_record:
        s = res.short_record
        st.divider()
        st.markdown("**短记录逐拍摘要**（记录较短，补充 30 秒窗筛查未覆盖的信息）")
        if s.get("too_few_beats"):
            st.info("心拍过少，无法做逐拍摘要。")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("总心拍数", s["n_beats"])
            c2.metric("平均心率", f"{s['hr_mean']:.0f} bpm")
            c3.metric("疑似早搏", s["suspected_ectopy"])
            c4.metric("宽 QRS", s["suspected_wide_qrs"])
            st.caption(f"心率范围 {s['hr_min']:.0f}–{s['hr_max']:.0f} bpm ｜ "
                       f"节律{'整齐' if s['regular'] else '不齐'}（RR CV {s['rr_cv']:.3f}）")
            if s["beat_details"]:
                beat_df = pd.DataFrame([
                    {"时间(s)": b["time_s"],
                     "前RR(s)": b["rr_prev_s"] if b["rr_prev_s"] is not None else "-",
                     "后RR(s)": b["rr_next_s"] if b["rr_next_s"] is not None else "-",
                     "QRS宽(ms)": round(b["width_s"] * 1000) if b["width_s"] else 0,
                     "形态相关": b["morph_corr"],
                     "标记": {"normal": "正常", "suspected_ectopy": "疑似早搏"}.get(b["flag"], b["flag"])}
                    for b in s["beat_details"]
                ])
                st.dataframe(beat_df, use_container_width=True, hide_index=True, key="beat_details_df")
        if res.db_label:
            st.info(f"**数据库参考标注**：{res.db_label}")
        st.divider()

    st.markdown("**上报事件**")
    alerts = [e for e in d.events if e.severity in ("caution", "refer")]
    alerts_df = pd.DataFrame(
        [{"类型": R.KIND_CN.get(e.kind, e.kind), "时间(分)": round(e.start / 60, 1),
          "分级": R.LEVEL_CN.get(e.severity, e.severity),
          "置信度": round(e.confidence, 2), "依据": e.detail} for e in alerts],
        columns=["类型", "时间(分)", "分级", "置信度", "依据"]
    )
    st.dataframe(alerts_df, use_container_width=True, hide_index=True, key="alerts_df")

    infos = [e for e in d.events if e.severity not in ("caution", "refer")]
    if infos:
        from collections import Counter

        cnt = Counter(e.kind for e in infos)
        st.markdown("**仅计数提示（不上报为报警）**：" +
                    "、".join(f"{R.KIND_CN.get(k, k)} {v} 次" for k, v in cnt.items()))
    else:
        st.markdown("**仅计数提示（不上报为报警）**：（无）")

    with st.expander(f"被防误报策略压掉的候选（{len(d.suppressed)} 个，可审计）", expanded=False):
        supp_df = pd.DataFrame(
            [{"类型": R.KIND_CN.get(e.kind, e.kind), "时间(分)": round(e.start / 60, 1),
              "压掉原因": why} for e, why in d.suppressed[:200]],
            columns=["类型", "时间(分)", "压掉原因"]
        )
        st.dataframe(supp_df, use_container_width=True, hide_index=True, key="suppressed_df")

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

# ================= 模拟实时心电监测 =================
import numpy as np
from scipy.signal import resample_poly
from fractions import Fraction

# 实时监测分析窗：必须长于 AF 滑窗（30 s）+ 一个完整 RR 周期，
# 否则 screen_af_windows 的 "t + win <= 最后一个 R 波" 一个窗都取不到，
# 房颤永远不会被评估（这是原来"跑起来但不出结论"的直接原因）。
LIVE_WINDOW_S = 60.0


@st.fragment(run_every=1.0)
def show_live_monitor():
    st.divider()
    st.subheader("📡 模拟实时心电监测")
    st.caption(
        "已有数据模拟设备输入：每次接收1秒，"
        f"每5秒分析最近 {LIVE_WINDOW_S:.0f} 秒。仅作原型演示，非心梗诊断。"
        "实时模式只做单窗判定；跨窗持续确认请看上方离线分析。"
    )

    # 独立保存监测状态，不使用上方整段分析的结果。
    if "_live_monitor" not in st.session_state:
        st.session_state["_live_monitor"] = {
            "raw": None,
            "fs": None,
            "cursor": 0,
            "running": False,
            "last_analysis": 0,
            "hr": None,
            "coverage": None,
            "events": [],
            "seen": set(),
            "message": "请先点击“载入当前数据”。",
            "error": "",
            "cost": None,
            "label": "",
        }

    m = st.session_state["_live_monitor"]
    b1, b2, b3, b4 = st.columns(4)

    load_clicked = b1.button(
        "① 载入当前数据", key="live_load"
    )
    start_clicked = b2.button(
        "▶ 开始 / 继续", key="live_start"
    )
    pause_clicked = b3.button(
        "⏸ 暂停", key="live_pause"
    )
    reset_clicked = b4.button(
        "↺ 重新开始", key="live_reset"
    )

    if load_clicked:
        m["running"] = False
        try:
            if not source:
                raise ValueError("请先在左侧选择数据源。")

            # 文件上传分支单独读取，避免原项目将采样率固定为250。
            if source.startswith("file:"):
                raw = np.load(
                    source.split(":", 1)[1],
                    allow_pickle=False,
                )
                fs = float(
                    st.session_state.get("_upload_fs", 250)
                )
                a = int(float(start) * fs)
                b = (
                    a + int(float(minutes) * 60 * fs)
                    if minutes > 0 else len(raw)
                )
                raw = raw[a:b]
            else:
                raw, fs = pipeline.load_source(
                    source,
                    minutes=float(minutes) if minutes > 0 else None,
                    start_s=float(start),
                )

            raw = np.asarray(raw, dtype=float)
            if raw.ndim != 1:
                raise ValueError("当前版本需要单导联的一维数据。")
            if not np.isfinite(fs) or fs <= 0:
                raise ValueError("采样率无效。")
            if len(raw) < int(30 * fs):
                raise ValueError("请至少选择30秒数据。")
            if not np.all(np.isfinite(raw)):
                raise ValueError("数据包含缺失值或无穷值，请先清理。")

            m.update({
                "raw": raw,
                "fs": float(fs),
                "cursor": 0,
                "running": False,
                "last_analysis": 0,
                "hr": None,
                "coverage": None,
                "events": [],
                "seen": set(),
                "message": "数据已就绪，点击“开始 / 继续”。",
                "error": "",
                "cost": None,
                "label": str(source),
                "layers": layers,
                "offset": float(start),
            })
        except Exception as exc:
            m["raw"] = None
            m["error"] = f"载入失败：{exc}"

    if pause_clicked:
        m["running"] = False

    if reset_clicked and m["raw"] is not None:
        m.update({
            "cursor": 0,
            "running": False,
            "last_analysis": 0,
            "hr": None,
            "coverage": None,
            "events": [],
            "seen": set(),
            "message": "已归零，点击“开始 / 继续”。",
            "error": "",
            "cost": None,
        })

    if start_clicked:
        if m["raw"] is None:
            m["error"] = "请先点击“载入当前数据”。"
        elif m["cursor"] >= len(m["raw"]):
            m["message"] = "数据已结束，请先点击“重新开始”。"
        else:
            m["running"] = True
            m["error"] = ""

    # 按数据时间推进：一次仅交付1秒，不跳过数据。
    # 分析较慢时会延长墙钟时间，因此这是模拟流，不是硬实时系统。
    if m["running"] and m["raw"] is not None:
        fs = m["fs"]
        n = len(m["raw"])
        m["cursor"] = min(
            n, m["cursor"] + max(1, int(round(fs)))
        )
        cursor = m["cursor"]

        if cursor < int(round(LIVE_WINDOW_S * fs)):
            m["message"] = (
                f"积累数据中：{cursor / fs:.0f}/{LIVE_WINDOW_S:.0f} 秒；"
                "凑满一个分析窗后开始输出提示。"
            )
        else:
            due = (
                cursor - m["last_analysis"] >= int(round(5 * fs))
                or cursor == n
            )

            if due:
                # 只截取已收到的数据，最多最近 LIVE_WINDOW_S 秒。
                left = max(0, cursor - int(round(LIVE_WINDOW_S * fs)))
                received = m["raw"][left:cursor].copy()
                begin = time.perf_counter()

                try:
                    # 原分析管线的file输入默认250Hz，
                    # 因此先正确转换当前窗口的采样率。
                    ratio = Fraction(
                        250.0 / fs
                    ).limit_denominator(10000)
                    window = resample_poly(
                        received,
                        ratio.numerator,
                        ratio.denominator,
                    )

                    # 每个窗口使用独立临时文件，分析后自动清理。
                    with tempfile.TemporaryDirectory(
                        prefix="ecg_live_"
                    ) as tmp_dir:
                        path = os.path.join(tmp_dir, "window.npy")
                        np.save(path, window)

                        current = pipeline.analyze(
                            "file:" + path,
                            layers=m["layers"],
                            verbose=False,
                            profile=profile,
                            # 单窗场景：60 s 只够 1 个 30 s AF 窗，
                            # 跨窗二次确认（离线默认 2 窗）在这里做不到
                            screen_overrides={"af_min_windows": 1},
                        )

                    m["cost"] = time.perf_counter() - begin
                    m["last_analysis"] = cursor

                    hr = float(current.hrv.get("hr", np.nan))
                    m["hr"] = (
                        hr if np.isfinite(hr) and hr > 0 else None
                    )
                    m["coverage"] = float(current.decision.coverage)

                    # 记录本窗口内的提示事件。
                    # 检出时间与事件发生时间分别保存。
                    for event in current.decision.events:
                        if event.severity not in ("caution", "refer"):
                            continue

                        event_start = left / fs + float(event.start)
                        event_end = left / fs + float(event.end)

                        # 重叠滑窗可能重复检出同一事件。
                        duplicate = any(
                            old["_kind"] == event.kind
                            and event_start <= old["_end"] + 1.0
                            and event_end >= old["_start"] - 1.0
                            for old in m["events"]
                        )
                        if duplicate:
                            continue

                        m["events"].append({
                            "_kind": event.kind,
                            "_start": event_start,
                            "_end": event_end,
                            "类型": R.KIND_CN.get(
                                event.kind, event.kind
                            ),
                            "事件起点（秒）": round(event_start, 1),
                            "检出时刻（秒）": round(cursor / fs, 1),
                            "分级": R.LEVEL_CN.get(
                                event.severity, event.severity
                            ),
                            "依据": event.detail,
                        })

                    m["message"] = (
                        f"最近一次分析覆盖 "
                        f"{left / fs:.0f}–{cursor / fs:.0f} 秒。"
                    )
                    m["error"] = ""

                except Exception as exc:
                    m["running"] = False
                    m["hr"] = None
                    m["coverage"] = None
                    m["error"] = f"分析失败，已暂停：{exc}"

        if cursor >= n:
            m["running"] = False
            m["message"] += " 数据已全部接收。"

    if m["error"]:
        st.error(m["error"])

    if m["raw"] is None:
        st.info(m["message"])
        return

    fs = m["fs"]
    cursor = m["cursor"]
    elapsed = cursor / fs
    total = len(m["raw"]) / fs

    status = (
        "运行中" if m["running"]
        else "已结束" if cursor >= len(m["raw"])
        else "暂停 / 待开始"
    )
    st.write(
        f"状态：{status} ｜ 已接收 {elapsed:.1f}/{total:.1f} 秒"
    )
    st.caption(
        f"已载入来源：{m['label']} ｜ "
        f"原记录起点：{m['offset']:.0f} 秒。"
        "下方时间均相对于该起点。"
    )
    st.progress(min(elapsed / total, 1.0))

    # 波形只使用已经交付给监测器的原始数据。
    plot_left = max(0, cursor - int(round(10 * fs)))
    if cursor - plot_left >= 2:
        frame = pd.DataFrame({
            "时间（秒）": np.arange(plot_left, cursor) / fs,
            "心电幅值": m["raw"][plot_left:cursor],
        })
        st.line_chart(
            frame,
            x="时间（秒）",
            y="心电幅值",
            height=300,
        )
    else:
        st.info("点击开始，接收并显示心电数据。")

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "最近分析窗平均心率",
        f"{m['hr']:.0f} bpm" if m["hr"] is not None else "等待分析",
    )
    c2.metric(
        "最近分析窗有效占比",
        f"{m['coverage'] * 100:.0f}%"
        if m["coverage"] is not None else "等待分析",
    )
    c3.metric("累计提示事件", len(m["events"]))

    st.info(m["message"])
    if m["cost"] is not None:
        st.caption(
            f"上次窗口分析耗时：{m['cost']:.2f}秒。"
            "分析期间输入暂缓，不丢弃样本；"
            "运行速度可能慢于真实时间。"
        )

    st.markdown("**随数据接收生成的事件列表**")
    if m["events"]:
        visible = [
            {k: v for k, v in row.items() if not k.startswith("_")}
            for row in m["events"]
        ]
        st.dataframe(
            pd.DataFrame(visible),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.write("暂无提示事件；不表示已排除疾病。")

    st.caption(
        "本版逐窗调用原有离线算法：每次只保留最近 "
        f"{LIVE_WINDOW_S:.0f} 秒，房颤按单窗判定；"
        "离线版的跨窗二次确认在这里不生效，可靠性低于离线分析。"
    )


show_live_monitor()