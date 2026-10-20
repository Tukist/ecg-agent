"""LLM 解说插件（DeepSeek / OpenAI 兼容接口）。

定位：算法结果 → 大模型改写成"人话版"解读 + 回答追问。
**不参与任何算法决策**——判不判、报不报，全部由五层策略决定；LLM 只负责表达。

硬约束（写进 system prompt，防止模型越界）：
  1. 只能说"提示 / 疑似 / 建议就医复核"，禁止"诊断 / 确诊 / 预警 / 患病风险"
  2. 只能引用给定数据，不得编造数字；不确定就说不确定
  3. 不给用药/治疗建议
  4. 明确说明这不是诊断、需医生复核

API Key 读取顺序：环境变量 DEEPSEEK_API_KEY → 项目根目录的 .ds_key 文件（一行）。
没有 key 时所有函数抛 LLMUnavailable，调用方应回退到模板报告。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"

SYSTEM_PROMPT = """你是一个心电分析助手，负责把算法输出的结构化结果改写成普通人能看懂的解读。

硬约束（必须遵守）：
1. 只能使用"提示""疑似""可能""建议就医复核"这类措辞；禁止使用"诊断""确诊""预警""患病""存在XX病"这类表述。
2. 只能引用下面给出的数据，不得编造任何数字；数据里没有的信息，就说不确定或没测。
3. 不给用药建议、不做治疗指导。
4. 结尾必须写明：本结果由算法自动生成，仅供健康提示、不构成诊断，需医生复核。
5. 已知的限制（如算法做不到的事）要如实转述，不要美化。

写作要求：3 段以内，每段不超过 100 字，用中文，口语化但准确。"""


class LLMUnavailable(RuntimeError):
    """没有配置 API Key，或接口不可用。"""


def api_key() -> str | None:
    k = os.environ.get("DEEPSEEK_API_KEY")
    if k and k.strip():
        return k.strip()
    f = PROJECT_ROOT / ".ds_key"
    if f.exists():
        v = f.read_text(encoding="utf-8").strip()
        if v:
            return v
    return None


def build_context(res) -> str:
    """把 AnalysisResult 压缩成给大模型看的结构化摘要（省 token、防歧义）。"""
    d = res.decision
    h = res.hrv
    lines = []
    lines.append(f"【记录】来源 {res.source}，时长 {res.duration_s/60:.1f} 分钟，"
                 f"有效监测时长占比 {d.coverage*100:.0f}%（信号质量通过门控的窗口比例）")
    lines.append(f"【心率】平均 {h['hr']:.0f} bpm；SDNN {h['sdnn']*1000:.0f} ms；"
                 f"RMSSD {h['rmssd']*1000:.0f} ms；RR 变异系数 {h['cv_rr']:.3f}；"
                 f"样本熵 {h['sample_entropy']:.2f}")
    lines.append(f"【总体分级】{d.level}（normal=未见需即刻关注；caution=关注；refer=建议尽快就医复核）")
    # 只把"可行动"事件（关注/建议就医）逐条给模型；info 级（早搏计数）只给汇总，
    # 否则几百条早搏会把 prompt 塞满、还容易让模型抓错重点
    alerts = [e for e in d.events if e.severity in ("caution", "refer")]
    infos = [e for e in d.events if e.severity not in ("caution", "refer")]
    lines.append(f"【上报事件】可行动（关注/建议就医）{len(alerts)} 个；仅计数提示（info）{len(infos)} 个")
    for i, e in enumerate(alerts[:15], 1):
        lines.append(f"  {i}. {e.kind} 于 {e.start/60:.1f} min，分级 {e.severity}，"
                     f"置信度 {e.confidence:.2f}，依据：{e.detail}")
    if len(alerts) > 15:
        lines.append(f"  （另有 {len(alerts)-15} 条未逐条列出）")
    if infos:
        from collections import Counter

        cnt = Counter(e.kind for e in infos)
        kinds_cn = {"ectopy": "疑似早搏", "af_suspect": "疑似房颤样", "pause": "长间歇",
                    "tachy": "心动过速", "brady": "心动过缓"}
        desc = "、".join(f"{kinds_cn.get(k, k)} {v} 次" for k, v in cnt.items())
        lines.append(f"  info 级（只计数、不上报为报警）：{desc}")
    if d.suppressed:
        from collections import Counter

        cnt = Counter(e.kind for e, _ in d.suppressed)
        lines.append(f"【被防误报策略压掉的候选 {len(d.suppressed)} 个】"
                     f"按类型：{dict(cnt)}；示例原因：{d.suppressed[0][1]}")
    lines.append("【五层策略决策日志】")
    lines.extend("  " + s for s in d.log)
    lines.append("【算法已知限制（必须如实转述）】"
                 "① 室上性早搏基本检不出；② 房颤样筛查只是提示、不能区分房颤与频发房早；"
                 "③ 阈值来自公开数据集，未在自研设备上标定；④ 运动伪差下质量门控会偏宽松。")
    return "\n".join(lines)


def chat(messages: list[dict], api_key_override: str | None = None,
         base_url: str = DEFAULT_BASE_URL, model: str = DEFAULT_MODEL,
         timeout: int = 90) -> str:
    key = api_key_override or api_key()
    if not key:
        raise LLMUnavailable(
            "未找到 DeepSeek API Key。设置环境变量后重试：\n"
            "  cmd:        set DEEPSEEK_API_KEY=sk-xxxx\n"
            "  PowerShell: $env:DEEPSEEK_API_KEY=\"sk-xxxx\"\n"
            "  或把 key 写进项目根目录的 .ds_key 文件（一行）")

    r = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": model, "messages": messages, "temperature": 0.3, "stream": False},
        timeout=timeout,
    )
    if r.status_code != 200:
        raise LLMUnavailable(f"接口返回 {r.status_code}：{r.text[:300]}")
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()


def narrate(res) -> str:
    """把分析结果改写成"人话版"解读。"""
    return chat([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content":
            "下面是一次心电分析的完整结果，请改写成给受检者看的解读。\n\n" + build_context(res)},
    ])


def ask(res, question: str) -> str:
    """基于本次分析结果回答追问。"""
    return chat([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content":
            "以下是一次心电分析的结构化结果：\n\n" + build_context(res) +
            f"\n\n问题：{question}\n\n请基于上面的数据回答；数据里没有的信息明说不确定。"},
    ])


def dry_run_payload(res, question: str | None = None) -> dict:
    """不调接口，只返回将要发送的 messages（用于自检/演示讲解）。"""
    user = (build_context(res) if question is None else
            build_context(res) + f"\n\n问题：{question}")
    return {"model": DEFAULT_MODEL,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": user}]}


__all__ = ["LLMUnavailable", "api_key", "build_context", "chat", "narrate", "ask", "dry_run_payload"]
