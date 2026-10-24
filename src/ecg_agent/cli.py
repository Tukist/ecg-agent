"""命令行入口：ecg-agent

用法示例：
    # 基础分析（纯本地，不联网）
    python -m ecg_agent.cli analyze --source ppg-dalia:S1 --minutes 30
    python -m ecg_agent.cli analyze --source mitdb:203 --start 600 --minutes 30 --layers 裸阈值

    # 带大模型解读（联网调 DeepSeek）
    python -m ecg_agent.cli analyze --source mitdb:114 --minutes 30 --llm

    # 追问（智能体问答：基于本次分析结果回答）
    python -m ecg_agent.cli ask --source mitdb:114 --minutes 30 --question "为什么报了6个事件？"
    python -m ecg_agent.cli ask --source ppg-dalia:S1 --minutes 30        # 不带问题=进入交互问答
"""

from __future__ import annotations

import argparse
import sys

from .agent import pipeline
from .agent import report as R
from .layers import strategy as S
from .subject import SubjectProfile


def _print_result(res):
    print("\n===== 智能体决策日志 =====")
    for line in res.log:
        print(" ", line)
    print("\n===== 结论（提示级，非诊断）=====")
    d = res.decision
    print(f"  受检者：{d.population.get('summary', '未提供')} | "
          f"人群阈值：{d.population.get('threshold_text', '-')}")
    print(f"  分级：{R.LEVEL_CN.get(d.level, d.level)} | 上报事件 {d.n_alerts} 个 | "
          f"被压掉候选 {len(d.suppressed)} 个 | 有效监测占比 {d.coverage*100:.1f}%")
    alerts = [e for e in d.events if e.severity in ("caution", "refer")]
    infos = [e for e in d.events if e.severity not in ("caution", "refer")]
    for e in alerts:
        print(f"  · [{e.severity}] {R.KIND_CN.get(e.kind, e.kind)} "
              f"@ {e.start/60:.1f} min（{e.detail}）")
    if not alerts:
        print("  · （无需要关注的疑似事件）")
    if infos:
        from collections import Counter

        cnt = Counter(e.kind for e in infos)
        desc = "、".join(f"{R.KIND_CN.get(k, k)} {v} 次" for k, v in cnt.items())
        print(f"  · [info] 仅计数提示（不上报为报警）：{desc}")


def _profile(args):
    """命令行参数 → 受检者档案（全部可选）。"""
    return SubjectProfile(age=args.age, sex=args.sex, bmi=args.bmi)


def _add_subject_args(p):
    p.add_argument("--age", type=float, default=None,
                   help="受检者年龄（岁，>0；不填按成人默认阈值）")
    p.add_argument("--sex", default=None, choices=["男", "女", "male", "female"],
                   help="受检者性别（可选；女性过速上限 +2 bpm，弱证据修正）")
    p.add_argument("--bmi", type=float, default=None,
                   help="BMI（kg/m²，可选；>25 时过速上限每 5 单位 +1 bpm，上限 +4）")


def cmd_analyze(args):
    res = pipeline.analyze(args.source, minutes=args.minutes, start_s=args.start,
                           layers=args.layers, profile=_profile(args))
    _print_result(res)

    llm_text = None
    if args.llm:
        from .agent import llm

        print("\n===== AI 解读（DeepSeek 生成，仅供参考）=====")
        try:
            llm_text = llm.narrate(res)
            print(llm_text)
            print("\n（本段由大模型改写，不参与算法判断，仅供健康提示、不构成诊断）")
        except llm.LLMUnavailable as e:
            print(f"  [跳过] {e}")
        except Exception as e:  # 网络等异常不让主流程挂掉
            print(f"  [跳过] 大模型调用失败：{type(e).__name__}: {e}")

    if args.out:
        md, fig = R.render_all(res, args.out, stem=args.stem, llm_text=llm_text)
        print(f"\n报告已写出：\n  {md}\n  {fig}")
    return 0


def cmd_ask(args):
    from .agent import llm

    res = pipeline.analyze(args.source, minutes=args.minutes, start_s=args.start,
                           layers=args.layers, verbose=False, profile=_profile(args))
    _print_result(res)

    if not llm.api_key():
        print("\n[错误] 未配置 DeepSeek API Key：" + str(llm.LLMUnavailable("")))
        print("      设置后重试：cmd 里执行  set DEEPSEEK_API_KEY=sk-xxxx")
        return 2

    def answer(q: str):
        print("\n" + "-" * 60)
        try:
            print(llm.ask(res, q))
        except Exception as e:
            print(f"[调用失败] {type(e).__name__}: {e}")

    if args.question:
        answer(args.question)
        return 0

    print("\n进入问答模式（基于本次分析结果回答；输入 q 退出）")
    while True:
        try:
            q = input("\n问> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q or q.lower() in ("q", "quit", "exit", "退出"):
            break
        answer(q)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ecg-agent", description="心电分析智能体（提示级，非诊断）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="分析一段心电记录并出报告")
    a.add_argument("--source", required=True,
                   help="ppg-dalia:S1 / mitdb:100 / file:path.npy")
    a.add_argument("--minutes", type=float, default=None, help="分析前 N 分钟（默认全部）")
    a.add_argument("--start", type=float, default=0.0, help="起始秒（配合 --minutes 定位片段）")
    _add_subject_args(a)
    a.add_argument("--layers", default="完整五层", choices=list(S.PRESETS.keys()),
                   help="五层策略配置（用于消融演示）")
    a.add_argument("--out", default="reports", help="报告输出目录（留空=不出文件）")
    a.add_argument("--stem", default="report", help="报告文件名前缀")
    a.add_argument("--llm", action="store_true", help="追加 DeepSeek 大模型解读（需 API Key）")
    a.set_defaults(func=cmd_analyze)

    b = sub.add_parser("ask", help="基于分析结果追问（智能体问答，需 API Key）")
    b.add_argument("--source", required=True)
    b.add_argument("--minutes", type=float, default=None)
    b.add_argument("--start", type=float, default=0.0)
    _add_subject_args(b)
    b.add_argument("--layers", default="完整五层", choices=list(S.PRESETS.keys()))
    b.add_argument("--question", "-q", default=None, help="问题；不带则进入交互问答")
    b.set_defaults(func=cmd_ask)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
