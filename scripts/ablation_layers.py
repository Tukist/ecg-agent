"""五层防误报策略消融实验（课程重点的定量证据）。

两条证据线：

  A. 真实数据（mitdb）：
     · 报警率：只统计落在"窦律时段"（(N 节律段）内的可行动报警，折算成次/小时
     · 检出率：房颤记录中，AFIB/AFL 时段内是否被正确提示（事件级）

  B. 噪声压力测试（模拟可穿戴场景）：
     · 在干净记录上按已知时间位置注入 4 类干扰（基线漂移/工频/肌电/运动伪差，
       见 algorithms/quality.py synth_noise_bank），复现"误报的技术本源"
     · 统计各配置在干扰段内产生的报警数（误报）与干扰段外是否新增报警

用法：
    python scripts/ablation_layers.py [--max-records 20]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import wfdb  # noqa: E402

from ecg_agent.algorithms import quality as Q  # noqa: E402
from ecg_agent.algorithms.preprocess import prepare_for_quality, preprocess  # noqa: E402
from ecg_agent.algorithms.qrs import pan_tompkins, xqrs  # noqa: E402
from ecg_agent.io import mitdb  # noqa: E402
from ecg_agent.layers import strategy as S  # noqa: E402


def rhythm_segments(rec: str, duration_s: float | None = None):
    ann = wfdb.rdann(os.path.join(mitdb.MITDB_DIR, rec), "atr")
    return mitdb.rhythm_timeline(ann, 360.0, duration_s=duration_s)


def overlap(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def run_presets(xc, fs, r1, r2, qw, presets):
    out = {}
    for name in presets:
        out[name] = S.run(xc, fs, r1, r2, cfg=S.preset(name), quality_windows=qw)
    return out


def part_a(recs: list[str], presets: list[str]):
    rows = []
    for rec in recs:
        t0 = time.time()
        x, fs, _ = mitdb.read(rec, channel=0)
        dur = len(x) / fs
        xc, fs = preprocess(x, fs, target_fs=250.0)
        x_q, fs_q = prepare_for_quality(x, fs, target_fs=250.0)
        r1 = pan_tompkins(xc, fs)
        r2 = xqrs(xc, fs)
        qw = Q.assess(x_q, fs_q, r_peaks=r1)
        segs = rhythm_segments(rec, duration_s=dur)
        normal = [(a, b) for a, b, k in segs if k == "normal"]
        normal_h = sum(b - a for a, b in normal) / 3600.0
        af_segs = [(a, b) for a, b, k in segs if k == "af"]
        af_h = sum(b - a for a, b in af_segs) / 3600.0
        decs = run_presets(xc, fs, r1, r2, qw, presets)
        for name, dec in decs.items():
            alerts_normal = 0
            af_hit = 0
            for e in dec.events:
                if e.severity not in ("caution", "refer"):
                    continue
                ov_n = sum(overlap(e.start, e.end, a, b) for a, b in normal)
                if ov_n > 0.5 * max(0.1, e.end - e.start):
                    alerts_normal += 1
                ov_af = sum(overlap(e.start, e.end, a, b) for a, b in af_segs)
                if e.kind == "af_suspect" and ov_af > 0.5 * max(0.1, e.end - e.start):
                    af_hit += 1
            rows.append({"record": rec, "preset": name, "normal_hours": round(normal_h, 3),
                         "af_hours": round(af_h, 3), "alerts_in_normal": alerts_normal,
                         "af_events_in_af": af_hit, "suppressed": len(dec.suppressed),
                         "coverage": round(dec.coverage, 3)})
        print(f"  [A] {rec} 完成（{time.time()-t0:.0f}s；窦律 {normal_h:.2f} h，房颤 {af_h:.2f} h）")
    return rows


def part_b(presets: list[str], rec: str = "100", duration_s: int = 1800, n_artifacts: int = 12):
    """噪声压力测试：在干净记录上分段注入干扰（默认 1800 s，12 段）。"""
    """噪声压力测试：在干净记录上分段注入干扰。"""
    x, fs, _ = mitdb.read(rec, channel=0)
    x = x[: duration_s * int(fs)]
    # 原始（仅去直流）信号：质量评估与噪声注入都在这一路上做
    x_raw, fs = prepare_for_quality(x, fs, target_fs=250.0)
    rng = np.random.default_rng(7)
    bank = Q.synth_noise_bank(fs, duration_s, seed=7)
    kinds = ["baseline", "mains", "emg", "motion"]
    # 均匀布点、随机挑干扰类型，每段 40–60 s
    step = duration_s // (n_artifacts + 1)
    spans = []
    x_noisy = x_raw.copy()
    # 强度按"能击穿 QRS 检测"标定：肌电/运动伪差是带内噪声（带通挡不住），
    # 基线/工频虽然被带通滤掉，但会被质量门控（在原始信号上评估）抓到
    # 强度按"能击穿 QRS 检测"标定（实测 ×6 以下太温和，检测器抗得住）
    scale = {"baseline": 4.0, "mains": 8.0, "emg": 12.0, "motion": 8.0}
    for i in range(n_artifacts):
        k = kinds[i % len(kinds)]
        t0 = step * (i + 1)
        ln = int(rng.uniform(40, 60) * fs)
        a = int(t0 * fs)
        b = min(len(x_noisy), a + ln)
        noise = bank[k][a:b] * scale[k]
        x_noisy[a:b] += noise[: b - a]
        spans.append((t0, b / fs, k))

    # 质量评估：在"注入后、未带通"的原始信号上做（基线/工频/肌电都还在）
    qw = Q.assess(x_noisy.copy(), fs, r_peaks=None)
    # 检测链路：走正式预处理（带通），与真实设备一致
    x_det, fs_det = preprocess(x_noisy, fs, target_fs=None)
    r1 = pan_tompkins(x_det, fs_det)
    r2 = xqrs(x_det, fs_det)
    # 质量门控对干扰段的识别情况
    for k in kinds:
        qs = [w["quality"] for w in qw
              if any(s[2] == k and w["t_end"] > s[0] and w["t_start"] < s[1] for s in spans)]
        if qs:
            print(f"  干扰类型 {k:>8s}: 干扰段平均质量分 {np.mean(qs):.2f}"
                  f"（<0.6 的比例 {np.mean(np.array(qs) < 0.6)*100:.0f}%）")

    rows = []
    for name in presets:
        dec = S.run(x_det, fs_det, r1, r2, cfg=S.preset(name), quality_windows=qw)
        ev_art = ev_clean = al_art = al_clean = 0
        for e in dec.events:
            ov = sum(overlap(e.start, e.end, a, b) for a, b, _ in spans)
            in_art = ov > 0.5 * max(0.1, e.end - e.start)
            if in_art:
                ev_art += 1
            else:
                ev_clean += 1
            if e.severity in ("caution", "refer"):
                if in_art:
                    al_art += 1
                else:
                    al_clean += 1
        rows.append({"preset": name,
                     "candidates": len(dec.events) + len(dec.suppressed),
                     "surviving": len(dec.events),
                     "surviving_in_artifact": ev_art, "surviving_in_clean": ev_clean,
                     "alerts_total": dec.n_alerts,
                     "alerts_in_artifact": al_art, "alerts_in_clean": al_clean,
                     "suppressed": len(dec.suppressed), "coverage": round(dec.coverage, 3)})
    return rows, spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-records", type=int, default=20)
    ap.add_argument("--only-b", action="store_true", help="只跑噪声压力测试（B 线，跳过 A 线）")
    ap.add_argument("--out-prefix", default="results/ablation")
    args = ap.parse_args()

    presets = list(S.PRESETS.keys())
    os.makedirs("results", exist_ok=True)
    rows = []
    if not args.only_b:
        recs = mitdb.record_ids()[: args.max_records]
        print(f"A. 真实数据消融：{len(recs)} 条记录 {recs}")
        # 保证房颤记录在集内
        af_recs = [r for r in mitdb.record_ids() if r in ("201", "202", "203", "210", "221", "222")]
        recs = sorted(set(recs) | set(af_recs), key=lambda r: int(r))
        rows = part_a(recs, presets)
        with open(f"{args.out_prefix}_layers.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

    print("\nB. 噪声压力测试（记录 100 + 注入 4 类干扰）")
    rows_b, spans = part_b(presets)

    with open(f"{args.out_prefix}_noise.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_b[0].keys()))
        w.writeheader(); w.writerows(rows_b)

    # —— 汇总 ——
    summary = []
    if rows:
        print(f"\nA. 各配置（窦律时段报警率 / 房颤时段检出率）")
        print(f"{'配置':>10} {'窦律报警/小时':>14} {'窦律报警总数':>12} {'房颤检出率':>12} {'压掉事件':>10}")
        for name in presets:
            rr = [r for r in rows if r["preset"] == name]
            hours = sum(r["normal_hours"] for r in rr) or 1
            alerts = sum(r["alerts_in_normal"] for r in rr)
            af_records = [r for r in rr if r["af_hours"] > 0.05]
            hit = sum(1 for r in af_records if r["af_events_in_af"] > 0)
            det = hit / len(af_records) if af_records else 0.0
            supp = sum(r["suppressed"] for r in rr)
            summary.append({"preset": name, "alerts_per_hour": alerts / hours, "alerts_total": alerts,
                            "af_detect_rate": det, "n_af_records": len(af_records), "suppressed": supp})
            print(f"{name:>10} {alerts/hours:>14.2f} {alerts:>12} {det*100:>11.1f}% {supp:>10}")
        with open(f"{args.out_prefix}_summary.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            w.writeheader(); w.writerows(summary)

    print(f"\nB. 噪声压力测试（{len(spans)} 段注入干扰）")
    print(f"{'配置':>10} {'候选事件':>8} {'存活':>6} {'干扰段内存活':>12} {'可行动报警':>10} {'被压掉':>8}")
    for r in rows_b:
        print(f"{r['preset']:>10} {r['candidates']:>8} {r['surviving']:>6} "
              f"{r['surviving_in_artifact']:>12} {r['alerts_total']:>10} {r['suppressed']:>8}")

    print("\n已写出", f"{args.out_prefix}_noise.csv",
          ("/ " + f"{args.out_prefix}_layers.csv + ' / ' + args.out_prefix + '_summary.csv'") if rows else "")


if __name__ == "__main__":
    main()
