"""MIT-BIH 心律失常数据库（mitdb）读取与标注解析。

数据结构：
  · 48 条记录，30 分钟，360 Hz，2 通道，逐拍人工标注（.atr）
  · 节律标注：符号 '+' / '~' 的 aux_note 记录节律变化，如
      '(N' 正常窦性、'(AFIB' 房颤、'(AFL' 房扑、'(VT' 室速、'(B' 二联律 …… 
    aux_note 常带尾部 \\x00 或被截断（'(AFIB' / '(AFL'），解析时统一清洗。

逐拍标注符号 → AAMI 五类的映射见 algorithms/metrics.py。
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# 项目根目录（src/ecg_agent/io/mitdb.py → 上 3 级），保证在任何工作目录下都能找到数据
PROJECT_ROOT = Path(__file__).resolve().parents[3]
MITDB_DIR = str(PROJECT_ROOT / "data" / "mitdb" / "mit-bih-arrhythmia-database-1.0.0")
BEAT_SYMS = set("NLRBAaJSVrFejnE/fQ?")
# AAMI 评价排除的记录（含起搏节律，EC57 规定不作心律分类评价）
AAMI_EXCLUDE = {"102", "104", "107", "217"}

RHYTHM_ALIASES = {
    "N": "normal", "NOD": "normal",
    "AFIB": "af", "AFL": "af",
    "B": "other", "BII": "other", "T": "other", "P": "other", "PREX": "other",
    "SVTA": "other", "VT": "other", "VFL": "other", "IVR": "other", "AB": "other",
    "SBR": "other", "NSR": "normal",
}


def record_ids(with_atr_only: bool = True) -> list[str]:
    out = []
    for name in sorted(os.listdir(MITDB_DIR)):
        if not name.endswith(".hea"):
            continue
        rec = name[:-4]
        if with_atr_only and not os.path.exists(os.path.join(MITDB_DIR, rec + ".atr")):
            continue
        out.append(rec)
    return out


def read(rec: str, channel: int = 0, physical: bool = True):
    """返回 (signal, fs, ann)。signal 为单通道。"""
    import wfdb

    r = wfdb.rdrecord(os.path.join(MITDB_DIR, rec), physical=physical)
    ann = wfdb.rdann(os.path.join(MITDB_DIR, rec), "atr")
    return r.p_signal[:, channel], float(r.fs), ann


def beat_annotations(ann) -> tuple[np.ndarray, list[str]]:
    """逐拍标注（样本下标, 符号）。"""
    syms = list(ann.symbol)
    mask = np.array([s in BEAT_SYMS for s in syms])
    return np.asarray(ann.sample)[mask], [s for s in syms if s in BEAT_SYMS]


def _clean_aux(s: str) -> str:
    s = s.replace("\x00", "").strip()
    s = s.lstrip("([")
    # 有些标注被截断成 '(AFL'、'(SVTA' 等，取字母前缀匹配即可
    return s


def rhythm_timeline(ann, fs: float, duration_s: float | None = None) -> list[tuple[float, float, str]]:
    """节律段列表 [(t_start_s, t_end_s, kind)]，kind ∈ {normal, af, other}。

    解析规则：符号 '+' / '~' 的标注表示该时刻起节律切换，直到下一次切换为止。
    duration_s：记录总时长（秒）——不传则末段以 inf 结尾（统计时长时会出问题，
    调用方务必传真实时长）。
    """
    segs = []
    cur_kind, cur_start = None, 0.0
    for samp, s, aux in zip(ann.sample, ann.symbol, ann.aux_note):
        if s not in ("+", "~"):
            continue
        name = _clean_aux(aux)
        if not name:
            continue
        key = None
        for k in RHYTHM_ALIASES:
            if name.upper().startswith(k) and len(k) >= len(key or ""):
                key = k
        if key is None:
            continue
        kind = RHYTHM_ALIASES[key]
        t = float(samp) / fs
        if cur_kind is not None and t > cur_start:
            segs.append((cur_start, t, cur_kind))
        cur_kind, cur_start = kind, t
    if cur_kind is not None:
        end = float(duration_s) if duration_s is not None else np.inf
        if end > cur_start:
            segs.append((cur_start, end, cur_kind))
    return segs


def label_window(segs: list[tuple[float, float, str]], t0: float, t1: float) -> str | None:
    """判断窗口完全落在哪种节律段内；跨段则返回 None（不用于评价）。"""
    kinds = set()
    for a, b, k in segs:
        if b <= t0 or a >= t1:
            continue
        # 段必须完整覆盖窗口
        if a <= t0 + 1e-6 and b >= t1 - 1e-6:
            return k
        kinds.add(k)
    return None if kinds else None
