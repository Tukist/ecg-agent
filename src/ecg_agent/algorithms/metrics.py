"""评价指标：心拍配对（ANSI/AAMI EC57 口径，容差 ±150 ms）与分类指标。"""

from __future__ import annotations

import bisect

import numpy as np


def match_beats(det: np.ndarray, ref: np.ndarray, fs: float, tol: float = 0.15):
    """把检测心拍与参考标注配对。

    规则：每个参考拍最多配一个检测拍，|Δt| ≤ tol；一个检测拍也最多配一个参考。
    先贪心匹配最近邻，再统计 TP/FP/FN。
    """
    det = np.sort(np.asarray(det, dtype=np.int64))
    ref = np.sort(np.asarray(ref, dtype=np.int64))
    tol_s = tol
    ref_s = ref / fs
    used_ref = np.zeros(len(ref), dtype=bool)
    tp = 0
    for d in det / fs:
        k = bisect.bisect_left(ref_s.tolist(), d)
        best, best_dt = -1, None
        for j in (k - 1, k, k + 1):
            if 0 <= j < len(ref_s) and not used_ref[j]:
                dt = abs(ref_s[j] - d)
                if dt <= tol_s and (best_dt is None or dt < best_dt):
                    best, best_dt = j, dt
        if best >= 0:
            used_ref[best] = True
            tp += 1
    fp = len(det) - tp
    fn = int((~used_ref).sum())
    return tp, fp, fn


def beat_metrics(det: np.ndarray, ref: np.ndarray, fs: float, tol: float = 0.15) -> dict:
    tp, fp, fn = match_beats(det, ref, fs, tol)
    se = tp / (tp + fn) if tp + fn else 0.0
    ppv = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * se * ppv / (se + ppv) if se + ppv else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "se": se, "ppv": ppv, "f1": f1,
            "n_ref": len(ref), "n_det": len(det)}


AAMI_MAP = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",  # 正常/束支阻滞
    "A": "S", "a": "S", "J": "S", "S": "S",            # 室上性
    "V": "V", "E": "V",                                 # 室性
    "F": "F",                                           # 融合
    "/": "Q", "f": "Q", "Q": "Q",                       # 未分类/起搏
}
