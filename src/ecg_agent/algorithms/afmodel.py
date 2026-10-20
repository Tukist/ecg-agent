"""房颤筛查模型的加载与打分（RR 域逻辑回归）。

模型由 scripts/train_af_model.py 训练并保存（按记录分组交叉验证）。
若模型文件不存在，调用方应回退到 rules 版（rhythm.af_score_rules）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "af_logreg.joblib"

_CACHE: dict = {}


def load(path: str | Path | None = None):
    """加载模型（带缓存）。返回 dict(pipeline, features, cv_thr) 或 None。"""
    p = Path(path) if path else DEFAULT_MODEL
    key = str(p)
    if key in _CACHE:
        return _CACHE[key]
    if not p.exists():
        _CACHE[key] = None
        return None
    try:
        import joblib

        _CACHE[key] = joblib.load(p)
    except Exception:
        _CACHE[key] = None
    return _CACHE[key]


def score(feats: dict, path: str | Path | None = None) -> float | None:
    """给单个窗口特征打房颤概率；模型不可用时返回 None。"""
    m = load(path)
    if not m:
        return None
    x = np.array([[float(feats[f]) for f in m["features"]]])
    return float(m["pipeline"].predict_proba(x)[0, 1])
