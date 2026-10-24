"""读取外部数据库自带的诊断标签，仅作为「数据库参考标注」展示，不参与算法判断。

支持：
  · PTB-XL（ptbxl_database.csv + scp_statements.csv）
  · 中文 12 导联心律失常库（ConditionNames_SNOMED-CT.csv + .hea 里的 #Dx）
"""

from __future__ import annotations

import ast
import csv
import os
import re
from pathlib import Path


# ------------------------------------------------------------------ PTB-XL

def _load_ptbxl_index(csv_path: str) -> dict[str, dict]:
    """建立 filename_lr -> {ecg_id, scp_codes} 索引。"""
    index: dict[str, dict] = {}
    if not os.path.exists(csv_path):
        return index
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fn = row.get("filename_lr", "")
            if not fn:
                continue
            scp_raw = row.get("scp_codes", "")
            try:
                scp = ast.literal_eval(scp_raw) if scp_raw else {}
            except Exception:
                scp = {}
            index[fn] = {
                "ecg_id": row.get("ecg_id", ""),
                "scp_codes": scp if isinstance(scp, dict) else {},
            }
    return index


def _load_scp_descriptions(csv_path: str) -> dict[str, str]:
    """SCP code -> 描述。"""
    out: dict[str, str] = {}
    if not os.path.exists(csv_path):
        return out
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = row.get("", "") or row.get("statement", "")
            desc = row.get("description", "") or row.get("SCP-ECG Statement Description", "")
            if code:
                out[code] = desc
    return out


def ptbxl_label(record_path: str) -> str | None:
    """根据 PTB-XL 记录路径查找参考标签。

    record_path 可以带或不带 _lr/_hr 后缀；函数会优先按传入后缀匹配，
    未匹配则尝试补 _lr 再查。
    """
    p = Path(record_path)
    root = p.parents[2] if len(p.parts) >= 3 else p.parent
    db_csv = root / "ptbxl_database.csv"
    stmt_csv = root / "scp_statements.csv"
    if not db_csv.exists():
        return None

    index = _load_ptbxl_index(str(db_csv))

    # 把绝对路径转成 csv 里的相对路径
    rel = "/".join(p.relative_to(root).with_suffix("").parts)
    for candidate in (rel, rel + "_lr", rel + "_hr"):
        info = index.get(candidate)
        if info:
            break
    else:
        return None

    scp = info.get("scp_codes", {})
    if not scp:
        return None

    descs = _load_scp_descriptions(str(stmt_csv)) if stmt_csv.exists() else {}
    parts = []
    for code, conf in scp.items():
        d = descs.get(code, code)
        parts.append(f"{d} ({conf:.0f}%)")
    return "；".join(parts) if parts else None


# ------------------------------------------------------------------ 中文 12 导联

def _load_snomed_map(csv_path: str) -> dict[str, str]:
    """SNOMED CT code -> Full Name。"""
    out: dict[str, str] = {}
    if not os.path.exists(csv_path):
        return out
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = row.get("Snomed_CT", "").strip()
            name = row.get("Full Name", "").strip()
            if code and name:
                out[code] = name
    return out


def _read_dx_from_hea(hea_path: str) -> list[str]:
    """从 .hea 文件里读取 #Dx: 行。"""
    codes = []
    if not os.path.exists(hea_path):
        return codes
    with open(hea_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("#Dx:"):
                rest = line.replace("#Dx:", "").strip()
                codes = [c.strip() for c in rest.split(",") if c.strip()]
                break
    return codes


def chinese12_label(record_path: str) -> str | None:
    """根据中文 12 导联记录路径查找参考标签。"""
    p = Path(record_path)
    root = p.parents[3] if len(p.parts) >= 4 else p.parent
    csv_path = root / "ConditionNames_SNOMED-CT.csv"
    if not csv_path.exists():
        return None

    hea_path = str(p.with_suffix(".hea"))
    codes = _read_dx_from_hea(hea_path)
    if not codes:
        return None

    mapping = _load_snomed_map(str(csv_path))
    names = [mapping.get(c, c) for c in codes]
    return "；".join(names) if names else None


# ------------------------------------------------------------------ generic

def lookup(source: str) -> str | None:
    """根据 source 字符串自动判断数据库并返回参考标签。"""
    if source.startswith("wfdb:"):
        body = source[len("wfdb:"):]
        rec_path, _, _ = body.rpartition("|")
        if not rec_path:
            return None
        lowered = rec_path.lower()
        if "ptb-xl" in lowered:
            return ptbxl_label(rec_path)
        if "a-large-scale-12-lead" in lowered or "12-lead-electrocardiogram" in lowered:
            return chinese12_label(rec_path)
    return None
