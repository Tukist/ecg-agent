"""本地外部 WFDB 数据库的交互式选择支持。

提供统一接口：
  · 列出已发现的数据库
  · 列出某个数据库下的所有记录
  · 列出某条记录的所有导联/通道
  · 把（数据库名, 记录ID, 通道）转成 pipeline 可识别的 wfdb: 来源字符串

支持的数据库：
  · PTB-XL（records100/records500）
  · 中文 12 导联心律失常库（CPSC2018）
  · European ST-T
  · BUT-QDB

默认扫描位置：D:/data box（Windows 常见下载路径）。
"""

from __future__ import annotations

import os
from pathlib import Path

import wfdb


# 默认数据库根目录，可扩展为列表按顺序查找
DEFAULT_ROOTS = [
    Path("D:/data box"),
    Path("D:/date box"),
    Path("C:/data"),
]


# 数据库元数据：文件夹名、RECORDS 文件相对路径、记录发现策略
_DB_META = {
    "PTB-XL": {
        "folder": "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3",
        "records_file": "RECORDS",
        "record_filter": lambda r: r.endswith("_lr") and not r.startswith("records500"),
    },
    "中文 12 导联": {
        "folder": "a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0",
        "records_file": "RECORDS",
        "record_filter": lambda r: r.startswith("WFDBRecords/") and r.endswith(".hea") is False,
    },
    "European ST-T": {
        "folder": "european-st-t-database-1.0.0",
        "records_file": "RECORDS",
        "record_filter": None,
    },
    "BUT-QDB": {
        "folder": "brno-university-of-technology-ecg-quality-database-but-qdb-1.0.0",
        "records_file": "RECORDS",
        "record_filter": lambda r: r.endswith("_ECG"),
    },
}


def _find_db_root(folder_name: str) -> Path | None:
    """在默认根目录中查找指定数据库文件夹。"""
    for root in DEFAULT_ROOTS:
        candidate = root / folder_name
        if candidate.exists():
            return candidate
    return None


def _all_records_from_scan(db_root: Path, meta: dict) -> list[str]:
    """当 RECORDS 文件缺失或格式不对时，退回文件系统扫描。"""
    records = []
    folder = db_root.name.lower()
    if "ptb-xl" in folder:
        for hea in sorted(db_root.rglob("records100/*/*.hea")):
            rel = hea.relative_to(db_root).with_suffix("").as_posix()
            records.append(rel)
    elif "chinese" in folder or "12-lead" in folder or "large-scale" in folder:
        for hea in sorted(db_root.rglob("WFDBRecords/*/*/*.hea")):
            rel = hea.relative_to(db_root).with_suffix("").as_posix()
            records.append(rel)
    elif "european" in folder or "st-t" in folder:
        for hea in sorted(db_root.glob("*.hea")):
            records.append(hea.stem)
    elif "but-qdb" in folder or "quality" in folder:
        for hea in sorted(db_root.rglob("*_ECG.hea")):
            rel = hea.relative_to(db_root).with_suffix("").as_posix()
            records.append(rel)
    return records


def list_databases() -> list[str]:
    """返回本机实际存在的数据库名称列表。"""
    return [name for name, meta in _DB_META.items() if _find_db_root(meta["folder"]) is not None]


def list_records(db_name: str) -> list[str]:
    """返回数据库下所有可用记录的显示 ID。"""
    if db_name not in _DB_META:
        raise ValueError(f"未知数据库：{db_name}")

    meta = _DB_META[db_name]
    db_root = _find_db_root(meta["folder"])
    if db_root is None:
        return []

    records_file = db_root / meta["records_file"]
    if records_file.exists():
        try:
            with open(records_file, "r", encoding="utf-8") as f:
                raw = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        except Exception:
            raw = []
    else:
        raw = []

    if not raw:
        raw = _all_records_from_scan(db_root, meta)

    rec_filter = meta.get("record_filter")
    if rec_filter:
        raw = [r for r in raw if rec_filter(r)]

    # 中文 12 导联 RECORDS 里写的是目录，需要展开成目录下的实际记录文件名
    if db_name == "中文 12 导联":
        expanded = []
        for r in raw:
            dir_path = db_root / r
            if dir_path.is_dir():
                for hea in sorted(dir_path.glob("*.hea")):
                    rel = hea.relative_to(db_root).with_suffix("").as_posix()
                    expanded.append(rel)
            else:
                expanded.append(r)
        raw = expanded

    # 去重并保持稳定顺序
    seen = set()
    out = []
    for r in raw:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _record_path(db_name: str, record_id: str) -> Path:
    """根据数据库名和记录显示 ID，返回记录的绝对路径前缀（不含扩展名）。"""
    meta = _DB_META[db_name]
    db_root = _find_db_root(meta["folder"])
    if db_root is None:
        raise FileNotFoundError(f"找不到数据库根目录：{db_name}")

    candidate = db_root / record_id
    # 中文 12 导联早期 RECORDS 文件写的是目录；list_records 已展开为完整记录路径
    if db_name == "中文 12 导联" and not candidate.with_suffix(".hea").exists():
        # 兼容旧式目录记录 ID：扫描目录下 .hea
        heas = sorted(candidate.glob("*.hea"))
        if not heas:
            raise FileNotFoundError(f"目录下无记录：{candidate}")
        return heas[0].with_suffix("")

    return candidate.resolve()


def list_channels(db_name: str, record_id: str) -> list[str]:
    """返回某条记录的所有通道/导联名。"""
    rec_path = _record_path(db_name, record_id)
    try:
        rec = wfdb.rdrecord(str(rec_path), physical=True)
        sig_names = [str(n) for n in rec.sig_name]
        # 去重，保留顺序
        seen = set()
        out = []
        for name in sig_names:
            if name not in seen:
                seen.add(name)
                out.append(name)
        return out
    except Exception as e:
        raise RuntimeError(f"读取记录 {rec_path} 的导联信息失败：{e}")


def build_source(db_name: str, record_id: str, channel: str) -> str:
    """生成 pipeline 可识别的 wfdb: 来源字符串。"""
    rec_path = _record_path(db_name, record_id)
    # 统一用正斜杠，wfdb 在 Windows 下也能正确解析
    path_str = rec_path.as_posix()
    return f"wfdb:{path_str}|{channel}"


def guess_default_channel(channels: list[str]) -> str:
    """根据常见惯例推荐默认导联。"""
    preferred = ["II", "MLIII", "V1", "V2", "V5", "ECG", "lead II"]
    for p in preferred:
        for ch in channels:
            if ch.upper() == p.upper():
                return ch
    return channels[0] if channels else "0"
