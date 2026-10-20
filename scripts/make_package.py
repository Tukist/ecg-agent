"""打包脚本：生成可发给同学的 zip（代码 + 文档 + 少量示例数据，不含密钥/大文件）。

用法：
    python scripts/make_package.py            # 默认带 5 条 mitdb 记录 + 1 例穿戴缓存
    python scripts/make_package.py --no-data  # 只打代码和文档（~2 MB）

产出：dist/ecg_agent_v1.0.zip

排除规则（重要）：
  .venv/  data/_kaggle/  data/_ms_cache/  data/mitdb（除白名单记录）  __pycache__
  .ds_key  .env  reports/_web/
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 打进去的 mitdb 记录（覆盖：干净 / 噪声 / 房颤 / 室早）
DEMO_RECORDS = ["100", "114", "201", "203", "222"]
DEMO_SUBJECTS = ["S1"]

INCLUDE = [
    "src", "scripts", "docs", "results", "models", "reports",
    "app.py", "README.md", "requirements.txt", ".gitignore", "start_web.bat",
]
EXCLUDE_DIRS = {".venv", "__pycache__", "_kaggle", "_ms_cache", "_web", ".git"}
EXCLUDE_FILES = {".ds_key", ".env", "report.md", "report_fig.png"}


def _skip(path: Path) -> bool:
    parts = set(path.parts)
    if parts & EXCLUDE_DIRS:
        return True
    if path.name in EXCLUDE_FILES or path.name.endswith(".pyc"):
        return True
    return False


def build(with_data: bool = True) -> Path:
    stage = ROOT / "dist" / "_stage" / "ecg_agent_v1.0"
    if stage.parent.exists():
        shutil.rmtree(stage.parent, ignore_errors=True)
    stage.mkdir(parents=True)

    n = 0
    for item in INCLUDE:
        src = ROOT / item
        if not src.exists():
            continue
        if src.is_file():
            shutil.copy(src, stage / item)
            n += 1
            continue
        for p in src.rglob("*"):
            if p.is_dir() or _skip(p) or _skip(p.relative_to(ROOT)):
                continue
            rel = p.relative_to(ROOT)
            dst = stage / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(p, dst)
            n += 1

    if with_data:
        md = ROOT / "data" / "mitdb" / "mit-bih-arrhythmia-database-1.0.0"
        out = stage / "data" / "mitdb" / "mit-bih-arrhythmia-database-1.0.0"
        out.mkdir(parents=True, exist_ok=True)
        for rec in DEMO_RECORDS:
            for suf in (".dat", ".hea", ".atr"):
                f = md / f"{rec}{suf}"
                if f.exists():
                    shutil.copy(f, out / f.name)
                    n += 1
        cache = ROOT / "data" / "cache" / "ppg_dalia"
        cout = stage / "data" / "cache" / "ppg_dalia"
        cout.mkdir(parents=True, exist_ok=True)
        for sid in DEMO_SUBJECTS:
            f = cache / f"{sid}_ecg.npy"
            if f.exists():
                shutil.copy(f, cout / f.name)
                n += 1
        m = cache / "manifest.csv"
        if m.exists():
            shutil.copy(m, cout / m.name)

    zip_path = ROOT / "dist" / "ecg_agent_v1.0.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(stage.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(stage.parent))
    shutil.rmtree(stage.parent, ignore_errors=True)
    size_mb = zip_path.stat().st_size / 1e6
    print(f"已打包 {n} 个文件 → {zip_path}（{size_mb:.1f} MB）")
    return zip_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-data", action="store_true", help="不打示例数据")
    a = ap.parse_args()
    build(with_data=not a.no_data)
