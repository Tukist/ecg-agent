"""数据准备脚本：下载/检查项目需要的公开数据。

用法：
    python scripts/setup_data.py            # 检查现状 + 给出下一步提示
    python scripts/setup_data.py --mitdb    # 从 ModelScope 下载 MIT-BIH（约 77 MB，国内 20 MB/s）
    python scripts/setup_data.py --check    # 只检查

MIT-BIH：48 条 30 分钟记录，逐拍+节律人工标注 → 所有指标都来自它。
PPG-DaLiA：真实穿戴 15 例 × 2.5 h（只有 ECG 无心律标注）→ 长时程演示用。
    原始数据体积大（每例 ~1.7 GB），本脚本不自动下；如果拿到原始 h5，
    放到项目同级的 PPG_FieldStudy/ 下（或设环境变量 PPG_DALIA_ROOT），
    再跑 scripts/extract_ppg_dalia.py 生成 250 Hz 缓存。
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecg_agent.io import mitdb, respiban  # noqa: E402


def check():
    ok = True
    n = len(mitdb.record_ids())
    print(f"[MIT-BIH] {n} 条记录 → {mitdb.MITDB_DIR}")
    if n < 40:
        print("          ⚠ 记录不全（完整为 48 条），跑评测前建议先 --mitdb")
        ok = False
    caches = [s for s in respiban.subject_ids() if respiban.cache_path(s).exists()]
    print(f"[PPG-DaLiA 缓存] {len(caches)}/15 例 → {respiban.CACHE_DIR}")
    if not caches:
        print("          ⚠ 没有缓存：长时程演示不可用（不影响 MIT-BIH 的全部指标）")
    print(f"模型文件: {'存在' if (ROOT / 'models' / 'af_logreg.joblib').exists() else '缺失（房颤筛查会回退规则版）'}")
    return ok


def fetch_mitdb():
    try:
        from modelscope import snapshot_download
    except ImportError:
        print("需要先装 modelscope：pip install -i https://pypi.tuna.tsinghua.edu.cn/simple modelscope")
        return False
    print("从 ModelScope 下载 MIT-BIH（约 77 MB）…")
    p = snapshot_download("OmniData/MIT-BIH_Arrhythmia_Database", repo_type="dataset",
                          cache_dir=str(ROOT / "data" / "_ms_cache"))
    zip_path = Path(p) / "raw" / "mit-bih-arrhythmia-database-1.0.0.zip"
    if not zip_path.exists():
        cands = list(Path(p).rglob("*.zip"))
        if not cands:
            print("下载目录里没找到 zip，请检查：", p)
            return False
        zip_path = cands[0]
    out = ROOT / "data" / "mitdb"
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out)
    print(f"已解压到 {out}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mitdb", action="store_true", help="下载 MIT-BIH")
    ap.add_argument("--check", action="store_true", help="只检查")
    args = ap.parse_args()

    if args.mitdb:
        fetch_mitdb()
    ok = check()
    print()
    if ok:
        print("数据就绪。下一步：")
        print("  网页端:  .venv\\Scripts\\python.exe -m streamlit run app.py --server.port 8600")
        print("  命令行:  set PYTHONPATH=src && .venv\\Scripts\\python.exe -m ecg_agent.cli analyze --source mitdb:100 --minutes 10")
    else:
        print("按上面的提示补齐数据后重跑本脚本。")


if __name__ == "__main__":
    main()
