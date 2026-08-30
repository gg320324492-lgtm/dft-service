r"""dft-service 产物清理 — 扫 data/jobs, 删超过 N 天的终态任务目录

    python scripts/cleanup.py --days 7 --dry-run
    python scripts/cleanup.py --days 7 --purge-rows   # 连 DB 行一起删

只动终态任务目录; 运行中/排队中的任务绝不碰。与 `dft cleanup` 子命令等价。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dft_cli import run_cleanup  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=7,
                    help="保留天数 (默认 7, 目录 mtime 距今超过才删)")
    ap.add_argument("--dry-run", action="store_true", help="只看不删")
    ap.add_argument("--purge-rows", action="store_true",
                    help="连 dft_jobs DB 行一起删")
    args = ap.parse_args()
    removed, kept = run_cleanup(args.days, args.dry_run, args.purge_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
