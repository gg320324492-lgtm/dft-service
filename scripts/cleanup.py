r"""dft-service 产物清理 — 扫 data/jobs, 删超过 N 天的终态任务目录

    python scripts/cleanup.py --days 7 --dry-run
    python scripts/cleanup.py --days 7 --purge-rows   # 连 DB 行一起删
    python scripts/cleanup.py --days 7 --archive --backup   # 删前 zip 留底 + DB 备份

只动终态任务目录; 运行中/排队中的任务绝不碰。与 `dft cleanup` 子命令等价。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dft_cli import backup_db, run_cleanup  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=7,
                    help="保留天数 (默认 7, 目录 mtime 距今超过才删)")
    ap.add_argument("--dry-run", action="store_true", help="只看不删")
    ap.add_argument("--purge-rows", action="store_true",
                    help="连 dft_jobs DB 行一起删")
    ap.add_argument("--archive", nargs="?", const="data/archives", default=None,
                    help="删除前 zip 归档 (默认 data/archives) — 缺口 #35")
    ap.add_argument("--backup", action="store_true",
                    help="SQLite 在线备份到 data/backups/ — 缺口 #35")
    ap.add_argument("--keep-backups", dest="keep_backups", type=int, default=7)
    args = ap.parse_args()
    run_cleanup(args.days, args.dry_run, args.purge_rows, archive=args.archive)
    if args.backup:
        backup_db(keep=args.keep_backups, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
