"""cleanup 测试 (缺口 #35/#18/#10) — 全在 tmp_path 里跑, 不碰真实 data/"""
import os
import sqlite3
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dft_cli  # noqa: E402


def _make_env(tmp_path, monkeypatch):
    monkeypatch.setattr(dft_cli, "ROOT", tmp_path)
    monkeypatch.setattr(dft_cli, "_g16_scratch_dir", lambda: None)  # 隔离真实 g16 目录
    data = tmp_path / "data"
    (data / "jobs").mkdir(parents=True)
    conn = sqlite3.connect(data / "dft_service.db")
    conn.execute("CREATE TABLE dft_jobs (id TEXT PRIMARY KEY, status TEXT)")
    conn.execute("INSERT INTO dft_jobs VALUES ('abc123', 'success')")
    conn.execute("INSERT INTO dft_jobs VALUES ('run0000', 'running')")
    conn.commit()
    conn.close()
    done = data / "jobs" / "pyscf_abc123"
    done.mkdir()
    (done / "result.json").write_text('{"status": "success"}', encoding="utf-8")
    live = data / "jobs" / "pyscf_run0000"
    live.mkdir()
    (live / "result.json").write_text("{}", encoding="utf-8")
    old = time.time() - 40 * 86400
    for d in (done, live):
        os.utime(d, (old, old))  # 两个目录都"超期" — 只有终态的该被删
    return data


def test_cleanup_archive_and_skip_running(tmp_path, monkeypatch):
    data = _make_env(tmp_path, monkeypatch)
    removed, kept = dft_cli.run_cleanup(
        days=7, archive=str(data / "archives"))
    assert removed == 1 and kept == 1
    assert not (data / "jobs" / "pyscf_abc123").exists()
    assert (data / "jobs" / "pyscf_run0000").exists()  # running 绝不碰
    zf = data / "archives" / "pyscf_abc123.zip"
    assert zf.exists()
    with zipfile.ZipFile(zf) as z:
        assert "pyscf_abc123/result.json" in z.namelist()  # 留底完整


def test_cleanup_archive_failure_keeps_dir(tmp_path, monkeypatch):
    """归档失败 → 跳过删除 (宁留勿丢)"""
    _make_env(tmp_path, monkeypatch)
    real_zip = dft_cli._zip_dir

    def boom(src, dst):
        raise OSError("disk on fire")

    monkeypatch.setattr(dft_cli, "_zip_dir", boom)
    data = tmp_path / "data"
    dft_cli.run_cleanup(days=7, archive=str(data / "archives"))
    assert (data / "jobs" / "pyscf_abc123").exists()  # zip 失败不删
    monkeypatch.setattr(dft_cli, "_zip_dir", real_zip)


def test_backup_db_online(tmp_path, monkeypatch):
    _make_env(tmp_path, monkeypatch)
    b1 = dft_cli.backup_db(keep=2)
    assert b1 and b1.exists()
    conn = sqlite3.connect(b1)
    assert conn.execute("SELECT COUNT(*) FROM dft_jobs").fetchone()[0] == 2
    conn.close()
    # 轮转: 造 3 份旧备份, keep=2 只留最近 2 份
    bdir = b1.parent
    import datetime
    for i in range(3):
        old = bdir / f"dft_service-2026010{i}-0000.db"
        old.write_bytes(b"x")
    dft_cli.backup_db(keep=2)
    assert len(list(bdir.glob("dft_service-*.db"))) == 2
