"""缺口 #4 — 启动清扫: queued/running 幽灵任务标 interrupted"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dft_service import taskstore  # noqa: E402
from dft_service.db import SessionFactory  # noqa: E402
from dft_service.models import DFTJob  # noqa: E402


def _insert_row(task_id: str, status: str):
    import asyncio

    async def _go():
        async with SessionFactory() as session:
            # merge = 幂等 upsert, 重跑测试不撞主键
            await session.merge(DFTJob(
                id=task_id, tool="gaussian", smiles="O",
                params={}, status=status,
                error_msg=None, finish_time=None,
            ))
            await session.commit()

    asyncio.run(_go())


def _get_status(task_id: str):
    import asyncio

    async def _go():
        async with SessionFactory() as session:
            row = await session.get(DFTJob, task_id)
            return (row.status, row.error_msg, row.finish_time) if row else None

    return asyncio.run(_go())


def test_sweep_marks_stale_running_interrupted():
    _insert_row("sweeprun01", "running")
    _insert_row("sweepque02", "queued")
    _insert_row("sweepokk03", "success")  # 终态不受影响

    n = asyncio_run(taskstore.mark_interrupted_on_startup())
    assert n >= 2

    st = _get_status("sweeprun01")
    assert st[0] == "interrupted" and "restarted" in st[1] and st[2] is not None
    st = _get_status("sweepque02")
    assert st[0] == "interrupted"
    st = _get_status("sweepokk03")
    assert st[0] == "success" and st[2] is None


def asyncio_run(coro):
    import asyncio

    return asyncio.run(coro)
