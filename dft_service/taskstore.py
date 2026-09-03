"""任务存储 — 内存 dict (快) + SQLite (持久) 双写

修复 microbubble 版两个缺口:
- /status 重启后不再 404: 内存 miss → DB 回退
- 新增 /jobs 列表查询 (按 tool / status 过滤 + 分页)
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select

from dft_service.db import SessionFactory
from dft_service.models import DFTJob, new_task_id

logger = logging.getLogger("dft_service.taskstore")

# 内存态 (per-process), 重启后自然回退 DB
_TASKS: dict[str, dict[str, Any]] = {}

_TERMINAL = {"success", "failed", "unavailable", "completed_with_warnings",
             "cancelled", "interrupted", "timeout"}

# 缺口 #21: 内存记录只进不出会让长期运行 + 批量场景无界增长。
# 终态任务内存副本保留 24h (覆盖 CLI 轮询窗口), 超量按插入顺序逐出;
# DB 仍是权威, 被逐出任务经 get_task 的 DB 回退照常可查。
_MEM_KEEP_S = 86400.0
_MEM_MAX = 2000


def _evict_mem() -> int:
    """清扫过期/超量的终态内存条目, 返回逐出数 (非终态任务永不动)"""
    now = datetime.now(timezone.utc)
    removed = 0
    for tid in [
        t for t, r in _TASKS.items()
        if r["status"] in _TERMINAL
        and (
            (ft := r.get("finish_time")) is None
            or (now - datetime.fromisoformat(ft)).total_seconds() > _MEM_KEEP_S
        )
    ]:
        _TASKS.pop(tid, None)
        removed += 1
    for tid in list(_TASKS):  # dict 保持插入顺序 = 提交顺序
        if len(_TASKS) <= _MEM_MAX:
            break
        if _TASKS[tid]["status"] in _TERMINAL:
            _TASKS.pop(tid, None)
            removed += 1
    return removed


def create_task(
    tool: str, smiles: str, params: dict, submitter: str | None = None,
    callback_url: str | None = None,
) -> dict[str, Any]:
    """登记新任务 (内存 + DB), 返回内存记录"""
    _evict_mem()  # 缺口 #21: 提交时摊还清扫过期终态内存条目 (DB 仍是权威)
    task_id = new_task_id()
    rec: dict[str, Any] = {
        "task_id": task_id,
        "tool": tool,
        "smiles": smiles,
        "params": params,
        "status": "queued",
        "submitter": submitter,
        # 缺口 #37: 仅存内存 (不落 DB) — 重启丢回调可接受, 由 _execute 持有引用
        "callback_url": callback_url,
        "submit_time": datetime.now(timezone.utc).isoformat(),
        "finish_time": None,
        "result": None,
        "error_msg": None,
        "log_path": None,
    }
    _TASKS[task_id] = rec
    # 真正落库由调用方在事件循环内调 persist_new_task 完成
    rec["_needs_db_insert"] = True
    return rec


async def persist_new_task(rec: dict[str, Any]) -> None:
    """在事件循环内落库新任务"""
    if not rec.get("_needs_db_insert"):
        return
    rec["_needs_db_insert"] = False
    try:
        async with SessionFactory() as session:
            session.add(DFTJob(
                id=rec["task_id"], submitter=rec.get("submitter"),
                tool=rec["tool"], smiles=rec["smiles"],
                params=rec["params"], status=rec["status"],
                submit_time=datetime.fromisoformat(rec["submit_time"]),
            ))
            await session.commit()
    except Exception:
        logger.exception("Failed to persist task %s", rec["task_id"])


async def mark_interrupted_on_startup() -> int:
    """启动清扫: 上次进程死于中途的任务标为 interrupted (缺口 #4)

    服务重启时 driver 子进程随进程消亡, SQLite 里的 queued/running 是假状态。
    全部标 interrupted + error_msg, 避免 /jobs 里积累永远"运行中"的幽灵任务。
    返回受影响行数。
    """
    try:
        from sqlalchemy import func, update

        async with SessionFactory() as session:
            result = await session.execute(
                update(DFTJob)
                .where(DFTJob.status.in_(("queued", "running")))
                .values(status="interrupted",
                        error_msg="service restarted mid-run",
                        finish_time=func.now())
            )
            await session.commit()
            return result.rowcount or 0
    except Exception:
        logger.exception("startup sweep failed")
        return 0


async def cancel_task(task_id: str) -> bool:
    """标记任务为 cancelled (缺口 #6)。返回是否发生了标记 (已是终态则 False)"""
    rec = _TASKS.get(task_id)
    if rec is not None:
        if rec["status"] in _TERMINAL:
            return False
        rec["status"] = "cancelled"
        rec["error_msg"] = "cancelled by user"
        rec["finish_time"] = datetime.now(timezone.utc).isoformat()

    try:
        async with SessionFactory() as session:
            row = await session.get(DFTJob, task_id)
            if row is None:
                return bool(rec is not None)
            if row.status in _TERMINAL:
                return False
            row.status = "cancelled"
            row.error_msg = "cancelled by user"
            row.finish_time = datetime.now(timezone.utc)
            await session.commit()
            return True
    except Exception:
        logger.exception("cancel_task DB update failed for %s", task_id)
        return False


async def finish_task(
    task_id: str, status: str, result: dict | None, error_msg: str | None = None,
) -> None:
    """任务结束: 更新内存 + DB。已取消的任务不覆盖 (取消竞态保护)"""
    cur = _TASKS.get(task_id, {}).get("status")
    if cur == "cancelled":
        logger.info("task %s already cancelled, discarding result", task_id)
        return
    finish_iso = datetime.now(timezone.utc).isoformat()
    rec = _TASKS.get(task_id)
    if rec is not None:
        rec["status"] = status
        rec["result"] = result
        rec["error_msg"] = error_msg
        rec["finish_time"] = finish_iso
        if result:
            rec["log_path"] = result.get("log_path") or rec.get("log_path")

    try:
        async with SessionFactory() as session:
            row = await session.get(DFTJob, task_id)
            if row is not None:
                row.status = status
                row.result = result
                row.error_msg = error_msg
                row.finish_time = datetime.now(timezone.utc)
                if result:
                    row.log_path = result.get("log_path") or row.log_path
                await session.commit()
    except Exception:
        logger.exception("Failed to persist finish state for %s", task_id)


async def get_task(task_id: str, include_result: bool = False) -> dict | None:
    """内存优先, DB 回退 (修复重启 404)"""
    rec = _TASKS.get(task_id)
    if rec is not None:
        out = {k: v for k, v in rec.items() if not k.startswith("_")}
        if not include_result:
            out.pop("result", None)
        return out
    try:
        async with SessionFactory() as session:
            row = await session.get(DFTJob, task_id)
            if row is None:
                return None
            return row.to_dict(include_result=include_result)
    except Exception:
        logger.exception("DB fallback failed for %s", task_id)
        return None


async def stats() -> dict:
    """缺口 #36: 运营统计 (SQLite 聚合, 全历史 + 近 24h)

    每后端: 总数/各状态计数/成功率 (success+warnings / 已终结)/平均耗时 (分钟);
    全局: 排队深度 (queued+running) 与近 24h 提交量。
    """
    try:
        from sqlalchemy import case
        from sqlalchemy import func as f

        dur_min = (f.julianday(DFTJob.finish_time) - f.julianday(DFTJob.submit_time)) * 1440.0
        finished = DFTJob.status.in_(_TERMINAL)
        ok = DFTJob.status.in_(("success", "completed_with_warnings"))
        q = (
            select(
                DFTJob.tool,
                f.count().label("n_total"),
                f.sum(case((finished, 1), else_=0)).label("n_finished"),
                f.sum(case((ok, 1), else_=0)).label("n_ok"),
                f.avg(case((ok, dur_min), else_=None)).label("avg_min"),
                f.sum(case((DFTJob.status == "failed", 1), else_=0)).label("n_failed"),
                f.sum(case((DFTJob.status == "timeout", 1), else_=0)).label("n_timeout"),
            )
            .group_by(DFTJob.tool)
        )
        now_epoch = datetime.now(timezone.utc).timestamp()
        async with SessionFactory() as session:
            rows = (await session.execute(q)).all()
            queue_depth = (await session.execute(
                select(func.count()).select_from(DFTJob)
                .where(DFTJob.status.in_(("queued", "running")))
            )).scalar() or 0
            recent = (await session.execute(
                select(func.count()).select_from(DFTJob)
                .where(DFTJob.submit_time >= func.datetime(
                    (now_epoch - 86400), "unixepoch"))
            )).scalar() or 0
        return {
            "status": "success",
            "per_tool": sorted(
                [
                    {
                        "tool": r.tool,
                        "n_total": r.n_total,
                        "n_finished": int(r.n_finished or 0),
                        "n_ok": int(r.n_ok or 0),
                        "n_failed": int(r.n_failed or 0),
                        "n_timeout": int(r.n_timeout or 0),
                        "success_rate": (round(r.n_ok / r.n_finished, 3)
                                         if r.n_finished else None),
                        "avg_minutes": round(r.avg_min, 1) if r.avg_min else None,
                    }
                    for r in rows
                ],
                key=lambda t: -t["n_total"],
            ),
            "queue_depth": queue_depth,
            "submissions_last_24h": recent,
        }
    except Exception:
        logger.exception("stats aggregation failed")
        return {"status": "failed", "error_msg": "stats query failed"}


async def list_jobs(
    tool: str | None = None, status: str | None = None,
    limit: int = 50, offset: int = 0,
) -> dict:
    """任务列表 (DB 查询, 按提交时间倒序)"""
    limit = min(max(limit, 1), 200)
    conditions = []
    if tool:
        conditions.append(DFTJob.tool == tool)
    if status:
        conditions.append(DFTJob.status == status)
    try:
        async with SessionFactory() as session:
            q = select(DFTJob).order_by(DFTJob.submit_time.desc())
            count_q = select(func.count()).select_from(DFTJob)
            if conditions:
                q = q.where(*conditions)
                count_q = count_q.where(*conditions)
            total = (await session.execute(count_q)).scalar() or 0
            rows = (await session.execute(q.limit(limit).offset(offset))).scalars().all()
            return {
                "total": total,
                "limit": limit,
                "offset": offset,
                "jobs": [r.to_dict() for r in rows],
            }
    except Exception:
        logger.exception("list_jobs query failed")
        return {"total": 0, "limit": limit, "offset": offset, "jobs": []}
