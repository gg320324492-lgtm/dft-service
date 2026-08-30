"""DFTJob 独立持久化模型 — SQLite (aiosqlite)

与 microbubble-agent 的 dft_jobs 表 (PostgreSQL JSONB/UUID) 字段对齐,
但类型全部换成 SQLite 友好形态: id 是 16 位 hex 字符串, params/result 用 JSON。
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


def new_task_id() -> str:
    return uuid.uuid4().hex[:16]


class DFTJob(Base):
    """DFT/MD 异步任务记录"""

    __tablename__ = "dft_jobs"

    id: Mapped[str] = mapped_column(String(16), primary_key=True)
    submitter: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    tool: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    smiles: Mapped[str] = mapped_column(Text, nullable=False)
    params: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="queued", index=True)
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    log_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_msg: Mapped[str | None] = mapped_column(Text, nullable=True)
    submit_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    finish_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_dft_jobs_tool_status", "tool", "status"),
        Index("ix_dft_jobs_submitter_submit", "submitter", "submit_time"),
    )

    def to_dict(self, include_result: bool = False) -> dict:
        d = {
            "task_id": self.id,
            "submitter": self.submitter,
            "tool": self.tool,
            "smiles": self.smiles,
            "params": self.params,
            "status": self.status,
            "error_msg": self.error_msg,
            "log_path": self.log_path,
            "submit_time": self.submit_time.isoformat() if self.submit_time else None,
            "finish_time": self.finish_time.isoformat() if self.finish_time else None,
        }
        if include_result:
            d["result"] = self.result
        return d
