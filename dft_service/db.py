"""async SQLAlchemy 引擎 + session factory (SQLite)"""
from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from dft_service.config import settings
from dft_service.models import Base

engine = create_async_engine(settings.db_url, echo=False)
SessionFactory: async_sessionmaker[AsyncSession] = async_sessionmaker(
    engine, expire_on_commit=False
)


async def init_db() -> None:
    """建表 (幂等)"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
