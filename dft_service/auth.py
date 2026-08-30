"""X-API-Key 鉴权 — DFT_SERVICE_API_KEY 未设置时完全放行 (本地/内网模式)"""
from __future__ import annotations

import fastapi
from fastapi import Header, HTTPException

from dft_service.config import settings


async def require_api_key(
    x_api_key: str | None = fastapi.Header(default=None, alias="X-API-Key"),
) -> None:
    if settings.api_key is None:
        return
    if not x_api_key or x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")
