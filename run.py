"""启动入口 — python run.py"""
from __future__ import annotations

import uvicorn

from dft_service.config import settings

if __name__ == "__main__":
    uvicorn.run("dft_service.main:app", host=settings.host, port=settings.port)
