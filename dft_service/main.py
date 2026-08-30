"""FastAPI app 工厂 + lifespan"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from dft_service import __version__
from dft_service.auth import require_api_key
from dft_service.config import settings
from dft_service.db import init_db
from dft_service.api import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("dft_service.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    if settings.api_key is None:
        logger.warning(
            "DFT_SERVICE_API_KEY 未设置 — 鉴权关闭 (本地/内网模式); "
            "对外部署务必设置 API key"
        )
    logger.info(
        "dft-service v%s ready | output=%s | scichem=%s",
        __version__, settings.output_root, settings.scichem_python,
    )
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="dft-service",
        description="微纳米气泡课题组独立 DFT/MD 计算服务 "
                    "(Gaussian / GROMACS / MACE / PySCF / Psi4)",
        version=__version__,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
    app.include_router(router)

    @app.get("/health", tags=["健康"])
    async def health():
        return {
            "status": "ok",
            "service": "dft-service",
            "version": __version__,
            "auth": settings.api_key is not None,
        }

    # 保持 require_api_key 引用 (文档依赖; 未鉴权模式它是 no-op)
    _ = require_api_key
    return app


app = create_app()
