"""FastAPI app 工厂 + lifespan"""
from __future__ import annotations

import logging
import logging.handlers
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from dft_service import __version__, taskstore
from dft_service.auth import require_api_key
from dft_service.config import settings
from dft_service.db import init_db
from dft_service.api import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def _setup_file_logging() -> None:
    """缺口 #11: 文件日志 (5MB × 3 轮转), 后台排查不再依赖 stdout"""
    log_dir = settings.service_root / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        log_dir / "dft-service.log",
        maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(handler)
logger = logging.getLogger("dft_service.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _setup_file_logging()
    await init_db()
    # 缺口 #4: 重启清扫 — 上次中途死掉的任务不可能复活, 全部标 interrupted
    interrupted = await taskstore.mark_interrupted_on_startup()
    if interrupted:
        logger.warning("marked %d stale queued/running task(s) as interrupted",
                       interrupted)
    if settings.api_key is None:
        logger.warning(
            "DFT_SERVICE_API_KEY 未设置 — 鉴权关闭 (本地/内网模式); "
            "对外部署务必设置 API key"
        )
    _start_health_prewarm()  # 缺口 #26
    logger.info(
        "dft-service v%s ready | output=%s | scichem=%s",
        __version__, settings.output_root, settings.scichem_python,
    )
    yield


def _start_health_prewarm() -> None:
    """缺口 #26: 后台线程预热探测缓存 (daemon, 不阻塞启动)。

    /dft/tools 冷缓存要现场起子进程探测 (scichem import mace 最长 120s,
    WSL 发行版逐个探), 首个探针可能几分钟。启动即预热填 TTL 缓存。
    """
    import threading

    def _warm():
        try:
            from dft_service.runners.tool_definitions import list_available_tools
            body = list_available_tools()
            logger.info("health prewarm: %d/%d tools available",
                        body["available_count"], body["count"])
        except Exception:
            logger.exception("health prewarm failed (non-fatal)")

    threading.Thread(target=_warm, name="health-prewarm", daemon=True).start()


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
