r"""dft-service 配置 — 全部环境变量可覆盖, 零 .env 依赖也能跑默认值

环境变量 (前缀 DFT_SERVICE_):
- DFT_SERVICE_HOST            监听地址 (默认 127.0.0.1)
- DFT_SERVICE_PORT            监听端口 (默认 8620)
- DFT_SERVICE_API_KEY         X-API-Key 鉴权; 未设置 = 鉴权关闭 (启动时 warning)
- DFT_SERVICE_DB_URL          SQLite 路径 (默认 E:/dft-service/data/dft_service.db)
- DFT_SERVICE_OUTPUT_ROOT     job 输出根目录 (默认 E:/dft-service/data/jobs)
- DFT_SERVICE_SCISOFTWARE     E:\sci-software 根 (默认 E:/sci-software)
- DFT_SERVICE_SCICHEM_PYTHON  scichem python 路径 (默认 <SCISOFTWARE>/conda-envs/scichem/python.exe)
- DFT_SERVICE_GAUSSIAN_BIN    g16.exe 路径 (默认 <SCISOFTWARE>/g16w/g16.exe)
- DFT_SERVICE_WSL_DISTRO      WSL 发行版 (默认 auto: 探测装了 gmx 的那个)
- DFT_SERVICE_CORS_ORIGINS    CORS 允许源, 逗号分隔 (默认 *, 实验室内网用)
"""
from __future__ import annotations

import os
from pathlib import Path

_SERVICE_ROOT = Path(__file__).resolve().parent.parent


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


class Settings:
    """进程级单例配置 (启动时读一次环境变量)"""

    def __init__(self) -> None:
        self.host: str = _env("DFT_SERVICE_HOST", "127.0.0.1")
        self.port: int = int(_env("DFT_SERVICE_PORT", "8620"))
        self.api_key: str | None = _env("DFT_SERVICE_API_KEY", "") or None

        self.service_root: Path = _SERVICE_ROOT
        self.output_root: Path = Path(
            _env("DFT_SERVICE_OUTPUT_ROOT", _SERVICE_ROOT / "data" / "jobs")
        )
        db_path = Path(
            _env("DFT_SERVICE_DB_URL", _SERVICE_ROOT / "data" / "dft_service.db")
        )
        self.db_url: str = f"sqlite+aiosqlite:///{db_path.as_posix()}"

        self.scisoftware_base: Path = Path(
            _env("DFT_SERVICE_SCISOFTWARE", "E:/sci-software")
        )
        self.scichem_python: Path = Path(
            _env(
                "DFT_SERVICE_SCICHEM_PYTHON",
                self.scisoftware_base / "conda-envs" / "scichem" / "python.exe",
            )
        )
        self.gaussian_bin: Path = Path(
            _env("DFT_SERVICE_GAUSSIAN_BIN", self.scisoftware_base / "g16w" / "g16.exe")
        )
        self.workflows_dir: Path = Path(
            _env("DFT_SERVICE_WORKFLOWS", self.scisoftware_base / "workflows")
        )

        wsl = _env("DFT_SERVICE_WSL_DISTRO", "")
        self.wsl_distro: str | None = wsl or None  # None = auto-detect (runners.paths)

        cors = _env("DFT_SERVICE_CORS_ORIGINS", "*")
        self.cors_origins: list[str] = [o.strip() for o in cors.split(",") if o.strip()]

        # 输出目录保证存在
        self.output_root.mkdir(parents=True, exist_ok=True)


settings = Settings()
