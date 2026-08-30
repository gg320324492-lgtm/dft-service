"""driver 子进程执行器 — scichem python 跑 driver, 读 result.json

service 本体不 import 任何科学计算包 (rdkit/mace/pyscf/psi4 全在 scichem 环境)。
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

from dft_service.config import settings

logger = logging.getLogger("dft_service.executor")

DRIVERS_DIR = Path(__file__).resolve().parent / "drivers"


async def execute_driver(
    tool: str, workdir: Path, driver: str, params: dict[str, Any], timeout_s: float,
) -> dict[str, Any]:
    """跑 driver → 返回 result dict (永不抛异常)"""
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "params.json").write_text(
        json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    driver_path = DRIVERS_DIR / driver
    if not driver_path.exists():
        return {"status": "failed", "error_msg": f"driver missing: {driver}"}
    if not settings.scichem_python.exists():
        return {
            "status": "unavailable",
            "error_msg": f"scichem python not found: {settings.scichem_python}",
        }

    env = os.environ.copy()
    env["DFT_SERVICE_SCICHEM_PYTHON"] = str(settings.scichem_python)
    cmd = [str(settings.scichem_python), str(driver_path), str(workdir)]

    def _run() -> subprocess.CompletedProcess:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s, env=env,
        )

    try:
        proc = await asyncio.to_thread(_run)
    except subprocess.TimeoutExpired:
        return {
            "status": "failed",
            "error_msg": f"driver timeout after {timeout_s:.0f}s",
            "work_dir": str(workdir),
        }
    except Exception as e:  # noqa: BLE001 — 编排层兜底
        return {"status": "failed", "error_msg": repr(e), "work_dir": str(workdir)}

    result: dict[str, Any] | None = None
    result_file = workdir / "result.json"
    if result_file.exists():
        try:
            result = json.loads(result_file.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("corrupt result.json in %s", workdir)
    if result is None and proc.stdout:
        # 双通道兜底: 从 stdout 尾部找 JSON
        try:
            start = proc.stdout.rindex("{")
            result = json.loads(proc.stdout[start:])
        except Exception:
            result = None

    if result is None:
        result = {
            "status": "failed",
            "error_msg": "driver produced no result.json",
            "stdout_tail": (proc.stdout or "")[-500:],
            "stderr_tail": (proc.stderr or "")[-500:],
            "returncode": proc.returncode,
        }
    if result.get("status") == "failed" and not result.get("stderr_tail"):
        if proc.stderr:
            result["stderr_tail"] = proc.stderr[-500:]
    result.setdefault("work_dir", str(workdir))
    return result


async def execute_driver_wsl(
    tool: str, workdir: Path, driver: str, params: dict[str, Any],
    timeout_s: float, distro: str,
) -> dict[str, Any]:
    r"""WSL 回退执行 — wsl.exe -d <distro> python3 <driver> <workdir>

    路径走 win_to_wsl_path 转换 (E:\x → /mnt/e/x); workdir 的 params.json /
    result.json 在 Windows 侧读写, WSL 内经 /mnt 挂载访问, 无 tee 无管道坑。
    """
    import re

    from dft_service.runners.paths import win_to_wsl_path

    def _convert(value):
        """params 里的 Windows 绝对路径值 → WSL 路径 (仅转换真实存在的路径)"""
        if isinstance(value, str) and re.match(r"^[A-Za-z]:[\\/]", value) \
                and os.path.exists(value):
            return win_to_wsl_path(value)
        return value

    def _deep_convert(obj):
        if isinstance(obj, dict):
            return {k: _deep_convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_deep_convert(v) for v in obj]
        return _convert(obj)

    params = _deep_convert(params)

    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "params.json").write_text(
        json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    driver_path = DRIVERS_DIR / driver
    if not driver_path.exists():
        return {"status": "failed", "error_msg": f"driver missing: {driver}"}

    cmd = [
        "wsl.exe", "-d", distro, "python3",
        win_to_wsl_path(driver_path), win_to_wsl_path(workdir),
    ]

    def _run() -> subprocess.CompletedProcess:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s,
        )

    try:
        proc = await asyncio.to_thread(_run)
    except subprocess.TimeoutExpired:
        return {
            "status": "failed",
            "error_msg": f"WSL driver timeout after {timeout_s:.0f}s",
            "work_dir": str(workdir),
        }
    except Exception as e:  # noqa: BLE001
        return {"status": "failed", "error_msg": repr(e), "work_dir": str(workdir)}

    result: dict[str, Any] | None = None
    result_file = workdir / "result.json"
    if result_file.exists():
        try:
            result = json.loads(result_file.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("corrupt result.json in %s", workdir)
    if result is None and proc.stdout:
        try:
            start = proc.stdout.rindex("{")
            result = json.loads(proc.stdout[start:])
        except Exception:
            result = None
    if result is None:
        result = {
            "status": "failed",
            "error_msg": "WSL driver produced no result.json",
            "stdout_tail": (proc.stdout or "")[-500:],
            "stderr_tail": (proc.stderr or "")[-500:],
            "returncode": proc.returncode,
        }
    if result.get("status") == "failed" and not result.get("stderr_tail") and proc.stderr:
        result["stderr_tail"] = proc.stderr[-500:]
    result.setdefault("work_dir", str(workdir))
    return result


def make_workdir(tool: str, task_id: str) -> Path:
    d = settings.output_root / f"{tool}_{task_id}"
    d.mkdir(parents=True, exist_ok=True)
    return d
