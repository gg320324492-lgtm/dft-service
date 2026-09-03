r"""driver 子进程执行器 — scichem python / WSL 跑 driver, Popen + 树杀 + 超时清理

Batch2 #6/#7: subprocess.run → Popen 登记 (_POPEN), 支持:
- 取消: kill_task_process() 树杀 (Windows taskkill /T /F 连杀 g16.exe 等孙进程)
- WSL 残留清理: wsl pkill -f <workdir.name> (gmx/driver cmdline 都含唯一 workdir)
- 超时: TimeoutExpired 分支复用同一清理机制, 不再留孤儿进程
- 编码: 全部显式 utf-8 + errors='replace' (中文 Windows GBK 解码 wsl.exe UTF-16 会炸)
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

# task_id → (Popen, wsl_cleanup (tag, distro) | None); cancel / 超时清理用
_POPEN: dict[str, tuple[subprocess.Popen, tuple[str, str] | None]] = {}


def _wsl_pkill(tag: str, distro: str) -> None:
    """WSL 内按 tag 杀残留计算进程 — gmx / pyscf driver 的 cmdline 都含唯一
    workdir 名 (/tmp/<workdir.name>/...), pkill -f 按名匹配, 不会误伤别的任务"""
    if os.name != "nt":
        return
    try:
        subprocess.run(
            ["wsl.exe", "-d", distro, "-u", "root", "pkill", "-9", "-f", tag],
            capture_output=True, timeout=20,
        )
    except Exception:
        logger.warning("wsl pkill failed for tag %r", tag)


def kill_task_process(task_id: str) -> bool:
    """取消/超时时树杀任务进程。返回是否有已登记进程被杀。

    - Windows: taskkill /T /F 杀整棵进程树 (driver python → g16.exe 孙进程)
    - WSL: pkill -f <tag> 杀 WSL 侧残留 (只杀 wsl.exe 会留 gmx 继续吃 CPU)
    """
    entry = _POPEN.pop(task_id, None)
    if entry is None:
        return False
    proc, wsl = entry
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output=True, timeout=20,
            )
        else:
            proc.terminate()
    except Exception:
        logger.warning("tree-kill failed for %s (pid=%s)", task_id, proc.pid)
    if wsl:
        _wsl_pkill(*wsl)
    return True


def _parse_driver_output(proc: subprocess.Popen, workdir: Path,
                         stdout: str, stderr: str) -> dict[str, Any]:
    """读 result.json (主) / stdout 尾部 JSON (兜底), 都没有 → failed"""
    result: dict[str, Any] | None = None
    result_file = workdir / "result.json"
    if result_file.exists():
        try:
            result = json.loads(result_file.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("corrupt result.json in %s", workdir)
    if result is None and stdout:
        try:
            start = stdout.rindex("{")
            result = json.loads(stdout[start:])
        except Exception:
            result = None
    if result is None:
        result = {
            "status": "failed",
            "error_msg": "driver produced no result.json",
            "stdout_tail": (stdout or "")[-500:],
            "stderr_tail": (stderr or "")[-500:],
            "returncode": proc.returncode,
        }
    if result.get("status") == "failed" and not result.get("stderr_tail") and stderr:
        result["stderr_tail"] = stderr[-500:]
    result.setdefault("work_dir", str(workdir))
    return result


async def _communicate(proc: subprocess.Popen,
                       timeout_s: float) -> tuple[str, str] | None:
    """带超时的 communicate; 超时返回 None (调用方负责清理)"""
    try:
        stdout, stderr = await asyncio.wait_for(
            asyncio.to_thread(proc.communicate), timeout=timeout_s)
        return stdout or "", stderr or ""
    except asyncio.TimeoutExpired:
        return None


async def execute_driver(
    tool: str, workdir: Path, driver: str, params: dict[str, Any],
    timeout_s: float, *, task_id: str = "",
    wsl_cleanup: tuple[str, str] | None = None,
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

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace", env=env,
    )
    if task_id:
        _POPEN[task_id] = (proc, wsl_cleanup)

    try:
        out = await _communicate(proc, timeout_s)
        if out is None:  # 超时: 树杀 + WSL 清理
            if task_id:
                kill_task_process(task_id)
            elif wsl_cleanup:
                _wsl_pkill(*wsl_cleanup)
            return {
                "status": "timeout",  # #19: 一等状态 (CLI 退出码 2 依赖)
                "error_msg": f"driver timeout after {timeout_s:.0f}s "
                             "(process tree killed)",
                "work_dir": str(workdir),
            }
        stdout, stderr = out
    except Exception as e:  # noqa: BLE001 — 编排层兜底
        return {"status": "failed", "error_msg": repr(e), "work_dir": str(workdir)}
    finally:
        if task_id:
            _POPEN.pop(task_id, None)

    return _parse_driver_output(proc, workdir, stdout, stderr)


async def execute_driver_wsl(
    tool: str, workdir: Path, driver: str, params: dict[str, Any],
    timeout_s: float, distro: str, *, task_id: str = "",
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

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace",
    )
    # WSL 清理 tag = workdir 名 (唯一), driver/gmx cmdline 都含它
    if task_id:
        _POPEN[task_id] = (proc, (workdir.name, distro))

    try:
        out = await _communicate(proc, timeout_s)
        if out is None:
            if task_id:
                kill_task_process(task_id)
            return {
                "status": "timeout",  # #19
                "error_msg": f"WSL driver timeout after {timeout_s:.0f}s "
                             "(process tree killed)",
                "work_dir": str(workdir),
            }
        stdout, stderr = out
    except Exception as e:  # noqa: BLE001
        return {"status": "failed", "error_msg": repr(e), "work_dir": str(workdir)}
    finally:
        if task_id:
            _POPEN.pop(task_id, None)

    return _parse_driver_output(proc, workdir, stdout, stderr)


def make_workdir(tool: str, task_id: str) -> Path:
    d = settings.output_root / f"{tool}_{task_id}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def read_progress(tool: str | None, task_id: str) -> dict[str, Any] | None:
    """缺口 #22: 回读 driver 写的 progress.json (running 任务进度)。

    文件不存在 / 解析失败 → None (进度是 best-effort 旁路, 绝不影响 status)。
    """
    if not tool:
        return None
    pf = settings.output_root / f"{tool}_{task_id}" / "progress.json"
    try:
        return json.loads(pf.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — 无进度文件是常态
        return None
