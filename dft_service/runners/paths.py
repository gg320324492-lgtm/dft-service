"""运行环境探测 — scichem python / WSL 发行版 / g16.exe / 包可用性

全部探测结果带 TTL 缓存; 子进程探测永不抛异常 (失败 = unavailable)。
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from dft_service.config import settings

_CACHE: dict[str, tuple[float, object]] = {}
_CACHE_TTL_S = 300.0


def _cached(key: str, fn):
    """简单 TTL 缓存"""
    now = time.monotonic()
    hit = _CACHE.get(key)
    if hit and (now - hit[0]) < _CACHE_TTL_S:
        return hit[1]
    value = fn()
    _CACHE[key] = (now, value)
    return value


def workflows_available() -> bool:
    """E:\\sci-software\\workflows 是否有 python 脚本"""
    w = settings.workflows_dir
    return w.is_dir() and any(w.glob("*.py"))


def gaussian_binary_exists() -> bool:
    return settings.gaussian_bin.exists()


def scichem_python_exists() -> bool:
    return settings.scichem_python.exists()


def _scichem_import_check(module: str) -> bool:
    """scichem python 能否 import 某 module (子进程, 60s 超时, mace/torch 首次 import 慢)"""
    if not scichem_python_exists():
        return False
    try:
        proc = subprocess.run(
            [str(settings.scichem_python), "-c", f"import {module}"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
        )
        return proc.returncode == 0
    except Exception:
        return False


def scichem_has(module: str) -> bool:
    return _cached(f"scichem_has:{module}", lambda: _scichem_import_check(module))


def _list_wsl_distros() -> list[str]:
    try:
        proc = subprocess.run(
            ["wsl.exe", "-l", "-q"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15,
        )
        # WSL 输出是 UTF-16, 可能带 \x00; git-bash 下 text=True 已经 decode 过一部分
        raw = (proc.stdout or "") + (proc.stderr or "")
        names = [
            ln.strip().replace("\x00", "")
            for ln in raw.splitlines()
        ]
        return [n for n in names if n and not n.startswith("Wsl")]
    except Exception:
        return []


def detect_wsl_gromacs_distro() -> str | None:
    """探测装了 gmx 的 WSL 发行版; 找不到返回 None"""
    if os.name != "nt":
        return None

    def _detect():
        preferred = settings.wsl_distro  # 显式配置优先
        candidates = ([preferred] if preferred else []) + [
            d for d in _list_wsl_distros() if d != preferred
        ]
        for distro in candidates:
            if not distro:
                continue
            try:
                proc = subprocess.run(
                    ["wsl.exe", "-d", distro, "bash", "-c", "command -v gmx"],
                    capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15,
                )
                out = (proc.stdout or "").replace("\x00", "")
                if proc.returncode == 0 and "gmx" in out:
                    return distro
            except Exception:
                continue
        return None

    return _cached("wsl_gromacs_distro", _detect)


def win_to_wsl_path(path: str | Path) -> str:
    r"""Windows 路径 → WSL 路径 (E:\x\y → /mnt/e/x/y)"""
    p = Path(path).resolve()
    drive = p.drive.rstrip(":").lower()
    return f"/mnt/{drive}{p.as_posix()[2:]}"


def check_wsl_gromacs(distro: str) -> bool:
    try:
        proc = subprocess.run(
            ["wsl.exe", "-d", distro, "bash", "-c", "command -v gmx"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15,
        )
        out = (proc.stdout or "").replace("\x00", "")
        return proc.returncode == 0 and "gmx" in out
    except Exception:
        return False


def check_wsl_python3_module(distro: str, module: str) -> bool:
    """WSL 发行版的 python3 能否 import 某 module (PySCF WSL 回退用)"""
    try:
        proc = subprocess.run(
            ["wsl.exe", "-d", distro, "python3", "-c", f"import {module}"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
        )
        return proc.returncode == 0
    except Exception:
        return False


def detect_wsl_pyscf_distro() -> str | None:
    """探测 python3 里有 pyscf 的 WSL 发行版 (PySCF WSL 回退); 找不到返回 None"""
    if os.name != "nt":
        return None

    def _detect():
        preferred = settings.wsl_distro
        candidates = ([preferred] if preferred else []) + [
            d for d in _list_wsl_distros() if d != preferred
        ]
        for distro in candidates:
            if not distro:
                continue
            if check_wsl_python3_module(distro, "pyscf"):
                return distro
        return None

    return _cached("wsl_pyscf_distro", _detect)
