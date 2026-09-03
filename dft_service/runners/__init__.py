"""5 个工具的 runner — 每个: params 校验组装 + execute_driver + health_check"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from dft_service.config import settings
from dft_service.runners import paths
from dft_service.runners.executor import execute_driver, make_workdir


async def run_gaussian(task_id: str, p: dict[str, Any], timeout_s: float) -> dict:
    params = {
        "smiles": p.get("smiles"),
        "xyz_content": p.get("xyz_content"),  # 缺口 #29 内联几何 (二选一)
        "xc": p.get("xc", "B3LYP"),
        "basis": p.get("basis", "6-31G(d)"),
        "job": p.get("job", "opt"),
        "solvent": p.get("solvent", "none"),
        "extra_route": p.get("extra_route", ""),  # 缺口 #28 逃生舱
        # None = 未提供, driver 侧从 SMILES 推断 (缺口 #2); int() 强转会炸 None
        "charge": p.get("charge"),
        "multiplicity": p.get("multiplicity"),
        "nproc": int(p.get("nproc", 8)),
        "mem": p.get("mem", "8GB"),
        "timeout_s": timeout_s,
        "gaussian_bin": str(settings.gaussian_bin),
    }
    return await execute_driver(
        "gaussian", make_workdir("gaussian", task_id),
        "gaussian_driver.py", params, timeout_s + 120, task_id=task_id,
    )


async def run_gromacs(task_id: str, p: dict[str, Any], timeout_s: float) -> dict:
    distro = paths.detect_wsl_gromacs_distro()
    if distro is None:
        return {
            "status": "unavailable",
            "error_msg": "WSL 里没有找到装了 gmx 的发行版 — "
                         "装 GROMACS 或设 DFT_SERVICE_WSL_DISTRO",
        }
    params = {
        "smiles": p["smiles"],
        "n_molecules": int(p.get("n_molecules", 100)),
        "box_nm": float(p.get("box_nm", 3.0)),
        "time_ns": float(p.get("time_ns", 1.0)),
        "temperature_K": float(p.get("temperature_K", 300.0)),
        "analyze": bool(p.get("analyze")),  # 缺口 #31 后处理开关
        "wsl_distro": distro,
    }
    workdir = make_workdir("gromacs", task_id)
    # WSL 清理标签: driver 内 gmx 的 cmdline 都含 /tmp/<workdir.name>
    return await execute_driver(
        "gromacs", workdir,
        "gromacs_driver.py", params, timeout_s,
        task_id=task_id, wsl_cleanup=(workdir.name, distro),
    )


async def run_mace(task_id: str, p: dict[str, Any], timeout_s: float) -> dict:
    params = {
        "smiles": p.get("smiles"),
        "xyz_content": p.get("xyz_content"),  # 缺口 #29 内联几何 (二选一)
        "fmax_ev_A": float(p.get("fmax_ev_A", 0.05)),
        "max_steps": int(p.get("max_steps", 200)),
        "model": p.get("model", "medium"),
        "device": p.get("device", "auto"),
    }
    return await execute_driver(
        "mace", make_workdir("mace", task_id),
        "mace_driver.py", params, timeout_s, task_id=task_id,
    )


async def run_pyscf(task_id: str, p: dict[str, Any], timeout_s: float) -> dict:
    """两段式编排 (rdkit 与 pyscf 可以在不同环境):

    Stage A 几何生成: SMILES → xyz — 需要 rdkit, 跑 scichem python
      (缺口 #29: 传 xyz_content 时跳过本阶段, 直接用内联几何)
    Stage B PySCF 计算: 优先 scichem; 无 pyscf 时 WSL python3 回退 (xyz 输入, 无需 rdkit)
    """
    from dft_service.runners.executor import execute_driver_wsl

    backend = None
    if paths.scichem_has("pyscf"):
        backend = "scichem"
    else:
        wsl_distro = paths.detect_wsl_pyscf_distro()
        if wsl_distro:
            backend = f"wsl:{wsl_distro}"

    workdir = make_workdir("pyscf", task_id)
    smiles = p.get("smiles")

    if p.get("xyz_content"):
        # 缺口 #29: 内联几何 → 直接落 input.xyz, 跳过 geom 阶段
        if backend is None:
            return {
                "status": "unavailable",
                "error_msg": "pyscf 不可用: scichem 无 pyscf 且 WSL 无 pyscf",
                "stage": "backend-select",
            }
        if p.get("charge") is None or p.get("multiplicity") is None:
            return {"status": "failed", "stage": "validate",
                    "error_msg": "xyz 输入必须显式提供 charge 和 multiplicity"}
        xyz_path = workdir / "input.xyz"
        xyz_path.write_text(p["xyz_content"], encoding="utf-8")
        charge = int(p["charge"])
        multiplicity = int(p["multiplicity"])
    else:
        # Stage A — 几何 (scichem rdkit)
        geom_params = {"smiles": smiles}
        geom = await execute_driver(
            "pyscf", workdir,
            "geom_driver.py", geom_params, min(timeout_s, 300), task_id=task_id,
        )
        if geom.get("status") != "success":
            # #19: 几何阶段超时按 timeout 传播, 其余按 failed
            return {
                "status": "timeout" if geom.get("status") == "timeout" else "failed",
                "error_msg": f"geometry generation failed: {geom.get('error_msg')}",
                "stage": "geom",
                "detail": geom,
            }
        if not paths.scichem_has("rdkit") and backend is None:
            return {
                "status": "unavailable",
                "error_msg": "pyscf 不可用: scichem 无 pyscf 且 WSL 无 pyscf",
                "stage": "backend-select",
            }
        xyz_path = geom["xyz_path"]
        charge = int(p.get("charge") if p.get("charge") is not None
                     else geom["charge"])
        multiplicity = int(p.get("multiplicity") if p.get("multiplicity") is not None
                           else geom["multiplicity"])

    # Stage B — PySCF 计算
    calc_params = {
        "smiles": smiles or "<inline-xyz>",
        "xyz_file": str(xyz_path),
        "charge": charge,
        "multiplicity": multiplicity,
        "method": p.get("method", "B3LYP"),
        "basis": p.get("basis", "6-31G*"),
        "operation": p.get("operation", "energy"),
        "solvent": p.get("solvent", "none"),
        "max_opt_steps": int(p.get("max_opt_steps", 50)),
        "backend": backend,
        "solvation_energy": bool(p.get("solvation_energy")),  # 缺口 #30
    }
    if backend == "scichem":
        return await execute_driver(
            "pyscf", workdir, "pyscf_driver.py", calc_params, timeout_s,
            task_id=task_id,
        )
    return await execute_driver_wsl(
        "pyscf", workdir, "pyscf_driver.py", calc_params, timeout_s,
        backend.split(":", 1)[1], task_id=task_id,
    )


async def run_psi4(task_id: str, p: dict[str, Any], timeout_s: float) -> dict:
    params = {
        "smiles": p["smiles"],
        "method": p.get("method", "B3LYP"),
        "basis": p.get("basis", "6-31G*"),
        "operation": p.get("operation", "energy"),
        "charge": p.get("charge"),
        "multiplicity": p.get("multiplicity"),
        "nproc": int(p.get("nproc", 8)),
        "mem": p.get("mem", "8GB"),
    }
    return await execute_driver(
        "psi4", make_workdir("psi4", task_id),
        "psi4_driver.py", params, timeout_s, task_id=task_id,
    )


# ------------------------------------------------------------------
# health checks
# ------------------------------------------------------------------
def health_gaussian() -> dict:
    return {
        "tool": "gaussian",
        "workflows_available": paths.workflows_available(),
        "g16_exists": paths.gaussian_binary_exists(),
        "g16_path": str(settings.gaussian_bin),
        "scichem_rdkit": paths.scichem_has("rdkit"),
    }


def health_gromacs() -> dict:
    distro = paths.detect_wsl_gromacs_distro()
    return {
        "tool": "gromacs",
        "workflows_available": paths.workflows_available(),
        "wsl_distro": distro,
        "wsl_gromacs": distro is not None,
    }


def health_mace() -> dict:
    return {
        "tool": "mace",
        "scichem_python": paths.scichem_python_exists(),
        "mace": paths.scichem_has("mace"),
        "rdkit": paths.scichem_has("rdkit"),
    }


def health_pyscf() -> dict:
    scichem_ok = paths.scichem_has("pyscf")
    wsl_distro = None if scichem_ok else paths.detect_wsl_pyscf_distro()
    return {
        "tool": "pyscf",
        "scichem_python": paths.scichem_python_exists(),
        "pyscf_scichem": scichem_ok,
        "pyscf_wsl_distro": wsl_distro,
        "backend": "scichem" if scichem_ok else (f"wsl:{wsl_distro}" if wsl_distro else None),
        "rdkit": paths.scichem_has("rdkit"),
    }


def health_psi4() -> dict:
    return {
        "tool": "psi4",
        "scichem_python": paths.scichem_python_exists(),
        "psi4": paths.scichem_has("psi4"),
        "rdkit": paths.scichem_has("rdkit"),
    }
