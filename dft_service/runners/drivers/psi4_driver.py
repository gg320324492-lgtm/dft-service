r"""Psi4 driver — 复用 E:\sci-software\workflows\psi4_runner.py (scichem 里 psi4 1.9.1 实测可用)

支持 single / optimize / properties 三种任务 + 线程数/内存控制。
charge/mult 未给时由 psi4_runner.smiles_to_geometry 从 SMILES 自动推断。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from _driver_common import load_params, run_driver, write_progress

sys.path.insert(0, "E:/sci-software/workflows")  # noqa: S104


def compute(params: dict, workdir: Path) -> dict:
    import psi4  # noqa: PLC0415 — 可用性哨兵 (scichem 有 1.9.1)
    import psi4_runner  # noqa: PLC0415 — workflows CLI 模块直接复用

    t0 = time.time()
    psi4.set_num_threads(int(params.get("nproc", 8)))
    psi4.set_memory(f"{params.get('mem', '8GB')}")
    psi4.core.clean_options()

    smiles = params["smiles"]
    method = params.get("method", "B3LYP")
    basis = params.get("basis", "6-31G*")
    operation = params.get("operation", "energy")

    geom, auto_charge, auto_mult = psi4_runner.smiles_to_geometry(smiles)
    charge = params.get("charge")
    charge = int(charge) if charge is not None else int(auto_charge)
    multiplicity = params.get("multiplicity")
    multiplicity = int(multiplicity) if multiplicity is not None else int(auto_mult)
    # smiles_to_geometry 不吃显式 charge/mult — set 到 molecule 上
    # (psi4_runner.run_* 内部 set_molecular_charge / set_multiplicity)

    if operation == "optimize":
        write_progress(workdir, "optimize", method=method, basis=basis,
                       elapsed_s=round(time.time() - t0))
        res = psi4_runner.run_optimize(geom, method, basis, charge, multiplicity)
    elif operation == "properties":
        write_progress(workdir, "properties", method=method, basis=basis,
                       elapsed_s=round(time.time() - t0))
        res = psi4_runner.run_properties(geom, method, basis, charge, multiplicity)
    else:
        write_progress(workdir, "single_point", method=method, basis=basis,
                       elapsed_s=round(time.time() - t0))
        res = psi4_runner.run_single_point(geom, method, basis, charge, multiplicity)

    out: dict = {
        "tool": "psi4",
        "smiles": smiles,
        "method": method,
        "basis": basis,
        "operation": operation,
        "charge": charge,
        "multiplicity": multiplicity,
        "work_dir": str(workdir),
        "status": "success",
    }
    out.update(res)  # energy_hartree / time_s / dipole_debye / homo_lumo_gap_eV ...
    out["elapsed_s"] = round(time.time() - t0, 2)
    return out


if __name__ == "__main__":
    run_driver(Path(sys.argv[1]), compute)
