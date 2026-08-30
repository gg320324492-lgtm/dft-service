"""GROMACS MD driver — 复用 E:\sci-software\workflows\gromacs_runner.py

流程: prep_system (SMILES→体系) → energy_minimize → run_md
WSL 发行版由 runner 探测后传进来 (Ubuntu-24.04), 不再硬编码 "Ubuntu"。
小盒子 (box < 2.2nm) 自动缩截断半径, 否则固定 1.0nm 截断 grompp 必报错。
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

from _driver_common import load_params, run_driver

sys.path.insert(0, "E:/sci-software/workflows")  # noqa: S104

_CUTOFF_KEYS = ("rlist", "rcoulomb", "rvdw")


def _make_mdp(kind: str, workdir: Path, box_nm: float,
              nsteps: int | None = None, temp: float | None = None) -> Path:
    """生成 mdp — 盒子太小时缩截断 (cutoff < 盒子一半, 留 0.1nm 余量)"""
    from gromacs_runner import _MD_MDP_TEMPLATE, _MINI_MDP

    if kind == "em":
        text = _MINI_MDP
    else:
        nstout = min(5000, max(250, (nsteps or 500000) // 10))
        text = _MD_MDP_TEMPLATE.format(nsteps=nsteps, temp=temp, nstout=nstout)
    cutoff = round(min(1.0, box_nm / 2 - 0.1), 2)
    if cutoff < 1.0:
        for key in _CUTOFF_KEYS:
            text = re.sub(rf"^{key}\s*=.*$", f"{key:<15} = {cutoff}", text, flags=re.M)
    path = workdir / f"{kind}.mdp"
    path.write_text(text, encoding="utf-8")
    return path


def compute(params: dict, workdir: Path) -> dict:
    from gromacs_runner import energy_minimize, prep_system, run_md  # noqa: PLC0415

    t0 = time.time()
    distro = params["wsl_distro"]
    box_nm = float(params.get("box_nm", 3.0))
    time_ns = float(params.get("time_ns", 1.0))
    temp = float(params.get("temperature_K", 300.0))
    out: dict = {
        "tool": "gromacs",
        "smiles": params["smiles"],
        "n_molecules": params.get("n_molecules", 100),
        "box_nm": box_nm,
        "time_ns": time_ns,
        "temperature_K": temp,
        "wsl_distro": distro,
        "work_dir": str(workdir),
    }

    # 1) prep
    paths = prep_system(
        params["smiles"],
        n_mol=params.get("n_molecules", 100),
        box_size=box_nm,
        output_dir=workdir,
    )
    out["gro_path"] = str(paths["gro"])
    out["top_path"] = str(paths["top"])

    # 2) energy minimize (小盒子自适应 mdp + 显式拓扑 + 唯一 WSL 工作目录)
    # wsl_work 用 workdir 名 (含唯一 task_id): 取消/超时可按名 pkill 不误伤
    em = energy_minimize(
        paths["gro"], _make_mdp("em", workdir, box_nm),
        workdir / "em", wsl_distro=distro, top_path=paths["top"],
        wsl_work=f"/tmp/{workdir.name}",
    )
    out["em_gro"] = str(em["gro"])
    out["em_log"] = str(em["log"])

    # 3) run_md (nsteps 由 run_md 内部按 1ns=500000 步换算)
    md = run_md(
        em["gro"], workdir / "md",
        time_ns=time_ns, temperature_k=temp,
        wsl_distro=distro,
        mdp_path=_make_mdp(
            "md", workdir, box_nm,
            nsteps=int(time_ns * 500000), temp=temp,
        ),
        top_path=paths["top"],
        wsl_work=f"/tmp/{workdir.name}",
    )
    out["trajectory_path"] = str(md["xtc"])
    out["md_log"] = str(md["log"])
    out["final_gro"] = str(md["gro"])
    out["status"] = "success"
    out["elapsed_s"] = round(time.time() - t0, 2)
    return out


if __name__ == "__main__":
    run_driver(Path(sys.argv[1]), compute)
