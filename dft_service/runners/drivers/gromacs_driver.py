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

from _driver_common import load_params, run_driver, write_progress

sys.path.insert(0, "E:/sci-software/workflows")  # noqa: S104

_CUTOFF_KEYS = ("rlist", "rcoulomb", "rvdw")


def _analyze_md(md_paths: dict, workdir: Path, distro: str) -> dict:
    """缺口 #31 v1: MD 后处理 — gmx rms/energy → 统计量 (+ matplotlib 出图)。

    自写而非复用 workflows.analyze_md: 后者 RMSD 写死 "Protein" 组,
    气泡/纯溶剂体系没有该组会失败。这里统一用 "System" 组。
    分析失败不拖垮已成功的 MD, 只回 rmsd_analysis_failed / energy_analysis_failed。
    """
    from gromacs_runner import _copy_from_wsl, _copy_to_wsl, _read_xvg, _wsl_run

    out: dict = {}
    adir = workdir / "analyze"
    adir.mkdir(exist_ok=True)
    wsl_work = f"/tmp/{workdir.name}_an"
    _wsl_run(f"mkdir -p {wsl_work}", wsl_distro=distro)
    tpr_wsl = _copy_to_wsl(md_paths["tpr"], wsl_work, wsl_distro=distro)
    xtc_wsl = _copy_to_wsl(md_paths["xtc"], wsl_work, wsl_distro=distro)
    edr_wsl = _copy_to_wsl(md_paths["edr"], wsl_work, wsl_distro=distro)

    # RMSD (System 拟合 + System 计算, 对任意体系通用)
    try:
        rmsd_wsl = f"{wsl_work}/rmsd.xvg"
        _wsl_run(f'echo "System System" | gmx rms -s {tpr_wsl} -f {xtc_wsl} '
                 f"-o {rmsd_wsl} -xvg none", cwd=wsl_work, wsl_distro=distro,
                 timeout=180)
        host_rmsd = _copy_from_wsl(rmsd_wsl, adir, wsl_distro=distro)
        df = _read_xvg(host_rmsd)
        col = df.columns[-1]
        vals = df[col].astype(float)
        out["rmsd_xvg"] = str(host_rmsd)
        out["rmsd_avg_nm"] = round(float(vals.mean()), 4)
        out["rmsd_max_nm"] = round(float(vals.max()), 4)
    except Exception as e:  # noqa: BLE001
        out["rmsd_analysis_failed"] = str(e)[:200]

    # 势能 / 温度 (gmx energy)
    try:
        en_wsl = f"{wsl_work}/energy.xvg"
        _wsl_run(f'echo "Potential Temperature" | gmx energy -f {edr_wsl} '
                 f"-o {en_wsl} -xvg none",
                 cwd=wsl_work, wsl_distro=distro, timeout=120)
        host_en = _copy_from_wsl(en_wsl, adir, wsl_distro=distro)
        df = _read_xvg(host_en)
        for c in df.columns:
            if "Potential" in c or "col_1" in c:
                out["potential_avg_kj_mol"] = round(float(df[c].mean()), 2)
            if "Temperature" in c or "col_2" in c:
                out["temperature_avg_K"] = round(float(df[c].mean()), 2)
        out["energy_xvg"] = str(host_en)
    except Exception as e:  # noqa: BLE001
        out["energy_analysis_failed"] = str(e)[:200]

    # 出图 (scichem matplotlib, 无则静默跳过)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if "rmsd_xvg" in out:
            df = _read_xvg(out["rmsd_xvg"])
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.plot(df[df.columns[0]], df[df.columns[-1]])
            ax.set_xlabel("time (ps)"); ax.set_ylabel("RMSD (nm)")
            ax.set_title("RMSD")
            fig.savefig(adir / "rmsd.png", dpi=110, bbox_inches="tight")
            plt.close(fig)
            out["rmsd_png"] = str(adir / "rmsd.png")
    except Exception:  # noqa: BLE001 — 无 matplotlib/无显示 全静默
        pass
    return out


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
    write_progress(workdir, "prep", n_molecules=out["n_molecules"],
                   box_nm=box_nm, elapsed_s=round(time.time() - t0))
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
    # 缺口 #22: 三阶段顺序阻塞, run_md 内是单次 gmx mdrun (不改 workflows
    # 拿不到 mdrun 中间步), 故只报"当前处于哪个 stage"这一粒度
    write_progress(workdir, "energy_minimize", elapsed_s=round(time.time() - t0))
    em = energy_minimize(
        paths["gro"], _make_mdp("em", workdir, box_nm),
        workdir / "em", wsl_distro=distro, top_path=paths["top"],
        wsl_work=f"/tmp/{workdir.name}",
    )
    out["em_gro"] = str(em["gro"])
    out["em_log"] = str(em["log"])

    # 3) run_md (nsteps 由 run_md 内部按 1ns=500000 步换算)
    write_progress(workdir, "md", time_ns=time_ns,
                   elapsed_s=round(time.time() - t0))
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

    # 缺口 #31: 分析后处理 (可选) — 失败只降级 warning, 不推翻已成功的 MD
    if params.get("analyze"):
        write_progress(workdir, "analyze", elapsed_s=round(time.time() - t0))
        try:
            out.update(_analyze_md(md, workdir, distro))
        except Exception as e:  # noqa: BLE001
            out["analysis_failed"] = f"{type(e).__name__}: {e}"[:300]

    out["status"] = "success"
    out["elapsed_s"] = round(time.time() - t0, 2)
    return out


if __name__ == "__main__":
    run_driver(Path(sys.argv[1]), compute)
