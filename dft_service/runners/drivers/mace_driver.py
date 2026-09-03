"""MACE-MP driver — SMILES → xyz → relax_trajectory (真实 converged/n_steps)

修复 microbubble 版缺口 #4: 非 trajectory 路径 converged 硬编码 True / n_steps 抄 max_steps。
现在统一走 mace_relaxation.relax_trajectory, 它返回真实的
{converged, n_steps, final_energy, trajectory_file}。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from _driver_common import load_params, parse_xyz_text, run_driver, write_progress

sys.path.insert(0, "E:/sci-software/workflows")  # noqa: S104


def _smiles_to_xyz(smiles: str, xyz_path: Path) -> int:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise RuntimeError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) != 0:
        raise RuntimeError(f"Failed to embed 3D: {smiles}")
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        AllChem.UFFOptimizeMolecule(mol, maxIters=500)
    with open(xyz_path, "w", encoding="utf-8") as f:
        f.write(f"{mol.GetNumAtoms()}\n\n")
        conf = mol.GetConformer()
        for atom in mol.GetAtoms():
            p = conf.GetAtomPosition(atom.GetIdx())
            f.write(f"{atom.GetSymbol()} {p.x:.6f} {p.y:.6f} {p.z:.6f}\n")
    return mol.GetNumAtoms()


def compute(params: dict, workdir: Path) -> dict:
    import mace_relaxation  # noqa: PLC0415
    from mace_relaxation import relax_trajectory  # noqa: PLC0415

    # 兼容 shim: workflows._get_calculator 假设本地有 mace_mp_medium.model 文件 +
    # MACECalculator.from_model API (mace 0.3.16 均不存在)。
    # 换成 0.3.x 标准工厂 mace_mp() — 它自己解析 medium/small/large 并命中
    # ~/.cache/mace 缓存; 优化/轨迹逻辑仍全部复用 mace_relaxation。
    import torch

    def _patched_get_calculator(model: str = "medium", device: str = "auto"):
        from mace.calculators import mace_mp

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return mace_mp(model=model, device=device, default_dtype="float64")

    mace_relaxation._get_calculator = _patched_get_calculator

    t0 = time.time()
    smiles = params.get("smiles")
    xyz_path = workdir / "input.xyz"
    if params.get("xyz_content"):
        # 缺口 #29: 内联几何 → 规范化重写 (ase.io.read 需要标准 2 行头)
        write_progress(workdir, "geometry", source="xyz_content")
        atoms, coords = parse_xyz_text(params["xyz_content"])
        with open(xyz_path, "w", encoding="utf-8") as f:
            f.write(f"{len(atoms)}\ninline-xyz\n")
            for a, c in zip(atoms, coords):
                f.write(f"{a} {c[0]:.8f} {c[1]:.8f} {c[2]:.8f}\n")
        n_atoms = len(atoms)
    else:
        write_progress(workdir, "geometry", smiles=smiles)
        n_atoms = _smiles_to_xyz(smiles, xyz_path)

    traj_path = workdir / "trajectory.extxyz"
    # 缺口 #22: relax_trajectory 是单次阻塞调用 (不改 workflows 拿不到 BFGS 每步),
    # 报"进入 relax"这一粒度; MACE 通常秒级, 足够
    write_progress(workdir, "relax", max_steps=int(params.get("max_steps", 200)),
                   elapsed_s=round(time.time() - t0))
    res = relax_trajectory(
        xyz_path, traj_path,
        fmax=float(params.get("fmax_ev_A", 0.05)),
        steps=int(params.get("max_steps", 200)),
        model=params.get("model", "medium"),
        device=params.get("device", "auto"),
    )
    elapsed = time.time() - t0

    out: dict = {
        "tool": "mace",
        "smiles": smiles or f"<inline-xyz:{n_atoms} atoms>",
        "n_atoms": n_atoms,
        "fmax_ev_A": params.get("fmax_ev_A", 0.05),
        "max_steps": params.get("max_steps", 200),
        "model": params.get("model", "medium"),
        "device": params.get("device", "auto"),
        "converged": bool(res["converged"]),
        "n_steps": int(res["n_steps"]),
        "energy_ev": float(res["final_energy"]),
        "trajectory_path": str(res["trajectory_file"]),
        "work_dir": str(workdir),
        "elapsed_s": round(elapsed, 2),
    }
    if not out["converged"]:
        out["status"] = "completed_with_warnings"
        out["warning"] = (
            f"not converged in {out['max_steps']} steps (fmax={out['fmax_ev_A']} eV/A)"
        )
    return out


if __name__ == "__main__":
    run_driver(Path(sys.argv[1]), compute)
