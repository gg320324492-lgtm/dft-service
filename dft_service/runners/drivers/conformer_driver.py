"""构象搜索 driver (缺口 #32) — ETKDG 多构象 → MMFF 预优化 → MACE 弛豫排序

微纳米气泡课题常用: 溶液中分子的有效构象集 + 相对能量 (kJ/mol)。
纯编排现有能力 (rdkit + mace 同在 scichem), 不动 workflows。

params: smiles, n_conformers(≤50), top_k(≤10), fmax_ev_A, max_steps, model, device
产物: workdir/conformers/conf_<rank>.xyz (top_k) + result.conformers 列表
排序键 = MACE 单点弛豫能; rel_kj_mol 相对全局最低构象。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from _driver_common import load_params, run_driver, write_progress

sys.path.insert(0, "E:/sci-software/workflows")  # noqa: S104 — 借 mace_relaxation 的 env 预处理

_EV_PER_KJ_MOL = 96.48533212


def _embed_conformers(smiles: str, n: int):
    """SMILES → (mol, [cid]) — ETKDGv3 多构象 + MMFF/UFF 预优化 (纯 rdkit, 可单测)"""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise RuntimeError(f"Failed to parse SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    ep = AllChem.ETKDGv3()
    ep.randomSeed = 0xF00D
    cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=n, params=ep))
    if not cids:
        raise RuntimeError(f"ETKDG embedding produced 0 conformers for {smiles}")
    try:
        AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=200)
    except Exception:
        AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=200)
    return mol, cids


def _conf_to_atoms(mol, cid):
    """rdkit conformer → ase.Atoms (含 pbc=False)"""
    import numpy as np
    from ase import Atoms

    conf = mol.GetConformer(cid)
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    positions = np.array([[conf.GetAtomPosition(i).x,
                           conf.GetAtomPosition(i).y,
                           conf.GetAtomPosition(i).z]
                          for i in range(mol.GetNumAtoms())])
    return Atoms(symbols=symbols, positions=positions, pbc=False)


def compute(params: dict, workdir: Path) -> dict:
    t0 = time.time()
    smiles = params["smiles"]
    n_conf = int(params.get("n_conformers", 20))
    top_k = int(params.get("top_k", 5))
    fmax = float(params.get("fmax_ev_A", 0.05))
    steps = int(params.get("max_steps", 100))
    model = params.get("model", "medium")
    device = params.get("device", "auto")

    try:
        import torch  # noqa: F401
        from mace.calculators import mace_mp  # noqa: F401
    except ImportError as e:
        return {"status": "unavailable",
                "error_msg": f"构象搜索需要 scichem 环境有 mace+torch: {e}",
                "hint": "pip install mace-torch (scichem 环境)"}

    write_progress(workdir, "embed", n_conformers=n_conf)
    mol, cids = _embed_conformers(smiles, n_conf)

    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    calc = mace_mp(model=model, device=device, default_dtype="float64")

    from ase.optimize import BFGS

    cdir = workdir / "conformers"
    cdir.mkdir(exist_ok=True)
    relaxed: list[dict] = []
    for i, cid in enumerate(cids, 1):
        write_progress(workdir, "relax", conf=i, n_total=len(cids),
                       elapsed_s=round(time.time() - t0))
        atoms = _conf_to_atoms(mol, cid)
        atoms.calc = calc
        try:
            dyn = BFGS(atoms, logfile=None)
            conv = dyn.run(fmax=fmax, steps=steps)
        except Exception:  # noqa: BLE001 — 单构象炸不拖垮整批
            conv = False
        if not conv or atoms.get_positions().shape != (mol.GetNumAtoms(), 3):
            continue
        xyz_path = cdir / f"conf_{i:03d}.xyz"
        atoms.write(str(xyz_path))
        relaxed.append({
            "embed_index": i,
            "energy_ev": float(atoms.get_potential_energy()),
            "n_steps": int(dyn.nsteps),
            "xyz_path": str(xyz_path),
            "smiles": smiles,
        })
    if not relaxed:
        return {"status": "failed",
                "error_msg": f"全部 {len(cids)} 个构象弛豫失败"}

    relaxed.sort(key=lambda r: r["energy_ev"])
    emin = relaxed[0]["energy_ev"]
    for rank, r in enumerate(relaxed[:top_k], 1):
        r["rank"] = rank
        r["rel_kj_mol"] = round((r["energy_ev"] - emin) * _EV_PER_KJ_MOL, 2)
        # 重命名为 rank 序号 (稳定可读)
        newp = cdir / f"top_{rank}.xyz"
        Path(r["xyz_path"]).replace(newp)
        r["xyz_path"] = str(newp)
    for r in relaxed[top_k:]:  # 落选构象文件删掉, 只留 top_k
        try:
            Path(r.pop("xyz_path")).unlink()
        except OSError:
            pass

    return {
        "status": "success",
        "tool": "conformers",
        "smiles": smiles,
        "model": model,
        "device": device,
        "n_embedded": len(cids),
        "n_relaxed": len(relaxed),
        "top_k": min(top_k, len(relaxed)),
        "n_atoms": len(mol.GetAtoms()),
        "conformers": relaxed[:top_k],
        "all_energies_ev": [r["energy_ev"] for r in relaxed],
        "work_dir": str(workdir),
        "elapsed_s": round(time.time() - t0, 2),
    }


if __name__ == "__main__":
    run_driver(Path(sys.argv[1]), compute)
