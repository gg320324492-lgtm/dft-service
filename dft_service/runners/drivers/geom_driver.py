"""几何生成 driver — SMILES → input.xyz + 推断 charge/mult

独立成段: 让 PySCF 计算可以在没有 rdkit 的环境跑 (如 WSL python3 有 pyscf 无 rdkit)。
"""
from __future__ import annotations

import sys
from pathlib import Path

from _driver_common import load_params, run_driver


def compute(params: dict, workdir: Path) -> dict:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    smiles = params["smiles"]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise RuntimeError(f"Invalid SMILES: {smiles}")
    charge = Chem.GetFormalCharge(mol)
    n_radical = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
    multiplicity = 1 + 2 * n_radical if n_radical > 0 else 1

    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) != 0:
        raise RuntimeError(f"Failed to embed 3D: {smiles}")
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        AllChem.UFFOptimizeMolecule(mol, maxIters=500)

    n_atoms = mol.GetNumAtoms()
    conf = mol.GetConformer()
    with open(workdir / "input.xyz", "w", encoding="utf-8") as f:
        f.write(f"{n_atoms}\n\n")
        for atom in mol.GetAtoms():
            p = conf.GetAtomPosition(atom.GetIdx())
            f.write(f"{atom.GetSymbol()} {p.x:.8f} {p.y:.8f} {p.z:.8f}\n")

    return {
        "tool": "geom",
        "smiles": smiles,
        "n_atoms": n_atoms,
        "charge": charge,
        "multiplicity": multiplicity,
        "xyz_path": str(workdir / "input.xyz"),
    }


if __name__ == "__main__":
    run_driver(Path(sys.argv[1]), compute)
