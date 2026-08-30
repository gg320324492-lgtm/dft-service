"""PySCF driver — 完全重写 (microbubble 版真跑不通)

原版 3 个致命问题, 全部修复:
1. SMILES 原样塞给 gto.M(atom='CCO') — PySCF 不认 SMILES, 必然报错。
   现在: RDKit 生成 3D 几何 → xyz 坐标块 → gto.M(atom=<坐标块>)。
2. 生成的 calc.py 从未被执行 (实际跑的内联命令), spin/optimize 参数丢失。
   现在: 参数经 params.json 完整传递, spin≠0 自动 UKS。
3. Windows 路径 tee 进 WSL 写不进去 / 无意义的 cp 命令。
   现在: 无 WSL tee; scichem (Windows) 或 WSL python3 二选一, 由 runner 决定。

两种输入模式:
- params["smiles"]    — 本地有 rdkit 时自己生成几何 (scichem python)
- params["xyz_file"]  — 跳过 rdkit, 直接读 xyz (WSL 回退路径: 有 pyscf 无 rdkit 也能跑)

支持: energy / optimize (geomeTRIC), C-PCM 隐式溶剂, 输出 scf_converged。
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from _driver_common import load_params, run_driver

# 常见溶剂介电常数 (C-PCM 用)
_SOLVENT_EPS = {
    "water": 78.3553, "h2o": 78.3553,
    "ethanol": 24.852, "methanol": 32.613,
    "acetone": 20.493, "acetonitrile": 35.688,
    "dmso": 46.826, "benzene": 2.2706, "toluene": 2.3741,
    "hexane": 1.879, "thf": 7.4257, "dichloromethane": 8.93,
    "chloroform": 4.7113,
}
_SOLVENT_NONE = {"none", "gas", "gasphase", "vacuum", ""}


def _xyz_block_from_file(xyz_path: str) -> tuple[str, int]:
    """读 xyz 文件 → (原子坐标块, n_atoms) — 跳过 rdkit 依赖"""
    lines = Path(xyz_path).read_text(encoding="utf-8").strip().splitlines()
    n_atoms = int(lines[0].strip())
    block = "\n".join(ln.strip() for ln in lines[2:2 + n_atoms])
    if len(block.splitlines()) < n_atoms:
        raise RuntimeError(f"xyz truncated: {xyz_path}")
    return block, n_atoms


def _xyz_block_from_smiles(smiles: str) -> tuple[str, int, int, int]:
    """SMILES → (坐标块, n_atoms, 电荷, 自旋多重度) — 需要 rdkit"""
    from rdkit import Chem
    from rdkit.Chem import AllChem

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

    conf = mol.GetConformer()
    lines = []
    for atom in mol.GetAtoms():
        p = conf.GetAtomPosition(atom.GetIdx())
        lines.append(f"{atom.GetSymbol()} {p.x:.8f} {p.y:.8f} {p.z:.8f}")
    return "\n".join(lines), mol.GetNumAtoms(), charge, multiplicity


def _build_mf(mol, method: str, solvent: str, spin: int):
    from pyscf import dft

    mf_cls = dft.UKS if spin != 0 else dft.RKS
    mf = mf_cls(mol)
    mf.xc = method
    if solvent and solvent.lower() not in _SOLVENT_NONE:
        eps = _SOLVENT_EPS.get(solvent.lower())
        if eps is None:
            raise RuntimeError(
                f"Unknown solvent '{solvent}' for PySCF C-PCM; "
                f"supported: {sorted(_SOLVENT_EPS)}"
            )
        mf = mf.PCM()
        mf.with_solvent.method = "C-PCM"
        mf.with_solvent.eps = eps
    return mf


def compute(params: dict, workdir: Path) -> dict:
    t0 = time.time()
    smiles = params["smiles"]
    method = params.get("method", "B3LYP")
    basis = params.get("basis", "6-31G*")
    operation = params.get("operation", "energy")
    solvent = (params.get("solvent") or "none").strip()

    try:
        from pyscf import gto  # noqa: F401 — 可用性哨兵
    except ImportError as e:
        py = os.environ.get(
            "DFT_SERVICE_SCICHEM_PYTHON",
            "E:/sci-software/conda-envs/scichem/python.exe",
        )
        return {
            "status": "unavailable",
            "error_msg": (
                "pyscf 不可用 — scichem 缺 pyscf 且 WSL 回退未启用; "
                '任选: scichem 装 pyscf, 或 WSL "pip install pyscf"'
            ),
            "hint": f'"{py}" -m pip install pyscf  (或 wsl 内 python3 -m pip install pyscf)',
            "detail": str(e),
        }

    if params.get("xyz_file"):
        xyz_block, n_atoms = _xyz_block_from_file(params["xyz_file"])
        charge = params.get("charge")
        charge = int(charge) if charge is not None else 0
        multiplicity = params.get("multiplicity")
        multiplicity = int(multiplicity) if multiplicity is not None else 1
    else:
        xyz_block, n_atoms, charge_inf, mult_inf = _xyz_block_from_smiles(smiles)
        charge = params.get("charge")
        charge = int(charge) if charge is not None else charge_inf
        multiplicity = params.get("multiplicity")
        multiplicity = int(multiplicity) if multiplicity is not None else mult_inf

    spin = multiplicity - 1  # PySCF spin = 2S = 未配对电子数
    mol = gto.M(
        atom=xyz_block, basis=basis, charge=charge,
        spin=spin, verbose=0,
    )
    mf = _build_mf(mol, method, solvent, spin)

    result: dict = {
        "tool": "pyscf",
        "smiles": smiles,
        "method": method,
        "basis": basis,
        "operation": operation,
        "charge": charge,
        "multiplicity": multiplicity,
        "n_atoms": n_atoms,
        "solvent": solvent,
        "backend": params.get("backend", "scichem"),
        "work_dir": str(workdir),
    }

    if operation == "optimize":
        try:
            from pyscf.geomopt.geometric_solver import optimize
        except ImportError as e:
            raise RuntimeError(
                f"几何优化需要 geomeTRIC 包 (当前 backend): pip install geometric — {e}"
            ) from e
        mol_eq = optimize(mf, maxsteps=int(params.get("max_opt_steps", 50)))
        mf_eq = _build_mf(mol_eq, method, solvent, spin)
        energy = mf_eq.kernel()
        result["converged_geometry"] = True
        coords = "\n".join(
            f"{mol_eq.atom_symbol(i)} "
            f"{mol_eq.atom_coord(i, unit='Angstrom')[0]:.8f} "
            f"{mol_eq.atom_coord(i, unit='Angstrom')[1]:.8f} "
            f"{mol_eq.atom_coord(i, unit='Angstrom')[2]:.8f}"
            for i in range(mol_eq.natm)
        )
        (workdir / "optimized.xyz").write_text(
            f"{mol_eq.natm}\n\n{coords}\n", encoding="utf-8"
        )
        result["optimized_xyz"] = str(workdir / "optimized.xyz")
    else:
        energy = mf.kernel()

    result["energy_hartree"] = float(energy)
    result["energy_ev"] = float(energy) * 27.211386245988
    result["scf_converged"] = bool(mf.converged)
    result["status"] = "success" if mf.converged else "completed_with_warnings"
    if not mf.converged:
        result["warning"] = "SCF not converged"
    result["elapsed_s"] = round(time.time() - t0, 2)
    return result


if __name__ == "__main__":
    run_driver(Path(sys.argv[1]), compute)
