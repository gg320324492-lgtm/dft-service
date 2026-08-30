"""Gaussian driver — rdkit 3D → .gjf (含 SCRF 溶剂!) → g16.exe → parse log

修复 microbubble 版缺口 #1: solvent 只写 title 不生效。
现在 solvent 默认 none (气相), 指定时真正写 SCRF=(SMD,Solvent=XXX) 路由关键字。
修复缺口 #2: charge/multiplicity/nproc/mem 全部透传。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from _driver_common import load_params, run_driver

sys.path.insert(0, "E:/sci-software/workflows")  # noqa: S104 — submit_gjf/parse_log 复用


# Gaussian 16 SMD 支持的溶剂名 → 规范写法 (覆盖常用; 未列出的按首字母大写透传)
_SOLVENT_MAP = {
    "water": "Water", "h2o": "Water",
    "ethanol": "Ethanol", "etoh": "Ethanol",
    "methanol": "Methanol", "meoh": "Methanol",
    "acetone": "Acetone",
    "acetonitrile": "Acetonitrile", "meCN": "Acetonitrile",
    "dmso": "DMSO",
    "dichloromethane": "Dichloromethane", "dcm": "Dichloromethane",
    "chloroform": "Chloroform",
    "toluene": "Toluene",
    "hexane": "n-Hexane", "n-hexane": "n-Hexane",
    "benzene": "Benzene",
    "thf": "THF",
    "dmf": "DMF",
    "ammonia": "Ammonia",
    "octanol": "n-Octanol", "n-octanol": "n-Octanol",
}
_SOLVENT_NONE = {"none", "gas", "gasphase", "vacuum", ""}


def _smiles_to_coords(smiles: str):
    """rdkit SMILES → ([元素], [(x,y,z), ...]) — ETKDGv3 + MMFF/UFF"""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise RuntimeError(f"Failed to parse SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) != 0:
        raise RuntimeError(f"Failed to embed 3D: {smiles}")
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        AllChem.UFFOptimizeMolecule(mol, maxIters=500)
    conf = mol.GetConformer()
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    coords = [
        (conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z)
        for i in range(mol.GetNumAtoms())
    ]
    return atoms, coords


def _infer_charge_mult(smiles: str) -> tuple[int, int]:
    """从 SMILES 推断 (电荷, 自旋多重度) — 与 pyscf geom_driver 同逻辑"""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise RuntimeError(f"Failed to parse SMILES: {smiles}")
    charge = Chem.GetFormalCharge(mol)
    n_radical = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
    multiplicity = 1 + 2 * n_radical if n_radical > 0 else 1
    return charge, multiplicity


def _gen_gjf(workdir: Path, smiles: str, p: dict) -> Path:
    """自建 gjf 生成 (支持 SCRF) — 不用 workflows.gen_gjf 因为它 route 写死"""
    atoms, coords = _smiles_to_coords(smiles)

    # job 可能归一化为空串 (单点是 Gaussian 默认), 过滤防双空格
    route_parts = [f"{p['xc']}/{p['basis']}"]
    if p.get("job"):
        route_parts.append(p["job"])
    solvent = (p.get("solvent") or "none").strip()
    if solvent.lower() not in _SOLVENT_NONE:
        g16_name = _SOLVENT_MAP.get(solvent.lower(), solvent.capitalize())
        route_parts.append(f"SCRF=(SMD,Solvent={g16_name})")
    route = "# " + " ".join(route_parts)

    stem = "input"
    lines = [
        f"%nproc={p.get('nproc', 8)}",
        f"%mem={p.get('mem', '8GB')}",
        f"%chk={stem}.chk",
        route,
        "",
        f"{smiles} {p['xc']}/{p['basis']} {p['job']} solvent={solvent}",
        "",
        f"{p['charge']} {p['multiplicity']}",
    ]
    lines += [
        f"{a:2s}  {c[0]:14.8f}  {c[1]:14.8f}  {c[2]:14.8f}"
        for a, c in zip(atoms, coords)
    ]
    gjf_path = workdir / f"{stem}.gjf"
    gjf_path.write_text("\n".join(lines) + "\n\n", encoding="utf-8")
    return gjf_path


def compute(params: dict, workdir: Path) -> dict:
    from gaussian_runner import parse_log, submit_gjf  # E:\sci-software\workflows

    t0 = time.time()

    # 缺口 #2: charge/multiplicity 未提供时从 SMILES 推断 —
    # 旧行为默默按 0/1 跑, 阳离子/自由基返回看似成功的错误能量
    charge = params.get("charge")
    multiplicity = params.get("multiplicity")
    if charge is None or multiplicity is None:
        inf_charge, inf_mult = _infer_charge_mult(params["smiles"])
        charge = int(charge) if charge is not None else inf_charge
        multiplicity = int(multiplicity) if multiplicity is not None else inf_mult
    params = {**params, "charge": int(charge), "multiplicity": int(multiplicity)}

    gjf_path = _gen_gjf(workdir, params["smiles"], params)

    gaussian_bin = params.get("gaussian_bin")
    log_path = submit_gjf(
        gjf_path, gaussian_path=gaussian_bin, timeout=float(params.get("timeout_s", 7200)),
    )
    parsed = parse_log(log_path)
    elapsed = time.time() - t0

    result = {
        "status": "success" if parsed.converged else "completed_with_warnings",
        "energy_hartree": parsed.energy_hartree,
        "energy_ev": parsed.energy_ev,
        "n_opt_steps": parsed.n_opt_steps,
        "converged": parsed.converged,
        "charge": params["charge"],
        "multiplicity": params["multiplicity"],
        "log_path": str(log_path),
        "gjf_path": str(gjf_path),
        "work_dir": str(workdir),
        "smiles": params["smiles"],
        "xc": params["xc"],
        "basis": params["basis"],
        "job": params["job"],
        "solvent": params.get("solvent", "none"),
        "elapsed_s": round(elapsed, 2),
        "extra": parsed.extra or {},
    }
    if parsed.error_msg:
        result["error_msg"] = parsed.error_msg
        result["status"] = "failed"
    return result


if __name__ == "__main__":
    run_driver(Path(sys.argv[1]), compute)
