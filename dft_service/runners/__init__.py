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
        "analyze_options": p.get("analyze_options"),  # 缺口 #34 分析项
        "water_model": p.get("water_model", "demo"),  # 遗留修复: spce 真水路径
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


_HA2KJ_MOL = 2625.499639  # Hartree → kJ/mol


async def run_reaction(task_id: str, p: dict[str, Any], timeout_s: float) -> dict:
    """缺口 #33: 反应能垒工作流编排 — 各物种 gaussian opt freq (#27) → ΔE/ΔG 汇总

    reactants/products 各带计量数; 子任务独立 workdir (gaussian_<subid>),
    父任务取消级联杀子 (executor.register_children)。
    try_ts=true 时额外提交 QST2 复合任务 (一键尝试, 失败如实报告)。
    """
    import asyncio

    from dft_service.models import new_task_id
    from dft_service.runners.executor import (
        execute_driver, make_workdir, register_children, unregister_children)
    from dft_service.runners.tool_definitions import availability_map

    if not availability_map().get("gaussian"):
        return {"status": "unavailable",
                "error_msg": "反应工作流需要 Gaussian (opt freq 热化学链) — "
                             "/dft/tools 里 gaussian 不可用"}

    sides: dict[str, list[dict]] = {
        "reactants": list(p.get("reactants") or []),
        "products": list(p.get("products") or []),
    }
    child_ids: list[str] = []
    register_children(task_id, child_ids)

    async def one_species(spec: dict, side: str, idx: int) -> dict:
        sub_id = new_task_id()
        child_ids.append(sub_id)  # gather 追加, 取消级联用 (list 写原子性足够)
        gp = {
            "xc": p.get("xc", "B3LYP"), "basis": p.get("basis", "6-31G(d)"),
            "job": "opt freq", "solvent": p.get("solvent", "none"),
            "charge": spec.get("charge"), "multiplicity": spec.get("multiplicity"),
            "nproc": p.get("nproc", 8), "mem": p.get("mem", "8GB"),
            "timeout_s": p.get("species_timeout_s") or max(timeout_s / 4, 600),
        }
        if spec.get("smiles"):
            gp["smiles"] = spec["smiles"]
        else:
            gp["xyz_content"] = spec["xyz_content"]
        res = await run_gaussian(sub_id, gp, gp["timeout_s"])
        thermo = (res.get("thermochemistry") or {})
        out = {
            "side": side, "index": idx,
            "label": spec.get("label") or ("reac" if side == "reactants" else "prod") + str(idx),
            "count": int(spec.get("count", 1)),
            "sub_task_id": sub_id,
            "status": res.get("status"),
            "energy_hartree": res.get("energy_hartree"),
            "gibbs_hartree": thermo.get("sum_elec_thermal_gibbs_hartree"),
            "n_imaginary": res.get("n_imaginary"),
            "smiles": res.get("smiles"),
            "error_msg": res.get("error_msg"),
        }
        return out

    jobs = [one_species(s, side, i)
            for side, lst in sides.items() for i, s in enumerate(lst)]
    try:
        species_results = await asyncio.gather(*jobs)
    finally:
        unregister_children(task_id)

    failed = [s for s in species_results
              if s["status"] not in ("success", "completed_with_warnings")]
    result: dict[str, Any] = {
        "tool": "reaction", "smiles": " + ".join(
            f"{s['count']}*{s['smiles'] or s['label']}" for s in species_results),
        "species": species_results,
        "xc": p.get("xc"), "basis": p.get("basis"),
        "solvent": p.get("solvent", "none"),
        "work_dir": str(settings.output_root / f"reaction_{task_id}"),
    }
    if failed:
        result["status"] = "failed"
        result["error_msg"] = "以下物种计算失败: " + "; ".join(
            f"{s['label']}({s['status']}: {(s['error_msg'] or '')[:100]})"
            for s in failed)
        return result

    e_reac = sum(s["count"] * s["energy_hartree"]
                 for s in species_results if s["side"] == "reactants")
    e_prod = sum(s["count"] * s["energy_hartree"]
                 for s in species_results if s["side"] == "products")
    de = e_prod - e_reac
    result["delta_electronic_hartree"] = round(de, 6)
    result["delta_electronic_kj_mol"] = round(de * _HA2KJ_MOL, 2)
    warnings = []
    if all(s.get("gibbs_hartree") is not None for s in species_results):
        g_reac = sum(s["count"] * s["gibbs_hartree"]
                     for s in species_results if s["side"] == "reactants")
        g_prod = sum(s["count"] * s["gibbs_hartree"]
                     for s in species_results if s["side"] == "products")
        dg = g_prod - g_reac
        result["delta_gibbs_hartree"] = round(dg, 6)
        result["delta_gibbs_kj_mol"] = round(dg * _HA2KJ_MOL, 2)
    else:
        warnings.append("部分物种缺热化学 Gibbs (freq 段解析失败?) — 只报电子能差")
    for s in species_results:
        if s.get("n_imaginary"):
            warnings.append(f"{s['label']} 有 {s['n_imaginary']} 个虚频 "
                            f"(非极小值, 热化学量按极小值假设)")
        if s["status"] == "completed_with_warnings":
            warnings.append(f"{s['label']} 计算带警告 (SCC 未收敛?)")
    result["warnings"] = warnings
    result["status"] = "success"

    # ---- TS 尝试 (可选; 一键 qst2, 诚实报告) ----
    if p.get("try_ts"):
        ts = await _try_qst2(task_id, p, species_results, register_children,
                             child_ids)
        result["ts_attempt"] = ts
    return result


async def _try_qst2(task_id, p, species_results, register_children, child_ids) -> dict:
    """QST2 一键尝试 — 需要反应物/产物各恰一种且总电荷/自旋两侧一致。

    几何来源: xyz 物种直接用; SMILES 物种先跑 geom_driver 内联嵌入。
    成功率对体系极敏感 (设计稿明说), 失败原样回 status + 日志路径。
    """
    from dft_service.models import new_task_id
    from dft_service.runners.executor import execute_driver, make_workdir

    reac = [s for s in species_results if s["side"] == "reactants"]
    prod = [s for s in species_results if s["side"] == "products"]
    if len(reac) != 1 or len(prod) != 1:
        return {"status": "skipped",
                "reason": "QST2 需要反应物/产物各恰一种 (多物种请手动拼复合物)"}
    spec_a, spec_b = reac[0], prod[0]
    if spec_a.get("count", 1) != 1 or spec_b.get("count", 1) != 1:
        return {"status": "skipped", "reason": "QST2 不支持计量数 ≠1"}
    xyz_a = (p.get("reactants") or [{}])[0].get("xyz_content")
    xyz_b = (p.get("products") or [{}])[0].get("xyz_content")
    try:
        if not xyz_a:
            xyz_a = await _embed_smiles(spec_a["smiles"], task_id)
        if not xyz_b:
            xyz_b = await _embed_smiles(spec_b["smiles"], task_id)
    except Exception as e:  # noqa: BLE001
        return {"status": "failed", "reason": f"几何嵌入失败: {e}"}
    sub_id = new_task_id()
    child_ids.append(sub_id)
    gp = {
        "smiles": None, "xyz_content": xyz_a, "xyz_b_content": xyz_b,
        "xc": p.get("xc", "B3LYP"), "basis": p.get("basis", "6-31G(d)"),
        "job": "qst2", "solvent": p.get("solvent", "none"),
        "charge": (p.get("reactants") or [{}])[0].get("charge") or 0,
        "multiplicity": (p.get("reactants") or [{}])[0].get("multiplicity") or 1,
        "nproc": p.get("nproc", 8), "mem": p.get("mem", "8GB"),
        "timeout_s": max(p.get("species_timeout_s") or 3600, 600),
    }
    res = await execute_driver(
        "gaussian", make_workdir("gaussian", sub_id),
        "gaussian_driver.py", gp, gp["timeout_s"] + 120, task_id=sub_id)
    return {
        "status": res.get("status"), "sub_task_id": sub_id,
        "energy_hartree": res.get("energy_hartree"),
        "note": "QST2 给 TS 能量估计; 确认需对该几何单独跑 freq (应恰 1 虚频)",
        "log_path": res.get("log_path"), "error_msg": res.get("error_msg"),
    }


async def _embed_smiles(smiles: str, parent_id: str) -> str:
    """SMILES → xyz 文本 (scichem geom_driver, 同 pyscf Stage A)"""
    from dft_service.models import new_task_id
    from dft_service.runners.executor import execute_driver, make_workdir

    sub_id = new_task_id()
    res = await execute_driver(
        "pyscf", make_workdir("pyscf", f"{parent_id}_emb{sub_id[:6]}"),
        "geom_driver.py", {"smiles": smiles}, 120, task_id=sub_id)
    if res.get("status") != "success":
        raise RuntimeError(res.get("error_msg") or "geom_driver failed")
    from pathlib import Path as _P
    return _P(res["xyz_path"]).read_text(encoding="utf-8")


async def run_conformers(task_id: str, p: dict[str, Any], timeout_s: float) -> dict:
    """缺口 #32: 构象搜索 — ETKDG 多构象 → MACE 弛豫排序 (scichem 内一把做完)"""
    from dft_service.runners.tool_definitions import availability_map

    if not availability_map().get("mace"):
        return {
            "status": "unavailable",
            "error_msg": "构象搜索需要 scichem 的 mace+rdkit (当前不可用) — "
                         "先确保 /dft/tools 里 mace ✓",
        }
    params = {
        "smiles": p["smiles"],
        "n_conformers": int(p.get("n_conformers", 20)),
        "top_k": int(p.get("top_k", 5)),
        "fmax_ev_A": float(p.get("fmax_ev_A", 0.05)),
        "max_steps": int(p.get("max_steps", 100)),
        "model": p.get("model", "medium"),
        "device": p.get("device", "auto"),
    }
    return await execute_driver(
        "conformers", make_workdir("conformers", task_id),
        "conformer_driver.py", params, timeout_s, task_id=task_id,
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
