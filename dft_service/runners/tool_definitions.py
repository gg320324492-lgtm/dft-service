"""工具健康聚合 + auto 智能选路 — 补齐 microbubble 版 "multimodel 统一接口" 缺口"""
from __future__ import annotations

from typing import Any

from dft_service.runners import (
    health_gaussian,
    health_gromacs,
    health_mace,
    health_psi4,
    health_pyscf,
)


def availability_map() -> dict[str, bool]:
    """{tool: available} — available 判定与 health details 解耦"""
    g = health_gaussian()
    m = health_mace()
    p = health_pyscf()
    psi = health_psi4()
    gm = health_gromacs()
    return {
        "gaussian": bool(g.get("workflows_available") and g.get("g16_exists")
                         and g.get("scichem_rdkit")),
        "gromacs": bool(gm.get("wsl_gromacs") and gm.get("workflows_available")),
        "mace": bool(m.get("mace") and m.get("rdkit")),
        "pyscf": bool(p.get("backend") and p.get("rdkit")),
        "psi4": bool(psi.get("psi4") and psi.get("rdkit")),
    }


def list_available_tools() -> dict[str, Any]:
    healths = [
        health_gaussian(), health_gromacs(), health_mace(),
        health_pyscf(), health_psi4(),
    ]
    avail = availability_map()
    tools = [{"name": h["tool"], "available": avail[h["tool"]], "details": h}
             for h in healths]
    return {
        "status": "success",
        "tools": tools,
        "count": len(tools),
        "available_count": sum(1 for t in tools if t["available"]),
        "rich_block_type": "dft_tools",
    }


# ------------------------------------------------------------------
# auto 选路 (修复 "multimodel 统一接口" 只有名字没实现)
# ------------------------------------------------------------------
_TASK_ALIASES = {
    "energy": "energy", "sp": "energy", "single": "energy", "single_point": "energy",
    "optimize": "optimize", "opt": "optimize", "geometry": "optimize",
    "freq": "freq", "frequency": "freq", "frequencies": "freq",
    "properties": "properties", "prop": "properties",
    "md": "md", "dynamics": "md", "molecular_dynamics": "md",
}


def select_backend(task: str, quality: str = "auto") -> tuple[str | None, str]:
    """按任务类型 + 质量要求选后端 → (tool, 说明); 无可用后端 → (None, 原因)"""
    task = _TASK_ALIASES.get(task.strip().lower(), task.strip().lower())
    quality = (quality or "auto").strip().lower()
    avail = availability_map()

    if task == "md":
        if avail["gromacs"]:
            return "gromacs", "分子动力学 → GROMACS (WSL)"
        return None, "task=md 需要 GROMACS (当前 WSL gmx 不可用)"

    if task in ("energy", "optimize") and quality == "fast":
        if avail["mace"]:
            return "mace", "fast 模式 → MACE 机器学习力场 (秒级)"
        if avail["psi4"] or avail["pyscf"] or avail["gaussian"]:
            return _accurate_fallback(avail), "MACE 不可用, 回退量子化学后端"
        return None, "无任何可用后端 (mace/psi4/pyscf/gaussian 全不可用)"

    # accurate / auto / freq / properties → 量子化学
    tool = _accurate_fallback(avail)
    if tool:
        why = {
            "gaussian": "Gaussian 16W (商业金标准, SMD 溶剂支持)",
            "psi4": "Psi4 1.9.1 (开源, 支持 properties: 偶极/HOMO-LUMO)",
            "pyscf": "PySCF (纯开源 BSD)",
        }[tool]
        return tool, f"量子化学任务 → {why}"
    return None, "无任何可用量子化学后端 (psi4/pyscf/gaussian 全不可用)"


def _accurate_fallback(avail: dict[str, bool]) -> str | None:
    for t in ("gaussian", "psi4", "pyscf"):
        if avail[t]:
            return t
    return None
