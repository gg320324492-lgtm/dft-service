"""FastAPI 路由 — 端点形状与 microbubble 版兼容 (前端/代理无感迁移)

- GET  /dft/tools              健康检查 (免鉴权)
- POST /dft/gaussian           Gaussian DFT (solvent 真生效 + charge/mult/nproc/mem)
- POST /dft/gromacs            GROMACS MD (WSL 发行版自动探测)
- POST /dft/mace               MACE 快速优化 (真实 converged/n_steps)
- POST /dft/pyscf              PySCF (RDKit 3D + C-PCM 溶剂 + UKS 开壳层)
- POST /dft/psi4               Psi4 (energy/optimize/properties)
- POST /dft/auto               智能选路 → 派发到最快可用后端
- GET  /dft/status/{task_id}   状态 (内存 miss → DB 回退, 重启不丢)
- GET  /dft/result/{task_id}   结果 (同上)
- GET  /dft/jobs               任务列表 (tool/status 过滤 + 分页)
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from dft_service import taskstore
from dft_service.auth import require_api_key
from dft_service.runners import (
    run_gaussian,
    run_gromacs,
    run_mace,
    run_psi4,
    run_pyscf,
)
from dft_service.runners.tool_definitions import list_available_tools, select_backend

logger = logging.getLogger("dft_service.api")

router = APIRouter(prefix="/dft", tags=["DFT/MD 计算"])

RUNNERS = {
    "gaussian": run_gaussian,
    "gromacs": run_gromacs,
    "mace": run_mace,
    "pyscf": run_pyscf,
    "psi4": run_psi4,
}

# task → Gaussian 路由关键字 (SP 是文档化合法关键字; properties 无独立关键字, 用 sp)
_GAUSS_JOB = {"energy": "sp", "optimize": "opt", "opt": "opt", "freq": "freq",
              "frequency": "freq", "properties": "sp", "prop": "sp"}


# ------------------------------------------------------------------
# request / response models
# ------------------------------------------------------------------
class GaussianRequest(BaseModel):
    smiles: str = Field(..., min_length=1)
    xc: str = "B3LYP"
    basis: str = "6-31G(d)"
    job: str = Field("opt", description="opt / sp / freq")
    solvent: str = Field("none", description="SCRF 溶剂 (water/ethanol/...; none=气相)")
    charge: Optional[int] = Field(None, ge=-10, le=10,
                                  description="缺省从 SMILES 自动推断")
    multiplicity: Optional[int] = Field(None, ge=1, le=10,
                                        description="缺省从 SMILES 自动推断")
    nproc: int = Field(8, ge=1)
    mem: str = "8GB"
    timeout_s: float = Field(7200.0, ge=10.0)


class GromacsRequest(BaseModel):
    smiles: str = Field(..., min_length=1)
    n_molecules: int = Field(100, ge=1)
    box_nm: float = Field(3.0, gt=0.0)
    time_ns: float = Field(1.0, gt=0.0)
    temperature_K: float = Field(300.0, ge=0.0)
    timeout_s: float = Field(14400.0, ge=10.0)


class MaceRequest(BaseModel):
    smiles: str = Field(..., min_length=1)
    fmax_ev_A: float = Field(0.05, gt=0.0)
    max_steps: int = Field(200, ge=1)
    model: str = "medium"
    device: str = Field("auto", description="cuda / cpu / auto")
    timeout_s: float = Field(900.0, ge=10.0)


class PyscfRequest(BaseModel):
    smiles: str = Field(..., min_length=1)
    method: str = "B3LYP"
    basis: str = "6-31G*"
    operation: str = Field("energy", description="energy / optimize")
    solvent: str = Field("none", description="C-PCM 溶剂 (water/ethanol/...; none=气相)")
    charge: Optional[int] = Field(None, description="缺省从 SMILES 推断")
    multiplicity: Optional[int] = Field(None, description="缺省从 SMILES 推断")
    max_opt_steps: int = Field(50, ge=1)
    timeout_s: float = Field(1800.0, ge=10.0)


class Psi4Request(BaseModel):
    smiles: str = Field(..., min_length=1)
    method: str = "B3LYP"
    basis: str = "6-31G*"
    operation: str = Field("energy", description="energy / optimize / properties")
    charge: Optional[int] = None
    multiplicity: Optional[int] = None
    nproc: int = Field(8, ge=1)
    mem: str = "8GB"
    timeout_s: float = Field(3600.0, ge=10.0)


class AutoRequest(BaseModel):
    smiles: str = Field(..., min_length=1)
    task: str = Field("energy", description="energy / optimize / freq / properties / md")
    quality: str = Field("auto", description="fast (MACE) / accurate (量子化学) / auto")
    xc: str = "B3LYP"
    basis: str = "6-31G*"
    solvent: str = "none"
    charge: Optional[int] = None
    multiplicity: Optional[int] = None
    timeout_s: Optional[float] = None


class TaskIdResponse(BaseModel):
    task_id: str
    status: str
    submit_time: str
    tool: str


# ------------------------------------------------------------------
# 内部: 提交 + 后台执行
# ------------------------------------------------------------------
async def _execute(tool: str, task_id: str, p: dict[str, Any], timeout_s: float) -> None:
    try:
        result = await RUNNERS[tool](task_id, p, timeout_s)
    except Exception as e:  # noqa: BLE001 — 执行器兜底
        logger.exception("task %s (%s) crashed", task_id, tool)
        result = {"status": "failed", "error_msg": repr(e)}
    status = result.get("status", "failed")
    if status not in ("success", "failed", "unavailable", "completed_with_warnings"):
        status = "failed"
    await taskstore.finish_task(task_id, status, result, result.get("error_msg"))


async def _submit(
    tool: str, smiles: str, p: dict[str, Any],
    timeout_s: float, submitter: Optional[str],
) -> TaskIdResponse:
    rec = taskstore.create_task(tool, smiles, p, submitter)
    await taskstore.persist_new_task(rec)
    task_id = rec["task_id"]
    asyncio.create_task(_execute(tool, task_id, p, timeout_s))
    return TaskIdResponse(
        task_id=task_id, status="queued",
        submit_time=rec["submit_time"], tool=tool,
    )


# ------------------------------------------------------------------
# 端点
# ------------------------------------------------------------------
@router.get("/tools")
async def tools() -> dict:
    """5 工具健康状态 (免鉴权 — 探针友好)"""
    return list_available_tools()


@router.post("/gaussian", response_model=TaskIdResponse,
             dependencies=[Depends(require_api_key)])
async def submit_gaussian(req: GaussianRequest, submitter: Optional[str] = None):
    return await _submit("gaussian", req.smiles, req.model_dump(), req.timeout_s, submitter)


@router.post("/gromacs", response_model=TaskIdResponse,
             dependencies=[Depends(require_api_key)])
async def submit_gromacs(req: GromacsRequest, submitter: Optional[str] = None):
    return await _submit("gromacs", req.smiles, req.model_dump(), req.timeout_s, submitter)


@router.post("/mace", response_model=TaskIdResponse,
             dependencies=[Depends(require_api_key)])
async def submit_mace(req: MaceRequest, submitter: Optional[str] = None):
    return await _submit("mace", req.smiles, req.model_dump(), req.timeout_s, submitter)


@router.post("/pyscf", response_model=TaskIdResponse,
             dependencies=[Depends(require_api_key)])
async def submit_pyscf(req: PyscfRequest, submitter: Optional[str] = None):
    return await _submit("pyscf", req.smiles, req.model_dump(), req.timeout_s, submitter)


@router.post("/psi4", response_model=TaskIdResponse,
             dependencies=[Depends(require_api_key)])
async def submit_psi4(req: Psi4Request, submitter: Optional[str] = None):
    return await _submit("psi4", req.smiles, req.model_dump(), req.timeout_s, submitter)


@router.post("/auto", dependencies=[Depends(require_api_key)])
async def submit_auto(req: AutoRequest, submitter: Optional[str] = None):
    """智能选路: task+quality → 最快可用后端, 派发并返回 task_id + 选路理由"""
    tool, reason = select_backend(req.task, req.quality)
    if tool is None:
        return {
            "status": "unavailable",
            "reason": reason,
            "availability": list_available_tools(),
        }
    if req.task.lower() in ("freq", "frequency", "frequencies") and tool != "gaussian":
        return {
            "status": "unavailable",
            "reason": f"freq 任务当前仅 Gaussian 支持 (选中 {tool})",
        }

    task_norm = req.task.strip().lower()
    operation = "optimize" if task_norm in ("optimize", "opt", "geometry") else (
        "properties" if task_norm in ("properties", "prop") else "energy"
    )
    p: dict[str, Any] = {
        "smiles": req.smiles,
        "solvent": req.solvent,
        "charge": req.charge,
        "multiplicity": req.multiplicity,
    }
    if tool == "gaussian":
        # 缺口 #1 修复 (2026-08-30): req.task 原样进路由会把 "energy"/"optimize"
        # 拼成非法 Gaussian 关键字 → Error termination。按映射表归一化。
        p.update({
            "xc": req.xc, "basis": req.basis,
            "job": _GAUSS_JOB.get(task_norm, "sp"),
            "solvent": req.solvent,
        })
        timeout = req.timeout_s or 7200.0
    elif tool == "psi4":
        p.update({"method": req.xc, "basis": req.basis, "operation": operation})
        timeout = req.timeout_s or 3600.0
    elif tool == "pyscf":
        p.update({"method": req.xc, "basis": req.basis, "operation": operation})
        timeout = req.timeout_s or 1800.0
    elif tool == "mace":
        p = {"smiles": req.smiles}
        timeout = req.timeout_s or 900.0
    else:  # gromacs
        p = {"smiles": req.smiles}
        timeout = req.timeout_s or 14400.0

    resp = await _submit(tool, req.smiles, p, timeout, submitter)
    return {**resp.model_dump(), "backend": tool, "reason": reason}


@router.get("/status/{task_id}", dependencies=[Depends(require_api_key)])
async def task_status(task_id: str):
    rec = await taskstore.get_task(task_id, include_result=False)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"task {task_id} not found")
    return {
        "task_id": task_id,
        "status": rec.get("status"),
        "tool": rec.get("tool"),
        "submit_time": rec.get("submit_time"),
        "finish_time": rec.get("finish_time"),
    }


@router.get("/result/{task_id}", dependencies=[Depends(require_api_key)])
async def task_result(task_id: str):
    rec = await taskstore.get_task(task_id, include_result=True)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"task {task_id} not found")
    status = rec.get("status")
    if status in ("queued", "running"):
        return {"task_id": task_id, "status": status, "message": "task still running"}
    return {
        "task_id": task_id,
        "status": status,
        "result": rec.get("result"),
        "error_msg": rec.get("error_msg"),
        "submit_time": rec.get("submit_time"),
        "finish_time": rec.get("finish_time"),
    }


@router.get("/jobs", dependencies=[Depends(require_api_key)])
async def jobs(
    tool: Optional[str] = None, status: Optional[str] = None,
    limit: int = 50, offset: int = 0,
):
    """任务列表 (重启后仍可查 — 数据在 SQLite)"""
    return await taskstore.list_jobs(tool=tool, status=status, limit=limit, offset=offset)
