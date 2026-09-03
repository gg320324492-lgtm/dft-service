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
from pydantic import BaseModel, Field, model_validator

from dft_service import taskstore
from dft_service.auth import require_api_key
from dft_service.runners import (
    run_conformers,
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
    "conformers": run_conformers,  # 缺口 #32 工作流 (非独立后端, 走 mace)
}

# task → Gaussian 路由关键字 (SP 是文档化合法关键字; properties 无独立关键字, 用 sp)
# 缺口 #27: opt_freq 一次任务同时拿几何 + 频率/热化学量 (省一半机时)
_GAUSS_JOB = {"energy": "sp", "optimize": "opt", "opt": "opt", "freq": "freq",
              "frequency": "freq", "properties": "sp", "prop": "sp",
              "opt_freq": "opt freq", "opt+freq": "opt freq",
              "optfreq": "opt freq", "freqopt": "opt freq"}

# 缺口 #28: extra_route 白名单 — 只留 route 行合法字符, 挡换行/#/%/\ (gjf 注入面)。
# 允许 opt=calclevel、int=ultrafine、iop(6/7=...) 之类: 字母数字 = , ( ) + * . 空格制表
# (- 放末尾, 类内 . + ( ) 皆字面量; extra_route 用 * 允许留空)
_ROUTE_SAFE_RE = r"^[A-Za-z0-9=,()+.*\- \t]+$"
_ROUTE_SAFE_OR_EMPTY_RE = r"^[A-Za-z0-9=,()+.*\- \t]*$"


# ------------------------------------------------------------------
# request / response models
# ------------------------------------------------------------------
# 缺口 #20: 所有参数带上下限 — 防 timeout_s=10⁹ 类输入无限占住信号量。
# 上限按课题组机器 (16 核 / 单卡 GPU / WSL 4 线程) 的合理用量放宽。
_TIMEOUT_LE = 604800.0  # 7 天 (gromacs 长 MD)
_XYZ_MAX = 200_000      # 内联 xyz 尺寸上限 (~4000 原子)


class _CallbackMixin(BaseModel):
    """缺口 #37: callback_url — 任务终态时服务端 POST 回该 URL (fire-and-forget)"""
    callback_url: Optional[str] = Field(
        None, max_length=2000,
        description="任务完成/取消后 POST {task_id,status,tool,result,error_msg} 到此 URL")


class _XyzCapableMixin(BaseModel):
    """缺口 #29: smiles / xyz_content 二选一 (gaussian/pyscf/mace 支持内联几何)

    xyz 无 SMILES 可推 charge/multiplicity → 必须显式给 (mace 无此概念不含该检查)。
    """
    smiles: Optional[str] = Field(None, min_length=1, max_length=2000)
    xyz_content: Optional[str] = Field(None, max_length=_XYZ_MAX,
                                       description="内联 xyz 几何 (与 smiles 二选一)")

    @model_validator(mode="after")
    def _require_one_input(self):
        if bool(self.smiles) == bool(self.xyz_content):
            raise ValueError("smiles 与 xyz_content 二选一 (且仅一个)")
        return self


class _XyzNeedsChargeMixin(_XyzCapableMixin):
    @model_validator(mode="after")
    def _xyz_requires_charge(self):
        if not self.smiles and (self.charge is None or self.multiplicity is None):
            raise ValueError("xyz 输入无 SMILES 可推断, 请显式提供 charge 和 multiplicity")
        return self


class GaussianRequest(_CallbackMixin, _XyzNeedsChargeMixin):
    xc: str = Field("B3LYP", pattern=_ROUTE_SAFE_RE)
    basis: str = Field("6-31G(d)", pattern=_ROUTE_SAFE_RE)
    job: str = Field("opt", pattern=_ROUTE_SAFE_RE,
                     description="opt / sp / freq / opt freq (联跑一次拿几何+热化学)")
    solvent: str = Field("none", pattern=r"^[A-Za-z\-]{1,30}$",
                         description="SCRF 溶剂 (water/ethanol/...; none=气相)")
    extra_route: str = Field(
        "", pattern=_ROUTE_SAFE_OR_EMPTY_RE, max_length=200,
        description="追加进路由行的额外关键字 (缺口 #28 逃生舱): "
                    "int=ultrafine / scf=(qc,xqd) / geom=check / gpush 等; "
                    "换行/#/% 被白名单挡")
    charge: Optional[int] = Field(None, ge=-10, le=10,
                                  description="缺省从 SMILES 自动推断")
    multiplicity: Optional[int] = Field(None, ge=1, le=10,
                                        description="缺省从 SMILES 自动推断")
    nproc: int = Field(8, ge=1, le=64)
    mem: str = Field("8GB", pattern=r"^\d{1,5}(?i:MB|GB|TB)$")
    timeout_s: float = Field(7200.0, ge=10.0, le=_TIMEOUT_LE)


class GromacsRequest(_CallbackMixin):
    smiles: str = Field(..., min_length=1, max_length=2000)
    n_molecules: int = Field(100, ge=1, le=20000)
    box_nm: float = Field(3.0, gt=0.0, le=30.0)
    time_ns: float = Field(1.0, gt=0.0, le=1000.0)
    temperature_K: float = Field(300.0, ge=0.0, le=2000.0)
    analyze: bool = Field(False, description="缺口 #31: MD 后 gmx rms/energy "
                                             "统计 + PNG 出图")
    timeout_s: float = Field(14400.0, ge=10.0, le=_TIMEOUT_LE)


class MaceRequest(_CallbackMixin, _XyzCapableMixin):
    fmax_ev_A: float = Field(0.05, gt=0.0, le=10.0)
    max_steps: int = Field(200, ge=1, le=10000)
    model: str = "medium"
    device: str = Field("auto", description="cuda / cpu / auto")
    timeout_s: float = Field(900.0, ge=10.0, le=_TIMEOUT_LE)


class PyscfRequest(_CallbackMixin, _XyzNeedsChargeMixin):
    method: str = "B3LYP"
    basis: str = "6-31G*"
    operation: str = Field("energy", description="energy / optimize")
    solvent: str = Field("none", description="C-PCM 溶剂 (water/ethanol/...; none=气相)")
    charge: Optional[int] = Field(None, ge=-10, le=10, description="缺省从 SMILES 推断")
    multiplicity: Optional[int] = Field(None, ge=1, le=10, description="缺省从 SMILES 推断")
    max_opt_steps: int = Field(50, ge=1, le=1000)
    solvation_energy: bool = Field(
        False, description="缺口 #30: true 时同几何双算 (气相+C-PCM), "
                           "返回 delta_solvation_kj_mol; 需 solvent 非 none")
    timeout_s: float = Field(1800.0, ge=10.0, le=_TIMEOUT_LE)


class Psi4Request(_CallbackMixin):
    smiles: str = Field(..., min_length=1, max_length=2000)
    method: str = "B3LYP"
    basis: str = "6-31G*"
    operation: str = Field("energy", description="energy / optimize / properties")
    charge: Optional[int] = Field(None, ge=-10, le=10)
    multiplicity: Optional[int] = Field(None, ge=1, le=10)
    nproc: int = Field(8, ge=1, le=64)
    mem: str = Field("8GB", pattern=r"^\d{1,5}(?i:MB|GB|TB)$")
    timeout_s: float = Field(3600.0, ge=10.0, le=_TIMEOUT_LE)


class ConformersRequest(_CallbackMixin):
    """缺口 #32: 构象搜索 (ETKDG + MACE) — 返回 top_k 构象 xyz 与相对能量"""
    smiles: str = Field(..., min_length=1, max_length=2000)
    n_conformers: int = Field(20, ge=2, le=100)
    top_k: int = Field(5, ge=1, le=20)
    fmax_ev_A: float = Field(0.05, gt=0.0, le=10.0)
    max_steps: int = Field(100, ge=1, le=5000)
    model: str = "medium"
    device: str = Field("auto", description="cuda / cpu / auto")
    timeout_s: float = Field(1800.0, ge=10.0, le=_TIMEOUT_LE)


class AutoRequest(_CallbackMixin):
    smiles: str = Field(..., min_length=1, max_length=2000)
    task: str = Field("energy", description="energy / optimize / freq / properties / md")
    quality: str = Field("auto", description="fast (MACE) / accurate (量子化学) / auto")
    xc: str = "B3LYP"
    basis: str = "6-31G*"
    solvent: str = Field("none", pattern=r"^[A-Za-z\-]{1,30}$")  # 直通 gaussian SCRF
    charge: Optional[int] = Field(None, ge=-10, le=10)
    multiplicity: Optional[int] = Field(None, ge=1, le=10)
    timeout_s: Optional[float] = Field(None, ge=10.0, le=_TIMEOUT_LE)


class TaskIdResponse(BaseModel):
    task_id: str
    status: str
    submit_time: str
    tool: str


def _label(req) -> str:
    """smiles 或内联几何的展示标签 (DB smiles 列 NOT NULL)"""
    return getattr(req, "smiles", None) or "<inline-xyz>"


# ------------------------------------------------------------------
# 内部: 提交 + 后台执行
# ------------------------------------------------------------------
# 缺口 #8: 每工具并发上限 — mace 单并发防 GPU 显存争抢, 其余 2; 超出的
# 任务保持 queued 自动排队。auto 继承所选后端的限制。
_SEMAPHORES: dict[str, asyncio.Semaphore] = {
    "gaussian": asyncio.Semaphore(2),
    "gromacs": asyncio.Semaphore(2),
    "mace": asyncio.Semaphore(1),
    "pyscf": asyncio.Semaphore(2),
    "psi4": asyncio.Semaphore(2),
    "conformers": asyncio.Semaphore(1),  # 与 mace 共用 GPU, 单并发
}


async def _post_callback(url: str, body: dict) -> None:
    """缺口 #37: fire-and-forget 回调 — 失败只 warning, 绝不影响任务状态"""
    if not url.startswith(("http://", "https://")):
        logger.warning("callback url rejected (non-http): %r", url[:80])
        return
    try:
        import httpx

        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(url, json=body)
    except Exception as e:  # noqa: BLE001
        logger.warning("callback failed task=%s url=%s: %r",
                       body.get("task_id"), url[:120], e)


async def _fire_callback(callback_url: str | None, tool: str, task_id: str) -> None:
    if not callback_url:
        return
    rec = await taskstore.get_task(task_id, include_result=True) or {}
    asyncio.create_task(_post_callback(callback_url, {
        "task_id": task_id, "tool": tool, "status": rec.get("status"),
        "result": rec.get("result"), "error_msg": rec.get("error_msg"),
    }))


async def _execute(tool: str, task_id: str, p: dict[str, Any], timeout_s: float,
                   callback_url: str | None = None) -> None:
    # 取消竞态保护: 任务在排队期间被 DELETE → 直接放弃执行
    cur = await taskstore.get_task(task_id)
    if cur is not None and cur.get("status") == "cancelled":
        logger.info("task %s cancelled before dispatch, skipping", task_id)
        return
    try:
        async with _SEMAPHORES[tool]:
            # 拿到并发额度后再查一次 (可能在信号量排队期间被取消)
            cur = await taskstore.get_task(task_id)
            if cur is not None and cur.get("status") == "cancelled":
                logger.info("task %s cancelled while queued, skipping", task_id)
                return
            result = await RUNNERS[tool](task_id, p, timeout_s)
    except Exception as e:  # noqa: BLE001 — 执行器兜底
        logger.exception("task %s (%s) crashed", task_id, tool)
        result = {"status": "failed", "error_msg": repr(e)}
    status = result.get("status", "failed")
    # #19: timeout 是一等状态 (executor/driver 超时分支产生), 不再被归一成 failed
    if status not in ("success", "failed", "unavailable",
                      "completed_with_warnings", "timeout"):
        status = "failed"
    await taskstore.finish_task(task_id, status, result, result.get("error_msg"))
    await _fire_callback(callback_url, tool, task_id)  # 缺口 #37


async def _submit(
    tool: str, smiles: str, p: dict[str, Any],
    timeout_s: float, submitter: Optional[str],
) -> TaskIdResponse:
    callback_url = p.pop("callback_url", None)  # 缺口 #37: 不进 driver params/DB
    rec = taskstore.create_task(tool, smiles, p, submitter, callback_url)
    await taskstore.persist_new_task(rec)
    task_id = rec["task_id"]
    asyncio.create_task(_execute(tool, task_id, p, timeout_s, callback_url))
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
    return await _submit("gaussian", _label(req), req.model_dump(), req.timeout_s, submitter)


@router.post("/gromacs", response_model=TaskIdResponse,
             dependencies=[Depends(require_api_key)])
async def submit_gromacs(req: GromacsRequest, submitter: Optional[str] = None):
    return await _submit("gromacs", req.smiles, req.model_dump(), req.timeout_s, submitter)


@router.post("/mace", response_model=TaskIdResponse,
             dependencies=[Depends(require_api_key)])
async def submit_mace(req: MaceRequest, submitter: Optional[str] = None):
    return await _submit("mace", _label(req), req.model_dump(), req.timeout_s, submitter)


@router.post("/pyscf", response_model=TaskIdResponse,
             dependencies=[Depends(require_api_key)])
async def submit_pyscf(req: PyscfRequest, submitter: Optional[str] = None):
    return await _submit("pyscf", _label(req), req.model_dump(), req.timeout_s, submitter)


@router.post("/psi4", response_model=TaskIdResponse,
             dependencies=[Depends(require_api_key)])
async def submit_psi4(req: Psi4Request, submitter: Optional[str] = None):
    return await _submit("psi4", req.smiles, req.model_dump(), req.timeout_s, submitter)


@router.post("/auto", dependencies=[Depends(require_api_key)])
async def submit_auto(req: AutoRequest, submitter: Optional[str] = None):
    """智能选路: task+quality → 最快可用后端, 派发并返回 task_id + 选路理由

    返回带 warnings 列表 (缺口 #13): 后端不支持你传的参数时明确告知,
    不静默丢弃。
    """
    warnings: list[str] = []
    tool, reason = select_backend(req.task, req.quality)
    if tool is None:
        return {
            "status": "unavailable",
            "reason": reason,
            "availability": list_available_tools(),
        }
    task_norm = req.task.strip().lower()
    from dft_service.runners.tool_definitions import _TASK_ALIASES
    task_alias = _TASK_ALIASES.get(task_norm, task_norm)
    # 缺口 #27: freq / opt_freq 仅 Gaussian 支持 (pyscf/psi4 无频率解析)
    if task_alias in ("freq", "opt_freq") and tool != "gaussian":
        return {
            "status": "unavailable",
            "reason": f"{req.task} 任务当前仅 Gaussian 支持 (选中 {tool})",
        }

    operation = "optimize" if task_alias in ("optimize",) else (
        "properties" if task_alias == "properties" else "energy"
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
            "job": _GAUSS_JOB.get(task_alias, "sp"),  # 用归一化别名: geometry→opt 等
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

    # 缺口 #13: 被选后端不支持的参数明确告警, 不静默丢弃
    p["callback_url"] = req.callback_url  # #37: _submit 会 pop 掉, 不进 runner
    solvent = (req.solvent or "none").lower()
    if solvent not in ("", "none", "gas", "gasphase", "vacuum"):
        if tool in ("mace", "gromacs"):
            warnings.append(
                f"{tool} 不支持溶剂模型, solvent={req.solvent} 被忽略 (需溶剂请选 "
                "gaussian/pyscf/psi4)")
    if tool in ("mace", "gromacs") and (
            req.charge not in (None, 0) or req.multiplicity not in (None, 1)):
        warnings.append(
            f"{tool} 无电荷/自旋概念, charge/multiplicity 被忽略")
    if req.task.strip().lower() in ("properties", "prop") and tool == "gaussian":
        warnings.append("Gaussian 无独立 properties 任务, 已降级为 sp 单点 "
                        "(偶极/HOMO-LUMO 请用 psi4)")

    resp = await _submit(tool, req.smiles, p, timeout, submitter)
    return {**resp.model_dump(), "backend": tool, "reason": reason,
            "warnings": warnings}


@router.post("/conformers", response_model=TaskIdResponse,
             dependencies=[Depends(require_api_key)])
async def submit_conformers(req: ConformersRequest, submitter: Optional[str] = None):
    """构象搜索工作流 (缺口 #32): ETKDG N 构象 → MMFF 预优化 → MACE 弛豫排序"""
    return await _submit("conformers", req.smiles, req.model_dump(),
                         req.timeout_s, submitter)


@router.get("/status/{task_id}", dependencies=[Depends(require_api_key)])
async def task_status(task_id: str):
    rec = await taskstore.get_task(task_id, include_result=False)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"task {task_id} not found")
    out = {
        "task_id": task_id,
        "status": rec.get("status"),
        "tool": rec.get("tool"),
        "submit_time": rec.get("submit_time"),
        "finish_time": rec.get("finish_time"),
    }
    # 缺口 #22: 未终结任务回读 driver 进度 (progress.json, 无则 null)
    if out["status"] in ("queued", "running"):
        from dft_service.runners.executor import read_progress

        out["progress"] = read_progress(out["tool"], task_id)
    return out


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


@router.get("/stats", dependencies=[Depends(require_api_key)])
async def stats():
    """缺口 #36: 各后端成功率/平均耗时/排队深度 (SQLite 聚合)"""
    return await taskstore.stats()


@router.get("/jobs", dependencies=[Depends(require_api_key)])
async def jobs(
    tool: Optional[str] = None, status: Optional[str] = None,
    limit: int = 50, offset: int = 0,
):
    """任务列表 (重启后仍可查 — 数据在 SQLite)"""
    return await taskstore.list_jobs(tool=tool, status=status, limit=limit, offset=offset)


@router.delete("/jobs/{task_id}", dependencies=[Depends(require_api_key)])
async def cancel_job(task_id: str):
    """取消任务 (缺口 #6)

    - queued: 直接标 cancelled, 派发前会被跳过
    - running: 树杀进程树 (Windows taskkill /T 连杀 g16 孙进程;
      WSL 侧按唯一 workdir 名 pkill 清残留 gmx/pyscf)
    - 终态: 幂等返回当前状态
    """
    from dft_service.runners.executor import kill_task_process

    rec = await taskstore.get_task(task_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"task {task_id} not found")
    status = rec.get("status")
    if status in ("cancelled", "interrupted", "success", "failed",
                  "unavailable", "completed_with_warnings", "timeout"):
        return {"task_id": task_id, "status": status,
                "process_killed": False, "message": "task already finished"}
    killed = kill_task_process(task_id)
    await taskstore.cancel_task(task_id)
    await _fire_callback(rec.get("callback_url"), rec.get("tool"), task_id)  # #37
    return {
        "task_id": task_id,
        "status": "cancelled",
        "process_killed": killed,
        "message": "running process tree killed" if killed
                   else "marked cancelled (no live process)",
    }
