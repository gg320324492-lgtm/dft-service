r"""dft MCP server — 把 dft-service 暴露为 Claude Code 原生工具 (stdio)

注册 (user 作用域, 所有会话可用):
    claude mcp add --scope user dft -- E:\dft-service\.venv\Scripts\python.exe E:\dft-service\mcp_server.py

工具:
- dft_tools()                        5 后端健康状态
- dft_wait(tool, params, timeout_s)  提交 + 轮询到完成 (短任务)
- dft_submit(task, params)           长任务只提交 → task_id
- dft_result(task_id) / dft_status(task_id) / dft_cancel(task_id)
- dft_list(tool, status, limit)      任务列表
"""
from __future__ import annotations

import os
import time
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

BASE_URL = os.environ.get("DFT_SERVICE_URL", "http://127.0.0.1:8620").rstrip("/")
API_KEY = os.environ.get("DFT_SERVICE_API_KEY", "")
POLL_S = 3.0

mcp = FastMCP("dft")


def _client() -> httpx.Client:
    return httpx.Client(
        base_url=BASE_URL,
        headers={"X-API-Key": API_KEY} if API_KEY else {},
        timeout=httpx.Timeout(connect=5.0, read=30.0, write=30.0, pool=5.0),
    )


def _call(method: str, path: str, payload: dict | None = None) -> dict:
    try:
        with _client() as c:
            r = c.request(method, path, json=payload) if payload is not None \
                else c.request(method, path)
            r.raise_for_status()
            return r.json()
    except Exception as e:  # noqa: BLE001 — MCP 边界兜底
        return {"status": "unavailable", "error_msg": repr(e)}


@mcp.tool()
def dft_tools() -> dict:
    """5 个 DFT/MD 后端 (gaussian/gromacs/mace/pyscf/psi4) 健康状态"""
    return _call("GET", "/dft/tools")


@mcp.tool()
def dft_wait(tool: str, params: dict[str, Any], timeout_s: float = 540.0) -> dict:
    """提交 DFT 任务并阻塞到完成 (短任务; params 见各后端 schema)。

    tool: gaussian / gromacs / mace / pyscf / psi4 / auto
    params 示例: {"smiles": "O", "basis": "sto-3g"} /
                {"smiles": "CCO", "job": "opt", "solvent": "water"} /
                {"task": "optimize", "quality": "fast"} (auto)
    返回 status=success/failed/... + energy_hartree 等字段。
    """
    task = _call("POST", f"/dft/{tool}", params)
    if task.get("status") in ("unavailable", "failed") or not task.get("task_id"):
        return task
    task_id = task["task_id"]
    deadline = time.monotonic() + timeout_s
    while True:
        rec = _call("GET", f"/dft/result/{task_id}")
        if rec.get("status") not in ("queued", "running", None):
            result = rec.get("result") or {}
            result.setdefault("task_id", task_id)
            return result
        if time.monotonic() > deadline:
            return {"status": "timeout", "task_id": task_id,
                    "error_msg": f"MCP 等待 {timeout_s:.0f}s 未完成 — 用 "
                                 "dft_submit 两段式或加大 timeout_s"}
        time.sleep(POLL_S)


@mcp.tool()
def dft_submit(tool: str, params: dict[str, Any]) -> dict:
    """长任务只提交不等待 → {task_id}; 稍后用 dft_result 查"""
    return _call("POST", f"/dft/{tool}", params)


@mcp.tool()
def dft_status(task_id: str) -> dict:
    """查任务状态 (queued/running/success/failed/...)"""
    return _call("GET", f"/dft/status/{task_id}")


@mcp.tool()
def dft_result(task_id: str) -> dict:
    """拿任务结果 (已完成返回完整 result; 未完成返回提示)"""
    return _call("GET", f"/dft/result/{task_id}")


@mcp.tool()
def dft_cancel(task_id: str) -> dict:
    """取消任务 (树杀运行中进程, 含 WSL 侧清理)"""
    return _call("DELETE", f"/dft/jobs/{task_id}")


@mcp.tool()
def dft_list(tool: str = "", status: str = "", limit: int = 20) -> dict:
    """任务列表 (可按 tool/status 过滤)"""
    q = f"/dft/jobs?limit={limit}"
    if tool:
        q += f"&tool={tool}"
    if status:
        q += f"&status={status}"
    return _call("GET", q)


if __name__ == "__main__":
    mcp.run(transport="stdio")
