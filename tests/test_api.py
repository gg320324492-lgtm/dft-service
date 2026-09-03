"""dft-service 测试 — 全部 mock driver 执行, 不真跑 g16/wsl/mace

覆盖:
- /health + /dft/tools 形状
- API key 鉴权 (401 / 放行 / 关闭模式)
- gaussian 提交→轮询→结果 全流程 (含 solvent 参数透传断言)
- gromacs WSL 不可用 → status=unavailable
- /auto 选路 (fast→mace / accurate→gaussian / 全不可用)
- /dft/jobs 列表 + 过滤 + 分页
- 进程内 _TASKS 清空后 status/result 仍可查 (重启持久化 = 缺口 #5 修复验证)
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# 环境变量必须在 import dft_service 之前设置
_TMP = Path(__file__).parent / ".tmp"
_TMP.mkdir(exist_ok=True)
os.environ["DFT_SERVICE_API_KEY"] = "test-key-123"
os.environ["DFT_SERVICE_OUTPUT_ROOT"] = str(_TMP / "jobs")
os.environ["DFT_SERVICE_DB_URL"] = str(_TMP / "test.db")
os.environ["DFT_SERVICE_SCISOFTWARE"] = str(_TMP / "nonexistent-sci")

# 清掉可能的残留 DB 保证幂等
for f in (_TMP / "test.db",):
    if f.exists():
        f.unlink()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dft_service import taskstore  # noqa: E402
from dft_service.main import app  # noqa: E402
from dft_service.runners import paths as runner_paths  # noqa: E402

HEADERS = {"X-API-Key": "test-key-123"}


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """每个测试: 清内存任务 + 屏蔽真实环境探测 (全部视为不可用)"""
    taskstore._TASKS.clear()
    monkeypatch.setattr(runner_paths, "workflows_available", lambda: False)
    monkeypatch.setattr(runner_paths, "gaussian_binary_exists", lambda: False)
    monkeypatch.setattr(runner_paths, "scichem_has", lambda m: False)
    monkeypatch.setattr(runner_paths, "detect_wsl_gromacs_distro", lambda: None)
    monkeypatch.setattr(runner_paths, "detect_wsl_pyscf_distro", lambda: None)
    yield
    taskstore._TASKS.clear()


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


async def _fake_driver_success(tool: str, workdir, driver, params, timeout_s, **_):
    from dft_service.runners.executor import make_workdir

    make_workdir(tool, "fake")  # 确保 workdir 参数真实
    return {
        "status": "success",
        "tool": tool,
        "smiles": params.get("smiles"),
        "params_echo": params,
        "energy_hartree": -76.4,
        "log_path": str(workdir / "fake.log"),
        "work_dir": str(workdir),
    }


def _poll_result(client, task_id: str, tries: int = 50) -> dict:
    import time

    for _ in range(tries):
        r = client.get(f"/dft/result/{task_id}", headers=HEADERS).json()
        if r.get("status") not in ("queued", "running"):
            return r
        time.sleep(0.05)
    raise AssertionError(f"task {task_id} did not finish")


# ---------------------------------------------------------------
# 基础
# ---------------------------------------------------------------
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "dft-service"
    assert body["auth"] is True  # 测试环境设了 API key


def test_tools_shape(client):
    r = client.get("/dft/tools")
    assert r.status_code == 200
    body = r.json()
    names = {t["name"] for t in body["tools"]}
    assert names == {"gaussian", "gromacs", "mace", "pyscf", "psi4"}
    assert body["count"] == 5
    assert body["available_count"] == 0  # 测试环境全部屏蔽


# ---------------------------------------------------------------
# 鉴权 (缺口 #7)
# ---------------------------------------------------------------
def test_submit_requires_api_key(client):
    r = client.post("/dft/gaussian", json={"smiles": "O"})
    assert r.status_code == 401
    r = client.post("/dft/gaussian", json={"smiles": "O"},
                    headers={"X-API-Key": "wrong"})
    assert r.status_code == 401


def test_status_and_jobs_require_key(client):
    assert client.get("/dft/status/nope").status_code == 401
    assert client.get("/dft/jobs").status_code == 401
    assert client.get("/dft/tools").status_code == 200  # tools 免鉴权


# ---------------------------------------------------------------
# gaussian 全流程 (solvent 透传 = 缺口 #1/#2)
# ---------------------------------------------------------------
def test_gaussian_full_flow(client, monkeypatch):
    import dft_service.runners as runners_pkg

    captured = {}

    async def fake(tool, workdir, driver, params, timeout_s, **_):
        captured["tool"] = tool
        captured["params"] = params
        return await _fake_driver_success(tool, workdir, driver, params, timeout_s)

    monkeypatch.setattr(runners_pkg, "execute_driver", fake)

    r = client.post("/dft/gaussian", headers=HEADERS, json={
        "smiles": "CCO",
        "xc": "M06-2X",
        "basis": "def2-TZVP",
        "job": "sp",
        "solvent": "water",
        "charge": 1,
        "multiplicity": 2,
        "nproc": 16,
        "mem": "16GB",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["tool"] == "gaussian"
    assert body["status"] == "queued"
    task_id = body["task_id"]

    # 等后台任务完成 (mock 即时返回)
    import time
    for _ in range(100):
        rr = client.get(f"/dft/result/{task_id}", headers=HEADERS).json()
        if rr.get("status") not in ("queued", "running"):
            break
        time.sleep(0.05)
    assert rr["status"] == "success"
    assert rr["result"]["energy_hartree"] == -76.4

    # driver 收到的参数: solvent/charge/multiplicity/nproc/mem 全透传
    p = captured["params"]
    assert p["solvent"] == "water"
    assert p["charge"] == 1
    assert p["multiplicity"] == 2
    assert p["nproc"] == 16
    assert p["mem"] == "16GB"
    assert p["xc"] == "M06-2X"
    assert p["basis"] == "def2-TZVP"
    assert p["job"] == "sp"


# ---------------------------------------------------------------
# gromacs 不可用路径
# ---------------------------------------------------------------
def test_gromacs_unavailable_without_wsl(client):
    r = client.post("/dft/gromacs", headers=HEADERS, json={"smiles": "CCO"})
    assert r.status_code == 200
    task_id = r.json()["task_id"]

    import time
    for _ in range(100):
        rr = client.get(f"/dft/result/{task_id}", headers=HEADERS).json()
        if rr.get("status") not in ("queued", "running"):
            break
        time.sleep(0.05)
    assert rr["status"] == "unavailable"
    assert "WSL" in rr["result"]["error_msg"] or "gmx" in rr["result"]["error_msg"]


# ---------------------------------------------------------------
# /auto 选路 (缺口 #8)
# ---------------------------------------------------------------
def test_auto_fast_selects_mace(client, monkeypatch):
    import dft_service.runners as runners_pkg
    import dft_service.runners.tool_definitions as td

    async def fake(tool, workdir, driver, params, timeout_s, **_):
        return {"status": "success", "tool": tool}

    monkeypatch.setattr(runners_pkg, "execute_driver", fake)
    monkeypatch.setattr(td, "availability_map", lambda: {
        "gaussian": True, "gromacs": True, "mace": True, "pyscf": True, "psi4": True,
    })

    r = client.post("/dft/auto", headers=HEADERS, json={
        "smiles": "O", "task": "optimize", "quality": "fast",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["backend"] == "mace"
    assert "MACE" in body["reason"]


def test_auto_accurate_freq_selects_gaussian(client, monkeypatch):
    import dft_service.runners as runners_pkg
    import dft_service.runners.tool_definitions as td

    async def fake(tool, workdir, driver, params, timeout_s, **_):
        return {"status": "success", "tool": tool}

    monkeypatch.setattr(runners_pkg, "execute_driver", fake)
    monkeypatch.setattr(td, "availability_map", lambda: {
        "gaussian": True, "gromacs": False, "mace": True, "pyscf": True, "psi4": True,
    })

    r = client.post("/dft/auto", headers=HEADERS, json={
        "smiles": "O", "task": "freq",
    })
    assert r.status_code == 200
    assert r.json()["backend"] == "gaussian"


def test_auto_all_unavailable(client):
    r = client.post("/dft/auto", headers=HEADERS, json={
        "smiles": "O", "task": "energy", "quality": "fast",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "unavailable"
    assert "reason" in body


def test_auto_md_selects_gromacs(client, monkeypatch):
    import dft_service.runners as runners_pkg
    import dft_service.runners.tool_definitions as td

    async def fake(tool, workdir, driver, params, timeout_s, **_):
        return {"status": "success", "tool": tool}

    monkeypatch.setattr(runners_pkg, "execute_driver", fake)
    monkeypatch.setattr(td, "availability_map", lambda: {
        "gaussian": False, "gromacs": True, "mace": False, "pyscf": False, "psi4": False,
    })

    r = client.post("/dft/auto", headers=HEADERS, json={"smiles": "O", "task": "md"})
    assert r.json()["backend"] == "gromacs"


# ---------------------------------------------------------------
# 缺口 #13: auto 丢参数告警
# ---------------------------------------------------------------
def test_auto_warns_when_solvent_dropped_for_mace(client, monkeypatch):
    import dft_service.runners as runners_pkg
    import dft_service.runners.tool_definitions as td

    async def fake(tool, workdir, driver, params, timeout_s, **_):
        return {"status": "success", "tool": tool}

    monkeypatch.setattr(runners_pkg, "execute_driver", fake)
    monkeypatch.setattr(td, "availability_map", lambda: {
        "gaussian": False, "gromacs": False, "mace": True, "pyscf": False, "psi4": False,
    })

    r = client.post("/dft/auto", headers=HEADERS, json={
        "smiles": "O", "task": "optimize", "quality": "fast", "solvent": "water",
    })
    body = r.json()
    assert body["backend"] == "mace"
    assert body["warnings"] and "mace" in body["warnings"][0] \
        and "solvent" in body["warnings"][0]


def test_auto_no_warning_when_solvent_supported(client, monkeypatch):
    import dft_service.runners as runners_pkg
    import dft_service.runners.tool_definitions as td

    async def fake(tool, workdir, driver, params, timeout_s, **_):
        return {"status": "success", "tool": tool}

    monkeypatch.setattr(runners_pkg, "execute_driver", fake)
    monkeypatch.setattr(td, "availability_map", lambda: {
        "gaussian": True, "gromacs": False, "mace": False, "pyscf": False, "psi4": False,
    })

    r = client.post("/dft/auto", headers=HEADERS, json={
        "smiles": "O", "task": "energy", "quality": "accurate", "solvent": "water",
    })
    body = r.json()
    assert body["backend"] == "gaussian"
    assert body["warnings"] == []  # gaussian 支持溶剂, 不告警


# ---------------------------------------------------------------
# 缺口 #1 回归: auto→gaussian job 归一化 (energy/optimize 不是合法关键字)
# ---------------------------------------------------------------
def _auto_capture(client, monkeypatch, task: str, avail: dict) -> dict:
    import time

    import dft_service.runners as runners_pkg
    import dft_service.runners.tool_definitions as td

    captured = {}

    async def fake(tool, workdir, driver, params, timeout_s, **_):
        captured["tool"] = tool
        captured["params"] = params
        return {"status": "success", "tool": tool}

    monkeypatch.setattr(runners_pkg, "execute_driver", fake)
    monkeypatch.setattr(td, "availability_map", lambda: avail)
    r = client.post("/dft/auto", headers=HEADERS,
                    json={"smiles": "O", "task": task, "quality": "accurate"})
    assert r.status_code == 200
    # _execute 是响应后才跑的后台任务, 等它落地
    for _ in range(100):
        if "params" in captured:
            break
        time.sleep(0.05)
    return captured


def test_auto_gaussian_energy_maps_to_sp(client, monkeypatch):
    cap = _auto_capture(client, monkeypatch, "energy", {
        "gaussian": True, "gromacs": False, "mace": False, "pyscf": False, "psi4": False,
    })
    assert cap["tool"] == "gaussian"
    assert cap["params"]["job"] == "sp"  # 不再是非法关键字 "energy"


def test_auto_gaussian_optimize_maps_to_opt(client, monkeypatch):
    cap = _auto_capture(client, monkeypatch, "optimize", {
        "gaussian": True, "gromacs": False, "mace": False, "pyscf": False, "psi4": False,
    })
    assert cap["params"]["job"] == "opt"  # 不再是非法关键字 "optimize"


def test_auto_properties_prefers_psi4(client, monkeypatch):
    cap = _auto_capture(client, monkeypatch, "properties", {
        "gaussian": True, "gromacs": False, "mace": False, "pyscf": True, "psi4": True,
    })
    assert cap["tool"] == "psi4"  # properties 是 Psi4 独有能力


# ---------------------------------------------------------------
# /dft/jobs 列表 (缺口 #6)
# ---------------------------------------------------------------
def test_jobs_list_and_filter(client, monkeypatch):
    import dft_service.runners as runners_pkg

    async def fake(tool, workdir, driver, params, timeout_s, **_):
        return {"status": "success", "tool": tool}

    monkeypatch.setattr(runners_pkg, "execute_driver", fake)

    client.post("/dft/gaussian", headers=HEADERS, json={"smiles": "O"})
    client.post("/dft/mace", headers=HEADERS, json={"smiles": "CCO"})
    import time
    time.sleep(0.3)  # 等两个后台任务落库

    r = client.get("/dft/jobs", headers=HEADERS).json()
    assert r["total"] >= 2
    assert {j["tool"] for j in r["jobs"]} >= {"gaussian", "mace"}

    r2 = client.get("/dft/jobs?tool=gaussian", headers=HEADERS).json()
    assert all(j["tool"] == "gaussian" for j in r2["jobs"])
    assert r2["total"] >= 1

    r3 = client.get("/dft/jobs?status=success&limit=1", headers=HEADERS).json()
    assert len(r3["jobs"]) <= 1
    assert r3["limit"] == 1


# ---------------------------------------------------------------
# 重启持久化 (缺口 #5: 内存清空后 DB 回退)
# ---------------------------------------------------------------
def test_status_survives_memory_loss(client, monkeypatch):
    import dft_service.runners as runners_pkg

    async def fake(tool, workdir, driver, params, timeout_s, **_):
        return {"status": "success", "tool": tool, "energy_hartree": -76.4}

    monkeypatch.setattr(runners_pkg, "execute_driver", fake)

    r = client.post("/dft/gaussian", headers=HEADERS, json={"smiles": "O"}).json()
    task_id = r["task_id"]
    import time
    for _ in range(100):
        rr = client.get(f"/dft/result/{task_id}", headers=HEADERS).json()
        if rr.get("status") not in ("queued", "running"):
            break
        time.sleep(0.05)
    assert rr["status"] == "success"

    # 模拟进程重启: 内存态清空
    taskstore._TASKS.clear()

    # /status 不再 404 — DB 回退
    # 注: finish_task 先更内存后写 DB, 刚完成瞬间 DB 可能还在路上, 轮询等它
    s = {"status": "queued"}
    for _ in range(100):
        s = client.get(f"/dft/status/{task_id}", headers=HEADERS).json()
        if s.get("status") == "success":
            break
        time.sleep(0.05)
    assert s["status"] == "success"

    rr2 = client.get(f"/dft/result/{task_id}", headers=HEADERS).json()
    assert rr2["result"]["energy_hartree"] == -76.4


def test_status_404_unknown(client):
    r = client.get("/dft/status/deadbeefdeadbeef", headers=HEADERS)
    assert r.status_code == 404


# ---------------------------------------------------------------
# 取消 (缺口 #6)
# ---------------------------------------------------------------
def test_cancel_unknown_404(client):
    r = client.delete("/dft/jobs/nope123456789", headers=HEADERS)
    assert r.status_code == 404


def test_cancel_finished_task_idempotent(client, monkeypatch):
    import dft_service.runners as runners_pkg
    import time

    async def fake(tool, workdir, driver, params, timeout_s, **_):
        return {"status": "success", "tool": tool}

    monkeypatch.setattr(runners_pkg, "execute_driver", fake)
    tid = client.post("/dft/mace", headers=HEADERS,
                      json={"smiles": "O"}).json()["task_id"]
    for _ in range(100):
        rr = client.get(f"/dft/status/{tid}", headers=HEADERS).json()
        if rr.get("status") not in ("queued", "running"):
            break
        time.sleep(0.05)

    r = client.delete(f"/dft/jobs/{tid}", headers=HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"      # 终态不被改写
    assert body["process_killed"] is False


def test_cancel_queued_task_semantics():
    """taskstore 层: queued → cancelled; finish_task 不覆盖取消态"""
    import asyncio

    async def _go():
        rec = taskstore.create_task("gaussian", "O", {}, None)
        await taskstore.persist_new_task(rec)
        tid = rec["task_id"]
        assert await taskstore.cancel_task(tid) is True
        # 已取消: 再取消幂等 False
        assert await taskstore.cancel_task(tid) is False
        # 后台执行完回来写结果: 不允许覆盖取消态
        await taskstore.finish_task(tid, "success", {"energy_hartree": -1.0})
        cur = await taskstore.get_task(tid, include_result=True)
        return cur

    cur = asyncio.run(_go())
    assert cur["status"] == "cancelled"
    assert cur["result"] is None  # 取消后结果被丢弃


def test_kill_task_process_unregistered_is_noop():
    from dft_service.runners.executor import kill_task_process

    assert kill_task_process("never000registered") is False


# ---------------------------------------------------------------
# 鉴权关闭模式
# ---------------------------------------------------------------
def test_auth_disabled_mode(client, monkeypatch):
    from dft_service.config import settings
    from dft_service.auth import require_api_key

    monkeypatch.setattr(settings, "api_key", None)
    # require_api_key 读 settings.api_key → None 时直接放行
    import asyncio

    async def _call():
        await require_api_key(x_api_key=None)

    asyncio.run(_call())  # 不抛 = 放行


# ---------------------------------------------------------------
# #19: timeout 一等状态
# ---------------------------------------------------------------
def test_timeout_status_survives_api(client, monkeypatch):
    """executor 超时返回的 status=timeout 不被 _execute 白名单归一成 failed"""
    import dft_service.runners as runners_pkg

    async def fake_timeout(tool, workdir, driver, params, timeout_s, **_):
        return {"status": "timeout",
                "error_msg": f"driver timeout after {timeout_s:.0f}s",
                "work_dir": str(workdir)}

    monkeypatch.setattr(runners_pkg, "execute_driver", fake_timeout)
    # 用 gaussian (单段直通); pyscf 是两段编排, geom 超时被包成 failed 属预期
    r = client.post("/dft/gaussian", headers=HEADERS, json={"smiles": "O"})
    task_id = r.json()["task_id"]

    body = _poll_result(client, task_id)
    assert body["status"] == "timeout"
    assert "timeout" in (body.get("error_msg") or "").lower()
    # timeout 是终态: cancel 幂等返回 already finished
    r2 = client.delete(f"/dft/jobs/{task_id}", headers=HEADERS)
    assert r2.json()["message"] == "task already finished"


# ---------------------------------------------------------------
# #20: 参数边界
# ---------------------------------------------------------------
def test_param_bounds_rejected(client, monkeypatch):
    """超限/非法格式 → 422; 正常值不受影响 (mem 大小写兼容)"""
    import dft_service.runners as runners_pkg

    async def fake(tool, workdir, driver, params, timeout_s, **_):
        return {"status": "success", "tool": tool}

    monkeypatch.setattr(runners_pkg, "execute_driver", fake)

    bad_calls = [
        ("/dft/gaussian", {"smiles": "O", "timeout_s": 10**9}),      # 超 7 天
        ("/dft/gaussian", {"smiles": "O", "mem": "8; rm -rf"}),      # gjf 注入面
        ("/dft/gaussian", {"smiles": "O", "nproc": 9999}),           # 超核数上限
        ("/dft/gromacs", {"smiles": "O", "box_nm": 100}),            # 超盒子上限
        ("/dft/mace", {"smiles": "O", "max_steps": 10**6}),          # 超步数上限
    ]
    for path, payload in bad_calls:
        r = client.post(path, headers=HEADERS, json=payload)
        assert r.status_code == 422, f"{path} {payload} 应 422, 实得 {r.status_code}"

    ok = client.post("/dft/gaussian", headers=HEADERS,
                     json={"smiles": "O", "timeout_s": 7200, "mem": "16gb", "nproc": 8})
    assert ok.status_code == 200
    client.delete(f"/dft/jobs/{ok.json()['task_id']}", headers=HEADERS)


# ---------------------------------------------------------------
# #21: 终态内存驱逐
# ---------------------------------------------------------------
def test_taskstore_mem_eviction():
    """终态任务内存副本过期即逐出 (DB 回退仍可查); 运行中任务永不动"""
    import asyncio
    from datetime import datetime, timedelta, timezone

    async def _go():
        rec = taskstore.create_task("pyscf", "O", {}, None)
        await taskstore.persist_new_task(rec)
        tid = rec["task_id"]
        await taskstore.finish_task(tid, "success", {"energy_hartree": -1.0})
        assert tid in taskstore._TASKS

        # 手工把 finish_time 老化 25 小时 → 下次 create_task 触发逐出
        old = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
        taskstore._TASKS[tid]["finish_time"] = old

        running = taskstore.create_task("pyscf", "C", {}, None)  # queued, 不被逐出
        assert tid not in taskstore._TASKS
        assert running["task_id"] in taskstore._TASKS

        cur = await taskstore.get_task(tid)  # DB 回退
        assert cur is not None and cur["status"] == "success"

    try:
        asyncio.run(_go())
    finally:
        taskstore._TASKS.clear()
