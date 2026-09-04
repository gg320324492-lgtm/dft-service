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
# #27 / #28: opt_freq 联跑 + extra_route 白名单
# ---------------------------------------------------------------
def test_extra_route_and_job_validation(client, monkeypatch):
    """注入面 422; 合法 extra_route / opt freq 透传到 driver"""
    import dft_service.runners as runners_pkg

    captured = {}

    async def fake(tool, workdir, driver, params, timeout_s, **_):
        captured["params"] = params
        return {"status": "success", "tool": tool}

    monkeypatch.setattr(runners_pkg, "execute_driver", fake)

    for bad in ({"smiles": "O", "extra_route": "int=ultrafine\n#evil"},
                {"smiles": "O", "extra_route": "%mkchg /usr"},
                {"smiles": "O", "job": "opt\nsp"},
                {"smiles": "O", "solvent": "water) x(1"}):
        r = client.post("/dft/gaussian", headers=HEADERS, json=bad)
        assert r.status_code == 422, f"{bad} 应 422"

    r = client.post("/dft/gaussian", headers=HEADERS, json={
        "smiles": "O", "job": "opt freq", "extra_route": "int=ultrafine"})
    assert r.status_code == 200
    body = _poll_result(client, r.json()["task_id"])
    assert body["status"] == "success"
    assert captured["params"]["job"] == "opt freq"
    assert captured["params"]["extra_route"] == "int=ultrafine"


def test_auto_opt_freq_maps_to_gaussian(client, monkeypatch):
    """缺口 #27: auto task=opt+freq → gaussian, job='opt freq'"""
    import dft_service.runners as runners_pkg
    import dft_service.runners.tool_definitions as td

    captured = {}

    async def fake(tool, workdir, driver, params, timeout_s, **_):
        captured["tool"] = tool
        captured["params"] = params
        return {"status": "success", "tool": tool}

    monkeypatch.setattr(runners_pkg, "execute_driver", fake)
    monkeypatch.setattr(td, "availability_map", lambda: {
        "gaussian": True, "gromacs": False, "mace": False, "pyscf": True, "psi4": True,
    })
    r = client.post("/dft/auto", headers=HEADERS,
                    json={"smiles": "O", "task": "opt+freq", "quality": "accurate"})
    assert r.json()["backend"] == "gaussian"
    _poll_result(client, r.json()["task_id"])
    assert captured["params"]["job"] == "opt freq"


def test_auto_opt_freq_guards_non_gaussian(client, monkeypatch):
    """gaussian 不可用时 opt_freq 明确 unavailable, 不落到 pyscf (其无频率解析)"""
    import dft_service.runners.tool_definitions as td

    monkeypatch.setattr(td, "availability_map", lambda: {
        "gaussian": False, "gromacs": False, "mace": False, "pyscf": True, "psi4": True,
    })
    r = client.post("/dft/auto", headers=HEADERS,
                    json={"smiles": "O", "task": "opt_freq"})
    body = r.json()
    assert body["status"] == "unavailable"
    assert "Gaussian" in body["reason"]


# ---------------------------------------------------------------
# #29: 内联 xyz 输入 (smiles / xyz_content 二选一)
# ---------------------------------------------------------------
def test_xyz_content_validation(client, monkeypatch):
    import dft_service.runners as runners_pkg

    captured = {}

    async def fake(tool, workdir, driver, params, timeout_s, **_):
        captured["params"] = params
        return {"status": "success", "tool": tool}

    monkeypatch.setattr(runners_pkg, "execute_driver", fake)

    # 都不给 / 都给 → 422
    r = client.post("/dft/gaussian", headers=HEADERS, json={})
    assert r.status_code == 422
    r = client.post("/dft/gaussian", headers=HEADERS,
                    json={"smiles": "O", "xyz_content": "1\n\nO 0 0 0"})
    assert r.status_code == 422
    # 只给 xyz 但缺 charge/mult → 422
    r = client.post("/dft/gaussian", headers=HEADERS,
                    json={"xyz_content": "1\n\nO 0 0 0"})
    assert r.status_code == 422
    # 合法 xyz 请求 → 202 语义 (queued) 且 smiles 列落占位标签
    r = client.post("/dft/gaussian", headers=HEADERS, json={
        "xyz_content": "2\nwater\nO 0 0 0\nH 0 0 0.96\n",
        "charge": 0, "multiplicity": 1, "job": "sp"})
    assert r.status_code == 200
    assert r.json()["status"] == "queued"
    body = _poll_result(client, r.json()["task_id"])
    assert body["status"] == "success"
    assert captured["params"]["xyz_content"].startswith("2\nwater")

    # pyscf 同样支持 (smiles 可缺)
    r = client.post("/dft/pyscf", headers=HEADERS, json={
        "xyz_content": "2\n\nO 0 0 0\nH 0 0 0.96\n", "charge": 0,
        "multiplicity": 1, "basis": "sto-3g"})
    assert r.status_code == 200
    # gromacs 不收 xyz (拓扑必须 SMILES)
    r = client.post("/dft/gromacs", headers=HEADERS,
                    json={"xyz_content": "1\n\nO 0 0 0"})
    assert r.status_code == 422


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


# ---------------------------------------------------------------
# #22: 进度回读
# ---------------------------------------------------------------
def test_status_returns_progress_for_running(client):
    """running 任务的 /status 带 progress.json 内容; 无进度文件 → progress=null"""
    import asyncio
    import json
    from dft_service.runners.executor import make_workdir

    async def _setup():
        rec = taskstore.create_task("gaussian", "O", {}, None)
        await taskstore.persist_new_task(rec)
        rec["status"] = "running"  # 模拟后台执行中
        wd = make_workdir("gaussian", rec["task_id"])
        (wd / "progress.json").write_text(json.dumps(
            {"stage": "running", "opt_step": 4, "scf_cycles": 12}), encoding="utf-8")
        return rec["task_id"], wd

    tid, wd = asyncio.run(_setup())
    try:
        st = client.get(f"/dft/status/{tid}", headers=HEADERS).json()
        assert st["status"] == "running"
        assert st["progress"]["opt_step"] == 4
        assert st["progress"]["scf_cycles"] == 12

        # 无进度文件 → null (不影响 status 本身)
        (wd / "progress.json").unlink()
        st2 = client.get(f"/dft/status/{tid}", headers=HEADERS).json()
        assert st2["progress"] is None
    finally:
        import shutil
        shutil.rmtree(wd, ignore_errors=True)


# ---------------------------------------------------------------
# #36: stats 聚合
# ---------------------------------------------------------------
def test_stats_endpoint(client):
    """各后端计数/成功率 + 排队深度 (直接向 DB 插样本行)"""
    import asyncio
    from datetime import datetime, timedelta, timezone

    from dft_service.db import SessionFactory
    from dft_service.models import DFTJob, new_task_id

    now = datetime.now(timezone.utc)

    async def _seed():
        async with SessionFactory() as s:
            for tool, status, dt in [
                ("pyscf", "success", 2), ("pyscf", "success", 4),
                ("pyscf", "failed", 1), ("gaussian", "success", 10),
                ("mace", "running", None),
            ]:
                s.add(DFTJob(
                    id=new_task_id(), tool=tool, smiles="O", params={},
                    status=status,
                    submit_time=now - timedelta(minutes=(dt or 1) + 20),
                    finish_time=(now - timedelta(minutes=dt) if dt else None),
                ))
            await s.commit()

    asyncio.run(_seed())
    body = client.get("/dft/stats", headers=HEADERS).json()
    assert body["status"] == "success"
    by = {t["tool"]: t for t in body["per_tool"]}
    assert by["pyscf"]["n_total"] >= 3
    assert by["pyscf"]["n_ok"] >= 2
    assert by["pyscf"]["n_failed"] >= 1
    assert by["pyscf"]["success_rate"] == pytest.approx(2 / 3, abs=0.15)
    assert by["pyscf"]["avg_minutes"] is not None
    assert body["queue_depth"] >= 1  # 至少那行 mace running


# ---------------------------------------------------------------
# #37: webhook 回调
# ---------------------------------------------------------------
def test_callback_fired_on_finish(client, monkeypatch):
    """任务终态 → POST callback_url, payload 含 status/result; 非法 URL 不发"""
    import httpx

    sent = []

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None):
            sent.append((url, json))

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)

    import dft_service.runners as runners_pkg

    async def fake(tool, workdir, driver, params, timeout_s, **_):
        assert "callback_url" not in params  # 不进 driver
        return {"status": "success", "tool": tool, "energy_hartree": -1.0}

    monkeypatch.setattr(runners_pkg, "execute_driver", fake)

    r = client.post("/dft/mace", headers=HEADERS, json={
        "smiles": "O", "callback_url": "http://127.0.0.1:9/hook"})
    tid = r.json()["task_id"]
    _poll_result(client, tid)

    import time
    for _ in range(50):  # 等异步 create_task 完成投递
        if sent:
            break
        time.sleep(0.05)
    assert sent and sent[0][0] == "http://127.0.0.1:9/hook"
    body = sent[0][1]
    assert body["task_id"] == tid and body["status"] == "success"
    assert body["result"]["energy_hartree"] == -1.0

    # 非 http URL → 拒发 (SSRF 面收紧)
    sent.clear()
    r = client.post("/dft/mace", headers=HEADERS, json={
        "smiles": "O", "callback_url": "file:///etc/passwd"})
    _poll_result(client, r.json()["task_id"])
    time.sleep(0.2)
    assert not sent


def test_callback_fired_on_cancel(client, monkeypatch):
    import httpx

    sent = []

    class FakeClient:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, url, json=None): sent.append((url, json))

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    import asyncio

    gate = asyncio.Event()
    import dft_service.runners as runners_pkg

    async def hang(tool, workdir, driver, params, timeout_s, **_):
        await gate.wait()
        return {"status": "success"}

    monkeypatch.setattr(runners_pkg, "execute_driver", hang)
    r = client.post("/dft/gaussian", headers=HEADERS, json={
        "smiles": "O", "callback_url": "http://127.0.0.1:9/hook"})
    tid = r.json()["task_id"]
    import time
    time.sleep(0.2)  # 让任务进入 running
    client.delete(f"/dft/jobs/{tid}", headers=HEADERS)
    for _ in range(50):
        if sent:
            break
        time.sleep(0.05)
    gate.set()
    assert sent and sent[0][1]["status"] == "cancelled"


# ---------------------------------------------------------------
# #32: 构象搜索端点形状
# ---------------------------------------------------------------
def test_conformers_submit_shape(client, monkeypatch):
    import dft_service.runners as runners_pkg
    import dft_service.runners.tool_definitions as td

    captured = {}

    async def fake(tool, workdir, driver, params, timeout_s, **_):
        captured.update(tool=tool, driver=driver, params=params)
        return {"status": "success", "tool": tool, "conformers": []}

    monkeypatch.setattr(runners_pkg, "execute_driver", fake)
    monkeypatch.setattr(td, "availability_map", lambda: {
        "gaussian": False, "gromacs": False, "mace": True, "pyscf": False,
        "psi4": False,
    })
    r = client.post("/dft/conformers", headers=HEADERS,
                    json={"smiles": "CCCCO", "n_conformers": 8, "top_k": 3})
    assert r.status_code == 200
    body = _poll_result(client, r.json()["task_id"])
    assert body["status"] == "success"
    assert captured["tool"] == "conformers"
    assert captured["driver"] == "conformer_driver.py"
    assert captured["params"]["n_conformers"] == 8
    # 超限 → 422
    r = client.post("/dft/conformers", headers=HEADERS,
                    json={"smiles": "O", "n_conformers": 5000})
    assert r.status_code == 422


# ---------------------------------------------------------------
# #34: analyze_options 校验与透传
# ---------------------------------------------------------------
def test_analyze_options_validation(client, monkeypatch):
    import dft_service.api as api_mod

    captured = {}

    async def fake(task_id, p, timeout_s):
        captured["p"] = p
        return {"status": "success", "tool": "gromacs"}

    monkeypatch.setitem(api_mod.RUNNERS, "gromacs", fake)
    # 非法项 → 422
    r = client.post("/dft/gromacs", headers=HEADERS, json={
        "smiles": "O", "analyze": True, "analyze_options": ["rms", "quantum_tunneling"]})
    assert r.status_code == 422
    # 合法项 → 透传到 runner params
    r = client.post("/dft/gromacs", headers=HEADERS, json={
        "smiles": "O", "analyze": True,
        "analyze_options": ["rms", "density", "rdf", "hbond"]})
    assert r.status_code == 200
    _poll_result(client, r.json()["task_id"])
    assert captured["p"]["analyze_options"] == ["rms", "density", "rdf", "hbond"]


# ---------------------------------------------------------------
# #33: reaction 工作流
# ---------------------------------------------------------------
def test_reaction_orchestration_math(client, monkeypatch):
    """2 H2O -> dimer: ΔE/ΔG 按计量数聚合; 缺 Gibbs 时只报电子能差"""
    import asyncio
    import dft_service.api as api_mod
    import dft_service.runners as runners_pkg
    import dft_service.runners.tool_definitions as td

    monkeypatch.setattr(td, "availability_map", lambda: {
        "gaussian": True, "gromacs": False, "mace": False, "pyscf": False,
        "psi4": False,
    })

    async def fake_gauss(task_id, p, timeout_s):
        if "reactants" and p.get("smiles") == "O":
            return {"status": "success", "energy_hartree": -76.4,
                    "thermochemistry": {"sum_elec_thermal_gibbs_hartree": -76.32},
                    "smiles": "O", "n_imaginary": 0}
        return {"status": "success", "energy_hartree": -152.9,
                "thermochemistry": {"sum_elec_thermal_gibbs_hartree": -152.75},
                "smiles": "<inline-xyz:6 atoms>", "n_imaginary": 0}

    monkeypatch.setattr(runners_pkg, "run_gaussian", fake_gauss)

    captured = {}

    async def fake_exec(task_id, p, timeout_s):
        captured["result"] = await runners_pkg.run_reaction(task_id, p, timeout_s)
        return captured["result"]

    monkeypatch.setitem(api_mod.RUNNERS, "reaction", fake_exec)
    r = client.post("/dft/reaction", headers=HEADERS, json={
        "reactants": [{"smiles": "O", "count": 2}],
        "products": [{"xyz_content": "6\n\nO 0 0 0\nH 0 0 1\nH 0 1 0\n"
                                      "O 0 0 3\nH 0 0 4\nH 0 1 3",
                       "charge": 0, "multiplicity": 1}],
    })
    assert r.status_code == 200
    _poll_result(client, r.json()["task_id"])
    res = captured["result"]
    assert res["status"] == "success"
    # ΔE = -152.9 - 2*(-76.4) = -0.1 Ha → -262.5 kJ/mol
    assert res["delta_electronic_hartree"] == pytest.approx(-0.1, abs=1e-6)
    assert res["delta_electronic_kj_mol"] == pytest.approx(-262.55, abs=0.1)
    # ΔG = -152.75 - 2*(-76.32) = -0.11 Ha
    assert res["delta_gibbs_hartree"] == pytest.approx(-0.11, abs=1e-6)
    assert len(res["species"]) == 2


def test_reaction_species_validation(client):
    """xyz species 缺 charge → 422; smiles+xyz 都给 → 422"""
    r = client.post("/dft/reaction", headers=HEADERS, json={
        "reactants": [{"xyz_content": "1\n\nO 0 0 0"}],
        "products": [{"smiles": "O"}]})
    assert r.status_code == 422
    r = client.post("/dft/reaction", headers=HEADERS, json={
        "reactants": [{"smiles": "O", "xyz_content": "1\n\nO 0 0 0",
                       "charge": 0, "multiplicity": 1}],
        "products": [{"smiles": "O"}]})
    assert r.status_code == 422


def test_kill_reaction_tree_cascades(monkeypatch):
    """父任务取消 → 级联杀登记的子任务进程"""
    import subprocess as sp
    from dft_service.runners import executor as ex

    killed_pids = []

    def fake_run(cmd, **kw):
        if cmd[:2] == ["taskkill", "/PID"]:
            killed_pids.append(cmd[2])

        class R:
            returncode = 0
        return R()

    monkeypatch.setattr(sp, "run", fake_run)
    monkeypatch.setattr(ex.os, "name", "nt")

    class P:
        def __init__(self, pid):
            self.pid = pid

    ex._POPEN["parent01"] = (P(111), None)
    ex._POPEN["child0001"] = (P(222), None)
    ex._POPEN["child0002"] = (P(333), None)
    ex.register_children("parent01", ["child0001", "child0002"])
    n = ex.kill_reaction_tree("parent01")
    assert n == 2  # 存活子进程数
    assert set(killed_pids) == {"222", "333", "111"}
    assert not ex._CHILDREN.get("parent01")


# ---------------------------------------------------------------
# 遗留修复: SPC/E 真水模型路径
# ---------------------------------------------------------------
def test_water_model_spce_validation(client, monkeypatch):
    import dft_service.api as api_mod

    captured = {}

    async def fake(task_id, p, timeout_s):
        captured["p"] = p
        return {"status": "success", "tool": "gromacs"}

    monkeypatch.setitem(api_mod.RUNNERS, "gromacs", fake)
    # 非纯水 + spce → 422
    r = client.post("/dft/gromacs", headers=HEADERS, json={
        "smiles": "CCO", "water_model": "spce"})
    assert r.status_code == 422
    # 纯水 + spce → 透传
    r = client.post("/dft/gromacs", headers=HEADERS, json={
        "smiles": "O", "water_model": "spce", "box_nm": 2.2})
    assert r.status_code == 200
    _poll_result(client, r.json()["task_id"])
    assert captured["p"]["water_model"] == "spce"
    # 非法值 → 422
    r = client.post("/dft/gromacs", headers=HEADERS, json={
        "smiles": "O", "water_model": "tip4p-fb"})
    assert r.status_code == 422
