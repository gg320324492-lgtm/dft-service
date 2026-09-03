"""dft_cli.run_batch 测试 (缺口 #23) — fake api_call, 不碰真实服务"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dft_cli  # noqa: E402


def _install_fake_api(monkeypatch, fail_on=None):
    """模拟服务端: submit 分 task_id; status 第 2 轮全部终态; result 回能量"""
    state = {"poll_round": 0, "submitted": []}

    def fake_api_call(client, method, path, payload=None):
        if method == "POST" and path.startswith("/dft/"):
            smi = payload["smiles"]
            state["submitted"].append(smi)
            if fail_on and smi in fail_on:
                return {"status": "unavailable", "error_msg": f"no backend for {smi}"}
            return {"task_id": f"t_{smi}", "status": "queued"}
        if "/status/" in path:
            tid = path.rsplit("/", 1)[1]
            state["poll_round"] += 1
            # 每个任务第 2 次被问就完成 (poll_round 是全局计数, 简单用奇偶)
            done = state.setdefault("seen", {}).get(tid, 0)
            state["seen"][tid] = done + 1
            return {"status": "running" if done < 1 else "success"}
        if "/result/" in path:
            tid = path.rsplit("/", 1)[1]
            return {"status": "success",
                    "result": {"status": "success", "tool": "pyscf",
                               "energy_hartree": -1.0, "elapsed_s": 2.5,
                               "smiles": tid[2:]}}
        raise AssertionError(path)

    monkeypatch.setattr(dft_cli, "api_call", fake_api_call)
    monkeypatch.setattr(dft_cli, "POLL_INTERVAL_S", 0)  # 去真实等待
    return state


def test_run_batch_submits_all_before_polling(tmp_path, monkeypatch):
    state = _install_fake_api(monkeypatch)
    rows = dft_cli.run_batch(None, "pyscf", {"basis": "sto-3g"},
                             ["O", "S", "CO"], timeout_s=30, quiet=True)
    # 全部一次性提交 (旧串行版不可能做到)
    assert state["submitted"] == ["O", "S", "CO"]
    assert len(rows) == 3
    assert all(r["status"] == "success" for r in rows)


def test_run_batch_csv_incremental(tmp_path, monkeypatch):
    _install_fake_api(monkeypatch)
    summary = tmp_path / "s.csv"
    rows = dft_cli.run_batch(None, "pyscf", {}, ["O", "S"],
                             timeout_s=30, summary=str(summary), quiet=True)
    with open(summary, newline="", encoding="utf-8") as f:
        got = list(csv.DictReader(f))
    assert len(got) == 2 and len(rows) == 2
    assert {r["smiles"] for r in got} == {"O", "S"}
    assert got[0]["energy_hartree"] == "-1.0"


def test_run_batch_submit_failure_recorded(tmp_path, monkeypatch):
    _install_fake_api(monkeypatch, fail_on={"X"})
    rows = dft_cli.run_batch(None, "gromacs", {}, ["O", "X"],
                             timeout_s=30, quiet=True)
    by = {r["smiles"]: r for r in rows}
    assert by["X"]["status"] == "unavailable"
    assert "no backend" in by["X"]["error_msg"]
    assert by["O"]["status"] == "success"


def test_run_batch_resume_skips_finished(tmp_path, monkeypatch):
    """CSV 已有终态行不再重提交; 未终态行按原 task_id 重轮询"""
    summary = tmp_path / "s.csv"
    with open(summary, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=dft_cli._BATCH_CSV_FIELDS)
        w.writeheader()
        w.writerow({**{k: "" for k in dft_cli._BATCH_CSV_FIELDS},
                    "smiles": "O", "status": "success",
                    "energy_hartree": "-1.0", "task_id": "t_O"})
        w.writerow({**{k: "" for k in dft_cli._BATCH_CSV_FIELDS},
                    "smiles": "S", "status": "timeout", "task_id": "t_S"})

    state = _install_fake_api(monkeypatch)
    rows = dft_cli.run_batch(None, "pyscf", {}, ["O", "S", "CO"],
                             timeout_s=30, summary=str(summary),
                             resume=True, quiet=True)
    # O 已终态不重跑; S 之前 timeout (非终态? timeout 是终态 → 也应跳过)
    # 实现约定: _TERMINAL 含 timeout → S 被跳过重提交, 但它的行已在 CSV
    # CO 新提交
    assert "O" not in state["submitted"]
    assert "CO" in state["submitted"]
    assert len([r for r in rows if r["smiles"] == "O"]) == 1  # 不重复
