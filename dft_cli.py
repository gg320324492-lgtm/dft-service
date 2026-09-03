r"""dft CLI — dft-service 命令行入口 (Claude Code / 终端直用)

用法示例:
    python dft_cli.py tools
    python dft_cli.py wait gaussian --smiles O --job opt
    python dft_cli.py wait pyscf --smiles O --basis sto-3g --solvent water
    python dft_cli.py wait auto --task optimize --quality fast
    python dft_cli.py submit gromacs --smiles O --time-ns 1      # 长任务只提交
    python dft_cli.py result <task_id>
    python dft_cli.py cancel <task_id>
    python dft_cli.py list --status success --tool pyscf
    python dft_cli.py cleanup --days 7 --dry-run

退出码: 0 成功 / 1 失败 / 2 超时 / 3 服务不可用 / 4 已取消 / 5 其他错误
(Claude Code Bash 工具可用退出码判断, 无需解析输出)

长任务模式: Bash 工具单次 ~10 分钟上限, 超过用
    submit 提交 → (稍后任意时刻) result <task_id> 轮询
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import httpx

# 中文 Windows 控制台默认 GBK, 编不出 ✓/✗ 等字符 → UnicodeEncodeError。
# CLI 输出统一 UTF-8 + replace (Claude Code/终端读 bytes, UTF-8 本来就是目标编码)
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent
DEFAULT_URL = os.environ.get("DFT_SERVICE_URL", "http://127.0.0.1:8620")
DEFAULT_KEY = os.environ.get("DFT_SERVICE_API_KEY", "")
POLL_INTERVAL_S = 3.0

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_TIMEOUT = 2
EXIT_UNAVAILABLE = 3
EXIT_CANCELLED = 4
EXIT_ERROR = 5

_TERMINAL = {"success", "completed_with_warnings", "failed",
             "unavailable", "cancelled", "interrupted", "timeout"}

# CLI 旗标 → API payload 字段 (只收显式提供的; 服务端 pydantic 忽略多余字段)
_FLAG_MAP = {
    "smiles": "smiles",
    "xc": "xc",
    "method": "method",
    "basis": "basis",
    "job": "job",
    "extra_route": "extra_route",
    "operation": "operation",
    "solvent": "solvent",
    "charge": "charge",
    "multiplicity": "multiplicity",
    "nproc": "nproc",
    "mem": "mem",
    "n_molecules": "n_molecules",
    "box_nm": "box_nm",
    "time_ns": "time_ns",
    "temperature_k": "temperature_K",
    "fmax": "fmax_ev_A",
    "max_steps": "max_steps",
    "model": "model",
    "device": "device",
    "task": "task",
    "quality": "quality",
    "max_opt_steps": "max_opt_steps",
    "timeout_s": "timeout_s",
}


def make_client(args: argparse.Namespace) -> httpx.Client:
    return httpx.Client(
        base_url=(getattr(args, "url", None) or DEFAULT_URL).rstrip("/"),
        headers={"X-API-Key": getattr(args, "api_key", None) or DEFAULT_KEY},
        timeout=httpx.Timeout(connect=5.0, read=30.0, write=30.0, pool=5.0),
    )


def api_call(client: httpx.Client, method: str, path: str,
             payload: dict | None = None) -> dict:
    try:
        r = client.request(method, path, json=payload) if payload is not None \
            else client.request(method, path)
        r.raise_for_status()
        return r.json()
    except httpx.ConnectError as e:
        return {"status": "unavailable",
                "error_msg": f"dft-service 不可达 ({client.base_url}) — "
                             "先启动: E:\\dft-service\\start.bat", "detail": repr(e)}
    except httpx.HTTPStatusError as e:
        try:
            return e.response.json()
        except Exception:
            return {"status": "failed", "error_msg": f"HTTP {e.response.status_code}"}


def build_payload(ns: argparse.Namespace) -> dict:
    payload: dict = {}
    for flag, field in _FLAG_MAP.items():
        val = getattr(ns, flag, None)
        if val is not None:
            payload[field] = val
    if ns.data:  # --data 整包逃生舱, 覆盖旗标
        payload.update(json.loads(ns.data))
    return payload


def human_summary(res: dict) -> str:
    icon = "✓" if res.get("status") == "success" else (
        "!" if res.get("status") == "completed_with_warnings" else "✗")
    lines = [f"{icon} [{res.get('status')}] tool={res.get('tool')} "
             f"task_id={res.get('task_id')} elapsed={res.get('elapsed_s')}s"]
    for k in ("energy_hartree", "energy_ev", "energy_eV", "converged",
              "scf_converged", "n_steps", "n_atoms", "n_imaginary",
              "lowest_freq_cm_1", "dipole_debye", "homo_lumo_gap_eV",
              "charge", "multiplicity", "backend"):
        if res.get(k) is not None:
            lines.append(f"  {k} = {res[k]}")
    thermo = res.get("thermochemistry")
    if isinstance(thermo, dict):
        for k, v in thermo.items():
            lines.append(f"  {k} = {v}")
    for k in ("work_dir", "log_path", "trajectory_path", "optimized_xyz"):
        if res.get(k):
            lines.append(f"  {k}: {res[k]}")
    for k in ("warning", "error_msg"):
        if res.get(k):
            lines.append(f"  {k}: {str(res[k])[:300]}")
    return "\n".join(lines)


def exit_code_for(status: str | None) -> int:
    return {
        "success": EXIT_OK, "completed_with_warnings": EXIT_OK,
        "failed": EXIT_FAILED, "timeout": EXIT_TIMEOUT,
        "unavailable": EXIT_UNAVAILABLE, "cancelled": EXIT_CANCELLED,
    }.get(status or "", EXIT_ERROR)


def wait_one(client: httpx.Client, tool: str, payload: dict,
             timeout_s: float, quiet: bool = False) -> dict:
    """提交 + 轮询到终态 (或超时)。返回服务端 result dict。"""
    task = api_call(client, "POST", f"/dft/{tool}", payload)
    if task.get("status") == "unavailable":
        return task
    task_id = task.get("task_id")
    if not task_id:
        return {"status": "failed", "error_msg": f"异常响应: {task}"}

    deadline = time.monotonic() + timeout_s
    n = 0
    while True:
        # #22: 中间态查 /status (带 progress.json 回读), 终态再取 /result
        st = api_call(client, "GET", f"/dft/status/{task_id}")
        if st.get("status") not in ("queued", "running", None):
            rec = api_call(client, "GET", f"/dft/result/{task_id}")
            result = rec.get("result") or {}
            result.setdefault("task_id", task_id)
            result.setdefault("tool", rec.get("tool"))
            return result
        if time.monotonic() > deadline:
            return {"status": "timeout", "task_id": task_id,
                    "error_msg": f"CLI 等待 {timeout_s:.0f}s 未完成 — "
                                 "任务仍在服务端, 稍后 result "
                                 f"{task_id} 再查"}
        n += 1
        if not quiet:
            # 进程度打 stderr — stdout 永远只有结果 (--json 可直接管道)
            prog = st.get("progress") or {}
            bits = " ".join(f"{k}={prog[k]}"
                            for k in ("stage", "opt_step", "scf_cycles",
                                      "time_ns", "elapsed_s")
                            if prog.get(k) is not None)
            print(f"\r  poll#{n} status={st.get('status')} "
                  + (f"[{bits}] " if bits else "")
                  + f"waited={int(n * POLL_INTERVAL_S)}s   ",
                  end="", file=sys.stderr, flush=True)
        time.sleep(POLL_INTERVAL_S)


# ------------------------------------------------------------------
# 子命令
# ------------------------------------------------------------------
def cmd_tools(args, client) -> int:
    body = api_call(client, "GET", "/dft/tools")
    if args.json:
        print(json.dumps(body, ensure_ascii=False, indent=2))
        return EXIT_OK
    print(f"available {body.get('available_count')}/{body.get('count')}:")
    for t in body.get("tools", []):
        mark = "✓" if t.get("available") else "✗"
        detail = t.get("details", {})
        extra = {k: v for k, v in detail.items()
                 if k in ("wsl_distro", "backend", "g16_path", "psi4", "mace")}
        print(f"  {mark} {t['name']:9s} {extra if extra else ''}")
    return EXIT_OK


_BATCH_CSV_FIELDS = ["smiles", "status", "energy_hartree", "elapsed_s",
                     "task_id", "error_msg"]


def _batch_row(smiles: str, res: dict) -> dict:
    return {
        "smiles": smiles, "status": res.get("status"),
        "energy_hartree": res.get("energy_hartree"),
        "elapsed_s": res.get("elapsed_s"),
        "task_id": res.get("task_id"),
        "error_msg": (res.get("error_msg") or "")[:120],
    }


def run_batch(client: httpx.Client, tool: str, payload: dict, mols: list[str],
              timeout_s: float, summary: str | None = None,
              resume: bool = False, quiet: bool = False) -> list[dict]:
    """缺口 #23: 批量筛选并发化 — 全部 submit 后轮询 (服务端信号量自然限流),

    旧版串行逐个等, 吞吐被客户端人为砍半。CSV 增量写出 (每条终态立即落盘),
    --resume 时读取已有 summary: 已终态的 smiles 跳过不重跑, 未终态的只重轮询
    不重提交 (task_id 已在 CSV 里)。返回全部行。timeout_s 为整批总时限。
    """
    rows: list[dict] = []
    pending: list[tuple[str, str]] = []  # (smiles, task_id)
    summary_path = Path(summary) if summary else None

    # resume 分类: 服务端已给最终答案的 → 跳过不重跑; timeout/queued/running
    # (CLI 放弃了但服务端可能还在算) → 按原 task_id 重轮询; unavailable/failed
    # 提交类 (无 task_id) → 落回 todo 重提交
    _RESUME_FINAL = {"success", "completed_with_warnings", "failed",
                     "cancelled", "interrupted"}
    if resume and summary_path and summary_path.exists():
        with open(summary_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("smiles") in mols and r.get("status") in _RESUME_FINAL:
                    rows.append(dict(r))
                elif r.get("task_id"):
                    pending.append((r["smiles"], r["task_id"]))
        if not quiet:
            print(f"resume: {len(rows)} 已完成, {len(pending)} 重轮询")

    done_smiles = {r["smiles"] for r in rows}
    pending_smiles = {s for s, _ in pending}
    todo = [m for m in dict.fromkeys(mols)  # 去重保持顺序
            if m not in done_smiles and m not in pending_smiles]
    for m in todo:
        resp = api_call(client, "POST", f"/dft/{tool}", {**payload, "smiles": m})
        if resp.get("task_id"):
            pending.append((m, resp["task_id"]))
            if not quiet:
                print(f"submitted {m} → {resp['task_id']}")
        else:  # 提交即失败 (校验/不可用), 直接落行
            row = _batch_row(m, {"status": resp.get("status", "failed"),
                                 "error_msg": resp.get("error_msg") or str(resp)})
            rows.append(row)
            if summary_path:
                _append_csv_row(summary_path, row)

    if summary_path and not summary_path.exists():
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=_BATCH_CSV_FIELDS).writeheader()

    deadline = time.monotonic() + timeout_s
    while pending:
        for m, tid in list(pending):
            st = api_call(client, "GET", f"/dft/status/{tid}")
            status = st.get("status")
            if status in ("queued", "running", None):
                continue
            pending.remove((m, tid))
            rec = api_call(client, "GET", f"/dft/result/{tid}")
            res = rec.get("result") or {}
            res.setdefault("task_id", tid)
            res.setdefault("status", status)
            row = _batch_row(m, res)
            rows.append(row)
            if summary_path:
                _append_csv_row(summary_path, row)
            if not quiet:
                print(f"[{len(rows)}/{len(mols)}] {m} → {status}")
        if not pending:
            break
        if time.monotonic() > deadline:
            for m, tid in pending:
                row = _batch_row(m, {"status": "timeout", "task_id": tid,
                                     "error_msg": "batch deadline exceeded "
                                                  "(任务仍在服务端, 可 --resume 续等)"})
                rows.append(row)
                if summary_path:
                    _append_csv_row(summary_path, row)
            break
        time.sleep(POLL_INTERVAL_S)
    return rows


def _append_csv_row(path: Path, row: dict) -> None:
    """增量落一行 (崩溃安全: --resume 的数据源就是这份文件)"""
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_BATCH_CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in _BATCH_CSV_FIELDS})


def cmd_wait(args, client) -> int:
    payload = build_payload(args)
    timeout_s = args.timeout

    if args.smiles_file:  # 缺口 #15: 批量 → #23: 并发 + 断点续跑
        mols = [ln.strip() for ln in Path(args.smiles_file).read_text(
            encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")]
        rows = run_batch(client, args.tool, payload, mols, timeout_s,
                         summary=args.summary, resume=getattr(args, "resume", False),
                         quiet=args.json)
        if args.json:
            print(json.dumps(rows, ensure_ascii=False, indent=2, default=str))
        if args.summary:
            print(f"CSV → {args.summary}")
        bad = sum(1 for r in rows if r["status"] != "success")
        return EXIT_FAILED if bad else EXIT_OK

    res = wait_one(client, args.tool, payload, timeout_s)
    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=2, default=str))
    else:
        print(human_summary(res))
    return exit_code_for(res.get("status"))


def cmd_submit(args, client) -> int:
    resp = api_call(client, "POST", f"/dft/{args.tool}", build_payload(args))
    if args.json:
        print(json.dumps(resp, ensure_ascii=False, indent=2))
    else:
        print(f"task_id={resp.get('task_id')} status={resp.get('status')} "
              f"tool={resp.get('tool') or args.tool}")
        if resp.get("reason"):
            print(f"reason: {resp['reason']}")
    return EXIT_UNAVAILABLE if resp.get("status") == "unavailable" else EXIT_OK


def cmd_status(args, client) -> int:
    body = api_call(client, "GET", f"/dft/status/{args.task_id}")
    print(json.dumps(body, ensure_ascii=False, indent=2))
    return EXIT_OK


def cmd_result(args, client) -> int:
    body = api_call(client, "GET", f"/dft/result/{args.task_id}")
    if body.get("result") and not args.json:
        res = body["result"]
        res.setdefault("task_id", body["task_id"])
        print(human_summary(res))
        return exit_code_for(body.get("status"))
    print(json.dumps(body, ensure_ascii=False, indent=2))
    return exit_code_for(body.get("status"))


def cmd_cancel(args, client) -> int:
    body = api_call(client, "DELETE", f"/dft/jobs/{args.task_id}")
    print(json.dumps(body, ensure_ascii=False, indent=2))
    return EXIT_OK if body.get("status") in ("cancelled",) or \
        body.get("message") == "task already finished" else EXIT_ERROR


def cmd_list(args, client) -> int:
    q = f"?limit={args.limit}&offset={args.offset}"
    if args.tool:
        q += f"&tool={args.tool}"
    if args.status:
        q += f"&status={args.status}"
    body = api_call(client, "GET", f"/dft/jobs{q}")
    if args.json:
        print(json.dumps(body, ensure_ascii=False, indent=2))
        return EXIT_OK
    print(f"total={body.get('total')} (limit={body.get('limit')}, "
          f"offset={body.get('offset')})")
    for j in body.get("jobs", []):
        print(f"  {j['task_id']}  {j['tool']:9s} {j['status']:25s} "
              f"{j['smiles'][:30]:30s} {j['submit_time'][:19]}")
    return EXIT_OK


def _g16_scratch_dir() -> Path | None:
    """Gaussian 安装目录 (g16.exe 的父目录) — 缺口 #18 残留清扫用。

    driver 正常/超时路径自己清, 但 cancel/kill 树杀时 driver 子进程直接死亡,
    dft_job_*.gjf/.out/.chk 会残留在安装目录。这里按 mtime 超期兜底清扫
    (运行中任务的 mtime 一直刷新, 天然不会被 cutoff 命中)。
    """
    gb = os.environ.get("DFT_SERVICE_GAUSSIAN_BIN", r"E:\sci-software\g16w\g16.exe")
    d = Path(gb).parent
    return d if d.is_dir() else None


def _sweep_g16_scratch(cutoff: float, dry_run: bool) -> int:
    d = _g16_scratch_dir()
    if d is None:
        return 0
    n = 0
    for f in d.glob("dft_job_*"):
        if not f.is_file():
            continue
        try:
            stale = f.stat().st_mtime < cutoff
        except OSError:
            continue
        if not stale:
            continue
        if dry_run:
            print(f"[dry-run] would remove g16 scratch {f.name}")
        else:
            try:
                f.unlink()
                print(f"removed g16 scratch {f.name}")
            except OSError:
                pass
        n += 1
    return n


def run_cleanup(days: int = 7, dry_run: bool = False,
                purge_rows: bool = False) -> tuple[int, int]:
    """缺口 #10: 清理过期 job 目录 (直接读 SQLite + 扫目录, 无需服务在跑)。

    只动终态任务; 运行中/排队中的目录绝不碰。返回 (removed, kept)。
    """
    db_path = ROOT / "data" / "dft_service.db"
    jobs_root = ROOT / "data" / "jobs"
    if not db_path.exists():
        print("no DB yet — nothing to clean")
        return 0, 0
    conn = sqlite3.connect(db_path)
    rows = dict(conn.execute("SELECT id, status FROM dft_jobs").fetchall())
    cutoff = time.time() - days * 86400
    victims: list[Path] = []
    for d in sorted(jobs_root.glob("*")) if jobs_root.exists() else []:
        if not d.is_dir():
            continue
        task_id = d.name.rsplit("_", 1)[-1]
        status = rows.get(task_id)
        if status not in _TERMINAL:
            continue
        if d.stat().st_mtime < cutoff:
            victims.append(d)
    for d in victims:
        if dry_run:
            print(f"[dry-run] would remove {d.name}")
        else:
            import shutil

            shutil.rmtree(d, ignore_errors=True)
            print(f"removed {d.name}")
    if purge_rows and not dry_run:
        ids = [d.name.rsplit("_", 1)[-1] for d in victims]
        if ids:
            marks = ",".join("?" * len(ids))
            conn.execute(f"DELETE FROM dft_jobs WHERE id IN ({marks})", ids)
            conn.commit()
            print(f"purged {len(ids)} DB rows")
    total = sum(1 for _ in jobs_root.glob("*")) if jobs_root.exists() else 0
    kept = total - len(victims)
    print(f"{len(victims)} job dir(s) older than {days}d (total {total}, kept {kept})")
    conn.close()
    n_scratch = _sweep_g16_scratch(cutoff, dry_run)  # 缺口 #18: g16 目录残留兜底
    if n_scratch:
        print(f"{n_scratch} g16 scratch file(s) older than {days}d")
    return len(victims), kept


def cmd_cleanup(args, client) -> int:  # noqa: ARG001 — 本地操作, 不需要 client
    removed, _ = run_cleanup(args.days, args.dry_run, args.purge_rows)
    return EXIT_OK


# ------------------------------------------------------------------
# 参数解析
# ------------------------------------------------------------------
def add_tool_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--smiles")
    p.add_argument("--smiles-file", help="批量: 每行一个 SMILES (# 注释)")
    p.add_argument("--summary", help="批量 CSV 输出路径")
    p.add_argument("--xc", help="gaussian 泛函")
    p.add_argument("--method", help="pyscf/psi4 方法")
    p.add_argument("--basis")
    p.add_argument("--job", help="gaussian: opt/sp/freq/opt freq")
    p.add_argument("--extra-route", dest="extra_route",
                   help="gaussian 路由追加关键字 (int=ultrafine / scf qc 等)")
    p.add_argument("--operation", help="pyscf/psi4: energy/optimize/properties")
    p.add_argument("--solvent", help="gaussian SMD / pyscf C-PCM 溶剂")
    p.add_argument("--charge", type=int)
    p.add_argument("--multiplicity", type=int)
    p.add_argument("--nproc", type=int)
    p.add_argument("--mem")
    p.add_argument("--n-molecules", dest="n_molecules", type=int)
    p.add_argument("--box-nm", dest="box_nm", type=float)
    p.add_argument("--time-ns", dest="time_ns", type=float)
    p.add_argument("--temperature-k", dest="temperature_k", type=float)
    p.add_argument("--fmax", type=float, help="mace fmax (eV/A)")
    p.add_argument("--max-steps", dest="max_steps", type=int)
    p.add_argument("--model", help="mace: small/medium/large")
    p.add_argument("--device", help="mace: cuda/cpu/auto")
    p.add_argument("--task", help="auto: energy/optimize/freq/properties/md")
    p.add_argument("--quality", help="auto: fast/accurate/auto")
    p.add_argument("--max-opt-steps", dest="max_opt_steps", type=int)
    p.add_argument("--timeout-s", dest="timeout_s", type=float,
                   help="服务端任务超时 (秒)")
    p.add_argument("--data", help='整包 JSON 逃生舱, 如 \'{"smiles":"O","job":"sp"}\'')


def build_parser() -> argparse.ArgumentParser:
    # 共享旗标 (SUPPRESS 默认: 子命令未显式给时保留顶层解析值)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true",
                        default=argparse.SUPPRESS, help="输出原始 JSON")
    common.add_argument("--url", default=argparse.SUPPRESS)
    common.add_argument("--api-key", default=argparse.SUPPRESS)

    ap = argparse.ArgumentParser(prog="dft", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--api-key", default=DEFAULT_KEY)
    ap.add_argument("--json", action="store_true", help="输出原始 JSON")
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name, help_ in [("wait", "提交 + 轮询到完成 (阻塞, 人类可读摘要)"),
                        ("submit", "只提交, 返回 task_id (长任务用)")]:
        sp = sub.add_parser(name, help=help_, parents=[common])
        sp.add_argument("tool", choices=["gaussian", "gromacs", "mace",
                                         "pyscf", "psi4", "auto"])
        if name == "wait":
            sp.add_argument("--timeout", type=float, default=540.0,
                            help="CLI 侧最长等待秒 (默认 540, 适配 Bash 上限); "
                                 "批量模式为整批总时限")
            sp.add_argument("--resume", action="store_true",
                            help="批量: 复用已有 --summary CSV, 跳过已终态分子, "
                                 "未完成的只重轮询原 task_id")
        add_tool_args(sp)
        sp.set_defaults(func=cmd_wait if name == "wait" else cmd_submit)

    sp = sub.add_parser("tools", help="5 后端健康状态", parents=[common])
    sp.set_defaults(func=cmd_tools)
    sp = sub.add_parser("status", help="查任务状态", parents=[common])
    sp.add_argument("task_id")
    sp.set_defaults(func=cmd_status)
    sp = sub.add_parser("result", help="拿任务结果", parents=[common])
    sp.add_argument("task_id")
    sp.set_defaults(func=cmd_result)
    sp = sub.add_parser("cancel", help="取消任务 (树杀运行中进程)",
                        parents=[common])
    sp.add_argument("task_id")
    sp.set_defaults(func=cmd_cancel)
    sp = sub.add_parser("list", help="任务列表", parents=[common])
    sp.add_argument("--tool")
    sp.add_argument("--status")
    sp.add_argument("--limit", type=int, default=20)
    sp.add_argument("--offset", type=int, default=0)
    sp.set_defaults(func=cmd_list)
    sp = sub.add_parser("cleanup", help="清理过期 job 目录 (本地操作)",
                        parents=[common])
    sp.add_argument("--days", type=int, default=7)
    sp.add_argument("--dry-run", action="store_true")
    sp.add_argument("--purge-rows", action="store_true",
                    help="连 DB 行一起删 (默认只删目录)")
    sp.set_defaults(func=cmd_cleanup)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # 全局 --json 只挂顶层; subparser 里也允许
    if not hasattr(args, "json"):
        args.json = False
    try:
        with make_client(args) as client:
            return args.func(args, client)
    except KeyboardInterrupt:
        print("\ninterrupted")
        return EXIT_ERROR
    except Exception as e:  # noqa: BLE001
        print(f"CLI error: {e!r}")
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
