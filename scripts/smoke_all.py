r"""五后端真算冒烟验收 — 把 8/30 人工实证固化为脚本 (缺口 #24)

改任何 driver / 部署新版后跑一遍, 专拦 test_gaussian_driver 这类 mock 测不到的
组装 bug (如 #17 freq 必崩)。

    python scripts/smoke_all.py                        # 默认集: pyscf/mace/psi4 (秒级)
    python scripts/smoke_all.py --with-gaussian        # + Gaussian sp (依赖 license)
    python scripts/smoke_all.py --with-gaussian --gaussian-freq   # 也跑 freq (#17 回归)
    python scripts/smoke_all.py --with-gromacs         # + GROMACS 短 MD (约 1-2 分钟)
    python scripts/smoke_all.py --url http://127.0.0.1:8621        # 指到其他实例

前提: 目标 dft-service 实例已启动 (start.bat 或 run.py)。
退出码: 0 全 PASS / 1 有 FAIL / 3 服务不可达。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dft_cli import EXIT_UNAVAILABLE, api_call, make_client  # noqa: E402

# (name, tool, payload, 单跑时限s, 判活字段)
_DEFAULT_CASES = [
    ("pyscf sp sto-3g", "pyscf",
     {"smiles": "O", "basis": "sto-3g", "operation": "energy"}, 300,
     "energy_hartree"),
    ("mace relax H2O", "mace",
     {"smiles": "O", "max_steps": 50}, 600,
     "energy_ev"),
    ("psi4 sp sto-3g", "psi4",
     {"smiles": "O", "basis": "sto-3g", "operation": "energy"}, 600,
     "energy_hartree"),
]
_GAUSSIAN_CASES = [
    ("gaussian sp sto-3g", "gaussian",
     {"smiles": "O", "basis": "sto-3g", "job": "sp"}, 900,
     "energy_hartree"),
]
_GAUSSIAN_FREQ_CASES = [
    # #17 回归: freq 组装链路 (频率/热化学量提取)
    ("gaussian freq sto-3g", "gaussian",
     {"smiles": "O", "basis": "sto-3g", "job": "freq"}, 1200,
     "frequencies_cm_1"),
]
_GROMACS_CASES = [
    # demo 级拓扑 (GROMOS 水盒), 只验链路不验科学量; analyze 顺带验 #31
    ("gromacs 10x0.01ns", "gromacs",
     {"smiles": "O", "n_molecules": 10, "box_nm": 2.6, "time_ns": 0.01,
      "analyze": True}, 1800,
     "rmsd_avg_nm"),
]


def run_case(client, name: str, tool: str, payload: dict,
             budget_s: float, key: str) -> tuple[str, str, str]:
    """提交一个冒烟用例 → (name, PASS/FAIL/SKIP, 详情)"""
    t0 = time.monotonic()
    resp = api_call(client, "POST", f"/dft/{tool}", payload)
    task_id = resp.get("task_id")
    if not task_id:
        return name, "FAIL", f"submit 异常: {resp}"
    deadline = t0 + budget_s
    while True:
        st = api_call(client, "GET", f"/dft/status/{task_id}")
        status = st.get("status")
        if status not in ("queued", "running", None):
            break
        if time.monotonic() > deadline:
            return name, "FAIL", f"超 {budget_s:.0f}s 未终态 (task {task_id})"
        time.sleep(3)
    rec = api_call(client, "GET", f"/dft/result/{task_id}")
    result = rec.get("result") or {}
    elapsed = round(time.monotonic() - t0, 1)
    if status == "unavailable":
        return name, "SKIP", "后端不可用 (不计失败)"
    if status != "success":
        return name, "FAIL", (f"status={status}: "
                              f"{(result.get('error_msg') or '')[:200]}")
    if key not in result:
        return name, "FAIL", f"结果缺 {key} 字段 (组装 bug?)"
    val = result[key]
    if isinstance(val, list):
        val = f"{key[:12]}={val[:3]}…({len(val)})"
    return name, "PASS", f"{key}={val} ({elapsed}s)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=None, help="服务地址 (默认 $DFT_SERVICE_URL/8620)")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--with-gaussian", action="store_true")
    ap.add_argument("--gaussian-freq", action="store_true",
                    help="含 gaussian freq 用例 (#17 回归, 较慢)")
    ap.add_argument("--with-gromacs", action="store_true")
    args = ap.parse_args()

    class _A:  # make_client 的鸭子接口
        url = args.url
        api_key = args.api_key

    cases = list(_DEFAULT_CASES)
    if args.with_gaussian:
        cases += _GAUSSIAN_CASES
        if args.gaussian_freq:
            cases += _GAUSSIAN_FREQ_CASES
    if args.with_gromacs:
        cases += _GROMACS_CASES

    with make_client(_A()) as client:
        health = api_call(client, "GET", "/health")
        if health.get("status") != "ok":
            print(f"✗ dft-service 不可达 — 先启动 (start.bat) 或 --url 指定。{health.get('error_msg', '')}")
            return EXIT_UNAVAILABLE
        print(f"smoke_all → {client.base_url} | {len(cases)} case(s)\n")
        results = [run_case(client, *c) for c in cases]

    width = max(len(r[0]) for r in results) + 1
    n_fail = 0
    for name, verdict, detail in results:
        icon = {"PASS": "✓", "FAIL": "✗", "SKIP": "-"}[verdict]
        print(f"{icon} {name:<{width}} {verdict:5s} {detail}")
        n_fail += verdict == "FAIL"
    print(f"\n{len(results) - n_fail} ok, {n_fail} fail "
          f"(SKIP=后端不可用, 不算失败)")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
