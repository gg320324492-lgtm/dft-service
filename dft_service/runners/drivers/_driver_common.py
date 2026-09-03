"""driver 公共骨架 — 每个 driver 以 `scichem_python <driver> <workdir>` 运行

约定:
- 读   <workdir>/params.json   (runner 写入的参数)
- 写   <workdir>/result.json   (永远写, 哪怕失败 — status=failed + error_msg)
- 打印 result.json 内容到 stdout (双通道, runner 优先读文件)
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))  # _driver_common 可 import


def load_params(workdir: Path) -> dict:
    return json.loads((workdir / "params.json").read_text(encoding="utf-8"))


def write_result(workdir: Path, result: dict) -> None:
    out = workdir / "result.json"
    out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(out.read_text(encoding="utf-8"))


def write_progress(workdir: Path, stage: str, **fields) -> None:
    """缺口 #22: 长任务进度上报 — 原子写 <workdir>/progress.json。

    /dft/status 在任务 running 时回读该文件 (读失败 = null, 不影响任务)。
    永不抛异常: 进度是 best-effort 旁路, 炸了也不能带崩计算。
    """
    try:
        payload = {"stage": stage, "updated_at": time.time(), **fields}
        tmp = workdir / "progress.json.tmp"
        tmp.write_text(json.dumps(payload, ensure_ascii=False, default=str),
                       encoding="utf-8")
        os.replace(tmp, workdir / "progress.json")
    except Exception:  # noqa: BLE001
        pass


def run_driver(workdir: Path, compute) -> None:
    """统一入口: compute(params) -> result dict; 任何异常包成 failed"""
    try:
        params = load_params(workdir)
        result = compute(params, workdir)
        result.setdefault("status", "success")
    except Exception as e:  # noqa: BLE001 — driver 边界必须兜住一切
        result = {
            "status": "failed",
            "error_msg": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(limit=6),
        }
    write_result(workdir, result)
