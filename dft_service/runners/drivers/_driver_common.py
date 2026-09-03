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


def parse_xyz_text(text: str) -> tuple[list[str], list[tuple[float, float, float]]]:
    """xyz 文本 → (元素列表, 坐标列表) — 缺口 #29 内联几何输入共用

    支持标准 2 行头 (n_atoms + comment, comment 可空可省) 与无头裸坐标块
    (每行 `El x y z [额外列...]`)。坐标单位 Å (不转换)。抛 ValueError 由
    driver 边界兜成 failed。
    """
    lines = text.splitlines()
    try:  # 首非空行是整数 → 视为原子数声明 (标准 xyz: 声明行 + 注释行 + n 原子行)
        i = next(k for k, ln in enumerate(lines) if ln.strip())
        n = int(lines[i].split()[0])
        keep = [ln for ln in lines[i + 2:]
                if ln.strip() and not ln.strip().startswith("#")]
        if len(keep) < n:  # 也可能没有注释行 (声明后直接原子)
            keep = [ln for ln in lines[i + 1:]
                    if ln.strip() and not ln.strip().startswith("#")]
        atom_lines = keep[:n]
        if not atom_lines or not atom_lines[0].split()[0][:1].isalpha():
            raise ValueError  # 声明行其实是坐标首行 (裸块) → 非真 header
    except (StopIteration, ValueError):
        atom_lines = [ln for ln in lines
                      if ln.strip() and not ln.strip().startswith("#")]
    atoms: list[str] = []
    coords: list[tuple[float, float, float]] = []
    for i, ln in enumerate(atom_lines, 1):
        parts = ln.split()
        if len(parts) < 4:
            raise ValueError(f"xyz 第 {i} 行字段不足: {ln!r}")
        el = parts[0]
        if not el[:1].isalpha():
            raise ValueError(f"xyz 第 {i} 行首字段非元素符号: {ln!r}")
        atoms.append(el.capitalize() if len(el) > 1 else el.upper())
        try:
            coords.append((float(parts[1]), float(parts[2]), float(parts[3])))
        except ValueError as e:
            raise ValueError(f"xyz 第 {i} 行坐标非法: {ln!r}") from e
    if not atoms:
        raise ValueError("xyz 为空")
    return atoms, coords


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
