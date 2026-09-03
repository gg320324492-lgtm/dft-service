"""Gaussian driver — rdkit 3D → .gjf (含 SCRF 溶剂!) → g16.exe → parse log

修复 microbubble 版缺口 #1: solvent 只写 title 不生效。
现在 solvent 默认 none (气相), 指定时真正写 SCRF=(SMD,Solvent=XXX) 路由关键字。
修复缺口 #2: charge/multiplicity/nproc/mem 全部透传。
"""
from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

from _driver_common import load_params, run_driver, write_progress

sys.path.insert(0, "E:/sci-software/workflows")  # noqa: S104 — submit_gjf/parse_log 复用


# Gaussian 16 SMD 支持的溶剂名 → 规范写法 (覆盖常用; 未列出的按首字母大写透传)
_SOLVENT_MAP = {
    "water": "Water", "h2o": "Water",
    "ethanol": "Ethanol", "etoh": "Ethanol",
    "methanol": "Methanol", "meoh": "Methanol",
    "acetone": "Acetone",
    "acetonitrile": "Acetonitrile", "meCN": "Acetonitrile",
    "dmso": "DMSO",
    "dichloromethane": "Dichloromethane", "dcm": "Dichloromethane",
    "chloroform": "Chloroform",
    "toluene": "Toluene",
    "hexane": "n-Hexane", "n-hexane": "n-Hexane",
    "benzene": "Benzene",
    "thf": "THF",
    "dmf": "DMF",
    "ammonia": "Ammonia",
    "octanol": "n-Octanol", "n-octanol": "n-Octanol",
}
_SOLVENT_NONE = {"none", "gas", "gasphase", "vacuum", ""}

# 缺口 #3: freq 任务的频率与热化学量标记 (Gaussian log 标准输出行)
_THERMO_KEYS = {
    "zero_point_correction_hartree": "Zero-point correction=",
    "thermal_correction_energy_hartree": "Thermal correction to Energy=",
    "thermal_correction_enthalpy_hartree": "Thermal correction to Enthalpy=",
    "thermal_correction_gibbs_hartree": "Thermal correction to Gibbs Free Energy=",
    "sum_elec_zpe_hartree": "Sum of electronic and zero-point Energies=",
    "sum_elec_thermal_energy_hartree": "Sum of electronic and thermal Energies=",
    "sum_elec_thermal_enthalpy_hartree": "Sum of electronic and thermal Enthalpies=",
    "sum_elec_thermal_gibbs_hartree": "Sum of electronic and thermal Free Energies=",
}


def _parse_freq_thermo(log_text: str) -> dict:
    """从 Gaussian freq 输出提取频率 (cm⁻¹) 与热化学量 (Hartree)。

    纯函数, 无外部依赖, 可独立单测。负频率 = 虚频 (TS 特征)。
    """
    freqs = [
        float(v)
        for chunk in re.findall(r"Frequencies\s+--\s+([-\d.\s]+)", log_text)
        for v in chunk.split()
    ]
    thermo = {}
    for key, marker in _THERMO_KEYS.items():
        m = re.search(re.escape(marker) + r"\s*(-?\d+\.\d+)", log_text)
        if m:
            thermo[key] = float(m.group(1))
    out: dict = {"n_frequencies": len(freqs), "n_imaginary": sum(1 for f in freqs if f < 0)}
    if freqs:
        out["lowest_freq_cm_1"] = min(freqs)
        out["frequencies_cm_1"] = freqs
    if thermo:
        out["thermochemistry"] = thermo
    return out


def _smiles_to_coords(smiles: str):
    """rdkit SMILES → ([元素], [(x,y,z), ...]) — ETKDGv3 + MMFF/UFF"""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise RuntimeError(f"Failed to parse SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) != 0:
        raise RuntimeError(f"Failed to embed 3D: {smiles}")
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        AllChem.UFFOptimizeMolecule(mol, maxIters=500)
    conf = mol.GetConformer()
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    coords = [
        (conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z)
        for i in range(mol.GetNumAtoms())
    ]
    return atoms, coords


def _infer_charge_mult(smiles: str) -> tuple[int, int]:
    """从 SMILES 推断 (电荷, 自旋多重度) — 与 pyscf geom_driver 同逻辑"""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise RuntimeError(f"Failed to parse SMILES: {smiles}")
    charge = Chem.GetFormalCharge(mol)
    n_radical = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
    multiplicity = 1 + 2 * n_radical if n_radical > 0 else 1
    return charge, multiplicity


def _build_route(p: dict) -> str:
    """拼 Gaussian 路由行 (缺口 #28: extra_route 逃生舱追加在末尾)

    各字段已在 API 层用 _ROUTE_SAFE_RE 白名单过滤 (防换行/#/% 注入 gjf),
    这里信任输入。
    """
    # 纵深防御 (auto 路径不经 GaussianRequest 校验): 路由各元素必须纯安全字符,
    # 挡换行/#/% 注入 gjf (route 行之外的 %chk/%mem 行同理不可被越权改写)
    _elem_re = re.compile(r"[A-Za-z0-9=,()+.*\- ]{1,60}")
    for key in ("xc", "basis", "job"):
        val = str(p.get(key) or "")
        if val and not _elem_re.fullmatch(val):
            raise ValueError(f"路由字段 {key} 含非法字符: {val!r}")

    route_parts = [f"{p['xc']}/{p['basis']}"]
    if p.get("job"):
        route_parts.append(p["job"])
    solvent = (p.get("solvent") or "none").strip()
    if solvent.lower() not in _SOLVENT_NONE:
        g16_name = _SOLVENT_MAP.get(solvent.lower(), solvent.capitalize())
        # 溶剂名必须纯字母/连字符
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9\-]{0,30}", g16_name):
            raise ValueError(f"非法溶剂名: {solvent!r}")
        route_parts.append(f"SCRF=(SMD,Solvent={g16_name})")
    extra = (p.get("extra_route") or "").strip()
    if extra:
        if not re.fullmatch(r"[A-Za-z0-9=,()+.*\- \t]{1,200}", extra):
            raise ValueError(f"extra_route 含非法字符 (换行/#/% 等): {extra!r}")
        route_parts.append(extra)
    return "# " + " ".join(route_parts)


def _gen_gjf(workdir: Path, smiles: str, p: dict, chk_stem: str) -> Path:
    """自建 gjf 生成 (支持 SCRF) — 不用 workflows.gen_gjf 因为它 route 写死

    chk_stem: %chk 用全局唯一名 (缺口 #18) — g16 的 cwd 是安装目录,
    写死 input.chk 会让并发任务互相覆盖。
    """
    atoms, coords = _smiles_to_coords(smiles)

    route = _build_route(p)
    solvent = (p.get("solvent") or "none").strip()

    stem = "input"
    title = " ".join(smiles.split())[:100]  # 压掉换行 (smiles 直通 gjf 标题注释行)
    lines = [
        f"%nproc={p.get('nproc', 8)}",
        f"%mem={p.get('mem', '8GB')}",
        f"%chk={chk_stem}.chk",
        route,
        "",
        f"{title} {p['xc']}/{p['basis']} {p['job']} solvent={solvent}",
        "",
        f"{p['charge']} {p['multiplicity']}",
    ]
    lines += [
        f"{a:2s}  {c[0]:14.8f}  {c[1]:14.8f}  {c[2]:14.8f}"
        for a, c in zip(atoms, coords)
    ]
    gjf_path = workdir / f"{stem}.gjf"
    gjf_path.write_text("\n".join(lines) + "\n\n", encoding="utf-8")
    return gjf_path


def _cleanup_g16_dir(g16_dir: Path, stem: str) -> None:
    """删除 g16 安装目录里本任务的残留 (gjf/out/chk/其他 dft_job_<stem> 派生文件)

    正常路径与超时路径都调; 取消 (树杀 driver) 时由 executor 的 win_cleanup 兜底,
    scripts/cleanup.py 再对 mtime 超期孤儿做最终清扫。
    """
    for f in g16_dir.glob(f"{stem}.*"):
        try:
            f.unlink()
        except OSError:
            pass


def compute(params: dict, workdir: Path) -> dict:
    import subprocess
    import time as _time

    from gaussian_runner import parse_log  # 只复用解析; 提交自建 (见下)

    t0 = time.time()

    # 缺口 #2: charge/multiplicity 未提供时从 SMILES 推断 —
    # 旧行为默默按 0/1 跑, 阳离子/自由基返回看似成功的错误能量
    charge = params.get("charge")
    multiplicity = params.get("multiplicity")
    if charge is None or multiplicity is None:
        inf_charge, inf_mult = _infer_charge_mult(params["smiles"])
        charge = int(charge) if charge is not None else inf_charge
        multiplicity = int(multiplicity) if multiplicity is not None else inf_mult
    params = {**params, "charge": int(charge), "multiplicity": int(multiplicity)}

    # workdir 名含唯一 task_id; dft_job_ 前缀让 cleanup 能精确识别本服务的残留
    stem = f"dft_job_{workdir.name}"
    gjf_path = _gen_gjf(workdir, params["smiles"], params, chk_stem=stem)

    # ------------------------------------------------------------------
    # 提交 (2026-08-30 重写) — workflows.submit_gjf 的两个 Windows 不兼容:
    #   ① cwd=job 目录时 l1.exe 链起不来 (exit 127/静默死), 实证配方是
    #      cwd = g16 安装目录 (8/5 B1 验收 .out 头部 Output=D:\G16W\*.out 为证)
    #   ② G16W 输出扩展名是 .out 不是 .log, submit_gjf 轮询 .log 永远等不到
    # 配方: cwd=g16 目录 + 唯一 stem (并发安全, 信号量限 gaussian≤2) +
    #       干净 PATH (防 MSYS 污染) + 轮询 .out + 产物拷回 + 残留清理
    # ------------------------------------------------------------------
    g16_exe = Path(params.get("gaussian_bin") or r"D:\G16W\g16.exe").resolve()
    g16_dir = g16_exe.parent
    if not g16_exe.exists():
        return {
            "status": "unavailable",
            "error_msg": f"g16.exe not found: {g16_exe}",
        }

    # stem 已在 _gen_gjf 前定义 (dft_job_<workdir.name>), %chk/提交/清理三处共用
    write_progress(workdir, "submit", backend="g16")
    (g16_dir / f"{stem}.gjf").write_bytes(gjf_path.read_bytes())

    env = {
        "PATH": r"C:\Windows\System32;C:\Windows;" + str(g16_dir),
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", r"C:\Windows"),
        "GAUSS_EXEDIR": str(g16_dir),
        "GAUSS_SCRDIR": str(workdir),
    }
    proc = subprocess.Popen(
        [str(g16_exe), f"{stem}.gjf"], cwd=str(g16_dir),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env,
    )

    out_path = g16_dir / f"{stem}.out"
    deadline = _time.time() + float(params.get("timeout_s", 7200))
    n_tick = 0
    while _time.time() < deadline:
        if proc.poll() is not None and proc.returncode != 0:
            _cleanup_g16_dir(g16_dir, stem)  # 缺口 #18: 异常退出也清残留
            return {
                "status": "failed",
                "error_msg": f"Gaussian exited with code {proc.returncode}",
                "stage": "submit",
            }
        if out_path.exists():
            try:
                tail = out_path.read_text(encoding="utf-8", errors="ignore")[-4096:]
            except OSError:
                tail = ""
            if "Normal termination" in tail or "Error termination" in tail:
                _time.sleep(1.0)  # 让缓冲写完
                break
            # 缺口 #22: 每 ~15s 报一次进度 (opt 步数 / 累计 SCF 周期 / 末行)
            n_tick += 1
            if n_tick % 5 == 0:
                write_progress(
                    workdir, "running", backend="g16",
                    elapsed_s=round(_time.time() - t0),
                    opt_step=(lambda m: int(m.group(1)) if m else None)(
                        re.search(r"Step number\s+(\d+)", tail)),
                    scf_done=len(re.findall(r"SCF Done", tail)),
                    last_line=(tail.strip().splitlines() or [""])[-1][:120],
                )
        _time.sleep(3.0)
    else:
        proc.kill()
        _cleanup_g16_dir(g16_dir, stem)  # 缺口 #18: 超时残留
        return {
            "status": "timeout",  # #19: 一等状态 (原为 failed, CLI 退出码恒 1)
            "error_msg": f"Gaussian timeout after {params.get('timeout_s')}s",
            "stage": "submit",
        }

    # 产物拷回 workdir (out + chk), 再清安装目录残留
    local_out = workdir / "input.out"
    local_out.write_bytes(out_path.read_bytes())
    g16_chk = g16_dir / f"{stem}.chk"
    if g16_chk.exists():
        try:
            (workdir / "input.chk").write_bytes(g16_chk.read_bytes())
        except OSError:
            pass
    _cleanup_g16_dir(g16_dir, stem)
    log_for_parse = local_out

    parsed = parse_log(log_for_parse)
    elapsed = time.time() - t0

    result = {
        "status": "success" if parsed.converged else "completed_with_warnings",
        "tool": "gaussian",
        "energy_hartree": parsed.energy_hartree,
        "energy_ev": parsed.energy_ev,
        "n_opt_steps": parsed.n_opt_steps,
        "converged": parsed.converged,
        "charge": params["charge"],
        "multiplicity": params["multiplicity"],
        "log_path": str(local_out),
        "gjf_path": str(gjf_path),
        "work_dir": str(workdir),
        "smiles": params["smiles"],
        "xc": params["xc"],
        "basis": params["basis"],
        "job": params["job"],
        "route": _build_route(params),  # 缺口 #28: 回显最终路由 (含 SCRF/extra)
        "solvent": params.get("solvent", "none"),
        "elapsed_s": round(elapsed, 2),
        "extra": parsed.extra or {},
    }

    # 缺口 #3/#17 (2026-09-04): freq 任务提取频率 + 热化学量。
    # 必须放在 result 赋值之后 — 旧代码在之前 update 导致 UnboundLocalError,
    # freq 任务全部崩在组装阶段。error_msg 检查放最后, 保证失败状态优先。
    if "freq" in (params.get("job") or "").lower():
        freq_data = _parse_freq_thermo(
            local_out.read_text(encoding="utf-8", errors="ignore"))
        result.update(freq_data)
        if freq_data.get("n_imaginary"):
            result["warning"] = (
                f"{freq_data['n_imaginary']} 个虚频 — 若优化目标是极小值"
                f"(非过渡态), 该结构未收敛到极小值"
            )

    if parsed.error_msg:
        result["error_msg"] = parsed.error_msg
        result["status"] = "failed"
    return result


if __name__ == "__main__":
    run_driver(Path(sys.argv[1]), compute)
