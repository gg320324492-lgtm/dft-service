"""gaussian_driver.compute() 组装链路测试 (缺口 #17/#25)

driver 顶层只依赖 stdlib + _driver_common, service venv 可直接 import;
rdkit / g16.exe / gaussian_runner 全部 stub, 不真跑计算。
"""
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

_DRIVERS = Path(__file__).resolve().parents[1] / "dft_service" / "runners" / "drivers"
if str(_DRIVERS) not in sys.path:
    sys.path.insert(0, str(_DRIVERS))

import gaussian_driver as gd  # noqa: E402

# 假 G16W 输出: 3 个频率 (含 1 虚频) + 全套热化学标记 + 正常终止
_FREQ_OUT = """
 Some header
 Frequencies --    1650.12   1650.13
 Frequencies --    -120.45
 Zero-point correction=                           0.056789 (Hartree/Particle)
 Thermal correction to Energy=                    0.059362
 Thermal correction to Enthalpy=                  0.060306
 Thermal correction to Gibbs Free Energy=         0.040142
 Sum of electronic and zero-point Energies=     -76.301331
 Sum of electronic and thermal Energies=        -76.298758
 Sum of electronic and thermal Enthalpies=      -76.297814
 Sum of electronic and thermal Free Energies=   -76.317978

 Normal termination of Gaussian 16
"""

_PLAIN_OUT = """
 SCF Done:  E(RB3LYP) =  -76.4000000000     A.U. after    9 cycles
                                                               V = -76.4000000000
 Normal termination of Gaussian 16
"""


class _Parsed:
    def __init__(self, converged=True, error_msg=None):
        self.converged = converged
        self.error_msg = error_msg
        self.energy_hartree = -76.4
        self.energy_ev = -76.4 * 27.211386245988
        self.n_opt_steps = 0
        self.extra = {}


@pytest.fixture()
def g16_env(tmp_path, monkeypatch):
    """stub rdkit 辅助 + gaussian_runner.parse_log + g16.exe + Popen"""
    monkeypatch.setattr(gd, "_smiles_to_coords",
                        lambda s: (["O", "H", "H"], [(0, 0, 0), (0, 0, 1), (0, 1, 0)]))
    monkeypatch.setattr(gd, "_infer_charge_mult", lambda s: (0, 1))

    state = {"parsed": _Parsed(), "out_text": _FREQ_OUT}
    mod = types.ModuleType("gaussian_runner")
    mod.parse_log = lambda p: state["parsed"]
    monkeypatch.setitem(sys.modules, "gaussian_runner", mod)

    g16_dir = tmp_path / "g16"
    g16_dir.mkdir()
    exe = g16_dir / "g16.exe"
    exe.write_bytes(b"")

    class FakeProc:
        returncode = 0

        def poll(self):
            return 0

        def kill(self):
            pass

    def fake_popen(cmd, **kw):
        # cmd = [g16_exe, "job_<stem>.gjf"]; cwd = g16_dir
        stem = Path(cmd[1]).stem
        (g16_dir / f"{stem}.out").write_text(state["out_text"], encoding="utf-8")
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    def make_params(**over):
        p = {"smiles": "O", "xc": "B3LYP", "basis": "6-31G(d)", "job": "freq",
             "solvent": "none", "timeout_s": 60, "gaussian_bin": str(exe)}
        p.update(over)
        return p

    workdir = tmp_path / "gaussian_test1234"
    workdir.mkdir()
    return workdir, make_params, state, g16_dir


def test_freq_merges_thermo_into_result(g16_env):
    """#17 回归: freq 任务 result 必须含频率与热化学量 (旧代码 UnboundLocalError)"""
    workdir, make_params, _, _ = g16_env
    res = gd.compute(make_params(), workdir)
    assert res["status"] == "success"
    assert res["n_frequencies"] == 3
    assert res["n_imaginary"] == 1
    assert res["lowest_freq_cm_1"] == pytest.approx(-120.45)
    assert res["thermochemistry"]["zero_point_correction_hartree"] == pytest.approx(0.056789)
    assert "虚频" in res["warning"]
    assert (workdir / "input.out").exists()


def test_sp_task_skips_freq_block(g16_env):
    workdir, make_params, state, _ = g16_env
    state["out_text"] = _PLAIN_OUT
    res = gd.compute(make_params(job="sp"), workdir)
    assert res["status"] == "success"
    assert "n_frequencies" not in res


def test_parse_error_beats_freq_warning(g16_env):
    """parsed.error_msg → status=failed 优先于 freq 合并 (freq 数据仍可回查)"""
    workdir, make_params, state, _ = g16_env
    state["parsed"] = _Parsed(converged=False, error_msg="SCF not converged")
    res = gd.compute(make_params(), workdir)
    assert res["status"] == "failed"
    assert res["error_msg"] == "SCF not converged"
    assert res["n_frequencies"] == 3


def test_g16_missing_is_unavailable(tmp_path, monkeypatch, g16_env):
    """license/二进制缺失 → unavailable ≠ failed (CLAUDE.md 错误哲学)"""
    workdir, make_params, _, _ = g16_env
    res = gd.compute(make_params(gaussian_bin=str(tmp_path / "nope" / "g16.exe")),
                     workdir)
    assert res["status"] == "unavailable"


# ---------------------------------------------------------------
# #25: gjf 路由组装 (纯字符串逻辑, 只 stub 坐标)
# ---------------------------------------------------------------
def _gjf_text(tmp_path, monkeypatch, params):
    monkeypatch.setattr(gd, "_smiles_to_coords",
                        lambda s: (["O"], [(0.0, 0.0, 0.0)]))
    wd = tmp_path / "gaussian_gjf"
    wd.mkdir()
    p = {"xc": "B3LYP", "basis": "6-31G(d)", "job": "opt", "solvent": "none",
         "charge": 0, "multiplicity": 1, "nproc": 8, "mem": "8GB", **params}
    path = gd._gen_gjf(wd, p.get("smiles", "O"), p, chk_stem="dft_job_gaussian_xyz")
    return path.read_text(encoding="utf-8")


def test_gjf_solvent_writes_scrf(tmp_path, monkeypatch):
    """缺口 #1: solvent 真写 SCRF=(SMD,...) 而非只进标题 (缺口 #18: %chk 唯一名)"""
    txt = _gjf_text(tmp_path, monkeypatch, {"smiles": "O", "solvent": "water"})
    assert "SCRF=(SMD,Solvent=Water)" in txt
    assert "%chk=dft_job_gaussian_xyz.chk" in txt  # 并发安全唯一名, 非 input.chk
    assert "input.chk" not in txt


def test_gjf_gas_phase_no_scrf(tmp_path, monkeypatch):
    txt = _gjf_text(tmp_path, monkeypatch, {"solvent": "none"})
    assert "SCRF" not in txt


def test_gjf_charge_mult_line(tmp_path, monkeypatch):
    """缺口 #2: charge/multiplicity 透传到坐标块首行"""
    txt = _gjf_text(tmp_path, monkeypatch, {"charge": 1, "multiplicity": 2})
    assert "\n1 2\n" in txt


def test_gjf_unknown_solvent_capitalized(tmp_path, monkeypatch):
    """未在映射表的溶剂按首字母大写透传 (Gaussian SMD 名称约定)"""
    txt = _gjf_text(tmp_path, monkeypatch, {"solvent": "pyridine"})
    assert "Solvent=Pyridine" in txt


def test_gjf_extra_route(tmp_path, monkeypatch):
    """缺口 #28: extra_route 追加进路由行 (逃生舱)"""
    txt = _gjf_text(tmp_path, monkeypatch,
                    {"extra_route": "int=ultrafine scf=(qc,maxcycle=200)"})
    assert "# B3LYP/6-31G(d) opt int=ultrafine scf=(qc,maxcycle=200)" in txt


def test_opt_freq_route(tmp_path, monkeypatch):
    """缺口 #27: opt freq 联跑 — 路由同时含两关键字"""
    txt = _gjf_text(tmp_path, monkeypatch, {"job": "opt freq"})
    assert "# B3LYP/6-31G(d) opt freq" in txt


def test_extra_route_rejects_newline_injection():
    """缺口 #28: 换行/#/% 注入被 driver 层兜底拒绝 (auto 路径不过 API 校验)"""
    p = {"xc": "B3LYP", "basis": "6-31G(d)", "job": "sp", "solvent": "none",
         "extra_route": "int=ultrafine\n# pop"}
    with pytest.raises(ValueError, match="非法字符"):
        gd._build_route(p)


def test_solvent_rejects_injection():
    p = {"xc": "B3LYP", "basis": "6-31G(d)", "job": "sp",
         "solvent": "Water) weird("}
    with pytest.raises(ValueError, match="溶剂名"):
        gd._build_route(p)


# ---------------------------------------------------------------
# #29: 内联 xyz 输入
# ---------------------------------------------------------------
def test_parse_xyz_text_std_and_bare():
    from _driver_common import parse_xyz_text
    std = parse_xyz_text("2\nwater\nO 0 0 0\nH 0 0 0.96\n")
    assert std[0] == ["O", "H"] and std[1][1] == (0.0, 0.0, 0.96)
    bare = parse_xyz_text("O 0.0 0.0 0.0\nh 0.0 0.0 0.96\ncL 1 1 1")
    assert bare[0] == ["O", "H", "Cl"]
    with pytest.raises(ValueError):
        parse_xyz_text("1\nx\nO 0 0\n")       # 字段不足
    with pytest.raises(ValueError):
        parse_xyz_text("")                     # 空


def test_gjf_from_xyz_content(tmp_path, monkeypatch):
    """xyz 输入不走 rdkit (_smiles_to_coords 故意炸) 也能出正确 gjf 坐标块"""
    monkeypatch.setattr(gd, "_smiles_to_coords",
                        lambda s: (_ for _ in ()).throw(AssertionError("不该调 rdkit")))
    wd = tmp_path / "gaussian_xyz"
    wd.mkdir()
    p = {"xc": "B3LYP", "basis": "6-31G(d)", "job": "sp", "solvent": "none",
         "charge": -1, "multiplicity": 1, "nproc": 4, "mem": "4GB"}
    path = gd._gen_gjf(wd, None, p, chk_stem="dft_job_x",
                       atoms_coords=([("O"), ("H")],
                                     [(0.0, 0.0, 0.0), (0.0, 0.0, 0.96)]))
    txt = path.read_text(encoding="utf-8")
    assert "-1 1" in txt
    atom_lines = [ln for ln in txt.splitlines() if ln.strip().startswith(("O ", "O  "))
                  or (ln.split()[:1] == ["O"] and "0.00000000" in ln)]
    assert len(atom_lines) == 1 and atom_lines[0].split()[1:] == \
        ["0.00000000", "0.00000000", "0.00000000"]
    assert "inline-xyz" in txt  # 标题占位


def test_compute_xyz_requires_charge(g16_env):
    """缺 charge/mult → failed 带明确指引 (API 已拦, driver 纵深兜底)"""
    workdir, make_params, _, _ = g16_env
    res = gd.compute(make_params(smiles=None,
                                 xyz_content="1\n\nO 0 0 0"), workdir)
    assert res["status"] == "failed"
    assert "charge" in res["error_msg"]


# ---------------------------------------------------------------
# #22: write_progress 原子写
# ---------------------------------------------------------------
def test_write_progress_atomic(tmp_path):
    import _driver_common as dc
    import json
    dc.write_progress(tmp_path, "running", opt_step=3)
    payload = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert payload["stage"] == "running"
    assert payload["opt_step"] == 3
    assert "updated_at" in payload
    assert not (tmp_path / "progress.json.tmp").exists()  # rename 完成, 无残留 tmp


def test_opt_freq_waits_for_process_exit(g16_env, monkeypatch):
    """#27 回归: 多步任务第 1 步末就写 "Normal termination"+
    "Proceeding to internal job step 2" — 完成判定必须等进程真正退出,
    否则拷回半成品日志 (频率段缺失, 2026-09-04 实测踩中)"""
    workdir, make_params, state, g16_dir = g16_env
    # 文件从第一次轮询起就带 "Normal termination", 但进程还没退出
    calls = {"poll": 0}

    class StillRunning:
        returncode = 0

        def poll(self):
            calls["poll"] += 1
            return None if calls["poll"] < 3 else 0

        def kill(self):
            pass

    def fake_popen(cmd, **kw):
        stem = Path(cmd[1]).stem
        (g16_dir / f"{stem}.out").write_text(state["out_text"], encoding="utf-8")
        return StillRunning()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(time, "sleep", lambda s: None)  # 免真实等待
    res = gd.compute(make_params(job="opt freq"), workdir)
    assert res["status"] == "success"
    assert calls["poll"] >= 3  # 旧代码 poll=1 时就会被中间标记骗走
