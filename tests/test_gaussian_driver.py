"""gaussian_driver.compute() 组装链路测试 (缺口 #17/#25)

driver 顶层只依赖 stdlib + _driver_common, service venv 可直接 import;
rdkit / g16.exe / gaussian_runner 全部 stub, 不真跑计算。
"""
import subprocess
import sys
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
