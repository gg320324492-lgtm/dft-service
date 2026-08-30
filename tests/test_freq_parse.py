r"""gaussian_driver._parse_freq_thermo 纯函数单测 — 缺口 #3 (freq 结果解析)

driver 文件顶层只有 stdlib import (rdkit 在函数内), 可直接 import。
"""
from __future__ import annotations

import sys
from pathlib import Path

DRIVERS = Path(__file__).resolve().parent.parent / "dft_service" / "runners" / "drivers"
sys.path.insert(0, str(DRIVERS))

from gaussian_driver import _parse_freq_thermo  # noqa: E402

_SYNTHETIC_LOG = """
 SCF Done:  E(RB3LYP) =  -76.410532121     A.U. after   10 cycles
 Step number   4
 Frequencies --    1827.3211              3801.4523              3918.1204
 Red. masses --       1.0873                 1.0624                 1.0481
 Frequencies --    3924.5321
 Red. masses --       1.0481
 Zero-point correction=                           0.021453 (Hartree/Particle)
 Thermal correction to Energy=                    0.023724
 Thermal correction to Enthalpy=                  0.024668
 Thermal correction to Gibbs Free Energy=         0.004567
 Sum of electronic and zero-point Energies=           -76.412345
 Sum of electronic and thermal Energies=              -76.410074
 Sum of electronic and thermal Enthalpies=            -76.409130
 Sum of electronic and thermal Free Energies=         -76.429231
 Normal termination of Gaussian 16 at ...
"""

_SYNTHETIC_TS_LOG = """
 Frequencies --  -512.3341               1421.2213              2870.1102
 Zero-point correction=                           0.018000
 Normal termination of Gaussian 16
"""


def test_parse_freq_counts_and_values():
    d = _parse_freq_thermo(_SYNTHETIC_LOG)
    assert d["n_frequencies"] == 4  # 3 + 1
    assert d["n_imaginary"] == 0
    assert d["lowest_freq_cm_1"] == 1827.3211
    assert d["frequencies_cm_1"][0] == 1827.3211
    assert d["frequencies_cm_1"][-1] == 3924.5321


def test_parse_thermochemistry_all_eight_keys():
    d = _parse_freq_thermo(_SYNTHETIC_LOG)
    t = d["thermochemistry"]
    assert t["zero_point_correction_hartree"] == 0.021453
    assert t["thermal_correction_energy_hartree"] == 0.023724
    assert t["thermal_correction_enthalpy_hartree"] == 0.024668
    assert t["thermal_correction_gibbs_hartree"] == 0.004567
    assert t["sum_elec_zpe_hartree"] == -76.412345
    assert t["sum_elec_thermal_energy_hartree"] == -76.410074
    assert t["sum_elec_thermal_enthalpy_hartree"] == -76.409130
    assert t["sum_elec_thermal_gibbs_hartree"] == -76.429231


def test_parse_imaginary_frequencies_flagged():
    d = _parse_freq_thermo(_SYNTHETIC_TS_LOG)
    assert d["n_imaginary"] == 1
    assert d["lowest_freq_cm_1"] == -512.3341
    assert "thermochemistry" not in d or "zero_point_correction_hartree" \
        in d.get("thermochemistry", {})


def test_parse_empty_log_returns_counts_zero():
    d = _parse_freq_thermo("no freq data here\n")
    assert d == {"n_frequencies": 0, "n_imaginary": 0}
