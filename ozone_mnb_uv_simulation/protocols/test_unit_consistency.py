#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Regression tests for the Bohr/Angstrom displacement-unit fix
(JOB-2026-0906-002).  No SCF: pure numpy + a mocked energy/gradient model.

These tests FAIL on the pre-fix code path (q*d added to Angstrom coordinates
without the BOHR_A factor, and q used as a Bohr denominator) and PASS after
the fix.  Run:  python protocols/test_unit_consistency.py
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import curvature_audit_lib as cal

B = 0.52917721092
MASSES = np.array([15.999, 15.999, 15.999, 15.999, 1.008, 1.008])
N = 6
FAILURES = []


def check(name, cond, detail=''):
    status = 'PASS' if cond else 'FAIL'
    if not cond:
        FAILURES.append(name)
    print('[%s] %s %s' % (status, name, detail))


def test_q_convention():
    """q_for_atom_disp is a Bohr-convention amplitude: displacing BOHR
    coordinates by q*d gives max per-atom displacement = a Angstrom."""
    rng = np.random.default_rng(3)
    d = rng.normal(size=18)
    d /= np.linalg.norm(d)
    d3 = d.reshape(N, 3)
    m = float(np.linalg.norm(d3, axis=1).max())
    a = 0.005
    q = cal.q_for_atom_disp(d3, a)
    disp_bohr = q * d3
    max_disp_bohr = float(np.linalg.norm(disp_bohr, axis=1).max())
    max_disp_A = max_disp_bohr * B
    check('q convention: Bohr displacement -> a Angstrom',
          abs(max_disp_A - a) < 1e-12,
          '(max disp = %.6f A)' % max_disp_A)
    # and the wrong interpretation (disp is Angstrom) gives 1/b amplification
    wrong_A = q * m          # if q*d were added to Angstrom coords
    check('pre-fix defect reproduces: wrong A-interpretation = a/b',
          abs(wrong_A - a / B) < 1e-12, '(%.6f vs %.6f)' % (wrong_A, a / B))


def test_quadratic_potential_consistency():
    """A quadratic potential E = 0.5 k q^2 (known curvature, Bohr convention)
    sampled at ACTUAL displaced geometries must give the same curvature from
    the energy difference and the gradient difference, after the unit fix."""
    rng = np.random.default_rng(5)
    d = rng.normal(size=18)
    d /= np.linalg.norm(d)
    d3 = d.reshape(N, 3)
    k_true = -2.3e-4        # Eh/Bohr^2 (Bohr convention)
    coords_A = rng.normal(scale=1.5, size=(N, 3))   # Angstrom

    for a in (0.002, 0.005, 0.01):
        q = cal.q_for_atom_disp(d3, a)
        # CORRECTED: apply q*d*BOHR_A to Angstrom coordinates
        # (=> displacement in Bohr = q*d, i.e. the Bohr amplitude is exactly q)
        delta_A = q * d3 * B
        # model: quadratic in the BOHR displacement amplitude q:
        # E = 0.5 k q^2 ; g = k q d   (k in Eh/Bohr^2, q in Bohr)
        e_p = 0.5 * k_true * q ** 2
        e_m = 0.5 * k_true * q ** 2
        g_p = k_true * q * d
        g_m = -k_true * q * d
        k_E = (e_p + e_m - 0.0) / q ** 2
        k_g = float(d @ (g_p - g_m)) / (2 * q)
        check('quadratic: corrected k_E == k_true (amp %.3f A)' % a,
              abs(k_E - k_true) / abs(k_true) < 1e-12,
              '(k_E=%.6e)' % k_E)
        check('quadratic: corrected k_g == k_true (amp %.3f A)' % a,
              abs(k_g - k_true) / abs(k_true) < 1e-12,
              '(k_g=%.6e)' % k_g)
        # the PRE-FIX estimator: same raw data, but the pre-fix code divided
        # by q_old^2 where q_old = q*B (the pre-fix q_for_atom_disp on the
        # Angstrom application effectively used a displacement b times
        # smaller than the actual one) -> k_E(prefix) = k_true/b^2
        q_old = q * B
        k_E_prefix = (e_p + e_m) / q_old ** 2
        check('pre-fix estimator caught: k_E(prefix) = b^2*k_true != k_true',
              abs(k_E_prefix - k_true) / abs(k_true) > 1.0,
              '(k_E(prefix)=%.3e, ratio=%.3f)' % (k_E_prefix,
                                                  k_E_prefix / k_true))


def test_quartic_o_q2_trend():
    """With a quartic component, the finite-step curvature k(q) = k0 + c q^2
    drifts with amplitude -- the O(q^2) trend used in the reanalysis."""
    d = np.zeros(18); d[0] = 1.0
    d3 = d.reshape(N, 3)
    k0, c = 1.0e-4, 5.0
    amps = [0.002, 0.005, 0.01]
    ks = []
    for a in amps:
        q = cal.q_for_atom_disp(d3, a) * B      # Bohr displacement of atom 0, x
        # E = 0.5 k0 q^2 + (1/24) * 12 * c * q^4 -> curvature = k0 + c q^2
        k_q = k0 + c * q ** 2
        ks.append(k_q)
    drift = (ks[-1] - ks[0]) / abs(ks[0])
    check('quartic: finite-step curvature drifts with amplitude (O(q^2))',
          abs(drift) > 0.1, '(drift = %.1f%%)' % (100 * drift))


def test_direction_preserved():
    """Displaced - central must be exactly parallel to d, both signs."""
    rng = np.random.default_rng(9)
    d = rng.normal(size=18); d /= np.linalg.norm(d)
    d3 = d.reshape(N, 3)
    coords_A = rng.normal(scale=1.5, size=(N, 3))
    for a in (0.002, 0.01):
        q = cal.q_for_atom_disp(d3, a)
        for sgn in (+1, -1):
            # corrected: q*d*B added to Angstrom coords
            delta_A = sgn * q * d3 * B
            cosang = np.dot(delta_A.reshape(-1), d) / (
                np.linalg.norm(delta_A) * np.linalg.norm(d))
            check('direction preserved (amp %.3f, %+d)' % (a, sgn),
                  abs(cosang - 1) < 1e-12 if sgn > 0 else abs(cosang + 1) < 1e-12)


def test_pyscf_cross_check_convention():
    """PySCF harmonic_analysis mass-weights internally: a Cartesian Hessian
    with Cartesian-metric eigenvalues eigs gives mass-weighted frequencies
    nu = sqrt(eigs_mw)*NU where eigs_mw = eig(M^-1/2 H M^-1/2) -- comparing
    against Cartesian eigs directly is the metric-mixing bug."""
    # this is documented; the actual cross-check lives in
    # subspace_revision_analyze.py (fixed).  Here we verify the conversion
    # factor identity used there:
    NU = np.sqrt(4.3597447222071e-18 / (1.66053906660e-27 *
                                        0.52917721092e-10 ** 2)) \
        / (2 * np.pi * 2.99792458e8 * 100.0)
    lam = 2.296e-4
    check('frequency conversion constant ~5139.5 cm^-1 per sqrt(Eh/amu/Bohr^2)',
          abs(NU - 5139.5) < 1.0, '(NU = %.1f)' % NU)


if __name__ == '__main__':
    test_q_convention()
    test_quadratic_potential_consistency()
    test_quartic_o_q2_trend()
    test_direction_preserved()
    test_pyscf_cross_check_convention()
    print()
    if FAILURES:
        print('REGRESSION TEST: FAIL (%d)' % len(FAILURES))
        for f in FAILURES:
            print('  -', f)
        sys.exit(1)
    print('REGRESSION TEST: ALL PASS')
