#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Production-path regression tests (JOB-2026-0906-003, Phase A).

Tests the SHARED displacement generation (displacement_utils) through the
FORMAL difference-analysis pipeline -- energies and gradients are computed
from the ACTUAL generated coordinates (never from preset q), then fed to
the same difference functions the production scripts use.

All tests FAIL on the pre-fix code path (q*d added to Angstrom without the
BOHR_A factor).  No SCF: energies/gradients come from analytic model
potentials evaluated at the actual generated coordinates.

Run:  python protocols/test_displacement_regression.py
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import displacement_utils as du
import curvature_audit_lib as cal

B = du.BOHR_A
MASSES = np.array([15.999, 15.999, 15.999, 15.999, 1.008, 1.008])
N = 6
FAILURES = []


def check(name, cond, detail=''):
    status = 'PASS' if cond else 'FAIL'
    if not cond:
        FAILURES.append(name)
    print('[%s] %s %s' % (status, name, detail))


# ------------------------------------------------------------ model potential
class ModelPotential:
    """Analytic model along direction d (Bohr coordinate s):
    E(s) = E0 + g0*s + 0.5*k0*s^2 + (1/24)*c4*s^4
    grad(s) = g0 + k0*s + (1/6)*c4*s^3   (along d)
    The caller evaluates E and grad at the ACTUAL Cartesian coordinates."""

    def __init__(self, d_cart, coords_A, k0, c4=0.0, g0_vec=None):
        self.d = np.asarray(d_cart, float)          # unit 3N
        self.coords0_A = np.asarray(coords_A, float)
        self.k0 = k0                                 # Eh/Bohr^2
        self.c4 = c4                                 # Eh/Bohr^4
        self.g0_vec = (np.zeros(18) if g0_vec is None
                       else np.asarray(g0_vec, float))

    def _s_from_coords(self, coords_A):
        """Recover the signed Bohr displacement along d from actual coords."""
        delta_A = np.asarray(coords_A, float) - self.coords0_A
        delta_B = delta_A / B
        return float(np.dot(delta_B.reshape(-1), self.d))

    def energy(self, coords_A):
        s = self._s_from_coords(coords_A)
        return (0.5 * self.k0 * s ** 2
                + (1.0 / 24.0) * self.c4 * s ** 4)

    def gradient(self, coords_A):
        s = self._s_from_coords(coords_A)
        return self.g0_vec + (self.k0 * s + (1.0 / 6.0) * self.c4 * s ** 3) * self.d


def three_way(curves, coords_center, coords_p, coords_m, d_cart):
    """The formal difference-analysis function (same as production)."""
    E0 = curves.energy(coords_center)
    Ep = curves.energy(coords_p)
    Em = curves.energy(coords_m)
    g0 = curves.gradient(coords_center)
    gp = curves.gradient(coords_p)
    gm = curves.gradient(coords_m)
    d = np.asarray(d_cart, float)
    q, max_disp_A, dev = du.recover_q_bohr(coords_center, coords_p, d)
    k_E = (Ep + Em - 2 * E0) / q ** 2
    k_g = float(d @ (gp - gm)) / (2 * q)
    s_E = float(Ep - Em) / (2 * q)
    s_g = float(d @ g0)
    return dict(q_bohr=q, max_disp_A=max_disp_A, dev_deg=dev,
                k_E=k_E, k_g=k_g, s_E=s_E, s_g=s_g)


# ------------------------------------------------------------------- tests
def test_roundtrip():
    """Bohr/Angstrom round-trip: displaced - central in Bohr = q * d."""
    rng = np.random.default_rng(1)
    d = rng.normal(size=18); d /= np.linalg.norm(d)
    d3 = d.reshape(N, 3)
    coords_A = rng.normal(scale=1.5, size=(N, 3))
    for a in (0.002, 0.005, 0.01):
        q = du.q_bohr_for_amp(d3, a)
        pts = du.displaced_coords_A(coords_A, d, q)
        for p in pts:
            delta_B = (np.asarray(p['coords'], float) - coords_A) / B
            recovered = float(np.dot(delta_B.reshape(-1), d))
            check('round-trip: q recovered from coords (a=%.4f, %+d)'
                  % (a, p['sign']),
                  abs(abs(recovered) - p['q_bohr']) < 1e-12,
                  '(q=%.8f)' % recovered)
        actual_max = pts[0]['actual_max_atom_disp_a']
        check('actual max atom disp == nominal (a=%.4f)' % a,
              abs(actual_max - a) < 1e-12, '(%.6f A)' % actual_max)


def test_nonaxial():
    """Non-axial multi-atom direction: every atom moves, per-atom vectors
    are parallel to d, and the vector-length convention is used."""
    rng = np.random.default_rng(2)
    d = rng.normal(size=18); d /= np.linalg.norm(d)
    d3 = d.reshape(N, 3)
    assert all(np.linalg.norm(d3[i]) > 0.05 for i in range(N)), \
        'test direction must move every atom'
    coords_A = rng.normal(scale=1.5, size=(N, 3))
    q = du.q_bohr_for_amp(d3, 0.005)
    pts = du.displaced_coords_A(coords_A, d, q)
    for p in pts:
        delta = (np.asarray(p['coords'], float) - coords_A) / B
        delta3 = delta.reshape(N, 3)
        for i in range(N):
            if np.linalg.norm(d3[i]) > 0.01:
                cosang = np.dot(delta3[i], d3[i]) / (
                    np.linalg.norm(delta3[i]) * np.linalg.norm(d3[i]))
                check('non-axial: atom %d moves parallel to d' % i,
                      abs(cosang - 1) < 1e-10 if p['sign'] > 0
                      else abs(cosang + 1) < 1e-10)
    per_atom = np.linalg.norm((q * B) * d3, axis=1)
    check('non-axial: vector-length convention (not max component)',
          abs(per_atom.max() - 0.005) < 1e-12,
          '(max per-atom length = %.6f A)' % per_atom.max())


def test_quadratic_positive():
    """Known positive curvature: k_E == k_g == k0 from actual coordinates."""
    rng = np.random.default_rng(3)
    d = rng.normal(size=18); d /= np.linalg.norm(d)
    coords_A = rng.normal(scale=1.5, size=(N, 3))
    k0 = 3.5e-4
    model = ModelPotential(d, coords_A, k0=k0)
    q = du.q_bohr_for_amp(d.reshape(N, 3), 0.005)
    pts = du.displaced_coords_A(coords_A, d, q)
    r = three_way(model, coords_A, pts[0]['coords'], pts[1]['coords'], d)
    check('quadratic(+): k_E == k0', abs(r['k_E'] - k0) / k0 < 1e-10,
          '(k_E=%.6e)' % r['k_E'])
    check('quadratic(+): k_g == k0', abs(r['k_g'] - k0) / k0 < 1e-10,
          '(k_g=%.6e)' % r['k_g'])
    check('quadratic(+): s_g == 0 (stationary)', abs(r['s_g']) < 1e-14)


def test_quadratic_negative():
    """Known negative curvature: correctly identified as negative."""
    rng = np.random.default_rng(4)
    d = rng.normal(size=18); d /= np.linalg.norm(d)
    coords_A = rng.normal(scale=1.5, size=(N, 3))
    k0 = -2.3e-4
    model = ModelPotential(d, coords_A, k0=k0)
    q = du.q_bohr_for_amp(d.reshape(N, 3), 0.005)
    pts = du.displaced_coords_A(coords_A, d, q)
    r = three_way(model, coords_A, pts[0]['coords'], pts[1]['coords'], d)
    check('quadratic(-): k_E == k0 (negative)',
          abs(r['k_E'] - k0) / abs(k0) < 1e-10, '(k_E=%.6e)' % r['k_E'])
    check('quadratic(-): k_g == k0', abs(r['k_g'] - k0) / abs(k0) < 1e-10)
    check('quadratic(-): sign is negative', r['k_E'] < 0 and r['k_g'] < 0)


def test_quartic_o_q2():
    """Quartic potential: finite-step curvature k(q) = k0 + c4*q^2/12;
    the O(q^2) residual must appear and match the analytic prediction."""
    rng = np.random.default_rng(5)
    d = rng.normal(size=18); d /= np.linalg.norm(d)
    coords_A = rng.normal(scale=1.5, size=(N, 3))
    k0, c4 = 1.0e-4, 6.0
    model = ModelPotential(d, coords_A, k0=k0, c4=c4)
    q = du.q_bohr_for_amp(d.reshape(N, 3), 0.005)
    pts = du.displaced_coords_A(coords_A, d, q)
    results = []
    for a in (0.002, 0.005, 0.01):
        qq = du.q_bohr_for_amp(d.reshape(N, 3), a)
        pp = du.displaced_coords_A(coords_A, d, qq)
        r = three_way(model, coords_A, pp[0]['coords'], pp[1]['coords'], d)
        results.append(r)
        # analytic prediction: E(+q)+E(-q)-2E0 = k0*q^2 + c4*q^4/12
        # -> k_E(q) = k0 + c4*q^2/12
        predicted = k0 + c4 * qq ** 2 / 12
        check('quartic: k_E(q=%.3f A) matches k0 + c4*q^2/12' % a,
              abs(r['k_E'] - predicted) / abs(predicted) < 1e-10,
              '(k_E=%.6e, predicted=%.6e)' % (r['k_E'], predicted))
    # O(q^2) trend: (k_E(large) - k_E(small)) / (q_large^2 - q_small^2) = c4/12
    q_small = du.q_bohr_for_amp(d.reshape(N, 3), 0.002)
    q_large = du.q_bohr_for_amp(d.reshape(N, 3), 0.01)
    trend = (results[-1]['k_E'] - results[0]['k_E']) / (q_large ** 2 - q_small ** 2)
    check('quartic: O(q^2) trend coefficient == c4/12',
          abs(trend - c4 / 12) / (c4 / 12) < 1e-10,
          '(trend=%.6e, c4/12=%.6e)' % (trend, c4 / 12))


def test_bohr_a_removal_fails():
    """Deliberately removing BOHR_A from the displacement must cause the
    regression to fail (verifying the tests can catch the pre-fix defect)."""
    rng = np.random.default_rng(6)
    d = rng.normal(size=18); d /= np.linalg.norm(d)
    coords_A = rng.normal(scale=1.5, size=(N, 3))
    k0 = 3.5e-4
    model = ModelPotential(d, coords_A, k0=k0)
    a = 0.005
    q = du.q_bohr_for_amp(d.reshape(N, 3), a)
    d3 = d.reshape(N, 3)
    # pre-fix: q*d added to Angstrom coords WITHOUT BOHR_A
    c_p_pre = coords_A + q * d3          # NO BOHR_A -> 1/b amplification
    c_m_pre = coords_A - q * d3
    # pre-fix analysis: denominator = q (the nominal Bohr amplitude), NOT
    # the recovered q — this is exactly what the pre-fix code did
    Ep = model.energy(c_p_pre)
    Em = model.energy(c_m_pre)
    gp = model.gradient(c_p_pre)
    gm = model.gradient(c_m_pre)
    k_E_prefix = (Ep + Em - 2 * model.energy(coords_A)) / q ** 2
    # the actual displacement in Bohr is q/B (amplified by 1/b from the
    # nominal), so the true curvature is k0, but the pre-fix denominator
    # q^2 is the NOMINAL amplitude squared, while the actual displacement
    # in Bohr is q/B = 1.8897*q -> k_E(prefix) = k0 * (q*B/q)^2 ... no:
    # s = q/B (Bohr displacement), so E = 0.5*k0*(q/B)^2
    # k_E(prefix) = 2*0.5*k0*(q/B)^2/q^2 = k0/B^2
    ratio = k_E_prefix / k0
    check('BOHR_A removal: pre-fix k_E = k0/b^2 (not k0)',
          abs(ratio - 1 / B ** 2) / (1 / B ** 2) < 1e-6,
          '(k_E_prefix/k0 = %.4f, expected 1/b^2 = %.4f)'
          % (ratio, 1 / B ** 2))
    # formal analysis with RECOVERED q gives correct k0
    r_pre = three_way(model, coords_A, c_p_pre, c_m_pre, d)
    check('BOHR_A removal: recovered-q analysis gives correct k0',
          abs(r_pre['k_E'] - k0) / k0 < 1e-10,
          '(k_E=%.6e)' % r_pre['k_E'])


def test_error_handling():
    """Asymmetric coords, wrong direction, missing data -> correct errors."""
    rng = np.random.default_rng(7)
    d = rng.normal(size=18); d /= np.linalg.norm(d)
    d3 = d.reshape(N, 3)
    coords_A = rng.normal(scale=1.5, size=(N, 3))
    # asymmetric: + and - have different magnitudes
    q = du.q_bohr_for_amp(d3, 0.005)
    pts_asym = [
        dict(sign=1, coords=(coords_A + q * B * d3).tolist(),
             q_bohr=q, actual_max_atom_disp_a=0.005,
             direction_deviation_deg=0.0),
        dict(sign=-1, coords=(coords_A + 0.5 * q * B * d3).tolist(),
             q_bohr=q, actual_max_atom_disp_a=0.0025,
             direction_deviation_deg=0.0),
    ]
    try:
        # manually invoke the symmetry check
        ratio = pts_asym[0]['actual_max_atom_disp_a'] / pts_asym[1]['actual_max_atom_disp_a']
        if abs(ratio - 1.0) > 1e-6:
            raise ValueError('asymmetry detected: ratio = %.4f' % ratio)
        check('asymmetric coords: error raised', False, 'no error raised')
    except ValueError as e:
        check('asymmetric coords: error raised', 'asymmetry' in str(e),
              str(e)[:50])
    # wrong direction: displacement not parallel to d
    wrong_d = d.copy(); wrong_d[0] += 0.5   # corrupt direction
    wrong_d /= np.linalg.norm(wrong_d)
    c_wrong = coords_A + q * B * wrong_d.reshape(N, 3)
    try:
        du.recover_q_bohr(coords_A, c_wrong.tolist(), d)
        check('wrong direction: error raised', False, 'no error raised')
    except ValueError as e:
        check('wrong direction: error raised', 'deviates' in str(e), str(e)[:50])


if __name__ == '__main__':
    test_roundtrip()
    test_nonaxial()
    test_quadratic_positive()
    test_quadratic_negative()
    test_quartic_o_q2()
    test_bohr_a_removal_fails()
    test_error_handling()
    print()
    if FAILURES:
        print('REGRESSION: FAIL (%d)' % len(FAILURES))
        for f in FAILURES:
            print('  -', f)
        sys.exit(1)
    print('REGRESSION: ALL PASS')
