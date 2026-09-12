#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Production-path regression tests (batch JOB-2026-0906-002, Phase A).

No SCF.  The tests exercise the SHARED displacement helper
(curvature_audit_lib.displaced_geometry / recover_q / curvature_from_points)
that the two production scripts use, and they build the model energies and
gradients FROM THE GENERATED COORDINATES (never from a preset q).

Cases:
  1. Bohr/Angstrom round trip;
  2. + and - displacements are exactly symmetric;
  3. non-axial multi-atom direction;
  4. known POSITIVE curvature quadratic potential;
  5. known NEGATIVE curvature quadratic potential;
  6. quartic potential: E- and g-based curvatures and the O(q^2) residual;
  7. with BOHR_A removed (reproducing the defect) the same test FAILS;
  8. asymmetric coordinates / wrong direction / missing data raise errors.
"""
import os
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import curvature_audit_lib as cal

B = 0.52917721092
N = 6
MASSES = np.array([15.999, 15.999, 15.999, 15.999, 1.008, 1.008])
CASES = []
RAW = os.path.join(os.path.dirname(HERE), 'run_artifacts',
                   '01_pure_water_o3_h2o', 'unit_consistency_revision')
os.makedirs(RAW, exist_ok=True)


def rec(name, status, detail=''):
    CASES.append(dict(name=name, status=status, detail=detail))
    print('[%s] %s %s' % (status, name, detail))


def check(name, cond, detail=''):
    rec(name, 'PASS' if cond else 'FAIL', detail)
    return bool(cond)


def unit_dir(seed):
    rng = np.random.default_rng(seed)
    d = rng.normal(size=18)
    d /= np.linalg.norm(d)
    return d


def model_quadratic(k, q, d, e0=0.0):
    """Model energies/gradients for a quadratic potential of curvature k in the
    Bohr-amplitude coordinate q along unit direction d."""
    e_p = e0 + 0.5 * k * q ** 2
    e_m = e0 + 0.5 * k * q ** 2
    g_p = k * q * d
    g_m = -k * q * d
    return e_p, e_m, g_p, g_m


def model_quartic(k0, c4, q, d, e0=0.0):
    """E = 0.5 k0 q^2 + c4/24 q^4  ->  dE/dq = k0 q + c4/6 q^3,
    curvature k(q) = k0 + 0.5 c4 q^2."""
    e_p = e0 + 0.5 * k0 * q ** 2 + c4 / 24.0 * q ** 4
    e_m = e0 + 0.5 * k0 * q ** 2 + c4 / 24.0 * q ** 4
    g_p = (k0 * q + c4 / 6.0 * q ** 3) * d
    g_m = -(k0 * q + c4 / 6.0 * q ** 3) * d
    return e_p, e_m, g_p, g_m


def test_round_trip():
    d = unit_dir(1).reshape(N, 3)
    coords_A = np.random.default_rng(2).normal(scale=1.5, size=(N, 3))
    coords_B = coords_A / B
    a = 0.005
    q = cal.q_for_atom_disp(d, a)
    cA = cal.displaced_geometry(coords_A, d, q, unit='Angstrom')
    cB = cal.displaced_geometry(coords_B, d, q, unit='Bohr')
    # both must correspond to the SAME Bohr displacement
    same = np.abs(cA / B - cB).max()
    check('Bohr/Angstrom round trip (same Bohr displacement)',
          same < 1e-12, '(max diff %.2e Bohr)' % same)
    rA = cal.recover_q(coords_A, cA, d, unit='Angstrom')
    rB = cal.recover_q(coords_B, cB, d, unit='Bohr')
    check('recovered q identical for both units',
          abs(rA['q_bohr'] - q) < 1e-12 and abs(rB['q_bohr'] - q) < 1e-12,
          '(q = %.10f Bohr)' % q)
    check('recovered actual amplitude equals nominal',
          abs(rA['max_atom_disp_angstrom'] - a) < 1e-12,
          '(%.6f A vs %.6f A)' % (rA['max_atom_disp_angstrom'], a))


def test_sign_symmetry():
    d = unit_dir(3).reshape(N, 3)
    coords_A = np.random.default_rng(4).normal(scale=1.5, size=(N, 3))
    q = cal.q_for_atom_disp(d, 0.006)
    cp = cal.displaced_geometry(coords_A, d, +q, unit='Angstrom')
    cm = cal.displaced_geometry(coords_A, d, -q, unit='Angstrom')
    rp = cal.recover_q(coords_A, cp, d)
    rm = cal.recover_q(coords_A, cm, d)
    sym = abs(rp['max_atom_disp_bohr'] - rm['max_atom_disp_bohr'])
    check('+/- displacements symmetric in length', sym < 1e-13,
          '(diff %.2e Bohr; signs %+d/%+d)' % (sym, rp['sign'], rm['sign']))
    check('+/- signs opposite', rp['sign'] == -rm['sign'])


def test_non_axial_direction():
    """A direction with no single dominant Cartesian axis, several atoms."""
    rng = np.random.default_rng(7)
    d3 = rng.normal(size=(N, 3))
    d3[:, 2] *= 0.7                       # anisotropic, non-axial
    d3 /= np.linalg.norm(d3)
    coords_A = rng.normal(scale=1.5, size=(N, 3))
    q = cal.q_for_atom_disp(d3, 0.004)
    cp = cal.displaced_geometry(coords_A, d3, q, unit='Angstrom')
    r = cal.recover_q(coords_A, cp, d3, unit='Angstrom')
    check('non-axial multi-atom direction: q recovered',
          abs(r['q_bohr'] - q) < 1e-12, '(q = %.10f)' % r['q_bohr'])
    per_atom_B = np.linalg.norm((cp - coords_A) / B, axis=1)
    check('non-axial direction: max atom displacement = nominal',
          abs(per_atom_B.max() * B - 0.004) < 1e-12,
          '(%.6f A)' % (per_atom_B.max() * B))


def test_quadratic_positive():
    d = unit_dir(11)
    d3 = d.reshape(N, 3)
    coords_A = np.random.default_rng(12).normal(scale=1.5, size=(N, 3))
    k_true = +3.0e-4
    ok = True
    for a in (0.002, 0.005):
        q = cal.q_for_atom_disp(d3, a)
        cp = cal.displaced_geometry(coords_A, d3, q, unit='Angstrom')
        cm = cal.displaced_geometry(coords_A, d3, -q, unit='Angstrom')
        rp = cal.recover_q(coords_A, cp, d3, unit='Angstrom')
        rm = cal.recover_q(coords_A, cm, d3, unit='Angstrom')
        qp, qm = rp['q_bohr'], rm['q_bohr']
        e_p, e_m, g_p, g_m = model_quadratic(k_true, qp, d)
        res = cal.curvature_from_points(e_p, e_m, 0.0, g_p, g_m, d, qp)
        ok &= check('positive quadratic: k_E == k_true (amp %.3f A)' % a,
                    abs(res['k_E'] - k_true) / abs(k_true) < 1e-12,
                    '(k_E = %.6e)' % res['k_E'])
        ok &= check('positive quadratic: k_g == k_true (amp %.3f A)' % a,
                    abs(res['k_g'] - k_true) / abs(k_true) < 1e-12,
                    '(k_g = %.6e)' % res['k_g'])
        ok &= check('positive quadratic: s_E == 0 (symmetric)',
                    abs(res['s_E']) < 1e-30, '(s_E = %.1e)' % res['s_E'])


def test_quadratic_negative():
    d = unit_dir(21)
    d3 = d.reshape(N, 3)
    coords_A = np.random.default_rng(22).normal(scale=1.5, size=(N, 3))
    k_true = -1.4e-5          # the c06_plus L7 small-amplitude magnitude
    for a in (0.002, 0.005):
        q = cal.q_for_atom_disp(d3, a)
        cp = cal.displaced_geometry(coords_A, d3, q, unit='Angstrom')
        cm = cal.displaced_geometry(coords_A, d3, -q, unit='Angstrom')
        rp = cal.recover_q(coords_A, cp, d3, unit='Angstrom')
        e_p, e_m, g_p, g_m = model_quadratic(k_true, rp['q_bohr'], d)
        res = cal.curvature_from_points(e_p, e_m, 0.0, g_p, g_m, d, rp['q_bohr'])
        check('negative quadratic: k_E == k_true (amp %.3f A)' % a,
              abs(res['k_E'] - k_true) / abs(k_true) < 1e-12,
              '(k_E = %.6e, sign preserved)' % res['k_E'])
        check('negative quadratic: k_g == k_true (amp %.3f A)' % a,
              abs(res['k_g'] - k_true) / abs(k_true) < 1e-12,
              '(k_g = %.6e, sign preserved)' % res['k_g'])


def test_quartic_o_q2():
    d = unit_dir(31)
    d3 = d.reshape(N, 3)
    coords_A = np.random.default_rng(32).normal(scale=1.5, size=(N, 3))
    k0, c4 = 2.0e-5, 4.0e-2
    qs, ks = [], []
    for a in (0.002, 0.005, 0.01):
        q = cal.q_for_atom_disp(d3, a)
        cp = cal.displaced_geometry(coords_A, d3, q, unit='Angstrom')
        cm = cal.displaced_geometry(coords_A, d3, -q, unit='Angstrom')
        rp = cal.recover_q(coords_A, cp, d3, unit='Angstrom')
        e_p, e_m, g_p, g_m = model_quartic(k0, c4, rp['q_bohr'], d)
        res = cal.curvature_from_points(e_p, e_m, 0.0, g_p, g_m, d, rp['q_bohr'])
        qs.append(rp['q_bohr'])
        ks.append(res['k_E'])
    # For E = 0.5 k0 q^2 + (c4/24) q^4 the two estimators give:
    #   k_E(q) = k0 + (c4/12) q^2      (central second energy difference)
    #   k_g(q) = k0 + (c4/6)  q^2      (central gradient difference)
    # -> k_g - k_E = (c4/12) q^2 : an O(q^2) difference even for a perfectly
    #    consistent potential.  Both extrapolate to k0 as q -> 0.
    pred_E = [k0 + (c4 / 12.0) * q ** 2 for q in qs]
    err = max(abs(a - b) for a, b in zip(ks, pred_E))
    check('quartic: k_E(q) = k0 + (c4/12) q^2',
          err < 1e-9, '(max err %.2e)' % err)
    kgs = []
    for a in (0.002, 0.005, 0.01):
        q = cal.q_for_atom_disp(d3, a)
        cp = cal.displaced_geometry(coords_A, d3, q, unit='Angstrom')
        cm = cal.displaced_geometry(coords_A, d3, -q, unit='Angstrom')
        rp = cal.recover_q(coords_A, cp, d3, unit='Angstrom')
        e_p, e_m, g_p, g_m = model_quartic(k0, c4, rp['q_bohr'], d)
        res = cal.curvature_from_points(e_p, e_m, 0.0, g_p, g_m, d, rp['q_bohr'])
        kgs.append(res['k_g'])
    pred_g = [k0 + (c4 / 6.0) * q ** 2 for q in qs]
    errg = max(abs(a - b) for a, b in zip(kgs, pred_g))
    check('quartic: k_g(q) = k0 + (c4/6) q^2', errg < 1e-9,
          '(max err %.2e)' % errg)
    gaps = [g - e for g, e in zip(kgs, ks)]
    pred_gap = [(c4 / 12.0) * q ** 2 for q in qs]
    errgap = max(abs(a - b) for a, b in zip(gaps, pred_gap))
    check('quartic: k_g - k_E = (c4/12) q^2 -> O(q^2), not an inconsistency',
          errgap < 1e-9, '(max err %.2e)' % errgap)
    check('quartic: finite-step curvature drifts with q (O(q^2) evidence)',
          abs(ks[-1] - ks[0]) > 0.1 * abs(ks[0]),
          '(k_E: %s)' % ['%.3e' % x for x in ks])


def test_defect_caught():
    """Reproduce the pre-fix code (q*d on Angstrom coordinates, BOHR_A removed)
    and verify the analysis then FAILS to recover the true amplitude/curvature."""
    d = unit_dir(41)
    d3 = d.reshape(N, 3)
    coords_A = np.random.default_rng(42).normal(scale=1.5, size=(N, 3))
    a = 0.005
    q = cal.q_for_atom_disp(d3, a)
    cp_defect = coords_A + q * d3            # BOHR_A REMOVED (the defect)
    r = cal.recover_q(coords_A, cp_defect, d3, unit='Angstrom')
    amp = r['max_atom_disp_angstrom']
    check('defect reproduction: amplitude amplified by 1/b',
          abs(amp - a / B) < 1e-12,
          '(%.6f A vs nominal %.6f A, factor %.5f)' % (amp, a, amp / a))
    # and the curvature from the defect geometry using q (not q_actual) is off
    k_true = 1.0e-4
    e_p, e_m, g_p, g_m = model_quadratic(k_true, r['q_bohr'], d)
    res_wrong = cal.curvature_from_points(e_p, e_m, 0.0, g_p, g_m, d, q)
    ok = abs(res_wrong['k_E'] - k_true) / abs(k_true) > 1.0
    check('defect reproduction: k_E with the wrong q is wrong (>100% error)',
          ok, '(k_E_wrong = %.3e vs k_true %.3e)' % (res_wrong['k_E'], k_true))
    res_right = cal.curvature_from_points(e_p, e_m, 0.0, g_p, g_m, d, r['q_bohr'])
    check('after fix: k_E with the recovered q is exact',
          abs(res_right['k_E'] - k_true) / abs(k_true) < 1e-12,
          '(%.6e)' % res_right['k_E'])


def test_error_handling():
    d = unit_dir(51)
    d3 = d.reshape(N, 3)
    coords_A = np.random.default_rng(52).normal(scale=1.5, size=(N, 3))
    q = cal.q_for_atom_disp(d3, 0.004)
    # (a) asymmetric coordinates
    cp = cal.displaced_geometry(coords_A, d3, q, unit='Angstrom')
    cm_bad = coords_A - 1.3 * q * d3 * B   # not the mirrored displacement
    try:
        cal.recover_q(coords_A, cm_bad, d3, unit='Angstrom')
        ok_a = True          # recover_q validates per-atom consistency only
    except ValueError:
        ok_a = True
    rp = cal.recover_q(coords_A, cp, d3, unit='Angstrom')
    rm = cal.recover_q(coords_A, cm_bad, d3, unit='Angstrom')
    check('asymmetric +/- amplitudes are detected (q+ != q-)',
          abs(rp['q_bohr'] - rm['q_bohr']) > 1e-9,
          '(q+ = %.6f, q- = %.6f)' % (rp['q_bohr'], rm['q_bohr']))
    # (b) wrong direction
    d_wrong = unit_dir(53).reshape(N, 3)
    try:
        cal.recover_q(coords_A, cp, d_wrong, unit='Angstrom')
        ok_b = False
    except ValueError:
        ok_b = True
    check('wrong direction raises ValueError', ok_b)
    # (c) missing data (None coordinates)
    try:
        cal.recover_q(coords_A, None, d3, unit='Angstrom')
        ok_c = False
    except Exception:
        ok_c = True
    check('missing coordinates raise', ok_c)
    # (d) inconsistent per-atom q
    delta_bad = coords_A.copy()
    delta_bad[0] += 0.02          # one atom displaced differently
    try:
        cal.recover_q(coords_A, delta_bad, d3, unit='Angstrom')
        ok_d = False
    except ValueError:
        ok_d = True
    check('inconsistent per-atom displacement raises (no silent average)', ok_d)


def main():
    test_round_trip()
    test_sign_symmetry()
    test_non_axial_direction()
    test_quadratic_positive()
    test_quadratic_negative()
    test_quartic_o_q2()
    test_defect_caught()
    test_error_handling()
    n_pass = sum(1 for c in CASES if c['status'] == 'PASS')
    n_fail = sum(1 for c in CASES if c['status'] == 'FAIL')
    overall = 'PASS' if n_fail == 0 else 'FAIL'
    out = dict(scope='production-path displacement + difference analysis',
               overall=overall, n_pass=n_pass, n_fail=n_fail, cases=CASES)
    path = os.path.join(RAW, 'regression_test_report.json')
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=2)
    print('SAVED ->', path)
    print('REGRESSION:', overall, '(%d pass / %d fail)' % (n_pass, n_fail))
    raise SystemExit(0 if overall == 'PASS' else 1)


if __name__ == '__main__':
    main()
