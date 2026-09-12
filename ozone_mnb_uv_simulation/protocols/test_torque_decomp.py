#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JOB-2026-0906-009 Phase A: no-SCF regression for the repaired torque
decomposition (torque_decomp.py).

T1  Angstrom/Bohr input consistency (converted results identical).
T2  Center-shift relation: T_g(c+d) = T_g(c) - d x F  (F = net force).
T3  Rotational covariance: decompose(R r, R g) has T_g -> R T_g and
    identical internal norms.
T4  Orthogonality and squared-norm additivity of the trans/rot/internal
    projections.
T5  Unit-conversion guard: computing the cross product with ANGSTROM
    coordinates (the 008 defect) gives a DIFFERENT T_g -- the test fails
    if the unit conversion is removed.
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torque_decomp as td

PASS = []


def check(name, ok, detail=''):
    PASS.append(bool(ok))
    print('[%s] %s %s' % ('PASS' if ok else 'FAIL', name, detail))


def main():
    rng = np.random.default_rng(42)
    masses = np.array([15.999, 15.999, 15.999, 15.999, 1.008, 1.008])
    c_A = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.24], [0.0, 0.0, -1.24],
                    [3.1, 0.2, 0.0], [3.9, 0.7, 0.1], [2.5, -0.4, 0.3]])
    g = rng.normal(scale=1e-4, size=(6, 3))

    dA = td.decompose_force(g, c_A, masses, unit='Angstrom')
    dB = td.decompose_force(g, c_A / td.BOHR_A, masses, unit='Bohr')
    # T1
    check('T1 Angstrom/Bohr inputs give identical T_g',
          np.allclose(dA['T_g'], dB['T_g'], atol=1e-14),
          '(diff %.2e)' % np.abs(np.array(dA['T_g'])
                                 - np.array(dB['T_g'])).max())
    check('T1 Angstrom/Bohr inputs give identical projections',
          all(np.allclose(dA[k], dB[k], atol=1e-14)
              for k in ('g_trans_vec', 'g_rot_vec', 'g_int_vec')))

    # T2 center-shift: T_g(c+d) = T_g(c) - d x F
    F = np.asarray(dA['net_force'], float)
    d = np.array([0.3, -0.2, 0.15])            # Bohr shift of the center
    rc = np.asarray(dA['center_bohr'], float)
    dA2 = td.decompose_force(g, c_A, masses, unit='Angstrom',
                             center=rc + d)
    lhs = np.asarray(dA2['T_g'], float)
    rhs = np.asarray(dA['T_g'], float) - np.cross(d, F)
    check('T2 T_g(c+d) = T_g(c) - d x F',
          np.allclose(lhs, rhs, atol=1e-14),
          '(max diff %.2e)' % np.abs(lhs - rhs).max())

    # T3 rotational covariance
    axis = np.array([0.3, -0.5, 0.8])
    th = 0.7
    R = td.rodrigues(axis, th)
    rc = np.asarray(dA['center_bohr'], float)
    rB = c_A / td.BOHR_A - rc
    rB_rot = (R @ rB.T).T + rc
    g_rot = td.rotate_vectors(g, R)
    dR = td.decompose_force(g_rot, rB_rot * td.BOHR_A, masses,
                            unit='Angstrom', center=rc)
    T_cov = R.dot(np.asarray(dA['T_g'], float))
    check('T3 T_g is rotationally covariant: T_g(Rr,Rg) = R T_g',
          np.allclose(dR['T_g'], T_cov, atol=1e-12),
          '(max diff %.2e)' % np.abs(np.asarray(dR['T_g']) - T_cov).max())
    check('T3 internal norms invariant under rigid rotation',
          abs(dR['g_int_norm'] - dA['g_int_norm']) < 1e-14)

    # T4 orthogonality and squared-norm additivity
    n_t = dA['g_trans_norm']; n_r = dA['g_rot_norm']
    n_i = dA['g_int_norm']; n_g = dA['g_norm']
    check('T4 squared norms add: ||g||^2 = ||t||^2+||r||^2+||i||^2',
          abs((n_t ** 2 + n_r ** 2 + n_i ** 2) - n_g ** 2)
          < 1e-18 + 1e-9 * n_g ** 2,
          '(LHS %.6e RHS %.6e)' % (n_t ** 2 + n_r ** 2 + n_i ** 2, n_g ** 2))
    t = np.asarray(dA['g_trans_vec'], float).reshape(-1)
    rr_ = np.asarray(dA['g_rot_vec'], float).reshape(-1)
    ii = np.asarray(dA['g_int_vec'], float).reshape(-1)
    check('T4 projections mutually orthogonal',
          abs(t.dot(rr_)) + abs(t.dot(ii)) + abs(rr_.dot(ii)) < 1e-20)

    # T5 unit guard: the 008 defect (Angstrom cross product) must differ
    g6 = g
    com_A = np.asarray(dA['center_bohr'], float) * td.BOHR_A  # SAME center
    rA = c_A - com_A                       # WRONG: Angstrom coords
    T_wrong = np.cross(rA, g6).sum(axis=0)
    T_right = np.asarray(dA['T_g'], float)
    check('T5 Angstrom-coordinate cross product differs from the Bohr '
          'T_g (defect detectable)',
          not np.allclose(T_wrong, T_right, atol=1e-12),
          '(ratio of norms %.6f, expect %.6f)'
          % (np.linalg.norm(T_wrong) / np.linalg.norm(T_right),
             td.BOHR_A))
    check('T5 removing the unit conversion scales T_g by exactly BOHR',
          abs(np.linalg.norm(T_wrong) / np.linalg.norm(T_right)
              - td.BOHR_A) < 1e-9)

    print('TORQUE REGRESSION: %s (%d/%d pass)'
          % ('PASS' if all(PASS) else 'FAIL', sum(PASS), len(PASS)))
    import json
    out = os.path.join(HERE, '..', 'run_artifacts', '01_pure_water_o3_h2o',
                       'smd_rotation_audit', 'torque_regression.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(dict(all_pass=bool(all(PASS)), n_pass=sum(PASS),
                   n_total=len(PASS)), open(out, 'w'), indent=2)
    print('saved ->', out)
    sys.exit(0 if all(PASS) else 1)


if __name__ == '__main__':
    main()
