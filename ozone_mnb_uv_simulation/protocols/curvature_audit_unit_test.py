#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Curvature-audit unit test (no DFT): a quadratic potential of KNOWN
curvature validates sign, normalization, the mass-weighted->Cartesian
conversion, the projection and the three-way estimator.

Construction: the 6-dim mass-weighted TR subspace is built FIRST (from the
real geometry/masses), an orthonormal basis of its complement (12-dim internal
space) is derived, and a synthetic Hessian is assembled with KNOWN eigenvalues
in that basis -- one negative soft mode (-1.234e-4) plus stiff modes, and ~0
on the TR space.  The exact quadratic model E(q)=E0+0.5 k q^2 / g(q)=k q d then
round-trips through the audit estimators.
"""
import os
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import curvature_audit_lib as cal

ROOT = os.path.dirname(HERE)
FV = os.path.join(ROOT, 'run_artifacts', '01_pure_water_o3_h2o', 'curvature_audit')
os.makedirs(FV, exist_ok=True)

MASSES = np.array([15.999, 15.999, 15.999, 15.999, 1.008, 1.008])
N = len(MASSES)
LAM_SOFT = -1.234e-4


def main():
    rng = np.random.default_rng(7)
    coords = rng.normal(scale=1.5, size=(N, 3))
    V = cal.tr_subspace(MASSES, coords)
    P = cal.projector(V)

    # orthonormal basis of the internal (projected) space
    wp, Up = np.linalg.eigh(P)
    internal = Up[:, wp > 0.5]                      # 12 columns
    tr = Up[:, wp <= 0.5]                           # 6 columns
    assert internal.shape[1] == 12, internal.shape

    # synthetic Hessian: known eigenvalues on the internal space, ~0 on TR
    eigs = np.array([LAM_SOFT, 1e-3, 5e-3, 1e-2, 3e-2, 5e-2, 1e-1, 2e-1,
                     4e-1, 5e-1, 6e-1, 8e-1])
    H_mw = (internal * eigs) @ internal.T
    H_mw = 0.5 * (H_mw + H_mw.T)

    v_soft = internal[:, 0]                          # exact soft eigenvector
    overlap_soft = cal.external_overlap(v_soft, V)   # ~0 by construction
    d_mw = v_soft.copy()                             # projection = identity here
    k_true = float(d_mw @ H_mw @ d_mw)
    assert abs(k_true - LAM_SOFT) < 1e-12, k_true

    d_cart = (cal.mw_vector(MASSES) * d_mw).reshape(N, 3)
    qs = [cal.q_for_atom_disp(d_cart, a) for a in (0.005, 0.01, 0.02)]
    rows, ok_all = [], True
    for a, q in zip((0.005, 0.01, 0.02), qs):
        e0, e_p, e_m = 0.0, 0.5 * k_true * q ** 2, 0.5 * k_true * q ** 2
        g0 = np.zeros(3 * N)
        g_p, g_m = k_true * q * d_mw, -k_true * q * d_mw
        pr = cal.curvature_three_way(e_p, e_m, e0, g_p, g_m, g0, d_mw, q)
        pr.update(q_bohr=q, max_atom_disp_a=a, k_H_true=k_true,
                  err_k_E=abs(pr['k_E'] - k_true), err_k_g=abs(pr['k_g'] - k_true))
        ok_all &= (pr['err_k_E'] < 1e-12 and pr['err_k_g'] < 1e-12)
        rows.append(pr)
        print('amp=%.3fA q=%.4f  k_E=%.6e k_g=%.6e k_H=%.6e  exact round-trip'
              % (a, q, pr['k_E'], pr['k_g'], k_true))

    # withdrawn old denominator: (E+ + E- - 2E0)/(2q^2) = k/2  (bug reproduced)
    q = qs[0]
    e_p, e_m = 0.5 * k_true * q ** 2, 0.5 * k_true * q ** 2
    k_old = (e_p + e_m - 2 * e0) / (2 * q ** 2)
    bug_confirmed = bool(abs(k_old - k_true / 2) < 1e-12)
    print('old denominator: k_old=%.6e vs k_true/2=%.6e -> bug reproduced: %s'
          % (k_old, k_true / 2, bug_confirmed))

    # positive sign round-trip along a stiff internal direction
    k_pos = eigs[5]
    d2 = internal[:, 5]
    pr = cal.curvature_three_way(0.5 * k_pos * q ** 2, 0.5 * k_pos * q ** 2, 0.0,
                                 k_pos * q * d2, -k_pos * q * d2,
                                 np.zeros(3 * N), d2, q)
    sign_ok = pr['k_E'] > 0 and pr['k_g'] > 0 and abs(pr['k_E'] - k_pos) < 1e-12
    print('positive round-trip:', 'PASS' if sign_ok else 'FAIL')

    # max-atom-displacement convention: per-atom VECTOR length, not component
    d_t = np.zeros((N, 3)); d_t[0] = (0.3, 0.0, 0.0)
    a_vec = cal.max_atom_disp_a(d_t, 1.0)
    conv_ok = abs(a_vec - 0.3 * cal.BOHR_A) < 1e-12
    print('vector-length convention: %.4f A (component convention 0.3000 A) %s'
          % (a_vec, 'PASS' if conv_ok else 'FAIL'))

    w_proj = np.linalg.eigvalsh(cal.projector(V) @ H_mw @ cal.projector(V))
    n_external_removed = int(np.sum(np.abs(w_proj) < 1e-8))

    summary = dict(test='curvature_audit_unit_test',
                   external_overlap_soft=overlap_soft,
                   n_internal_modes=int(internal.shape[1]),
                   n_external_removed=n_external_removed,
                   lowest_projected_eig=float(w_proj[0]),
                   quadratic_roundtrip_pass=bool(ok_all),
                   old_denominator_bug_reproduced=bug_confirmed,
                   positive_sign_roundtrip=bool(sign_ok),
                   vector_length_convention=bool(conv_ok),
                   lam_soft=LAM_SOFT, rows=rows)
    path = os.path.join(FV, 'unit_test.json')
    with open(path, 'w') as fh:
        json.dump(summary, fh, indent=2)
    print('SAVED ->', path)
    all_pass = (ok_all and bug_confirmed and sign_ok and conv_ok
                and n_external_removed == 6 and internal.shape[1] == 12)
    print('UNIT_TEST', 'PASS' if all_pass else 'FAIL')
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
