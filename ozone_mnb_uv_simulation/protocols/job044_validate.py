#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-044 no-SCF validation of the corrected post-processing.

(a) recover the JOB-026 saved NH3 / O3 reference frequencies from their
    saved Hessian matrices;
(b) synthetic-matrix tests: unit conversion, 6 external directions excluded,
    internal negative eigenvalues retained, invariance of the internal
    spectrum under a rigid rotation+translation of geometry AND Hessian.

No quantum backend is called.
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, '/mnt/e/dft-service/ozone_mnb_uv_simulation/protocols')
import curvature_audit_lib as cal

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
BD026 = ROOT + '/run_artifacts/02_nh3o3_reference/freq_check'
OUT = ROOT + '/run_artifacts/02_nh3o3_reference/c1_freq_fix044'
os.makedirs(OUT, exist_ok=True)

HARTREE2J = 4.3597447222071e-18
AMU_KG = 1.66053906660e-27
BOHR_M = 0.52917721092e-10
C_CM = 2.99792458e10
NU_AMU = np.sqrt(HARTREE2J / (AMU_KG * BOHR_M**2)) / (2 * np.pi * C_CM)

ok = True
V = {}


def rep(name, cond, detail=''):
    global ok
    ok &= bool(cond)
    print('[%s] %s %s' % ('PASS' if cond else 'FAIL', name, detail))


def mw_mat(H21, masses):
    mw = np.repeat(np.asarray(masses, float) ** -0.5, 3)
    Hm = np.asarray(H21, float) * np.outer(mw, mw)
    return 0.5 * (Hm + Hm.T)


def tr_basis_raw(masses, coords_bohr):
    masses = np.asarray(masses, float)
    coords = np.asarray(coords_bohr, float)
    n = len(masses)
    sqrtm = np.sqrt(masses)
    com = (masses[:, None] * coords).sum(0) / masses.sum()
    r = coords - com
    B = np.zeros((3 * n, 6))
    for i in range(n):
        B[3*i:3*i+3, 0:3] = np.eye(3) * sqrtm[i]
        for k in range(3):
            e = np.zeros(3); e[k] = 1.0
            B[3*i:3*i+3, 3+k] = np.cross(r[i], e) * sqrtm[i]
    return B


def subspaces(masses, coords_bohr, tol=1e-8):
    Braw = tr_basis_raw(masses, coords_bohr)
    sv = np.linalg.svd(Braw, compute_uv=False)
    rank = int((sv > tol * max(1.0, sv[0])).sum())
    V = cal.tr_subspace(masses, coords_bohr)
    n = len(masses)
    Q, _ = np.linalg.qr(np.hstack([V, np.eye(3 * n)]))
    return V, Q[:, 6:], sv, rank


def internal_freqs(H21, masses, coords_bohr):
    V, U, sv, rank = subspaces(masses, coords_bohr)
    Hm = mw_mat(H21, masses)
    Hint = U.T @ Hm @ U
    Hint = 0.5 * (Hint + Hint.T)
    lam = np.linalg.eigvalsh(Hint)
    nu = np.array([np.sign(x) * np.sqrt(abs(x)) * NU_AMU for x in lam])
    return nu, U, V, lam, rank


# ---------------------------------------------------------------- (a) 026
print('=== (a) JOB-026 reference recovery (no SCF) ===')
r26 = json.load(open(BD026 + '/freq_check_results.json'))
ref = {'NH3': (np.array(r26['results']['NH3']['frequencies']['my_freq_cm']),
               ['N', 'H', 'H', 'H']),
       'O3': (np.array(r26['results']['O3']['frequencies']['my_freq_cm']),
              ['O', 'O', 'O'])}
for sp, (expect, syms) in ref.items():
    d = r26['results'][sp]
    C_A = np.asarray(d['endpoint_check']['coords_passed_to_object'], float)
    masses = np.asarray(d['masses']['values'], float)
    H = np.load(BD026 + '/%s_hess_combined.npy' % sp)
    if H.ndim == 4:
        n = H.shape[0]
        H21 = H.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)
    else:
        H21 = H
        n = H21.shape[0] // 3
    coords_bohr = C_A / 0.52917721092
    nu, U, Vtr, lam, rank = internal_freqs(H21, masses, coords_bohr)
    diff = np.abs(np.sort(nu) - np.sort(expect)).max()
    print('  %s: n_internal=%d (expected %d), rank_TR=%d'
          % (sp, len(nu), 3 * n - 6, rank))
    print('     recovered: %s' % np.round(np.sort(nu), 2))
    print('     026 ref  : %s' % np.round(np.sort(expect), 2))
    rep('  %s: recovers 026 reference frequencies' % sp,
        diff < 1e-3 and len(nu) == 3 * n - 6,
        '(max diff %.2e cm^-1)' % diff)
    V[sp] = dict(recovered=[float(x) for x in np.sort(nu)],
                 reference=[float(x) for x in np.sort(expect)],
                 max_diff_cm1=float(diff), n_internal=int(len(nu)),
                 tr_rank=rank)

# ---------------------------------------------------------------- (b) synth
print()
print('=== (b) synthetic-matrix tests ===')
rng = np.random.default_rng(20260908)
SYMS7 = ['N', 'H', 'H', 'H', 'O', 'O', 'O']
masses7 = np.array([14.0, 1.0, 1.0, 1.0, 16.0, 16.0, 16.0])
coords = rng.normal(scale=2.0, size=(7, 3))     # Bohr
Vtr, U, sv, rank = subspaces(masses7, coords)
rep('external rank == 6 (SVD on 7-atom nonlinear geometry)', rank == 6,
    '(rank=%d)' % rank)

# prescribed internal spectrum (15 values) including ONE negative
eigs = np.array([-1.9e-4, 1.4e-4, 1.9e-4, 5.6e-4, 1.06e-3, 1.31e-3,
                 2.34e-2, 3.74e-2, 6.75e-2, 8.0e-2, 2.35e-1, 2.38e-1,
                 1.04, 1.12, 1.13])
Hm_syn = (U * eigs) @ U.T
Hm_syn = 0.5 * (Hm_syn + Hm_syn.T)
# -> Cartesian Hessian (un-mass-weight)
sm = np.repeat(np.sqrt(masses7), 3)
H21_syn = Hm_syn * np.outer(sm, sm)
nu_syn, U2, V2, lam_syn, rank2 = internal_freqs(H21_syn, masses7, coords)
expect_syn = np.sort(np.array([np.sign(x) * np.sqrt(abs(x)) * NU_AMU
                               for x in eigs]))
got_syn = np.sort(nu_syn)
d_syn = np.abs(got_syn - expect_syn).max()
print('  prescribed : %s' % np.round(expect_syn, 3))
print('  recovered  : %s' % np.round(got_syn, 3))
rep('synthetic: all 15 prescribed eigenvalues recovered (unit conversion)',
    len(nu_syn) == 15 and d_syn < 1e-6, '(max diff %.2e cm^-1)' % d_syn)
rep('synthetic: internal negative eigenvalue RETAINED (not discarded)',
    int((nu_syn < 0).sum()) == 1 and abs(min(nu_syn) - expect_syn[0]) < 1e-6,
    '(lowest = %.3f cm^-1)' % min(nu_syn))
# external directions must carry ~zero curvature
ext_curv = np.abs(np.linalg.eigvalsh(Vtr.T @ mw_mat(H21_syn, masses7) @ Vtr))
rep('synthetic: 6 external directions carry ~zero curvature',
    ext_curv.max() < 1e-10, '(max |lambda_ext| = %.2e)' % ext_curv.max())

# rigid rotation + translation of BOTH geometry and Hessian
def rand_rot(rng):
    A = rng.normal(size=(3, 3))
    Qa, Ra = np.linalg.qr(A)
    if np.linalg.det(Qa) < 0:
        Qa[:, 0] *= -1
    return Qa

R = rand_rot(rng)
t = rng.normal(scale=3.0, size=3)
coords2 = coords @ R.T + t
Rfull = np.kron(np.eye(7), R)
H21_2 = Rfull @ H21_syn @ Rfull.T
nu_2, _, _, _, rank_2 = internal_freqs(H21_2, masses7, coords2)
d_rt = np.abs(np.sort(nu_2) - got_syn).max()
rep('rigid rotation+translation of geometry AND Hessian leaves the '
    'internal spectrum unchanged', d_rt < 1e-6,
    '(max diff %.2e cm^-1)' % d_rt)
V['synthetic'] = dict(prescribed=[float(x) for x in expect_syn],
                      recovered=[float(x) for x in got_syn],
                      max_diff_cm1=float(d_syn),
                      n_negative_recovered=int((nu_syn < 0).sum()),
                      external_curvature_max=float(ext_curv.max()),
                      rigid_motion_max_diff_cm1=float(d_rt),
                      n_internal=int(len(nu_syn)))

json.dump(V, open(OUT + '/validation_results.json', 'w'), indent=2)
print()
print('NO-SCF VALIDATION:', 'ALL PASS' if ok else 'FAILED')
if not ok:
    sys.exit(1)
