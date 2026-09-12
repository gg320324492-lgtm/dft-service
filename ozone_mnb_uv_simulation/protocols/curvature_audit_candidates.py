#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Curvature-audit step C (JOB-2026-0905-009): three-layer curvature
comparison for the disputed lowest modes of the two candidates, at the
re-optimised endpoints and unified settings (grid level 6, SCF 1e-12/1e-9).

For each candidate:
  * analytic Hessian matrix  (persisted .npy) and FD Hessian matrix at
    step 0.02 Bohr (persisted .npy), both on the SAME masses/geometry;
  * mass-weighted TR projection -> 12 internal modes; external overlap,
    pre/post-projection eigenvalues recorded;
  * disputed modes: the projected lowest mode of the analytic Hessian AND of
    the FD Hessian (if the two directions differ -- overlap check);
  * for each disputed direction d (unit, 3N):
      k_H = d^T H d, and for q in {0.005, 0.01, 0.02} A (max per-atom
      displacement-VECTOR length): k_E, k_g, slope d.g0, first-difference
      slope.  Results below the energy error floor are flagged 不可分辨.
Resume-safe per structure.
"""
import os
import sys
import json
import time
import numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import curvature_audit_lib as cal
import d2_full
import conformer_utils as cu

ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, 'run_artifacts', '01_pure_water_o3_h2o')
FV = os.path.join(ART, 'final_minima_validation')
CA = os.path.join(ART, 'curvature_audit')
os.makedirs(CA, exist_ok=True)

SYMS = ['O', 'O', 'O', 'O', 'H', 'H']
TARGETS = ['c14_plus', 'c06_plus']
FD_STEP = 0.02
AMPS_A = [0.005, 0.01, 0.02]
DE_E_FLOOR = 1e-9            # Eh, energy reproducibility/error floor (documented)
N_THREADS = 8


def make_mf(mol):
    mf = d2_full.make_mf_d2(mol)
    mf.grids.level = 6
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-9
    return mf


def eg_at(mol, coords_bohr):
    m = mol.copy()
    m.set_geom_(coords_bohr, unit='Bohr')
    m.build()
    mf = make_mf(m)
    mf.kernel()
    assert mf.converged
    return float(mf.e_tot), np.asarray(mf.nuc_grad_method().kernel()).reshape(-1)


def hess_block(mol):
    mf = make_mf(mol)
    mf.kernel()
    h = np.asarray(mf.Hessian().kernel())
    e = float(mf.e_tot)
    g = np.asarray(mf.nuc_grad_method().kernel()).reshape(-1)
    return e, g, h


def fd_hess_block(mol, step):
    c0 = mol.atom_coords(unit='Bohr')
    n = mol.natm
    g = np.zeros((n, n, 3, 3))
    for i in range(n):
        for k in range(3):
            for s in (+1, -1):
                c = c0.copy()
                c[i, k] += s * step
                _, gg = eg_at(mol, c)
                g[i, :, k] += s * gg.reshape(n, 3) / (2 * step)
    return 0.5 * (g + g.transpose(1, 0, 3, 2))


def projected_modes(h_block, masses, coords_bohr):
    """REVISITED (internal-subspace revision batch): build the mass-weighted
    TR basis V (3N x 6), its orthonormal COMPLEMENT U (3N x 12) via QR, and
    diagonalize H_internal = U^T H_mw U (12 x 12).  Eigenvectors are mapped
    back to the 18-dim mass-weighted space through U.

    The previous implementation diagonalized the full P H_mw P and took the
    lowest eigenvalues, which selects the EXTERNAL zero modes whenever the
    internal spectrum is positive (the exact failure the revision brief
    describes).  Deleting small-|eigenvalue| entries is NOT a substitute:
    real soft modes may also be near zero.  Internal-mode selection must come
    from the subspace reduction itself.
    """
    H_mw = cal.mw_hessian_matrix(h_block, masses)
    V = cal.tr_subspace(masses, coords_bohr)
    # orthonormal complement of V in the 3N space
    Qfull, _ = np.linalg.qr(np.hstack([V, np.eye(3 * len(masses))]))
    # columns of Qfull orthogonal to V span the complement; pick via projection
    P = cal.projector(V)
    wp, Up_full = np.linalg.eigh(P)
    U = Up_full[:, wp > 0.5]                      # 3N x 12, U^T U = I, V^T U = 0
    H_int = U.T @ H_mw @ U
    H_int = 0.5 * (H_int + H_int.T)
    w_int, v_int = np.linalg.eigh(H_int)
    v_mw = U @ v_int                              # back to 18-dim mw space
    return dict(V=V, U=U, H_mw=H_mw, H_int=H_int,
                w_int=w_int, v_int=v_int, v_mw=v_mw)


def run_target(tid):
    t0 = time.time()
    out_path = os.path.join(CA, 'candidate_curvature_%s.json' % tid)
    if os.path.exists(out_path):
        print('[skip]', tid, flush=True)
        with open(out_path) as fh:
            return json.load(fh)

    coords = np.asarray(json.load(open(os.path.join(
        FV, 'stepC_reopt_%s.json' % tid)))['new_geometry'], float)
    masses = np.array([15.999, 15.999, 15.999, 15.999, 1.008, 1.008])
    mol = cu.mol_from_coords(coords, SYMS, verbose=0)

    # ---- analytic Hessian (persist matrix; reuse persisted one if present)
    ana_path = os.path.join(CA, 'hessian_analytic_%s.npy' % tid)
    fd_path = os.path.join(CA, 'hessian_fd0.02_%s.npy' % tid)
    if os.path.exists(ana_path):
        h_an = np.load(ana_path)
        mf0 = make_mf(mol)
        mf0.kernel()
        e0 = float(mf0.e_tot)
        g0 = np.asarray(mf0.nuc_grad_method().kernel()).reshape(-1)
        print('[%s] analytic Hessian reused from cache' % tid, flush=True)
    else:
        e0, g0, h_an = hess_block(mol)
        np.save(ana_path, h_an)
    # ---- FD Hessian at 0.02 Bohr (persist matrix; reuse if present)
    if os.path.exists(fd_path):
        h_fd = np.load(fd_path)
        print('[%s] FD Hessian reused from cache' % tid, flush=True)
    else:
        t1 = time.time()
        h_fd = fd_hess_block(mol, FD_STEP)
        np.save(fd_path, h_fd)
        print('[%s] FD Hessian done (%.0f s)' % (tid, time.time() - t1),
              flush=True)

    # ---- projected spectra
    pan = projected_modes(h_an, masses, mol.atom_coords(unit='Bohr'))
    pfd = projected_modes(h_fd, masses, mol.atom_coords(unit='Bohr'))
    # NOTE: eigenvectors are COLUMNS of v (eigh ascending) -- the lowest mode
    # is v[:, 0].  (A previous run sliced v[:12][0], i.e. ROW 0 = a mixture of
    # all modes; those probe results were invalid and are superseded.)
    spec = dict(
        analytic=dict(pre_projection_eigenvalues=[float(x) for x in
                     np.linalg.eigvalsh(pan['H_mw'])],
                     internal_eigenvalues=[float(x) for x in pan['w'][:12]],
                     lowest_internal_mode=pan['v'][:, 0].tolist(),
                     lowest_external_overlap=cal.external_overlap(
                         pan['v'][:, 0], pan['V'])),
        fd=dict(pre_projection_eigenvalues=[float(x) for x in
             np.linalg.eigvalsh(pfd['H_mw'])],
             internal_eigenvalues=[float(x) for x in pfd['w'][:12]],
             lowest_internal_mode=pfd['v'][:, 0].tolist(),
             lowest_external_overlap=cal.external_overlap(
                 pfd['v'][:, 0], pfd['V'])))

    # ---- disputed directions -- ALL THREE layers evaluated in the CARTESIAN
    # metric along the SAME direction (the earlier run compared d^T H_mw d
    # against Cartesian-space probes: a metric mismatch, superseded).
    H_cart9 = np.asarray(h_an).transpose(0, 2, 1, 3).reshape(3 * len(masses), 3 * len(masses))
    H_cart9_fd = np.asarray(h_fd).transpose(0, 2, 1, 3).reshape(3 * len(masses), 3 * len(masses))
    mw = cal.mw_vector(masses)
    disp = {}
    d_mw_an = pan['v'][:, 0]
    d_cart_an = mw * d_mw_an
    d_cart_an /= np.linalg.norm(d_cart_an)
    disp['analytic_lowest'] = dict(
        d_cart=d_cart_an,
        lam_mw=float(pan['w'][0]),
        k_H=float(d_cart_an @ H_cart9 @ d_cart_an),
        norm_factor=float(np.linalg.norm(mw * d_mw_an) ** 2),  # lam_mw/norm^2 = k_H
        source='analytic')
    ov = float(abs(np.dot(pfd['v'][:, 0], d_mw_an)))
    d_mw_fd = pfd['v'][:, 0]
    d_cart_fd = mw * d_mw_fd
    d_cart_fd /= np.linalg.norm(d_cart_fd)
    entry_fd = dict(
        d_cart=d_cart_fd,
        lam_mw=float(pfd['w'][0]),
        k_H=float(d_cart_fd @ H_cart9_fd @ d_cart_fd),
        norm_factor=float(np.linalg.norm(mw * d_mw_fd) ** 2),
        source='fd(0.02)',
        overlap_with_analytic=ov,
        overlap_with_analytic_cart=float(abs(np.dot(d_cart_fd, d_cart_an))))
    # keep the FD direction only if it is a genuinely different Cartesian
    # direction from the analytic one
    if entry_fd['overlap_with_analytic_cart'] < 0.99:
        disp['fd_lowest'] = entry_fd

    # ---- three-layer probe per disputed direction (Cartesian metric)
    probes = {}
    for name, info in disp.items():
        d = info['d_cart']
        d3 = d.reshape(-1, 3)
        rows = []
        for a in AMPS_A:
            q = cal.q_for_atom_disp(d3, a)
            dq = (d * q).reshape(-1, 3)
            cp = mol.atom_coords(unit='Bohr') + dq
            cm = mol.atom_coords(unit='Bohr') - dq
            ep, gp = eg_at(mol, cp)
            em, gm = eg_at(mol, cm)
            pr = cal.curvature_three_way(ep, em, e0, gp, gm, g0, d, q)
            # energy-noise floor for this q
            pr['k_E_noise_floor'] = 3.0 * DE_E_FLOOR / q ** 2
            pr['resolvable_E'] = bool(abs(pr['k_E']) > pr['k_E_noise_floor'])
            pr['amp_a'] = a
            rows.append(pr)
            print('[%s/%s] amp=%.3fA  k_E=%.3e(floor %.1e, resolvable=%s)  '
                  'k_g=%.3e  k_H=%.3e  slope_g=%.2e'
                  % (tid, name, a, pr['k_E'], pr['k_E_noise_floor'],
                     pr['resolvable_E'], pr['k_g'],
                     info['k_H'], pr['slope_g']), flush=True)
        probes[name] = dict(k_H=info['k_H'], lam_mw=info['lam_mw'],
                            norm_factor=info['norm_factor'],
                            source=info.get('source'),
                            overlap_with_analytic=info.get('overlap_with_analytic'),
                            overlap_with_analytic_cart=info.get(
                                'overlap_with_analytic_cart'),
                            rows=rows)

    res = dict(id=tid, geometry=coords.tolist(),
               masses=masses.tolist(),
               e0=e0, max_grad=g0.max().__float__(),
               spectra=spec, disputed_probes=probes,
               seconds=round(time.time() - t0, 1))
    with open(out_path, 'w') as fh:
        json.dump(res, fh, indent=2)
    print('[%s] SAVED %s (%.0f s)' % (tid, out_path, res['seconds']), flush=True)
    return res


def _init():
    import pyscf.lib as lib
    lib.num_threads(N_THREADS)


def main():
    done, todo = [], []
    for t in TARGETS:
        if os.path.exists(os.path.join(CA, 'candidate_curvature_%s.json' % t)):
            with open(os.path.join(CA, 'candidate_curvature_%s.json' % t)) as fh:
                done.append(json.load(fh))
        else:
            todo.append(t)
    results = done
    if todo:
        with Pool(len(todo), initializer=_init) as pool:
            results.extend(pool.map(run_target, todo))
    results.sort(key=lambda r: TARGETS.index(r['id']))
    with open(os.path.join(CA, 'candidate_curvature_summary.json'), 'w') as fh:
        json.dump(dict(job='JOB-2026-0905-009',
                       step='curvature_audit_C_candidates',
                       settings=dict(grid_level=6, scf_tol=1e-12,
                                     scf_tol_grad=1e-9, fd_step=FD_STEP,
                                     amps_a=AMPS_A,
                                     energy_error_floor_eh=DE_E_FLOOR),
                       targets=results), fh, indent=2)
    print('SAVED candidate_curvature_summary.json', flush=True)


if __name__ == '__main__':
    main()
