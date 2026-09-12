#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JOB-2026-0906-013 (Task B): diagnostic internal-curvature Hessian at the
c06 order-47 NEW endpoint (011 re-optimisation endpoint).

AUTHORISED as a non-stationary-point diagnostic; the formal stationary
acceptance criterion (unprojected max|g| <= 1e-5 Eh/Bohr) is UNCHANGED and
this endpoint REMAINS not passed.

Budget: 1 center recheck + 72 displaced points (18 Cartesian directions x
h = 0.002/0.004 Bohr x +/-).  No auto-retry, no expansion.

Reproducibility tolerance FIXED BEFORE ANY RUN (section III):
    |dE|      <= 1e-9  Eh
    max|dg|   <= 1e-8  Eh/Bohr
If the center recheck violates these, the batch STOPS before any Hessian
point is computed.

Per displaced point: verify actual displacement & actual settings, save
immediately.  Disqualified points (SCF failure / config rollback / missing /
non-finite) never enter the qualified matrix; the batch aborts (no retry).
The UNSYMMETRIZED Hessian is saved first, its antisymmetric residual
reported, and only then symmetrized (raw matrices are never overwritten).
"""
import os
import sys
import json
import time
import hashlib
import traceback
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, 'run_artifacts', '01_pure_water_o3_h2o')
OUT = os.path.join(ART, 'smd_diag_curvature')
sys.path.insert(0, HERE)

from pyscf import gto
import d2_full
from grad_factory import make_mf_d2_gr, method_fingerprint
from smd_surface_refine_config import GRID_LEVEL, TARGET_ORDER
import torque_decomp as td

SYMS = ['O', 'O', 'O', 'O', 'H', 'H']
MASS = [15.9949146] * 4 + [1.007825] * 2
ORDER = TARGET_ORDER                    # 47
BOHR_A = td.BOHR_A
H_LIST = [0.002, 0.004]                 # Bohr
TOL_E = 1e-9                            # FIXED BEFORE ANY RUN
TOL_G = 1e-8                            # FIXED BEFORE ANY RUN
DISP_TOL = 1e-8                         # Bohr, actual vs intended
GMAX_FORMAL = 1e-5                      # unchanged acceptance (reference only)


def sha(b):
    return hashlib.sha256(b).hexdigest()[:16]


def mol_from_coords(coords_A):
    atom = "; ".join("%s %.10f %.10f %.10f" % (s, x, y, z)
                     for s, (x, y, z) in zip(SYMS, coords_A))
    return gto.M(atom=atom, basis='def2-TZVP', charge=0, spin=0,
                 verbose=0, max_memory=4000)


def build_mf(coords_A, order):
    mol = mol_from_coords(coords_A)
    mf = make_mf_d2_gr(mol, solvent='water', grid_response=True,
                       grid_level=GRID_LEVEL)
    mf.with_solvent.lebedev_order = order
    return mol, mf


def surf_pts_of(mf):
    surf = getattr(mf.with_solvent, 'surface', None)
    if surf is None:
        return None
    area = surf.get('area') if isinstance(surf, dict) else None
    return None if area is None else int(np.asarray(area).size)


def eval_at(coords_A, order, tag):
    """One full SMD energy+gradient evaluation with hard config checks."""
    mol, mf = build_mf(coords_A, order)
    # settings BEFORE kernel
    pre = dict(lebedev=int(mf.with_solvent.lebedev_order),
               grid_level=int(mf.grids.level))
    if pre['lebedev'] != order or pre['grid_level'] != GRID_LEVEL:
        raise RuntimeError('CONFIG ROLLBACK before kernel (%s): %s' % (tag, pre))
    t0 = time.time()
    mf.kernel()
    g = np.asarray(mf.nuc_grad_method().kernel(), float).reshape(6, 3)
    ss = mf.scf_summary
    # settings AFTER kernel
    post = dict(lebedev=int(mf.with_solvent.lebedev_order),
                grid_level=int(mf.grids.level),
                grid_points=(int(mf.grids.weights.size)
                             if mf.grids.weights is not None else None),
                surface_points=surf_pts_of(mf))
    if post['lebedev'] != order or post['grid_level'] != GRID_LEVEL:
        raise RuntimeError('CONFIG ROLLBACK after kernel (%s): %s' % (tag, post))
    e = float(mf.e_tot)
    rec = dict(
        tag=tag,
        coords_angstrom=np.asarray(coords_A, float).tolist(),
        coords_sha_full=sha(np.asarray(coords_A, float).tobytes()),
        coords_unit='Angstrom',
        e_total=e,
        gradient_full_6x3=g.tolist(),
        grad_max=float(np.abs(g).max()),
        grad_rms=float(np.sqrt((g ** 2).mean())),
        e_solvent=(float(ss['e_solvent']) if 'e_solvent' in ss else None),
        e_cds=(float(ss['e_cds']) if 'e_cds' in ss else None),
        e_d2_analytic=float(d2_full.d2_energy(mol)),
        scf_converged=bool(mf.converged),
        finite=bool(np.isfinite(e) and np.isfinite(g).all()),
        surface_order_actual=post['lebedev'],
        surface_points=post['surface_points'],
        dft_grid_level=post['grid_level'],
        dft_grid_points=post['grid_points'],
        method_fingerprint=method_fingerprint(mf),
        seconds=round(time.time() - t0, 1))
    if not rec['scf_converged']:
        raise RuntimeError('SCF not converged (%s)' % tag)
    if not rec['finite']:
        raise RuntimeError('non-finite E/g (%s)' % tag)
    return rec


def main():
    os.makedirs(OUT, exist_ok=True)
    # ---------- fixed input: 011 order47 NEW endpoint via formal index ----------
    idx = json.load(open(os.path.join(ART, 'smd_acceptance_evidence',
                                      'endpoint_index.json')))
    e47 = [x for x in idx['entries'] if x['id'] == 'c06_order47_reopt'][0]
    src = json.load(open(os.path.join(ROOT, e47['source_file'])))
    center = np.asarray(src['endpoint_coords_angstrom'], float)
    ref_e = float(src['independent_verification']['e_total'])
    ref_g = np.asarray(src['independent_verification']['gradient_full_6x3'],
                       float)
    idx_sha = hashlib.sha256(center.tobytes()).hexdigest()[:16]
    src_sha = e47['coords_sha_full_precision']
    assert idx_sha == src_sha == e47.get('coords_sha_full_precision'), \
        'endpoint index hash mismatch'
    print('center = 011 order47 endpoint (index-verified, sha=%s)' % idx_sha,
          flush=True)
    print('acceptance note: center max|g| = %.4e (formal threshold %.1e, '
          'NOT passed; authorized non-stationary diagnostic)'
          % (float(np.abs(ref_g).max()), GMAX_FORMAL), flush=True)

    # ---------- center recheck (tolerances fixed above) ----------
    c = eval_at(center, ORDER, 'center_order47')
    dE = abs(c['e_total'] - ref_e)
    dG = float(np.abs(np.asarray(c['gradient_full_6x3'], float)
                      - ref_g).max())
    c['dE_vs_011'] = dE
    c['dG_vs_011'] = dG
    c['tol_E'] = TOL_E
    c['tol_G'] = TOL_G
    c['reproducibility_pass'] = bool(dE <= TOL_E and dG <= TOL_G)
    print('CENTER RECHECK: dE=%.3e (tol %.0e)  dG=%.3e (tol %.0e)  pass=%s'
          % (dE, TOL_E, dG, TOL_G, c['reproducibility_pass']), flush=True)
    json.dump(c, open(os.path.join(OUT, 'center_order47.json'), 'w'),
              indent=2)
    if not c['reproducibility_pass']:
        print('STOP: center reproducibility FAILED -- investigate before '
              'any Hessian point (no points computed).', flush=True)
        return

    # ---------- 72 displaced points ----------
    g0 = np.asarray(c['gradient_full_6x3'], float)   # (6,3)
    e0 = c['e_total']
    points = []
    raw = {h: np.full((18, 18), np.nan) for h in H_LIST}
    n_ok = 0
    try:
        for i in range(6):
            for a in range(3):
                k = 3 * i + a
                for h in H_LIST:
                    for sgn in (+1, -1):
                        tag = 'dir%02d_atom%d_c%d_h%0.3f_sgn%+d' % (
                            k, i, a, h, sgn)
                        disp = np.zeros((6, 3))
                        disp[i, a] = sgn * h          # Bohr
                        coords_d = center + disp * BOHR_A   # Angstrom
                        actual = (np.asarray(
                            coords_d, float) - center) / BOHR_A
                        d_err = float(np.abs(actual - disp).max())
                        rec = eval_at(coords_d, ORDER, tag)
                        rec.update(dir_idx=k, atom=i, cart=a, h_bohr=h,
                                   sign=sgn,
                                   intended_disp_bohr=disp.tolist(),
                                   actual_disp_error_bohr=d_err)
                        if d_err > DISP_TOL:
                            raise RuntimeError('displacement mismatch %s: '
                                               '%.2e' % (tag, d_err))
                        points.append(rec)
                        n_ok += 1
                        print('[%2d/72] %s  E=%.9f  max|g|=%.3e  '
                              'surf=%s dft=%s  ok'
                              % (n_ok, tag, rec['e_total'], rec['grad_max'],
                                 rec['surface_points'],
                                 rec['dft_grid_points']), flush=True)
                        json.dump(points, open(
                            os.path.join(OUT, 'hessian_points.json'), 'w'),
                            indent=2)
        # ---------- assemble RAW (unsymmetrized) Hessians ----------
        gflat = {}
        for p in points:
            key = (p['dir_idx'], p['h_bohr'], p['sign'])
            gflat[key] = np.asarray(p['gradient_full_6x3'], float).reshape(-1)
        hess = {}
        for h in H_LIST:
            H = np.full((18, 18), np.nan)
            for k in range(18):
                gp = gflat[(k, h, +1)]
                gm = gflat[(k, h, -1)]
                H[k, :] = (gp - gm) / (2.0 * h)
            if not np.isfinite(H).all():
                raise RuntimeError('non-finite entries in Hessian h=%s' % h)
            asym = float(np.abs(H - H.T).max())
            rel_asym = float(np.abs(H - H.T).max()
                             / (np.abs(H).max() + 1e-300))
            Hs = 0.5 * (H + H.T)
            hess[h] = dict(
                h_bohr=h,
                antisym_residual_max=asym,
                antisym_relative=rel_asym,
                raw_hessian_18x18=H.tolist(),
                sym_hessian_18x18=Hs.tolist())
            print('h=%0.3f: |H-H^T|max=%.3e (rel %.2e)  raw saved first'
                  % (h, asym, rel_asym), flush=True)
        out = dict(
            job='JOB-2026-0906-013 Task B',
            center='c06_order47_reopt endpoint (011), index-verified',
            center_sha=idx_sha,
            config=dict(solvent='SMD(water)', order=ORDER,
                        grid_level=GRID_LEVEL, xc='wb97xd + project -D2',
                        basis='def2-TZVP', grid_response=True,
                        scf='1e-12/1e-9'),
            tolerance_fixed_before_run=dict(dE=TOL_E, dG=TOL_G,
                                            repro_pass=True),
            center_recheck=dict(e_total=e0, grad_max=float(np.abs(g0).max()),
                        dE=dE, dG=dG),
            n_points_ok=n_ok, n_points_expected=72,
            hessians=hess,
            note='raw (unsymmetrized) matrices saved verbatim; symmetrized '
                 'copies separate; antisymmetric residual reported before '
                 'symmetrization; raw never overwritten')
        json.dump(out, open(os.path.join(OUT, 'hessian_batch.json'), 'w'),
                  indent=2)
        print('HESSIAN BATCH COMPLETE: %d/72 points, matrices saved'
              % n_ok, flush=True)
    except Exception as e:
        traceback.print_exc()
        json.dump(dict(job='JOB-2026-0906-013 Task B',
                       aborted_at=len(points), error=repr(e),
                       completed_points=points),
                  open(os.path.join(OUT, 'hessian_batch.json'), 'w'),
                  indent=2)
        print('BATCH ABORTED at point %d: %s' % (len(points), e), flush=True)


if __name__ == '__main__':
    main()
