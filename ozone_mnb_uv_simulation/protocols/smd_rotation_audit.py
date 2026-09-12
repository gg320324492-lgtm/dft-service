#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JOB-2026-0906-009: rigid-rotation dependence & residual-torque source audit.

For each candidate (c14: 008 continuation endpoint; c06: 006 step-40
endpoint):
  * axis = corrected-centre T_g direction (fixed once from the ORIGINAL
    centre; never re-chosen);
  * EXACT finite rigid rotation of all atoms (Rodrigues; NOT a first-order
    displacement) about the mass-COM by theta = 0, +/-0.1, +/-0.2 degrees;
  * per orientation a FRESH production SMD(water)/L8 object is built and
    converged (no scanner reuse -> no cache-fake invariance); recorded:
    full gradient, energy, e_solvent/e_cds/e_d2, SCF state, DFT grid and
    solvent surface configuration + point counts;
  * the gradient is rotated BACK into the original frame (g_back = R^T g')
    and compared with the original gradient;
  * central slope [E(+th)-E(-th)]/(2 th_rad) vs axis . T_g(0);
  * dispersion energy rotational invariance and dispersion-gradient
    covariance (analytic positive control).

c06 single-factor controls on the same five rotated geometries:
  A. DFT grid L8 -> L9 (solvent surface untouched);
  B. DFT stays L8, solvent surface lebedev_order 29 -> 31 (next supported
     order; 302 -> 350 points per sphere), verified by the ACTUAL surface
     point count.
Budget: 10 baseline + 10 control evaluations.  No optimisation, no
threshold change, no frequencies.
"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, 'run_artifacts', '01_pure_water_o3_h2o')
OUT = os.path.join(ART, 'smd_rotation_audit')
sys.path.insert(0, HERE)

from pyscf import gto
from pyscf.solvent import smd
from pyscf.dft import gen_grid as dft_gen_grid
import d2_full
from grad_factory import make_mf_d2_gr
import torque_decomp as td

SYMS = ['O', 'O', 'O', 'O', 'H', 'H']
MASS = np.array([15.999, 15.999, 15.999, 15.999, 1.008, 1.008])
GRID_LEVEL = 8
THETAS_DEG = [0.0, 0.1, -0.1, 0.2, -0.2]


def build(coords_A, grid_level=GRID_LEVEL, lebedev_order=None):
    atom = "; ".join("%s %.10f %.10f %.10f" % (s, x, y, z)
                     for s, (x, y, z) in zip(SYMS, coords_A))
    mol = gto.M(atom=atom, basis='def2-TZVP', charge=0, spin=0,
                verbose=0, max_memory=4000)
    mf = make_mf_d2_gr(mol, solvent='water', grid_response=True,
                       grid_level=grid_level)
    if lebedev_order is not None:
        mf.with_solvent.lebedev_order = lebedev_order
    return mol, mf


def eval_full(mol, mf):
    t0 = time.time()
    mf.kernel()
    g = np.asarray(mf.nuc_grad_method().kernel(), float).reshape(6, 3)
    ss = mf.scf_summary
    ws = mf.with_solvent
    surf_pts = None
    if getattr(ws, 'surface', None) is not None:
        try:
            surf_pts = int(np.asarray(ws.surface['area']).size)
        except Exception:
            surf_pts = 'unreadable'
    return dict(e_total=float(mf.e_tot),
                gradient_full_6x3=g.tolist(),
                grad_max=float(np.abs(g).max()),
                scf_converged=bool(mf.converged),
                finite=bool(np.isfinite(mf.e_tot) and np.isfinite(g).all()),
                e_solvent=(float(ss['e_solvent']) if 'e_solvent' in ss
                           else None),
                e_cds=(float(ss['e_cds']) if 'e_cds' in ss else None),
                e_d2_analytic=float(d2_full.d2_energy(mol)),
                d2_in_scf_summary=float(ss.get('d2_dispersion',
                                               float('nan'))),
                dft_grid_points=(int(mf.grids.weights.size)
                                 if mf.grids.weights is not None else None),
                solvent_surface_points=surf_pts,
                dft_grid_level=int(mf.grids.level),
                lebedev_order=int(ws.lebedev_order),
                seconds=round(time.time() - t0, 1))


def audit_candidate(tid, coords_A, axis, T_g, controls=False):
    # axis and T_g are fixed by the caller from the unit-repaired
    # decomposition (endpoint_decomp_fixed.json); never re-chosen here
    T_g = np.asarray(T_g, float)
    T_norm = float(np.linalg.norm(T_g))
    axis = T_g / T_norm
    com_B = td.com_bohr(coords_A / td.BOHR_A, MASS)
    rows = []
    R_records = []
    for th_deg in THETAS_DEG:
        th = np.deg2rad(th_deg)
        R = td.rodrigues(axis, th)
        # verification of R
        rtR = float(np.abs(R.T.dot(R) - np.eye(3)).max())
        det = float(np.linalg.det(R))
        c_rot = td.rotate_coords(coords_A / td.BOHR_A, R, com_B) * td.BOHR_A
        # pairwise-distance preservation
        D0 = np.linalg.norm(coords_A[:, None, :] - coords_A[None, :, :],
                            axis=2)
        D1 = np.linalg.norm(c_rot[:, None, :] - c_rot[None, :, :], axis=2)
        ddev = float(np.abs(D0 - D1).max())
        assert rtR < 1e-12 and abs(det - 1) < 1e-12 and ddev < 1e-10, (
            rtR, det, ddev)
        R_records.append(dict(theta_deg=th_deg, axis=axis.tolist(),
                              center_bohr=com_B.tolist(),
                              R=R.tolist(), RT_R_err=rtR, det=det,
                              pairwise_dist_max_dev_A=ddev,
                              coords_angstrom=c_rot.tolist()))
        mol, mf = build(c_rot)
        r = eval_full(mol, mf)
        r['theta_deg'] = th_deg
        # rotate the gradient BACK into the original frame: g_back = R^T g'
        g_new = np.asarray(r['gradient_full_6x3'], float)
        g_back = (R.T @ g_new.T).T
        r['gradient_rotated_back_6x3'] = g_back.tolist()
        # dispersion positive control
        g_d2_rot = td.rotate_vectors(
            np.asarray(d2_full.d2_grad(mol), float).reshape(6, 3), R)
        # dispersion positive control: analytic d2 is exactly covariant
        d2_grad_orig = np.asarray(d2_full.d2_grad(mol), float).reshape(6, 3)
        r['d2_grad_covariance_resid'] = float(np.abs(
            (R.T @ g_d2_rot.T).T - d2_grad_orig).max())
        rows.append(r)
        print('[%s] th=%+5.1f° E=%.9f max|g|=%.3e surf_pts=%s dft=%s'
              % (tid, th_deg, r['e_total'], r['grad_max'],
                 r['solvent_surface_points'], r['dft_grid_points']),
              flush=True)
    # orientation dependence on the original-frame gradients
    g0 = np.asarray(rows[0]['gradient_full_6x3'], float).reshape(-1)
    for r in rows[1:]:
        gb = np.asarray(r['gradient_rotated_back_6x3'], float).reshape(-1)
        r['grad_back_vs_orig_maxabs'] = float(np.abs(gb - g0).max())
    dEs = [rows[i]['e_total'] - rows[0]['e_total']
           for i in range(len(rows))]
    # central slopes
    def slope(i_plus, i_minus, th_deg):
        th = np.deg2rad(th_deg)
        return float((rows[i_plus]['e_total'] - rows[i_minus]['e_total'])
                     / (2 * th))
    i_p1 = THETAS_DEG.index(0.1); i_m1 = THETAS_DEG.index(-0.1)
    i_p2 = THETAS_DEG.index(0.2); i_m2 = THETAS_DEG.index(-0.2)
    slope_01 = slope(i_p1, i_m1, 0.1)
    slope_02 = slope(i_p2, i_m2, 0.2)
    axis_dot_Tg = float(axis.dot(T_g))
    return dict(
        axis=axis.tolist(), T_g=T_g.tolist(), T_g_norm=T_norm,
        center_bohr=com_B.tolist(),
        rotations=R_records, evaluations=rows,
        dE_vs_theta=dEs,
        central_slope=[dict(theta_deg=0.1, slope_Eh_per_rad=slope_01,
                            axis_dot_Tg=axis_dot_Tg,
                            resid=slope_01 - axis_dot_Tg),
                       dict(theta_deg=0.2, slope_Eh_per_rad=slope_02,
                            axis_dot_Tg=axis_dot_Tg,
                            resid=slope_02 - axis_dot_Tg)],
        grad_back_vs_orig_max=[r.get('grad_back_vs_orig_maxabs')
                               for r in rows[1:]])


def main():
    os.makedirs(OUT, exist_ok=True)
    fx = json.load(open(os.path.join(OUT, 'endpoint_decomp_fixed.json')))
    pc = json.load(open(os.path.join(ART, 'smd_restart',
                                     'phaseC_smd_optimize.json')))
    c08 = json.load(open(os.path.join(ART, 'smd_controlled_opt',
                                      'c14_cont_result.json')))
    geoms = {
        'c14_plus': (np.asarray(c08['endpoint_coords_angstrom'], float),
                     '008 continuation endpoint (cumulative 120 steps)'),
        'c06_plus': (np.asarray(pc['c06_plus']['endpoint_coords_angstrom'],
                                float),
                     '006 step-40 endpoint')}
    out = dict(job='JOB-2026-0906-009', grid_level=GRID_LEVEL,
               thetas_deg=THETAS_DEG, candidates={})
    key_map = {'c14_plus': 'c14_plus_cont120', 'c06_plus': 'c06_plus_step40'}
    for tid, (coords_A, src) in geoms.items():
        print('=== rotation audit: %s (%s) ===' % (tid, src), flush=True)
        # axis from the FIXED decomposition T_g
        T_g = np.asarray(fx[key_map[tid]]['T_g'], float)
        axis = T_g / np.linalg.norm(T_g)
        T_g_c = np.asarray(fx[key_map[tid]]['T_g'], float)
        res = audit_candidate(tid, coords_A, axis, T_g_c, controls=False)
        res['start_source'] = src
        out['candidates'][tid] = res
        for c in res['central_slope']:
            print('[%s] slope(%.1f°)=%+.4e Eh/rad vs axis·T_g=%+.4e '
                  '(resid %+.3e)'
                  % (tid, c['theta_deg'], c['slope_Eh_per_rad'],
                     c['axis_dot_Tg'], c['resid']), flush=True)
        print('[%s] grad-back-vs-orig max: %s'
              % (tid, ['%.2e' % x for x in res['grad_back_vs_orig_max']]),
              flush=True)

    # ---- c06 single-factor controls ----
    coords_A, src = geoms['c06_plus']
    T_g = np.asarray(fx['c06_plus_step40']['T_g'], float)
    axis = T_g / np.linalg.norm(T_g)
    controls = {}
    for cname, kw in (('A_dft_L9', dict(grid_level=9)),
                      ('B_surface_leb31', dict(grid_level=GRID_LEVEL,
                                               lebedev_order=31))):
        print('=== c06 control %s ===' % cname, flush=True)
        rows = []
        for th_deg in THETAS_DEG:
            th = np.deg2rad(th_deg)
            R = td.rodrigues(axis, th)
            c_rot = td.rotate_coords(coords_A / td.BOHR_A, R, td.com_bohr(
                coords_A / td.BOHR_A, MASS)) * td.BOHR_A
            mol, mf = build(c_rot, **kw)
            r = eval_full(mol, mf)
            r['theta_deg'] = th_deg
            g_new = np.asarray(r['gradient_full_6x3'], float)
            g_back = (R.T @ g_new.T).T
            r['gradient_rotated_back_6x3'] = g_back.tolist()
            rows.append(r)
            print('[c06-%s] th=%+5.1f° E=%.9f max|g|=%.3e surf=%s dft=%s'
                  % (cname, th_deg, r['e_total'], r['grad_max'],
                     r['solvent_surface_points'], r['dft_grid_points']),
                  flush=True)
        g0 = np.asarray(rows[0]['gradient_full_6x3'], float).reshape(-1)
        for r in rows[1:]:
            gb = np.asarray(r['gradient_rotated_back_6x3'],
                            float).reshape(-1)
            r['grad_back_vs_orig_maxabs'] = float(np.abs(gb - g0).max())
        # orientation dependence relative to its own theta=0
        dE = [rows[i]['e_total'] - rows[0]['e_total']
              for i in range(len(rows))]
        gdep = [r.get('grad_back_vs_orig_maxabs') for r in rows[1:]]
        controls[cname] = dict(evaluations=rows, dE_vs_theta=dE,
                               grad_back_vs_orig_max=gdep,
                               config=dict(dft_grid_level=rows[0][
                                   'dft_grid_level'],
                                   lebedev_order=rows[0]['lebedev_order'],
                                   solvent_surface_points=rows[0][
                                       'solvent_surface_points']))
        print('[c06-%s] dE=%s  grad-dep=%s'
              % (cname, ['%.2e' % x for x in dE],
                 ['%.2e' % x for x in gdep]), flush=True)
    out['c06_controls'] = controls

    with open(os.path.join(OUT, 'rotation_audit_data.json'), 'w') as fh:
        json.dump(out, fh, indent=2)
    print('saved -> rotation_audit_data.json')


if __name__ == '__main__':
    main()
