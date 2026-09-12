#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JOB-2026-0906-011 (Task B, pre-optimization path verification, NO optimization
here).  Confirms the ACTUAL call path that will feed the re-optimization:

  1. Production object at the 010 order-47 theta=0 geometry: recompute E/g,
     compare to the 010 theta=0 record (tolerance FIXED before comparison;
     missing fields never pass by default).
  2. The ACTUAL gradient scanner built from that mf reads surface order 47
     and DFT grid level 8.
  3. Independent native SMD + explicit -D2 dispersion baseline:
        e_native(SMD, no D2) + d2_energy(mol)  ==  production e_tot
     and  wrapped grad - dft_only_grad(mf)      ==  d2_grad(mol)
     so D2 is counted exactly once and the SMD path is the real one.
  4. Settings read BEFORE and AFTER the kernel (catch any config rollback).

Read-only on historical files; writes only into smd_order47_opt/.
"""
import os
import sys
import json
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, 'run_artifacts', '01_pure_water_o3_h2o')
OUT = os.path.join(ART, 'smd_order47_opt')
sys.path.insert(0, HERE)

from pyscf import gto, dft
from pyscf.solvent import smd
import d2_full
from grad_factory import make_mf_d2_gr, method_fingerprint, dft_only_grad
from smd_surface_refine_config import GRID_LEVEL, BASELINE_ORDER, TARGET_ORDER

SYMS = ['O', 'O', 'O', 'O', 'H', 'H']
ORDER = TARGET_ORDER                      # 47
TOL_E = 1e-8                              # fixed before any comparison
TOL_G = 1e-7
SCF_TOL = 1e-12
SCF_TOL_GRAD = 1e-9


def mol_from_coords(coords_A):
    atom = "; ".join("%s %.10f %.10f %.10f" % (s, x, y, z)
                     for s, (x, y, z) in zip(SYMS, coords_A))
    return gto.M(atom=atom, basis='def2-TZVP', charge=0, spin=0,
                 verbose=0, max_memory=4000)


def build_production(coords_A, order):
    mol = mol_from_coords(coords_A)
    mf = make_mf_d2_gr(mol, solvent='water', grid_response=True,
                       grid_level=GRID_LEVEL)
    mf.with_solvent.lebedev_order = order
    return mol, mf


def build_native_smd(coords_A, order):
    """Independent SMD object WITHOUT the project -D2 wrapper."""
    mol = mol_from_coords(coords_A)
    base = dft.RKS(mol)
    mf = smd.smd_for_scf(base, solvent_obj=smd.SMD(mol, solvent='water'))
    mf.xc = 'wb97xd'
    mf.grids.level = GRID_LEVEL
    mf.conv_tol = SCF_TOL
    mf.conv_tol_grad = SCF_TOL_GRAD
    mf.max_cycle = 200
    mf.with_solvent.lebedev_order = order
    return mol, mf


def surf_pts_of(mf):
    surf = getattr(mf.with_solvent, 'surface', None)
    if surf is None:
        return None
    area = surf.get('area') if isinstance(surf, dict) else None
    if area is None:
        return None
    return int(np.asarray(area).size)


def main():
    os.makedirs(OUT, exist_ok=True)
    o10 = json.load(open(os.path.join(ART, 'smd_surface_refine',
                                      'order47_five_points.json')))
    p0 = [p for p in o10['points'] if p['theta_deg'] == 0.0][0]
    coords0 = np.asarray(p0['coords_angstrom'], float)
    rec = dict(e_total=p0['e_total'],
               grad=np.asarray(p0['gradient_full_6x3'], float),
               surface_order_actual=p0['surface_order_actual'],
               dft_grid_points=p0['dft_grid_points'],
               surface_points=p0['surface_points'],
               scf_converged=p0['scf_converged'],
               e_solvent=p0['e_solvent'], e_cds=p0['e_cds'])

    # ---- (1) production recompute at order 47 ----
    mol, mf = build_production(coords0, ORDER)
    cfg_before = dict(
        lebedev=int(mf.with_solvent.lebedev_order),
        grid_level=int(mf.grids.level),
        grid_points=(int(mf.grids.weights.size)
                     if mf.grids.weights is not None else None),
        xc=mf.xc, d2_attached=bool(getattr(mf, '_has_full_d2', False)),
        solvent=('smd' if getattr(mf, '_d2_parent_cls', None) is not None
                 and 'SMD' in type(mf._d2_parent_cls).__name__ else 'gas'))
    t0 = time.time()
    mf.kernel()
    g = np.asarray(mf.nuc_grad_method().kernel(), float).reshape(6, 3)
    e_prod = float(mf.e_tot)
    s_pts = surf_pts_of(mf)
    cfg_after = dict(
        lebedev=int(mf.with_solvent.lebedev_order),
        grid_points=(int(mf.grids.weights.size)
                     if mf.grids.weights is not None else None),
        surface_points=s_pts,
        scf_converged=bool(mf.converged),
        finite=bool(np.isfinite(e_prod) and np.isfinite(g).all()))

    # compare to 010 theta=0 record (tolerance fixed above)
    dE = abs(e_prod - rec['e_total'])
    dG = float(np.abs(g - rec['grad']).max())
    compare = dict(
        dE=float(dE), dG=dG,
        tol_E=TOL_E, tol_G=TOL_G,
        e_pass=bool(dE <= TOL_E), g_pass=bool(dG <= TOL_G),
        surf_ord_match=bool(int(mf.with_solvent.lebedev_order)
                            == rec['surface_order_actual']),
        dft_pts_match=bool(cfg_after['grid_points'] == rec['dft_grid_points']),
        scf_match=bool(cfg_after['scf_converged'] == rec['scf_converged']))

    # ---- (2) actual gradient scanner reads order 47 / L8 ----
    gsc = mf.nuc_grad_method().as_scanner()
    scanner_lebedev = int(gsc.base.with_solvent.lebedev_order)
    scanner_gridlvl = int(gsc.base.grids.level)
    scanner = dict(lebedev=scanner_lebedev, grid_level=scanner_gridlvl,
                   lebedev_ok=bool(scanner_lebedev == ORDER),
                   grid_ok=bool(scanner_gridlvl == GRID_LEVEL))

    # ---- (3) independent native SMD + explicit -D2 baseline ----
    mol_n, mf_n = build_native_smd(coords0, ORDER)
    mf_n.kernel()
    e_native = float(mf_n.e_tot)
    e_d2 = float(d2_full.d2_energy(mol_n))
    e_recon = e_native + e_d2
    dE_recon = abs(e_recon - e_prod)
    # D2-once in the gradient
    g_native = np.asarray(dft_only_grad(mf).kernel(), float).reshape(6, 3)
    g_d2 = d2_full.d2_grad(mol)
    dG_d2 = float(np.abs(g - (g_native + g_d2)).max())
    baseline = dict(
        e_native_SMD_noD2=e_native,
        e_d2_analytic=e_d2,
        e_recon=e_recon,
        dE_recon_vs_production=float(dE_recon),
        dE_recon_pass=bool(dE_recon <= 1e-9),
        dG_D2_once_maxdiff=dG_d2,
        dG_D2_once_pass=bool(dG_d2 <= 1e-9),
        e_cds_production=float(mf.scf_summary.get('e_cds', float('nan'))),
        e_cds_native=float(mf_n.scf_summary.get('e_cds', float('nan'))),
        e_solvent_production=float(mf.scf_summary.get('e_solvent', float('nan'))),
        e_solvent_native=float(mf_n.scf_summary.get('e_solvent', float('nan'))))

    # ---- gate decision ----
    rollback_guard = bool(
        cfg_before['lebedev'] == cfg_after['lebedev'] == ORDER
        and cfg_before['grid_level'] == GRID_LEVEL
        and cfg_after['grid_points'] is not None)
    gate_ok = bool(
        compare['e_pass'] and compare['g_pass'] and compare['surf_ord_match']
        and compare['dft_pts_match'] and compare['scf_match']
        and scanner['lebedev_ok'] and scanner['grid_ok']
        and baseline['dE_recon_pass'] and baseline['dG_D2_once_pass'])

    result = dict(
        job='JOB-2026-0906-011 Task B',
        geometry='010 order47 theta=0 (original c06 step-40, original orientation)',
        order=ORDER, grid_level=GRID_LEVEL,
        baseline_order=BASELINE_ORDER,
        target_order=TARGET_ORDER,
        config_before=cfg_before,
        config_after=cfg_after,
        production=dict(e_total=e_prod, grad_max=float(np.abs(g).max()),
                        grad_rms=float(np.sqrt((g ** 2).mean())),
                        surface_points=s_pts,
                        seconds=round(time.time() - t0, 1)),
        record_010_theta0=dict(
            e_total=rec['e_total'],
            grad_max=float(np.abs(rec['grad']).max()),
            surface_order_actual=rec['surface_order_actual'],
            dft_grid_points=rec['dft_grid_points'],
            surface_points=rec['surface_points'],
            scf_converged=rec['scf_converged']),
        comparison_to_010=compare,
        scanner_actual=scanner,
        independent_baseline=baseline,
        rollback_guard=rollback_guard,
        gate=('PASS' if (gate_ok and rollback_guard) else 'FAIL'),
        note=('All checks must pass before launching the <=30-step '
              're-optimization; missing fields never pass by default.'))
    out = os.path.join(OUT, 'preopt_verify.json')
    json.dump(result, open(out, 'w'), indent=2)
    print('TASK B GATE:', result['gate'], '->', out)
    print(json.dumps(dict(order=ORDER, comparison=compare, scanner=scanner,
                          baseline_dE_recon=baseline['dE_recon_vs_production'],
                          baseline_dG_d2=baseline['dG_D2_once_maxdiff'],
                          rollback_guard=rollback_guard), indent=2))
    return result


if __name__ == '__main__':
    main()
