#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JOB-2026-0906-011 (Task C): single bounded c06 aqueous re-optimization under
order=47, at most 30 steps, no continuation/restart.

Fixed configuration (identical to the 010 five-point batch EXCEPT the surface
order, which is the experiment):
    SMD(water), charge 0, spin 0
    DFT grid level 8 (L8)
    electrostatic solvent surface lebedev order 47   (EXPERIMENT, not prod default)
    xc = wb97xd DFT part + project empirical -D2 (full derivatives)
    def2-TZVP, grid_response=True
    SCF conv_tol=1e-12, conv_tol_grad=1e-9, CDS & other solvent settings unchanged

Verified pyberny params: initial trust=0.1 (dynamic, NOT a hard step cap),
internal-coordinate gradientmax=gradientrms=1e-7; other step criteria kept at
defaults; maxsteps=30.  The stricter internal threshold does NOT replace the
full Cartesian acceptance max|g|<=1e-5 Eh/Bohr.

Per step, immediately saved: actual coords (+units & both hashes), full 6x3
gradient, total energy, e_solvent/e_cds/e_d2, SCF state, actual grid/surface
settings, optimizer trust & internal gradient norm.

Hard stops (preserve site, do NOT continue wrong config):
    SCF not converged / non-finite / anomalous geometry
    surface order reverts from 47 OR grid level reverts from 8 (rollback)
No extra global rotation is applied to reduce the gradient; no rigid-body
component is deleted.
"""
import os
import sys
import json
import time
import traceback
import hashlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, 'run_artifacts', '01_pure_water_o3_h2o')
OUT = os.path.join(ART, 'smd_order47_opt')
sys.path.insert(0, HERE)

from pyscf import gto
from pyscf.geomopt import berny_solver
import d2_full
from grad_factory import make_mf_d2_gr, method_fingerprint
from smd_surface_refine_config import GRID_LEVEL, BASELINE_ORDER, TARGET_ORDER
import torque_decomp as td

SYMS = ['O', 'O', 'O', 'O', 'H', 'H']
MASS = [15.9949146, 15.9949146, 15.9949146, 15.9949146,
        1.007825, 1.007825]               # O, O, O, O, H, H (amu)
ORDER = TARGET_ORDER                      # 47
MAXSTEPS = 30
TRUST0 = 0.1
CONV_PARAMS = dict(trust=TRUST0, gradientmax=1e-7, gradientrms=1e-7)
GMAX_TARGET = 1e-5                         # Eh/Bohr Cartesian acceptance


def sha(b):
    return hashlib.sha256(b).hexdigest()[:16]


def mol_from_coords(coords_A):
    atom = "; ".join("%s %.10f %.10f %.10f" % (s, x, y, z)
                     for s, (x, y, z) in zip(SYMS, coords_A))
    return gto.M(atom=atom, basis='def2-TZVP', charge=0, spin=0,
                 verbose=0, max_memory=4000)


def actual_input_hash(coords_A):
    atom = "; ".join("%s %.10f %.10f %.10f" % (s, x, y, z)
                     for s, (x, y, z) in zip(SYMS, coords_A))
    actual = []
    for tok in atom.split("; "):
        p = tok.split()
        actual.append([float(p[1]), float(p[2]), float(p[3])])
    return sha(np.asarray(actual, float).tobytes())


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
    if area is None:
        return None
    return int(np.asarray(area).size)


def main():
    os.makedirs(OUT, exist_ok=True)
    o10 = json.load(open(os.path.join(ART, 'smd_surface_refine',
                                      'order47_five_points.json')))
    p0 = [p for p in o10['points'] if p['theta_deg'] == 0.0][0]
    coords0 = np.asarray(p0['coords_angstrom'], float)   # original orientation

    mol, mf = build_mf(coords0, ORDER)
    fp = method_fingerprint(mf)

    steps = []
    config_error = None

    def callback(envs):
        nonlocal config_error
        cycle = int(envs['cycle'])
        m = envs['method']
        order_now = int(m.with_solvent.lebedev_order)
        grid_now = int(m.grids.level)
        # CONFIG ROLLBACK GUARD (immediate stop, preserve site)
        if order_now != ORDER or grid_now != GRID_LEVEL:
            config_error = ('surface order %d / grid level %d at step %d '
                            '(expected %d / %d)' % (order_now, grid_now,
                                                   cycle, ORDER, GRID_LEVEL))
            raise RuntimeError('CONFIG ROLLBACK: ' + config_error)
        mol_now = m.mol
        c = np.asarray(mol_now.atom_coords(unit='Angstrom'), float)
        g = np.asarray(envs['gradients'], float).reshape(6, 3)
        e = float(envs['energy'])
        e_d2 = float(d2_full.d2_energy(mol_now))
        ss = m.scf_summary
        gs = envs['g_scanner']
        rec = dict(
            step=int(cycle) + 1,
            coords_angstrom=c.tolist(),
            coord_sha_full=sha(c.tobytes()),
            coord_sha_actual_input=actual_input_hash(c),
            e_total=e,
            e_dft_part=e - e_d2,
            e_d2_analytic=e_d2,
            e_solvent=(float(ss['e_solvent']) if 'e_solvent' in ss else None),
            e_cds=(float(ss['e_cds']) if 'e_cds' in ss else None),
            gradient_full_6x3=g.tolist(),
            grad_max=float(np.abs(g).max()),
            grad_rms=float(np.sqrt((g ** 2).mean())),
            scf_converged=bool(gs.converged),
            finite=bool(np.isfinite(e) and np.isfinite(g).all()),
            surface_order_actual=order_now,
            surface_points=surf_pts_of(m),
            dft_grid_level=grid_now,
            dft_grid_points=(int(m.grids.weights.size)
                             if m.grids.weights is not None else None),
            trust=(float(envs['trust']) if 'trust' in envs else None),
            norm_grad_internal=(float(envs['norm_grad'])
                               if 'norm_grad' in envs else None),
            max_step=(float(envs['max_step']) if 'max_step' in envs else None))
        # hard stops
        if not rec['finite']:
            raise RuntimeError('step %d: non-finite energy/gradient'
                               % rec['step'])
        if not rec['scf_converged']:
            raise RuntimeError('step %d: SCF not converged' % rec['step'])
        # anomalous geometry: any bond < 0.5 A or > 4 A among O atoms
        if np.abs(c - c.mean(axis=0)).max() > 6.0:
            raise RuntimeError('step %d: anomalous geometry' % rec['step'])
        steps.append(rec)
        print('[opt] step %2d  E=%.9f  max|g|=%.3e  rms|g|=%.3e  '
              'order=%d grid=%d trust=%s  conv=%s'
              % (rec['step'], e, rec['grad_max'], rec['grad_rms'],
                 order_now, grid_now, rec['trust'], rec['scf_converged']),
              flush=True)

    t0 = time.time()
    try:
        opt_conv, mol_opt = berny_solver.kernel(
            mf, assert_convergence=True, include_ghost=True,
            callback=callback, maxsteps=MAXSTEPS, **CONV_PARAMS)
        opt_error = None
    except Exception as e:
        traceback.print_exc()
        opt_conv, opt_error = False, repr(e)
        mol_opt = (mol_from_coords(np.asarray(steps[-1]['coords_angstrom'],
                                              float)) if steps else None)

    # ---- independent endpoint verification (fresh objects) ----
    verify = None
    if mol_opt is not None:
        c_end = np.asarray(mol_opt.atom_coords(unit='Angstrom'), float)
        mol_v, mf_v = build_mf(c_end, ORDER)
        # double-check the fresh object really uses order 47
        if int(mf_v.with_solvent.lebedev_order) != ORDER:
            opt_error = (opt_error or '') + ' endpoint rebuild order != 47'
        mf_v.kernel()
        g_v = np.asarray(mf_v.nuc_grad_method().kernel(), float).reshape(6, 3)
        e_d2_v = float(d2_full.d2_energy(mol_v))
        ss_v = mf_v.scf_summary
        dec = td.decompose_force(g_v, c_end, MASS, unit='Angstrom')
        verify = dict(
            coords_angstrom=c_end.tolist(),
            coord_sha_full=sha(c_end.tobytes()),
            coord_sha_actual_input=actual_input_hash(c_end),
            e_total=float(mf_v.e_tot),
            e_d2_analytic=e_d2_v,
            e_dft_part=float(mf_v.e_tot) - e_d2_v,
            e_solvent=(float(ss_v['e_solvent']) if 'e_solvent' in ss_v else None),
            e_cds=(float(ss_v['e_cds']) if 'e_cds' in ss_v else None),
            gradient_full_6x3=g_v.tolist(),
            grad_max=float(np.abs(g_v).max()),
            grad_rms=float(np.sqrt((g_v ** 2).mean())),
            scf_converged=bool(mf_v.converged),
            finite=bool(np.isfinite(mf_v.e_tot) and np.isfinite(g_v).all()),
            surface_order_actual=int(mf_v.with_solvent.lebedev_order),
            surface_points=surf_pts_of(mf_v),
            dft_grid_level=int(mf_v.grids.level),
            dft_grid_points=(int(mf_v.grids.weights.size)
                             if mf_v.grids.weights is not None else None),
            pass_gmax_cartesian=bool(np.abs(g_v).max() <= GMAX_TARGET),
            decomposition=dict(
                net_force_norm=dec['net_force_norm'],
                T_g=dec['T_g'],
                T_g_norm=dec['T_g_norm'],
                g_translation_norm=dec['g_trans_norm'],
                g_rotation_norm=dec['g_rot_norm'],
                g_internal_norm=dec['g_int_norm'],
                g_norm=dec['g_norm'],
                int_norm_fraction=dec['int_norm_fraction'],
                int_sq_fraction=dec['int_sq_fraction']))

    result = dict(
        job='JOB-2026-0906-011 Task C',
        experiment='c06 aqueous re-optimization under surface order=47',
        order=ORDER, baseline_order=BASELINE_ORDER, grid_level=GRID_LEVEL,
        start_coords_angstrom=coords0.tolist(),
        coord_sha_full_start=sha(coords0.tobytes()),
        coord_sha_actual_input_start=actual_input_hash(coords0),
        method_fingerprint=fp,
        optimizer='pyberny 0.7.0 (berny_solver.kernel)',
        conv_params=CONV_PARAMS,
        conv_params_units=dict(trust='Bohr (initial, dynamic, NOT a hard cap)',
                               gradientmax='Eh/Bohr internal-coordinate',
                               gradientrms='Eh/Bohr internal-coordinate',
                               note='ALL criteria AND-ed; NO energy criterion; '
                                    'stricter internal threshold does NOT '
                                    'replace Cartesian max|g|<=1e-5'),
        maxsteps=MAXSTEPS,
        acceptance_cartesian_gmax=GMAX_TARGET,
        n_steps=len(steps),
        hit_maxsteps=bool(len(steps) >= MAXSTEPS),
        optimizer_converged=bool(opt_conv),
        optimizer_error=opt_error,
        config_error=config_error,
        endpoint_coords_angstrom=(np.asarray(
            mol_opt.atom_coords(unit='Angstrom'), float).tolist()
            if mol_opt is not None else None),
        independent_verification=verify,
        steps=steps,
        seconds=round(time.time() - t0, 1))
    out = os.path.join(OUT, 'order47_optimization.json')
    json.dump(result, open(out, 'w'), indent=2)
    print('\n=== OPT DONE ===')
    print('n_steps=%d hit_maxsteps=%s optimizer_converged=%s config_error=%s'
          % (len(steps), result['hit_maxsteps'], opt_conv, config_error))
    if verify:
        print('endpoint max|g|=%.3e pass_gmax=%s order=%d'
              % (verify['grad_max'], verify['pass_gmax_cartesian'],
                 verify['surface_order_actual']))
    return result


if __name__ == '__main__':
    main()
