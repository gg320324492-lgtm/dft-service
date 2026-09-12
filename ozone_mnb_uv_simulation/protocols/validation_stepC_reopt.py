#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Final-minima-validation step C (JOB-2026-0905-009): re-optimise the two
candidate structures at UNIFIED accuracy and compute the analytic Hessian at
the new endpoints.

Unified accuracy (explicitly set and recorded):
  grid level 6, SCF conv_tol = 1e-12, conv_tol_grad = 1e-9 (orbital-gradient
  threshold), full-derivative -D2 via make_mf_d2;
  geomeTRIC convergence_gmax = 1e-5 Eh/Bohr (brief: max nuclear gradient
  <= 1e-5), grms = 5e-6, displacement criteria disabled.
Acceptance: a FRESH make_mf_d2 single point must reproduce max|grad| <= 1e-5;
failure is recorded as evidence (structure marked 待定), the criterion is
NEVER relaxed.  Old->new coordinate correspondence (revised RMSD + max atom
shift) is saved; the analytic Hessian of the new endpoint is computed at the
same unified settings.
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
import d2_full
import conformer_utils as cu
import run_baseline as rb
from pyscf.geomopt.geometric_solver import optimize

ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, 'run_artifacts', '01_pure_water_o3_h2o')
FV = os.path.join(ART, 'final_minima_validation')
os.makedirs(FV, exist_ok=True)

SYMS = ['O', 'O', 'O', 'O', 'H', 'H']
TARGETS = ['c14_plus', 'c06_plus']
GRID_LEVEL = 6
SCF_TOL = 1e-12
SCF_TOL_GRAD = 1e-9          # orbital-gradient threshold, explicit
GMAX_TARGET = 1e-5           # Eh/Bohr, brief acceptance threshold
OPT_CONV = dict(maxsteps=400, assert_convergence=False,
                convergence_gmax=GMAX_TARGET, convergence_grms=5e-6,
                convergence_dmax=10.0, convergence_drms=10.0,
                convergence_energy=1e-8)


def make_mf(mol):
    mf = d2_full.make_mf_d2(mol)
    mf.grids.level = GRID_LEVEL
    mf.conv_tol = SCF_TOL
    mf.conv_tol_grad = SCF_TOL_GRAD
    return mf


def fresh_check(coords):
    mol = cu.mol_from_coords(coords, SYMS, verbose=0)
    mf = make_mf(mol)
    mf.kernel()
    g = np.asarray(mf.nuc_grad_method().kernel())
    return float(mf.e_tot), float(np.abs(g).max()), bool(mf.converged)


def run_target(tid):
    t0 = time.time()
    out_path = os.path.join(FV, 'stepC_reopt_%s.json' % tid)
    if os.path.exists(out_path):
        print('[skip] %s' % tid, flush=True)
        with open(out_path) as fh:
            return json.load(fh)

    old = np.asarray(json.load(open(
        os.path.join(ART, 'mode_follow', '%s.json' % tid)))['geometry_angstrom'], float)
    mol = cu.mol_from_coords(old, SYMS, verbose=0)
    mf = make_mf(mol)
    t1 = time.time()
    mol_opt = optimize(mf, **OPT_CONV)
    coords = np.asarray(mol_opt.atom_coords(unit='Angstrom'), dtype=float)
    opt_seconds = round(time.time() - t1, 1)

    # fresh-object acceptance check
    e_new, gmax_new, scf_ok = fresh_check(coords)
    converged = bool(gmax_new <= GMAX_TARGET and scf_ok)

    # old->new correspondence (revised registration)
    rms_old_new = cu.best_rmsd(old, coords)
    max_shift = float(np.abs(coords - old).max())

    # analytic Hessian at the new endpoint, same unified settings
    mol_new = cu.mol_from_coords(coords, SYMS, verbose=0)
    mf_h = make_mf(mol_new)
    mf_h.kernel()
    h_raw = mf_h.Hessian().kernel()
    asym = float(np.abs(h_raw - h_raw.transpose(1, 0, 3, 2)).max())
    h = 0.5 * (h_raw + h_raw.transpose(1, 0, 3, 2))
    ha = rb.pyscf_thermo.harmonic_analysis(mol_new, h, imaginary_freq=False)
    modes = np.sort(np.asarray(ha['freq_wavenumber'], dtype=float))

    res = dict(id=tid, grid_level=GRID_LEVEL, scf_tol=SCF_TOL,
               scf_tol_grad=SCF_TOL_GRAD, gmax_target=GMAX_TARGET,
               old_geometry=old.tolist(), new_geometry=coords.tolist(),
               old_new_rmsd_angstrom=rms_old_new, old_new_max_shift=max_shift,
               opt_seconds=opt_seconds,
               fresh_e_total=e_new, fresh_max_gradient=gmax_new,
               fresh_scf_converged=scf_ok,
               gradient_target_met=converged,
               hessian_pre_symmetry_asymmetry=asym,
               hessian_projected=dict(
                   freq_error=int(ha.get('freq_error', -1)),
                   n_imag=int(np.sum(modes < -1e-6)),
                   lowest6=[round(float(f), 2) for f in modes[:6]]),
               status=('reoptimised_converged' if converged
                       else 'GRADIENT_TARGET_NOT_MET_pending'),
               seconds=round(time.time() - t0, 1))
    with open(out_path, 'w') as fh:
        json.dump(res, fh, indent=2)
    print('[%s] %s  fresh max|grad|=%.2e (target %.0e)  rms(old,new)=%.4f  '
          'n_imag=%d lowest=%s  (%.0f s)'
          % (tid, res['status'], gmax_new, GMAX_TARGET, rms_old_new,
             res['hessian_projected']['n_imag'],
             res['hessian_projected']['lowest6'][:3], res['seconds']), flush=True)
    return res


def _init():
    import pyscf.lib as lib
    lib.num_threads(8)


def main():
    done = []
    todo = []
    for t in TARGETS:
        if os.path.exists(os.path.join(FV, 'stepC_reopt_%s.json' % t)):
            with open(os.path.join(FV, 'stepC_reopt_%s.json' % t)) as fh:
                done.append(json.load(fh))
        else:
            todo.append(t)
    results = done
    if todo:
        with Pool(len(todo), initializer=_init) as pool:
            results.extend(pool.map(run_target, todo))
    results.sort(key=lambda r: TARGETS.index(r['id']))
    with open(os.path.join(FV, 'stepC_reopt_summary.json'), 'w') as fh:
        json.dump(dict(job='JOB-2026-0905-009',
                       step='final_validation_C_reoptimisation',
                       settings=dict(grid_level=GRID_LEVEL, scf_tol=SCF_TOL,
                                     scf_tol_grad=SCF_TOL_GRAD,
                                     gmax_target=GMAX_TARGET),
                       targets=results), fh, indent=2)
    print('SAVED stepC summary', flush=True)


if __name__ == '__main__':
    main()
