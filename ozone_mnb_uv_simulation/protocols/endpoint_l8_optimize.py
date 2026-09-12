#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JOB-2026-0906-005 Phase B: ONE bounded L8 re-optimisation of c14_plus.

Start: the ACCEPTED L7 attempt2 endpoint (endpoints_fixed.json) -- NOT an
old search structure.  Everything identical to the verified Phase-C path
EXCEPT the grid, which is raised to level 8:
  gas phase, xc='wb97xd' (no -D2 in libxc) + project empirical full-derivative
  -D2 (Chai-Head-Gordon, verbatim), def2-TZVP, grid level 8,
  SCF conv_tol=1e-12 / conv_tol_grad=1e-9, grid_response=True enforced by
  grad_factory.make_mf_d2_gr (D2 exactly once),
  optimizer pyscf.geomopt.berny_solver.kernel (pyberny 0.7.0),
  CONFIRMED internal-coordinate thresholds gradientmax=gradientrms=1e-6
  (stepmax/steprms defaults), maxsteps=60.

ONE run only: no restart, no extension, no threshold change.

Per step the FULL 6x3 gradient is saved immediately (004's evidence gap is
closed here), together with actual coords, energy, dispersion component,
SCF state, actual grid level and method fingerprints.

Afterwards: independent freshly built mol+mf+gradient at the endpoint must
reach SCF-converged, finite, max|g| <= 1e-5 Eh/Bohr; all optimizer criteria
are recorded.  If not met: site preserved, marked pending, STOP (no retry).
"""
import os, sys, json, time, traceback
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, 'run_artifacts', '01_pure_water_o3_h2o')
OUT = os.path.join(ART, 'c14_l8_check')
sys.path.insert(0, HERE)

from pyscf import gto
from pyscf.geomopt import berny_solver
import d2_full
from grad_factory import make_mf_d2_gr, method_fingerprint

SYMS = ['O', 'O', 'O', 'O', 'H', 'H']
MAXSTEPS = 60
GRID_LEVEL = 8
GMAX_TARGET = 1e-5


def mol_from_coords(coords_A):
    atom = "; ".join("%s %.10f %.10f %.10f" % (s, x, y, z)
                     for s, (x, y, z) in zip(SYMS, coords_A))
    return gto.M(atom=atom, basis='def2-TZVP', charge=0, spin=0,
                 verbose=0, max_memory=4000)


def main():
    os.makedirs(OUT, exist_ok=True)
    fx = json.load(open(os.path.join(ART, 'endpoint_frequency',
                                     'endpoints_fixed.json')))
    start = np.asarray(fx['endpoints']['c14_plus']['coords_angstrom'], float)
    mol = mol_from_coords(start)
    mf = make_mf_d2_gr(mol, solvent=None, grid_response=True)
    mf.grids.level = GRID_LEVEL

    steps = []

    def callback(envs):
        m = envs['mol']
        grad = np.asarray(envs['gradients'], float).reshape(6, 3)
        rec = dict(
            step=int(envs['cycle']) + 1,
            coords_angstrom=np.asarray(m.atom_coords(unit='Angstrom'),
                                       float).tolist(),
            e_total=float(envs['energy']),
            gradient_full_6x3=grad.tolist(),          # FULL gradient saved
            grad_max=float(np.abs(grad).max()),
            grad_rms=float(np.sqrt((grad ** 2).mean())),
            scf_converged=bool(envs['g_scanner'].converged),
            d2_in_scf_summary=float(mf.scf_summary.get('d2_dispersion',
                                                       float('nan'))),
            e_d2_analytic=float(d2_full.d2_energy(m)),
            grid_level_actual=GRID_LEVEL,
            grid_response_actual=bool(
                getattr(envs['g_scanner'], 'grid_response')))
        if not np.isfinite(rec['e_total']) or not np.isfinite(grad).all():
            raise RuntimeError('step %d: non-finite energy/gradient'
                               % rec['step'])
        steps.append(rec)
        with open(os.path.join(OUT, 'l8_trajectory_partial.json'), 'w') as fh:
            json.dump(steps, fh, indent=2)            # incremental save
        print('[c14 L8] step %2d  E=%.9f  max|g|=%.3e  (%s)'
              % (rec['step'], rec['e_total'], rec['grad_max'],
                 'conv' if rec['scf_converged'] else 'NOT-CONV'), flush=True)

    t0 = time.time()
    try:
        opt_conv, mol_opt = berny_solver.kernel(
            mf, assert_convergence=True, include_ghost=True,
            callback=callback, maxsteps=MAXSTEPS,
            gradientmax=1e-6, gradientrms=1e-6)
        opt_error = None
    except Exception as e:
        traceback.print_exc()
        opt_conv, mol_opt, opt_error = False, None, repr(e)
        mol_opt = mol_from_coords(
            np.asarray(steps[-1]['coords_angstrom'], float)) if steps else None

    verify = None
    if mol_opt is not None:
        c_end = np.asarray(mol_opt.atom_coords(unit='Angstrom'), float)
        mol_v = mol_from_coords(c_end)
        mf_v = make_mf_d2_gr(mol_v, solvent=None, grid_response=True)
        mf_v.grids.level = GRID_LEVEL
        mf_v.kernel()
        g_v = np.asarray(mf_v.nuc_grad_method().kernel(), float).reshape(6, 3)
        verify = dict(
            e_total=float(mf_v.e_tot),
            gradient_full_6x3=g_v.tolist(),
            grad_max=float(np.abs(g_v).max()),
            grad_rms=float(np.sqrt((g_v ** 2).mean())),
            scf_converged=bool(mf_v.converged),
            finite=bool(np.isfinite(mf_v.e_tot) and np.isfinite(g_v).all()),
            d2_in_scf_summary=float(mf_v.scf_summary.get('d2_dispersion')),
            e_d2_analytic=float(d2_full.d2_energy(mol_v)),
            grid_level_actual=GRID_LEVEL,
            pass_gmax=bool(np.abs(g_v).max() <= GMAX_TARGET),
            dE_vs_last_step=float(mf_v.e_tot - steps[-1]['e_total'])
            if steps else None)

    result = dict(
        job='JOB-2026-0906-005 Phase B',
        candidate='c14_plus', grid_level=GRID_LEVEL,
        start_source='endpoints_fixed.json (L7 attempt2 accepted endpoint)',
        start_coords_angstrom=start.tolist(),
        maxsteps=MAXSTEPS, one_run_only=True,
        conv_params=dict(gradientmax=1e-6, gradientrms=1e-6,
                         stepmax='default 1.8e-3', steprms='default 1.2e-3'),
        conv_params_units='internal-coordinate gradients Eh per internal '
                          'unit (berny.py L294); ALL criteria AND-ed',
        method_fingerprint=method_fingerprint(mf),
        n_steps=len(steps),
        optimizer_converged=bool(opt_conv),
        optimizer_error=opt_error,
        final_criteria_last_step=dict(
            grad_max=steps[-1]['grad_max'] if steps else None,
            grad_rms=steps[-1]['grad_rms'] if steps else None),
        independent_verification=verify,
        steps=steps,
        endpoint_coords_angstrom=(
            np.asarray(mol_opt.atom_coords(unit='Angstrom'), float).tolist()
            if mol_opt is not None else None),
        seconds=round(time.time() - t0, 1))
    verdict = 'pending'
    if verify and opt_conv and verify['pass_gmax'] and verify['scf_converged'] \
            and verify['finite']:
        verdict = 'converged_stationary_candidate_L8'
    elif len(steps) >= MAXSTEPS:
        verdict = 'maxsteps_reached_not_converged'
    result['verdict'] = verdict
    with open(os.path.join(OUT, 'l8_optimize_result.json'), 'w') as fh:
        json.dump(result, fh, indent=2)
    print('verdict=%s  n_steps=%d  indep max|g|=%s'
          % (verdict, len(steps),
             ('%.3e' % verify['grad_max']) if verify else 'n/a'))


if __name__ == '__main__':
    main()
