#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JOB-2026-0906-006 Phase C: ONE bounded SMD(water)/L8 optimisation per
candidate, starting from the gas-phase registry structures.

Path: exactly the Phase-B-verified production path -- SMD(water),
def2-TZVP, grid level 8, xc='wb97xd' (libxc, no -D2) + project empirical
full-derivative -D2, grid_response=True, SCF 1e-12/1e-9,
optimizer pyberny 0.7.0 with CONFIRMED internal-coordinate thresholds
gradientmax=gradientrms=1e-6 (stepmax/steprms defaults), maxsteps=60.
ONE run per candidate: no restart, no extension, no threshold change.

Per step saved IMMEDIATELY: actual coords, total energy, solvent and
dispersion components, FULL 6x3 gradient, SCF state, method fingerprint.

Endpoint: independent freshly built SMD object must reach SCF-converged,
finite, max|g| <= 1e-5 Eh/Bohr with all optimizer criteria satisfied;
otherwise the candidate is marked NOT converged and the site is preserved
(no retry).  Dissociation / non-finite values abort the run.
"""
import os, sys, json, time, traceback
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, 'run_artifacts', '01_pure_water_o3_h2o')
OUT = os.path.join(ART, 'smd_restart')
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


def optimize(tid, start_coords):
    mol = mol_from_coords(start_coords)
    mf = make_mf_d2_gr(mol, solvent='water', grid_response=True)
    mf.grids.level = GRID_LEVEL
    steps = []

    def callback(envs):
        m = envs['mol']
        grad = np.asarray(envs['gradients'], float).reshape(6, 3)
        ss = mf.scf_summary
        rec = dict(
            step=int(envs['cycle']) + 1,
            coords_angstrom=np.asarray(m.atom_coords(unit='Angstrom'),
                                       float).tolist(),
            e_total=float(envs['energy']),
            gradient_full_6x3=grad.tolist(),
            grad_max=float(np.abs(grad).max()),
            grad_rms=float(np.sqrt((grad ** 2).mean())),
            scf_converged=bool(envs['g_scanner'].converged),
            e_solvent=(float(ss['e_solvent']) if 'e_solvent' in ss
                       else None),
            e_cds=(float(ss['e_cds']) if 'e_cds' in ss else None),
            e_d2_analytic=float(d2_full.d2_energy(m)),
            d2_in_scf_summary=float(ss.get('d2_dispersion', float('nan'))),
            grid_level_actual=GRID_LEVEL,
            grid_response_actual=bool(getattr(envs['g_scanner'],
                                              'grid_response')),
            solvent='SMD(water)')
        if not np.isfinite(rec['e_total']) or not np.isfinite(grad).all():
            raise RuntimeError('step %d: non-finite energy/gradient'
                               % rec['step'])
        steps.append(rec)
        with open(os.path.join(OUT, 'smd_traj_%s.json' % tid), 'w') as fh:
            json.dump(steps, fh, indent=2)
        print('[%s-SMD] step %2d  E=%.9f  max|g|=%.3e  e_solv=%s'
              % (tid, rec['step'], rec['e_total'], rec['grad_max'],
                 ('%.3e' % rec['e_solvent'])
                 if rec['e_solvent'] is not None else 'n/a'), flush=True)

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
        mf_v = make_mf_d2_gr(mol_v, solvent='water', grid_response=True)
        mf_v.grids.level = GRID_LEVEL
        mf_v.kernel()
        g_v = np.asarray(mf_v.nuc_grad_method().kernel(), float).reshape(6, 3)
        ss = mf_v.scf_summary
        verify = dict(
            e_total=float(mf_v.e_tot),
            gradient_full_6x3=g_v.tolist(),
            grad_max=float(np.abs(g_v).max()),
            grad_rms=float(np.sqrt((g_v ** 2).mean())),
            scf_converged=bool(mf_v.converged),
            finite=bool(np.isfinite(mf_v.e_tot) and np.isfinite(g_v).all()),
            e_solvent=(float(ss['e_solvent']) if 'e_solvent' in ss else None),
            e_cds=(float(ss['e_cds']) if 'e_cds' in ss else None),
            e_d2_analytic=float(d2_full.d2_energy(mol_v)),
            d2_in_scf_summary=float(ss.get('d2_dispersion', float('nan'))),
            grid_level_actual=GRID_LEVEL,
            pass_gmax=bool(np.abs(g_v).max() <= GMAX_TARGET),
            dE_vs_last_step=float(mf_v.e_tot - steps[-1]['e_total'])
            if steps else None)

    return dict(
        candidate=tid, grid_level=GRID_LEVEL, solvent='SMD(water)',
        start_source='gas_phase_registry.json primary_reference',
        start_coords_angstrom=start_coords.tolist(),
        maxsteps=MAXSTEPS, one_run_only=True,
        conv_params=dict(gradientmax=1e-6, gradientrms=1e-6,
                         stepmax='default 1.8e-3', steprms='default 1.2e-3'),
        method_fingerprint=method_fingerprint(mf),
        n_steps=len(steps),
        optimizer_converged=bool(opt_conv),
        optimizer_error=opt_error,
        independent_verification=verify,
        steps=steps,
        endpoint_coords_angstrom=(
            np.asarray(mol_opt.atom_coords(unit='Angstrom'), float).tolist()
            if mol_opt is not None else None),
        seconds=round(time.time() - t0, 1))


def main():
    reg = json.load(open(os.path.join(OUT, 'gas_phase_registry.json')))
    results = {}
    for tid in ('c14_plus', 'c06_plus'):
        start = np.asarray(
            reg['structures'][tid]['primary_reference']['coords_angstrom'],
            float)
        print('=== Phase C: %s (SMD/L8) ===' % tid, flush=True)
        results[tid] = optimize(tid, start)
        v = results[tid]['independent_verification']
        ok = bool(results[tid]['optimizer_converged'] and v
                  and v['pass_gmax'] and v['scf_converged'] and v['finite'])
        results[tid]['verdict'] = ('converged_stationary_candidate_SMD'
                                   if ok else
                                   ('maxsteps_reached_not_converged'
                                    if results[tid]['n_steps'] >= MAXSTEPS
                                    else 'stopped_early_not_converged'))
        print('[%s] verdict=%s  n_steps=%d  indep max|g|=%s'
              % (tid, results[tid]['verdict'], results[tid]['n_steps'],
                 ('%.3e' % v['grad_max']) if v else 'n/a'), flush=True)
        with open(os.path.join(OUT, 'phaseC_smd_optimize.json'), 'w') as fh:
            json.dump(results, fh, indent=2)
    print('=== PHASE C DONE ===')


if __name__ == '__main__':
    main()
