#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase C: bounded (maxsteps=60) gas-phase endpoint re-optimisation for the
c14_plus and c06_plus candidates (JOB-2026-0906-003).

Method: wB97X-D (libxc DFT part) + project full-derivative -D2, def2-TZVP,
grid level 7, SCF conv_tol=1e-12 / conv_tol_grad=1e-9, gas phase.
Gradient path: grad_factory.make_mf_d2_gr -> grid_response=True AND project D2
exactly once (verified in Phase B, scanner_verify_c14_plus_L7_center.json).

Optimizer: pyscf.geomopt.berny_solver.kernel (pyberny 0.7.0).  CONFIRMED
convergence parameters (BernyParams, atomic units, ALL criteria AND-ed):
    gradientmax  (Eh/Bohr)   default 0.45e-3  -> set 1e-5 (acceptance target)
    gradientrms  (Eh/Bohr)   default 0.15e-3  -> set 1e-5
    stepmax / steprms (internal-coord step, rad for angles) -> defaults kept
There is NO energy criterion in pyberny 0.7.0; convergence therefore means
ALL gradient+step criteria hold simultaneously (not a single threshold).
Step limit: maxsteps=60, no restart/relaxation on non-convergence.

Final verification: INDEPENDENT freshly built mol+mf+gradient object at the
endpoint must reach max|g| <= 1e-5 Eh/Bohr.
"""
import os, sys, json, time, traceback
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pyscf import gto
from pyscf.geomopt import berny_solver
import d2_full
from grad_factory import make_mf_d2_gr, method_fingerprint

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'run_artifacts', '01_pure_water_o3_h2o',
                   'phaseC_reoptimization')
RES = os.path.join(HERE, '..', 'results', '01_water_matrices',
                   'pure_water_o3_h2o', 'phaseC_reoptimization')
SYMS = ['O', 'O', 'O', 'O', 'H', 'H']
MAXSTEPS = 60
GMAX_TARGET = 1e-5           # Eh/Bohr CARTESIAN acceptance (unchanged)
DC_DIR = os.path.join(HERE, '..', 'run_artifacts', '01_pure_water_o3_h2o',
                      'directional_confirmation')

# Attempt 2 parameter correction (JOB-2026-0906-003):
# pyberny 0.7.0 evaluates gradientmax/gradientrms on the INTERNAL-coordinate
# gradient dot(B_inv.T, g) (berny.py L294), not the Cartesian gradient, so
# attempt 1 (gradientmax=1e-5) converged by internal criteria while the
# Cartesian acceptance max|g|<=1e-5 was NOT met (3.13e-5 / 1.38e-5).
# Attempt 2 TIGHTENS the internal thresholds 10x (1e-6) from the ORIGINAL
# start geometries; the Cartesian acceptance and maxsteps=60 are unchanged.
# No threshold relaxation, no continuation/append, no extra attempts after
# this one regardless of outcome.
import sys as _sys
ATTEMPT = int(_sys.argv[1]) if len(_sys.argv) > 1 else 2
CONV_PARAMS = dict(gradientmax=1e-6, gradientrms=1e-6) if ATTEMPT >= 2 \
    else dict(gradientmax=GMAX_TARGET, gradientrms=GMAX_TARGET)


def mol_from_coords(coords_A):
    atom = "; ".join("%s %.10f %.10f %.10f" % (s, x, y, z)
                     for s, (x, y, z) in zip(SYMS, coords_A))
    return gto.M(atom=atom, basis='def2-TZVP', charge=0, spin=0,
                 verbose=0, max_memory=4000)


def load_center(tid):
    p = os.path.join(DC_DIR, 'point_%s_L7_central.json' % tid)
    rec = json.load(open(p))
    return np.asarray(rec['coords_angstrom'], float)


def optimize_candidate(tid, t_start):
    coords0 = load_center(tid)
    mol = mol_from_coords(coords0)
    mf = make_mf_d2_gr(mol, solvent=None, grid_response=True)

    steps = []

    def callback(envs):
        cycle = envs['cycle']
        energy = float(envs['energy'])
        grad = np.asarray(envs['gradients'], float).reshape(-1)
        gs = envs['g_scanner']
        m = envs['mol']
        e_d2 = float(d2_full.d2_energy(m))
        rec = dict(
            step=int(cycle) + 1,
            coords_angstrom=np.asarray(m.atom_coords(unit='Angstrom'),
                                       float).tolist(),
            e_total=energy,
            e_d2_analytic=e_d2,
            e_dft_part=energy - e_d2,
            d2_in_scf_summary=float(
                (envs['method'].scf_summary.get('d2_dispersion')
                 if hasattr(envs.get('method', None), 'scf_summary')
                 else mf.scf_summary.get('d2_dispersion')) or np.nan),
            grad_max=float(np.abs(grad).max()),
            grad_rms=float(np.sqrt((grad ** 2).mean())),
            scf_converged=bool(gs.converged),
        )
        # hard stop on non-finite values (anomalous geometry / SCF blow-up)
        if not np.isfinite(energy) or not np.isfinite(grad).all():
            raise RuntimeError('step %d: non-finite energy/gradient' % rec['step'])
        steps.append(rec)
        print('[%s] step %2d  E=%.9f  max|g|=%.3e  rms|g|=%.3e  conv=%s'
              % (tid, rec['step'], energy, rec['grad_max'], rec['grad_rms'],
                 rec['scf_converged']), flush=True)

    t0 = time.time()
    try:
        opt_conv, mol_opt = berny_solver.kernel(
            mf, assert_convergence=True, include_ghost=True,
            callback=callback, maxsteps=MAXSTEPS, **CONV_PARAMS)
        opt_error = None
    except Exception as e:                                # keep the site
        traceback.print_exc()
        opt_conv, mol_opt, opt_error = False, None, repr(e)
        # endpoint = last recorded step coords (site preserved)
        mol_opt = mol_from_coords(np.asarray(steps[-1]['coords_angstrom'],
                                             float)) if steps else None

    # ---- independent endpoint verification (fresh objects) ----
    verify = None
    if mol_opt is not None:
        c_end = np.asarray(mol_opt.atom_coords(unit='Angstrom'), float)
        mol_v = mol_from_coords(c_end)
        mf_v = make_mf_d2_gr(mol_v, solvent=None, grid_response=True)
        mf_v.kernel()
        g_v = mf_v.nuc_grad_method().kernel()
        g_v = np.asarray(g_v, float).reshape(-1)
        e_d2_v = float(d2_full.d2_energy(mol_v))
        verify = dict(
            e_total=float(mf_v.e_tot),
            e_d2_analytic=e_d2_v,
            e_dft_part=float(mf_v.e_tot) - e_d2_v,
            d2_in_scf_summary=float(mf_v.scf_summary.get('d2_dispersion')),
            grad_max=float(np.abs(g_v).max()),
            grad_rms=float(np.sqrt((g_v ** 2).mean())),
            scf_converged=bool(mf_v.converged),
            pass_gmax=bool(np.abs(g_v).max() <= GMAX_TARGET),
        )
        if steps:
            verify['dE_vs_last_step'] = (verify['e_total']
                                         - steps[-1]['e_total'])
    return dict(
        candidate=tid,
        start_coords_angstrom=coords0.tolist(),
        n_steps=len(steps),
        optimizer_converged=bool(opt_conv),
        optimizer_error=opt_error,
        maxsteps=MAXSTEPS,
        conv_params=CONV_PARAMS,
        conv_params_units=dict(gradientmax='Eh/Bohr', gradientrms='Eh/Bohr',
                               stepmax='internal-coord (rad for angles)',
                               steprms='internal-coord (rad for angles)',
                               note='pyberny 0.7.0 BernyParams; ALL criteria '
                                    'AND-ed; NO energy criterion'),
        final_criteria_last_step=dict(
            grad_max=steps[-1]['grad_max'] if steps else None,
            grad_rms=steps[-1]['grad_rms'] if steps else None),
        independent_verification=verify,
        steps=steps,
        endpoint_coords_angstrom=(
            np.asarray(mol_opt.atom_coords(unit='Angstrom'), float).tolist()
            if mol_opt is not None else None),
        seconds=round(time.time() - t0, 1))


def rmsd_dedup(res):
    """Crude endpoint dedup check (same atom order, centroid-aligned)."""
    ends = [(r['candidate'], r['endpoint_coords_angstrom'])
            for r in res if r.get('endpoint_coords_angstrom')]
    if len(ends) < 2:
        return None
    (ta, ca), (tb, cb) = ends
    ca, cb = np.asarray(ca), np.asarray(cb)
    ca = ca - ca.mean(axis=0)
    cb = cb - cb.mean(axis=0)
    rmsd = float(np.sqrt(((ca - cb) ** 2).sum(axis=1).mean()))
    # intermolecular distance O3-centroid <-> H2O-centroid for context
    def gap(c):
        return float(np.linalg.norm(c[:3].mean(axis=0) - c[3:].mean(axis=0)))
    return dict(candidate_a=ta, candidate_b=tb, rmsd_A=rmsd,
                same_structure=bool(rmsd < 0.1),
                gap_a_A=gap(ca), gap_b_A=gap(cb),
                note='no rotation/permutation registration; same atom order '
                     'SYMS=O3+H2O; identical only if RMSD ~ 0')


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(RES, exist_ok=True)
    results = []
    suffix = '' if ATTEMPT <= 1 else '_attempt%d' % ATTEMPT
    for tid in ('c14_plus', 'c06_plus'):
        print('=== Phase C attempt %d: %s ===' % (ATTEMPT, tid), flush=True)
        results.append(optimize_candidate(tid, time.time()))
        # persist incrementally so a crash preserves the finished candidate
        with open(os.path.join(OUT, 'phaseC_steps_%s%s.json'
                               % (tid, suffix)), 'w') as fh:
            json.dump(results[-1], fh, indent=2)

    dedup = rmsd_dedup(results)
    summary = dict(
        job='JOB-2026-0906-003 Phase C',
        method_fingerprint=None,          # filled below
        results=[{k: v for k, v in r.items() if k != 'steps'} for r in results],
        trajectories={r['candidate']: r['steps'] for r in results},
        endpoint_dedup=dedup,
        verdicts={r['candidate']: dict(
            optimizer_converged=r['optimizer_converged'],
            n_steps=r['n_steps'],
            hit_maxsteps=bool(r['n_steps'] >= MAXSTEPS),
            independent_max_g=(r['independent_verification']['grad_max']
                               if r['independent_verification'] else None),
            pass_gmax=(r['independent_verification']['pass_gmax']
                       if r['independent_verification'] else False),
            status=('converged_stationary_candidate'
                    if r['optimizer_converged']
                    and r['independent_verification']
                    and r['independent_verification']['pass_gmax']
                    else ('maxsteps_reached_not_converged'
                          if r['n_steps'] >= MAXSTEPS
                          else 'optimizer_converged_cartesian_acceptance_failed')))
            for r in results},
    )
    # method fingerprint from a fresh mf at the c14 start geometry
    mol0 = mol_from_coords(np.asarray(results[0]['start_coords_angstrom'],
                                      float))
    mf0 = make_mf_d2_gr(mol0, solvent=None, grid_response=True)
    summary['method_fingerprint'] = method_fingerprint(mf0)
    summary['attempt'] = ATTEMPT
    summary['attempt_note'] = (
        'attempt 1: gradientmax=gradientrms=1e-5 -> optimizer converged by '
        'INTERNAL-coordinate criteria but Cartesian acceptance failed '
        '(see phaseC_summary_attempt1.json); '
        'attempt 2: internal thresholds tightened to 1e-6 (NO relaxation), '
        'same Cartesian acceptance max|g|<=1e-5, same maxsteps=60, '
        'fresh start from original geometries'
    ) if ATTEMPT >= 2 else 'attempt 1 (internal thresholds = Cartesian target)'

    suffix = '' if ATTEMPT <= 1 else '_attempt%d' % ATTEMPT
    with open(os.path.join(OUT, 'phaseC_summary%s.json' % suffix), 'w') as fh:
        json.dump(summary, fh, indent=2)
    print('\n=== PHASE C DONE ===')
    for tid, v in summary['verdicts'].items():
        print('  %-9s status=%s  n_steps=%d  indep max|g|=%s'
              % (tid, v['status'], v['n_steps'],
                 ('%.3e' % v['independent_max_g'])
                 if v['independent_max_g'] is not None else 'n/a'))
    if dedup:
        print('  endpoint dedup: RMSD=%.4f A  same=%s' % (dedup['rmsd_A'],
                                                          dedup['same_structure']))


if __name__ == '__main__':
    main()
