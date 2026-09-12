#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Directional confirmation batch (JOB-2026-0905-009): targeted finite-size
probes along the two VALID disputed internal modes, paired at grid levels 6
and 7.

Per the batch brief:
  * read the VERIFIED coordinates (stepC re-optimised endpoints) and the
    VERIFIED disputed directions (internal_subspace_revision analysis);
  * verify each direction: unit 3N norm, mass-weighted vector orthogonal to
    the external (TR) subspace, identical to the persisted vector;
  * per candidate: central geometry + 6 displaced points (max per-atom
    displacement-VECTOR length 0.002 / 0.005 / 0.01 A, both signs) = 7 points;
  * two grids (level 6 / level 7), everything else identical: same functional,
    basis, full-derivative -D2, pruning, SCF conv_tol=1e-12 /
    conv_tol_grad=1e-9, electronic state;
  * per point: total energy + DFT/D2 components, FULL nuclear gradient,
    SCF convergence, <S^2>, actual grid point count, PySCF version;
  * central point recomputes max(abs(g)) -- fills the missing gradient
    evidence;
  * non-converged points are recorded but excluded from curvature fitting.

Raw per-point records: run_artifacts/01_pure_water_o3_h2o/directional_confirmation/
Resume-safe per point.  Report assembly is in
protocols/directional_confirmation_report.py.
"""
import os
import sys
import json
import time
import hashlib
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
IS = os.path.join(ART, 'internal_subspace_revision')
DC = os.path.join(ART, 'directional_confirmation')
os.makedirs(DC, exist_ok=True)

SYMS = ['O', 'O', 'O', 'O', 'H', 'H']
MASSES = np.array([15.999, 15.999, 15.999, 15.999, 1.008, 1.008])
BOHR_A = 0.52917721092
AMPS_A = [0.002, 0.005, 0.01]
GRID_LEVELS = [6, 7]
SCF_TOL = 1e-12
SCF_TOL_GRAD = 1e-9
N_THREADS = 8

# disputed direction per candidate: (construction, direction id)
DISPUTED = {'c14_plus': 'analytic', 'c06_plus': 'fd'}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:16]


def load(p, default=None):
    if not os.path.exists(p):
        return default
    with open(p) as fh:
        return json.load(fh)


def build_task_list():
    """Stage 0: read verified coordinates/directions, verify, hash, save."""
    isx = load(os.path.join(IS, 'internal_subspace_analysis.json'), {})
    tasks = []
    verification = {}
    for tid, construction in DISPUTED.items():
        stepc = load(os.path.join(FV, 'stepC_reopt_%s.json' % tid), {})
        coords = np.asarray(stepc['new_geometry'], float)
        rec = isx['candidate_reanalysis'][tid]
        d_cart = np.asarray(rec[construction]['lowest_internal_direction_cart'], float)
        # verification 1: unit 3N norm
        norm = float(np.linalg.norm(d_cart))
        # verification 2: mass-weighted vector orthogonal to external subspace
        v_mw = np.repeat(np.sqrt(MASSES), 3) * d_cart   # unnormalised mw vector
        V = cal.tr_subspace(MASSES, mol_coords_bohr(coords))
        mw_orth = float(np.abs(V.T @ v_mw).max())
        # verification 3: identical to the persisted verified direction
        persisted = np.asarray(rec[construction]['lowest_internal_direction_cart'],
                               float)
        consistent = bool(np.abs(persisted - d_cart).max() == 0.0)
        lam_mw = float(rec[construction]['lowest_internal_eigenvalue'])
        verification[tid] = dict(
            construction=construction, norm_1=norm, norm_pass=bool(abs(norm - 1) < 1e-12),
            mw_external_overlap_max=mw_orth,
            orthogonality_pass=bool(mw_orth < 1e-10),
            identical_to_persisted=consistent,
            lam_mw=lam_mw,
            nu_cm1=float(np.sign(lam_mw) * np.sqrt(abs(lam_mw))
                         * np.sqrt(4.3597447222071e-18 / (1.66053906660e-27
                                                          * 0.52917721092e-10 ** 2))
                         / (2 * np.pi * 2.99792458e8 * 100.0)))
        d3 = d_cart.reshape(6, 3)
        # central point per grid
        for gl in GRID_LEVELS:
            tasks.append(dict(candidate=tid, grid_level=gl, kind='central',
                              amp_a=None, sign=0, coords=coords.tolist(),
                              d_cart=d_cart.tolist()))
        # displaced points
        for a in AMPS_A:
            q = cal.q_for_atom_disp(d3, a)
            for sgn in (+1, -1):
                c = cal.displaced_geometry(
                    coords, d_cart.reshape(6, 3), sgn * q,
                    unit='Angstrom')
                for gl in GRID_LEVELS:
                    tasks.append(dict(candidate=tid, grid_level=gl, kind='displaced',
                                      amp_a=a, sign=sgn, coords=c.tolist(),
                                      d_cart=d_cart.tolist()))
    task_list = dict(
        job='JOB-2026-0905-009',
        step='directional_confirmation_task_list',
        settings=dict(grid_levels=GRID_LEVELS, scf_tol=SCF_TOL,
                      scf_tol_grad=SCF_TOL_GRAD,
                      amplitudes_a=AMPS_A,
                      amplitude_convention='max per-atom displacement-VECTOR length',
                      unit_fix='2026-09-06: q is a Bohr-convention amplitude; displaced coords in Angstrom now apply q*BOHR_A*d (was q*d — 1/b amplification defect, JOB-2026-0906-002)',
                      functional='wb97xd (libxc) + full-derivative -D2 (ghost n/a)',
                      basis='def2-TZVP'),
        verification=verification,
        n_tasks=len(tasks),
        sources=dict(
            coordinates='final_minima_validation/stepC_reopt_<id>.json new_geometry',
            directions='internal_subspace_revision/internal_subspace_analysis.json '
                       'candidate_reanalysis.<id>.<construction>.'
                       'lowest_internal_direction_cart',
            hashes=dict(
                internal_subspace_analysis=sha256(os.path.join(
                    IS, 'internal_subspace_analysis.json')),
                **{'stepC_reopt_%s' % t: sha256(os.path.join(
                    FV, 'stepC_reopt_%s.json' % t)) for t in DISPUTED},
                script=sha256(os.path.abspath(__file__)))),
        tasks=tasks)
    path = os.path.join(DC, 'directional_task_list.json')
    with open(path, 'w') as fh:
        json.dump(task_list, fh, indent=2)
    print('SAVED ->', path)
    for t, v in verification.items():
        print('VERIFY %s: norm_pass=%s orth_pass=%s consistent=%s (nu=%.1f cm-1)'
              % (t, v['norm_pass'], v['orthogonality_pass'],
                 v['identical_to_persisted'], v['nu_cm1']), flush=True)
    return task_list


def mol_coords_bohr(coords_a):
    return np.asarray(coords_a, float) / BOHR_A


def point_key(t):
    return '%s_L%d_%s' % (t['candidate'], t['grid_level'],
                          'central' if t['kind'] == 'central'
                          else 'amp%g_%+d' % (t['amp_a'], t['sign']))


def run_point(t):
    key = point_key(t)
    path = os.path.join(DC, 'point_%s.json' % key)
    if os.path.exists(path):
        print('[skip]', key, flush=True)
        with open(path) as fh:
            return json.load(fh)
    import pyscf
    from pyscf import gto
    t_start = time.time()
    coords = np.asarray(t['coords'], float)
    s = "; ".join("%s %.10f %.10f %.10f" % (s_, x, y, z)
                  for s_, (x, y, z) in zip(SYMS, coords))
    mol = gto.M(atom=s, basis='def2-TZVP', charge=0, spin=0, verbose=0,
                max_memory=4000)
    mf = d2_full.make_mf_d2(mol)
    mf.grids.level = t['grid_level']
    mf.conv_tol = SCF_TOL
    mf.conv_tol_grad = SCF_TOL_GRAD
    mf.kernel()
    e_tot = float(mf.e_tot)
    e_d2 = d2_full.d2_energy(mol)
    g = np.asarray(mf.nuc_grad_method().kernel()).reshape(-1)
    converged = bool(mf.converged)
    try:
        ss, _ = mf.spin_square()
        s2 = float(ss)
    except Exception:
        s2 = 0.0
    n_grid = int(mf.grids.weights.size)
    rec = dict(key=key, candidate=t['candidate'], grid_level=t['grid_level'],
               kind=t['kind'], amp_a=t['amp_a'], sign=t['sign'],
               coords_angstrom=t['coords'],
               e_total=e_tot, e_d2_hartree=float(e_d2),
               e_dft_part_hartree=e_tot - float(e_d2),
               gradient=g.tolist(), max_abs_grad=float(np.abs(g).max()),
               scf_converged=converged, spin_s2=s2,
               n_grid_points=n_grid, pyscf_version=pyscf.__version__,
               scf_tol=SCF_TOL,
               scf_tol_grad=SCF_TOL_GRAD,
               seconds=round(time.time() - t_start, 1))
    with open(path, 'w') as fh:
        json.dump(rec, fh, indent=2)
    print('[%s] L%d E=%.8f max|g|=%.2e conv=%s ngrid=%d (%.0f s)'
          % (key, t['grid_level'], e_tot, rec['max_abs_grad'], converged,
             n_grid, rec['seconds']), flush=True)
    return rec


def _init():
    import pyscf.lib as lib
    lib.num_threads(N_THREADS)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--report-only', action='store_true')
    args = ap.parse_args()

    task_list = build_task_list()
    tasks = task_list['tasks']
    if args.report_only:
        results = {point_key(t): load(os.path.join(DC, 'point_%s.json' % point_key(t)))
                   for t in tasks}
        results = {k: v for k, v in results.items() if v is not None}
    else:
        keys_done = set()
        for t in tasks:
            p = os.path.join(DC, 'point_%s.json' % point_key(t))
            if os.path.exists(p):
                keys_done.add(point_key(t))
        todo = [t for t in tasks if point_key(t) not in keys_done]
        results = [load(os.path.join(DC, 'point_%s.json' % point_key(t)))
                   for t in tasks if point_key(t) in keys_done]
        if todo:
            with Pool(min(4, len(todo)), initializer=_init) as pool:
                results.extend(pool.map(run_point, todo))
        results = {point_key(t): load(os.path.join(
            DC, 'point_%s.json' % point_key(t))) for t in tasks}
        results = {k: v for k, v in results.items() if v is not None}
    from directional_confirmation_report import assemble
    assemble(task_list, list(results.values()))


if __name__ == '__main__':
    main()
