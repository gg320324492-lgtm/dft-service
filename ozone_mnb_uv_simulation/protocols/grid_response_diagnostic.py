#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Grid-response paired diagnostic (JOB-2026-0905-009).

Hypothesis under test: the systematic k_E (energy-layer) vs k_g (gradient-layer)
curvature gap along the disputed internal modes is caused by the DFT numerical-
integration grid response that the DEFAULT analytic gradient neglects
(Gradients.grid_response = False).

Design (per the batch brief):
  * level 7, SCF conv_tol=1e-12 / conv_tol_grad=1e-9, full-derivative -D2;
  * per candidate (c14_plus, c06_plus): central + 0.002 A + 0.005 A, both
    signs = 5 geometry points (same coordinates/directions as the previous
    directional batch);
  * each point: ONE converged SCF, then TWO independent Gradients objects
    built from the same converged density -- grid_response=False and True;
    everything else identical; both gradients contain the project -D2 exactly
    once (D2 is grid-independent and added on top of the parent kernel);
  * recorded per point: SCF total energy, both gradients (full), both
    grid_response actual values, SCF state, <S^2>, grid point counts, version,
    D2 component.

Analysis (in the report script): s_g(False)/s_g(True) vs s_E; k_g(False)/
k_g(True) vs k_E at both amplitudes; does d.[g(True)-g(False)] explain the
gap; energy invariance under the switch (control that only one factor changed).
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
BOHR_A = 0.52917721092
import d2_full
import conformer_utils as cu

ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, 'run_artifacts', '01_pure_water_o3_h2o')
FV = os.path.join(ART, 'final_minima_validation')
IS = os.path.join(ART, 'internal_subspace_revision')
GR = os.path.join(ART, 'grid_response_diagnostic')
os.makedirs(GR, exist_ok=True)

SYMS = ['O', 'O', 'O', 'O', 'H', 'H']
MASSES = np.array([15.999, 15.999, 15.999, 15.999, 1.008, 1.008])
GRID_LEVEL = 7
SCF_TOL = 1e-12
SCF_TOL_GRAD = 1e-9
AMPS_A = [0.002, 0.005]
DISPUTED = {'c14_plus': 'analytic', 'c06_plus': 'fd'}
N_THREADS = 8
BOHR_A = 0.52917721092


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:16]


def make_mf(mol):
    mf = d2_full.make_mf_d2(mol)
    mf.grids.level = GRID_LEVEL
    mf.conv_tol = SCF_TOL
    mf.conv_tol_grad = SCF_TOL_GRAD
    return mf


def run_point(task):
    tid, kind, amp_a, sgn, coords = (task['candidate'], task['kind'],
                                     task['amp_a'], task['sign'],
                                     task['coords'])
    key = '%s_%s' % (tid, kind if kind == 'central'
                     else 'amp%g_%+d' % (amp_a, sgn))
    path = os.path.join(GR, 'point_%s.json' % key)
    if os.path.exists(path):
        print('[skip]', key, flush=True)
        with open(path) as fh:
            return json.load(fh)
    import pyscf
    from pyscf import gto
    t0 = time.time()
    s = "; ".join("%s %.10f %.10f %.10f" % (s_, x, y, z)
                  for s_, (x, y, z) in zip(SYMS, coords))
    mol = gto.M(atom=s, basis='def2-TZVP', charge=0, spin=0, verbose=0,
                max_memory=4000)
    mf = make_mf(mol)
    mf.kernel()
    e_scf = float(mf.e_tot)
    e_d2 = d2_full.d2_energy(mol)
    converged = bool(mf.converged)
    n_grid = int(mf.grids.weights.size)
    try:
        ss, _ = mf.spin_square()
        s2 = float(ss)
    except Exception:
        s2 = 0.0

    grads = {}
    for flag in (False, True):
        gobj = mf.nuc_grad_method()
        gobj.grid_response = flag            # set on the ACTUAL Gradients object
        g = gobj.kernel()
        grads['grid_response_%s' % flag] = dict(
            grid_response_actual=bool(getattr(gobj, 'grid_response')),
            gradient=np.asarray(g).reshape(-1).tolist(),
            max_abs_grad=float(np.abs(g).max()))
        # the SCF energy must be untouched by the gradient switch
        assert abs(float(mf.e_tot) - e_scf) < 1e-12, 'energy changed by switch!'

    rec = dict(key=key, candidate=tid, kind=kind, amp_a=amp_a, sign=sgn,
               coords_angstrom=coords,
               e_scf_total=e_scf, e_d2_hartree=float(e_d2),
               e_dft_part_hartree=e_scf - float(e_d2),
               scf_converged=converged, spin_s2=s2,
               n_grid_points=n_grid, grid_level=GRID_LEVEL,
               scf_tol=SCF_TOL, scf_tol_grad=SCF_TOL_GRAD,
               pyscf_version=pyscf.__version__,
               grads=grads, seconds=round(time.time() - t0, 1))
    with open(path, 'w') as fh:
        json.dump(rec, fh, indent=2)
    print('[%s] E=%.8f  max|g| F=%.2e T=%.2e  d|g|=%.2e  ngrid=%d (%.0f s)'
          % (key, e_scf, grads['grid_response_False']['max_abs_grad'],
             grads['grid_response_True']['max_abs_grad'],
             abs(grads['grid_response_True']['max_abs_grad'] -
                 grads['grid_response_False']['max_abs_grad']),
             n_grid, rec['seconds']), flush=True)
    return rec


def _init():
    import pyscf.lib as lib
    lib.num_threads(N_THREADS)


def main():
    isx = load(os.path.join(IS, 'internal_subspace_analysis.json'), {})
    tasks = []
    for tid, construction in DISPUTED.items():
        stepc = load(os.path.join(FV, 'stepC_reopt_%s.json' % tid), {})
        coords0 = np.asarray(stepc['new_geometry'], float)
        d_cart = np.asarray(isx['candidate_reanalysis'][tid][construction]
                            ['lowest_internal_direction_cart'], float)
        d3 = d_cart.reshape(6, 3)
        # record the direction metadata for provenance
        tasks.append(dict(candidate=tid, kind='central', amp_a=None, sign=0,
                          coords=coords0.tolist(), d_cart=d_cart.tolist(),
                          construction=construction,
                          direction_mw=isx['candidate_reanalysis'][tid]
                          [construction]['lowest_internal_direction_mw']))
        for a in AMPS_A:
            q = cal.q_for_atom_disp(d3, a)
            for sgn in (+1, -1):
                c = cal.displaced_geometry(
                    coords0, d_cart.reshape(6, 3), sgn * q,
                    unit='Angstrom')
                tasks.append(dict(candidate=tid, kind='displaced', amp_a=a,
                                  sign=sgn, coords=c.tolist(),
                                  d_cart=d_cart.tolist(),
                                  construction=construction,
                                  direction_mw=isx['candidate_reanalysis'][tid]
                                  [construction]['lowest_internal_direction_mw']))
    # task-list provenance
    tl = dict(job='JOB-2026-0905-009',
              step='grid_response_diagnostic_task_list',
              settings=dict(grid_level=GRID_LEVEL, scf_tol=SCF_TOL,
                            scf_tol_grad=SCF_TOL_GRAD, amplitudes_a=AMPS_A,
                            unit_fix='2026-09-06: q*BOHR_A*d applied to Angstrom coords (unit defect fixed, JOB-2026-0906-002)'
                            note='level 7 points reuse the previous directional '
                                 'batch geometries/directions/settings; density '
                                 'was NOT persisted there -> full SCF recompute '
                                 'of the 10 points'),
              hashes=dict(
                  internal_subspace_analysis=sha256(os.path.join(
                      IS, 'internal_subspace_analysis.json')),
                  **{'stepC_reopt_%s' % t: sha256(os.path.join(
                      FV, 'stepC_reopt_%s.json' % t)) for t in DISPUTED},
                  script=sha256(os.path.abspath(__file__))),
              verification={}, tasks=tasks)
    # verify directions (norm / external orthogonality / consistency)
    for t in tasks:
        if t['kind'] != 'central':
            continue
        d = np.asarray(t['d_cart'], float)
        v_mw = np.repeat(np.sqrt(MASSES), 3) * d
        V = cal.tr_subspace(MASSES, np.asarray(t['coords'], float) / BOHR_A)
        persisted = np.asarray(t['direction_mw'], float)
        dcart_persist = np.asarray(
            isx['candidate_reanalysis'][t['candidate']][t['construction']]
            ['lowest_internal_direction_cart'], float)
        tl['verification'][t['candidate']] = dict(
            norm_1=float(np.linalg.norm(d)),
            norm_pass=bool(abs(np.linalg.norm(d) - 1) < 1e-12),
            mw_external_overlap_max=float(np.abs(V.T @ v_mw).max()),
            orthogonality_pass=bool(np.abs(V.T @ v_mw).max() < 1e-10),
            consistent_with_persisted=bool(
                np.abs(d / np.linalg.norm(d) -
                       dcart_persist / np.linalg.norm(dcart_persist)).max() < 1e-10),
            lam_mw=float(isx['candidate_reanalysis'][t['candidate']]
                         [t['construction']]['lowest_internal_eigenvalue']))
    tl_path = os.path.join(GR, 'grid_response_task_list.json')
    with open(tl_path, 'w') as fh:
        json.dump(tl, fh, indent=2, ensure_ascii=False)
    print('SAVED ->', tl_path, flush=True)
    for t, v in tl['verification'].items():
        print('VERIFY %s: norm=%.4f orth=%.1e consistent=%s (nu=%.1f cm-1)'
              % (t, v['norm_1'], v['mw_external_overlap_max'],
                 v['consistent_with_persisted'],
                 np.sign(v['lam_mw']) * np.sqrt(abs(v['lam_mw'])) * 5139.5),
              flush=True)

    done = {}
    results = dict(done)
    for t in tasks:
        k = '%s_%s' % (t['candidate'], t['kind'] if t['kind'] == 'central'
                       else 'amp%g_%+d' % (t['amp_a'], t['sign']))
        p = os.path.join(GR, 'point_%s.json' % k)
        if os.path.exists(p):
            done[k] = json.load(open(p))
    todo = [t for t in tasks if ('%s_%s' % (t['candidate'],
            t['kind'] if t['kind'] == 'central'
            else 'amp%g_%+d' % (t['amp_a'], t['sign']))) not in done]
    print('points to compute: %d (of %d)' % (len(todo), len(tasks)), flush=True)
    results = dict(done)
    if todo:
        with Pool(min(4, len(todo)), initializer=_init) as pool:
            new = pool.map(run_point, todo)
        for r in new:
            results[r['key']] = r
    with open(os.path.join(GR, 'grid_response_points.json'), 'w') as fh:
        json.dump(dict(settings=tl['settings'], verification=tl['verification'],
                       hashes=tl['hashes'], points=list(results.values())),
                  fh, indent=2)
    print('SAVED grid_response_points.json (%d points)' % len(results),
          flush=True)


def load(p, default=None):
    if not os.path.exists(p):
        return default
    with open(p) as fh:
        return json.load(fh)


if __name__ == '__main__':
    main()
