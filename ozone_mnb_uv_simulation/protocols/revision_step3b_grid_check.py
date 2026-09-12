#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Acceptance-revision step #3b (JOB-2026-0905-009): grid / SCF convergence
check of the analytic full-derivative Hessian for the flat lowest modes.

Motivation: the step-3 cross-check found that the sign of the LOWEST mode
flips between the analytic Hessian and an independent finite-difference
Hessian (c14: n_imag 1 vs 0; c14_plus / c12_plus: 0 vs 1), with relative
Hessian differences ~3e-3.  These structures sit on a very flat PES
(lowest |nu| ~ 50-80 cm-1, curvature near zero), so tiny curvature errors
flip the mode sign.  Suspects:
  (a) PySCF grid pruning is discontinuous under atomic displacement -> the
      small-step FD Hessian is polluted by gradient noise ~1e-5/(2h) ~ 1e-3;
  (b) the analytic DFT Hessian's own grid/SCF accuracy.

Per the revision brief, negative modes are NEVER deleted or dismissed as
noise; instead the analytic Hessian is recomputed at a finer grid (level 6)
and tighter SCF (1e-12) and compared against the level-5 result, and the FD
Hessian is repeated with a LARGER step (0.05 Bohr) where pruning noise is
suppressed.  The verdict records whether n_imag and the lowest modes are
stable across all constructions.
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
from pyscf import gto

ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, 'run_artifacts', '01_pure_water_o3_h2o')
REV = os.path.join(ART, 'acceptance_revision')
os.makedirs(REV, exist_ok=True)

SYMS = ['O', 'O', 'O', 'O', 'H', 'H']
TARGETS = ['c14_plus', 'c12_plus', 'c14']
OUT = os.path.join(REV, 'revision_grid_check.json')


def freqs_of(mol, h, e_total):
    ha = rb.pyscf_thermo.harmonic_analysis(mol, h, imaginary_freq=False)
    modes = np.sort(np.asarray(ha['freq_wavenumber'], dtype=float))
    return dict(n_imag=int(np.sum(modes < -1e-6)),
                freq_error=int(ha.get('freq_error', -1)),
                lowest6=[round(float(f), 2) for f in modes[:6]])


def analytic_at(mol, grid_level, scf_conv):
    mf = d2_full.make_mf_d2(mol)
    mf.grids.level = grid_level
    mf.conv_tol = scf_conv
    mf.kernel()
    e = float(mf.e_tot)
    h_raw = mf.Hessian().kernel()
    asym = float(np.abs(h_raw - h_raw.transpose(1, 0, 3, 2)).max())
    h = 0.5 * (h_raw + h_raw.transpose(1, 0, 3, 2))
    return e, asym, h


def run_target(tid):
    t0 = time.time()
    coords = json.load(open(os.path.join(
        ART, 'gas_freq', '%s.json' % tid)))['geometry_angstrom']
    mol = cu.mol_from_coords(coords, SYMS, verbose=0)
    res = dict(id=tid, variants={})

    # level-5 baseline is already in revision_hessian_crosscheck.json; recompute
    # the finer-grid and tighter-SCF variants here
    e6, asym6, h6 = analytic_at(mol, 6, 1e-12)
    res['variants']['analytic_grid6_scf1e-12'] = dict(
        e_total=e6, pre_symmetry_asymmetry=asym6,
        projected=freqs_of(mol, h6, e6))
    print('[%s] grid6: n_imag=%d lowest=%s  (%.1f s)'
          % (tid, res['variants']['analytic_grid6_scf1e-12']['projected']['n_imag'],
             res['variants']['analytic_grid6_scf1e-12']['projected']['lowest6'][:3],
             time.time() - t0), flush=True)

    # large-step FD (0.05 Bohr): pruning noise suppressed
    t1 = time.time()
    c0 = mol.atom_coords(unit='Bohr')
    n = mol.natm
    g = np.zeros((n, n, 3, 3))
    step = 0.05
    for i in range(n):
        for k in range(3):
            for s in (+1, -1):
                c = c0.copy()
                c[i, k] += s * step
                m = mol.copy()
                m.set_geom_(c, unit='Bohr')
                m.build()
                mf = d2_full.make_mf_d2(m)
                mf.kernel()
                g[i, :, k] += s * mf.nuc_grad_method().kernel().reshape(n, 3) / (2 * step)
    h_fd = 0.5 * (g + g.transpose(1, 0, 3, 2))
    res['variants']['fd_step0.05'] = dict(
        projected=freqs_of(mol, h_fd, e6),
        max_abs_H_diff_vs_grid6=float(np.abs(h6 - h_fd).max()),
        relative_H_diff=float(np.abs(h6 - h_fd).max() / np.abs(h6).max()))
    print('[%s] fd0.05: n_imag=%d lowest=%s  max|dH|_grid6=%.2e  (%.1f s)'
          % (tid, res['variants']['fd_step0.05']['projected']['n_imag'],
             res['variants']['fd_step0.05']['projected']['lowest6'][:3],
             res['variants']['fd_step0.05']['max_abs_H_diff_vs_grid6'],
             time.time() - t1), flush=True)

    res['seconds'] = round(time.time() - t0, 1)
    return res


def _init():
    import pyscf.lib as lib
    lib.num_threads(8)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default=None)
    args = ap.parse_args()
    targets = args.only.split(',') if args.only else TARGETS

    done = {}
    if os.path.exists(OUT):
        with open(OUT) as fh:
            done = {r['id']: r for r in json.load(fh).get('targets', [])}
    todo = [t for t in targets if t not in done]
    results = list(done.values())
    if todo:
        with Pool(len(todo), initializer=_init) as pool:
            results.extend(pool.map(run_target, todo))
    results.sort(key=lambda r: TARGETS.index(r['id']))

    verdict = {}
    for r in results:
        nims = [r['variants'][v]['projected']['n_imag'] for v in r['variants']]
        r['n_imag_stable_across_variants'] = bool(len(set(nims)) == 1)
        verdict[r['id']] = dict(variants=list(r['variants']),
                                n_imag_values=nims,
                                stable=bool(len(set(nims)) == 1))
    with open(OUT, 'w') as fh:
        json.dump(dict(job='JOB-2026-0905-009',
                       step='revision_grid_check',
                       note=('imaginary modes are never removed; stability is the '
                             'criterion.  A structure whose n_imag flips between '
                             'constructions is marked UNSTABLE and its minimum '
                             'status stays 待定 (pending).'),
                       targets=results, verdict=verdict), fh, indent=2)
    print('SAVED ->', OUT)
    for k, v in verdict.items():
        print('GRID-VERDICT %s: n_imag across variants=%s -> %s'
              % (k, v['n_imag_values'], 'STABLE' if v['stable'] else 'UNSTABLE'))


if __name__ == '__main__':
    main()
