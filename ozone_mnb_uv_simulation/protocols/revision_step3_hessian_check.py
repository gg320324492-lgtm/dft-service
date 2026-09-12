#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Acceptance-revision step #3 (JOB-2026-0905-009): frequency reliability
cross-check on c14_plus, c12_plus and c14.

Per the revision brief:
  1. provenance check -- coordinates, energies and gradients in the three
     record sets (gas_opt/, gas_freq/, mode_follow/) must refer to the same
     geometry and the same script/PySCF version;
  2. at the FIXED endpoint geometry, compare the full-derivative ANALYTIC
     Hessian against an INDEPENDENT central-difference Hessian of the total
     gradient (DFT + analytic -D2), at two step sizes (0.005 and 0.02 Bohr);
  3. record pre-symmetrisation asymmetry, projected frequencies, negative
     mode counts and the key (lowest) modes for every Hessian;
  4. imaginary modes are never deleted or dismissed as noise; if the
     frequencies are sensitive to step size / SCF / grid, a convergence
     check is run for the affected structure.

Outputs: run_artifacts/01_pure_water_o3_h2o/acceptance_revision/
         revision_hessian_crosscheck.json
Resume-safe: per-structure results are cached; rerun skips finished ones.
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
STEPS = [0.005, 0.02]          # Bohr, two independent central-difference steps
N_THREADS = 8

OUT_JSON = os.path.join(REV, 'revision_hessian_crosscheck.json')


# --------------------------------------------------------------------- helpers
def load_record(rel):
    with open(os.path.join(ART, rel)) as fh:
        return json.load(fh)


def provenance(tid):
    """Cross-check coordinates/energies across the three record families."""
    opt = load_record('gas_opt/%s.json' % tid)
    frq = load_record('gas_freq/%s.json' % tid)
    rows = {}
    recs = {'gas_opt': opt, 'gas_freq': frq}
    mf_path = os.path.join(ART, 'mode_follow', '%s.json' % tid)
    if os.path.exists(mf_path):
        recs['mode_follow'] = load_record('mode_follow/%s.json' % tid)
    for k, r in recs.items():
        rows[k] = dict(
            coords=(r.get('optimised_coords_angstrom') or r.get('geometry_angstrom')),
            e_total=r.get('e_total_hartree'),
            max_grad=r.get('max_gradient_hartree_bohr'),
            pyscf=r.get('pyscf_version'))
    base_c = np.asarray(rows['gas_freq']['coords'], float)
    geom_dev = {k: float(np.abs(np.asarray(v['coords'], float) - base_c).max())
                for k, v in rows.items() if v['coords'] is not None}
    energies = {k: v['e_total'] for k, v in rows.items() if v['e_total'] is not None}
    e_dev = (max(energies.values()) - min(energies.values())) if energies else None
    return dict(records=rows, max_geometry_deviation_angstrom=geom_dev,
                energy_spread_hartree=e_dev,
                same_geometry=bool(max(geom_dev.values()) < 1e-6) if geom_dev else None,
                same_energy=bool(e_dev is not None and e_dev < 1e-8))


def projected_freqs(mol, h, e_total):
    ha = rb.pyscf_thermo.harmonic_analysis(mol, h, imaginary_freq=False)
    modes = np.sort(np.asarray(ha['freq_wavenumber'], dtype=float))
    return dict(freq_error=int(ha.get('freq_error', -1)),
                n_imag=int(np.sum(modes < -1e-6)),
                lowest6=[round(float(f), 2) for f in modes[:6]],
                modes_all=[round(float(f), 3) for f in modes])


def analytic_hessian(mol):
    mf = d2_full.make_mf_d2(mol)
    mf.kernel()
    e = float(mf.kernel()) if not mf.converged else float(mf.e_tot)
    h_raw = mf.Hessian().kernel()
    asym = float(np.abs(h_raw - h_raw.transpose(1, 0, 3, 2)).max())
    h = 0.5 * (h_raw + h_raw.transpose(1, 0, 3, 2))
    return e, asym, h


def fd_hessian(mol, step):
    """Independent Hessian: central difference of the TOTAL gradient
    (make_mf_d2 gradients = DFT + analytic -D2)."""
    c0 = mol.atom_coords(unit='Bohr')
    n = mol.natm
    g = np.zeros((n, n, 3, 3))
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
                gg = mf.nuc_grad_method().kernel().reshape(n, 3)
                g[i, :, k] += s * gg / (2.0 * step)
    asym = float(np.abs(g - g.transpose(1, 0, 3, 2)).max())
    return 0.5 * (g + g.transpose(1, 0, 3, 2)), asym


def run_target(tid):
    t0 = time.time()
    res = dict(id=tid, provenance=provenance(tid), steps={})
    coords = res['provenance']['records']['gas_freq']['coords']
    mol = cu.mol_from_coords(coords, SYMS, verbose=0)

    e_a, asym_a, h_a = analytic_hessian(mol)
    fa = projected_freqs(mol, h_a, e_a)
    res['analytic'] = dict(e_total=e_a, pre_symmetry_asymmetry=asym_a,
                           projected=fa)

    for step in STEPS:
        h_fd, asym_fd = fd_hessian(mol, step)
        dh = float(np.abs(h_a - h_fd).max())
        rel = float(np.abs(h_a - h_fd).max() / np.abs(h_a).max())
        f_fd = projected_freqs(mol, h_fd, e_a)
        # frequency deltas for the lowest 6 modes (matched by index after sort)
        n6 = min(6, len(fa['modes_all']), len(f_fd['modes_all']))
        dnu = [round(f_fd['modes_all'][i] - fa['modes_all'][i], 2) for i in range(n6)]
        res['steps'][str(step)] = dict(
            pre_symmetry_asymmetry=asym_fd,
            max_abs_H_diff_hartree_bohr2=dh,
            relative_H_diff=rel,
            projected=f_fd,
            lowest6_freq_delta_cm1=dnu)
        print('[%s] step=%.3f  max|dH|=%.3e  rel=%.2e  n_imag(a=%d,fd=%d)  dnu=%s'
              % (tid, step, dh, rel, fa['n_imag'], f_fd['n_imag'], dnu), flush=True)

    res['seconds'] = round(time.time() - t0, 1)
    with open(os.path.join(REV, '_crosscheck_%s.json' % tid), 'w') as fh:
        json.dump(res, fh, indent=2)
    return res


def _init_worker():
    import pyscf.lib as lib
    lib.num_threads(N_THREADS)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--only', default=None)
    args = ap.parse_args()
    targets = args.only.split(',') if args.only else TARGETS

    done = {}
    if os.path.exists(OUT_JSON) and not args.force:
        with open(OUT_JSON) as fh:
            done = {r['id']: r for r in json.load(fh).get('targets', [])}

    todo = [t for t in targets if args.force or t not in done]
    results = list(done.values())
    if todo:
        with Pool(len(todo), initializer=_init_worker) as pool:
            results.extend(pool.map(run_target, todo))
    results.sort(key=lambda r: TARGETS.index(r['id']))

    # sensitivity verdict
    verdict = {}
    for r in results:
        rels = [s['relative_H_diff'] for s in r['steps'].values()]
        dnus = [max(abs(x) for x in s['lowest6_freq_delta_cm1']) for s in r['steps'].values()]
        verdict[r['id']] = dict(
            max_rel_H_diff=max(rels),
            max_lowest6_freq_shift_cm1=max(dnus),
            n_imag_consistent=len(set([r['analytic']['projected']['n_imag']] +
                                      [s['projected']['n_imag'] for s in r['steps'].values()])) == 1)
    out = dict(job='JOB-2026-0905-009',
               step='revision_hessian_crosscheck',
               method='analytic full-derivative Hessian vs central-difference of the '
                      'TOTAL gradient (DFT + analytic -D2), steps %s Bohr' % STEPS,
               targets=results, verdict=verdict)
    with open(OUT_JSON, 'w') as fh:
        json.dump(out, fh, indent=2)
    print('SAVED ->', OUT_JSON)
    for k, v in verdict.items():
        print('VERDICT %s: max_rel_dH=%.2e  max_dnu(6 lowest)=%.1f cm-1  n_imag_consistent=%s'
              % (k, v['max_rel_H_diff'], v['max_lowest6_freq_shift_cm1'],
                 v['n_imag_consistent']))


if __name__ == '__main__':
    main()
