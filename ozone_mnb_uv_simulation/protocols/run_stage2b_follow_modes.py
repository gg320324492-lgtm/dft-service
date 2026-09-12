#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage 2b (JOB-2026-0905-009): follow imaginary modes of saddle-point
stationary structures to locate the true minima they connect to.

Motivation: with the corrected frequency sign handling (imaginary_freq=False)
only c05 among the 14 trial structures is a genuine minimum; the near-degenerate
global-lowest structures (c14/c09/c11/c12, ...) each carry ONE significant
imaginary mode (|nu_i| ~ 45-120 cm-1, far above numerical noise).  A saddle
below a minimum implies a lower true minimum exists that the trial pool never
reached, so per the job brief ("keep the global lowest and all distinct minima
within 3 kcal/mol") we displace along the imaginary mode in BOTH directions,
re-optimise (geomeTRIC, correct convergence keys), and confirm each endpoint
with a full-derivative -D2 Hessian frequency analysis.

Only saddles with exactly one imaginary mode inside the 3 kcal/mol window are
followed (higher-order saddles reflect poor trial geometries, not missing
minima).  The final true-minimum set is de-duplicated together with c05 using
run_stage2_gas_freq.deduplicate, and stage2_minima.json is rewritten so the
downstream stages (SMD / binding / CCSD(T)) consume the corrected set.
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
import run_stage2_gas_freq as s2
from pyscf import gto
from pyscf.geomopt import geometric_solver
from pyscf.geomopt.geometric_solver import optimize

ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, 'run_artifacts', '01_pure_water_o3_h2o')
GAS_OPT = os.path.join(ART, 'gas_opt')
GAS_FREQ = os.path.join(ART, 'gas_freq')
OUT = os.path.join(ART, 'mode_follow')
os.makedirs(OUT, exist_ok=True)

SYMS = ['O', 'O', 'O', 'O', 'H', 'H']
N_PROC = 8
N_THREADS = 4
DISPLACE_MAX_A = 0.25          # max atom displacement along the imaginary mode
OPT_CONV = dict(maxsteps=400, assert_convergence=False,
                convergence_gmax=3e-5, convergence_grms=1.5e-5,
                convergence_dmax=10.0, convergence_drms=10.0,
                convergence_energy=1e-7)


def _init_worker():
    import pyscf.lib as lib
    lib.num_threads(N_THREADS)


def imaginary_mode_cartesian(mol, h):
    """Mass-weighted Hessian -> Cartesian displacement of the most imaginary
    mode (same mass-weighting convention as run_baseline.vib_freqs)."""
    n = mol.natm
    H = h.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)
    mw = np.repeat(mol.atom_mass_list(isotope_avg=True) ** -0.5, 3)
    Heff = H * np.outer(mw, mw)
    w, v = np.linalg.eigh(Heff)
    imode = int(np.argmin(w))                     # most negative eigenvalue
    dx = mw * v[:, imode]                         # mass-weighted -> Cartesian
    dx = dx.reshape(n, 3)
    scale = DISPLACE_MAX_A / np.abs(dx).max()
    return dx * scale, float(w[imode])


def follow_one(task):
    """Displace saddle `tid` along +/- its imaginary mode, re-optimise each
    endpoint and run the full -D2 frequency check.  Returns two records."""
    tid, direction = task
    label = '%s_%s' % (tid, direction)
    t0 = time.time()
    rec = dict(label=label, source=tid, direction=direction)
    try:
        g = cu.load_gas_opt(GAS_OPT)[tid]
        coords0 = cu.coords_of(g)
        mol0 = cu.mol_from_coords(coords0, SYMS, verbose=0)
        mf0 = d2_full.make_mf_d2(mol0)
        mf0.kernel()
        h0 = cu.gas_full_hessian(mol0)
        dx, eig = imaginary_mode_cartesian(mol0, h0)
        rec['imag_eigenvalue_hartree_amu_bohr2'] = eig
        start = coords0 + (dx if direction == 'plus' else -dx)

        mols = gto.M(atom="; ".join("%s %.10f %.10f %.10f" % (s_, x, y, z)
                                    for s_, (x, y, z) in zip(SYMS, start)),
                     basis=d2_full.BASIS, charge=0, spin=0, verbose=0)
        mf = d2_full.make_mf_d2(mols)
        mol_opt = optimize(mf, **OPT_CONV)
        coords = np.asarray(mol_opt.atom_coords(unit='Angstrom'), dtype=float)

        # strict single point with a FRESH object (same convention as Stage 1)
        molf = cu.mol_from_coords(coords, SYMS, verbose=0)
        mff = d2_full.make_mf_d2(molf)
        e_tot = float(mff.kernel())
        grad = np.asarray(mff.nuc_grad_method().kernel())
        rec['max_gradient_hartree_bohr'] = float(np.abs(grad).max())
        rec['optimisation_converged'] = bool(rec['max_gradient_hartree_bohr'] < 5e-5
                                             and mff.converged)
        rec['e_total_hartree'] = e_tot
        rec['geometry_angstrom'] = [[round(float(x), 8) for x in row]
                                    for row in coords]

        # full -D2 frequency check at the endpoint
        h = cu.gas_full_hessian(molf)
        vib = cu.analyse_vibrations(molf, h, e_tot, sigma=1)
        rec['n_imaginary'] = int(vib['n_imag'])
        rec['freq_error'] = int(vib.get('freq_error', -1))
        rec['lowest_frequency_cm1'] = float(vib['modes'][0])
        rec['frequencies_cm1'] = [round(float(f), 3) for f in vib['modes']]
        rec['zpe_hartree'] = vib['zpe_hartree']
        th = vib['thermo']
        rec['thermo_hartree'] = {'ZPE': th['ZPE'] / cu.rb.HA2JMOL,
                                 'E_0K': th['E_0K'] / cu.rb.HA2JMOL,
                                 'H_298': th['H_298'] / cu.rb.HA2JMOL,
                                 'G_298': th['G_298'] / cu.rb.HA2JMOL}
        topo = cu.hbond_topology(coords)
        rec['topology'] = topo['description']
        rec['topology_signature'] = [list(topo['hbonds']), sorted(topo['ow_contacts'])]
        rec['is_minimum'] = bool(rec['n_imaginary'] == 0)
        rec['status'] = ('converged_minimum' if (rec['is_minimum'] and rec['optimisation_converged'])
                         else ('converged_saddle' if rec['optimisation_converged'] else 'NOT_CONVERGED'))
    except Exception as exc:
        import traceback
        rec['status'] = 'FAILED'
        rec['error'] = '%s: %s' % (type(exc).__name__, exc)
        with open(os.path.join(OUT, 'logs', '%s.log' % label), 'w') as fh:
            os.makedirs(os.path.join(OUT, 'logs'), exist_ok=True)
            traceback.print_exc(file=fh)
    rec['seconds'] = round(time.time() - t0, 1)
    with open(os.path.join(OUT, '%s.json' % label), 'w') as fh:
        json.dump(rec, fh, indent=2)
    print('[follow %s] %s E=%.8f n_imag=%s %.1f s' % (
        label, rec.get('status'), rec.get('e_total_hartree', float('nan')),
        rec.get('n_imaginary'), rec.get('seconds', 0)), flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    os.makedirs(os.path.join(OUT, 'logs'), exist_ok=True)

    with open(os.path.join(ART, 'stage2_minima.json')) as fh:
        s2m = json.load(fh)

    # saddles to follow: exactly one imaginary mode, inside the window,
    # not already deduplicated away as a structural duplicate
    elim = s2m.get('eliminated', {})
    saddles = []
    for i, reason in elim.items():
        if 'saddle point: 1 imaginary' in reason:
            saddles.append(i)
    tasks = []
    for tid in saddles:
        for direction in ('plus', 'minus'):
            label = '%s_%s' % (tid, direction)
            if not args.force and os.path.exists(os.path.join(OUT, '%s.json' % label)):
                continue
            tasks.append((tid, direction))
    print('following %d saddles x 2 directions -> %d runs: %s'
          % (len(saddles), len(tasks), [t[0] for t in tasks][::2]), flush=True)

    t0 = time.time()
    results = []
    if tasks:
        with Pool(N_PROC, initializer=_init_worker) as pool:
            results = pool.map(follow_one, tasks)

    # ---- merge newly found minima with c05 and re-deduplicate
    freq_files = sorted(os.listdir(GAS_FREQ))
    minima_pool = {}
    for fn in freq_files:
        if not fn.endswith('.json'):
            continue
        with open(os.path.join(GAS_FREQ, fn)) as fh:
            d = json.load(fh)
        if d.get('n_imaginary') == 0 and 'e_total_hartree' in d and d['id'].startswith('c'):
            minima_pool[d['id']] = d
    for r in results:
        if r.get('status') == 'converged_minimum':
            r['id'] = r['label']            # dedup/summary key
            minima_pool[r['label']] = r
    print('true-minimum pool before dedup: %s' % sorted(minima_pool), flush=True)

    global_min_id, kept, elim2 = s2.deduplicate(dict(minima_pool))

    # ---- provenance bookkeeping for the report
    follow_summary = dict(
        job='JOB-2026-0905-009', stage='2b_mode_following',
        method='displace +/- %.2f A along the most imaginary mode (mass-weighted '
               'eigenvector), re-optimise with the Stage-1 geomeTRIC convergence, '
               'confirm with the full-derivative -D2 Hessian' % DISPLACE_MAX_A,
        saddles_followed=saddles,
        runs=results,
        n_runs=len(results),
        n_runs_failed=sum(1 for r in results if r.get('status') == 'FAILED'),
        endpoints_true_minima=sorted(r['label'] for r in results
                                     if r.get('status') == 'converged_minimum'),
        endpoints_saddles=sorted(r['label'] for r in results
                                 if r.get('status') == 'converged_saddle'),
        endpoints_not_converged=sorted(r['label'] for r in results
                                       if r.get('status') == 'NOT_CONVERGED'),
        minimum_pool_before_dedup=sorted(minima_pool),
        kept_after_dedup=kept,
        eliminated_after_dedup=elim2,
        wall_seconds=round(time.time() - t0, 1))
    with open(os.path.join(ART, 'stage2b_follow.json'), 'w') as fh:
        json.dump(follow_summary, fh, indent=2)

    # ---- rewrite stage2_minima.json with the corrected true-minimum set
    s2m['kept_minima'] = sorted(kept, key=lambda i: minima_pool[i]['e_total_hartree'])
    s2m['kept_relative_kcal'] = {
        i: round((minima_pool[i]['e_total_hartree']
                  - minima_pool[global_min_id]['e_total_hartree']) * cu.HARTREE2KCAL, 4)
        for i in s2m['kept_minima']}
    s2m['gas_global_minimum'] = global_min_id
    s2m['gas_global_energy_hartree'] = minima_pool[global_min_id]['e_total_hartree']
    s2m['n_distinct_minima'] = len(s2m['kept_minima'])
    merged_elim = dict(elim2)
    for i, reason in elim.items():                 # keep the saddle bookkeeping
        if 'saddle point' in reason:
            merged_elim.setdefault(i, reason)
    s2m['eliminated'] = merged_elim
    s2m['mode_following'] = dict(
        performed=True,
        saddles_followed=saddles,
        endpoints_true_minima=follow_summary['endpoints_true_minima'],
        endpoints_saddles=follow_summary['endpoints_saddles'],
        note=('trial structures with imaginary modes are saddle points; they '
              'were displaced along the imaginary mode and re-optimised to '
              'locate the true minima of the flat association PES'))
    with open(os.path.join(ART, 'stage2_minima.json'), 'w') as fh:
        json.dump(s2m, fh, indent=2)
    print('STAGE2B_DONE global=%s kept=%s wall=%.1f s'
          % (global_min_id, s2m['kept_minima'], time.time() - t0), flush=True)


if __name__ == '__main__':
    import argparse
    main()
