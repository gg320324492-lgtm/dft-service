#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage 2 (JOB-2026-0905-009): gas-phase conformer de-duplication + full
derivative -D2 harmonic analysis.

Pipeline:
  1. load every Stage-1 gas optimisation (c01..c14 + monomers h2o, o3);
  2. for each, build a fresh PySCF mol at the optimised geometry, compute the
     total energy, the analytic DFT Hessian + analytic -D2 Hessian, and run the
     vibrational / thermochemical analysis (reusing run_baseline.thermochemistry);
     record <S^2> and the number of imaginary frequencies (true-minimum test);
  3. de-duplicate the conformers by (RMSD with C2 ozone-terminal permutation)
     AND (H-bond topology signature) AND (total energy, 0.1 kcal/mol window);
  4. retain the global minimum and every DISTINCT minimum within 3 kcal/mol of
     the gas global minimum; the rest are tagged with their elimination reason.

Outputs (per species):  run_artifacts/01_pure_water_o3_h2o/gas_freq/<id>.json
Aggregates:             run_artifacts/01_pure_water_o3_h2o/stage2_minima.json
"""
import os
import sys
import json
import time
import argparse
import numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import d2_full
import conformer_utils as cu

ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, 'run_artifacts', '01_pure_water_o3_h2o')
GAS_OPT = os.path.join(ART, 'gas_opt')
OUT = os.path.join(ART, 'gas_freq')
os.makedirs(OUT, exist_ok=True)

SYMS = ['O', 'O', 'O', 'O', 'H', 'H']
CONF_IDS = ['c%02d' % i for i in range(1, 15)]
MONO_IDS = ['h2o', 'o3']
N_PROC = 8
N_THREADS = 4

RMSD_TOL = 0.10           # Angstrom
ETOL_KCAL = 0.10          # kcal/mol
WINDOW_KCAL = 3.0         # keep minima within this of the gas global min


def _init_worker():
    import pyscf.lib as lib
    lib.num_threads(N_THREADS)


def analyse_one(tid, coords, syms, is_monomer=False):
    t0 = time.time()
    mol = cu.mol_from_coords(coords, syms, verbose=0)
    mf = d2_full.make_mf_d2(mol)
    e_total = float(mf.kernel())
    mol.build()  # ensure consistent
    h = cu.gas_full_hessian(mol)
    asym = float(np.abs(h - h.transpose(1, 0, 3, 2)).max())
    sigma = 2 if is_monomer else 1
    vib = cu.analyse_vibrations(mol, h, e_total, sigma)
    rec = dict(
        id=tid,
        kind=('monomer' if is_monomer else 'conformer'),
        n_basis_functions=int(mol.nao_nr()),
        e_total_hartree=e_total,
        d2_hartree=d2_full.d2_energy(mol),
        hessian_asymmetry=asym,
        hessian_method='analytic DFT + analytic -D2 (finite-diff of analytic grad)',
        n_imaginary=int(vib['n_imag']),
        lowest_frequency_cm1=float(vib['modes'][0]),
        frequencies_cm1=[round(float(f), 3) for f in vib['modes']],
        is_minimum=bool(vib['n_imag'] == 0),
        zpe_hartree=vib['zpe_hartree'],
        zpe_kcal_mol=vib['zpe_hartree'] * cu.HARTREE2KCAL,
        thermo={k: (round(float(v), 8) if isinstance(v, (int, float)) else v)
                for k, v in vib['thermo'].items()},
        seconds=round(time.time() - t0, 1),
    )
    # thermo in Hartree for binding bookkeeping
    th = vib['thermo']
    rec['thermo_hartree'] = {
        'ZPE': th['ZPE'] / cu.rb.HA2JMOL,
        'E_0K': th['E_0K'] / cu.rb.HA2JMOL,
        'H_298': th['H_298'] / cu.rb.HA2JMOL,
        'G_298': th['G_298'] / cu.rb.HA2JMOL,
    }
    # <S^2>
    try:
        ss, sz = mf.spin_square()
        rec['spin_s2'] = float(ss)
    except Exception:
        rec['spin_s2'] = 0.0
    rec['geometry_angstrom'] = [[round(float(x), 8) for x in row]
                                for row in coords]
    if len(coords) == 6:
        topo = cu.hbond_topology(coords)
        rec['topology'] = topo['description']
        rec['topology_signature'] = [list(topo['hbonds']), sorted(topo['ow_contacts'])]
    else:
        rec['topology'] = 'monomer (no H-bond topology defined)'
        rec['topology_signature'] = None
    return rec


def run_task(tid):
    path = os.path.join(OUT, '%s.json' % tid)
    try:
        recs = cu.load_gas_opt(GAS_OPT)
        rec = recs[tid]
        coords = cu.coords_of(rec)
        is_mono = (tid in MONO_IDS)
        if tid == 'h2o':
            syms = ['O', 'H', 'H']
        elif tid == 'o3':
            syms = ['O', 'O', 'O']
        else:
            syms = SYMS
        out = analyse_one(tid, coords, syms, is_mono)
    except Exception as exc:
        import traceback
        out = dict(id=tid, status='FAILED', error='%s: %s' % (type(exc).__name__, exc))
        with open(os.path.join(OUT, 'logs', '%s.log' % tid), 'w') as fh:
            traceback.print_exc(file=fh)
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=2)
    print('[stage2 %s] n_imag=%s  E=%.8f  %.1f s' % (
        tid, out.get('n_imaginary'), out.get('e_total_hartree', float('nan')),
        out.get('seconds', 0)), flush=True)
    return out


def deduplicate(conf_recs):
    """Return (kept_ids, eliminations) for the conformers."""
    ids = sorted(conf_recs, key=lambda i: conf_recs[i]['e_total_hartree'])
    global_min_id = ids[0]
    global_e = conf_recs[global_min_id]['e_total_hartree']
    for i in ids:
        conf_recs[i]['rel_kcal'] = (conf_recs[i]['e_total_hartree'] - global_e) * cu.HARTREE2KCAL
    # cluster by topology signature -- the record stores the JSON-serialisable
    # list form, so convert to the hashable canonical form here
    clusters = {}
    for i in ids:
        ts = conf_recs[i]['topology_signature']
        sig = (tuple(map(tuple, ts[0])), tuple(ts[1]))
        clusters.setdefault(sig, []).append(i)
    kept = []
    elim = {}
    for sig, group in clusters.items():
        group = sorted(group, key=lambda i: conf_recs[i]['e_total_hartree'])
        rep = group[0]
        kept.append(rep)
        for other in group[1:]:
            rms = cu.best_rmsd(conf_recs[rep]['geometry_angstrom'],
                               conf_recs[other]['geometry_angstrom'])
            de = (conf_recs[other]['e_total_hartree']
                  - conf_recs[rep]['e_total_hartree']) * cu.HARTREE2KCAL
            if rms < RMSD_TOL and abs(de) < ETOL_KCAL:
                elim[other] = ('duplicate of %s (RMSD=%.3f A, dE=%.3f kcal/mol, '
                               'same topology)' % (rep, rms, de))
            else:
                kept.append(other)
    # window filter + true-minimum requirement (JOB brief: keep the global
    # lowest and all DISTINCT MINIMA within 3 kcal/mol -- structures with
    # imaginary modes are saddle points and are eliminated with the value of
    # the imaginary frequency recorded)
    final = []
    for i in sorted(kept, key=lambda i: conf_recs[i]['e_total_hartree']):
        rel = conf_recs[i]['rel_kcal']
        n_img = int(conf_recs[i].get('n_imaginary', 0))
        if n_img > 0:
            nu_i = conf_recs[i].get('lowest_frequency_cm1')
            elim.setdefault(i, 'saddle point: %d imaginary mode(s) '
                              '(nu_i = %s cm-1), rel %.3f kcal/mol -- not a minimum'
                            % (n_img, nu_i, rel))
            continue
        if rel <= WINDOW_KCAL:
            final.append(i)
        else:
            elim.setdefault(i, 'energy %.3f kcal/mol above gas global min (> %.1f window)'
                            % (rel, WINDOW_KCAL))
    return global_min_id, final, elim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default=None)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    os.makedirs(os.path.join(OUT, 'logs'), exist_ok=True)

    all_ids = CONF_IDS + MONO_IDS
    if args.only:
        all_ids = [i for i in all_ids if i in set(args.only.split(','))]
    if not args.force:
        all_ids = [i for i in all_ids
                   if not os.path.exists(os.path.join(OUT, '%s.json' % i))]

    t0 = time.time()
    results = []
    if all_ids:
        with Pool(N_PROC, initializer=_init_worker) as pool:
            results = pool.map(run_task, all_ids)
    # merge results that already existed on disk (skipped ids) so the
    # deduplication summary still sees the full conformer pool
    for i in CONF_IDS + MONO_IDS:
        if i in all_ids:
            continue
        p = os.path.join(OUT, '%s.json' % i)
        if os.path.exists(p):
            with open(p) as fh:
                results.append(json.load(fh))
    conf_recs = {r['id']: r for r in results if 'e_total_hartree' in r and r['id'] in CONF_IDS}
    mono_recs = {r['id']: r for r in results if 'e_total_hartree' in r and r['id'] in MONO_IDS}

    global_min_id, kept, elim = deduplicate(conf_recs)
    summary = dict(
        job='JOB-2026-0905-009', stage='2_gas_freq_dedup',
        method='wB97X-D/def2-TZVP full-derivative -D2',
        rmsd_tol_angstrom=RMSD_TOL, energy_tol_kcal=ETOL_KCAL,
        window_kcal=WINDOW_KCAL,
        gas_global_minimum=global_min_id,
        gas_global_energy_hartree=conf_recs[global_min_id]['e_total_hartree'],
        kept_minima=sorted(kept, key=lambda i: conf_recs[i]['e_total_hartree']),
        kept_relative_kcal={i: round(conf_recs[i]['rel_kcal'], 4) for i in kept},
        eliminated={i: elim[i] for i in elim},
        n_conformers_searched=len(CONF_IDS),
        n_distinct_minima=len(kept),
        wall_seconds=round(time.time() - t0, 1),
    )
    with open(os.path.join(ART, 'stage2_minima.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)
    # also dump per-conformer table for the report
    table = {}
    for i in CONF_IDS:
        if i in conf_recs:
            r = conf_recs[i]
            table[i] = dict(e_total=r['e_total_hartree'],
                            rel_kcal=round(r.get('rel_kcal', 0), 4),
                            n_imag=r['n_imaginary'],
                            lowest_freq=round(r['lowest_frequency_cm1'], 2),
                            topology=r['topology'],
                            kept=(i in kept),
                            eliminated_reason=elim.get(i))
    with open(os.path.join(ART, 'stage2_conformer_table.json'), 'w') as fh:
        json.dump(table, fh, indent=2)
    print('STAGE2_DONE global=%s kept=%s eliminated=%d wall=%.1f s' % (
        global_min_id, kept, len(elim), time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
