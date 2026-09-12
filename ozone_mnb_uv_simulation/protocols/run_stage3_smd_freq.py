#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage 3 (JOB-2026-0905-009): SMD(water) comparison of the retained gas-phase
minima.

For every distinct gas-phase minimum kept by Stage 2 we:
  * re-optimise the structure in SMD(H2O) with make_mf_d2(solvent='water')
    (geomeTRIC, delocalised internal coords);
  * compute the harmonic frequencies from a NUMERICAL Hessian built from the
    analytic gradients (so the solvent contribution is correctly differentiated
    -- an analytic solvent Hessian is not guaranteed in PySCF);
  * de-duplicate the SMD results again (RMSD + topology);
  * report the gas -> SMD change in ordering, geometry and H-bond topology.

The SMD results are a *local solvation-structure trend only*; they must NOT be
read as aqueous-phase binding constants (no free-energy-of-solvation cycle is
performed here).
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
from pyscf.geomopt import geometric_solver
from pyscf.geomopt.geometric_solver import optimize

ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, 'run_artifacts', '01_pure_water_o3_h2o')
GAS_OPT = os.path.join(ART, 'gas_opt')
OUT = os.path.join(ART, 'smd_freq')
os.makedirs(OUT, exist_ok=True)

SYMS = ['O', 'O', 'O', 'O', 'H', 'H']
SOLVENT = 'water'
N_PROC = 8
N_THREADS = 4
# geomeTRIC (delocalised internal coords) -- berny refuses the near-linear
# O_w-H...O hydrogen-bond chain with a CoordinateError, and SMD re-optimisation
# from the gas-phase minima would hit the same failure.  Convergence is passed
# as TOP-LEVEL kwargs with geomeTRIC's real key names (convergence_gmax etc.);
# dmax/drms = 10 A disables premature displacement convergence.
OPT_CONV = dict(maxsteps=400, assert_convergence=False,
                convergence_gmax=3e-5, convergence_grms=1.5e-5,
                convergence_dmax=10.0, convergence_drms=10.0,
                convergence_energy=1e-7)
RMSD_TOL = 0.10
ETOL_KCAL = 0.10


def _init_worker():
    import pyscf.lib as lib
    lib.num_threads(N_THREADS)


def run_one(tid):
    t_start = time.time()
    rec = dict(id=tid, solvent=SOLVENT)
    try:
        g = cu.load_gas_opt(GAS_OPT)[tid]
        coords0 = cu.coords_of(g)
        mol0 = cu.mol_from_coords(coords0, SYMS, verbose=0)
        mf = d2_full.make_mf_d2(mol0, solvent=SOLVENT)
        mol_opt = optimize(mf, **OPT_CONV)
        coords = np.asarray(mol_opt.atom_coords(unit='Angstrom'), dtype=float)
        rec['optimised_coords_angstrom'] = [[round(float(x), 8) for x in row]
                                            for row in coords]

        # strict single point + SMD numerical Hessian (full potential)
        molf = cu.mol_from_coords(coords, SYMS, verbose=0)
        mff = d2_full.make_mf_d2(molf, solvent=SOLVENT)
        e_total = float(mff.kernel())
        rec['e_total_hartree'] = e_total
        rec['d2_hartree'] = d2_full.d2_energy(molf)
        t0 = time.time()
        h = cu.smd_num_hessian(molf, solvent=SOLVENT)
        rec['hessian_seconds'] = round(time.time() - t0, 1)
        rec['hessian_method'] = 'numerical (central diff of analytic grad), SMD'
        rec['hessian_asymmetry'] = float(np.abs(h - h.transpose(1, 0, 3, 2)).max())
        vib = cu.analyse_vibrations(molf, h, e_total)
        rec['n_imaginary'] = int(vib['n_imag'])
        rec['lowest_frequency_cm1'] = float(vib['modes'][0])
        rec['frequencies_cm1'] = [round(float(f), 3) for f in vib['modes']]
        rec['is_minimum'] = bool(vib['n_imag'] == 0)
        rec['zpe_hartree'] = vib['zpe_hartree']
        rec['zpe_kcal_mol'] = vib['zpe_hartree'] * cu.HARTREE2KCAL
        th = vib['thermo']
        rec['thermo_hartree'] = {'ZPE': th['ZPE'] / cu.rb.HA2JMOL,
                                 'E_0K': th['E_0K'] / cu.rb.HA2JMOL,
                                 'H_298': th['H_298'] / cu.rb.HA2JMOL,
                                 'G_298': th['G_298'] / cu.rb.HA2JMOL}
        topo = cu.hbond_topology(coords)
        rec['topology'] = topo['description']
        rec['topology_signature'] = [list(topo['hbonds']), sorted(topo['ow_contacts'])]
        rec['status'] = 'converged' if rec['is_minimum'] else 'SADDLE'
    except Exception as exc:
        import traceback
        rec['status'] = 'FAILED'
        rec['error'] = '%s: %s' % (type(exc).__name__, exc)
        os.makedirs(os.path.join(OUT, 'logs'), exist_ok=True)
        with open(os.path.join(OUT, 'logs', '%s.log' % tid), 'w') as fh:
            traceback.print_exc(file=fh)
    rec['seconds'] = round(time.time() - t_start, 1)
    with open(os.path.join(OUT, '%s.json' % tid), 'w') as fh:
        json.dump(rec, fh, indent=2)
    print('[stage3 %s] %s  n_imag=%s  E=%.8f  %.1f s' % (
        tid, rec.get('status'), rec.get('n_imaginary'),
        rec.get('e_total_hartree', float('nan')), rec.get('seconds', 0)), flush=True)
    return rec


def deduplicate_smd(recs):
    """Re-dedup SMD optimised structures (some gas minima may collapse)."""
    ids = sorted(recs, key=lambda i: recs[i]['e_total_hartree'])
    kept, elim = [], {}
    used = []
    for i in ids:
        dup_of = None
        for j in used:
            rms = cu.best_rmsd(recs[j]['optimised_coords_angstrom'],
                               recs[i]['optimised_coords_angstrom'])
            same_sig = (tuple(map(tuple, recs[j]['topology_signature'][0])),
                        tuple(recs[j]['topology_signature'][1])) == \
                       (tuple(map(tuple, recs[i]['topology_signature'][0])),
                        tuple(recs[i]['topology_signature'][1]))
            de = (recs[i]['e_total_hartree'] - recs[j]['e_total_hartree']) * cu.HARTREE2KCAL
            if rms < RMSD_TOL and same_sig and abs(de) < ETOL_KCAL:
                dup_of = j
                break
        if dup_of is None:
            kept.append(i)
            used.append(i)
        else:
            elim[i] = 'collapsed onto SMD structure of %s (RMSD small, same topology)' % dup_of
    return kept, elim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    os.makedirs(os.path.join(OUT, 'logs'), exist_ok=True)

    with open(os.path.join(ART, 'stage2_minima.json')) as fh:
        s2 = json.load(fh)
    kept_ids = list(s2['kept_minima'])
    if not args.force:
        kept_ids = [i for i in kept_ids
                    if not os.path.exists(os.path.join(OUT, '%s.json' % i))]

    t0 = time.time()
    with Pool(N_PROC, initializer=_init_worker) as pool:
        results = pool.map(run_one, kept_ids)
    recs = {r['id']: r for r in results if 'e_total_hartree' in r}
    smd_kept, smd_elim = deduplicate_smd(recs)

    # gas -> SMD ordering comparison
    gas = cu.load_gas_opt(GAS_OPT)
    order = []
    for i in sorted(recs, key=lambda i: recs[i]['e_total_hartree']):
        g_e = gas[i]['e_total_hartree']
        s_e = recs[i]['e_total_hartree']
        order.append(dict(id=i, gas_e=g_e, smd_e=s_e,
                          delta_e_smd_minus_gas=(s_e - g_e) * cu.HARTREE2KCAL,
                          gas_topology=cu.hbond_topology(cu.coords_of(gas[i]))['description'],
                          smd_topology=recs[i]['topology'],
                          smd_kept=(i in smd_kept)))
    summary = dict(
        job='JOB-2026-0905-009', stage='3_smd_freq',
        solvent_model='SMD(%s)' % SOLVENT,
        method='wB97X-D/def2-TZVP full-derivative -D2',
        note='local hydration-structure trend only; NOT an aqueous binding constant',
        smd_minima=smd_kept,
        smd_eliminated={i: smd_elim[i] for i in smd_elim},
        gas_vs_smd_ordering=order,
        wall_seconds=round(time.time() - t0, 1),
    )
    with open(os.path.join(ART, 'stage3_smd.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)
    print('STAGE3_DONE smd_kept=%s wall=%.1f s' % (smd_kept, time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
