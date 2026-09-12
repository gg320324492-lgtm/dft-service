#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage 5 (JOB-2026-0905-009): CCSD(T)/aug-cc-pVTZ single-point interaction
energies for the two lowest distinct gas-phase minima.

For each selected complex the script also evaluates the two monomers
(O3, H2O) at the same basis set (shared across complexes), so the
CCSD(T) interaction energy is consistent.  The T1 diagnostic is reported
for every complex as a sanity check on single-reference behaviour.

Resource policy: if the basis/run cannot be afforded it MUST stop and
report -- it must never be silently downgraded to a cheaper method.
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
import conformer_utils as cu

ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, 'run_artifacts', '01_pure_water_o3_h2o')
GAS_OPT = os.path.join(ART, 'gas_opt')
OUT = os.path.join(ART, 'ccsdt')
os.makedirs(OUT, exist_ok=True)

BASIS = 'aug-cc-pVTZ'
N_PROC = 4
N_THREADS = 8
MAX_MEM_MB = 4000

SYMS_COMPLEX = ['O', 'O', 'O', 'O', 'H', 'H']


def _init_worker():
    import pyscf.lib as lib
    lib.num_threads(N_THREADS)


def ccsdt_sp(tid, coords, syms):
    """RHF -> CCSD(T) single point; returns total energy and T1 diagnostic."""
    import pyscf
    from pyscf import gto, scf, cc
    t0 = time.time()
    s = "; ".join("%s %.10f %.10f %.10f" % (s_, x, y, z)
                  for s_, (x, y, z) in zip(syms, coords))
    mol = gto.M(atom=s, basis=BASIS, charge=0, spin=0, verbose=4)
    mf = scf.RHF(mol)
    mf.max_memory = MAX_MEM_MB
    mf.kernel()
    mycc = cc.CCSD(mf)
    mycc.max_memory = MAX_MEM_MB
    mycc.kernel()
    et = mycc.ccsd_t()
    e_tot = mycc.e_tot + et
    # T1 diagnostic (RHF t1 is (nocc, nvir)); single-reference if T1 < 0.02
    t1 = float(np.max(np.abs(mycc.t1)))
    res = dict(
        id=tid, basis=BASIS, pyscf_version=pyscf.__version__,
        n_basis_functions=int(mol.nao_nr()),
        e_hf=float(mf.e_tot),
        e_ccsd=float(mycc.e_tot),
        e_ccsd_t=float(et),
        e_ccsdt_total=float(e_tot),
        t1_diagnostic=t1,
        converged_ccsd=bool(mycc.converged),
        seconds=round(time.time() - t0, 1),
    )
    with open(os.path.join(OUT, '%s.json' % tid), 'w') as fh:
        json.dump(res, fh, indent=2)
    print('[stage5 %s] E(CCSD(T))=% .10f  T1=%.4f  %.1f s' % (
        tid, e_tot, res['t1_diagnostic'], res['seconds']), flush=True)
    return res


def monomer_coords(spec):
    """Optimised monomer geometry (wB97X-D/def2-TZVP) from Stage 1, so the
    CCSD(T) interaction energy is consistent with the DFT Stage-4 value
    (same monomer geometries)."""
    coords = cu.coords_of(cu.load_gas_opt(GAS_OPT)[spec])
    if spec == 'o3':
        return np.asarray(coords, dtype=float), ['O', 'O', 'O']
    if spec == 'h2o':
        return np.asarray(coords, dtype=float), ['O', 'H', 'H']
    raise ValueError(spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    with open(os.path.join(ART, 'stage2_minima.json')) as fh:
        s2 = json.load(fh)
    kept = sorted(s2['kept_minima'],
                  key=lambda i: s2['kept_relative_kcal'][i])
    chosen = kept[:2]
    print('CCSD(T) targets (two lowest distinct minima): %s' % chosen, flush=True)

    tasks = []
    for tid in chosen:
        coords = cu.coords_of(cu.load_gas_opt(GAS_OPT)[tid])
        tasks.append((tid, coords, SYMS_COMPLEX))
    for m in ('o3', 'h2o'):
        c, s = monomer_coords(m)
        tasks.append((m, c, s))

    if not args.force:
        tasks = [t for t in tasks if not os.path.exists(os.path.join(OUT, '%s.json' % t[0]))]

    t0 = time.time()
    with Pool(N_PROC, initializer=_init_worker) as pool:
        results = pool.starmap(ccsdt_sp, tasks)
    res = {r['id']: r for r in results}

    # assemble interaction energies for the two complexes
    o3 = res['o3']['e_ccsdt_total']
    h2o = res['h2o']['e_ccsdt_total']
    bind = []
    for tid in chosen:
        ec = res[tid]['e_ccsdt_total']
        bind.append(dict(
            id=tid,
            e_ccsdt_complex=ec, e_ccsdt_o3=o3, e_ccsdt_h2o=h2o,
            e_int_ccsdt_kcal_mol=round((ec - o3 - h2o) * cu.HARTREE2KCAL, 4),
            t1_diagnostic=res[tid]['t1_diagnostic'],
            single_reference_ok=bool(res[tid]['t1_diagnostic'] < 0.02)))
    out = dict(job='JOB-2026-0905-009', stage='5_ccsdt', basis=BASIS,
               n_proc=N_PROC, n_threads=N_THREADS, max_memory_mb=MAX_MEM_MB,
               targets=chosen, monomer_energies=dict(o3=o3, h2o=h2o),
               interaction=bind, wall_seconds=round(time.time() - t0, 1))
    with open(os.path.join(ART, 'ccsdt.json'), 'w') as fh:
        json.dump(out, fh, indent=2)
    print('STAGE5_DONE  wall=%.1f s' % (time.time() - t0), flush=True)
    for b in bind:
        print('  %s  E_int(CCSD(T))=% .3f kcal/mol  T1=%.4f  single-ref=%s'
              % (b['id'], b['e_int_ccsdt_kcal_mol'], b['t1_diagnostic'],
                 b['single_reference_ok']), flush=True)


if __name__ == '__main__':
    main()
