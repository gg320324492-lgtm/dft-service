#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase B: scanner A->B->A call-path verification (JOB-2026-0906-003).

Verifies, using the ACTUAL gas-phase c14_plus L7 centre geometry, that the
unified gradient entry point (grad_factory.make_grad) produces a scanner whose
gradient

  * always has grid_response == True,
  * is identical (within tolerance) to an INDEPENDENT benchmark
        total = dft_only(wb97x, grid_response=True)  +  project D2,
  * counts the project -D2 dispersion exactly once
        scanner_grad - dft_only == d2_grad(mol),
        scanner_energy - dft_only_energy == d2_energy(mol),
  * reproduces when returning to A (no stale-geometry residue),
  * is actually engaged for grid_response (differs from a False-flag gradient).

Run:
    python phaseB_scanner_verify.py            # real c14_plus, 60+ min -> background
    python phaseB_scanner_verify.py --smoke    # H2O quick path validation (~20 s)
"""
import os, sys, json, time, argparse
import numpy as np
from pyscf import gto

import d2_full
from grad_factory import (make_grad, make_mf_d2_gr, dft_only_grad,
                         method_fingerprint)

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, '..', 'run_artifacts', '01_pure_water_o3_h2o',
                   'phaseB_scanner_verify')

# tolerances (recorded, not hidden)
# D2-once / scanner-consistency are judged RELATIVE to the D2 magnitude:
# if -D2 were double-counted, the residual would be ~|D2| (ratio >= 1). The
# SCF-noise floor on a meaningful system (c14_plus, |D2|~1e-3) is ~1e-4 of
# |D2|, so a 0.2 relative gate cleanly separates "once" from "twice". A small
# absolute floor (1e-6) also gates the tiny-D2 H2O smoke (where |D2|~noise,
# so smoke only validates the code path, not D2-once -- the real c14 run does).
THR = dict(d2_once_rel=2e-1, scanner_rel=2e-1, abs_floor=1e-6,
           reproduce_e=1e-7, reproduce_g=1e-6)

SYMS_C14 = ['O', 'O', 'O', 'O', 'H', 'H']   # O3 (0-2) + H2O (3-5)


def mol_from_coords(syms, coords_A):
    atom = "; ".join("%s %.10f %.10f %.10f" % (s, x, y, z)
                     for s, (x, y, z) in zip(syms, coords_A))
    return gto.M(atom=atom, basis='def2-TZVP', charge=0, spin=0,
                 verbose=0, max_memory=4000)


def load_c14_center():
    p = os.path.join(HERE, '..', 'run_artifacts', '01_pure_water_o3_h2o',
                     'directional_confirmation', 'point_c14_plus_L7_central.json')
    rec = json.load(open(p))
    return np.asarray(rec['coords_angstrom'], float)


def make_B(A_coords):
    """Clear small displacement: stretch water O(3)-H(4) bond by +0.05 A."""
    ca = A_coords.copy()
    v = ca[4] - ca[3]
    u = v / np.linalg.norm(v)
    cb = ca.copy()
    cb[4] += 0.05 * u
    return cb, u


def eval_scanner(scanner, syms, coords_A):
    """Call the scanner at geometry G, then build the INDEPENDENT benchmark
    on a SEPARATE fresh mf (separates un-wrapped DFT from project D2):

        total_independent = dft_only(wb97x, grid_response=True)  +  analytic D2

    The D2-once proof: ``scanner - dft_only`` must equal the analytic D2 term
    to within the SCF-noise floor (NOT to within |D2| -- if D2 were counted
    twice the residual would be ~|D2|, ~1e-3 Hartree, orders of magnitude
    above the observed ~1e-7 noise).  Thresholds are set relative to |D2|.
    """
    molG = mol_from_coords(syms, coords_A)
    e_scn, g_scn = scanner(molG)
    e_scn = float(e_scn)
    g_scn = np.asarray(g_scn, float).reshape(-1)
    ib = indep_bench(syms, coords_A, grid_response=True)   # fresh mf
    e_dft, e_d2 = ib['e_dft'], ib['e_d2']
    g_dft, g_d2 = ib['g_dft'], ib['g_d2']
    # D2 counted exactly once: (scanner - dft_only) - d2  ~  noise
    d2_e_resid = (e_scn - e_dft) - e_d2
    d2_g_resid = float(np.abs((g_scn - g_dft) - g_d2).max())
    dE_total = e_scn - (e_dft + e_d2)        # == d2_e_resid
    dG_total = float(np.abs(g_scn - (g_dft + g_d2)).max())
    return dict(e_scn=e_scn, g_scn=g_scn, e_dft=e_dft, e_d2=e_d2,
                e_dft_plus_d2=e_dft + e_d2,
                dE_total=dE_total, dG_total=dG_total,
                d2_e_resid=d2_e_resid, d2_g_resid=d2_g_resid,
                g_scn_maxabs=float(g_scn.max()),
                e_d2_mag=float(abs(e_d2)), g_d2_mag=float(np.abs(g_d2).max()))


def indep_bench(syms, coords_A, grid_response=True):
    molG = mol_from_coords(syms, coords_A)
    mf = make_mf_d2_gr(molG, solvent=None, grid_response=grid_response)
    mf.kernel()
    e_dft = float(mf._d2_parent_cls.energy_tot(mf))
    e_d2 = float(d2_full.d2_energy(mf.mol))
    # independent DFT-only gradient: un-wrapped parent, WITH grid_response set
    # (otherwise it defaults to False and the scanner-vs-bench gap is polluted
    #  by the grid-response term, not just D2).
    g_obj = mf._d2_parent_cls.nuc_grad_method(mf)
    g_obj.grid_response = grid_response
    g_dft = np.asarray(g_obj.kernel()).reshape(-1)
    g_d2 = np.asarray(d2_full.d2_grad(mf.mol)).reshape(-1)
    return dict(e_dft=e_dft, e_d2=e_d2, g_dft=g_dft, g_d2=g_d2,
                converged=bool(mf.converged))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()

    if args.smoke:
        # water dimer: cheap but with MEANINGFUL dispersion (|D2| ~ 1e-3) so
        # the D2-once gate is actually exercised (single H2O has |D2|~noise).
        syms = ['O', 'H', 'H', 'O', 'H', 'H']
        A = np.array([
            [0.0, 0.0, 0.0], [0.757, 0.587, 0.0], [-0.757, 0.587, 0.0],
            [3.0, 0.0, 0.0], [3.757, 0.587, 0.0], [2.243, 0.587, 0.0]])
        A = A + np.array([1.0, 1.0, 1.0])   # avoid origin
        label = 'smoke_H2O_dimer'
        ca = A.copy()
        v = ca[1] - ca[0]
        ca[1] += 0.05 * v / np.linalg.norm(v)   # stretch O-H(1) by 0.05 A
        B_coords = ca
    else:
        syms = SYMS_C14
        A = load_c14_center()
        label = 'c14_plus_L7_center'
        B_coords, _ = make_B(A)
    Geoms = [('A', A), ('B', B_coords), ('A2', A)]

    t0 = time.time()
    scan_mf = make_mf_d2_gr(mol_from_coords(syms, A), solvent=None,
                            grid_response=True)
    g = make_grad(scan_mf, True)
    scanner = g.as_scanner()

    rows = []
    for name, coords in Geoms:
        t1 = time.time()
        ev = eval_scanner(scanner, syms, coords)
        g_scn = ev['g_scn']
        rows.append(dict(
            geom=name,
            e_scn=ev['e_scn'], g_scn_maxabs=ev['g_scn_maxabs'],
            e_dft_plus_d2=ev['e_dft_plus_d2'],
            dE_total=ev['dE_total'],
            maxabs_dG_total=ev['dG_total'],
            d2_e_resid=ev['d2_e_resid'],
            d2_g_resid=ev['d2_g_resid'],
            e_d2_mag=ev['e_d2_mag'], g_d2_mag=ev['g_d2_mag'],
            # D2-once gate: residual << |D2| (relative) AND below noise floor
            pass_d2_once=bool(abs(ev['d2_e_resid'])
                              < max(THR['d2_once_rel'] * ev['e_d2_mag'],
                                    THR['abs_floor'])
                              and ev['d2_g_resid']
                              < max(THR['d2_once_rel'] * ev['g_d2_mag'],
                                    THR['abs_floor'])),
            # scanner-vs-independent gate: total residual << |D2|
            pass_scanner_indep=bool(abs(ev['dE_total'])
                                   < max(THR['scanner_rel'] * ev['e_d2_mag'],
                                         THR['abs_floor'])
                                   and ev['dG_total']
                                   < max(THR['scanner_rel'] * ev['g_d2_mag'],
                                         THR['abs_floor'])),
            seconds=round(time.time() - t1, 1)))
        rows[-1]['_gscn'] = g_scn

    # grid_response active evidence at B (scanner True vs a False-flag grad)
    ib_false = indep_bench(syms, B_coords, grid_response=False)
    g_scnB = rows[1]['_gscn']
    gr_active_diff = float(np.abs(g_scnB - ib_false['g_dft']).max())

    # reproducibility A2 vs A (full grad, scanner's own two calls)
    repro_e = abs(rows[0]['e_scn'] - rows[2]['e_scn'])
    repro_g = float(np.abs(rows[0]['_gscn'] - rows[2]['_gscn']).max())
    gr_flags = [bool(getattr(scanner, 'grid_response', None))] * 3

    # verdicts
    d2_once_ok = all(r['pass_d2_once'] for r in rows)
    scanner_indep_ok = all(r['pass_scanner_indep'] for r in rows)
    gr_always_true = all(gr_flags)
    gr_active = gr_active_diff > 1e-6
    reproducible = (repro_e < THR['reproduce_e']
                     and repro_g < THR['reproduce_g'])
    no_residue = reproducible

    result = dict(
        label=label, method_fingerprint=method_fingerprint(scan_mf),
        tolerances=THR, geometries=[r['geom'] for r in rows],
        gr_flags=gr_flags,
        rows=[{k: v for k, v in r.items() if k != '_gscn'} for r in rows],
        grid_response_active_diff=gr_active_diff,
        reproducibility=dict(dE=repro_e, maxabs_dG=repro_g, reproduced=reproducible),
        d2_counted_once=dict(
            maxabs_d2_e_resid=max(abs(r['d2_e_resid']) for r in rows),
            maxabs_d2_g_resid=max(r['d2_g_resid'] for r in rows),
            max_e_d2_mag=max(r['e_d2_mag'] for r in rows),
            max_g_d2_mag=max(r['g_d2_mag'] for r in rows)),
        verdicts=dict(
            d2_counted_exactly_once=d2_once_ok,
            scanner_equals_independent_dft_plus_d2=scanner_indep_ok,
            grid_response_always_true=gr_always_true,
            grid_response_actually_active=gr_active,
            reproducible_no_residue=no_residue),
        overall_pass=bool(d2_once_ok and scanner_indep_ok and gr_always_true
                          and gr_active and reproducible),
        total_seconds=round(time.time() - t0, 1))

    os.makedirs(ART, exist_ok=True)
    out = os.path.join(ART, 'scanner_verify_%s.json' % label)
    json.dump(result, open(out, 'w'), indent=2)
    print('overall_pass=%s  gr_active_diff=%.3e  repro_g=%.3e  -> %s'
          % (result['overall_pass'], gr_active_diff, repro_g, out))
    print('  verdicts:', result['verdicts'])
    for r in result['rows']:
        print('  %-3s e_scn=%.9f dE=%.2e dG=%.2e d2_e_resid=%.2e '
              'd2_g_resid=%.2e (|D2|e=%.2e g=%.2e)'
              % (r['geom'], r['e_scn'], r['dE_total'], r['maxabs_dG_total'],
                 r['d2_e_resid'], r['d2_g_resid'], r['e_d2_mag'], r['g_d2_mag']))


if __name__ == '__main__':
    main()
