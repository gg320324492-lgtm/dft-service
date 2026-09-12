#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Endpoint-freq batch Step 1 (JOB-2026-0906-004):
fix the official endpoints from phaseC_summary_attempt2.json, record
source/hash/order/units/fingerprint, VERIFY the real ozone central/terminal
and water-H indices from the actual geometries, and run the independent
centre check (L7 energy + FULL 6x3 grid_response=True gradient) at each
endpoint.  Gate: SCF converged, finite, max|g| <= 1e-5 Eh/Bohr.
"""
import os, sys, json, hashlib, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, 'run_artifacts', '01_pure_water_o3_h2o')
OUT = os.path.join(ART, 'endpoint_frequency')
sys.path.insert(0, HERE)

from pyscf import gto
from grad_factory import make_mf_d2_gr, method_fingerprint

SYMS = ['O', 'O', 'O', 'O', 'H', 'H']
GMAX = 1e-5


def sha256_16(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()[:16]


def verify_indices(c):
    """Verify ozone central/terminal and water indices from bond lengths."""
    c = np.asarray(c, float)
    dO = lambda i, j: float(np.linalg.norm(c[i] - c[j]))
    pairs = {(i, j): dO(i, j) for i in range(3) for j in range(i + 1, 3)}
    pairs = {(i, j): dO(i, j) for i in range(3) for j in range(i + 1, 3)}
    # central ozone O = the one with TWO short O-O contacts (< 1.6 A)
    central = None
    for k in range(3):
        others = [l for l in range(3) if l != k]
        if all(dO(k, l) < 1.6 for l in others):
            central = k
    term = [l for l in range(3) if l != central]
    ow = 3
    dH = [dO(ow, 4), dO(ow, 5)]
    # H bonded to water O (short O-H)
    h_bonded = [4, 5]
    return dict(
        ozone_central_O=central, ozone_terminal_Os=term,
        o_o_distances={'%d-%d' % k: round(v, 4) for k, v in pairs.items()},
        water_O=ow, water_Hs=h_bonded, o_h_distances=[round(x, 4) for x in dH],
        intermolecular_Ow_O3_centroid_A=round(
            float(np.linalg.norm(c[ow] - c[:3].mean(0))), 4))


def centre_check(tid, coords_A):
    atom = "; ".join("%s %.10f %.10f %.10f" % (s, x, y, z)
                     for s, (x, y, z) in zip(SYMS, coords_A))
    mol = gto.M(atom=atom, basis='def2-TZVP', charge=0, spin=0,
                verbose=0, max_memory=4000)
    mf = make_mf_d2_gr(mol, solvent=None, grid_response=True)
    mf.kernel()
    g = np.asarray(mf.nuc_grad_method().kernel(), float).reshape(6, 3)
    import d2_full
    e_d2 = float(d2_full.d2_energy(mol))
    return dict(
        e_total=float(mf.e_tot), e_d2_analytic=e_d2,
        e_dft_part=float(mf.e_tot) - e_d2,
        d2_in_scf_summary=float(mf.scf_summary.get('d2_dispersion')),
        gradient_full_6x3=g.tolist(),
        grad_max=float(np.abs(g).max()),
        grad_rms=float(np.sqrt((g ** 2).mean())),
        scf_converged=bool(mf.converged),
        finite=bool(np.isfinite(mf.e_tot) and np.isfinite(g).all()),
        n_grid_points=int(mf.grids.weights.size))


def main():
    os.makedirs(OUT, exist_ok=True)
    src = os.path.join(ART, 'phaseC_reoptimization',
                       'phaseC_summary_attempt2.json')
    pc2 = json.load(open(src))
    rec = dict(
        job='JOB-2026-0906-004',
        source_file=os.path.relpath(src, ROOT),
        source_hash=sha256_16(src),
        source_note='official endpoints = phaseC attempt2 independent-'
                    'verification endpoints (NOT attempt1, NOT old geometries)',
        atom_order=SYMS,
        atom_order_note='SYMS convention of all Phase-A/B/C batches: '
                        'indices 0-2 = ozone OOO, 3 = water O, 4-5 = water H',
        coords_unit='Angstrom',
        method_fingerprint=method_fingerprint(
            make_mf_d2_gr(gto.M(atom='O 0 0 0; H 0 0 1.1; H 0 0 -1.1',
                                basis='def2-TZVP', verbose=0,
                                max_memory=2000), grid_response=True)),
        endpoints={})
    for tid in ('c14_plus', 'c06_plus'):
        cand = next(r for r in pc2['results'] if r['candidate'] == tid)
        coords = np.asarray(cand['endpoint_coords_angstrom'], float)
        idx = verify_indices(coords)
        print('[%s] indices: ozone central=%d terminal=%s  O-O=%s  '
              'O-H(w)=%s  gap(Ow-O3c)=%.3f A'
              % (tid, idx['ozone_central_O'], idx['ozone_terminal_Os'],
                 idx['o_o_distances'], idx['o_h_distances'],
                 idx['intermolecular_Ow_O3_centroid_A']), flush=True)
        t0 = time.time()
        cc = centre_check(tid, coords)
        gm = cc['grad_max']
        gate = bool(cc['scf_converged'] and cc['finite'] and gm <= GMAX)
        print('[%s] centre check: E=%.9f  max|g|=%.3e  rms|g|=%.3e  '
              'conv=%s  ngrid=%d  gate=%s (%.0f s)'
              % (tid, cc['e_total'], gm, cc['grad_rms'], cc['scf_converged'],
                 cc['n_grid_points'], gate, time.time() - t0), flush=True)
        rec['endpoints'][tid] = dict(
            coords_angstrom=coords.tolist(),
            source_step_count=cand['n_steps'],
            source_independent_max_g=cand['independent_verification']['grad_max'],
            indices=idx,
            centre_check=cc, centre_gate_pass=gate)
    with open(os.path.join(OUT, 'endpoints_fixed.json'), 'w') as fh:
        json.dump(rec, fh, indent=2)
    print('all gates pass:', all(r['centre_gate_pass']
                                 for r in rec['endpoints'].values()))
    print('saved ->', os.path.join(OUT, 'endpoints_fixed.json'))


if __name__ == '__main__':
    main()
