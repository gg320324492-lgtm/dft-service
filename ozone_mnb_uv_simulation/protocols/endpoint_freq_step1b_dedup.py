#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Formal endpoint dedup (JOB-2026-0906-004 Step 1b).

Replaces the Phase-C centroid-only check with the project's formal dedup:
Kabsch rigid-body registration anchored on the O3 frame centroid + equivalent
atom permutations (terminal ozone Os 1<->2, water Hs 4<->5) -- indices
VERIFIED from the actual geometries in endpoints_fixed.json (central O = 0
with two 1.23-1.24 A O-O contacts; water O = 3, H = 4/5 at 0.958-0.960 A).
Reports the BEST mapping (rotation, permutation, per-atom deviations), not
only the minimum RMSD.
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, 'run_artifacts', '01_pure_water_o3_h2o',
                   'endpoint_frequency')
sys.path.insert(0, HERE)
from conformer_utils import kabsch_rotate   # formal project function

O3 = (0, 1, 2)          # verified: central O = 0, terminals = 1, 2
PERMS = [None, [(1, 2)], [(4, 5)], [(1, 2), (4, 5)]]


def rmsd_for(P, Q, perm):
    Qp = Q.copy()
    if perm:
        for i, j in perm:
            Qp[[i, j]] = Qp[[j, i]]
    cP, cQ = P[list(O3)].mean(0), Qp[list(O3)].mean(0)
    R = kabsch_rotate(P[list(O3)] - cP, Qp[list(O3)] - cQ)
    Pa = (P - cP) @ R.T + cQ
    dev = np.sqrt(((Pa - Qp) ** 2).sum(axis=1))     # per-atom distance
    return float(np.sqrt((dev ** 2).mean())), R, Qp, dev


def best_mapping(coords_a, coords_b):
    best = None
    for perm in PERMS:
        rmsd, R, Qp, dev = rmsd_for(coords_a, coords_b, perm)
        if best is None or rmsd < best['rmsd_A']:
            best = dict(rmsd_A=rmsd, permutation=(perm or []),
                        rotation_matrix=R.tolist(),
                        permuted_reference=Qp.tolist(),
                        per_atom_deviation_A=dev.tolist())
    return best


def main():
    fx = json.load(open(os.path.join(OUT, 'endpoints_fixed.json')))
    ca = np.asarray(fx['endpoints']['c14_plus']['coords_angstrom'], float)
    cb = np.asarray(fx['endpoints']['c06_plus']['coords_angstrom'], float)
    m_ab = best_mapping(ca, cb)          # c14 -> onto c06 reference
    m_ba = best_mapping(cb, ca)          # reverse direction (sanity)
    # also compare each endpoint against its OWN pre-optimisation start
    src = os.path.join(ROOT, 'run_artifacts', '01_pure_water_o3_h2o',
                       'phaseC_reoptimization', 'phaseC_summary_attempt2.json')
    pc2 = json.load(open(src))
    res = {}
    for tid, c in (('c14_plus', ca), ('c06_plus', cb)):
        start = np.asarray(next(r for r in pc2['results']
                                if r['candidate'] == tid)['start_coords_angstrom'],
                           float)
        res[tid + '_vs_own_start'] = best_mapping(start, c)
    out = dict(
        job='JOB-2026-0906-004',
        method='Kabsch registration anchored on O3-frame centroid + '
               'equivalent permutations {terminal Os 1<->2, water Hs 4<->5}; '
               'indices verified from geometry (central O=0, 1.23-1.24 A '
               'contacts; water H 4/5 at 0.958-0.960 A)',
        supersedes='phaseC centroid-only RMSD check (no rotation/permutation) '
                   'and any claim of a formal registration there',
        c14_plus_vs_c06_plus=dict(
            c14_as_target=m_ab,
            c06_as_target=m_ba,
            same_structure=bool(m_ab['rmsd_A'] < 0.1),
            note='same_structure iff RMSD ~ 0 after registration+permutation'),
        endpoints_vs_own_starts=res)
    with open(os.path.join(OUT, 'formal_dedup.json'), 'w') as fh:
        json.dump(out, fh, indent=2)
    print('c14 vs c06: RMSD=%.4f A (perm=%s) same=%s'
          % (m_ab['rmsd_A'], m_ab['permutation'],
             out['c14_plus_vs_c06_plus']['same_structure']))
    print('reverse: RMSD=%.4f A (perm=%s)'
          % (m_ba['rmsd_A'], m_ba['permutation']))
    for k, v in res.items():
        print('%s: RMSD=%.4f A (perm=%s)' % (k, v['rmsd_A'], v['permutation']))


if __name__ == '__main__':
    main()
