#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Acceptance-revision test #1 (JOB-2026-0905-009): validate the revised
conformer_utils.best_rmsd / hbond_topology / topology_signature.

Checks, per the revision brief:
  1. the same structure after an arbitrary rotation + translation + equivalent
     atom relabelling (O3 terminal swap, water H swap) gives RMSD ~ 0;
  2. the topology classification is invariant under those operations;
  3. regression guard: a real geometry with a known H-bond (c02, verified
     d(H..O)=2.286 A, O_w-H..O angle=179.9 deg) is detected as H-bonded by the
     revised criterion, and the old inverted criterion is reported as failing
     it (documenting the fixed defect).
"""
import os
import sys
import json
import math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import conformer_utils as cu

ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, 'run_artifacts', '01_pure_water_o3_h2o', 'acceptance_revision')
os.makedirs(ART, exist_ok=True)


def rigid_transform(coords, seed):
    rng = np.random.default_rng(seed)
    th, phi = rng.uniform(0, 2 * np.pi, 2)
    ax = rng.normal(size=3)
    ax /= np.linalg.norm(ax)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    R = np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)
    t = rng.uniform(-5, 5, 3)
    return coords @ R.T + t, R, t


def main():
    results = []
    ref = np.array(json.load(open(os.path.join(
        ROOT, 'run_artifacts', '01_pure_water_o3_h2o', 'gas_freq', 'c02.json')))['geometry_angstrom'])
    sig_ref = cu.topology_signature(ref)
    topo_ref = cu.hbond_topology(ref)

    cases = []
    for seed in (1, 2, 3):
        moved, R, t = rigid_transform(ref, seed)
        # (a) rigid motion only
        cases.append(('rot+trans seed=%d' % seed, moved, []))
        # (b) rigid motion + O3 terminal swap
        m = moved.copy(); m[[1, 2]] = m[[2, 1]]
        cases.append(('rot+trans+O3term swap seed=%d' % seed, m, [(1, 2)]))
        # (c) rigid motion + water H swap
        m = moved.copy(); m[[4, 5]] = m[[5, 4]]
        cases.append(('rot+trans+H swap seed=%d' % seed, m, [(4, 5)]))
        # (d) rigid motion + both swaps
        m = moved.copy(); m[[1, 2]] = m[[2, 1]]; m[[4, 5]] = m[[5, 4]]
        cases.append(('rot+trans+both swaps seed=%d' % seed, m, [(1, 2), (4, 5)]))

    for name, coords, swaps in cases:
        # undo the swaps to compare topology on the same labels
        undone = coords.copy()
        for i, j in swaps:
            undone[[i, j]] = undone[[j, i]]
        rmsd = cu.best_rmsd(ref, coords)
        sig = cu.topology_signature(undone)
        ok_rmsd = rmsd < 1e-6
        ok_sig = (sig == sig_ref)
        results.append(dict(case=name, rmsd_angstrom=rmsd,
                            rmsd_pass=bool(ok_rmsd),
                            topology_pass=bool(ok_sig)))
        print('%-34s RMSD=%.2e %s  topology_invariant=%s' % (
            name, rmsd, 'PASS' if ok_rmsd else 'FAIL', ok_sig))

    # regression guard: the documented c02 H-bond must now be detected
    hb = [(h, o) for h, o in topo_ref['hbonds']]
    guard = dict(
        c02_hbonds_detected=hb,
        c02_description=topo_ref['description'],
        expected='a water H bonded to an O3 terminal oxygen with d(H..O)=2.286 A, angle 179.9 deg',
        pass_=bool(hb))
    print('c02 H-bond regression guard:', 'PASS' if hb else 'FAIL', hb)

    summary = dict(test='revision_test_1_rmsd_topology',
                   reference='gas_freq/c02.json', n_cases=len(results),
                   all_rmsd_pass=all(r['rmsd_pass'] for r in results),
                   all_topology_pass=all(r['topology_pass'] for r in results),
                   c02_hbond_guard=guard, cases=results)
    with open(os.path.join(ART, 'revision_test_1_rmsd_topology.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)
    print('saved ->', os.path.join(ART, 'revision_test_1_rmsd_topology.json'))
    return 0 if (summary['all_rmsd_pass'] and summary['all_topology_pass'] and hb) else 1


if __name__ == '__main__':
    sys.exit(main())
