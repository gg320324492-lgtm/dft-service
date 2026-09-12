#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-065: C2 mirror-branch equivalence check + JOB-064 report
corrections.

ZERO quantum evaluations.  Offline only:

(a) 064 report corrections (from the commander review):
    - the six-point table is regenerated WITH ALL SIX ROWS (the report
      text had omitted delta=-0.10 and +0.10; the JSON/ledger always had
      them);
    - the minimum interfragment distances are corrected to the manifest
      measured values 1.5427/1.6086/1.6779/1.7501/1.9023/2.1454 A;
    - the collision statement becomes "passes the CURRENT collision
      criterion (threshold 1.2 A)" - 1.54 A is NOT called roomy;
    - scope restricted to the coplanar project model, POSITIVE-normal
      3H branch; the mirror branch was NOT computed;
    - a machine-readable correction note is added additively to the 064
      directory (original quantum data and ledger untouched);
(b) mirror construction: from the 064 delta=0 ACTUAL geometry, 3H is
    reflected through the N-2H-4H contact plane (O3, N, 2H, 4H
    unchanged; internal and contact distances unchanged); the mirror
    coordinates are saved as an OFFLINE CANDIDATE ONLY;
(c) equivalence enumeration: 6 NH3-H permutations x 2 end-O swaps,
    PROPER rotations only (Kabsch, det=+1); mirror transformations are
    NOT allowed to impersonate rotations (a deliberate improper-
    transform control is computed and rejected);
(d) verdict: symmetric-equivalent (RMSD ~ 0) / independent mirror
    branch (RMSD clearly nonzero) / cannot-determine.
"""
import os, sys, json, hashlib, itertools, types
import numpy as np

sys.modules['pyscf'] = None
sys.modules['pyscf.hessian'] = None
sys.modules['d2_full'] = types.ModuleType('d2_full')

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
R_REF = ROOT + '/run_artifacts/02_nh3o3_reference'
R64 = R_REF + '/c2_scan064'
OUT = R_REF + '/c2_mirror065'
BPA = 1.0 / 0.52917721092
SYMS = ['N', 'H', 'H', 'H', 'O', 'O', 'O']


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for blk in iter(lambda: fh.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()[:16]


def save_json(path, obj):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(obj, fh, indent=2, default=str)
    os.replace(tmp, path)


def check_number():
    import glob, re
    used = set()
    for f in glob.glob(ROOT + '/jobs/JOB-*.md'):
        m = re.match(r'JOB-2026-0906-(\d{3})', os.path.basename(f))
        if m:
            used.add(m.group(1))
    nxt = '%03d' % (max(int(u) for u in used) + 1)
    return dict(used_max=max(used), next_free=nxt,
                pass_=bool(nxt == '065' and '065' not in used))


def kabsch(P, Q):
    """Best PROPER rotation (det=+1) mapping P onto Q (both centered)."""
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    V, S, Wt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(V @ Wt))
    return V @ np.diag([1.0, 1.0, d]) @ Wt


def rmsd_after(R, P, Q):
    return float(np.sqrt((np.linalg.norm((P @ R.T) - Q, axis=1) ** 2)
                         .mean()))


def contact_plane_normal(X):
    """Unit normal of the plane through N(0), 2H(1), 4H(3), signed
    toward 3H (index 2)."""
    n = np.cross(X[1] - X[0], X[3] - X[0])
    n = n / np.linalg.norm(n)
    if n @ (X[2] - X[0]) < 0:
        n = -n
    return n


def save_xyz(path, comment, coords_bohr):
    with open(path, 'w') as fh:
        fh.write('7\n%s\n' % comment)
        for s, r in zip(SYMS, np.asarray(coords_bohr, float)):
            fh.write('%s %.12f %.12f %.12f\n'
                     % (s, r[0], r[1], r[2]))


def main():
    num = check_number()
    os.makedirs(OUT, exist_ok=True)

    # ---- source: the 064 delta=0 ACTUAL geometry ------------------------
    src_path = R64 + '/eval_scan_d+00.json'
    rec0 = json.load(open(src_path))
    X0 = np.asarray(rec0['coords_actual_angstrom'], float)  # Angstrom
    man64 = json.load(open(R64 + '/input_manifest.json'))
    g0 = [g for g in man64['geometries']
          if abs(g['delta_A']) < 1e-12][0]

    # ---- (a) 064 corrections (additive; originals untouched) ------------
    mins = [g['min_interfrag_A'] for g in man64['geometries']]
    corr = dict(
        job='JOB-064 corrections (issued with JOB-065 prep)',
        review='notes/job064_commander_review_2026-09-10.md',
        kept='six points all completed SCF+full gradient (6/6, 0 '
             'failures); energies strictly monotonic decreasing over '
             'delta=-0.20..+0.60; five negative secant slopes, zero '
             'sign changes; no radial minimum within the sampled range; '
             'delta=+0.60 still +1.53e-2 Eh above the monomer sum; '
             'original quantum data and ledger untouched',
        corrections=[
            dict(item='the report table omitted delta=-0.10 and +0.10',
                 fix='full six-row table regenerated from '
                     'c2_scan064_results.json (the JSON and ledger '
                     'always contained all six)'),
            dict(item='minimum interfragment distances corrected',
                 actual_A=mins,
                 range_A=[min(mins), max(mins)],
                 withdrawn='"1.68-2.53 A" (inaccurate)'),
            dict(item='collision statement narrowed',
                 fix='the six points PASS the current collision '
                     'criterion (threshold 1.2 A); 1.5427 A is NOT '
                     'called roomy or collision-free in any absolute '
                     'sense'),
            dict(item='branch scope restricted',
                 fix='results apply ONLY to the coplanar project model, '
                     'POSITIVE-normal 3H branch; the mirror branch was '
                     'not computed'),
        ])
    tmp = R64 + '/job064_correction_note.json.tmp'
    with open(tmp, 'w') as fh:
        json.dump(corr, fh, indent=2, default=str)
    os.replace(tmp, R64 + '/job064_correction_note.json')

    # ---- (b) mirror construction ----------------------------------------
    n = contact_plane_normal(X0)
    Xm = X0.copy()
    Xm[2] = X0[2] - 2.0 * ((X0[2] - X0[0]) @ n) * n   # reflect 3H
    # in-plane atoms (N, 2H, 4H, O3) exactly unchanged?
    inplane_idx = [0, 1, 3, 4, 5, 6]
    inplane_dev = float(max(float(np.abs(Xm[i] - X0[i]).max())
                            for i in inplane_idx))
    h3_disp = float(np.linalg.norm(Xm[2] - X0[2]))
    # internal NH3 distances preserved?
    dm0 = np.linalg.norm(X0[:4, None, :] - X0[None, :4, :], axis=2)
    dmm = np.linalg.norm(Xm[:4, None, :] - Xm[None, :4, :], axis=2)
    nh_dev = float(np.abs(dm0 - dmm).max())
    # contact distances unchanged?
    c0 = (float(np.linalg.norm(X0[1] - X0[4])),
          float(np.linalg.norm(X0[3] - X0[6])))
    cm = (float(np.linalg.norm(Xm[1] - Xm[4])),
          float(np.linalg.norm(Xm[3] - Xm[6])))

    save_xyz(OUT + '/c2_posnormal_branch_delta0.xyz',
             'JOB-064 delta=0 actual geometry (positive-normal 3H '
             'branch); order N,H,H,H,O,O,O', X0 * BPA)
    save_xyz(OUT + '/c2_mirror_candidate_delta0.xyz',
             'OFFLINE CANDIDATE ONLY (JOB-065): 3H-reflected mirror of '
             'the delta=0 geometry; NOT a literature structure; NOT '
             'authorized as a quantum-chemistry input', Xm * BPA)

    # ---- (c) equivalence enumeration -------------------------------------
    # allowed correspondences: NH3 H permutations (rows 1,2,3) x end-O
    # swap (rows 4,6); N(0) and central O(5) fixed.
    best = None
    table = []
    for hperm in itertools.permutations([1, 2, 3]):
        for oswap in (False, True):
            omap = {4: 6, 6: 4} if oswap else {4: 4, 6: 6}
            perm = [0, hperm[0], hperm[1], hperm[2],
                    omap[4], 5, omap[6]]
            # P (mirror rows reordered) -> Q (original rows)
            Q = X0[perm]
            R = kabsch(Xm, Q)
            r = rmsd_after(R, Xm, Q)
            det = float(np.linalg.det(R))
            # contact preservation under this correspondence
            # mirror contact 5O...2H (rows 4,1) maps to original pair
            # (perm[4], perm[1])
            d_map_1 = float(np.linalg.norm(Q[4] - Q[1]))
            d_map_2 = float(np.linalg.norm(Q[6] - Q[3]))
            table.append(dict(
                perm_nh3_rows=list(hperm), endO_swap=oswap,
                mapped_pairs=dict(orig_pair_for_5O_2H=[
                    perm[4], perm[1]], orig_pair_for_7O_4H=[
                    perm[6], perm[3]]),
                contact_distances_on_original_A=[d_map_1, d_map_2],
                rmsd_A=r, det=det))
            if det > 0 and (best is None or r < best['rmsd_A']):
                best = dict(perm_nh3_rows=list(hperm), endO_swap=oswap,
                            perm=perm, rmsd_A=r, det=det,
                            rotation=R.tolist(),
                            contact_pair_5O2H_maps_to=[perm[4], perm[1]],
                            contact_pair_7O4H_maps_to=[perm[6], perm[3]],
                            contacts_preserved=bool(
                                abs(d_map_1 - c0[0]) < 1e-10
                                and abs(d_map_2 - c0[1]) < 1e-10))
    table.sort(key=lambda t: t['rmsd_A'])

    # improper-transform control (FORBIDDEN, computed only to show the
    # distinction): a mirror operation would give ~0 RMSD trivially
    R_improper = np.diag([1.0, 1.0, -1.0])
    rmsd_improper = rmsd_after(R_improper, Xm, X0)

    if best and best['rmsd_A'] < 1e-6:
        verdict = 'symmetric_equivalent'
    elif best and best['rmsd_A'] > 0.05:
        verdict = 'independent_mirror_branch'
    else:
        verdict = 'cannot_determine'

    results = dict(
        job='JOB-2026-0906-065 C2 mirror-branch equivalence check + '
            'JOB-064 report corrections',
        number_check=num,
        review_064='notes/job064_commander_review_2026-09-10.md',
        budget=dict(all_quantum_calls=0),
        corrections_064=corr,
        mirror_construction=dict(
            source='eval_scan_d+00.json (the 064 delta=0 ACTUAL '
                   'geometry)',
            operation='3H (project index 2) reflected through the '
                      'N-2H-4H contact plane; O3, N, 2H, 4H unchanged',
            inplane_maxdev_A=inplane_dev,
            h3_displacement_A=h3_disp,
            nh3_internal_maxdev_A=nh_dev,
            contacts_A=dict(before=c0, after=cm),
            status='OFFLINE CANDIDATE ONLY - not a literature '
                   'structure, not authorized as a quantum input'),
        equivalence=dict(
            enumerations='6 NH3-H permutations x 2 end-O swaps = 12 '
                         'correspondences, PROPER rotations only '
                         '(Kabsch, det=+1); improper transforms '
                         'rejected as impersonation',
            improper_control_rmsd_A=rmsd_improper,
            best=best,
            table=table,
            verdict=verdict,
            interpretation={
                'symmetric_equivalent':
                    'the mirror branch is the same configuration under '
                    'legal relabeling + proper rotation; rescanning it '
                    'would duplicate the 064 energies',
                'independent_mirror_branch':
                    'the mirror branch is a distinct configuration; '
                    'whether to scan it is a COMMANDER decision',
                'cannot_determine':
                    'the evidence does not allow a unique judgment; no '
                    'scan is selected',
            }[verdict])
        )

    save_json(os.path.join(OUT, 'c2_mirror065_results.json'), results)
    print('[065] mirror rmsd (best proper) = %.6e A | verdict: %s'
          % (best['rmsd_A'] if best else float('nan'), verdict))
    print('[065] improper control rmsd = %.4f A (rejected as '
          'impersonation)' % rmsd_improper)
    print('[065] 064 corrections written (additive)', flush=True)


if __name__ == '__main__':
    main()
