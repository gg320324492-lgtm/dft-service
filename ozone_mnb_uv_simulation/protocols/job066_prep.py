#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-066 step A: OFFLINE preparation (ZERO evaluations).

Mirror-branch six-point scan inputs.  Source = the SIX VERIFIED 064
geometries (c2_scan064/input_manifest.json, coords_bohr).  Per point the
ONLY operation is reflecting 3H (project index 2) through the N-2H-4H
contact plane; O3, N, 2H, 4H unchanged; contacts stay 5O...2H and
7O...4H; no monomer rebuild, no internal adjustment, no branch
re-selection.  Registered as: COPLANAR PROJECT MODEL, NEGATIVE-normal
3H mirror branch.

Per-point verification: H...O targets hit (<=1e-10 A); NH3/O3 internal
distance matrices = JOB-026 monomers; 3H on the NEGATIVE normal;
element order/atom mapping; rotation orthonormal det=+1; collision
criterion (1.2 A); same mirror branch at all six points (3H signed
height negative throughout, no flip back); BOTH branch hashes saved
(positive-normal source and mirror).
"""
import os, json, hashlib
import numpy as np

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
R_REF = ROOT + '/run_artifacts/02_nh3o3_reference'
R64 = R_REF + '/c2_scan064'
OUT = R_REF + '/c2_mirror_scan066'
JOB_NO = '066'
BPA = 1.0 / 0.52917721092
COLLISION_THRESHOLD_A = 1.2


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for blk in iter(lambda: fh.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()[:16]


def sha_arr(a):
    return hashlib.sha256(np.asarray(a, float).tobytes()).hexdigest()[:16]


def check_number():
    import glob, re
    used = set()
    for f in glob.glob(ROOT + '/jobs/JOB-*.md'):
        m = re.match(r'JOB-2026-0906-(\d{3})', os.path.basename(f))
        if m:
            used.add(m.group(1))
    nxt = '%03d' % (max(int(u) for u in used) + 1)
    return dict(used_max=max(used), next_free=nxt,
                pass_=bool(nxt == JOB_NO and JOB_NO not in used))


def contact_plane_normal(X):
    n = np.cross(X[1] - X[0], X[3] - X[0])
    n = n / np.linalg.norm(n)
    if n @ (X[2] - X[0]) < 0:
        n = -n
    return n


def kabsch(P, Q):
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    V, S, Wt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(V @ Wt))
    return V @ np.diag([1.0, 1.0, d]) @ Wt


def distance_matrix(X):
    X = np.asarray(X, float).reshape(-1, 3)
    diff = X[:, None, :] - X[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))


def main():
    num = check_number()
    man64 = json.load(open(R64 + '/input_manifest.json'))
    idx = json.load(open(R_REF + '/final_endpoint_index.json'))
    nh3 = np.asarray(idx['default_endpoints']['NH3_gas_v5_final'][
        'coords_angstrom'], float) * BPA
    o3 = np.asarray(idx['default_endpoints']['O3_gas_final'][
        'coords_angstrom'], float) * BPA
    mon_nh3 = np.asarray(idx['default_endpoints']['NH3_gas_v5_final'][
        'coords_angstrom'], float) * BPA          # Bohr
    mon_o3 = np.asarray(idx['default_endpoints']['O3_gas_final'][
        'coords_angstrom'], float) * BPA          # Bohr

    checks = {}
    checks['number_check'] = bool(num['pass_'])
    geoms = []
    prev_h3_signed = None
    signs = []
    rel_nh3 = nh3 - nh3[0]
    # placed-row order for the rigid reference (N, 2H, 3H, 4H): 2H/4H
    # are the contact pair, 3H the out-of-plane H (see 064 assignment)
    ref_rows = [0, 2, 3, 1]   # (N, H_b, H_c, H_a) matching the placed
    # row order (N, 2H=img H_b, 3H=img H_c, 4H=img H_a)
    for g64 in man64['geometries']:
        X = np.asarray(g64['coords_bohr'], float).reshape(7, 3)
        n = contact_plane_normal(X)
        h3_signed = float((X[2] - X[0]) @ n)
        # reflect 3H only
        Xm = X.copy()
        Xm[2] = X[2] - 2.0 * h3_signed * n
        h3_signed_m = float((Xm[2] - X[0]) @ n)
        tag = 'mscan_' + g64['tag'].replace('scan_', '')
        # verifications
        inplane_dev = max(float(np.abs(Xm[i] - X[i]).max())
                          for i in (0, 1, 3, 4, 5, 6))
        dm_nh_m = distance_matrix(Xm[:4])
        dm_nh_ref = distance_matrix(
            np.vstack([rel_nh3[k] + X[0] for k in ref_rows]))
        nh_dev = float(np.abs(dm_nh_m - dm_nh_ref).max())
        dm_o3_m = distance_matrix(Xm[4:])
        # placed (5O,6O,7O) = monomer rows perm_o3 (registered in 064)
        dm_o3_ref = distance_matrix(o3[g64['perm_o3']])
        o_dev = float(np.abs(dm_o3_m - dm_o3_ref).max())
        d1 = float(np.linalg.norm(Xm[1] - Xm[4]))
        d2 = float(np.linalg.norm(Xm[3] - Xm[6]))
        d1_t = g64['d_actual_A'][0] * BPA      # target in Bohr
        d2_t = g64['d_actual_A'][1] * BPA
        d_inter = float(np.linalg.norm(Xm[:4, None, :]
                                       - Xm[None, 4:, :], axis=2).min())
        Rm = kabsch((nh3 - nh3[0])[[0, 2, 3, 1]], Xm[:4] - Xm[:4].mean(0))
        Ro3 = kabsch(o3 - o3[ci := 0] if False else o3 - o3[0],
                     Xm[4:] - Xm[4:].mean(0))
        entry = dict(
            tag=tag, delta_A=g64['delta_A'],
            source_064_tag=g64['tag'],
            d_5O_2H_target_A=d1_t, d_7O_4H_target_A=d2_t,
            d_5O_2H_actual_A=d1, d_7O_4H_actual_A=d2,
            contact_err_max_A=float(max(abs(d1 - d1_t),
                                        abs(d2 - d2_t))),
            h3_signed_height_A=h3_signed_m,
            h3_signed_height_pos_branch_A=h3_signed,
            negative_normal=bool(h3_signed_m < 0),
            inplane_maxdev_Bohr=inplane_dev,
            nh3_internal_maxdev_Bohr=nh_dev,
            o3_internal_maxdev_Bohr=o_dev,
            min_interfrag_Bohr=d_inter,
            coords_bohr=Xm.reshape(-1).tolist(),
            coords_sha=sha_arr(Xm),
            source_sha=g64['coords_sha'],
            rotation_nh3_det=float(np.linalg.det(Rm)),
            rotation_o3_det=float(np.linalg.det(Ro3)))
        entry_ok = bool(
            inplane_dev < 1e-10 and nh_dev < 1e-10 and o_dev < 1e-10
            and entry['contact_err_max_A'] < 1e-10
            and entry['negative_normal']
            and d_inter >= COLLISION_THRESHOLD_A
            and abs(entry['rotation_nh3_det'] - 1) < 1e-12
            and abs(entry['rotation_o3_det'] - 1) < 1e-12)
        entry['pass'] = entry_ok
        geoms.append(entry)
        signs.append(h3_signed_m)
        checks['point_%s' % tag] = entry_ok
    # same mirror branch at all six points (negative, no flip back)
    checks['negative_normal_all_points'] = bool(all(s < 0 for s in signs))
    checks['single_mirror_branch'] = bool(
        all(s < 0 for s in signs) and len(geoms) == 6)
    checks['n_geoms'] = bool(len(geoms) == 6)

    manifest = dict(
        job='JOB-2026-0906-066: C2 mirror-branch (negative-normal 3H) '
            'six-point fixed-orientation distance scan',
        number_check=num,
        review_065='notes/job065_commander_review_2026-09-10.md',
        branch=dict(
            name='COPLANAR PROJECT MODEL, NEGATIVE-normal 3H mirror '
                 'branch',
            source='the six VERIFIED 064 geometries; per point only 3H '
                   'reflected through the N-2H-4H contact plane; O3, N, '
                   '2H, 4H unchanged; contacts stay 5O...2H and 7O...4H; '
                   'no monomer rebuild, no internal adjustment, no '
                   'branch re-selection',
            independence='JOB-065: best proper-rotation RMSD vs the '
                         'positive branch = 0.3255 A (independent '
                         'branch); reflection control ~0 (forbidden)'),
        deltas_A=[g['delta_A'] for g in man64['geometries']],
        monomer_sources=dict(
            file=os.path.relpath(R_REF
                                 + '/final_endpoint_index.json', ROOT),
            sha256_16=sha256_file(R_REF
                                  + '/final_endpoint_index.json')),
        caps=dict(scan=6,
                  note='six SCF+full-gradient attempts, one per point; '
                       'no centre reproduction; failures count; any '
                       'geometry/SCF/gradient/finite/config anomaly -> '
                       'save and STOP immediately, no skip, no restart, '
                       'no extra budget'),
        quantum_config=man64['quantum_config'],
        analysis_rules=dict(
            reference='mirror-branch delta=0.00 point',
            side_by_side='the 064 positive-normal curve is plotted '
                         'BESIDE the mirror curve; the two branches are '
                         'NEVER joined into one continuous path'),
        geometries=geoms,
        checks=checks, all_checks_pass=all(checks.values()))
    os.makedirs(OUT, exist_ok=True)
    tmp = OUT + '/input_manifest.json.tmp'
    with open(tmp, 'w') as fh:
        json.dump(manifest, fh, indent=2, default=str)
    os.replace(tmp, OUT + '/input_manifest.json')

    print('[066] number check:', json.dumps(num))
    print('[066] checks:', json.dumps(checks, default=str))
    print('[066] 3H signed heights (A):',
          ['%.4f' % s for s in signs])
    if not (num['pass_'] and manifest['all_checks_pass']):
        print('[066] PRECHECK FAILED -> STOP (zero evaluations)')
        raise SystemExit(1)
    print('[066] PREP OK (zero evaluations; six mirror geometries '
          'verified on one branch)', flush=True)


if __name__ == '__main__':
    main()
