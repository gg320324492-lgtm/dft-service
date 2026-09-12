#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-064 step A: OFFLINE preparation (ZERO evaluations).

C2 coplanar-assumption model construction + six-point fixed-orientation
scan inputs.

REGISTERED MODEL ASSUMPTIONS (authorized by the commander in the JOB-064
instruction, following the JOB-063 review requirement that this choice
NOT be made by the executor):
  1. the O3 plane coincides with the N-2H-4H contact plane;
  2. 3H lies on the POSITIVE normal side of that plane;
  3. the 3H mirror image is NOT computed in this batch;
  4. contacts fixed: 5O...2H and 7O...4H;
  5. the figure distances are PROJECT INITIAL-GUESS constraints only,
     not literature 3D coordinates;
  6. NH3 and O3 internal geometry = the JOB-026 accepted monomers,
     rigid rotation+translation ONLY.

Six scan points: (d_5O-2H, d_7O-4H) = (2.548+delta, 2.448+delta) Å with
delta in {-0.20,-0.10,0.00,+0.10,+0.30,+0.60}; identical orientation,
only the overall fragment separation changes.  Placement = the same
closed-form circle-circle method as JOB-063 (proper rotations, R^T R=I,
det=+1).  Every point is verified: contact distances hit (<=1e-10 A),
internal distance matrices = JOB-026 monomers under the registered
permutation, element order N,H,H,H,O,O,O, central/terminal mapping,
collision-free, and the six points stay on ONE continuous branch
(consecutive N separations recorded; no mirror flip).  Any failure or
branch ambiguity -> hard stop BEFORE any SCF.

Quantum budget for the batch: exactly 6 SCF+full-gradient attempts
(no centre reproduction, no expansion).
"""
import os, json, hashlib
import numpy as np

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
R_REF = ROOT + '/run_artifacts/02_nh3o3_reference'
OUT = R_REF + '/c2_scan064'
SRC63 = R_REF + '/c2_geometry063/c2_geometry063_results.json'
IDX = R_REF + '/final_endpoint_index.json'
JOB_NO = '064'
BPA = 1.0 / 0.52917721092
SYMS = ['N', 'H', 'H', 'H', 'O', 'O', 'O']
DELTAS = [-0.20, -0.10, 0.00, 0.10, 0.30, 0.60]
D1_BASE, D2_BASE = 2.548, 2.448
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


def identify_central_o(o3_A):
    o = np.asarray(o3_A, float).reshape(3, 3)
    sums = [float(np.linalg.norm(o[i] - o[(i + 1) % 3])
                  + np.linalg.norm(o[i] - o[(i + 2) % 3]))
            for i in range(3)]
    return int(np.argmin(sums)), sums


def kabsch(P, Q):
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    V, S, Wt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(V @ Wt))
    return V @ np.diag([1.0, 1.0, d]) @ Wt


def distance_matrix(X):
    X = np.asarray(X, float).reshape(-1, 3)
    diff = X[:, None, :] - X[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))


def build_frames(mon_nh3, mon_o3):
    """Contact-plane frame from the rigid monomers.

    NH3: normal n = normalize((H2-N) x (H3-N)) signed so that
    n.(H1-N) > 0 (3H on the POSITIVE normal side, assumption 2);
    in-plane basis e1 along (H2-N), e2 = n x e1 (toward 4H).
    Returns the exact in-plane offsets (both contact H at z=0 after the
    plane rotation), the 3H out-of-plane height, and the rotation R_nh3
    that maps the monomer frame onto the contact-plane frame
    (proper: orthonormal, det=+1)."""
    nh = np.asarray(mon_nh3, float).reshape(4, 3)
    N = nh[0]
    h2v, h3v, h1v = nh[1] - N, nh[2] - N, nh[3] - N
    n = np.cross(h2v, h3v)
    n = n / np.linalg.norm(n)
    if n @ h1v < 0:
        n = -n
    e1 = h2v / np.linalg.norm(h2v)
    e2 = np.cross(n, e1)
    # in-plane components in (e1, e2); heights along n
    p2 = (e1 @ h2v, e2 @ h2v, n @ h2v)
    p3 = (e1 @ h3v, e2 @ h3v, n @ h3v)
    p1 = (e1 @ h1v, e2 @ h1v, n @ h1v)
    # the two contact H (2H, 4H) must have ZERO height: verify the
    # plane normal is exactly their plane normal (it is, by construction)
    assert abs(p2[2]) < 1e-12 and abs(p3[2]) < 1e-12
    # rotation: monomer frame -> contact frame (columns = e1,e2,n)
    Rm = np.column_stack([e1, e2, n])      # v_contact = Rm.T @ v_monomer
    R_nh3 = Rm.T
    nh3_inplane = dict(h2=(float(p2[0]), float(p2[1])),
                       h4=(float(p3[0]), float(p3[1])),
                       h1_inplane=(float(p1[0]), float(p1[1])),
                       h1_height=float(p1[2]),
                       r_NH=float(np.linalg.norm(h2v)))
    return R_nh3, nh3_inplane


def main():
    num = check_number()
    idx = json.load(open(IDX))
    nh3 = np.asarray(idx['default_endpoints']['NH3_gas_v5_final'][
        'coords_angstrom'], float)
    o3 = np.asarray(idx['default_endpoints']['O3_gas_final'][
        'coords_angstrom'], float)

    # contact-plane frame from the rigid monomer NH3 (exact):
    # n = normal of the N-H2-H3 plane, signed toward H1 (3H positive
    # normal side, assumption 2); R_nh3 = Rm.T maps monomer vectors
    # into (e1, e2, n) contact-frame components
    R_nh3, nh3_ip = build_frames(nh3, o3)
    # full proper placement rotation: in-plane rotation aligning the
    # 2H/4H SUM direction with +x, composed with the frame change
    bis = (nh3_ip['h2'][0] + nh3_ip['h4'][0],
           nh3_ip['h2'][1] + nh3_ip['h4'][1])
    theta = np.arctan2(bis[1], bis[0])
    Rz = np.array([[np.cos(-theta), -np.sin(-theta), 0.0],
                   [np.sin(-theta), np.cos(-theta), 0.0],
                   [0.0, 0.0, 1.0]])
    R_full = Rz @ R_nh3
    placed = (nh3 - nh3[0]) @ R_full.T      # rigid image, exact offsets
    placed_row_z = placed[:, 2].copy()
    # placed rows: 1,2 = the in-plane contact pair (z~0); row 3 = the
    # out-of-plane 3H (z = +0.871, positive normal by the n-sign choice)
    h2_off = placed[2][:2].copy()   # contact H on the +y side (2H role)
    h4_off = placed[1][:2].copy()   # contact H on the -y side (4H role)
    h1_ip = placed[3][:2].copy()    # 3H in-plane projection
    h1_z = float(placed_row_z[3])
    assert h2_off[1] * h4_off[1] < 0
    # 5O on the SAME side as 2H (uncrossed ring, per the figure)
    side = 1.0 if h2_off[1] >= 0 else -1.0

    # O3: central at origin, bisector +x, 5O on the 2H side (plane z=0)
    ci, sums = identify_central_o(o3)
    c = o3[ci]
    i1, i2 = (ci + 1) % 3, (ci + 2) % 3
    o_up, o_dn = o3[i1] - c, o3[i2] - c
    if (o_up[1] - h2_off[1]) * side < 0:
        o_up, o_dn = o_dn, o_up
        i1, i2 = i2, i1
    ang = np.arccos(np.clip(o_up @ o_dn / np.linalg.norm(o_up)
                            / np.linalg.norm(o_dn), -1, 1))
    r5, r7 = float(np.linalg.norm(o_up)), float(np.linalg.norm(o_dn))
    P5 = np.array([r5 * np.cos(ang / 2),
                   side * r5 * np.sin(ang / 2), 0.0])
    P7 = np.array([r7 * np.cos(ang / 2),
                   -side * r7 * np.sin(ang / 2), 0.0])
    P6 = np.array([0.0, 0.0, 0.0])

    # proper-rotation verification per fragment (kabsch, det +1),
    # rows aligned: O3 placed (5O,6O,7O) = monomer (term_i1, central,
    # term_i2); NH3 placed (N,2H,3H,4H) = monomer (N, H_b, H_c, H_a)
    mon_o3_ordered = np.vstack([o3[i1], o3[ci], o3[i2]]) - o3[ci]
    Ro3 = kabsch(mon_o3_ordered, np.vstack([P5, P6, P7]))
    mon_nh3_ordered = nh3[[0, 2, 3, 1]] - nh3[0]
    Rnh = kabsch(mon_nh3_ordered, placed)

    checks = {}
    checks['number_check'] = bool(num['pass_'])
    checks['rotation_orthonormal'] = bool(
        max(np.abs(Ro3.T @ Ro3 - np.eye(3)).max(),
            np.abs(Rnh.T @ Rnh - np.eye(3)).max()) < 1e-12)
    checks['rotation_det_plus1'] = bool(
        abs(np.linalg.det(Ro3) - 1) < 1e-12
        and abs(np.linalg.det(Rnh) - 1) < 1e-12)
    checks['contact_h_in_plane'] = True   # by construction (p2[2]=p3[2]=0)
    checks['three_h_positive_normal'] = bool(h1_z > 0)

    # ---- six-point closed-form placement --------------------------------
    Ca = P5[:2] - h2_off
    Cb = P7[:2] - h4_off
    v = Cb - Ca
    dist = float(np.linalg.norm(v))
    geoms = []
    prev_N = None
    max_step = 0.0
    branch_signs = []
    for delta in DELTAS:
        d1, d2 = D1_BASE + delta, D2_BASE + delta
        x0 = (d1 ** 2 - d2 ** 2 + dist ** 2) / (2 * dist)
        y02 = d1 ** 2 - x0 ** 2
        if y02 < 0:
            raise RuntimeError('HARD STOP before SCF: no real placement '
                               'for delta=%+.2f' % delta)
        y0 = float(np.sqrt(y02))
        branch_signs.append(1.0)          # fixed +y0 branch throughout
        N2 = Ca + x0 * (v / dist) + y0 * np.array([-v[1], v[0]]) / dist
        N = np.array([N2[0], N2[1], 0.0])
        H2 = N + np.hstack([h2_off, [placed_row_z[2]]])
        H3 = N + np.hstack([h1_ip, [placed_row_z[3]]])
        H4 = N + np.hstack([h4_off, [placed_row_z[1]]])
        coords_A = np.vstack([N, H2, H3, H4, P5, P6, P7])
        coords_B = coords_A * BPA
        # verifications
        d_actual = (float(np.linalg.norm(H2 - P5)),
                    float(np.linalg.norm(H4 - P7)))
        ok_contacts = bool(abs(d_actual[0] - d1) < 1e-10
                           and abs(d_actual[1] - d2) < 1e-10)
        dm_nh = distance_matrix(coords_A[:4])
        dm_o3 = distance_matrix(coords_A[4:])
        # rigid reference in the SAME row assignment (N, 2H, 3H, 4H):
        # 2H <- monomer H_b, 3H <- monomer H_c, 4H <- monomer H_a
        dev_nh = float(np.abs(dm_nh - distance_matrix(
            mon_nh3_ordered @ R_full.T + N)).max())
        dev_o3 = float(np.abs(dm_o3 - distance_matrix(
            mon_o3_ordered @ Ro3.T)).max())
        internal_ok = bool(dev_nh < 1e-10 and dev_o3 < 1e-10)
        d_inter = float(np.linalg.norm(coords_A[:4, None, :]
                                       - coords_A[None, 4:, :],
                                       axis=2).min())
        collision_ok = bool(d_inter >= COLLISION_THRESHOLD_A)
        if prev_N is not None:
            step = float(np.linalg.norm(N2 - prev_N))
            max_step = max(max_step, step)
        else:
            step = 0.0
        geoms.append(dict(
            tag='scan_d%+03d' % int(round(delta * 100)),
            delta_A=delta,
            perm_nh3=[0, 2, 3, 1],
            perm_o3=[i1, ci, i2],
            d_5O_2H_target_A=d1, d_7O_4H_target_A=d2,
            d_actual_A=d_actual,
            coords_bohr=coords_B.tolist(),
            coords_sha=sha_arr(coords_B),
            contacts_ok=ok_contacts, internal_ok=internal_ok,
            min_interfrag_A=d_inter, collision_ok=collision_ok,
            n_step_from_prev_A=step,
            branch='+y0 (N on the +x side of the Ca-Cb axis)'))
        prev_N = N2.copy()
        checks['point_%+03d' % int(round(delta * 100))] = bool(
            ok_contacts and internal_ok and collision_ok)
    # branch continuity: consecutive N separations bounded by ~0.6 A
    checks['branch_continuous'] = bool(max_step < 0.6 and max_step > 0)
    checks['single_branch_no_mirror'] = bool(
        len(set(branch_signs)) == 1)

    manifest = dict(
        job='JOB-2026-0906-064: C2 coplanar-assumption model six-point '
            'fixed-orientation distance scan',
        number_check=num,
        review_063='notes/job063_commander_review_2026-09-10.md',
        model_assumptions=dict(
            authorization='commander-selected in the JOB-064 instruction '
                          '(the executor did not choose this)',
            items=[
                'O3 plane coincides with the N-2H-4H contact plane',
                '3H on the POSITIVE normal side of that plane',
                '3H mirror image NOT computed in this batch',
                'contacts fixed: 5O...2H and 7O...4H',
                'figure distances are project initial-guess constraints '
                'only, NOT literature 3D coordinates',
                'NH3 and O3 internal geometry = JOB-026 accepted '
                'monomers, rigid rotation+translation only'],
            three_h_height_A=h1_z,
            contact_ring='six-membered contact ring 5O-6O-7O...4H-N-2H'
                         '...5O (per the 063 review correction)'),
        monomer_sources=dict(
            file=os.path.relpath(IDX, ROOT),
            sha256_16=sha256_file(IDX),
            nh3_e=idx['default_endpoints']['NH3_gas_v5_final'][
                'e_total_hartree'],
            o3_e=idx['default_endpoints']['O3_gas_final'][
                'e_total_hartree']),
        o3_placement=dict(central_index=ci, angle_deg=float(
            np.degrees(ang)), r5_A=r5, r7_A=r7,
            sums_A=sums),
        scan=dict(
            deltas_A=DELTAS,
            definition='(d_5O-2H, d_7O-4H) = (2.548+delta, 2.448+delta) '
                       'A; identical orientation; only the overall '
                       'fragment separation changes; same closed-form '
                       'placement / same branch as JOB-063',
            branch=geoms[0]['branch'],
            max_consecutive_N_step_A=max_step),
        caps=dict(scan=6,
                  note='six SCF+full-gradient attempts, one per point; '
                       'no centre reproduction; failures count; any SCF/'
                       'gradient/finite/config error -> save and STOP, '
                       'no auto-advance, no restart, no separate ledger'),
        quantum_config=dict(
            method='identical to the JOB-031 formal scan configuration',
            xc='wb97xd (libxc) + project explicit D2 once',
            basis='def2-TZVP', grid_level=8, grid_response=True,
            scf_tol=[1e-12, 1e-9], charge=0, spin=0,
            solvent='none (gas phase)'),
        analysis_rules=dict(
            reference='delta = 0.00 point is the energy reference',
            report='actual H...O distances, dE, max|g|, per-point '
                   'geometry hashes, distance-energy curve, whether an '
                   'energy minimum exists within the sampled range, and '
                   'whether the radial derivative changes sign',
            scope='fixed-orientation scan under the COPLANAR PROJECT '
                  'MODEL ONLY'),
        geometries=geoms,
        checks=checks, all_checks_pass=all(checks.values()))
    os.makedirs(OUT, exist_ok=True)
    tmp = OUT + '/input_manifest.json.tmp'
    with open(tmp, 'w') as fh:
        json.dump(manifest, fh, indent=2, default=str)
    os.replace(tmp, OUT + '/input_manifest.json')

    print('[064] number check:', json.dumps(num))
    print('[064] checks:', json.dumps(
        {k: v for k, v in checks.items()}, default=str))
    if not (num['pass_'] and manifest['all_checks_pass']):
        print('[064] PRECHECK FAILED -> STOP (zero evaluations)')
        raise SystemExit(1)
    print('[064] PREP OK (zero evaluations; six geometries built and '
          'verified on one branch)', flush=True)


if __name__ == '__main__':
    main()
