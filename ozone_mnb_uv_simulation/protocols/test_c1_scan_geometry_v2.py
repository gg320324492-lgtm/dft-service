import os, sys, json, tempfile
import numpy as np
sys.path.insert(0, '/mnt/e/dft-service/ozone_mnb_uv_simulation/protocols')
from c1_scan_geometry_v2 import (load_monomers, build_scan_geometries,
                                 identify_central)

ok = True
NH3, O3 = load_monomers()
dmat = lambda P: np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)


def rigid_dist_eq(A, B, tol=1e-9):
    """Same physical geometry up to rotation+translation."""
    cA = A - A.mean(axis=0)
    cB = B - B.mean(axis=0)
    # compare all pairwise distances (rotation+translation invariant)
    return bool(np.abs(dmat(cA) - dmat(cB)).max() < tol)


# ---------- T1: global rotation + translation -> equivalent path ----------
rng = np.random.default_rng(42)
Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
if np.linalg.det(Q) < 0:
    Q[:, 0] *= -1
t = rng.normal(size=3) * 2.0
NH3_t = NH3 @ Q.T + t
O3_t = O3 @ Q.T + t
m1 = build_scan_geometries(NH3, O3, [2.8, 3.6])
m2 = build_scan_geometries(NH3_t, O3_t, [2.8, 3.6])
eq = all(rigid_dist_eq(np.asarray(a['coords']), np.asarray(b['coords']))
         for a, b in zip(m1['points'], m2['points']))
t1 = eq and m1['all_assertions_pass'] and m2['all_assertions_pass']
print('[T1] global rotation+translation -> equivalent path: %s'
      % ('PASS' if t1 else 'FAIL'))
ok &= t1

# ---------- T2: terminal swap -> same physical geometry ----------
O3_swap = O3[[0, 2, 1]]   # swap the two terminal rows (keep central first)
m3 = build_scan_geometries(NH3, O3_swap, [3.0])
m0 = build_scan_geometries(NH3, O3, [3.0])
# tolerance note: the JOB-026 endpoint's two O-O bonds differ at the ~1e-6 A
# level (numerical asymmetry); swapping terminals relocates that asymmetry,
# so the rebuilt geometry differs from the original by ~2.4e-6 A -- the
# pipeline faithfully preserves the input asymmetry (rigid placement).
t2 = (np.abs(dmat(np.asarray(m3['points'][0]['coords']))
             - dmat(np.asarray(m0['points'][0]['coords']))).max() < 1e-5)
print('[T2] terminal swap -> same physical geometry: %s'
      % ('PASS' if t2 else 'FAIL'))
ok &= t2

# ---------- T3: atom order permutation -> central O still identified ----------
for perm in ([2, 0, 1], [1, 2, 0], [2, 1, 0]):
    O3_p = O3[list(perm)]
    central, terms, cok, _ = identify_central(O3_p)
    # map back: the identified central must be the same physical atom
    same = (O3_p[central] - O3[0]).max() < 1e-12  # physical central is row 0
    if not same:
        break
t3 = same
print('[T3] permuted atom order -> central O still identified: %s'
      % ('PASS' if t3 else 'FAIL'))
ok &= t3

# ---------- T4: flipped sign / central-side placement must FAIL ----------
# build a WRONG geometry: N on the central-O side (negative s)
m4 = build_scan_geometries(NH3, O3, [3.0])
C = np.asarray(m4['points'][0]['coords'], float)
N_wrong = C[5] - (C[0] - C[5]) * 0.5    # N pulled to the central-O side
C_wrong = C.copy()
C_wrong[0] = N_wrong
# verification quantities for the wrong geometry:
lp = C_wrong[0] - C_wrong[1:4].mean(axis=0)
lp /= np.linalg.norm(lp)
mid = C_wrong[5:7].mean(axis=0)
to_mid = mid - C_wrong[0]; to_mid /= np.linalg.norm(to_mid)
ang_wrong = float(np.degrees(np.arccos(np.clip(lp @ to_mid, -1, 1))))
# (N-M).u check: with the correct u = (M-C)/|M-C| of the O3 fragment
u = (mid - C[5]) / np.linalg.norm(mid - C[5])
nm_u = float((C_wrong[0] - mid) @ u)
t4 = (nm_u < 0) and (ang_wrong > 90.0)   # side test and direction test both fail
print('[T4] central-side placement: (N-M).u=%.3f <0, lone-pair angle=%.1f deg '
      '-> verification FAILS as required: %s' % (nm_u, ang_wrong,
                                                 'PASS' if t4 else 'FAIL'))
ok &= t4

# ---------- T5: six-point distances and monotonicity ----------
m5 = build_scan_geometries(NH3, O3, [2.6, 2.8, 3.0, 3.2, 3.6, 4.2])
devs = [p['d_dev'] for p in m5['points']]
svals = [p['s'] for p in m5['points']]
t5 = (max(devs) < 1e-9 and all(svals[i] < svals[i + 1] for i in range(5)))
print('[T5] six-point: max|d_actual-d_target|=%.2e, s monotonic: %s'
      % (max(devs), 'PASS' if t5 else 'FAIL'))
ok &= t5

print('C1 GEOMETRY V2 REGRESSION:', 'ALL PASS' if ok else 'FAILED')
if not ok:
    raise SystemExit(1)
