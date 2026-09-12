#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-045 step A (offline, no SCF):
  * read mode 0 from the 044 saved analysis, determine its convention;
  * build the fixed unit direction q (21 Cartesian components, ||q||2 = 1);
  * cross-check the centre geometry against 041 attempt 5 / 043 endpoint;
  * generate the four displacement geometries t = +-0.01, +-0.02 Bohr
    along the straight line R(t) = R0 + t q (no rotation registration, no
    re-projection, no internal relaxation).
"""
import os, sys, json, hashlib
import numpy as np

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
OUT = ROOT + '/run_artifacts/02_nh3o3_reference/c1_negmode_scan045'
os.makedirs(OUT, exist_ok=True)
SYMS = ['N', 'H', 'H', 'H', 'O', 'O', 'O']
MASSES = np.array([14.0, 1.0, 1.0, 1.0, 16.0, 16.0, 16.0])


def sha_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def sha_arr(a):
    return hashlib.sha256(np.asarray(a, float).tobytes()).hexdigest()[:16]


out = dict(job='JOB-2026-0906-045 step A: direction + displacement inputs')

# ---------- 1) centre geometry: 041 attempt 5 ----------
f41 = ROOT + '/run_artifacts/02_nh3o3_reference/c1_cart_exec_v3/eval_records.json'
d41 = json.load(open(f41))
rec5 = None
for k, v in d41.items():
    if v.get('attempt') == 5 and v.get('status') == 'evaluated':
        rec5, key5 = v, k
        break
C_A = np.asarray(rec5['coords_angstrom'], float)          # Angstrom
C_B = np.asarray(rec5['coords_bohr'], float)              # Bohr
g41 = np.asarray(rec5['grad_full'], float).reshape(-1)    # Eh/Bohr

# ---------- 2) 043 endpoint record (cross-check) ----------
f43 = ROOT + '/run_artifacts/02_nh3o3_reference/c1_nondiag_freq/nondiag_results.json'
d43 = json.load(open(f43))
er = d43['endpoint_reproduction']
g43 = np.asarray(er['grad_full'], float).reshape(-1)
out['centre_cross_check'] = dict(
    source_041_attempt5=dict(file=f41, sha256=sha_file(f41),
                             point_key=key5,
                             coords_A_hash=sha_arr(C_A),
                             e_total=rec5['e_total'],
                             grad_max=rec5['grad_max']),
    source_043_endpoint=dict(
        file=f43, sha256=sha_file(f43),
        e_total=er['e_total'], grad_max=er['grad_max'],
        dE_vs_041=er['dE_vs_041'], dgrad_max_vs_041=er['dgrad_max_vs_041'],
        dcoords_vs_041=er['dcoords_vs_041'], gates_pass=er['gates_pass'],
        config=er.get('config')),
    grad_041_vs_043_max_abs_diff=float(np.abs(g41 - g43).max()),
    note='043 endpoint reproduced the 041 attempt-5 geometry with '
         'dcoords = 0.0 and dgrad ~ 4e-13; the 042 last step is NOT used')

# ---------- 3) mode 0 direction from 044 ----------
f44 = ROOT + '/run_artifacts/02_nh3o3_reference/c1_freq_fix044/freq_fix044_results.json'
d44 = json.load(open(f44))
m0 = None
for m in d44['c1_analysis']['modes']:
    if m['freq_cm1'] < 0:
        m0 = m
        break
d_vec = np.asarray(m0['cart_norm_mode'], float)     # already Cartesian
# convention check: if d = M^-1/2 v with ||v|| = 1, then ||M^1/2 d|| = 1
sm = np.repeat(np.sqrt(MASSES), 3)
mw_norm = float(np.linalg.norm(d_vec * sm))
out['mode0'] = dict(
    source=dict(file=f44, sha256=sha_file(f44), mode=int(m0['mode']),
                freq_cm1=m0['freq_cm1'],
                external_overlap=m0['external_overlap']),
    saved_field='cart_norm_mode',
    convention='saved field is ALREADY the Cartesian direction '
               '(d = M^-1/2 v_mw); it is NOT divided by mass again',
    convention_check=dict(
        norm_of_M_sqrt_times_d=mw_norm,
        expected_1_if_cartesian=1.0,
        check_pass=bool(abs(mw_norm - 1.0) < 1e-9)),
    raw_norm=float(np.linalg.norm(d_vec)))

# unit direction q, sign fixed: largest-|component| entry positive
q = d_vec / np.linalg.norm(d_vec)
imax = int(np.argmax(np.abs(q)))
if q[imax] < 0:
    q = -q
out['direction'] = dict(
    definition='q = d / ||d||_2 over all 21 Cartesian components',
    norm=float(np.linalg.norm(q)),
    sign_convention='component %d made positive' % imax,
    q_vector=[float(x) for x in q],
    q_hash=sha_arr(q),
    per_atom_displacement_of_q_Bohr=[float(np.linalg.norm(q[3*i:3*i+3]))
                                     for i in range(7)],
    withdrawal='the mode vector is NOT an executed physical displacement; '
               'the 044 statement "3 H displaced 0.548/0.566/0.566 Bohr" '
               '(raw normalised components written as actual displacements) '
               'is WITHDRAWN')

# ---------- 4) four displacement geometries ----------
steps = [0.01, -0.01, 0.02, -0.02]
geoms = []
for t in steps:
    Rt = C_B + t * q.reshape(7, 3)        # Bohr, straight line
    d_actual = Rt - C_B
    geoms.append(dict(
        t_Bohr=t,
        coords_bohr=[list(map(float, r)) for r in Rt],
        coords_angstrom=[list(map(float, r * 0.52917721092)) for r in Rt],
        coords_bohr_sha=sha_arr(Rt),
        step_norm_Bohr=float(np.linalg.norm(d_actual)),
        max_atom_displacement_Bohr=float(max(
            np.linalg.norm(d_actual[3*i:3*i+3]) for i in range(7))),
        sign_check=bool(abs(np.linalg.norm(d_actual) - abs(t)) < 1e-12),
        element_order=SYMS,
        unit='Bohr (stored) / Angstrom (derived)'))
out['displacement_geometries'] = geoms
out['scan_protocol'] = dict(
    line='R(t) = R0 + t q, t in Bohr, q unit-norm Cartesian',
    steps_Bohr=steps,
    no_rotation_registration=True, no_reprojection=True,
    no_internal_relaxation=True,
    no_finite_angle_conversion=True)

json.dump(out, open(OUT + '/input_manifest.json', 'w'), indent=2)
print('centre: 041 attempt5 key=%s  E=%.9f' % (key5[:16], rec5['e_total']))
print('  043 endpoint dE=%.2e dgrad=%.2e dC=%.1e gates=%s'
      % (er['dE_vs_041'], er['dgrad_max_vs_041'], er['dcoords_vs_041'],
         er['gates_pass']))
print('  grad 041 vs 043 max diff: %.2e' % out['centre_cross_check'][
    'grad_041_vs_043_max_abs_diff'])
print()
print('mode0: freq=%.3f cm^-1  external_overlap=%.2e'
      % (m0['freq_cm1'], m0['external_overlap']))
print('  convention check ||M^1/2 d|| = %.10f (expect 1) -> %s'
      % (mw_norm, out['mode0']['convention_check']['check_pass']))
print('  raw ||d|| = %.6f ; ||q|| = %.10f'
      % (out['mode0']['raw_norm'], out['direction']['norm']))
print('  per-atom |q| (Bohr): %s'
      % np.round(out['direction']['per_atom_displacement_of_q_Bohr'], 4))
print()
print('geometries:')
for g in geoms:
    print('  t=%+.3f  step_norm=%.6f  max_atom_disp=%.6f  sign_ok=%s  sha=%s'
          % (g['t_Bohr'], g['step_norm_Bohr'],
             g['max_atom_displacement_Bohr'], g['sign_check'],
             g['coords_bohr_sha']))
print()
print('saved ->', OUT + '/input_manifest.json')
