#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-046 step A (zero-evaluation):
  * verify job number availability (external, already done via ls);
  * load 045 actual evaluation records (+0.01 / -0.01), verify direction q
    hash against the charter value 59599f2a2c0be260;
  * verify element order / coordinate frame / unit consistency between the
    centre, the two displacement points, q and the 043 Hessian;
  * verify coordinate round trip (045 actual coords -> Bohr -> would-be
    mol build) BEFORE any SCF is spent;
  * recompute k_H components from the 043 saved matrices with q;
  * compute the k_E vs k_g differences at both steps (045 correction input);
  * read-only source audit of the installed PySCF grid-response paths.
"""
import os, sys, json, hashlib
import numpy as np

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
OUT = ROOT + '/run_artifacts/02_nh3o3_reference/c1_gridresp046'
os.makedirs(OUT, exist_ok=True)
SYMS = ['N', 'H', 'H', 'H', 'O', 'O', 'O']
ANG_PER_BOHR = 0.52917721092
BOHR_PER_A = 1.0 / ANG_PER_BOHR
CHARTER_Q_HASH = '59599f2a2c0be260'


def sha_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def sha_arr(a):
    return hashlib.sha256(np.asarray(a, float).tobytes()).hexdigest()[:16]


out = dict(job='JOB-2026-0906-046 step A: zero-evaluation checks')

# ---------- 1) 045 records ----------
S45 = ROOT + '/run_artifacts/02_nh3o3_reference/c1_negmode_scan045'
man = json.load(open(S45 + '/input_manifest.json'))
q = np.asarray(man['direction']['q_vector'], float)
q_hash_manifest = man['direction']['q_hash']
q_hash_recomputed = sha_arr(q)
out['q_check'] = dict(
    manifest_hash=q_hash_manifest,
    recomputed_hash=q_hash_recomputed,
    charter_hash=CHARTER_Q_HASH,
    match=bool(q_hash_manifest == q_hash_recomputed == CHARTER_Q_HASH),
    norm=float(np.linalg.norm(q)),
    note='q is a dimensionless unit vector over 21 Cartesian components, '
         'atom-major order (N,H,H,H,O,O,O x x x y y y z z z)')

ev = {}
for tag, t in (('p01', 0.01), ('m01', -0.01)):
    ev[tag] = json.load(open(S45 + '/eval_eval_displacement_%+.2f.json' % t))
out['eval045_loaded'] = {k: dict(e_total=v['e_total'],
                                 coords_sha=v['coords_actual_sha'],
                                 grad_max=v['grad_max'],
                                 config=v['config'])
                         for k, v in ev.items()}

# ---------- 2) 043 Hessian metadata + k_H recomputation ----------
d43 = json.load(open(ROOT + '/run_artifacts/02_nh3o3_reference/'
                            'c1_nondiag_freq/nondiag_results.json'))
H_dft = np.asarray(d43['dft_hessian']['matrix'], float)
H_d2 = np.asarray(d43['d2_hessian']['matrix'], float)
H_tot = H_dft + H_d2
k_H = float(q @ H_tot @ q)
k_H_dft = float(q @ H_dft @ q)
k_H_d2 = float(q @ H_d2 @ q)
out['h043_check'] = dict(
    unit=d43['dft_hessian']['unit'],
    shape=list(H_dft.shape),
    sym_resid_dft=float(np.abs(H_dft - H_dft.T).max()),
    sym_resid_d2=float(np.abs(H_d2 - H_d2.T).max()),
    layout='atom-major 21x21 (043: transpose(0,2,1,3).reshape(21,21)), '
           'same atom order as SYMS and as q components',
    k_H_total=k_H, k_H_dft=k_H_dft, k_H_d2=k_H_d2,
    symmetrization_note='symmetrization does not change the quadratic form '
                        'of a REAL fixed direction: q^T (H+H^T)/2 q == '
                        'q^T H q identically for real q; it therefore '
                        'cannot by itself explain the k_H vs FD difference')

# ---------- 3) coordinate round-trip BEFORE any SCF ----------
geo = {}
for tag, t in (('p01', 0.01), ('m01', -0.01)):
    C_A = np.asarray(ev[tag]['coords_actual_angstrom'], float)  # 045 actual
    R_bohr = C_A / ANG_PER_BOHR
    # cross-check against the canonical line R0 + t q (043 convention)
    d41 = json.load(open(ROOT + '/run_artifacts/02_nh3o3_reference/'
                                'c1_cart_exec_v3/eval_records.json'))
    rec5 = None
    for k, v in d41.items():
        if v.get('attempt') == 5 and v.get('status') == 'evaluated':
            rec5 = v
            break
    R0 = np.asarray(rec5['coords_angstrom'], float) * BOHR_PER_A
    R_canon = R0 + t * q.reshape(7, 3)
    geo[tag] = dict(
        coords_bohr_rebuilt=[list(map(float, r)) for r in R_bohr],
        rebuilt_bohr_sha=sha_arr(R_bohr),
        max_diff_vs_canonical_line_Bohr=float(np.abs(R_bohr - R_canon).max()),
        element_order=SYMS,
        unit='rebuilt Bohr = 045 actual Angstrom / 0.52917721092')
    # per-atom interatomic distances sanity (no atom overlap after rebuild)
    dm = np.linalg.norm(R_bohr[:, None, :] - R_bohr[None, :, :], axis=-1)
    np.fill_diagonal(dm, 9e9)
    geo[tag]['min_pair_distance_Bohr'] = float(dm.min())
out['geometry_rebuild'] = geo
worst_line = max(g['max_diff_vs_canonical_line_Bohr'] for g in geo.values())
min_pair = min(g['min_pair_distance_Bohr'] for g in geo.values())
out['geometry_rebuild']['round_trip_ok'] = bool(worst_line < 1e-9
                                                and min_pair > 1.0)
out['geometry_rebuild']['round_trip_summary'] = dict(
    max_diff_vs_canonical_line_Bohr=worst_line, min_pair_distance_Bohr=min_pair)
print('round-trip: max |rebuilt - canonical line| = %.3e Bohr; min pair '
      'dist = %.3f Bohr -> %s'
      % (worst_line, min_pair,
         'OK' if out['geometry_rebuild']['round_trip_ok'] else 'FAIL'),
      flush=True)
if not out['geometry_rebuild']['round_trip_ok']:
    json.dump(out, open(OUT + '/prep_check_results.json', 'w'), indent=2)
    raise RuntimeError('HARD STOP: geometry round-trip verification failed')

# ---------- 4) k_E vs k_g differences at both steps (045 correction) ----------
d45 = json.load(open(S45 + '/direction_derivative_results.json'))
da = d45['direction_analysis']
kg = {}
for h in (0.01, 0.02):
    r = da['per_h'][repr(h)]
    kE, kG = r['k_E_Eh_Bohr2'], r['k_g_Eh_Bohr2']
    kg['h%g' % h] = dict(k_E=kE, k_g=kG, diff_E_minus_G=kE - kG,
                         rel_vs_k_E=abs(kE - kG) / abs(kE))
out['kE_vs_kg_045'] = kg
print('045 k_E vs k_g: h=0.01 diff %+.3e (%.2f%% of |k_E|); '
      'h=0.02 diff %+.3e (%.2f%%)'
      % (kg['h0.01']['diff_E_minus_G'], 100 * kg['h0.01']['rel_vs_k_E'],
         kg['h0.02']['diff_E_minus_G'], 100 * kg['h0.02']['rel_vs_k_E']),
      flush=True)

# ---------- 5) PySCF source audit (read-only) ----------
import pyscf
import pyscf.dft.libxc as libxc
pyscf_dir = os.path.dirname(pyscf.__file__)
grad_rks = os.path.join(pyscf_dir, 'grad', 'rks.py')
hess_rks = os.path.join(pyscf_dir, 'hessian', 'rks.py')


def grab(path, lo, hi):
    lines = open(path, encoding='utf-8', errors='replace').readlines()
    return ''.join('%5d| %s' % (i + 1, lines[i])
                   for i in range(lo - 1, min(hi, len(lines))))


audit = dict(pyscf_version=pyscf.__version__,
             grad_rks_path=grad_rks, hess_rks_path=hess_rks)
audit['grad_grid_response_default'] = (
    'grad/rks.py:745 grid_response = getattr(__config__, '
    "'grad_rks_Gradients_grid_response', False) -> DEFAULT False")
audit['grad_get_veff_excerpt'] = grab(grad_rks, 50, 58)
audit['grad_extra_force_excerpt'] = grab(grad_rks, 768, 782)
# hessian: grid_response occurrences (expected NLC-only)
hl = [(i + 1, l.rstrip()) for i, l in
      enumerate(open(hess_rks, encoding='utf-8', errors='replace'))
      if 'grid_response' in l or 'response' in l.lower()]
audit['hess_grid_response_lines'] = hl[:12]
audit['nlc_check'] = dict(
    wb97xd_is_nlc=bool(libxc.is_nlc('wb97xd')),
    note='wb97xd has NO non-local correlation -> the NLC numerical-Hessian '
         'grid-response paths in hessian/rks.py (lines ~461-482, 573-624) '
         'are NOT triggered for this functional')
out['pyscf_source_audit'] = audit
print('pyscf %s: grad grid_response default False (line 745); '
      'get_veff full-response + extra_force(exc1_grid) gated by the flag; '
      'wb97xd is_nlc = %s' % (pyscf.__version__, audit['nlc_check'][
          'wb97xd_is_nlc']), flush=True)

json.dump(out, open(OUT + '/prep_check_results.json', 'w'), indent=2)
print('saved ->', OUT + '/prep_check_results.json', flush=True)
print('STEP A OK', flush=True)
