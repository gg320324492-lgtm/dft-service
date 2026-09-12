#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-058 step A: OFFLINE preparation (ZERO evaluations).

FD-Hessian curvature check of the 057 stationary-point candidate:
- centre = 057 independent recheck record (E=-282.00172193965955,
  gmax=6.847299345879123e-6), geometry correspondence with opt_05 verified;
- 42 central-difference displacements R0 +/- h e_j (h=0.001 Bohr), all
  inputs pre-built, hash-registered, per-pair checks (only coordinate j
  changes; +/- signs; step; element order; units);
- caps: centre 1, stability 1, fd 42 (SCF+gradient total 43);
- nominal masses [14,1,1,1,16,16,16] amu (NAMED as nominal, matching 052).
Also registers the 057 corrections (notes/job058_057_correction.md).
"""
import os, json, hashlib, glob, re
import numpy as np

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
R57 = ROOT + '/run_artifacts/02_nh3o3_reference/c1_fdopt057'
OUT = ROOT + '/run_artifacts/02_nh3o3_reference/c1_fdhess058'
JOBS = ROOT + '/jobs'
JOB_NO = '058'
CENTRE_FILE = R57 + '/eval_recheck_recheck.json'
SYMS = ['N', 'H', 'H', 'H', 'O', 'O', 'O']
BPA = 1.0 / 0.52917721092
H_STEP = 0.001
MASSES_NOMINAL = [14.0, 1.0, 1.0, 1.0, 16.0, 16.0, 16.0]


def save_json_atomic(path, obj):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(obj, fh, indent=2, default=str)
    os.replace(tmp, path)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for blk in iter(lambda: fh.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def sha_arr(a):
    return hashlib.sha256(np.asarray(a, float).tobytes()).hexdigest()[:16]


def check_number():
    used = set()
    for f in glob.glob(JOBS + '/JOB-*.md'):
        m = re.match(r'JOB-2026-0906-(\d{3})', os.path.basename(f))
        if m:
            used.add(m.group(1))
    nxt = '%03d' % (max(int(u) for u in used) + 1)
    return dict(used_max=max(used), next_free=nxt,
                pass_=bool(nxt == JOB_NO and JOB_NO not in used))


def main():
    num = check_number()
    cen = json.load(open(CENTRE_FILE))
    m57 = json.load(open(R57 + '/input_manifest.json'))
    src57 = json.load(open(m57['source']['source_path_wsl']))
    checks = {}
    checks['e_matches'] = bool(abs(float(cen['e_total'])
                                   - (-282.00172193965955)) <= 1e-12)
    checks['gmax_matches'] = bool(abs(float(cen['grad_max'])
                                      - 6.847299345879123e-6) <= 1e-15)
    checks['converged_finite'] = bool(cen['converged']
                                      and cen['all_finite'])
    checks['gmax_le_1e5'] = bool(float(cen['grad_max']) <= 1e-5)
    checks['config_complete'] = bool(
        cen['config'].get('grid_response') is True
        and cen['config'].get('grid_level') == 8
        and cen['config'].get('d2_attached') is True)
    # correspondence with 057 opt_05 (the meeting record)
    o05 = json.load(open(R57 + '/eval_opt_opt_05.json'))
    checks['recheck_vs_opt05_coords_maxdiff_Bohr'] = float(np.abs(
        np.asarray(cen['coords_actual_angstrom'], float).reshape(-1) * BPA
        - np.asarray(o05['coords_actual_angstrom'], float)
        .reshape(-1) * BPA).max())
    checks['recheck_vs_opt05_match'] = bool(
        checks['recheck_vs_opt05_coords_maxdiff_Bohr'] <= 1e-9)
    # centre execution coords (x_bohr saved in the recheck record)
    R0 = np.asarray(cen['x_bohr'], float).reshape(7, 3)
    # 42 displacements R0 +/- h e_j
    geoms = []
    for j in range(21):
        i, a = divmod(j, 3)
        for sgn, tag in ((+1, 'p'), (-1, 'm')):
            R = R0.copy()
            R[i, a] += sgn * H_STEP
            d = (R - R0).reshape(-1)
            geoms.append(dict(
                j=j, sign=tag, h_Bohr=H_STEP,
                tag_full='fd_%s%02d' % (tag, j),
                coords_bohr=R.tolist(), coords_bohr_sha=sha_arr(R),
                displacement_max_Bohr=float(np.abs(d).max()),
                displacement_along_ej=float(
                    d @ np.eye(21)[j]),
                element_order=SYMS, unit='Bohr (full-precision floats)'))
    # per-pair checks
    pair_ok = True
    for j in range(21):
        gp = geoms[2 * j]
        gm = geoms[2 * j + 1]
        Rp = np.asarray(gp['coords_bohr'], float).reshape(7, 3)
        Rm = np.asarray(gm['coords_bohr'], float).reshape(7, 3)
        diff = (Rp - Rm).reshape(-1)
        # only coordinate j differs, by exactly 2h
        mask = np.zeros(21)
        mask[j] = 2 * H_STEP
        if float(np.abs(diff - mask).max()) > 1e-15:
            pair_ok = False
    checks['pairs_only_j_changes'] = bool(pair_ok)
    checks['n_geoms'] = len(geoms)
    all_pass = all(bool(v) for v in checks.values() if isinstance(v, bool))
    man = dict(
        job='JOB-2026-0906-058 full-gradient central-difference Hessian '
            'check of the 057 stationary-point candidate',
        number_check=num,
        caps=dict(centre=1, stability=1, fd=42, scf_grad_total=43,
                  note='failures count; no borrowing; no re-initialising '
                       'the ledger to clear failures; on exception or '
                       'non-convergence: save and STOP, no fix-and-resume '
                       'without new authorization'),
        correction_057='notes/job058_057_correction.md',
        centre=dict(path_wsl=CENTRE_FILE, sha256=sha256_file(CENTRE_FILE),
                    e_total=float(cen['e_total']),
                    grad_max=float(cen['grad_max']),
                    x_bohr=cen['x_bohr'],
                    coords_actual_angstrom=cen['coords_actual_angstrom'],
                    grad=cen['grad'], config=cen['config'],
                    element_order=SYMS),
        centre_gate=dict(dE_max=1e-8, dgrad_max=1e-7, dcoords_max_A=1e-9,
                         gmax_le=1e-5, extra='SCF converged, finite, '
                         'config identical'),
        direction_note='central differences along Cartesian e_j; h=0.001 '
                       'Bohr; single step size (sensitivity NOT verified '
                       'in this batch)',
        masses=dict(values=MASSES_NOMINAL,
                    name='NOMINAL mass numbers (not isotope masses); '
                         'consistent with 052 for comparison'),
        displacement_geometries=geoms,
        checks=checks, all_checks_pass=all_pass,
        hessian_spec=dict(
            column_j='H[:,j] = [g(R0+h e_j) - g(R0-h e_j)]/(2h); full '
                     'gradients already include D2 once and '
                     'grid_response (no extra D2 matrix added)',
            order='H_raw saved FIRST with its antisymmetric residual, then '
                  'H_sym=(H_raw+H_raw.T)/2; missing points -> NO column '
                  'interpolation and NO mode analysis',
            post='TR projection rank-6 / internal 15; nominal masses; '
                 'negative eigenvalues kept; PySCF post-processing '
                 'cross-check (post-processing validation only)'),
        analysis_limits='a negative internal mode is reported as "this '
                        'batch FD matrix predicts negative curvature" '
                        '(no transition-state claim, no mode following); '
                        'all-positive = "this single-step FD matrix shows '
                        'no internal negative curvature"; single step '
                        'size; frequencies NOT compared across geometries '
                        'without atom-mapping/registration/overlap checks')
    save_json_atomic(OUT + '/input_manifest.json', man)
    print('[058] number check:', json.dumps(num))
    print('[058] checks:', json.dumps(
        {k: v for k, v in checks.items()}, default=str))
    if not (num['pass_'] and all_pass):
        print('[058] PRECHECK FAILED -> STOP (zero-evaluation stage)')
        raise SystemExit(1)
    print('[058] PREP OK (zero evaluations consumed)', flush=True)


if __name__ == '__main__':
    main()
