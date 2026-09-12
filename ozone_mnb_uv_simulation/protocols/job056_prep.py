#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-056 step A: OFFLINE preparation (ZERO evaluations).

Refine the 0.04-0.06 Bohr sign-change interval found by JOB-055 with
three new points t = 0.045/0.050/0.055 Bohr on the SAME new line
(R(t) = R*_055 + t q; NOT merged with the 053 old-centre path).

The 055 centre, t=0.04 and t=0.06 records are REUSED read-only (no centre
SCF in this batch).  Verifies: numbering, q hash / Cartesian unit / sign,
centre coords, the 055 t=0.04/0.06 actual coords and config vs the 055
manifest geometries (bitwise), fragments of the new points.
"""
import os, json, hashlib, glob, re
import numpy as np

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
R55 = ROOT + '/run_artifacts/02_nh3o3_reference/c1_dir_scan055'
OUT = ROOT + '/run_artifacts/02_nh3o3_reference/c1_refine056'
JOBS = ROOT + '/jobs'
JOB_NO = '056'
SYMS = ['N', 'H', 'H', 'H', 'O', 'O', 'O']
BPA = 1.0 / 0.52917721092
NEW_T = (0.045, 0.050, 0.055)


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
    m55 = json.load(open(R55 + '/input_manifest.json'))
    q = np.asarray(m55['direction']['q'], float).reshape(-1)
    R0 = np.asarray(m55['centre']['x_bohr'], float).reshape(7, 3)
    checks = {}
    # q: hash / Cartesian unit / sign preserved
    checks['q_hash_match'] = bool(m55['direction']['q_sha'] == sha_arr(q))
    checks['q_norm_1'] = bool(abs(float(np.linalg.norm(q)) - 1.0) < 1e-12)
    # centre: 055 manifest centre vs the 055 centre record (bitwise)
    c55 = json.load(open(R55 + '/eval_centre.json'))
    xc55 = np.asarray(c55['coords_actual_angstrom'], float) * BPA
    cd = float(np.abs(xc55.reshape(-1) - R0.reshape(-1)).max())
    checks['centre_coords_maxdiff_Bohr'] = cd
    checks['centre_coords_match'] = bool(cd <= 1e-9)
    checks['centre_e_ref'] = bool(abs(float(c55['e_total'])
                                      - (-282.0017216061232)) <= 1e-9)
    # 055 endpoint records: t=0.04 and t=0.06
    endpoints = {}
    for t, f in ((0.04, 'eval_scan_t040.json'), (0.06,
                                                 'eval_scan_t060.json')):
        rec = json.load(open(os.path.join(R55, f)))
        g = [g for g in m55['scan_geometries'] if g['t_Bohr'] == t][0]
        xd = float(np.abs(np.asarray(rec['coords_actual_angstrom'], float)
                          .reshape(-1) * BPA
                          - np.asarray(g['coords_bohr'],
                                       float).reshape(-1)).max())
        checks['t%03d_coords_maxdiff_Bohr' % int(t * 1000)] = xd
        checks['t%03d_coords_match' % int(t * 1000)] = bool(xd <= 1e-9)
        checks['t%03d_config_match' % int(t * 1000)] = bool(
            rec['config'] == c55['config'])
        checks['t%03d_e_ref' % int(t * 1000)] = bool(
            abs(float(rec['e_total']) - (-282.001721770 if t == 0.04
                                         else -282.001721758)) < 1e-6)
        endpoints['%g' % t] = dict(
            path=os.path.join(R55, f), sha256=sha256_file(
                os.path.join(R55, f)),
            e_total=float(rec['e_total']),
            grad_max=float(rec['grad_max']),
            grad=rec['grad'], x_bohr=rec['x_bohr'],
            t_Bohr=t, config=rec['config'])
    # three new points
    geoms = []
    for t in NEW_T:
        R = R0 + t * q.reshape(7, 3)
        d = (R - R0).reshape(-1)
        Rm = R.reshape(7, 3)
        dNH = [float(np.linalg.norm(Rm[i] - Rm[0])) for i in (1, 2, 3)]
        dOO = float(np.linalg.norm(Rm[5] - Rm[4]))
        dmin = float(min(np.linalg.norm(Rm[i] - Rm[j])
                         for i in range(7) for j in range(i + 1, 7)))
        geoms.append(dict(
            t_Bohr=t, coords_bohr=R.tolist(), coords_bohr_sha=sha_arr(R),
            step_norm_Bohr=float(np.linalg.norm(d)),
            displacement_along_q_Bohr=float(d @ q),
            NH_bonds_Bohr=dNH, O3_bond_Bohr=dOO,
            min_interatomic_distance_Bohr=dmin, element_order=SYMS,
            unit='Bohr (full-precision floats)'))
    checks['new_norms_match_t'] = bool(all(
        abs(g['step_norm_Bohr'] - g['t_Bohr']) < 1e-12
        and abs(g['displacement_along_q_Bohr'] - g['t_Bohr']) < 1e-12
        for g in geoms))
    checks['new_fragments_intact'] = bool(all(
        min(g['NH_bonds_Bohr']) > 1.5 and g['O3_bond_Bohr'] > 2.0
        and g['min_interatomic_distance_Bohr'] > 1.5 for g in geoms))
    all_pass = all(bool(v) for v in checks.values() if isinstance(v, bool))
    man = dict(
        job='JOB-2026-0906-056 refinement of the 0.04-0.06 Bohr '
            'sign-change interval (three new points on the 055 line)',
        number_check=num,
        caps=dict(refine=3, centre_new=0,
                  note='failures count; no borrowing; no retry; no auto '
                       'extra points; on exception or non-convergence: '
                       'save and STOP; the 055 centre is reused - NO '
                       'centre SCF in this batch'),
        centre=dict(x_bohr=R0.tolist(), coords_bohr_sha=sha_arr(R0),
                    e_total=float(c55['e_total']),
                    grad_max=float(c55['grad_max']),
                    grad=c55['grad'], config=c55['config'],
                    source='055 eval_centre.json (reused, not re-evaluated)'),
        direction=dict(q=q.tolist(), q_sha=m55['direction']['q_sha'],
                       already_cartesian=True,
                       note='no mass conversion; same sign/order as 055; '
                            'refinement of the 055 line ONLY, not merged '
                            'with the 053 old-centre path'),
        reused_055_records=dict(
            centre_path=R55 + '/eval_centre.json',
            t040=endpoints['0.04'], t060=endpoints['0.06'],
            note='read-only reuse; a(t) at t=0/0.04/0.06 from the saved '
                 'full gradients'),
        new_geometries=geoms,
        checks=checks, all_checks_pass=all_pass,
        analysis_spec=dict(
            merge='new points + the 055 saved t=0/0.04/0.06 records as ONE '
                  'line analysis table (t, dE vs the 055 centre, gmax, '
                  'a(t)=g.q, adjacent secant slopes)',
            rules=['secant slopes are NOT endpoint analytic derivatives',
                   'if a(t) still changes sign inside 0.04-0.06: register '
                   'only a NARROWER fixed-line sign-change interval',
                   'no sign change -> report as-is, no auto extra points',
                   'even an exact 1-D low point is NOT a full-DOF minimum '
                   '(transverse gradients require independent checks)']))
    save_json_atomic(OUT + '/input_manifest.json', man)
    print('[056] number check:', json.dumps(num))
    print('[056] checks:', json.dumps(
        {k: v for k, v in checks.items()}, default=str))
    if not (num['pass_'] and all_pass):
        print('[056] PRECHECK FAILED -> STOP (zero-evaluation stage)')
        raise SystemExit(1)
    print('[056] PREP OK (zero evaluations consumed; centre reused)',
          flush=True)


if __name__ == '__main__':
    main()
