#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-054 step A: OFFLINE preparation (ZERO evaluations).

(a) batch-number check: 054 next unoccupied;
(b) 053 corrections registered (notes/job054_053_correction.md; see also
    the banners on the 053 report and job record);
(c) source verification: the 053 actually-evaluated +0.02 Bohr point
    (eval_disp_p02.json in the resume-01 directory) - identity as the 053
    displacement geometry (ulp-level vs the 053 manifest), saved actual
    coordinates (x_bohr), full gradient, config; hashes recorded;
    registration: "geometry continuation, optimizer state reset";
(d) manifest for the optimizer: stop target max|g|<=1e-6 (acceptance gate
    1e-5 UNCHANGED; the start itself may exceed 1e-5 and is NOT rejected
    for that); caps: start_repro 1, opt 25, recheck 1, total <= 27.
No SCF / gradient: reads existing records only.
"""
import os, json, hashlib
import numpy as np

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
R53 = ROOT + '/run_artifacts/02_nh3o3_reference/c1_negmode_dir053'
RR = ROOT + '/run_artifacts/02_nh3o3_reference/c1_negmode_dir053_resume01'
OUT = ROOT + '/run_artifacts/02_nh3o3_reference/c1_negmode_relax054'
JOBS = ROOT + '/jobs'
JOB_NO = '054'
SRC_FILE = RR + '/eval_disp_p02.json'
SYMS = ['N', 'H', 'H', 'H', 'O', 'O', 'O']
BPA = 1.0 / 0.52917721092


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


def check_number():
    import glob, re
    used = set()
    for f in glob.glob(JOBS + '/JOB-*.md'):
        m = re.match(r'JOB-2026-0906-(\d{3})', os.path.basename(f))
        if m:
            used.add(m.group(1))
    nxt = '%03d' % (max(int(u) for u in used) + 1)
    return dict(used_max=max(used), next_free=nxt,
                pass_=bool(nxt == JOB_NO and JOB_NO not in used))


def verify_source():
    src = json.load(open(SRC_FILE))
    m53 = json.load(open(R53 + '/input_manifest.json'))
    checks = {}
    checks['e_matches'] = bool(float(src['e_total'])
                               == -282.0017215435013)
    checks['gmax_matches'] = bool(float(src['grad_max'])
                                  == 1.5224820525232119e-5)
    checks['converged_finite'] = bool(src['converged'] and src['all_finite'])
    checks['x_bohr_saved'] = 'x_bohr' in src
    checks['config_complete'] = bool(
        src['config'].get('grid_response') is True
        and src['config'].get('grid_level') == 8
        and src['config'].get('d2_attached') is True)
    # identity: saved coords == the 053 displacement geometry at t=+0.02
    g02 = [g for g in m53['displacement_geometries']
           if g['t_Bohr'] == 0.02][0]
    xd = float(np.abs(np.asarray(src['x_bohr'], float).reshape(-1)
                      - np.asarray(g02['coords_bohr'], float)
                      .reshape(-1)).max())
    checks['identity_vs_053_geometry_maxdiff_Bohr'] = xd
    checks['identity_ulp'] = bool(xd == 0.0)
    # source is a displacement record of 053 (tag field; the persisted
    # eval file carries tag+bitwise coords, t_Bohr lived on the in-memory
    # copy - identity is proven by the bitwise match above + the tag)
    checks['tag_is_p02'] = bool(src.get('tag') == 'disp_p02')
    all_pass = all(bool(v) for v in checks.values()
                   if isinstance(v, bool))
    return src, checks, all_pass


def main():
    num = check_number()
    src, checks, ok = verify_source()
    man = dict(
        job='JOB-2026-0906-054 full-DOF relaxation from the 053 '
            'energy-lowered +0.02 Bohr point',
        number_check=num,
        caps=dict(start_repro=1, opt=25, recheck=1, total=27,
                  note='failures count within their category; no '
                       'borrowing; on exception: save and STOP, no retry, '
                       'no param change, no offline repair continuation'),
        stop_target=dict(gm_stop=1e-6,
                         gm_acceptance_unchanged=1e-5,
                         note='optimizer controlled stop at max|g|<=1e-6; '
                              'reaching 1e-5 does NOT stop early; the '
                              'start itself may exceed 1e-5 and is not '
                              'rejected for that; this stricter target '
                              'does not guarantee removing the negative '
                              'curvature nor convergence'),
        source=dict(source_path_wsl=SRC_FILE,
                    source_sha256=sha256_file(SRC_FILE),
                    e_total=float(src['e_total']),
                    grad_max=float(src['grad_max']),
                    x_bohr=src['x_bohr'],
                    coords_actual_angstrom=src['coords_actual_angstrom'],
                    grad=src['grad'],
                    config=src['config'],
                    element_order=SYMS,
                    continuation_mode='geometry continuation, optimizer '
                                      'state reset'),
        source_checks=checks, all_checks_pass=ok,
        bfgs=dict(method='BFGS', jac=True, gtol=1e-6, norm='inf',
                  xrtol=0, c1=1e-4, c2=0.9,
                  hess_inv0='21x21 identity (Bohr^2/Eh, algorithmic model, '
                            'not a computed Hessian; no old molecular '
                            'Hessian used)',
                  maxiter=200,
                  variables='21 Cartesian, Bohr; gradient Eh/Bohr; '
                            'unconstrained, no projection, no per-step '
                            'registration'))
    save_json_atomic(OUT + '/input_manifest.json', man)
    print('[054] number check:', json.dumps(num))
    print('[054] source checks:', json.dumps(
        {k: v for k, v in checks.items()}, default=str))
    if not (num['pass_'] and ok):
        print('[054] PRECHECK FAILED -> STOP (zero-evaluation stage)')
        raise SystemExit(1)
    print('[054] PREP OK (zero evaluations consumed)', flush=True)


if __name__ == '__main__':
    main()
