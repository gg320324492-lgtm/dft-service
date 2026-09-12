#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-057 step A: OFFLINE preparation (ZERO evaluations).

Full-DOF standard-BFGS optimization from the JOB-056 actual t=0.050 Bohr
lowest-sampled-energy point (eval_refine_t050.json), stop target = the
ORIGINAL acceptance gate max|g|<=1e-5; independent recheck on success.

Verifies: numbering, source identity (tag + coords vs the 056 manifest
geometry at ulp level), saved hashes, config; records the environment
(interpreter / scipy / pyscf versions and module paths).  Registration:
"geometry continuation, BFGS optimizer state reset".
"""
import os, sys, json, hashlib, glob, re
import numpy as np

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
R56 = ROOT + '/run_artifacts/02_nh3o3_reference/c1_refine056'
OUT = ROOT + '/run_artifacts/02_nh3o3_reference/c1_fdopt057'
JOBS = ROOT + '/jobs'
JOB_NO = '057'
SRC_FILE = R56 + '/eval_refine_t050.json'
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
    used = set()
    for f in glob.glob(JOBS + '/JOB-*.md'):
        m = re.match(r'JOB-2026-0906-(\d{3})', os.path.basename(f))
        if m:
            used.add(m.group(1))
    nxt = '%03d' % (max(int(u) for u in used) + 1)
    return dict(used_max=max(used), next_free=nxt,
                pass_=bool(nxt == JOB_NO and JOB_NO not in used))


def environment():
    import scipy
    import pyscf
    return dict(python=sys.version.split()[0],
                python_executable=sys.executable,
                scipy_version=scipy.__version__,
                scipy_module=scipy.__file__,
                pyscf_version=pyscf.__version__,
                pyscf_module=pyscf.__file__)


def main():
    num = check_number()
    src = json.load(open(SRC_FILE))
    m56 = json.load(open(R56 + '/input_manifest.json'))
    checks = {}
    checks['e_matches'] = bool(abs(float(src['e_total'])
                                   - (-282.0017217785996)) <= 1e-12)
    checks['gmax_matches'] = bool(abs(float(src['grad_max'])
                                      - 1.8902393805165122e-4) <= 1e-15)
    checks['converged_finite'] = bool(src['converged'] and src['all_finite'])
    checks['x_bohr_saved'] = 'x_bohr' in src
    checks['tag_is_refine_t050'] = bool(src.get('tag') == 'refine_t050')
    checks['config_complete'] = bool(
        src['config'].get('grid_response') is True
        and src['config'].get('grid_level') == 8
        and src['config'].get('d2_attached') is True)
    # identity: saved coords vs the 056 manifest geometry at t=0.050
    g050 = [g for g in m56['new_geometries'] if g['t_Bohr'] == 0.05][0]
    xd = float(np.abs(np.asarray(src['coords_actual_angstrom'], float)
                      .reshape(-1) * BPA
                      - np.asarray(g050['coords_bohr'],
                                   float).reshape(-1)).max())
    checks['identity_maxdiff_Bohr'] = xd
    checks['identity_match'] = bool(xd <= 1e-9)
    all_pass = all(bool(v) for v in checks.values() if isinstance(v, bool))
    man = dict(
        job='JOB-2026-0906-057 full-DOF BFGS optimization from the 056 '
            't=0.050 one-dimensional low point',
        number_check=num,
        caps=dict(start_repro=1, opt=25, recheck=1, total=27,
                  note='failures count; no borrowing; stop target = the '
                       'ORIGINAL acceptance gate max|g|<=1e-5; on budget '
                       'exhaustion / precision loss / line-search failure '
                       'or any exception: save and STOP, no recheck, no '
                       'param change, no other optimizer'),
        source=dict(source_path_wsl=SRC_FILE,
                    source_sha256=sha256_file(SRC_FILE),
                    e_total=float(src['e_total']),
                    grad_max=float(src['grad_max']),
                    x_bohr=src['x_bohr'],
                    coords_actual_angstrom=src['coords_actual_angstrom'],
                    grad=src['grad'], config=src['config'],
                    element_order=SYMS,
                    continuation_mode='geometry continuation, BFGS '
                                      'optimizer state RESET',
                    note='the interpolated t~0.0484 coords were NOT used; '
                         'the actual t=0.050 geometry is used as-is'),
        source_checks=checks, all_checks_pass=all_pass,
        environment=environment(),
        bfgs=dict(method='BFGS', jac=True, gtol=1e-6, norm='inf', xrtol=0,
                  c1=1e-4, c2=0.9,
                  hess_inv0='21x21 identity (Bohr^2/Eh, algorithmic model, '
                            'NOT a molecular Hessian)', maxiter=200,
                  variables='21 Cartesian, Bohr; gradient Eh/Bohr; '
                            'unconstrained, no projection, no per-step '
                            'registration'),
        analysis_spec=dict(compare='vs the 056 t=0.050 start: min-gradient '
                                   'point, lowest-energy point, last '
                                   'accepted point, max gradient change, '
                                   'fragment internal changes, NH3...O3 '
                                   'contact distance and orientation',
                           limits='energy lowering is NOT a binding free '
                                  'energy; a 1-D low point or optimizer '
                                  'success is NOT a stable complex; even '
                                  'passing the gate only registers a '
                                  'stationary-point candidate (stability '
                                  'and curvature in later batches)'))
    save_json_atomic(OUT + '/input_manifest.json', man)
    print('[057] number check:', json.dumps(num))
    print('[057] source checks:', json.dumps(
        {k: v for k, v in checks.items()}, default=str))
    print('[057] environment:', json.dumps(man['environment']))
    if not (num['pass_'] and all_pass):
        print('[057] PRECHECK FAILED -> STOP (zero-evaluation stage)')
        raise SystemExit(1)
    print('[057] PREP OK (zero evaluations consumed)', flush=True)


if __name__ == '__main__':
    main()
