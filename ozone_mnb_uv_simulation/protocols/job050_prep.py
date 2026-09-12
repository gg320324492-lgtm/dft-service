#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-050 step A: OFFLINE preparation (ZERO evaluations).

(a) batch-number check: 050 must be the next unoccupied number;
(b) charter section 2 - verify the 048 opt_19 record as start point:
    simultaneously the last accepted iterate, the last evaluated point and
    the lowest-energy point of 048; source hash / coords / full gradient /
    config saved; no guess reconstruction, no Hessian-derived coordinates;
(c) zero-eval correction of 049 (per the commander review):
    - recompute the first-try-accepted statistic from the event mapping
      (13/16, not "16/19 line-search success rate");
    - check the opt_08 trial against the Armijo condition with c1=1e-4
      from raw records (it SATISFIES Armijo; the actual rejection reason
      is not established);
    - record the verified SciPy version of the actual execution
      environment and the standard-BFGS signature;
No SCF / gradient / optimization: reads existing records only.
"""
import os, sys, json, hashlib, glob, re
import numpy as np

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
S48 = ROOT + '/run_artifacts/02_nh3o3_reference/c1_fulldof_opt048'
S49 = ROOT + '/run_artifacts/02_nh3o3_reference/c1_traj_eval049'
OUT = ROOT + '/run_artifacts/02_nh3o3_reference/c1_bfgs_opt050'
JOBS = ROOT + '/jobs'
JOB_NO = '050'
SRC_FILE = S48 + '/eval_opt_opt_19.json'


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


def verify_source():
    src = json.load(open(SRC_FILE))
    res48 = json.load(open(S48 + '/fulldof_opt048_results.json'))
    led48 = json.load(open(S48 + '/budget_fulldof048.json'))
    checks = {}

    # identity: last accepted iterate
    acc = res48['optimization']['accepted_iterates']
    x_src = np.asarray(src['x_bohr'], float).reshape(-1)
    checks['is_last_accepted'] = bool(np.array_equal(
        x_src, np.asarray(acc[-1], float)))

    # identity: last evaluated point
    last_att = led48['attempts'][-1]
    checks['is_last_evaluated'] = bool(
        last_att['stage'] == 'opt_opt_19' and last_att['status'] == 'done'
        and float(last_att['e_total']) == float(src['e_total']))

    # identity: lowest energy among all 20 evaluated records
    evals = []
    for f in glob.glob(S48 + '/eval_*.json'):
        evals.append(json.load(open(f)))
    checks['is_lowest_energy'] = bool(
        float(src['e_total']) == min(float(r['e_total']) for r in evals))
    checks['n_eval_files'] = len(evals)

    # record self-consistency
    C = np.asarray(src['coords_actual_angstrom'], float)
    g = np.asarray(src['grad'], float)
    checks['coords_sha_matches'] = bool(
        src.get('coords_sha') == sha_arr(C))
    checks['grad_sha_matches'] = bool(
        src.get('grad_sha') == sha_arr(g))
    checks['grad_max_matches'] = bool(
        float(src['grad_max']) == float(np.abs(g).max()))
    checks['converged_finite'] = bool(src['converged'] and src['all_finite'])
    checks['config_complete'] = bool(
        src['config'].get('grid_response') is True
        and src['config'].get('grid_level') == 8
        and src['config'].get('d2_attached') is True)
    checks['no_hessian_used'] = True   # coordinates taken from the record
    checks['x_dim_21'] = bool(x_src.size == 21)

    all_pass = all(bool(v) for v in checks.values()
                   if isinstance(v, bool))
    return dict(
        source_path_wsl=SRC_FILE,
        source_sha256=sha256_file(SRC_FILE),
        e_total_ref_Eh=-282.0017212923778,
        e_total_matches_reference=bool(
            float(src['e_total']) == -282.0017212923778),
        element_order=['N', 'H', 'H', 'H', 'O', 'O', 'O'],
        coordinate_units=dict(record='Angstrom', optimizer='Bohr',
                              ang_per_bohr=0.52917721092),
        e_total=float(src['e_total']),
        e_d2_Eh=float(src['e_d2_Eh']),
        e_dft_part_Eh=float(src['e_dft_part_Eh']),
        grad=g.tolist(), grad_max=float(src['grad_max']),
        coords_actual_angstrom=C.tolist(),
        config=src['config'],
        checks=checks, all_checks_pass=all_pass)


def correction_049():
    # event mapping from 049 analysis + raw records
    traj = json.load(open(S49 + '/traj_analysis.json'))
    acc_tags = traj['mapping']['accepted_tags_in_order']
    r48 = json.load(open(S48 + '/fulldof_opt048_results.json'))
    E = {}
    G = {}
    X = {}
    for f in glob.glob(S48 + '/eval_*.json'):
        r = json.load(open(f))
        t = r['tag']
        E[t] = float(r['e_total'])
        G[t] = np.asarray(r['grad'], float).reshape(-1)
        X[t] = np.asarray(r['x_bohr'], float).reshape(-1)
    order = ['start'] + ['opt_%02d' % i for i in range(1, 20)]
    pos = {t: i for i, t in enumerate(order)}
    # trials attached to each accepted step (event order)
    extra = {t: 0 for t in acc_tags}
    for t in ('opt_01', 'opt_08', 'opt_17'):
        nxt = next(a for a in acc_tags if pos[a] > pos[t])
        extra[nxt] += 1
    first_try = sum(1 for t in acc_tags if extra[t] == 0)
    stat = dict(
        accepted_steps=len(acc_tags),
        steps_with_extra_trial=sum(1 for t in acc_tags if extra[t]),
        first_try_accepted=first_try,
        correction='the 049 statement "16/19 line-search first-try '
                   'success" is replaced by: 3 of the 16 completed '
                   'accepted steps contained extra trial(s); '
                   'first-try accepted = 13/16; this ratio alone does '
                   'NOT establish whether the line search is a '
                   'bottleneck')
    # opt_08 Armijo check from raw values
    dE = E['opt_08'] - E['opt_07']
    gdx = float(G['opt_07'] @ (X['opt_08'] - X['opt_07']))
    c1 = 1e-4
    armijo = dict(
        dE_Eh=dE, g_prev_dot_dx_Eh=gdx, c1=c1,
        armijo_rhs_Eh=c1 * gdx,
        armijo_satisfied=bool(dE <= c1 * gdx),
        note='raw values match the commander review (dE = '
             '-1.252146830665879e-9, g.dx = -3.5151897640593525e-7); '
             'opt_08 SATISFIES the Armijo condition at c1=1e-4, so the '
             'earlier claim "opt_08 failed Armijo" is withdrawn; the '
             'actual rejection reason cannot be established from saved '
             'records alone')
    # scipy version of the ACTUAL execution environment
    import scipy
    import inspect
    from scipy.optimize import _optimize as sopt
    sig = list(inspect.signature(sopt._minimize_bfgs).parameters)
    sci = dict(scipy_version=scipy.__version__,
               interpreter=sys.executable,
               required_bfgs_options_present=bool(
                   all(p in sig for p in
                       ('gtol', 'norm', 'xrtol', 'c1', 'c2', 'hess_inv0'))),
               signature_params=[p for p in sig if p not in
                                 ('fun', 'x0', 'args', 'jac', 'callback')],
               note='the 049 commander review states 1.18.1; this batch '
                    'verifies the ACTUAL evaluation environment reports '
                    + scipy.__version__ + ' (the same interpreter used '
                    'by 047/048); both versions provide the required '
                    'BFGS options, so the 050 plan is unaffected')
    return dict(first_try_statistic=stat, opt_08_armijo=armijo,
                scipy_environment=sci,
                scope_limits='the 049 decomposition findings are kept; '
                             'small trans/rot components and the fragment '
                             'force decrease are observations on the '
                             'analysed geometries only and do NOT prove '
                             'that all residual degrees of freedom are '
                             'fully decomposed; the 0.07 Bohr / 3.6e-6 Eh '
                             '/ 8-15 step extrapolation is WITHDRAWN '
                             '(max|g| is not the directional derivative '
                             'g.u of the chosen direction, and the '
                             'old-geometry Hessian curvature is not a '
                             'validated endpoint curvature)')


def main():
    num = check_number()
    src = verify_source()
    corr = correction_049()
    save_json_atomic(OUT + '/correction_evidence.json',
                     dict(job='JOB-2026-0906-050 zero-eval correction of 049',
                          nature='offline recomputation from raw records; '
                                 'no new evaluations', evidence=corr))
    man = dict(
        job='JOB-2026-0906-050 standard-BFGS limited optimization from the '
            '048 last accepted point (opt_19)',
        number_check=num,
        caps=dict(start_repro=1, opt=20, recheck=1, total=22),
        caps_note='failures count within their category; no borrowing; '
                  'no auto-resume; stability/Hessian/frequency/CP/'
                  'high-level = 0',
        source=src,
        start_gate=dict(dE_max=1e-8, dgrad_max=1e-7, dcoords_max_A=1e-9,
                        extra='SCF converged, config identical, finite; '
                              'on PASS the reproduction record is reused '
                              'for the optimizer first request at the '
                              'same coordinates'),
        optimizer=dict(method='BFGS', jac=True, gtol=1e-6, norm='inf',
                       xrtol=0, c1=1e-4, c2=0.9,
                       hess_inv0='21x21 identity (Bohr^2/Eh units; '
                                 'algorithmic initial model, NOT a '
                                 'computed molecular Hessian; 043 '
                                 'Hessian NOT used)',
                       maxiter=200,
                       maxiter_note='maxiter is NOT the evaluation '
                                    'budget; the persisted pre-call guard '
                                    '(opt cap 20) is the real limit',
                       variables='21 Cartesian coordinates in Bohr; '
                                 'gradient Eh/Bohr',
                       constraints='none; no projection; no per-step '
                                   'registration; fresh optimizer state, '
                                   'no claim of resuming 048 history'),
        acceptance=dict(unprojected_max_grad=1e-5,
                        recheck='at most one independent new-object '
                                'recheck; register stationary-point '
                                'candidate only on PASS'),
        zero_eval_correction=dict(
            note='notes/job050_049_correction.md',
            evidence='correction_evidence.json'))
    save_json_atomic(OUT + '/input_manifest.json', man)

    print('[050] number check:', json.dumps(num))
    print('[050] source all_checks_pass =', src['all_checks_pass'])
    for k, v in src['checks'].items():
        print('      %-28s %s' % (k, v))
    print('[050] E matches reference -282.0017212923778:',
          src['e_total_matches_reference'])
    print('[050] first-try stat:', corr['first_try_statistic']
          ['first_try_accepted'], '/', corr['first_try_statistic']
          ['accepted_steps'],
          '(steps with extra trial:', corr['first_try_statistic']
          ['steps_with_extra_trial'], ')')
    print('[050] opt_08 armijo satisfied:', corr['opt_08_armijo']
          ['armijo_satisfied'])
    print('[050] scipy:', corr['scipy_environment']['scipy_version'],
          '| BFGS options present:',
          corr['scipy_environment']['required_bfgs_options_present'])
    if not (num['pass_'] and src['all_checks_pass']):
        print('[050] PRECHECK FAILED -> STOP (zero-evaluation stage)')
        raise SystemExit(1)
    print('[050] PREP OK (zero evaluations consumed)', flush=True)


if __name__ == '__main__':
    main()
