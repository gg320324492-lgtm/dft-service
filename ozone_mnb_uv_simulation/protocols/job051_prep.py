#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-051 step A: OFFLINE preparation (ZERO evaluations).

(a) batch-number check: 051 must be the next unoccupied number;
(b) zero-eval correction of 050 (per the commander review), verified from
    raw records:
    - 048 and 050 have DIFFERENT starting points -> the "same-start
      comparison" of descent speed / optimizer ranking is withdrawn;
    - the 22 optimizer requests = 1 start-cache reuse + 20 evaluated &
      accepted + 1 budget-BLOCKED request (not evaluated; NOT a
      line-search rejection) -> "1 extra trial in 20 steps" withdrawn;
    - SCF time increase: observation only (no iteration-log evidence for
      attribution);
    - the 'unclosed file' warning is kept verbatim (no source attribution,
      no harmlessness claim);
    - environment: actual interpreter /usr/bin/python3, SciPy 1.18.0 and
      PySCF 2.14.0 with module paths; other-interpreter records kept
      separately;
    - 049's "100% internal" wording is superseded by "internal-dominant,
      external components not strictly zero".
(c) charter section 4 - verify the 050 opt_20 record as the unique
    continuation start: simultaneously 050's last accepted point, last
    completed evaluation and lowest-energy point; register "geometry
    continuation, optimizer state RESET" (050 saved no inverse-Hessian
    history).
No SCF / gradient / optimization: reads existing records only.
"""
import os, sys, json, hashlib, glob, re
import numpy as np

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
S48 = ROOT + '/run_artifacts/02_nh3o3_reference/c1_fulldof_opt048'
S50 = ROOT + '/run_artifacts/02_nh3o3_reference/c1_bfgs_opt050'
OUT = ROOT + '/run_artifacts/02_nh3o3_reference/c1_bfgs_cont051'
JOBS = ROOT + '/jobs'
JOB_NO = '051'
SRC_FILE = S50 + '/eval_opt_opt_20.json'


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


def correction_050():
    res50 = json.load(open(S50 + '/bfgs_opt050_results.json'))
    o50 = res50['optimization']
    n_req = len(o50['trial_sequence'])
    n_acc = len(o50['accepted_iterates'])
    n_new = o50['n_new_evals']
    n_blocked = n_req - n_acc - (1 if o50['reuse_start'] else 0)
    # the blocked request has NO evaluated record at its coordinates
    recs = []
    for f in glob.glob(S50 + '/eval_opt_opt_*.json'):
        r = json.load(open(f))
        recs.append(np.asarray(r['x_bohr'], float).reshape(-1))
    blocked = [np.asarray(x, float) for x in o50['trial_sequence']]
    # blocked = requests with no bitwise match among evaluated records
    unmatched = []
    for x in blocked:
        if not any(np.array_equal(x, r) for r in recs):
            unmatched.append(x)
    n_unmatched = len(unmatched) - (1 if o50['reuse_start'] else 0)
    # the last unmatched request equals the last entry of trial_sequence
    last_is_blocked = bool(
        o50['trial_sequence']
        and not any(np.array_equal(np.asarray(o50['trial_sequence'][-1],
                                              float), r) for r in recs))
    requests = dict(
        total_requests=n_req, start_cache_reuse=1 if o50['reuse_start'] else 0,
        evaluated_and_accepted=n_acc, evaluated_new=n_new,
        budget_blocked_not_evaluated=n_unmatched,
        last_request_is_blocked=last_is_blocked,
        correction='the 22 requests are: 1 start-cache reuse, 20 evaluated '
                   'and accepted, 1 budget-BLOCKED and NOT evaluated; the '
                   'blocked request has NO energy/gradient and is NOT a '
                   'line-search rejection; the 050 statements "1 extra '
                   'trial in 20 accepted steps" and the derived "line '
                   'search more stable" judgment are WITHDRAWN',
        scf_times='seconds recorded per evaluation (75.9 - 391.3 s, '
                  'increasing late in the run); OBSERVATION ONLY - no '
                  'iteration-log evidence for attribution to a flatter '
                  'valley, a numerical floor, or more SCF iterations',
        unclosed_file_warning='unclosed file <_io.BufferedRandom '
                              'name=\'/tmp/tmpqk4zksbf\'> (kept verbatim; '
                              'no source attribution, no harmlessness '
                              'claim)',
        comparison='048 started from the 045 +0.02 Bohr displacement '
                   'point; 050 started from 048 opt_19 -> NOT a same-start '
                   'comparison; the descent-speed and optimizer-ranking '
                   'statements are WITHDRAWN; each batch lists only its '
                   'own actual descent and minimum gradient (048: '
                   '-1.4276e-6 Eh, min 3.669e-5; 050: -1.7706e-7 Eh, '
                   'min 1.308e-5)',
        internal_wording='049 "100% internal space" superseded by '
                         '"internal components dominant, external '
                         'components not strictly zero" (trans ~1e-13, '
                         'rot ~1e-6 Eh/Bohr, never exactly zero)')
    import scipy
    import pyscf
    env = dict(
        actual_interpreter=sys.executable,
        scipy_version=scipy.__version__,
        scipy_module=scipy.__file__,
        pyscf_version=pyscf.__version__,
        pyscf_module=pyscf.__file__,
        numpy_version=np.__version__,
        note='this is the SAME interpreter that executed 050; the earlier '
             'commander-review record of SciPy 1.18.1 came from a DIFFERENT '
             'Python environment and is kept as a separate record, not '
             'overwritten; no installation or upgrade performed')
    return dict(requests=requests, environment=env)


def verify_source():
    src = json.load(open(SRC_FILE))
    res50 = json.load(open(S50 + '/bfgs_opt050_results.json'))
    led50 = json.load(open(S50 + '/budget_bfgs050.json'))
    checks = {}
    acc = res50['optimization']['accepted_iterates']
    x_src = np.asarray(src['x_bohr'], float).reshape(-1)
    checks['is_last_accepted'] = bool(np.array_equal(
        x_src, np.asarray(acc[-1], float)))
    last_att = led50['attempts'][-1]
    checks['is_last_evaluated'] = bool(
        last_att['stage'] == 'opt_opt_20' and last_att['status'] == 'done'
        and float(last_att['e_total']) == float(src['e_total']))
    evals = [json.load(open(f)) for f in glob.glob(S50 + '/eval_*.json')]
    checks['is_lowest_energy'] = bool(
        float(src['e_total']) == min(float(r['e_total']) for r in evals))
    checks['n_eval_files_050'] = len(evals)
    C = np.asarray(src['coords_actual_angstrom'], float)
    g = np.asarray(src['grad'], float)
    checks['coords_sha_matches'] = bool(src.get('coords_sha') == sha_arr(C))
    checks['grad_sha_matches'] = bool(src.get('grad_sha') == sha_arr(g))
    checks['grad_max_matches'] = bool(
        float(src['grad_max']) == float(np.abs(g).max()))
    checks['converged_finite'] = bool(src['converged'] and src['all_finite'])
    checks['config_complete'] = bool(
        src['config'].get('grid_response') is True
        and src['config'].get('grid_level') == 8
        and src['config'].get('d2_attached') is True)
    checks['x_dim_21'] = bool(x_src.size == 21)
    checks['not_opt06_or_guess_or_hessian'] = bool(
        float(src['e_total']) == -282.00172146935444)
    all_pass = all(bool(v) for v in checks.values()
                   if isinstance(v, bool))
    return dict(
        source_path_wsl=SRC_FILE,
        source_sha256=sha256_file(SRC_FILE),
        e_total_ref_Eh=-282.00172146935444,
        grad_max_ref=2.0018948453250075e-05,
        e_total_matches_reference=bool(
            float(src['e_total']) == -282.00172146935444),
        element_order=['N', 'H', 'H', 'H', 'O', 'O', 'O'],
        coordinate_units=dict(record='Angstrom', optimizer='Bohr',
                              ang_per_bohr=0.52917721092),
        e_total=float(src['e_total']),
        e_d2_Eh=float(src['e_d2_Eh']),
        e_dft_part_Eh=float(src['e_dft_part_Eh']),
        grad=g.tolist(), grad_max=float(src['grad_max']),
        coords_actual_angstrom=C.tolist(),
        config=src['config'],
        continuation=dict(mode='geometry continuation, optimizer state '
                             'RESET; 050 saved no inverse-Hessian history '
                             '-> no claim of resuming optimizer history'),
        checks=checks, all_checks_pass=all_pass)


def main():
    num = check_number()
    src = verify_source()
    corr = correction_050()
    save_json_atomic(OUT + '/correction_evidence.json',
                     dict(job='JOB-2026-0906-051 zero-eval correction of 050',
                          nature='offline recomputation from raw records; '
                                 'no new evaluations', evidence=corr))
    man = dict(
        job='JOB-2026-0906-051 standard-BFGS limited continuation from the '
            '050 last accepted point (opt_20)',
        number_check=num,
        caps=dict(start_repro=1, opt=20, recheck=1, total=22),
        caps_note='failures count within their category; no borrowing; '
                  'no auto-resume; stability/Hessian/frequency/CP/'
                  'high-level/other-species = 0',
        source=src,
        start_gate=dict(dE_max=1e-8, dgrad_max=1e-7, dcoords_max_A=1e-9,
                        extra='config identical, SCF converged, finite; '
                              'on PASS the reproduction record is reused '
                              'for the optimizer first request at the same '
                              'coordinates (no repeated SCF)'),
        optimizer=dict(method='BFGS', jac=True, gtol=1e-6, norm='inf',
                       xrtol=0, c1=1e-4, c2=0.9,
                       hess_inv0='21x21 identity (Bohr^2/Eh; algorithmic '
                                 'initial approximation, NOT a molecular '
                                 'Hessian; 043 matrix NOT used)',
                       maxiter=200,
                       maxiter_note='algorithm parameter only; the real '
                                    'limit is the persisted pre-call opt '
                                    'cap 20',
                       variables='21 Cartesian coordinates in Bohr; '
                                 'gradient Eh/Bohr',
                       constraints='none; no projection; no per-step '
                                   'registration',
                       method_and_units='identical to 050: D2 attached '
                                        'once, grid level 8, original SCF '
                                        'tolerances, grid_response=True',
                       bookkeeping='optimizer requests / completed '
                                   'evaluations / accepted iterates / '
                                   'budget-BLOCKED requests recorded '
                                   'SEPARATELY; a blocked request is never '
                                   'counted as a line-search rejection'),
        acceptance=dict(unprojected_max_grad=1e-5,
                        recheck='at most one independent new-object '
                                'recheck; register stationary-point '
                                'candidate only on PASS'),
        zero_eval_correction=dict(
            note='notes/job051_050_correction.md',
            evidence='correction_evidence.json'))
    save_json_atomic(OUT + '/input_manifest.json', man)

    print('[051] number check:', json.dumps(num))
    print('[051] source all_checks_pass =', src['all_checks_pass'])
    for k, v in src['checks'].items():
        print('      %-32s %s' % (k, v))
    print('[051] E matches reference -282.00172146935444:',
          src['e_total_matches_reference'])
    rq = corr['requests']
    print('[051] 050 requests: total=%d reuse=%d accepted=%d blocked=%d '
          '(last blocked: %s)' % (rq['total_requests'],
                                  rq['start_cache_reuse'],
                                  rq['evaluated_and_accepted'],
                                  rq['budget_blocked_not_evaluated'],
                                  rq['last_request_is_blocked']))
    env = corr['environment']
    print('[051] env: %s | scipy %s | pyscf %s'
          % (env['actual_interpreter'], env['scipy_version'],
             env['pyscf_version']))
    if not (num['pass_'] and src['all_checks_pass']):
        print('[051] PRECHECK FAILED -> STOP (zero-evaluation stage)')
        raise SystemExit(1)
    print('[051] PREP OK (zero evaluations consumed)', flush=True)


if __name__ == '__main__':
    main()
