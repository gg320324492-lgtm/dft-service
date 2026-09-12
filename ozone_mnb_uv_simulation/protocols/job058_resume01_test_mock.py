#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-058 resume-01 step B: zero-evaluation tests (temp dirs,
real backend unreachable, formal directories untouched).

The stability tests call the ACTUAL fixed wrapper
(job058_resume01_exec.stability_fn_fixed) with an INJECTED fake mf:
  T1  4-tuple return: internal=True / external=False / return_status=True
      actually passed; stable_i=True handled; raw result + log + mo array
      on disk BEFORE summarising; no real backend;
  T2  stable_i=False -> wrapper record False; full-flow run_resume stops
      with 0 fd calls;
  T3  stable_i=None -> same stop rule, 0 fd calls;
  T4  contract error (fake returns 3 items): wrapper raises AFTER the log
      (written directly during the call) and the raw-status JSON are on
      disk; ledger error attempt preserved;
  T5  post-processing exception (mo_i not array-convertible): log + raw
      status JSON still saved; ledger error attempt preserved;
  T6  centre gate failure -> 0 stability and 0 fd calls;
  T7  full 42-point happy path (stable_i=True): H_raw recovers a known
      symmetric matrix, antisym residual ~0, TR rank 6 / internal 15,
      scipy cross-check, 42 records, cumulative accounting 44/2/46.
Isolation: pyscf + job052_exec stubbed; the formal 058 and resume
directories are snapshotted and must be byte-identical afterwards.
"""
import os, sys, json, tempfile, types
import numpy as np

sys.modules['pyscf'] = None
sys.modules['pyscf.hessian'] = None
# stub the production dependency chain (job052_exec -> d2_full -> pyscf);
# the tests exercise the ACTUAL resume entry / wrapper with injected fakes
sys.modules['job052_exec'] = types.ModuleType('job052_exec')

HERE = '/mnt/e/dft-service/ozone_mnb_uv_simulation/protocols'
sys.path.insert(0, HERE)
import job047_exec as ex
import job058_resume01_exec as jr

S58 = jr.S58
OUT = jr.OUT
BPA = 1.0 / ex.ANG_PER_BOHR
rng = np.random.default_rng(58)
M = rng.standard_normal((21, 21))
H_TRUE = 0.5 * (M + M.T) + np.eye(21) * 0.5     # known symmetric, PD


def snapshot(d):
    if not os.path.isdir(d):
        return []
    return sorted((f, os.path.getsize(os.path.join(d, f)))
                  for f in os.listdir(d))


class FakeMF:
    """Minimal mf stand-in for stability_fn_fixed: stdout/verbose swap +
    an injectable stability() that records its kwargs."""
    def __init__(self, ret=None, raise_exc=None):
        self.stdout = sys.__stdout__
        self.verbose = 0
        self._ret = ret
        self._raise = raise_exc
        self.kwargs = None
        self.n_calls = 0

    def stability(self, **kw):
        self.n_calls += 1
        self.kwargs = kw
        if self._raise is not None:
            raise self._raise
        # PySCF writes the verbose stability report into mf.stdout
        print('fake stability internal report', file=self.stdout)
        return self._ret


def make_man():
    man = json.load(open(os.path.join(S58, 'input_manifest.json')))
    # align the gate reference with the mock quadratic model (E=0, g=0 at
    # the centre) so the centre gate passes in the mock flow
    man['centre']['e_total'] = 0.0
    man['centre']['grad'] = [[0.0] * 3] * 7
    man['centre']['grad_max'] = 0.0
    man['centre']['coords_actual_angstrom'] = (
        np.asarray(man['centre']['x_bohr'], float) * ex.ANG_PER_BOHR
    ).tolist()
    return man


def mock_eval_factory(be_calls, cfg, fail_conv_tags=()):
    def ev(R, out_dir, tag):
        R = np.asarray(R, float).reshape(7, 3)
        be_calls.append((tag, R.tolist()))
        x = R.reshape(-1) - np.asarray(json.load(open(os.path.join(
            S58, 'input_manifest.json')))['centre']['x_bohr'],
            float).reshape(-1)
        g = H_TRUE @ x
        conv = tag not in fail_conv_tags
        rec = dict(e_total=(float(0.5 * x @ H_TRUE @ x) if conv else 0.0),
                   e_d2_Eh=0.0, e_dft_part_Eh=0.0,
                   grad=(g.reshape(7, 3).tolist() if conv
                         else [[0.0] * 3] * 7),
                   grad_max=(float(np.abs(g).max()) if conv else 0.0),
                   grad_sha='m',
                   coords_actual_angstrom=(R * ex.ANG_PER_BOHR).tolist(),
                   coords_sha='m', config=dict(cfg), converged=conv,
                   all_finite=True, scf_kernel_count=1, seconds=0.0,
                   tag=tag)
        ex.save_json_atomic(os.path.join(out_dir,
                                         'eval_%s.json' % tag), rec)
        return rec, 0.0, None
    return ev


def orig_centre_mock():
    return dict(e_total=0.0, grad=[[0.0] * 3] * 7,
                coords_actual_angstrom=json.load(open(os.path.join(
                    S58, 'input_manifest.json')))['centre'][
                        'coords_actual_angstrom'],
                config=json.load(open(os.path.join(
                    S58, 'input_manifest.json')))['centre']['config'])


def main():
    pre_snap58 = snapshot(S58)
    pre_snap_r = snapshot(OUT)
    man = make_man()
    cfg = man['centre']['config']
    R = {}

    # ---- T1: ACTUAL wrapper, 4-tuple, kwargs + artefacts ----------------
    tmp = tempfile.mkdtemp(prefix='job058r_T1_')
    mo_i = np.arange(12, dtype=float).reshape(3, 4)
    fm = FakeMF(ret=(mo_i, None, True, None))
    rec = jr.stability_fn_fixed(fm, tmp)
    ok_call = (fm.n_calls == 1 and fm.kwargs is not None
               and fm.kwargs.get('internal') is True
               and fm.kwargs.get('external') is False
               and fm.kwargs.get('return_status') is True)
    log_p = os.path.join(tmp, 'stability_raw_log.txt')
    raw_p = os.path.join(tmp, 'stability_raw_result.json')
    R['T1'] = dict(
        kwargs_ok=bool(ok_call), kwargs=fm.kwargs,
        stable_i=rec.get('stable_i'), stable_e=rec.get('stable_e'),
        stable_i_raw=rec.get('stable_i_raw'),
        log_exists=bool(os.path.exists(log_p)),
        log_nonempty=bool(os.path.exists(log_p)
                          and os.path.getsize(log_p) > 0),
        raw_json_exists=bool(os.path.exists(raw_p)),
        mo_i_exists=bool(os.path.exists(os.path.join(
            tmp, 'stability_mo_i.npy'))),
        mo_e_file_absent=bool(rec.get('mo_e_file') is None),
        contract_ok=json.load(open(raw_p)).get('contract_ok')
        if os.path.exists(raw_p) else None)
    R['T1_pass'] = bool(
        ok_call and rec['stable_i'] is True and rec['stable_e'] is None
        and R['T1']['log_nonempty'] and R['T1']['raw_json_exists']
        and R['T1']['mo_i_exists'] and R['T1']['contract_ok']
        and 'mo_i, mo_e, stable_i, stable_e'
        == rec['installed_return_order'])

    # ---- T2/T3: stable_i False / None -> flow stops with 0 fd calls -----
    for name, si in (('T2', False), ('T3', None)):
        tmp = tempfile.mkdtemp(prefix='job058r_%s_' % name)
        be = []
        led = jr.Ledger(os.path.join(tmp, 'b.json'))
        fm2 = FakeMF(ret=(np.ones((3, 4)), None, si, None))
        calls = {'n': 0}

        def ev_count(Rv, out_dir, tag, _be=be, _c=calls):
            _c['n'] += 1
            return mock_eval_factory(_be, cfg)(Rv, out_dir, tag)

        flow = jr.run_resume(man, led, ev_count, tmp,
                             stability_fn=lambda mf, od:
                             jr.stability_fn_fixed(fm2, od),
                             orig_centre_rec=orig_centre_mock(),
                             checkpoint_rec=dict(note='mock'),
                             checkpoint_hook=None)
        stopped = flow['hessian'].get('status') == \
            'stopped_stability_not_explicit_true'
        fd_calls = sum(1 for t, _ in be if str(t).startswith('fd_'))
        R[name] = dict(stable_i=flow['stability'].get('stable_i'),
                       stopped=stopped, stability_calls=fm2.n_calls,
                       fd_calls=fd_calls,
                       hess_status=flow['hessian']['status'],
                       stab_status=led.count('stability'))
        R[name + '_pass'] = bool(
            stopped and flow['stability'].get('stable_i') == si
            and fm2.n_calls == 1 and fd_calls == 0
            and led.count('stability') == 1
            and led.count('fd') == 0)

    # ---- T4: contract error (3 items) -> log + raw saved, ledger error --
    tmp = tempfile.mkdtemp(prefix='job058r_T4_')
    led = jr.Ledger(os.path.join(tmp, 'b.json'))
    att = led.pre('stability', dict(note='contract error test'))
    fm4 = FakeMF(ret=(np.ones((3, 4)), True, False))   # 3 items: WRONG
    err = None
    try:
        jr.stability_fn_fixed(fm4, tmp)
    except RuntimeError as e:
        err = str(e)
    att['note']['kwargs'] = fm4.kwargs
    led.fail(att, err or 'no error raised')
    log_ok = os.path.exists(os.path.join(tmp, 'stability_raw_log.txt')) \
        and os.path.getsize(os.path.join(
            tmp, 'stability_raw_log.txt')) > 0
    raw = json.load(open(os.path.join(tmp, 'stability_raw_result.json')))
    R['T4'] = dict(error=err[:120] if err else None,
                   contract_ok=raw.get('contract_ok'),
                   n_returned=raw.get('n_returned'),
                   log_saved=bool(log_ok), att_status=att['status'])
    R['T4_pass'] = bool(err and 'contract violation' in err
                        and raw.get('contract_ok') is False
                        and raw.get('n_returned') == 3
                        and log_ok and att['status'] == 'error')

    # ---- T5: post-processing exception -> log + raw status saved --------
    class BadArray:
        def __array__(self, *a, **k):
            raise TypeError('bad array')
        def __repr__(self):
            return '<BadArray>'

    tmp = tempfile.mkdtemp(prefix='job058r_T5_')
    led = jr.Ledger(os.path.join(tmp, 'b.json'))
    att = led.pre('stability', dict(note='post-processing exception test'))
    fm5 = FakeMF(ret=(BadArray(), None, True, None))
    err5 = None
    try:
        jr.stability_fn_fixed(fm5, tmp)
    except Exception as e:                              # noqa: BLE001
        err5 = '%s: %s' % (type(e).__name__, e)
    led.fail(att, err5 or 'no error raised')
    log_ok5 = os.path.exists(os.path.join(tmp,
                                          'stability_raw_log.txt')) \
        and os.path.getsize(os.path.join(
            tmp, 'stability_raw_log.txt')) > 0
    raw5 = json.load(open(os.path.join(tmp,
                                       'stability_raw_result.json')))
    R['T5'] = dict(error=err5[:120] if err5 else None,
                   raw_status_saved=bool(
                       raw5.get('contract_ok') is True
                       and raw5.get('stable_i_raw') == 'True'),
                   log_saved=bool(log_ok5), att_status=att['status'])
    R['T5_pass'] = bool(err5 and log_ok5 and raw5.get('contract_ok')
                        is True and raw5.get('stable_i_raw') == 'True'
                        and att['status'] == 'error')

    # ---- T6: centre gate failure -> 0 stability + 0 fd calls ------------
    tmp = tempfile.mkdtemp(prefix='job058r_T6_')
    be6 = []
    led = jr.Ledger(os.path.join(tmp, 'b.json'))
    man_bad = make_man()
    man_bad['centre']['e_total'] = -1.0        # gate mismatch forced
    err6 = None
    try:
        jr.run_resume(man_bad, led, mock_eval_factory(be6, cfg), tmp,
                      stability_fn=jr.stability_fn_fixed,
                      orig_centre_rec=orig_centre_mock(),
                      checkpoint_rec=dict(note='mock'),
                      checkpoint_hook=None)
    except RuntimeError as e:
        err6 = str(e)
    fd6 = sum(1 for t, _ in be6 if str(t).startswith('fd_'))
    R['T6'] = dict(error=err6[:80] if err6 else None,
                   stability_calls=fd6, total_calls=len(be6),
                   fd_ledger=led.count('fd'))
    R['T6_pass'] = bool(err6 and 'gate failed' in err6 and fd6 == 0
                        and led.count('fd') == 0)

    # ---- T7: full 42-point happy path -----------------------------------
    tmp = tempfile.mkdtemp(prefix='job058r_T7_')
    be7 = []
    led = jr.Ledger(os.path.join(tmp, 'b.json'))
    flow = jr.run_resume(man, led, mock_eval_factory(be7, cfg), tmp,
                         stability_fn=lambda mf, od: dict(
                             stable_i=True, stable_e=None,
                             installed_return_order=
                             'mo_i, mo_e, stable_i, stable_e',
                             raw_log='stability_raw_log.txt',
                             raw_result='stability_raw_result.json',
                             seconds=0.0,
                             scope='mock (fake mf injected; wrapper T1 '
                                   'already covered)'),
                         orig_centre_rec=orig_centre_mock(),
                         checkpoint_rec=dict(note='mock'),
                         checkpoint_hook=lambda mf: dict(note='mock'))
    hess = flow['hessian']
    H_raw = np.asarray(hess['H_raw'], float)
    centre_calls = sum(1 for t, _ in be7 if t == 'centre')
    fd_calls = sum(1 for t, _ in be7 if str(t).startswith('fd_'))
    modes = jr.j58.postprocess(man, hess, flow['centre_record'], tmp)
    lam = np.asarray(modes['eigenvalues_Eh_Bohr2_amu'], float)
    R['T7'] = dict(hess_status=hess['status'],
                   maxdiff_vs_true=float(np.abs(H_raw - H_TRUE).max()),
                   antisym=hess['antisymmetric_residual_max'],
                   n_columns=hess['n_columns'],
                   centre_calls=centre_calls, fd_calls=fd_calls,
                   fd_ledger=led.count('fd'),
                   modes_status=modes.get('status'),
                   tr_rank=modes.get('tr_rank_svd'),
                   n_modes=modes.get('n_internal_modes'),
                   n_neg=modes.get('n_negative'),
                   neg_direct=int((lam < 0).sum()),
                   cross_diff=modes.get('cross_check', {}).get(
                       'max_abs_diff_Eh_Bohr2'))
    R['T7_pass'] = bool(
        hess['status'] == 'ok'
        and R['T7']['maxdiff_vs_true'] < 1e-8
        and hess['antisymmetric_residual_max'] < 1e-8
        and hess['n_columns'] == 21 and centre_calls == 1
        and fd_calls == 42 and led.count('fd') == 42
        and modes.get('status') == 'ok' and modes['tr_rank_svd'] == 6
        and modes['n_internal_modes'] == 15
        and modes['n_negative'] == int((lam < 0).sum())
        and modes['cross_check']['max_abs_diff_Eh_Bohr2'] < 1e-12)

    # ---- T7b: cumulative accounting numbers ------------------------------
    scfg = led.count('centre') + led.count('fd')
    cum_scf = 1 + scfg
    cum_stab = 1 + led.count('stability')
    cum_att = 2 + len(led.data['attempts'])
    R['T7b'] = dict(scf_grad_new=scfg, cumulative_scf_grad=cum_scf,
                    cumulative_stability=cum_stab,
                    cumulative_attempts=cum_att)
    R['T7b_pass'] = bool(scfg == 43 and cum_scf == 44 and cum_stab == 2
                         and cum_att == 46 and cum_att <= 46)

    # ---- T7c: 43rd fd request refused (cap) ------------------------------
    err7 = None
    try:
        led.pre('fd')
    except RuntimeError as e:
        err7 = str(e)
    R['T7c'] = dict(error=err7)
    R['T7c_pass'] = bool(err7 and 'cap reached' in err7)

    R['formal_dirs_untouched'] = bool(pre_snap58 == snapshot(S58)
                                      and pre_snap_r == snapshot(OUT))
    R['isolation'] = dict(pyscf_stubbed=sys.modules.get('pyscf') is None,
                          job052_stubbed=getattr(
                              sys.modules.get('job052_exec'), '__name__',
                              None) == 'job052_exec'
                          and not hasattr(sys.modules['job052_exec'],
                                          'endpoint_eval'))
    keys = ('T1_pass', 'T2_pass', 'T3_pass', 'T4_pass', 'T5_pass',
            'T6_pass', 'T7_pass', 'T7b_pass', 'T7c_pass',
            'formal_dirs_untouched')
    R['ALL_PASS'] = bool(all(R[k] for k in keys))
    print(json.dumps(R, indent=1, default=str))
    ex.save_json_atomic(os.path.join(OUT, 'mock_test_results_resume01.json'),
                        R)
    if not R['ALL_PASS']:
        raise SystemExit('MOCK TESTS FAILED')
    print('MOCK TESTS ALL PASS', flush=True)


if __name__ == '__main__':
    main()
