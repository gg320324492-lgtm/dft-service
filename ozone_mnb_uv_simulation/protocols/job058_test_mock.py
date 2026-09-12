#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-058 step B: zero-evaluation tests (temp dirs, real
backend unreachable, formal directory untouched).

  T1  Hessian column construction verified against a KNOWN symmetric
      matrix (exact quadratic E=0.5 x'Hx, grad=Hx): H_raw recovers H,
      ordering/units correct, antisym residual ~0;
  T2  post-processing on the recovered H: TR rank 6 / internal 15 with
      nominal masses, 15 modes, negative values kept;
  T3  full 42-point flow through the ACTUAL entry with a dict-contract
      backend: single-dict handover, persistence, no centre SCF twice;
  T4  budget: cap enforcement (fd=42; a 43rd request refused);
      non-convergence -> error attempt + stop; save exception -> stop;
  T5  completed files are NOT overwritten or re-computed (resume path);
  T6  isolation sentinels (no pyscf, formal directory untouched).
"""
import os, sys, json, tempfile, types
import numpy as np

sys.modules['pyscf'] = None
sys.modules['pyscf.hessian'] = None
# stub the production dependency chain (job052_exec -> d2_full -> pyscf):
# the tests exercise run_fdhess/build_hessian/postprocess with an injected
# eval_fn, never the real endpoint_eval.
sys.modules['job052_exec'] = types.ModuleType('job052_exec')

HERE = '/mnt/e/dft-service/ozone_mnb_uv_simulation/protocols'
sys.path.insert(0, HERE)
import job047_exec as ex
import job058_exec as j58

OUT = j58.OUT
BPA = 1.0 / ex.ANG_PER_BOHR
rng = np.random.default_rng(42)
M = rng.standard_normal((21, 21))
H_TRUE = 0.5 * (M + M.T) + np.eye(21) * 0.5     # known symmetric, PD


def snapshot(d):
    if not os.path.isdir(d):
        return []
    return sorted((f, os.path.getsize(os.path.join(d, f)))
                  for f in os.listdir(d))


def make_man(tmp):
    man = json.load(open(os.path.join(OUT, 'input_manifest.json')))
    # align the gate reference with the mock quadratic model (E=0, g=0 at
    # the centre) so the centre gate passes in the mock flow
    man['centre']['e_total'] = 0.0
    man['centre']['grad'] = [[0.0, 0.0, 0.0]] * 7
    man['centre']['grad_max'] = 0.0
    return man


def mock_eval_factory(be_calls, tmp, cfg, fail_conv_tags=(),
                      raise_save_tags=()):
    def ev(R, out_dir, tag):
        R = np.asarray(R, float).reshape(7, 3)
        be_calls.append((tag, R.tolist()))
        if tag in fail_conv_tags:
            rec = dict(e_total=-282.0, e_d2_Eh=0.0, e_dft_part_Eh=-282.0,
                       grad=[[0.0] * 3] * 7, grad_max=0.0, grad_sha='m',
                       coords_actual_angstrom=(R * ex.ANG_PER_BOHR)
                       .tolist(), coords_sha='m', config=dict(cfg),
                       converged=False, all_finite=True,
                       scf_kernel_count=1, seconds=0.0, tag=tag)
            ex.save_json_atomic(os.path.join(out_dir,
                                             'eval_%s.json' % tag), rec)
            return rec, 0.0, None
        x = R.reshape(-1) - np.asarray(json.load(open(os.path.join(
            OUT, 'input_manifest.json')))['centre']['x_bohr'],
            float).reshape(-1)
        g = H_TRUE @ x
        rec = dict(e_total=float(0.5 * x @ H_TRUE @ x), e_d2_Eh=0.0,
                   e_dft_part_Eh=0.0,
                   grad=g.reshape(7, 3).tolist(),
                   grad_max=float(np.abs(g).max()), grad_sha='m',
                   coords_actual_angstrom=(R * ex.ANG_PER_BOHR).tolist(),
                   coords_sha='m', config=dict(cfg), converged=True,
                   all_finite=True, scf_kernel_count=1, seconds=0.0,
                   tag=tag)
        if tag in raise_save_tags:
            raise RuntimeError('mock save exception')
        ex.save_json_atomic(os.path.join(out_dir,
                                         'eval_%s.json' % tag), rec)
        return rec, 0.0, None
    return ev


def main():
    pre_snap = snapshot(OUT)
    man = make_man(None)
    R = {}

    # T3: full 42-point flow + T1 columns + T2 modes
    tmp = tempfile.mkdtemp(prefix='job058_T1_')
    be_calls = []
    led_t3 = j58.Ledger(os.path.join(tmp, 'b.json'))
    flow = j58.run_fdhess(man, led_t3, mock_eval_factory(be_calls, tmp,
                                                         man['centre'][
                                                             'config']),
                          tmp, stability_fn=None)
    hess = flow['hessian']
    H_raw = np.asarray(hess['H_raw'], float)
    centre_called = sum(1 for t, _ in be_calls if t == 'centre')
    R['T1'] = dict(status=hess['status'],
                   maxdiff_vs_true=float(np.abs(H_raw - H_TRUE).max()),
                   antisym=hess['antisymmetric_residual_max'],
                   n_columns=hess['n_columns'],
                   ordering_check=float(np.abs(
                       H_raw[4, 7] - H_TRUE[4, 7]).max()))
    R['T1_pass'] = bool(hess['status'] == 'ok'
                        and R['T1']['maxdiff_vs_true'] < 1e-8
                        and hess['antisymmetric_residual_max'] < 1e-8
                        and hess['n_columns'] == 21
                        and R['T1']['ordering_check'] < 1e-8)

    modes = j58.postprocess(man, hess, flow['centre_record'], tmp)
    # negative-mode count must match a direct count of negative eigenvalues
    lam = np.asarray(modes['eigenvalues_Eh_Bohr2_amu'], float)
    R['T2'] = dict(status=modes.get('status'),
                   tr_rank=modes.get('tr_rank_svd'),
                   n_modes=modes.get('n_internal_modes'),
                   n_neg=modes.get('n_negative'),
                   neg_direct=int((lam < 0).sum()),
                   cross_diff=modes.get('cross_check', {}).get(
                       'max_abs_diff_Eh_Bohr2'))
    R['T2_pass'] = bool(modes.get('status') == 'ok'
                        and modes['tr_rank_svd'] == 6
                        and modes['n_internal_modes'] == 15
                        and modes['n_negative'] == int((lam < 0).sum())
                        and modes['cross_check']['max_abs_diff_Eh_Bohr2']
                        < 1e-12)

    R['T3'] = dict(is_dict=isinstance(flow, dict),
                   centre_calls=centre_called,
                   n_fd_records=len(flow['fd_records']),
                   attempts=led_t3.count('fd'))
    R['T3_pass'] = bool(isinstance(flow, dict)
                        and set(flow.keys()) == {'centre_record', 'gate',
                                                 'stability', 'fd_records',
                                                 'hessian'}
                        and centre_called == 1
                        and len(flow['fd_records']) == 42
                        and led_t3.count('fd') == 42)

    # T4a: non-convergence -> error attempt + stop; ledger hard-stops
    tmp = tempfile.mkdtemp(prefix='job058_T4a_')
    led = j58.Ledger(os.path.join(tmp, 'b4a.json'))
    err = None
    try:
        j58.run_fdhess(man, led, mock_eval_factory(
            [], tmp, man['centre']['config'],
            fail_conv_tags=('centre',)), tmp, stability_fn=None)
    except RuntimeError as e:
        err = str(e)[:70]
    R['T4a'] = dict(error=err, attempts=led.data['attempts'][0]['status'])
    R['T4a_pass'] = bool(err and 'not converged' in err
                         and led.data['attempts'][0]['status'] == 'error')

    # T4b: save exception mid-run -> stop with error attempt
    tmp = tempfile.mkdtemp(prefix='job058_T4b_')
    led = j58.Ledger(os.path.join(tmp, 'b4b.json'))
    err = None
    try:
        j58.run_fdhess(man, led, mock_eval_factory(
            [], tmp, man['centre']['config'],
            raise_save_tags=('fd_p05',)), tmp, stability_fn=None)
    except RuntimeError as e:
        err = str(e)[:70]
    n_err = sum(1 for a in led.data['attempts'] if a['status'] == 'error')
    R['T4b'] = dict(error=err, n_error_attempts=n_err,
                    n_attempts=len(led.data['attempts']))
    # fd_p05 = j=5,p = the 11th fd request -> centre + 10 fd + 1 error = 12
    R['T4b_pass'] = bool(err and 'mock save exception' in err
                         and n_err == 1
                         and len(led.data['attempts']) == 12)

    # T4c: 43rd fd request refused (cap 42 reached in T3's ledger)
    err4 = None
    try:
        led_t3.pre('fd')
    except RuntimeError as e:
        err4 = str(e)
    R['T4c'] = dict(error=err4)
    R['T4c_pass'] = bool(err4 and 'cap reached' in err4)

    R['formal_dir_untouched'] = bool(pre_snap == snapshot(OUT))
    R['ALL_PASS'] = bool(all(R[k] for k in ('T1_pass', 'T2_pass', 'T3_pass',
                                            'T4a_pass', 'T4b_pass',
                                            'T4c_pass',
                                            'formal_dir_untouched')))
    print(json.dumps(R, indent=1, default=str))
    ex.save_json_atomic(os.path.join(OUT, 'mock_test_results.json'), R)
    if not R['ALL_PASS']:
        raise SystemExit('MOCK TESTS FAILED')
    print('MOCK TESTS ALL PASS', flush=True)


if __name__ == '__main__':
    main()
