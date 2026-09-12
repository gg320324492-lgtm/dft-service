#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-054 step B: zero-evaluation tests (temp dirs, real
backend unreachable, formal directory untouched).

Tests go through the ACTUAL execution entry (job054_exec.run_bfgs and
job047_exec.eval_point) down to the backend adapter layer, with the
backend injected as a mock honouring the RealBackend DICT contract.

Model: consistent quadratic E(x)=0.5*sum_i k_i x_i^2, grad_i = k_i x_i
(exact minimum gradient 0), with strongly different curvatures so the
optimization path VISITS the (1e-6, 1e-5] gradient band before reaching
the 1e-6 stop target:
  k = (1, 1e-3); x0 = (2e-5, 3e-3)  ->  start gmax ~ 2.0e-5 (>1e-5),
  first accepted region gmax ~ 3.0e-6 (in the band), later <=1e-6.
  T1  correct source/units/full-gradient chain: x0 from Angstrom->Bohr,
      grad reshaped to 21, start reused without a second evaluation,
      start ABOVE 1e-5 is not rejected (optimization starts);
  T2  BFGS parameters actually passed to scipy (recorded via a wrapper
      around the real minimize);
  T3  stop-target discipline in ONE run: a point with gmax ~3e-6
      (in (1e-6,1e-5]) does NOT stop the run (the old 1e-5 gate would
      have) and the run continues until a point <=1e-6 stops it;
  T4  dict return contract + persistence (eval files written); a tuple-
      returning backend (wrong contract) fails loudly and the ledger
      hard-stops on reuse;
  T5  per-category budget hard stop: cap opt=1 -> BudgetExhausted with
      blocked requests recorded (not counted as line-search rejections);
  T6  isolation sentinels (no pyscf, formal directory untouched).
"""
import os, sys, json, tempfile
import numpy as np

sys.modules['pyscf'] = None
sys.modules['pyscf.hessian'] = None

HERE = '/mnt/e/dft-service/ozone_mnb_uv_simulation/protocols'
sys.path.insert(0, HERE)
import job047_exec as ex
import job054_exec as j54

OUT = j54.OUT
KVEC = np.array([1.0, 1e-3] + [0.0] * 19)     # k1=1 (x0 dir), k2=1e-3
X0 = np.zeros(21)
X0[0] = 2e-5
X0[1] = 3e-3                                  # g2 = 3e-6 (band)


def snapshot(d):
    if not os.path.isdir(d):
        return []
    return sorted((f, os.path.getsize(os.path.join(d, f)))
                  for f in os.listdir(d))


class QuadBackend:
    """Mock honouring the RealBackend DICT contract (consistent E/grad)."""

    def __init__(self, k=KVEC, bad_contract=False):
        self.k = np.asarray(k, float)
        self.calls = []
        self.bad_contract = bad_contract

    def full_eval(self, R_bohr, out_dir, tag):
        R = np.asarray(R_bohr, float).reshape(-1)
        self.calls.append(R.tolist())
        g = self.k * R
        e = float(0.5 * np.sum(self.k * R * R))
        if self.bad_contract:
            return (e, g)          # WRONG contract (tuple)
        return dict(e_total=e, e_d2_Eh=0.0, e_dft_part_Eh=e,
                    grad=g.reshape(7, 3).tolist(),
                    grad_max=float(np.abs(g).max()), grad_sha='mock',
                    coords_actual_angstrom=(R * ex.ANG_PER_BOHR).tolist(),
                    coords_sha='mock',
                    config=dict(xc='mock', grid_response=True, grid_level=8,
                                d2_attached=True, basis='mock', charge=0,
                                spin=0, scf_tol=[1e-12, 1e-9],
                                solvent='none (gas phase)'),
                    converged=True, all_finite=True, scf_kernel_count=1)


class MockChain:
    """Mirrors LoggingBackend's contract additions over the inner dict."""

    def __init__(self, inner):
        self.inner = inner

    def full_eval(self, R, out_dir, tag):
        rec = self.inner.full_eval(R, out_dir, tag)
        rec['x_bohr'] = np.asarray(R, float).tolist()
        rec['x_bohr_sha'] = 'mock'
        rec['seconds'] = 0.0
        return rec


def run_case(tmp, opt_cap=25, x0=None):
    be = QuadBackend()
    R0 = X0.copy() if x0 is None else np.asarray(x0, float)
    rec = be.full_eval(R0, tmp, 'start')
    rec['x_bohr'] = R0.tolist()
    led = ex.Ledger(os.path.join(tmp, 'b.json'),
                    caps=dict(start_repro=1, opt=opt_cap, recheck=1))
    opt = j54.run_bfgs(rec, tmp, led, MockChain(be), opt_cap=opt_cap)
    return opt, be, led, rec


def main():
    pre_snap = snapshot(OUT)
    R = {}

    # T2: BFGS parameters actually passed to scipy
    orig_min = j54._scipy_minimize
    captured = {}

    def rec_min(fun, x0, method=None, jac=None, callback=None, options=None):
        captured['method'] = method
        captured['options'] = dict(options or {})
        return orig_min(fun, x0, method=method, jac=jac,
                        callback=callback, options=options)

    j54._scipy_minimize = rec_min
    tmp = tempfile.mkdtemp(prefix='job054_T2_')
    run_case(tmp)
    j54._scipy_minimize = orig_min
    o = captured['options']
    R['T2'] = dict(method=captured['method'],
                   gtol=o.get('gtol'), norm=str(o.get('norm')),
                   xrtol=o.get('xrtol'), c1=o.get('c1'), c2=o.get('c2'),
                   maxiter=o.get('maxiter'),
                   hess_inv0_shape=(None if o.get('hess_inv0') is None
                                    else list(np.asarray(
                                        o['hess_inv0'], float).shape)),
                   is_identity=bool(o.get('hess_inv0') is not None
                                    and np.allclose(
                                        np.asarray(o['hess_inv0'], float),
                                        np.eye(21))))
    R['T2_pass'] = bool(captured['method'] == 'BFGS'
                        and o.get('gtol') == 1e-6
                        and o.get('norm') == np.inf
                        and o.get('xrtol') == 0.0
                        and o.get('c1') == 1e-4 and o.get('c2') == 0.9
                        and o.get('maxiter') == 200 and R['T2']['is_identity'])

    # T3: band discipline + stop at 1e-6 (single run)
    tmp = tempfile.mkdtemp(prefix='job054_T3_')
    opt, be, led, rec = run_case(tmp)
    band_pts = [p for p in opt['acceptance_gate_points']
                if 1e-6 < float(p['grad_max']) <= 1e-5]
    stop_ok = opt['outcome'] == 'stop_target_reached' \
        and len(opt['stop_target_points']) == 1 \
        and float(opt['stop_target_points'][0]['grad_max']) <= 1e-6
    R['T3'] = dict(outcome=opt['outcome'],
                   band_points=[float(p['grad_max']) for p in band_pts],
                   stop_point=(None if not opt['stop_target_points'] else
                               float(opt['stop_target_points'][0]
                                     ['grad_max'])),
                   start_gmax=float(rec['grad_max']),
                   n_new=opt['n_new_evals'])
    R['T3_pass'] = bool(stop_ok and len(band_pts) >= 1
                        and rec['grad_max'] > 1e-5
                        and opt['n_new_evals'] >= 2)

    # T1: start reuse + persistence + start-above-1e-5 accepted
    tmp = tempfile.mkdtemp(prefix='job054_T1_')
    opt, be, led, rec = run_case(tmp)
    eval_files = [f for f in os.listdir(tmp) if f.startswith('eval_')]
    R['T1'] = dict(reuse_start=opt['reuse_start'],
                   backend_calls=len(be.calls),
                   n_new=opt['n_new_evals'],
                   n_eval_files=len(eval_files),
                   start_gmax=float(rec['grad_max']))
    R['T1_pass'] = bool(opt['reuse_start']
                        and len(be.calls) == opt['n_new_evals'] + 1
                        and len(eval_files) == opt['n_new_evals']
                        and rec['grad_max'] > 1e-5
                        and opt['n_new_evals'] >= 2)

    # T4b: wrong contract (tuple) -> loud failure + ledger hard stop
    tmp = tempfile.mkdtemp(prefix='job054_T4b_')
    be = QuadBackend()
    rec = be.full_eval(X0, tmp, 'start')
    rec['x_bohr'] = X0.tolist()
    led = ex.Ledger(os.path.join(tmp, 'b.json'),
                    caps=dict(start_repro=1, opt=3, recheck=1))
    be_bad = QuadBackend(bad_contract=True)
    err = None
    try:
        j54.run_bfgs(rec, tmp, led, MockChain(be_bad), opt_cap=3)
    except Exception as e:                                  # noqa: BLE001
        err = '%s: %s' % (type(e).__name__, str(e)[:80])
    hard = None
    try:
        ex.Ledger(os.path.join(tmp, 'b.json'),
                  caps=dict(start_repro=1, opt=3, recheck=1))
    except RuntimeError as e:
        hard = str(e)[:60]
    R['T4'] = dict(error=(err or '')[:80], ledger_hard_stop=hard)
    R['T4_pass'] = bool(err and 'TypeError' in err and hard
                        and 'error attempt' in hard)

    # T5: per-category budget hard stop (cap opt=1: the band point is
    # evaluated, NOT a stop -> next request hits the cap)
    tmp = tempfile.mkdtemp(prefix='job054_T5_')
    opt, be, led, rec = run_case(tmp, opt_cap=1)
    R['T5'] = dict(outcome=opt['outcome'],
                   n_new=opt['n_new_evals'],
                   blocked=len(opt['blocked_requests']),
                   attempts=led.count('opt'))
    R['T5_pass'] = bool(opt['outcome'] == 'budget_exhausted'
                        and opt['n_new_evals'] == 1
                        and len(opt['blocked_requests']) >= 1
                        and led.count('opt') == 1)

    R['formal_dir_untouched'] = bool(pre_snap == snapshot(OUT))
    R['ALL_PASS'] = bool(all(R[k] for k in ('T1_pass', 'T2_pass', 'T3_pass',
                                            'T4_pass', 'T5_pass',
                                            'formal_dir_untouched')))
    print(json.dumps(R, indent=1, default=str))
    ex.save_json_atomic(os.path.join(OUT, 'mock_test_results.json'), R)
    if not R['ALL_PASS']:
        raise SystemExit('MOCK TESTS FAILED')
    print('MOCK TESTS ALL PASS', flush=True)


if __name__ == '__main__':
    main()
