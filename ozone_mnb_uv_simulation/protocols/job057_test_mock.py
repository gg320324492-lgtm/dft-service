#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-057 step B: zero-evaluation tests (temp dirs, real
backend unreachable, formal directory untouched).

Tests go through the ACTUAL entry (job051_exec.run_bfgs, the validated
component with the 1e-5 gate) with an injected dict-contract backend:
  T1  start above 1e-5 is NOT rejected; the first same-coords request
      reuses the accepted start record (no second SCF);
  T2  BFGS receives all specified parameters (recorded via a wrapper);
  T3  a point reaching max|g|<=1e-5 triggers the controlled stop and
      populates meeting_points (which is what invokes the recheck);
  T4  budget hard stop (cap 1): BudgetExhausted, blocked request recorded;
      a wrong-contract backend fails loudly and the ledger hard-stops;
  T5  isolation sentinels (no pyscf, formal directory untouched).
"""
import os, sys, json, tempfile
import numpy as np

sys.modules['pyscf'] = None
sys.modules['pyscf.hessian'] = None

HERE = '/mnt/e/dft-service/ozone_mnb_uv_simulation/protocols'
sys.path.insert(0, HERE)
import job047_exec as ex
import job051_exec as j51

OUT = os.path.join(os.path.dirname(j51.OUT), 'c1_fdopt057')
E1 = np.zeros(21)
E1[0] = 1.0


def snapshot(d):
    if not os.path.isdir(d):
        return []
    return sorted((f, os.path.getsize(os.path.join(d, f)))
                  for f in os.listdir(d))


class QuadBackend:
    """Mock honouring the RealBackend DICT contract; consistent quadratic
    E=0.5|x-xs|^2 (min gradient 0)."""

    def __init__(self, bad_contract=False):
        self.calls = []
        self.bad_contract = bad_contract

    def full_eval(self, R_bohr, out_dir, tag):
        Rp = np.asarray(R_bohr, float).reshape(-1)
        self.calls.append(Rp.tolist())
        g = Rp
        e = float(0.5 * Rp @ Rp)
        if self.bad_contract:
            return (e, g)
        return dict(e_total=e, e_d2_Eh=0.0, e_dft_part_Eh=e,
                    grad=g.reshape(7, 3).tolist(),
                    grad_max=float(np.abs(g).max()), grad_sha='mock',
                    coords_actual_angstrom=(Rp * ex.ANG_PER_BOHR).tolist(),
                    coords_sha='mock',
                    config=dict(xc='mock', grid_response=True, grid_level=8,
                                d2_attached=True, basis='mock', charge=0,
                                spin=0, scf_tol=[1e-12, 1e-9],
                                solvent='none (gas phase)'),
                    converged=True, all_finite=True, scf_kernel_count=1)


class MockChain:
    def __init__(self, inner):
        self.inner = inner

    def full_eval(self, R, out_dir, tag):
        rec = self.inner.full_eval(R, out_dir, tag)
        rec['x_bohr'] = np.asarray(R, float).tolist()
        rec['x_bohr_sha'] = 'mock'
        rec['seconds'] = 0.0
        return rec


def make_start(tmp, x0):
    be = QuadBackend()
    rec = be.full_eval(x0, tmp, 'start')
    rec['x_bohr'] = np.asarray(x0, float).tolist()
    return rec, be


def main():
    pre_snap = snapshot(OUT)
    R = {}
    X0 = 2e-4 * E1          # start gmax = 2e-4 > 1e-5 (NOT rejected)

    # T2: BFGS parameters actually passed
    orig_min = j51._scipy_minimize
    captured = {}

    def rec_min(fun, x0, method=None, jac=None, callback=None, options=None):
        captured['method'] = method
        captured['options'] = dict(options or {})
        return orig_min(fun, x0, method=method, jac=jac,
                        callback=callback, options=options)

    j51._scipy_minimize = rec_min
    tmp = tempfile.mkdtemp(prefix='job057_T2_')
    rec, be = make_start(tmp, X0)
    led = ex.Ledger(os.path.join(tmp, 'b.json'),
                    caps=dict(start_repro=1, opt=25, recheck=1))
    opt = j51.run_bfgs(rec, tmp, led, MockChain(be), opt_cap=25)
    j51._scipy_minimize = orig_min
    o = captured['options']
    R['T2'] = dict(method=captured['method'], gtol=o.get('gtol'),
                   norm=str(o.get('norm')), xrtol=o.get('xrtol'),
                   c1=o.get('c1'), c2=o.get('c2'), maxiter=o.get('maxiter'),
                   identity=bool(np.allclose(
                       np.asarray(o.get('hess_inv0'), float), np.eye(21))))
    R['T2_pass'] = bool(captured['method'] == 'BFGS'
                        and o.get('gtol') == 1e-6
                        and o.get('norm') == np.inf
                        and o.get('xrtol') == 0.0
                        and o.get('c1') == 1e-4 and o.get('c2') == 0.9
                        and o.get('maxiter') == 200 and R['T2']['identity'])

    # T1+T3: start reuse; threshold at 1e-5 -> meeting point
    tmp = tempfile.mkdtemp(prefix='job057_T1_')
    rec, be = make_start(tmp, X0)
    led = ex.Ledger(os.path.join(tmp, 'b1.json'),
                    caps=dict(start_repro=1, opt=25, recheck=1))
    opt = j51.run_bfgs(rec, tmp, led, MockChain(be), opt_cap=25)
    R['T1'] = dict(start_gmax=float(rec['grad_max']),
                   outcome=opt['outcome'],
                   reuse=opt['reuse_start'],
                   backend_calls=len(be.calls),
                   n_new=opt['n_new_evals'],
                   meeting=opt['meeting_points'])
    R['T1_pass'] = bool(rec['grad_max'] > 1e-5
                        and opt['outcome'] == 'threshold_reached'
                        and opt['reuse_start']
                        and len(be.calls) == opt['n_new_evals'] + 1
                        and len(opt['meeting_points']) == 1
                        and opt['meeting_points'][0]['grad_max'] <= 1e-5)

    # T4a: budget hard stop (cap 1) - constant-gradient linear model
    # (E = g0.x, grad = g0 always 2e-4 > 1e-5: no threshold possible)
    tmp = tempfile.mkdtemp(prefix='job057_T4_')

    class LinearBackend(QuadBackend):
        def full_eval(self, R_bohr, out_dir, tag):
            Rp = np.asarray(R_bohr, float).reshape(-1)
            self.calls.append(Rp.tolist())
            g = 2e-4 * E1
            e = float(g @ Rp)
            if self.bad_contract:
                return (e, g)
            return dict(e_total=e, e_d2_Eh=0.0, e_dft_part_Eh=e,
                        grad=g.reshape(7, 3).tolist(),
                        grad_max=float(np.abs(g).max()), grad_sha='mock',
                        coords_actual_angstrom=(Rp
                                                 * ex.ANG_PER_BOHR)
                        .tolist(), coords_sha='mock',
                        config=dict(xc='mock', grid_response=True,
                                    grid_level=8, d2_attached=True,
                                    basis='mock', charge=0, spin=0,
                                    scf_tol=[1e-12, 1e-9],
                                    solvent='none (gas phase)'),
                        converged=True, all_finite=True,
                        scf_kernel_count=1)
    rec, be = make_start(tmp, X0)
    led = ex.Ledger(os.path.join(tmp, 'b4.json'),
                    caps=dict(start_repro=1, opt=1, recheck=1))
    lin = LinearBackend()
    opt = j51.run_bfgs(rec, tmp, led, MockChain(lin), opt_cap=1)
    R['T4'] = dict(outcome=opt['outcome'], n_new=opt['n_new_evals'],
                   blocked=len(opt['blocked_requests']),
                   calls=len(lin.calls))
    R['T4_pass'] = bool(opt['outcome'] == 'budget_exhausted'
                        and opt['n_new_evals'] == 1
                        and len(opt['blocked_requests']) >= 1)

    # T4b: wrong contract -> loud failure + ledger hard stop
    tmp = tempfile.mkdtemp(prefix='job057_T4b_')
    rec, be = make_start(tmp, X0)
    led = ex.Ledger(os.path.join(tmp, 'b4b.json'),
                    caps=dict(start_repro=1, opt=3, recheck=1))
    bad = QuadBackend(bad_contract=True)
    err = None
    try:
        j51.run_bfgs(rec, tmp, led, MockChain(bad), opt_cap=3)
    except Exception as e:                                  # noqa: BLE001
        err = '%s: %s' % (type(e).__name__, str(e)[:60])
    hard = None
    try:
        ex.Ledger(os.path.join(tmp, 'b4b.json'),
                  caps=dict(start_repro=1, opt=3, recheck=1))
    except RuntimeError as e:
        hard = str(e)[:50]
    R['T4b'] = dict(error=err, hard=hard)
    R['T4b_pass'] = bool(err and hard and 'error attempt' in hard)

    R['formal_dir_untouched'] = bool(pre_snap == snapshot(OUT))
    R['ALL_PASS'] = bool(all(R[k] for k in ('T1_pass', 'T2_pass', 'T4_pass',
                                            'T4b_pass',
                                            'formal_dir_untouched')))
    print(json.dumps(R, indent=1, default=str))
    ex.save_json_atomic(os.path.join(OUT, 'mock_test_results.json'), R)
    if not R['ALL_PASS']:
        raise SystemExit('MOCK TESTS FAILED')
    print('MOCK TESTS ALL PASS', flush=True)


if __name__ == '__main__':
    main()
