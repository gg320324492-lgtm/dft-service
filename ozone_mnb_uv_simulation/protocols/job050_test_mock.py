#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-050 step B: mock-backend tests (temp dirs, no pyscf).

Per charter section 4, the tests cover exactly the actual changes:
  T1  the real call receives method='BFGS' and the prescribed options
      (gtol=1e-6, norm=inf, xrtol=0, c1=1e-4, c2=0.9, hess_inv0=I21,
      maxiter=200) -- verified via a recording spy around minimize;
  T2  21-dim coordinates, (7,3) gradient shape, and energy-gradient
      pairing from the SAME geometry (recomputed from the mock potential);
  T3  start reuse: the optimizer first request at the start coordinates
      reuses the reproduction record (no repeated SCF);
  T4  all line-search evaluations are ledger-counted; budget exhaustion
      or exception = real hard stop (no recheck, no borrowing);
  T5  gradient threshold -> controlled stop -> exactly one independent
      recheck -> registered only on PASS (failure case covered);
  T6  tests cannot call the real backend nor touch the formal directory.
"""
import os, sys, json, tempfile, shutil
import numpy as np

for _m in ('pyscf', 'd2_full', 'grad_factory'):
    sys.modules[_m] = None          # real backend unreachable in tests

HERE = '/mnt/e/dft-service/ozone_mnb_uv_simulation/protocols'
sys.path.insert(0, HERE)
import job047_exec as ex
import job050_exec as ex50

BPA = 1.0 / ex.ANG_PER_BOHR

X0 = (np.arange(21, dtype=float).reshape(7, 3) * 0.05 + 1.0)
U = np.zeros((7, 3))
U[4, 0] = 1.0                                # unit direction


class MockBackend:
    """E(s) = E0 + A s + B s^2 + C s^4 along the line; g = E'(s) * U."""

    def __init__(self, E0, A, B, C):
        self.E0, self.A, self.B, self.C = E0, A, B, C
        self.scf_calls = 0
        self.cfg = dict(xc='mock', d2_attached=True, grid_level=8,
                        scf_tol=[1e-12, 1e-9], basis='mock', charge=0,
                        spin=0, grid_response=True,
                        solvent='none (gas phase)')

    def full_eval(self, R_bohr, out_dir, tag):
        R = np.asarray(R_bohr, float)
        s = float(np.dot((R - X0).reshape(-1), U.reshape(-1)))
        e = self.E0 + self.A * s + self.B * s * s + self.C * s ** 4
        de = self.A + 2 * self.B * s + 4 * self.C * s ** 3
        g = (de * U.reshape(-1)).reshape(7, 3)
        self.scf_calls += 1
        return dict(e_total=float(e), e_d2_Eh=0.0, e_dft_part_Eh=float(e),
                    grad=g.tolist(), grad_max=float(np.abs(g).max()),
                    grad_sha='mock',
                    coords_actual_angstrom=(R * ex.ANG_PER_BOHR).tolist(),
                    coords_sha='mock', config=dict(self.cfg),
                    chkfile=dict(path='mock.chk', exists=True, size=0),
                    mo_summary=dict(n_mo=100, nocc=17),
                    converged=True, all_finite=True, scf_kernel_count=1)


def make_source_record(backend):
    R = X0 + 0.0
    rec = backend.full_eval(R, tempfile.mkdtemp(), 'src')
    rec['tag'] = 'opt_19'
    rec['category'] = 'opt'
    return rec


def snapshot(d):
    if not os.path.isdir(d):
        return []
    return sorted((f, os.path.getsize(os.path.join(d, f)))
                  for f in os.listdir(d))


def recheck(meeting_rec, tmp, ledger, backend):
    return ex.eval_point('recheck', np.asarray(
        meeting_rec['coords_actual_angstrom'], float) * BPA,
        tmp, ledger, backend, 'recheck',
        gate_fn=ex.recheck_gate_fn(meeting_rec))


def verify_pairing(tmp, be):
    """Energy-gradient pairing: each saved opt record must reproduce the
    mock potential (E and g) computed from ITS OWN coordinates."""
    import glob as _g
    for f in _g.glob(tmp + '/eval_opt_opt_*.json'):
        r = json.load(open(f))
        x = np.asarray(r['x_bohr'], float).reshape(-1)
        s = float(np.dot(x - X0.reshape(-1), U.reshape(-1)))
        e = be.E0 + be.A * s + be.B * s * s + be.C * s ** 4
        de = be.A + 2 * be.B * s + 4 * be.C * s ** 3
        g = (de * U.reshape(-1)).reshape(7, 3)
        if abs(e - float(r['e_total'])) > 1e-12:
            return False
        if np.abs(g - np.asarray(r['grad'], float)).max() > 1e-12:
            return False
        if np.asarray(r['grad'], float).shape != (7, 3):
            return False
        if np.asarray(r['x_bohr'], float).reshape(-1).size != 21:
            return False
    return True


def wrap(be):
    return ex50.LoggingBackend(be)


def main():
    pre_snap = snapshot(ex50.OUT)
    R = {}
    spy = dict(calls=[])

    real_minimize = ex50._scipy_minimize

    def spy_minimize(fun, x0, **kw):
        spy['calls'].append(dict(method=kw.get('method'),
                                 options=dict(kw.get('options', {})),
                                 jac=kw.get('jac')))
        return real_minimize(fun, x0, **kw)

    ex50._scipy_minimize = spy_minimize

    # ---- T1/T2/T3/T5a: full happy path with spy, reuse, threshold, recheck
    tmp = tempfile.mkdtemp(prefix='job050_T1_')
    be = MockBackend(-282.0, 2e-4, -0.004, 0.25)
    src = make_source_record(be)
    scf_after_src = be.scf_calls
    led = ex.Ledger(os.path.join(tmp, 'budget.json'),
                    caps=dict(start_repro=1, opt=20, recheck=1))
    rec_s = ex.eval_point('start', np.asarray(
        src['coords_actual_angstrom'], float) * BPA, tmp, led, be,
        'start_repro', gate_fn=ex.centre_gate_fn(src))
    opt = ex50.run_bfgs(rec_s, tmp, led, wrap(be))
    n_opt = opt['n_new_evals']
    cand = None
    if opt['meeting_points']:
        mp = opt['meeting_points'][0]
        mrec = json.load(open('%s/eval_%s_%s.json'
                              % (tmp, mp['category'], mp['tag'])))
        rr = recheck(mrec, tmp, led, be)
        cand = bool(rr['gate']['gates_pass'])
    c1 = spy['calls'][0] if spy['calls'] else {}
    exp_opts = dict(gtol=1e-6, norm=float('inf'), xrtol=0.0, c1=1e-4,
                    c2=0.9, maxiter=200,
                    hess_inv0=[[1.0 if i == j else 0.0
                                for j in range(21)] for i in range(21)])
    opts_ok = bool(c1.get('method') == 'BFGS' and c1.get('jac') is True
                   and all(np.array_equal(np.asarray(c1['options'].get(k)),
                                          np.asarray(v))
                           for k, v in exp_opts.items()))
    R['T1'] = dict(spy_method=c1.get('method'), opts_ok=opts_ok,
                   outcome=opt['outcome'], n_new=n_opt,
                   reuse=opt['reuse_start'], recheck_pass=cand,
                   attempts={c: led.count(c) for c in
                             ('start_repro', 'opt', 'recheck')})
    R['T1_pass'] = bool(opts_ok
                        and opt['outcome'] == 'threshold_reached'
                        and opt['reuse_start'] is True
                        and cand is True
                        and be.scf_calls == scf_after_src + 1 + n_opt + 1
                        and R['T1']['attempts'] == dict(start_repro=1,
                                                        opt=n_opt,
                                                        recheck=1))
    R['T2'] = dict(pairing_ok=verify_pairing(tmp, be),
                   trial_count=len(opt['trial_sequence']),
                   accepted_count=len(opt['accepted_iterates']))
    R['T2_pass'] = bool(R['T2']['pairing_ok']
                        and R['T2']['trial_count'] >= n_opt)

    # ---- T4: budget exhaustion = real hard stop, all trials counted
    tmp = tempfile.mkdtemp(prefix='job050_T4_')
    be = MockBackend(-282.0, 2e-4, -0.004, 0.25)
    src = make_source_record(be)
    led = ex.Ledger(os.path.join(tmp, 'budget.json'),
                    caps=dict(start_repro=1, opt=3, recheck=1))
    rec_s = ex.eval_point('start', np.asarray(
        src['coords_actual_angstrom'], float) * BPA, tmp, led, be,
        'start_repro', gate_fn=ex.centre_gate_fn(src))
    opt = ex50.run_bfgs(rec_s, tmp, led, wrap(be), opt_cap=3)
    no_borrow = False
    try:
        led.pre_eval('opt', dict(note='cap probe'))
    except RuntimeError:
        no_borrow = True
    R['T4'] = dict(outcome=opt['outcome'],
                   n_new=opt['n_new_evals'],
                   trials=len(opt['trial_sequence']),
                   accepted=len(opt['accepted_iterates']),
                   attempts={c: led.count(c) for c in
                             ('start_repro', 'opt', 'recheck')},
                   no_borrow=no_borrow)
    R['T4_pass'] = bool(opt['outcome'] == 'budget_exhausted'
                        and opt['n_new_evals'] == 3
                        and R['T4']['attempts'] == dict(start_repro=1,
                                                        opt=3, recheck=0)
                        and no_borrow)

    # ---- T5b: recheck failure -> not registered
    tmp = tempfile.mkdtemp(prefix='job050_T5_')
    be = MockBackend(-282.0, 2e-4, -0.004, 0.25)
    src = make_source_record(be)
    led = ex.Ledger(os.path.join(tmp, 'budget.json'),
                    caps=dict(start_repro=1, opt=20, recheck=1))
    rec_s = ex.eval_point('start', np.asarray(
        src['coords_actual_angstrom'], float) * BPA, tmp, led, be,
        'start_repro', gate_fn=ex.centre_gate_fn(src))
    opt = ex50.run_bfgs(rec_s, tmp, led, wrap(be))
    mp = opt['meeting_points'][0]
    mrec = json.load(open('%s/eval_%s_%s.json'
                          % (tmp, mp['category'], mp['tag'])))
    mrec['e_total'] += 1e-3
    err5 = None
    try:
        recheck(mrec, tmp, led, be)
    except RuntimeError as e:
        err5 = str(e)
    R['T5'] = dict(hard_stop_raised=bool(err5 and 'gate failed' in err5),
                   attempts={c: led.count(c) for c in
                             ('start_repro', 'opt', 'recheck')},
                   last_status=led.data['attempts'][-1]['status'])
    R['T5_pass'] = bool(R['T5']['hard_stop_raised']
                        and R['T5']['attempts']['recheck'] == 1
                        and R['T5']['last_status'] == 'error')

    ex50._scipy_minimize = real_minimize
    R['formal_dir_untouched'] = bool(pre_snap == snapshot(ex50.OUT))
    R['ALL_PASS'] = bool(all(R[k] for k in ('T1_pass', 'T2_pass', 'T4_pass',
                                            'T5_pass',
                                            'formal_dir_untouched')))
    print(json.dumps(R, indent=1, default=str))
    ex.save_json_atomic(os.path.join(ex50.OUT, 'mock_test_results.json'), R)
    if not R['ALL_PASS']:
        raise SystemExit('MOCK TESTS FAILED')
    print('MOCK TESTS ALL PASS', flush=True)


if __name__ == '__main__':
    main()
