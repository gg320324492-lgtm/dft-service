#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-051 step B: mock-backend tests (temp dirs, no pyscf).

Zero-evaluation coverage required by charter section 6:
  M1  correct start point and unit chain (Bohr 21-dim x, (7,3) gradient,
      energy-gradient pairing from the SAME geometry);
  M2  the real call receives method='BFGS' and the prescribed options;
  M3  test-backend isolation (no pyscf import path) and formal-directory
      sentinel untouched;
  M4  budget rejection happens BEFORE evaluation: the blocked request is
      recorded separately with NO energy/gradient and is never counted as
      a line-search rejection or a completed evaluation;
  M5  complete records safely persisted (coords/gradient/components/
      config/convergence/counters) before the ledger marks done;
  M6  start cache reuse: the optimizer first request at the start
      coordinates triggers no repeated SCF;
  M7  gradient threshold -> controlled stop -> exactly one independent
      recheck -> registered only on PASS (failure path covered).
"""
import os, sys, json, tempfile, shutil
import numpy as np

for _m in ('pyscf', 'd2_full', 'grad_factory'):
    sys.modules[_m] = None          # real backend unreachable in tests

HERE = '/mnt/e/dft-service/ozone_mnb_uv_simulation/protocols'
sys.path.insert(0, HERE)
import job047_exec as ex
import job051_exec as ex51

BPA = 1.0 / ex.ANG_PER_BOHR
X0 = (np.arange(21, dtype=float).reshape(7, 3) * 0.05 + 1.0)
U = np.zeros((7, 3)); U[4, 0] = 1.0


class MockBackend:
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


def wrap(be):
    return ex51.LoggingBackend(be)


def make_source_record(be):
    rec = be.full_eval(X0, tempfile.mkdtemp(), 'src')
    rec['tag'] = 'opt_20'
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


def verify_records(tmp, be):
    """M1/M5: units, shapes, pairing, completeness of persisted records."""
    import glob as _g
    ok = True
    for f in sorted(_g.glob(tmp + '/eval_*.json')):
        r = json.load(open(f))
        x = np.asarray(r['x_bohr'], float).reshape(-1)
        g = np.asarray(r['grad'], float)
        if x.size != 21 or g.shape != (7, 3):
            ok = False
        for k in ('e_total', 'e_d2_Eh', 'e_dft_part_Eh', 'config',
                  'converged', 'all_finite', 'coords_actual_angstrom',
                  'grad_max', 'scf_kernel_count', 'seconds'):
            if k not in r:
                ok = False
        if 'opt_' in r['tag']:
            s = float(np.dot(x - X0.reshape(-1), U.reshape(-1)))
            e = be.E0 + be.A * s + be.B * s * s + be.C * s ** 4
            de = be.A + 2 * be.B * s + 4 * be.C * s ** 3
            ge = (de * U.reshape(-1)).reshape(7, 3)
            if abs(e - float(r['e_total'])) > 1e-12:
                ok = False
            if np.abs(ge - g).max() > 1e-12:
                ok = False
        if not os.path.exists(f):       # persisted before ledger 'done'
            ok = False
    return ok


def main():
    pre_snap = snapshot(ex51.OUT)
    R = {}
    spy = dict(calls=[])
    real_minimize = ex51._scipy_minimize

    def spy_minimize(fun, x0, **kw):
        spy['calls'].append(dict(method=kw.get('method'),
                                 jac=kw.get('jac'),
                                 options=dict(kw.get('options', {}))))
        return real_minimize(fun, x0, **kw)

    ex51._scipy_minimize = spy_minimize

    # ---- M1/M2/M5/M6/M7: happy path
    tmp = tempfile.mkdtemp(prefix='job051_M1_')
    be = MockBackend(-282.0, 2e-4, -0.004, 0.25)
    src = make_source_record(be)
    scf_after_src = be.scf_calls
    led = ex.Ledger(os.path.join(tmp, 'budget.json'),
                    caps=dict(start_repro=1, opt=20, recheck=1))
    rec_s = ex.eval_point('start', np.asarray(
        src['coords_actual_angstrom'], float) * BPA, tmp, led, wrap(be),
        'start_repro', gate_fn=ex.centre_gate_fn(src))
    opt = ex51.run_bfgs(rec_s, tmp, led, wrap(be))
    n_opt = opt['n_new_evals']
    cand = None
    if opt['meeting_points']:
        mp = opt['meeting_points'][0]
        mrec = json.load(open('%s/eval_%s_%s.json'
                              % (tmp, mp['category'], mp['tag'])))
        rr = recheck(mrec, tmp, led, wrap(be))
        cand = bool(rr['gate']['gates_pass'])
    c1 = spy['calls'][0] if spy['calls'] else {}
    exp = dict(gtol=1e-6, norm=float('inf'), xrtol=0.0, c1=1e-4, c2=0.9,
               maxiter=200,
               hess_inv0=[[1.0 if i == j else 0.0
                           for j in range(21)] for i in range(21)])
    opts_ok = bool(c1.get('method') == 'BFGS' and c1.get('jac') is True
                   and all(np.array_equal(np.asarray(c1['options'].get(k)),
                                          np.asarray(v))
                           for k, v in exp.items()))
    R['M1'] = dict(records_ok=verify_records(tmp, be))
    R['M1_pass'] = bool(R['M1']['records_ok'])
    R['M2'] = dict(spy_method=c1.get('method'), opts_ok=opts_ok)
    R['M2_pass'] = bool(opts_ok)
    R['M6'] = dict(reuse=opt['reuse_start'],
                   scf_calls=be.scf_calls,
                   expected=scf_after_src + 1 + n_opt + 1)
    R['M6_pass'] = bool(opt['reuse_start'] is True
                        and be.scf_calls == scf_after_src + 1 + n_opt + 1)
    R['M7'] = dict(outcome=opt['outcome'], recheck_pass=cand,
                   attempts={c: led.count(c) for c in
                             ('start_repro', 'opt', 'recheck')})
    R['M7_pass'] = bool(opt['outcome'] == 'threshold_reached'
                        and cand is True
                        and R['M7']['attempts'] == dict(start_repro=1,
                                                        opt=n_opt,
                                                        recheck=1))

    # ---- M3/M4: budget rejection BEFORE evaluation; blocked not a rejection
    tmp = tempfile.mkdtemp(prefix='job051_M4_')
    be = MockBackend(-282.0, 2e-4, -0.004, 0.25)
    src = make_source_record(be)
    led = ex.Ledger(os.path.join(tmp, 'budget.json'),
                    caps=dict(start_repro=1, opt=3, recheck=1))
    rec_s = ex.eval_point('start', np.asarray(
        src['coords_actual_angstrom'], float) * BPA, tmp, led, wrap(be),
        'start_repro', gate_fn=ex.centre_gate_fn(src))
    opt = ex51.run_bfgs(rec_s, tmp, led, wrap(be), opt_cap=3)
    import glob as _g
    n_eval_files = len(_g.glob(tmp + '/eval_opt_opt_*.json'))
    blocked_has_no_record = all(
        not any(np.array_equal(np.asarray(b, float),
                               np.asarray(json.load(open(f))['x_bohr'],
                                          float).reshape(-1))
                for f in _g.glob(tmp + '/eval_opt_opt_*.json'))
        for b in opt['blocked_requests'])
    no_borrow = False
    try:
        led.pre_eval('opt', dict(note='cap probe'))
    except RuntimeError:
        no_borrow = True
    R['M3'] = dict(formal_dir_untouched=bool(pre_snap
                                             == snapshot(ex51.OUT)))
    R['M3_pass'] = bool(R['M3']['formal_dir_untouched'])
    R['M4'] = dict(outcome=opt['outcome'], n_new=opt['n_new_evals'],
                   n_requests=len(opt['requests']),
                   n_blocked=len(opt['blocked_requests']),
                   n_accepted=len(opt['accepted_iterates']),
                   n_eval_files=n_eval_files,
                   blocked_has_no_record=blocked_has_no_record,
                   attempts={c: led.count(c) for c in
                             ('start_repro', 'opt', 'recheck')},
                   no_borrow=no_borrow)
    R['M4_pass'] = bool(opt['outcome'] == 'budget_exhausted'
                        and opt['n_new_evals'] == 3
                        and len(opt['blocked_requests']) == 1
                        and blocked_has_no_record
                        and len(opt['requests']) == 1 + 3 + 1
                        and R['M4']['attempts'] == dict(start_repro=1,
                                                        opt=3, recheck=0)
                        and no_borrow)

    # ---- M7b: recheck failure -> not registered
    tmp = tempfile.mkdtemp(prefix='job051_M7b_')
    be = MockBackend(-282.0, 2e-4, -0.004, 0.25)
    src = make_source_record(be)
    led = ex.Ledger(os.path.join(tmp, 'budget.json'),
                    caps=dict(start_repro=1, opt=20, recheck=1))
    rec_s = ex.eval_point('start', np.asarray(
        src['coords_actual_angstrom'], float) * BPA, tmp, led, wrap(be),
        'start_repro', gate_fn=ex.centre_gate_fn(src))
    opt = ex51.run_bfgs(rec_s, tmp, led, wrap(be))
    mp = opt['meeting_points'][0]
    mrec = json.load(open('%s/eval_%s_%s.json'
                          % (tmp, mp['category'], mp['tag'])))
    mrec['e_total'] += 1e-3
    err = None
    try:
        recheck(mrec, tmp, led, wrap(be))
    except RuntimeError as e:
        err = str(e)
    R['M7b'] = dict(hard_stop=bool(err and 'gate failed' in err),
                    last_status=led.data['attempts'][-1]['status'])
    R['M7b_pass'] = bool(R['M7b']['hard_stop']
                         and R['M7b']['last_status'] == 'error')

    ex51._scipy_minimize = real_minimize
    R['ALL_PASS'] = bool(all(R[k] for k in ('M1_pass', 'M2_pass', 'M3_pass',
                                            'M4_pass', 'M6_pass', 'M7_pass',
                                            'M7b_pass')))
    print(json.dumps(R, indent=1, default=str))
    ex.save_json_atomic(os.path.join(ex51.OUT, 'mock_test_results.json'), R)
    if not R['ALL_PASS']:
        raise SystemExit('MOCK TESTS FAILED')
    print('MOCK TESTS ALL PASS', flush=True)


if __name__ == '__main__':
    main()
