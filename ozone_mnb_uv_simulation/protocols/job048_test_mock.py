#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-048 step B: mock-backend tests (temp dirs, no pyscf).

Per charter section 5, ONLY the actual changes of this batch are tested
(the validated 047 evaluation chain is reused, not rewritten):
  T1  source switch: start = an externally supplied actually-evaluated
      record (045 +0.02), gate vs THAT record -> PASS -> optimization with
      start reuse (no repeated SCF) -> threshold stop -> recheck -> candidate
  T2  start reproduction gate failure -> HARD STOP, optimizer not started
      (opt 0/19), ledger attempt marked error
  T3  start already meets max|g|<=1e-5 -> NO optimization, direct
      conditional recheck (charter 4/6), opt 0/19
  T4  budget categories start_repro=1 / opt=19 / recheck=1 enforced
      independently; failures count in-category; no borrowing; opt cap
      reached -> controlled stop, no recheck, no auto-resume
  T5  recheck gate failure -> NOT registered (status recheck_failed)
  T6  formal artifact directory untouched by tests
"""
import os, sys, json, tempfile, shutil
import numpy as np

for _m in ('pyscf', 'd2_full', 'grad_factory'):
    sys.modules[_m] = None          # real backend unreachable in tests

HERE = '/mnt/e/dft-service/ozone_mnb_uv_simulation/protocols'
sys.path.insert(0, HERE)
import job047_exec as ex          # reuse the validated pipeline functions

FORMAL_OUT = ex.OUT
BPA = 1.0 / ex.ANG_PER_BOHR

# fixed straight line: x(s) = X0 + s * u  (unit vector over 21 components)
X0 = (np.arange(21, dtype=float).reshape(7, 3) * 0.05 + 1.0)
U = np.zeros(21)
U[4] = 1.0                                   # unit direction
U = U.reshape(7, 3)


class MockBackend:
    """E(s) = E0 + A s + B s^2 + C s^4 along the line; g = E'(s) * U."""

    def __init__(self, E0, A, B, C):
        self.E0, self.A, self.B, self.C = E0, A, B, C
        self.scf_calls = 0
        self.cfg = dict(xc='mock', d2_attached=True, grid_level=8,
                        scf_tol=[1e-12, 1e-9], basis='mock', charge=0,
                        spin=0, grid_response=True, solvent='none (gas phase)')

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


def make_source_record(backend, s_start):
    """An 'actually evaluated' external source record at X0 + s_start*U."""
    R = X0 + s_start * U
    rec = backend.full_eval(R, tempfile.mkdtemp(), 'src')
    rec['tag'] = 'displacement_+0.02'
    rec['category'] = 'eval'
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


def main():
    pre_snap = snapshot(FORMAL_OUT)
    R = {}

    # ---- T1: source switch + reuse + threshold stop + recheck pass
    tmp = tempfile.mkdtemp(prefix='job048_T1_')
    be = MockBackend(-282.0, 2e-4, -0.004, 0.25)
    src = make_source_record(be, 0.0)          # "045 +0.02" stand-in
    scf_after_src = be.scf_calls
    led = ex.Ledger(os.path.join(tmp, 'budget.json'),
                    caps=dict(start_repro=1, opt=19, recheck=1))
    rec_s = ex.eval_point('start', np.asarray(
        src['coords_actual_angstrom'], float) * BPA, tmp, led, be,
        'start_repro', gate_fn=ex.centre_gate_fn(src))
    opt = ex.run_optimization(rec_s, tmp, led, be, opt_cap=19, maxiter=19)
    n_opt = opt['n_new_evals']
    cand = None
    if opt['meeting_points']:
        mp = opt['meeting_points'][0]
        mrec = json.load(open('%s/eval_%s_%s.json'
                              % (tmp, mp['category'], mp['tag'])))
        rr = recheck(mrec, tmp, led, be)
        cand = bool(rr['gate']['gates_pass'])
    R['T1'] = dict(gate_pass=rec_s['gate']['gates_pass'],
                   outcome=opt['outcome'], n_new=n_opt,
                   reuse=opt['reuse_start'], status=cand,
                   attempts={c: led.count(c) for c in
                             ('start_repro', 'opt', 'recheck')})
    R['T1_pass'] = bool(rec_s['gate']['gates_pass']
                        and opt['outcome'] == 'threshold_reached'
                        and opt['reuse_start'] is True
                        and cand is True
                        # SCF budget: 1(src)+1(start repro)+n_opt+1(recheck)
                        and be.scf_calls == scf_after_src + 1 + n_opt + 1
                        and R['T1']['attempts'] == dict(start_repro=1,
                                                        opt=n_opt,
                                                        recheck=1))

    # ---- T2: start gate failure -> hard stop, optimizer never started
    tmp = tempfile.mkdtemp(prefix='job048_T2_')
    be = MockBackend(-282.0, 2e-4, -0.004, 0.25)
    src = make_source_record(be, 0.0)
    src['e_total'] = src['e_total'] + 1e-3      # mismatch beyond gate
    led = ex.Ledger(os.path.join(tmp, 'budget.json'),
                    caps=dict(start_repro=1, opt=19, recheck=1))
    err = None
    try:
        ex.eval_point('start', np.asarray(
            src['coords_actual_angstrom'], float) * BPA, tmp, led, be,
            'start_repro', gate_fn=ex.centre_gate_fn(src))
    except RuntimeError as e:
        err = str(e)
    R['T2'] = dict(error=(err or '')[:80],
                   attempts={c: led.count(c) for c in
                             ('start_repro', 'opt', 'recheck')},
                   status=led.data['attempts'][0]['status'])
    R['T2_pass'] = bool(err and 'gate failed' in err
                        and R['T2']['attempts'] == dict(start_repro=1,
                                                        opt=0, recheck=0)
                        and R['T2']['status'] == 'error')

    # ---- T3: start already meets gate -> direct recheck (opt 0)
    tmp = tempfile.mkdtemp(prefix='job048_T3_')
    be = MockBackend(-282.0, 0.0, -0.005, 0.25)   # E'(0)=0 -> gmax=0
    src = make_source_record(be, 0.0)
    led = ex.Ledger(os.path.join(tmp, 'budget.json'),
                    caps=dict(start_repro=1, opt=19, recheck=1))
    rec_s = ex.eval_point('start', np.asarray(
        src['coords_actual_angstrom'], float) * BPA, tmp, led, be,
        'start_repro', gate_fn=ex.centre_gate_fn(src))
    opt = ex.run_optimization(rec_s, tmp, led, be, opt_cap=19, maxiter=19)
    rr = None
    if opt['meeting_points']:
        mp = opt['meeting_points'][0]
        mrec = json.load(open('%s/eval_%s_%s.json'
                              % (tmp, mp['category'], mp['tag'])))
        rr = recheck(mrec, tmp, led, be)
    R['T3'] = dict(outcome=opt['outcome'], status=rr['gate']['gates_pass'],
                   attempts={c: led.count(c) for c in
                             ('start_repro', 'opt', 'recheck')})
    R['T3_pass'] = bool(opt['outcome'] == 'start_already_meets_gate'
                        and rr is not None and rr['gate']['gates_pass']
                        and R['T3']['attempts'] == dict(start_repro=1,
                                                        opt=0, recheck=1))

    # ---- T4: opt cap reached -> controlled stop, no recheck, no borrowing
    tmp = tempfile.mkdtemp(prefix='job048_T4_')
    be = MockBackend(-282.0, 2e-4, -1e-4, 0.25)   # very flat: gate far away
    src = make_source_record(be, 0.0)
    led = ex.Ledger(os.path.join(tmp, 'budget.json'),
                    caps=dict(start_repro=1, opt=2, recheck=1))
    rec_s = ex.eval_point('start', np.asarray(
        src['coords_actual_angstrom'], float) * BPA, tmp, led, be,
        'start_repro', gate_fn=ex.centre_gate_fn(src))
    opt = ex.run_optimization(rec_s, tmp, led, be, opt_cap=2, maxiter=19)
    no_borrow = False
    try:
        led.pre_eval('opt', dict(note='cap probe'))
    except RuntimeError:
        no_borrow = True
    R['T4'] = dict(outcome=opt['outcome'],
                   attempts={c: led.count(c) for c in
                             ('start_repro', 'opt', 'recheck')},
                   no_borrow=no_borrow)
    R['T4_pass'] = bool(opt['outcome'] == 'budget_exhausted'
                        and R['T4']['attempts'] == dict(start_repro=1,
                                                        opt=2, recheck=0)
                        and no_borrow)

    # ---- T5: recheck failure -> not registered
    tmp = tempfile.mkdtemp(prefix='job048_T5_')
    be = MockBackend(-282.0, 2e-4, -0.004, 0.25)
    src = make_source_record(be, 0.0)
    led = ex.Ledger(os.path.join(tmp, 'budget.json'),
                    caps=dict(start_repro=1, opt=19, recheck=1))
    rec_s = ex.eval_point('start', np.asarray(
        src['coords_actual_angstrom'], float) * BPA, tmp, led, be,
        'start_repro', gate_fn=ex.centre_gate_fn(src))
    opt = ex.run_optimization(rec_s, tmp, led, be, opt_cap=19, maxiter=19)
    mp = opt['meeting_points'][0]
    mrec = json.load(open('%s/eval_%s_%s.json'
                          % (tmp, mp['category'], mp['tag'])))
    mrec['e_total'] += 1e-3                     # tamper: record check fails
    err5 = None
    try:
        rr = recheck(mrec, tmp, led, be)
    except RuntimeError as e:
        err5 = str(e)
        rr = dict(gate=dict(gates_pass=False))
    R['T5'] = dict(gates_pass=rr['gate']['gates_pass'],
                   hard_stop_raised=bool(err5),
                   attempts={c: led.count(c) for c in
                             ('start_repro', 'opt', 'recheck')})
    R['T5_pass'] = bool(rr['gate']['gates_pass'] is False
                        and err5 and 'gate failed' in err5
                        and R['T5']['attempts']['recheck'] == 1
                        and led.data['attempts'][-1]['status'] == 'error')

    R['formal_dir_untouched'] = bool(pre_snap == snapshot(FORMAL_OUT))
    R['ALL_PASS'] = bool(all(R[k] for k in ('T1_pass', 'T2_pass', 'T3_pass',
                                            'T4_pass', 'T5_pass',
                                            'formal_dir_untouched')))
    print(json.dumps(R, indent=1, default=str))
    ex.save_json_atomic(os.path.join(FORMAL_OUT, 'mock_test_results.json'), R)
    if not R['ALL_PASS']:
        raise SystemExit('MOCK TESTS FAILED')
    print('MOCK TESTS ALL PASS', flush=True)


if __name__ == '__main__':
    main()
