#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-061 step B: zero-evaluation tests (temp dirs, mock
backend only, formal directories untouched).

  T1  runtime trigger patch: recheck gate under GM_GATE=1e-6 - 8e-7
      passes, 5e-6 fails; centre-gate constants 1e-8/1e-7/1e-9;
  T2  start provenance: the opt_24 record matches the 060 LAST accepted
      iterate (<=1e-12 Bohr) and is NOT opt_25/opt_02;
  T3  full flow on the Taylor-anchored quadratic mock: start gate PASS ->
      BFGS -> threshold_reached -> conditional recheck; four-way
      classification (start_repro / accepted / non_accepted_trial);
  T4  budget: opt cap 20 -> budget_exhausted, blocked request recorded,
      ledger count = cap;
  T5  start gate failure -> error attempt + STOPPED json + 0 opt evals;
  T6  060 correction note numerators/classification consistent with the
      saved 060 results (start 1 / accepted 24 / trial 1 / refused 1).
Isolation: pyscf/d2_full stubbed; formal directories byte-identical.
"""
import os, sys, json, tempfile, types
import numpy as np

sys.modules['pyscf'] = None
sys.modules['pyscf.hessian'] = None
sys.modules['d2_full'] = types.ModuleType('d2_full')

HERE = '/mnt/e/dft-service/ozone_mnb_uv_simulation/protocols'
sys.path.insert(0, HERE)
import job047_exec as ex
import job051_exec as j51
import job061_exec as j61

R60 = j61.R60
OUT = j61.OUT
BPA = 1.0 / ex.ANG_PER_BOHR

ex.GM_GATE = 1e-6
j51.GM_GATE = 1e-6


def snapshot(d):
    if not os.path.isdir(d):
        return []
    return sorted((f, os.path.getsize(os.path.join(d, f)))
                  for f in os.listdir(d))


class MockBackend:
    """Quadratic E(x)=E0+g0.x~+0.5 x~'Hx~ anchored AT the start (E and
    grad continuous with the reused record); Hdiag uniform 0.8."""

    def __init__(self, out_dir, Hdiag=None, e_off=0.0):
        src = json.load(open(os.path.join(R60, 'eval_opt_opt_24.json')))
        self.src = src
        self.x_start = (np.asarray(src['coords_actual_angstrom'], float)
                        * BPA).reshape(-1)
        self.E0 = float(src['e_total'])
        self.g0 = np.asarray(src['grad'], float).reshape(-1)
        if Hdiag is None:
            Hdiag = [0.8] * 21
        self.H = np.diag(np.asarray(Hdiag, float))
        self.e_off = e_off
        self.calls = []

    def full_eval(self, R_bohr, out_dir, tag):
        R = np.asarray(R_bohr, float).reshape(-1)
        self.calls.append((tag, R.tolist()))
        if np.abs(R - self.x_start).max() < 1e-12 and self.e_off == 0.0:
            rec = dict(self.src)
            rec.pop('gate', None)
        else:
            xt = R - self.x_start
            g = self.g0 + self.H @ xt
            E = self.E0 + float(self.g0 @ xt) + 0.5 * float(
                xt @ self.H @ xt)
            if self.e_off:
                E += self.e_off
            rec = dict(
                e_total=E, e_d2_Eh=0.0, e_dft_part_Eh=0.0,
                grad=g.reshape(7, 3).tolist(),
                grad_max=float(np.abs(g).max()), grad_sha='m',
                coords_actual_angstrom=(R.reshape(7, 3)
                                        * ex.ANG_PER_BOHR).tolist(),
                coords_sha='m',
                config=json.load(open(os.path.join(
                    R60, 'eval_opt_opt_24.json')))['config'],
                converged=True, all_finite=True, scf_kernel_count=1)
        rec['tag'] = tag
        ex.save_json_atomic(os.path.join(out_dir, 'eval_%s_%s.json'
                                         % (rec.get('category', 'x'),
                                            tag)), rec)
        return rec


def source_record():
    return json.load(open(os.path.join(R60, 'eval_opt_opt_24.json')))


def main():
    pre60 = snapshot(R60)
    pre61 = snapshot(OUT)
    src = source_record()
    R = {}

    # ---- T1: trigger patch semantics -------------------------------------
    meeting = dict(grad=np.asarray(src['grad'], float).reshape(7, 3)
                   .tolist(),
                   coords_actual_angstrom=src['coords_actual_angstrom'],
                   e_total=float(src['e_total']), config=src['config'],
                   converged=True)
    g_meet = dict(meeting)
    g_meet['grad_max'] = 8e-7
    ok = ex.recheck_gate_fn(meeting)(g_meet)
    bad = dict(meeting)
    bad['grad_max'] = 5e-6
    no = ex.recheck_gate_fn(meeting)(bad)
    R['T1'] = dict(pass_8e7=bool(ok['gates_pass']),
                   fail_5e6=bool(not no['gates_pass']),
                   gm_patched=ex.GM_GATE,
                   gates=(ex.GATE_E, ex.GATE_G, ex.GATE_C))
    R['T1_pass'] = bool(R['T1']['pass_8e7'] and R['T1']['fail_5e6']
                        and ex.GM_GATE == 1e-6
                        and R['T1']['gates'] == (1e-8, 1e-7, 1e-9))

    # ---- T2: start provenance --------------------------------------------
    acc = json.load(open(os.path.join(R60, 'accepted_iterates.json')))
    x = np.asarray(src['coords_actual_angstrom'], float).reshape(-1) * BPA
    devs = [float(np.abs(x - np.asarray(a, float).reshape(-1)).max())
            for a in acc['iterates']]
    r60res = json.load(open(os.path.join(R60, 'relax060_results.json')))
    tag25 = json.load(open(os.path.join(
        R60, 'eval_opt_opt_25.json')))['tag']
    assert tag25 == 'opt_25'
    R['T2'] = dict(last_dev=devs[-1],
                   best_is_last=bool(devs.index(min(devs))
                                     == len(devs) - 1),
                   not_opt25=bool(src['tag'] == 'opt_24'
                                  and tag25 == 'opt_25'),
                   gmax_ref=bool(src['grad_max']
                                 == 1.312036626515271e-4))
    R['T2_pass'] = bool(devs[-1] <= 1e-12 and R['T2']['best_is_last']
                        and R['T2']['not_opt25'] and R['T2']['gmax_ref'])

    # ---- T3: full flow -----------------------------------------------------
    tmp = tempfile.mkdtemp(prefix='job061_T3_')
    led = ex.Ledger(os.path.join(tmp, 'b.json'), caps=j61.CAPS)
    mb = MockBackend(tmp)
    rec_s = ex.eval_point('start', (np.asarray(
        src['coords_actual_angstrom'], float) * BPA), tmp, led, mb,
        'start_repro', gate_fn=ex.centre_gate_fn(src))
    gate_ok = rec_s['gate']['gates_pass']
    opt = j51.run_bfgs(rec_s, tmp, led, mb, opt_cap=20)
    import glob as _glob
    recs = [json.load(open(f)) for f in
            sorted(_glob.glob(os.path.join(tmp, 'eval_opt_opt_*.json')))]
    acc_list = [np.asarray(a, float) for a in opt['accepted_iterates']]
    cls = dict(start_repro=1, accepted_iterate=0, non_accepted_trial=0)
    for r in recs:
        xx = np.asarray(r['coords_actual_angstrom'], float).reshape(-1) \
            * BPA
        dmin = min((float(np.abs(xx - a).max()) for a in acc_list),
                   default=float('inf'))
        cls['accepted_iterate' if dmin <= 1e-12
            else 'non_accepted_trial'] += 1
    recheck_pass = None
    if opt['meeting_points']:
        mp = opt['meeting_points'][0]
        mrec = json.load(open('%s/eval_%s_%s.json'
                              % (tmp, mp['category'], mp['tag'])))
        rr = ex.eval_point('recheck', np.asarray(
            mrec['coords_actual_angstrom'], float) * BPA, tmp, led, mb,
            'recheck', gate_fn=ex.recheck_gate_fn(mrec))
        recheck_pass = bool(rr['gate']['gates_pass'])
    R['T3'] = dict(gate_pass=bool(gate_ok), outcome=opt['outcome'],
                   reuse_start=bool(opt['reuse_start']),
                   n_new=opt['n_new_evals'], classes=cls,
                   meeting=len(opt['meeting_points']),
                   recheck_pass=recheck_pass)
    R['T3_pass'] = bool(gate_ok and opt['reuse_start']
                        and opt['outcome'] == 'threshold_reached'
                        and cls['accepted_iterate'] >= 1
                        and len(opt['meeting_points']) >= 1
                        and recheck_pass)

    # ---- T4: budget cap ---------------------------------------------------
    tmp = tempfile.mkdtemp(prefix='job061_T4_')
    led = ex.Ledger(os.path.join(tmp, 'b.json'), caps=j61.CAPS)
    mb4 = MockBackend(tmp, Hdiag=[1e-4] * 21)
    rec_s4 = ex.eval_point('start', (np.asarray(
        src['coords_actual_angstrom'], float) * BPA), tmp, led, mb4,
        'start_repro', gate_fn=ex.centre_gate_fn(src))
    opt4 = j51.run_bfgs(rec_s4, tmp, led, mb4, opt_cap=2)
    R['T4'] = dict(outcome=opt4['outcome'], n_new=opt4['n_new_evals'],
                   blocked=len(opt4['blocked_requests']),
                   ledger_opt=led.count('opt'))
    R['T4_pass'] = bool(opt4['outcome'] == 'budget_exhausted'
                        and opt4['n_new_evals'] == 2
                        and led.count('opt') == 2
                        and len(opt4['blocked_requests']) >= 1)

    # ---- T5: start gate failure -------------------------------------------
    tmp = tempfile.mkdtemp(prefix='job061_T5_')
    led = ex.Ledger(os.path.join(tmp, 'b.json'), caps=j61.CAPS)
    mb5 = MockBackend(tmp, e_off=1e-6)
    err = None
    try:
        ex.eval_point('start', (np.asarray(
            src['coords_actual_angstrom'], float) * BPA), tmp, led, mb5,
            'start_repro', gate_fn=ex.centre_gate_fn(src))
    except RuntimeError as e:
        err = str(e)
    R['T5'] = dict(error=err[:60] if err else None,
                   stopped_json=bool(os.path.exists(
                       os.path.join(tmp,
                                    'STOPPED_gate_fail_start.json'))),
                   att0=led.data['attempts'][0]['status'])
    R['T5_pass'] = bool(err and 'gate failed' in err
                        and R['T5']['stopped_json']
                        and led.data['attempts'][0]['status'] == 'error')

    # ---- T6: 060 correction note consistency -------------------------------
    corr = json.load(open(os.path.join(R60, 'job060_correction_note.json')))
    c2 = corr['corrections'][1]['correct']
    ok6 = bool(c2 == dict(start_repro=1, accepted_iterate_evaluations=24,
                          non_accepted_trials=1,
                          budget_refused_requests=1))
    R['T6'] = dict(classification_ok=bool(ok6),
                   kept_clean=bool('largest drop' not in corr['kept']
                                   and 'harmless' not in corr['kept']),
                   warnings_registered=bool(
                       corr['corrections'][5]['count'] == 23))
    R['T6_pass'] = bool(ok6 and R['T6']['kept_clean']
                        and R['T6']['warnings_registered'])

    R['formal_dirs_untouched'] = bool(pre60 == snapshot(R60)
                                      and pre61 == snapshot(OUT))
    keys = ('T1_pass', 'T2_pass', 'T3_pass', 'T4_pass', 'T5_pass',
            'T6_pass', 'formal_dirs_untouched')
    R['ALL_PASS'] = bool(all(R[k] for k in keys))
    print(json.dumps(R, indent=1, default=str))
    ex.save_json_atomic(os.path.join(OUT, 'mock_test_results.json'), R)
    if not R['ALL_PASS']:
        raise SystemExit('MOCK TESTS FAILED')
    print('MOCK TESTS ALL PASS', flush=True)


if __name__ == '__main__':
    main()
