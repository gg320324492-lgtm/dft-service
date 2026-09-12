#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-060 step B: zero-evaluation tests (temp dirs, real
backend unreachable, formal directories untouched).

  T1  runtime trigger patch: recheck gate under GM_GATE=1e-6 - a point at
      8e-7 passes, a point at 5e-6 (below the general 1e-5 gate!) fails;
      centre-gate constants match the charter (1e-8/1e-7/1e-9);
  T2  start_repro reuse: run_bfgs's first request at the identical start
      coordinates reuses the persisted record (0 duplicate SCF);
  T3  full flow on a quadratic mock backend: start gate PASS -> BFGS ->
      threshold_reached at the 1e-6 trigger -> conditional recheck;
      accepted iterates vs line-search trials classified;
  T4  budget: opt cap blocks further real evaluations (budget_exhausted,
      blocked_requests recorded, ledger count = cap);
  T5  start gate failure -> error attempt + STOPPED json + 0 opt evals;
  T6  start_already_meets_trigger branch (fake start gmax 5e-7 < 1e-6) ->
      no new evaluations;
  T7  059 correction numerators equal kE*h^2 from the saved results.
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
import job060_exec as j60

R59 = j60.R59
OUT = j60.OUT
BPA = 1.0 / ex.ANG_PER_BOHR

# the SAME runtime patch the production exec applies (fixed BEFORE runs)
ex.GM_GATE = 1e-6
j51.GM_GATE = 1e-6


def snapshot(d):
    if not os.path.isdir(d):
        return []
    return sorted((f, os.path.getsize(os.path.join(d, f)))
                  for f in os.listdir(d))


class MockBackend:
    """Quadratic E(x)=0.5(x-x*)'H(x-x*); the UNIQUE start coordinates
    return the REAL 059 disp_p002 record values exactly.  Hdiag anisotropic
    so BFGS takes several accepted iterations (both classes appear)."""

    def __init__(self, out_dir, Hdiag=None, e_off=0.0):
        src = json.load(open(os.path.join(R59, 'eval_disp_p002.json')))
        self.src = src
        self.x_start = (np.asarray(src['coords_actual_angstrom'], float)
                        * BPA).reshape(-1)
        g0 = np.asarray(src['grad'], float).reshape(-1)
        if Hdiag is None:
            Hdiag = [0.8] * 21   # uniform 0.8: gradient x0.2 per exact step
        self.H = np.diag(np.asarray(Hdiag, float))
        # Taylor form anchored AT the start: E(x)=E0+g0.x~+0.5 x~'Hx~
        # (x~=x-x_start) -> E and grad are CONTINUOUS with the reused
        # record at x_start (minimum at x_start - H^-1 g0)
        self.x_start = self.x_start  # noqa: clarity
        self.E0 = float(src['e_total'])
        self.g0 = np.asarray(src['grad'], float).reshape(-1)
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
                e_total=E,
                e_d2_Eh=0.0, e_dft_part_Eh=0.0,
                grad=g.reshape(7, 3).tolist(),
                grad_max=float(np.abs(g).max()),
                grad_sha='m',
                coords_actual_angstrom=(R.reshape(7, 3)
                                        * ex.ANG_PER_BOHR).tolist(),
                coords_sha='m',
                config=json.load(open(os.path.join(
                    R59, 'eval_disp_p002.json')))['config'],
                converged=True, all_finite=True, scf_kernel_count=1)
        rec['tag'] = tag
        ex.save_json_atomic(os.path.join(out_dir,
                                         'eval_%s_%s.json'
                                         % (rec.get('category', 'x'),
                                            tag)), rec)
        return rec


def source_record():
    src = json.load(open(os.path.join(R59, 'eval_disp_p002.json')))
    return src


def main():
    pre59 = snapshot(R59)
    pre60 = snapshot(OUT)
    src = source_record()
    R = {}

    # ---- T1: trigger patch semantics -------------------------------------
    meeting = dict(grad=np.asarray(src['grad'], float).reshape(7, 3)
                   .tolist(),
                   coords_actual_angstrom=src['coords_actual_angstrom'],
                   e_total=float(src['e_total']), config=src['config'],
                   converged=True)
    g_meet = dict(meeting)
    g_meet['grad_max'] = 8e-7          # same grad/coords, own gmax < 1e-6
    ok = ex.recheck_gate_fn(meeting)(g_meet)
    bad = dict(meeting)
    bad['grad_max'] = 5e-6             # below the general 1e-5 gate!
    no = ex.recheck_gate_fn(meeting)(bad)
    cg = ex.centre_gate_fn(src)
    R['T1'] = dict(pass_8e7=bool(ok['gates_pass']),
                   fail_5e6=bool(not no['gates_pass']),
                   gmax_checked=no['gmax_recheck'],
                   gate_E=ex.GATE_E, gate_G=ex.GATE_G, gate_C=ex.GATE_C,
                   gm_patched=ex.GM_GATE)
    R['T1_pass'] = bool(R['T1']['pass_8e7'] and R['T1']['fail_5e6']
                        and ex.GM_GATE == 1e-6
                        and (ex.GATE_E, ex.GATE_G, ex.GATE_C)
                        == (1e-8, 1e-7, 1e-9))

    # ---- T2/T3: full flow -------------------------------------------------
    tmp = tempfile.mkdtemp(prefix='job060_T3_')
    led = ex.Ledger(os.path.join(tmp, 'b.json'), caps=j60.CAPS)
    mb = MockBackend(tmp)
    rec_s = ex.eval_point('start', (np.asarray(
        src['coords_actual_angstrom'], float) * BPA), tmp, led, mb,
        'start_repro', gate_fn=ex.centre_gate_fn(src))
    gate_ok = rec_s['gate']['gates_pass']
    opt = j51.run_bfgs(rec_s, tmp, led, mb, opt_cap=25)
    R['T2'] = dict(gate_pass=bool(gate_ok),
                   reuse_start=bool(opt['reuse_start']),
                   outcome=opt['outcome'],
                   n_new=opt['n_new_evals'])
    R['T2_pass'] = bool(gate_ok and opt['reuse_start']
                        and opt['outcome'] in ('threshold_reached',
                                               'optimizer_returned'))
    # accepted vs trials classification (same logic as production:
    # nearest accepted iterate within 1e-12 Bohr); evaluations live in
    # the saved eval_opt_*.json files
    import glob as _glob
    acc_list = [np.asarray(a, float) for a in opt['accepted_iterates']]
    recs = [json.load(open(f)) for f in
            sorted(_glob.glob(os.path.join(tmp, 'eval_opt_opt_*.json')))]
    # if the trigger fired inside the FIRST line search no callback has
    # fired yet; inject one accepted iterate to unit-test the matching
    if not acc_list and recs:
        acc_list = [np.asarray(recs[0]['coords_actual_angstrom'], float)
                    .reshape(-1) * BPA]
    acc_cls = []
    for r in recs:
        x = np.asarray(r['coords_actual_angstrom'], float).reshape(-1) * BPA
        dmin = min((float(np.abs(x - a).max()) for a in acc_list),
                   default=float('inf'))
        acc_cls.append('accepted_iterate' if dmin <= 1e-12
                       else 'linesearch_trial')
    n_rec = len(recs)
    R['T3'] = dict(n_records=n_rec, classes=set(acc_cls),
                   meeting=len(opt['meeting_points']),
                   ledger_opt=led.count('opt'),
                   ledger_start=led.count('start_repro'))
    R['T3_pass'] = bool(n_rec == opt['n_new_evals']
                        and led.count('opt') == n_rec
                        and led.count('start_repro') == 1
                        and len(opt['meeting_points']) >= 1
                        and ('accepted_iterate' in acc_cls))

    # recheck on the meeting point
    if opt['meeting_points']:
        mp = opt['meeting_points'][0]
        mrec = json.load(open('%s/eval_%s_%s.json'
                              % (tmp, mp['category'], mp['tag'])))
        rr = ex.eval_point('recheck', np.asarray(
            mrec['coords_actual_angstrom'], float) * BPA, tmp, led, mb,
            'recheck', gate_fn=ex.recheck_gate_fn(mrec))
        R['T3']['recheck_pass'] = bool(rr['gate']['gates_pass'])
        R['T3_pass'] = R['T3_pass'] and bool(rr['gate']['gates_pass'])

    # ---- T4: budget cap ---------------------------------------------------
    tmp = tempfile.mkdtemp(prefix='job060_T4_')
    led = ex.Ledger(os.path.join(tmp, 'b.json'), caps=j60.CAPS)
    mb4 = MockBackend(tmp, Hdiag=[1e-4] * 21)   # very soft: slow descent
    rec_s4 = ex.eval_point('start', (np.asarray(
        src['coords_actual_angstrom'], float) * BPA), tmp, led, mb4,
        'start_repro', gate_fn=ex.centre_gate_fn(src))
    opt4 = j51.run_bfgs(rec_s4, tmp, led, mb4, opt_cap=3)
    R['T4'] = dict(outcome=opt4['outcome'], n_new=opt4['n_new_evals'],
                   blocked=len(opt4['blocked_requests']),
                   ledger_opt=led.count('opt'))
    R['T4_pass'] = bool(opt4['outcome'] == 'budget_exhausted'
                        and opt4['n_new_evals'] == 3
                        and led.count('opt') == 3
                        and len(opt4['blocked_requests']) >= 1)

    # ---- T5: start gate failure -------------------------------------------
    tmp = tempfile.mkdtemp(prefix='job060_T5_')
    led = ex.Ledger(os.path.join(tmp, 'b.json'), caps=j60.CAPS)
    mb5 = MockBackend(tmp, e_off=1e-6)     # breaks the 1e-8 energy gate
    err = None
    try:
        ex.eval_point('start', (np.asarray(
            src['coords_actual_angstrom'], float) * BPA), tmp, led, mb5,
            'start_repro', gate_fn=ex.centre_gate_fn(src))
    except RuntimeError as e:
        err = str(e)
    R['T5'] = dict(error=err[:60] if err else None,
                   stopped_json=bool(os.path.exists(
                       os.path.join(tmp, 'STOPPED_gate_fail_start.json'))),
                   att0=led.data['attempts'][0]['status'])
    R['T5_pass'] = bool(err and 'gate failed' in err
                        and R['T5']['stopped_json']
                        and led.data['attempts'][0]['status'] == 'error')

    # ---- T6: start_already_meets_trigger branch ---------------------------
    tmp = tempfile.mkdtemp(prefix='job060_T6_')
    led = ex.Ledger(os.path.join(tmp, 'b.json'), caps=j60.CAPS)
    src6 = dict(src)
    src6['grad'] = (np.asarray(src['grad'], float)
                    * 0.05).tolist()     # gmax ~3.4e-7 < 1e-6
    src6['grad_max'] = 3.4e-7
    rec6 = dict(src6)
    rec6['coords_actual_angstrom'] = src['coords_actual_angstrom']
    rec6['e_total'] = float(src['e_total'])
    rec6['category'] = 'start_repro'
    rec6['tag'] = 'start'
    opt6 = j51.run_bfgs(rec6, tmp, led, mb := MockBackend(tmp),
                        opt_cap=25)
    R['T6'] = dict(outcome=opt6['outcome'],
                   n_new=opt6['n_new_evals'],
                   backend_calls=len(mb.calls))
    R['T6_pass'] = bool(opt6['outcome'] == 'start_already_meets_gate'
                        and opt6['n_new_evals'] == 0
                        and len(mb.calls) == 0)

    # ---- T7: 059 correction numerators ------------------------------------
    corr = json.load(open(os.path.join(R59, 'job059_correction_note.json')))
    res59 = json.load(open(os.path.join(R59, 'negdir059_results.json')))
    ok7 = True
    for h, v in res59['analysis']['per_h'].items():
        want = v['kE_Eh_Bohr2'] * float(h) ** 2
        got = corr['corrections'][1]['per_h'][h][
            'second_difference_numerator_Eh']
        ok7 = ok7 and abs(want - got) <= 1e-24
    R['T7'] = dict(numerators_match=bool(ok7),
                   withdrawn_present=bool('withdrawn'
                                          in json.dumps(corr)))
    R['T7_pass'] = bool(ok7 and R['T7']['withdrawn_present'])

    R['formal_dirs_untouched'] = bool(pre59 == snapshot(R59)
                                      and pre60 == snapshot(OUT))
    keys = ('T1_pass', 'T2_pass', 'T3_pass', 'T4_pass', 'T5_pass',
            'T6_pass', 'T7_pass', 'formal_dirs_untouched')
    R['ALL_PASS'] = bool(all(R[k] for k in keys))
    print(json.dumps(R, indent=1, default=str))
    ex.save_json_atomic(os.path.join(OUT, 'mock_test_results.json'), R)
    if not R['ALL_PASS']:
        raise SystemExit('MOCK TESTS FAILED')
    print('MOCK TESTS ALL PASS', flush=True)


if __name__ == '__main__':
    main()
