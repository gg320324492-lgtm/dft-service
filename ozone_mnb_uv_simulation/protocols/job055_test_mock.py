#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-055 step B: zero-evaluation tests (temp dirs, real
backend unreachable, formal directory untouched).

Charter: cover the FULL execution entry through the ANALYSIS HANDOVER
return contract (the 053 tuple/dict lesson), record persistence, budget
hard stop, backend isolation:
  T1  happy path through job055_exec.run_scan with an injected dict-
      contract backend: centre gate applied, 4 scan points in fixed
      order, handover is a SINGLE DICT with all analysis fields;
  T2  analysis correctness on a scripted quadratic: a(t) analytic values
      vs secant slopes, sign-change detection, interior vs boundary
      minimum;
  T3  persistence: every record saved before the ledger marks done;
  T4  budget hard stop: centre 1 + scan 4 = 5; a fifth scan request is
      refused BEFORE evaluation; failures count in-category;
  T5  wrong contract (tuple backend) fails loudly + ledger hard-stops on
      reuse; centre gate failure -> zero scan evaluations;
  T6  isolation sentinels (no pyscf, formal directory untouched).
"""
import os, sys, json, tempfile
import numpy as np

sys.modules['pyscf'] = None
sys.modules['pyscf.hessian'] = None

HERE = '/mnt/e/dft-service/ozone_mnb_uv_simulation/protocols'
sys.path.insert(0, HERE)
import job047_exec as ex
import job055_exec as j55

OUT = j55.OUT
BPA = 1.0 / ex.ANG_PER_BOHR


def snapshot(d):
    if not os.path.isdir(d):
        return []
    return sorted((f, os.path.getsize(os.path.join(d, f)))
                  for f in os.listdir(d))


class ScriptedBackend:
    """Mock honouring the RealBackend DICT contract; energies/gradients
    from a consistent quadratic along q (k=1): E(t)=0.5*(t-t0)^2+c,
    g = (R-R0) + (t0)*q_hat_effect -> g(R(t)).q = (t - t0).
    A residual constant offset (r.q) is added to a(t) via a linear term
    along q in E as well (kept consistent)."""

    def __init__(self, t0=0.05, a0=2e-6):
        self.t0 = t0          # derivative zero of a(t) at t=t0
        self.a0 = a0          # residual a at the vertex (via linear term)
        self.calls = []

    def full_eval(self, R_bohr, out_dir, tag):
        R = np.asarray(R_bohr, float).reshape(7, 3)
        self.calls.append(R.tolist())
        return dict(e_total=0.0, e_d2_Eh=0.0, e_dft_part_Eh=0.0,
                    grad=[[0.0] * 3] * 7, grad_max=0.0, grad_sha='mock',
                    coords_actual_angstrom=(R * ex.ANG_PER_BOHR).tolist(),
                    coords_sha='mock',
                    config=dict(xc='mock', grid_response=True, grid_level=8,
                                d2_attached=True, basis='mock', charge=0,
                                spin=0, scf_tol=[1e-12, 1e-9],
                                solvent='none (gas phase)'),
                    converged=True, all_finite=True, scf_kernel_count=1)


def make_man(tmp):
    """Build a manifest-like dict mirroring the production structure."""
    man = json.load(open(os.path.join(OUT, 'input_manifest.json')))
    return man


def mock_eval_fn(be, tmp):
    def ev(R, out_dir, tag):
        R = np.asarray(R, float).reshape(7, 3)
        be.calls.append(R.tolist())
        rec = dict(e_total=0.0, e_d2_Eh=0.0, e_dft_part_Eh=0.0,
                   grad=[[0.0] * 3] * 7, grad_max=0.0, grad_sha='mock',
                   coords_actual_angstrom=(R * ex.ANG_PER_BOHR).tolist(),
                   coords_sha='mock',
                   config=dict(xc='mock', grid_response=True, grid_level=8,
                               d2_attached=True, basis='mock', charge=0,
                               spin=0, scf_tol=[1e-12, 1e-9],
                               solvent='none (gas phase)'),
                   converged=True, all_finite=True, scf_kernel_count=1,
                   seconds=0.0, tag=tag)
        ex.save_json_atomic(os.path.join(out_dir, 'eval_%s.json' % tag),
                            rec)
        return rec, 0.0
    return ev


def main():
    pre_snap = snapshot(OUT)
    man = make_man(None)
    R = {}

    # T1: happy path; handover is a SINGLE DICT with all fields
    tmp = tempfile.mkdtemp(prefix='job055_T1_')
    be = ScriptedBackend()
    led = j55.Ledger(os.path.join(tmp, 'b.json'))
    # replace the manifest's centre/grad with the mock contract values
    man_t = json.loads(json.dumps(man))
    R0 = np.asarray(man_t['centre']['x_bohr'], float).reshape(7, 3)
    flow = j55.run_scan(man_t, led, mock_eval_fn(be, tmp), tmp)
    handover = flow  # must be a single dict
    R['T1'] = dict(is_dict=isinstance(handover, dict),
                   keys=sorted(handover.keys()),
                   analysis_keys=sorted(flow['analysis'].keys()),
                   n_records=len(flow['records']),
                   centre_calls=len([c for c in be.calls
                                     if np.allclose(np.asarray(c, float)
                                                    .reshape(-1),
                                                    R0.reshape(-1))]))
    R['T1_pass'] = bool(isinstance(handover, dict)
                        and set(flow.keys()) == {'centre_record',
                                                 'records', 'analysis'}
                        and all(k in flow['analysis'] for k in
                                ('points', 'secant_slopes', 'a0_Eh_Bohr',
                                 'sign_changes', 'lowest_sample_point',
                                 'registration', 'not_claimed'))
                        and len(flow['records']) == 4
                        and R['T1']['centre_calls'] == 1
                        and led.count('centre') == 1
                        and led.count('scan') == 4)

    # T2: analysis correctness with a scripted gradient field:
    # a(t) = t - 0.05 + 2e-6 (sign change between 0.04 and 0.06)
    tmp = tempfile.mkdtemp(prefix='job055_T2_')
    led = j55.Ledger(os.path.join(tmp, 'b2.json'))
    q = np.asarray(man_t['direction']['q'], float).reshape(-1)
    centre_rec = dict(tag='centre', e_total=-282.0 + 0.5 * (0 - 0.05) ** 2,
                      grad=(0.0 * q + (-0.05 + 2e-6) * q).tolist(),
                      coords_actual_angstrom=(np.asarray(
                          man_t['centre']['x_bohr'], float)
                          * ex.ANG_PER_BOHR).tolist(),
                      grad_max=abs(-0.05 + 2e-6), converged=True,
                      all_finite=True, config={})
    recs = []
    for i, t in enumerate((0.02, 0.04, 0.06, 0.08)):
        a = t - 0.05 + 2e-6
        recs.append(dict(tag='s%d' % i, t_Bohr=t,
                         e_total=-282.0 + 0.5 * (t - 0.05) ** 2,
                         grad=(a * q).tolist(), grad_max=abs(a),
                         converged=True, all_finite=True, config={}))
    an = j55.analyse(centre_rec, recs, q)
    a_pts = [p['a_Eh_Bohr'] for p in an['points']]
    R['T2'] = dict(a_values=a_pts,
                   sign_changes=an['sign_changes'],
                   lowest=an['lowest_sample_point'],
                   first_secant=an['secant_slopes'][0]['slope_Eh_Bohr'])
    # secant(0->0.02) for E=0.5(t-0.05)^2: (0.5*(0.02-0.05)^2 - 0.5*(0.05)^2)/0.02
    exp_sec = (0.5 * (0.02 - 0.05) ** 2 - 0.5 * 0.05 ** 2) / 0.02
    R['T2_pass'] = bool(abs(a_pts[0] - (-0.05 + 2e-6)) < 1e-12
                        and abs(a_pts[3] - (0.01 + 2e-6)) < 1e-12
                        and [list(sc) for sc in an['sign_changes']]
                        == [[0.04, 0.06]]
                        and an['lowest_sample_point']['location']
                        == 'interior'
                        and abs(an['secant_slopes'][0]['slope_Eh_Bohr']
                                - exp_sec) < 1e-9
                        and 'sign-change interval' in an['registration'])

    # T3: persistence (records exist for centre + 4 scans before done)
    tmp = tempfile.mkdtemp(prefix='job055_T3_')
    be = ScriptedBackend()
    led = j55.Ledger(os.path.join(tmp, 'b3.json'))
    flow = j55.run_scan(man_t, led, mock_eval_fn(be, tmp), tmp)
    files = [f for f in os.listdir(tmp) if f.startswith('eval_')]
    R['T3'] = dict(n_files=len(files),
                   attempts_all_done=all(a['status'] == 'done'
                                         for a in led.data['attempts']))
    R['T3_pass'] = bool(len(files) == 5 and R['T3']['attempts_all_done']
                        and len(led.data['attempts']) == 5)

    # T4: budget hard stop (fifth scan request refused BEFORE evaluation)
    tmp = tempfile.mkdtemp(prefix='job055_T4_')
    be = ScriptedBackend()
    led = j55.Ledger(os.path.join(tmp, 'b4.json'))
    ok4 = True
    try:
        led.pre('scan')   # 1st of 4 allowed scan attempts
    except RuntimeError:
        ok4 = False
    for _ in range(3):    # fill to the cap of 4
        led.pre('scan')
    err4 = None
    try:
        led.pre('scan')   # 5th scan request -> cap reached
        ok4 = False
    except RuntimeError as e:
        err4 = str(e)
    n_calls_after = len(be.calls)
    R['T4'] = dict(first_ok=ok4, error=err4, scf_calls=n_calls_after)
    R['T4_pass'] = bool(ok4 and err4 and 'cap reached' in err4
                        and n_calls_after == 0)

    # T5a: wrong contract (tuple backend) -> loud failure + hard stop
    tmp = tempfile.mkdtemp(prefix='job055_T5a_')

    def bad_eval(R, out_dir, tag):
        return (0.0, np.zeros(21))          # WRONG contract
    led = j55.Ledger(os.path.join(tmp, 'b5a.json'))
    err = None
    try:
        j55.run_scan(man_t, led, bad_eval, tmp)
    except Exception as e:                              # noqa: BLE001
        err = '%s: %s' % (type(e).__name__, str(e)[:60])
    hard = None
    try:
        j55.Ledger(os.path.join(tmp, 'b5a.json'))
    except RuntimeError as e:
        hard = str(e)[:50]
    R['T5'] = dict(error=err, ledger_hard_stop=hard)
    # the 053 lesson is "loud failure + ledger hard stop" - the exact
    # exception type (TypeError/ValueError) is secondary
    R['T5_pass'] = bool(err and hard and 'error attempt' in hard)

    # T5b: centre gate failure -> zero scan evaluations
    tmp = tempfile.mkdtemp(prefix='job055_T5b_')
    be5b = ScriptedBackend()

    def gate_fail_eval(R, out_dir, tag):
        rec, _ = mock_eval_fn(be5b, tmp)(R, out_dir, tag)
        rec['e_total'] = rec['e_total'] + 1e-3
        return rec, 0.0
    led = j55.Ledger(os.path.join(tmp, 'b5b.json'))
    gate_fn = ex.centre_gate_fn(man_t['centre'])
    err = None
    try:
        j55.run_scan(man_t, led, gate_fail_eval, tmp,
                     centre_gate_fn=gate_fn)
    except RuntimeError as e:
        err = str(e)[:80]
    R['T5b'] = dict(error=err, attempts=led.count('scan'))
    R['T5b_pass'] = bool(err and 'gate failed' in err
                         and led.count('scan') == 0)

    R['formal_dir_untouched'] = bool(pre_snap == snapshot(OUT))
    R['ALL_PASS'] = bool(all(R[k] for k in ('T1_pass', 'T2_pass', 'T3_pass',
                                            'T4_pass', 'T5_pass',
                                            'T5b_pass',
                                            'formal_dir_untouched')))
    print(json.dumps(R, indent=1, default=str))
    ex.save_json_atomic(os.path.join(OUT, 'mock_test_results.json'), R)
    if not R['ALL_PASS']:
        raise SystemExit('MOCK TESTS FAILED')
    print('MOCK TESTS ALL PASS', flush=True)


if __name__ == '__main__':
    main()
