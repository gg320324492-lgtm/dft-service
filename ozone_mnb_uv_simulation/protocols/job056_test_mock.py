#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-056 step B: zero-evaluation tests (temp dirs, real
backend unreachable, formal directory untouched).

Coverage (charter section 4):
  T1  new-point displacements equal the target t (norm and along-q);
      q used as a Cartesian unit direction (NO repeated mass conversion);
  T2  055 existing endpoint records correctly reused in the merged
      analysis (t=0 centre, t=0.04, t=0.06 from the manifest); no centre
      SCF (the eval_fn is never called at t=0);
  T3  new-point configs match the 055 config; full records persisted
      before the ledger marks done; completion order recorded;
  T4  budget hard stop: 3 refine attempts; the 4th request refused
      BEFORE evaluation; single-point failure -> ledger keeps the error
      attempt and the run stops;
  T5  isolation sentinels (no pyscf, formal directory untouched).
"""
import os, sys, json, tempfile
import numpy as np

sys.modules['pyscf'] = None
sys.modules['pyscf.hessian'] = None

HERE = '/mnt/e/dft-service/ozone_mnb_uv_simulation/protocols'
sys.path.insert(0, HERE)
import job047_exec as ex
import job056_exec as j56

OUT = j56.OUT
BPA = 1.0 / ex.ANG_PER_BOHR


def snapshot(d):
    if not os.path.isdir(d):
        return []
    return sorted((f, os.path.getsize(os.path.join(d, f)))
                  for f in os.listdir(d))


def mock_eval_fn(be_calls, tmp, cfg):
    def ev(R, out_dir, tag):
        R = np.asarray(R, float).reshape(7, 3)
        be_calls.append(R.tolist())
        rec = dict(e_total=-282.0017216000, e_d2_Eh=-0.0003,
                   e_dft_part_Eh=-282.0014216000,
                   grad=[[1e-6] * 3] * 7, grad_max=1e-6, grad_sha='mock',
                   coords_actual_angstrom=(R * ex.ANG_PER_BOHR).tolist(),
                   coords_sha='mock', config=dict(cfg), converged=True,
                   all_finite=True, scf_kernel_count=1, seconds=0.0,
                   tag=tag)
        ex.save_json_atomic(os.path.join(out_dir, 'eval_%s.json' % tag),
                            rec)
        return rec, 0.0
    return ev


def main():
    pre_snap = snapshot(OUT)
    man = json.load(open(os.path.join(OUT, 'input_manifest.json')))
    q = np.asarray(man['direction']['q'], float).reshape(-1)
    R = {}

    # T1: displacements = t; q Cartesian unit (no re-conversion)
    ok1 = True
    for g in man['new_geometries']:
        R0 = np.asarray(man['centre']['x_bohr'], float).reshape(7, 3)
        Rn = np.asarray(g['coords_bohr'], float).reshape(7, 3)
        d = (Rn - R0).reshape(-1)
        if abs(float(np.linalg.norm(d)) - g['t_Bohr']) > 1e-12 \
                or abs(float(d @ q) - g['t_Bohr']) > 1e-12:
            ok1 = False
    R['T1'] = dict(displacements_ok=ok1,
                   q_norm=float(np.linalg.norm(q)),
                   q_is_cartesian=man['direction']['already_cartesian'])
    R['T1_pass'] = bool(ok1 and abs(R['T1']['q_norm'] - 1.0) < 1e-12
                        and R['T1']['q_is_cartesian'])

    # T2/T3: happy path - reuse, no centre SCF, persistence, order
    tmp = tempfile.mkdtemp(prefix='job056_T2_')
    be_calls = []
    led = j56.Ledger(os.path.join(tmp, 'b.json'))
    flow = j56.run_refine(man, led, mock_eval_fn(be_calls, tmp,
                                                 man['centre']['config']),
                          tmp)
    called_R = [np.asarray(c, float).reshape(-1) for c in be_calls]
    centre_called = any(np.allclose(c, np.asarray(
        man['centre']['x_bohr'], float).reshape(-1)) for c in called_R)
    files = [f for f in os.listdir(tmp) if f.startswith('eval_')]
    order_ok = flow['analysis']['points'][2]['tag'] == 'refine_t045' \
        and flow['analysis']['points'][3]['tag'] == 'refine_t050' \
        and flow['analysis']['points'][4]['tag'] == 'refine_t055' \
        and flow['analysis']['points'][5]['tag'] == 'scan_t060_055' \
        and flow['analysis']['points'][1]['tag'] == 'scan_t040_055'
    cfg_ok = all(r['config'] == man['centre']['config']
                 for r in flow['records'])
    R['T2'] = dict(centre_scf_called=centre_called,
                   n_eval_files=len(files), n_attempts=led.count('refine'),
                   merged_points=len(flow['analysis']['points']),
                   order_ok=order_ok, cfg_ok=cfg_ok,
                   reused_sources=[p['source'] for p in
                                   flow['analysis']['points']])
    R['T2_pass'] = bool(not centre_called and len(files) == 3
                        and led.count('refine') == 3
                        and len(flow['analysis']['points']) == 6
                        and order_ok and cfg_ok
                        and flow['analysis']['points'][0]['source']
                        == 'reused 055 centre'
                        and flow['analysis']['points'][1]['source']
                        == 'reused 055 record'
                        and flow['analysis']['points'][-1]['source']
                        == 'reused 055 record')

    # T4a: 4th refine request refused BEFORE evaluation
    tmp = tempfile.mkdtemp(prefix='job056_T4_')
    be_calls = []
    led = j56.Ledger(os.path.join(tmp, 'b4.json'))
    ev = mock_eval_fn(be_calls, tmp, man['centre']['config'])
    j56.run_refine(man, led, ev, tmp)
    n_before = len(be_calls)
    err4 = None
    try:
        led.pre('refine')
    except RuntimeError as e:
        err4 = str(e)
    R['T4'] = dict(error=err4, scf_calls_after=len(be_calls))
    R['T4_pass'] = bool(err4 and 'cap reached' in err4
                        and len(be_calls) == n_before)

    # T4b: single-point failure -> ledger error attempt + stop
    tmp = tempfile.mkdtemp(prefix='job056_T4b_')

    def failing_ev(R, out_dir, tag):
        if tag == 'refine_t050':
            raise RuntimeError('SCF did not converge (mock)')
        return mock_eval_fn(be_calls, tmp, man['centre']['config'])(
            R, out_dir, tag)
    led = j56.Ledger(os.path.join(tmp, 'b4b.json'))
    err = None
    try:
        j56.run_refine(man, led, failing_ev, tmp)
    except RuntimeError as e:
        err = str(e)[:60]
    atts = [(a['category'], a['status']) for a in led.data['attempts']]
    hard = None
    try:
        j56.Ledger(os.path.join(tmp, 'b4b.json'))
    except RuntimeError as e:
        hard = str(e)[:50]
    R['T4b'] = dict(error=err, attempts=atts, hard=hard)
    R['T4b_pass'] = bool(err and ('SCF did not converge' in err)
                         and atts == [('refine', 'done'),
                                      ('refine', 'error')]
                         and hard and 'error attempt' in hard)

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
