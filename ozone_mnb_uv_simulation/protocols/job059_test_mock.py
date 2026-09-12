#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-059 step B: zero-evaluation tests (temp dirs, real
backend unreachable, formal directories untouched).

  T1  differencing formulas under a NON-ZERO centre gradient: quadratic
      model E(x)=0.5 x'Hx + g0.x with random g0 != 0 through the ACTUAL
      analyse() -> kE and kg recover q^T H q exactly, a0 = g0.q,
      slope = g0.q (exact for a quadratic), near-resolution flags only
      for genuinely tiny |dE|;
  T2  kH uses the SAVED matrix route (q^T H q) and the lam0/d_norm_sq
      relation holds;
  T3  full flow through the ACTUAL run_batch with injected eval: centre
      dual gate, 4 displacements in preregistered order, immediate
      per-point save, 5 attempts;
  T4  budget: 6th displacement request refused; centre non-convergence
      -> error attempt + stop with 0 displacement calls; displacement
      save exception -> error attempt + stop;
  T5  centre gate failure -> 0 displacement calls;
  T6  direction conventions: q Euclidean-normalised (no double mass
      division), sign = largest-|component| positive, hash stable.
Isolation: pyscf + job052_exec stubbed; the 058/058r formal directories
are snapshotted and must be byte-identical afterwards.
"""
import os, sys, json, tempfile, types, hashlib
import numpy as np

sys.modules['pyscf'] = None
sys.modules['pyscf.hessian'] = None
sys.modules['job052_exec'] = types.ModuleType('job052_exec')

HERE = '/mnt/e/dft-service/ozone_mnb_uv_simulation/protocols'
sys.path.insert(0, HERE)
import job047_exec as ex
import job059_exec as j59

R58R = j59.R58R
OUT = j59.OUT
BPA = 1.0 / ex.ANG_PER_BOHR
rng = np.random.default_rng(59)


def snapshot(d):
    if not os.path.isdir(d):
        return []
    return sorted((f, os.path.getsize(os.path.join(d, f)))
                  for f in os.listdir(d))


def make_man(tmp_q=None):
    man = json.load(open(os.path.join(OUT, 'input_manifest.json')))
    if tmp_q is None:
        tmp_q = np.asarray(man['direction']['q'], float).reshape(-1)
    M = rng.standard_normal((21, 21))
    H = 0.5 * (M + M.T) + np.eye(21) * 0.5
    # mock model: keep q as the direction, generic H
    man['direction']['q'] = tmp_q.tolist()
    man['_H_mock'] = H.tolist()
    # centre refs aligned to the mock model (E0=0, g0 != 0 along q)
    g0 = np.asarray(man['centre']['grad'], float).reshape(-1)
    g0_mock = g0 * 0.5                       # non-zero, still <= 1e-5 gmax
    man['centre']['e_total'] = 0.0
    man['centre']['grad'] = g0_mock.reshape(7, 3).tolist()
    man['centre']['grad_max'] = float(np.abs(g0_mock).max())
    man['centre']['coords_actual_angstrom'] = (
        np.asarray(man['centre']['x_bohr'], float) * ex.ANG_PER_BOHR
    ).tolist()
    cc = man['centre']['cross_check_ref']
    cc['e_total'] = 0.0
    cc['grad'] = g0_mock.reshape(7, 3).tolist()
    cc['coords_actual_angstrom'] = man['centre']['coords_actual_angstrom']
    return man


def mock_eval_factory(cfg, H, q, x0, g0, fail_conv_tags=(),
                      raise_tags=()):
    def ev(R, out_dir, tag):
        R = np.asarray(R, float).reshape(7, 3)
        x = (R.reshape(-1) - x0)
        g = H @ x + g0
        E = float(0.5 * x @ H @ x + g0 @ x)
        conv = tag not in fail_conv_tags
        rec = dict(e_total=(E if conv else 0.0), e_d2_Eh=0.0,
                   e_dft_part_Eh=0.0,
                   grad=(g.reshape(7, 3).tolist() if conv
                         else [[0.0] * 3] * 7),
                   grad_max=float(np.abs(g).max()), grad_sha='m',
                   coords_actual_angstrom=(R * ex.ANG_PER_BOHR).tolist(),
                   coords_sha='m', config=dict(cfg), converged=conv,
                   all_finite=True, scf_kernel_count=1, seconds=0.0,
                   tag=tag)
        if tag in raise_tags:
            raise RuntimeError('mock save exception')
        ex.save_json_atomic(os.path.join(out_dir,
                                         'eval_%s.json' % tag), rec)
        return rec, 0.0, None
    return ev


def main():
    pre58 = snapshot(R58R)
    pre59 = snapshot(OUT)
    man = make_man()
    cfg = man['centre']['config']
    H = np.asarray(man['_H_mock'], float)
    q = np.asarray(man['direction']['q'], float).reshape(-1)
    x0 = np.asarray(man['centre']['x_bohr'], float).reshape(-1)
    g0 = np.asarray(man['centre']['grad'], float).reshape(-1)
    R = {}

    # ---- T1/T2: formulas with NON-ZERO centre gradient ------------------
    tmp = tempfile.mkdtemp(prefix='job059_T1_')
    be = []
    led = j59.Ledger(os.path.join(tmp, 'b.json'))
    flow = j59.run_batch(man, led,
                         mock_eval_factory(cfg, H, q, x0, g0), tmp)
    ana = j59.analyse(man, flow['centre_record'], flow['disp_records'],
                      H, H)
    kH_true = float(q @ H @ q)
    kE1 = ana['per_h']['0.001']['kE_Eh_Bohr2']
    kE2 = ana['per_h']['0.002']['kE_Eh_Bohr2']
    kg1 = ana['per_h']['0.001']['kg_Eh_Bohr2']
    kg2 = ana['per_h']['0.002']['kg_Eh_Bohr2']
    R['T1'] = dict(a0=ana['a0'], a0_expect=float(g0 @ q),
                   slope1=ana['per_h']['0.001']['energy_slope_Eh_Bohr'],
                   kE1=kE1, kE2=kE2, kg1=kg1, kg2=kg2, kH_true=kH_true,
                   err_max=float(max(abs(kE1 - kH_true), abs(kE2 - kH_true),
                                     abs(kg1 - kH_true), abs(kg2 - kH_true))),
                   a0_err=float(abs(ana['a0'] - float(g0 @ q))),
                   slope_err=float(abs(
                       ana['per_h']['0.001']['energy_slope_Eh_Bohr']
                       - float(g0 @ q))))
    R['T1_pass'] = bool(R['T1']['err_max'] < 1e-6
                        and R['T1']['a0_err'] < 1e-12
                        and R['T1']['slope_err'] < 1e-8
                        and ana['a0'] != 0.0
                        and not ana['per_h']['0.002']['near_resolution'])
    d = np.asarray(man['direction']['d_raw'], float).reshape(-1)
    lam0_mock = float(d @ H @ d)
    R['T2'] = dict(kH_sym=ana['kH']['qTHq_H_sym'],
                   lam0_over_norm2=lam0_mock / float(d @ d))
    R['T2_pass'] = bool(abs(ana['kH']['qTHq_H_sym']
                            - lam0_mock / float(d @ d)) < 1e-10)

    # ---- T3: flow structure ---------------------------------------------
    n_centre = sum(1 for a in led.data['attempts']
                   if a['category'] == 'centre')
    n_disp = sum(1 for a in led.data['attempts']
                 if a['category'] == 'displacement')
    order = [a['note']['tag'] for a in led.data['attempts']
             if a['category'] == 'displacement']
    all_status = set(a['status'] for a in led.data['attempts'])
    R['T3'] = dict(n_centre=n_centre, n_disp=n_disp, order=order,
                   all_done=all_status == {'done'},
                   n_records=len(flow['disp_records']))
    R['T3_pass'] = bool(n_centre == 1 and n_disp == 4
                        and order == ['disp_p001', 'disp_m001',
                                      'disp_p002', 'disp_m002']
                        and all_status == {'done'}
                        and len(flow['disp_records']) == 4
                        and all(os.path.exists(
                            os.path.join(tmp, 'eval_%s.json' % t))
                            for t in ['centre'] + order))

    # ---- T4a: 6th displacement refused -----------------------------------
    err = None
    try:
        led.pre('displacement')
    except RuntimeError as e:
        err = str(e)
    R['T4a'] = dict(error=err)
    R['T4a_pass'] = bool(err and 'cap reached' in err)

    # ---- T4b: centre non-convergence -> stop, 0 disp ---------------------
    tmp = tempfile.mkdtemp(prefix='job059_T4b_')
    led = j59.Ledger(os.path.join(tmp, 'b.json'))
    err = None
    try:
        j59.run_batch(man, led,
                      mock_eval_factory(cfg, H, q, x0, g0,
                                        fail_conv_tags=('centre',)), tmp)
    except RuntimeError as e:
        err = str(e)
    n_disp_b = sum(1 for a in led.data['attempts']
                   if a['category'] == 'displacement')
    R['T4b'] = dict(error=err[:60] if err else None, n_disp=n_disp_b,
                    att0=led.data['attempts'][0]['status'])
    R['T4b_pass'] = bool(err and 'not converged' in err
                         and n_disp_b == 0
                         and led.data['attempts'][0]['status'] == 'error')

    # ---- T4c: displacement exception -> error attempt + stop -------------
    tmp = tempfile.mkdtemp(prefix='job059_T4c_')
    led = j59.Ledger(os.path.join(tmp, 'b.json'))
    err = None
    try:
        j59.run_batch(man, led,
                      mock_eval_factory(cfg, H, q, x0, g0,
                                        raise_tags=('disp_p002',)), tmp)
    except RuntimeError as e:
        err = str(e)
    n_att = len(led.data['attempts'])
    n_err = sum(1 for a in led.data['attempts']
                if a['status'] == 'error')
    R['T4c'] = dict(error=err[:60] if err else None, n_attempts=n_att,
                    n_err=n_err)
    R['T4c_pass'] = bool(err and 'mock save exception' in err
                         and n_att == 4 and n_err == 1)

    # ---- T5: centre gate failure -> 0 disp -------------------------------
    tmp = tempfile.mkdtemp(prefix='job059_T5_')
    led = j59.Ledger(os.path.join(tmp, 'b.json'))
    man_bad = make_man()
    man_bad['centre']['e_total'] = -1.0
    err = None
    try:
        j59.run_batch(man_bad, led,
                      mock_eval_factory(man_bad['centre']['config'], H, q,
                                        x0, g0), tmp)
    except RuntimeError as e:
        err = str(e)
    n_disp_d = sum(1 for a in led.data['attempts']
                   if a['category'] == 'displacement')
    R['T5'] = dict(error=err[:60] if err else None, n_disp=n_disp_d)
    R['T5_pass'] = bool(err and 'gate failed' in err and n_disp_d == 0)

    # ---- T6: direction conventions ---------------------------------------
    qq = np.asarray(man['direction']['q'], float).reshape(-1)
    d_raw = np.asarray(json.load(open(os.path.join(
        OUT, 'input_manifest.json')))['direction']['d_raw'], float)
    sha_a = hashlib.sha256(qq.tobytes()).hexdigest()[:16]
    sha_man = json.load(open(os.path.join(
        OUT, 'input_manifest.json')))['direction']['q_sha']
    imax = int(np.argmax(np.abs(qq)))
    R['T6'] = dict(q_norm=float(np.linalg.norm(qq)),
                   largest_positive=bool(qq[imax] > 0),
                   sha_stable=bool(sha_a == sha_man),
                   no_double_mass_div=bool(abs(
                       np.linalg.norm(qq) - 1.0) < 1e-12),
                   raw_norm_gt1=bool(np.linalg.norm(d_raw) > 1.0
                                     or np.linalg.norm(d_raw) < 1.0))
    R['T6_pass'] = bool(abs(R['T6']['q_norm'] - 1.0) < 1e-12
                        and R['T6']['largest_positive']
                        and R['T6']['sha_stable']
                        and R['T6']['no_double_mass_div']
                        and R['T6']['raw_norm_gt1'])

    R['formal_dirs_untouched'] = bool(pre58 == snapshot(R58R)
                                      and pre59 == snapshot(OUT))
    keys = ('T1_pass', 'T2_pass', 'T3_pass', 'T4a_pass', 'T4b_pass',
            'T4c_pass', 'T5_pass', 'T6_pass', 'formal_dirs_untouched')
    R['ALL_PASS'] = bool(all(R[k] for k in keys))
    print(json.dumps(R, indent=1, default=str))
    ex.save_json_atomic(os.path.join(OUT, 'mock_test_results.json'), R)
    if not R['ALL_PASS']:
        raise SystemExit('MOCK TESTS FAILED')
    print('MOCK TESTS ALL PASS', flush=True)


if __name__ == '__main__':
    main()
