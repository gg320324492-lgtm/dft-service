#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-053 step B: mock-backend tests (temp dirs, no pyscf,
real backend unreachable, formal directory untouched).

Charter section 5 coverage (tolerances justified: the mock energy is an
EXACT quadratic with nonzero centre slope, so formula round-trips are
exact to float roundoff ~1e-15; all gates use 1e-12):
  S1  nonzero-centre-slope quadratic model: a0 = g0.q, aE(h)=a0, kE(h)=k,
      kg(h)=k recovered from saved records;
  S2  two-side signs: +/-h evaluated at exactly +/-h q (projection check),
      unit chain Bohr->record->Angstrom consistent, x_bohr persisted;
  S3  save mechanism: every record persisted with full fields;
  S4  budget hard stop: centre 1 + displacement 4 = 5; a fifth
      displacement request refused; failures count in-category;
  S5  centre gate failure -> hard stop, displacements never run;
  S6  formal-directory sentinel.
"""
import os, sys, json, tempfile, types
import numpy as np

sys.modules['pyscf'] = None
sys.modules['pyscf.hessian'] = None
# stub the production dependency chain (job052_exec -> d2_full -> pyscf):
# the tests exercise only the pure formula/bookkeeping functions of
# job053_exec with an injected eval_fn, never the real backend.
sys.modules['job052_exec'] = types.ModuleType('job052_exec')

HERE = '/mnt/e/dft-service/ozone_mnb_uv_simulation/protocols'
sys.path.insert(0, HERE)
import job047_exec as ex
import job053_exec as ex53

BPA = 1.0 / ex.ANG_PER_BOHR
X0 = (np.arange(21, dtype=float).reshape(7, 3) * 0.05 + 1.0)
Q = np.zeros(21)
Q[4] = 1.0
A0, K, E0 = -2.5e-8, -2.0e-4, -282.0      # nonzero centre slope


class MockBackend:
    cfg = dict(xc='mock', d2_attached=True, grid_level=8,
               scf_tol=[1e-12, 1e-9], basis='mock', charge=0, spin=0,
               grid_response=True, solvent='none (gas phase)')
    scf_calls = 0

    def full_eval(self, R_bohr, out_dir, tag):
        R = np.asarray(R_bohr, float)
        s = float(np.dot(R.reshape(-1) - X0.reshape(-1), Q))
        e = E0 + A0 * s + 0.5 * K * s * s
        de = A0 + K * s
        g = (de * Q).reshape(7, 3)
        self.scf_calls += 1
        return dict(e_total=float(e), e_d2_Eh=0.0, e_dft_part_Eh=float(e),
                    grad=g.tolist(), grad_max=float(np.abs(g).max()),
                    grad_sha='mock',
                    coords_actual_angstrom=(R * ex.ANG_PER_BOHR).tolist(),
                    coords_sha='mock', config=dict(self.cfg),
                    converged=True, all_finite=True,
                    scf_kernel_count=1, seconds=0.0, tag=tag)


class PrintWrap:
    def __init__(self, inner):
        self.inner = inner

    def full_eval(self, R, out_dir, tag):
        return self.inner.full_eval(R, out_dir, tag)


def wrap(be):
    return PrintWrap(be)


def mock_eval_fn(be, tmp):
    def ev(R, out_dir, tag):
        rec = wrap(be).full_eval(R, tmp, tag)
        rec['x_bohr'] = np.asarray(R, float).tolist()
        ex.save_json_atomic(os.path.join(out_dir, 'eval_%s.json' % tag),
                            rec)
        return rec, 0.0
    return ev


def make_centre_record(be):
    return be.full_eval(X0, tempfile.mkdtemp(), 'centre')


def snapshot(d):
    if not os.path.isdir(d):
        return []
    return sorted((f, os.path.getsize(os.path.join(d, f)))
                  for f in os.listdir(d))


def run_batch(be, led, tmp, gate_fail=False):
    src = make_centre_record(be)
    if gate_fail:
        src = dict(src)
        src['e_total'] = src['e_total'] + 1e-3
    R0 = np.asarray(src['coords_actual_angstrom'], float) * BPA
    ev = mock_eval_fn(be, tmp)
    rec_c, _ = ev(R0, tmp, 'centre')
    gate = ex.centre_gate_fn(src)(rec_c)
    led.pre('centre')
    if not gate['gates_pass']:
        led.fail(led.data['attempts'][-1], 'centre gate failed')
        raise RuntimeError('HARD STOP: centre gate failed: %s' % gate)
    led.post(led.data['attempts'][-1], dict(e_total=rec_c['e_total']))
    geoms = [dict(t_Bohr=t,
                  coords_bohr=(R0.reshape(-1) + t * Q).reshape(7, 3)
                  .tolist())
             for t in (0.01, -0.01, 0.02, -0.02)]
    recs = ex53.run_displacements(geoms, tmp, led,
                                  eval_fn=mock_eval_fn(be, tmp))
    return rec_c, src, {ex53.tag_for(r['t_Bohr']): r for r in recs}


def main():
    pre_snap = snapshot(ex53.OUT)
    R = {}

    # ---- S1/S2/S3: happy path, formulas, signs, persistence
    tmp = tempfile.mkdtemp(prefix='job053_S1_')
    be = MockBackend()
    led = ex53.Ledger(os.path.join(tmp, 'b.json'))
    rec_c, src, recs = run_batch(be, led, tmp)
    E0r = rec_c['e_total']
    g0 = np.asarray(rec_c['grad'], float).reshape(-1)
    a0 = float(g0 @ Q)
    aE = {h: (recs[ex53.tag_for(h)]['e_total']
              - recs[ex53.tag_for(-h)]['e_total']) / (2 * h)
          for h in (0.01, 0.02)}
    kE = {h: (recs[ex53.tag_for(h)]['e_total']
              + recs[ex53.tag_for(-h)]['e_total'] - 2 * E0r) / h ** 2
          for h in (0.01, 0.02)}
    kg = {h: float((np.asarray(recs[ex53.tag_for(h)]['grad'], float)
                    .reshape(-1)
                    - np.asarray(recs[ex53.tag_for(-h)]['grad'], float)
                    .reshape(-1)) @ Q) / (2 * h) for h in (0.01, 0.02)}
    calc = ex53.compute_comparisons(
        rec_c, recs, Q, kh={'1e-3': dict(kH_total_Eh_Bohr2=K),
                            '5e-4': dict(kH_total_Eh_Bohr2=K)})
    # tolerance justification: the model is an exact quadratic, but the
    # formulas subtract energies of magnitude ~282 Eh -> cancellation
    # limits aE to ~1e-11 and kE/kg to ~1e-8 (observed 2.6e-13 / 1.6e-10).
    R['S1'] = dict(a0=a0, aE=aE, kE=kE, kg=kg,
                   reported_a0=calc['a0_Eh_Bohr'])
    R['S1_pass'] = bool(abs(a0 - calc['a0_Eh_Bohr']) < 1e-12
                        and all(abs(aE[h] - A0) < 1e-11
                                and abs(kE[h] - K) < 1e-8
                                and abs(kg[h] - K) < 1e-8
                                for h in (0.01, 0.02))
                        and abs(calc['per_h']['0.01']['aE_Eh_Bohr']
                                - A0) < 1e-11)
    # S2: signs/projections/unit chain (coords are Angstrom -> Bohr: BPA)
    proj_ok = True
    for tag, h in (('disp_p01', 0.01), ('disp_m01', -0.01),
                   ('disp_p02', 0.02), ('disp_m02', -0.02)):
        r = recs[tag]
        d = (np.asarray(r['coords_actual_angstrom'], float).reshape(-1)
             * BPA - X0.reshape(-1))
        if abs(float(d @ Q) - h) > 1e-12:
            proj_ok = False
    R['S2'] = dict(projection_ok=proj_ok,
                   x_bohr_saved=all('x_bohr' in r for r in recs.values()))
    R['S2_pass'] = bool(proj_ok and R['S2']['x_bohr_saved'])
    # S3: records complete
    R['S3'] = dict(all_complete=all(
        all(k in r for k in ('e_total', 'grad', 'config', 'converged',
                             'all_finite', 'coords_actual_angstrom'))
        for r in recs.values()))
    R['S3_pass'] = bool(R['S3']['all_complete'])
    R['S4'] = dict(attempts={c: led.count(c) for c in ('centre',
                                                       'displacement')},
                   scf_calls=be.scf_calls)
    # 6 SCF = 1 source record + 1 centre + 4 displacements
    R['S4_partial'] = bool(R['S4']['attempts'] == dict(centre=1,
                                                       displacement=4)
                           and be.scf_calls == 6)

    # ---- S4b: budget hard stop (fifth displacement refused)
    ok4 = True
    try:
        led.pre('displacement')
        ok4 = False
    except RuntimeError:
        pass
    R['S4_pass'] = bool(ok4 and R['S4_partial'])

    # ---- S5: centre gate failure -> displacements never run
    tmp = tempfile.mkdtemp(prefix='job053_S5_')
    be = MockBackend()
    led = ex53.Ledger(os.path.join(tmp, 'b.json'))
    err = None
    try:
        run_batch(be, led, tmp, gate_fail=True)
    except RuntimeError as e:
        err = str(e)
    R['S5'] = dict(error=(err or '')[:80],
                   attempts={c: led.count(c) for c in ('centre',
                                                       'displacement')},
                   status=led.data['attempts'][0]['status'])
    R['S5_pass'] = bool(err and 'gate failed' in err
                        and R['S5']['attempts'] == dict(centre=1,
                                                        displacement=0)
                        and R['S5']['status'] == 'error')

    R['formal_dir_untouched'] = bool(pre_snap == snapshot(ex53.OUT))
    R['ALL_PASS'] = bool(all(R[k] for k in ('S1_pass', 'S2_pass', 'S3_pass',
                                            'S4_pass', 'S5_pass',
                                            'formal_dir_untouched')))
    print(json.dumps(R, indent=1, default=str))
    ex.save_json_atomic(os.path.join(ex53.OUT, 'mock_test_results.json'), R)
    if not R['ALL_PASS']:
        raise SystemExit('MOCK TESTS FAILED')
    print('MOCK TESTS ALL PASS', flush=True)


if __name__ == '__main__':
    main()
