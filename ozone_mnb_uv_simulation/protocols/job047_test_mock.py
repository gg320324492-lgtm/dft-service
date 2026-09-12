#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-047 step B: mock-backend tests (temp dir, no pyscf).

Exercises the IDENTICAL pipeline functions (eval_point / select_branch /
run_optimization / recheck gate) used by the real run:
  S1  both sides lower, energies near-equal -> tie rule picks R+; branch
      optimization reaches the gradient gate -> recheck -> candidate
  S2  single side lowers -> that branch optimized -> recheck -> candidate
  S3  displacement point already meets max|g|<=1e-5 -> NO optimization,
      direct recheck (charter 5)
  S4  no side lowers -> save & stop, no expansion
  S5  opt cap reached -> BudgetExhausted -> not converged; caps are not
      borrowed across categories
  S6  centre reproduction gate failure -> hard stop
Plus: start-point reuse (no repeated SCF), ledger accounting, formal-dir
sentinel.
"""
import os, sys, json, tempfile, shutil
import numpy as np

for _m in ('pyscf', 'd2_full', 'grad_factory'):
    sys.modules[_m] = None          # real backend unreachable in tests

HERE = '/mnt/e/dft-service/ozone_mnb_uv_simulation/protocols'
sys.path.insert(0, HERE)
import job047_exec as ex

FORMAL_OUT = ex.OUT
SYMS = ex.SYMS
BPA = 1.0 / ex.ANG_PER_BOHR

R0_BOHR = (np.arange(21, dtype=float).reshape(7, 3) * 0.05 + 1.0)
Q = np.asarray(json.load(open(os.path.join(
    ex.S45, 'input_manifest.json')))['direction']['q_vector'], float)
QF = Q.reshape(-1)


class MockBackend:
    """1-D potential along q: E(s) = E0 + A s + B s^2 + C s^4,
    s = (R - R0) . q ; gradient g = (dE/ds) q."""

    def __init__(self, E0, A, B, C):
        self.E0, self.A, self.B, self.C = E0, A, B, C
        self.scf_calls = 0
        self.cfg = dict(xc='mock', d2_attached=True, grid_level=8,
                        scf_tol=[1e-12, 1e-9], basis='mock', charge=0,
                        spin=0, grid_response=True, solvent='none')

    def full_eval(self, R_bohr, out_dir, tag):
        R = np.asarray(R_bohr, float)
        s = float(np.dot(R.reshape(-1) - R0_BOHR.reshape(-1), QF))
        e = self.E0 + self.A * s + self.B * s * s + self.C * s ** 4
        de = self.A + 2 * self.B * s + 4 * self.C * s ** 3
        g = (de * QF).reshape(7, 3)
        self.scf_calls += 1
        C_act = R * ex.ANG_PER_BOHR
        return dict(e_total=float(e), e_d2_Eh=0.0, e_dft_part_Eh=float(e),
                    grad=g.tolist(), grad_max=float(np.abs(g).max()),
                    grad_sha='mock', coords_actual_angstrom=C_act.tolist(),
                    coords_sha='mock', config=dict(self.cfg),
                    chkfile=dict(path='mock.chk', exists=True, size=0),
                    mo_summary=dict(n_mo=100, nocc=17),
                    converged=True, all_finite=True, scf_kernel_count=1)


def mock_centre_ref(backend):
    return dict(e_total=backend.E0,
                grad=(backend.A * QF).reshape(7, 3).tolist(),
                coords_actual_angstrom=(R0_BOHR * ex.ANG_PER_BOHR).tolist(),
                config=dict(backend.cfg), converged=True)


def snapshot(d):
    if not os.path.isdir(d):
        return []
    return sorted((f, os.path.getsize(os.path.join(d, f)))
                  for f in os.listdir(d))


def run_scenario(name, A, B, C, opt_cap=None, expect_error=False):
    tmp = tempfile.mkdtemp(prefix='job047_%s_' % name)
    caps = dict(ex.Ledger.DEFAULT_CAPS)
    if opt_cap is not None:
        caps['opt'] = opt_cap
    ledger = ex.Ledger(os.path.join(tmp, 'budget.json'), caps=caps)
    be = MockBackend(-282.0, A, B, C)
    ref = mock_centre_ref(be)
    out = dict(temp=tmp)
    try:
        rec_c = ex.eval_point('centre', R0_BOHR, tmp, ledger, be, 'centre',
                              gate_fn=ex.centre_gate_fn(ref))
        R_p = R0_BOHR + 0.10 * Q.reshape(7, 3)
        R_m = R0_BOHR - 0.10 * Q.reshape(7, 3)
        rec_p = ex.eval_point('disp_p10', R_p, tmp, ledger, be, 'disp')
        rec_m = ex.eval_point('disp_m10', R_m, tmp, ledger, be, 'disp')
        chosen, info = ex.select_branch(rec_c, rec_p, rec_m)
        out['chosen'] = chosen
        out['decision'] = info['decision']
        out['near_equal'] = info.get('near_equal')
        out['dE_plus'] = info['dE_plus']
        out['dE_minus'] = info['dE_minus']
        out['disp_gmax_p'] = rec_p['grad_max']
        out['disp_gmax_m'] = rec_m['grad_max']
        if chosen is None:
            out['status'] = 'stopped_no_side_lowered'
            out['attempts'] = {c: ledger.count(c)
                               for c in ('centre', 'disp', 'opt', 'recheck')}
            out['scf_calls'] = be.scf_calls
            return out, ledger, be
        chosen_rec = rec_p if chosen == 'p10' else rec_m
        opt = ex.run_optimization(chosen_rec, tmp, ledger, be,
                                  opt_cap=(opt_cap or ex.OPT_CAP))
        out['outcome'] = opt['outcome']
        out['n_new_evals'] = opt['n_new_evals']
        out['reuse_start'] = opt['reuse_start']
        out['meeting'] = opt['meeting_points']
        if opt['meeting_points']:
            mp = opt['meeting_points'][0]
            mrec = json.load(open('%s/eval_%s_%s.json'
                                  % (tmp, mp['category'], mp['tag'])))
            rr = ex.eval_point('recheck', np.asarray(
                mrec['coords_actual_angstrom'], float) * BPA,
                tmp, ledger, be, 'recheck',
                gate_fn=ex.recheck_gate_fn(mrec))
            out['recheck_pass'] = bool(rr['gate']['gates_pass'])
            out['status'] = ('stationary_point_candidate'
                             if rr['gate']['gates_pass']
                             else 'recheck_failed')
        else:
            out['status'] = 'not_converged'
            out['recheck_pass'] = None
    except RuntimeError as e:
        out['error'] = str(e)
        out['status'] = 'hard_stop'
    out['attempts'] = {c: ledger.count(c)
                       for c in ('centre', 'disp', 'opt', 'recheck')}
    out['scf_calls'] = be.scf_calls
    return out, ledger, be


def main():
    pre_snap = snapshot(FORMAL_OUT)
    R = {}

    # S1: both lower (symmetric) -> tie -> prefer positive -> optimize
    o1, l1, be1 = run_scenario('S1', A=0.0, B=-0.004, C=0.25)
    R['S1'] = dict(chosen=o1['chosen'], decision=o1['decision'],
                   near_equal=o1['near_equal'], outcome=o1.get('outcome'),
                   status=o1.get('status'), reuse=o1.get('reuse_start'),
                   n_new=o1.get('n_new_evals'), attempts=o1['attempts'])
    R['S1_pass'] = bool(o1['chosen'] == 'p10'
                        and o1['decision'] == 'tie_rule_prefer_positive'
                        and o1['near_equal'] is True
                        and o1.get('status') == 'stationary_point_candidate'
                        and o1.get('reuse_start') is True
                        and o1['attempts']['opt'] <= 20
                        and o1['attempts']['recheck'] == 1
                        and be1.scf_calls == 4 + o1['attempts']['opt'])

    # S2: single side lowers -> m10
    o2, l2, be2 = run_scenario('S2', A=1.5e-4, B=-0.004, C=0.25)
    R['S2'] = dict(chosen=o2['chosen'], decision=o2['decision'],
                   status=o2.get('status'), attempts=o2['attempts'])
    R['S2_pass'] = bool(o2['chosen'] == 'm10'
                        and o2['decision'] == 'single_lowering_side'
                        and o2.get('status') == 'stationary_point_candidate')

    # S3: displacement point already meets the gate -> no optimization
    o3, l3, be3 = run_scenario('S3', A=0.0, B=-0.005, C=0.25)
    R['S3'] = dict(chosen=o3['chosen'], outcome=o3.get('outcome'),
                   status=o3.get('status'), attempts=o3['attempts'],
                   disp_gmax_p=o3['disp_gmax_p'])
    R['S3_pass'] = bool(o3['chosen'] == 'p10'
                        and o3.get('outcome') == 'start_already_meets_gate'
                        and o3['attempts']['opt'] == 0
                        and o3.get('status') == 'stationary_point_candidate')

    # S4: no side lowers -> stop
    o4, l4, be4 = run_scenario('S4', A=0.0, B=5e-7, C=0.0)
    R['S4'] = dict(decision=o4['decision'], status=o4.get('status'),
                   attempts=o4['attempts'])
    R['S4_pass'] = bool(o4['decision'] == 'stop_no_side_lowered'
                        and o4.get('status') == 'stopped_no_side_lowered'
                        and o4['attempts']['opt'] == 0
                        and o4['attempts']['recheck'] == 0)

    # S5: opt cap exhausted -> not converged; caps not borrowed
    o5, l5, be5 = run_scenario('S5', A=0.0, B=-0.004, C=0.25, opt_cap=1)
    R['S5'] = dict(outcome=o5.get('outcome'), status=o5.get('status'),
                   attempts=o5['attempts'])
    cap_ok = True
    try:
        l5.pre_eval('opt', dict(note='should raise'))
        cap_ok = False
    except RuntimeError:
        pass
    R['S5_pass'] = bool(o5.get('outcome') == 'budget_exhausted'
                        and o5.get('status') == 'not_converged'
                        and o5['attempts']['recheck'] == 0
                        and cap_ok)

    # S6: centre gate failure -> hard stop
    tmp6 = tempfile.mkdtemp(prefix='job047_S6_')
    be6 = MockBackend(-282.0, 0.0, -0.004, 0.25)
    ref6 = mock_centre_ref(be6)
    ref6['e_total'] = ref6['e_total'] + 1e-3     # mismatch beyond gate
    led6 = ex.Ledger(os.path.join(tmp6, 'budget.json'))
    err6 = None
    try:
        ex.eval_point('centre', R0_BOHR, tmp6, led6, be6, 'centre',
                      gate_fn=ex.centre_gate_fn(ref6))
    except RuntimeError as e:
        err6 = str(e)
    R['S6'] = dict(error=(err6 or '')[:80],
                   centre_status=led6.data['attempts'][0]['status'])
    R['S6_pass'] = bool(err6 and 'gate failed' in err6
                        and led6.data['attempts'][0]['status'] == 'error')

    R['formal_dir_untouched'] = bool(pre_snap == snapshot(FORMAL_OUT))
    R['ALL_PASS'] = bool(all(R[k] for k in ('S1_pass', 'S2_pass', 'S3_pass',
                                            'S4_pass', 'S5_pass', 'S6_pass',
                                            'formal_dir_untouched')))
    print(json.dumps(R, indent=1, default=str))
    ex.save_json_atomic(os.path.join(FORMAL_OUT, 'mock_test_results.json'),
                        R)
    if not R['ALL_PASS']:
        raise SystemExit('MOCK TESTS FAILED')
    print('MOCK TESTS ALL PASS', flush=True)


if __name__ == '__main__':
    main()
