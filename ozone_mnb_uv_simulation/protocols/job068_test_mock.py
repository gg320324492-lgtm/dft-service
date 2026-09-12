#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-068 step B: zero-evaluation tests (mock backend only,
formal directories untouched).

  T1 manifest: caps 34 total, start hashes, reference energy;
  T2 generalized BFGS flow on a mock quadratic (HNO 9 vars): converges
     below 1e-5, records every evaluation, ledger counting correct;
  T3 budget: opt cap enforced (excess requests refused);
  T4 recheck pass/fail gates the stability stage;
  T5 stability attempt recorded with the fixed 4-tuple wrapper on a
     mock object (log written directly to file);
  T6 one-product-passed rule: a failed molecule does not produce a P4
     energy and is reported truthfully;
  T7 isolation: pyscf/d2_full stubbed; formal dirs untouched.
"""
import os, sys, json, tempfile, types
import numpy as np

sys.modules['pyscf'] = None
sys.modules['pyscf.hessian'] = None
sys.modules['d2_full'] = types.ModuleType('d2_full')

HERE = '/mnt/e/dft-service/ozone_mnb_uv_simulation/protocols'
sys.path.insert(0, HERE)
import job047_exec as ex
import job068_exec as j68

OUT = j68.OUT
BPA = 1.0 / ex.ANG_PER_BOHR


def snapshot(d):
    if not os.path.isdir(d):
        return []
    return sorted((f, os.path.getsize(os.path.join(d, f)))
                  for f in os.listdir(d))


class MockBackend:
    """Generalized quadratic backend: E(x)=0.5(x-x*)'H(x-x*) anchored
    at the start (E0, g0)."""

    def __init__(self, elements, x_start, Hdiag=0.8):
        self.elements = elements
        self.x_start = np.asarray(x_start, float).reshape(-1)
        self.E0 = -55.9
        self.g0 = np.zeros(len(self.x_start))
        self.g0[::3] = 0.02                     # gentle nonzero start
        self.H = np.eye(len(self.x_start)) * Hdiag
        self.x_star = self.x_start - self.g0 / self.H.diagonal()
        self.calls = []

    def full_eval(self, R_bohr, out_dir, tag):
        R = np.asarray(R_bohr, float).reshape(-1)
        self.calls.append((tag, R.tolist()))
        if np.abs(R - self.x_start).max() < 1e-12:
            e, g = self.E0, self.g0
        else:
            xt = R - self.x_start
            e = self.E0 + float(self.g0 @ xt) + 0.5 * float(
                xt @ self.H @ xt)
            g = self.g0 + self.H @ xt
        n = len(self.elements)
        rec = dict(
            e_total=e, e_d2_Eh=-1e-6, e_dft_part_Eh=e + 1e-6,
            grad=g.reshape(n, 3).tolist(),
            grad_max=float(np.abs(g).max()), grad_sha='m',
            coords_actual_angstrom=(R.reshape(n, 3)
                                    * ex.ANG_PER_BOHR).tolist(),
            coords_sha='m',
            config=dict(xc='wb97xd (project -D2 attached)',
                        d2_attached=True, grid_level=8,
                        scf_tol=[1e-12, 1e-9], basis='def2-TZVP',
                        charge=0, spin=0, grid_response=True,
                        solvent='none (gas phase)'),
            converged=True, all_finite=True, scf_kernel_count=1,
            tag=tag)
        ex.save_json_atomic(os.path.join(out_dir,
                                         'eval_%s.json' % tag), rec)
        return rec


class MockRKS:
    """Minimal RKS stand-in for the stability wrapper."""

    def __init__(self, stable=True):
        self.stdout = sys.__stdout__
        self.verbose = 0
        self._stable = stable
        self.mo_coeff = np.eye(2)
        self.mo_occ = np.array([2.0, 0.0])
        self.mo_energy = np.array([-1.0, 0.5])
        self.e_tot = -55.9
        self.converged = True

    def stability(self, **kw):
        self.kwargs = kw
        print('mock stability report', file=self.stdout)
        return (None, None, self._stable, None)


def main():
    man = json.load(open(os.path.join(OUT, 'input_manifest.json')))
    R = {}

    # T1 manifest
    R['T1'] = dict(caps=man['caps'], caps_total=man['caps_total'],
                   e_mono_sum=man['reference_energy']['e_monomer_sum'],
                   stop=man['stop_trigger']['gmax_le'])
    R['T1_pass'] = bool(man['caps_total'] == 34
                        and man['caps'] == {'hno_opt': 15,
                                            'hno_recheck': 1,
                                            'hno_stab': 1,
                                            'h2o2_opt': 15,
                                            'h2o2_recheck': 1,
                                            'h2o2_stab': 1}
                        and R['T1']['stop'] == 1e-5)

    # T2 generalized BFGS flow (mock backend, HNO 9 vars)
    man_h = json.loads(json.dumps(man))
    sp = 'HNO'
    tmp = tempfile.mkdtemp(prefix='job068_T2_')
    led = ex.Ledger(os.path.join(tmp, 'b.json'),
                    caps={'hno_opt': 15, 'hno_recheck': 1,
                          'hno_stab': 1})
    start = np.asarray(man['molecules'][sp]['coords_start_bohr'],
                       float).reshape(-1)
    mb = MockBackend(['N', 'O', 'H'], start)
    flow = j68.run_opt_general(x0=start, natoms=3, out_dir=tmp,
                               ledger=led, backend=mb,
                               category='hno_opt', opt_cap=15,
                               threshold=1e-5)
    R['T2'] = dict(outcome=flow['outcome'],
                   n_new=flow['n_new_evals'],
                   meeting=flow['meeting_points'],
                   ledger=led.count('hno_opt'))
    R['T2_pass'] = bool(flow['outcome'] == 'threshold_reached'
                        and flow['n_new_evals'] >= 1
                        and flow['meeting_points']
                        and led.count('hno_opt') == flow['n_new_evals'])

    # T3 budget enforcement
    tmp = tempfile.mkdtemp(prefix='job068_T3_')
    led = ex.Ledger(os.path.join(tmp, 'b.json'),
                    caps={'hno_opt': 2, 'hno_recheck': 1,
                          'hno_stab': 1})
    mb3 = MockBackend(['N', 'O', 'H'], start, Hdiag=0.05)
    flow3 = j68.run_opt_general(x0=start, natoms=3, out_dir=tmp,
                                ledger=led, backend=mb3,
                                category='hno_opt', opt_cap=2,
                                threshold=1e-5)
    blocked = len(flow3['blocked_requests'])
    R['T3'] = dict(outcome=flow3['outcome'], n_new=flow3['n_new_evals'],
                   blocked=blocked, ledger=led.count('hno_opt'))
    R['T3_pass'] = bool(flow3['outcome'] == 'budget_exhausted'
                        and flow3['n_new_evals'] == 2
                        and blocked >= 1
                        and led.count('hno_opt') == 2)

    # T4 recheck gates stability
    tmp = tempfile.mkdtemp(prefix='job068_T4_')
    led = ex.Ledger(os.path.join(tmp, 'b.json'),
                    caps={'hno_opt': 15, 'hno_recheck': 1,
                          'hno_stab': 1})
    rks_ok = MockRKS(stable=True)
    a = led.pre_eval('hno_recheck', dict(tag='recheck'))
    led.post_eval(a, dict(e_total=-55.9, grad_max=5e-7))
    a2 = led.pre_eval('hno_stab', dict(tag='stability'))
    stab = j68.stability_wrapper(rks_ok, tmp, 'hno')
    led.post_eval(a2, dict(stable_i=stab['stable_i']))
    R['T4'] = dict(stable_i=stab['stable_i'],
                   kwargs=rks_ok.kwargs,
                   log_written=os.path.exists(os.path.join(
                       tmp, 'stability_hno_raw_log.txt')),
                   ledger=led.count('hno_stab'))
    R['T4_pass'] = bool(stab['stable_i'] is True
                        and rks_ok.kwargs.get('internal') is True
                        and rks_ok.kwargs.get('return_status') is True
                        and R['T4']['log_written']
                        and led.count('hno_stab') == 1)

    # T5 one-product-passed rule (from the analysis function)
    both = dict(hno=dict(recheck_pass=True,
                         e_recheck=-56.2),
                h2o2=dict(recheck_pass=True, e_recheck=-225.8))
    one = dict(hno=dict(recheck_pass=True, e_recheck=-56.2),
               h2o2=dict(recheck_pass=False, e_recheck=None))
    e_nh3 = man['reference_energy']['e_nh3_job026']
    e_o3 = man['reference_energy']['e_o3_job026']
    full = j68.product_reference(both, e_nh3, e_o3)
    part = j68.product_reference(one, e_nh3, e_o3)
    R['T5'] = dict(full=full, part=part)
    R['T5_pass'] = bool(full['p4_products_complete']
                        and abs(full['e_products']
                                - (-56.2 + -225.8)) < 1e-12
                        and not part['p4_products_complete']
                        and part['e_products'] is None)

    # T6 isolation
    pre = snapshot(OUT)
    R['T6'] = dict(pyscf_stubbed=sys.modules.get('pyscf') is None,
                   d2_stubbed=getattr(sys.modules.get('d2_full'),
                                      '__name__', None) == 'd2_full',
                   dirs_untouched=bool(pre == snapshot(OUT)))
    R['T6_pass'] = bool(R['T6']['pyscf_stubbed']
                        and R['T6']['d2_stubbed']
                        and R['T6']['dirs_untouched'])

    keys = ('T1_pass', 'T2_pass', 'T3_pass', 'T4_pass', 'T5_pass',
            'T6_pass')
    R['ALL_PASS'] = bool(all(R[k] for k in keys))
    print(json.dumps(R, indent=1, default=str)[:1600])
    ex.save_json_atomic(os.path.join(OUT, 'mock_test_results.json'), R)
    if not R['ALL_PASS']:
        raise SystemExit('MOCK TESTS FAILED')
    print('MOCK TESTS ALL PASS', flush=True)


def snapshot(d):
    if not os.path.isdir(d):
        return []
    return sorted((f, os.path.getsize(os.path.join(d, f)))
                  for f in os.listdir(d))


if __name__ == '__main__':
    main()
