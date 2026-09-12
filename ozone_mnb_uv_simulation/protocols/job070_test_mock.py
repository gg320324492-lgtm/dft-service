#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-070 step B: zero-evaluation tests (mock backend only,
formal directories untouched).

  T1 manifest: caps 34 (2 radicals x [opt 15 + recheck 1 + stab 1]),
     spin=1 UKS pinned, <S2> reference present, S13 qualitative
     comparison block present, template hashes, number check;
  T2 opt flow on a mock quadratic UKS backend (HOO 9 vars): threshold
     stop, ledger counting, <S2> fields recorded per evaluation;
  T3 per-category independence: hoo_* and h2no_* counters are separate
     (no borrowing), caps enforced;
  T4 stability wrapper on a mock UKS object: 4-tuple contract, log
     written directly to file, kernel counted INSIDE the stab attempt
     (fresh converged object pattern);
  T5 recheck gate: centre gate refuses a drifted record;
  T6 P1 fragment rule: one radical failed -> NO fragment energy;
     both passed -> computed and labelled qualitative-only;
  T7 isolation: formal dirs untouched.
"""
import os, sys, json, tempfile, types
import numpy as np

sys.modules['pyscf'] = None
sys.modules['pyscf.data'] = None
sys.modules['d2_full'] = types.ModuleType('d2_full')

HERE = '/mnt/e/dft-service/ozone_mnb_uv_simulation/protocols'
sys.path.insert(0, HERE)
import job047_exec as ex
import job068_exec as j68
import job070_exec as j70

OUT = j70.OUT
R = {}


def snapshot(d):
    if not os.path.isdir(d):
        return []
    return sorted((f, os.path.getsize(os.path.join(d, f)))
                  for f in os.listdir(d))


class MockUKSBackend:
    """Quadratic UKS stand-in: E(x)=0.5(x-x*)'H(x-x*), records <S2>."""

    def __init__(self, elements, Hdiag=0.8, s2=0.76):
        self.elements = list(elements)
        n3 = 3 * len(elements)
        self.E0 = -55.9
        self.g0 = np.zeros(n3)
        self.g0[::3] = 0.02
        self.H = np.eye(n3) * Hdiag
        self.x_start = np.zeros(n3)
        self.x_star = self.x_start - self.g0 / self.H.diagonal()
        self.s2 = s2
        self.kernel_calls = 0

    def full_eval(self, R_bohr, out_dir, tag):
        x = np.asarray(R_bohr, float).reshape(-1) - self.x_star
        e = self.E0 + 0.5 * float(x @ self.H @ x)
        g = (self.H @ x).reshape(-1, 3)
        n = len(self.elements)
        rec = dict(
            e_total=e, e_d2_Eh=-1e-6, e_dft_part_Eh=e + 1e-6,
            grad=g.tolist(), grad_max=float(np.abs(g).max()),
            grad_sha='m',
            coords_actual_angstrom=(np.asarray(R_bohr, float)
                                    .reshape(n, 3)
                                    * ex.ANG_PER_BOHR).tolist(),
            coords_sha='m',
            config=dict(xc='wb97xd (project -D2 attached)',
                        d2_attached=True, grid_level=8,
                        scf_tol=[1e-12, 1e-9], basis='def2-TZVP',
                        charge=0, spin=1, uks=True,
                        grid_response=True,
                        solvent='none (gas phase)'),
            converged=True, all_finite=True, scf_kernel_count=1,
            s2_total=self.s2, mult_measured=1.752,
            s2_reference_doublet=0.75,
            s2_delta=self.s2 - 0.75, tag=tag)
        ex.save_json_atomic(os.path.join(out_dir,
                                         'eval_%s.json' % tag), rec)
        return rec

    def new_mf(self, R_bohr, out_dir, tag):
        return None, MockUKS(self.s2, self)


class MockUKS:
    """Minimal converged-UKS stand-in for the stability wrapper."""

    def __init__(self, s2, owner):
        self.stdout = sys.__stdout__
        self.verbose = 0
        self._s2 = s2
        self._owner = owner
        self.mo_coeff = np.eye(2)
        self.mo_occ = np.array([1.0, 0.0])
        self.mo_energy = np.array([-1.0, 0.5])
        self.e_tot = -55.9
        self.converged = True

    def kernel(self):
        self._owner.kernel_calls += 1
        return self.e_tot

    def spin_square(self):
        return (self._s2, 1.752)

    def stability(self, **kw):
        self.kwargs = kw
        print('mock ukS stability report', file=self.stdout)
        return (None, None, True, None)


def main():
    man = json.load(open(os.path.join(OUT, 'input_manifest.json')))
    before = snapshot(OUT)

    # ---------------- T1 manifest ----------------
    R['T1'] = dict(caps=man['caps'], caps_total=man['caps_total'],
                   spin=man['method']['spin'],
                   uks=man['method']['uks'],
                   s2ref=man['electron_state'][
                       's2_reference_doublet'],
                   s13=man['p1_fragment_energy'][
                       'qualitative_comparison'][
                       'product_vs_R_ccsdt_kcalmol'],
                   num=man['number_check']['pass_'])
    R['T1_pass'] = bool(
        man['caps_total'] == 34
        and man['caps'] == {'hoo_opt': 15, 'hoo_recheck': 1,
                            'hoo_stab': 1, 'h2no_opt': 15,
                            'h2no_recheck': 1, 'h2no_stab': 1}
        and R['T1']['spin'] == 1 and R['T1']['uks'] is True
        and R['T1']['s2ref'] == 0.75 and R['T1']['s13'] == -5.82
        and R['T1']['num'])

    # ---------------- T2 opt flow (mock UKS, HOO) ----------------
    tmp2 = tempfile.mkdtemp(prefix='job070_T2_')
    led2 = ex.Ledger(os.path.join(tmp2, 'b.json'),
                     caps={'hoo_opt': 15, 'hoo_recheck': 1,
                           'hoo_stab': 1})
    mb = MockUKSBackend(['O', 'O', 'H'])
    mb.x_start = np.zeros(9)
    flow = j68.run_opt_general(x0=mb.x_start + 0.05, natoms=3,
                               out_dir=tmp2, ledger=led2,
                               backend=mb, category='hoo_opt',
                               opt_cap=15, threshold=1e-5)
    recs = [json.load(open(os.path.join(tmp2, f)))
            for f in sorted(os.listdir(tmp2))
            if f.startswith('eval_hoo_opt_')]
    R['T2'] = dict(outcome=flow['outcome'], n_new=flow['n_new_evals'],
                   n_records=len(recs),
                   s2_recorded=all('s2_total' in r for r in recs),
                   counts=led2.count('hoo_opt'))
    R['T2_pass'] = bool(flow['outcome'] == 'threshold_reached'
                        and flow['n_new_evals'] >= 1
                        and len(recs) == flow['n_new_evals']
                        and R['T2']['s2_recorded']
                        and led2.count('hoo_opt')
                        == flow['n_new_evals'])

    # ---------------- T3 per-category independence ----------------
    tmp3 = tempfile.mkdtemp(prefix='job070_T3_')
    led3 = ex.Ledger(os.path.join(tmp3, 'b.json'),
                     caps={'hoo_opt': 1, 'h2no_opt': 1})
    a = led3.pre_eval('hoo_opt', dict(tag='a'))
    led3.post_eval(a, dict(e_total=-1.0))
    b = led3.pre_eval('h2no_opt', dict(tag='b'))
    led3.post_eval(b, dict(e_total=-1.0))
    refused = False
    try:
        led3.pre_eval('hoo_opt', dict(tag='x'))
    except RuntimeError:
        refused = True
    ok_h2no = False
    try:
        led3.pre_eval('h2no_opt', dict(tag='y'))
        ok_h2no = False          # cap also reached for h2no
    except RuntimeError:
        ok_h2no = True
    R['T3'] = dict(hoo_refused=refused, h2no_refused=ok_h2no,
                   counts={c: led3.count(c) for c in
                           ('hoo_opt', 'h2no_opt')})
    R['T3_pass'] = bool(refused and ok_h2no
                        and R['T3']['counts'] == {'hoo_opt': 1,
                                                  'h2no_opt': 1})

    # ---------------- T4 stability wrapper + kernel-inside ----------------
    tmp4 = tempfile.mkdtemp(prefix='job070_T4_')
    mb4 = MockUKSBackend(['N', 'O', 'H', 'H'])
    _, mf4 = mb4.new_mf(None, tmp4, 'stab_h2no')
    mf4.kernel()          # main() converges the fresh object BEFORE
    mf4.kernel()          # stability; two calls -> counter == 2
    stab = j68.stability_wrapper(mf4, tmp4, 'h2no')
    log_exists = os.path.isfile(os.path.join(
        tmp4, 'stability_h2no_raw_log.txt'))
    R['T4'] = dict(stable_i=stab['stable_i'],
                   stable_e=stab['stable_e'],
                   kernel_calls_inside=mb4.kernel_calls,
                   internal=stab and mf4.kwargs.get('internal'),
                   log_exists=log_exists,
                   order=stab['installed_order'])
    R['T4_pass'] = bool(stab['stable_i'] is True
                        and stab['stable_e'] is None
                        and mb4.kernel_calls == 2
                        and mf4.kwargs.get('internal') is True
                        and mf4.kwargs.get('external') is False
                        and log_exists)

    # ---------------- T5 recheck gate ----------------
    mrec = dict(grad=[[0.0, 0.0, 0.0]] * 3,
                coords_actual_angstrom=[[0.0, 0.0, 0.0],
                                        [1.3, 0.0, 0.0],
                                        [0.2, 0.9, 0.0]],
                e_total=-55.9,
                config=dict(xc='wb97xd (project -D2 attached)'))
    gate_fn = ex.centre_gate_fn(mrec)
    rec_same = dict(grad=[[0.0, 0.0, 0.0]] * 3,
                    coords_actual_angstrom=mrec[
                        'coords_actual_angstrom'],
                    e_total=-55.9 + 1e-13,
                    config=mrec['config'], converged=True,
                    all_finite=True)
    rec_drift = dict(rec_same, coords_actual_angstrom=[
        [0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.2, 0.9, 0.0]])
    R['T5'] = dict(same=gate_fn(rec_same)['gates_pass'],
                   drift=gate_fn(rec_drift)['gates_pass'])
    R['T5_pass'] = bool(R['T5']['same'] and not R['T5']['drift'])

    # ---------------- T6 P1 fragment rule ----------------
    good = dict(HOO=dict(recheck_pass=True, e_recheck=-75.6),
                H2NO=dict(recheck_pass=True, e_recheck=-130.2))
    bad = dict(HOO=dict(recheck_pass=True, e_recheck=-75.6),
               H2NO=dict(recheck_pass=False))
    e_mono = -281.99812183548585
    fr_good = j70.p1_fragment_reference(good, e_mono)
    fr_bad = j70.p1_fragment_reference(bad, e_mono)
    R['T6'] = dict(good_complete=fr_good['p1_fragments_complete'],
                   good_dE_kcal=fr_good['dE_P1frag_kcal_mol'],
                   bad_complete=fr_bad['p1_fragments_complete'],
                   bad_e=fr_bad['e_fragments'],
                   qualitative=('QUALITATIVE'
                                in str(fr_good.get(
                                    'interpretation', ''))))
    R['T6_pass'] = bool(fr_good['p1_fragments_complete']
                        and abs(fr_good['e_fragments']
                                - (-75.6 - 130.2)) < 1e-12
                        and fr_bad['p1_fragments_complete'] is False
                        and fr_bad['e_fragments'] is None
                        and R['T6']['qualitative'])

    # ---------------- T7 isolation ----------------
    after = snapshot(OUT)
    R['T7'] = dict(before=len(before), after=len(after))
    R['T7_pass'] = bool(len(after) == len(before))

    R['all_pass'] = all(bool(R['T%d_pass' % i]) for i in range(1, 8))
    ex.save_json_atomic(os.path.join(tempfile.gettempdir(),
                                     'job070_mock_results.json'), R)
    print(json.dumps({k: v for k, v in R.items()
                      if k.endswith('_pass') or k == 'all_pass'},
                     indent=1))
    if not R['all_pass']:
        raise RuntimeError('MOCK TESTS FAILED: see '
                           + tempfile.gettempdir()
                           + '/job070_mock_results.json')


if __name__ == '__main__':
    main()
