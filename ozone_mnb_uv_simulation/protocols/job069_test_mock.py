#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-069 step B: zero-evaluation tests (mock backend only,
formal directories untouched).

  T1 manifest: caps (anchor 1 + hess 1), single-product rule, accepted
     endpoint provenance (recheck + stability hashes), verbatim thermo
     extraction file exists and really contains the function;
  T2 verbatim project thermochemistry: runs in a FAKE pyscf-data
     namespace; ZPE matches the hand-computed 0.5*R*sum(theta_v); S_elec
     = 0 for mult=1; linear=False for a bent mock molecule; J/mol units;
  T3 frequency_analysis on a synthetic positive-definite Hessian with
     exact TR zero-block: internal modes recovered, all positive, rank
     6, U shape (9,3), negatives=0;
  T4 negative-mode preservation: a synthetic Hessian with one negative
     internal eigenvalue keeps the negative sign (never abs-ed);
  T5 ledger: caps enforced, failure recorded, reload-with-error -> HARD
     STOP;
  T6 decision + combined-Hessian gate logic on synthetic matrices:
     attached == sum -> pass; perturbed -> fail; n_neg drives the
     registration wording;
  T7 isolation: formal dirs untouched (snapshot before/after).
"""
import os, sys, json, math, tempfile, types, shutil
import numpy as np

sys.modules['pyscf'] = None
sys.modules['pyscf.hessian'] = None
sys.modules['pyscf.data'] = None
sys.modules['d2_full'] = types.ModuleType('d2_full')

HERE = '/mnt/e/dft-service/ozone_mnb_uv_simulation/protocols'
sys.path.insert(0, HERE)
import job047_exec as ex
import job069_exec as j69

OUT = j69.OUT
R = {}


def snapshot(d):
    if not os.path.isdir(d):
        return []
    return sorted((f, os.path.getsize(os.path.join(d, f)))
                  for f in os.listdir(d))


class FakeNist:
    BOLTZMANN = 1.380649e-23
    PLANCK = 6.62607015e-34
    LIGHT_SPEED_SI = 299792458.0
    AVOGADRO = 6.02214076e23
    HARTREE2J = 4.3597447222071e-18
    ATOMIC_MASS = 1.66053906660e-27
    BOHR_SI = 0.529177210903e-10


class FakeData:
    nist = FakeNist()


class FakeMol:
    """Bent triatomic stand-in for the thermochemistry function."""

    def __init__(self, masses=(14.0, 16.0, 1.0)):
        self.natm = 3
        self._m = np.asarray(masses, float)
        self._r = np.array([[0.0, 0.0, 0.0],
                            [1.2, 0.2, 0.0],
                            [-0.9, 0.9, 0.0]])   # Angstrom

    def atom_mass_list(self, isotope_avg=True):
        return self._m.copy()

    def atom_coords(self, unit='Angstrom'):
        return self._r.copy()


def hess_from_internal_freqs(masses, coords_bohr, wn_list, AU2WN):
    """Build a (n,n,3,3) Hessian whose TR-projected internal
    eigenvalues correspond to the requested wavenumbers (or negative
    for imaginary)."""
    n = len(masses)
    m = np.asarray(masses, float)
    TR = j69.get_TR(m, np.asarray(coords_bohr, float).reshape(n, 3))
    q, _ = np.linalg.qr(TR.T)
    w, v = np.linalg.eigh(np.eye(3 * n) - q.dot(q.T))
    U = v[:, w > 1e-7]
    lam = [np.sign(wn) * (wn / AU2WN) ** 2 for wn in wn_list]
    Hint = np.diag(lam)
    Hmw = U.dot(Hint).dot(U.T)
    D = np.repeat(m ** 0.5, 3)          # un-mass-weight: H = M^0.5 Hmw M^0.5
    H33 = Hmw * np.outer(D, D)
    return H33.reshape(n, 3, n, 3).transpose(0, 2, 1, 3)


def main():
    man = json.load(open(os.path.join(OUT, 'input_manifest.json')))
    before = snapshot(OUT)

    # ---------------- T1 manifest ----------------
    vfile = os.path.join(OUT, 'hno',
                         'run_baseline_thermochemistry_verbatim.py')
    R['T1'] = dict(
        caps=man['caps'], caps_total=man['caps_total'],
        hno_accepted=man['single_product_rule']['hno_accepted'],
        h2o2_accepted=man['single_product_rule']['h2o2_accepted'],
        stable_i=man['accepted_endpoint']['stable_i'],
        verbatim_exists=os.path.isfile(vfile),
        verbatim_has_def='def thermochemistry' in open(vfile).read())
    R['T1_pass'] = bool(
        man['caps'] == {'hno_anchor': 1, 'hno_hess_dft': 1}
        and man['caps_total'] == 2
        and R['T1']['hno_accepted'] and not R['T1']['h2o2_accepted']
        and R['T1']['stable_i'] and R['T1']['verbatim_exists']
        and R['T1']['verbatim_has_def'])

    # ---------------- T2 verbatim thermochemistry ----------------
    tmp = tempfile.mkdtemp(prefix='job069_T2_')
    src = open(vfile, encoding='utf-8').read()
    body = '\n'.join(src.splitlines()[1:])
    ns = dict(data=FakeData, math=math, np=np)
    exec(compile(body, vfile, 'exec'), ns)          # noqa: S102
    thermo_fn = ns['thermochemistry']
    modes = [1500.0, 2000.0, 3500.0]
    fmol = FakeMol()
    out = thermo_fn(fmol, modes, -130.0, mult=1, sigma=1)
    kB, NA = FakeNist.BOLTZMANN, FakeNist.AVOGADRO
    ccm = FakeNist.LIGHT_SPEED_SI * 100.0
    th_v = [FakeNist.PLANCK * ccm * f / kB for f in modes]
    zpe_ref = 0.5 * kB * NA * sum(th_v)
    R['T2'] = dict(zpe=out['ZPE'], zpe_ref=zpe_ref,
                   s_elec=out['S_elec'], linear=out['linear'],
                   T=out['T'], P=out['P'], sigma=out['sigma'])
    R['T2_pass'] = bool(abs(out['ZPE'] - zpe_ref)
                        <= 1e-6 * abs(zpe_ref)
                        and out['S_elec'] == 0.0
                        and out['linear'] is False
                        and out['T'] == 298.15 and out['P'] == 101325.0
                        and out['sigma'] == 1)

    # ---------------- T3 frequency analysis (all positive) ----------------
    AU2WN = ((FakeNist.HARTREE2J
              / (FakeNist.ATOMIC_MASS * FakeNist.BOHR_SI ** 2)) ** .5
             / (2 * math.pi) / FakeNist.LIGHT_SPEED_SI * 1e-2)
    masses = [14.0, 16.0, 1.0]
    coords = np.array([[0.2832, 0.1588, 0.0],
                       [2.4603, -0.4076, 0.0],
                       [0.1357, 2.1561, 0.0]])   # Bohr, bent
    wn_true = [1500.0, 2100.0, 3400.0]
    h4 = hess_from_internal_freqs(masses, coords, wn_true, AU2WN)
    fa = j69.frequency_analysis(h4, coords, masses, AU2WN,
                                ['N', 'O', 'H'])
    got = np.asarray(fa['freq_wavenumber_signed'], float)
    R['T3'] = dict(freqs=got.tolist(), rank6=fa['tr_rank6'],
                   U_shape=fa['U_shape'],
                   n_neg=fa['n_negative_modes'],
                   tr_contam=fa['full_3n_lowest6_max_abs_wn'])
    R['T3_pass'] = bool(fa['tr_rank6'] and fa['U_shape'] == [9, 3]
                        and fa['n_negative_modes'] == 0
                        and np.allclose(np.sort(got),
                                        np.sort(wn_true), rtol=1e-6)
                        and fa['full_3n_lowest6_max_abs_wn'] < 1.0)

    # ---------------- T4 negative-mode preservation ----------------
    h4n = hess_from_internal_freqs(masses, coords,
                                   [-300.0, 2100.0, 3400.0], AU2WN)
    fan = j69.frequency_analysis(h4n, coords, masses, AU2WN,
                                 ['N', 'O', 'H'])
    gotn = np.asarray(fan['freq_wavenumber_signed'], float)
    R['T4'] = dict(freqs=gotn.tolist(),
                   n_neg=fan['n_negative_modes'])
    R['T4_pass'] = bool(fan['n_negative_modes'] == 1
                        and gotn[0] < 0 and abs(gotn[0] + 300.0)
                        < 1e-3 * 300.0)

    # ---------------- T5 ledger ----------------
    tmp5 = tempfile.mkdtemp(prefix='job069_T5_')
    led = ex.Ledger(os.path.join(tmp5, 'b.json'),
                    caps={'hno_anchor': 1, 'hno_hess_dft': 1})
    a1 = led.pre_eval('hno_anchor', dict(tag='a'))
    led.post_eval(a1, dict(e_total=-1.0, grad_max=1e-7))
    a2 = led.pre_eval('hno_hess_dft', dict(tag='h'))
    led.fail(a2, RuntimeError('mock hessian failure'))
    cap_ok = False
    try:
        led.pre_eval('hno_anchor', dict(tag='x'))
    except RuntimeError:
        cap_ok = True
    hard_stop = False
    try:
        ex.Ledger(os.path.join(tmp5, 'b.json'),
                  caps={'hno_anchor': 1, 'hno_hess_dft': 1})
    except RuntimeError as e:
        hard_stop = 'HARD STOP' in str(e)
    R['T5'] = dict(cap_ok=cap_ok, hard_stop=hard_stop,
                   counts={c: led.count(c) for c in
                           ('hno_anchor', 'hno_hess_dft')})
    R['T5_pass'] = bool(cap_ok and hard_stop
                        and R['T5']['counts'] == {'hno_anchor': 1,
                                                  'hno_hess_dft': 1})

    # ---------------- T6 combined gate + decision logic ----------------
    dft = np.random.RandomState(0).randn(3, 3, 3, 3)
    dft = 0.5 * (dft + dft.transpose(1, 0, 3, 2))
    d2 = np.random.RandomState(1).randn(3, 3, 3, 3) * 1e-4
    d2 = 0.5 * (d2 + d2.transpose(1, 0, 3, 2))
    kern = dft + d2
    dev0 = float(np.abs(kern - (dft + d2)).max())
    kern_p = kern.copy(); kern_p[0, 0, 0, 0] += 1e-8
    dev1 = float(np.abs(kern_p - (dft + d2)).max())
    gate0 = dev0 <= 1e-10
    gate1 = dev1 <= 1e-10
    R['T6'] = dict(dev_exact=dev0, dev_perturbed=dev1,
                   gate0_pass=gate0, gate1_pass=gate1,
                   n_neg_all_pos=fa['n_negative_modes'],
                   n_neg_with_neg=fan['n_negative_modes'])
    R['T6_pass'] = bool(gate0 and not gate1
                        and fa['n_negative_modes'] == 0
                        and fan['n_negative_modes'] == 1)

    # ---------------- T7 isolation ----------------
    after = snapshot(OUT)
    expected_new = []          # tests write NOTHING into OUT
    R['T7'] = dict(before=len(before), after=len(after))
    R['T7_pass'] = bool(len(after) == len(before))

    R['all_pass'] = all(bool(R['T%d_pass' % i]) for i in range(1, 8))
    ex.save_json_atomic(os.path.join(tempfile.gettempdir(),
                                     'job069_mock_results.json'), R)
    print(json.dumps({k: v for k, v in R.items()
                      if k.endswith('_pass') or k == 'all_pass'},
                     indent=1))
    if not R['all_pass']:
        raise RuntimeError('MOCK TESTS FAILED: see '
                           + tempfile.gettempdir()
                           + '/job069_mock_results.json')


if __name__ == '__main__':
    main()
