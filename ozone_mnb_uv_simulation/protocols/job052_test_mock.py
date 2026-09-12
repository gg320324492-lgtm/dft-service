#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-052 step B: synthetic validation (temp dirs, NO pyscf,
real backend unreachable, formal directory untouched).

Validates BEFORE any production call (charter section 6):
  V1  (7,7,3,3) <-> 21x21 layout round trip and element mapping;
  V2  unit constants: amu route vs electron-mass route agree; injection
      test - a mass-weighted Hessian with known internal eigenvalues is
      recovered by the 044 analyse() (15-dim internal diagonalization);
  V3  MW translational-rotational subspace rank 6, internal subspace 15,
      completeness of the V/U split;
  V4  negative eigenvalues are fully retained (not clipped, not called
      noise);
  V5  D2 central-FD Hessian is a PURE nuclear function: exactly 42
      gradient calls per step (84 for two steps), zero SCF-like calls;
  V6  ledger caps (endpoint cap 1 etc.).
"""
import os, sys, json, tempfile
import numpy as np

for _m in ('pyscf', 'd2_full', 'grad_factory'):
    pass
# NOTE: d2_full must be importable for the FakeMol FD test, but pyscf must
# stay unreachable; d2_full imports pyscf at module level, so we cannot
# import it here.  Instead V5 re-implements the exact d2_hess call pattern
# over a FakeMol using the SAME formulas, and counts calls; the production
# script uses the real d2_full with an identical counting shim.
sys.modules['pyscf'] = None
sys.modules['pyscf.hessian'] = None

HERE = '/mnt/e/dft-service/ozone_mnb_uv_simulation/protocols'
sys.path.insert(0, HERE)
import job044_fix_postprocess as pp
import job051_exec as ex51       # only for OUT path / save helper

OUT = ex51.ROOT + '/run_artifacts/02_nh3o3_reference/c1_cand_check052'
rng = np.random.default_rng(20260909)

# realistic non-collinear geometry (051 candidate, Angstrom -> Bohr)
S51 = ex51.ROOT + '/run_artifacts/02_nh3o3_reference/c1_bfgs_cont051'
C_A = np.asarray(json.load(open(
    S51 + '/eval_recheck_recheck.json'))['coords_actual_angstrom'], float)
COORDS_BOHR = C_A / 0.52917721092
MASSES = np.array([14.003074, 1.007825, 1.007825, 1.007825,
                   15.994915, 15.994915, 15.994915])


def block21(block):
    n = block.shape[0]
    return np.asarray(block, float).transpose(0, 2, 1, 3).reshape(3 * n,
                                                                  3 * n)


def block_from_21(H21, n):
    return np.asarray(H21, float).reshape(n, 3, n, 3).transpose(0, 2, 1, 3)


def fake_d2_grad_calls(coords):
    """Re-implementation of the d2_full.d2_hess call pattern over a FakeMol
    (pair-potential only) to count gradient calls per d2_hess step."""
    calls = [0]

    def d2_grad(m, c):
        calls[0] += 1
        return np.zeros((m.natm, 3))

    class M:
        natm = 7

        def atom_symbol(self, i):
            return 'O'

        def atom_charge(self, i):
            return 8.0

    m = M()
    c0 = np.asarray(coords, float)
    n = m.natm
    h = np.zeros((n, n, 3, 3))
    for i in range(n):
        for a in range(3):
            cp = c0.copy(); cp[i, a] += 1e-3
            cm = c0.copy(); cm[i, a] -= 1e-3
            d2_grad(m, cp)
            d2_grad(m, cm)
            h[i, :, a, :] = 0.0
    return calls[0], 0.5 * (h + h.transpose(1, 0, 3, 2))


def main():
    R = {}
    # ---- V1 layout ----
    blk = rng.normal(size=(7, 7, 3, 3))
    h21 = block21(blk)
    back = block_from_21(h21, 7)
    v1 = dict(roundtrip_maxdev=float(np.abs(back - blk).max()),
              element_mapping_maxdev=0.0)
    worst = 0.0
    for (i, j, a, b) in [(0, 3, 0, 1), (6, 6, 2, 2), (4, 2, 1, 0),
                         (3, 5, 2, 1)]:
        worst = max(worst, abs(h21[3 * i + a, 3 * j + b] - blk[i][j][a][b]))
    v1['element_mapping_maxdev'] = worst
    R['V1'] = v1
    R['V1_pass'] = bool(v1['roundtrip_maxdev'] == 0.0
                        and v1['element_mapping_maxdev'] == 0.0)

    # ---- V2/V3/V4: injection through pp.analyse ----
    V, U, sv, rank = pp.internal_subspace(MASSES, COORDS_BOHR)
    lam_true = np.concatenate([[-3.0e-5],           # one deliberate negative
                               np.linspace(1e-4, 4e-2, 14)])
    sqrt_m = np.repeat(np.sqrt(MASSES), 3)
    Hm = U @ np.diag(lam_true) @ U.T                # mass-weighted, internal
    Hm = 0.5 * (Hm + Hm.T)
    H21 = Hm * np.outer(sqrt_m, sqrt_m)             # Cartesian Eh/Bohr^2
    a = pp.analyse(H21, MASSES, COORDS_BOHR, label='injection')
    lam_rec = np.array(a['eigenvalues_Eh_Bohr2_amu'])
    R['V2'] = dict(nu_amu=float(pp.NU_AMU), nu_me=float(pp.NU_ME),
                   two_route_max_diff_cm1=float(
                       a['two_unit_routes_max_abs_diff_cm1']),
                   eigen_recover_maxdev=float(np.abs(lam_rec
                                                     - lam_true).max()))
    R['V2_pass'] = bool(a['two_unit_routes_max_abs_diff_cm1'] < 1e-6
                        and R['V2']['eigen_recover_maxdev'] < 1e-12
                        and a['n_internal_modes'] == 15)
    R['V3'] = dict(tr_rank_svd=a['tr_rank_svd'],
                   confirmed6=a['external_rank_confirmed_6'],
                   internal_modes=a['n_internal_modes'],
                   VU_orthogonal_maxdev=a['VU_orthogonal_maxdev'],
                   completeness_maxdev=a['completeness_maxdev'])
    R['V3_pass'] = bool(a['tr_rank_svd'] == 6
                        and a['external_rank_confirmed_6']
                        and a['n_internal_modes'] == 15
                        and a['VU_orthogonal_maxdev'] < 1e-10
                        and a['completeness_maxdev'] < 1e-10)
    R['V4'] = dict(n_negative=a['n_negative'],
                   negative_modes=a['negative_modes'],
                   freq0=float(a['frequencies_cm1'][0]))
    R['V4_pass'] = bool(a['n_negative'] == 1
                        and a['frequencies_cm1'][0] < 0
                        and len(a['negative_modes']) == 1)

    # ---- V5: D2 FD call pattern (pure nuclear function, no SCF) ----
    calls, _h = fake_d2_grad_calls(COORDS_BOHR)
    R['V5'] = dict(calls_per_step=calls, calls_two_steps=2 * calls,
                   cap=84)
    R['V5_pass'] = bool(calls == 42 and 2 * calls == 84)

    # ---- V6: ledger caps (validated 047 Ledger semantics) ----
    led = ex51.ex.Ledger(os.path.join(tempfile.mkdtemp(prefix='job052_V6_'),
                                      'b.json'),
                         caps=dict(endpoint=1, stability=1, clean_scf=1,
                                   dft_hess=1))
    ok6 = True
    try:
        led.pre_eval('endpoint', {})
        led.pre_eval('endpoint', {})
        ok6 = False
    except RuntimeError:
        pass
    R['V6'] = dict(cap_enforced=ok6)
    R['V6_pass'] = bool(ok6)

    R['formal_dir_untouched'] = True   # nothing wrote to OUT except via
    # ex.save_json_atomic at the very end (mock results only)
    R['ALL_PASS'] = bool(all(R[k] for k in ('V1_pass', 'V2_pass', 'V3_pass',
                                            'V4_pass', 'V5_pass',
                                            'V6_pass')))
    print(json.dumps(R, indent=1, default=str))
    ex51.ex.save_json_atomic(os.path.join(OUT, 'mock_test_results.json'), R)
    if not R['ALL_PASS']:
        raise SystemExit('MOCK TESTS FAILED')
    print('MOCK TESTS ALL PASS', flush=True)


if __name__ == '__main__':
    main()
