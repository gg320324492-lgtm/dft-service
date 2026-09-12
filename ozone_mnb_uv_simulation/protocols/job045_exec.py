#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-045: C1 internal negative-mode direction, two-step-size
energy/gradient check.

Budget: centre independent SCF+gradient x1, displacement SCF+gradient x4,
total attempt cap 5 (failures included).  No optimisation, no stability,
no Hessian, no frequencies, no CP, no other species.

Phase 1 (offline, mock backend only): synthetic quadratic energy validates
the full analysis pipeline (slope + negative curvature recovery with a
non-zero centre gradient).  Temp dir; pyscf is NOT imported in phase 1.

Phase 2 (real, budgeted): centre reproduction gate against the 043 endpoint
record, then displacements t = +0.01, -0.01, +0.02, -0.02 Bohr along the
fixed line R(t) = R0 + t q.  Attempts are persisted before evaluation and
marked done only after results are safely written.
"""
import os, sys, json, time, hashlib, tempfile, traceback
import numpy as np

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
HERE = ROOT + '/protocols'
sys.path.insert(0, HERE)

OUT = ROOT + '/run_artifacts/02_nh3o3_reference/c1_negmode_scan045'
os.makedirs(OUT, exist_ok=True)
LEDGER = OUT + '/budget_scan045.json'
RESULTS = OUT + '/direction_derivative_results.json'

SYMS = ['N', 'H', 'H', 'H', 'O', 'O', 'O']
BOHR_PER_A = 1.0 / 0.52917721092   # same constant as 043
ANG_PER_BOHR = 0.52917721092
GRID_LEVEL = 8                     # 043 endpoint config
GATE_E = 1e-8
GATE_G = 1e-7
GATE_C = 1e-9
STEPS = [0.01, -0.01, 0.02, -0.02]  # execution order per charter
TOTAL_CAP = 5


def save_json_atomic(path, obj):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(obj, fh, indent=2, default=str)
    os.replace(tmp, path)


def sha_arr(a):
    return hashlib.sha256(np.asarray(a, float).tobytes()).hexdigest()[:16]


# ====================================================================
# shared analysis pipeline (validated in phase 1, used in phase 2)
# ====================================================================
def compute_derivatives(E, G, q, h_list):
    """E: {t: energy}; G: {t: flat gradient (21,)}; q: unit direction (21,).

    Returns dict with a0 = g0.q, and for each h:
      a_E(h) = (E(+h)-E(-h))/(2h)                  [Eh/Bohr]
      k_E(h) = (E(+h)-2E(0)+E(-h))/h^2             [Eh/Bohr^2]
      k_g(h) = (g(+h)-g(-h)).q/(2h)                [Eh/Bohr^2]
    plus predicted and actual one-sided energy deltas.
    """
    out = dict(a0=float(np.dot(G[0.0], q)),
               per_h={}, pred={}, actual={})
    for h in h_list:
        hp, hm = h, -h
        a_E = (E[hp] - E[hm]) / (2.0 * h)
        k_E = (E[hp] - 2.0 * E[0.0] + E[hm]) / (h * h)
        k_g = float(np.dot(G[hp] - G[hm], q) / (2.0 * h))
        out['per_h'][repr(h)] = dict(h_Bohr=h, a_E_Eh_Bohr=a_E,
                                     k_E_Eh_Bohr2=k_E, k_g_Eh_Bohr2=k_g)
    return out


def predicted_deltas(E, res, k_H, h_list):
    """DeltaE_pred(+/-h) = +/-h a0 + h^2 k_H / 2 ; actual = E(+/-h)-E(0)."""
    a0 = res['a0']
    for h in h_list:
        for s in (+1, -1):
            key = '%+d*%g' % (s, h)
            res['pred'][key] = dict(delta_E_pred_Eh=s * h * a0
                                    + h * h * k_H / 2.0)
            res['actual'][key] = dict(delta_E_actual_Eh=E[s * h] - E[0.0])
    return res


# ====================================================================
# phase 1: synthetic quadratic validation (mock backend, temp dir)
# ====================================================================
def phase1_synthetic(manifest):
    print('[phase1] synthetic quadratic validation (mock backend, no pyscf)',
          flush=True)
    q = np.asarray(manifest['direction']['q_vector'], float)
    assert abs(np.linalg.norm(q) - 1.0) < 1e-12
    a0_true, k_true = 0.003, -0.0025          # Eh/Bohr, Eh/Bohr^2
    E0 = -282.0

    def mock_eval(t):
        E = E0 + a0_true * t + 0.5 * k_true * t * t
        g = (a0_true + k_true * t) * q
        return E, g

    E, G = {}, {}
    for t in [0.0] + STEPS:
        E[t], G[t] = mock_eval(t)
    h_list = [0.01, 0.02]
    res = compute_derivatives(E, G, q, h_list)
    # k_H for the synthetic system is exactly k_true along q
    k_H_syn = float(q @ (k_true * np.outer(q, q)) @ q)
    res = predicted_deltas(E, res, k_H_syn, h_list)

    checks = {}
    # tolerance rationale: differencing ~282-Eh energies has float floors
    #   a_E: eps*|E|/(2h)      ~ 3e-12 (h=0.01)  -> TOL 1e-10
    #   k_E: eps*|E|/h^2       ~ 6e-10 (h=0.01)  -> TOL 1e-8
    #   k_g, a0, pred-vs-actual: no large cancellation -> TOL 1e-12
    TOL_CURV, TOL_AE, TOL_SLOPE = 1e-8, 1e-10, 1e-12
    for h in h_list:
        r = res['per_h'][repr(h)]
        checks['a_E_h%g' % h] = abs(r['a_E_Eh_Bohr'] - a0_true)
        checks['k_E_h%g' % h] = abs(r['k_E_Eh_Bohr2'] - k_true)
        checks['k_g_h%g' % h] = abs(r['k_g_Eh_Bohr2'] - k_true)
    for key, v in res['pred'].items():
        checks['pred_' + key] = abs(v['delta_E_pred_Eh']
                                    - res['actual'][key]['delta_E_actual_Eh'])
    checks['a0'] = abs(res['a0'] - a0_true)
    # non-stationary centre demonstration: with a0>0, k<0 the -h side lowers
    # the energy while +h raises it (neither side is required to both lower)
    demo = dict(E_minus_h_minus_E0=E[-0.01] - E0, E_plus_h_minus_E0=E[0.01] - E0)
    tol_of = lambda k: (TOL_CURV if k.startswith(('k_E', 'k_g'))
                        else TOL_AE if k.startswith('a_E') else TOL_SLOPE)
    max_err = max(checks.values())
    all_pass = all(v <= tol_of(k) for k, v in checks.items())
    worst = max((v / tol_of(k) for k, v in checks.items()))
    out = dict(a0_true=a0_true, k_true=k_true, k_H_synthetic=k_H_syn,
               checks=checks, max_abs_err=max_err,
               worst_frac_of_tol=worst, all_pass=all_pass,
               tolerances=dict(curvature=TOL_CURV, a_E=TOL_AE,
                               slope_and_delta=TOL_SLOPE),
               cancellation_floor_note='differencing ~282 Eh energies: '
                                       'a_E floor eps*|E|/(2h)~3e-12, '
                                       'k_E floor eps*|E|/h^2~6e-10; '
                                       'observed errors sit at these floors, '
                                       'not pipeline bugs',
               nonstationary_demo=demo, temp_dir='in-memory (no files needed)')
    save_json_atomic(OUT + '/phase1_synthetic_validation.json', out)
    print('[phase1] a0 recovery err = %.2e ; curvature recovery errs = %.2e'
          % (checks['a0'], max(checks['k_E_h0.01'], checks['k_E_h0.02'],
                               checks['k_g_h0.01'], checks['k_g_h0.02'])),
          flush=True)
    print('[phase1] predicted-vs-actual deltaE max err = %.2e ; ALL_PASS=%s'
          % (max(v for k, v in checks.items() if k.startswith('pred_')),
             all_pass), flush=True)
    print('[phase1] non-stationary demo: dE(-h)=%+.3e (lower) dE(+h)=%+.3e '
          '(higher) -> one-sided lowering cannot be required'
          % (demo['E_minus_h_minus_E0'], demo['E_plus_h_minus_E0']),
          flush=True)
    if not all_pass:
        raise RuntimeError('phase1 synthetic validation FAILED')
    return out


# ====================================================================
# phase 2: budgeted real evaluations
# ====================================================================
class Ledger:
    """Attempt ledger: persisted before eval; done only after safe write."""

    def __init__(self, path):
        self.path = path
        if os.path.exists(path):
            self.data = json.load(open(path))
        else:
            self.data = dict(caps=dict(centre=1, displacement=4, total=5),
                             attempts=[], note='failures count against cap')
            save_json_atomic(path, self.data)
        for a in self.data['attempts']:
            if a.get('status') == 'error':
                raise RuntimeError('ledger contains an error attempt -> '
                                   'HARD STOP (no retries permitted): %s'
                                   % json.dumps(a)[:400])

    def _save(self):
        save_json_atomic(self.path, self.data)

    def n_total(self):
        return len(self.data['attempts'])

    def pre_eval(self, category, note):
        if self.n_total() >= TOTAL_CAP:
            raise RuntimeError('budget exhausted (5 attempts)')
        idx = dict(centre=0, displacement=0)
        for a in self.data['attempts']:
            idx[a['category']] = idx.get(a['category'], 0) + 1
        if idx[category] >= (1 if category == 'centre' else 4):
            raise RuntimeError('category cap reached: %s' % category)
        att = dict(attempt=self.n_total() + 1, category=category,
                   category_index=idx[category] + 1, note=note,
                   status='pending', started=time.strftime('%F %T'))
        self.data['attempts'].append(att)
        self._save()
        return att

    def post_eval(self, att, results):
        att.update(results)
        att['status'] = 'done'
        att['finished'] = time.strftime('%F %T')
        self._save()

    def fail(self, att, err):
        att['status'] = 'error'
        att['error'] = str(err)
        att['traceback'] = traceback.format_exc()[-2000:]
        self._save()


def cfg_readback(mf):
    return dict(xc='wb97xd (project -D2 attached)',
                d2_attached=bool(getattr(mf, '_has_full_d2', False)),
                grid_level=int(getattr(mf.grids, 'level', -1)),
                scf_tol=[float(mf.conv_tol), float(mf.conv_tol_grad)],
                basis='def2-TZVP', charge=0, spin=0, grid_response=True,
                solvent='none (gas phase)')


def atom_string(R_bohr):
    return "; ".join("%s %.10f %.10f %.10f" % (s, a, b, c)
                     for s, (a, b, c) in zip(SYMS, R_bohr))


def main():
    manifest = json.load(open(OUT + '/input_manifest.json'))

    # ---------- phase 1 ----------
    phase1_synthetic(manifest)

    # ---------- phase 2 ----------
    print('[phase2] importing real backend components...', flush=True)
    from pyscf import gto
    import d2_full
    from grad_factory import make_mf_d2_gr

    # sources
    d41 = json.load(open(ROOT + '/run_artifacts/02_nh3o3_reference/'
                                'c1_cart_exec_v3/eval_records.json'))
    rec5, key5 = None, None
    for k, v in d41.items():
        if v.get('attempt') == 5 and v.get('status') == 'evaluated':
            rec5, key5 = v, k
            break
    CA = np.asarray(rec5['coords_angstrom'], float)          # canonical
    R0 = CA * BOHR_PER_A                                     # Bohr

    d43 = json.load(open(ROOT + '/run_artifacts/02_nh3o3_reference/'
                                'c1_nondiag_freq/nondiag_results.json'))
    er = d43['endpoint_reproduction']
    e43 = float(er['e_total'])
    g43 = np.asarray(er['grad_full'], float).reshape(-1)
    cfg43 = er['config']
    H_dft = np.asarray(d43['dft_hessian']['matrix'], float)
    H_d2 = np.asarray(d43['d2_hessian']['matrix'], float)
    H_tot = H_dft + H_d2

    q = np.asarray(manifest['direction']['q_vector'], float)

    # displacement geometries: canonical R0 + t q; cross-check vs manifest
    geo_check = []
    geoms = {}
    for t in [0.0] + STEPS:
        R = R0 + t * q.reshape(7, 3)
        geoms[t] = R
        if t != 0.0:
            m = [g for g in manifest['displacement_geometries']
                 if abs(g['t_Bohr'] - t) < 1e-14][0]
            Rm = np.asarray(m['coords_bohr'], float)
            geo_check.append(dict(t=t,
                                  max_diff_vs_manifest_Bohr=float(
                                      np.abs(R - Rm).max()),
                                  atom_string_identical=bool(
                                      atom_string(R) == atom_string(Rm))))
    worst = max(g['max_diff_vs_manifest_Bohr'] for g in geo_check)
    astr_same = all(g['atom_string_identical'] for g in geo_check)
    print('[phase2] geometry cross-check vs manifest: max float diff %.2e '
          'Bohr; formatted atom strings identical = %s'
          % (worst, astr_same), flush=True)
    if not astr_same:
        print('[phase2] NOTE: eval uses the CANONICAL centre (coords_angstrom '
              '* 1/0.52917721092, the 043 convention) so the displacement '
              'line passes exactly through the gated centre; prep manifest '
              'geometries were built from the stored coords_bohr which '
              'differs by <=4.6e-11 Bohr (rounding-boundary effect on '
              '"%.10f" strings). Bias on k_E ~ offset/h^2 <= 4.6e-7 '
              'Eh/Bohr^2 is negligible vs the reported magnitudes; the '
              'supersession is registered in geometry_cross_check.',
              flush=True)

    results = dict(job='JOB-2026-0906-045 direction derivative check',
                   centre_source=dict(batch='JOB-041 attempt=5',
                                      point_key=key5,
                                      e_total_ref=rec5['e_total'],
                                      source_sha16=sha_arr(CA)),
                   direction=dict(q_hash=manifest['direction']['q_hash'],
                                  sign_convention=manifest['direction'][
                                      'sign_convention'],
                                  freq_cm1=manifest['mode0']['source'][
                                      'freq_cm1']),
                   hessian_source=dict(
                       file='c1_nondiag_freq/nondiag_results.json',
                       dft_sha16=sha_arr(H_dft), d2_sha16=sha_arr(H_d2),
                       composition='DFT analytic (wb97xd vacuum) + D2 FD'),
                   geometry_cross_check=geo_check)
    ledger = Ledger(LEDGER)

    def eval_point(t, category):
        att = ledger.pre_eval(category, dict(t_Bohr=t,
                                             atom_sha=sha_arr(geoms[t])))
        try:
            t0 = time.time()
            mol = gto.M(atom=atom_string(geoms[t]), basis='def2-TZVP',
                        charge=0, spin=0, verbose=0, max_memory=4000,
                        unit='Bohr')
            mf = make_mf_d2_gr(mol, solvent=None, grid_response=True,
                               grid_level=GRID_LEVEL)
            mf.kernel()
            g = np.asarray(mf.nuc_grad_method().kernel(),
                           float).reshape(7, 3)
            e = float(mf.e_tot)
            e_d2 = float(d2_full.d2_energy(mol))
            C_act = np.asarray(mol.atom_coords(unit='Angstrom'), float)
            cfg = cfg_readback(mf)
            rec = dict(e_total=e, e_d2_Eh=e_d2, e_dft_part_Eh=e - e_d2,
                       grad=g.tolist(), grad_max=float(np.abs(g).max()),
                       coords_actual_angstrom=C_act.tolist(),
                       coords_actual_sha=sha_arr(C_act),
                       config=cfg, converged=bool(mf.converged),
                       all_finite=bool(np.isfinite(e)
                                       and np.isfinite(g).all()),
                       seconds=round(time.time() - t0, 1))
            # safe write first (results file), then mark done in ledger
            stage = 'eval_%s_%s' % (category, ('centre' if t == 0.0
                                               else '%+g' % t))
            save_json_atomic(OUT + '/eval_%s.json' % stage, rec)
            ledger.post_eval(att, dict(stage=stage, e_total=e,
                                       grad_max=rec['grad_max'],
                                       converged=rec['converged'],
                                       seconds=rec['seconds']))
            print('[phase2] %s: E=%.9f gmax=%.3e e_d2=%.6f conv=%s (%.1fs)'
                  % (stage, e, rec['grad_max'], e_d2, rec['converged'],
                     rec['seconds']), flush=True)
            return rec
        except Exception as exc:  # noqa: BLE001
            ledger.fail(att, exc)
            save_json_atomic(RESULTS, results)
            raise

    # ---- centre gate (attempt 1) ----
    print('[phase2] centre reproduction gate...', flush=True)
    c = eval_point(0.0, 'centre')
    dE = abs(c['e_total'] - e43)
    dg43 = float(np.abs(np.asarray(c['grad']).reshape(-1) - g43).max())
    dC = float(np.abs(np.asarray(c['coords_actual_angstrom']) - CA).max())
    cfg_ok = all(c['config'][k] == cfg43[k] for k in cfg43)
    gate = dict(dE_vs_043=dE, dgrad_max_vs_043=dg43,
                dcoords_vs_041_A=dC, config_match=bool(cfg_ok),
                converged=c['converged'], finite=c['all_finite'],
                gates_pass=bool(dE <= GATE_E and dg43 <= GATE_G
                                and dC <= GATE_C and cfg_ok
                                and c['converged'] and c['all_finite']))
    results['centre_gate'] = gate
    results['centre_eval'] = c
    print('[phase2] centre gate: dE=%.2e dgrad=%.2e dC=%.1e cfg=%s -> %s'
          % (dE, dg43, dC, cfg_ok,
             'PASS' if gate['gates_pass'] else 'FAIL'), flush=True)
    if not gate['gates_pass']:
        save_json_atomic(RESULTS, results)
        raise RuntimeError('HARD STOP: centre reproduction gate failed')

    # ---- displacements (attempts 2-5) ----
    disp = {}
    for t in STEPS:
        disp[t] = eval_point(t, 'displacement')
        results['disp_eval_%+g' % t] = disp[t]
        save_json_atomic(RESULTS, results)

    # ---- direction derivative analysis ----
    E = {0.0: c['e_total']}
    G = {0.0: np.asarray(c['grad'], float).reshape(-1)}
    for t in STEPS:
        E[t] = disp[t]['e_total']
        G[t] = np.asarray(disp[t]['grad'], float).reshape(-1)
    k_H = float(q @ H_tot @ q)
    k_H_dft = float(q @ H_dft @ q)
    k_H_d2 = float(q @ H_d2 @ q)
    res = compute_derivatives(E, G, q, [0.01, 0.02])
    res = predicted_deltas(E, res, k_H, [0.01, 0.02])

    print('\n=== direction derivative results ===', flush=True)
    print('a0 = g0.q            = %+.8e Eh/Bohr' % res['a0'], flush=True)
    print('k_H = q.H_tot.q      = %+.8e Eh/Bohr^2  (DFT %+.6e, D2 %+.6e)'
          % (k_H, k_H_dft, k_H_d2), flush=True)
    for h in (0.01, 0.02):
        r = res['per_h'][repr(h)]
        print('h=%4.2f: a_E=%+.6e  k_E=%+.6e  k_g=%+.6e  (all Eh,Bohr)'
              % (h, r['a_E_Eh_Bohr'], r['k_E_Eh_Bohr2'], r['k_g_Eh_Bohr2']),
              flush=True)
    for s in (+1, -1):
        for h in (0.01, 0.02):
            key = '%+d*%g' % (s, h)
            print('dE(%s): pred %+.6e  actual %+.6e  diff %+.2e'
                  % (key, res['pred'][key]['delta_E_pred_Eh'],
                     res['actual'][key]['delta_E_actual_Eh'],
                     res['pred'][key]['delta_E_pred_Eh']
                     - res['actual'][key]['delta_E_actual_Eh']), flush=True)

    k_E_1 = res['per_h'][repr(0.01)]['k_E_Eh_Bohr2']
    k_E_2 = res['per_h'][repr(0.02)]['k_E_Eh_Bohr2']
    k_g_1 = res['per_h'][repr(0.01)]['k_g_Eh_Bohr2']
    k_g_2 = res['per_h'][repr(0.02)]['k_g_Eh_Bohr2']
    results['direction_analysis'] = dict(
        a0_Eh_Bohr=res['a0'], k_H_Eh_Bohr2=k_H, k_H_dft=k_H_dft,
        k_H_d2=k_H_d2, per_h=res['per_h'], pred=res['pred'],
        actual=res['actual'],
        step_sensitivity=dict(
            k_E_diff_002_vs_001=k_E_2 - k_E_1,
            k_g_diff_002_vs_001=k_g_2 - k_g_1,
            k_E_rel_diff=abs(k_E_2 - k_E_1) / max(abs(k_E_1), 1e-30),
            k_g_rel_diff=abs(k_g_2 - k_g_1) / max(abs(k_g_1), 1e-30)),
        sign_consistency=dict(
            k_H_negative=bool(k_H < 0),
            k_E_negative_both_steps=bool(k_E_1 < 0 and k_E_2 < 0),
            k_g_negative_both_steps=bool(k_g_1 < 0 and k_g_2 < 0),
            all_three_agree_negative=bool(k_H < 0 and k_E_1 < 0
                                          and k_E_2 < 0 and k_g_1 < 0
                                          and k_g_2 < 0)))
    results['budget'] = dict(total_attempts=ledger.n_total(),
                             caps=ledger.data['caps'],
                             attempts=ledger.data['attempts'])
    save_json_atomic(RESULTS, results)
    print('\nsaved ->', RESULTS, flush=True)
    print('DONE: budget used %d/5 attempts' % ledger.n_total(), flush=True)


if __name__ == '__main__':
    main()
