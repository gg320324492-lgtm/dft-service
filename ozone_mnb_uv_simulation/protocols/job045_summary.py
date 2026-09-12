#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-045 summary: read direction_derivative_results.json, print the full
comparison table and derived quantities for the report (read-only)."""
import json
import numpy as np

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
OUT = ROOT + '/run_artifacts/02_nh3o3_reference/c1_negmode_scan045'
d = json.load(open(OUT + '/direction_derivative_results.json'))

E = {0.0: d['centre_eval']['e_total']}
G = {0.0: np.asarray(d['centre_eval']['grad'], float).reshape(-1)}
for t in (0.01, -0.01, 0.02, -0.02):
    E[t] = d['disp_eval_%+g' % t]['e_total']
    G[t] = np.asarray(d['disp_eval_%+g' % t]['grad'], float).reshape(-1)
da = d['direction_analysis']
q = None
m = json.load(open(OUT + '/input_manifest.json'))
q = np.asarray(m['direction']['q_vector'], float)

print('=== budget ===')
for a in d['budget']['attempts']:
    print('  attempt %d %-13s %-7s E=%s gmax=%s %ss'
          % (a['attempt'], a['category'], a['status'],
             ('%.12f' % a['e_total']) if a.get('e_total') else '-',
             ('%.3e' % a['grad_max']) if a.get('grad_max') else '-',
             a.get('seconds', '-')))
print('  total: %d/5' % d['budget']['total_attempts'])

print('\n=== energies vs centre ===')
E0 = E[0.0]
for t in (0.01, -0.01, 0.02, -0.02):
    print('  t=%+5.2f  E=%-18.12f  dE=%+.6e Eh (%+.3e kcal/mol)'
          % (t, E[t], E[t] - E0, (E[t] - E0) * 627.5095))
print('  all four displacement sides LOWER the energy (a0 ~ -2.5e-8 is tiny)')

print('\n=== centre gate ===')
g = d['centre_gate']
print('  dE=%.2e dgrad=%.2e dC=%.1e cfg=%s conv=%s -> %s'
      % (g['dE_vs_043'], g['dgrad_max_vs_043'], g['dcoords_vs_041_A'],
         g['config_match'], g['converged'],
         'PASS' if g['gates_pass'] else 'FAIL'))

print('\n=== slope consistency (FD vs centre gradient projection) ===')
print('  a0 = g0.q            = %+.8e Eh/Bohr' % da['a0_Eh_Bohr'])
for h in ('0.01', '0.02'):
    print('  a_E(h=%s)             = %+.8e  (diff %+.1e)'
          % (h, da['per_h'][repr(float(h))]['a_E_Eh_Bohr'],
             da['per_h'][repr(float(h))]['a_E_Eh_Bohr'] - da['a0_Eh_Bohr']))

print('\n=== curvature comparison ===')
print('  k_H (saved Hessian)  = %+.6e Eh/Bohr^2  (DFT %+.6e, D2 %+.6e)'
      % (da['k_H_Eh_Bohr2'], da['k_H_dft'], da['k_H_d2']))
for h in ('0.01', '0.02'):
    r = da['per_h'][repr(float(h))]
    print('  h=%s: k_E=%+.6e  k_g=%+.6e  k_H/k_E=%.3f  k_H/k_g=%.3f'
          % (h, r['k_E_Eh_Bohr2'], r['k_g_Eh_Bohr2'],
             da['k_H_Eh_Bohr2'] / r['k_E_Eh_Bohr2'],
             da['k_H_Eh_Bohr2'] / r['k_g_Eh_Bohr2']))
ss = da['step_sensitivity']
print('  step sensitivity: k_E diff %+.3e (rel %.1f%%), k_g diff %+.3e '
      '(rel %.1f%%)' % (ss['k_E_diff_002_vs_001'],
                        100 * ss['k_E_rel_diff'],
                        ss['k_g_diff_002_vs_001'],
                        100 * ss['k_g_rel_diff']))

print('\n=== equivalent frequency along q ===')
LAM = (72.186 / 5140.487) ** 2        # |lambda_mw| from 044 mode 0
mu_eff = abs(da['k_H_Eh_Bohr2']) / LAM
print('  mu_eff = |k_H|/lambda_mw = %.4f amu' % mu_eff)
print('  Hessian equivalent : -72.186 cm^-1 (by construction)')
for h in ('0.01', '0.02'):
    ke = da['per_h'][repr(float(h))]['k_E_Eh_Bohr2']
    print('  FD h=%s equivalent: %+.2f cm^-1'
          % (h, -5140.487 * (abs(ke) / mu_eff) ** 0.5))

print('\n=== predicted vs actual (k_H quadratic model) ===')
for key in ('+1*0.01', '-1*0.01', '+1*0.02', '-1*0.02'):
    p = da['pred'][key]['delta_E_pred_Eh']
    a = da['actual'][key]['delta_E_actual_Eh']
    print('  dE(%s): pred %+.4e  actual %+.4e  ratio pred/act %+.3f'
          % (key, p, a, p / a))

print('\n=== sign consistency ===')
print(' ', json.dumps(da['sign_consistency']))
