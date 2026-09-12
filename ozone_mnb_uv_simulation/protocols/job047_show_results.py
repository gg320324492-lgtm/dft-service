#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Show 047 displacement record details + 4-point even/odd decomposition
(read-only, offline)."""
import json
import numpy as np

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
OUT = ROOT + '/run_artifacts/02_nh3o3_reference/c1_bidir_opt047'
S45 = ROOT + '/run_artifacts/02_nh3o3_reference/c1_negmode_scan045'

for tag in ('disp_p10', 'disp_m10'):
    d = json.load(open('%s/eval_disp_%s.json' % (OUT, tag)))
    print('%s: E=%.12f gmax=%.4e e_d2=%.6e conv=%s cfg_ok=%s'
          % (tag, d['e_total'], d['grad_max'], d['e_d2_Eh'],
             d['converged'], d['all_finite']))

res = json.load(open(OUT + '/bidir_opt047_results.json'))
print('\nbranch_selection:', json.dumps(res['branch_selection'],
                                        indent=1))
print('status:', res['status'])

# 4-point decomposition along q: 045 +-0.01 and 047 +-0.10
c = json.load(open(S45 + '/eval_eval_centre_centre.json'))
d45 = json.load(open(S45 + '/direction_derivative_results.json'))
E0 = c['e_total']
e_p01 = json.load(open(S45 + '/eval_eval_displacement_+0.01.json'))['e_total']
e_m01 = json.load(open(S45 + '/eval_eval_displacement_-0.01.json'))['e_total']
e_p10 = json.load(open(OUT + '/eval_disp_disp_p10.json'))['e_total']
e_m10 = json.load(open(OUT + '/eval_disp_disp_m10.json'))['e_total']
print('\nE-centre  = %+.6e' % 0.0)
print('dE(+0.01) = %+.6e   dE(-0.01) = %+.6e' % (e_p01 - E0, e_m01 - E0))
print('dE(+0.10) = %+.6e   dE(-0.10) = %+.6e' % (e_p10 - E0, e_m10 - E0))
even = [(e_p01 + e_m01) / 2 - E0, (e_p10 + e_m10) / 2 - E0]
odd = [(e_p01 - e_m01) / 2, (e_p10 - e_m10) / 2]
print('even part at 0.01/0.10: %+.4e / %+.4e' % (even[0], even[1]))
print('odd  part at 0.01/0.10: %+.4e / %+.4e (linear slope a0 = %.3e)'
      % (odd[0], odd[1], odd[1] / 0.10))
# even-part quartic fit: 0.5 k2 s^2 + 0.25 k4 s^4
A = np.array([[0.5 * 0.01 ** 2, 0.25 * 0.01 ** 4],
              [0.5 * 0.10 ** 2, 0.25 * 0.10 ** 4]])
sol = np.linalg.solve(A, np.array(even))
k2, k4 = sol
print('even quartic fit: k2 = %+.4e Eh/Bohr^2, k4 = %+.4e Eh/Bohr^4'
      % (k2, k4))
if k2 < 0 and k4 > 0:
    s_star = np.sqrt(-k2 / k4)
    E_star = 0.5 * k2 * s_star ** 2 + 0.25 * k4 * s_star ** 4
    print('  fit suggests even-part minima at s* = %+.4f Bohr, depth '
          '%+.3e Eh (FIT OBSERVATION, not evaluated points)' % (s_star, E_star))
