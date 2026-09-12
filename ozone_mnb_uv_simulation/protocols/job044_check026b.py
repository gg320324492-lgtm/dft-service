import json
import numpy as np

r = json.load(open('run_artifacts/02_nh3o3_reference/freq_check/freq_check_results.json'))
res = r['results']
for sp in ['NH3', 'O3']:
    d = res[sp]
    print('===', sp, '===')
    f = d['frequencies']
    print('  internal_expected:', f['internal_expected'])
    print('  n_internal_modes:', f['n_internal_modes'])
    print('  my_freq_cm:', [round(x, 2) for x in f['my_freq_cm']])
    print('  negative_modes:', f['negative_modes'])
    print('  pyscf_cross_check_freq_cm:',
          [round(x, 2) for x in f['pyscf_cross_check_freq_cm']]
          if f.get('pyscf_cross_check_freq_cm') else None)
    print('  pyscf_vs_my_max_diff_cm:', f.get('pyscf_vs_my_max_diff_cm'))
    print('  tr_basis_rank:', f.get('tr_basis_rank'))
    print('  tr_basis_ortho_maxdev:', f.get('tr_basis_ortho_maxdev'))
    print('  projection_dims:', f.get('projection_dims'))
    print('  cross_check_note:', str(f.get('cross_check_note'))[:200])
    print('  masses:', d['masses'])
    h = d['hessian']
    print('  matrices_saved:', h.get('matrices_saved'))
    print('  composition:', str(h.get('composition'))[:200])
    print('  antisym_residual_max:', h.get('antisym_residual_max'))
    print('  d2_step_check:', str(h.get('d2_step_check'))[:200])
    print()
PYEOF_MARK = None
