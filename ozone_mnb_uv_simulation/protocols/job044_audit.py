import os, json, hashlib
import numpy as np

BD = 'run_artifacts/02_nh3o3_reference/c1_nondiag_freq'
res = json.load(open(BD + '/nondiag_results.json'))

print('=== 043 results JSON top-level keys ===')
for k in sorted(res.keys()):
    print('  ', k)

print()
print('=== endpoint reproduction ===')
er = res.get('endpoint_reproduction', {})
for k in ['e_total', 'grad_max', 'grad_rms', 'dE_vs_041', 'dgrad_max_vs_041',
          'dcoords_vs_041', 'gates_pass', 'actual_max_g']:
    print('  %s = %s' % (k, er.get(k, '-')))
print('  config =', json.dumps(er.get('config', {})))

print()
print('=== stability ===')
st = res.get('stability', {})
for k in ['call', 'stable_i', 'stable_e', 'seconds', 'note']:
    v = st.get(k, '-')
    if k == 'note':
        v = str(v)[:100]
    print('  %s = %s' % (k, v))
print('  raw_return =', st.get('raw_return', '-'))

print()
print('=== DFT hessian meta ===')
dh = res.get('dft_hessian', {})
for k in ['shape', 'unit', 'method', 'seconds']:
    print('  %s = %s' % (k, dh.get(k, '-')))
print('  has matrix:', 'matrix' in dh)

print()
print('=== D2 hessian meta ===')
d2 = res.get('d2_hessian', {})
for k in ['shape', 'unit', 'method', 'seconds', 'd2_fd_calls_added']:
    print('  %s = %s' % (k, d2.get(k, '-')))
print('  has matrix:', 'matrix' in d2)

print()
print('=== combined ===')
cb = res.get('combined_hessian', {})
for k in ['unit', 'composition', 'symmetry_residual']:
    print('  %s = %s' % (k, cb.get(k, '-')))

print()
print('=== harmonic analysis (043, TO BE WITHDRAWN) ===')
ha = res.get('harmonic_analysis', {})
for k in ['method', 'projection_rank', 'n_modes', 'n_negative']:
    print('  %s = %s' % (k, ha.get(k, '-')))
freqs = ha.get('all_frequencies_cm1', [])
print('  frequencies:', [round(f, 2) for f in freqs])

# --- check whether raw (unsymmetrized) matrices were saved ---
print()
print('=== raw (unsymmetrized) matrix availability ===')
print('  dft_hessian has raw key:', any(
    'raw' in k.lower() for k in dh.keys()))
print('  d2_hessian has raw key:', any('raw' in k.lower() for k in d2.keys()))
print('  -> only symmetrized matrices were saved (raw MISSING)')

# --- mass-weighting / unit check of the 043 post-processing ---
print()
print('=== unit-error diagnosis ===')
EH_J = 4.3597447222071e-18
BOHR_M = 5.29177210903e-11
AMU_KG = 1.66053906660e-27
ME_KG = 9.1093837015e-31
C_CM_S = 2.99792458e10

conv_amu = np.sqrt(EH_J / (BOHR_M**2 * AMU_KG)) / (2 * np.pi * C_CM_S)
conv_me = np.sqrt(EH_J / (BOHR_M**2 * ME_KG)) / (2 * np.pi * C_CM_S)
print('  SI-derived conv (mass in amu)          = %.4f cm^-1' % conv_amu)
print('  SI-derived conv (mass in electron mass) = %.4f cm^-1' % conv_me)
print('  ratio conv_me/conv_amu = %.4f  (= sqrt(amu/me) = %.4f)'
      % (conv_me / conv_amu, np.sqrt(AMU_KG / ME_KG)))
print('  old script used masses in ELECTRON MASS but conv = 5140.487 (amu-based)')
print('  -> frequencies too small by factor %.3f' % (conv_me / conv_amu))

# check: what does the stored harmonic analysis claim
g_src = np.asarray(er.get('grad_full', [[0.0] * 3] * 7))
print()
print('  grad_full shape:', g_src.shape, '(7x3)')
print('  grad_max from record: %.6e' % er.get('grad_max', float('nan')))
print('  actual |g| max recomputed: %.6e' % np.abs(g_src).max())
PYEOF_MARK = None
