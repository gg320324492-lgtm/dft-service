import os, sys, json, hashlib
import numpy as np
ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
REF = ROOT + '/run_artifacts/02_nh3o3_reference/gas_monomer_references'

def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16]

# --- script provenance ---
scripts = {}
for f in ('gas_monomer_reference.py', 'nh3_opt_continue2.py',
          'gas_monomer_precheck.py', 'nh3_consistency_diag.py', 'sp_ledger.py'):
    p = ROOT + '/protocols/' + f
    scripts[f] = dict(sha256_16=sha(p), size=os.path.getsize(p))
src = open(ROOT + '/protocols/nh3_opt_continue2.py').read()
scripts['nh3_opt_continue2.py']['contains_kernel_patch'] = ('.kernel = ' in src)
scripts['nh3_opt_continue2.py']['executed_version_note'] = (
    'current on-disk version == v5-executed version (last edit preceded the '
    'v5 run); the v3/v4-executed variants (BASE_EVALS/MAXSTEPS/filenames '
    'different) were edited in place and are NOT preserved as separate files '
    '- recorded as post-hoc status, originals not recoverable')
src1 = open(ROOT + '/protocols/gas_monomer_reference.py').read()
scripts['gas_monomer_reference.py']['contains_kernel_patch'] = ('gobj.kernel = k' in src1 or '.kernel = k' in src1)
scripts['gas_monomer_reference.py']['executed_version_note'] = (
    'current on-disk version == v1-executed version (contains the defective '
    'binding patch; historical evidence, to be superseded by v2 entry)')

# --- result file hashes ---
results = {}
for f in ('monomer_results.json', 'nh3_steps.json', 'nh3_continue_steps.json',
          'nh3_v3_steps.json', 'nh3_v3_result.json', 'nh3_v4_result.json',
          'nh3_v5_steps.json', 'nh3_v5_result.json',
          'diagnostic_consistency.json', 'attempts.json',
          'gas_monomer_run.log', 'nh3_continue_run.log', 'nh3_v3_run.log',
          'nh3_v4_run.log', 'nh3_v5_run.log'):
    p = REF + '/' + f
    if os.path.exists(p):
        results[f] = dict(sha256_16=sha(p), size=os.path.getsize(p))

# --- chain verification for NH3 v3 -> v4 -> v5 ---
v3 = json.load(open(REF + '/nh3_v3_result.json'))
v4 = json.load(open(REF + '/nh3_v4_result.json'))
v5 = json.load(open(REF + '/nh3_v5_result.json'))
c3 = np.asarray(v3['v3_final']['coords_angstrom'], float)
c4 = np.asarray(v4['v3_final']['coords_angstrom'], float)
c5 = np.asarray(v5['v3_final']['coords_angstrom'], float)
chain = dict(
    v3_to_v4_input_match=bool(np.abs(c3 - c4).max() < 1e-12),
    v4_to_v5_input_match=bool(np.abs(c4 - c5).max() < 1e-12),
    v5_final_e=v5['v3_final']['e_total'],
    v5_final_grad_max=v5['v3_final']['grad_max'],
    v5_scf_converged=v5['v3_final']['scf_converged'])

# O3 endpoint
mr = json.load(open(REF + '/monomer_results.json'))
o3 = mr['o3']['endpoint_recheck']
o3_lines = open(ROOT + '/inputs/nh3o3_phase2/o3_gas.xyz').read().splitlines()
o3_in = [[l.split()[0], float(l.split()[1]), float(l.split()[2]), float(l.split()[3])]
         for l in o3_lines[1:1 + int(o3_lines[0].split()[0])]]
co3 = np.asarray(o3['coords_angstrom'], float)
i3 = np.asarray([[a[1], a[2], a[3]] for a in o3_in], float)
o3_check = dict(recheck_coords_vs_input_maxdiff=float(np.abs(co3 - i3).max()),
                e_total=o3['e_total'], grad_max=o3['grad_max'])

# --- final endpoint index ---
nh3_v5 = v5['v3_final']
idx = dict(
    job='JOB-2026-0906-025 final endpoint index',
    default_endpoints={
        'NH3_gas_v5_final': dict(
            species='NH3', phase='gas',
            coords_angstrom=nh3_v5['coords_angstrom'],
            e_total_hartree=nh3_v5['e_total'],
            grad_max_eh_bohr=nh3_v5['grad_max'],
            criterion='unprojected max|g| <= 1e-5',
            status='stationary-point candidate (NO Hessian/frequency; not an '
                   'accepted minimum)',
            provenance='nh3_opt_continue2.py (v5, unpatched callback-only '
                       'path) -> nh3_v5_result.json',
            default_read=True),
        'O3_gas_final': dict(
            species='O3', phase='gas',
            coords_angstrom=o3['coords_angstrom'],
            e_total_hartree=o3['e_total'],
            grad_max_eh_bohr=o3['grad_max'],
            criterion='unprojected max|g| <= 1e-5',
            status='stationary-point candidate (NO Hessian/frequency)',
            provenance='gas_monomer_reference.py (v1) -> '
                       'monomer_results.json :: o3.endpoint_recheck',
            default_read=True)},
    superseded_endpoints={
        'NH3_gas_v1_frozen_final': dict(
            note='v1 run: driver wrapper defect froze the gradient at the '
                 'initial geometry; endpoint (r=1.0486 A, max|g|=4.634e-2 '
                 'fresh / 4.357e-2 optimizer) NOT a stationary candidate',
            coords_source='nh3_steps.json step 15',
            default_read=False, superseded_by='NH3_gas_v5_final'),
        'NH3_gas_v2_disp_start': dict(note='diagnostic displaced start',
                                      default_read=False),
        'NH3_gas_v3_endpoint_6.5e-5': dict(
            note='v3 endpoint: pyberny internal criteria met but unprojected '
                 'Cartesian max|g|=6.484e-5 > 1e-5 -> superseded by v5',
            default_read=False, superseded_by='NH3_gas_v5_final')},
    rule='superseded/failed endpoints retained for evidence; readers must '
         'use default_endpoints only',
    chain_verification=chain, o3_check=o3_check,
    script_provenance=scripts, result_file_hashes=results)

json.dump(idx, open(ROOT + '/run_artifacts/02_nh3o3_reference/final_endpoint_index.json', 'w'),
          indent=2)
print(json.dumps(dict(chain=chain, o3_check=o3_check,
                      nh3_patch_absent=not scripts['nh3_opt_continue2.py']['contains_kernel_patch'],
                      v1_patch_present=scripts['gas_monomer_reference.py']['contains_kernel_patch'],
                      v5_e=chain['v5_final_e'], v5_g=chain['v5_final_grad_max']), indent=2))
