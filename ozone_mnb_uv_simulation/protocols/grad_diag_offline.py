import os, sys, json, hashlib
import numpy as np
HERE = '/mnt/e/dft-service/ozone_mnb_uv_simulation/protocols'
sys.path.insert(0, HERE)
from torque_decomp import decompose_force

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
OUT = ROOT + '/run_artifacts/02_nh3o3_reference/grad_diag'
os.makedirs(OUT, exist_ok=True)

# ---------- 1) extract JOB-034 last completed evaluation ----------
r34 = json.load(open(ROOT + '/run_artifacts/02_nh3o3_reference/'
                            'c1_full_opt_cont/exec_results_formal.json'))
steps = r34['trajectory']
last = steps[-1]
assert last['step'] == 20, 'unexpected last step'
coords = np.asarray(last['coords_angstrom'], float)
e = last['e_total']
pk = hashlib.sha256(coords.tobytes()).hexdigest()[:16]
# REGISTERED GAP: the 033/034 trajectory step records store grad_max and
# grad_rms but NOT the full gradient vector -- the batch-planned extraction
# of the full gradient from the record is NOT possible; the direction q is
# therefore constructed from the INDEPENDENT center gradient (per the batch
# instruction), and the full-vector decomposition runs on that center
# gradient after reproduction.
gap = dict(grad_full_in_record=False,
           recorded=['grad_max', 'grad_rms'],
           consequence='full-vector decomposition moved to the independent '
                       'center gradient (post reproduction); full-vector '
                       'comparison vs JOB-034 impossible -> scalar gates '
                       '(E, grad_max, grad_rms) used for reproduction')
print('source: c1_full_opt_cont/exec_results_formal.json step %d' % last['step'])
print('coords hash:', pk)
print('E:', e, '| grad_max:', last['grad_max'], '| grad_rms:', last['grad_rms'])

# source & hash provenance
prov = dict(source_file='run_artifacts/02_nh3o3_reference/c1_full_opt_cont/'
                         'exec_results_formal.json (JOB-034)',
            source_step=last['step'],
            coords_sha16=pk,
            e_total=e,
            grad_max_recorded=last['grad_max'],
            grad_rms_recorded=last['grad_rms'],
            record_gap=gap,
            note='record is the last COMPLETED evaluation of JOB-034 '
                 '(identified by trajectory + n_opt_steps=20 + step=20)')

# ---------- 2) gradient decomposition: DEFERRED to the independent center
# gradient (executed by grad_diag_run.py after the center reproduction) ----------
dec_out = dict(status='deferred to center gradient (see grad_diag_run.py; '
                      'the 034 record has no full gradient vector)')

# ---------- 3) cross-checks: thresholds, scanner path, coords-gradient corr. ----------
r33 = json.load(open(ROOT + '/run_artifacts/02_nh3o3_reference/'
                            'c1_full_opt/exec_results_formal.json'))
same_conv = r33['conv_params'] == r34['conv_params']
same_cfg = (r33['conv_params']['gradientmax'] == 1e-6
            and r34['conv_params']['gradientmax'] == 1e-6)
# scanner call path: both drivers used BudgetedScanner (import from
# gas_monomer_reference_v2) -- verify by grep of the driver sources
import io
p33 = open(ROOT + '/protocols/c1_full_opt.py').read()
p34 = open(ROOT + '/protocols/c1_full_opt_cont.py').read()
same_path = ('BudgetedScanner' in p33) and ('BudgetedScanner' in p34) \
    and ('kernel = ' not in p33) and ('kernel = ' not in p34)
# end-segment coords-gradient correlation (last 10 steps of 034)
g10 = [s['grad_max'] for s in steps[-10:]]
st10 = [s['step_size_A'] for s in steps[-10:]]
corr = float(np.corrcoef(g10, st10)[0, 1])

cross = dict(conv_params_identical=same_conv,
             strict_thresholds=same_cfg,
             scanner_path_same_and_no_kernel_patch=same_path,
             grad_max_vs_step_size_corr_last10=corr,
             note='correlation is descriptive only; no attribution')

# ---------- 4) attribution correction (034 -> 035) ----------
attr = dict(
    withdrawn=['"梯度地板稳定出现"（归因性表述）',
               '"极平坦势面扩散"（归因性表述）'],
    replaced_by='actual observations: end-segment energy changes are small '
               '(per-step |dE| ~ 1e-6..1e-8 Eh), max|g| oscillates in '
               '3.2e-4..1.0e-3 across two independently initialised '
               'optimizers, and the geometry is still changing (contacts '
               '3.23->3.13 A, orientation angle 20->29.5 deg)')

out = dict(job='JOB-2026-0906-035 offline gradient decomposition',
           provenance=prov, decomposition=dec_out, largest_gradient=None,
           cross_checks=cross, attribution_correction=attr,
           acceptance_status_note='decomposing/removing external components '
                                  'does NOT change the acceptance status: '
                                  'the endpoint remains an unconverged '
                                  'optimizer endpoint (no recheck run)')
json.dump(out, open(OUT + '/offline_gradient_decomposition.json', 'w'),
          indent=2)
print(json.dumps(dict(dec=dec_out, cross=cross), indent=2)[:1500])
print('OFFLINE ANALYSIS SAVED')
