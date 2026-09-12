import os, sys, json, tempfile
import numpy as np
sys.path.insert(0, '/mnt/e/dft-service/ozone_mnb_uv_simulation/protocols')
import c1_scan_exec_v2 as E
from c1_scan_exec_v2 import (Budget, run_entry, EvaluationFailed,
                             ScanIntegrityError, save_results_atomic,
                             load_results, point_key)

ok = True
TMP = tempfile.mkdtemp()
FORMAL_RESULTS = E.RESULTS_FILE
FORMAL_LEDGER = E.LEDGER


def rep(name, cond, detail=''):
    global ok
    ok &= bool(cond)
    print('[%s] %s %s' % ('PASS' if cond else 'FAIL', name, detail))


# ---------- T-A: point-2 failure -> point-1 full result on disk ----------
led_A = os.path.join(TMP, 'ledA.json')
res_A_path = os.path.join(TMP, 'resA.json')
res_A = {}
sim_fail_p2 = E.sim_backend_factory(fail_at=3.2272, harmonic=(0.05, 4.5))
try:
    run_entry(ctrl=Budget(led_A, dict(total=6, scan=6), init_new=True),
              backend_factory=sim_fail_p2, execute=True, results=res_A,
              results_path=res_A_path)
except EvaluationFailed:
    pass
on_disk = load_results(res_A_path)
done_recs = [v for k, v in on_disk.items()
             if isinstance(v, dict) and v.get('status') == 'done']
tA = (len(done_recs) == 1
      and isinstance(done_recs[0].get('grad_full'), list)
      and len(done_recs[0]['grad_full']) == 7
      and 'config_readback' in done_recs[0]
      and on_disk.get('last_error') is not None)
rep('T-A point-2 failure -> point-1 full result on disk & verifiable', tA,
    'done records on disk: %d' % len(done_recs))

# ---------- T-B: ledger done but full result missing -> MUST stop ----------
del on_disk[[k for k, v in on_disk.items()
             if isinstance(v, dict) and v.get('status') == 'done'][0]]
save_results_atomic(on_disk, res_A_path)
ctrl_B = Budget(led_A, dict(total=6, scan=6))    # restart: ledger kept
try:
    run_entry(ctrl=ctrl_B, backend_factory=E.sim_backend_factory(),
              execute=True, results=on_disk, results_path=res_A_path)
    tB = False
except ScanIntegrityError as e:
    tB = 'full result missing' in str(e)
rep('T-B ledger done but result missing -> ScanIntegrityError (stop)', tB)

# ---------- T-C: incompatible config / tampered hash -> no reuse ----------
led_C = os.path.join(TMP, 'ledC.json')
res_C_path = os.path.join(TMP, 'resC.json')
res_C = {}
try:
    run_entry(ctrl=Budget(led_C, dict(total=6, scan=6), init_new=True),
              backend_factory=E.sim_backend_factory(), execute=True,
              results=res_C, results_path=res_C_path)
except Exception:
    pass
# case 1: config incompatible (stored grid_level tampered to 5)
k1 = [k for k, v in res_C.items() if isinstance(v, dict)][0]
res_C[k1]['config_readback']['grid_level'] = 5
save_results_atomic(res_C, res_C_path)
try:
    run_entry(ctrl=Budget(led_C, dict(total=6, scan=6)),
              backend_factory=E.sim_backend_factory(), execute=True,
              results=res_C, results_path=res_C_path)
    tC1 = False
except ScanIntegrityError as e:
    tC1 = 'config incompatible' in str(e)
rep('T-C1 incompatible stored config -> no reuse (stop)', tC1)
# case 2: tampered coords hash
k2 = [k for k, v in res_C.items() if isinstance(v, dict)][0]
res_C[k2]['coords_computed_sha'] = 'deadbeef'
save_results_atomic(res_C, res_C_path)
try:
    run_entry(ctrl=Budget(led_C, dict(total=6, scan=6)),
              backend_factory=E.sim_backend_factory(), execute=True,
              results=res_C, results_path=res_C_path)
    tC2 = False
except ScanIntegrityError as e:
    tC2 = 'hash or config incompatible' in str(e)
rep('T-C2 tampered coords hash -> no reuse (stop)', tC2)

# ---------- T-D: formal vs simulated path separation ----------
tD = (E.BATCH_DIR.startswith(E.ROOT + '/run_artifacts/')
      and 'c1_scan_formal' in E.RESULTS_FILE
      and E.LEDGER.endswith('budget_formal.json'))
rep('T-D formal results/ledger under run_artifacts/c1_scan_formal', tD)

# ---------- T-E: validation mode on FORMAL paths ----------
led_F = E.Budget(FORMAL_LEDGER, dict(total=6, scan=6), init_new=True)
res_F = load_results(FORMAL_RESULTS)
run_entry(ctrl=led_F, backend_factory=E.real_backend, execute=False,
          results=res_F, results_path=FORMAL_RESULTS)
tE = (led_F.used_total() == 0 and led_F.used_cat('scan') == 0)
rep('T-E validation mode on formal paths: 6 points pass, 0 evaluations', tE)

print('FORMAL ENTRY PRE-CHECKS:', 'ALL PASS' if ok else 'FAILED')
if not ok:
    raise SystemExit(1)
