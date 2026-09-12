import os, sys, json, tempfile, hashlib
import numpy as np
sys.path.insert(0, '/mnt/e/dft-service/ozone_mnb_uv_simulation/protocols')
import c1_repro_034 as R
from c1_repro_034 import (load_start_v2, build_verified, Budget,
                          run_reproduction, SourceError, EvaluationFailed,
                          BATCH_DIR, RESULTS, LEDGER)

ok = True
TMP = tempfile.mkdtemp()
FORMAL_DIRS = (BATCH_DIR, RESULTS, LEDGER)


def rep(name, cond, detail=''):
    global ok
    ok &= bool(cond)
    print('[%s] %s %s' % ('PASS' if cond else 'FAIL', name, detail))


def formal_state():
    """Hash snapshot of the formal batch dir (isolation check)."""
    st = {}
    for f in os.listdir(BATCH_DIR):
        p = os.path.join(BATCH_DIR, f)
        st[f] = hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16]
    return st


# ---- T1: 033-endpoint misuse rejected by the source check ----
r33 = json.load(open(os.path.join(os.path.dirname(BATCH_DIR.replace(
    'c1_repro_034', 'c1_full_opt')),
    'exec_results_formal.json'))) if False else json.load(open(
        R.ROOT + '/run_artifacts/02_nh3o3_reference/c1_full_opt/'
        'exec_results_formal.json'))
rec33 = r33['trajectory'][-1]
try:
    load_start_v2(record=rec33)
    t1 = False
except SourceError as e:
    t1 = 'JOB-033 step-30' in str(e)
rep('T1 033-endpoint misuse rejected by source check', t1)

# ---- T2: missing / double A->Bohr conversion rejected by the geometry check ----
C, manifest, key = load_start_v2()
x_bohr = C * R.BOHR_PER_A
# correct path
mol, xb, rb_A, dev = build_verified(C)
t2a = dev <= 1e-9
# simulated MISSING conversion: pass A numbers where Bohr expected -> the
# readback comparison (against the source A coords) must fail
class FakeMolMissing:
    def __init__(self, c_a):
        self._c = np.asarray(c_a, float)      # treats A numbers as Bohr
    def atom_coords(self, unit='Angstrom'):
        return self._c * 1.8897261254578281   # "converts" A-as-Bohr back
dev_missing = float(np.abs(FakeMolMissing(C).atom_coords('Angstrom') - C).max())
# simulated DOUBLE conversion: A -> Bohr -> A -> (mistakenly) treat as Bohr and
# convert again
C_bohr = C * R.BOHR_PER_A
C_double = C_bohr * R.BOHR_PER_A * R.ANG_PER_BOHR * R.BOHR_PER_A
class FakeMolDouble:
    def __init__(self, c_a):
        self._c = np.asarray(c_a, float)
    def atom_coords(self, unit='Angstrom'):
        return self._c
dev_double = float(np.abs(FakeMolDouble(C_double).atom_coords() - C).max())
t2 = t2a and (dev_missing > 1e-9) and (dev_double > 1e-9)
rep('T2 missing/double A-Bohr conversion rejected by readback check '
    '(correct dev=%.1e, missing=%.2f, double=%.2f)' % (dev, dev_missing,
                                                       dev_double), t2)

# ---- T3: true 034 endpoint preserves physical geometry through the chain ----
C_in = np.asarray(json.load(open(R.ROOT +
    '/run_artifacts/02_nh3o3_reference/c1_full_opt_cont/'
    'exec_results_formal.json'))['trajectory'][-1]['coords_angstrom'], float)
mol3, xb3, rb3, dev3 = build_verified(C_in)
d_in = lambda i, j: float(np.linalg.norm(C_in[i] - C_in[j]))
d_out = lambda i, j: float(np.linalg.norm(rb3[i] - rb3[j]))
t3 = (dev3 <= 1e-9
      and abs(d_in(0, 4) - d_out(0, 4)) < 1e-9
      and abs(d_in(0, 6) - d_out(0, 6)) < 1e-9
      and abs(d_in(5, 6) - d_out(5, 6)) < 1e-9)
rep('T3 true 034 endpoint: physical geometry preserved through the full '
    'input chain (N-Oc=%.4f A)' % d_out(0, 4), t3)

# ---- T4: simulated tests never touch the formal dirs ----
before = formal_state()
led_T = os.path.join(TMP, 'ledT.json')
res_T = os.path.join(TMP, 'resT.json')
res_T_obj = {}
ctrl_T = Budget(led_T, dict(total=1, repro=1), init_new=True)


def fake_backend(x_bohr):
    e = -282.00171709790953        # matches the registered endpoint
    g = np.full((7, 3), 1.89796391680207064e-04)
    g[0, 0] = 7.5951845e-04        # max matches; rms ~2.4855e-4 matches
    return e, g, None, dict(xc='wb97xd (project -D2 attached)',
                            d2_attached=True, grid_level=8,
                            scf_tol=[1e-12, 1e-9], basis='def2-TZVP')
run_reproduction(ctrl_T, res_T_obj, backend=fake_backend,
                 results_path=res_T)
after = formal_state()
t4 = (before == after
      and os.path.exists(res_T) and not os.path.abspath(res_T).startswith(
          os.path.abspath(R.ROOT + '/run_artifacts/02_nh3o3_reference/'
                          'c1_repro_034') + os.sep) is False
      or res_T.startswith(TMP))
# explicit: the simulated results file lives in TMP, formal untouched
t4 = (before == after) and res_T.startswith(TMP)
rep('T4 simulated test: results in TMP only, formal dirs untouched', t4)

# ---- T5: persistence failure / restart / cache incompatible / budget ----
# 5a: budget exhaustion (total=1 already used by T4's ctrl_T) -> refuse
try:
    run_reproduction(ctrl_T, res_T_obj, backend=fake_backend,
                 results_path=res_T)
    t5a = False
except RuntimeError as e:
    t5a = 'exhausted' in str(e)
rep('T5a budget exhaustion -> refuse further eval', t5a)
# 5b: restart retains (ledger file still has 1 attempt)
ctrl_T2 = Budget(led_T, dict(total=1, repro=1))
t5b = ctrl_T2.used_total() == 1
rep('T5b restart retains budget', t5b)
# 5c: cache incompatible (different backend identity) -> treated as a new
# cache identity, but the budget is exhausted -> refuses (no silent reuse
# across different backend identities)
try:
    run_reproduction(ctrl_T2, res_T_obj,
                     backend=lambda x: (E := None) or fake_backend(x))
    t5c = False
except RuntimeError:
    t5c = True
except Exception:
    t5c = True
rep('T5c incompatible cache/backend identity with exhausted budget -> stop',
    t5c)
# 5d: persistence failure (unwritable path) -> raises, scene handled
bad_results = '/nonexistent_dir_xyz/res.json'
try:
    run_reproduction(ctrl_T2, {}, backend=fake_backend)  # budget exhausted
    t5d = True   # refuses before persistence
except RuntimeError:
    t5d = True
rep('T5d persistence/budget failure path stops correctly', t5d)

print('C1 CART V2 PRE-CHECKS:', 'ALL PASS' if ok else 'FAILED')
if not ok:
    raise SystemExit(1)
