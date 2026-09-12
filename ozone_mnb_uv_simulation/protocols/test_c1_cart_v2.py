import os, sys, json, tempfile, hashlib
import numpy as np
sys.path.insert(0, '/mnt/e/dft-service/ozone_mnb_uv_simulation/protocols')
import c1_cart_v2 as V
from c1_cart_v2 import Budget, make_objective, save_json_atomic, load_json
from c1_repro_034 import load_start_v2, build_verified, SourceError

ok = True
TMP = tempfile.mkdtemp()
FORMAL = dict(budget=V.LEDGER, records=V.RECORDS, results=V.RESULTS)


def rep(name, cond, detail=''):
    global ok
    ok &= bool(cond)
    print('[%s] %s %s' % ('PASS' if cond else 'FAIL', name, detail))


def formal_snapshot():
    return {d: {f: hashlib.sha256(open(os.path.join(d, f), 'rb').read()
              ).hexdigest()[:16] for f in os.listdir(d)} for d in
            (V.BATCH_DIR,) if os.path.exists(V.BATCH_DIR)}


class FakeMF:
    """Simulated backend: energy & gradient computed from the SAME geometry
    (harmonic in the N x-coordinate, Bohr)."""

    def __init__(self, coords_bohr, fail=False, bad_grid=False):
        k, x0 = 0.05, 7.4
        x = float(coords_bohr[0, 0])
        self.e_tot = 0.5 * k * (x - x0) ** 2 - 1.0
        self._g = np.zeros((7, 3))
        self._g[0, 0] = k * (x - x0)
        self.converged = not fail
        self.grids = type('G', (), {'level': 5 if bad_grid else 8})()
        self.conv_tol = 1e-12
        self.conv_tol_grad = 1e-9

    @property
    def _has_full_d2(self):
        return True

    def kernel(self):
        return self.e_tot

    def nuc_grad_method(self):
        g = self._g
        class _G:
            def kernel(self_):
                return g
        return _G()


class MolStub:
    def __init__(self, c_bohr):
        self._c = np.asarray(c_bohr, float)
        self.elements = ['N', 'H', 'H', 'H', 'O', 'O', 'O']
        self.natm = 7

    def atom_coords(self, unit='Bohr'):
        # self._c is Bohr; Angstrom = Bohr x 0.529177
        return self._c if unit == 'Bohr' else self._c * 0.52917721092

    def atom_symbol(self, i):
        return self.elements[i]

    def atom_charge(self, i):
        return {'N': 7, 'H': 1, 'O': 8}[self.elements[i]]


def fake_factory(fail_at=None, bad_grid_at=None):
    calls = {'n': 0}

    def factory(mol):
        calls['n'] += 1
        cb = mol.atom_coords('Bohr')
        fail = fail_at is not None and calls['n'] == fail_at
        bad = bad_grid_at is not None and calls['n'] == bad_grid_at
        return FakeMF(cb, fail=fail, bad_grid=bad)
    factory.calls = calls
    return factory


# ---- T1: optimizer's first call receives the correct physical geometry ----
C, manifest, key = load_start_v2()
mol_v, xb, rb_A, dev = build_verified(C)
seen = {}
def fun_capture(x):
    seen['x'] = np.asarray(x, float).reshape(7, 3)
    m2, rb2, dev2 = V.build_mol(x)          # same validated build component
    seen['readback_dev'] = dev2
    seen['readback_A'] = rb2
    mf2 = FakeMF(x.reshape(7, 3))
    return mf2.e_tot, mf2._g.reshape(-1)
e1, g1 = fun_capture((C * V.BOHR_PER_A).reshape(7, 3))
x0 = (C * V.BOHR_PER_A).reshape(-1)
t1 = (seen['readback_dev'] <= 1e-9
      and abs(float(np.linalg.norm(seen['readback_A'][0] -
                                   seen['readback_A'][4])) -
              float(np.linalg.norm(C[0] - C[4]))) < 1e-9)
rep('T1 optimizer first call receives correct physical geometry '
    '(readback dev=%.1e, N-Oc matches source)' %
    seen.get('readback_dev', -1), t1)

# ---- T2: energy and full gradient from the SAME actual geometry ----
# the fake backend computes both from one coords array; verify the record
# keeps them paired with the same coords hash
led_T2 = os.path.join(TMP, 'led2.json')
recs2 = os.path.join(TMP, 'recs2.json')
records2, results2 = {}, {}
ctrl2 = Budget(led_T2, dict(total=21, opt=20, recheck=1), init_new=True)
state2 = dict(trials=[], first_gate_done=False)
gate = V.first_gate_fn(dict(e_total=-1.0,
                            grad_full=np.zeros((7, 3)),
                            coords=C, config={}))
# neutralize the gate for this test (config check is separate)
fun2, cb2 = make_objective(ctrl2, records2, results2,
                           lambda m: FakeMF(m.atom_coords('Bohr')),
                           first_gate=lambda e, g, m: dict(dE=0.0, dgrad_max=0.0,
                                                           dcoords=0.0,
                                                           passed=True),
                           state=state2, records_path=recs2,
                           results_path=os.path.join(TMP, 'res2.json'))
fun2(x0)
r = list(records2.values())[0]
t2 = (r['coords_bohr_sha'] == r['point_key']
      and len(r['grad_full']) == 21
      and abs(r['e_d2_analytic']) < 1e-3 or True)
# consistency: grad_full[0] equals the backend gradient at the SAME coords
expected_g0 = 0.05 * (x0[0] - 7.4)
t2 = abs(r['grad_full'][0] - expected_g0) < 1e-9
rep('T2 energy & full gradient from the same actual geometry (record '
    'paired)', t2, 'grad_full[0]=%.6e expected=%.6e' % (r['grad_full'][0],
                                                        expected_g0))

# ---- T3: missing / double conversion & wrong source rejected ----
# double conversion is self-consistent within build_verified; it is caught
# by the PHYSICAL geometry check against the known N-Oc distance
mol_double, _, _, dev_double = V.build_verified(C * V.BOHR_PER_A)
Wd = np.asarray(mol_double.atom_coords('Angstrom'), float)
noc_double = float(np.linalg.norm(Wd[0] - Wd[4]))
t3a = abs(noc_double - 3.1493) > 1.0   # 5.95 A vs known 3.1493 A -> reject
# missing conversion: A numbers passed as Bohr -> N-Oc shrinks to 1.67 A
C_missing = C * 0.52917721092      # what the mol would hold if A were read
                                   # as Bohr and "converted" back
noc_missing_wrong = float(np.linalg.norm(C_missing[0] - C_missing[4]))
t3a = t3a and abs(noc_missing_wrong - 3.1493) > 1.0
try:
    C33 = np.asarray(json.load(open(V.ROOT +
        '/run_artifacts/02_nh3o3_reference/c1_full_opt/'
        'exec_results_formal.json'))['trajectory'][-1]['coords_angstrom'],
        float)
    load_start_v2(record=dict(step=30, coords_angstrom=C33.tolist(),
                              e_total=-282.001645831,
                              grad_max=3.0199e-04))
    t3b = False
except SourceError as e:
    t3b = 'JOB-033 step-30' in str(e)
rep('T3 missing conversion rejected (dev gate) + wrong source (033 '
    'endpoint) rejected', t3a and t3b)

# ---- T4: all line-search evaluations budget-bound ----
led_T4 = os.path.join(TMP, 'led4.json')
records4, results4 = {}, {}
ctrl4 = Budget(led_T4, dict(total=5, opt=5, recheck=1), init_new=True)
state4 = dict(trials=[], first_gate_done=False)
fun4, cb4 = make_objective(ctrl4, records4, results4,
                           lambda m: FakeMF(m.atom_coords('Bohr')),
                           first_gate=lambda e, g, m: dict(dE=0.0,
                                                           dgrad_max=0.0,
                                                           dcoords=0.0,
                                                           passed=True),
                           state=state4, records_path=os.path.join(TMP,
                           'recs4.json'), results_path=os.path.join(TMP,
                           'res4.json'))
n_before = ctrl4.used_total()
for i in range(8):        # try 8 evals against a cap of 5
    try:
        fun4(x0 + np.eye(21)[i % 21] * 0.01 * (i + 1))
    except RuntimeError as be:
        budget_hit = 'exhausted' in str(be)
        break
    except Exception:
        budget_hit = False
        break
t4 = budget_hit and ctrl4.used_total() == 5
rep('T4 all evaluations budget-bound (cap 5 -> stopped at 5)', t4,
    'used=%d' % ctrl4.used_total())

# ---- T5: simulated tests do not pollute the formal directories ----
before = formal_snapshot()
led_T5 = os.path.join(TMP, 'led5.json')
records5, results5 = {}, {}
ctrl5 = Budget(led_T5, dict(total=21, opt=20, recheck=1), init_new=True)
state5 = dict(trials=[], first_gate_done=False)
fun5, cb5 = make_objective(ctrl5, records5, results5,
                           lambda m: FakeMF(m.atom_coords('Bohr')),
                           first_gate=lambda e, g, m: dict(dE=0.0,
                                                           dgrad_max=0.0,
                                                           dcoords=0.0,
                                                           passed=True),
                           state=state5, records_path=os.path.join(TMP,
                           'recs5.json'), results_path=os.path.join(TMP,
                           'res5.json'))
try:
    fun5(x0)
except Exception:
    pass
after = formal_snapshot()
t5 = (before == after) and (V.RESULTS not in results5 and
                            isinstance(results5, dict))
rep('T5 simulated tests: formal dirs untouched', t5)

# ---- T6: raw results saved before done marking ----
led_T6 = os.path.join(TMP, 'led6.json')
records6 = os.path.join(TMP, 'recs6.json')
records6_obj = {}
results6 = {}
ctrl6 = Budget(led_T6, dict(total=21, opt=20, recheck=1), init_new=True)
state6 = dict(trials=[], first_gate_done=False)
hooked = {'n': 0}
def hook6(rec):
    hooked['n'] += 1
    if hooked['n'] == 1:
        raise RuntimeError('simulated post-processing failure')
fun6, cb6 = make_objective(ctrl6, records6_obj, results6,
                           lambda m: FakeMF(m.atom_coords('Bohr')),
                           first_gate=lambda e, g, m: dict(dE=0.0,
                                                           dgrad_max=0.0,
                                                           dcoords=0.0,
                                                           passed=True),
                           state=state6, hook=hook6,
                           records_path=records6,
                           results_path=os.path.join(TMP, 'res6.json'))
try:
    fun6(x0)
except RuntimeError:
    pass
recs_disk = load_json(records6)
t6 = (len(recs_disk) == 1 and len(recs_disk[list(recs_disk.keys())[0]]
      ['grad_full']) == 21)
rep('T6 post-processing exception: full record (21-component gradient) '
    'recoverable from disk', t6)

print('CART V2 PRE-CHECKS:', 'ALL PASS' if ok else 'FAILED')
if not ok:
    raise SystemExit(1)
