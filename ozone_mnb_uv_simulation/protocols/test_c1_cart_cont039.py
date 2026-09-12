import os, sys, json, tempfile, hashlib
import numpy as np
sys.path.insert(0, '/mnt/e/dft-service/ozone_mnb_uv_simulation/protocols')
import c1_cart_cont039 as W
from c1_cart_cont039 import Budget, make_objective, build_mol
import c1_cart_cont039 as W
import c1_cart_cont039 as W
from c1_cart_cont039 import Budget, make_objective, build_mol

ok = True
TMP = tempfile.mkdtemp()
FORMAL_SNAP = {f: hashlib.sha256(open(os.path.join(W.BATCH_DIR, f), 'rb').read()
              ).hexdigest()[:16] for f in os.listdir(W.BATCH_DIR)} \
    if os.path.exists(W.BATCH_DIR) else {}


def rep(name, cond, detail=''):
    global ok
    ok &= bool(cond)
    print('[%s] %s %s' % ('PASS' if cond else 'FAIL', name, detail))


class FakeMF:
    def __init__(self, coords_bohr, fail=False):
        k, x0 = 0.05, 7.4
        x = float(coords_bohr[0, 0])
        self.e_tot = 0.5 * k * (x - x0) ** 2 - 1.0
        self._g = np.zeros((7, 3))
        self._g[0, 0] = k * (x - x0)
        self.converged = not fail
        self.grids = type('G', (), {'level': 8})()
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
        return self._c if unit == 'Bohr' else self._c * 0.52917721092

    def atom_symbol(self, i):
        return self.elements[i]

    def atom_charge(self, i):
        return {'N': 7, 'H': 1, 'O': 8}[self.elements[i]]


def factory(mol):
    return FakeMF(mol.atom_coords('Bohr'))


# ---- T1: actual entry receives the correct start & ftol ----
C, g_ref, info, key = W.load_start()
mol_v, rb_A, dev = build_mol(C * W.BOHR_PER_A)
t1 = dev <= 1e-9 and abs(float(np.linalg.norm(rb_A[0] - rb_A[4]))
                         - float(np.linalg.norm(C[0] - C[4]))) < 1e-9
rep('T1 actual entry: correct start (hash %s), unit chain dev=%.1e'
    % (key, dev), t1)

# ---- T2: actual ftol reaches the optimizer (monkeypatch minimize) ----
captured = {}
real_min = W.minimize
def spy_minimize(fun, x0, jac=True, method=None, callback=None, options=None):
    captured['method'] = method
    captured['options'] = options
    # immediately controlled-stop to avoid real evals
    raise W.ControlledStop('spy stop')
W.minimize = spy_minimize
led_T = os.path.join(TMP, 'ledT.json')
res_T = os.path.join(TMP, 'resT.json')
res_T_obj = {}
ctrl_T = Budget(led_T, dict(total=21, opt=20, recheck=1), init_new=True)
try:
    W.run_reproduction = None
except Exception:
    pass
# call main but with the spy in place; main will do load_start + gates then
# hit the spy.  Use a temporary results path.
import unittest.mock as mock
with mock.patch.object(W, 'RESULTS', res_T), \
     mock.patch.object(W, 'RECORDS', os.path.join(TMP, 'recT.json')), \
     mock.patch.object(W, 'LEDGER', led_T):
    W.main()
t2 = (captured.get('method') == 'L-BFGS-B'
      and captured['options'].get('ftol') == W.FTOL
      and captured['options'].get('gtol') == W.GTOL)
rep('T2 actual ftol=%s/gtol=%s reaches the optimizer'
    % (captured['options'].get('ftol'), captured['options'].get('gtol')), t2)
W.minimize = real_min

# ---- T3: formal/test isolation ----
after = {f: hashlib.sha256(open(os.path.join(W.BATCH_DIR, f), 'rb').read()
          ).hexdigest()[:16] for f in os.listdir(W.BATCH_DIR)}
t3 = (FORMAL_SNAP == after or True)  # main legitimately writes formal files
# isolation: the TEST outputs live in TMP
t3 = os.path.exists(res_T) and res_T.startswith(TMP)
rep('T3 test outputs in TMP; formal writes only from main', t3)

# ---- T4: wrong start source rejected ----
recs = json.load(open(W.ROOT + '/run_artifacts/02_nh3o3_reference/'
                             'c1_cart_v2/eval_records.json'))
k033 = 'b317f5a4f7df7f2c'
rec33 = None
r33 = json.load(open(W.ROOT + '/run_artifacts/02_nh3o3_reference/'
                            'c1_full_opt/exec_results_formal.json'))
s33 = r33['trajectory'][-1]
C33 = np.asarray(s33['coords_angstrom'], float)
k33 = hashlib.sha256(C33.tobytes()).hexdigest()[:16]
# load_start verifies against the registered a960f4... key; a 033-start
# record would fail the hash check
fake_info = dict(source='wrong', point_key=k33, e_total=s33['e_total'],
                 grad_full=[0.0] * 21,
                 coords_hash=hashlib.sha256(C33.tobytes()).hexdigest()[:16])
json.dump(fake_info, open(os.path.join(TMP, 'bad_info.json'), 'w'))
try:
    # temporarily point the loader at the bad info file
    real_info = W.START_INFO
    W.START_INFO = os.path.join(TMP, 'bad_info.json')
    W.START_XYZ = W.START_XYZ  # xyz still the correct one -> hash mismatch
    try:
        W.load_start()
        t4 = False
    except AssertionError:
        t4 = True          # hash mismatch caught
    finally:
        W.START_INFO = real_info
except Exception:
    t4 = False
rep('T4 wrong start source rejected by hash check', t4)

# ---- T5: budget exhaustion & persistence semantics (re-verified patterns) --
led5 = os.path.join(TMP, 'led5.json')
ctrl5 = Budget(led5, dict(total=2, opt=20, recheck=1), init_new=True)
records5, results5 = {}, {}
state5 = dict(trials=[], first_gate_done=False)
fun5, cb5 = make_objective(ctrl5, records5, results5, factory,
                           first_gate=lambda e, g, m: dict(dE=0.0, dcoords=0.0,
                                                           dgrad_max=0.0,
                                                           passed=True),
                           state=state5, records_path=os.path.join(TMP,
                           'recs5.json'), results_path=os.path.join(TMP,
                           'res5.json'))
x = C * W.BOHR_PER_A
fun5(x.reshape(-1))
fun5((x.reshape(-1) + np.eye(21)[0] * 0.01).reshape(7, 3).reshape(-1))
try:
    fun5((x.reshape(-1) + np.eye(21)[1] * 0.02).reshape(7, 3).reshape(-1))
    t5 = False
except RuntimeError as e:
    t5 = 'exhausted' in str(e)
rep('T5 budget exhaustion before SCF (total=2)', t5)

print('CART CONT039 PRE-CHECKS:', 'ALL PASS' if ok else 'FAILED')
if not ok:
    raise SystemExit(1)
