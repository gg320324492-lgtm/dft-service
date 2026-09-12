import os, sys, json, tempfile
import numpy as np
sys.path.insert(0, '/mnt/e/dft-service/ozone_mnb_uv_simulation/protocols')
import c1_cart_opt as M
from c1_cart_opt import Budget, make_objective, save_json_atomic, load_json

ok = True
TMP = tempfile.mkdtemp()


def rep(name, cond, detail=''):
    global ok
    ok &= bool(cond)
    print('[%s] %s %s' % ('PASS' if cond else 'FAIL', name, detail))


class FakeMF:
    """Simulated harmonic backend: E = 0.5k(x-x0)^2 along the N x-coordinate
    (Bohr), true Bohr gradient; can fail or have a wrong grid."""

    def __init__(self, coords_bohr, fail=False, bad_grid=False):
        k, x0 = 0.05, 7.5
        x = float(coords_bohr[0, 0])
        self.e_tot = 0.5 * k * (x - x0) ** 2 - 1.0
        self._g = np.zeros((7, 3))
        self._g[0, 0] = k * (x - x0)
        self.converged = not fail
        self.grids = type('G', (), {'level': 5 if bad_grid else 8})()
        self._fail = fail
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
    def __init__(self, c):
        self._c = np.asarray(c, float)
        self.elements = ['N', 'H', 'H', 'H', 'O', 'O', 'O']
        self.natm = 7

    def atom_coords(self, unit='Bohr'):
        return self._c * 1.8897261254578281 if unit == 'Bohr' else self._c

    def atom_symbol(self, i):
        return self.elements[i]

    def atom_charge(self, i):
        return {'N': 7, 'H': 1, 'O': 8}[self.elements[i]]

    def fake_factory(fail_at=None, bad_grid_at=None):
        calls = {'n': 0}
        def factory(coords_bohr):
            calls['n'] += 1
            fail = fail_at is not None and calls['n'] == fail_at
            bad = bad_grid_at is not None and calls['n'] == bad_grid_at
            return MolStub(coords_bohr), FakeMF(coords_bohr, fail=fail,
                                                bad_grid=bad)
        return factory


def fake_factory(fail_at=None, bad_grid_at=None):
    calls = {'n': 0}

    def factory(coords_bohr):
        calls['n'] += 1
        fail = fail_at is not None and calls['n'] == fail_at
        bad = bad_grid_at is not None and calls['n'] == bad_grid_at
        return MolStub(coords_bohr), FakeMF(coords_bohr, fail=fail,
                                            bad_grid=bad)
    return factory


# ---------- T-A: post-processing exception -> record still recoverable ----------
led = os.path.join(TMP, 'ledA.json')
recs_path = os.path.join(TMP, 'recA.json')
records = load_json(recs_path)
results = {}
ctrl = Budget(led, dict(total=21, opt=20, recheck=1), init_new=True)

calls = {'n': 0}
def hook_boom(rec):
    calls['n'] += 1
    if calls['n'] == 1:
        raise RuntimeError('simulated post-processing failure')

fun = make_objective(ctrl, records, results,
                     lambda c: (MolStub(c), FakeMF(c)), hook=hook_boom,
                     records_path=recs_path)
x0 = np.zeros(21); x0[0] = 8.0
try:
    fun(x0)
    tA1 = False
except RuntimeError as e:
    tA1 = 'post-processing' in str(e)
# the record WAS persisted before the hook (safe-write first)
on_disk = load_json(recs_path)
k0 = list(on_disk.keys())[0]
rec0 = on_disk[k0]
tA1 = tA1 and (len(rec0['grad_full']) == 21
               and rec0['status'] == 'evaluated'
               and np.isfinite(rec0['e_total']))
rep('T-A1 post-processing exception: full record still on disk & readable',
    tA1)
# ledger attempt NOT marked done (crash before post_eval)
ledA = json.load(open(led))
n_done = sum(1 for a in ledA['attempts'] if a['status'] == 'done')
# correct semantics: the full record WAS safely persisted before done
# marking (batch requirement), so done marking is legitimate; the
# post-processing failure happened after and the record remains readable
rep('T-A2 ledger done marked only after safe record persistence', n_done == 1)
# restart: budget retained + record reused WITHOUT new SCF
ctrl2 = Budget(led, dict(total=21, opt=20, recheck=1))
records2 = load_json(recs_path)
fun2 = make_objective(ctrl2, records2, {}, lambda c: (MolStub(c), FakeMF(c)),
                      records_path=recs_path)
e, g = fun2(x0)
tA3 = (ctrl2.used_total() == 1   # the original attempt counted; no new eval
       and abs(e - rec0['e_total']) < 1e-15
       and abs(g[0] - rec0['grad_full'][0]) < 1e-15)
rep('T-A3 restart: budget retained + verified record reused (no new SCF)',
    tA3, 'attempts=%d' % ctrl2.used_total())

# ---------- T-B: config change -> no reuse ----------
recs_path_B = os.path.join(TMP, 'recB.json')
records_B = load_json(recs_path_B)
results_B = {}
ctrl_B = Budget(os.path.join(TMP, 'ledB.json'), dict(total=21, opt=20,
                                                     recheck=1),
                init_new=True)
fun_B = make_objective(ctrl_B, records_B, results_B,
                       fake_factory(bad_grid_at=None),
                       records_path=recs_path_B)
e, g = fun_B(x0)
# tamper the stored config
k = list(records_B.keys())[0]
records_B[k]['config']['grid_level'] = 5
save_json_atomic(recs_path_B, records_B)
records_B2 = load_json(recs_path_B)
calls_before = ctrl_B.used_total()
fun_B2 = make_objective(ctrl_B, records_B2, {}, fake_factory(),
                        records_path=recs_path_B)
e2, g2 = fun_B2(x0)
tB = ctrl_B.used_total() == calls_before + 1
rep('T-B config change -> no reuse (fresh evaluation)', tB,
    'attempts %d -> %d' % (calls_before, ctrl_B.used_total()))

# ---------- T-C: budget exhaustion before eval ----------
ctrl_C = Budget(os.path.join(TMP, 'ledC.json'), dict(total=2, opt=20),
                init_new=True)
fun_C = make_objective(ctrl_C, {}, {}, fake_factory())
fun_C(x0); fun_C(x0 * 0 + np.arange(21) * 0.01)
try:
    fun_C(np.arange(21) * 0.02)
    tC = False
except M.BudgetExhausted as e:
    tC = 'TOTAL' in str(e)
rep('T-C total cap 2 -> third eval rejected before SCF', tC)

print('PERSISTENCE REGRESSION:', 'ALL PASS' if ok else 'FAILED')
if not ok:
    raise SystemExit(1)
