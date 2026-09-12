import os, sys, json, tempfile
import numpy as np
sys.path.insert(0, '/mnt/e/dft-service/ozone_mnb_uv_simulation/protocols')
import c1_scan_exec_v2 as E
from c1_scan_exec_v2 import EvaluationFailed

BOHR = E.BOHR_PER_A
ok = True
results = []

def rep(name, cond, detail=''):
    global ok
    ok &= bool(cond)
    results.append((name, cond))
    print('[%s] %s %s' % ('PASS' if cond else 'FAIL', name, detail))


# ================= T1: unit conversion with REAL Bohr/A factor =================
# Energy defined in Bohr coords: E = 0.5*k*(x_B - x0_B)^2 ; x_B = x_A * BOHR.
kk, x0_B = 0.07, 9.0 * BOHR          # x0 at 9 A (in Bohr)

class BohrBackend:
    """Backend whose energy is DEFINED in Bohr coordinates; receives coords
    in Angstrom from the entry and converts internally."""
    def __init__(self, coords_A, fail=False):
        x_B = float(coords_A[0, 0]) * BOHR
        self.e_total = 0.5 * kk * (x_B - x0_B) ** 2 - 1.0
        g = np.zeros((7, 3))
        g[0, 0] = kk * (x_B - x0_B)          # dE/dx_B  [Eh/Bohr]
        self._g = g
        self.converged = not fail
        self.grids = type('G', (), {'level': E.GRID_LEVEL})()
        self.conv_tol = 1e-12
        self.conv_tol_grad = 1e-9

    @property
    def _has_full_d2(self):
        return True

    def kernel(self):
        return self.e_tot

    def nuc_grad_method(self):
        outer = self
        class _G:
            def kernel(self_):
                return outer._g
        return _G()


def bohr_factory(coords_A):
    return None, BohrBackend(coords_A)


# entry-side conversion under test
g_proj, dE_ds_A = E.dE_ds_A_from_projection.__wrapped__(None, None) \
    if False else (None, None)
# direct use of the two helpers on a synthetic gradient:
g_fake = np.zeros((7, 3))
x_A = 4.0
x_B = x_A * BOHR
g_fake[0, 0] = kk * (x_B - x0_B)
g_proj, dE_ds_A = E.dE_ds_A_from_projection(g_fake, np.array([1.0, 0.0, 0.0]))
# independent FD in ANGSTROM coords (energy evaluated via the Bohr definition)
hA = 1e-5
E_plus = 0.5 * kk * ((x_A + hA) * BOHR - x0_B) ** 2 - 1.0
E_minus = 0.5 * kk * ((x_A - hA) * BOHR - x0_B) ** 2 - 1.0
dE_ds_A_fd = (E_plus - E_minus) / (2 * hA)
rep('T1a dE/ds_A = g_proj x Bohr_per_A (independent FD in A)',
    abs(dE_ds_A - dE_ds_A_fd) / abs(dE_ds_A_fd) < 1e-6,
    'g_proj*1.8897=%.9f vs FD(A)=%.9f' % (dE_ds_A, dE_ds_A_fd))
# the old (dividing) implementation would give dE_ds_A/1.8897^2 -> detectably wrong
rep('T1b dividing implementation is detectably wrong',
    abs(dE_ds_A / BOHR ** 2 - dE_ds_A_fd) / abs(dE_ds_A_fd) > 0.5)

# ================= T2: local dd/ds vs geometry root-finding FD =================
from c1_scan_geometry_v2 import load_monomers
NH3, O3 = load_monomers()
from c1_scan_geometry_v2 import build_scan_geometries
m = build_scan_geometries(NH3, O3, [3.0])
C = np.asarray(m['points'][0]['coords'], float)
u = np.array([1.0, 0.0, 0.0])
t_rows = [5, 6]
dd_ds = E.dd_ds_local(C, u, t_rows)
ds_dd = E.ds_dd_from_local(dd_ds)
# FD via the geometry root-finder: shift N by +ds along u, rebuild d_target
h_s = 1e-6
N = C[0] + h_s * u
d_plus = float(np.mean([np.linalg.norm(N - C[j]) for j in t_rows]))
N = C[0] - h_s * u
d_minus = float(np.mean([np.linalg.norm(N - C[j]) for j in t_rows]))
dd_ds_fd = (d_plus - d_minus) / (2 * h_s)
rep('T2 local dd/ds matches small-step geometry FD',
    abs(dd_ds - dd_ds_fd) / abs(dd_ds_fd) < 1e-9,
    'local=%.12f FD=%.12f' % (dd_ds, dd_ds_fd))

# ================= T3: standard XYZ reader (independent cross-check) ==========
import tempfile
tmp = tempfile.mkdtemp()
p3 = os.path.join(tmp, 't.xyz')
E.write_standard_xyz(p3, E.SYMS_EXPECT, C, 'test comment line')
s1, C1 = E.read_xyz_primary(p3)
s2, C2 = E.read_xyz_independent(p3)
rep('T3 standard XYZ: comment line + both readers agree',
    abs(C1 - C2).max() < 1e-12 and abs(C1 - C).max() < 1e-12)
# negative: 8-atom file must be rejected
E.write_standard_xyz(os.path.join(tmp, 'bad.xyz'), ['N'] * 8,
                     np.zeros((8, 3)), 'bad')
try:
    E.read_xyz_primary(os.path.join(tmp, 'bad.xyz'))
    neg = False
except AssertionError:
    neg = True
rep('T3b non-7-atom XYZ rejected', neg)

# ================= T4: budget totals, sharing, restart, exception =================
tmp2 = tempfile.mkdtemp()
bp = os.path.join(tmp2, 'b.json')
ctrl = E.Budget(bp, dict(total=2, scan=6), init_new=True)
ctrl.pre_eval('scan', dict(i=1)); ctrl.post_eval(0, dict(e=1))
ctrl.pre_eval('scan', dict(i=2)); ctrl.post_eval(1, dict(e=1))
# third: scan cap (6) has room but TOTAL (2) exhausted -> rejected before eval
try:
    ctrl.pre_eval('scan', dict(i=3))
    t4a = False
except RuntimeError as e:
    t4a = 'TOTAL' in str(e)
rep('T4a total cap 2 exhausted -> third rejected before eval', t4a)
# different category shares the same total budget
try:
    ctrl.pre_eval('other', dict(i=4))
    t4b = False
except RuntimeError:
    t4b = True
rep('T4b other category shares total budget', t4b)
# exception during eval: attempt + error retained
ctrl2 = E.Budget(os.path.join(tmp2, 'b2.json'), dict(total=10, scan=10),
                 init_new=True)
idx = ctrl2.pre_eval('scan', dict(note='boom'))
ctrl2.mark_error(idx, 'simulated failure')
l2 = json.load(open(os.path.join(tmp2, 'b2.json')))
rec = l2['attempts'][idx]
rep('T4c exception keeps attempt + error record',
    rec['status'] == 'error' and 'simulated failure' in
    rec['errors'][0]['error'])
# restart retains
ctrl2r = E.Budget(os.path.join(tmp2, 'b2.json'), dict(total=10, scan=10))
rep('T4d restart retains counts',
    ctrl2r.used_total() == 1 and ctrl2r.used_cat('scan') == 1)
# missing ledger refuses
try:
    E.Budget(os.path.join(tmp2, 'missing.json'), dict(total=10, scan=10))
    t4e = False
except RuntimeError as e:
    t4e = 'refusing' in str(e)
rep('T4e missing ledger refuses silent quota', t4e)

# ================= T5: real entry end-to-end (simulated backend) =============
tmp3 = tempfile.mkdtemp()
led3 = os.path.join(tmp3, 'led.json')
# fail_at must match the N x-coordinate at d=3.2 (=3.6551), not the
# contact-distance parameter
sim = E.sim_backend_factory(fail_at=3.6551, harmonic=(0.05, 4.5))
ctrl3 = E.Budget(led3, dict(total=6, scan=6), init_new=True)
res3 = {}
try:
    E.run_entry(ctrl3, sim, execute=True, results=res3)
except EvaluationFailed:
    pass
done_keys = [k for k, v in res3.items() if isinstance(v, dict)]
stopped = 'last_error' in res3
n_attempt = ctrl3.used_total()
# expected: 2.6/2.8/3.0 done (3 attempts); d=3.2 SCF fails (4th attempt);
# hard stop -> 3.6/4.2 NOT attempted
t5 = (stopped and len(done_keys) == 3 and n_attempt == 4)
rep('T5a end-to-end: 3 done, fail at d=3.2 -> hard stop before 3.6/4.2', t5,
    'done=%d attempts=%d' % (len(done_keys), n_attempt))
# per-point full record present
r0 = next(v for v in res3.values() if isinstance(v, dict))
full = all(kk in r0 for kk in ('coords_computed', 'grad_full', 'e_d2_analytic',
                               'e_dft_part', 'config_readback', 'dE_ds_Eh_per_A',
                               'dd_ds', 'ds_dd', 'dE_dd_Eh_per_A'))
rep('T5b per-point record has full gradient/coords/config/components', full)
# restart with a FIXED backend: remaining points evaluated, done reused
# restart with a FIXED backend: d=3.2 re-evaluated, 3.6/4.2 evaluated; the
# failed d=3.2 attempt has consumed budget, so the LAST point (4.2) is
# honestly rejected by the exhausted total cap (attempts = 6 = cap)
sim_ok = E.sim_backend_factory(harmonic=(0.05, 4.5))
res3b = dict(res3)
try:
    E.run_entry(ctrl3, sim_ok, execute=True, results=res3b)
except RuntimeError:
    pass          # honest budget exhaustion at 4.2
n_attempt_after = ctrl3.used_total()
done_after = sum(1 for k, v in res3b.items() if isinstance(v, dict))
t5c = (n_attempt_after == 6 and done_after == 5)
rep('T5c restart: attempts=6 (cap), done=5, 4.2 honestly rejected by '
    'exhausted budget', t5c)
# validation-only mode does not evaluate
led4 = os.path.join(tmp3, 'led4.json')
ctrl4 = E.Budget(led4, dict(total=6, scan=6), init_new=True)
res4 = {}
E.run_entry(ctrl4, sim_ok, execute=False, results=res4)
rep('T5d validation-only mode: zero evaluations', ctrl4.used_total() == 0)

print('SCAN EXEC V3 TESTS:', 'ALL PASS' if ok else 'FAILED')
if not ok:
    raise SystemExit(1)
