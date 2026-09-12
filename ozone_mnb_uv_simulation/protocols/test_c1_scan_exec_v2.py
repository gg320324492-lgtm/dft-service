import os, sys, json, tempfile
import numpy as np
sys.path.insert(0, '/mnt/e/dft-service/ozone_mnb_uv_simulation/protocols')
from c1_scan_exec_v2 import (Budget, load_verified_points, run_point,
                             dE_dd_from_projection, ScanIntegrityError,
                             EvaluationFailed, GEOM_MANIFEST)

ok = True

# ---------- T-A: hash/distance re-verification on the verified manifest ----------
m, pts = load_verified_points()
tA = len(pts) == 6
print('[T-A] manifest re-verification: %d points, hashes+distances OK -> %s'
      % (len(pts), 'PASS' if tA else 'FAIL'))
ok &= tA

# ---------- T-B: dE/dd chain factor with a simulated E(s) ----------
# simulated scan: E(s) = 0.5*a*s^2 (a in Eh/Bohr^2 of the simulation),
# d(s) = sqrt(s^2 + w^2)  ->  dE/dd = dE/ds * s/d  (chain factor)
a, w = 0.05, 1.0618
s0 = 2.8
g_proj = a * s0                      # dE/ds in simulated Eh/Bohr
ds_dd = float(np.sqrt(s0 ** 2 + w ** 2)) / s0   # analytic ds/dd = d/s
dE_dd = dE_dd_from_projection(g_proj, ds_dd, bohr_per_A=1.0)
# numeric reference: E as a function of d
f_E = lambda d: 0.5 * a * (d ** 2 - w ** 2)
h = 1e-4
d0 = float(np.sqrt(s0 ** 2 + w ** 2))
num = (f_E(d0 + h) - f_E(d0 - h)) / (2 * h)
tB = abs(dE_dd - num) / max(abs(num), 1e-300) < 1e-9
print('[T-B] dE/dd chain factor: analytic=%.9f numeric=%.9f -> %s'
      % (dE_dd, num, 'PASS' if tB else 'FAIL'))
ok &= tB

# ---------- T-C: SCF failure actually stops (simulated) ----------
tmp = tempfile.mkdtemp()
ctrl = Budget(os.path.join(tmp, 'b.json'),
              dict(total=6, scan=6), init_new=True)


class FakeMol:
    grids = dict(level=8)


class FakeMF:
    def __init__(self, conv=True, finite=True):
        self.grids = type('G', (), {'level': 8})()
        self._d2 = True
        self.converged = conv
        self._finite = finite
        self.e_tot = -1.0

    def kernel(self, **kw):
        return self.e_tot

    @property
    def _has_full_d2(self):
        return self._d2

    def nuc_grad_method(self):
        g = np.full((7, 3), 1e-4) if self._finite \
            else np.full((7, 3), np.nan)
        return type('G', (), {'kernel': lambda self_: g})()


try:
    run_point(None, FakeMF(conv=False), ctrl, 'scan', 2.6)
    tC = False
except EvaluationFailed as e:
    tC = 'SCF not converged' in str(e)
rec = ctrl.data['attempts'][-1]
tC &= (rec['status'] == 'error')
print('[T-C] SCF failure -> EvaluationFailed + attempt marked error: %s'
      % ('PASS' if tC else 'FAIL'))
ok &= tC

# ---------- T-D: non-finite actually stops ----------
try:
    run_point(None, FakeMF(conv=True, finite=False), ctrl, 'scan', 2.8)
    tD = False
except EvaluationFailed as e:
    tD = 'non-finite' in str(e)
print('[T-D] non-finite -> hard stop: %s' % ('PASS' if tD else 'FAIL'))
ok &= tD

# ---------- T-E: config mismatch actually stops ----------
class BadCfgMF(FakeMF):
    def __init__(self):
        FakeMF.__init__(self)
        self.grids = type('G', (), {'level': 5})()   # silent fallback simulation


try:
    run_point(None, BadCfgMF(), ctrl, 'scan', 3.0)
    tE = False
except EvaluationFailed as e:
    tE = 'config mismatch' in str(e)
print('[T-E] config mismatch (grid fallback) -> hard stop: %s'
      % ('PASS' if tE else 'FAIL'))
ok &= tE

# ---------- T-F: budget exhaustion raises before eval ----------
used = ctrl.used('scan')
caps_left = 6 - used
for i in range(caps_left + 1):
    try:
        ctrl.pre_eval('scan', dict(i=i))
    except RuntimeError as e:
        tF = 'exhausted' in str(e)
        break
else:
    tF = False
print('[T-F] budget exhaustion raises (remaining evals not silently '
      'extended): %s' % ('PASS' if tF else 'FAIL'))
ok &= tF

print('SCAN EXEC V2 TESTS (no SCF):', 'ALL PASS' if ok else 'FAILED')
if not ok:
    raise SystemExit(1)
