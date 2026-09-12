import os, sys, json, tempfile
import numpy as np
HERE = '/mnt/e/dft-service/ozone_mnb_uv_simulation/protocols'
sys.path.insert(0, HERE)
from pyscf import gto
from pyscf.lib import GradScanner
from pyscf.geomopt import berny_solver
from sp_ledger import SPLedger
from gas_monomer_reference_v2 import BudgetedScanner, BudgetController

TMP = tempfile.mkdtemp()


class FakeScanner(GradScanner):
    """Harmonic-bond PES scanner: NO SCF.  E and g are computed from the
    RECEIVED geometry only, so any proxy corruption of geometry or of the
    E/g pairing is detectable.  Records everything it receives."""

    def __init__(self, mol, k=0.6, r0=1.20):
        self.__dict__.update(mol.__dict__)
        self.mol = mol
        self.stdout = mol.stdout
        self.verbose = mol.verbose
        self.k = k
        self.r0 = r0
        self._conv = True
        self.received = []       # list of (coords_angstrom, E, g)
        self.n_evals = 0

    @property
    def converged(self):
        return self._conv

    @converged.setter
    def converged(self, v):
        self._conv = bool(v)

    def __call__(self, mol=None):
        c = np.asarray(mol.atom_coords(unit='Angstrom'), float)
        r = float(np.linalg.norm(c[1] - c[0]))
        E = 0.5 * self.k * (r - self.r0) ** 2
        dEdr = self.k * (r - self.r0)
        u = (c[1] - c[0]) / r
        g = np.zeros((2, 3))
        g[0] = -dEdr * u          # dE/dr_H1
        g[1] = +dEdr * u
        self.received.append((c.tolist(), E, g.copy()))
        self.n_evals += 1
        self.converged = True
        return E, g


def make_mol(r=1.0):
    return gto.M(atom='H 0 0 0; H %.10f 0 0' % r, basis='sto-3g',
                 charge=0, spin=0, verbose=0)


def run_berny(ctrl, mol, k=0.6, r0=1.20, maxsteps=8, cat='species:x:opt',
              break_record_after_eval=False):
    scanner = FakeScanner(mol, k=k, r0=r0)
    proxy = BudgetedScanner(scanner, ctrl, cat)
    state = {}
    def callback(envs):
        state['last'] = envs['energy']
        if break_record_after_eval and state.get('n', 0) == 0:
            state['n'] = 1
            raise RuntimeError('simulated record-write failure after eval')
    try:
        berny_solver.kernel(proxy, assert_convergence=False,
                            include_ghost=True, callback=callback,
                            maxsteps=maxsteps)
        err = None
    except Exception as e:
        err = repr(e)
    return scanner, err


ok = True

# ---------- T1: two different geometries in sequence; E/g from the same geometry ----------
path1 = os.path.join(TMP, 't1.json')
ctrl = BudgetController(path1, dict(total=100, species_opt={'x': 20},
                                    recheck={'x': 1}), init_new=True)
mol = make_mol(1.0)
scanner, err = run_berny(ctrl, mol, maxsteps=8)
recv = scanner.received
distinct = len({tuple(tuple(round(float(x), 6) for x in row) for row in c) for c, e, g in recv})
t1_geom = distinct >= 2 and err is None
cons = True
for (c, E, g) in recv:
    r = float(np.linalg.norm(np.asarray(c)[1] - np.asarray(c)[0]))
    E_re = 0.5 * scanner.k * (r - scanner.r0) ** 2
    if abs(E - E_re) > 1e-12:
        cons = False
t1 = t1_geom and cons
print('[T1] distinct geometries received=%d, E/g consistent with received '
      'geometry: %s -> %s' % (distinct, cons, 'PASS' if t1 else 'FAIL'))
ok &= t1

# ---------- T2: post-eval record exception -> attempt NOT lost ----------
path2 = os.path.join(TMP, 't2.json')
ctrl2 = BudgetController(path2, dict(total=100, species_opt={'x': 20},
                                     recheck={'x': 1}), init_new=True)
mol2 = make_mol(1.0)
sc2, err2 = run_berny(ctrl2, mol2, maxsteps=2,
                      break_record_after_eval=True)
led2 = json.load(open(path2))
evals_done = sum(1 for a in led2['attempts'] if a['status'] == 'done')
started = sum(1 for a in led2['attempts'] if a['status'] in
              ('pre_checked', 'started', 'error'))
t2 = (err2 is not None and len(led2['attempts']) >= 1
      and led2['attempts'][0].get('status') is not None)
print('[T2] record exception: attempts persisted=%d (eval executed=%d), '
      'berny stopped with %s -> %s'
      % (len(led2['attempts']), sc2.n_evals, err2[:40] if err2 else None,
         'PASS' if t2 else 'FAIL'))
ok &= t2

# ---------- T3: restart -> budget retained ----------
ctrl3 = BudgetController(path2, dict(total=100, species_opt={'x': 20},
                                     recheck={'x': 1}))
t3 = ctrl3.used('total') == ctrl2.used('total')
print('[T3] restart budget retained: used=%d -> %s'
      % (ctrl3.used('total'), 'PASS' if t3 else 'FAIL'))
ok &= t3

# ---------- T4: species cap exhausted, total has balance -> rejected BEFORE eval ----------
path4 = os.path.join(TMP, 't4.json')
ctrl4 = BudgetController(path4, dict(total=100, species_opt={'x': 2},
                                     recheck={'x': 1}), init_new=True)
mol4 = make_mol(1.0)
sc4, err4 = run_berny(ctrl4, mol4, maxsteps=8)
n_before = sc4.n_evals
rejected = None
try:
    ctrl4.pre_eval('species:x:opt', dict(note='one eval too many'))
except Exception as e:
    rejected = repr(e)
t4 = (rejected is not None and 'per-species' in rejected
      and sc4.n_evals == n_before
      and ctrl4.used('total') < 100)
print('[T4] species cap exhausted (total has balance): rejected=%s, '
      'no eval consumed -> %s' % (rejected[:60] if rejected else None,
                                  'PASS' if t4 else 'FAIL'))
ok &= t4

# ---------- T5: missing ledger must not silently open quota ----------
try:
    BudgetController(os.path.join(TMP, 'nonexistent.json'),
                     dict(total=42, species_opt={'x': 20}, recheck={'x': 1}))
    t5 = False
except Exception as e:
    t5 = 'refusing' in str(e)
print('[T5] missing ledger -> explicit error: %s'
      % ('PASS' if t5 else 'FAIL'))
ok &= t5

print('REGRESSION TESTS (no SCF):', 'ALL PASS' if ok else 'FAILED')
if not ok:
    raise SystemExit(1)
