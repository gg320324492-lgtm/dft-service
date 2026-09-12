import os, sys, json, hashlib, time
import numpy as np
HERE = '/mnt/e/dft-service/ozone_mnb_uv_simulation/protocols'
sys.path.insert(0, HERE)
from pyscf import gto
from scipy.optimize import minimize
import d2_full
from grad_factory import make_mf_d2_gr

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
BATCH_DIR = ROOT + '/run_artifacts/02_nh3o3_reference/c1_cart_opt'
os.makedirs(BATCH_DIR, exist_ok=True)
LEDGER = BATCH_DIR + '/budget_cart.json'
RECORDS = BATCH_DIR + '/eval_records.json'
RESULTS = BATCH_DIR + '/cart_opt_results.json'
CAPS = dict(total=21, opt=20, recheck=1)
SYMS = ['N', 'H', 'H', 'H', 'O', 'O', 'O']
GRID_LEVEL = 8
GMAX_RECHECK = 1e-5
GTOL = 1e-6
OPT_SETTINGS = dict(method='L-BFGS-B', gtol=GTOL, ftol=2.22e-9,
                    maxcor=10, maxiter=15000,
                    variables='all 21 Cartesian coordinates (Bohr)',
                    constraints='none; no gradient projection; no rotation '
                                'registration',
                    note='maxiter is NOT the evaluation limit; the hard '
                         'limit is the pre-eval persistent ledger')


class EvaluationFailed(RuntimeError):
    pass


class BudgetExhausted(RuntimeError):
    pass


# ==================== persistence (full-record, atomic) ====================
def save_json_atomic(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(obj, fh, indent=2, default=str)
    os.replace(tmp, path)


def load_json(path):
    if os.path.exists(path):
        return json.load(open(path))
    return {}


def point_key(x):
    return hashlib.sha256(np.asarray(x, float).tobytes()).hexdigest()[:16]


def record_sha(records, key):
    payload = json.dumps(records[key], sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ==================== budget ====================
class Budget:
    """TOTAL counts ALL actual attempts; per-category caps additional."""

    def __init__(self, path, caps, init_new=False):
        self.path = path
        self.caps = caps
        if os.path.exists(path):
            self.data = json.load(open(path))
        elif init_new:
            self.data = dict(caps=caps, attempts=[])
            self._save()
        else:
            raise RuntimeError('ledger %s missing (init_new=True to start '
                               'a new batch explicitly)' % path)
        self.data.setdefault('attempts', [])

    def used_total(self):
        return len(self.data['attempts'])

    def used_cat(self, cat):
        return sum(1 for a in self.data['attempts']
                   if a.get('category') == cat
                   and a.get('status') != 'rejected')

    def pre_eval(self, cat, meta):
        if self.used_total() + 1 > self.caps['total']:
            raise BudgetExhausted('TOTAL budget exhausted (%d/%d)'
                                  % (self.used_total(), self.caps['total']))
        if self.used_cat(cat) + 1 > self.caps[cat]:
            raise BudgetExhausted('%s budget exhausted (%d/%d)'
                                  % (cat, self.used_cat(cat), self.caps[cat]))
        self.data['attempts'].append(dict(category=cat, status='pre_checked',
                                          meta=meta))
        self._save()
        return len(self.data['attempts']) - 1

    def mark_error(self, idx, error):
        rec = self.data['attempts'][idx]
        rec['status'] = 'error'
        rec.setdefault('errors', []).append(dict(error=error))
        self._save()

    def post_eval(self, idx, fields):
        rec = self.data['attempts'][idx]
        rec.update(fields)
        rec['status'] = 'done'
        self._save()

    def find_recoverable(self, key):
        """A pre_checked attempt whose full record exists and verifies."""
        for i, a in enumerate(self.data['attempts']):
            if a.get('meta', {}).get('point_key') == key:
                return i, a
        return None, None

    def _save(self):
        tmp = self.path + '.tmp'
        with open(tmp, 'w') as fh:
            json.dump(self.data, fh, indent=2, default=str)
        os.replace(tmp, self.path)


# ==================== config ====================
def cfg_readback(mf):
    return dict(xc='wb97xd (project -D2 attached)',
                d2_attached=bool(getattr(mf, '_has_full_d2', False)),
                grid_level=int(getattr(mf.grids, 'level', -1)),
                scf_tol=[float(mf.conv_tol), float(mf.conv_tol_grad)],
                basis='def2-TZVP', charge=0, spin=0, grid_response=True,
                solvent='none (gas phase)')


def config_compatible(cfg):
    return (cfg.get('d2_attached') is True
            and cfg.get('grid_level') == GRID_LEVEL
            and cfg.get('basis') == 'def2-TZVP'
            and cfg.get('scf_tol') == [1e-12, 1e-9])


# ==================== structure analysis ====================
def structure_analysis(coords_A):
    C = np.asarray(coords_A, float)
    d = lambda i, j: float(np.linalg.norm(C[i] - C[j]))
    N = C[0]
    Hc = C[1:4].mean(axis=0)
    lp = N - Hc; lp /= np.linalg.norm(lp)
    mid = C[5:7].mean(axis=0)
    to_mid = mid - N; to_mid /= np.linalg.norm(to_mid)
    import math
    ang = float(math.degrees(math.acos(np.clip(lp @ to_mid, -1, 1))))
    return dict(N_O=[round(d(0, 4), 4), round(d(0, 5), 4), round(d(0, 6), 4)],
                N_O_central=round(d(0, 4), 4),
                min_interfragment=round(min(d(i, j) for i in range(4)
                                            for j in (4, 5, 6)), 4),
                lonepair_vs_terminal_midpoint_deg=round(ang, 2),
                NH3_bonds=[round(d(0, j), 4) for j in (1, 2, 3)],
                O3_bonds=[round(d(4, 5), 4), round(d(4, 6), 4)])


# ==================== objective (full persistence) ====================
def make_objective(ctrl, records, results, backend_factory, hook=None,
                   records_path=RECORDS):
    """backend_factory(coords_bohr(7,3)) -> (mol, mf).
    hook: optional callable(rec) for test injection (may raise)."""
    cache = {}

    def fun(x_bohr):
        x = np.asarray(x_bohr, float).reshape(7, 3)
        key = point_key(x)
        # ---- restart / duplicate reuse: full verification required ----
        if key in cache:
            return cache[key]
        rec = records.get(key)
        if rec is not None:
            if config_compatible(rec.get('config', {})) \
                    and rec.get('coords_bohr_sha') == key \
                    and rec.get('scf_converged'):
                print('[cart] reuse verified record for %s' % key, flush=True)
                cache[key] = (rec['e_total'], np.asarray(rec['grad_full']))
                return cache[key]
            # incompatible record: do NOT reuse, do NOT delete -- new eval
            print('[cart] record for %s incompatible -> fresh evaluation'
                  % key, flush=True)
        idx = ctrl.pre_eval('opt', dict(point_key=key))
        mol, mf = backend_factory(x)
        try:
            mf.kernel()
            if not mf.converged:
                ctrl.mark_error(idx, 'SCF not converged')
                raise EvaluationFailed('SCF not converged')
            g = np.asarray(mf.nuc_grad_method().kernel(), float).reshape(-1)
            e = float(mf.e_tot)
            if not (np.isfinite(e) and np.isfinite(g).all()):
                ctrl.mark_error(idx, 'non-finite')
                raise EvaluationFailed('non-finite E/g')
            e_d2 = float(d2_full.d2_energy(mol))
            coords_A = np.asarray(mol.atom_coords(unit='Angstrom'), float)
            cfg = cfg_readback(mf)
            if not config_compatible(cfg):
                ctrl.mark_error(idx, 'config mismatch')
                raise EvaluationFailed('config mismatch')
            # ---- FULL record persisted BEFORE done marking ----
            rec = dict(
                attempt=idx, point_key=key,
                coords_bohr=x.tolist(), coords_bohr_sha=key,
                coords_angstrom=coords_A.tolist(),
                elements=list(SYMS), coord_unit='Angstrom (stored) / Bohr '
                '(optimization variables)',
                e_total=e, e_d2_analytic=e_d2, e_dft_part=e - e_d2,
                grad_full=g.tolist(), grad_unit='Eh/Bohr',
                grad_max=float(np.abs(g).max()),
                grad_rms=float(np.sqrt((g ** 2).mean())),
                config=cfg, scf_converged=True, status='evaluated')
            records[key] = rec
            save_json_atomic(records_path, records)  # safe write FIRST
            rsha = record_sha(records, key)
            ctrl.post_eval(idx, dict(point_key=key, e_total=e,
                                     grad_max=rec['grad_max'],
                                     record_file=RECORDS,
                                     record_sha=rsha))
            if hook is not None:
                hook(rec)                            # test injection point
            if rec['grad_max'] <= GMAX_RECHECK:
                results['recheck_candidate'] = dict(
                    point_key=key, coords_angstrom=coords_A.tolist(),
                    grad_max=rec['grad_max'], attempt=idx,
                    note='saved point reached unprojected max|g|<=1e-5')
                save_json_atomic(RESULTS, results)
            print('[cart] eval %2d E=%.9f max|g|=%.3e key=%s'
                  % (ctrl.used_cat('opt'), e, rec['grad_max'], key),
                  flush=True)
            cache[key] = (e, g)
            return e, g
        except Exception as ex:
            results['last_error'] = dict(attempt=idx, error=repr(ex))
            save_json_atomic(RESULTS, results)
            raise
    return fun


# ==================== start geometry ====================
def load_start():
    """JOB-034 last completed evaluation actual geometry (verified)."""
    info = json.load(open(ROOT + '/inputs/nh3o3_phase2/c1_cont/'
                                 'cont_start_info.json'))
    lines = [l for l in open(ROOT + '/inputs/nh3o3_phase2/c1_cont/'
                                    'cont_start.xyz').read().splitlines()
             if l.strip()]
    n = int(lines[0])
    C = np.asarray([[float(x) for x in l.split()[1:4]]
                    for l in lines[2:2 + n]], float)
    key = point_key(C)
    assert key == info['coords_hash'], 'start hash mismatch'
    # file identity: cont_start.xyz is derived from JOB-033/034 (c1_full_opt
    # --cont) step 30; the record it came from lives in c1_full_opt_cont
    r34 = json.load(open(ROOT + '/run_artifacts/02_nh3o3_reference/'
                                'c1_full_opt_cont/exec_results_formal.json'))
    assert 'trajectory' in r34 and r34.get('n_opt_steps') == 20
    return C, dict(source='JOB-034 (c1_full_opt_cont) step 20 last '
                          'completed evaluation, via verified cont_start.xyz',
                   coords_hash=key, e_total_ref=info['e_total'],
                   grad_max_ref=info['grad_max'])


def build_mol(coords_bohr):
    atom = "; ".join("%s %.10f %.10f %.10f" % (s, x, y, z)
                     for s, (x, y, z) in zip(SYMS, coords_bohr))
    mol = gto.M(atom=atom, basis='def2-TZVP', charge=0, spin=0, verbose=0,
                max_memory=4000, unit='Bohr')
    mf = make_mf_d2_gr(mol, solvent=None, grid_response=True,
                       grid_level=GRID_LEVEL)
    return mol, mf


def main():
    ctrl = Budget(LEDGER, CAPS, init_new=not os.path.exists(LEDGER))
    records = load_json(RECORDS)
    results = load_json(RESULTS)
    coords0, src = load_start()
    results['start'] = src
    print('[cart] start verified: %s (hash %s)' % (src['source'],
                                                   src['coords_hash']),
          flush=True)
    print('[cart] settings (pre-run): %s' % OPT_SETTINGS, flush=True)
    fun = make_objective(ctrl, records, results, build_mol)

    stopped = {}
    def guarded(x):
        return fun(x)

    x0 = coords0.reshape(-1)
    try:
        res = minimize(guarded, x0, jac=True, method='L-BFGS-B',
                       options=dict(gtol=GTOL, ftol=2.22e-9, maxcor=10,
                                    maxiter=15000))
        results['scipy'] = dict(success=bool(res.success),
                                message=str(res.message), nit=int(res.nit),
                                fun=float(res.fun))
        print('[cart] scipy returned: %s (nit=%d)' % (res.message, res.nit),
              flush=True)
    except BudgetExhausted as be:
        results['budget_stop'] = str(be)
        stopped['reason'] = str(be)
        print('[cart] BUDGET STOP: %s' % be, flush=True)
    except EvaluationFailed as ef:
        results['eval_failed'] = str(ef)
        print('[cart] HARD STOP: %s' % ef, flush=True)
        save_json_atomic(RESULTS, results)
        raise

    # recheck ONLY if a saved point reached max|g| <= 1e-5
    cand = results.get('recheck_candidate')
    if cand is not None and ctrl.used_cat('recheck') < CAPS['recheck']:
        idx = ctrl.pre_eval('recheck', dict(point_key=cand['point_key']))
        mol_v, mf_v = build_mol(np.asarray(cand['coords_angstrom'], float)
                                * 1.8897261254578281)
        mf_v.kernel()
        g_v = np.asarray(mf_v.nuc_grad_method().kernel(), float).reshape(-1)
        gm = float(np.abs(g_v).max())
        ctrl.post_eval(idx, dict(e_total=float(mf_v.e_tot), grad_max=gm))
        results['recheck'] = dict(e_total=float(mf_v.e_tot), grad_max=gm,
                                  passes=bool(gm <= GMAX_RECHECK),
                                  note='independent new-object endpoint '
                                       'recheck; NO stability/Hessian/freq '
                                       'this batch')
        if results['recheck']['passes']:
            results['verdict'] = ('stationary-point candidate via Cartesian '
                                  'strategy; electronic stability and '
                                  'frequency UNCHECKED')
        print('[cart] RECHECK E=%.9f max|g|=%.3e passes=%s'
              % (results['recheck']['e_total'], gm,
                 results['recheck']['passes']), flush=True)

    results['budget'] = dict(total_used=ctrl.used_total(),
                             opt_used=ctrl.used_cat('opt'),
                             recheck_used=ctrl.used_cat('recheck'),
                             caps=CAPS)
    results['stopped'] = stopped
    save_json_atomic(RESULTS, results)
    print('CART OPT DONE: opt evals=%d recheck=%d'
          % (ctrl.used_cat('opt'), ctrl.used_cat('recheck')), flush=True)


if __name__ == '__main__':
    main()
