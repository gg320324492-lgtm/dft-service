import os, sys, json, hashlib, time
import numpy as np
HERE = '/mnt/e/dft-service/ozone_mnb_uv_simulation/protocols'
sys.path.insert(0, HERE)
from scipy.optimize import minimize
from c1_repro_034 import (Budget, cfg_readback, config_compatible, point_key,
                          save_json_atomic, load_json, EvaluationFailed,
                          SYMS, GRID_LEVEL, BOHR_PER_A, ANG_PER_BOHR,
                          GATE_E, GATE_G)
from pyscf import gto
import d2_full
from grad_factory import make_mf_d2_gr

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
BATCH_DIR = ROOT + '/run_artifacts/02_nh3o3_reference/c1_cart_cont042'
os.makedirs(BATCH_DIR, exist_ok=True)
LEDGER = BATCH_DIR + '/budget_cont042.json'
RECORDS = BATCH_DIR + '/eval_records.json'
RESULTS = BATCH_DIR + '/cont042_results.json'
CAPS = dict(total=16, opt=15, recheck=1)
GMAX_RECHECK = 1e-5
GTOL = 1e-6
FTOL = 1e-13
SYMS = ['N', 'H', 'H', 'H', 'O', 'O', 'O']
# 041 attempt 5 reference (min-gradient point)
REF = dict(
    point_key='50571026adb2204d',
    E=-282.001719851,
    grad_max=3.494e-05,
    source='JOB-041 c1_cart_exec_v3 eval_records.json attempt=5')


class ControlledStop(RuntimeError):
    pass


def build_mol(coords_bohr):
    atom = "; ".join("%s %.10f %.10f %.10f" % (s, x, y, z)
                     for s, (x, y, z) in zip(SYMS, coords_bohr))
    mol = gto.M(atom=atom, basis='def2-TZVP', charge=0, spin=0, verbose=0,
                max_memory=4000, unit='Bohr')
    readback_A = np.asarray(mol.atom_coords(unit='Angstrom'), float)
    expected_A = np.asarray(coords_bohr, float) * ANG_PER_BOHR
    dev = float(np.abs(readback_A - expected_A).max())
    if dev > 1e-9:
        raise RuntimeError('unit-chain check failed: dev %.2e A' % dev)
    return mol, readback_A, dev


def cfg_readback(mf):
    return dict(xc='wb97xd (project -D2 attached)',
                d2_attached=bool(getattr(mf, '_has_full_d2', False)),
                grid_level=int(getattr(mf.grids, 'level', -1)),
                scf_tol=[float(mf.conv_tol), float(mf.conv_tol_grad)],
                basis='def2-TZVP', charge=0, spin=0, grid_response=True,
                solvent='none (gas phase)')


def config_compatible(cfg):
    return all(cfg.get(k) == v for k, v in
               dict(basis='def2-TZVP', charge=0, spin=0,
                    grid_level=GRID_LEVEL, scf_tol=[1e-12, 1e-9],
                    d2_attached=True, grid_response=True,
                    solvent='none (gas phase)').items())


def save_json_atomic(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(obj, fh, indent=2, default=str)
    os.replace(tmp, path)


def load_json(path):
    if os.path.exists(path):
        return json.load(open(path))
    return {}


def point_key(arr):
    return hashlib.sha256(np.asarray(arr, float).tobytes()).hexdigest()[:16]


def record_sha(records, key):
    payload = json.dumps(records[key], sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def load_start():
    """JOB-041 attempt 5 min-gradient point (Bohr + full gradient)."""
    doc = json.load(open(ROOT + '/run_artifacts/02_nh3o3_reference/'
                                'c1_cart_exec_v3/eval_records.json'))
    rec = doc[REF['point_key']]
    assert rec['attempt'] == 5, 'attempt mismatch'
    assert abs(rec['e_total'] - REF['E']) < 1e-9, 'E mismatch'
    assert abs(rec['grad_max'] - REF['grad_max']) < 1e-3, 'grad_max mismatch'
    C_A = np.asarray(rec['coords_angstrom'], float)
    XB = np.asarray(rec['coords_bohr'], float)
    g_ref = np.asarray(rec['grad_full'], float)
    return C_A, XB, g_ref, rec


class Budget:
    def __init__(self, path, caps, init_new=False):
        self.path = path
        self.caps = caps
        if os.path.exists(path):
            self.data = json.load(open(path))
        elif init_new:
            self.data = dict(caps=caps, attempts=[])
            self._save()
        else:
            raise RuntimeError('ledger %s missing' % path)
        self.data.setdefault('attempts', [])

    def used_total(self):
        return len(self.data['attempts'])

    def used_cat(self, cat):
        return sum(1 for a in self.data['attempts']
                   if a.get('category') == cat
                   and a.get('status') != 'rejected')

    def pre_eval(self, cat, meta):
        if self.used_total() + 1 > self.caps['total']:
            raise RuntimeError('TOTAL budget exhausted (%d/%d)'
                               % (self.used_total(), self.caps['total']))
        if self.used_cat(cat) + 1 > self.caps[cat]:
            raise RuntimeError('%s budget exhausted' % cat)
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

    def _save(self):
        tmp = self.path + '.tmp'
        with open(tmp, 'w') as fh:
            json.dump(self.data, fh, indent=2, default=str)
        os.replace(tmp, self.path)


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


def main():
    ctrl = Budget(LEDGER, CAPS, init_new=not os.path.exists(LEDGER))
    records = load_json(RECORDS)
    results = load_json(RESULTS)
    C_A, XB, g_ref, src_rec = load_start()
    results['start'] = dict(
        source='JOB-041 attempt=5 min-gradient point (50571026adb2204d)',
        e_total_ref=REF['E'], grad_max_ref=REF['grad_max'],
        continuation='geometry continuation; optimizer state RESET',
        coords_hash=point_key(C_A))
    save_json_atomic(RESULTS, results)
    print('[cont42] start: attempt=5 key=50571026adb2204d E=%.9f gmax=%.3e'
          % (src_rec['e_total'], src_rec['grad_max']), flush=True)
    print('[cont42] settings (pre-run): L-BFGS-B gtol=1e-6 ftol=1e-13 '
          'maxcor=10 maxiter=15000', flush=True)

    # first-eval gate reference: E and full gradient from the 041 record
    first_gate_ref = dict(E=src_rec['e_total'],
                          grad_full=np.asarray(src_rec['grad_full'], float),
                          coords=C_A)
    state = dict(trials=[], accepted_keys=[], first_gate_done=False)
    eval_count = {'n': 0}

    def fun(x_bohr):
        x = np.asarray(x_bohr, float).reshape(7, 3)
        key = point_key(x)
        idx = ctrl.pre_eval('opt', dict(point_key=key))
        mol, mf = None, None
        try:
            atom = "; ".join("%s %.10f %.10f %.10f" % (s, a, b, c)
                             for s, (a, b, c) in zip(SYMS, x))
            mol = gto.M(atom=atom, basis='def2-TZVP', charge=0, spin=0,
                        verbose=0, max_memory=4000, unit='Bohr')
            mf = make_mf_d2_gr(mol, solvent=None, grid_response=True,
                               grid_level=GRID_LEVEL)
            cfg = cfg_readback(mf)
            if not config_compatible(cfg):
                ctrl.mark_error(idx, 'config mismatch')
                raise EvaluationFailed('config mismatch')
            mf.kernel()
            if not mf.converged:
                ctrl.mark_error(idx, 'SCF not converged')
                raise EvaluationFailed('SCF not converged')
            g = np.asarray(mf.nuc_grad_method().kernel(),
                           float).reshape(-1, 3)
            e = float(mf.e_tot)
            if not (np.isfinite(e) and np.isfinite(g).all()):
                ctrl.mark_error(idx, 'non-finite')
                raise EvaluationFailed('non-finite E/g')
            eval_count['n'] += 1
            C_calc = np.asarray(mol.atom_coords(unit='Angstrom'), float)
            # first-eval gate (attempt-5 reproduction)
            if not state.get('first_gate_done'):
                state['first_gate_done'] = True
                dE = abs(e - first_gate_ref['E'])
                dC = float(np.abs(C_calc - first_gate_ref['coords']).max())
                dg = float(np.abs(g.reshape(-1)
                                  - first_gate_ref['grad_full'].reshape(-1)
                                  ).max())
                fg = dict(dE=dE, dcoords=dC, dgrad_max=dg,
                          passed=bool(dE <= GATE_E and dg <= GATE_G
                                      and dC <= 1e-9))
                state['first_gate'] = fg
                if not fg['passed']:
                    ctrl.mark_error(idx, 'first-eval gate failed: %s' % fg)
                    raise EvaluationFailed('first-eval gate failed: %s' % fg)
                print('[cont42] FIRST-EVAL GATE PASSED: dE=%.2e dC=%.1e '
                      'dg=%.2e' % (dE, dC, dg), flush=True)
            e_d2 = float(d2_full.d2_energy(mol))
            rec = dict(attempt=idx, point_key=key,
                       coords_bohr=x.tolist(), coords_bohr_sha=key,
                       coords_angstrom=C_calc.tolist(),
                       elements=list(SYMS),
                       e_total=e, e_d2_analytic=e_d2, e_dft_part=e - e_d2,
                       grad_full=g.tolist(), grad_unit='Eh/Bohr',
                       grad_max=float(np.abs(g).max()),
                       grad_rms=float(np.sqrt((g ** 2).mean())),
                       config=cfg, scf_converged=True, status='evaluated',
                       structure=structure_analysis(C_calc),
                       accepted_iteration=None)
            records[key] = rec
            save_json_atomic(RECORDS, records)
            ctrl.post_eval(idx, dict(point_key=key, e_total=e,
                                     grad_max=rec['grad_max']))
            state['trials'].append(key)
            print('[cont42] eval %2d E=%.9f max|g|=%.3e N-Oc=%.4f'
                  % (eval_count['n'], e, rec['grad_max'],
                     rec['structure']['N_O_central']), flush=True)
            if rec['grad_max'] <= GMAX_RECHECK:
                results['recheck_candidate'] = dict(
                    point_key=key, coords_angstrom=C_calc.tolist(),
                    grad_max=rec['grad_max'], attempt=idx)
                save_json_atomic(RESULTS, results)
                raise ControlledStop('recheck trigger at %s (max|g|=%.2e)'
                                     % (key, rec['grad_max']))
            return e, g.reshape(-1)
        except ControlledStop:
            raise
        except Exception as ex:
            results['last_error'] = dict(attempt=idx, error=repr(ex))
            save_json_atomic(RESULTS, results)
            raise

    def callback(xk):
        key = point_key(np.asarray(xk, float).reshape(7, 3))
        state['accepted_keys'].append(key)
        if key in records:
            records[key]['accepted_iteration'] = True
            save_json_atomic(RECORDS, records)

    x0 = (C_A * BOHR_PER_A).reshape(-1)
    scipy_out = None
    stop_reason = None
    try:
        res = minimize(fun, x0, jac=True, method='L-BFGS-B', callback=callback,
                       options=dict(gtol=GTOL, ftol=FTOL, maxcor=10,
                                    maxiter=15000))
        scipy_out = dict(success=bool(res.success), message=str(res.message),
                         nit=int(res.nit))
        stop_reason = 'scipy returned: %s' % res.message
        print('[cont42] scipy returned: %s (nit=%d)' % (res.message, res.nit),
              flush=True)
    except ControlledStop as cs:
        stop_reason = str(cs)
        print('[cont42] CONTROLLED STOP: %s' % cs, flush=True)
    except EvaluationFailed as ef:
        stop_reason = 'HARD STOP: %s' % ef
        results['eval_failed'] = str(ef)
        save_json_atomic(RESULTS, results)
        print('[cont42] HARD STOP: %s' % ef, flush=True)
        raise
    except RuntimeError as re_:
        stop_reason = 'BUDGET STOP: %s' % re_
        results['budget_stop'] = str(re_)
        print('[cont42] BUDGET STOP: %s' % re_, flush=True)

    cand = results.get('recheck_candidate')
    if cand is not None and ctrl.used_cat('recheck') < CAPS['recheck']:
        idx = ctrl.pre_eval('recheck', dict(point_key=cand['point_key']))
        atom = "; ".join("%s %.10f %.10f %.10f" % (s, a, b, c)
                         for s, (a, b, c) in zip(
                             SYMS, cand['coords_angstrom']))
        mol_v = gto.M(atom=atom, basis='def2-TZVP', charge=0, spin=0,
                      verbose=0, max_memory=4000)
        mf_v = make_mf_d2_gr(mol_v, solvent=None, grid_response=True,
                             grid_level=GRID_LEVEL)
        mf_v.kernel()
        g_v = np.asarray(mf_v.nuc_grad_method().kernel(),
                         float).reshape(-1, 3)
        e_v = float(mf_v.e_tot)
        gm = float(np.abs(g_v).max())
        C_v = np.asarray(mol_v.atom_coords(unit='Angstrom'), float)
        cfg = cfg_readback(mf_v)
        passes = bool(mf_v.converged and np.isfinite(e_v)
                      and np.isfinite(g_v).all()
                      and config_compatible(cfg)
                      and np.abs(C_v - np.asarray(
                          cand['coords_angstrom'], float)).max() <= 1e-9
                      and gm <= 1e-5)
        rec = dict(attempt=idx, point_key=cand['point_key'],
                   coords_angstrom=C_v.tolist(), elements=list(SYMS),
                   e_total=e_v, grad_full=g_v.tolist(),
                   grad_unit='Eh/Bohr', grad_max=gm,
                   grad_rms=float(np.sqrt((g_v ** 2).mean())),
                   config=cfg, scf_converged=bool(mf_v.converged),
                   coords_match=bool(np.abs(C_v - np.asarray(
                       cand['coords_angstrom'], float)).max() <= 1e-9),
                   recheck_passes=passes, status='recheck_evaluated')
        records[cand['point_key'] + '_recheck'] = rec
        save_json_atomic(RECORDS, records)
        ctrl.post_eval(idx, dict(e_total=e_v, grad_max=gm))
        results['recheck'] = dict(e_total=e_v, grad_max=gm, passes=passes,
                                  note='independent new-object recheck with '
                                       'full save mechanism; stability and '
                                       'frequency UNCHECKED')
        if passes:
            results['verdict'] = ('stationary-point candidate; stability and '
                                  'frequency NOT accepted')
        else:
            results['verdict'] = 'recheck NOT passed - registered, stopped'
        print('[cont42] RECHECK E=%.9f max|g|=%.3e passes=%s'
              % (e_v, gm, passes), flush=True)

    results['scipy'] = scipy_out
    results['stop_reason'] = stop_reason
    results['budget'] = dict(total_attempts=ctrl.used_total(),
                             opt_used=ctrl.used_cat('opt'),
                             recheck_used=ctrl.used_cat('recheck'),
                             scf_completed=eval_count['n'],
                             caps=CAPS)
    results['n_accepted'] = len(state.get('accepted_keys', []))
    save_json_atomic(RESULTS, results)
    print('CONT42 DONE: attempts=%d scf=%d opt=%d recheck=%d stop=%s'
          % (ctrl.used_total(), eval_count['n'], ctrl.used_cat('opt'),
             ctrl.used_cat('recheck'), stop_reason), flush=True)


if __name__ == '__main__':
    main()
