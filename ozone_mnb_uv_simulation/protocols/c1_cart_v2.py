import os, sys, json, hashlib, time
import numpy as np
HERE = '/mnt/e/dft-service/ozone_mnb_uv_simulation/protocols'
sys.path.insert(0, HERE)
from scipy.optimize import minimize
# reuse validated components (no rewritten build/D2/gradient logic)
from c1_repro_034 import (load_start_v2, build_verified, Budget,
                          cfg_readback, config_compatible, point_key,
                          save_json_atomic, load_json, EvaluationFailed,
                          SYMS, GRID_LEVEL, BOHR_PER_A, ANG_PER_BOHR,
                          GATE_E, GATE_G)
from pyscf import gto
import d2_full
from grad_factory import make_mf_d2_gr

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
BATCH_DIR = ROOT + '/run_artifacts/02_nh3o3_reference/c1_cart_v2'
os.makedirs(BATCH_DIR, exist_ok=True)
LEDGER = BATCH_DIR + '/budget_cart_v2.json'
RECORDS = BATCH_DIR + '/eval_records.json'
RESULTS = BATCH_DIR + '/cart_v2_results.json'
REPRO_037 = ROOT + '/run_artifacts/02_nh3o3_reference/c1_repro_034/repro_result.json'
CAPS = dict(total=21, opt=20, recheck=1)
GMAX_RECHECK = 1e-5
GTOL = 1e-6
OPT_SETTINGS = dict(method='L-BFGS-B', gtol=GTOL, ftol=2.22e-9, maxcor=10,
                    maxiter=15000,
                    variables='all 21 Cartesian coordinates (Bohr)',
                    constraints='none; no gradient projection; no stepwise '
                                'rotation registration',
                    scipy_version=__import__('scipy').__version__,
                    note='maxiter is NOT the evaluation limit; the hard '
                         'limit is the pre-eval persistent ledger')


class ControlledStop(RuntimeError):
    """A saved point reached the recheck trigger -> stop the optimizer in a
    controlled way (NOT a failure)."""


def build_mol(coords_bohr):
    """Same validated component as JOB-037 (no rewritten logic): Bohr in,
    Angstrom readback verified against the variables."""
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


def make_objective(ctrl, records, results, backend_factory, first_gate,
                   state, hook=None, records_path=RECORDS,
                   results_path=RESULTS):
    """Objective for L-BFGS-B: returns (E [Eh], grad [Eh/Bohr]) from the
    SAME actual geometry; full record persisted before done marking."""
    state['accepted_keys'] = []
    cache = {}

    def callback(xk):
        """Accepted iteration point (distinct from line-search trials)."""
        key = point_key(np.asarray(xk, float).reshape(7, 3))
        state['accepted_keys'].append(key)
        if key in records:
            records[key]['accepted_iteration'] = True
            save_json_atomic(records_path, records)

    def fun(x_bohr):
        x = np.asarray(x_bohr, float).reshape(7, 3)
        key = point_key(x)
        if key in cache:
            return cache[key]
        idx = ctrl.pre_eval('opt', dict(point_key=key))
        mol, readback_A, dev = build_mol(x)
        cfg = cfg_readback(mf) if False else None
        try:
            mf = backend_factory(mol)
            cfg = cfg_readback(mf)
            if not config_compatible(cfg):
                ctrl.mark_error(idx, 'config mismatch: %s' % cfg)
                raise EvaluationFailed('config mismatch')
            mf.kernel()
            if not mf.converged:
                ctrl.mark_error(idx, 'SCF not converged')
                raise EvaluationFailed('SCF not converged')
            g = np.asarray(mf.nuc_grad_method().kernel(),
                           float).reshape(-1)
            e = float(mf.e_tot)
            if not (np.isfinite(e) and np.isfinite(g).all()):
                ctrl.mark_error(idx, 'non-finite')
                raise EvaluationFailed('non-finite E/g')
            e_d2 = float(d2_full.d2_energy(mol))
            # first evaluation = MANDATORY start reproduction gate
            if not state.get('first_gate_done'):
                state['first_gate_done'] = True
                fg = first_gate(e, g, mol)
                state['first_gate'] = fg
                if not fg['passed']:
                    ctrl.mark_error(idx, 'first-eval reproduction gate '
                                         'failed: %s' % fg)
                    raise EvaluationFailed('first-eval reproduction gate '
                                           'failed: %s' % fg)
                print('[cart2] FIRST-EVAL GATE PASSED: dE=%.2e '
                      'dgrad=%.2e' % (fg['dE'], fg['dgrad_max']), flush=True)
            rec = dict(
                attempt=idx, point_key=key,
                coords_bohr=x.tolist(), coords_bohr_sha=key,
                coords_angstrom=readback_A.tolist(),
                elements=list(SYMS),
                unit='variables Bohr; stored Angstrom; gradient Eh/Bohr; '
                     'energy Eh',
                e_total=e, e_d2_analytic=e_d2, e_dft_part=e - e_d2,
                grad_full=g.tolist(), grad_max=float(np.abs(g).max()),
                grad_rms=float(np.sqrt((g ** 2).mean())),
                config=cfg, scf_converged=True, status='evaluated',
                accepted_iteration=None,     # set by the callback if accepted
                step_size_A=None)
            if state.get('accepted_keys') and key == state['accepted_keys'][-1]:
                rec['accepted_iteration'] = True
            records[key] = rec
            save_json_atomic(records_path, records)  # safe write FIRST
            ctrl.post_eval(idx, dict(point_key=key, e_total=e,
                                     grad_max=rec['grad_max'],
                                     record_sha=hashlib.sha256(
                                         json.dumps(rec, sort_keys=True,
                                                    default=str).encode()
                                     ).hexdigest()[:16]))
            state['trials'].append(dict(key=key, step=rec))
            if hook is not None:
                hook(rec)                       # test injection point
            if rec['grad_max'] <= GMAX_RECHECK:
                results['recheck_candidate'] = dict(
                    point_key=key, coords_angstrom=readback_A.tolist(),
                    grad_max=rec['grad_max'], attempt=idx,
                    note='saved eval point reached unprojected '
                         'max|g|<=1e-5; record complete; config verified')
                save_json_atomic(results_path, results)
                raise ControlledStop('recheck trigger reached at %s '
                                     '(max|g|=%.2e) - controlled stop'
                                     % (key, rec['grad_max']))
            state['last_coords'] = readback_A
            cache[key] = (e, g)
            return e, g
        except ControlledStop:
            raise
        except Exception as ex:
            results['last_error'] = dict(attempt=idx, error=repr(ex))
            save_json_atomic(results_path, results)
            raise
    return fun, callback


def load_first_gate_reference():
    """037's SAVED full gradient + energy (NOT the 034 historical vector)."""
    doc = json.load(open(REPRO_037))
    rec = doc['reproduction']
    return dict(e_total=rec['e_total'],
                grad_full=np.asarray(rec['grad_full'], float),
                coords=np.asarray(rec['coords_computed'], float),
                config=rec['config'],
                source='JOB-037 repro_records.json (saved full vector)')


def first_gate_fn(ref):
    def first_gate(e, g, mol):
        dE = abs(e - ref['e_total'])
        C_calc = np.asarray(mol.atom_coords(unit='Angstrom'), float)
        dC = float(np.abs(C_calc - ref['coords']).max())
        dg = float(np.abs(g.reshape(-1) - ref['grad_full'].reshape(-1)).max())
        cfg_ok = True      # config verified separately from the actual object
        return dict(dE=dE, dgrad_max=dg, dcoords=dC, passed=bool(
            dE <= GATE_E and dg <= GATE_G and dC <= 1e-9 and cfg_ok))
    return first_gate


def main():
    ctrl = Budget(LEDGER, CAPS, init_new=not os.path.exists(LEDGER))
    records = load_json(RECORDS)
    results = load_json(RESULTS)
    coords_A, manifest, key = load_start_v2()
    results['start'] = manifest
    save_json_atomic(RESULTS, results)
    x0 = (coords_A * BOHR_PER_A).reshape(-1)     # variables in Bohr
    ref = load_first_gate_reference()
    # cross-check the start against the 037 reproduction record
    assert np.abs(coords_A - ref['coords']).max() <= 1e-9, \
        'start coords disagree with the 037 reproduction record'
    state = dict(trials=[], first_gate_done=False)
    fun, callback = make_objective(ctrl, records, results,
                                   lambda mol: make_mf_d2_gr(
                                       mol, solvent=None, grid_response=True,
                                       grid_level=GRID_LEVEL),
                                   first_gate_fn(ref), state)
    print('[cart2] settings (pre-run): %s' % OPT_SETTINGS, flush=True)
    scipy_out = None
    stop_reason = None
    try:
        res = minimize(fun, x0, jac=True, method='L-BFGS-B',
                       callback=callback,
                       options=dict(gtol=GTOL, ftol=2.22e-9, maxcor=10,
                                    maxiter=15000))
        scipy_out = dict(success=bool(res.success), message=str(res.message),
                         nit=int(res.nit))
        stop_reason = 'scipy returned: %s' % res.message
        print('[cart2] scipy returned: %s (nit=%d)' % (res.message, res.nit),
              flush=True)
    except ControlledStop as cs:
        stop_reason = str(cs)
        print('[cart2] CONTROLLED STOP: %s' % cs, flush=True)
    except EvaluationFailed as ef:
        stop_reason = 'HARD STOP: %s' % ef
        results['eval_failed'] = str(ef)
        print('[cart2] HARD STOP: %s' % ef, flush=True)
        save_json_atomic(RESULTS, results)
        raise
    except RuntimeError as re_:
        stop_reason = 'BUDGET/LEDGER STOP: %s' % re_
        results['budget_stop'] = str(re_)
        print('[cart2] BUDGET/LEDGER STOP: %s' % re_, flush=True)

    # ---- conditional independent endpoint recheck ----
    cand = results.get('recheck_candidate')
    if cand is not None and ctrl.used_cat('recheck') < CAPS['recheck']:
        idx = ctrl.pre_eval('recheck', dict(point_key=cand['point_key']))
        mol_v, rb_A, dev = build_mol(np.asarray(cand['coords_angstrom'],
                                                float) * BOHR_PER_A)
        mf_v = make_mf_d2_gr(mol_v, solvent=None, grid_response=True,
                             grid_level=GRID_LEVEL)
        mf_v.kernel()
        g_v = np.asarray(mf_v.nuc_grad_method().kernel(), float).reshape(-1)
        gm = float(np.abs(g_v).max())
        ctrl.post_eval(idx, dict(e_total=float(mf_v.e_tot), grad_max=gm))
        passes = bool(gm <= GMAX_RECHECK)
        results['recheck'] = dict(e_total=float(mf_v.e_tot), grad_max=gm,
                                  passes=passes,
                                  note='independent new-object endpoint '
                                       'recheck at the same gate')
        if passes:
            results['verdict'] = ('stationary-point candidate via the '
                                  'Cartesian strategy; electronic stability '
                                  'and frequency UNCHECKED')
        else:
            results['verdict'] = ('recheck did not reproduce the gradient '
                                  'gate - registered, stopped')
        print('[cart2] RECHECK E=%.9f max|g|=%.3e passes=%s'
              % (results['recheck']['e_total'], gm, passes), flush=True)

    results['scipy'] = scipy_out
    results['stop_reason'] = stop_reason
    results['settings'] = OPT_SETTINGS
    results['budget'] = dict(total_used=ctrl.used_total(),
                             opt_used=ctrl.used_cat('opt'),
                             recheck_used=ctrl.used_cat('recheck'),
                             caps=CAPS)
    results['n_accepted'] = len(state.get('accepted_keys', []))
    results['accepted_keys'] = state.get('accepted_keys', [])
    save_json_atomic(RESULTS, results)
    print('CART V2 DONE: opt=%d recheck=%d accepted=%d stop=%s'
          % (ctrl.used_cat('opt'), ctrl.used_cat('recheck'),
             results['n_accepted'], stop_reason), flush=True)


if __name__ == '__main__':
    main()
