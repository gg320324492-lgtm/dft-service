import os, sys, json, hashlib, time
import numpy as np
HERE = '/mnt/e/dft-service/ozone_mnb_uv_simulation/protocols'
sys.path.insert(0, HERE)
from scipy.optimize import minimize
# reuse validated components (no rewritten build/D2/gradient logic)
from c1_repro_034 import (Budget, cfg_readback, config_compatible, point_key,
                          save_json_atomic, load_json, EvaluationFailed,
                          SYMS, GRID_LEVEL, BOHR_PER_A, ANG_PER_BOHR,
                          GATE_E, GATE_G)
from pyscf import gto
import d2_full
from grad_factory import make_mf_d2_gr

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
BATCH_DIR = ROOT + '/run_artifacts/02_nh3o3_reference/c1_cart_cont039'
os.makedirs(BATCH_DIR, exist_ok=True)
LEDGER = BATCH_DIR + '/budget_cont039.json'
RECORDS = BATCH_DIR + '/eval_records.json'
RESULTS = BATCH_DIR + '/cont039_results.json'
START_XYZ = ROOT + '/inputs/nh3o3_phase2/c1_cont039/cont039_start.xyz'
START_INFO = ROOT + '/inputs/nh3o3_phase2/c1_cont039/cont039_start_info.json'
CAPS = dict(total=21, opt=20, recheck=1)
GMAX_RECHECK = 1e-5
GTOL = 1e-6
FTOL = 1e-13                    # TIGHTENED energy stop condition (038: 2.22e-9)
OPT_SETTINGS = dict(method='L-BFGS-B', gtol=GTOL, ftol=FTOL, maxcor=10,
                    maxiter=15000,
                    variables='all 21 Cartesian coordinates (Bohr)',
                    constraints='none; no gradient projection; no stepwise '
                                'rotation registration',
                    energy_zero='not shifted; no objective or gradient '
                                'scaling',
                    scipy_version=__import__('scipy').__version__,
                    continuation='geometry continuation; optimizer state '
                                 'RESET (no trust-radius or Hessian-history '
                                 'recovery claimed)',
                    note='tightened SOFTWARE stop condition; NOT a loosened '
                         'acceptance standard; no convergence guarantee; '
                         'maxiter is NOT the evaluation limit')


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


def load_start():
    """JOB-038 last ACCEPTED point (a960f4a403fbd3ec) -- from the original
    full record, verified to belong to the accepted list."""
    info = json.load(open(START_INFO))
    lines = [l for l in open(START_XYZ).read().splitlines() if l.strip()]
    n = int(lines[0])
    C = np.asarray([[float(x) for x in l.split()[1:4]]
                    for l in lines[2:2 + n]], float)
    key = hashlib.sha256(C.tobytes()).hexdigest()[:16]
    assert key == info['coords_hash'], 'continuation start hash mismatch'
    assert info['point_key'] == 'a960f4a403fbd3ec', 'unexpected source point'
    # verify it belongs to the 038 accepted list
    recs = json.load(open(ROOT + '/run_artifacts/02_nh3o3_reference/'
                                 'c1_cart_v2/eval_records.json'))
    src = recs.get(info['point_key'])
    assert src is not None and src.get('accepted_iteration'), \
        'start point not in the 038 accepted list'
    assert abs(src['e_total'] - info['e_total']) < 1e-15
    assert src['grad_full'] == info['grad_full'], 'gradient reference mismatch'
    g_ref = np.asarray(info['grad_full'], float)
    return C, g_ref, info, key


def make_objective(ctrl, records, results, backend_factory, first_gate,
                   state, hook=None, records_path=RECORDS,
                   results_path=RESULTS):
    state['accepted_keys'] = []
    cache = {}

    def callback(xk):
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
        rec = records.get(key)
        if rec is not None:
            if config_compatible(rec.get('config', {})) \
                    and rec.get('coords_bohr_sha') == key \
                    and rec.get('scf_converged'):
                print('[cont39] reuse verified record for %s' % key,
                      flush=True)
                cache[key] = (rec['e_total'], np.asarray(rec['grad_full']))
                return cache[key]
        idx = ctrl.pre_eval('opt', dict(point_key=key))
        mol, readback_A, dev = build_mol(x)
        cfg = None
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
            g = np.asarray(mf.nuc_grad_method().kernel(), float).reshape(-1)
            e = float(mf.e_tot)
            if not (np.isfinite(e) and np.isfinite(g).all()):
                ctrl.mark_error(idx, 'non-finite')
                raise EvaluationFailed('non-finite E/g')
            e_d2 = float(d2_full.d2_energy(mol))
            if not state.get('first_gate_done'):
                state['first_gate_done'] = True
                fg = first_gate(e, g, mol)
                state['first_gate'] = fg
                if not fg['passed']:
                    ctrl.mark_error(idx, 'first-eval gate failed: %s' % fg)
                    raise EvaluationFailed('first-eval gate failed: %s' % fg)
                print('[cont39] FIRST-EVAL GATE PASSED: dE=%.2e dcoords=%.1e '
                      'dgrad=%.2e' % (fg['dE'], fg['dcoords'], fg['dgrad_max']),
                      flush=True)
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
                accepted_iteration=None)
            if state.get('accepted_keys') and key == state['accepted_keys'][-1]:
                rec['accepted_iteration'] = True
            records[key] = rec
            save_json_atomic(records_path, records)   # safe write FIRST
            ctrl.post_eval(idx, dict(point_key=key, e_total=e,
                                     grad_max=rec['grad_max'],
                                     record_sha=hashlib.sha256(
                                         json.dumps(rec, sort_keys=True,
                                                    default=str).encode()
                                     ).hexdigest()[:16]))
            if hook is not None:
                hook(rec)
            state['trials'].append(dict(key=key, step=rec))
            if rec['grad_max'] <= 1e-5:
                results['recheck_candidate'] = dict(
                    point_key=key, coords_angstrom=readback_A.tolist(),
                    grad_max=rec['grad_max'], attempt=idx)
                save_json_atomic(results_path, results)
                raise ControlledStop('recheck trigger reached at %s '
                                     '(max|g|=%.2e)' % (key, rec['grad_max']))
            cache[key] = (e, g)
            return e, g
        except ControlledStop:
            raise
        except Exception as ex:
            results['last_error'] = dict(attempt=idx, error=repr(ex))
            save_json_atomic(results_path, results)
            raise
    return fun, callback


def first_gate_fn(ref):
    """ref: dict(E, grad_full(7,3), coords(7,3)) of the 038 accepted point."""
    def first_gate(e, g, mol):
        dE = abs(e - ref['E'])
        C_calc = np.asarray(mol.atom_coords(unit='Angstrom'), float)
        dC = float(np.abs(C_calc - ref['coords']).max())
        dg = float(np.abs(g.reshape(-1) - ref['grad_full'].reshape(-1)).max())
        return dict(dE=dE, dcoords=dC, dgrad_max=dg, passed=bool(
            dE <= GATE_E and dg <= GATE_G and dC <= 1e-9))
    return first_gate


def main():
    ctrl = Budget(LEDGER, CAPS, init_new=not os.path.exists(LEDGER))
    records = load_json(RECORDS)
    results = load_json(RESULTS)
    coords_A, g_ref, info, key = load_start()
    results['start'] = dict(source='JOB-038 last accepted point '
                                   'a960f4a403fbd3ec (verified against the '
                                   'accepted list; hash %s)' % key,
                            continuation='geometry continuation; optimizer '
                                         'state RESET',
                            e_total_ref=info['e_total'],
                            grad_max_ref=info['grad_max'])
    save_json_atomic(RESULTS, results)
    x0 = (coords_A * BOHR_PER_A).reshape(-1)
    ref = dict(E=info['e_total'],
               grad_full=np.asarray(info['grad_full'], float),
               coords=coords_A)
    state = dict(trials=[], first_gate_done=False)
    fun, callback = make_objective(ctrl, records, results,
                                   lambda mol: make_mf_d2_gr(
                                       mol, solvent=None, grid_response=True,
                                       grid_level=GRID_LEVEL),
                                   first_gate_fn(ref), state)
    print('[cont39] settings (pre-run): %s' % OPT_SETTINGS, flush=True)
    scipy_out = None
    stop_reason = None
    try:
        res = minimize(fun, x0, jac=True, method='L-BFGS-B', callback=callback,
                       options=dict(gtol=GTOL, ftol=FTOL, maxcor=10,
                                    maxiter=15000))
        scipy_out = dict(success=bool(res.success), message=str(res.message),
                         nit=int(res.nit))
        stop_reason = 'scipy returned: %s' % res.message
        print('[cont39] scipy returned: %s (nit=%d)' % (res.message, res.nit),
              flush=True)
    except ControlledStop as cs:
        stop_reason = str(cs)
        print('[cont39] CONTROLLED STOP: %s' % cs, flush=True)
    except EvaluationFailed as ef:
        stop_reason = 'HARD STOP: %s' % ef
        results['eval_failed'] = str(ef)
        save_json_atomic(RESULTS, results)
        print('[cont39] HARD STOP: %s' % ef, flush=True)
        raise
    except RuntimeError as re_:
        stop_reason = 'BUDGET/LEDGER STOP: %s' % re_
        results['budget_stop'] = str(re_)
        print('[cont39] BUDGET/LEDGER STOP: %s' % re_, flush=True)

    cand = results.get('recheck_candidate')
    if cand is not None and ctrl.used_cat('recheck') < CAPS['recheck']:
        idx = ctrl.pre_eval('recheck', dict(point_key=cand['point_key']))
        mol_v, rb_A, dev = build_mol(np.asarray(cand['coords_angstrom'],
                                                float) * BOHR_PER_A)
        mf_v = make_mf_d2_gr(mol_v, solvent=None, grid_response=True,
                             grid_level=GRID_LEVEL)
        mf_v.kernel()
        g_v = np.asarray(mf_v.nuc_grad_method().kernel(), float).reshape(-1)
        e_v = float(mf_v.e_tot)
        gm = float(np.abs(g_v).max())
        grms = float(np.sqrt((g_v ** 2).mean()))
        C_v = np.asarray(mol_v.atom_coords(unit='Angstrom'), float)
        cfg = cfg_readback(mf_v)
        # full save mechanism for the recheck too
        rec = dict(attempt=idx, point_key=cand['point_key'],
                   coords_angstrom=C_v.tolist(), elements=list(SYMS),
                   e_total=e_v, grad_full=g_v.tolist(),
                   grad_unit='Eh/Bohr', grad_max=gm, grad_rms=grms,
                   config=cfg, scf_converged=bool(mf_v.converged),
                   finite=bool(np.isfinite(e_v) and np.isfinite(g_v).all()),
                   coords_match=bool(np.abs(C_v - np.asarray(
                       cand['coords_angstrom'], float)).max() <= 1e-9),
                   status='recheck_evaluated')
        passes = bool(mf_v.converged and rec['finite'] and rec['coords_match']
                      and config_compatible(cfg) and gm <= 1e-5)
        rec['recheck_passes'] = passes
        records[cand['point_key'] + '_recheck'] = rec
        save_json_atomic(RECORDS, records)
        ctrl.post_eval(idx, dict(e_total=e_v, grad_max=gm))
        results['recheck'] = dict(e_total=e_v, grad_max=gm, grad_rms=grms,
                                  passes=passes,
                                  note='independent new-object recheck with '
                                       'the full save mechanism; stability '
                                       'and frequency UNCHECKED')
        if passes:
            results['verdict'] = ('stationary-point candidate; stability '
                                  'and frequency NOT accepted')
        else:
            results['verdict'] = 'recheck NOT passed - registered, stopped'
        print('[cont39] RECHECK E=%.9f max|g|=%.3e passes=%s'
              % (e_v, gm, passes), flush=True)

    results['scipy'] = scipy_out
    results['stop_reason'] = stop_reason
    results['settings'] = OPT_SETTINGS
    results['budget'] = dict(total_attempts=ctrl.used_total(),
                             opt_used=ctrl.used_cat('opt'),
                             recheck_used=ctrl.used_cat('recheck'),
                             caps=CAPS)
    results['n_accepted'] = len(state.get('accepted_keys', []))
    save_json_atomic(RESULTS, results)
    print('CONT39 DONE: attempts=%d opt=%d recheck=%d stop=%s'
          % (ctrl.used_total(), ctrl.used_cat('opt'),
             ctrl.used_cat('recheck'), stop_reason), flush=True)


if __name__ == '__main__':
    main()
