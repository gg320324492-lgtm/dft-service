import os, sys, json, hashlib, time, argparse
import numpy as np
HERE = '/mnt/e/dft-service/ozone_mnb_uv_simulation/protocols'
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from scipy.optimize import minimize
# reuse validated components (no rewritten build/D2/gradient logic)
from c1_repro_034 import (Budget, cfg_readback, config_compatible, point_key, SourceError,
                          save_json_atomic, load_json, EvaluationFailed,
                          SYMS, GRID_LEVEL, BOHR_PER_A, ANG_PER_BOHR,
                          GATE_E, GATE_G)
from pyscf import gto
import d2_full
from grad_factory import make_mf_d2_gr

# NO module-level side effects: no directory creation, no ledger, no compute.
ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
# registry of FORMAL locations (the write guard refuses these for tests)
BATCH_DIR = ROOT + '/run_artifacts/02_nh3o3_reference/c1_cart_exec_v3'
FORMAL_DIRS = (BATCH_DIR,)
LEDGER = BATCH_DIR + '/budget_cart_v3.json'
RECORDS = BATCH_DIR + '/eval_records.json'
RESULTS = BATCH_DIR + '/cart_v3_results.json'
CAPS = dict(total=21, opt=20, recheck=1, repro=1)
# registered obsolete-source hashes (JOB-037 traceback audit)
HASH_033_STEP30 = 'b317f5a4f7df7f2c'
HASH_034_STEP20 = '5c9554e0c8b2125e'
GMAX_RECHECK = 1e-5
GTOL = 1e-6
FTOL = 1e-13
SYMS = ['N', 'H', 'H', 'H', 'O', 'O', 'O']
CONFIG_FINGERPRINT = dict(basis='def2-TZVP', charge=0, spin=0,
                          grid_level=GRID_LEVEL, scf_tol=[1e-12, 1e-9],
                          d2_attached=True, grid_response=True,
                          solvent='none (gas phase)')


class PathGuardError(RuntimeError):
    pass


class StartGateError(RuntimeError):
    """Start acceptance failed -> the optimizer must NOT start."""


def guard_path(path, allow_formal=False):
    """Reject resolved paths inside formal directories before any write."""
    rp = os.path.abspath(path)
    for fd in FORMAL_DIRS:
        if rp.startswith(os.path.abspath(fd) + os.sep):
            if not allow_formal:
                raise PathGuardError('refusing to write into the formal '
                                     'directory: %s' % rp)
    return rp


def save_json_atomic(path, obj, allow_formal=False):
    guard_path(path, allow_formal=allow_formal)
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


def file_sha(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()[:16]


def build_mol(coords_bohr):
    """Validated component (JOB-037): Bohr in, Angstrom readback verified."""
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
    return all(cfg.get(k) == v for k, v in CONFIG_FINGERPRINT.items())


def real_backend(mol):
    return make_mf_d2_gr(mol, solvent=None, grid_response=True,
                         grid_level=GRID_LEVEL)


# ==================== budget ====================
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
            raise RuntimeError('ledger %s missing: refusing to silently '
                               'create quota' % path)
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


# ==================== start acceptance (BEFORE the optimizer) ====================
def run_start_acceptance(ctrl, records, results, backend_factory,
                         coords_source, manifest, records_path,
                         results_path, allow_formal_writes=False):
    """One independent SCF + full gradient AT THE START COORDS, verified
    against the manifest reference scalars.  Returns the acceptance
    credential on success; raises StartGateError on failure (the optimizer
    is never started in that case)."""
    guard_path(records_path, allow_formal=allow_formal_writes)
    guard_path(results_path, allow_formal=allow_formal_writes)
    idx = ctrl.pre_eval('repro', dict(point_key=manifest['coords_sha16']))
    x_bohr = np.asarray(coords_source, float) * BOHR_PER_A
    mol, readback_A, dev = build_mol(x_bohr)
    if dev > 1e-9:
        ctrl.mark_error(idx, 'unit-chain dev %.2e' % dev)
        raise StartGateError('unit-chain check failed')
    mf = backend_factory(mol)
    cfg = cfg_readback(mf)
    if not config_compatible(cfg):
        ctrl.mark_error(idx, 'config mismatch')
        raise StartGateError('config mismatch: %s' % cfg)
    mf.kernel()
    if not mf.converged:
        ctrl.mark_error(idx, 'SCF not converged')
        raise StartGateError('SCF not converged')
    g = np.asarray(mf.nuc_grad_method().kernel(), float).reshape(-1, 3)
    e = float(mf.e_tot)
    if not (np.isfinite(e) and np.isfinite(g).all()):
        ctrl.mark_error(idx, 'non-finite')
        raise StartGateError('non-finite E/g')
    C_calc = np.asarray(mol.atom_coords(unit='Angstrom'), float)
    dC = float(np.abs(C_calc - np.asarray(coords_source, float)).max())
    g_ref_full = np.asarray(manifest['ref_scalars'].get(
        'grad_full', np.zeros((7, 3))), float).reshape(-1)
    gates = dict(dE=abs(e - manifest['ref_scalars']['E']),
                 dgrad_max=abs(float(np.abs(g).max())
                               - manifest['ref_scalars']['grad_max']),
                 dgrad_rms=abs(float(np.sqrt((g ** 2).mean()))
                               - manifest['ref_scalars']['grad_rms']),
                 dcoords=dC,
                 dgrad_full_max=float(np.abs(g.reshape(-1)
                                             - g_ref_full).max()))
    passed = bool(gates['dE'] <= GATE_E and gates['dgrad_max'] <= GATE_G
                  and gates['dgrad_rms'] <= GATE_G and gates['dcoords'] <= 1e-9
                  and gates['dgrad_full_max'] <= GATE_G)
    key_vars = point_key(x_bohr)     # optimizer-variable hash (Bohr)
    rec = dict(attempt=idx, point_key=key_vars,
               coords_bohr_sha=key_vars,
               coords_source_sha16=manifest['coords_sha16'],
               coords_source=np.asarray(coords_source, float).tolist(),
               coords_computed=C_calc.tolist(), elements=list(SYMS),
               e_total=e, grad_full=g.tolist(),
               grad_unit='Eh/Bohr', grad_max=float(np.abs(g).max()),
               grad_rms=float(np.sqrt((g ** 2).mean())),
               gates=gates, gates_pass=passed, config=cfg,
               scf_converged=True,
               accepted_start=passed,
               label='start acceptance record (full vector; NOT claimed as '
                     'a full-vector/bitwise reproduction of JOB-034)')
    records[key_vars] = rec
    save_json_atomic(records_path, records, allow_formal=allow_formal_writes)  # safe write FIRST
    results['start_acceptance'] = dict(passed=passed, gates=gates,
                                       point_key=key_vars,
                                       record_sha=record_sha(records,
                                                             key_vars))
    save_json_atomic(results_path, results, allow_formal=allow_formal_writes)
    ctrl.post_eval(idx, dict(e_total=e, grad_max=rec['grad_max'],
                             point_key=manifest['coords_sha16']))
    print('[v3] START ACCEPTANCE: %s (dE=%.2e dgrad_max=%.2e dgrad_rms=%.2e '
          'dcoords=%.1e)' % ('PASS' if passed else 'FAIL', gates['dE'],
                             gates['dgrad_max'], gates['dgrad_rms'],
                             gates['dcoords']), flush=True)
    if not passed:
        raise StartGateError('start acceptance FAILED - optimizer not '
                             'started')
    return rec


def record_sha(records, key):
    payload = json.dumps(records[key], sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ==================== optimizer (after the gate) ====================
def make_objective(ctrl, records, results, backend_factory, state,
                   records_path, results_path, allow_formal_writes=False,
                   hook=None):
    """Trial points get GENERAL checks only (coords/config/SCF/finiteness/
    budget) -- no start energy/gradient comparison.  The accepted start
    record is reused via its acceptance credential (no duplicate SCF)."""
    guard_path(records_path, allow_formal=allow_formal_writes)
    guard_path(results_path, allow_formal=allow_formal_writes)
    cache = {}
    state.setdefault('accepted_keys', [])

    def callback(xk):
        key = point_key(np.asarray(xk, float).reshape(7, 3))
        state['accepted_keys'].append(key)
        if key in records:
            records[key]['accepted_iteration'] = True
            save_json_atomic(records_path, records,
                             allow_formal=allow_formal_writes)

    def fun(x_bohr):
        x = np.asarray(x_bohr, float).reshape(7, 3)
        key = point_key(x)
        if key in cache:
            return cache[key]
        rec = records.get(key)
        if rec is not None and rec.get('accepted_start'):
            # an acceptance credential: verify hash & config before reuse
            if rec.get('coords_bohr_sha') != key \
                    or not config_compatible(rec.get('config', {})):
                raise StartGateError('acceptance credential corrupted '
                                     '(hash/config mismatch) - refusing '
                                     'silent reuse')
            print('[v3] reusing the accepted start record for %s' % key,
                  flush=True)
            cache[key] = (rec['e_total'], np.asarray(rec['grad_full']))
            return cache[key]
        idx = ctrl.pre_eval('opt', dict(point_key=key))
        try:
            mol, readback_A, dev = build_mol(x)
            mf = backend_factory(mol)
            cfg = cfg_readback(mf)
            if not config_compatible(cfg):
                ctrl.mark_error(idx, 'config mismatch')
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
            rec = dict(attempt=idx, point_key=key,
                       coords_bohr=x.tolist(), coords_bohr_sha=key,
                       coords_angstrom=readback_A.tolist(),
                       elements=list(SYMS),
                       e_total=e, e_d2_analytic=e_d2, e_dft_part=e - e_d2,
                       grad_full=g.tolist(), grad_unit='Eh/Bohr',
                       grad_max=float(np.abs(g).max()),
                       grad_rms=float(np.sqrt((g ** 2).mean())),
                       config=cfg, scf_converged=True, status='evaluated',
                       accepted_iteration=None)
            records[key] = rec
            save_json_atomic(records_path, records,
                             allow_formal=allow_formal_writes)
            ctrl.post_eval(idx, dict(point_key=key, e_total=e,
                                     grad_max=rec['grad_max']))
            if hook is not None:
                hook(rec)
            state['trials'].append(key)
            if rec['grad_max'] <= GMAX_RECHECK:
                results['recheck_candidate'] = dict(
                    point_key=key, coords_angstrom=readback_A.tolist(),
                    grad_max=rec['grad_max'], attempt=idx)
                save_json_atomic(results_path, results,
                                 allow_formal=allow_formal_writes)
                raise ControlledStop('recheck trigger at %s (max|g|=%.2e)'
                                     % (key, rec['grad_max']))
            cache[key] = (e, g)
            return e, g
        except ControlledStop:
            raise
        except Exception as ex:
            results['last_error'] = dict(attempt=idx, error=repr(ex))
            save_json_atomic(results_path, results,
                             allow_formal=allow_formal_writes)
            raise
    return fun, callback


class ControlledStop(RuntimeError):
    pass


def run_optimizer(ctrl, records, results, backend_factory, state,
                  records_path, results_path, x0_bohr,
                  allow_formal_writes=False, hook=None,
                  start_credential_key=None):
    # the optimizer must NOT start without a verified start acceptance
    if start_credential_key is None:
        raise StartGateError('no start acceptance credential provided - '
                             'optimizer not started')
    cred = records.get(start_credential_key)
    if cred is None or not cred.get('accepted_start') \
            or not config_compatible(cred.get('config', {})):
        raise StartGateError('start acceptance credential missing or '
                             'incompatible - optimizer not started')
    fun, callback = make_objective(ctrl, records, results, backend_factory,
                                   state, hook=hook,
                                   records_path=records_path,
                                   results_path=results_path,
                                   allow_formal_writes=allow_formal_writes)
    return minimize(fun, x0_bohr, jac=True, method='L-BFGS-B',
                    callback=callback,
                    options=dict(gtol=GTOL, ftol=FTOL, maxcor=10,
                                 maxiter=15000))


def load_start_038(src_path=None):
    """JOB-041 source: the JOB-038 LAST ACCEPTED point (a960f4a403fbd3ec)
    read from its full record (coords + full gradient).  The old
    cont_start.xyz (JOB-033 step 30) and the 034 step-20 source are NOT
    used."""
    src_path = src_path or (ROOT + '/run_artifacts/02_nh3o3_reference/'
                            'c1_cart_v2/eval_records.json')
    key_expect = 'a960f4a403fbd3ec'
    doc = json.load(open(src_path))
    rec = doc.get(key_expect)
    if rec is None:
        raise SourceError('REJECTED: the JOB-038 last accepted point %s is '
                          'missing from the source file' % key_expect)
    if not rec.get('accepted_iteration'):
        raise SourceError('REJECTED: the source point is not an accepted '
                          'iteration')
    C = np.asarray(rec['coords_angstrom'], float)
    # the record stores the BOHR variables (coords_bohr) and their hash --
    # the authoritative key basis (the A-coords went through PySCF's internal
    # Bohr->A conversion, so their hash differs)
    XB = np.asarray(rec['coords_bohr'], float)
    key = point_key(XB)
    if key != key_expect or rec.get('coords_bohr_sha') != key:
        raise SourceError('REJECTED: coords hash mismatch (%s != %s)'
                          % (key, key_expect))
    # cross-check the stored A-coords against the stored Bohr coords
    dA = float(np.abs(XB * ANG_PER_BOHR - C).max())
    if dA > 1e-9:
        raise SourceError('REJECTED: stored A-coords inconsistent with the '
                          'stored Bohr coords (dev %.2e A)' % dA)
    ref = dict(E=-282.0017197416054, grad_max=1.935298044405478e-04,
              grad_rms=rec['grad_rms'])
    if abs(rec['e_total'] - ref['E']) > 1e-12:
        raise SourceError('REJECTED: E does not match the registered value')
    if abs(rec['grad_max'] - ref['grad_max']) > 1e-18:
        raise SourceError('REJECTED: grad_max does not match the registered '
                          'value')
    if key == HASH_033_STEP30 or key == HASH_034_STEP20:
        raise SourceError('REJECTED: obsolete source (033 step30 / 034 '
                          'step20)')
    manifest = dict(
        batch='JOB-038 (c1_cart_v2) last accepted iteration',
        step='last accepted (nit 4 of 4)',
        coord_unit_source='Angstrom',
        elements=list(SYMS),
        source_file=src_path,
        source_file_sha16=file_sha(src_path),
        coords_sha16=key,
        ref_scalars=ref,
        grad_full_reference=rec['grad_full'],
        label='JOB-038 last accepted point (full-record source; not '
              'cont_start.xyz, not 039 leftover cache)')
    g_ref = np.asarray(rec['grad_full'], float)
    return C, manifest, key, g_ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--execute', action='store_true', required=True,
                    help='explicit execution flag (no silent default)')
    ap.add_argument('--backend', choices=['real'], required=True,
                    help='explicit real-backend selection')
    ap.add_argument('--init-new-ledger', action='store_true')
    args = ap.parse_args()
    ctrl = Budget(LEDGER, CAPS, init_new=args.init_new_ledger)
    records = load_json(RECORDS)
    results = load_json(RESULTS)
    coords, manifest, key, g_ref_038 = load_start_038()
    manifest['ref_scalars']['grad_full'] = g_ref_038.tolist()
    # restart recovery: if a verified passing acceptance record already
    # exists (JOB-041's own earlier acceptance), reuse it -- do not re-run
    key_vars = point_key((coords * BOHR_PER_A).reshape(7, 3))
    existing = records.get(key_vars)
    accepted_ok = (existing is not None
                   and existing.get('accepted_start') is True
                   and existing.get('gates_pass') is True
                   and existing.get('scf_converged') is True
                   and config_compatible(existing.get('config', {}))
                   and existing.get('coords_bohr_sha') == key_vars)
    if accepted_ok:
        print('[v3] verified passing start acceptance found -- reused '
              '(restart recovery)', flush=True)
    else:
        run_start_acceptance(ctrl, records, results, real_backend, coords,
                             manifest, RECORDS, RESULTS,
                             allow_formal_writes=True)
    key_vars = point_key((coords * BOHR_PER_A).reshape(7, 3))
    res = run_optimizer(ctrl, records, results, real_backend,
                        dict(trials=[], accepted_keys=[]), RECORDS, RESULTS,
                        (coords * BOHR_PER_A).reshape(-1),
                        start_credential_key=key_vars,
                        allow_formal_writes=True)
    results['scipy'] = dict(success=bool(res.success), message=str(res.message),
                            nit=int(res.nit))
    save_json_atomic(RESULTS, results, allow_formal_writes=True)
    print('V3 EXECUTION DONE', flush=True)


if __name__ == '__main__':
    main()
