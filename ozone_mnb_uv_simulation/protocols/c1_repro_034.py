import os, sys, json, hashlib, time
import numpy as np
HERE = '/mnt/e/dft-service/ozone_mnb_uv_simulation/protocols'
sys.path.insert(0, HERE)
from pyscf import gto
from grad_factory import make_mf_d2_gr

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
SRC_034 = ROOT + '/run_artifacts/02_nh3o3_reference/c1_full_opt_cont/exec_results_formal.json'
BATCH_DIR = ROOT + '/run_artifacts/02_nh3o3_reference/c1_repro_034'
os.makedirs(BATCH_DIR, exist_ok=True)
LEDGER = BATCH_DIR + '/budget_repro.json'
RECORDS = BATCH_DIR + '/repro_records.json'
RESULTS = BATCH_DIR + '/repro_result.json'
CAPS = dict(total=1, repro=1)
SYMS = ['N', 'H', 'H', 'H', 'O', 'O', 'O']
GRID_LEVEL = 8
# use PySCF's OWN internal Bohr constant so the A->Bohr->A round trip is
# exact (a CODATA value differing at the 9th digit produced ~2e-9 A readback
# deviation and tripped the 1e-9 chain gate)
from pyscf.data.nist import BOHR as ANG_PER_BOHR_PYSIF  # 0.52917721092
ANG_PER_BOHR = ANG_PER_BOHR_PYSIF
BOHR_PER_A = 1.0 / ANG_PER_BOHR
GATE_E = 1e-8
GATE_G = 1e-7
# registered known endpoint hashes (from the 037 traceback audit)
HASH_034_STEP20 = '5c9554e0c8b2125e'
HASH_033_STEP30 = 'b317f5a4f7df7f2c'


class SourceError(RuntimeError):
    """Start-source / unit-chain verification failure -> hard stop."""


class EvaluationFailed(RuntimeError):
    pass


def save_json_atomic(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(obj, fh, indent=2, default=str)
    os.replace(tmp, path)


def point_key(arr):
    return hashlib.sha256(np.asarray(arr, float).tobytes()).hexdigest()[:16]


def file_sha(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()[:16]


# ==================== source-checked start loading ====================
def load_json(path):
    if os.path.exists(path):
        return json.load(open(path))
    return {}


def load_start_v2(record=None, src_path=SRC_034):
    """Read the 034 last COMPLETED evaluation coords DIRECTLY (never via
    cont_start.xyz).  record: injectable trajectory record for tests.
    Rejects the 033 endpoint and any unregistered source."""
    if record is None:
        r = json.load(open(src_path))
        # file identity: must be the JOB-034 continuation (20 steps)
        assert 'trajectory' in r and r.get('n_opt_steps') == 20, \
            'source file is not the JOB-034 continuation'
        rec = r['trajectory'][-1]
        assert rec['step'] == 20, 'source file last step != 20'
        src_file_sha = file_sha(src_path)
    else:
        rec = record
        src_file_sha = 'injected-test'
    C = np.asarray(rec['coords_angstrom'], float)
    key = point_key(C)
    # ---- source checks ----
    if key == HASH_033_STEP30:
        raise SourceError('REJECTED: this is the JOB-033 step-30 endpoint '
                          '(the 036 misuse); the intended start is the 034 '
                          'step-20 endpoint')
    if key != HASH_034_STEP20:
        raise SourceError('REJECTED: unregistered start source (hash %s; '
                          'expected %s)' % (key, HASH_034_STEP20))
    # scalar cross-check against the record itself
    assert abs(rec['e_total'] - (-282.00171709790953)) < 1e-9, \
        'start energy does not match the registered 034 endpoint'
    assert abs(rec['grad_max'] - 0.0007595184552366901) < 1e-15, \
        'start grad_max does not match the registered 034 endpoint'
    manifest = dict(
        batch='JOB-034 (c1_full_opt_cont)',
        step=rec['step'],
        coord_unit_source='Angstrom',
        elements=list(SYMS),
        source_file=src_path,
        source_file_sha16=src_file_sha,
        coords_sha16=key,
        ref_scalars=dict(E=-282.00171709790953,
                         grad_max=0.0007595184552366901,
                         grad_rms=0.00024855024228963236),
        label='文献启发、经几何与来源核验的项目初猜（034 末步，非作者坐标）')
    return C, manifest, key


# ==================== unit-verified geometry build ====================
def build_verified(coords_A, backend='pyscf'):
    """Explicit A->Bohr conversion; read back Angstrom from the ACTUAL
    object and compare with the source (max dev <= 1e-9 A)."""
    x_bohr = np.asarray(coords_A, float) * BOHR_PER_A     # explicit
    atom = "; ".join("%s %.10f %.10f %.10f" % (s, x, y, z)
                     for s, (x, y, z) in zip(SYMS, x_bohr))
    mol = gto.M(atom=atom, basis='def2-TZVP', charge=0, spin=0, verbose=0,
                max_memory=4000, unit='Bohr')
    readback_A = np.asarray(mol.atom_coords(unit='Angstrom'), float)
    dev = float(np.abs(readback_A - np.asarray(coords_A, float)).max())
    if dev > 1e-9:
        raise SourceError('unit-chain check failed: readback dev %.2e A > '
                          '1e-9 (missing or double conversion?)' % dev)
    return mol, x_bohr, readback_A, dev


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


# ==================== single reproduction ====================
def run_reproduction(ctrl, results, backend=None, results_path=RESULTS):
    """One independent SCF + full gradient at the verified 034 endpoint.
    backend: None = real PySCF; injectable for tests."""
    coords, manifest, key = load_start_v2()
    results['start'] = manifest
    save_json_atomic(results_path, results)
    mol, x_bohr, readback_A, dev = build_verified(coords)
    results['unit_chain'] = dict(readback_dev_A=dev, passed=dev <= 1e-9,
                                 coords_bohr=x_bohr.tolist())
    save_json_atomic(results_path, results)
    cache_key = hashlib.sha256(json.dumps(dict(
        unit='Bohr', coords=[list(map(float, p)) for p in x_bohr],
        elements=SYMS, config='wb97xd+projectD2/def2-TZVP/L8/1e-12,1e-9',
        backend_id=(getattr(backend, '__name__', None) or
                    repr(backend))[:40] if backend else 'pyscf'), sort_keys=True).encode()
    ).hexdigest()[:16]
    results['cache_key'] = cache_key
    idx = ctrl.pre_eval('repro', dict(point_key=key, cache_key=cache_key,
                                      unit_chain_dev_A=dev))
    try:
        mf = None
        if backend is None:
            mf = make_mf_d2_gr(mol, solvent=None, grid_response=True,
                               grid_level=GRID_LEVEL)
            mf.kernel()
            g = np.asarray(mf.nuc_grad_method().kernel(), float).reshape(-1, 3)
            e = float(mf.e_tot)
            e_d2 = None  # the project attachment already includes -D2 in e
            cfg = cfg_readback(mf)
        else:
            e, g, e_d2, cfg = backend(x_bohr)
        if not mf_converged_ok(mf, backend):
            ctrl.mark_error(idx, 'SCF not converged')
            raise EvaluationFailed('SCF not converged')
        if not (np.isfinite(e) and np.isfinite(g).all()):
            ctrl.mark_error(idx, 'non-finite')
            raise EvaluationFailed('non-finite E/g')
        if not config_compatible(cfg):
            ctrl.mark_error(idx, 'config mismatch: %s' % cfg)
            raise EvaluationFailed('config mismatch')
        # coords actually computed (read back from the object) vs source
        C_calc = np.asarray(mol.atom_coords(unit='Angstrom'), float)
        coords_match = bool(np.abs(C_calc - coords).max() <= 1e-9)
        gates = dict(
            dE=abs(e - manifest['ref_scalars']['E']),
            dgrad_max=abs(float(np.abs(g).max())
                          - manifest['ref_scalars']['grad_max']),
            dgrad_rms=abs(float(np.sqrt((g ** 2).mean()))
                          - manifest['ref_scalars']['grad_rms']))
        gates_pass = bool(coords_match and gates['dE'] <= GATE_E
                          and gates['dgrad_max'] <= GATE_G
                          and gates['dgrad_rms'] <= GATE_G)
        # full record persisted BEFORE done marking
        rec = dict(attempt=idx, point_key=key, cache_key=cache_key,
                   coords_source=coords.tolist(),
                   coords_computed=C_calc.tolist(),
                   coords_match=coords_match,
                   elements=list(SYMS), coord_unit='Angstrom (stored) / '
                   'Bohr (variables)',
                   e_total=e,
                   e_d2_note='project -D2 already included in e_total '
                             '(attached path)',
                   grad_full=g.tolist(), grad_unit='Eh/Bohr',
                   grad_max=float(np.abs(g).max()),
                   grad_rms=float(np.sqrt((g ** 2).mean())),
                   gates=gates, gates_pass=gates_pass,
                   config=cfg, scf_converged=True, status='evaluated',
                   label='full vector saved as a NEW record; NOT claimed as '
                         'a full-vector or bitwise reproduction of JOB-034 '
                         '(historical full vector missing)')
        results['reproduction'] = rec
        save_json_atomic(results_path, results)     # safe write FIRST
        ctrl.post_eval(idx, dict(e_total=e, grad_max=rec['grad_max'],
                                 point_key=key,
                                 results_file=results_path,
                                 results_sha=hashlib.sha256(
                                     open(results_path, 'rb').read()
                                 ).hexdigest()[:16]))
        results['verdict'] = ('034 true endpoint REPRODUCED (gates passed)'
                              if gates_pass else
                              '034 true endpoint NOT reproduced (gates '
                              'failed) - scene saved, stopped')
        print('[repro] gates: dE=%.2e dgrad_max=%.2e dgrad_rms=%.2e -> %s'
              % (gates['dE'], gates['dgrad_max'], gates['dgrad_rms'],
                 'PASS' if gates_pass else 'FAIL'), flush=True)
    except Exception as ex:
        results['last_error'] = repr(ex)
        save_json_atomic(results_path, results)
        print('[repro] HARD STOP: %s' % ex, flush=True)
        raise
    save_json_atomic(results_path, results)
    return results


def mf_converged_ok(mf, backend):
    if backend is not None:
        return True                      # injected backends self-report
    return bool(mf.converged) if mf is not None else False


def main():
    ctrl = Budget(LEDGER, CAPS, init_new=not os.path.exists(LEDGER))
    results = load_json(RESULTS) if os.path.exists(RESULTS) else {}
    run_reproduction(ctrl, results, backend=None)
    print('REPRODUCTION BATCH DONE: attempts=%d' % ctrl.used_total(),
          flush=True)


if __name__ == '__main__':
    main()
