import os, sys, json, hashlib, time, argparse
import numpy as np
HERE = '/mnt/e/dft-service/ozone_mnb_uv_simulation/protocols'
sys.path.insert(0, HERE)
from sp_ledger import SPLedger
from pyscf import gto
import d2_full
from grad_factory import make_mf_d2_gr

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
GEOM_MANIFEST = ROOT + '/inputs/nh3o3_phase2/c1_scan_v2_std/scan_geometry_manifest_std.json'
XYZ_DIR = os.path.dirname(GEOM_MANIFEST)
BATCH_DIR = ROOT + '/run_artifacts/02_nh3o3_reference/c1_scan_formal'
os.makedirs(BATCH_DIR, exist_ok=True)
RESULTS_FILE = BATCH_DIR + '/exec_results_formal.json'
LEDGER = BATCH_DIR + '/budget_formal.json'
CAPS = dict(total=6, scan=6)
BOHR_PER_A = 1.8897261254578281     # Bohr per Angstrom
SYMS_EXPECT = ['N', 'H', 'H', 'H', 'O', 'O', 'O']
GRID_LEVEL = 8
CONFIG_FINGERPRINT = dict(basis='def2-TZVP', charge=0, spin=0,
                          grid_level=GRID_LEVEL, grid_response=True,
                          scf_tol=[1e-12, 1e-9], solvent='none (gas phase)')


class ScanIntegrityError(RuntimeError):
    """Manifest/hash/distance/ledger-result conflict -> hard stop."""


class EvaluationFailed(RuntimeError):
    """SCF failure / non-finite / config mismatch -> hard stop."""


# ==================== full-precision reference energies ====================
def load_reference_energies():
    """Read full-precision monomer reference energies from the JOB-026
    original result files (NOT hand-written rounded constants)."""
    fr = json.load(open(ROOT + '/run_artifacts/02_nh3o3_reference/'
                               'freq_check/freq_check_results.json'))
    e_nh3 = fr['results']['NH3']['endpoint_check']['e_total']
    e_o3 = fr['results']['O3']['endpoint_check']['e_total']
    return e_nh3, e_o3, dict(source='freq_check_results.json (JOB-026 '
                                    'endpoint rechecks, full precision)')


# ==================== XYZ I/O ====================
def write_standard_xyz(path, symbols, coords, comment):
    lines = [str(len(symbols)), comment]
    for s, (x, y, z) in zip(symbols, coords):
        lines.append('%-2s %20.12f %20.12f %20.12f' % (s, x, y, z))
    open(path, 'w').write('\n'.join(lines) + '\n')


def read_xyz_primary(path):
    lines = [l for l in open(path).read().splitlines() if l.strip() != '']
    n = int(lines[0].split()[0])
    assert n == 7, 'expected exactly 7 atoms, got %d' % n
    assert len(lines) >= n + 2, 'XYZ too short'
    syms, C = [], []
    for l in lines[2:2 + n]:
        p = l.split()
        syms.append(p[0])
        C.append([float(p[1]), float(p[2]), float(p[3])])
    C = np.asarray(C, float)
    assert np.isfinite(C).all(), 'non-finite coordinate'
    assert syms == SYMS_EXPECT, 'element order mismatch: %s' % syms
    return syms, C


def read_xyz_independent(path):
    tokens = open(path).read().split('\n')
    n = int(tokens[0].strip())
    rows = [b.split() for b in tokens[2:2 + n]]
    syms = [r[0] for r in rows]
    C = np.array([[float(v) for v in r[1:4]] for r in rows])
    assert C.shape == (7, 3) and n == 7
    assert syms == SYMS_EXPECT
    assert np.isfinite(C).all()
    return syms, C


# ==================== local derivatives (per geometry) ====================
def dE_ds_A_from_projection(g_full, u_dir):
    """g_proj [Eh/Bohr] = sum over NH3 atoms of g_i . u;
    dE/ds [Eh/A] = g_proj * BOHR_PER_A (multiply: 1 Bohr = 1/1.8897 A)."""
    g_proj = float(np.sum(g_full[:4] @ u_dir))     # Eh/Bohr
    return g_proj, g_proj * BOHR_PER_A


def dd_ds_local(coords, u_dir, term_rows):
    """dd/ds = 1/2 * sum_j [ (N-O_j).u / |N-O_j| ] -- local to this geometry."""
    N = coords[0]
    tot = 0.0
    for j in term_rows:
        v = N - coords[j]
        r = float(np.linalg.norm(v))
        tot += float(v @ u_dir) / r
    return 0.5 * tot


def ds_dd_from_local(dd_ds):
    assert abs(dd_ds) > 1e-12, 'dd/ds ~ 0'
    return 1.0 / dd_ds


def dE_dd_A(dE_ds_A, ds_dd):
    return dE_ds_A * ds_dd


# ==================== budget ====================
class Budget:
    """TOTAL counts ALL actual attempts (any category); per-category caps
    additional.  Missing ledger is an error unless explicitly initialised."""

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
                               'create new quota (init_new=True)' % path)
        self.data.setdefault('attempts', [])

    def used_total(self):
        return len(self.data['attempts'])       # ALL actual attempts

    def used_cat(self, cat):
        return sum(1 for a in self.data['attempts']
                   if a.get('category') == cat
                   and a.get('status') != 'rejected')

    def pre_eval(self, cat, meta):
        if self.used_total() + 1 > self.caps['total']:
            raise RuntimeError('TOTAL budget exhausted (%d/%d) - rejected '
                               'before evaluation' % (self.used_total(),
                                                      self.caps['total']))
        if self.used_cat(cat) + 1 > self.caps[cat]:
            raise RuntimeError('%s budget exhausted (%d/%d)'
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

    def find_done(self, key):
        for a in self.data['attempts']:
            if a.get('status') == 'done' \
                    and a.get('meta', {}).get('point_key') == key:
                return a
        return None

    def _save(self):
        tmp = self.path + '.tmp'
        with open(tmp, 'w') as fh:
            json.dump(self.data, fh, indent=2, default=str)
        os.replace(tmp, self.path)


# ==================== backends ====================
def real_backend(coords_angstrom):
    atom = "; ".join("%s %.10f %.10f %.10f" % (s, x, y, z)
                     for s, (x, y, z) in zip(SYMS_EXPECT, coords_angstrom))
    mol = gto.M(atom=atom, basis='def2-TZVP', charge=0, spin=0, verbose=0,
                max_memory=4000)
    mf = make_mf_d2_gr(mol, solvent=None, grid_response=True,
                       grid_level=GRID_LEVEL)
    return mol, mf


class SimulatedBackend:
    def __init__(self, e_tot, grad, fail_scf=False, fail_finite=False):
        self.e_tot = e_tot
        self._grad = grad
        self.fail_scf = fail_scf
        self.fail_finite = fail_finite
        self.converged = not fail_scf
        self.grids = type('G', (), {'level': GRID_LEVEL})()

    @property
    def _has_full_d2(self):
        return True

    @property
    def conv_tol(self):
        return 1e-12

    @property
    def conv_tol_grad(self):
        return 1e-9

    def kernel(self):
        if self.fail_scf:
            self.converged = False
        return self.e_tot

    def nuc_grad_method(self):
        g = self._grad
        class _G:
            def kernel(self_):
                return g
        return _G()


def sim_backend_factory(fail_at=None, harmonic=(0.05, 4.5)):
    kk, x0 = harmonic

    class MolStub:
        def __init__(self, c):
            self._c = np.asarray(c, float)
            self.elements = SYMS_EXPECT
            self.natm = len(SYMS_EXPECT)

        def atom_coords(self, unit='Bohr'):
            return self._c * BOHR_PER_A if unit == 'Bohr' else self._c

        def atom_symbol(self, i):
            return SYMS_EXPECT[i]

        def atom_charge(self, i):
            return {'N': 7, 'H': 1, 'O': 8}[SYMS_EXPECT[i]]

    def factory(coords_A):
        x_A = float(coords_A[0, 0])
        e = 0.5 * kk * (x_A - x0) ** 2 - 1.0
        g = np.zeros((7, 3))
        g[0, 0] = kk * (x_A - x0)
        fail = fail_at is not None and abs(x_A - fail_at) < 5e-3
        return MolStub(coords_A), SimulatedBackend(e, g, fail_scf=fail)
    return factory


# ==================== config / keys ====================
def cfg_readback(mf):
    return dict(grid_level=int(getattr(mf.grids, 'level', -1)),
                d2_attached=bool(getattr(mf, '_has_full_d2', False)),
                scf_tol=[float(mf.conv_tol), float(mf.conv_tol_grad)],
                basis='def2-TZVP', charge=0, spin=0,
                grid_response=True, solvent='none (gas phase)')


def config_compatible(cfg):
    return (cfg.get('grid_level') == GRID_LEVEL
            and cfg.get('d2_attached') is True
            and cfg.get('basis') == CONFIG_FINGERPRINT['basis']
            and cfg.get('scf_tol') == CONFIG_FINGERPRINT['scf_tol'])


def cache_key(coords, syms, cfg):
    payload = json.dumps(dict(coords=[list(map(float, p))
                                      for p in np.asarray(coords, float)],
                              syms=list(syms), cfg=cfg), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def point_key(coords):
    return hashlib.sha256(np.asarray(coords, float).tobytes()).hexdigest()[:16]


# ==================== results persistence (per-point, atomic) ====================
def save_results_atomic(results, path):
    tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(results, fh, indent=2, default=str)
    os.replace(tmp, path)


def load_results(path):
    if os.path.exists(path):
        return json.load(open(path))
    return {}


# ==================== entry flow ====================
def run_entry(ctrl, backend_factory, execute=False, results=None,
              results_path=None):
    """Real entry flow with per-point atomic persistence.  execute=False ->
    validation-only.  Any failure raises after saving the scene (error state
    + all existing results).  NEVER auto-continues to the next point."""
    results = {} if results is None else results
    rpath = results_path
    m = json.load(open(GEOM_MANIFEST))
    e_nh3, e_o3, ref_src = load_reference_energies()
    pts = []
    for pt in m['points']:
        p = os.path.join(os.path.dirname(GEOM_MANIFEST), pt['xyz_file'])
        syms, C = read_xyz_primary(p)
        _, C2 = read_xyz_independent(p)
        assert np.abs(C - C2).max() < 1e-12, 'reader disagreement'
        t_rows = [4 + t for t in pt['o3_row_mapping']['terminal_rows']]
        c_row = 4 + pt['o3_row_mapping']['central_row']
        d_avg = float(np.mean([np.linalg.norm(C[0] - C[j]) for j in t_rows]))
        assert abs(d_avg - pt['actual_avg_d']) < 1e-9, \
            'distance mismatch: %s' % pt['xyz_file']
        M = C[t_rows].mean(axis=0)
        Cv = C[c_row]
        u_dir = (M - Cv) / np.linalg.norm(M - Cv)
        assert (C[0] - M) @ u_dir > 0, \
            'N not on terminal side (%s)' % pt['xyz_file']
        pts.append(dict(pt=pt, C_src=C, d_avg=d_avg, t_rows=t_rows,
                        c_row=c_row, u_dir=u_dir))
    for item in pts:
        pt, C_src, d_avg, t_rows, c_row = (item['pt'], item['C_src'],
                                           item['d_avg'], item['t_rows'],
                                           item['c_row'])
        if not execute:
            print('[entry] d=%.1f validation-only (no evaluation)'
                  % pt['target_d'], flush=True)
            continue
        key = point_key(C_src)
        done = ctrl.find_done(key)
        if done is not None:
            rec = results.get(key)
            if rec is None:
                save_results_atomic(results, rpath)
                raise ScanIntegrityError(
                    'ledger done but full result missing for d=%.1f - stop, '
                    'no auto-recalc' % pt['target_d'])
            if rec.get('coords_computed_sha') != key \
                    or not config_compatible(rec.get('config_readback', {})):
                save_results_atomic(results, rpath)
                raise ScanIntegrityError(
                    'ledger/result conflict for d=%.1f (hash or config '
                    'incompatible) - stop' % pt['target_d'])
            print('[entry] d=%.1f reused verified result' % pt['target_d'],
                  flush=True)
            continue
        idx = ctrl.pre_eval('scan', dict(point_key=key,
                                         target_d=pt['target_d'],
                                         results_path=rpath))
        try:
            mol, mf = backend_factory(C_src)
            coords_calc = np.asarray(mol.atom_coords(unit='Angstrom'), float)
            coords_sha = point_key(coords_calc)
            cfg = cfg_readback(mf)
            if not config_compatible(cfg):
                ctrl.mark_error(idx, 'config mismatch: %s' % cfg)
                raise EvaluationFailed('config mismatch at d=%.1f'
                                       % pt['target_d'])
            mf.kernel()
            if not mf.converged:
                ctrl.mark_error(idx, 'SCF not converged')
                raise EvaluationFailed('SCF not converged at d=%.1f'
                                       % pt['target_d'])
            g = np.asarray(mf.nuc_grad_method().kernel(), float).reshape(-1, 3)
            e = float(mf.e_tot)
            if not (np.isfinite(e) and np.isfinite(g).all()):
                ctrl.mark_error(idx, 'non-finite E/g')
                raise EvaluationFailed('non-finite at d=%.1f' % pt['target_d'])
            Cc = coords_calc
            M_c = Cc[t_rows].mean(axis=0)
            C_c = Cc[c_row]
            u_calc = (M_c - C_c) / np.linalg.norm(M_c - C_c)
            g_proj, dE_ds_A = dE_ds_A_from_projection(g, u_calc)
            dd_ds = dd_ds_local(Cc, u_calc, t_rows)
            ds_dd = ds_dd_from_local(dd_ds)
            dE_dd = dE_dd_A(dE_ds_A, ds_dd)
            e_d2 = float(d2_full.d2_energy(mol))
            ck = cache_key(coords_calc, SYMS_EXPECT, cfg)
            rec = dict(
                point_key=key, target_d=pt['target_d'],
                coords_source=C_src.tolist(), coords_computed=coords_calc.tolist(),
                coords_computed_sha=coords_sha, cache_key=ck,
                atom_order=SYMS_EXPECT,
                fragments=dict(NH3=[0, 1, 2, 3],
                               O3_rows=dict(central=c_row, terminals=t_rows)),
                e_total=e, e_d2_analytic=e_d2, e_dft_part=e - e_d2,
                dE_nonCP=e - e_nh3 - e_o3,
                ref_energies=dict(E_NH3=e_nh3, E_O3=e_o3,
                                  source=ref_src['source']),
                grad_full=g.tolist(), grad_max=float(np.abs(g).max()),
                u_from_actual_O3=u_calc.tolist(),
                g_proj_Eh_per_Bohr=g_proj, dE_ds_Eh_per_A=dE_ds_A,
                dd_ds=dd_ds, ds_dd=ds_dd, dE_dd_Eh_per_A=dE_dd,
                d_N_terminals=[float(np.linalg.norm(Cc[0] - Cc[j]))
                               for j in t_rows],
                d_N_central=float(np.linalg.norm(Cc[0] - Cc[c_row])),
                scf_converged=True, config_readback=cfg, status='done')
            ctrl.post_eval(idx, dict(e_total=e, grad_max=rec['grad_max'],
                                     point_key=key, results_path=rpath,
                                     results_sha=hashlib.sha256(
                                         open(rpath, 'rb').read()
                                     ).hexdigest()[:16] if
                                     os.path.exists(rpath) else None))
            results[key] = rec
            save_results_atomic(results, rpath)
            print('[entry] d=%.1f E=%.9f max|g|=%.3e dE/ds=%.4f Eh/A saved'
                  % (pt['target_d'], e, rec['grad_max'], dE_ds_A), flush=True)
        except Exception as ex:
            results['last_error'] = dict(where='d=%.1f' % pt['target_d'],
                                         error=repr(ex))
            save_results_atomic(results, rpath)
            print('[entry] HARD STOP: %s' % ex, flush=True)
            raise        # never auto-continue to next point
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--execute', action='store_true',
                    help='actually evaluate (default: validation-only)')
    ap.add_argument('--init-new-ledger', action='store_true')
    args = ap.parse_args()
    ctrl = Budget(LEDGER, CAPS, init_new=args.init_new_ledger)
    results = load_results(RESULTS_FILE)
    run_entry(ctrl, real_backend, execute=args.execute, results=results,
              results_path=RESULTS_FILE)
    print('entry finished')


if __name__ == '__main__':
    main()
