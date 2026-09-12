import os, sys, json, hashlib, time
import numpy as np
HERE = '/mnt/e/dft-service/ozone_mnb_uv_simulation/protocols'
sys.path.insert(0, HERE)
from sp_ledger import SPLedger
from pyscf import gto
import d2_full
from grad_factory import make_mf_d2_gr

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
GEOM_MANIFEST = ROOT + '/inputs/nh3o3_phase2/c1_scan_v2_std/scan_geometry_manifest_std.json'
OUT_DIR = ROOT + '/inputs/nh3o3_phase2/c1_scan_v2'
LEDGER = ROOT + '/run_artifacts/02_nh3o3_reference/c1_adduct/budget_scan_exec_v3.json'
CAPS = dict(total=6, scan=6)
BOHR_PER_A = 1.8897261254578281     # Bohr per Angstrom
SYMS_EXPECT = ['N', 'H', 'H', 'H', 'O', 'O', 'O']
GRID_LEVEL = 8
E_NH3_REF = -56.564219135911        # JOB-026 full precision
E_O3_REF = -225.433902699575


class ScanIntegrityError(RuntimeError):
    """Manifest/hash/distance verification failure -> hard stop."""


class EvaluationFailed(RuntimeError):
    """SCF failure / non-finite / config mismatch -> hard stop."""


# ============================ XYZ I/O ============================
def write_standard_xyz(path, symbols, coords, comment):
    """Standard XYZ: line 1 = atom count, line 2 = comment, then coords (A)."""
    lines = [str(len(symbols)), comment]
    for s, (x, y, z) in zip(symbols, coords):
        lines.append('%-2s %20.12f %20.12f %20.12f' % (s, x, y, z))
    open(path, 'w').write('\n'.join(lines) + '\n')


def read_xyz_primary(path):
    """Primary strict reader: count / element order / exactly-7 / finiteness."""
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
    """Independent generic reader (different parsing path) for cross-check."""
    tokens = open(path).read().split('\n')
    n = int(tokens[0].strip())
    body = tokens[2:2 + n]
    rows = [b.split() for b in body]
    syms = [r[0] for r in rows]
    C = np.array([[float(v) for v in r[1:4]] for r in rows])
    assert C.shape == (7, 3) and n == 7
    assert syms == SYMS_EXPECT
    assert np.isfinite(C).all()
    return syms, C


# ==================== local derivatives (per geometry) ====================
def dE_ds_A_from_projection(g_full, u_dir):
    """g_proj [Eh/Bohr] = sum over NH3 atoms of g_i . u;
    dE/ds [Eh/A] = g_proj * BOHR_PER_A  (1 Bohr = 1/1.8897 A)."""
    g_proj = float(np.sum(g_full[:4] @ u_dir))     # Eh/Bohr
    return g_proj, g_proj * BOHR_PER_A


def dd_ds_local(coords, u_dir, term_rows):
    """dd/ds = 1/2 * sum_j [ (N-O_j).u / |N-O_j| ] -- local to this geometry,
    terminal asymmetry preserved (no secant across sparse samples)."""
    N = coords[0]
    tot = 0.0
    for j in term_rows:
        v = N - coords[j]
        r = float(np.linalg.norm(v))
        tot += float(v @ u_dir) / r
    return 0.5 * tot


def ds_dd_from_local(dd_ds):
    assert abs(dd_ds) > 1e-12, 'dd/ds ~ 0: ds/dd undefined'
    return 1.0 / dd_ds


def dE_dd_A(dE_ds_A, ds_dd):
    return dE_ds_A * ds_dd


# ==================== budget ====================
class Budget:
    """Persistent budget: TOTAL counts ALL actual attempts (any category);
    per-category caps are enforced additionally.  Missing ledger is an
    error unless explicitly initialised."""

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
                               'create new quota (init_new=True to start a '
                               'new batch explicitly)' % path)
        self.data.setdefault('attempts', [])

    def used_total(self):
        return len(self.data['attempts'])       # ALL actual attempts

    def used_cat(self, cat):
        return sum(1 for a in self.data['attempts']
                   if a.get('category') == cat and a.get('status') != 'rejected')

    def pre_eval(self, cat, meta):
        if self.used_total() + 1 > self.caps['total']:
            raise RuntimeError('TOTAL budget exhausted (%d/%d) - rejected '
                               'before evaluation' % (self.used_total(),
                                                      self.caps['total']))
        if self.used_cat(cat) + 1 > self.caps[cat]:
            raise RuntimeError('%s budget exhausted (%d/%d) - rejected '
                               'before evaluation'
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
    """No-SCF fake mf: harmonic radial energy along x (Bohr-frame value
    interpreted via the entry's own unit handling).  Can be told to fail."""

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
        g = self._grad if self.fail_finite is False and self.converged \
            else np.full_like(self._grad, np.nan if self.fail_finite else 0.0)
        outer = self

        class _G:
            def kernel(self_):
                return g
        return _G()


def sim_backend_factory(fail_at=None, harmonic=None):
    """Return a backend_factory for the simulated PES.  fail_at: d value at
    which SCF fails."""
    kk, x0 = harmonic if harmonic else (0.05, 4.0)

    class MolStub:
        def __init__(self, c):
            self._c = np.asarray(c, float)
            self.elements = SYMS_EXPECT
            self.natm = len(SYMS_EXPECT)
        def atom_symbol(self, i):
            return SYMS_EXPECT[i]
        def atom_charge(self, i):
            return {'N': 7, 'H': 1, 'O': 8}[SYMS_EXPECT[i]]
        def atom_coords(self, unit='Bohr'):
            return self._c * BOHR_PER_A if unit == 'Bohr' else self._c

    def factory(coords_A):
        x_A = float(coords_A[0, 0])
        e = 0.5 * kk * (x_A - x0) ** 2 - 1.0
        g = np.zeros((7, 3))
        g[0, 0] = kk * (x_A - x0)
        fail = fail_at is not None and abs(x_A - fail_at) < 5e-3
        return MolStub(coords_A), SimulatedBackend(e, g, fail_scf=fail)
    return factory


# ==================== entry flow ====================
def point_key(coords):
    return hashlib.sha256(np.asarray(coords, float).tobytes()).hexdigest()[:16]


def cfg_readback(mf):
    return dict(grid_level=int(getattr(mf.grids, 'level', -1)),
                d2_attached=bool(getattr(mf, '_has_full_d2', False)),
                scf_tol=[float(mf.conv_tol), float(mf.conv_tol_grad)])


def run_entry(ctrl, backend_factory, execute=False, results=None):
    """Real entry flow.  execute=False -> validation-only (no evaluation).
    ANY evaluation failure raises after saving the scene -- the entry NEVER
    automatically continues to the next point (JOB-2026-0906-030)."""
    results = {} if results is None else results
    m = json.load(open(GEOM_MANIFEST))
    pts = []
    for pt in m['points']:
        p = os.path.join(os.path.dirname(GEOM_MANIFEST), pt['xyz_file'])
        syms, C = read_xyz_primary(p)
        _, C2 = read_xyz_independent(p)
        assert np.abs(C - C2).max() < 1e-12, 'reader disagreement'
        t_rows = [4 + t for t in pt['o3_row_mapping']['terminal_rows']]
        d_avg = float(np.mean([np.linalg.norm(C[0] - C[j]) for j in t_rows]))
        assert abs(d_avg - pt['actual_avg_d']) < 1e-9, 'distance mismatch'
        pts.append((pt, C, d_avg, t_rows))
    u_dir = np.asarray([1.0, 0.0, 0.0], float)   # recorded bisector direction
    u_dir = u_dir / np.linalg.norm(u_dir)

    for pt, coords, d_avg, t_rows in pts:
        key = point_key(coords)
        if ctrl.find_done(key) is not None:
            print('[entry] d=%.1f already done -> reused' % pt['target_d'],
                  flush=True)
            continue
        if not execute:
            print('[entry] d=%.1f validation-only (no evaluation)'
                  % pt['target_d'], flush=True)
            continue
        idx = ctrl.pre_eval('scan', dict(point_key=key,
                                         target_d=pt['target_d']))
        try:
            mol, mf = backend_factory(coords)
            lvl = int(getattr(mf.grids, 'level', -1))
            if lvl != GRID_LEVEL or not getattr(mf, '_has_full_d2', False):
                ctrl.mark_error(idx, 'config mismatch')
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
            g_proj, dE_ds_A = dE_ds_A_from_projection(g, u_dir)
            dd_ds = dd_ds_local(coords, u_dir, t_rows)
            ds_dd = ds_dd_from_local(dd_ds)
            dE_dd = dE_dd_A(dE_ds_A, ds_dd)
            e_d2 = float(d2_full.d2_energy(mol))
            rec = dict(point_key=key, target_d=pt['target_d'],
                       coords_computed=coords.tolist(),
                       e_total=e, e_d2_analytic=e_d2, e_dft_part=e - e_d2,
                       grad_full=g.tolist(), grad_max=float(np.abs(g).max()),
                       g_proj_Eh_per_Bohr=g_proj,
                       dE_ds_Eh_per_A=dE_ds_A,
                       dd_ds=dd_ds, ds_dd=ds_dd, dE_dd_Eh_per_A=dE_dd,
                       dE_nonCP=e - E_NH3_REF - E_O3_REF,
                       scf_converged=True, config_readback=cfg_readback(mf),
                       status='done')
            ctrl.post_eval(idx, dict(e_total=e,
                                     grad_max=float(np.abs(g).max())))
            results[key] = rec
            print('[entry] d=%.1f E=%.9f max|g|=%.3e dE/ds=%.4f Eh/A'
                  % (pt['target_d'], e, rec['grad_max'], dE_ds_A), flush=True)
        except EvaluationFailed:
            results['last_error'] = 'evaluation failed at d=%.1f' \
                % pt['target_d']
            print('[entry] HARD STOP: evaluation failed at d=%.1f'
                  % pt['target_d'], flush=True)
            raise                       # never auto-continue to next point
        except Exception as ex:
            results['last_error'] = repr(ex)
            print('[entry] HARD STOP: %s' % ex, flush=True)
            raise                       # never auto-continue to next point
    return results


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--execute', action='store_true',
                    help='actually evaluate (default: validation-only)')
    ap.add_argument('--init-new-ledger', action='store_true')
    args = ap.parse_args()
    ctrl = Budget(LEDGER, CAPS, init_new=args.init_new_ledger)
    rfile = OUT_DIR + '/exec_results_v3.json'
    results = json.load(open(rfile)) if os.path.exists(rfile) else {}
    run_entry(ctrl, real_backend, execute=args.execute, results=results)
    json.dump(results, open(rfile, 'w'), indent=2)


if __name__ == '__main__':
    main()
