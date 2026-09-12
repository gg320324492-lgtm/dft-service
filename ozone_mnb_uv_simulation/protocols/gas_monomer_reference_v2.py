import os, sys, json, time, hashlib
import numpy as np
HERE = '/mnt/e/dft-service/ozone_mnb_uv_simulation/protocols'
sys.path.insert(0, HERE)
from sp_ledger import SPLedger
from pyscf import gto
from pyscf.geomopt import berny_solver
from pyscf.lib import GradScanner
import d2_full
from grad_factory import make_mf_d2_gr

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
OUT = ROOT + '/run_artifacts/02_nh3o3_reference/gas_monomer_references'
LEDGER = OUT + '/attempts_v2.json'
SYMS = {'nh3': ['N', 'H', 'H', 'H'], 'o3': ['O', 'O', 'O']}
GRID_LEVEL = 8

CAPS = dict(total=42,
            species_opt={'nh3': 20, 'o3': 20},
            recheck={'nh3': 1, 'o3': 1})


class LedgerMissingError(RuntimeError):
    pass


class BudgetExceeded(RuntimeError):
    pass


class BudgetController:
    """Persistent, category-resolved budget guard (JOB-2026-0906-025).

    Caps: total E/G evals; per-species optimization evals; per-species
    endpoint-recheck evals.  Attempts are persisted BEFORE the evaluation
    (pre_eval) and updated afterwards (post_eval).  Restart reloads the
    ledger; a MISSING ledger file is an error unless explicitly initialised.
    """

    def __init__(self, path, caps, init_new=False):
        self.path = path
        self.caps = caps
        if os.path.exists(path):
            self.data = json.load(open(path))
        elif init_new:
            self.data = dict(caps=caps, attempts=[], results={},
                             initialised='explicit init_new')
            self._save()
        else:
            raise LedgerMissingError(
                'ledger %s missing: refusing to silently open a new quota '
                '(use init_new=True to start a new batch explicitly)' % path)
        self.data.setdefault('attempts', [])
        self.data.setdefault('results', {})

    def used(self, cat):
        return sum(1 for a in self.data['attempts']
                   if a.get('category') == cat and a.get('status') != 'rejected')

    def pre_eval(self, cat, meta):
        """Check ALL relevant caps, then persist the attempt BEFORE eval."""
        total_used = self.used('total')
        if total_used + 1 > self.caps['total']:
            raise BudgetExceeded('TOTAL budget exhausted (%d/%d)'
                                 % (total_used, self.caps['total']))
        if cat.startswith('species:'):
            sid, kind = cat.split(':')[1], cat.split(':')[2]
            if kind == 'opt':
                cap = self.caps['species_opt'][sid]
                used = self.used(cat)
                if used + 1 > cap:
                    raise BudgetExceeded(
                        'per-species OPT budget exhausted for %s (%d/%d) '
                        'even though total budget has balance - rejected '
                        'before evaluation' % (sid, used, cap))
            elif kind == 'recheck':
                cap = self.caps['recheck'][sid]
                used = self.used(cat)
                if used + 1 > cap:
                    raise BudgetExceeded(
                        'per-species RECHECK budget exhausted for %s (%d/%d)'
                        % (sid, used, cap))
        idx = len(self.data['attempts'])
        self.data['attempts'].append(dict(category=cat, status='pre_checked',
                                          meta=meta, n=total_used + 1,
                                          pre_eval_at=time.time()))
        self._save()
        return idx

    def post_eval(self, idx, fields):
        rec = self.data['attempts'][idx]
        rec.update(fields)
        rec['status'] = 'done'
        self._save()

    def mark_error(self, idx, error):
        rec = self.data['attempts'][idx]
        rec['status'] = 'error'
        rec.setdefault('errors', []).append(dict(error=error,
                                                 at=time.time()))
        self._save()

    def _save(self):
        tmp = self.path + '.tmp'
        with open(tmp, 'w') as fh:
            json.dump(self.data, fh, indent=2, default=str)
        os.replace(tmp, self.path)


class BudgetedScanner(GradScanner):
    """Proxy around a real GradScanner.

    Subclasses pyscf.lib.GradScanner so berny_solver accepts it via the
    isinstance branch; __call__ performs the PRE-EVAL budget guard and
    persistence, then delegates to the real scanner (which evaluates at the
    geometry passed in - no stale geometry binding, no kernel patching).
    """

    def __init__(self, scanner, ctrl, cat):
        self.__dict__.update(scanner.__dict__)
        self._scanner = scanner
        self._ctrl = ctrl
        self._cat = cat

    def __call__(self, mol=None):
        coords = np.asarray(mol.atom_coords(unit='Angstrom'), float).tolist()
        idx = self._ctrl.pre_eval(self._cat,
                                  dict(coords_sha=hashlib.sha256(
                                      json.dumps(coords).encode()
                                  ).hexdigest()[:16],
                                      coords=coords))
        try:
            e, g = self._scanner(mol)
        except Exception as ex:
            self._ctrl.mark_error(idx, repr(ex))
            raise
        g = np.asarray(g, float)
        self._ctrl.post_eval(idx, dict(
            e_total=float(e), grad_max=float(np.abs(g).max()),
            grad_rms=float(np.sqrt((g ** 2).mean())),
            scf_converged=bool(getattr(self._scanner, 'converged', True))))
        return e, g


def read_xyz(path):
    lines = open(path).read().splitlines()
    n = int(lines[0].split()[0])
    return [[p.split()[0], float(p.split()[1]), float(p.split()[2]),
             float(p.split()[3])] for p in lines[1:1 + n]]


def build_mol(sid, coords=None):
    atoms = read_xyz(ROOT + '/inputs/nh3o3_phase2/%s_gas.xyz' % sid)
    if coords is not None:
        atoms = [[a[0], c[0], c[1], c[2]] for a, c in zip(atoms, coords)]
    atom = "; ".join("%s %.10f %.10f %.10f" % (a[0], float(a[1]), float(a[2]),
                                               float(a[3])) for a in atoms)
    return gto.M(atom=atom, basis='def2-TZVP', charge=0, spin=0, verbose=0,
                 max_memory=4000)


def readback(mf):
    return dict(grid_level=int(mf.grids.level),
                d2_attached=bool(getattr(mf, '_has_full_d2', False)),
                scf_tol=[float(mf.conv_tol), float(mf.conv_tol_grad)],
                solvent='none (gas phase)')


def main():
    ledger = SPLedger(LEDGER, cap=CAPS['total'])  # legacy ledger (read-only)
    ctrl = BudgetController(OUT + '/budget_v2.json', CAPS)
    print('budget v2: total used=%d/%d' % (ctrl.used('total'),
                                           CAPS['total']), flush=True)
    results = {}
    for sid in ('nh3', 'o3'):
        mol = build_mol(sid)
        mf = make_mf_d2_gr(mol, solvent=None, grid_response=True,
                           grid_level=GRID_LEVEL)
        cfg = readback(mf)
        assert cfg['grid_level'] == GRID_LEVEL and cfg['d2_attached']
        cat_opt = 'species:%s:opt' % sid
        scanner = mf.nuc_grad_method().as_scanner()
        proxy = BudgetedScanner(scanner, ctrl, cat_opt)
        steps = []

        def callback(envs):
            m = envs['mol']
            grad = np.asarray(envs['gradients'],
                              float).reshape(len(m.elements), 3)
            steps.append(dict(step=int(envs['cycle']) + 1,
                              coords_angstrom=np.asarray(
                                  m.atom_coords(unit='Angstrom'),
                                  float).tolist(),
                              e_total=float(envs['energy']),
                              grad_max=float(np.abs(grad).max()),
                              scf_converged=bool(envs['g_scanner'].converged)))
            print('[%s] step %d E=%.9f max|g|=%.3e conv=%s'
                  % (sid, steps[-1]['step'], steps[-1]['e_total'],
                     steps[-1]['grad_max'], steps[-1]['scf_converged']),
                  flush=True)
            json.dump(steps, open(OUT + '/%s_v2_steps.json' % sid, 'w'),
                      indent=2)

        opt_error = None
        try:
            opt_conv, mol_opt = berny_solver.kernel(
                proxy, assert_convergence=True, include_ghost=True,
                callback=callback, maxsteps=CAPS['species_opt'][sid])
        except Exception as e:
            opt_error = repr(e)
            print('[%s] berny exception: %s' % (sid, opt_error), flush=True)
            continue

        # endpoint recheck: fresh objects, guarded SP (category recheck)
        cat_rc = 'species:%s:recheck' % sid
        c_end = np.asarray(mol_opt.atom_coords(unit='Angstrom'), float)
        coords_sha = hashlib.sha256(json.dumps(
            c_end.tolist()).encode()).hexdigest()[:16]
        idx = ctrl.pre_eval(cat_rc, dict(coords_sha=coords_sha))
        mol_v = build_mol(sid, c_end)
        mf_v = make_mf_d2_gr(mol_v, solvent=None, grid_response=True,
                             grid_level=GRID_LEVEL)
        try:
            mf_v.kernel()
            g_v = np.asarray(mf_v.nuc_grad_method().kernel(), float)
            ctrl.post_eval(idx, dict(e_total=float(mf_v.e_tot),
                                     grad_max=float(np.abs(g_v).max())))
            results[sid] = dict(
                coords_angstrom=c_end.tolist(),
                e_total=float(mf_v.e_tot),
                e_d2_analytic=float(d2_full.d2_energy(mol_v)),
                grad_max=float(np.abs(g_v).max()),
                scf_converged=bool(mf_v.converged),
                stationary_candidate=bool(
                    mf_v.converged and np.abs(g_v).max() <= 1e-5),
                label='stationary-point candidate (NO Hessian/frequency)',
                opt_error=opt_error, config_readback=cfg)
            print('[%s] RECHECK E=%.9f max|g|=%.3e candidate=%s'
                  % (sid, results[sid]['e_total'], results[sid]['grad_max'],
                     results[sid]['stationary_candidate']), flush=True)
        except Exception as e:
            ctrl.mark_error(idx, repr(e))
            raise
    json.dump(results, open(OUT + '/monomer_results_v2.json', 'w'), indent=2)
    if all(k in results for k in ('nh3', 'o3')):
        e_sum = results['nh3']['e_total'] + results['o3']['e_total']
        results['sum_of_monomer_reference_energies_hartree'] = e_sum
        json.dump(results, open(OUT + '/monomer_results_v2.json', 'w'),
                  indent=2)
        print('SUM of monomer reference energies = %.15f Eh' % e_sum,
              flush=True)
    print('V2 DRIVER DONE: total evals=%d/%d'
          % (ctrl.used('total'), CAPS['total']), flush=True)


if __name__ == '__main__':
    main()
