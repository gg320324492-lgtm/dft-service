import os, sys, json, time, hashlib
import numpy as np
HERE = '/mnt/e/dft-service/ozone_mnb_uv_simulation/protocols'
sys.path.insert(0, HERE)
from sp_ledger import SPLedger
from pyscf import gto
from pyscf.geomopt import berny_solver
import d2_full
from grad_factory import make_mf_d2_gr

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
OUT = ROOT + '/run_artifacts/02_nh3o3_reference/gas_monomer_references'
os.makedirs(OUT, exist_ok=True)
LEDGER = OUT + '/attempts.json'
CAP = 42
MAXOPT = 20          # per-monomer optimizer E/G eval cap
GMAX = 1e-5          # unprojected endpoint criterion (reference)
GRID_LEVEL = 8

def sha(o):
    if isinstance(o, (bytes, bytearray)):
        return hashlib.sha256(o).hexdigest()[:16]
    return hashlib.sha256(json.dumps(o, sort_keys=True).encode()).hexdigest()[:16]

ledger = SPLedger(LEDGER, cap=CAP)
print('ledger: attempts=%d cap=%d budget_left=%d'
      % (ledger.attempts_used(), ledger.cap, ledger.budget_left()), flush=True)

manifest = json.load(open(ROOT + '/references/literature/manifest.json'))
prov = json.load(open(ROOT + '/run_artifacts/02_nh3o3_reference/precheck_noSCF.json'))

SYMS = {'nh3': ['N', 'H', 'H', 'H'], 'o3': ['O', 'O', 'O']}
inputs = prov['inputs_roundtrip']
def read_xyz(path):
    lines = open(path).read().splitlines()
    n = int(lines[0].split()[0])
    return [[p.split()[0], float(p.split()[1]), float(p.split()[2]), float(p.split()[3])]
            for p in lines[1:1 + n]]
ATOMS = {'nh3': read_xyz(ROOT + '/inputs/nh3o3_phase2/nh3_gas.xyz'),
         'o3': read_xyz(ROOT + '/inputs/nh3o3_phase2/o3_gas.xyz')}

def build_mol(sid):
    atoms = ATOMS[sid]
    atom = "; ".join("%s %.10f %.10f %.10f" % (a[0], float(a[1]), float(a[2]), float(a[3]))
                     for a in atoms)
    return gto.M(atom=atom, basis='def2-TZVP', charge=0, spin=0, verbose=0,
                 max_memory=4000)

def readback(mf):
    return dict(grid_level=int(mf.grids.level),
                d2_attached=bool(getattr(mf, '_has_full_d2', False)),
                scf_tol=[float(mf.conv_tol), float(mf.conv_tol_grad)],
                solvent='none (gas phase)')

results = {}
for sid in ('nh3', 'o3'):
    tag = sid + '_gas_opt'
    mol = build_mol(sid)
    mf = make_mf_d2_gr(mol, solvent=None, grid_response=True, grid_level=GRID_LEVEL)
    cfg = readback(mf)
    assert cfg['grid_level'] == GRID_LEVEL and cfg['d2_attached']
    print('[%s] config readback: %s' % (sid, cfg), flush=True)

    steps = []
    state = {'n': 0}

    def counted_grad():
        # budget check + attempt persisted BEFORE the compute (019 mechanism)
        if ledger.budget_left() <= 0 or state['n'] >= MAXOPT:
            raise RuntimeError('BUDGET EXHAUSTED for %s (opt evals=%d, cap=%d)'
                               % (tag, state['n'], MAXOPT))
        key = make_key = json.dumps([tag, state['n'] + 1], sort_keys=True)
        idx = ledger.start_attempt(key, dict(tag=tag, kind='opt_eval',
                                             n=state['n'] + 1))
        gobj = _orig_ngm()
        _k = gobj.kernel
        def k(**kw):
            g = _k(**kw)
            state['n'] += 1
            ledger.update(idx, status='done',
                          grad_max=float(np.abs(np.asarray(g, float)).max()))
            return g
        gobj.kernel = k
        return gobj
    _orig_ngm = mf.nuc_grad_method
    mf.nuc_grad_method = counted_grad

    def callback(envs):
        m = envs['mol']
        grad = np.asarray(envs['gradients'], float).reshape(len(m.elements), 3)
        rec = dict(step=int(envs['cycle']) + 1,
                   coords_angstrom=np.asarray(m.atom_coords(unit='Angstrom'),
                                              float).tolist(),
                   e_total=float(envs['energy']),
                   e_d2_analytic=float(d2_full.d2_energy(m)),
                   grad_max=float(np.abs(grad).max()),
                   grad_rms=float(np.sqrt((grad ** 2).mean())),
                   scf_converged=bool(envs['g_scanner'].converged),
                   evals_so_far=state['n'])
        steps.append(rec)
        print('[%s] step %2d E=%.9f max|g|=%.3e evals=%d conv=%s'
              % (sid, rec['step'], rec['e_total'], rec['grad_max'],
                 rec['evals_so_far'], rec['scf_converged']), flush=True)
        json.dump(steps, open(OUT + '/%s_steps.json' % sid, 'w'), indent=2)

    t0 = time.time()
    opt_error = None
    mol_opt = None
    try:
        opt_conv, mol_opt = berny_solver.kernel(
            mf, assert_convergence=True, include_ghost=True,
            callback=callback, maxsteps=MAXOPT)
    except Exception as e:
        opt_error = repr(e)
        print('[%s] berny exception: %s' % (sid, opt_error), flush=True)
        mol_opt = None
        if steps:
            # preserve the site: last recorded step coords
            class _M:
                elements = SYMS[sid]
                def atom_coords(self, unit='Angstrom'):
                    return np.asarray(steps[-1]['coords_angstrom'], float)
                def atom_coords_list(self):
                    return steps[-1]['coords_angstrom']
            class _MO:
                def __init__(self, els, c):
                    self.elements = els
                    self._c = np.asarray(c, float)
                def atom_coords(self, unit='Angstrom'):
                    return self._c
            mol_opt = _MO(SYMS[sid], steps[-1]['coords_angstrom'])

    # endpoint recheck with a FRESH object (independent of optimizer internals)
    recheck = None
    stability = None
    if mol_opt is not None:
        c_end = np.asarray(mol_opt.atom_coords(unit='Angstrom'), float)
        mol_v = build_mol(sid)
        mol_v.set_geom_(c_end, unit='Angstrom')
        mf_v = make_mf_d2_gr(mol_v, solvent=None, grid_response=True,
                             grid_level=GRID_LEVEL)
        if ledger.budget_left() > 0:
            idx = ledger.start_attempt(json.dumps([tag, 'endpoint_recheck'],
                                                  sort_keys=True),
                                       dict(tag=tag, kind='endpoint_recheck'))
            t1 = time.time()
            mf_v.kernel()
            g_v = np.asarray(mf_v.nuc_grad_method().kernel(), float).reshape(-1, 3)
            ledger.update(idx, status='done',
                          grad_max=float(np.abs(g_v).max()),
                          seconds=round(time.time() - t1, 1))
            recheck = dict(
                coords_angstrom=c_end.tolist(),
                e_total=float(mf_v.e_tot),
                e_d2_analytic=float(d2_full.d2_energy(mol_v)),
                grad_max=float(np.abs(g_v).max()),
                grad_rms=float(np.sqrt((g_v ** 2).mean())),
                scf_converged=bool(mf_v.converged),
                finite=bool(np.isfinite(mf_v.e_tot) and np.isfinite(g_v).all()))
            print('[%s] ENDPOINT RECHECK E=%.9f max|g|=%.3e conv=%s'
                  % (sid, recheck['e_total'], recheck['grad_max'],
                     recheck['scf_converged']), flush=True)
            # RKS internal stability analysis (if tool supports; no following)
            try:
                t2 = time.time()
                st = mf_v.stability()
                stability = dict(
                    tool='pyscf RHF/RKS internal stability (mf.stability)',
                    seconds=round(time.time() - t2, 1),
                    result=str(st[1]),
                    note='internal stability only; no UKS switch, no '
                         'instability following; does not exclude '
                         'multireference')
                print('[%s] RKS stability: %s (%.1fs)'
                      % (sid, str(st[1]), stability['seconds']), flush=True)
            except Exception as se:
                stability = dict(tool='pyscf mf.stability',
                                 error=repr(se),
                                 note='listed as pending verification')
                print('[%s] stability analysis unavailable: %s'
                      % (sid, se), flush=True)

    last = steps[-1] if steps else None
    results[sid] = dict(
        tag=tag, input_source=prov['inputs_roundtrip'],
        input_source_label=prov and json.load(open(
            ROOT + '/run_artifacts/02_nh3o3_reference/precheck_noSCF.json')),
        config_readback=cfg, opt_error=opt_error,
        n_opt_evals=state['n'], n_steps=len(steps),
        last_step=last, endpoint_recheck=recheck, stability=stability,
        stationary_candidate=(bool(recheck and recheck['finite']
                                   and recheck['scf_converged']
                                   and recheck['grad_max'] <= GMAX)
                              if recheck else False),
        label='stationary-point candidate (unprojected grad check only; '
              'NO Hessian/frequency -> NOT an accepted minimum; no ZPE/H/G)')
    json.dump(results, open(OUT + '/monomer_results.json', 'w'), indent=2)

# sum of same-method separated monomer reference energies
if all('endpoint_recheck' in results.get(k, {}) and
       results[k]['endpoint_recheck'] for k in ('nh3', 'o3')):
    e_sum = results['nh3']['endpoint_recheck']['e_total'] + \
        results['o3']['endpoint_recheck']['e_total']
    results['sum_of_monomer_reference_energies'] = dict(
        e_total_hartree=e_sum,
        note='E(NH3) + E(O3) under the SAME batch method; NO adduct '
             'computed -> NO binding energy')
json.dump(results, open(OUT + '/monomer_results.json', 'w'), indent=2)
print('BATCH DONE: attempts=%d cap=%d' % (ledger.attempts_used(), ledger.cap),
      flush=True)
