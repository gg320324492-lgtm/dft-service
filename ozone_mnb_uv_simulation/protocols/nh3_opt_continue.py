import os, sys, json, time
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
LEDGER = OUT + '/attempts.json'
CAP = 42
BASE_EVALS = 21          # actual E/G evals already consumed (audited)
REMAIN = CAP - BASE_EVALS   # 21
MAXSTEPS = 5             # remaining per-monomer opt allowance (15/20 used)
SYMS = ['N', 'H', 'H', 'H']

ledger = SPLedger(LEDGER, cap=CAP)
print('ledger attempts recorded=%d (granularity corrected: actual evals consumed=21, remaining=%d)'
      % (ledger.attempts_used(), REMAIN), flush=True)

res = json.load(open(OUT + '/monomer_results.json'))
rc = np.asarray(res['nh3']['endpoint_recheck']['coords_angstrom'], float)

def build(coords):
    atom = "; ".join("%s %.10f %.10f %.10f" % (s, x, y, z)
                     for s, (x, y, z) in zip(SYMS, coords))
    return gto.M(atom=atom, basis='def2-TZVP', charge=0, spin=0, verbose=0,
                 max_memory=4000)

TOTAL = {'n': BASE_EVALS}

def counted_mf(coords):
    mol = build(coords)
    mf = make_mf_d2_gr(mol, solvent=None, grid_response=True, grid_level=8)
    _ngm = mf.nuc_grad_method
    def ngm():
        gobj = _ngm()
        _k = gobj.kernel
        def k(**kw):
            if TOTAL['n'] >= CAP:
                raise RuntimeError('BATCH BUDGET EXHAUSTED (42 E/G evals)')
            idx = ledger.start_attempt(
                json.dumps(['nh3_continue', TOTAL['n'] + 1], sort_keys=True),
                dict(tag='nh3_continue', kind='E/G eval', n=TOTAL['n'] + 1))
            g = _k(**kw)
            TOTAL['n'] += 1
            ledger.update(idx, status='done', n=TOTAL['n'])
            return g
        gobj.kernel = k
        return gobj
    mf.nuc_grad_method = ngm
    return mol, mf

steps = []
def callback(envs):
    m = envs['mol']
    grad = np.asarray(envs['gradients'], float).reshape(len(m.elements), 3)
    rec = dict(step=int(envs['cycle']) + 1,
               coords_angstrom=np.asarray(m.atom_coords(unit='Angstrom'),
                                          float).tolist(),
               e_total=float(envs['energy']),
               grad_max=float(np.abs(grad).max()),
               grad_rms=float(np.sqrt((grad ** 2).mean())),
               scf_converged=bool(envs['g_scanner'].converged),
               total_evals=TOTAL['n'])
    steps.append(rec)
    print('[cont] step %d E=%.9f max|g|=%.3e total_evals=%d conv=%s'
          % (rec['step'], rec['e_total'], rec['grad_max'],
             rec['total_evals'], rec['scf_converged']), flush=True)
    json.dump(steps, open(OUT + '/nh3_continue_steps.json', 'w'), indent=2)

# step 0: rebuild g at rc (deterministic, 1 eval) and form the displaced start
mol0, mf0 = counted_mf(rc)
mf0.kernel()
g0 = np.asarray(mf0.nuc_grad_method().kernel(), float).reshape(-1, 3)
ghat = g0 / np.linalg.norm(g0)
c_start = rc - 0.05 * ghat
print('[cont] start displaced geometry, |g| at rc was %.3e' % np.abs(g0).max(),
      flush=True)

opt_error = None
mol_opt = None
try:
    opt_conv, mol_opt = berny_solver.kernel(
        mf0, assert_convergence=True, include_ghost=True,
        callback=callback, maxsteps=MAXSTEPS)
except Exception as e:
    import traceback
    traceback.print_exc()
    opt_error = repr(e)
    print('[cont] berny exception: %s' % opt_error, flush=True)

# endpoint verification with fresh objects (1 eval)
final = None
if steps:
    c_end = np.asarray(steps[-1]['coords_angstrom'], float)
else:
    c_end = c_start
mol_v, mf_v = counted_mf(c_end)
mf_v.kernel()
g_v = np.asarray(mf_v.nuc_grad_method().kernel(), float).reshape(-1, 3)
final = dict(coords_angstrom=c_end.tolist(),
             e_total=float(mf_v.e_tot),
             e_d2_analytic=float(d2_full.d2_energy(mol_v)),
             grad_max=float(np.abs(g_v).max()),
             scf_converged=bool(mf_v.converged),
             opt_error=opt_error,
             stationary_candidate=bool(mf_v.converged and
                                       np.abs(g_v).max() <= 1e-5),
             label='stationary-point candidate (unprojected grad check; '
                   'NO Hessian/frequency)')
print('[cont] FINAL E=%.9f max|g|=%.3e evals_total=%d candidate=%s'
      % (final['e_total'], final['grad_max'], TOTAL['n'],
         final['stationary_candidate']), flush=True)

# stability at the final endpoint (no E/G evals)
try:
    t = time.time()
    st = mf_v.stability()
    final['stability'] = dict(tool='pyscf RHF/RKS internal stability',
                              seconds=round(time.time() - t, 1),
                              result=str(st[1]),
                              note='no UKS switch, no following; does not '
                                   'exclude multireference')
    print('[cont] RKS stability: %s' % str(st[1]), flush=True)
except Exception as se:
    final['stability'] = dict(error=repr(se))

json.dump(dict(continuation=final, total_evals=TOTAL['n'], cap=CAP),
          open(OUT + '/nh3_continuation_result.json', 'w'), indent=2)
print('CONTINUATION DONE: total_evals=%d/%d' % (TOTAL['n'], CAP), flush=True)
