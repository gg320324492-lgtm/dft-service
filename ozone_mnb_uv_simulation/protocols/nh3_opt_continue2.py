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
BASE_EVALS = 35
REMAIN = CAP - BASE_EVALS   # 14
MAXSTEPS = 3
CONV_PARAMS = dict(gradientmax=3e-5, gradientrms=1e-5, stepmax=6e-4, steprms=4e-4)
SYMS = ['N', 'H', 'H', 'H']

ledger = SPLedger(LEDGER, cap=CAP)
print('ledger attempts=%d | evals consumed=%d | remaining=%d'
      % (ledger.attempts_used(), BASE_EVALS, REMAIN), flush=True)

v4 = json.load(open(OUT + '/nh3_v4_result.json'))
rc = np.asarray(v4['v3_final']['coords_angstrom'], float)

def build(coords):
    atom = "; ".join("%s %.10f %.10f %.10f" % (s, x, y, z)
                     for s, (x, y, z) in zip(SYMS, coords))
    return gto.M(atom=atom, basis='def2-TZVP', charge=0, spin=0, verbose=0,
                 max_memory=4000)

TOTAL = {'n': BASE_EVALS}
steps = []

def callback(envs):
    TOTAL['n'] += 1
    if TOTAL['n'] > CAP:
        raise RuntimeError('BATCH BUDGET EXHAUSTED (%d evals)' % CAP)
    m = envs['mol']
    grad = np.asarray(envs['gradients'], float).reshape(len(m.elements), 3)
    # ledger: post-computation record (deviation from pre-call recording is
    # documented: pre-call patching freezes the scanner mol -> broken grads)
    idx = ledger.start_attempt(
        json.dumps(['nh3_continue4', TOTAL['n']], sort_keys=True),
        dict(tag='nh3_continue4', kind='E/G eval', n=TOTAL['n']))
    ledger.update(idx, status='done',
                  grad_max=float(np.abs(grad).max()),
                  e_total=float(envs['energy']))
    rec = dict(step=int(envs['cycle']) + 1,
               coords_angstrom=np.asarray(m.atom_coords(unit='Angstrom'),
                                          float).tolist(),
               e_total=float(envs['energy']),
               grad_max=float(np.abs(grad).max()),
               grad_rms=float(np.sqrt((grad ** 2).mean())),
               scf_converged=bool(envs['g_scanner'].converged),
               total_evals=TOTAL['n'])
    steps.append(rec)
    print('[v5] step %d E=%.9f max|g|=%.3e evals=%d conv=%s'
          % (rec['step'], rec['e_total'], rec['grad_max'],
             rec['total_evals'], rec['scf_converged']), flush=True)
    json.dump(steps, open(OUT + '/nh3_v5_steps.json', 'w'), indent=2)

opt_error = None
mol_opt = None
mol = build(rc)
mf = make_mf_d2_gr(mol, solvent=None, grid_response=True, grid_level=8)
try:
    opt_conv, mol_opt = berny_solver.kernel(
        mf, assert_convergence=True, include_ghost=True,
        callback=callback, maxsteps=MAXSTEPS, **CONV_PARAMS)
except Exception as e:
    import traceback
    traceback.print_exc()
    opt_error = repr(e)
    print('[v5] berny exception: %s' % opt_error, flush=True)

final = None
if steps:
    c_end = np.asarray(steps[-1]['coords_angstrom'], float)
    mol_v = build(c_end)
    mf_v = make_mf_d2_gr(mol_v, solvent=None, grid_response=True, grid_level=8)
    if TOTAL['n'] < CAP:
        TOTAL['n'] += 1
        t0 = time.time()
        mf_v.kernel()
        g_v = np.asarray(mf_v.nuc_grad_method().kernel(), float).reshape(-1, 3)
        final = dict(coords_angstrom=c_end.tolist(),
                     e_total=float(mf_v.e_tot),
                     e_d2_analytic=float(d2_full.d2_energy(mol_v)),
                     grad_max=float(np.abs(g_v).max()),
                     scf_converged=bool(mf_v.converged),
                     opt_error=opt_error,
                     stationary_candidate=bool(
                         mf_v.converged and np.abs(g_v).max() <= 1e-5),
                     label='stationary-point candidate (unprojected grad '
                           'check; NO Hessian/frequency)')
        print('[v5] FINAL E=%.9f max|g|=%.3e evals=%d candidate=%s'
              % (final['e_total'], final['grad_max'], TOTAL['n'],
                 final['stationary_candidate']), flush=True)
        try:
            t = time.time()
            st = mf_v.stability()
            final['stability'] = dict(
                tool='pyscf RHF/RKS internal stability',
                seconds=round(time.time() - t, 1), result=str(st[1]),
                note='no UKS switch, no following; does not exclude '
                     'multireference')
            print('[v5] RKS stability: %s' % str(st[1]), flush=True)
        except Exception as se:
            final['stability'] = dict(error=repr(se))
    json.dump(dict(v3_final=final, total_evals=TOTAL['n'], cap=CAP,
                   steps=steps),
              open(OUT + '/nh3_v5_result.json', 'w'), indent=2)
    print('V5 DONE: total_evals=%d/%d' % (TOTAL['n'], CAP), flush=True)
