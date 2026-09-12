import os, sys, json, time
import numpy as np
HERE = '/mnt/e/dft-service/ozone_mnb_uv_simulation/protocols'
sys.path.insert(0, HERE)
from sp_ledger import SPLedger
from pyscf import gto
import d2_full
from grad_factory import make_mf_d2_gr

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
OUT = ROOT + '/run_artifacts/02_nh3o3_reference/gas_monomer_references'
ledger = SPLedger(OUT + '/attempts.json', cap=42)
res = json.load(open(OUT + '/monomer_results.json'))
rc = np.asarray(res['nh3']['endpoint_recheck']['coords_angstrom'], float)
E1 = res['nh3']['endpoint_recheck']['e_total']
g1max = res['nh3']['endpoint_recheck']['grad_max']

SYMS = ['N', 'H', 'H', 'H']

def sp(coords, tag):
    if ledger.budget_left() <= 0:
        raise RuntimeError('budget exhausted')
    atom = "; ".join("%s %.10f %.10f %.10f" % (s, x, y, z)
                     for s, (x, y, z) in zip(SYMS, coords))
    mol = gto.M(atom=atom, basis='def2-TZVP', charge=0, spin=0, verbose=0,
                max_memory=4000)
    mf = make_mf_d2_gr(mol, solvent=None, grid_response=True, grid_level=8)
    idx = ledger.start_attempt(json.dumps([tag], sort_keys=True),
                               dict(tag=tag, kind='diag_sp'))
    t0 = time.time()
    mf.kernel()
    g = np.asarray(mf.nuc_grad_method().kernel(), float).reshape(-1, 3)
    ledger.update(idx, status='done', e_total=float(mf.e_tot),
                  grad_max=float(np.abs(g).max()),
                  seconds=round(time.time() - t0, 1))
    return float(mf.e_tot), g, bool(mf.converged)

out = {}
# 1) determinism: repeat the identical fresh calculation
E2, g2, c2 = sp(rc, 'diag_repeat_same_coords')
out['repeat'] = dict(E=E2, grad_max=float(np.abs(g2).max()),
                     dE_vs_recheck=E2 - E1,
                     recheck_grad_max=g1max)
# 2) displaced along -g2 (should go DOWNHILL if consistent)
ghat = g2 / (np.linalg.norm(g2) + 1e-300)
for sgn, name in ((-1.0, 'minus_g'), (+1.0, 'plus_g')):
    c = rc + sgn * 0.05 * ghat
    E, g, cv = sp(c, 'diag_disp_%s' % name)
    out['disp_%s' % name] = dict(E=E, dE=E - E1, grad_max=float(np.abs(g).max()),
                                 conv=cv)
out['predicted_downhill'] = -0.05 * float(np.linalg.norm(g2))  # Eh, approx
json.dump(out, open(OUT + '/diagnostic_consistency.json', 'w'), indent=2)
print(json.dumps(out, indent=2))
