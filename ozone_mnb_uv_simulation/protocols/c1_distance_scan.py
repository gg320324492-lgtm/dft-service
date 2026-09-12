import os, sys, json, time, hashlib
import numpy as np
HERE = '/mnt/e/dft-service/ozone_mnb_uv_simulation/protocols'
sys.path.insert(0, HERE)
from sp_ledger import SPLedger
from pyscf import gto
import d2_full
from grad_factory import make_mf_d2_gr

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
OUT = ROOT + '/run_artifacts/02_nh3o3_reference/c1_adduct'
LEDGER = OUT + '/budget_scan.json'
CAPS = dict(total=6, scan=6)

ledger = SPLedger(LEDGER, cap=CAPS['total'])

guess_lines = open(ROOT + '/inputs/nh3o3_phase2/c1_guess_S13fig1.xyz').read().splitlines()
G = np.asarray([[float(x) for x in l.split()[1:4]] for l in guess_lines[1:]], float)
SYMS = [l.split()[0] for l in guess_lines[1:]]
N0 = G[0].copy()
O3_FIXED = G[4:7].copy()                      # rows 5O, 7O, 6O
x_6O = O3_FIXED[2, 0]
z_term = float(abs(O3_FIXED[0, 2]))
x_term = float(O3_FIXED[0, 0])
# O3 angle-bisector direction (6O -> terminal midpoint), pre-recorded:
mid = O3_FIXED[:2].mean(axis=0)
u = mid - O3_FIXED[2]
u = np.array([1.0, 0.0, 0.0]) if np.linalg.norm(u) < 1e-9 else u / np.linalg.norm(u)
d0 = float(np.mean([np.linalg.norm(N0 - O3_FIXED[i]) for i in (0, 1)]))
print('bisector u=%s  d0(N-terminal avg)=%.6f A' % (u.tolist(), d0), flush=True)

TARGETS = [2.6, 2.8, 3.0, 3.2, 3.6, 4.2]
E_NH3 = -56.564219135911
E_O3 = -225.433902699575
D2_NH3 = d2_full.d2_energy(mol=None) if False else None
# monomer D2: pure-function evaluations at the JOB-026 monomer geometries
v5 = json.load(open(OUT.replace('c1_adduct', 'gas_monomer_references')
                    + '/nh3_v5_result.json'))
mr = json.load(open(OUT.replace('c1_adduct', 'gas_monomer_references')
                    + '/monomer_results.json'))
NH3_MONO = np.asarray(v5['v3_final']['coords_angstrom'], float)
O3_MONO = np.asarray(mr['o3']['endpoint_recheck']['coords_angstrom'], float)
mol_n = gto.M(atom="; ".join("%s %.10f %.10f %.10f" % (s, x, y, z) for s, (x, y, z)
                             in zip(['N', 'H', 'H', 'H'], NH3_MONO)),
              basis='def2-TZVP', verbose=0)
mol_o = gto.M(atom="; ".join("%s %.10f %.10f %.10f" % (s, x, y, z) for s, (x, y, z)
                             in zip(['O', 'O', 'O'], O3_MONO)),
              basis='def2-TZVP', verbose=0)
D2_NH3 = float(d2_full.d2_energy(mol_n))
D2_O3 = float(d2_full.d2_energy(mol_o))
print('monomer D2: NH3=%.6e O3=%.6e' % (D2_NH3, D2_O3), flush=True)

points = []
for d_t in TARGETS:
    xN = x_term + (d_t ** 2 - z_term ** 2) ** 0.5
    shift = xN - N0[0]
    coords = G.copy()
    coords[:4] = G[:4] + shift * u          # rigid translation of NH3 only
    # path verification: fragment internal geometry unchanged, no collision
    min_inter = min(float(np.linalg.norm(coords[i] - coords[j]))
                    for i in range(4) for j in (4, 5, 6))
    d_actual = float(np.mean([np.linalg.norm(coords[0] - O3_FIXED[i])
                              for i in (0, 1)]))
    meta = dict(target_d=d_t, actual_avg_d=d_actual,
                N_x=float(coords[0, 0]), translation=shift,
                direction=u.tolist(), min_interfragment=min_inter,
                path_check=('OK' if min_inter > 1.5 else 'COLLISION RISK'))
    print('[scan] d=%.1f  N_x=%.4f  shift=%.4f  min_inter=%.3f  %s'
          % (d_t, coords[0, 0], shift, min_inter, meta['path_check']),
          flush=True)
    if meta['path_check'] != 'OK':
        json.dump(dict(aborted=True, reason='path verification failed',
                       points=points, budget=ledger.attempts_used()),
                  open(OUT + '/scan_results.json', 'w'), indent=2)
        raise SystemExit('path verification failed -> stop')

    atom = "; ".join("%s %.10f %.10f %.10f" % (s, x, y, z)
                     for s, (x, y, z) in zip(SYMS, coords))
    mol = gto.M(atom=atom, basis='def2-TZVP', charge=0, spin=0, verbose=0,
                max_memory=4000)
    mf = make_mf_d2_gr(mol, solvent=None, grid_response=True, grid_level=8)
    # pre-eval budget: persist attempt BEFORE the evaluation (019 mechanism)
    idx = ledger.start_attempt(
        json.dumps(['c1_scan', d_t], sort_keys=True),
        dict(tag='c1_scan d=%.1f' % d_t, kind='E/G eval',
             actual_avg_d=d_actual, translation=shift))
    try:
        t0 = time.time()
        mf.kernel()
        g = np.asarray(mf.nuc_grad_method().kernel(), float).reshape(-1, 3)
        e = float(mf.e_tot)
        d2 = float(d2_full.d2_energy(mol))
        g_proj = float(np.sum(g[:4] @ u))     # path derivative (NH3 atoms move)
        ledger.update(idx, status='done', e_total=e, grad_max=float(np.abs(g).max()),
                      seconds=round(time.time() - t0, 1))
        points.append(dict(target_d=d_t, actual_avg_d=d_actual,
                           N_x=float(coords[0, 0]), translation=shift,
                           N_O6=float(np.linalg.norm(coords[0] - O3_FIXED[2])),
                           N_O5=float(np.linalg.norm(coords[0] - O3_FIXED[0])),
                           N_O7=float(np.linalg.norm(coords[0] - O3_FIXED[1])),
                           coords_angstrom=coords.tolist(),
                           e_total=e,
                           dE_nonCP=e - E_NH3 - E_O3,
                           d2_complex=d2,
                           d2_increment=d2 - D2_NH3 - D2_O3,
                           e_minus_d2=e - d2,
                           grad_max=float(np.abs(g).max()),
                           grad_proj_path=g_proj,
                           scf_converged=bool(mf.converged),
                           finite=bool(np.isfinite(e) and np.isfinite(g).all()),
                           coords_sha=hashlib.sha256(
                               np.asarray(coords, float).tobytes()).hexdigest()[:16]))
        print('   E=%.9f dE=%.6f d2_inc=%.6e g_proj=%.4f conv=%s'
              % (e, points[-1]['dE_nonCP'], points[-1]['d2_increment'],
                 g_proj, mf.converged), flush=True)
    except Exception as ex:
        ledger.mark_error(idx, repr(ex))
        print('EVAL FAILED at d=%.1f -> stop: %s' % (d_t, ex), flush=True)
        break

json.dump(dict(points=points, D2_NH3=D2_NH3, D2_O3=D2_O3,
               E_NH3_ref=E_NH3, E_O3_ref=E_O3,
               direction=u.tolist(), budget_used=ledger.attempts_used(),
               cap=CAPS['total']),
          open(OUT + '/scan_results.json', 'w'), indent=2)
print('SCAN DONE: %d/%d evals' % (ledger.attempts_used(), CAPS['total']),
      flush=True)
