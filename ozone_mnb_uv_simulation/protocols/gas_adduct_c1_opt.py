import os, sys, json, time, hashlib
import numpy as np
HERE = '/mnt/e/dft-service/ozone_mnb_uv_simulation/protocols'
sys.path.insert(0, HERE)
from sp_ledger import SPLedger
from pyscf import gto
from pyscf.geomopt import berny_solver
from pyscf.lib import GradScanner
from pyscf.hessian import thermo as h_thermo
import d2_full
from grad_factory import make_mf_d2_gr

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
OUT = ROOT + '/run_artifacts/02_nh3o3_reference/c1_adduct'
os.makedirs(OUT, exist_ok=True)
LEDGER = OUT + '/budget.json'
CAPS = dict(total=31, species_opt={'c1': 30}, recheck={'c1': 1})
STAB_CAP = 1
CONV_PARAMS = dict(gradientmax=1e-6, gradientrms=1e-6, stepmax=1.8e-3,
                   steprms=1.2e-3)   # strict internal-coord criteria, 024-proven
SYMS = ['N', 'H', 'H', 'H', 'O', 'O', 'O']   # S13 numbering 1N,2H,3H,4H,5O,7O,6O
GRID_LEVEL = 8
E_NH3_REF = -56.564219135911    # JOB-026 NH3_gas_v5_final (full precision)
E_O3_REF = -225.433902699575    # JOB-026 O3_gas_final

ledger = SPLedger(LEDGER, cap=CAPS['total'])
ctrl = BudgetController = None
# reuse BudgetController from v2 driver module
sys.path.insert(0, HERE)
import importlib
v2 = importlib.import_module('gas_monomer_reference_v2')
ctrl = v2.BudgetController(OUT + '/budget_c1.json', CAPS, init_new=True)
print('budget: total used=%d/%d' % (ctrl.used('total'), CAPS['total']),
      flush=True)

guess = [l.split() for l in
         open(ROOT + '/inputs/nh3o3_phase2/c1_guess_S13fig1.xyz').read()
         .splitlines()[1:]]
coords0 = np.asarray([[float(a[1]), float(a[2]), float(a[3])] for a in guess],
                     float)
atom = "; ".join("%s %.10f %.10f %.10f" % (a[0], float(a[1]), float(a[2]),
                                           float(a[3])) for a in guess)
mol = gto.M(atom=atom, basis='def2-TZVP', charge=0, spin=0, verbose=0,
            max_memory=4000)
coords_saved = np.asarray(mol.atom_coords(unit='Angstrom'), float)
coords_sha = hashlib.sha256(coords_saved.tobytes()).hexdigest()[:16]
mf = make_mf_d2_gr(mol, solvent=None, grid_response=True, grid_level=8)
cfg = dict(grid_level=int(mf.grids.level),
           d2_attached=bool(getattr(mf, '_has_full_d2', False)),
           scf_tol=[float(mf.conv_tol), float(mf.conv_tol_grad)],
           conv_params=CONV_PARAMS,
           note='project method calculation, NOT an S13 reproduction; '
                'conv params written before run and read back here')
print('[c1] config readback:', cfg, flush=True)

scanner = mf.nuc_grad_method().as_scanner()
proxy = v2.BudgetedScanner(scanner, ctrl, 'species:c1:opt')
steps = []

def callback(envs):
    TOTAL = None
    m = envs['mol']
    grad = np.asarray(envs['gradients'], float).reshape(len(m.elements), 3)
    steps.append(dict(step=int(envs['cycle']) + 1,
                      coords_angstrom=np.asarray(
                          m.atom_coords(unit='Angstrom'), float).tolist(),
                      e_total=float(envs['energy']),
                      grad_max=float(np.abs(grad).max()),
                      grad_rms=float(np.sqrt((grad ** 2).mean())),
                      scf_converged=bool(envs['g_scanner'].converged)))
    print('[c1] step %2d E=%.9f max|g|=%.3e conv=%s'
          % (steps[-1]['step'], steps[-1]['e_total'], steps[-1]['grad_max'],
             steps[-1]['scf_converged']), flush=True)
    json.dump(steps, open(OUT + '/c1_steps.json', 'w'), indent=2)

opt_error = None
mol_opt = None
try:
    opt_conv, mol_opt = berny_solver.kernel(
        proxy, assert_convergence=True, include_ghost=True,
        callback=callback, maxsteps=CAPS['species_opt']['c1'],
        **CONV_PARAMS)
except Exception as e:
    import traceback
    traceback.print_exc()
    opt_error = repr(e)
    print('[c1] berny exception: %s' % opt_error, flush=True)

# endpoint recheck with fresh objects (category recheck)
final = None
if mol_opt is not None:
    c_end = np.asarray(mol_opt.atom_coords(unit='Angstrom'), float)
    atom_v = "; ".join("%s %.10f %.10f %.10f"
                       % (s, x, y, z) for s, (x, y, z) in zip(SYMS, c_end))
    mol_v = gto.M(atom=atom_v, basis='def2-TZVP', charge=0, spin=0, verbose=0,
                  max_memory=4000)
    mf_v = make_mf_d2_gr(mol_v, solvent=None, grid_response=True,
                         grid_level=8)
    if ctrl.used('total') < CAPS['total']:
        idx = ctrl.pre_eval('species:c1:recheck',
                            dict(coords_sha=hashlib.sha256(
                                c_end.tobytes()).hexdigest()[:16]))
        try:
            mf_v.kernel()
            g_v = np.asarray(mf_v.nuc_grad_method().kernel(), float)
            ctrl.post_eval(idx, dict(e_total=float(mf_v.e_tot),
                                     grad_max=float(np.abs(g_v).max())))
            final = dict(coords_angstrom=c_end.tolist(),
                         e_total=float(mf_v.e_tot),
                         e_d2_analytic=float(d2_full.d2_energy(mol_v)),
                         grad_max=float(np.abs(g_v).max()),
                         grad_rms=float(np.sqrt((g_v ** 2).mean())),
                         scf_converged=bool(mf_v.converged),
                         opt_conv=opt_conv, opt_error=opt_error,
                         stationary_candidate=bool(
                             mf_v.converged and np.abs(g_v).max() <= 1e-5))
            print('[c1] RECHECK E=%.9f max|g|=%.3e candidate=%s'
                  % (final['e_total'], final['grad_max'],
                     final['stationary_candidate']), flush=True)
        except Exception as e:
            ctrl.mark_error(idx, repr(e))
            raise

    # fragment / orientation analysis
    C = c_end
    def dd(i, j):
        return float(np.linalg.norm(C[i] - C[j]))
    frag = dict(
        contacts=dict(N_O5=dd(0, 4), N_O7=dd(1 + 2, 4) if False else dd(0, 5),
                      N_O6=dd(0, 6)),
        O3_bonds=dict(O5_O6=dd(4, 6), O7_O6=dd(5, 6), O5_O7=dd(4, 5)),
        NH3_bonds=[dd(0, j) for j in (1, 2, 3)],
        min_interfragment=min(min(dd(i, j) for i in (0, 1, 2, 3))
                              for j in (4, 5, 6)))
    # compare with monomers
    nh3_mono = np.asarray(v5['v3_final']['coords_angstrom'], float)
    o3_mono = np.asarray(mr_o3())
    def mr_o3():
        return json.load(open(OUT.replace('c1_adduct',
                              'gas_monomer_references')
                              + '/monomer_results.json'))['o3'][
            'endpoint_recheck']['coords_angstrom']
    frag['NH3_bond_change_vs_monomer'] = [
        round(frag['NH3_bonds'][i] - float(np.linalg.norm(
            nh3_mono[0] - nh3_mono[i + 1])), 6) for i in range(3)]
    om = np.asarray(mr_o3(), float)
    do = [float(np.linalg.norm(om[0] - om[2])), float(np.linalg.norm(om[1] - om[2]))]
    frag['O3_bond_change_vs_monomer'] = [
        round(dd(4, 6) - do[0], 6), round(dd(5, 6) - do[1], 6)]
    # orientation correspondence with literature C1: N between/beside 5O,7O?
    frag['orientation_note'] = (
        'N-O5=%.4f N-O7=%.4f (S13 fig: 2.967/2.963); N-O6=%.4f; '
        'fragment preservation judged from bond changes' %
        (frag['contacts']['N_O5'], frag['contacts']['N_O7'],
         frag['contacts']['N_O6']))
    final['fragment_analysis'] = frag

    # stability (only if recheck passed), explicit booleans
    if final['stationary_candidate']:
        if 'stability' not in [k for k in ()]:
            pass
        cat = 'species:c1:stability'
        # stability is a separate cap (1), accounted in a dedicated counter
        idx = ctrl.pre_eval(cat, dict(note='stability analysis marker'))
        try:
            t0 = time.time()
            st = mf_v.stability(return_status=True)
            mo_i, mo_e, stable_i, stable_e = st
            final['stability'] = dict(
                call='mf_v.stability(return_status=True)',
                stable_i=(bool(stable_i) if stable_i is not None else None),
                stable_e=(bool(stable_e) if stable_e is not None else None),
                seconds=round(time.time() - t0, 1),
                note='internal only; no following/UKS/external')
            ctrl.post_eval(idx, dict(stable_i=final['stability']['stable_i']))
            print('[c1] stability: stable_i=%s stable_e=%s'
                  % (final['stability']['stable_i'],
                     final['stability']['stable_e']), flush=True)
        except Exception as se:
            ctrl.mark_error(idx, repr(se))
            final['stability'] = dict(error=repr(se))
            print('[c1] stability analysis failed: %s' % se, flush=True)

# energy record
if final and final.get('stationary_candidate'):
    dE = final['e_total'] - E_NH3_REF - E_O3_REF
    final['dE_candidate_nonCP_hartree'] = dE
    final['dE_candidate_nonCP_kcalmol'] = dE * 627.5094740631
    final['dE_label'] = ('stationary-point candidate electronic energy change '
                         'vs optimized monomers, NO CP correction; not a '
                         'frozen-fragment interaction energy, not enthalpy/'
                         'free energy; BSSE NOT evaluated; not directly '
                         'comparable with S13 CP-corrected values')
    print('[c1] dE_nonCP = %.6f Eh = %.3f kcal/mol'
          % (dE, final['dE_candidate_nonCP_kcalmol']), flush=True)

json.dump(dict(final=final, steps=steps, opt_error=opt_error,
               total_evals=ctrl.used('total'), caps=CAPS,
               conv_params=CONV_PARAMS),
          open(OUT + '/c1_opt_result.json', 'w'), indent=2)
print('C1 BATCH DONE: total_evals=%d/%d' % (ctrl.used('total'),
                                            CAPS['total']), flush=True)
