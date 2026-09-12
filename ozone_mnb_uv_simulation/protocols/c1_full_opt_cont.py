import os, sys, json, hashlib, time
import numpy as np
HERE = '/mnt/e/dft-service/ozone_mnb_uv_simulation/protocols'
sys.path.insert(0, HERE)
from pyscf import gto
from pyscf.geomopt import berny_solver
import d2_full
from grad_factory import make_mf_d2_gr
# JOB-025 fixed scanner adaptation + per-category budget (NO kernel patch)
from gas_monomer_reference_v2 import BudgetController, BudgetedScanner

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
BATCH_DIR = ROOT + '/run_artifacts/02_nh3o3_reference/c1_full_opt_cont'
os.makedirs(BATCH_DIR, exist_ok=True)
LEDGER = BATCH_DIR + '/budget_cont.json'
RESULTS = BATCH_DIR + '/exec_results_formal.json'
START_XYZ = ROOT + '/inputs/nh3o3_phase2/c1_cont/cont_start.xyz'
START_INFO = ROOT + '/inputs/nh3o3_phase2/c1_cont/cont_start_info.json'
CAPS = dict(total=21, species_opt={'c1c': 20}, recheck={'c1c': 1})
CONV_PARAMS = dict(gradientmax=1e-6, gradientrms=1e-6, stepmax=1.8e-3,
                   steprms=1.2e-3)   # identical to JOB-033, recorded pre-run
SYMS = ['N', 'H', 'H', 'H', 'O', 'O', 'O']
GRID_LEVEL = 8
GMAX = 1e-5


def save_results_atomic(results):
    tmp = RESULTS + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(results, fh, indent=2, default=str)
    os.replace(tmp, RESULTS)


def point_key(coords):
    return hashlib.sha256(np.asarray(coords, float).tobytes()).hexdigest()[:16]


def read_std_xyz(path):
    lines = [l for l in open(path).read().splitlines() if l.strip()]
    n = int(lines[0].split()[0])
    syms = [l.split()[0] for l in lines[2:2 + n]]
    C = np.asarray([[float(x) for x in l.split()[1:4]]
                    for l in lines[2:2 + n]], float)
    assert n == 7 and syms == SYMS and np.isfinite(C).all()
    return syms, C


def load_start():
    info = json.load(open(START_INFO))
    syms, C = read_std_xyz(START_XYZ)
    key = point_key(C)
    assert key == info['coords_hash'], 'continuation start hash mismatch'
    return C, info, key


def load_reference_energies():
    fr = json.load(open(ROOT + '/run_artifacts/02_nh3o3_reference/'
                               'freq_check/freq_check_results.json'))
    return (fr['results']['NH3']['endpoint_check']['e_total'],
            fr['results']['O3']['endpoint_check']['e_total'],
            'freq_check_results.json (JOB-026, full precision)')


def build_mol(coords):
    atom = "; ".join("%s %.10f %.10f %.10f" % (s, x, y, z)
                     for s, (x, y, z) in zip(SYMS, coords))
    return gto.M(atom=atom, basis='def2-TZVP', charge=0, spin=0, verbose=0,
                 max_memory=4000)


def structure_analysis(coords):
    C = np.asarray(coords, float)
    d = lambda i, j: float(np.linalg.norm(C[i] - C[j]))
    N = C[0]
    Hc = C[1:4].mean(axis=0)
    lp = N - Hc; lp /= np.linalg.norm(lp)
    mid = C[5:7].mean(axis=0)          # terminal rows 5,6
    to_mid = mid - N; to_mid /= np.linalg.norm(to_mid)
    import math
    ang = float(math.degrees(math.acos(np.clip(lp @ to_mid, -1, 1))))
    return dict(
        N_O=[round(d(0, 4), 4), round(d(0, 5), 4), round(d(0, 6), 4)],
        N_O_central=round(d(0, 4), 4),
        min_interfragment=round(min(d(i, j) for i in range(4)
                                    for j in (4, 5, 6)), 4),
        lonepair_vs_terminal_midpoint_deg=round(ang, 2),
        NH3_bonds=[round(d(0, j), 4) for j in (1, 2, 3)],
        O3_bonds=[round(d(4, 5), 4), round(d(4, 6), 4)])


def main():
    ctrl = BudgetController(LEDGER, CAPS, init_new=not os.path.exists(LEDGER))
    print('budget: attempts=%d (caps %s)' % (len(ctrl.data['attempts']),
                                             CAPS), flush=True)
    coords0, info, key0 = load_start()
    e_nh3, e_o3, ref_src = load_reference_energies()
    results = load_results(RESULTS) if os.path.exists(RESULTS) else {}
    results['start'] = dict(source='JOB-033 step 30 last-evaluated geometry '
                                   '(hash verified %s)' % key0,
                            continuation_mode='geometry continuation; '
                            'optimizer state RESET (no trust-radius or '
                            'Hessian-history recovery claimed)',
                            e_total_ref=info['e_total'],
                            grad_max_ref=info['grad_max'])
    print('[cont] start verified: %s' % results['start']['source'], flush=True)
    print('[cont] optimizer state: RESET (geometry continuation) - no trust '
          'radius or Hessian history recovery claimed', flush=True)
    print('[cont] conv params (recorded pre-run):', CONV_PARAMS, flush=True)

    mol = build_mol(coords0)
    mf = make_mf_d2_gr(mol, solvent=None, grid_response=True,
                       grid_level=GRID_LEVEL)
    steps = []
    results['trajectory'] = steps
    prev = {'coords': coords0}

    def callback(envs):
        m = envs['mol']
        grad = np.asarray(envs['gradients'], float).reshape(len(m.elements), 3)
        C = np.asarray(m.atom_coords(unit='Angstrom'), float)
        step_size = float(np.linalg.norm(C - prev['coords']))
        prev['coords'] = C
        rec = dict(step=int(envs['cycle']) + 1,
                   coords_angstrom=C.tolist(),
                   step_size_A=round(step_size, 6),
                   dE_vs_prev=None if len(steps) == 0 else
                   round(float(envs['energy']) - steps[-1]['e_total'], 12),
                   e_total=float(envs['energy']),
                   grad_max=float(np.abs(grad).max()),
                   grad_rms=float(np.sqrt((grad ** 2).mean())),
                   scf_converged=bool(envs['g_scanner'].converged),
                   structure=structure_analysis(C))
        steps.append(rec)
        print('[cont] step %2d E=%.9f max|g|=%.3e step=%.4f A N-Oc=%.3f'
              % (rec['step'], rec['e_total'], rec['grad_max'],
                 rec['step_size_A'], rec['structure']['N_O_central']),
              flush=True)
        save_results_atomic(results)     # per-step atomic persistence

    scanner = mf.nuc_grad_method().as_scanner()
    proxy = BudgetedScanner(scanner, ctrl, 'species:c1c:opt')
    opt_error = None
    mol_opt = None
    opt_conv = False
    try:
        opt_conv, mol_opt = berny_solver.kernel(
            proxy, assert_convergence=True, include_ghost=True,
            callback=callback, maxsteps=CAPS['species_opt']['c1c'],
            **CONV_PARAMS)
    except Exception as e:
        import traceback
        traceback.print_exc()
        opt_error = repr(e)
        print('[cont] berny exception: %s' % opt_error, flush=True)

    print('[cont] optimizer converged: %s (steps=%d)' % (opt_conv, len(steps)),
          flush=True)

    final = None
    if opt_conv and mol_opt is not None:
        # trigger 1 met: optimizer's own explicit convergence -> 1 recheck
        c_end = np.asarray(mol_opt.atom_coords(unit='Angstrom'), float)
        mol_v = build_mol(c_end)
        mf_v = make_mf_d2_gr(mol_v, solvent=None, grid_response=True,
                             grid_level=GRID_LEVEL)
        idx = ctrl.pre_eval('species:c1c:recheck',
                            dict(point_key=point_key(c_end)))
        t0 = time.time()
        mf_v.kernel()
        g_v = np.asarray(mf_v.nuc_grad_method().kernel(), float).reshape(-1, 3)
        ctrl.post_eval(idx, dict(e_total=float(mf_v.e_tot),
                                 grad_max=float(np.abs(g_v).max())))
        final = dict(coords_angstrom=c_end.tolist(),
                     e_total=float(mf_v.e_tot),
                     e_d2_analytic=float(d2_full.d2_energy(mol_v)),
                     e_dft_part=float(mf_v.e_tot) - float(d2_full.d2_energy(mol_v)),
                     grad_max=float(np.abs(g_v).max()),
                     scf_converged=bool(mf_v.converged),
                     structure=structure_analysis(c_end),
                     stationary_candidate=bool(
                         mf_v.converged and np.abs(g_v).max() <= GMAX),
                     label='stationary-point candidate pending frequency '
                           'check (NO frequency run)')
        print('[cont] RECHECK E=%.9f max|g|=%.3e candidate=%s'
              % (final['e_total'], final['grad_max'],
                 final['stationary_candidate']), flush=True)
        # trigger 2: recheck passed -> 1 stability analysis
        if final['stationary_candidate']:
            try:
                t = time.time()
                st = mf_v.stability(return_status=True)
                mo_i, mo_e, stable_i, stable_e = st
                final['stability'] = dict(
                    call='mf_v.stability(return_status=True)',
                    stable_i=(bool(stable_i) if stable_i is not None else None),
                    stable_e=(bool(stable_e) if stable_e is not None else None),
                    seconds=round(time.time() - t, 1),
                    note='internal only; no external search/following/UKS')
                print('[cont] stability: stable_i=%s'
                      % final['stability']['stable_i'], flush=True)
            except Exception as se:
                final['stability'] = dict(error=repr(se))
                print('[cont] stability analysis failed: %s' % se, flush=True)
        if final['stationary_candidate']:
            dE = final['e_total'] - e_nh3 - e_o3
            final['dE_nonCP_hartree'] = dE
            final['dE_nonCP_kcalmol'] = dE * 627.5094740631
            final['dE_label'] = ('vs full-precision monomer references; '
                                 'non-CP; includes fragment deformation; NOT '
                                 'a frozen-fragment interaction energy; no '
                                 'binding free energy claimed')
    else:
        results['incomplete'] = dict(
            reason='optimizer did not converge within 20 continuation evals; '
                   'recheck quota NOT used per trigger condition',
            opt_error=opt_error,
            end_segment=dict(
                last_steps=[dict(step=s['step'], e_total=s['e_total'],
                                 grad_max=s['grad_max'],
                                 step_size_A=s['step_size_A'])
                            for s in steps[-6:]],
                last_structure=steps[-1]['structure'] if steps else None))
        print('[cont] INCOMPLETE registered (no "how many more steps" '
              'extrapolation)', flush=True)

    results['final'] = final
    results['opt_conv'] = bool(opt_conv)
    results['opt_error'] = opt_error
    results['conv_params'] = CONV_PARAMS
    results['n_opt_steps'] = len(steps)
    save_results_atomic(results)
    print('CONT OPT DONE: opt_conv=%s steps=%d' % (opt_conv, len(steps)),
          flush=True)


if __name__ == '__main__':
    main()
