import os, sys, json, hashlib, time
import numpy as np
HERE = '/mnt/e/dft-service/ozone_mnb_uv_simulation/protocols'
sys.path.insert(0, HERE)
from pyscf import gto
from pyscf.geomopt import berny_solver
import d2_full
from grad_factory import make_mf_d2_gr
# JOB-025 fixed scanner adaptation + per-category budget (NO kernel patch)
from gas_monomer_reference_v2 import BudgetController, BudgetedScanner, \
    LedgerMissingError

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
BATCH_DIR = ROOT + '/run_artifacts/02_nh3o3_reference/c1_full_opt'
os.makedirs(BATCH_DIR, exist_ok=True)
LEDGER = BATCH_DIR + '/budget_full_opt.json'
RESULTS = BATCH_DIR + '/exec_results_formal.json'
CAPS = dict(total=31, species_opt={'c1f': 30}, recheck={'c1f': 1})
CONV_PARAMS = dict(gradientmax=1e-6, gradientrms=1e-6, stepmax=1.8e-3,
                   steprms=1.2e-3)   # strict internal criteria, fixed pre-run
SYMS = ['N', 'H', 'H', 'H', 'O', 'O', 'O']   # 1N,2H,3H,4H,5O,7O,6O (S13 numbering)
GRID_LEVEL = 8
GMAX = 1e-5


def save_results_atomic(results):
    tmp = RESULTS + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(results, fh, indent=2, default=str)
    os.replace(tmp, RESULTS)


def load_start_geometry():
    """JOB-031 d=3.0 ACTUALLY COMPUTED coords (no rebuild/rotation/bond change)."""
    r31 = json.load(open(ROOT + '/run_artifacts/02_nh3o3_reference/'
                                'c1_scan_formal/exec_results_formal.json'))
    pt = next(v for v in r31.values() if isinstance(v, dict)
              and v.get('target_d') == 3.0)
    coords = np.asarray(pt['coords_computed'], float)
    key = point_key(coords)
    # the record's point_key is the XYZ-source hash; the computed-coords hash
    # is stored separately (float round-trip differs at ~1e-12)
    assert key == pt['coords_computed_sha'], 'start geometry hash mismatch'
    assert list(coords.shape) == [7, 3] and np.isfinite(coords).all()
    # element order & fragment mapping from the record
    assert pt['atom_order'] == SYMS
    fm = pt['fragments']
    # assembled order = monomer row order: O3 central first (row 4)
    assert fm['NH3'] == [0, 1, 2, 3] and fm['O3_rows']['central'] == 4 \
        and fm['O3_rows']['terminals'] == [5, 6]
    # actual contact distance recheck
    d_avg = float(np.mean([np.linalg.norm(coords[0] - coords[j])
                           for j in (5, 6)]))   # terminal rows (central=row4)
    assert abs(d_avg - pt['target_d']) < 1e-6, 'contact distance mismatch'
    return coords, dict(source='JOB-031 exec_results_formal.json d=3.0 '
                              'computed coords (verified)',
                        point_key=key, actual_avg_d=d_avg,
                        label='文献启发、经几何核验的项目初猜（非作者三维坐标）')


def point_key(coords):
    return hashlib.sha256(np.asarray(coords, float).tobytes()).hexdigest()[:16]


def load_reference_energies():
    fr = json.load(open(ROOT + '/run_artifacts/02_nh3o3_reference/'
                               'freq_check/freq_check_results.json'))
    e_nh3 = fr['results']['NH3']['endpoint_check']['e_total']
    e_o3 = fr['results']['O3']['endpoint_check']['e_total']
    return e_nh3, e_o3, dict(source='freq_check_results.json (JOB-026)')


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
    mid = C[4:6].mean(axis=0)
    to_mid = mid - N; to_mid /= np.linalg.norm(to_mid)
    import math
    ang = float(math.degrees(math.acos(np.clip(lp @ to_mid, -1, 1))))
    return dict(
        N_O=[round(d(0, j), 4) for j in (4, 5, 6)],
        min_interfragment=round(min(d(i, j) for i in range(4)
                                    for j in (4, 5, 6)), 4),
        lonepair_vs_terminal_midpoint_deg=round(ang, 2),
        NH3_bonds=[round(d(0, j), 4) for j in (1, 2, 3)],
        O3_bonds=[round(d(4, 6), 4), round(d(5, 6), 4)])


def main():
    # budget: total 31, species opt 30, recheck 1 (stability separate, 1 op)
    from gas_monomer_reference_v2 import BudgetController as Budget
    ctrl = Budget(LEDGER, CAPS, init_new=not os.path.exists(LEDGER))
    print('budget: total used=%d/%d' % (len(ctrl.data['attempts']),
                                        CAPS['total']), flush=True)
    coords0, src = load_start_geometry()
    print('start geometry verified: %s' % src['source'], flush=True)
    results = load_results(RESULTS) if os.path.exists(RESULTS) else {}
    e_nh3, e_o3, ref_src = load_reference_energies()

    mol = build_mol(coords0)
    mf = make_mf_d2_gr(mol, solvent=None, grid_response=True,
                       grid_level=GRID_LEVEL)
    cfg = dict(grid_level=int(mf.grids.level),
               d2_attached=bool(getattr(mf, '_has_full_d2', False)),
               scf_tol=[float(mf.conv_tol), float(mf.conv_tol_grad)],
               conv_params=CONV_PARAMS,
               note='project method; thresholds fixed before run; all '
                    'degrees of freedom free')
    assert cfg['grid_level'] == GRID_LEVEL and cfg['d2_attached']
    print('[c1f] config readback:', cfg, flush=True)

    steps = []
    results['trajectory'] = steps
    results['start'] = dict(source_label=src['label'], **src)

    def callback(envs):
        m = envs['mol']
        grad = np.asarray(envs['gradients'], float).reshape(len(m.elements), 3)
        C = np.asarray(m.atom_coords(unit='Angstrom'), float)
        rec = dict(step=int(envs['cycle']) + 1,
                   coords_angstrom=C.tolist(),
                   e_total=float(envs['energy']),
                   grad_max=float(np.abs(grad).max()),
                   grad_rms=float(np.sqrt((grad ** 2).mean())),
                   scf_converged=bool(envs['g_scanner'].converged),
                   structure=structure_analysis(C))
        steps.append(rec)
        print('[c1f] step %2d E=%.9f max|g|=%.3e N-O=%s'
              % (rec['step'], rec['e_total'], rec['grad_max'],
                 rec['structure']['N_O']), flush=True)
        save_results_atomic(results)

    scanner = mf.nuc_grad_method().as_scanner()
    proxy = BudgetedScanner(scanner, ctrl, 'species:c1f:opt')
    opt_error = None
    mol_opt = None
    opt_conv = False
    try:
        opt_conv, mol_opt = berny_solver.kernel(
            proxy, assert_convergence=True, include_ghost=True,
            callback=callback, maxsteps=CAPS['species_opt']['c1f'],
            **CONV_PARAMS)
    except Exception as e:
        import traceback
        traceback.print_exc()
        opt_error = repr(e)
        print('[c1f] berny exception: %s' % opt_error, flush=True)

    print('[c1f] optimizer converged: %s (steps=%d, error=%s)'
          % (opt_conv, len(steps), opt_error), flush=True)

    # recheck ONLY if the optimizer itself explicitly converged
    final = None
    if opt_conv and mol_opt is not None:
        c_end = np.asarray(mol_opt.atom_coords(unit='Angstrom'), float)
        mol_v = build_mol(c_end)
        mf_v = make_mf_d2_gr(mol_v, solvent=None, grid_response=True,
                             grid_level=GRID_LEVEL)
        idx = ctrl.pre_eval('species:c1f:recheck',
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
                     label='stationary-point candidate (NO frequency check)')
        print('[c1f] RECHECK E=%.9f max|g|=%.3e candidate=%s'
              % (final['e_total'], final['grad_max'],
                 final['stationary_candidate']), flush=True)
        # stability ONLY if recheck passed
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
                print('[c1f] stability: stable_i=%s'
                      % final['stability']['stable_i'], flush=True)
            except Exception as se:
                final['stability'] = dict(error=repr(se))
                print('[c1f] stability analysis failed: %s' % se, flush=True)
        # energy difference for passing candidates
        if final['stationary_candidate']:
            dE = final['e_total'] - e_nh3 - e_o3
            final['dE_nonCP_hartree'] = dE
            final['dE_nonCP_kcalmol'] = dE * 627.5094740631
            final['dE_label'] = ('vs optimized monomers; includes fragment '
                                 'deformation; NOT a frozen-fragment '
                                 'interaction energy; BSSE not evaluated')
    else:
        # register incomplete: keep last step, no recheck quota used
        results['incomplete'] = dict(reason='optimizer did not converge '
                                     'within 30 evals; recheck quota NOT '
                                     'used per trigger condition',
                                     last_step=steps[-1] if steps else None,
                                     opt_error=opt_error)

    results['final'] = final
    results['opt_conv'] = bool(opt_conv)
    results['opt_error'] = opt_error
    results['conv_params'] = CONV_PARAMS
    results['n_opt_steps'] = len(steps)
    results['start_source'] = src
    save_results_atomic(results)
    print('FULL OPT DONE: opt_conv=%s steps=%d' % (opt_conv, len(steps)),
          flush=True)


if __name__ == '__main__':
    main()
