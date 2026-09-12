import os, sys, json, time, hashlib
import numpy as np
HERE = '/mnt/e/dft-service/ozone_mnb_uv_simulation/protocols'
sys.path.insert(0, HERE)
from pyscf import gto
from pyscf.hessian import thermo as h_thermo
import d2_full
from grad_factory import make_mf_d2_gr

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
OUT = ROOT + '/run_artifacts/02_nh3o3_reference/freq_check'
os.makedirs(OUT, exist_ok=True)
IDX = json.load(open(ROOT + '/run_artifacts/02_nh3o3_reference/final_endpoint_index.json'))

GRID_LEVEL = 8
GMAX = 1e-5
SYMS = {'NH3': ['N', 'H', 'H', 'H'], 'O3': ['O', 'O', 'O']}

# budget accounting (persisted before each operation)
budget = dict(scf_grad=0, stability=0, hessian=0, d2_fd_checks=0, cap=dict(
    scf_grad=2, stability=2, hessian=2))
BUDGET_FILE = OUT + '/budget.json'


def save_budget():
    json.dump(budget, open(BUDGET_FILE, 'w'), indent=2)


def pre(op):
    if budget[op] >= budget['cap'][op]:
        raise RuntimeError('BUDGET EXHAUSTED for %s (cap %d)'
                           % (op, budget['cap'][op]))
    budget[op] += 1
    save_budget()


# unit conversion for mass-weighted eigenvalues (Eh/Bohr^2/amu) -> cm^-1
from scipy.constants import h, c, electron_volt
AMU = 1.66053906892e-27
EH = 4.3597447222071e-18
BOHR = 5.29177210544e-11
CONV = np.sqrt(EH / (BOHR ** 2 * AMU)) / (2 * np.pi * c * 100.0)


def tr_basis(coords_bohr, masses):
    n = len(masses)
    cols = []
    r_mw = coords_bohr * np.sqrt(masses)[:, None]
    for a in range(3):
        v = np.zeros((n, 3))
        v[:, a] = np.sqrt(masses)
        cols.append(v.reshape(-1))
    for ax in range(3):
        e = np.zeros(3); e[ax] = 1.0
        v = np.cross(e, r_mw)
        cols.append(v.reshape(-1))
    T = np.column_stack(cols)
    U, S, Vt = np.linalg.svd(T)
    rank = int((S > 1e-8).sum())
    return U[:, :rank], rank, S


def my_freqs(h2d, coords_bohr, masses):
    n = len(masses)
    s = np.repeat(np.sqrt(masses), 3)
    mw = h2d / np.outer(s, s)
    mw = 0.5 * (mw + mw.T)
    B, rank, S = tr_basis(coords_bohr, masses)
    P = np.eye(3 * n) - B @ B.T
    Hp = P @ mw @ P
    Hp = 0.5 * (Hp + Hp.T)
    ortho = float(np.abs(B.T @ B - np.eye(rank)).max())
    w, v = np.linalg.eigh(Hp)
    lam_int = w[rank:]
    freq = np.sign(lam_int) * np.sqrt(np.abs(lam_int)) * CONV
    modes = v[:, rank:]
    return dict(eigenvalues=lam_int.tolist(), freq_cm=freq.tolist(),
                tr_rank=rank, tr_singular_values=S.tolist(),
                tr_basis_ortho_maxdev=ortho,
                projection_dims=(3 * n - rank), modes=modes.tolist())


results = {}
for sid in ('NH3', 'O3'):
    key = {'NH3': 'NH3_gas_v5_final', 'O3': 'O3_gas_final'}[sid]
    ep = IDX['default_endpoints'][key]
    coords_in = np.asarray(ep['coords_angstrom'], float)
    syms = SYMS[sid]
    atom = "; ".join("%s %.10f %.10f %.10f" % (s, x, y, z)
                     for s, (x, y, z) in zip(syms, coords_in))
    logf = OUT + '/%s_stdout.log' % sid
    mol = gto.M(atom=atom, basis='def2-TZVP', charge=0, spin=0, verbose=4,
                max_memory=4000, output=logf)
    coords_actual = np.asarray(mol.atom_coords(unit='Angstrom'), float)
    coord_hash = hashlib.sha256(coords_actual.tobytes()).hexdigest()[:16]
    mf = make_mf_d2_gr(mol, solvent=None, grid_response=True,
                       grid_level=GRID_LEVEL)
    cfg = dict(grid_level=int(mf.grids.level),
               d2_attached=bool(getattr(mf, '_has_full_d2', False)),
               scf_tol=[float(mf.conv_tol), float(mf.conv_tol_grad)])
    assert cfg['grid_level'] == GRID_LEVEL and cfg['d2_attached']

    # ---- 1) endpoint SCF + full gradient (budget: scf_grad) ----
    pre('scf_grad')
    t0 = time.time()
    mf.kernel()
    g = np.asarray(mf.nuc_grad_method().kernel(), float).reshape(-1, 3)
    grad_max = float(np.abs(g).max())
    print('[%s] endpoint SCF+grad E=%.9f max|g|=%.3e conv=%s (%.0fs)'
          % (sid, mf.e_tot, grad_max, mf.converged, time.time() - t0),
          flush=True)
    ep_check = dict(
        coords_passed_to_object=coords_actual.tolist(),
        coords_hash=coord_hash,
        coords_match_index=bool(np.abs(coords_actual - coords_in).max()
                                < 1e-10),
        atom_order=list(mol.elements), charge=mol.charge, spin=mol.spin,
        e_total=float(mf.e_tot), grad_max=grad_max,
        scf_converged=bool(mf.converged),
        grad_criterion_pass=bool(grad_max <= GMAX),
        config_readback=cfg)
    if not ep_check['grad_criterion_pass'] or not mf.converged:
        results[sid] = dict(endpoint_check=ep_check,
                            status='STOPPED (this monomer): gradient/SCF '
                                   'criterion failed; other monomer '
                                   'continues unless shared-implementation '
                                   'issue is identified')
        continue

    # ---- 2) internal stability with explicit status ----
    pre('stability')
    t0 = time.time()
    st = mf.stability(return_status=True)
    mo_i, mo_e, stable_i, stable_e = st
    stability = dict(
        call='mf.stability(return_status=True)',
        stable_i=(bool(stable_i) if stable_i is not None else None),
        stable_e=(bool(stable_e) if stable_e is not None else None),
        seconds=round(time.time() - t0, 1),
        raw_note='mo_i/mo_e arrays not saved (large); booleans and raw log '
                 'are the record; internal only, no external search, no '
                 'instability following, no UKS switch')
    print('[%s] stability: stable_i=%s stable_e=%s'
          % (sid, stability['stable_i'], stability['stable_e']), flush=True)
    if stable_i is not True:
        results[sid] = dict(endpoint_check=ep_check, stability=stability,
                            status='STOPPED (this monomer): internal '
                                   'stability not True / unclear; no '
                                   'Hessian for this monomer')
        continue

    # ---- 3) analytic DFT Hessian on the SAME converged object ----
    pre('hessian')
    t0 = time.time()
    hobj = mf.Hessian()
    hess4 = np.asarray(hobj.kernel(), float)          # (n,n,3,3), incl. D2-FD
    hess_seconds = round(time.time() - t0, 1)
    grid_after = int(mf.grids.level)
    n = mol.natm
    # antisymmetric residual BEFORE symmetrisation
    anti = float(np.abs(hess4 - hess4.transpose(1, 0, 3, 2)).max())
    hess4_sym = 0.5 * (hess4 + hess4.transpose(1, 0, 3, 2))
    # layout conversion (022 lesson): (n,3,n,3) -> (3n,3n)
    h2d = hess4_sym.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)
    # D2 part: central FD of analytic D2 gradient (default and half step)
    d2_default = d2_full.d2_hess(mol)
    d2_half = d2_full.d2_hess(mol, step=5e-4)
    d2_diff = float(np.abs(
        d2_default.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)
        - d2_half.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)).max())
    budget['d2_fd_checks'] += 2
    save_budget()
    # DFT part = combined - D2 (no double counting)
    d2_2d = d2_default.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)
    dft_2d = h2d - d2_2d
    # masses and projection
    masses = np.asarray(mol.atom_mass_list(), float)
    mine = my_freqs(h2d, mol.atom_coords(unit='Bohr'), masses)
    # PySCF post-processing cross-check (same matrix, same masses)
    pys = h_thermo.harmonic_analysis(mol, hess4_sym,
                                    imaginary_freq=False,
                                    mass=masses)
    pys_freq = [float(x) for x in pys['freq_wavenumber']]
    diff = float(np.abs(np.asarray(mine['freq_cm'])
                        - np.asarray(pys_freq)).max()) \
        if len(pys_freq) == len(mine['freq_cm']) else None
    internal_expected = 3 * n - 6
    neg = [f for f in mine['freq_cm'] if f < 0]
    results[sid] = dict(
        endpoint_check=ep_check, stability=stability,
        hessian=dict(
            seconds=hess_seconds, grid_level_before=GRID_LEVEL,
            grid_level_after=grid_after,
            layout='pyscf (natm,natm,3,3) -> transpose(0,2,1,3).reshape(3n,3n)',
            antisym_residual_max=anti,
            composition='combined(analytic DFT + central-FD D2); D2 = central '
                        'FD of analytic D2 gradient (d2_full.d2_hess, step in '
                        'Bohr); DFT_part = combined - D2 (no double counting); '
                        'NOT called fully analytic',
            d2_step_check=dict(default_1e_3_vs_half_5e_4_maxdiff=d2_diff),
            matrices_saved=dict(
                combined_sym='freq_check/%s_hess_combined.npy' % sid,
                d2_fd='freq_check/%s_hess_d2.npy' % sid,
                dft_part='freq_check/%s_hess_dft.npy' % sid)),
        response_solver=dict(
            note='RKS Hessian response solved inside hobj.kernel(); raw log '
                 'in %s_stdout.log (search "converged"); explicit response '
                 'convergence flags not exposed by the PySCF API' % sid),
        masses=dict(values=masses.tolist(),
                    convention='mol.atom_mass_list() = most abundant isotope'),
        frequencies=dict(
            internal_expected=internal_expected,
            n_internal_modes=len(mine['freq_cm']),
            my_freq_cm=mine['freq_cm'],
            my_eigenvalues=mine['eigenvalues'],
            tr_basis_rank=mine['tr_rank'],
            tr_basis_ortho_maxdev=mine['tr_basis_ortho_maxdev'],
            projection_dims=mine['projection_dims'],
            negative_modes=[f for f in mine['freq_cm'] if f < 0],
            pyscf_cross_check_freq_cm=pys_freq,
            pyscf_vs_my_max_diff_cm=diff,
            cross_check_note='PySCF harmonic_analysis with the SAME matrix '
                             'and SAME masses validates matrix handling and '
                             'mass conventions ONLY - not an independent '
                             'electronic-structure verification'),
        modes=mine['modes'],
        negative_modes_present=bool(neg),
        all_curvatures_positive=bool(len(neg) == 0))
    np.save(OUT + '/%s_hess_combined.npy' % sid, h2d)
    np.save(OUT + '/%s_hess_d2.npy' % sid, d2_2d)
    np.save(OUT + '/%s_hess_dft.npy' % sid, dft_2d)
    print('[%s] hessian done (%.0fs) anti=%.2e internal modes=%d freqs=%s'
          % (sid, hess_seconds, anti, len(mine['freq_cm']),
             ['%.1f' % f for f in mine['freq_cm']]), flush=True)

json.dump(dict(budget=budget, results=results),
          open(OUT + '/freq_check_results.json', 'w'), indent=2)
print('FREQ CHECK DONE:', json.dumps(budget), flush=True)
