#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-044: corrected frequency post-processing (final).

NO new SCF / gradient / stability / Hessian.  Matrix post-processing only.

Fixes relative to JOB-043:
  * unit: masses in amu + NU_CONV(amu)=5140.487 cm^-1 (SI-derived); the
    electron-mass route (lambda/AMU_PER_ME, NU_ME=219474.63) is verified to
    give the SAME frequencies (043 mixed the two -> 42.7x error);
  * projection: 6 mass-weighted TR directions (sqrt(m) factors, Bohr coords)
    built explicitly, rank confirmed by SVD, orthogonal complement (21x15)
    obtained by QR of [V|I]; the MW Hessian is restricted to that 15-dim
    internal subspace -> 15 internal modes.  The 6 lowest eigenvalues are
    NOT deleted by sorting.
  * PySCF cross-check uses imaginary_freq=False (True stores imaginary
    frequencies as COMPLEX numbers; casting them to float discards the
    imaginary part -> a spurious 0.0, which is what happened in 043) and
    mass=<same masses> so the comparison is apples-to-apples.
"""
import os, sys, json, hashlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import curvature_audit_lib as cal

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
BD043 = ROOT + '/run_artifacts/02_nh3o3_reference/c1_nondiag_freq'
OUT = ROOT + '/run_artifacts/02_nh3o3_reference/c1_freq_fix044'
os.makedirs(OUT, exist_ok=True)

HARTREE2J = 4.3597447222071e-18
AMU_KG = 1.66053906660e-27
ME_KG = 9.1093837015e-31
BOHR_M = 0.52917721092e-10
C_CM = 2.99792458e10
NU_AMU = np.sqrt(HARTREE2J / (AMU_KG * BOHR_M**2)) / (2 * np.pi * C_CM)
NU_ME = np.sqrt(HARTREE2J / (ME_KG * BOHR_M**2)) / (2 * np.pi * C_CM)
AMU_PER_ME = 1822.888486209
SYMS = ['N', 'H', 'H', 'H', 'O', 'O', 'O']


def nu_of(lam, conv=NU_AMU):
    return float(np.sign(lam) * np.sqrt(abs(lam)) * conv)


def mw_mat(H21, masses):
    mw = np.repeat(np.asarray(masses, float) ** -0.5, 3)
    Hm = np.asarray(H21, float) * np.outer(mw, mw)
    return 0.5 * (Hm + Hm.T)


def tr_basis_raw(masses, coords_bohr):
    masses = np.asarray(masses, float)
    coords = np.asarray(coords_bohr, float)
    n = len(masses)
    sqrtm = np.sqrt(masses)
    com = (masses[:, None] * coords).sum(0) / masses.sum()
    r = coords - com
    B = np.zeros((3 * n, 6))
    for i in range(n):
        B[3*i:3*i+3, 0:3] = np.eye(3) * sqrtm[i]
        for k in range(3):
            e = np.zeros(3); e[k] = 1.0
            B[3*i:3*i+3, 3+k] = np.cross(r[i], e) * sqrtm[i]
    return B


def internal_subspace(masses, coords_bohr, rank_tol=1e-8):
    n = len(masses)
    Braw = tr_basis_raw(masses, coords_bohr)
    sv = np.linalg.svd(Braw, compute_uv=False)
    rank = int((sv > rank_tol * max(1.0, sv[0])).sum())
    V = cal.tr_subspace(masses, coords_bohr)
    Q, _ = np.linalg.qr(np.hstack([V, np.eye(3 * n)]))
    return V, Q[:, 6:], sv, rank


def analyse(H21, masses, coords_bohr, g_cart=None, label=''):
    n = len(masses)
    d = dict(label=label, natm=n, n_dof=3 * n,
             internal_expected=3 * n - 6,
             masses=[float(x) for x in masses])
    V, U, sv, rank = internal_subspace(masses, coords_bohr)
    d['tr_singular_values'] = [float(x) for x in sv]
    d['tr_rank_svd'] = rank
    d['external_rank_confirmed_6'] = bool(rank == 6)
    d['V_orthonormal_maxdev'] = float(np.abs(V.T @ V - np.eye(V.shape[1])).max())
    d['U_orthonormal_maxdev'] = float(np.abs(U.T @ U - np.eye(U.shape[1])).max())
    d['VU_orthogonal_maxdev'] = float(np.abs(V.T @ U).max())
    d['completeness_maxdev'] = float(
        np.abs(V @ V.T + U @ U.T - np.eye(3 * n)).max())
    d['internal_basis_shape'] = list(U.shape)

    Hm = mw_mat(H21, masses)
    d['mw_symmetry_residual'] = float(np.abs(Hm - Hm.T).max())
    Hint = U.T @ Hm @ U
    Hint = 0.5 * (Hint + Hint.T)
    lam, vec = np.linalg.eigh(Hint)
    nu_amu = np.array([nu_of(x, NU_AMU) for x in lam])
    # independent unit route: electron mass
    lam_me = lam / AMU_PER_ME
    nu_me = np.array([nu_of(x, NU_ME) for x in lam_me])
    d['n_internal_modes'] = int(len(lam))
    d['eigenvalues_Eh_Bohr2_amu'] = [float(x) for x in lam]
    d['frequencies_cm1'] = [float(x) for x in nu_amu]
    d['frequencies_cm1_electron_mass_route'] = [float(x) for x in nu_me]
    d['two_unit_routes_max_abs_diff_cm1'] = float(np.abs(nu_amu - nu_me).max())
    d['NU_CONV_AMU'] = float(NU_AMU)
    d['NU_CONV_ME'] = float(NU_ME)
    d['NU_ratio'] = float(NU_ME / NU_AMU)
    neg = [i for i in range(len(lam)) if nu_amu[i] < 0]
    d['n_negative'] = len(neg)
    d['negative_modes'] = [dict(mode=i, freq_cm1=float(nu_amu[i]),
                                eigenvalue=float(lam[i])) for i in neg]
    modes = []
    for i in range(len(lam)):
        v_mw = U @ vec[:, i]
        v_cart = v_mw * np.repeat(np.asarray(masses, float) ** -0.5, 3)
        modes.append(dict(mode=i, freq_cm1=float(nu_amu[i]),
                          cart_norm_mode=[float(x) for x in v_cart],
                          external_overlap=float(
                              np.linalg.norm(V.T @ v_mw))))
    d['modes'] = modes

    if g_cart is not None:
        g = np.asarray(g_cart, float).reshape(-1)
        mw = np.repeat(np.asarray(masses, float) ** -0.5, 3)
        g_mw = g * mw
        c_int = U.T @ g_mw
        c_ext = V.T @ g_mw
        g_mw_rec = U @ c_int + V @ c_ext
        d['gradient'] = dict(
            cartesian_max_abs_Eh_Bohr=float(np.abs(g).max()),
            cartesian_norm=float(np.linalg.norm(g)),
            mw_gradient_unit='Eh/(Bohr*sqrt(amu))  (dE/dq, q=sqrt(m)*x)',
            mw_norm=float(np.linalg.norm(g_mw)),
            internal_components_Eh_Bohr_sqrtamu=[float(x) for x in c_int],
            external_components=[float(x) for x in c_ext],
            internal_abs_max=float(np.abs(c_int).max()),
            external_abs_max=float(np.abs(c_ext).max()),
            internal_norm=float(np.linalg.norm(c_int)),
            external_norm=float(np.linalg.norm(c_ext)),
            recon_residual_mw=float(np.linalg.norm(g_mw - g_mw_rec)),
            recon_residual_cart=float(
                np.linalg.norm(g - g_mw_rec / mw)),
            note='mass-weighted components are NOT directly comparable in '
                 'magnitude to the Cartesian max|g| (different units)')
    return d


def block_from_21(H21, n):
    return np.asarray(H21, float).reshape(n, 3, n, 3).transpose(0, 2, 1, 3)


def main():
    out = dict(job='JOB-2026-0906-044 corrected post-processing (final)')
    out['unit_constants'] = dict(
        NU_CONV_AMU_cm1=float(NU_AMU), NU_CONV_ME_cm1=float(NU_ME),
        ratio=float(NU_ME / NU_AMU), sqrt_amu_over_me=float(np.sqrt(AMU_PER_ME)),
        derivation='NU = sqrt(Eh_J/(mass_kg*BOHR_m^2))/(2*pi*c_cm); the two '
                   'routes must agree (verified)')

    r43 = json.load(open(BD043 + '/nondiag_results.json'))
    doc41 = json.load(open(ROOT +
                           '/run_artifacts/02_nh3o3_reference/'
                           'c1_cart_exec_v3/eval_records.json'))
    for k, v in doc41.items():
        if v.get('attempt') == 5 and v.get('status') == 'evaluated':
            C_A = np.asarray(v['coords_angstrom'], float)
            key5 = k
            break

    H_dft = np.asarray(r43['dft_hessian']['matrix'], float)
    H_d2 = np.asarray(r43['d2_hessian']['matrix'], float)
    H21 = H_dft + H_d2

    from pyscf import gto
    atom = "; ".join("%s %.10f %.10f %.10f" % (s, a, b, c)
                     for s, (a, b, c) in zip(SYMS, C_A))
    mol = gto.M(atom=atom, basis='def2-TZVP', charge=0, spin=0, verbose=0)
    masses = np.asarray(mol.atom_mass_list(), float)
    masses_avg = np.asarray(mol.atom_mass_list(isotope_avg=True), float)
    coords_bohr = np.asarray(mol.atom_coords(unit='Bohr'), float)
    g_new = np.asarray(r43['endpoint_reproduction']['grad_full'],
                       float).reshape(7, 3)

    out['matrix_source'] = dict(
        dft=dict(source='043 dft_hessian.matrix', shape=list(H_dft.shape),
                 unit='Eh/Bohr^2',
                 method=r43['dft_hessian'].get('method'),
                 raw_unsymmetrised_saved=False),
        d2=dict(source='043 d2_hessian.matrix', shape=list(H_d2.shape),
                unit='Eh/Bohr^2',
                method=r43['d2_hessian'].get('method'),
                raw_unsymmetrised_saved=False),
        combined=dict(composition='DFT analytic (wb97xd vacuum RKS, no D2) '
                                  '+ D2 central FD (step 1e-3 Bohr)',
                      no_double_counting=True,
                      antisym_residual_recorded=False,
                      note='raw unsymmetrised matrices NOT saved in 043; '
                           'the symmetrised-only residual (0) does NOT prove '
                           'raw symmetry'),
        geometry=dict(source='041 attempt=5 coords_angstrom',
                      coords_hash=key5),
        element_order=SYMS, layout='(3N x 3N) Cartesian',
        config=r43['endpoint_reproduction'].get('config'))
    out['masses'] = dict(mass_number_convention=[float(x) for x in masses],
                         isotope_avg_convention=[float(x) for x in masses_avg],
                         used='atom_mass_list() = most abundant isotope '
                              '(JOB-026 convention)')

    res = analyse(H21, masses, coords_bohr, g_cart=g_new,
                  label='C1 (JOB-041 attempt 5)')
    out['c1_analysis'] = res

    # sensitivity to the mass convention
    res_avg = analyse(H21, masses_avg, coords_bohr, label='C1 isotope_avg')
    out['c1_mass_sensitivity'] = dict(
        isotope_avg_freq=[float(x) for x in res_avg['frequencies_cm1']],
        max_abs_diff_vs_mass_number=float(np.abs(
            np.array(res_avg['frequencies_cm1'])
            - np.array(res['frequencies_cm1'])).max()))

    # ---- PySCF cross-check: same matrix, same geometry, SAME masses ----
    from pyscf.hessian import thermo as pyscf_thermo
    hblock = block_from_21(H21, 7)
    ha_t = pyscf_thermo.harmonic_analysis(mol, hblock, imaginary_freq=True,
                                          mass=masses)
    n_imag_pyscf = int(ha_t['freq_error'])
    ha = pyscf_thermo.harmonic_analysis(mol, hblock, imaginary_freq=False,
                                        mass=masses)
    pf = np.sort(np.asarray(ha['freq_wavenumber'], float))
    mine = np.sort(np.array(res['frequencies_cm1']))
    out['pyscf_cross_check'] = dict(
        pyscf_n_imaginary=n_imag_pyscf,
        pyscf_freq_cm1=[float(x) for x in pf],
        mine_freq_cm1=[float(x) for x in mine],
        max_abs_diff_cm1=float(np.abs(pf - mine).max()),
        note='imaginary_freq=False + mass=<same masses>; validates matrix '
             'handling / projection / mass convention ONLY')

    print('=== C1 internal modes (15, mass-number masses) ===')
    for i, f in enumerate(res['frequencies_cm1']):
        print('  mode %2d: %12.3f cm^-1%s'
              % (i, f, '   <-- NEGATIVE' if f < 0 else ''))
    print('  two unit routes max diff: %.3e cm^-1'
          % res['two_unit_routes_max_abs_diff_cm1'])
    print('  external rank (SVD) =', res['tr_rank_svd'],
          ' confirmed6 =', res['external_rank_confirmed_6'])
    print('  negative modes:', res['n_negative'])
    print('  PySCF n_imaginary:', n_imag_pyscf,
          ' max|diff| = %.3e cm^-1'
          % out['pyscf_cross_check']['max_abs_diff_cm1'])
    json.dump(out, open(OUT + '/freq_fix044_results.json', 'w'),
              indent=2, default=str)
    print('saved ->', OUT + '/freq_fix044_results.json')


if __name__ == '__main__':
    main()
