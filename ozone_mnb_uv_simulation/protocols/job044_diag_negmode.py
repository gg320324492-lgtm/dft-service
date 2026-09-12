#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-044 diagnostic: why does PySCF report 0.0 where the internal-subspace
analysis reports -72.19 cm^-1?  Fix the electron-mass route (divide, not
multiply) and identify PySCF's mass convention."""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import curvature_audit_lib as cal

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
BD043 = ROOT + '/run_artifacts/02_nh3o3_reference/c1_nondiag_freq'
SYMS = ['N', 'H', 'H', 'H', 'O', 'O', 'O']

HARTREE2J = 4.3597447222071e-18
AMU_KG = 1.66053906660e-27
ME_KG = 9.1093837015e-31
BOHR_M = 0.52917721092e-10
C_CM = 2.99792458e10
NU_AMU = np.sqrt(HARTREE2J / (AMU_KG * BOHR_M**2)) / (2 * np.pi * C_CM)
NU_ME = np.sqrt(HARTREE2J / (ME_KG * BOHR_M**2)) / (2 * np.pi * C_CM)
AMU_PER_ME = 1822.888486209


def nu_of(lam, conv=NU_AMU):
    return float(np.sign(lam) * np.sqrt(abs(lam)) * conv)


def mw_mat(H21, masses):
    mw = np.repeat(np.asarray(masses, float) ** -0.5, 3)
    Hm = np.asarray(H21, float) * np.outer(mw, mw)
    return 0.5 * (Hm + Hm.T)


def internal_basis(masses, coords_bohr):
    V = cal.tr_subspace(masses, coords_bohr)
    n = len(masses)
    Q, _ = np.linalg.qr(np.hstack([V, np.eye(3 * n)]))
    return V, Q[:, 6:]


def analyse(H21, masses, coords_bohr):
    V, U = internal_basis(masses, coords_bohr)
    Hm = mw_mat(H21, masses)
    Hint = U.T @ Hm @ U
    Hint = 0.5 * (Hint + Hint.T)
    lam, vec = np.linalg.eigh(Hint)
    nu_amu = [nu_of(x, NU_AMU) for x in lam]
    # ELECTRON-MASS ROUTE: lambda_me = lambda_amu / AMU_PER_ME
    lam_me = lam / AMU_PER_ME
    nu_me = [nu_of(x, NU_ME) for x in lam_me]
    return dict(lam=lam, nu_amu=np.array(nu_amu), nu_me=np.array(nu_me),
                V=V, U=U, vec=vec, Hm=Hm)


def block_from_21(H21, n):
    return np.asarray(H21, float).reshape(n, 3, n, 3).transpose(0, 2, 1, 3)


def main():
    r43 = json.load(open(BD043 + '/nondiag_results.json'))
    doc41 = json.load(open(ROOT +
                           '/run_artifacts/02_nh3o3_reference/'
                           'c1_cart_exec_v3/eval_records.json'))
    for k, v in doc41.items():
        if v.get('attempt') == 5 and v.get('status') == 'evaluated':
            C_A = np.asarray(v['coords_angstrom'], float)
            break
    H21 = (np.asarray(r43['dft_hessian']['matrix'], float)
           + np.asarray(r43['d2_hessian']['matrix'], float))

    from pyscf import gto
    atom = "; ".join("%s %.10f %.10f %.10f" % (s, a, b, c)
                     for s, (a, b, c) in zip(SYMS, C_A))
    mol = gto.M(atom=atom, basis='def2-TZVP', charge=0, spin=0, verbose=0)
    coords_bohr = np.asarray(mol.atom_coords(unit='Bohr'), float)

    m_num = np.asarray(mol.atom_mass_list(), float)
    m_avg = np.asarray(mol.atom_mass_list(isotope_avg=True), float)
    print('masses (atom_mass_list default)   :', m_num)
    print('masses (isotope_avg=True)         :', m_avg)
    print('difference                        :', np.round(m_avg - m_num, 6))

    print()
    print('=== unit-route consistency (fixed: lambda_me = lambda_amu/AMU_PER_ME) ===')
    a_num = analyse(H21, m_num, coords_bohr)
    print('  amu route  :', np.round(a_num['nu_amu'], 3))
    print('  me  route  :', np.round(a_num['nu_me'], 3))
    print('  max |diff| : %.3e cm^-1 (must be ~0)'
          % np.abs(a_num['nu_amu'] - a_num['nu_me']).max())

    # PySCF with both mass conventions
    from pyscf.hessian import thermo as pyscf_thermo
    hblock = block_from_21(H21, 7)
    print()
    print('=== PySCF harmonic_analysis (same matrix) ===')
    ha = pyscf_thermo.harmonic_analysis(mol, hblock, imaginary_freq=True)
    pf = np.sort(np.asarray(ha['freq_wavenumber'], float))
    print('  pyscf (its own masses)   :', np.round(pf, 3))

    # which mass convention matches pyscf?
    for tag, mm in [('mass numbers', m_num), ('isotope_avg', m_avg)]:
        aa = analyse(H21, mm, coords_bohr)
        mine = np.sort(aa['nu_amu'])
        print('  mine (%s):' % tag, np.round(mine, 3))
        print('     -> max|diff vs pyscf| = %.3e cm^-1'
              % np.abs(mine - pf).max())

    # --- the negative mode: is it internal? ---
    print()
    print('=== negative-mode diagnosis (mass-number convention) ===')
    lam = a_num['lam']
    nu = a_num['nu_amu']
    V, U = a_num['V'], a_num['U']
    vec = a_num['vec']
    i0 = int(np.argmin(nu))
    print('  lowest internal mode index: %d, nu = %.3f cm^-1, lambda = %.6e'
          % (i0, nu[i0], lam[i0]))
    v_int = vec[:, i0]
    v_mw = U @ v_int
    # overlap with the external (TR) subspace: should be ~0 by construction
    ext_overlap = float(np.linalg.norm(V.T @ v_mw))
    print('  |V^T v_mw| (external overlap) = %.3e  (internal => ~0)'
          % ext_overlap)
    print('  |v_mw| = %.6f (unit norm in MW space)'
          % float(np.linalg.norm(v_mw)))
    # Cartesian displacement pattern of this mode
    v_cart = v_mw * np.repeat(m_num ** -0.5, 3)
    disp = v_cart.reshape(7, 3)
    print('  Cartesian displacement pattern (Bohr, unit-norm MW mode):')
    for a in range(7):
        print('    %s  %+ .4f %+ .4f %+ .4f   |d|=%.4f'
              % (SYMS[a], disp[a, 0], disp[a, 1], disp[a, 2],
                 np.linalg.norm(disp[a])))

    # PySCF WITHOUT excluding trans/rot -> all 21 modes
    print()
    print('=== PySCF with exclude_trans_rota=False (all 21 modes) ===')
    ha_all = pyscf_thermo.harmonic_analysis(mol, hblock, imaginary_freq=True,
                                            exclude_trans_rota=False)
    pf_all = np.sort(np.asarray(ha_all['freq_wavenumber'], float))
    print('  all 21:', np.round(pf_all, 3))
    # my full-21 MW spectrum
    Hm = a_num['Hm']
    lam21 = np.linalg.eigvalsh(Hm)
    nu21 = np.sort(np.array([nu_of(x) for x in lam21]))
    print('  my 21 (MW, no projection):', np.round(nu21, 3))

    json.dump(dict(
        masses_mass_number=[float(x) for x in m_num],
        masses_isotope_avg=[float(x) for x in m_avg],
        nu_amu=[float(x) for x in a_num['nu_amu']],
        nu_me=[float(x) for x in a_num['nu_me']],
        two_route_max_diff=float(np.abs(a_num['nu_amu']
                                        - a_num['nu_me']).max()),
        pyscf_own_masses=[float(x) for x in pf],
        pyscf_all21=[float(x) for x in pf_all],
        my21_no_projection=[float(x) for x in nu21],
        negative_mode=dict(index=i0, nu_cm1=float(nu[i0]),
                           eigenvalue=float(lam[i0]),
                           external_overlap=ext_overlap,
                           cart_disp=[list(map(float, d)) for d in disp])),
        open(ROOT + '/run_artifacts/02_nh3o3_reference/c1_freq_fix044/'
             'negmode_diagnostic.json', 'w'), indent=2)
    print()
    print('saved negmode_diagnostic.json')


if __name__ == '__main__':
    main()
