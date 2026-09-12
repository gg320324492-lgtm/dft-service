#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JOB-2026-0906-013 (Task C part 1, OFFLINE linear algebra only).

Consumes hessian_batch.json + center_order47.json from Task B and produces:
  * mass-weighted Hessian H_mw (18x18)
  * external (translation+rotation) orthonormal basis Q_ext (18x6, rank 6)
  * internal orthogonal complement U (18x12, shape verified)
  * blocks: U'H_mwU (internal), Q'H_mwU (coupling), Q'H_mwQ (external)
  * all 12 internal eigenvalues/modes for BOTH step lengths
    (negative eigenvalues / imaginary frequencies kept verbatim)
  * PySCF cross-check (same masses, same projection convention) --
    validates MATRIX HANDLING ONLY, not the raw derivatives or any
    physical interpretation at this non-stationary point
  * two-step-length mode matching (overlap; near-degenerate -> subspace)
  * full gradient internal/external decomposition in the same convention

DIRECTION SELECTION RULE -- FIXED HERE, BEFORE ANY DIRECTION EVALUATION:
  primary matrix: h = 0.002 Bohr (smaller step)
  dir1  = lowest internal eigenvalue mode
  dir2  = most negative eigenvalue mode if any lambda < 0,
          otherwise the mode with the largest two-step frequency shift
          IF that shift exceeds 20 cm^-1 (fixed threshold),
          otherwise NO second direction.
"""
import os
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, 'run_artifacts', '01_pure_water_o3_h2o')
OUT = os.path.join(ART, 'smd_diag_curvature')
sys.path.insert(0, HERE)
import torque_decomp as td

SYMS = ['O', 'O', 'O', 'O', 'H', 'H']
MASS = np.array([15.9949146] * 4 + [1.007825] * 2)
BOHR_A = td.BOHR_A
SENS_THRESHOLD_CM = 20.0          # FIXED before any direction evaluation


def wavenumbers(lam):
    """Convert eigenvalues of the mass-weighted internal block
    [Eh / (Bohr^2 amu)] to cm^-1 (negative -> imaginary, sign kept)."""
    # PySCF convention: freq(cm^-1) = sign(l)*sqrt(|l|) * factor,
    # factor = sqrt(Eh/(Bohr^2 amu)) -> cm^-1
    from pyscf.data import nist
    hartree_j = nist.HARTREE2J
    amu_kg = nist.ATOMIC_MASS   # kg per amu
    bohr_m = nist.BOHR_SI
    c = nist.LIGHT_SPEED_SI
    # omega [rad/s] = sqrt(lam * Eh_J / (Bohr_m^2 * amu_kg))
    omega = np.sqrt(np.abs(lam) * hartree_j / (bohr_m ** 2 * amu_kg))
    cm1 = omega / (2 * np.pi * c) / 100.0 * np.sign(lam)
    return cm1


def main():
    hb = json.load(open(os.path.join(OUT, 'hessian_batch.json')))
    center_rec = json.load(open(os.path.join(OUT, 'center_order47.json')))
    assert hb['n_points_ok'] == 72, 'Hessian batch incomplete: %s' % hb['n_points_ok']
    center = np.asarray(center_rec['coords_angstrom'], float)   # Angstrom
    center_bohr = center / BOHR_A
    e0 = center_rec['e_total']
    g0 = np.asarray(center_rec['gradient_full_6x3'], float)     # (6,3) Eh/Bohr

    mats = {}
    results = {'job': 'JOB-2026-0906-013 Task C (analysis, offline)',
               'convention': dict(
                   mass_amu=MASS.tolist(),
                   displacement='Cartesian, Bohr (Hessian) / Angstrom (direction checks)',
                   mass_weighting='H_mw[i,j] = H[i,j]/sqrt(m_i m_j), '
                                  'i -> atom i//3, cart i%3',
                   external_basis='mass-weighted translations+rotations about '
                                  'the mass-centre, QR-orthonormalised (rank 6)',
                   note='at a NON-STATIONARY point the curvature '
                        'interpretation depends on the defined displacement '
                        'path; a projected positive spectrum does NOT prove '
                        'a true internal or full minimum')}
    for hkey in ('0.002', '0.004'):
        h = float(hkey)
        Hsym = np.asarray(hb['hessians'][hkey]['sym_hessian_18x18'], float)
        mroot = np.sqrt(np.repeat(MASS, 3))
        Hmw = Hsym / np.outer(mroot, mroot)

        # external basis (mass-weighted)
        com = (np.einsum('z,zx->x', MASS, center_bohr) / MASS.sum())
        rb = center_bohr - com
        cols = []
        for a in range(3):                       # translations
            v = np.zeros((6, 3))
            v[:, a] = np.sqrt(MASS)
            cols.append(v.reshape(-1))
        for a in range(3):                       # rotations
            e = np.zeros(3); e[a] = 1.0
            v = np.sqrt(MASS)[:, None] * np.cross(rb, np.tile(e, (6, 1)))
            cols.append(v.reshape(-1))
        B = np.array(cols).T                     # 18x6
        Q, R = np.linalg.qr(B)
        svals = np.linalg.svd(B, compute_uv=False)
        ext_rank = int((svals > 1e-7).sum())
        assert Q.shape == (18, 6) and ext_rank == 6, (Q.shape, ext_rank)
        # internal complement via projector eigendecomposition (deterministic)
        P = np.eye(18) - Q @ Q.T
        w, V = np.linalg.eigh(P)                 # ascending
        U = V[:, w > 0.5]                        # 18x12
        assert U.shape == (18, 12), U.shape
        ortho_err = float(np.abs(U.T @ U - np.eye(12)).max()
                          + np.abs(Q.T @ U).max())

        Hint = U.T @ Hmw @ U
        Hcouple = Q.T @ Hmw @ U
        Hext = Q.T @ Hmw @ Q
        lam, Vm = np.linalg.eigh(Hint)
        nu = wavenumbers(lam)
        modes_cart_mw = (U @ Vm).T          # 12 x 18, mass-weighted Cartesian

        # gradient decomposition in the same convention
        gmw = (g0.reshape(-1) / mroot)
        g_int = U.T @ gmw
        g_ext = Q.T @ gmw

        # PySCF cross-check (same masses via mol.mass, same projection idea)
        pyscf_check = None
        try:
            from pyscf import gto
            from pyscf.hessian import thermo
            atom = "; ".join("%s %.10f %.10f %.10f" % (s, x, y, z)
                             for s, (x, y, z) in zip(SYMS, center))
            mol = gto.M(atom=atom, basis='sto-3g', verbose=0)
            mol.mass = MASS.tolist()
            h4 = Hsym.reshape(6, 3, 6, 3).transpose(0, 2, 1, 3)
            # h4[p,q,x,y] = Hsym[3p+x, 3q+y]  (PySCF Hessian layout;
            # NOTE: a plain reshape(6,6,3,3) would SCRAMBLE the matrix)
            ha = thermo.harmonic_analysis(mol, h4)
            fw = [complex(v) for v in ha['freq_wavenumber']]
            signed = [(-abs(v.imag) if abs(v.imag) > 1e-10 * max(1.0, abs(v))
                       else float(v.real)) for v in fw]
            is_imag = [bool(abs(v.imag) > 1e-10 * max(1.0, abs(v)))
                       for v in fw]
            pyscf_check = dict(
                freq_wavenumber_signed_cm1=signed,
                imaginary_flags=is_imag,
                n_modes=int(len(signed)),
                note='PySCF harmonic_analysis on the SAME sym Hessian with '
                     'the SAME masses; negative = imaginary; cross-check of '
                     'matrix handling only')
        except Exception as e:
            pyscf_check = dict(error=repr(e))

        results[hkey] = dict(
            antisym_max=hb['hessians'][hkey]['antisym_residual_max'],
            antisym_rel=hb['hessians'][hkey]['antisym_relative'],
            Q_ext_shape=list(Q.shape), external_rank=ext_rank,
            U_shape=list(U.shape), orthogonality_err=ortho_err,
            internal_block=Hint.tolist(),
            coupling_block=Hcouple.tolist(),
            external_block=Hext.tolist(),
            coupling_rms=float(np.sqrt((Hcouple ** 2).mean())),
            eigenvalues=lam.tolist(),
            wavenumber_cm1=nu.tolist(),
            modes_internal_basis=Vm.T.tolist(),
            modes_cartesian_mw=modes_cart_mw.tolist(),
            gradient_internal_mw=g_int.tolist(),
            gradient_external_mw=g_ext.tolist(),
            gradient_internal_mw_norm=float(np.linalg.norm(g_int)),
            gradient_external_mw_norm=float(np.linalg.norm(g_ext)),
            pyscf_cross_check=pyscf_check)
        mats[hkey] = dict(H_mw=Hmw.tolist(), Q_ext=Q.T.tolist(), U=U.T.tolist())

    # ---------- two-step mode matching ----------
    l1 = np.asarray(results['0.002']['eigenvalues'])
    l2 = np.asarray(results['0.004']['eigenvalues'])
    V1 = np.asarray(results['0.002']['modes_internal_basis'])
    V2 = np.asarray(results['0.004']['modes_internal_basis'])
    O = V1 @ V2.T
    matches = []
    for i in range(12):
        j = int(np.argmax(np.abs(O[i])))
        second = np.max(np.abs(np.delete(O[i], j)))
        near_degenerate = bool(second > 0.2)
        matches.append(dict(
            mode_h002=i, mode_h004=j, overlap=float(O[i, j]),
            second_best_overlap=float(second),
            near_degenerate=near_degenerate,
            lambda_h002=float(l1[i]), lambda_h004=float(l2[j]),
            nu_h002_cm1=float(results['0.002']['wavenumber_cm1'][i]),
            nu_h004_cm1=float(results['0.004']['wavenumber_cm1'][j]),
            delta_nu_cm1=float(abs(results['0.002']['wavenumber_cm1'][i]
                                   - results['0.004']['wavenumber_cm1'][j]))))
    results['mode_matching'] = matches
    results['matching_note'] = ('two-step agreement is NOT a strict error '
                                'bound; near-degenerate modes compared by '
                                'subspace overlap (second-best > 0.2 flagged)')

    # ---------- direction selection (RULE FIXED ABOVE) ----------
    nu1 = results['0.002']['wavenumber_cm1']
    i_low = int(np.argmin(l1))
    directions = [dict(
        rank=1, rule='lowest internal eigenvalue (h=0.002 matrix)',
        mode_index=i_low, eigenvalue=float(l1[i_low]),
        wavenumber_cm1=float(nu1[i_low]))]
    neg = [i for i in range(12) if l1[i] < 0]
    dir2 = None
    if neg:
        i_n = int(np.argmin(l1))
        dir2 = dict(rank=2, rule='most negative eigenvalue (lambda < 0 '
                                 'present)', mode_index=i_n,
                    eigenvalue=float(l1[i_n]),
                    wavenumber_cm1=float(nu1[i_n]))
    else:
        shifts = [(m['mode_h002'], m['delta_nu_cm1']) for m in matches]
        m_max, d_max = max(shifts, key=lambda t: t[1])
        if d_max > SENS_THRESHOLD_CM:
            dir2 = dict(rank=2, rule='largest two-step frequency shift '
                                     '(> %g cm^-1 fixed threshold)'
                                     % SENS_THRESHOLD_CM,
                        mode_index=m_max,
                        eigenvalue=float(l1[m_max]),
                        wavenumber_cm1=float(nu1[m_max]),
                        delta_nu_cm1=float(d_max))
    directions.append(dir2 if dir2 else dict(
        rank=2, rule='no lambda < 0 and no shift > %g cm^-1 -> NO second '
                     'direction' % SENS_THRESHOLD_CM, mode_index=None))
    # convert to normalized Cartesian directions (max atom |d_i| = 1)
    mroot = np.sqrt(np.repeat(MASS, 3))
    for d in directions:
        if d.get('mode_index') is None:
            continue
        v = np.asarray(results['0.002']['modes_cartesian_mw'])[
            d['mode_index']]                       # mass-weighted Cartesian
        dcart = (v / mroot).reshape(6, 3)          # un-mass-weight
        norm = float(np.abs(np.linalg.norm(dcart, axis=1)).max())
        dcart = dcart / norm
        d['cartesian_direction_6x3'] = dcart.tolist()
        d['normalization'] = ('max atom displacement-vector length = 1; '
                              'unit: displacement along this direction is '
                              'measured in Angstrom (s * d)')
    results['direction_selection'] = dict(
        threshold_cm1=SENS_THRESHOLD_CM, directions=directions,
        n_directions=sum(1 for d in directions if d.get('mode_index') is not None))
    # dedup: if the dir2 rule selected the SAME mode as dir1 (negative mode
    # coincides with the lowest mode), keep ONE direction -- the rule's
    # intent is a second INDEPENDENT direction; record the dedup decision.
    sel = results['direction_selection']
    real_dirs = [d for d in directions if d.get('mode_index') is not None]
    if len(real_dirs) > 1 and real_dirs[0]['mode_index'] == real_dirs[1]['mode_index']:
        sel['dedup_note'] = ('dir2 rule (most negative lambda) selected the '
                             'same mode as dir1 (lowest lambda); deduplicated '
                             'to ONE direction, saving 8 duplicate '
                             'evaluations (batch authorises AT MOST two '
                             'directions)')
        sel['directions'] = [directions[0],
                             dict(rank=2, mode_index=None,
                                  rule='DEDUPLICATED into dir1 (same mode '
                                       'selected by both rules)')]
        sel['n_directions'] = 1
    results['interpretation_warning'] = (
        'At this NON-STATIONARY point (max|g| = %.3e > 1e-5, rotation-type '
        'residual dominant): curvature interpretation depends on the '
        'defined displacement path; a projected positive spectrum does NOT '
        'prove a true internal minimum or a full minimum; formal stationary '
        'acceptance remains NOT passed.'
        % float(np.abs(g0).max()))

    json.dump(results, open(os.path.join(OUT, 'analysis.json'), 'w'), indent=2)
    json.dump(mats, open(os.path.join(OUT, 'projection_matrices.json'), 'w'),
              indent=2)
    print('ANALYSIS -> analysis.json / projection_matrices.json')
    for hkey in ('0.002', '0.004'):
        nu = results[hkey]['wavenumber_cm1']
        print('h=%s internal wavenumbers [cm^-1]:' % hkey)
        print('  ', np.round(nu, 1).tolist())
        print('   coupling RMS: %.3e' % results[hkey]['coupling_rms'])
    print('directions:', [(d['rank'], d.get('mode_index'),
                           round(d.get('wavenumber_cm1', 0), 1))
                          for d in directions])


if __name__ == '__main__':
    main()
