#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Endpoint-freq batch Step 3 (JOB-2026-0906-004, REVISED in JOB-2026-0906-005
Phase A -- this revision supersedes the 004 analysis output):

Full internal frequency analysis from the difference Hessians.

REVISION (Phase A of JOB-2026-0906-005):
  * Each step size now builds the PySCF-layout Hessian FROM ITS OWN
    symmetrized Cartesian Hessian (the 004 revision wrongly reused the
    h=0.002 matrix for the h=0.004 cross-check, which made the h=0.004
    "cross-check difference" exactly equal the inter-step frequency change).
  * A same-matrix round-trip assertion guards the layout conversion.
  * Test hook ``override_hess_pqxy`` reproduces the old wrong-matrix path
    for the regression test (the same-matrix assertion is bypassed there and
    the mismatch is reported in the output).
  * The withdrawn explanation "h=0.004 difference comes from independent
    diagonalisation rounding" is replaced by the true same-matrix
    cross-check numbers.

For each candidate and each step size h in {0.002, 0.004} Bohr:
  * assemble the UNSYMMETRIZED 18x18 Hessian from the 36 gradient pairs,
    H[:, j] = [g(R+h e_j) - g(R-h e_j)]/(2h)  (conventional indexing);
  * report the antisymmetric residual (max, Frobenius, and RELATIVE to the
    mass-weighted internal scale), THEN symmetrize;
  * mass-weight with isotope-averaged masses (PySCF convention);
  * build the translation/rotation subspace (PySCF _get_TR conventions),
    verify rank 6, take U = orthonormal complement (18x12);
  * solve the 12x12 internal eigenproblem  U^T H_mw U  -- ALL 12 internal
    eigenvalues/modes kept (negatives preserved, never abs-ed/deleted/0-ed);
  * cross-check against pyscf.hessian.thermo.harmonic_analysis with the
    SAME matrix, same masses, same projection conventions;
  * comparisons between the two step sizes are reported SEPARATELY as:
    (a) Cartesian Hessian difference, (b) mass-weighted INTERNAL matrix
    difference in the SAME U basis, (c) mode-overlap matching and frequency
    changes.  None of these is claimed to be a strict error bound.
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, 'run_artifacts', '01_pure_water_o3_h2o')
OUT = os.path.join(ART, 'endpoint_frequency')
sys.path.insert(0, HERE)

from pyscf import gto
from pyscf.hessian import thermo as pyscf_thermo
from pyscf.data import nist

SYMS = ['O', 'O', 'O', 'O', 'H', 'H']
STEPS = [0.002, 0.004]
AU2WN = ((nist.HARTREE2J / (nist.ATOMIC_MASS * nist.BOHR_SI ** 2)) ** .5
         / (2 * np.pi) / nist.LIGHT_SPEED_SI * 1e-2)   # thermo.py L92-93


# ---- PySCF-identical TR construction (thermo.py _get_TR, replicated) ----
def get_TR(mass, coords):
    mass_center = np.einsum('z,zx->x', mass, coords) / mass.sum()
    coords = coords - mass_center
    massp = mass ** .5
    Tx = np.einsum('m,x->mx', massp, [1, 0, 0])
    Ty = np.einsum('m,x->mx', massp, [0, 1, 0])
    Tz = np.einsum('m,x->mx', massp, [0, 0, 1])
    im = np.einsum('m,mx,my->xy', mass, coords, coords)
    im = np.eye(3) * im.trace() - im
    w, paxes = np.linalg.eigh(im)
    w = w[::-1]; paxes = paxes[:, ::-1]
    ex, ey, ez = paxes.T
    cr = coords.dot(paxes)
    cx, cy, cz = cr.T
    Rx = massp[:, None] * (cy[:, None] * ez - cz[:, None] * ey)
    Ry = massp[:, None] * (cz[:, None] * ex - cx[:, None] * ez)
    Rz = massp[:, None] * (cx[:, None] * ey - cy[:, None] * ex)
    return np.vstack([Tx.ravel(), Ty.ravel(), Tz.ravel(),
                      Rx.ravel(), Ry.ravel(), Rz.ravel()])


def load_points(tid, h):
    pts = {}
    for sgn in (1, -1):
        for j in range(18):
            p = os.path.join(OUT, 'hessian_points',
                             'point_%s_h%.3f_dir%02d_sgn%+d.json' % (tid, h, j, sgn))
            r = json.load(open(p))
            assert r['scf_converged'] and r['grid_response_actual'], p
            assert r['fingerprint']['step_bohr'] == h
            pts[(j, sgn)] = r
    return pts


def assemble_hessian(tid, h):
    pts = load_points(tid, h)
    H = np.zeros((18, 18))
    for j in range(18):
        gp = np.asarray(pts[(j, 1)]['gradient_full_6x3'], float).reshape(-1)
        gm = np.asarray(pts[(j, -1)]['gradient_full_6x3'], float).reshape(-1)
        H[:, j] = (gp - gm) / (2 * h)
    anti = float(np.abs(H - H.T).max())
    anti_fro = float(np.linalg.norm(H - H.T))
    return H, anti, anti_fro, pts


def analyse(tid, h, mass, coords_B, override_hess_pqxy=None,
            skip_pyscf=False):
    """Full internal frequency analysis for ONE step size.

    The PySCF cross-check uses the Hessian built from THIS step's own
    symmetrized Cartesian matrix, guarded by a same-matrix round-trip
    assertion.  ``override_hess_pqxy`` (test-only) reproduces the 004
    wrong-matrix path; it bypasses the assertion and the output records
    ``same_matrix_as_eigenproblem=False`` so the regression test can show
    the cross-check failing loudly.
    """
    H18, anti_max, anti_fro, _ = assemble_hessian(tid, h)
    Hsym = 0.5 * (H18 + H18.T)
    m = mass
    # mass-weighted, PySCF convention (thermo.py L55):
    # h4[p,x,q,y] = H[p*3+x, q*3+y] / sqrt(m_p * m_q).
    # NOTE: h4 is ALREADY in (p,x,q,y) layout, so reshape(18,18) directly
    # gives [(p,x),(q,y)] -- the diagonal mass scaling D h D stays a
    # congruence (sign-preserving).  Do NOT apply PySCF's transpose here:
    # PySCF transposes its own (p,q,x,y)-layout array, ours is different.
    h4 = (Hsym.reshape(6, 3, 6, 3)
          / np.sqrt(m)[:, None, None, None]
          / np.sqrt(m)[None, None, :, None])
    h_mw = h4.reshape(18, 18)
    # PySCF-layout Hessian FROM THIS STEP'S OWN MATRIX
    hess_pqxy_own = Hsym.reshape(6, 3, 6, 3).transpose(0, 2, 1, 3)
    # same-matrix round-trip assertion: (p,q,x,y) -> (p,x,q,y) -> 18x18
    rt = hess_pqxy_own.transpose(0, 2, 1, 3).reshape(18, 18)
    assert np.allclose(rt, Hsym, atol=1e-12), \
        'SAME-MATRIX ASSERTION FAILED: PySCF-layout Hessian does not ' \
        'round-trip to this step\'s symmetrized Cartesian Hessian'
    # TR subspace (rank must be 6) and orthogonal complement U (18x12)
    TR = get_TR(m, coords_B)
    q, r = np.linalg.qr(TR.T)
    diag = np.abs(np.diag(r))
    rank6 = bool((diag > 1e-7).all()) and q.shape[1] == 6
    P = np.eye(18) - q.dot(q.T)
    w, v = np.linalg.eigh(P)
    U = v[:, w > 1e-7]                      # 18x12
    assert U.shape == (18, 12), U.shape
    assert rank6, 'TR subspace rank != 6'
    ortho_err = float(np.abs(U.T.dot(q)).max())
    # internal eigenproblem: ALL 12 eigenvalues/modes
    Hint = U.T.dot(h_mw).dot(U)
    ev, vec = np.linalg.eigh(Hint)          # ascending
    modes_cart = U.dot(vec)                 # (18, 12) mass-weighted modes
    freq_signed = np.sign(ev) * np.sqrt(np.abs(ev)) * AU2WN   # neg = imaginary
    # norm_mode[i, r, z]: mode i over (natm, 3), mass-weighted like PySCF
    # thermo.py L95: norm_mode = einsum('z,zri->izr', m**-.5, mode.reshape(6,3,-1))
    norm_mode = modes_cart.T.reshape(12, 6, 3) \
        * (m ** -0.5)[None, :, None]
    red_mass = 1.0 / np.einsum('izr,izr->i', norm_mode, norm_mode)
    # ---- PySCF cross-check (same masses, same projection convention) ----
    wrong_matrix = override_hess_pqxy is not None
    if skip_pyscf:
        pyscf_wn = np.full(12, np.nan)
        cross_max = None
        axis_check = dict(skipped=True)
    else:
        mol = gto.M(atom="; ".join("%s %.10f %.10f %.10f"
                                   % (s, x * 0.52917721092, y * 0.52917721092,
                                      z * 0.52917721092)
                                   for s, (x, y, z) in zip(SYMS, coords_B)),
                    basis='def2-TZVP', charge=0, spin=0, verbose=0,
                    max_memory=2000)
        ha = pyscf_thermo.harmonic_analysis(
            mol, hess_pqxy_own if not wrong_matrix else override_hess_pqxy,
            imaginary_freq=False)
        pyscf_wn = np.sort(np.asarray(ha['freq_wavenumber'], float))
        mine_wn = np.sort(freq_signed)
        cross_max = float(np.abs(pyscf_wn - mine_wn).max())
        axis_check = dict(norm_mode_shape=list(ha['norm_mode'].shape),
                          expected='(12, 6, 3) -> (nmode, natm, 3), per thermo.py L95')
    return dict(
        step_bohr=h,
        hessian_unsymmetrized=H18.tolist(),
        hessian_symmetrized=Hsym.tolist(),
        antisym_residual_max=anti_max, antisym_residual_fro=anti_fro,
        tr_rank6=rank6, U_shape=[18, 12], U=U.tolist(),
        orthogonality_err_max=ortho_err,
        mass_amu=m.tolist(), mass_convention='atom_mass_list(isotope_avg=True)',
        internal_eigenvalues_eh_bohr2_amu1=ev.tolist(),
        freq_wavenumber_signed=freq_signed.tolist(),
        n_negative_modes=int((ev < 0).sum()),
        lowest_eigenvalue=float(ev[0]), lowest_freq_wn=float(freq_signed[0]),
        modes_cart_massweighted=modes_cart.T.tolist(),   # (12, 18)
        norm_mode=norm_mode.tolist(),                    # (12, 6, 3)
        reduced_mass=red_mass.tolist(),
        internal_matrix_mw_Ubasis=Hint.tolist(),         # (12,12)
        pyscf_crosscheck=dict(
            max_abs_wn_diff=cross_max,
            same_matrix_as_eigenproblem=bool(not wrong_matrix),
            pyscf_freq_wavenumber=pyscf_wn.tolist(),
            axis_order_check=axis_check),
    )


def match_modes(a, b):
    """Hungarian matching by |overlap| of the 12 mass-weighted modes."""
    from scipy.optimize import linear_sum_assignment
    A = np.asarray(a['modes_cart_massweighted'])    # (12,18)
    B = np.asarray(b['modes_cart_massweighted'])
    S = np.abs(A @ B.T)
    i, j = linear_sum_assignment(-S)
    pairs = sorted(zip(i.tolist(), j.tolist()))
    return [dict(mode_h1=p[0], mode_h2=p[1],
                 overlap=float(S[p[0], p[1]]),
                 wn_h1=a['freq_wavenumber_signed'][p[0]],
                 wn_h2=b['freq_wavenumber_signed'][p[1]],
                 d_wn=float(b['freq_wavenumber_signed'][p[0]]
                            - a['freq_wavenumber_signed'][p[0]]))
            for p in pairs]


def main():
    fx = json.load(open(os.path.join(OUT, 'endpoints_fixed.json')))
    results = {}
    for tid in ('c14_plus', 'c06_plus'):
        coords_A = np.asarray(fx['endpoints'][tid]['coords_angstrom'], float)
        coords_B = coords_A / 0.52917721092
        mol = gto.M(atom="; ".join("%s %.10f %.10f %.10f"
                                   % (s, x, y, z)
                                   for s, (x, y, z) in zip(SYMS, coords_A)),
                    basis='def2-TZVP', charge=0, spin=0, verbose=0,
                    max_memory=2000)
        mass = mol.atom_mass_list(isotope_avg=True)
        per_h = {}
        for h in STEPS:
            per_h['%.3f' % h] = analyse(tid, h, mass, coords_B)
        # ---- step-size comparisons, SEPARATELY NAMED (005 Phase A.2) ----
        H1 = np.asarray(per_h['0.002']['hessian_symmetrized'])
        H2 = np.asarray(per_h['0.004']['hessian_symmetrized'])
        Hint1 = np.asarray(per_h['0.002']['internal_matrix_mw_Ubasis'])
        Hint2 = np.asarray(per_h['0.004']['internal_matrix_mw_Ubasis'])
        mt = match_modes(per_h['0.002'], per_h['0.004'])
        anti_scale = dict(
            antisym_max={k: per_h[k]['antisym_residual_max']
                         for k in ('0.002', '0.004')},
            antisym_fro={k: per_h[k]['antisym_residual_fro']
                         for k in ('0.002', '0.004')},
            internal_mw_fro_norm=float(np.linalg.norm(Hint1)),
            lowest_mode_wn_change=mt[0]['d_wn'],
            note='antisym residuals are reported against the mass-weighted '
                 'internal scale and the step-size trend; none of these '
                 'comparisons is a strict error bound')
        results[tid] = dict(
            per_step=per_h,
            step_comparison=dict(
                cartesian_hessian_max_diff=float(np.abs(H1 - H2).max()),
                cartesian_hessian_fro_diff=float(np.linalg.norm(H1 - H2)),
                internal_mw_matrix_max_diff_sameUbasis=
                    float(np.abs(Hint1 - Hint2).max()),
                mode_matching=mt,
                min_overlap=float(min(p['overlap'] for p in mt)),
                max_abs_d_wn_matched=max(abs(p['d_wn']) for p in mt),
                lowest_eigenvalue_h002=per_h['0.002']['lowest_eigenvalue'],
                lowest_eigenvalue_h004=per_h['0.004']['lowest_eigenvalue'],
                n_negative_h002=per_h['0.002']['n_negative_modes'],
                n_negative_h004=per_h['0.004']['n_negative_modes'],
                antisym_scale_context=anti_scale))
        for h in STEPS:
            a = per_h['%.3f' % h]
            print('[%s] h=%s: lowest lambda=%.4e (wn=%.2f) n_neg=%d '
                  'anti=%.2e  SAME-MATRIX pyscf xcheck=%.2e wn'
                  % (tid, h, a['lowest_eigenvalue'], a['lowest_freq_wn'],
                     a['n_negative_modes'], a['antisym_residual_max'],
                     a['pyscf_crosscheck']['max_abs_wn_diff']), flush=True)
        sc = results[tid]['step_comparison']
        print('[%s] cart diff=%.2e  internal(mw,U-basis) diff=%.2e  '
              'min overlap=%.6f  max|d_wn|=%.2f cm-1'
              % (tid, sc['cartesian_hessian_max_diff'],
                 sc['internal_mw_matrix_max_diff_sameUbasis'],
                 sc['min_overlap'], sc['max_abs_d_wn_matched']), flush=True)
    out = dict(job='JOB-2026-0906-005 Phase A (revised analysis; supersedes '
                   'the 004 frequency_analysis.json cross-check fields)',
               candidates=results)
    with open(os.path.join(OUT, 'frequency_analysis_revised.json'),
              'w') as fh:
        json.dump(out, fh, indent=2)
    print('saved -> frequency_analysis_revised.json')
    return out


if __name__ == '__main__':
    main()
