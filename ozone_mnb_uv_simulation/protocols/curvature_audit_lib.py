#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Curvature-audit shared library (JOB-2026-0905-009, curvature_audit batch).

Fixes relative to validation_stepD_softmode.py:
  * energy-directional curvature denominator corrected to q^2 (the old code
    divided by 2*q^2, underestimating k by 2 and the implied frequency by
    sqrt(2); the 'sqrt(2) conservative lower bound' statement is withdrawn);
  * 'max atom displacement' uses the LENGTH of each atom's displacement
    vector (max over atoms of |d_i|), not the max Cartesian component;
  * vibrational subspace: mass-weighted translation/rotation external space is
    built and projected out explicitly; internal space = 12 modes for the
    nonlinear hexamer; external-mode overlap is recorded for every disputed
    mode;
  * one single convention for masses/geometry/projection basis is used for
    the analytic Hessian, the finite-difference Hessian and the scans;
  * interface note (verified against the installed PySCF source):
    pyscf.hessian.thermo.harmonic_analysis returns norm_mode ALREADY in
    Cartesian displacement form (norm_mode = mass**-.5 (x) raw_mode), so the
    raw eigenvectors from numpy.eigh on the mass-weighted Hessian are
    mass-weighted coordinates and x_cart = M^-1/2 v q is the correct Cartesian
    conversion -- no double mass-weighting occurs when using raw eigenvectors.
"""
import numpy as np

BOHR_A = 0.52917721092


# ----------------------------------------------------------------- subspace
def mw_vector(masses):
    return np.repeat(np.asarray(masses, float) ** -0.5, 3)


def mw_hessian_matrix(H_block, masses):
    """(natm,natm,3,3) Hessian -> mass-weighted (3N,3N) matrix."""
    n = len(masses)
    mw = np.repeat(np.asarray(masses, float) ** -0.5, 3)
    H = np.asarray(H_block, float).transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)
    return H * np.outer(mw, mw)


def tr_subspace(masses, coords_bohr):
    """Orthonormal mass-weighted translation/rotation basis (3N x 6)."""
    masses = np.asarray(masses, float)
    coords = np.asarray(coords_bohr, float)
    n = len(masses)
    sqrtm = np.sqrt(masses)
    com = (masses[:, None] * coords).sum(0) / masses.sum()
    r = coords - com
    B = np.zeros((3 * n, 6))
    for i in range(n):
        B[3 * i:3 * i + 3, 0:3] = np.eye(3) * sqrtm[i]          # translations
        for k in range(3):                                       # rotations
            e = np.zeros(3)
            e[k] = 1.0
            B[3 * i:3 * i + 3, 3 + k] = np.cross(r[i], e) * sqrtm[i]
    Q, _ = np.linalg.qr(B)
    return Q


def projector(V):
    return np.eye(V.shape[0]) - V @ V.T


def external_overlap(vec, V):
    """|V^T v|_2 : how much of vector v lies in the external (TR) space."""
    return float(np.linalg.norm(V.T @ np.asarray(vec, float)))


def project_vector(vec, V):
    vec = np.asarray(vec, float)
    return vec - V @ (V.T @ vec)


def mode_cart_conversion(mw_vec, masses, q):
    """Mass-weighted eigenvector (unit norm) + amplitude q (amu^0.5*Bohr)
    -> Cartesian displacement per atom (Bohr).  x = M^-1/2 v q."""
    return (mw_vector(masses) * np.asarray(mw_vec, float)).reshape(-1, 3) * q


def max_atom_disp_a(d3n_reshaped_bohr, q):
    """Max per-atom displacement-vector LENGTH (Angstrom) for amplitude q."""
    per_atom = np.linalg.norm(np.asarray(d3n_reshaped_bohr, float), axis=1)
    return q * per_atom.max() * BOHR_A


def q_for_atom_disp(d3n_reshaped_bohr, target_a):
    """Amplitude q (amu^0.5*Bohr... in the units of d) that yields a max
    per-atom displacement of target_a Angstrom."""
    per_atom = np.linalg.norm(np.asarray(d3n_reshaped_bohr, float), axis=1)
    return (target_a / BOHR_A) / per_atom.max()


# ----------------------------------------------- displacement generation (shared)
def displaced_geometry(coords, d3, q_bohr, unit='Angstrom'):
    """Displace a geometry by q along d (SIDE-EFFECT FREE).

    q_bohr is a BOHR-convention amplitude (q * max|d_a| = amplitude in Bohr).
      unit='Angstrom' -> returns coords + q * d * BOHR_A   (Angstrom result)
      unit='Bohr'     -> returns coords + q * d            (Bohr result)
    Both are consistent: the displacement in Bohr equals q*d either way.
    The pre-fix defect was applying q*d to Angstrom coordinates without the
    BOHR_A factor (amplifying the amplitude by 1/b = 1.8897).
    """
    if unit not in ('Angstrom', 'Bohr'):
        raise ValueError('unit must be Angstrom or Bohr')
    coords = np.asarray(coords, float)
    d3 = np.asarray(d3, float)
    if unit == 'Angstrom':
        return coords + q_bohr * d3 * BOHR_A
    return coords + q_bohr * d3


def recover_q(coords_ref, coords_disp, d3, unit='Angstrom', tol=1e-8):
    """Recover the Bohr amplitude q from ACTUAL coordinates and validate:
    the displacement must be parallel to d (sign apart) and the recovered q
    must be unique per atom.  Raises on violation (no silent fallback)."""
    d3 = np.asarray(d3, float)
    delta = np.asarray(coords_disp, float) - np.asarray(coords_ref, float)
    if unit == 'Angstrom':
        delta_bohr = delta / BOHR_A
    elif unit == 'Bohr':
        delta_bohr = delta
    else:
        raise ValueError('unit must be Angstrom or Bohr')
    per_atom = np.linalg.norm(delta_bohr, axis=1)
    d_norm = np.linalg.norm(d3, axis=1)
    qs = []
    for a in range(len(d_norm)):
        if d_norm[a] < 1e-12:
            if per_atom[a] > 1e-10:
                raise ValueError('displacement on an atom with zero direction '
                                 'component (atom %d)' % a)
            continue
        qs.append(per_atom[a] / d_norm[a])
    if not qs:
        raise ValueError('no usable atom displacement')
    q = float(np.mean(qs))
    if max(abs(np.array(qs) - q)) > tol * max(1.0, abs(q)):
        raise ValueError('inconsistent per-atom q (spread %.3e)' %
                         max(abs(np.array(qs) - q)))
    # direction check (sign free)
    flat = delta_bohr.reshape(-1)
    dflat = d3.reshape(-1)
    cosang = float(np.dot(flat, dflat) /
                   (np.linalg.norm(flat) * np.linalg.norm(dflat)))
    if abs(abs(cosang) - 1.0) > 1e-8:
        raise ValueError('displacement not parallel to d (|cos| = %.10f)' % cosang)
    return dict(q_bohr=q, sign=int(np.sign(cosang)),
                max_atom_disp_bohr=float(per_atom.max()),
                max_atom_disp_angstrom=float(per_atom.max() * BOHR_A))


def curvature_from_points(e_plus, e_minus, e0, g_plus, g_minus, d, q_bohr):
    """Official difference estimators.  All arguments refer to the SAME
    direction d and the SAME Bohr amplitude q_bohr."""
    d = np.asarray(d, float)
    return dict(
        k_E=(e_plus + e_minus - 2 * e0) / q_bohr ** 2,
        k_g=float(d @ (np.asarray(g_plus) - np.asarray(g_minus))) / (2 * q_bohr),
        s_E=float(e_plus - e_minus) / (2 * q_bohr))


# ------------------------------------------------------- curvature estimators
def curvature_three_way(e_plus, e_minus, e0, g_plus, g_minus, g0, d, q):
    """All three curvature estimates along the SAME fixed direction d
    (unit 3N vector) and the SAME amplitude q (Bohr), plus slope indicators.

      k_E = [E(+q) + E(-q) - 2 E0] / q^2          (energy 2nd difference)
      k_g = d . [g(+q) - g(-q)] / (2q)            (gradient direction diff)
      k_H = d^T H d                                (analytic, caller-supplied)
    The odd (linear) terms cancel in k_E's numerator and in k_g's difference;
    the residual slope indicators are reported separately:
      slope_g = d . g0                             (non-stationarity)
      slope_E = [E(+q) - E(-q)] / (2q)             (= slope_g + O(q^2))
    """
    k_E = (e_plus + e_minus - 2.0 * e0) / q ** 2
    k_g = float(np.dot(d, (np.asarray(g_plus) - np.asarray(g_minus)))) / (2.0 * q)
    slope_g = float(np.dot(d, np.asarray(g0)))
    slope_E = float(e_plus - e_minus) / (2.0 * q)
    return dict(k_E=k_E, k_g=k_g, slope_g=slope_g, slope_E=slope_E)
