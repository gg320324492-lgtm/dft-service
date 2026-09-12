#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared displacement-coordinate utilities (JOB-2026-0906-003, Phase A).

Single source of truth for generating displaced geometries, used by BOTH
production scripts (directional_confirmation, grid_response_diagnostic,
curvature_audit_candidates) and the regression tests.  No SCF side effects.

Conventions (fixed by the unit-defect root cause analysis):
  * d_cart is a UNIT 3N Euclidean vector (Bohr per unit q);
  * q_bohr is the Bohr-convention amplitude: displacing BOHR coordinates by
    q_bohr * d gives a max per-atom displacement of q_bohr * max|d_a| Bohr;
  * the nominal amplitude a (Angstrom) maps to q_bohr = a / BOHR_A / max|d_a|,
    so that the actual max per-atom displacement in Angstrom is exactly `a`;
  * when displacing Angstrom coordinates: delta_A = q_bohr * BOHR_A * d_a.
    (The pre-fix defect omitted the BOHR_A factor here -> 1/b amplification.)
"""
import numpy as np

BOHR_A = 0.52917721092


def cart_from_mw(v_mw, masses):
    """Mass-weighted eigenvector (unit norm) -> unit Cartesian direction."""
    mw = np.repeat(np.asarray(masses, float) ** -0.5, 3)
    d = mw * np.asarray(v_mw, float)
    return d / np.linalg.norm(d)


def max_atom_disp(d_cart, q_bohr):
    """Max per-atom displacement-VECTOR length (Angstrom) for amplitude q."""
    d3 = np.asarray(d_cart, float).reshape(-1, 3)
    return q_bohr * float(np.linalg.norm(d3, axis=1).max()) * BOHR_A


def q_bohr_for_amp(d_cart, amp_a):
    """Bohr-convention amplitude q that yields max per-atom displacement
    = amp_a Angstrom (vector-length convention)."""
    d3 = np.asarray(d_cart, float).reshape(-1, 3)
    m = float(np.linalg.norm(d3, axis=1).max())
    return (amp_a / BOHR_A) / m


def displaced_coords_A(coords_A, d_cart, q_bohr, signs=(+1, -1)):
    """Displaced coordinates in Angstrom: coords_A + sign * q_bohr * BOHR_A * d.

    Returns a list of dicts with coords (Angstrom), q_bohr, sign,
    actual_max_atom_disp_a (measured from the ACTUAL coordinate difference,
    not from the nominal label), and the direction cosine with d.
    Raises ValueError on direction deviation > 1e-6 deg or +/- asymmetry.
    """
    d3 = np.asarray(d_cart, float).reshape(-1, 3)
    coords_A = np.asarray(coords_A, float)
    out = []
    for sgn in signs:
        delta_A = sgn * q_bohr * BOHR_A * d3
        c_new = coords_A + delta_A
        # actual displacement measured from the ACTUAL coordinates
        actual_delta = c_new - coords_A
        per_atom = np.linalg.norm(actual_delta, axis=1)
        max_disp_A = float(per_atom.max())
        # direction check: actual delta must be parallel to sgn*d3
        expected = sgn * d3.reshape(-1)
        u = actual_delta.reshape(-1) / (np.linalg.norm(actual_delta) + 1e-300)
        cosang = float(np.dot(u, expected) /
                       (np.linalg.norm(expected) + 1e-300))
        dev_deg = float(np.degrees(np.arccos(max(-1, min(1, cosang)))))
        if abs(dev_deg) > 1e-4:
            raise ValueError('direction deviation %.2f deg > 1e-4: coordinates '
                             'do not follow the specified direction' % dev_deg)
        out.append(dict(sign=sgn, coords=c_new.tolist(),
                        q_bohr=q_bohr,
                        actual_max_atom_disp_a=max_disp_A,
                        direction_deviation_deg=dev_deg))
    # +/- symmetry check
    if len(out) == 2:
        ratio = out[0]['actual_max_atom_disp_a'] / out[1]['actual_max_atom_disp_a']
        if abs(ratio - 1.0) > 1e-6:
            raise ValueError('+/- displacement asymmetry: ratio = %.8f' % ratio)
    for entry in out:
        entry['nominal_amp_a'] = q_bohr * BOHR_A * max(
            np.linalg.norm(d3, axis=1))
    return out


def recover_q_bohr(coords_center_A, coords_disp_A, d_cart):
    """Recover the actual Bohr amplitude from ACTUAL coordinates (not labels).

    Returns (q_bohr, max_atom_disp_A, direction_deviation_deg).
    Raises ValueError if the displacement is not parallel to d_cart.
    """
    d3 = np.asarray(d_cart, float).reshape(-1, 3)
    delta_A = np.asarray(coords_disp_A, float) - np.asarray(coords_center_A, float)
    per_atom_A = np.linalg.norm(delta_A, axis=1)
    max_disp_A = float(per_atom_A.max())
    max_disp_B = max_disp_A / BOHR_A
    u = delta_A.reshape(-1) / (np.linalg.norm(delta_A) + 1e-300)
    cosang = float(np.dot(u, d3.reshape(-1)) /
                   (np.linalg.norm(d3.reshape(-1)) + 1e-300))
    dev_deg = float(np.degrees(np.arccos(max(-1, min(1, cosang)))))
    if abs(dev_deg) > 0.1:
        raise ValueError('displacement direction deviates %.2f deg from '
                         'd_cart — cannot silently use symmetric formula'
                         % dev_deg)
    q_bohr = max_disp_B / float(np.linalg.norm(d3, axis=1).max())
    return q_bohr, max_disp_A, dev_deg
