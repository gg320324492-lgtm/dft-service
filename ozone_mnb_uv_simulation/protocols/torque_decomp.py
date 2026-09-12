#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fixed torque/force decomposition and rigid-rotation helpers
(JOB-2026-0906-008 -> 009 unit repair).

UNIT CONVENTION (repairs the 008 defect where the cross product was taken
with Angstrom coordinates but labelled Eh):
  * all cross products use BOHR coordinates;
  * T_g = sum_i (r_i - r_c) x g_i   with r in Bohr, g in Eh/Bohr
    -> T_g has units Eh (it is the gradient of the energy with respect to a
       rotation angle about the axis, i.e. dE/dtheta vector, Eh/rad);
  * the PHYSICAL torque on the molecule is -T_g (force = -gradient);
    the two signs are never mixed.
"""
import numpy as np

BOHR_A = 0.52917721092


def to_bohr(coords, unit='Angstrom'):
    c = np.asarray(coords, float)
    if unit == 'Angstrom':
        return c / BOHR_A
    elif unit == 'Bohr':
        return c.copy()
    raise ValueError('unknown unit: %r' % unit)


def com_bohr(coords_bohr, masses):
    m = np.asarray(masses, float)
    return np.einsum('z,zx->x', m, coords_bohr) / m.sum()


def decompose_force(g, coords, masses, unit='Angstrom', center='com'):
    """Decompose a nuclear gradient into translation / rotation / internal
    parts in UNWEIGHTED Cartesian force space.

    Parameters
    ----------
    g       : (natm, 3) gradient, Eh/Bohr (rows = atoms).
    coords  : (natm, 3) nuclear coordinates in `unit`.
    masses  : (natm,) masses (amu) -- only used to define the COM center.
    center  : 'com' or an explicit 3-vector in Bohr.

    Returns a dict with (all vector norms in Eh/Bohr; T_g in Eh):
      net_force      : sum_i g_i                     (Eh/Bohr)
      T_g            : sum_i (r_i - r_c) x g_i       (Eh, Bohr coords)
      physical_torque: -T_g                          (Eh)
      g_trans/g_rot/g_int vectors and norms, external rank, center.
    The external basis is unweighted (translations = unit vectors on every
    atom; rotations = e_a x (r_i - r_c) with r in BOHR), QR-orthonormalised.
    """
    r = to_bohr(coords, unit)
    g = np.asarray(g, float).reshape(-1, 3)
    natm = r.shape[0]
    if isinstance(center, str) and center == 'com':
        rc = com_bohr(r, masses)
    else:
        rc = np.asarray(center, float)
    rb = r - rc

    T = []
    for a in range(3):
        v = np.zeros((natm, 3)); v[:, a] = 1.0
        T.append(v.reshape(-1))
    R = []
    for a in range(3):
        e = np.zeros(3); e[a] = 1.0
        v = np.cross(rb, np.tile(e, (natm, 1)))     # rb in BOHR
        R.append(v.reshape(-1))
    M = np.vstack(T + R)                            # (6, 18) Bohr-scaled
    q, rr = np.linalg.qr(M.T)
    rank = int(np.sum(np.abs(np.diag(rr)) > 1e-7))
    q = q[:, :rank]

    gv = g.reshape(-1)
    g_ext = q.T.dot(gv)
    g_t_vec = q[:, :3].dot(g_ext[:3])
    g_r_vec = (q[:, 3:rank].dot(g_ext[3:rank])
               if rank > 3 else np.zeros_like(gv))
    g_int_vec = gv - q.dot(g_ext)

    T_g = np.cross(rb, g).sum(axis=0)               # Bohr x Eh/Bohr = Eh
    net_force = g.sum(axis=0)
    return dict(
        unit_used='coords converted to Bohr; g in Eh/Bohr',
        center_bohr=rc.tolist(),
        external_rank=rank,
        net_force=net_force.tolist(),
        net_force_norm=float(np.linalg.norm(net_force)),
        T_g=T_g.tolist(),
        T_g_norm=float(np.linalg.norm(T_g)),
        physical_torque=(-T_g).tolist(),
        g_trans_vec=g_t_vec.tolist(),
        g_trans_norm=float(np.linalg.norm(g_t_vec)),
        g_rot_vec=g_r_vec.tolist(),
        g_rot_norm=float(np.linalg.norm(g_r_vec)),
        g_int_vec=g_int_vec.tolist(),
        g_int_norm=float(np.linalg.norm(g_int_vec)),
        g_norm=float(np.linalg.norm(gv)),
        int_norm_fraction=float(np.linalg.norm(g_int_vec)
                                / (np.linalg.norm(gv) + 1e-300)),
        int_sq_fraction=float(np.linalg.norm(g_int_vec) ** 2
                              / (np.linalg.norm(gv) ** 2 + 1e-300)),
        note='"norm fraction" and "squared-norm fraction" are DIFFERENT '
             'quantities and are never added as percentages')


def rodrigues(axis, theta_rad):
    """Exact rotation matrix (column-vector convention: x' = R x)."""
    k = np.asarray(axis, float)
    k = k / np.linalg.norm(k)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    th = float(theta_rad)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def rotate_coords(coords_bohr, R, center_bohr):
    """Rigidly rotate coords (Bohr) about an axis through center_bohr."""
    c = np.asarray(coords_bohr, float)
    return (R @ (c - center_bohr).T).T + center_bohr


def rotate_vectors(v, R):
    """Rotate per-atom vector field (e.g. a gradient) into the rotated
    frame: v' = R v (per atom)."""
    v = np.asarray(v, float).reshape(-1, 3)
    return (R @ v.T).T
