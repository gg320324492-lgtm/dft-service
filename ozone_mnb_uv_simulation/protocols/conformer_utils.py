#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Shared helpers for the Phase-2A O3.H2O conformer pipeline (JOB-2026-0905-009).

Covers:
  * loading Stage-1 gas optimisations;
  * Kabsch RMSD between conformers (with C2 ozone terminal-oxygen permutation);
  * H-bond / oxygen-contact topology detection and signature;
  * analytic + D2 Hessian assembly and numerical-from-gradient Hessian (for SMD);
  * thermochemistry wrappers reusing run_baseline.thermochemistry;
  * Boys-Bernardi counterpoise (ghost) energy helper (D2 excludes ghosts).

All energies are returned in Hartree unless a kcal/mol field is named.
"""
import os
import sys
import json
import math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import d2_full
import run_baseline as rb

HARTREE2KCAL = rb.HARTREE2KCAL

ROOT = os.path.dirname(HERE)
GAS_OPT = os.path.join(ROOT, 'run_artifacts', '01_pure_water_o3_h2o', 'gas_opt')

# atom order in every O3.H2O structure handled here
O3_IDX = [0, 1, 2]          # central, terminal(+x), terminal(-x)
OW_IDX = 3                  # water oxygen
H1_IDX, H2_IDX = 4, 5       # water hydrogens

HBOND_DIST = 2.5            # H ... O  (Angstrom) H-bond donor cutoff
HBOND_ANGLE = 120.0         # O_w - H ... O  (degrees) minimum angle
OW_CONTACT_DIST = 3.3       # O_w ... O_ozone (Angstrom) contact cutoff


# --------------------------------------------------------------------- loading
def load_gas_opt(art_dir=GAS_OPT):
    """Return dict id -> Stage-1 record loaded from art_dir/<id>.json."""
    recs = {}
    if not os.path.isdir(art_dir):
        return recs
    for fn in sorted(os.listdir(art_dir)):
        if not fn.endswith('.json') or fn == 'stage1_summary.json':
            continue
        tid = fn[:-5]
        with open(os.path.join(art_dir, fn)) as fh:
            recs[tid] = json.load(fh)
    return recs


def coords_of(rec, key='optimised_coords_angstrom'):
    return np.asarray(rec[key], dtype=float)


# ------------------------------------------------------------------ geometry
def kabsch_rotate(P, Q):
    """Optimal rotation R (3x3, det +1) rotating P onto Q via SVD."""
    Pc = P - P.mean(0)
    Qc = Q - Q.mean(0)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    return R


def best_rmsd(coords_a, coords_b):
    """Minimum all-atom RMSD between two O3.H2O conformers (Angstrom).

    Revised (JOB-009 acceptance revision #1):
      * rigid-body fit anchored on the O3 frame centroid (idx 0,1,2) -- the
        previous version computed the rotation from the O3 frames but
        centred/translated on the FULL-molecule centroid, which is
        inconsistent whenever the water sits at a different relative
        position;
      * equivalence permutations cover the O3 terminal oxygens (1<->2) AND
        the two water hydrogens (4<->5): 4 label assignments, minimum kept.
    """
    a = np.asarray(coords_a, dtype=float)
    b = np.asarray(coords_b, dtype=float)
    o3 = O3_IDX

    def rmsd_for(P, Q, perm_b):
        Qp = Q.copy()
        if perm_b is not None:
            for i, j in perm_b:            # equivalent-atom relabelling of reference
                Qp[[i, j]] = Qp[[j, i]]
        # centre both on the O3-frame centroid, rotate P's O3 frame onto Q's
        cP = P[o3].mean(0)
        cQ = Qp[o3].mean(0)
        R = kabsch_rotate(P[o3] - cP, Qp[o3] - cQ)
        Pa = (P - cP) @ R.T + cQ
        return float(np.sqrt(((Pa - Qp) ** 2).sum() / len(P)))

    perms = [None, [(1, 2)], [(4, 5)], [(1, 2), (4, 5)]]
    return min(rmsd_for(a, b, p) for p in perms)


# --------------------------------------------------------- topology detection
def hbond_topology(coords):
    """Detect H-bonds (water H -> ozone O) and oxygen contacts (O_w ... O_ozone).

    H-bond angle criterion (revised, JOB-009 acceptance revision #1): the
    angle is measured AT THE HYDROGEN between the H->O_w vector and the
    H->O(acceptor) vector, i.e. the conventional O_w-H...O angle.  The
    previous implementation used the O_w->H vector instead of H->O_w, which
    measures the SUPPLEMENT of the H-bond angle and therefore REJECTED good
    nearly-linear H-bonds (verified on real geometries: c02 has a 179.9 deg
    H-bond, d(H..O)=2.286 A, that the old code classified as "none").

    Returns dict:
      hbonds      : list of (water_H_idx, ozone_O_idx)
      ow_contacts : list of ozone_O_idx in contact with the water oxygen
      description : human readable string
    """
    c = np.asarray(coords, dtype=float)
    hbonds = []
    for h in (H1_IDX, H2_IDX):
        for o in O3_IDX:
            dh = np.linalg.norm(c[h] - c[o])
            if dh > HBOND_DIST:
                continue
            # angle O_w - H ... O measured AT H: vectors H->O_w and H->O
            v1 = c[OW_IDX] - c[h]
            v2 = c[o] - c[h]
            cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
            ang = math.degrees(math.acos(max(-1, min(1, cosang))))
            if ang >= HBOND_ANGLE:
                hbonds.append((h, o))
    ow_contacts = []
    for o in O3_IDX:
        if np.linalg.norm(c[OW_IDX] - c[o]) <= OW_CONTACT_DIST:
            ow_contacts.append(o)
    # human readable
    names = {0: 'central', 1: 'term(+x)', 2: 'term(-x)'}
    hb_txt = ', '.join("H%d->O[%s]" % (h - 3, names[o]) for h, o in hbonds) or 'none'
    oc_txt = ', '.join(names[o] for o in ow_contacts) or 'none'
    desc = 'H-bonds: %s | O_w contacts: %s' % (hb_txt, oc_txt)
    return dict(hbonds=hbonds, ow_contacts=ow_contacts, description=desc)


def topology_signature(coords):
    """Canonical, permutation-insensitive topology signature.

    Revised: collapses BOTH equivalence classes -- O3 terminal oxygens
    (term(+x)/term(-x) -> 'term') AND the two water hydrogens (H1/H2 -> 'H')
    -- so structures differing only by equivalent-atom relabelling share one
    signature.
    """
    t = hbond_topology(coords)
    norm_hb = []
    for h, o in t['hbonds']:
        no = 'central' if o == 0 else 'term'       # collapse terminal-oxygen labels
        nh = 'H' if h in (H1_IDX, H2_IDX) else h   # collapse water-H labels
        norm_hb.append((nh, no))
    norm_oc = sorted(set('central' if o == 0 else 'term' for o in t['ow_contacts']))
    return (tuple(sorted(norm_hb)), tuple(norm_oc))


# ----------------------------------------------------------- Hessian assembly
def gas_full_hessian(mol):
    """Analytic DFT Hessian (PySCF) + analytic -D2 Hessian (project module).

    Returns the (natm, natm, 3, 3) Hessian in Hartree/Bohr^2, symmetrised.

    NOTE: attach_d2_hess already injects the -D2 Hessian into the analytic
    DFT Hessian kernel(), so mf.Hessian().kernel() already = DFT + D2.  We
    must NOT add d2_full.d2_hess(mol) again here (would double-count).
    """
    mf = d2_full.make_mf_d2(mol)
    mf.kernel()
    h = mf.Hessian().kernel()                 # analytic DFT Hessian + D2
    return 0.5 * (h + h.transpose(1, 0, 3, 2))


def smd_num_hessian(mol, solvent='water', step=0.01):
    """Numerical Hessian from analytic gradients (SMD or plain) in Hartree/Bohr^2.

    Robust for solvated systems where an analytic solvent Hessian is not
    guaranteed; every gradient evaluation carries DFT + full -D2 (via
    attach_d2_grad) and, for SMD, the PySCF solvent gradient.
    """
    c0 = mol.atom_coords(unit='Bohr')
    n = mol.natm
    g = np.zeros((n, n, 3, 3))
    for i in range(n):
        for k in range(3):
            for s in (+1, -1):
                c = c0.copy()
                c[i, k] += s * step
                m = mol.copy()
                m.set_geom_(c, unit='Bohr')
                m.build()
                mf = d2_full.make_mf_d2(m, solvent=solvent)
                mf.kernel()
                gg = mf.nuc_grad_method().kernel().reshape(n, 3)
                g[i, :, k] += s * gg / (2.0 * step)
    return 0.5 * (g + g.transpose(1, 0, 3, 2))


def analyse_vibrations(mol, h, e_total, sigma=1):
    """Full 3N vibrational analysis from a Hessian h (Hartree/Bohr^2).

    Returns dict with:
      modes         : sorted harmonic frequencies (cm^-1), incl. tiny T/R modes
      n_imag        : number of imaginary frequencies (< -1 cm^-1)
      freq_error    : PySCF's own count of imaginary modes (cross-check)
      zpe_hartree   : zero-point energy (Hartree)
      thermo        : run_baseline.thermochemistry dict (J/mol), electronic
                      datum = e_total (Hartree); sigma = rotational symmetry
                      number (2 for the C2v monomers H2O/O3, 1 for the
                      asymmetric complex)

    NOTE (JOB-2026-0905-009): pyscf.hessian.thermo.harmonic_analysis returns
    imaginary frequencies as COMPLEX numbers by default (imaginary_freq=True);
    casting that array to float silently maps every imaginary mode to 0.0 and
    fakes n_imag=0 (ComplexWarning).  We therefore call it with
    imaginary_freq=False so imaginary modes come back as NEGATIVE reals, and
    keep PySCF's freq_error as an independent cross-check of n_imag.
    """
    h = 0.5 * (h + h.transpose(1, 0, 3, 2))
    freq_raw = rb.vib_freqs(mol, h)
    nfreq = 3 * mol.natm - (5 if rb._is_linear(mol) else 6)
    ha = rb.pyscf_thermo.harmonic_analysis(mol, h, imaginary_freq=False)
    modes = np.sort(np.asarray(ha['freq_wavenumber'], dtype=float))
    n_imag = int(np.sum(modes < -1e-6))
    freq_error = int(ha.get('freq_error', -1))
    th = rb.thermochemistry(mol, modes, float(e_total), 1, sigma)
    # ZPE taken from the thermochemistry implementation (0.5*R*sum(theta_v))
    # to keep a single source of truth; the previous inline constant carried a
    # spurious Boltzmann factor (dimensionally wrong -> zpe exactly 0).
    zpe_hartree = th['ZPE'] / rb.HA2JMOL
    return dict(freq_raw=freq_raw, n_vib=nfreq, modes=modes, n_imag=n_imag,
                freq_error=freq_error, zpe_jmol=th['ZPE'],
                zpe_hartree=zpe_hartree, thermo=th)


# ----------------------------------------------------- counterpoise (CP) calc
def monomer_energy_in_complex_basis(mol_complex_coords, present, ghost, solvent=None):
    """Energy of a monomer (atoms `present`) evaluated at the complex geometry
    with the partner atoms `ghost` present only as basis functions (ghosts).

    present / ghost : lists of atom indices into the complex-coordinate array.
    Returns the PySCF RKS total energy (Hartree) with full -D2 (ghosts excluded
    from the D2 sum by d2_full._real_atoms).
    """
    elem = ['O', 'O', 'O', 'O', 'H', 'H']
    lines = []
    for i in present:
        x, y, z = mol_complex_coords[i]
        lines.append("%s %.10f %.10f %.10f" % (elem[i], x, y, z))
    for i in ghost:
        x, y, z = mol_complex_coords[i]
        lines.append("X-%s %.10f %.10f %.10f" % (elem[i], x, y, z))
    mol = gto_mol_from(lines, verbose=0)
    mf = d2_full.make_mf_d2(mol, solvent=solvent)
    return float(mf.kernel())


def gto_mol_from(atom_lines, verbose=0):
    from pyscf import gto
    return gto.M(atom="; ".join(atom_lines), basis=d2_full.BASIS,
                 charge=0, spin=0, verbose=verbose)


def mol_from_coords(coords, syms, basis=d2_full.BASIS, verbose=0):
    """Build a PySCF Mole from a coordinate array and symbol list."""
    from pyscf import gto
    s = "; ".join("%s %.10f %.10f %.10f" % (s_, x, y, z)
                  for s_, (x, y, z) in zip(syms, coords))
    return gto.M(atom=s, basis=basis, charge=0, spin=0, verbose=verbose)
