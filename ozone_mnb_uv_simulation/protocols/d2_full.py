#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Full-derivative Chai-Head-Gordon -D2 dispersion (Phase 2A, Part 1).

Goal: make -D2 a *consistent* part of the total potential, the nuclear
gradient, AND the frequency Hessian, so that a true "wB97X-D" geometry /
frequency run is possible inside this project (previously D2 entered only
as a post-optimisation single-point energy correction; see
notes/method_validation.md section 1).

Design constraints honoured:
  * dispersion formula IDENTICAL to run_baseline.chg_d2 (parameters copied
    verbatim; d2_energy() must reproduce chg_d2() to machine precision);
  * D2 depends only on nuclear coordinates -> the SCF density / Fock build
    is NOT modified;
  * no PySCF platform source is touched - D2 is injected by re-classing the
    mf / Gradients / Hessian objects into subclasses (attach_d2 below);
  * gradient is analytic; Hessian is a central finite difference of the
    analytic gradient (consistent by construction, symmetrised).
"""
import math
import numpy as np
from pyscf import gto, dft
from pyscf.solvent import smd
from pyscf.data import nist

# ---- parameters copied verbatim from run_baseline.chg_d2 (do NOT change here
# ---- without changing run_baseline.py in the same commit)
C6_TABLE = {'H': 0.14, 'C': 1.75, 'N': 1.23, 'O': 0.70, 'F': 0.75,
            'P': 7.84, 'S': 5.57, 'Cl': 5.07, 'Br': 12.47, 'I': 19.81}
RVDW_TABLE = {'H': 1.30, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'F': 1.47,
              'P': 1.80, 'S': 1.80, 'Cl': 1.75, 'Br': 1.85, 'I': 1.98}
CHG_S6, CHG_SR6, CHG_A, CHG_M = 1.0, 1.1, 6.0, 12

HA2JMOL = nist.HARTREE2J * nist.AVOGADRO   # J/mol per Hartree
BOHR_SI = nist.BOHR_SI                     # metres per Bohr
ANG_PER_BOHR = BOHR_SI / 1e-10             # Angstrom per Bohr

XC = 'wb97xd'          # libxc DFT part only (no -D2 inside libxc)
BASIS = 'def2-TZVP'
GRID_LEVEL = 5
SCF_CONV = 1e-11


def _atoms(mol):
    return [mol.atom_symbol(i) for i in range(mol.natm)]


def _coords_ang(mol, coords_bohr=None):
    if coords_bohr is None:
        return np.asarray(mol.atom_coords(unit='Angstrom'), dtype=float)
    return np.asarray(coords_bohr, dtype=float) * ANG_PER_BOHR


def _real_atoms(mol):
    """Index list of the REAL atoms (PySCF ghost atoms, e.g. 'X-O', carry zero
    nuclear charge).  Ghosts must not contribute dispersion: they are basis
    centres only (counterpoise calculations), so including them would create
    spurious -D2 atom pairs (JOB-2026-0905-009 requirement)."""
    return [i for i in range(mol.natm) if mol.atom_charge(i) != 0]


def _pairs(mol, r_ang):
    """Yield (i, j, c6[J*m^6], R0[m], R[m]) for all pairs of REAL atoms
    (ghost/basis-only centres are excluded)."""
    sym = _atoms(mol)
    real = _real_atoms(mol)
    for i in real:
        for j in real:
            if j <= i:
                continue
            c6 = math.sqrt(C6_TABLE[sym[i]] * C6_TABLE[sym[j]]) * 1e-54
            R0 = CHG_SR6 * (RVDW_TABLE[sym[i]] + RVDW_TABLE[sym[j]]) * 1e-10
            R = np.linalg.norm(r_ang[i] - r_ang[j]) * 1e-10
            yield i, j, c6, R0, R


def d2_energy(mol, coords_bohr=None):
    """-D2 dispersion energy in Hartree (identical to run_baseline.chg_d2)."""
    r = _coords_ang(mol, coords_bohr)
    e = 0.0
    for i, j, c6, R0, R in _pairs(mol, r):
        f = 1.0 / (1.0 + CHG_A * (R0 / R) ** CHG_M)
        e += -CHG_S6 * c6 / R ** 6 * f          # J/mol
    return e / HA2JMOL


def d2_grad(mol, coords_bohr=None):
    """Analytic -D2 nuclear gradient, Hartree/Bohr, shape (natm, 3).

    E_pair = -s6 * c6 * R^-6 * f(R),  f = 1 / (1 + a*(R0/R)^M)
    dE/dR  = -s6 * c6 * R^-7 * ( -6 f + M x / (1+x)^2 ),  x = a*(R0/R)^M
    """
    r = _coords_ang(mol, coords_bohr)
    n = mol.natm
    g_m_per_m = np.zeros((n, 3))               # J/(mol*m), w.r.t. metres
    for i, j, c6, R0, R in _pairs(mol, r):
        x = CHG_A * (R0 / R) ** CHG_M
        f = 1.0 / (1.0 + x)
        dEdR = -CHG_S6 * c6 / R ** 7 * (-6.0 * f + CHG_M * x / (1.0 + x) ** 2)
        u = (r[i] - r[j]) / (np.linalg.norm(r[i] - r[j]) + 1e-300)
        g_m_per_m[i] += dEdR * u
        g_m_per_m[j] -= dEdR * u
    # J/(mol*m) -> Hartree/Bohr:  (E/HA2JMOL) per (m/BOHR_SI) Bohr
    return g_m_per_m * (BOHR_SI / HA2JMOL)


def d2_hess(mol, coords_bohr=None, step=1e-3):
    """-D2 Hessian, Hartree/Bohr^2, layout (natm, natm, 3, 3) like PySCF.

    Central 2nd-order finite difference of the analytic d2_grad, i.e.
    guaranteed consistent with the gradient; symmetrised with the project
    convention transpose(1,0,3,2).
    """
    c0 = (np.asarray(mol.atom_coords(unit='Bohr'), dtype=float)
          if coords_bohr is None else np.asarray(coords_bohr, dtype=float))
    n = mol.natm
    h = np.zeros((n, n, 3, 3))
    for i in range(n):
        for a in range(3):
            cp = c0.copy(); cp[i, a] += step
            cm = c0.copy(); cm[i, a] -= step
            gp = d2_grad(mol, cp)
            gm = d2_grad(mol, cm)
            h[i, :, a, :] = (gp - gm) / (2.0 * step)
    return 0.5 * (h + h.transpose(1, 0, 3, 2))


def attach_d2_grad(g):
    """Re-class a Gradients object so its kernel() adds the analytic -D2
    gradient evaluated from self.mol AT CALL TIME (Part 1B v2)."""
    if getattr(g, '_has_full_d2', False):
        return g
    parent_cls = type(g)

    def kernel(self, *a, **k):
        return parent_cls.kernel(self, *a, **k) + d2_grad(self.mol)

    g.__class__ = type(parent_cls.__name__ + '_D2', (parent_cls,),
                       dict(kernel=kernel))
    g._has_full_d2 = True
    g._d2_parent_cls = parent_cls
    return g


def attach_d2_hess(hobj):
    """Re-class a Hessian object so its kernel() adds the -D2 Hessian
    evaluated from self.mol AT CALL TIME (Part 1B v2)."""
    if getattr(hobj, '_has_full_d2', False):
        return hobj
    parent_cls = type(hobj)

    def kernel(self, *a, **k):
        return parent_cls.kernel(self, *a, **k) + d2_hess(self.mol)

    hobj.__class__ = type(parent_cls.__name__ + '_D2', (parent_cls,),
                          dict(kernel=kernel))
    hobj._has_full_d2 = True
    hobj._d2_parent_cls = parent_cls
    return hobj


def attach_d2(mf):
    """Re-class an SCF object so -D2 enters energy, gradient AND Hessian
    consistently on every call path (v2, Part 1B fix).

    v2 design (replaces the v1 instance-attribute wrappers, which the
    production-path audit JOB-2026-0904-008 showed to be broken):
      * D2 enters the SCF driver EXACTLY ONCE through energy_tot:
        scf.hf.kernel computes every cycle energy (and the final return
        value / e_tot attribute) via the instance lookup mf.energy_tot,
        so overriding energy_tot makes kernel() return E(wB97X) + E(-D2)
        with no separate kernel wrapper (a kernel wrapper on top of the
        energy_tot route would double-count).
      * every override evaluates the D2 term from self.mol at CALL TIME,
        so scanner copies (SCF_Scanner / SCF_GradScanner copy __dict__ and
        rebind self.mol per geometry step, berny_solver mutates mol.copy())
        automatically use the current coordinates.  The v1 closures pinned
        bound methods of the original mf object and its attach-time mol,
        which broke both requirements.
      * nuc_grad_method / Gradients / Hessian factories return objects
        re-classed by attach_d2_grad / attach_d2_hess.
      * mf._d2_parent_cls stores the pre-attach class: the verification
        bypass for "wrapped - unwrapped == analytic D2 term" checks
        (the old type(mf).method(mf) bypass is meaningless once
        type(mf) IS the D2 subclass).
    """
    if getattr(mf, '_has_full_d2', False):
        return mf
    parent_cls = type(mf)

    def energy_tot(self, dm=None, h1e=None, vhf=None):
        e = parent_cls.energy_tot(self, dm, h1e, vhf)
        e_d2 = d2_energy(self.mol)
        self.scf_summary['d2_dispersion'] = e_d2
        return e + e_d2

    def nuc_grad_method(self):
        return attach_d2_grad(parent_cls.nuc_grad_method(self))

    def Hessian(self):
        return attach_d2_hess(parent_cls.Hessian(self))

    def undo_solvent(self):
        """Solvent-attached undo_solvent() with ALL project wrapper classes
        stripped down to the clean vacuum RKS/UKS.

        PySCF's _Solvation.undo_solvent() rebuilds the vacuum object by
        dropping ONLY the solvent classes from the MRO, which leaves the
        project D2 (and GR) subclasses in place.  Those subclasses' method
        closures reference the SMD parent class, so PySCF's
        make_grad_object -> base_method.undo_solvent().Gradients() path
        re-enters the solvent machinery and trips
        grad.pcm.make_grad_object's `assert isinstance(base_method,
        _Solvation)` (JOB-2026-0906-006 Phase B).  Worse, leaving a
        D2-wrapped vac_grad would DOUBLE-COUNT -D2, because the outer
        attach_d2_grad kernel wrapper already adds it exactly once.
        Stripping every project-added class down to the clean vacuum RKS
        loses nothing (-D2 enters through the outer wrapper) and keeps the
        accounting exact.
        """
        obj = parent_cls.undo_solvent(self)
        # strip project-added wrapper classes (SMD*/D2/GR) to clean PySCF
        while any(t in type(obj).__name__
                  for t in ('SMD', '_D2', '_GR')):
            mro = type(obj).__mro__
            if len(mro) < 2:
                break
            obj.__class__ = mro[1]
        return obj

    extras = dict(energy_tot=energy_tot,
                  nuc_grad_method=nuc_grad_method,
                  Gradients=nuc_grad_method,
                  Hessian=Hessian)
    if hasattr(parent_cls, 'undo_solvent'):
        extras['undo_solvent'] = undo_solvent
    mf.__class__ = type(parent_cls.__name__ + '_D2', (parent_cls,), extras)
    mf._has_full_d2 = True
    mf._d2_parent_cls = parent_cls
    return mf


def make_mf_d2(mol, solvent=None):
    """Build RKS/UKS (+ optional SMD) at wB97X/def2-TZVP with full -D2."""
    base = dft.UKS(mol) if mol.spin else dft.RKS(mol)
    mf = (smd.smd_for_scf(base, solvent_obj=smd.SMD(mol, solvent=solvent))
          if solvent else base)
    mf.xc = XC
    mf.conv_tol = SCF_CONV
    mf.max_cycle = 200
    mf.grids.level = GRID_LEVEL
    return attach_d2(mf)
