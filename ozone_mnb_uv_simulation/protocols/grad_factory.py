#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified gradient construction entry point for this project (Phase B/C).

Guarantees, by construction, exactly the two things the verification batch
requires:

1. The gradient object actually handed to the optimizer / scanner has
   ``grid_response = True`` (full XC grid response included).
2. The project -D2 dispersion enters the gradient EXACTLY ONCE, through the
   D2-attached ``nuc_grad_method`` built by :mod:`d2_full` (no edits to the
   dispersion formula, functional, basis or PySCF itself).

The two public helpers:

* :func:`make_grad`        -- return the D2-attached gradient with grid_response
                             set; the scanner used in Phase B is built from it.
* :func:`make_mf_d2_gr`    -- build the production mf whose ``nuc_grad_method``
                             *always* returns a gradient with grid_response set.
                             This is the object handed to the geometry
                             optimizer in Phase C, so the gradient the optimizer
                             receives is precisely the verified one.
* :func:`dft_only_grad`    -- independent benchmark: the un-wrapped DFT (wB97X)
                             gradient WITHOUT project D2, with grid_response set,
                             so the verification can show
                             ``wrapped - dft_only == d2_grad(mol)`` (D2 once).
"""
from pyscf import dft
from d2_full import make_mf_d2, d2_grad, d2_energy

# production method fingerprint (matches the derivative batches)
GRID_LEVEL = 7
SCF_TOL = 1e-12
SCF_TOL_GRAD = 1e-9
BASIS = 'def2-TZVP'
XC = 'wb97xd'


def make_grad(mf, grid_response=True):
    """Return the D2-attached gradient object with ``grid_response`` set.

    D2 enters exactly once: :func:`d2_full.make_mf_d2` already re-classes the
    mf so its ``nuc_grad_method`` returns a D2-attached Gradients object; this
    call applies that path and pins the response flag.
    """
    g = mf.nuc_grad_method()
    g.grid_response = grid_response
    return g


def make_mf_d2_gr(mol, solvent=None, grid_response=True,
                  grid_level=GRID_LEVEL, scf_tol=SCF_TOL,
                  scf_tol_grad=SCF_TOL_GRAD):
    """Build the production mf (D2-attached) whose ``nuc_grad_method`` always
    returns a gradient with ``grid_response`` set.

    This is the object handed to the geometry optimizer so the gradient it
    receives is the verified grid_response=True gradient.  The D2 term is
    still injected exactly once (via ``parent.nuc_grad_method`` -> the
    D2-attached path in :mod:`d2_full`).
    """
    mf = make_mf_d2(mol, solvent=solvent)
    mf.grids.level = grid_level
    mf.conv_tol = scf_tol
    mf.conv_tol_grad = scf_tol_grad
    parent = type(mf)

    def nuc_grad_method(self):
        g = parent.nuc_grad_method(self)
        g.grid_response = grid_response
        return g

    # keep the (rarely used) Gradients alias consistent with nuc_grad_method
    def Gradients(self):
        return nuc_grad_method(self)

    extras = dict(nuc_grad_method=nuc_grad_method, Gradients=Gradients)
    if hasattr(parent, 'undo_solvent'):
        extras['undo_solvent'] = parent.undo_solvent
    mf.__class__ = type(parent.__name__ + '_GR', (parent,), extras)
    # NOTE: do NOT re-assign mf._d2_parent_cls here.  make_mf_d2 (attach_d2)
    # already set it as an *instance* attribute pointing to the ORIGINAL RKS
    # (the un-wrapped DFT part).  Re-pointing it at RKS_D2 would make the
    # "independent" benchmark return a D2-containing energy/gradient and
    # defeat the D2-once check.  The inherited instance attribute is correct.
    mf._has_full_d2 = True
    return mf


def dft_only_grad(mf, grid_response=True):
    """Independent benchmark gradient: un-wrapped DFT (wB97X) WITHOUT project
    D2, with grid_response set.

    Used to verify D2 is counted exactly once:
        wrapped_grad - dft_only_grad == d2_grad(mol)
    """
    g = mf._d2_parent_cls.nuc_grad_method(mf)
    g.grid_response = grid_response
    return g


def _num(x):
    return float(x) if x is not None else None


def method_fingerprint(mf):
    """Record the method fingerprint from the ACTUAL object state.

    Solvent identification reads the LIVE solvent attachment
    (``mf.with_solvent``) and its configured attributes -- the solvent name,
    the SMD method, the electrostatic surface lebedev order and the
    discretisation scheme.  It never guesses from a parent-class name, so a
    gas-phase and an SMD object (or two SMD objects with different surface
    gears) can never produce the same full fingerprint.
    """
    ws = getattr(mf, 'with_solvent', None)
    if ws is None:
        solvent = dict(model='gas')
    else:
        order = getattr(ws, 'lebedev_order', None)
        eps = getattr(ws, 'eps', None)
        solvent = dict(
            model='smd',
            solvent=str(getattr(ws, 'solvent', None)),
            smd_method=str(getattr(ws, 'method', None)),
            lebedev_order=(int(order) if order is not None else None),
            surface_discretization=str(
                getattr(ws, 'surface_discretization_method', None)),
            eps=(float(eps) if eps is not None else None),
            built=bool(getattr(ws, 'surface', None) is not None))
    return dict(
        xc=XC, basis=BASIS, grid_level=int(mf.grids.level),
        scf_tol=_num(mf.conv_tol), scf_tol_grad=_num(mf.conv_tol_grad),
        solvent=solvent,
        d2_attached=bool(getattr(mf, '_has_full_d2', False)),
        d2_params=dict(s6=1.0, sr6=1.1, a=6.0, m=12),
        grid_response_default=True,
        note='solvent block read from the live with_solvent attachment '
             '(no class-name guessing); lebedev_order included so different '
             'surface gears never share a fingerprint')
