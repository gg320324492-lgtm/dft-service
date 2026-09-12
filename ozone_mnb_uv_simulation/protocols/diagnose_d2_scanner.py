#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Part 1B PRE-FIX diagnosis (JOB-2026-0904-008).

Runs the PRODUCTION optimizer construction path -- exactly the way
pyscf.geomopt.berny_solver.kernel builds it:

    mol_work = method.mol.copy()
    g_scanner = method.nuc_grad_method().as_scanner()
    # per cycle: mol_work.set_geom_(...);  energy, gradients = g_scanner(mol_work)

on the CURRENT attach_d2 (instance-attribute wrappers) and demonstrates the
two defects expected from the platform-source reading:

  D1) scanner coordinate freeze: the wrapped kernel/grad closures capture the
      attach-time mol object and bound methods of the ORIGINAL mf; scanner
      copies (SCF_Scanner / SCF_GradScanner copy __dict__) therefore evaluate
      BOTH the DFT part and the D2 part at the attach-time geometry, so the
      optimizer would see a constant potential.
  D2) kernel() return value counts D2 TWICE: hf.kernel computes every cycle
      energy via the instance attribute mf.energy_tot (already wrapped, +1x D2)
      and the kernel-level wrapper adds another 1x D2.  Part 1 check 4 compared
      (kernel return - e_tot attribute) which cancels these two opposite
      errors exactly -- a false pass.

Gas phase, 6-31G for speed: both defects are implementation-level and
basis-independent.  Writes
run_artifacts/01_pure_water_o3_h2o/d2_pre_fix_diagnosis.json
"""
import os, sys, json, math
import numpy as np
from pyscf import gto, dft

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d2_full

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, 'run_artifacts', '01_pure_water_o3_h2o')
BASIS = '6-31G'          # diagnosis-only basis (speed); defects are basis-independent
SHIFT_FAR = 5.0          # Angstrom, water displacement along x for the far geometry


def build(water_x):
    """O3 (C2v, exp geometry) + H2O; water translated by water_x along x."""
    r, ang = 1.2717, 117.79
    t = math.radians(ang / 2.0)
    atoms = [('O', (0.0, 0.0, 0.0)),
             ('O', (r * math.sin(t), r * math.cos(t), 0.0)),
             ('O', (-r * math.sin(t), r * math.cos(t), 0.0)),
             ('O', (water_x, 2.60, 1.05)),
             ('H', (water_x + 0.76, 2.95, 0.55)),
             ('H', (water_x - 0.76, 2.95, 0.55))]
    return gto.M(atom=atoms, basis=BASIS, charge=0, spin=0, verbose=0)


def make(mol):
    mf = dft.RKS(mol)
    mf.xc = 'wb97xd'
    mf.conv_tol = 1e-11
    mf.max_cycle = 200
    mf.grids.level = 4
    return d2_full.attach_d2(mf)


def main():
    res = dict(backend='PySCF / WSL2', basis=BASIS,
               implementation='attach_d2 v1 (instance-attribute wrappers)',
               construction_path='berny_solver: mol.copy() + nuc_grad_method().as_scanner()')

    mol_R1 = build(0.0)
    mol_R2 = build(SHIFT_FAR)
    e_d2_R1 = d2_full.d2_energy(mol_R1)
    e_d2_R2 = d2_full.d2_energy(mol_R2)
    res['d2_ref'] = dict(e_d2_R1=e_d2_R1, e_d2_R2=e_d2_R2,
                         d2_delta=float(abs(e_d2_R1 - e_d2_R2)))

    # ---------- D1: scanner coordinate freeze ----------
    mf1 = make(mol_R1)
    g_scanner = mf1.nuc_grad_method().as_scanner()   # berny_solver construction
    e1, g1 = g_scanner(mol_R1)                       # step at R1

    mol_work = mol_R1.copy()                         # berny_solver: mol = method.mol.copy()
    mol_work.set_geom_(np.asarray(mol_R2.atom_coords(unit='Angstrom')), unit='Angstrom')
    e2s, g2s = g_scanner(mol_work)                   # step at R2 via the SAME scanner

    mf2 = make(mol_R2)
    e2f = float(mf2.kernel())
    g2f = np.asarray(mf2.nuc_grad_method().kernel())

    res['scanner_freeze'] = dict(
        e_scanner_at_R1=float(e1),
        e_scanner_at_R2_moved=float(e2s),
        e_fresh_object_at_R2=e2f,
        e_scanner_R2_minus_R1=float(e2s - e1),
        max_g_scanner_R2_minus_R1=float(np.abs(np.asarray(g2s) - np.asarray(g1)).max()),
        e_scanner_R2_minus_fresh_R2=float(e2s - e2f),
        max_g_scanner_R2_minus_fresh_R2=float(np.abs(np.asarray(g2s) - g2f).max()),
        note=('scanner at the moved geometry must match the fresh object; '
              'equality with the R1 step proves the coordinate freeze'))

    # ---------- D2: kernel() return counts D2 twice ----------
    mf3 = make(mol_R1)
    e_kernel = float(mf3.kernel())                       # wrapped return value
    e_attr = float(mf3.e_tot)                            # attribute set inside hf.kernel
    e_bare = float(type(mf3).energy_tot(mf3))            # class bypass: pure DFT, same SCF
    ratio_kernel = (e_kernel - e_bare) / e_d2_R1
    ratio_attr = (e_attr - e_bare) / e_d2_R1
    res['kernel_double_count'] = dict(
        e_kernel=e_kernel, e_tot_attribute=e_attr, e_bare_class_bypass=e_bare,
        e_d2=e_d2_R1,
        kernel_minus_bare_over_d2=float(ratio_kernel),
        etot_attr_minus_bare_over_d2=float(ratio_attr),
        residual_vs_1x=float(abs((e_kernel - e_bare) - e_d2_R1)),
        residual_vs_2x=float(abs((e_kernel - e_bare) - 2.0 * e_d2_R1)),
        note=('ratio must be 1.0 for a correct implementation; '
              'ratio ~ 2.0 = double count; the Part 1 check compared '
              '(kernel return - e_tot attr) which equals exactly 1x D2 '
              'because the two opposite errors cancel'))

    # ---------- verdicts ----------
    res['verdict'] = dict(
        D1_scanner_frozen=bool(abs(e2s - e1) < 1e-7
                               and float(np.abs(np.asarray(g2s) - np.asarray(g1)).max()) < 1e-7),
        D1_propagation_broken=bool(abs(e2s - e2f) > 1e-6
                                   or float(np.abs(np.asarray(g2s) - g2f).max()) > 1e-6),
        D2_kernel_double_count=bool(abs(ratio_kernel - 2.0) < 1e-6),
        D2_attribute_single=bool(abs(ratio_attr - 1.0) < 1e-6),
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, 'd2_pre_fix_diagnosis.json')
    with open(out, 'w') as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps(res, indent=2))
    print('DIAGNOSIS_JSON = %s' % out)
    print('D1_frozen=%s  D1_broken=%s  D2_double=%s  D2_attr_single=%s' % (
        res['verdict']['D1_scanner_frozen'], res['verdict']['D1_propagation_broken'],
        res['verdict']['D2_kernel_double_count'], res['verdict']['D2_attribute_single']))


if __name__ == '__main__':
    main()
