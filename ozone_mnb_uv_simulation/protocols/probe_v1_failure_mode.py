#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Probe the v1 scanner failure mode (JOB-2026-0904-008 evidence detail)."""
import os, sys, json
import numpy as np
from pyscf import gto, dft

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d2_full
from diagnose_d2_scanner import build, make, SHIFT_FAR

mol_R1 = build(0.0)
mol_R2 = build(SHIFT_FAR)
mf1 = make(mol_R1)
g_scanner = mf1.nuc_grad_method().as_scanner()
e1, g1 = g_scanner(mol_R1)

scf_scanner = g_scanner.base
info = dict(
    scanner_class=type(g_scanner).__name__,
    scanner_mro=[c.__name__ for c in type(g_scanner).__mro__][:4],
    grad_scanner_has_kernel_instattr='kernel' in g_scanner.__dict__,
    scf_scanner_class=type(scf_scanner).__name__,
    scf_scanner_has_kernel_instattr='kernel' in scf_scanner.__dict__,
    scf_scanner_has_energy_tot_instattr='energy_tot' in scf_scanner.__dict__,
    e_nuc_R1=float(mol_R1.energy_nuc()),
    e_nuc_R2=float(mol_R2.energy_nuc()),
)

mol_work = mol_R1.copy()
mol_work.set_geom_(np.asarray(mol_R2.atom_coords(unit='Angstrom')), unit='Angstrom')
e2s, g2s = g_scanner(mol_work)
info.update(
    e_scanner_R2=float(e2s),
    scf_scanner_converged_after_R2=bool(scf_scanner.converged),
    scf_scanner_e_tot=float(scf_scanner.e_tot),
    scf_scanner_mol_is_mol_work=scf_scanner.mol is mol_work,
    mf1_mol_is_mol_R1=mf1.mol is mol_R1,
    mf1_e_tot=float(mf1.e_tot),
    mf1_mo_is_scanner_mo=mf1.mo_coeff is scf_scanner.mo_coeff,
    e_nuc_from_scanner_mol=float(scf_scanner.mol.energy_nuc()),
)
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'run_artifacts', '01_pure_water_o3_h2o', 'd2_pre_fix_probe.json')
with open(out, 'w') as fh:
    json.dump(info, fh, indent=2)
print(json.dumps(info, indent=2))
print('PROBE_JSON = %s' % out)
