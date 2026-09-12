#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Register the Stage-2b mode-follow endpoints into the standard gas_opt/ and
gas_freq/ record schemas so the downstream stages (SMD / binding / CCSD(T)),
which all load run_artifacts/.../gas_opt/<id>.json and gas_freq/<id>.json,
can consume them.  No recomputation -- reads the finished mode_follow/*.json.

Also fixes the registration gap that made Stage 3 drop 8 of the 9 kept minima
and Stage 4/5 crash with FileNotFoundError/KeyError ('c14_plus' etc.).
"""
import os
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import d2_full
import conformer_utils as cu

ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, 'run_artifacts', '01_pure_water_o3_h2o')
GAS_OPT = os.path.join(ART, 'gas_opt')
GAS_FREQ = os.path.join(ART, 'gas_freq')
FOLLOW = os.path.join(ART, 'mode_follow')
SYMS = ['O', 'O', 'O', 'O', 'H', 'H']

registered = []
for fn in sorted(os.listdir(FOLLOW)):
    if not fn.endswith('.json'):
        continue
    with open(os.path.join(FOLLOW, fn)) as fh:
        r = json.load(fh)
    label = r['label']
    coords = r['geometry_angstrom']
    mol = cu.mol_from_coords(coords, SYMS, verbose=0)

    common = dict(basis=d2_full.BASIS, xc=d2_full.XC,
                  grid_level=d2_full.GRID_LEVEL, scf_conv=d2_full.SCF_CONV,
                  solvent='none', charge=0, multiplicity=1,
                  pyscf_version=r.get('pyscf_version', '2.14.0'),
                  provenance=dict(stage='2b_mode_following', source_saddle=r['source'],
                                  direction=r['direction'],
                                  imag_eigenvalue_hartree_amu_bohr2=r.get(
                                      'imag_eigenvalue_hartree_amu_bohr2')),
                  spin_s2=0.0,
                  e_d2_hartree=d2_full.d2_energy(mol))

    # ---- gas_opt/<label>.json (Stage-1 schema, consumers: stage3/4/5)
    rec1 = dict(common)
    rec1.update(
        id=label, kind='mode_follow_endpoint',
        coverage='mode-follow endpoint of saddle %s (%s direction)' % (r['source'], r['direction']),
        rule=('displace +/-%.2f A along the most imaginary mode (mass-weighted '
              'eigenvector), re-optimised with geomeTRIC gmax=3e-5' % 0.25),
        opt_conv=dict(maxsteps=400, assert_convergence=False,
                      convergence_gmax=3e-5, convergence_grms=1.5e-5,
                      convergence_dmax=10.0, convergence_drms=10.0,
                      convergence_energy=1e-7),
        initial_coords_angstrom=None,          # displaced start not archived; see provenance
        mode_follow_source_coords=cu.coords_of(cu.load_gas_opt(GAS_OPT)[r['source']]).tolist(),
        optimised_coords_angstrom=coords,
        e_total_hartree=r['e_total_hartree'],
        max_gradient_hartree_bohr=r['max_gradient_hartree_bohr'],
        scf_converged=bool(r.get('optimisation_converged', False)),
        optimisation_converged=bool(r.get('optimisation_converged', False)),
        status=('converged' if r.get('optimisation_converged') else 'NOT_CONVERGED'),
        n_basis_functions=int(mol.nao_nr()))
    with open(os.path.join(GAS_OPT, '%s.json' % label), 'w') as fh:
        json.dump(rec1, fh, indent=2)

    # ---- gas_freq/<label>.json (Stage-2 schema, consumer: stage4 thermo)
    rec2 = dict(common)
    rec2.update(
        id=label, kind='mode_follow_endpoint',
        n_basis_functions=int(mol.nao_nr()),
        e_total_hartree=r['e_total_hartree'],
        hessian_method='analytic DFT + analytic -D2 (finite-diff of analytic grad)',
        n_imaginary=r['n_imaginary'],
        freq_error=r.get('freq_error'),
        lowest_frequency_cm1=r['lowest_frequency_cm1'],
        frequencies_cm1=r['frequencies_cm1'],
        is_minimum=bool(r['n_imaginary'] == 0),
        zpe_hartree=r['zpe_hartree'],
        zpe_kcal_mol=r['zpe_hartree'] * cu.HARTREE2KCAL,
        thermo=None,                            # full dict only in thermo_hartree form
        thermo_hartree=r['thermo_hartree'],
        geometry_angstrom=coords,
        topology=r['topology'],
        topology_signature=r['topology_signature'])
    with open(os.path.join(GAS_FREQ, '%s.json' % label), 'w') as fh:
        json.dump(rec2, fh, indent=2)
    registered.append(label)

print('registered %d endpoints: %s' % (len(registered), registered))
