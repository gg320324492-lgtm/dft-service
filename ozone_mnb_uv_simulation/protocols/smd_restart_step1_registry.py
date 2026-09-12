#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JOB-2026-0906-006 Phase A: gas-phase structure registry.

Primary references for the SMD restart batch:
  * c14_plus  -> the ACCEPTED L8 endpoint (JOB-2026-0906-005 Phase B);
                 the L7 endpoint and its evidence chain are retained.
  * c06_plus  -> the ACCEPTED L7 endpoint (JOB-2026-0906-004).
Records coordinates, units, provenance + file hashes, method settings,
frequency-evidence sources and the verdict SCOPE for each structure.
Energy non-portability: total energies from different grids / geometries
must NOT be combined into new precise conformation or binding energies;
historical CP / CCSD(T) / thermochemistry results keep their original
geometry + method labels and are NOT re-attached to these endpoints.
"""
import os, sys, json, hashlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, 'run_artifacts', '01_pure_water_o3_h2o')
OUT = os.path.join(ART, 'smd_restart')
SYMS = ['O', 'O', 'O', 'O', 'H', 'H']


def h16(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16]


def rel(p):
    return os.path.relpath(p, ROOT)


def main():
    os.makedirs(OUT, exist_ok=True)
    # sources
    p_l8 = os.path.join(ART, 'c14_l8_check', 'l8_optimize_result.json')
    p_fx = os.path.join(ART, 'endpoint_frequency', 'endpoints_fixed.json')
    p_l8freq = os.path.join(ART, 'endpoint_frequency',
                            'frequency_analysis_l8_c14.json')
    p_l7freq = os.path.join(ART, 'endpoint_frequency',
                            'frequency_analysis_revised.json')
    p_l8map = os.path.join(ART, 'c14_l8_check', 'formal_mapping.json')
    l8 = json.load(open(p_l8))
    fx = json.load(open(p_fx))
    reg = dict(
        job='JOB-2026-0906-006 Phase A',
        atom_order=SYMS,
        atom_order_note='0-2 ozone OOO (central=0, terminals 1,2 -- '
                        'geometry-verified), 3 water O, 4-5 water H',
        coords_unit='Angstrom',
        energy_portability=dict(
            rule='total energies from different grids or geometries must '
                 'NOT be combined into new precise conformation-energy or '
                 'binding-energy differences',
            historical_labels='CP / CCSD(T) / thermochemistry results keep '
                              'their ORIGINAL geometry + method labels and '
                              'are NOT re-attached to these endpoints'),
        global_minimum_claim='NONE -- both structures carry only '
                             'local-minimum evidence within the stated '
                             'check scopes',
        structures={})

    # ---- c14_plus: primary = L8 endpoint; L7 endpoint retained ----
    v = l8['independent_verification']
    reg['structures']['c14_plus'] = dict(
        primary_reference=dict(
            grid='L8', kind='SMD-restart start structure',
            coords_angstrom=l8['endpoint_coords_angstrom'],
            e_total_hartree=v['e_total'],
            grad_max=v['grad_max'], grad_rms=v['grad_rms'],
            scf_converged=v['scf_converged'],
            source_file=rel(p_l8), source_hash=h16(p_l8),
            optimizer='pyberny 0.7.0, 45/60 steps, gradientmax=gradientrms='
                      '1e-6 (internal-coord), one run',
            method=dict(xc="wb97xd (no -D2 in libxc) + project empirical "
                           "full-derivative -D2 (Chai-Head-Gordon, verbatim)",
                        basis='def2-TZVP', grid_level=8,
                        scf_tol=[1e-12, 1e-9], solvent='gas',
                        grid_response=True),
            verdict='local minimum SUPPORTED within L7/L8 checks '
                    '(JOB-2026-0906-005); not global minimum'),
        retained_evidence=dict(
            l7_endpoint=dict(
                coords_angstrom=fx['endpoints']['c14_plus']['coords_angstrom'],
                e_total_hartree=fx['endpoints']['c14_plus']['centre_check']
                ['e_total'],
                grad_max=fx['endpoints']['c14_plus']['centre_check']['grad_max'],
                source_file=rel(p_fx), source_hash=h16(p_fx)),
            l7_frequency_source=rel(p_l7freq),
            l7_frequency_hash=h16(p_l7freq),
            l7_frequency_verdict='12/12 internal modes positive at both '
                                 'step sizes (h=0.002/0.004 Bohr)',
            l8_frequency_source=rel(p_l8freq),
            l8_frequency_hash=h16(p_l8freq),
            l8_frequency_verdict='12/12 internal modes positive at both '
                                 'step sizes on the L8 endpoint',
            structure_preserved_L7_to_L8=dict(
                rmsd_A=json.load(open(p_l8map))
                ['c14_L7endpoint_vs_L8endpoint']['rmsd_A'],
                source=rel(p_l8map))))

    # ---- c06_plus: primary = L7 endpoint ----
    cc = fx['endpoints']['c06_plus']
    reg['structures']['c06_plus'] = dict(
        primary_reference=dict(
            grid='L7', kind='SMD-restart start structure',
            coords_angstrom=cc['coords_angstrom'],
            e_total_hartree=cc['centre_check']['e_total'],
            grad_max=cc['centre_check']['grad_max'],
            grad_rms=cc['centre_check']['grad_rms'],
            scf_converged=cc['centre_check']['scf_converged'],
            source_file=rel(p_fx), source_hash=h16(p_fx),
            method=dict(xc="wb97xd (no -D2 in libxc) + project empirical "
                           "full-derivative -D2 (Chai-Head-Gordon, verbatim)",
                        basis='def2-TZVP', grid_level=7,
                        scf_tol=[1e-12, 1e-9], solvent='gas',
                        grid_response=True),
            verdict='local minimum SUPPORTED within L7 checks '
                    '(JOB-2026-0906-004); L8 full spectrum NOT checked '
                    '(outside authorisation); not global minimum'),
        retained_evidence=dict(
            frequency_source=rel(p_l7freq),
            frequency_hash=h16(p_l7freq),
            frequency_verdict='12/12 internal modes positive at both step '
                              'sizes (h=0.002/0.004 Bohr, L7)',
            grid_robustness_scope='lowest direction only (L7/L8); NOT the '
                                  'full L8 internal spectrum'))

    out = os.path.join(OUT, 'gas_phase_registry.json')
    json.dump(reg, open(out, 'w'), indent=2)
    print('registry saved ->', rel(out))
    for tid, st in reg['structures'].items():
        pr = st['primary_reference']
        print(' %-9s grid=%s  E=%.9f  max|g|=%.3e'
              % (tid, pr['grid'], pr['e_total_hartree'], pr['grad_max']))


if __name__ == '__main__':
    main()
