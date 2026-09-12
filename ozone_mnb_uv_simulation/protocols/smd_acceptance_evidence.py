#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JOB-2026-0906-012 (OFFLINE ONLY -- no SCF, no optimisation, no Hessian,
no rotation scans).  Builds the formal aqueous endpoint index, the
registered structure comparison, and the unified evidence table from
EXISTING artifacts only.

Sources (never re-judged by filename):
  c06 order41 endpoint : smd_restart/phaseC_smd_optimize.json :: c06_plus
  c06 order47 endpoint : smd_order47_opt/order47_optimization.json
  c14 120-step endpoint: smd_controlled_opt/c14_cont_result.json
  fixed-unit decomp    : smd_rotation_audit/endpoint_decomp_fixed.json (009)
  five-point order47   : smd_surface_refine/order47_five_points.json  (010)

Outputs (run_artifacts/01_pure_water_o3_h2o/smd_acceptance_evidence/):
  endpoint_index.json      formal index (config/coords/hash/grad/status)
  registration.json        Kabsch + equivalent-permutation mapping
  evidence_table.json      geometry x config evidence rows
"""
import os
import sys
import json
import hashlib
import itertools
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, 'run_artifacts', '01_pure_water_o3_h2o')
OUT = os.path.join(ART, 'smd_acceptance_evidence')
sys.path.insert(0, HERE)
import torque_decomp as td

SYMS = ['O', 'O', 'O', 'O', 'H', 'H']
MASS = [15.9949146] * 4 + [1.007825] * 2
GMAX = 1e-5


def fsha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16]


def csha(coords):
    return hashlib.sha256(np.asarray(coords, float).tobytes()).hexdigest()[:16]


def kabsch_rmsd(P, Q, perm=None):
    """Proper-rotation (Kabsch) RMSD of P onto Q.  perm[i] = source index
    placed at target slot i (permutation of equivalent atoms)."""
    P = np.asarray(P, float)
    Q = np.asarray(Q, float)
    if perm is not None:
        P = P[list(perm)]
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt = Vt.copy()
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    diff = (R @ Pc.T).T - Qc
    return float(np.sqrt((diff ** 2).sum(axis=1).mean())), R


def register(P, Q):
    """Search equivalent-atom permutations (O3 group 6 perms x H swap 2)."""
    best = None
    for po in itertools.permutations([0, 1, 2]):
        for hs in ((3, 4, 5), (3, 5, 4)):
            perm = list(po) + list(hs)
            r, R = kabsch_rmsd(P, Q, perm)
            if best is None or r < best['rmsd_A']:
                best = dict(rmsd_A=r, perm=perm,
                            perm_O3=list(po), h_swapped=(hs == (3, 5, 4)))
    return best


def gap_and_contact(c):
    """O3-group centroid <-> H2O centroid gap; shortest O3..H2O atom contact."""
    c = np.asarray(c, float)
    gap = float(np.linalg.norm(c[:3].mean(axis=0) - c[3:].mean(axis=0)))
    dmin, pair = None, None
    for i in range(3):
        for j in range(3, 6):
            d = float(np.linalg.norm(c[i] - c[j]))
            if dmin is None or d < dmin:
                dmin, pair = d, (SYMS[i] + str(i), SYMS[j] + str(j))
    return dict(o3_water_gap_A=gap, min_o3_h2o_contact_A=dmin,
                min_contact_pair=pair)


def load(name):
    return json.load(open(os.path.join(ART, name)))


def main():
    os.makedirs(OUT, exist_ok=True)
    src_c06_41 = os.path.join(ART, 'smd_restart', 'phaseC_smd_optimize.json')
    src_c06_47 = os.path.join(ART, 'smd_order47_opt', 'order47_optimization.json')
    src_c14 = os.path.join(ART, 'smd_controlled_opt', 'c14_cont_result.json')
    p41 = load('smd_restart/phaseC_smd_optimize.json')['c06_plus']
    p47 = load('smd_order47_opt/order47_optimization.json')
    c14 = load('smd_controlled_opt/c14_cont_result.json')
    decomp009 = load('smd_rotation_audit/endpoint_decomp_fixed.json')
    o47pts = load('smd_surface_refine/order47_five_points.json')

    coords41 = np.asarray(p41['endpoint_coords_angstrom'], float)
    coords47 = np.asarray(p47['endpoint_coords_angstrom'], float)
    coords14 = np.asarray(c14['endpoint_coords_angstrom'], float)
    g41 = np.asarray(p41['independent_verification']['gradient_full_6x3'], float)
    g47 = np.asarray(p47['independent_verification']['gradient_full_6x3'], float)
    g14 = np.asarray(c14['independent_verification']['gradient_full_6x3'], float)

    # decompositions (corrected units, torque_decomp convention)
    d41 = td.decompose_force(g41, coords41, MASS, unit='Angstrom')
    d47 = td.decompose_force(g47, coords47, MASS, unit='Angstrom')
    d14 = td.decompose_force(g14, coords14, MASS, unit='Angstrom')

    # ---------------- endpoint index ----------------
    OLD_FP_NOTE = ('old method_fingerprint string in these files used the '
                   'parent-class-name heuristic and carried NO surface '
                   'order: the order-41 and order-47 records shared an '
                   'IDENTICAL fingerprint (verified below).  Historical '
                   'records are kept verbatim; the corrected fingerprint '
                   'block here is reconstructed from the recorded config '
                   'and linked to evidence.')
    fp41_old = p41.get('method_fingerprint')
    fp47_old = p47.get('method_fingerprint')
    fp14_old = c14.get('method_fingerprint')
    old_fp_identical_41_47 = (json.dumps(fp41_old, sort_keys=True)
                              == json.dumps(fp47_old, sort_keys=True))

    def entry(eid, path, job, rec, coords, g, dec, stop, surf_order,
              surf_order_src, freq, iv_extra):
        gm = float(np.abs(g).max())
        return dict(
            id=eid, source_file=os.path.relpath(path, ROOT),
            source_sha256=fsha(path), job=job, record=rec,
            coords_angstrom=np.asarray(coords, float).tolist(),
            coords_sha_full_precision=csha(coords),
            coords_unit='Angstrom',
            config=dict(
                solvent_model='SMD(water)', charge=0, spin=0,
                xc='wb97xd (libxc) + project -D2 (Chai-Head-Gordon, full '
                   'derivatives)', basis='def2-TZVP',
                dft_grid_level=8, grid_response=True,
                scf_conv_tol=1e-12, scf_conv_tol_grad=1e-9,
                surface_order=surf_order, surface_order_source=surf_order_src,
                corrected_fingerprint=dict(
                    model='smd', solvent='water',
                    lebedev_order=surf_order, grid_level=8,
                    d2_attached=True),
                old_fingerprint_verbatim=iv_extra.pop('old_fp', None)),
            gradient_full_6x3=np.asarray(g, float).tolist(),
            grad_max_unprojected=gm,
            grad_rms=float(np.sqrt((np.asarray(g, float) ** 2).mean())),
            decomposition=dict(
                g_translation_norm=dec['g_trans_norm'],
                g_rotation_norm=dec['g_rot_norm'],
                g_internal_norm=dec['g_int_norm'],
                T_g=dec['T_g'], T_g_norm=dec['T_g_norm']),
            optimizer_stop=stop,
            acceptance=dict(
                criterion='unprojected max|g| <= 1e-5 Eh/Bohr (UNCHANGED)',
                passed=bool(gm <= GMAX)),
            full_frequency=freq,
            **iv_extra)

    e41 = entry(
        'c06_order41_step40', src_c06_41, 'JOB-2026-0906-004 (SMD restart '
        'phaseC)', 'c06_plus.endpoint_coords_angstrom', coords41, g41, d41,
        dict(optimizer_converged=p41['optimizer_converged'],
             n_steps=p41['n_steps'], maxsteps=p41['maxsteps'],
             verdict=p41['verdict'],
             note='optimizer internal-coordinate criteria met; Cartesian '
                  'acceptance FAILED'),
        surf_order='missing (not recorded in source; produced before the '
                   'surface-order audit)',
        surf_order_src='missing', freq='none (no water-phase frequency)',
        iv_extra=dict(old_fp=fp41_old))
    e47 = entry(
        'c06_order47_reopt', src_c06_47, 'JOB-2026-0906-011',
        'endpoint_coords_angstrom + independent_verification', coords47,
        g47, d47,
        dict(optimizer_converged=p47['optimizer_converged'],
             n_steps=p47['n_steps'], maxsteps=p47['maxsteps'],
             hit_maxsteps=p47['hit_maxsteps'],
             note='optimizer internal-coordinate criteria met (28/30); '
                  'unprojected Cartesian max|g| = 2.503e-5 > 1e-5 -> NOT '
                  'passed; threshold NOT relaxed, no gear switch'),
        surf_order=47, surf_order_src='recorded per-step and verified '
                                      'independently (JOB-011)',
        freq='none (no water-phase frequency)',
        iv_extra=dict(old_fp=fp47_old))
    e14 = entry(
        'c14_cont120', src_c14, 'JOB-2026-0906-008 (c14 continuation)',
        'endpoint_coords_angstrom + independent_verification', coords14,
        g14, d14,
        dict(optimizer_converged=c14['optimizer_converged'],
             n_steps=c14['n_steps'], maxsteps=c14['maxsteps'],
             cumulative_optimizer_steps=120,
             verdict=c14['verdict'],
             note='maxsteps reached (60/60, cumulative 120); optimizer '
                  'internal criteria met; Cartesian acceptance FAILED '
                  '(1.2504e-5 > 1e-5)'),
        surf_order='missing (not recorded in source)',
        surf_order_src='missing', freq='none (no water-phase frequency)',
        iv_extra=dict(old_fp=fp14_old))

    index = dict(
        job='JOB-2026-0906-012',
        title='formal aqueous endpoint index (offline)',
        principle='entries are taken from actual saved records, NEVER from '
                  'filename tokens such as validated/pending',
        entries=[e41, e47, e14],
        supersession=dict(
            c14='this index supersedes any earlier ad-hoc citation of the '
                'c14 geometry; the 008 cumulative-120-step endpoint is the '
                'formal c14 aqueous endpoint',
            c06='BOTH c06 endpoints are retained: order41 step-40 (old '
                'production gear) AND order47 re-optimised (011); neither '
                'passes the unprojected acceptance'),
        fingerprint_correction=dict(
            old_scheme_defect='solvent guessed from parent-class name; no '
                              'surface order recorded',
            order41_order47_old_fingerprints_identical=old_fp_identical_41_47,
            cache_reuse_impact='none found: no script ever used the '
                               'fingerprint as a cache/dedup key (grep '
                               'audit); impact limited to provenance '
                               'coarseness + one display mislabel in the '
                               '011 preopt config_before (inline heuristic, '
                               'already documented there)',
            affected_files=[os.path.relpath(p, ROOT) for p in (
                src_c06_41, src_c06_47, src_c14,
                os.path.join(ART, 'smd_surface_refine', 'order47_five_points.json'),
                os.path.join(ART, 'smd_surface_refine',
                             'smd_surface_refinement_summary.json'),
                os.path.join(ART, 'c14_l8_check', 'l8_optimize_result.json'),
                os.path.join(ART, 'endpoint_frequency', 'endpoints_fixed.json'),
                os.path.join(ART, 'phaseB_scanner_verify',
                             'scanner_verify_c14_plus_L7_center.json'),
                os.path.join(ART, 'phaseB_scanner_verify',
                             'scanner_verify_smoke_H2O.json'),
                os.path.join(ART, 'phaseB_scanner_verify',
                             'scanner_verify_smoke_H2O_dimer.json'),
                os.path.join(ART, 'phaseC_reoptimization',
                             'phaseC_summary_attempt1.json'),
                os.path.join(ART, 'phaseC_reoptimization',
                             'phaseC_summary_attempt2.json'),
                os.path.join(ART, 'smd_controlled_opt', 'c06_decomp_diag.json'),
                os.path.join(ART, 'smd_restart', 'phaseB_smd_verification.json'))],
            fix='grad_factory.method_fingerprint now reads the live '
                'with_solvent attachment (solvent name, smd method, '
                'lebedev_order, discretisation); regression '
                'protocols/test_method_fingerprint.py 5/5 PASS'))

    # ---------------- registration ----------------
    reg = {}
    for a, b, key in ((coords47, coords41, 'c06_order47_vs_c06_order41'),
                      (coords47, coords14, 'c06_order47_vs_c14_cont120'),
                      (coords41, coords14, 'c06_order41_vs_c14_cont120')):
        best = register(a, b)
        extra = dict(gap_and_contact(a), target=gap_and_contact(b))
        reg[key] = dict(
            method='centroid + proper Kabsch rotation + equivalent-atom '
                   'permutation search (O3 group 6 perms x H swap 2)',
            best_perm=best['perm'], perm_O3=best['perm_O3'],
            h_swapped=best['h_swapped'], rmsd_A=best['rmsd_A'],
            **extra)
        if 'c14' in key:
            reg[key]['note'] = ('SUPERSEDES the JOB-011 report section 5 '
                                'unregistered c14 comparison '
                                '(RMSD 1.5998 A, no permutation/rotation); '
                                'this registered value replaces it.')

    # ---------------- evidence table ----------------
    # rotation-dependence evidence (from 009/010, at the c06 step-40 coords)
    th0_47 = [p for p in o47pts['points'] if p['theta_deg'] == 0.0][0]
    d_old47 = td.decompose_force(
        np.asarray(th0_47['gradient_full_6x3'], float),
        np.asarray(th0_47['coords_angstrom'], float), MASS, unit='Angstrom')
    axis = load('smd_rotation_audit/rotation_audit_data.json')[
        'candidates']['c06_plus']['axis']

    def row(rid, geom, cfg, g, dec, stop, rot, freq, supports, cannot):
        gm = float(np.abs(g).max())
        return dict(id=rid, geometry_version=geom, method_config=cfg,
                    grad_max_unprojected=gm,
                    g_norm=float(np.linalg.norm(g)),
                    translation_norm=dec['g_trans_norm'],
                    rotation_norm=dec['g_rot_norm'],
                    internal_norm=dec['g_int_norm'],
                    T_g_norm=dec['T_g_norm'],
                    optimizer_stop=stop,
                    rotation_measurements=rot,
                    full_frequency=freq,
                    supports=supports, cannot_support=cannot)
    rotmeas = ('measured at THESE coords (c06 step-40): fixed axis %s, '
               'theta=0,±0.1°,±0.2°; max|dE|: 31 1.56e-7 / 41 5.567e-7 / '
               '47 8.285e-8 (ratio 47/41 = 0.149, ordering 47<31<41 '
               'NON-monotonic); slope vs axis·T_g agreement 0.13-1.8%%' %
               (np.round(axis, 4).tolist(),))
    rotmeas_47end = ('NOT measured at these coords; the five-point rotation '
                     'scan was run at the OLD c06 step-40 geometry only')
    rotmeas_c14 = ('measured at THESE coords (JOB-013 correction: the 009 '
                   'rotation_audit_data.json c14_plus record IS the formal '
                   'c14 endpoint, verified: theta=0 coords match '
                   'c14_cont120 to <=5.6e-17, E/max|g| agree to ~1e-13, '
                   'record start_source cites "008 continuation endpoint, '
                   'cumulative 120 steps"): fixed axis [-0.1972, -0.9603, '
                   '0.1974], theta=0,+-0.1,+-0.2 deg; max|dE| = 6.900e-7 '
                   '(+3.357e-7/-3.143e-7 at 0.1, +6.900e-7/-6.043e-7 at '
                   '0.2); slope vs axis*T_g residuals -2.56e-7 (0.14%) and '
                   '-1.10e-6 (0.59%)')
    table = dict(
        job='JOB-2026-0906-012',
        acceptance_criterion='unprojected max|g| <= 1e-5 Eh/Bohr (UNCHANGED; '
                             'no current aqueous structure passes)',
        rows=[
            row('R1', 'c06 step-40 coords (order41 endpoint)',
                'SMD(water) L8, surface order missing (pre-audit), wb97xd+-D2, '
                'def2-TZVP, grid_response, SCF 1e-12/1e-9',
                g41, d41,
                'optimizer internal criteria met at step 40/60; Cartesian FAILED',
                rotmeas, 'none',
                ['internal-structure diagnostics',
                 'torque-energy slope relation (009/010)',
                 'sensitivity of residuals to surface discretisation'],
                ['formal stationary acceptance', 'final thermochemistry',
                 'reaction conclusions', 'any frequency-based science']),
            row('R2', 'c06 step-40 coords (SAME geometry as R1)',
                'SMD(water) L8, surface order 47, rest identical to R1',
                np.asarray(th0_47['gradient_full_6x3'], float), d_old47,
                'single-point evaluation (no optimisation at this row)',
                rotmeas + ' -- includes the order-47 five-point at these '
                          'coords (surface points 2747-2753)',
                'none',
                ['same-geometry order41-vs-47 SENSITIVITY CHECK (explicitly '
                 'NOT a convergence proof): max|g| 2.58e-5 vs 9.10e-5, '
                 'internal 1.15e-6 vs 2.00e-4, rotation 5.01e-5 vs 2.83e-5'],
                ['formal stationary acceptance', 'convergence claims',
                 'final thermochemistry']),
            row('R3', 'c06 order47 re-optimised endpoint (011, 28 steps)',
                'SMD(water) L8, surface order 47, rest identical',
                g47, d47,
                'optimizer internal criteria met (28/30, no maxsteps); '
                'unprojected Cartesian max|g| 2.503e-5 > 1e-5 -> NOT passed',
                rotmeas_47end, 'none',
                ['internal-structure diagnostics (internal residual 1.31e-7)',
                 'evidence that internal optimisation does NOT remove the '
                 'rotation-type external residual (5.02e-5)'],
                ['formal stationary acceptance', 'final thermochemistry',
                 'reaction conclusions', 'any frequency-based science']),
            row('R4', 'c14 continuation endpoint (008, cumulative 120 steps)',
                'SMD(water) L8, surface order missing (not recorded), '
                'wb97xd+-D2, def2-TZVP, grid_response, SCF 1e-12/1e-9',
                g14, d14,
                'maxsteps reached (60/60, cumulative 120); optimizer internal '
                'criteria met; Cartesian FAILED (1.2504e-5 > 1e-5)',
                rotmeas_c14, 'none',
                ['internal-structure diagnostics',
                 'formal c14 aqueous endpoint reference for registration'],
                ['formal stationary acceptance', 'final thermochemistry',
                 'reaction conclusions', 'any frequency-based science']),
        ],
        missing_policy='anything not recorded in the sources is listed as '
                       'missing; nothing is inferred or back-filled',
        non_single_factor_warning='R1/R2 differ ONLY in surface gear at the '
                                  'SAME geometry (sensitivity check); R3 is a '
                                  'DIFFERENT geometry; no pair of rows with '
                                  'different geometries or different rotation '
                                  'directions may be presented as a '
                                  'single-factor control')

    out1 = os.path.join(OUT, 'endpoint_index.json')
    out2 = os.path.join(OUT, 'registration.json')
    out3 = os.path.join(OUT, 'evidence_table.json')
    json.dump(index, open(out1, 'w'), indent=2)
    json.dump(reg, open(out2, 'w'), indent=2)
    json.dump(table, open(out3, 'w'), indent=2)
    print('ENDPOINT INDEX ->', out1)
    print('REGISTRATION  ->', out2)
    print('EVIDENCE TABLE->', out3)
    for k, v in reg.items():
        print('  %-32s RMSD=%.4f A  perm=%s' % (k, v['rmsd_A'],
                                                v['best_perm']))
    print('  old 41/47 fingerprints identical:', old_fp_identical_41_47)


if __name__ == '__main__':
    main()
