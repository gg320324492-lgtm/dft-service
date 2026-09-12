#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JOB-2026-0906-005 Phase C step 2: full internal frequency analysis at the
NEW c14_plus L8 endpoint (wrapper around the Phase-A-revised step3 code).

Reuses the REVISED analysis functions (each step size cross-checked against
PySCF with ITS OWN matrix, same-matrix assertion active).  Only the points
directory and centre geometry change.  Negative eigenvalues / imaginary
frequencies are preserved.
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, 'run_artifacts', '01_pure_water_o3_h2o')
sys.path.insert(0, HERE)

import endpoint_freq_step3_frequency as s3
from pyscf import gto

SYMS = s3.SYMS
STEPS = s3.STEPS


def main():
    # patch the points directory to the L8 set
    s3.OUT = os.path.join(ART, 'endpoint_frequency')
    pts_dir = os.path.join(ART, 'endpoint_frequency',
                           'hessian_points_l8_c14')
    orig_load = s3.load_points

    def load_points_l8(tid, h):
        pts = {}
        for sgn in (1, -1):
            for j in range(18):
                p = os.path.join(pts_dir,
                                 'point_c14L8_h%.3f_dir%02d_sgn%+d.json'
                                 % (h, j, sgn))
                r = json.load(open(p))
                assert r['scf_converged'] and r['grid_response_actual'], p
                assert r['point_accepted'], p
                assert r['fingerprint']['step_bohr'] == h
                pts[(j, sgn)] = r
        return pts
    s3.load_points = load_points_l8

    res = json.load(open(os.path.join(ART, 'c14_l8_check',
                                      'l8_optimize_result.json')))
    coords_A = np.asarray(res['endpoint_coords_angstrom'], float)
    coords_B = coords_A / 0.52917721092
    mol = gto.M(atom="; ".join("%s %.10f %.10f %.10f"
                               % (s, x, y, z)
                               for s, (x, y, z) in zip(SYMS, coords_A)),
                basis='def2-TZVP', charge=0, spin=0, verbose=0,
                max_memory=2000)
    mass = mol.atom_mass_list(isotope_avg=True)

    per_h = {}
    for h in STEPS:
        per_h['%.3f' % h] = s3.analyse('c14_plus_L8endpoint', h, mass,
                                       coords_B)
    H1 = np.asarray(per_h['0.002']['hessian_symmetrized'])
    H2 = np.asarray(per_h['0.004']['hessian_symmetrized'])
    I1 = np.asarray(per_h['0.002']['internal_matrix_mw_Ubasis'])
    I2 = np.asarray(per_h['0.004']['internal_matrix_mw_Ubasis'])
    mt = s3.match_modes(per_h['0.002'], per_h['0.004'])
    for h in STEPS:
        a = per_h['%.3f' % h]
        print('[c14L8] h=%s: lowest lambda=%.4e (wn=%.2f) n_neg=%d '
              'anti=%.2e SAME-MATRIX xcheck=%.2e wn'
              % (h, a['lowest_eigenvalue'], a['lowest_freq_wn'],
                 a['n_negative_modes'], a['antisym_residual_max'],
                 a['pyscf_crosscheck']['max_abs_wn_diff']), flush=True)
    sc = dict(cartesian_hessian_max_diff=float(np.abs(H1 - H2).max()),
              cartesian_hessian_fro_diff=float(np.linalg.norm(H1 - H2)),
              internal_mw_matrix_max_diff_sameUbasis=
                  float(np.abs(I1 - I2).max()),
              mode_matching=mt,
              min_overlap=float(min(p['overlap'] for p in mt)),
              max_abs_d_wn_matched=max(abs(p['d_wn']) for p in mt))
    print('[c14L8] cart diff=%.2e  internal diff=%.2e  min overlap=%.6f  '
          'max|d_wn|=%.2f cm-1'
          % (sc['cartesian_hessian_max_diff'],
             sc['internal_mw_matrix_max_diff_sameUbasis'],
             sc['min_overlap'], sc['max_abs_d_wn_matched']), flush=True)
    out = dict(job='JOB-2026-0906-005 Phase C (c14_plus L8 endpoint)',
               per_step=per_h, step_comparison=sc)
    with open(os.path.join(ART, 'endpoint_frequency',
                           'frequency_analysis_l8_c14.json'), 'w') as fh:
        json.dump(out, fh, indent=2)
    print('saved -> frequency_analysis_l8_c14.json')


if __name__ == '__main__':
    main()
