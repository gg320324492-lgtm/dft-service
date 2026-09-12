#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JOB-2026-0906-013 (Task C part 2): at most TWO independent direction checks
on the NEW c06 order-47 endpoint -- the same-geometry order41/47 comparison.
The old step-40 41/47 rotation data can NOT substitute for this.

Selection rule was fixed in smd_diag_analysis.py BEFORE any direction
evaluation.  Directions are fixed, normalized Cartesian straight-line
displacements (max atom |d_i| = 1); NO re-optimisation, NO realignment,
NO rotation after displacement.

Budget: 1 order41 center (order47 center reused from Task B record)
        + n_dirs x 2 gears x 2 amplitudes (0.002/0.004 Angstrom) x 2 signs
        = 1 + 16 evals at most.  No retries, no expansion.

Per (direction, gear) reported: FD energy slope vs center projected
gradient; FD energy curvature vs gradient-difference curvature; order-47
Hessian projection; sign/magnitude changes across amplitudes and gears.
Each gear uses its OWN center energy/gradient.
"""
import os
import sys
import json
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, 'run_artifacts', '01_pure_water_o3_h2o')
OUT = os.path.join(ART, 'smd_diag_curvature')
sys.path.insert(0, HERE)

import torque_decomp as td
BOHR_A = td.BOHR_A

AMPS_A = [0.002, 0.004]          # Angstrom (fixed by the batch instruction)


def main():
    import smd_diag_hessian as H  # reuse eval_at / mol builder / constants
    analysis = json.load(open(os.path.join(OUT, 'analysis.json')))
    center47 = json.load(open(os.path.join(OUT, 'center_order47.json')))
    hb = json.load(open(os.path.join(OUT, 'hessian_batch.json')))
    center = np.asarray(center47['coords_angstrom'], float)
    dirs = [d for d in analysis['direction_selection']['directions']
            if d.get('mode_index') is not None]
    assert 1 <= len(dirs) <= 2
    print('directions selected (rule fixed pre-evaluation):',
          [(d['rank'], d['mode_index'], round(d['wavenumber_cm1'], 1))
           for d in dirs], flush=True)

    # order41 center (1 new eval); order47 center reused from Task B
    c41 = H.eval_at(center, 41, 'center_order41')
    json.dump(c41, open(os.path.join(OUT, 'center_order41.json'), 'w'),
              indent=2)
    print('[center41] E=%.9f max|g|=%.3e order=%d'
          % (c41['e_total'], c41['grad_max'], c41['surface_order_actual']),
          flush=True)
    centers = {47: dict(e_total=center47['e_total'],
                        grad=np.asarray(center47['gradient_full_6x3'], float)),
               41: dict(e_total=c41['e_total'],
                        grad=np.asarray(c41['gradient_full_6x3'], float))}

    Hsym47 = np.asarray(hb['hessians']['0.002']['sym_hessian_18x18'], float)

    records = []
    comparisons = []
    n = 0
    try:
        for d in dirs:
            dvec = np.asarray(d['cartesian_direction_6x3'], float)
            dflat = dvec.reshape(-1)
            for gear in (47, 41):
                e0 = centers[gear]['e_total']
                g0 = centers[gear]['grad'].reshape(-1)
                proj_grad = float((g0 * dflat).sum()) / BOHR_A  # Eh/Angstrom
                hproj_bohr = float(dflat @ Hsym47 @ dflat)      # Eh/Bohr^2
                hproj = hproj_bohr / BOHR_A ** 2                # Eh/Angstrom^2
                per_amp = {}
                for amp in AMPS_A:
                    e_p = e_m = None
                    g_p = g_m = None
                    for sgn in (+1, -1):
                        coords_d = center + sgn * amp * dvec
                        intended = sgn * amp * dvec
                        tag = 'dir%d_gear%d_amp%0.3f_sgn%+d' % (
                            d['rank'], gear, amp, sgn)
                        rec = H.eval_at(coords_d, gear, tag)
                        actual = (np.asarray(coords_d, float) - center)
                        d_err = float(np.abs(actual - intended).max())
                        if d_err > 1e-8:
                            raise RuntimeError('displacement mismatch %s: %.2e'
                                               % (tag, d_err))
                        rec.update(dir_rank=d['rank'], gear=gear, amp_A=amp,
                                   sign=sgn, disp_error_A=d_err)
                        records.append(rec)
                        n += 1
                        print('[%2d/16] %s E=%.9f max|g|=%.3e order=%d ok'
                              % (n, tag, rec['e_total'], rec['grad_max'],
                                 rec['surface_order_actual']), flush=True)
                        json.dump(records, open(
                            os.path.join(OUT, 'direction_points.json'), 'w'),
                            indent=2)
                        if sgn > 0:
                            e_p, g_p = rec['e_total'], rec['gradient_full_6x3']
                        else:
                            e_m, g_m = rec['e_total'], rec['gradient_full_6x3']
                    g_p = np.asarray(g_p, float).reshape(-1)
                    g_m = np.asarray(g_m, float).reshape(-1)
                    slope_fd = (e_p - e_m) / (2 * amp)               # Eh/A
                    curv_e = (e_p - 2 * e0 + e_m) / amp ** 2         # Eh/A^2
                    curv_g = float(((g_p - g_m) * dflat).sum()
                                   / (2 * amp)) / BOHR_A             # Eh/A^2
                    per_amp[amp] = dict(
                        slope_fd_Eh_per_A=slope_fd,
                        curvature_energy_fd_Eh_per_A2=curv_e,
                        curvature_gradient_fd_Eh_per_A2=curv_g,
                        slope_vs_proj_grad_rel=abs(
                            slope_fd - proj_grad) / (abs(proj_grad) + 1e-300),
                        curv_e_vs_curv_g_rel=abs(
                            curv_e - curv_g) / (abs(curv_g) + 1e-300))
                comparisons.append(dict(
                    dir_rank=d['rank'], mode_index=d['mode_index'],
                    wavenumber_cm1_h002=d['wavenumber_cm1'], gear=gear,
                    center_e=e0, center_grad_max=float(np.abs(g0).max()),
                    projected_gradient_Eh_per_A=proj_grad,
                    hessian_projection_Eh_per_A2_order47matrix=hproj,
                    per_amp=per_amp,
                    note='fixed Cartesian straight-line path; no '
                         'realignment/rotation; each gear uses its OWN '
                         'center energy and gradient'))
        json.dump(dict(job='JOB-2026-0906-013 Task C (direction checks)',
                       n_direction_evals=n, n_center41=1,
                       directions=analysis['direction_selection']['directions'],
                       comparisons=comparisons, points=records),
                  open(os.path.join(OUT, 'direction_checks.json'), 'w'),
                  indent=2)
        print('DIRECTION CHECKS COMPLETE: %d points + 1 order41 center'
              % n, flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        json.dump(dict(aborted_at=n, error=repr(e), completed=records),
                  open(os.path.join(OUT, 'direction_checks.json'), 'w'),
                  indent=2)
        print('DIRECTION BATCH ABORTED: %s' % e, flush=True)


if __name__ == '__main__':
    main()
