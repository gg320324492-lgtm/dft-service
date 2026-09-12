#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JOB-2026-0906-007 Phase C: OFFLINE diagnosis of the saved 60/40-step SMD
trajectories (no re-optimisation, no new SCF).

Per candidate: per-step energy, max/RMS Cartesian gradient, adjacent-step
max atom displacement, and key intermolecular distances (Ow..O3-centroid
gap, min Ow-O).  For c14_plus: lowest-energy step, lowest-gradient step and
last step, with the structural motion accompanying the gradient rebound.
For c06_plus: the real stop reason (internal-coordinate criteria met per
the optimizer; the actual internal gradient values are NOT logged, recorded
as missing rather than inferred).  Produces ONE evidence-based follow-up
plan per candidate (proposal only, nothing executed).
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, 'run_artifacts', '01_pure_water_o3_h2o')
OUT = os.path.join(ART, 'smd_config_audit')


def geo_stats(c):
    c = np.asarray(c, float)
    gap = float(np.linalg.norm(c[3] - c[:3].mean(0)))
    d_oo = [float(np.linalg.norm(c[3] - c[k])) for k in range(3)]
    return dict(gap_Ow_O3c_A=round(gap, 4),
                min_Ow_O_A=round(min(d_oo), 4),
                max_Ow_O_A=round(max(d_oo), 4))


def diagnose(tid, r):
    steps = r['steps']
    n = len(steps)
    E = np.array([s['e_total'] for s in steps])
    gmax = np.array([s['grad_max'] for s in steps])
    grms = np.array([s['grad_rms'] for s in steps])
    C = np.array([s['coords_angstrom'] for s in steps])
    disp = [None] + [float(np.abs(C[i] - C[i - 1]).max())
                     for i in range(1, n)]
    gaps = [geo_stats(c) for c in C]

    i_E = int(np.argmin(E)); i_g = int(np.argmin(gmax))
    # classify: count energy increases / gradient rebounds
    dE = np.diff(E)
    n_rise = int((dE > 0).sum())
    biggest_rise = float(dE.max()) if n_rise else 0.0
    # oscillation metric: sign changes in dE among significant moves
    sig = np.abs(dE) > 1e-8
    sign_changes = int(np.sum(np.diff(np.sign(dE[sig])) != 0))
    # gradient rebound window (c14): steps around the gradient minimum
    reb = None
    if i_g < n - 1:
        reb = dict(
            lowest_grad_step=i_g + 1, lowest_grad=float(gmax[i_g]),
            next_steps=[{'step': k + 1, 'grad_max': float(gmax[k]),
                         'disp_from_prev_A': disp[k],
                         'gap_Ow_O3c_A': gaps[k]['gap_Ow_O3c_A']}
                        for k in range(i_g, min(i_g + 6, n))])
    table = [dict(step=s['step'], e_total=s['e_total'],
                  grad_max=s['grad_max'], grad_rms=s['grad_rms'],
                  max_disp_from_prev_A=disp[k],
                  **gaps[k])
             for k, s in enumerate(steps)]
    return dict(
        n_steps=n, table=table,
        lowest_energy_step=i_E + 1, lowest_energy=float(E[i_E]),
        lowest_gradient_step=i_g + 1, lowest_gradient=float(gmax[i_g]),
        last_step=dict(e_total=float(E[-1]), grad_max=float(gmax[-1]),
                       gap_Ow_O3c_A=gaps[-1]['gap_Ow_O3c_A']),
        energy_rises=n_rise, biggest_energy_rise=biggest_rise,
        energy_sign_changes_among_significant_moves=sign_changes,
        gradient_rebound=reb,
        displacement_stats=dict(max=float(np.max(disp[1:])),
                                median=float(np.median(disp[1:]))),
        E_total_change=float(E[-1] - E[0]))


def main():
    pc = json.load(open(os.path.join(ART, 'smd_restart',
                                     'phaseC_smd_optimize.json')))
    out = {}
    for tid in ('c14_plus', 'c06_plus'):
        out[tid] = diagnose(tid, pc[tid])
        d = out[tid]
        print('== %s: %d steps; E0=%.6f -> Emin=%.6f (step %d) -> Elast=%.6f'
              % (tid, d['n_steps'],
                 pc[tid]['steps'][0]['e_total'], d['lowest_energy'],
                 d['lowest_energy_step'], d['last_step']['e_total']))
        print('   grad: min=%.3e (step %d), last=%.3e; energy rises=%d '
              '(max %+.2e); sign changes=%d; max step disp=%.3f A'
              % (d['lowest_gradient'], d['lowest_gradient_step'],
                 d['last_step']['grad_max'], d['energy_rises'],
                 d['biggest_energy_rise'],
                 d['energy_sign_changes_among_significant_moves'],
                 d['displacement_stats']['max']))
        if d['gradient_rebound']:
            print('   rebound window:',
                  ['s%d g=%.2e disp=%.3f gap=%.2f'
                   % (x['step'], x['grad_max'], x['disp_from_prev_A'],
                      x['gap_Ow_O3c_A'])
                   for x in d['gradient_rebound']['next_steps'][:5]])
    json.dump(out, open(os.path.join(OUT, 'trajectory_diagnosis.json'),
                        'w'), indent=2)
    print('saved -> trajectory_diagnosis.json')


if __name__ == '__main__':
    main()
