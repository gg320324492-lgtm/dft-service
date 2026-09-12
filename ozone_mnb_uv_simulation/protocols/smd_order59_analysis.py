#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JOB-2026-0906-015 (Task C, offline): unified 41/47/59 fixed-direction
curvature comparison.

All three gears are recomputed from RAW saved points with the SAME code path
(the 41/47 raw points passed the reuse-compatibility gate in
smd_order59_check.py).  Each gear uses its OWN center energy/gradient.

Per amplitude per gear:
  * center full-gradient max component and direction projection
  * central-difference energy slope
  * energy second-difference curvature
  * gradient-difference curvature
  * |curvE - curvG| and the two-amplitude difference

Wording policy (fixed): only observed statements such as "the measured gears
agree in sign", "sign flip persists", "magnitude still gear-sensitive".
A single new gear, a monotonic change, or three-gear sign agreement is NOT a
convergence proof and supports no real-frequency / transition-state /
minimum / acceptance claim.
"""
import os
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, 'run_artifacts', '01_pure_water_o3_h2o')
OUT = os.path.join(ART, 'smd_diag_curvature')
sys.path.insert(0, HERE)
import torque_decomp as td

BOHR_A = td.BOHR_A
AMPS = [0.002, 0.004]


def gear_curvature(center, g_center, dvec, pts, gear):
    """Recompute per-amplitude quantities from raw displaced points of one
    gear (pts = list of dicts with amp_A, sign, e_total, gradient_full_6x3,
    coords, disp_error_A)."""
    dflat = dvec.reshape(-1)
    proj = float((np.asarray(g_center, float).reshape(-1) * dflat).sum()) \
        / BOHR_A                       # Eh/Angstrom
    out = dict(gear=gear,
               center_grad_max=float(np.abs(np.asarray(
                   g_center, float)).max()),
               projected_gradient_Eh_per_A=proj)
    per = {}
    for amp in AMPS:
        p_plus = [p for p in pts if p.get('gear') == gear
                  and abs(p['amp_A'] - amp) < 1e-12 and p['sign'] > 0]
        p_minus = [p for p in pts if p.get('gear') == gear
                   and abs(p['amp_A'] - amp) < 1e-12 and p['sign'] < 0]
        if not p_plus or not p_minus:
            continue
        pp, pm = p_plus[0], p_minus[0]
        e_p, e_m = pp['e_total'], pm['e_total']
        g_p = np.asarray(pp['gradient_full_6x3'], float).reshape(-1)
        g_m = np.asarray(pm['gradient_full_6x3'], float).reshape(-1)
        slope = (e_p - e_m) / (2 * amp)                                  # Eh/A
        curv_e = (e_p - OUT_E[gear] - (e_m - OUT_E[gear])) / amp ** 2 \
            if False else (e_p + e_m - 2 * OUT_E[gear]) / amp ** 2       # Eh/A^2
        curv_g = float(((g_p - g_m) * dflat).sum() / (2 * amp)) / BOHR_A
        per[str(amp)] = dict(
            slope_fd_Eh_per_A=slope,
            slope_vs_proj_rel=abs(slope - proj) / (abs(proj) + 1e-300),
            curvature_energy_fd_Eh_per_A2=curv_e,
            curvature_gradient_fd_Eh_per_A2=curv_g,
            curvE_curvG_absdiff=abs(curv_e - curv_g))
    out['per_amp'] = per
    amps = sorted(per)
    if len(amps) == 2:
        a1, a2 = per[amps[0]], per[amps[1]]
        out['two_amp_curvE_diff'] = abs(a1['curvature_energy_fd_Eh_per_A2']
                                        - a2['curvature_energy_fd_Eh_per_A2'])
        out['two_amp_curvG_diff'] = abs(a1['curvature_gradient_fd_Eh_per_A2']
                                        - a2['curvature_gradient_fd_Eh_per_A2'])
        out['sign_consistent_across_amps'] = bool(
            np.sign(a1['curvature_energy_fd_Eh_per_A2'])
            == np.sign(a2['curvature_energy_fd_Eh_per_A2'])
            == np.sign(a1['curvature_gradient_fd_Eh_per_A2'])
            == np.sign(a2['curvature_gradient_fd_Eh_per_A2']))
    return out


# center energies per gear (own center for each gear)
OUT_E = {}


def main():
    global OUT_E
    p59 = json.load(open(os.path.join(OUT, 'order59_points.json')))
    if len(p59) < 5 or any('e_total' not in p for p in p59):
        raise SystemExit('order59 points incomplete: %d' % len(p59))
    dpoints = json.load(open(os.path.join(OUT, 'direction_points.json')))
    c47 = json.load(open(os.path.join(OUT, 'center_order47.json')))
    c41 = json.load(open(os.path.join(OUT, 'center_order41.json')))
    analysis = json.load(open(os.path.join(OUT, 'analysis.json')))
    hb = json.load(open(os.path.join(OUT, 'hessian_batch.json')))

    center = np.asarray(c47['coords_angstrom'], float)
    dvec = np.asarray([d for d in analysis['direction_selection']['directions']
                       if d.get('mode_index') is not None][0]
                      ['cartesian_direction_6x3'], float)

    # gear 47 center = the order47 center record (JOB-013); its raw displaced
    # points live in direction_points.json (gear==47)
    OUT_E = {47: c47['e_total'], 41: c41['e_total'],
             59: [p for p in p59 if p['kind'] == 'center'][0]['e_total']}
    g_centers = {47: c47['gradient_full_6x3'],
                 41: c41['gradient_full_6x3'],
                 59: [p for p in p59 if p['kind'] == 'center'][0]
                     ['gradient_full_6x3']}

    results = dict(job='JOB-2026-0906-015 Task C (offline analysis)',
                   question='order-59 fixed-direction curvature evidence for '
                            'the c06 soft mode',
                   wording_policy='only observed statements: gears agree in '
                                  'sign / sign flip persists / magnitude '
                                  'still sensitive; single gear, monotonic '
                                  'change or three-gear agreement is NOT a '
                                  'convergence proof',
                   gears={})
    for gear, pts in ((41, dpoints), (47, dpoints), (59, p59)):
        results['gears'][str(gear)] = gear_curvature(
            center, g_centers[gear], dvec, pts, gear)

    # three-gear comparison table
    table = []
    for amp in AMPS:
        row = dict(amp_A=amp)
        for gear in (41, 47, 59):
            v = results['gears'][str(gear)]['per_amp'].get(str(amp))
            if v:
                row['gear%d' % gear] = dict(
                    curvE=v['curvature_energy_fd_Eh_per_A2'],
                    curvG=v['curvature_gradient_fd_Eh_per_A2'])
        table.append(row)
    results['three_gear_table'] = table

    # observed statements (wording policy enforced)
    signs = []
    for gear in (41, 47, 59):
        v = results['gears'][str(gear)]['per_amp']
        if v:
            s = set(np.sign(v[a]['curvature_energy_fd_Eh_per_A2'])
                    for a in v)
            signs.append((gear, s))
    obs = []
    sign47 = results['gears']['47']['per_amp']['0.002'][
        'curvature_energy_fd_Eh_per_A2']
    sign59 = results['gears']['59']['per_amp']['0.002'][
        'curvature_energy_fd_Eh_per_A2']
    sign41 = results['gears']['41']['per_amp']['0.002'][
        'curvature_energy_fd_Eh_per_A2']
    if np.sign(sign59) == np.sign(sign47):
        obs.append('order59 与 order47 所测档位同号（曲率观察）')
    else:
        obs.append('order59 与 order47 之间仍存在符号翻转（曲率观察）')
    if np.sign(sign59) == np.sign(sign41):
        obs.append('order59 与 order41 所测档位同号（曲率观察）')
    else:
        obs.append('order59 与 order41 之间仍存在符号翻转（曲率观察）')
    mags = [abs(results['gears'][str(g)]['per_amp']['0.002']
                ['curvature_energy_fd_Eh_per_A2']) for g in (41, 47, 59)]
    r = max(mags) / max(min(mags), 1e-300)
    if r > 3:
        obs.append('量值仍敏感（三档最大/最小比值 %.1f）' % r)
    obs.append('单一新增档位、单调变化或三档同号均不构成收敛证明；'
               '本批不宣布真实虚频、过渡态、完整极小值或通过驻点验收')
    results['observed_statements'] = obs
    results['fixed_direction_limit'] = (
        'fixed-direction curvature only characterises the local behaviour of '
        'THIS geometry, THIS direction and THIS numerical configuration; it '
        'does not revise acceptance standards and does not alone prove a '
        'full stationary point or a physical minimum')

    out = os.path.join(OUT, 'order59_analysis.json')
    json.dump(results, open(out, 'w'), indent=2)
    print('ANALYSIS ->', out)
    for gear in (41, 47, 59):
        g = results['gears'][str(gear)]
        print('gear%d: proj_grad=%+.4e' % (gear,
              g['projected_gradient_Eh_per_A']))
        for amp in sorted(g['per_amp']):
            v = g['per_amp'][amp]
            print('   amp%s: slope=%+.3e curvE=%+.4e curvG=%+.4e'
                  % (amp, v['slope_fd_Eh_per_A'],
                     v['curvature_energy_fd_Eh_per_A2'],
                     v['curvature_gradient_fd_Eh_per_A2']))
    for o in obs:
        print(' *', o)


if __name__ == '__main__':
    main()
