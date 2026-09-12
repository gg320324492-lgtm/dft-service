#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JOB-2026-0906-015 (Task B): order-59 fixed-direction curvature check at the
c06_order47_reopt endpoint.  EXACTLY 5 new E/g evaluations:
  1 center + 4 displaced points (+/-0.002, +/-0.004 Angstrom along the
  JOB-013 lowest-mode Cartesian direction).

Only the solvent surface gear changes (47 -> 59); everything else must match
JOB-013 item-by-item and is read back from the ACTUAL objects.

PRE-REUSE COMPATIBILITY GATE (before any new evaluation):
  * center coords hash == 011 endpoint / JOB-013 center hash
  * JOB-013 direction: max atom |d| = 1 (same normalization), finite
  * JOB-013 41/47 displaced points: saved coords == center + amp*d within
    1e-8 Angstrom (straight-line convention), units Angstrom
  * JOB-013 config stamps: surface order 47/41, DFT grid level 8
If any check fails in a way that cannot be explained -> STOP before new
calculations and report.

Any SCF failure / config rollback / non-finite point: save scene and stop,
no retry, no expansion.
"""
import os
import sys
import json
import hashlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, 'run_artifacts', '01_pure_water_o3_h2o')
OUT = os.path.join(ART, 'smd_diag_curvature')
sys.path.insert(0, HERE)

import smd_diag_hessian as H          # eval_at, constants, builders
import torque_decomp as td

BOHR_A = td.BOHR_A
ORDER = 59
AMPS = [0.002, 0.004]


def sha(b):
    return hashlib.sha256(b).hexdigest()[:16]


def main():
    # ---------- load JOB-013 artifacts ----------
    center47 = json.load(open(os.path.join(OUT, 'center_order47.json')))
    analysis = json.load(open(os.path.join(OUT, 'analysis.json')))
    dpoints = json.load(open(os.path.join(OUT, 'direction_points.json')))
    idx = json.load(open(os.path.join(ART, 'smd_acceptance_evidence',
                                      'endpoint_index.json')))
    e47 = [x for x in idx['entries'] if x['id'] == 'c06_order47_reopt'][0]

    center = np.asarray(center47['coords_angstrom'], float)
    gate = []
    # (1) center hash vs index and 013
    c_sha = sha(center.tobytes())
    gate.append(('center_sha==index', c_sha == e47['coords_sha_full_precision'],
                 c_sha))
    # (2) direction normalization (JOB-013 convention)
    dirs = [d for d in analysis['direction_selection']['directions']
            if d.get('mode_index') is not None]
    assert len(dirs) == 1
    dvec = np.asarray(dirs[0]['cartesian_direction_6x3'], float)
    norm_ok = bool(np.isfinite(dvec).all() and
                   abs(np.linalg.norm(dvec, axis=1).max() - 1.0) < 1e-12)
    gate.append(('direction_max_atom_norm==1', norm_ok,
                 sha(dvec.tobytes())))
    # (3) units + straight-line convention on the 013 41/47 points
    compat_pts = 0
    for p in dpoints:
        c = np.asarray(p['coords_angstrom'], float)
        intended = p['sign'] * p['amp_A'] * dvec
        err = float(np.abs(c - (center + intended)).max())
        ok = (p['disp_error_A'] <= 1e-8 and err <= 1e-8)
        if not ok:
            gate.append(('point %s straight-line' % p['tag'], False, err))
        else:
            compat_pts += 1
    gate.append(('013_displaced_points_straight_line', compat_pts == len(dpoints),
                 '%d/%d' % (compat_pts, len(dpoints))))
    # (4) 013 config stamps
    orders = set(p['surface_order_actual'] for p in dpoints)
    grids = set(p['dft_grid_level'] for p in dpoints)
    gate.append(('013_orders_41/47', orders == {41, 47}, str(sorted(orders))))
    gate.append(('013_grid_level_8', grids == {8}, str(sorted(grids))))
    # (5) center gradient/energy finite
    gate.append(('center_finite', bool(np.isfinite(center47['e_total'])
                                       and np.isfinite(
        np.asarray(center47['gradient_full_6x3'], float)).all()), None))

    failed = [g for g in gate if not g[1]]
    report = dict(gate=[dict(check=g[0], passed=g[1], detail=g[2])
                        for g in gate])
    json.dump(report, open(os.path.join(OUT, 'order59_reuse_gate.json'),
                           'w'), indent=2)
    for g in gate:
        print('GATE %-42s %s %s' % (g[0], 'PASS' if g[1] else 'FAIL', g[2]),
              flush=True)
    if failed:
        print('STOP: reuse compatibility gate FAILED -- no new evaluations '
              'run; investigate and report.', flush=True)
        return

    # ---------- 5 new evaluations at order 59 ----------
    points = []
    try:
        rec = H.eval_at(center, ORDER, 'center_order59')
        rec.update(kind='center', amp_A=0.0, sign=0,
                   coords_sha_full=sha(center.tobytes()),
                   direction_sha=sha(dvec.tobytes()))
        points.append(rec)
        print('[1/5] center_order59 E=%.9f max|g|=%.3e order=%d grid=%d pts=%s'
              % (rec['e_total'], rec['grad_max'], rec['surface_order_actual'],
                 rec['dft_grid_level'], rec['surface_points']), flush=True)
        json.dump(points, open(os.path.join(OUT, 'order59_points.json'),
                               'w'), indent=2)
        for amp in AMPS:
            for sgn in (+1, -1):
                coords_d = center + sgn * amp * dvec
                rec = H.eval_at(coords_d, ORDER,
                                'dir1_gear59_amp%0.3f_sgn%+d' % (amp, sgn))
                rec.update(kind='displaced', dir_rank=1, gear=59,
                           amp_A=amp, sign=sgn,
                           coords_sha_full=sha(np.asarray(coords_d,
                                                          float).tobytes()),
                           direction_sha=sha(dvec.tobytes()))
                actual = np.asarray(coords_d, float) - center
                err = float(np.abs(actual - sgn * amp * dvec).max())
                if err > 1e-8:
                    raise RuntimeError('displacement mismatch %.2e' % err)
                points.append(rec)
                print('[%d/5] amp%.3f sgn%+d E=%.9f max|g|=%.3e order=%d ok'
                      % (len(points), amp, sgn, rec['e_total'],
                         rec['grad_max'], rec['surface_order_actual']),
                      flush=True)
                json.dump(points, open(os.path.join(OUT, 'order59_points.json'),
                                       'w'), indent=2)
        print('ORDER59 BATCH COMPLETE: 5/5 points', flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        json.dump(dict(aborted=len(points), error=repr(e), points=points),
                  open(os.path.join(OUT, 'order59_points.json'), 'w'),
                  indent=2)
        print('ORDER59 BATCH ABORTED: %s' % e, flush=True)


if __name__ == '__main__':
    main()
