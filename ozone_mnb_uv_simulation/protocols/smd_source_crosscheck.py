#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JOB-2026-0906-011 (offline gate, NO SCF): independent source cross-check.

Reads c06 rotation coordinates, rotation matrices, angles, and the rotation
axis INDEPENDENTLY from the 009 audit source record and the 010 actual
output, compares them item by item, and records:
  * SHA-256 of BOTH source files,
  * max absolute comparison error per field (theta, coord, R, axis),
  * the distinction between the FULL-PRECISION task coordinates (what the
    010 JSON stores and hashes) and the FORMATTED coordinates actually
    passed to the PySCF Molecule object (the eval_point atom string uses
    %.10f), reporting the difference and the actual-input hash so a
    task-coordinate hash is NEVER presented as the actual-input hash.

A regression test (test_source_crosscheck.py) perturbs ONE coordinate
element and asserts the cross-check FAILS, proving the comparison is
sensitive.  Historical files are READ ONLY; nothing here re-runs 009/010.
"""
import os
import sys
import json
import hashlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, 'run_artifacts', '01_pure_water_o3_h2o')
SYMS = ['O', 'O', 'O', 'O', 'H', 'H']
SRC_009 = os.path.join(ART, 'smd_rotation_audit', 'rotation_audit_data.json')
OUT_010 = os.path.join(ART, 'smd_surface_refine', 'order47_five_points.json')
TOL = 1e-12


def sha(b):
    return hashlib.sha256(b).hexdigest()


def file_sha(path):
    return sha(open(path, 'rb').read())


def reconstruct_actual_input(coords_A):
    """Rebuild the coordinates PySCF actually received.

    In smd_surface_refine_run.py the molecule is built from
        atom = "; ".join("%s %.10f %.10f %.10f" % (s, x, y, z) ...)
    then PySCF parses that string back to float.  The resulting coordinates
    differ from the FULL-PRECISION task coordinates by %.10f rounding.
    """
    atom = "; ".join("%s %.10f %.10f %.10f" % (s, x, y, z)
                     for s, (x, y, z) in zip(SYMS, coords_A))
    actual = []
    for tok in atom.split("; "):
        p = tok.split()
        actual.append([float(p[1]), float(p[2]), float(p[3])])
    return np.asarray(actual, float)


def load_sources():
    src = json.load(open(SRC_009))
    c = src['candidates']['c06_plus']
    o = json.load(open(OUT_010))
    rot = {r['theta_deg']: r for r in c['rotations']}
    pts = {p['theta_deg']: p for p in o['points']}
    return c, rot, o, pts


def crosscheck(perturb=None):
    """Compare 009 source vs 010 output.

    `perturb` = (i, j, delta) modifies ONE coordinate element of the 009
    source COPY before comparison (regression test only; the source file is
    never touched).  Returns a result dict; raises AssertionError on any
    mismatch beyond TOL.
    """
    c, rot, o, pts = load_sources()
    src_coords = {th: np.asarray(r['coords_angstrom'], float).copy()
                  for th, r in rot.items()}
    if perturb is not None:
        i, j, delta = perturb
        th0 = sorted(src_coords)[0]
        src_coords[th0][i, j] += delta
    errs = []
    per_field = {}
    thetas = sorted(set(rot) & set(pts))
    if not thetas:
        raise AssertionError('SOURCE CROSS-CHECK FAILED: no common theta '
                             'between 009 source and 010 output')
    for th in thetas:
        c9 = src_coords[th]
        c10 = np.asarray(pts[th]['coords_angstrom'], float)
        d_coord = float(np.abs(c9 - c10).max())
        R9 = np.asarray(rot[th]['R'], float)
        R10 = np.asarray(pts[th]['rotation_matrix'], float)
        d_R = float(np.abs(R9 - R10).max())
        ax9 = np.asarray(rot[th]['axis'], float)
        ax10 = np.asarray(pts[th]['axis_from_009'], float)
        d_ax = float(np.abs(ax9 - ax10).max())
        d_th = abs(float(rot[th]['theta_deg']) - float(pts[th]['theta_deg']))
        per_field[th] = dict(d_coord=d_coord, d_R=d_R, d_axis=d_ax,
                             d_theta=float(d_th))
        for name, val in [('coord', d_coord), ('R', d_R), ('axis', d_ax),
                          ('theta', d_th)]:
            if val > TOL:
                errs.append('theta=%s %s mismatch %.3e' % (th, name, val))
    # full-precision task coords vs the %.10f-formatted actual input
    th0 = thetas[0]
    task_full = src_coords[th0]
    actual_in = reconstruct_actual_input(task_full)
    d_actual = float(np.abs(actual_in - task_full).max())
    task_sha = sha(task_full.tobytes())
    actual_sha = sha(actual_in.tobytes())
    # internal consistency of 010's own bookkeeping (reported, not gating)
    saved_coord_sha = pts[th0].get('coord_sha')
    internal_ok = (saved_coord_sha is not None and
                   saved_coord_sha == task_sha[:16])
    result = dict(
        file_sha_009=file_sha(SRC_009)[:16],
        file_sha_010=file_sha(OUT_010)[:16],
        per_field=per_field,
        task_full_vs_actual_input_maxdiff=d_actual,
        task_full_sha=task_sha[:16],
        actual_input_sha=actual_sha[:16],
        note_task_hash_is_full_precision=(
            '010 saved coord_sha = %s hashes the FULL-PRECISION task coords, '
            'NOT the 10-decimal-formatted actual input (actual-input sha = %s, '
            'diff max %.3e).  They are NOT the same and must not be conflated.'
            % (task_sha[:16], actual_sha[:16], d_actual)),
        internal_coord_sha_matches_task=bool(internal_ok),
        errors=errs)
    if errs:
        raise AssertionError('SOURCE CROSS-CHECK FAILED: %s' % errs)
    return result


def main(out_path=None):
    r = crosscheck()
    # regression: a single perturbed coordinate element MUST make it fail
    try:
        crosscheck(perturb=(0, 0, 1e-6))
        raise SystemExit('REGRESSION BROKEN: perturbed cross-check did NOT fail')
    except AssertionError:
        pass
    if out_path is None:
        out_path = os.path.join(ART, 'smd_order47_opt', 'source_crosscheck.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    r['gate'] = 'PASS'
    r['perturbation_regression'] = 'PASS (perturbed coord -> fails as required)'
    json.dump(r, open(out_path, 'w'), indent=2)
    print('CROSS-CHECK PASS ->', out_path)
    print(json.dumps({k: r[k] for k in
                      ('file_sha_009', 'file_sha_010', 'per_field',
                       'task_full_vs_actual_input_maxdiff',
                       'task_full_sha', 'actual_input_sha',
                       'internal_coord_sha_matches_task')}, indent=2))
    return r


if __name__ == '__main__':
    main()
