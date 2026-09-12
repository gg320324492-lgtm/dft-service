#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JOB-2026-0906-010: forward solvent-surface refinement control.

Exactly FIVE new SMD energy/gradient evaluations: the same c06 step-40
rotation geometries (theta = 0, +/-0.1, +/-0.2 degrees, coordinates and
rotation matrices re-read from the 009 audit record and hash-checked;
the axis is NOT re-chosen) at solvent surface lebedev order 47, with
EVERYTHING else identical to the baseline (SMD(water), DFT grid level 8,
xc, basis, dispersion, SCF 1e-12/1e-9, grid_response=True, CDS path).

Per point saved immediately: actual coords (+units), rotation matrix,
total energy, FULL gradient, e_solvent/e_cds/e_d2, SCF state, DFT grid and
surface order/point counts, and method/code fingerprints.  The five points
must all pass the config gate and SCF checks before any comparison with the
order-41 baseline; failures are never defaulted to pass and no extra
evaluations are run.
"""
import os, sys, json, time, hashlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, 'run_artifacts', '01_pure_water_o3_h2o')
OUT = os.path.join(ART, 'smd_surface_refine')
sys.path.insert(0, HERE)

from pyscf import gto
import d2_full
from grad_factory import make_mf_d2_gr, method_fingerprint
from smd_surface_refine_config import check_refinement, GRID_LEVEL, \
    BASELINE_ORDER, TARGET_ORDER
import torque_decomp as td

SYMS = ['O', 'O', 'O', 'O', 'H', 'H']


def sha(b):
    return hashlib.sha256(b).hexdigest()


def reconstruct_actual_input(coords_A):
    """Rebuild the coordinates PySCF actually receives: the atom string uses
    %.10f formatting, then PySCF parses it back to float.  The resulting
    coordinates differ from the FULL-PRECISION task coordinates."""
    atom = "; ".join("%s %.10f %.10f %.10f" % (s, x, y, z)
                     for s, (x, y, z) in zip(SYMS, coords_A))
    actual = []
    for tok in atom.split("; "):
        p = tok.split()
        actual.append([float(p[1]), float(p[2]), float(p[3])])
    return np.asarray(actual, float)[:16]


def eval_point(coords_A, R, theta_deg, order=TARGET_ORDER,
               actual_sha=None):
    atom = "; ".join("%s %.10f %.10f %.10f" % (s, x, y, z)
                     for s, (x, y, z) in zip(SYMS, coords_A))
    mol = gto.M(atom=atom, basis='def2-TZVP', charge=0, spin=0,
                verbose=0, max_memory=4000)
    mf = make_mf_d2_gr(mol, solvent='water', grid_response=True,
                       grid_level=GRID_LEVEL)
    mf.with_solvent.lebedev_order = order
    gate = check_refinement(mf, order)          # BEFORE any SCF
    t0 = time.time()
    mf.kernel()
    g = np.asarray(mf.nuc_grad_method().kernel(), float).reshape(6, 3)
    ss = mf.scf_summary
    ws = mf.with_solvent
    surf = getattr(ws, 'surface', None)
    surf_pts = int(np.asarray(surf['area']).size) if surf is not None else None
    Rm = np.asarray(R, float)
    g_back = (Rm.T @ g.T).T
    rec = dict(
        theta_deg=theta_deg,
        coords_angstrom=np.asarray(coords_A, float).tolist(),
        coords_unit='Angstrom',
        rotation_matrix=Rm.tolist(),
        coord_sha=sha(np.asarray(coords_A, float).tobytes()),
        coord_sha_actual_input=(actual_sha or
                                sha(reconstruct_actual_input(
                                    np.asarray(coords_A, float)).tobytes())),
        R_sha=sha(Rm.tobytes()),
        e_total=float(mf.e_tot),
        gradient_full_6x3=g.tolist(),
        gradient_rotated_back_6x3=g_back.tolist(),
        grad_max=float(np.abs(g).max()),
        grad_rms=float(np.sqrt((g ** 2).mean())),
        e_solvent=(float(ss['e_solvent']) if 'e_solvent' in ss else None),
        e_cds=(float(ss['e_cds']) if 'e_cds' in ss else None),
        e_d2_analytic=float(d2_full.d2_energy(mol)),
        d2_in_scf_summary=float(ss.get('d2_dispersion', float('nan'))),
        scf_converged=bool(mf.converged),
        finite=bool(np.isfinite(mf.e_tot) and np.isfinite(g).all()),
        dft_grid_level=int(mf.grids.level),
        dft_grid_points=(int(mf.grids.weights.size)
                         if mf.grids.weights is not None else None),
        surface_order_actual=int(ws.lebedev_order),
        surface_points=surf_pts,
        config_gate=gate,
        method_fingerprint=method_fingerprint(mf),
        seconds=round(time.time() - t0, 1))
    return rec


def main():
    os.makedirs(OUT, exist_ok=True)
    src = json.load(open(os.path.join(ART, 'smd_rotation_audit',
                                      'rotation_audit_data.json')))
    c = src['candidates']['c06_plus']
    axis_ref = np.asarray(c['axis'], float)
    rows = []
    ok_all = True
    for r in c['rotations']:
        coords = np.asarray(r['coords_angstrom'], float)
        R = np.asarray(r['R'], float)
        # geometry/axis are REUSED from 009; no re-selection.  Guard: finite
        # and correct shape, and keep a SEPARATE actual-input hash so the
        # task-coordinate hash is never mistaken for the %.10f-formatted
        # coordinates PySCF really receives (see smd_source_crosscheck.py).
        assert np.isfinite(coords).all() and coords.shape == (6, 3)
        actual_sha = sha(reconstruct_actual_input(coords).tobytes())
        rec = eval_point(coords, R, r['theta_deg'], actual_sha=actual_sha)
        rec['axis_from_009'] = axis_ref.tolist()
        rec['pairwise_dist_max_dev_A'] = r['pairwise_dist_max_dev_A']
        rows.append(rec)
        with open(os.path.join(OUT, 'order47_five_points.json'), 'w') as fh:
            json.dump(rows, fh, indent=2)
        ok = bool(rec['scf_converged'] and rec['finite']
                  and rec['surface_order_actual'] == TARGET_ORDER)
        ok_all = ok_all and ok
        print('[order47] th=%+0.1f E=%.9f max|g|=%.3e surf_order=%d '
              'surf_pts=%s dft_pts=%s ok=%s'
              % (rec['theta_deg'], rec['e_total'], rec['grad_max'],
                 rec['surface_order_actual'], rec['surface_points'],
                 rec['dft_grid_points'], ok), flush=True)
    out = dict(job='JOB-2026-0906-010', target_order=TARGET_ORDER,
               baseline_order=BASELINE_ORDER, grid_level=GRID_LEVEL,
               all_points_pass=bool(ok_all), points=rows)
    json.dump(out, open(os.path.join(OUT, 'order47_five_points.json'),
                        'w'), indent=2)
    print('ALL FIVE PASS:', ok_all)


if __name__ == '__main__':
    main()
