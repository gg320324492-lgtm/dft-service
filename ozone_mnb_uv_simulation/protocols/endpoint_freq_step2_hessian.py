#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Endpoint-freq batch Step 2 (JOB-2026-0906-004):
full central-difference Cartesian Hessian at the OFFICIAL endpoints.

For each candidate: displace along each of the 18 Cartesian directions with
h = 0.002 and 0.004 Bohr (both signs), 36 points per step size, 72 per
candidate (144 total).  H[:, j] = [g(R+h e_j) - g(R-h e_j)] / (2h)
-- i.e. column j = d(g)/d(x_j), the conventional Hessian indexing.

All gradients via the Phase-B-verified grad_factory path
(grid_response=True actually engaged, project -D2 exactly once).

Units are EXPLICIT: input coords from endpoints_fixed.json are Angstrom;
displacements are applied in BOHR via mol.set_geom_(..., unit='Bohr') and
verified against the actual mol coordinates read back (|actual - nominal|
< 1e-10 Bohr per component).

Every point is saved IMMEDIATELY with: nominal+actual coords, FULL gradient,
total energy, analytic + scf_summary dispersion, SCF state, actual
grid_response flag, method/code fingerprints.  On (re)start, existing points
are reused ONLY if method/code/coords fingerprints match; mismatched files
are recomputed and flagged.

Usage:  python endpoint_freq_step2_hessian.py [candidate|all]
"""
import os, sys, json, hashlib, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, 'run_artifacts', '01_pure_water_o3_h2o')
OUT = os.path.join(ART, 'endpoint_frequency', 'hessian_points')
sys.path.insert(0, HERE)

from pyscf import gto
import d2_full
from grad_factory import make_mf_d2_gr

SYMS = ['O', 'O', 'O', 'O', 'H', 'H']
STEPS_BOHR = [0.002, 0.004]
NDIR = 18


def sha(b):
    return hashlib.sha256(b).hexdigest()[:16]


def code_fingerprint():
    fps = {}
    for f in ('grad_factory.py', 'd2_full.py', 'endpoint_freq_step2_hessian.py'):
        fps[f] = sha(open(os.path.join(HERE, f), 'rb').read())
    return fps


def build_mol(coords_A):
    atom = "; ".join("%s %.10f %.10f %.10f" % (s, x, y, z)
                     for s, (x, y, z) in zip(SYMS, coords_A))
    return gto.M(atom=atom, basis='def2-TZVP', charge=0, spin=0,
                 verbose=0, max_memory=4000)


def main(tid):
    os.makedirs(OUT, exist_ok=True)
    fx = json.load(open(os.path.join(ART, 'endpoint_frequency',
                                     'endpoints_fixed.json')))
    ep = fx['endpoints'][tid]
    assert ep['centre_gate_pass'], 'centre gate failed for %s' % tid
    R_A = np.asarray(ep['coords_angstrom'], float)
    R_B = R_A / 0.52917721092          # nominal Bohr (display/conversion)

    mol = build_mol(R_A)
    mf = make_mf_d2_gr(mol, solvent=None, grid_response=True)
    g = mf.nuc_grad_method()           # D2-attached, grid_response=True
    scanner = g.as_scanner()
    assert bool(getattr(scanner, 'grid_response')), 'grid_response lost'

    fp = dict(method=fx['method_fingerprint'], code=code_fingerprint(),
              pyscf=__import__('pyscf').__version__,
              grid_level=7, scf_tol=[1e-12, 1e-9],
              coords_nominal_bohr_sha=sha(R_B.tobytes()))

    todo = [(h, j, sgn) for h in STEPS_BOHR for j in range(NDIR)
            for sgn in (1, -1)]
    t0 = time.time()
    n_new = n_skip = 0
    for k, (h, j, sgn) in enumerate(todo):
        key = '%s_h%.3f_dir%02d_sgn%+d' % (tid, h, j, sgn)
        path = os.path.join(OUT, 'point_%s.json' % key)
        e_j = np.zeros((6, 3)); e_j[j // 3, j % 3] = 1.0
        nominal_B = R_B + sgn * h * e_j
        rec_fp = dict(fp, step_bohr=h, direction=j, sign=sgn,
                      coords_nominal_bohr_sha=sha(nominal_B.tobytes()))
        if os.path.exists(path):
            old = json.load(open(path))
            if (old.get('fingerprint', {}) == rec_fp
                    and old.get('scf_converged')):
                n_skip += 1
                continue
            reused_mismatch = True
        else:
            reused_mismatch = False
        t1 = time.time()
        mol_p = build_mol(nominal_B * 0.52917721092)   # Angstrom input string
        # apply displacement in BOHR explicitly and verify
        mol_p.set_geom_(nominal_B, unit='Bohr')
        actual_B = np.asarray(mol_p.atom_coords(unit='Bohr'), float)
        dev = float(np.abs(actual_B - nominal_B).max())
        assert dev < 1e-10, 'displacement verification failed: %.2e' % dev
        e_tot, grad = scanner(mol_p)
        grad = np.asarray(grad, float).reshape(6, 3)
        rec = dict(
            key=key, candidate=tid, step_bohr=h, direction=j, sign=sgn,
            direction_atom=j // 3, direction_axis=j % 3,
            nominal_coords_bohr=nominal_B.tolist(),
            nominal_coords_angstrom=(nominal_B * 0.52917721092).tolist(),
            actual_coords_bohr=actual_B.tolist(),
            displacement_verification_max_dev_bohr=dev,
            coords_unit='Bohr (nominal & actual); Angstrom copy provided',
            e_total=float(e_tot),
            e_d2_analytic=float(d2_full.d2_energy(mol_p)),
            d2_in_scf_summary=float(
                scanner.base.scf_summary.get('d2_dispersion', float('nan'))),
            gradient_full_6x3=grad.tolist(),
            grad_max=float(np.abs(grad).max()),
            scf_converged=bool(scanner.converged),
            grid_response_actual=bool(getattr(scanner, 'grid_response')),
            fingerprint=rec_fp, reused_fingerprint_mismatch=reused_mismatch,
            seconds=round(time.time() - t1, 1))
        with open(path, 'w') as fh:
            json.dump(rec, fh, indent=2)
        n_new += 1
        print('[%s] %d/%d %s  E=%.9f  max|g|=%.3e  (%.1f s)'
              % (tid, k + 1, len(todo), key, e_tot, rec['grad_max'],
                 rec['seconds']), flush=True)
    print('[%s] DONE new=%d skipped=%d  (%.0f s total)'
          % (tid, n_new, n_skip, time.time() - t0), flush=True)


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    for tid in (('c14_plus', 'c06_plus') if which == 'all' else (which,)):
        main(tid)
