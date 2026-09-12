#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JOB-2026-0906-005 Phase C step 1: 72-point dual-step difference Hessian at
the NEW c14_plus L8 endpoint (only run if Phase B passed).

Same protocol as endpoint_freq_step2_hessian.py but at grid level 8 and
with the endpoint taken from the Phase-B independent verification
(c14_l8_check/l8_optimize_result.json).  Per-point immediate save with
fingerprints; failed points are excluded from matrix acceptance.
"""
import os, sys, json, hashlib, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, 'run_artifacts', '01_pure_water_o3_h2o')
OUT = os.path.join(ART, 'endpoint_frequency', 'hessian_points_l8_c14')
sys.path.insert(0, HERE)

from pyscf import gto
import d2_full
from grad_factory import make_mf_d2_gr, method_fingerprint

SYMS = ['O', 'O', 'O', 'O', 'H', 'H']
STEPS_BOHR = [0.002, 0.004]
NDIR = 18
GRID_LEVEL = 8


def sha(b):
    return hashlib.sha256(b).hexdigest()[:16]


def code_fingerprint():
    return {f: sha(open(os.path.join(HERE, f), 'rb').read())
            for f in ('grad_factory.py', 'd2_full.py',
                      'endpoint_freq_step2b_hessian_l8.py')}


def build_mol(coords_A):
    atom = "; ".join("%s %.10f %.10f %.10f" % (s, x, y, z)
                     for s, (x, y, z) in zip(SYMS, coords_A))
    return gto.M(atom=atom, basis='def2-TZVP', charge=0, spin=0,
                 verbose=0, max_memory=4000)


def main():
    os.makedirs(OUT, exist_ok=True)
    res = json.load(open(os.path.join(ART, 'c14_l8_check',
                                      'l8_optimize_result.json')))
    gate = (res.get('verdict') == 'converged_stationary_candidate_L8')
    print('Phase B gate:', res.get('verdict'), '-> proceed' if gate
          else '-> ABORT (Phase B not passed)')
    if not gate:
        sys.exit(2)
    R_A = np.asarray(res['endpoint_coords_angstrom'], float)
    R_B = R_A / 0.52917721092

    mol = build_mol(R_A)
    mf = make_mf_d2_gr(mol, solvent=None, grid_response=True)
    mf.grids.level = GRID_LEVEL
    g = mf.nuc_grad_method()
    scanner = g.as_scanner()
    assert bool(getattr(scanner, 'grid_response'))

    fp = dict(method=method_fingerprint(mf), code=code_fingerprint(),
              pyscf=__import__('pyscf').__version__,
              grid_level=GRID_LEVEL, scf_tol=[1e-12, 1e-9],
              coords_nominal_bohr_sha=sha(R_B.tobytes()),
              endpoint_source='c14_l8_check/l8_optimize_result.json')

    todo = [(h, j, sgn) for h in STEPS_BOHR for j in range(NDIR)
            for sgn in (1, -1)]
    t0 = time.time()
    n_new = n_skip = n_fail = 0
    for k, (h, j, sgn) in enumerate(todo):
        key = 'c14L8_h%.3f_dir%02d_sgn%+d' % (h, j, sgn)
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
        t1 = time.time()
        mol_p = build_mol(nominal_B * 0.52917721092)
        mol_p.set_geom_(nominal_B, unit='Bohr')
        actual_B = np.asarray(mol_p.atom_coords(unit='Bohr'), float)
        dev = float(np.abs(actual_B - nominal_B).max())
        assert dev < 1e-10
        try:
            e_tot, grad = scanner(mol_p)
            grad = np.asarray(grad, float).reshape(6, 3)
            ok = bool(scanner.converged) and np.isfinite(grad).all()
        except Exception as e:
            print('[FAIL] %s: %r' % (key, e), flush=True)
            n_fail += 1
            continue
        rec = dict(key=key, candidate='c14_plus_L8endpoint',
                   step_bohr=h, direction=j, sign=sgn,
                   direction_atom=j // 3, direction_axis=j % 3,
                   nominal_coords_bohr=nominal_B.tolist(),
                   actual_coords_bohr=actual_B.tolist(),
                   displacement_verification_max_dev_bohr=dev,
                   coords_unit='Bohr (nominal & actual)',
                   e_total=float(e_tot),
                   e_d2_analytic=float(d2_full.d2_energy(mol_p)),
                   d2_in_scf_summary=float(
                       scanner.base.scf_summary.get('d2_dispersion',
                                                    float('nan'))),
                   gradient_full_6x3=grad.tolist(),
                   grad_max=float(np.abs(grad).max()),
                   scf_converged=bool(scanner.converged),
                   point_accepted=bool(ok),
                   grid_response_actual=bool(getattr(scanner,
                                                     'grid_response')),
                   fingerprint=rec_fp, seconds=round(time.time() - t1, 1))
        with open(path, 'w') as fh:
            json.dump(rec, fh, indent=2)
        n_new += 1
        print('[c14L8] %d/%d %s  E=%.9f  max|g|=%.3e (%.1f s)%s'
              % (k + 1, len(todo), key, e_tot, rec['grad_max'],
                 rec['seconds'], '' if ok else '  [NOT ACCEPTED]'),
              flush=True)
    print('[c14L8] DONE new=%d skipped=%d failed=%d  (%.0f s)'
          % (n_new, n_skip, n_fail, time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
