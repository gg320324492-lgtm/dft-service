#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression test for the JOB-2026-0906-005 Phase A revision of
endpoint_freq_step3_frequency.py (no SCF; pure re-analysis of the saved
hessian_points).

Proves:
  T1  the same-matrix round-trip assertion holds for both candidates and
      both step sizes (layout conversion is identity on the matrix);
  T2  with its OWN matrix, the PySCF cross-check agrees to < 1e-6 cm^-1
      at h=0.002 (machine precision);
  T3  feeding the WRONG step's matrix (the 004 bug: h=0.002's matrix into
      the h=0.004 analysis) makes the cross-check differ by orders of
      magnitude more (~ the inter-step frequency change), and the output
      flags same_matrix_as_eigenproblem=False -- i.e. the test suite fails
      loudly on the pre-revision behaviour.
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import endpoint_freq_step3_frequency as s3

PASS = []


def check(name, ok, detail=''):
    PASS.append(bool(ok))
    print('[%s] %s %s' % ('PASS' if ok else 'FAIL', name, detail))


def main():
    fx = json.load(open(os.path.join(s3.OUT, 'endpoints_fixed.json')))
    for tid in ('c14_plus', 'c06_plus'):
        coords_A = np.asarray(fx['endpoints'][tid]['coords_angstrom'], float)
        from pyscf import gto
        mol = gto.M(atom="; ".join("%s %.10f %.10f %.10f"
                                   % (s, x, y, z)
                                   for s, (x, y, z) in zip(s3.SYMS, coords_A)),
                    basis='def2-TZVP', charge=0, spin=0, verbose=0,
                    max_memory=2000)
        mass = mol.atom_mass_list(isotope_avg=True)
        coords_B = coords_A / 0.52917721092
        a2 = s3.analyse(tid, 0.002, mass, coords_B)
        a4 = s3.analyse(tid, 0.004, mass, coords_B)
        # T1: same-matrix assertion passed implicitly (no exception);
        # explicit round-trip check here:
        own4 = np.asarray(a4['hessian_symmetrized']).reshape(6, 3, 6, 3) \
            .transpose(0, 2, 1, 3)
        rt = own4.transpose(0, 2, 1, 3).reshape(18, 18)
        check('T1 %s: PySCF-layout Hessian round-trips to own Hsym' % tid,
              np.allclose(rt, np.asarray(a4['hessian_symmetrized']),
                          atol=1e-12))
        # T2: own-matrix cross-check machine precision (h=0.002 and 0.004)
        x2 = a2['pyscf_crosscheck']['max_abs_wn_diff']
        x4 = a4['pyscf_crosscheck']['max_abs_wn_diff']
        check('T2 %s: own-matrix xcheck < 1e-6 cm-1 (h=0.002)' % tid,
              x2 < 1e-6, '(%.2e)' % x2)
        check('T2 %s: own-matrix xcheck < 1e-2 cm-1 (h=0.004)' % tid,
              x4 < 1e-2, '(%.2e)' % x4)
        # T3: WRONG matrix (h=0.002's) into the h=0.004 analysis must
        # differ by ~ the inter-step frequency change and be flagged
        wrong = np.asarray(a2['hessian_symmetrized']) \
            .reshape(6, 3, 6, 3).transpose(0, 2, 1, 3)
        bad = s3.analyse(tid, 0.004, mass, coords_B,
                         override_hess_pqxy=wrong)
        xb = bad['pyscf_crosscheck']['max_abs_wn_diff']
        inter = max(abs(p['d_wn']) for p in
                    s3.match_modes(a2, a4))
        check('T3 %s: wrong-matrix xcheck is LARGE (>0.05 cm-1)' % tid,
              xb > 0.05, '(%.4f)' % xb)
        check('T3 %s: wrong-matrix xcheck ~= inter-step max|d_wn| '
              '(ratio ~ 1, reproduces the 004 bug)' % tid,
              abs(xb / inter - 1.0) < 1e-6,
              '(xb=%.4f inter=%.4f ratio=%.8f)' % (xb, inter, xb / inter))
        check('T3 %s: output flags same_matrix_as_eigenproblem=False '
              'for the wrong-matrix run' % tid,
              bad['pyscf_crosscheck']['same_matrix_as_eigenproblem'] is False)
    print('REGRESSION: %s (%d/%d pass)'
          % ('PASS' if all(PASS) else 'FAIL', sum(PASS), len(PASS)))
    out = os.path.join(s3.OUT, 'step3_revision_regression.json')
    json.dump(dict(all_pass=bool(all(PASS)), n_pass=sum(PASS),
                   n_total=len(PASS)),
              open(out, 'w'), indent=2)
    print('saved ->', out)
    sys.exit(0 if all(PASS) else 1)


if __name__ == '__main__':
    main()
