#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JOB-2026-0906-010: no-SCF configuration gate for the forward solvent
surface refinement control (baseline order 41 -> target order 47).

Rules enforced BEFORE any calculation:
  * the lebedev order must be one the local PySCF actually supports
    (gen_grid.LEBEDEV_ORDER);
  * the ACTUAL value read from the constructed object must equal the
    requested target (never trust a script constant);
  * the target order must be STRICTLY ABOVE the baseline order (41), so a
    request for 31 -- or any coarsening -- fails pre-SCF;
  * if the assignment silently failed and the object still reports 41,
    the check fails.
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, 'run_artifacts', '01_pure_water_o3_h2o')
OUT = os.path.join(ART, 'smd_surface_refine')
sys.path.insert(0, HERE)

from pyscf import gto
from pyscf.dft import gen_grid
from grad_factory import make_mf_d2_gr

SYMS = ['O', 'O', 'O', 'O', 'H', 'H']
BASELINE_ORDER = 41
TARGET_ORDER = 47
GRID_LEVEL = 8


def supported_orders():
    return set(gen_grid.LEBEDEV_ORDER.keys())


def build(coords_A, order, grid_level=GRID_LEVEL):
    atom = "; ".join("%s %.10f %.10f %.10f" % (s, x, y, z)
                     for s, (x, y, z) in zip(SYMS, coords_A))
    mol = gto.M(atom=atom, basis='def2-TZVP', charge=0, spin=0,
                verbose=0, max_memory=4000)
    mf = make_mf_d2_gr(mol, solvent='water', grid_response=True,
                       grid_level=grid_level)
    if order is not None:
        mf.with_solvent.lebedev_order = order
    return mol, mf


def check_surface_config(mf, expected_order):
    """Basic gate: the order must be locally supported AND the ACTUAL value
    read from the constructed object must equal the expectation."""
    actual = int(mf.with_solvent.lebedev_order)
    errs = []
    if expected_order not in supported_orders():
        errs.append('order %s not supported by local PySCF' % expected_order)
    if actual != expected_order:
        errs.append('expected order %s but ACTUAL object reports %s'
                    % (expected_order, actual))
    if errs:
        raise AssertionError('SURFACE CONFIG CHECK FAILED: %s' % errs)
    return dict(order_expected=expected_order, order_actual=actual,
                points_per_sphere=gen_grid.LEBEDEV_ORDER[expected_order],
                pass_=True)


def check_refinement(mf, target_order, baseline_order=BASELINE_ORDER):
    """Refinement gate: basic gate PLUS target must be strictly ABOVE the
    baseline order (a coarsening request fails before any calculation)."""
    r = check_surface_config(mf, target_order)
    if target_order <= baseline_order:
        raise AssertionError(
            'SURFACE CONFIG CHECK FAILED: target order %s is not ABOVE '
            'baseline order %s (coarsening is not a refinement)'
            % (target_order, baseline_order))
    r['baseline_order'] = baseline_order
    r['points_per_sphere_baseline'] = gen_grid.LEBEDEV_ORDER[baseline_order]
    return r


def main():
    os.makedirs(OUT, exist_ok=True)
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.24], [0.0, 0.0, -1.24],
                       [3.1, 0.2, 0.0], [3.9, 0.7, 0.1], [2.5, -0.4, 0.3]])
    PASS = []

    def check(name, ok, detail=''):
        PASS.append(bool(ok))
        print('[%s] %s %s' % ('PASS' if ok else 'FAIL', name, detail))

    # T1 baseline 41 accepted by the config reader
    _, mf41 = build(coords, BASELINE_ORDER)
    try:
        check_surface_config(mf41, BASELINE_ORDER)   # basic gate only
        check('T1 baseline order 41 config check passes', True,
              '(actual=%d)' % int(mf41.with_solvent.lebedev_order))
    except AssertionError as e:
        check('T1 baseline order 41 config check passes', False, repr(e))

    # T2 target 47 passes and really takes effect
    _, mf47 = build(coords, TARGET_ORDER)
    try:
        r = check_refinement(mf47, TARGET_ORDER)     # refinement gate
        check('T2 target order 47 config check passes and is in effect',
              True, '(actual=%d, %d pts/sphere)'
              % (r['order_actual'], r['points_per_sphere']))
    except AssertionError as e:
        check('T2 target order 47 config check passes and is in effect',
              False, repr(e))

    # T3 a request for 31 must FAIL pre-SCF (coarsening)
    _, mf31 = build(coords, 31)
    try:
        check_refinement(mf31, 31)
        check('T3 order 31 fails pre-SCF (coarsening)', False)
    except AssertionError as e:
        check('T3 order 31 fails pre-SCF (coarsening)', True,
              '(%s)' % str(e)[:80])

    # T4 assignment silently failed -> object still reports 41 while 47
    # was requested -> must fail
    _, mf47b = build(coords, TARGET_ORDER)
    mf47b.with_solvent.lebedev_order = BASELINE_ORDER   # simulate failure
    try:
        check_refinement(mf47b, TARGET_ORDER)
        check('T4 failed assignment (actual still 41) fails', False)
    except AssertionError as e:
        check('T4 failed assignment (actual still 41) fails', True,
              '(%s)' % str(e)[:80])

    # T5 local support and monotonicity of the point counts
    sup = supported_orders()
    ok = (BASELINE_ORDER in sup and TARGET_ORDER in sup
          and gen_grid.LEBEDEV_ORDER[TARGET_ORDER]
          > gen_grid.LEBEDEV_ORDER[BASELINE_ORDER])
    check('T5 41 and 47 supported and 47 has more points', ok,
          '(%d -> %d pts/sphere)'
          % (gen_grid.LEBEDEV_ORDER[BASELINE_ORDER],
             gen_grid.LEBEDEV_ORDER[TARGET_ORDER]))

    print('SURFACE CONFIG REGRESSION: %s (%d/%d pass)'
          % ('PASS' if all(PASS) else 'FAIL', sum(PASS), len(PASS)))
    json.dump(dict(all_pass=bool(all(PASS)), n_pass=sum(PASS),
                   n_total=len(PASS), baseline_order=BASELINE_ORDER,
                   target_order=TARGET_ORDER),
              open(os.path.join(OUT, 'surface_config_regression.json'),
                   'w'), indent=2)
    sys.exit(0 if all(PASS) else 1)


if __name__ == '__main__':
    main()
