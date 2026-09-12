#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JOB-2026-0906-007 Phase A: no-SCF configuration regression.

Proves the pre-SCF configuration assertion FAILS when one side is
deliberately built at a different grid level (the 006 defect: production
L7 vs native L8), i.e. the mismatch is caught BEFORE any calculation and
cannot be masked by global defaults.
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, 'run_artifacts', '01_pure_water_o3_h2o')
sys.path.insert(0, HERE)

from smd_config_audit_verify import build, object_config, config_check

PASS = []


def check(name, ok, detail=''):
    PASS.append(bool(ok))
    print('[%s] %s %s' % ('PASS' if ok else 'FAIL', name, detail))


def main():
    coords = np.zeros((6, 3))
    coords[0] = [0, 0, 0]; coords[1] = [0, 0, 1.2]; coords[2] = [0, 0, -1.2]
    coords[3] = [3.0, 0, 0]; coords[4] = [3.757, 0.587, 0]
    coords[5] = [2.243, 0.587, 0]
    # T1: matching L8/L8 passes the pre-SCF assertion
    _, mf_p = build(coords, with_d2=True, grid_level=8)
    _, mf_n = build(coords, with_d2=False, grid_level=8)
    try:
        cfg = config_check(mf_p, mf_n, expect_grid_level=8)
        check('T1 L8/L8 same-config assertion passes', True,
              '(dft_grid_level=%d)' % cfg['production']['dft_grid_level'])
    except AssertionError as e:
        check('T1 L8/L8 same-config assertion passes', False, repr(e))
    # T2: deliberate L7 vs L8 MUST fail BEFORE any SCF
    _, mf_p7 = build(coords, with_d2=True, grid_level=7)
    failed = False
    try:
        config_check(mf_p7, mf_n, expect_grid_level=8)
    except AssertionError as e:
        failed = True
        check('T2 L7-vs-L8 config assertion fails pre-SCF', True,
              '(%s)' % str(e)[:100])
    check('T2 L7-vs-L8 config assertion fails pre-SCF', failed)
    # T3: the L7 object's ACTUAL config really is 7 (read from object)
    check('T3 production-L7 object reports grid_level=7',
          object_config(mf_p7)['dft_grid_level'] == 7)
    # T4: fingerprints come from objects, not constants
    c8 = object_config(mf_n)
    check('T4 native-L8 object reports grid_level=8',
          c8['dft_grid_level'] == 8)
    print('CONFIG REGRESSION: %s (%d/%d pass)'
          % ('PASS' if all(PASS) else 'FAIL', sum(PASS), len(PASS)))
    os.makedirs(os.path.join(ART, 'smd_config_audit'), exist_ok=True)
    out = os.path.join(ART, 'smd_config_audit', 'config_regression.json')
    json.dump(dict(all_pass=bool(all(PASS)), n_pass=sum(PASS),
                   n_total=len(PASS)),
              open(out, 'w'), indent=2)
    print('saved ->', out)
    sys.exit(0 if all(PASS) else 1)


if __name__ == '__main__':
    main()
