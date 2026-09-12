#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
No-SCF regression for the repaired method_fingerprint (JOB-2026-0906-012).

Covers:
  T1 gas-phase object  -> solvent.model == 'gas'
  T2 SMD(water) object -> model 'smd', solvent 'water', lebedev_order recorded
  T3 surface gears 41 vs 47 -> DIFFERENT full fingerprints (never equal)
  T4 fingerprint identical before attach-GR / after attach-GR / after
     scanner copy (configuration traceable through wrapping and copying)
  T5 fingerprint is JSON-serialisable (index requirement)

NO SCF is run anywhere in this test: objects are only constructed.
"""
import os
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from pyscf import gto
from grad_factory import make_mf_d2_gr, method_fingerprint
from d2_full import make_mf_d2

MOL = gto.M(atom="O 0 0 0; H 0 0 0.9; H 0 0.9 0", basis="sto-3g", verbose=0)


def fp_eq(a, b):
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def run():
    ok = True

    # T1 gas
    mfg = make_mf_d2_gr(MOL, solvent=None, grid_response=True, grid_level=8)
    fpg = method_fingerprint(mfg)
    t1 = fpg['solvent']['model'] == 'gas'
    print('[T1] gas labeled gas:', 'PASS' if t1 else 'FAIL', fpg['solvent'])
    ok &= t1

    # T2 SMD water
    mfs = make_mf_d2_gr(MOL, solvent='water', grid_response=True, grid_level=8)
    fps = method_fingerprint(mfs)
    t2 = (fps['solvent']['model'] == 'smd'
          and fps['solvent']['solvent'] == 'water'
          and fps['solvent']['lebedev_order'] is not None)
    print('[T2] smd labeled water + surface order:', 'PASS' if t2 else 'FAIL',
          fps['solvent'])
    ok &= t2

    # T3 different surface gears -> different fingerprints
    mfs47 = make_mf_d2_gr(MOL, solvent='water', grid_response=True,
                          grid_level=8)
    mfs47.with_solvent.lebedev_order = 47
    fp47 = method_fingerprint(mfs47)
    t3 = (not fp_eq(fps, fp47)
          and fps['solvent']['lebedev_order'] == 41
          and fp47['solvent']['lebedev_order'] == 47)
    print('[T3] gears 41 vs 47 differ:', 'PASS' if t3 else 'FAIL',
          fps['solvent']['lebedev_order'], 'vs',
          fp47['solvent']['lebedev_order'])
    ok &= t3

    # T4 traceability: plain D2 attach vs D2+GR attach vs scanner copy
    mfp = make_mf_d2(MOL, solvent='water')          # D2 only
    mfp.grids.level = 8
    mfp.conv_tol = 1e-12                            # align tolerances so the
    mfp.conv_tol_grad = 1e-9                        # configs are IDENTICAL
    fp_plain = method_fingerprint(mfp)
    gsc = mfs.nuc_grad_method().as_scanner()        # scanner copy
    fp_scan = method_fingerprint(gsc.base)
    t4 = fp_eq(fps, fp_plain) and fp_eq(fps, fp_scan)
    print('[T4] wrap/scanner traceable:', 'PASS' if t4 else 'FAIL')
    if not t4:
        for k in fps:
            if not fp_eq({k: fps[k]}, {k: fp_plain.get(k)}):
                print('   plain differs at', k, ':', fp_plain.get(k),
                      'vs', fps[k])
            if not fp_eq({k: fps[k]}, {k: fp_scan.get(k)}):
                print('   scan differs at', k, ':', fp_scan.get(k),
                      'vs', fps[k])
    ok &= t4

    # T5 JSON-serialisable
    try:
        json.dumps(fpg); json.dumps(fps); json.dumps(fp47)
        t5 = True
    except Exception as e:
        t5 = False
        print('   json error:', e)
    print('[T5] JSON-serialisable:', 'PASS' if t5 else 'FAIL')
    ok &= t5

    print('METHOD FINGERPRINT REGRESSION:', 'ALL PASS' if ok else 'FAILED')
    if not ok:
        raise SystemExit(1)
    return dict(gas=fpg, smd41=fps, smd47=fp47)


if __name__ == '__main__':
    run()
