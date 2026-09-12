#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression test for the 011 source cross-check gate.

Runs with:  python protocols/test_source_crosscheck.py
(or any test runner that imports and calls run()).

Two requirements:
  1. The unperturbed 009-source vs 010-output cross-check MUST pass.
  2. Perturbing ONE coordinate element MUST make the cross-check fail.
"""
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import smd_source_crosscheck as sc


def run():
    ok = True
    # (1) clean cross-check must pass
    try:
        r = sc.crosscheck()
        assert r['errors'] == [], 'unexpected errors: %s' % r['errors']
        print('[test] clean cross-check: PASS')
    except AssertionError as e:
        ok = False
        print('[test] clean cross-check: FAIL ->', e)

    # (2) a single perturbed coordinate element must fail the cross-check
    failed_as_required = False
    try:
        sc.crosscheck(perturb=(0, 0, 1e-6))
        print('[test] perturbation regression: FAIL (did not raise)')
    except AssertionError:
        failed_as_required = True
        print('[test] perturbation regression: PASS (raised as required)')
    except Exception as e:
        ok = False
        print('[test] perturbation regression: ERROR ->', repr(e))
        traceback.print_exc()
    if not failed_as_required:
        ok = False

    # (3) report the full-precision vs actual-input distinction explicitly
    r = sc.crosscheck()
    d = r['task_full_vs_actual_input_maxdiff']
    print('[test] task(full) vs actual(10-decimal) max diff = %.3e  (task=%s, actual=%s)'
          % (d, r['task_full_sha'], r['actual_input_sha']))
    assert r['task_full_sha'] != r['actual_input_sha'], \
        'task hash must differ from actual-input hash'
    print('[test] hash distinction (task != actual input): PASS')

    if not ok:
        raise SystemExit('SOURCE CROSS-CHECK REGRESSION: FAILED')
    print('SOURCE CROSS-CHECK REGRESSION: ALL PASS')


if __name__ == '__main__':
    run()
