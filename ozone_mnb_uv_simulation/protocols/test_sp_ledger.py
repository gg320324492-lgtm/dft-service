#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
No-SCF mechanism tests for the JOB-019 driver recording fix
(protocols/sp_ledger.py + the ledger flow used by paired_single_points.py).

Scenario A: post-computation read error -> the computed values are recorded
            in the attempt (scf_grad_done + error appended), nothing is
            saved as a result; a restart counts the attempt (budget kept)
            and re-runs the point.
Scenario B: restart budget retention -> a fresh SPLedger on an existing
            ledger file does NOT regain the full budget.
Scenario C: reuse -> a saved result with matching (tag, coords, config) is
            skipped without evaluation, after verification.
No PySCF, no quantitative computation anywhere in this test.
"""
import json
import os
import tempfile

import sp_ledger
from sp_ledger import SPLedger, make_key


class FakeDriver:
    """Mimics the paired_single_points main() flow with an injectable
    evaluator and a failing surface read."""

    def __init__(self, ledger_path, cap, fail_surface_read=False):
        self.ledger = SPLedger(ledger_path, cap=cap)
        self.fail_surface_read = fail_surface_read
        self.eval_calls = 0

    def run_point(self, tag, coords_sha, config, value):
        key = make_key(tag, coords_sha,
                       (config['surface_discretization_method'],
                        config['lebedev_order'], config['grid_level'],
                        config['d2_attached']))
        prior = self.ledger.get_saved(key)
        if prior is not None:
            assert prior['meta']['coords_sha'] == coords_sha
            assert prior['meta']['config'] == config
            return ('reused', prior['result']['value'])
        if self.ledger.budget_left() <= 0:
            return ('budget_exhausted', None)
        idx = self.ledger.start_attempt(key, dict(tag=tag,
                                                  coords_sha=coords_sha,
                                                  config=config))
        try:
            self.eval_calls += 1
            result = dict(value=value, grad_max=1.0, scf_converged=True,
                          finite=True)
            # post-compute state recorded BEFORE the fragile read
            self.ledger.update(idx, status='scf_grad_done', result=result)
            if self.fail_surface_read:
                raise AttributeError("no attribute 'with_solvent'")
            self.ledger.save_result(key, dict(meta=dict(tag=tag,
                                                        coords_sha=coords_sha,
                                                        config=config),
                                              result=result))
            self.ledger.update(idx, status='saved')
            try:
                if self.fail_surface_read:
                    raise RuntimeError('unreachable')
            except Exception as se:
                self.ledger.update(idx, surface_read_error=repr(se))
            return ('saved', value)
        except Exception as e:
            self.ledger.mark_error(idx, repr(e))
            return ('error_recorded', None)


def main():
    ok = True
    tmp = tempfile.mkdtemp()
    C = dict(surface_discretization_method='SWIG', lebedev_order=41,
             grid_level=8, d2_attached=True)
    csha = 'aa11'
    path = os.path.join(tmp, 'attempts.json')

    # Scenario A: compute-then-read error, then restart
    drv = FakeDriver(path, cap=4, fail_surface_read=True)
    st, _ = drv.run_point('p1', csha, C, -1.0)
    a_ok = st == 'error_recorded' and drv.eval_calls == 1
    led = json.load(open(path))
    a_ok &= (led['attempts'][0]['status'] == 'error'
             and 'errors' in led['attempts'][0]
             and led['attempts'][0]['result']['value'] == -1.0
             and len(led['results']) == 0)
    print('[A] post-compute error: %s | attempts=%d | result NOT saved'
          % (st, led['attempts'].__len__()))
    # restart: budget kept, point re-run and saved
    drv2 = FakeDriver(path, cap=4, fail_surface_read=False)
    st2, v2 = drv2.run_point('p1', csha, C, -1.0)
    a_ok &= (st2 == 'saved' and v2 == -1.0 and drv2.eval_calls == 1)
    a_ok &= drv2.ledger.attempts_used() == 2  # error attempt NOT overwritten
    print('[A] restart: %s | attempts now %d (budget retained, error record kept)'
          % (st2, drv2.ledger.attempts_used()))
    ok &= a_ok

    # Scenario B: restart budget retention
    pathB = os.path.join(tmp, 'attempts_b.json')
    ledB = SPLedger(pathB, cap=4)
    for i in range(3):
        idx = ledB.start_attempt('x%d' % i, {})
        ledB.update(idx, status='saved')
    fresh = SPLedger(pathB, cap=4)
    b_ok = (fresh.attempts_used() == 3 and fresh.budget_left() == 1)
    print('[B] restart budget retention: attempts=%d budget_left=%d -> %s'
          % (fresh.attempts_used(), fresh.budget_left(),
             'PASS' if b_ok else 'FAIL'))
    ok &= b_ok

    # Scenario C: reuse of a saved result
    drv3 = FakeDriver(path, cap=4, fail_surface_read=False)
    before = drv3.eval_calls
    st3, v3 = drv3.run_point('p1', csha, C, -1.0)
    c_ok = (st3 == 'reused' and v3 == -1.0
            and drv3.eval_calls == before)
    print('[C] reuse without evaluation: %s -> %s'
          % (st3, 'PASS' if c_ok else 'FAIL'))
    ok &= c_ok

    print('LEDGER MECHANISM TESTS:', 'ALL PASS' if ok else 'FAILED')
    if not ok:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
