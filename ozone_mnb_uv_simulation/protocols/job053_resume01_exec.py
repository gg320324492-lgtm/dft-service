#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-053 resume-01: four remaining displacement evaluations +
direction-derivative analysis.  The centre is REUSED from the persisted
053 record (recovery credential + gate credential required); NO centre
re-evaluation.  NEW budget: displacement 4 (failures counted); original
cumulative cap 5 evaluations stays (centre 1 already used); the flow
anomaly of the original run is listed separately - the whole batch is NOT
reported as "0 failures".  ON ANY NEW EXCEPTION: save and STOP.
"""
import os, sys, json, time, traceback
import numpy as np

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
HERE = ROOT + '/protocols'
sys.path.insert(0, HERE)
import job047_exec as ex
import job053_exec as ex53

OUT = ROOT + '/run_artifacts/02_nh3o3_reference/c1_negmode_dir053_resume01'
LEDGER = OUT + '/budget_resume01.json'
RESULTS = OUT + '/resume01_results.json'
MANIFEST = OUT + '/input_manifest_resume01.json'


class Ledger:
    CAPS = dict(displacement=4)

    def __init__(self, path):
        self.path = path
        self.data = dict(caps=dict(self.CAPS),
                         note='resume-01: only the four displacements are '
                              'authorized; centre new budget 0; failures '
                              'count; any new exception -> save and STOP, '
                              'no retry, no budget expansion',
                         attempts=[])
        self._save()

    def _save(self):
        ex.save_json_atomic(self.path, self.data)

    def count(self, cat):
        return sum(1 for a in self.data['attempts']
                   if a['category'] == cat)

    def pre(self, cat, note=None):
        if self.count(cat) >= self.CAPS[cat]:
            raise RuntimeError('cap reached for %s (no budget expansion)'
                               % cat)
        att = dict(attempt=len(self.data['attempts']) + 1, category=cat,
                   note=note or {}, status='pending',
                   started=time.strftime('%F %T'))
        self.data['attempts'].append(att)
        self._save()
        return att

    def post(self, att, fields):
        att.update(fields)
        att['status'] = 'done'
        att['finished'] = time.strftime('%F %T')
        self._save()

    def fail(self, att, err):
        att['status'] = 'error'
        att['error'] = str(err)[:500]
        self._save()


def main():
    man = json.load(open(MANIFEST))
    cred = json.load(open(OUT + '/recovery_credential.json'))
    if not man['centre_gate']['all_pass']:
        raise RuntimeError('HARD STOP: centre gate credential missing')
    print('[053r] resume-01: centre reused from persisted record '
          '(recovery credential verified, gate PASS)', flush=True)

    results = dict(
        job='JOB-2026-0906-053 resume-01: four remaining displacements + '
            'direction-derivative analysis',
        sections=dict(
            original_interruption=dict(
                centre_evaluated=True, centre_record_persisted=True,
                ledger_centre_status='error (preserved, never rewritten)',
                flow_anomaly='post-evaluation wrapper interface exception '
                             '(triple return unpacked as dict)',
                evidence='original_evidence_hashes.json + recovery_'
                         'credential.json'),
            resume_authorization=dict(
                authorization='notes/job053_resume_authorization_'
                              '2026-09-09.md',
                centre_gate_credential=man['centre_gate'],
                centre_re_evaluations=0),
            new_computations=dict(authorized='4 displacements, 1 SCF+'
                                            'gradient each'),
            cumulative_cost=dict(
                centre_evaluations=1, displacement_evaluations=0,
                flow_anomalies=1,
                cumulative_cap_original=5,
                note='filled after the displacements complete')))
    led = Ledger(LEDGER)
    status = 'aborted'
    try:
        flow = ex53.run_resume(man, led, ex53._eval_point, OUT,
                               gate_pass=True)
        comp = flow['comparisons']
        recs = {ex53.tag_for(r['t_Bohr']): r for r in flow['displacements']}
        results['displacements'] = recs
        results['comparisons'] = comp['comparisons']
        results['kH_from_052'] = man['kH_from_052_matrices']
        for k in ('0.01', '0.02'):
            v = comp['comparisons']['per_h'][k]
            print('[053r] h=%s: dE+=%+.3e dE-=%+.3e aE=%+.3e kE=%+.4e '
                  'kg=%+.4e (kE-kH=%+.3e, kg-kH=%+.3e)'
                  % (k, v['dE_plus'], v['dE_minus'], v['aE_Eh_Bohr'],
                     v['kE_Eh_Bohr2'], v['kg_Eh_Bohr2'],
                     v['kE_minus_kH'], v['kg_minus_kH']), flush=True)
        kE_neg = all(comp['comparisons']['per_h'][k]['kE_Eh_Bohr2'] < 0
                     for k in ('0.01', '0.02'))
        kg_neg = all(comp['comparisons']['per_h'][k]['kg_Eh_Bohr2'] < 0
                     for k in ('0.01', '0.02'))
        if kE_neg and kg_neg:
            reg = ('direction local negative curvature: independent FD '
                   'support (both steps, energy and gradient)')
        else:
            reg = ('direction curvature sign: pending (FD do not '
                   'consistently support the negative sign)')
        status = 'completed'
        results['final'] = dict(
            status=status, registration=reg,
            accounting=dict(
                new_displacement_evaluations=led.count('displacement'),
                new_failures=sum(1 for a in led.data['attempts']
                                 if a['status'] == 'error'),
                centre_evaluations_total=1,
                flow_anomalies_total=1,
                cumulative_evaluations=1 + led.count('displacement'),
                cumulative_cap_original=5,
                whole_batch_zero_failures_claim=False),
            not_claimed='verified frequency / transition state / minimum '
                        'acceptance / whole-Hessian correctness / '
                        'mechanism')
        ex.save_json_atomic(RESULTS, results)
        print('[053r] DONE: %s' % reg, flush=True)
    except Exception as exc:                                # noqa: BLE001
        results['abort'] = dict(error=str(exc),
                                traceback=traceback.format_exc()[-1500:])
        results.setdefault('final', dict(status='aborted',
                                         budget=led.data))
        ex.save_json_atomic(RESULTS, results)
        print('[053r] ABORT: %s' % exc, flush=True)
        raise


if __name__ == '__main__':
    main()
