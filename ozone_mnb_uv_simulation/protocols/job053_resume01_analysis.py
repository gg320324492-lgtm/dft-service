#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-053 resume-01 step C: OFFLINE analysis (ZERO evaluations).

The four authorized displacement SCF+gradient evaluations COMPLETED and
were persisted; the in-memory analysis handoff then crashed on a second
return-convention bug (run_resume returned a tuple).  This script
completes the analysis PURE OFFLINE from the persisted records (no new
evaluations, no budget), records the second control-flow anomaly, and
finalizes resume01_results.json.
"""
import os, sys, json
import numpy as np

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
R53 = ROOT + '/run_artifacts/02_nh3o3_reference/c1_negmode_dir053'
RR = ROOT + '/run_artifacts/02_nh3o3_reference/c1_negmode_dir053_resume01'


def save_json_atomic(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(obj, fh, indent=2, default=str)
    os.replace(tmp, path)


def main():
    man = json.load(open(RR + '/input_manifest_resume01.json'))
    res = json.load(open(RR + '/resume01_results.json'))
    q = np.asarray(man['direction']['q'], float).reshape(-1)
    kh = man['kH_from_052_matrices']

    centre = json.load(open(R53 + '/eval_centre.json'))
    recs = {}
    for tag in ('disp_p01', 'disp_m01', 'disp_p02', 'disp_m02'):
        recs[tag] = json.load(open(RR + '/eval_%s.json' % tag))

    E0 = float(centre['e_total'])
    g0 = np.asarray(centre['grad'], float).reshape(-1)
    a0 = float(g0 @ q)
    per_h = {}
    for h in (0.01, 0.02):
        p = recs['disp_p%02d' % int(h * 100)]
        m = recs['disp_m%02d' % int(h * 100)]
        Ep, Em = float(p['e_total']), float(m['e_total'])
        gp = np.asarray(p['grad'], float).reshape(-1)
        gm = np.asarray(m['grad'], float).reshape(-1)
        k = '%g' % h
        kh_tot = kh['1e-3' if h == 0.01 else '5e-4']['kH_total_Eh_Bohr2']
        per_h[k] = dict(
            h_Bohr=h,
            dE_plus=Ep - E0, dE_minus=Em - E0,
            aE_Eh_Bohr=(Ep - Em) / (2 * h),
            kE_Eh_Bohr2=(Ep + Em - 2 * E0) / h ** 2,
            kg_Eh_Bohr2=float((gp - gm) @ q / (2 * h)),
            kH_total_Eh_Bohr2=kh_tot)
        per_h[k]['kE_minus_kH'] = per_h[k]['kE_Eh_Bohr2'] - kh_tot
        per_h[k]['kg_minus_kH'] = per_h[k]['kg_Eh_Bohr2'] - kh_tot
    per_h['0.01']['kE_step_sensitivity'] = per_h['0.02']['kE_Eh_Bohr2'] \
        - per_h['0.01']['kE_Eh_Bohr2']
    per_h['0.01']['kg_step_sensitivity'] = per_h['0.02']['kg_Eh_Bohr2'] \
        - per_h['0.01']['kg_Eh_Bohr2']

    kE_neg = all(per_h[k]['kE_Eh_Bohr2'] < 0 for k in ('0.01', '0.02'))
    kg_neg = all(per_h[k]['kg_Eh_Bohr2'] < 0 for k in ('0.01', '0.02'))
    if kE_neg and kg_neg:
        reg = ('direction local negative curvature: independent FD support '
               '(both steps, energy and gradient); magnitude consistency '
               'with kH listed separately')
    else:
        reg = 'direction curvature sign: pending'

    analysis = dict(a0_Eh_Bohr=a0, per_h=per_h,
                    note='nonzero a0 retained; per-step/per-estimator '
                         'differences listed separately (never merged)')
    res['comparisons'] = analysis
    res['kH_from_052'] = kh
    res['registration'] = dict(registered=reg,
                               kE_negative_both_steps=kE_neg,
                               kg_negative_both_steps=kg_neg)
    # two control-flow anomalies recorded (see control_flow_anomalies)
    res['control_flow_anomalies'] = dict(
        count=2,
        list=[
            dict(where='after the centre evaluation (original run)',
                 what='post-eval wrapper exception: triple return unpacked '
                      'as dict; centre record was already persisted'),
            dict(where='after the four displacement evaluations '
                       '(resume-01)',
                 what='analysis handoff exception: run_resume returned '
                      '(dict, centre_rec) but the caller indexed a tuple; '
                      'ALL FOUR authorized displacement evaluations had '
                      'completed and persisted; this offline analysis '
                      'consumes no evaluations')])
    res['final'] = dict(
        status='completed',
        registration=reg,
        accounting=dict(
            centre_evaluations=1,
            displacement_evaluations=4,
            new_failures_resume01=0,
            control_flow_anomalies=2,
            cumulative_evaluations_vs_original_cap='5/5 (centre 1 + '
                                                   'displacements 4)',
            whole_batch_zero_failures_claim=False,
            zero_eval_offline_analysis='this step (no evaluations)'),
        not_claimed='verified frequency / transition state / minimum '
                    'acceptance / whole-Hessian correctness / mechanism')
    save_json_atomic(RR + '/resume01_analysis.json', analysis)
    save_json_atomic(RR + '/resume01_results.json', res)

    print('[053r] a0 = %+.6e Eh/Bohr' % a0)
    for k in ('0.01', '0.02'):
        v = per_h[k]
        print('[053r] h=%s: dE+=%+.4e dE-=%+.4e aE=%+.4e kE=%+.5e '
              'kg=%+.5e | kH=%+.5e (kE-kH=%+.3e kg-kH=%+.3e)'
              % (k, v['dE_plus'], v['dE_minus'], v['aE_Eh_Bohr'],
                 v['kE_Eh_Bohr2'], v['kg_Eh_Bohr2'], v['kH_total_Eh_Bohr2'],
                 v['kE_minus_kH'], v['kg_minus_kH']))
    print('[053r] registration:', reg)


if __name__ == '__main__':
    main()
