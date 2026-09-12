#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Show JOB-046 gate/eval record details (read-only)."""
import json

OUT = ('/mnt/e/dft-service/ozone_mnb_uv_simulation/run_artifacts/'
       '02_nh3o3_reference/c1_gridresp046')

for tag in ('p01', 'm01'):
    scf = json.load(open('%s/eval_scf_%s.json' % (OUT, tag)))
    gon = json.load(open('%s/eval_grad_on_%s.json' % (OUT, tag)))
    gof = json.load(open('%s/eval_grad_off_%s.json' % (OUT, tag)))
    d2 = json.load(open('%s/eval_d2grad_%s.json' % (OUT, tag)))
    print('=== %s ===' % tag)
    print('  SCF: E=%.12f conv=%s grid_level=%s tol=%s'
          % (scf['e_total'], scf['converged'],
             scf['config']['grid_level'], scf['config']['scf_tol']))
    print('       chkfile: %s (size %s)' % (scf['chkfile'].get('path'),
                                            scf['chkfile'].get('size')))
    print('       mo: %s occ, HOMO %.6f, gap %.6f'
          % (scf['mo_summary']['nocc'], scf['mo_summary']['homo'],
             scf['mo_summary'].get('gap_Eh', float('nan'))))
    print('  gate vs 045: dC=%.2e dE=%.2e dgrad=%.2e pass=%s'
          % (gon['reproduction_gate']['dcoords_vs_045_A'],
             gon['reproduction_gate']['dE_vs_045'],
             gon['reproduction_gate']['dgrad_max_vs_045'],
             gon['reproduction_gate']['gates_pass']))
    print('  g_on : class=%s grid_response=%s d2=%s gmax=%.3e'
          % (gon['config']['grad_class'], gon['config']['grid_response'],
             gon['config']['d2_attached'], gon['grad_max']))
    print('  g_off: class=%s grid_response=%s d2=%s gmax=%.3e'
          % (gof['config']['grad_class'], gof['config']['grid_response'],
             gof['config']['d2_attached'], gof['grad_max']))
    print('  d2grad: gmax=%.3e scf_unchanged=%s'
          % (d2['grad_max'], d2['scf_kernel_count_unchanged']))

led = json.load(open(OUT + '/budget_scan046.json'))
print('\nbudget attempts:')
for a in led['attempts']:
    print('  #%d %-8s %-8s %s' % (a['attempt'], a['category'],
                                  a['status'], a.get('stage', '')))
