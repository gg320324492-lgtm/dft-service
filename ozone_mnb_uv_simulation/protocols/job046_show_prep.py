#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Print key fields of the 046 prep check results (read-only)."""
import json

d = json.load(open('/mnt/e/dft-service/ozone_mnb_uv_simulation/'
                   'run_artifacts/02_nh3o3_reference/c1_gridresp046/'
                   'prep_check_results.json'))
print('q_check:', json.dumps(d['q_check'], indent=1))
h = d['h043_check']
print('k_H recomputed: total %.6e  dft %.6e  d2 %+.6e'
      % (h['k_H_total'], h['k_H_dft'], h['k_H_d2']))
print('sym resid dft %.1e  d2 %.1e' % (h['sym_resid_dft'], h['sym_resid_d2']))
print('hess unit:', h['unit'], 'shape:', h['shape'])
for tag in ('p01', 'm01'):
    g = d['geometry_rebuild'][tag]
    print('%s: vs canonical line %.3e Bohr, min pair %.3f Bohr'
          % (tag, g['max_diff_vs_canonical_line_Bohr'],
             g['min_pair_distance_Bohr']))
