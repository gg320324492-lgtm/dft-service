#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Quick field-name check on 043 results (read-only)."""
import json
import numpy as np

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
d = json.load(open(ROOT + '/run_artifacts/02_nh3o3_reference/'
                        'c1_nondiag_freq/nondiag_results.json'))
print('top keys:', list(d.keys()))
for k in ['dft_hessian', 'd2_hessian', 'hess_total']:
    if k in d:
        v = d[k]
        print(k, '->', list(v.keys()) if isinstance(v, dict) else type(v))
        if isinstance(v, dict) and 'matrix' in v:
            m = np.asarray(v['matrix'])
            print('   shape', m.shape,
                  'sym resid %.2e' % np.abs(m - m.T).max())
er = d['endpoint_reproduction']
print('endpoint keys:', list(er.keys()))
print('config:', er.get('config'))
