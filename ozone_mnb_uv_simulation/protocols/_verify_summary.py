import json
p = 'results/01_water_matrices/pure_water_o3_h2o/acceptance_revision/revision_summary.json'
d = json.load(open(p))
print('summary keys:', list(d.keys()))
print('candidates:', d['gas_candidates_pending_verification'],
      '| lowest:', d['gas_lowest_among_searched'])
print('freq verdict:', {k: v['stable'] for k, v in (d['frequency_verdict'] or {}).items()})
print('ccsdt rows:', [(r['id'], r['e_bind_ccsdt_nocp_kcal']) for r in d['ccsdt_rows']])
print('counts:', d['counts'])
