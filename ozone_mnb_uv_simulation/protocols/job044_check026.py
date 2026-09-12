import json
import numpy as np

r = json.load(open('run_artifacts/02_nh3o3_reference/freq_check/freq_check_results.json'))
res = r['results']
print('results keys:', list(res.keys()))
for sp in res.keys():
    print()
    print('===', sp, '===')
    d = res[sp]
    print('  sub-keys:', list(d.keys()) if isinstance(d, dict) else type(d))
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, dict):
                print('   ', k, '->', list(v.keys()))
            else:
                print('   ', k, '=', str(v)[:150])
PYEOF_MARK = None
