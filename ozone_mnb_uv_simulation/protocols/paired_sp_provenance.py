import json
import hashlib
import numpy as np

import os
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE) + os.sep
REG = ROOT + 'run_artifacts/01_pure_water_o3_h2o/smd_restart/gas_phase_registry.json'


def fsha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16]


def csha(c):
    return hashlib.sha256(np.asarray(c, float).tobytes()).hexdigest()[:16]


reg = json.load(open(REG))
out = {'registry_source_hash': fsha(REG),
       'candidates': {}}
for sid in ('c06_plus', 'c14_plus'):
    s = reg['structures'][sid]
    pr = s['primary_reference']
    entry = dict(
        primary_grid=pr['grid'], primary_kind=pr['kind'],
        registry_grad_max=pr['grad_max'],
        registry_e_total=pr.get('e_total_hartree'),
        method_label=pr['method'],
        verdict=pr['verdict'],
        optimizer=pr.get('optimizer'),
        atom_order=reg['atom_order'], coords_unit=reg['coords_unit'])
    # registry inline coords hash
    entry['registry_coords_sha'] = csha(pr['coords_angstrom'])
    # source file check
    src = ROOT + pr['source_file']
    entry['source_file'] = pr['source_file']
    entry['source_file_sha'] = fsha(src)
    entry['registry_source_hash'] = pr.get('source_hash')
    entry['source_hash_match'] = (pr.get('source_hash') is None or
                                  entry['source_file_sha'] == pr['source_hash'])
    d = json.load(open(src))
    if sid == 'c14_plus':
        ep = d.get('endpoint_coords_angstrom') or d.get('coords_angstrom')
        iv = d.get('independent_verification') or {}
        entry['file_coords_sha'] = csha(ep)
        entry['coords_match_registry'] = entry['file_coords_sha'] == entry['registry_coords_sha']
        entry['file_grad_max'] = iv.get('grad_max', d.get('grad_max'))
        entry['file_e_total'] = iv.get('e_total', d.get('e_total'))
        entry['file_surface_keys_note'] = 'gas L8 optimize result'
    else:
        ep = d['endpoints']['c06_plus']['coords_angstrom']
        entry['file_coords_sha'] = csha(ep)
        entry['coords_match_registry'] = entry['file_coords_sha'] == entry['registry_coords_sha']
        entry['file_source_note'] = d.get('source_note')
        entry['file_source_file'] = d.get('source_file')
        entry['file_source_hash'] = d.get('source_hash')
        # phaseC attempt2 grad for cross-check
        pc = json.load(open(ROOT + d['source_file']))
        v = pc['verdicts']['c06_plus']
        entry['phaseC_attempt2'] = v
    out['candidates'][sid] = entry

# acceptance-record links
out['acceptance_records'] = {
    'c06': 'phaseC_summary_attempt2.json :: verdicts.c06_plus (pass_gmax=True, L7)',
    'c14': 'phaseC_summary_attempt2.json (L7 pass) + c14_l8_check/l8_optimize_result.json (L8) + frequency_analysis_l8_c14.json (12/12 on L8 endpoint)'}
json.dump(out, open('run_artifacts/01_pure_water_o3_h2o/paired_sp/provenance.json', 'w'), indent=2)
for sid, e in out['candidates'].items():
    print('===', sid)
    print('  registry coords sha:', e['registry_coords_sha'],
          '| file coords sha:', e['file_coords_sha'],
          '| match:', e['coords_match_registry'])
    print('  source file sha:', e['source_file_sha'],
          '| registry source_hash:', e['registry_source_hash'],
          '| match:', e['source_hash_match'])
    print('  grid:', e['primary_grid'], '| grad_max(registry):', e['registry_grad_max'])
    print('  method:', json.dumps(e['method_label'], ensure_ascii=False)[:160])
    if 'phaseC_attempt2' in e:
        print('  phaseC attempt2:', e['phaseC_attempt2'])
