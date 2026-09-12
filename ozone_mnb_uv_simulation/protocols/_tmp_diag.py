import json
d = json.load(open('run_artifacts/01_pure_water_o3_h2o/unit_consistency_revision/unit_consistency_analysis.json'))
dc = d['directional_confirmation']['c14_plus']
print('m_max =', dc['max_atom_disp_directional_norm'])
print()
for p in dc['points_actual_displacement'][:6]:
    print('%-24s nom=%.3f sgn=%+d  maxdisp=%.6fA  q_act=%.6f  dev=%.2fdeg  E=%s' % (
        p['key'], p['amp_nominal_a'], p['sign'], p['max_atom_disp_actual_A'],
        p['q_actual_bohr'], p['direction_deviation_deg'],
        '%.10f' % p['e_total'] if p.get('e_total') else 'n/a'))
print()
for row in dc['corrected_analysis_false']:
    print('nominal %.3f: k_E(corr)=%+.4e' % (row['amp_nominal_a'], row['k_E']))
