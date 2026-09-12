import json
import numpy as np

HERE = '/mnt/e/dft-service/ozone_mnb_uv_simulation/'
d = json.load(open(HERE + 'run_artifacts/01_pure_water_o3_h2o/paired_sp/single_points.json'))
pts = {p['tag']: p for p in d['points']}
assert len(pts) == 4

KCAL = 627.5094740631  # Eh -> kcal/mol

E_gas = {'c06': pts['c06_L7endpoint_gas']['e_total'],
         'c14': pts['c14_L8endpoint_gas']['e_total']}
E_smd = {'c06': pts['c06_L7endpoint_smd']['e_total'],
         'c14': pts['c14_L8endpoint_smd']['e_total']}

# D2 consistency check (same geometry, two phases)
d2_check = {}
for tag_g, tag_s, sid in (('c06_L7endpoint_gas', 'c06_L7endpoint_smd', 'c06'),
                          ('c14_L8endpoint_gas', 'c14_L8endpoint_smd', 'c14')):
    d2g = pts[tag_g]['e_d2_analytic']
    d2s = pts[tag_s]['e_d2_analytic']
    d2_check[sid] = dict(gas=d2g, smd=d2s,
                         absdiff=abs(d2g - d2s),
                         consistent=bool(abs(d2g - d2s) < 1e-12))
    assert d2_check[sid]['consistent'], 'D2 mismatch for %s' % sid

out = dict(
    job='JOB-2026-0906-018 Task C (offline analysis)',
    sign_convention='DeltaE_solv_fixed = E_SMD(R) - E_gas(R); negative = '
                    'continuum water stabilizes the fixed geometry. '
                    'DeltaE_gas = E_gas(R_c06) - E_gas(R_c14) etc.',
    definitions=dict(
        dE_solv_fixed='solvent electronic-energy change at the FIXED '
                      'gas-phase geometry; NOT a solvation free energy, '
                      'NOT a binding energy, NOT an aqueous reaction energy',
        dE_gas='gas-phase relative electronic energy of the two candidates '
               'at their accepted geometries (different grids L7/L8 '
               'optimisation labels; single points both at L8)',
        dE_SMD_fixed='same comparison with SMD(water, order41, SWIG) at the '
                     'SAME fixed geometries'),
    energies_hartree=dict(E_gas=E_gas, E_SMD_fixed=E_smd),
    derived=dict(
        dE_solv_fixed=dict(
            c06=E_smd['c06'] - E_gas['c06'],
            c14=E_smd['c14'] - E_gas['c14']),
        dE_gas=E_gas['c06'] - E_gas['c14'],
        dE_SMD_fixed=E_smd['c06'] - E_smd['c14']),
    d2_consistency=d2_check,
    gradients=dict(
        c06_gas_max_g=pts['c06_L7endpoint_gas']['grad_max'],
        c14_gas_max_g=pts['c14_L8endpoint_gas']['grad_max'],
        c06_smd_max_g=pts['c06_L7endpoint_smd']['grad_max'],
        c14_smd_max_g=pts['c14_L8endpoint_smd']['grad_max'],
        c14_gas_reproduces_L8_acceptance=dict(
            e_matches=bool(abs(pts['c14_L8endpoint_gas']['e_total']
                               - (-301.87483796711086)) < 1e-9),
            grad_matches=bool(abs(pts['c14_L8endpoint_gas']['grad_max']
                                  - 2.5412224873388245e-06) < 1e-10)),
        note='gas L8 single-point gradients are checks under THIS batch '
             'configuration; they do NOT rewrite the original acceptance '
             'evidence (c06 accepted at L7, c14 at L8). SMD gradients show '
             'fixed-geometry forces; they do NOT trigger optimisation.'),
    kcal_per_eh=KCAL)

dv = out['derived']
dv['dE_solv_fixed_kcal'] = {k: v * KCAL for k, v in
                            dv['dE_solv_fixed'].items()}
dv['dE_gas_kcal'] = dv['dE_gas'] * KCAL
dv['dE_SMD_fixed_kcal'] = dv['dE_SMD_fixed'] * KCAL
dv['solvent_effect_on_relative_energy'] = (dv['dE_SMD_fixed']
                                           - dv['dE_gas'])
dv['solvent_effect_on_relative_energy_kcal'] = \
    dv['solvent_effect_on_relative_energy'] * KCAL

out['scope'] = dict(
    observation_only='all comparisons are observations under the CURRENT '
                     'model (wb97xd+-D2/def2-TZVP/L8, SMD water order41 '
                     'SWIG) and the FIXED gas-phase geometries',
    not_claimed=['robust aqueous ordering of the two candidates',
                 'solution-phase minima', 'solvation free energies',
                 'aqueous reaction energies or mechanisms'],
    ordering_note='gas: c14 lower by %.3f kcal/mol; SMD at fixed '
                  'geometries: c06 lower by %.3f kcal/mol -- an ordering '
                  'change is reported only as an observation; aqueous '
                  'numerical sensitivity is unresolved and geometries are '
                  'gas-phase minima'
                  % (abs(dv['dE_gas_kcal']), abs(dv['dE_SMD_fixed_kcal'])))

json.dump(out, open(HERE + 'run_artifacts/01_pure_water_o3_h2o/paired_sp/unified_energies.json', 'w'),
          indent=2)
print(json.dumps({k: out[k] for k in ('derived', 'd2_consistency')},
                 indent=2))
print('scope:', out['scope']['ordering_note'])
