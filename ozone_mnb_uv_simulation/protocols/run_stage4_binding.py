#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage 4 (JOB-2026-0905-009): gas-phase interaction energy of O3.H2O.

For every retained gas-phase minimum we report:
  * uncorrected electronic interaction energy  E_int = E(AB) - E(A) - E(B)
  * Boys-Bernardi counterpoise (CP) corrected  E_int^CP = E(AB) - E_A@AB - E_B@AB
    with the "ghost" partner represented as PySCF ghost atoms (X-O / X-H);
    the project -D2 term EXCLUDES ghost atoms (d2_full._real_atoms), so no
    spurious dispersion pairs are introduced by the ghost basis;
  * ZPE and thermal (H_298 / G_298) contributions, decomposed the same way;
  * the gas-phase association free energy at 1 atm and at 1 M standard state
    (O3 + H2O -> O3.H2O; the 1 atm -> 1 M correction is -1.894 kcal/mol per
    the job brief).

A verification of the ghost-D2 exclusion is written to the output so the CP
calculation can be audited.
"""
import os
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import d2_full
import conformer_utils as cu

ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, 'run_artifacts', '01_pure_water_o3_h2o')
GAS_OPT = os.path.join(ART, 'gas_opt')
GAS_FREQ = os.path.join(ART, 'gas_freq')

H2KCAL = cu.HARTREE2KCAL
STANDARD_STATE_CORR_KCAL = -1.894   # 1 atm -> 1 M for O3 + H2O -> O3.H2O


def load(path):
    with open(path) as fh:
        return json.load(fh)


def verify_ghost_d2():
    """Confirm ghosts are excluded from -D2: an O3+ghostH2O mol must carry the
    same D2 as isolated O3 at the same O3 coordinates, and no KeyError."""
    coords = cu.coords_of(load(os.path.join(GAS_OPT, 'c01.json')))
    # isolated O3 D2
    o3 = coords[:3]
    mol_o3 = cu.mol_from_coords(o3, ['O', 'O', 'O'], verbose=0)
    d2_iso = d2_full.d2_energy(mol_o3)
    # O3 + ghost H2O at the complex geometry
    elem = ['O', 'O', 'O', 'O', 'H', 'H']
    lines = ["%s %.10f %.10f %.10f" % (elem[i], *coords[i]) for i in range(3)]
    lines += ["X-%s %.10f %.10f %.10f" % (elem[i], *coords[i]) for i in (3, 4, 5)]
    mol_g = cu.gto_mol_from(lines, verbose=0)
    d2_g = d2_full.d2_energy(mol_g)
    # also confirm atom_charge of ghosts is 0 (the exclusion key)
    ghost_charges = [int(mol_g.atom_charge(i)) for i in (3, 4, 5)]
    return dict(d2_isolated_o3=d2_iso, d2_with_ghost_h2o=d2_g,
                d2_difference=d2_g - d2_iso,
                ghost_charges=ghost_charges,
                ghosts_excluded_ok=bool(abs(d2_g - d2_iso) < 1e-15
                                        and all(c == 0 for c in ghost_charges)))


def binding_for(tid):
    ab = load(os.path.join(GAS_OPT, '%s.json' % tid))
    coords = cu.coords_of(ab)
    o3 = load(os.path.join(GAS_OPT, 'o3.json'))
    h2o = load(os.path.join(GAS_OPT, 'h2o.json'))
    ab_e = ab['e_total_hartree']
    o3_e = o3['e_total_hartree']
    h2o_e = h2o['e_total_hartree']

    # CP: monomer energies at the complex geometry with ghost partner
    e_o3_ab = cu.monomer_energy_in_complex_basis(coords, present=[0, 1, 2],
                                                 ghost=[3, 4, 5])
    e_h2o_ab = cu.monomer_energy_in_complex_basis(coords, present=[3, 4, 5],
                                                  ghost=[0, 1, 2])

    e_int_unc = (ab_e - o3_e - h2o_e) * H2KCAL
    e_int_cp = (ab_e - e_o3_ab - e_h2o_ab) * H2KCAL
    bsse = e_int_unc - e_int_cp    # positive = BSSE stabilises the uncorrected value

    # ZPE / thermal from the Stage-2 harmonic analysis
    abf = load(os.path.join(GAS_FREQ, '%s.json' % tid))['thermo_hartree']
    o3f = load(os.path.join(GAS_FREQ, 'o3.json'))['thermo_hartree']
    h2of = load(os.path.join(GAS_FREQ, 'h2o.json'))['thermo_hartree']

    def decomp(term):
        return dict(value_kcal=round((abf[term] - o3f[term] - h2of[term]) * H2KCAL, 4),
                    complex=abf[term], o3=o3f[term], h2o=h2of[term])

    zpe = decomp('ZPE')
    g298 = decomp('G_298')
    h298 = decomp('H_298')

    # association free energy, 1 atm (gas-phase RRHO, run_baseline convention)
    g_assoc_1atm = g298['value_kcal']
    g_assoc_1m = g_assoc_1atm + STANDARD_STATE_CORR_KCAL

    return dict(
        id=tid,
        e_int_uncorrected_kcal_mol=round(e_int_unc, 4),
        e_int_cp_kcal_mol=round(e_int_cp, 4),
        bsse_kcal_mol=round(bsse, 4),
        e_complex_hartree=ab_e, e_o3_hartree=o3_e, e_h2o_hartree=h2o_e,
        e_o3_at_complex_geom_hartree=e_o3_ab, e_h2o_at_complex_geom_hartree=e_h2o_ab,
        zpe=zpe, h298=h298, g298=g298,
        deltaG_assoc_1atm_kcal_mol=round(g_assoc_1atm, 4),
        deltaG_assoc_1M_kcal_mol=round(g_assoc_1m, 4),
        standard_state_correction_kcal_mol=STANDARD_STATE_CORR_KCAL,
    )


def main():
    with open(os.path.join(ART, 'stage2_minima.json')) as fh:
        s2 = json.load(fh)
    kept = list(s2['kept_minima'])
    rows = [binding_for(tid) for tid in kept]
    ghost_audit = verify_ghost_d2()
    out = dict(job='JOB-2026-0905-009', stage='4_binding_energy',
               method='wB97X-D/def2-TZVP full-derivative -D2',
               monomers=dict(o3='gas_opt/o3.json', h2o='gas_opt/h2o.json'),
               standard_state_note=('O3 + H2O -> O3.H2O; 1 atm -> 1 M correction '
                                    '= %.3f kcal/mol (applied to the gas RRHO '
                                    'association free energy).' % STANDARD_STATE_CORR_KCAL),
               ghost_d2_audit=ghost_audit,
               binding=rows)
    with open(os.path.join(ART, 'binding.json'), 'w') as fh:
        json.dump(out, fh, indent=2)
    print('STAGE4_DONE  ghost_ok=%s  minima=%d' % (
        ghost_audit['ghosts_excluded_ok'], len(rows)), flush=True)
    for r in rows:
        print('  %s  E_int(noCP)=%.3f  E_int(CP)=%.3f  dG(1atm)=%.3f  dG(1M)=%.3f kcal/mol'
              % (r['id'], r['e_int_uncorrected_kcal_mol'], r['e_int_cp_kcal_mol'],
                 r['deltaG_assoc_1atm_kcal_mol'], r['deltaG_assoc_1M_kcal_mol']), flush=True)


if __name__ == '__main__':
    main()
