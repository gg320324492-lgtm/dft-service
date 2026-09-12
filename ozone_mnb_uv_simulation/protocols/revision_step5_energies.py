#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Acceptance-revision step #5 (JOB-2026-0905-009): energies under UNIFIED
definitions, for the REVISED candidate set.

Definitions (per the revision brief, applied identically to DFT and CCSD(T)):
  E_int(noCP)  = E(complex) - E(O3 frozen@complex) - E(H2O frozen@complex)
  E_int(CP)    = E(complex) - E(O3 frozen@complex + ghost H2O)
                              - E(H2O frozen@complex + ghost O3)
      -> CP and non-CP use the SAME (frozen-complex) geometry convention.
  BSSE         = E_int(noCP) - E_int(CP)   (<= 0 by the variational principle;
                 the uncorrected interaction energy is too negative by |BSSE|;
                 the sign convention is stated explicitly)
  E_def        = [E(O3 frozen) - E(O3 own optimum)] + [H2O analogue]
      -> NOT ghost-corrected (weak-complex deformation ~1e-2 kcal/mol, its
         BSSE negligible; standard convention, stated explicitly).  The same
         E_def is added to both the CP and non-CP interaction energies.
  E_bind(noCP) = E_int(noCP) + E_def     ( = E(complex) - E(opt monomers) )
  E_bind(CP)   = E_int(CP)  + E_def
  D2           : full-derivative -D2 is included in EVERY electronic term;
                 ghost atoms are excluded from D2 (verified).
  Free energy  : dG(1 atm / 1 M) comes from harmonic thermochemistry of the
                 SEPARATELY OPTIMISED monomers, i.e. it is paired with
                 E_bind(noCP) and is NOT CP-corrected (stated explicitly).
  CCSD(T)      : interaction / deformation / binding assembled under the SAME
                 definitions from the step-4 single points (non-CP; no
                 aug-cc-pVTZ CP correction computed for CCSD(T)).  Comparing
                 the non-CP CCSD(T) BINDING energy with the DFT CP-corrected
                 INTERACTION energy is invalid and is not done.

ERROR documented here: the superseded Stage-4 table mixed conventions -- its
"E_int(非CP)" column used the separately optimised monomers (i.e. it was the
BINDING energy) while its "E_int(CP)" column used frozen-complex monomers
(interaction energy), so the two columns -- and the derived "BSSE" -- were not
comparable.
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
REV = os.path.join(ART, 'acceptance_revision')
os.makedirs(REV, exist_ok=True)

SYMS_COMPLEX = ['O', 'O', 'O', 'O', 'H', 'H']
CANDIDATES = ['c14_plus', 'c06_plus']
H2KCAL = cu.HARTREE2KCAL
SS_CORR = -1.894     # 1 atm -> 1 M, kcal/mol (job-specified)


def load(rel):
    with open(os.path.join(ART, rel)) as fh:
        return json.load(fh)


def sp_energy(syms, coords, ghosts=None, tag=''):
    """DFT+D2 single point; `ghosts` = complex indices turned into ghosts."""
    coords = np.asarray(coords, float)
    if ghosts is None:
        s = "; ".join("%s %.10f %.10f %.10f" % (sy, x, y, z)
                      for sy, (x, y, z) in zip(syms, coords))
        mol = cu.mol_from_coords(coords, syms, verbose=0)
    else:
        elem = SYMS_COMPLEX
        lines = ["%s %.10f %.10f %.10f" % (elem[i], *coords[i])
                 for i in range(len(elem)) if i not in ghosts]
        lines += ["X-%s %.10f %.10f %.10f" % (elem[i], *coords[i]) for i in ghosts]
        mol = cu.gto_mol_from(lines, verbose=0)
    mf = d2_full.make_mf_d2(mol)
    mf.kernel()
    assert mf.converged, 'SCF not converged: %s' % tag
    return float(mf.e_tot)


def main():
    mono = {m: np.asarray(load('gas_opt/%s.json' % m)['optimised_coords_angstrom'], float)
            for m in ('o3', 'h2o')}
    e_mono_opt = {m: float(load('gas_opt/%s.json' % m)['e_total_hartree'])
                  for m in ('o3', 'h2o')}
    # thermochemistry of the separately optimised monomers (gas_freq)
    th_mono = {m: load('gas_freq/%s.json' % m)['thermo_hartree'] for m in ('o3', 'h2o')}

    rows = []
    for tid in CANDIDATES:
        d_mf = load('mode_follow/%s.json' % tid)
        d_fr = load('gas_freq/%s.json' % tid)
        coords = np.asarray(d_mf['geometry_angstrom'], float)
        e_ab = float(d_mf['e_total_hartree'])
        assert float(np.abs(np.asarray(d_fr['geometry_angstrom'], float) - coords).max()) < 1e-6, \
            'geometry mismatch between mode_follow and gas_freq for %s' % tid

        # frozen monomers at the complex geometry (noCP)
        e_o3_fr = sp_energy(['O', 'O', 'O'], coords[:3], tag='%s o3 frozen' % tid)
        e_h2o_fr = sp_energy(['O', 'H', 'H'], coords[3:], tag='%s h2o frozen' % tid)
        # frozen monomers with ghost partner (CP)
        e_o3_fr_cp = cu.monomer_energy_in_complex_basis(coords, present=[0, 1, 2], ghost=[3, 4, 5])
        e_h2o_fr_cp = cu.monomer_energy_in_complex_basis(coords, present=[3, 4, 5], ghost=[0, 1, 2])

        e_int_nocp = e_ab - e_o3_fr - e_h2o_fr
        e_int_cp = e_ab - e_o3_fr_cp - e_h2o_fr_cp
        bsse_signed = e_int_nocp - e_int_cp            # <= 0
        e_def = (e_o3_fr - e_mono_opt['o3']) + (e_h2o_fr - e_mono_opt['h2o'])
        e_bind_nocp = e_int_nocp + e_def
        e_bind_cp = e_int_cp + e_def

        # free energy (harmonic RRHO, separately optimised monomers, no CP)
        th_ab = d_fr['thermo_hartree']
        zpe = (th_ab['ZPE'] - th_mono['o3']['ZPE'] - th_mono['h2o']['ZPE']) * H2KCAL
        g298 = (th_ab['G_298'] - th_mono['o3']['G_298'] - th_mono['h2o']['G_298']) * H2KCAL
        h298 = (th_ab['H_298'] - th_mono['o3']['H_298'] - th_mono['h2o']['H_298']) * H2KCAL

        rows.append(dict(
            id=tid, e_complex_hartree=e_ab,
            e_int_nocp_kcal=round(e_int_nocp * H2KCAL, 4),
            e_int_cp_kcal=round(e_int_cp * H2KCAL, 4),
            bsse_signed_kcal=round(bsse_signed * H2KCAL, 4),
            bsse_magnitude_kcal=round(-bsse_signed * H2KCAL, 4),
            e_def_kcal=round(e_def * H2KCAL, 4),
            e_bind_nocp_kcal=round(e_bind_nocp * H2KCAL, 4),
            e_bind_cp_kcal=round(e_bind_cp * H2KCAL, 4),
            deformations=dict(
                o3_frozen_minus_opt_kcal=round((e_o3_fr - e_mono_opt['o3']) * H2KCAL, 4),
                h2o_frozen_minus_opt_kcal=round((e_h2o_fr - e_mono_opt['h2o']) * H2KCAL, 4)),
            thermal_nocp=dict(zpe_kcal=round(zpe, 4), h298_kcal=round(h298, 4),
                              g298_kcal=round(g298, 4),
                              dG_assoc_1atm_kcal=round(g298, 4),
                              dG_assoc_1M_kcal=round(g298 + SS_CORR, 4)),
            raw_hartree=dict(e_o3_frozen=e_o3_fr, e_h2o_frozen=e_h2o_fr,
                             e_o3_frozen_cp=e_o3_fr_cp, e_h2o_frozen_cp=e_h2o_fr_cp,
                             e_mono_o3_opt=e_mono_opt['o3'], e_mono_h2o_opt=e_mono_opt['h2o']),
        ))
        print('[%s] E_int(noCP)=%.3f  E_int(CP)=%.3f  BSSE=%.3f  E_def=%.3f  '
              'E_bind(noCP)=%.3f  E_bind(CP)=%.3f kcal/mol'
              % (tid, rows[-1]['e_int_nocp_kcal'], rows[-1]['e_int_cp_kcal'],
                 rows[-1]['bsse_signed_kcal'], rows[-1]['e_def_kcal'],
                 rows[-1]['e_bind_nocp_kcal'], rows[-1]['e_bind_cp_kcal']), flush=True)

    # ---------------- CCSD(T) assembly under the same definitions (non-CP)
    ccsdt_dir = os.path.join(REV, 'ccsdt_revision')
    cc = {}
    if os.path.isdir(ccsdt_dir):
        for fn in sorted(os.listdir(ccsdt_dir)):
            if fn.endswith('.json'):
                with open(os.path.join(ccsdt_dir, fn)) as fh:
                    cc[fn[:-5]] = json.load(fh)
    cc_rows = []
    for tid in CANDIDATES:
        need = ['complex_%s' % tid, 'o3_frozen_%s' % tid, 'h2o_frozen_%s' % tid,
                'o3_optimised', 'h2o_optimised']
        if not all(k in cc for k in need):
            print('[ccsdt] missing single points for %s -- skipped (step 4 not finished?)' % tid)
            continue
        e_ab = cc['complex_%s' % tid]['e_ccsdt_total']
        e_o3_fr = cc['o3_frozen_%s' % tid]['e_ccsdt_total']
        e_h2o_fr = cc['h2o_frozen_%s' % tid]['e_ccsdt_total']
        e_o3_op = cc['o3_optimised']['e_ccsdt_total']
        e_h2o_op = cc['h2o_optimised']['e_ccsdt_total']
        e_int = e_ab - e_o3_fr - e_h2o_fr
        e_def = (e_o3_fr - e_o3_op) + (e_h2o_fr - e_h2o_op)
        cc_rows.append(dict(
            id=tid,
            e_int_ccsdt_nocp_kcal=round(e_int * H2KCAL, 4),
            e_def_ccsdt_kcal=round(e_def * H2KCAL, 4),
            e_bind_ccsdt_nocp_kcal=round((e_int + e_def) * H2KCAL, 4),
            diagnostics={k: cc['complex_%s' % tid][k] for k in
                         ('max_abs_t1', 't1_diagnostic_pyscf', 'd1_diagnostic_janssen',
                          'n_correlated_electrons', 'frozen_orbitals',
                          'rhf_converged', 'ccsd_converged')}))
        print('[ccsdt %s] E_int=%.3f  E_def=%.3f  E_bind(noCP)=%.3f kcal/mol'
              % (tid, cc_rows[-1]['e_int_ccsdt_nocp_kcal'],
                 cc_rows[-1]['e_def_ccsdt_kcal'],
                 cc_rows[-1]['e_bind_ccsdt_nocp_kcal']), flush=True)

    out = dict(
        job='JOB-2026-0905-009',
        step='revision_unified_energies',
        definitions=dict(
            interaction='E(complex) - monomers FROZEN at the complex geometry',
            cp='Boys-Bernardi ghosts on the frozen partner; same geometry as noCP',
            bsse_sign='BSSE = E_int(noCP) - E_int(CP) <= 0; magnitude = -BSSE',
            deformation='frozen monomer minus own-optimised monomer; not ghost-corrected '
                        '(weak-complex deformation, BSSE negligible; standard convention)',
            binding='E_int + E_def, for both the noCP and CP interactions',
            d2='full-derivative -D2 included in every electronic term; ghosts excluded',
            free_energy='harmonic RRHO of separately optimised monomers; paired with '
                        'E_bind(noCP); NOT CP-corrected',
            ccsdt='same definitions, non-CP; no aug-cc-pVTZ CP correction computed'),
        superseded_error=('old Stage-4 "E_int(非CP)" used optimised monomers (binding energy) '
                          'while "E_int(CP)" used frozen monomers (interaction energy): the '
                          'two columns and the derived BSSE were not comparable'),
        standard_state_correction_1atm_to_1M_kcal=SS_CORR,
        dft_rows=rows, ccsdt_rows=cc_rows)
    path = os.path.join(REV, 'revision_unified_energies.json')
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=2)
    print('SAVED ->', path)


if __name__ == '__main__':
    main()
