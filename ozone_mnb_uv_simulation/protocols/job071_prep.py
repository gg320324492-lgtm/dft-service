#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-071: P1(2) overall spin-coupling preparation.

QUANTITATIVE BUDGET: 0 SCF, 0 gradient.  This batch ONLY assembles the
spin-coupling plan for the HOO. + H2NO. pair (34 electrons, charge 0)
from (a) the JOB-070 accepted radical endpoints (read-only) and (b)
registered S13 literature boundaries (JOB-067, read-only).  It does
NOT run the P1 complex, does NOT construct TS1/TS2/CP1, and does NOT
guess any 3D coordinates.

Outputs:
  * p1_spin071/p1_spin_prep071_results.json  (machine-readable plan)
  * report + task record are written separately by the operator.

All content is labelled: PROJECT METHOD MODEL, NOT an S13
reproduction.
"""
import os, sys, json, time

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
R_REF = ROOT + '/run_artifacts/02_nh3o3_reference'
OUT = R_REF + '/p1_spin071'
RES70 = R_REF + '/p1_radicals070/p1_radicals070_results.json'
MAN70 = R_REF + '/p1_radicals070/input_manifest.json'
RES69 = R_REF + '/p4_freq069/p4_freq069_results.json'
RES68 = R_REF + '/p4_products068/p4_products068_results.json'

VALENCE = {'H': 1, 'N': 7, 'O': 8}      # total electrons per element


def n_electrons(elements):
    return sum(VALENCE[e] for e in elements)


def main():
    os.makedirs(OUT, exist_ok=True)
    man70 = json.load(open(MAN70))
    res70 = json.load(open(RES70))
    res69 = json.load(open(RES69))
    res68 = json.load(open(RES68))

    # ---- 0. endpoint inventory (read-only) --------------------------
    endpoints = {}
    for sp in ('HOO', 'H2NO'):
        m = res70['molecules'].get(sp, {})
        endpoints[sp] = dict(
            accepted=bool(m.get('recheck_pass')),
            e_recheck=m.get('e_recheck'),
            stable_i=(m.get('stability') or {}).get('stable_i'),
            s2_total=(m.get('stability') or {}).get(
                's2_total',
                (m.get('recheck') or {}).get('s2_total')),
            s2_delta=(m.get('stability') or {}).get(
                's2_delta',
                (m.get('recheck') or {}).get('s2_delta')),
            elements=man70['molecules'][sp]['elements'],
            n_electrons=n_electrons(
                man70['molecules'][sp]['elements']))
    frag = res70.get('fragment_reference', {})

    # ---- 1. electron/charge bookkeeping ------------------------------
    n_hoo = n_electrons(['O', 'O', 'H'])            # 17
    n_h2no = n_electrons(['N', 'O', 'H', 'H'])      # 17
    n_total = n_hoo + n_h2no                        # 34
    n_reactant = n_electrons(['N', 'H', 'H', 'H',
                              'O', 'O', 'O'])       # 34
    bookkeeping = dict(
        HOO_electrons=n_hoo, H2NO_electrons=n_h2no,
        total_electrons=n_total, reactant_electrons=n_reactant,
        electrons_conserved=bool(n_total == n_reactant == 34),
        total_charge=0,
        spin_polarization_singlet='n_alpha = n_beta = 17',
        spin_polarization_triplet='n_alpha = 18, n_beta = 16 '
                                  '(mol.spin = 2)')

    # ---- 2. coupling candidates (prepared, NOT run) ------------------
    candidates = dict(
        bs_singlet_uks=dict(
            name='broken-symmetry open-shell singlet (UKS, '
                 'antiferromagnetically coupled doublets)',
            pyscf_setup='mol.spin = 0 with dft.UKS(mol) '
                        '(n_alpha = n_beta = 17, unequal spatial '
                        'orbitals); RKS is NOT applicable',
            s2_acceptance=dict(
                clean_singlet='|<S2>| <= 0.01 (rarely reached at BS)',
                bs_usable='0.01 < <S2> < 2.0: registered as a '
                          'spin-polarized singlet REFERENCE with the '
                          'exact <S2> recorded; NEVER reported as a '
                          'pure singlet',
                reject_as_singlet='<S2> >= 2.0 -> effectively '
                                  'triplet, reject as singlet '
                                  'candidate'),
            spin_density_check='fragment-wise spin populations must '
                               'have OPPOSITE signs on the two '
                               'radical centers and sum to ~0',
            fragment_spin_directions='HOO. unpaired electron in the '
                                     'O-O pi* system; H2NO. unpaired '
                                     'electron N-centered (N-O pi*); '
                                     'antiparallel pairing in the BS '
                                     'reference',
            budget_if_authorized='new JOB with its own ledger; NOT '
                                 'authorized in this window'),
        triplet_uks=dict(
            name='triplet (UKS, ferromagnetically coupled doublets)',
            pyscf_setup='mol.spin = 2 (n_alpha = 18, n_beta = 16) '
                        'with dft.UKS(mol)',
            s2_acceptance=dict(
                clean_triplet='|<S2> - 2.0| <= 0.01',
                contaminated='recorded truthfully; usable only as '
                             'documented'),
            spin_density_check='fragment-wise spin populations have '
                               'the SAME sign and sum to ~2',
            fragment_spin_directions='parallel unpaired electrons',
            budget_if_authorized='new JOB with its own ledger; NOT '
                                 'authorized in this window'),
        ranking_rule='compare E(BS-singlet) and E(triplet) at the '
                     'SAME geometry ONLY; the lower energy is the '
                     'better project-model reference state; neither '
                     'is an S13 quantity')

    # ---- 3. initial guess / occupation strategy ----------------------
    guesses = dict(
        a_fragment_superposition=dict(
            description='converge each doublet radical separately '
                        '(DONE in JOB-070), place fragments at a '
                        'large separation, and superpose the two '
                        'converged alpha/beta density blocks: '
                        'parallel alpha+alpha for the triplet '
                        'candidate, alpha+beta (opposite spin '
                        'blocks) for the BS-singlet candidate',
            implementation='init_guess from a custom density matrix '
                           'built from the two JOB-070 chkfiles; '
                           'registered per-tag; no re-convergence of '
                           'the fragments'),
        b_triplet_then_stability=dict(
            description='converge the triplet UKS first and run '
                        'internal stability; if the analysis finds a '
                        'lower singlet instability, follow it ONCE '
                        'with the installed-orbital update to reach '
                        'the BS reference',
            caveat='the followed state must be re-converged and its '
                   '<S2>/spin density re-checked; counts as new '
                   'attempts in its own ledger'),
        c_direct=dict(
            description='direct UKS from the default minao guess at '
                        'the target geometry',
            caveat='may land in either coupling; the <S2> and spin '
                   'density checks decide what was actually '
                   'converged; cheapest but least controlled'),
        occupation_rule=' occupancies follow from n_alpha/n_beta of '
                        'the candidate (17/17 or 18/16); no manual '
                        'HOMO/LUMO swapping is planned')

    # ---- 4. author-coordinate question --------------------------------
    author_coords = dict(
        separated_fragments='NOT needed: P1(2) as separated HOO. + '
                            'H2NO. endpoints were computed WITHOUT '
                            'author coordinates (JOB-070, project '
                            'method model)',
        cp1_complex='NEEDED (or an explicitly authorized project-model '
                    'construction): the S13 Figure-1 2D drawing does '
                    'not provide NH3/O3 depth orientation or the '
                    'H-transfer geometry of CP1 (JOB-063 precedent: '
                    '"partially constructible", missing dihedrals '
                    'hard stop)',
        ts2_or_path='NEEDED: TS2/TS1/IRC coordinates are absent from '
                    'the literature; guessing them is FORBIDDEN by '
                    'the project charter')

    # ---- 5. project model vs S13 differences --------------------------
    differences = [
        'method: wb97xd (libxc) + project explicit D2 / def2-TZVP '
        'vs S13 B3LYP geometries + CCSD(T) single points',
        'energy definition: project values are ELECTRONIC energies '
        '(no ZPE/thermal/CP) vs S13 Table 1 relative electronic '
        'energies AND a separate thermo table with dG0(298K)',
        'geometry source: JOB-067 templates -> project '
        'optimizations vs S13 published optimized structures',
        'scope: this project currently has ENDPOINTS only (R '
        'monomers, P4 HNO accepted, P1(2) radicals); S13 reports '
        'the full path R-C1-TS1-IN1-(TS2/TS8)-(CP1/CP3)-P',
        'no aqueous phase, no UV, no micro/nano bubble effects in '
        'either the project model or the S13 gas-phase table',
        'the S13 TS8 barrier text (51.08) vs table-derived (51.80) '
        'kcal/mol inconsistency remains unresolved (JOB-067)']

    results = dict(
        job='JOB-2026-0906-071: P1(2) overall spin-coupling '
            'preparation (ZERO evaluations)',
        quantitative_budget=dict(scf=0, gradients=0,
                                 d2_fd_calls=0),
        model_status=dict(
            status_is='project method model spin-coupling '
                      'preparation document',
            status_is_not='NOT an S13 reproduction; no P1 complex '
                          'run; no TS/CP construction; no coordinate '
                          'guessing'),
        endpoint_inventory=endpoints,
        p1_fragment_reference_070=frag,
        electron_bookkeeping=bookkeeping,
        coupling_candidates=candidates,
        initial_guess_strategy=guesses,
        author_coordinate_question=author_coords,
        project_vs_s13_differences=differences,
        s13_registered_values=(man70['p1_fragment_energy'][
            'qualitative_comparison']),
        related_results=dict(
            job068=os.path.relpath(RES68, ROOT),
            job069=os.path.relpath(RES69, ROOT),
            job070=os.path.relpath(RES70, ROOT)),
        next_step_recommendation=(
            'if the commander authorizes a P1-complex batch: ONE new '
            'JOB, own ledger, geometry = large-separation fragment '
            'superposition (strategy a) with candidates BS-singlet '
            'and triplet at IDENTICAL geometry, then optional rigid '
            'approach only after a defensible project-model '
            'construction assumption is authorized; author '
            'coordinates remain the preferred alternative'),
        finished=time.strftime('%F %T'))
    path = OUT + '/p1_spin_prep071_results.json'
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, path)
    print('[071] spin-coupling prep written:', path)
    print('[071] electrons conserved:', bookkeeping[
        'electrons_conserved'])
    print('[071] endpoints accepted:',
          {k: v['accepted'] for k, v in endpoints.items()})


if __name__ == '__main__':
    main()
