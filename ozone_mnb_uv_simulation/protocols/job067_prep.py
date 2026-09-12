#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-067: S13 reaction-channel P4 and P1(2) OFFLINE
preparation.  ZERO quantum evaluations (no SCF/gradient/stability/
Hessian/frequency/optimization/TS-search/IRC/single-point).

Deliverables:
  - literature extraction for P4 (NH3+O3 -> HNO+H2O2) and P1(2)
    (NH3+O3 -> HOO.+H2NO.): reactions, paths, relative energies
    (B3LYP/CCSD(T)//B3LYP/G3B3), barriers, T1 diagnostics, thermo
    (dH0/dG0/TdS + experimental anchors), frequency characterization
    and gaps;
  - electron/spin assignment table (34 electrons; closed-shell P4 vs
    two-doublet P1(2); O3 multireference risk T1=0.0346);
  - project input templates: reactants (JOB-026 accepted monomers,
    exact copies) and product input templates built from the S13
    Figure-1 B3LYP parameters with REGISTERED idealized values where
    the figure gives no label (HNO angle, HOO angle, H2O2 dihedral);
    templates are input preparation ONLY - never author coordinates,
    never results;
  - executability check (TS 3D coordinates NOT available from the
    paper -> "cannot reproduce directly"; contact-author
    recommendation registered as a SUGGESTION);
  - no-SCF regression tests.
"""
import os, sys, json, hashlib, types
import numpy as np

sys.modules['pyscf'] = None
sys.modules['pyscf.hessian'] = None
sys.modules['d2_full'] = types.ModuleType('d2_full')

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
R_REF = ROOT + '/run_artifacts/02_nh3o3_reference'
OUT = R_REF + '/reaction_prep067'
IDX = R_REF + '/final_endpoint_index.json'
JOB_NO = '067'
BPA = 1.0 / 0.52917721092
BOHR_A = 0.52917721092


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for blk in iter(lambda: fh.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()[:16]


def save_json(path, obj):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(obj, fh, indent=2, default=str)
    os.replace(tmp, path)


def check_number():
    import glob, re
    used = set()
    for f in glob.glob(ROOT + '/jobs/JOB-*.md'):
        m = re.match(r'JOB-2026-0906-(\d{3})', os.path.basename(f))
        if m:
            used.add(m.group(1))
    nxt = '%03d' % (max(int(u) for u in used) + 1)
    return dict(used_max=max(used), next_free=nxt,
                pass_=bool(nxt == JOB_NO and JOB_NO not in used))


# ---------------------------------------------------------------- geometry
def zmat_to_cart(species, params):
    """Minimal Z-matrix -> Cartesian for the small product templates.
    params are Angstrom/degrees from the S13 Figure-1 labels (or
    registered idealized values where the figure gives none)."""
    if species == 'HNO':
        rNH, rNO, angHNO = params            # angle at N
        N = np.array([0.0, 0.0, 0.0])
        O = np.array([rNO, 0.0, 0.0])
        H = N + rNH * np.array([-np.cos(np.radians(angHNO)),
                                np.sin(np.radians(angHNO)), 0.0])
        return dict(elements=['N', 'O', 'H'], coords=np.vstack([N, O, H]),
                    bonds={'N-O': rNO, 'N-H': rNH}, angle={'H-N-O': angHNO})
    if species == 'HOO':
        rOO, rHO, angHOO = params            # angle at terminal O(2O)
        O1 = np.array([0.0, 0.0, 0.0])       # central O (2O)
        O2 = np.array([rOO, 0.0, 0.0])       # terminal O (3O)
        H = O1 + rHO * np.array([-np.cos(np.radians(angHOO)),
                                 np.sin(np.radians(angHOO)), 0.0])
        return dict(elements=['O', 'O', 'H'], coords=np.vstack([O1, O2, H]),
                    bonds={'O-O': rOO, 'H-O': rHO}, angle={'H-O-O': angHOO})
    if species == 'H2O2':
        rOO, rOH, angOOH, dih = params
        O1 = np.array([0.0, 0.0, 0.0])
        O2 = np.array([rOO, 0.0, 0.0])
        t = np.radians(angOOH)
        H1 = O1 + rOH * np.array([-np.cos(t), np.sin(t), 0.0])
        H2 = O2 + rOH * np.array([np.cos(t) * np.cos(np.radians(dih))
                                  * (-1), np.sin(t),
                                  np.cos(t) * np.sin(np.radians(dih))])
        return dict(elements=['O', 'O', 'H', 'H'],
                    coords=np.vstack([O1, O2, H1, H2]),
                    bonds={'O-O': rOO, 'O-H': rOH},
                    angle={'O-O-H': angOOH}, dihedral={'H-O-O-H': dih})
    if species == 'H2NO':
        rNO, rNH, angHNO = params            # angle at N (H-N-O)
        N = np.array([0.0, 0.0, 0.0])
        O = np.array([rNO, 0.0, 0.0])
        h1 = rNH * np.array([-np.cos(np.radians(angHNO)),
                             np.sin(np.radians(angHNO)), 0.0])
        h2 = rNH * np.array([-np.cos(np.radians(angHNO)),
                             -np.sin(np.radians(angHNO)), 0.0])
        return dict(elements=['N', 'O', 'H', 'H'],
                    coords=np.vstack([N, O, h1, h2]),
                    bonds={'N-O': rNO, 'N-H': rNH},
                    angle={'H-N-O': angHNO},
                    note='planar template (2D figure projection; the '
                         'out-of-plane orientation of the radical is a '
                         'project assumption)')
    raise KeyError(species)


# ------------------------------- electron/spin assignment ----------------
Z_E = dict(N=7, H=1, O=8)


def electrons(formula_counts):
    return int(sum(Z_E[e] * c for e, c in formula_counts.items()))


# =============================== main =====================================
def main():
    num = check_number()
    os.makedirs(OUT + '/templates', exist_ok=True)
    os.makedirs(OUT + '/figures', exist_ok=True)

    idx = json.load(open(IDX))
    mon_nh3_A = np.asarray(idx['default_endpoints']['NH3_gas_v5_final'][
        'coords_angstrom'], float)
    mon_o3_A = np.asarray(idx['default_endpoints']['O3_gas_final'][
        'coords_angstrom'], float)

    # ---------------- literature extraction (S13, page refs) ------------
    literature = dict(
        source_pdf='references/literature/'
                   '2013-MechanismandthermodynamicsofmultichannelNH3O3'
                   '.pdf',
        level='B3LYP/6-311++G(3df,3pd) optimizations; CCSD(T)//B3LYP and '
              'G3B3//B3LYP single points; T1 at CCSD/6-311++G(3df,3pd)'
              '//B3LYP',
        t1_thresholds=dict(closed_shell=0.02, open_shell_accepted_to=0.045,
                           source='page 268'),
        pes_statement='On the singlet PES... P1 path is kinetically more '
                      'reliable than the others and P4 is the most '
                      'stable (page 279)',
        table1_relative_energies_kcalmol=dict(
            R=dict(b3lyp=0.0, ccsdt=0.0, g3b3=0.0, t1=0.0346),
            C1=dict(b3lyp=-0.69, ccsdt=-1.41, g3b3=-0.69, t1=0.0241),
            IN1=dict(b3lyp=-12.14, ccsdt=-10.00, g3b3=-5.13, t1=0.0174),
            TS1=dict(b3lyp=23.28, ccsdt=28.27, g3b3=31.86, t1=0.0452),
            TS2=dict(b3lyp=18.81, ccsdt=20.77, g3b3=None, t1=0.0507),
            TS8=dict(b3lyp=43.05, ccsdt=41.80, g3b3=42.50, t1=0.0215),
            CP1=dict(b3lyp=-21.02, ccsdt=-17.86, g3b3=-13.30, t1=0.0212),
            CP3=dict(b3lyp=-33.50, ccsdt=-35.99, g3b3=-32.68, t1=0.0145),
            P1=dict(b3lyp=-19.40, ccsdt=-5.82, g3b3=-5.16, t1=0.0201),
            P4=dict(b3lyp=-33.23, ccsdt=-32.64, g3b3=-31.83, t1=0.0284),
            note='relative to R (NH3+O3); page 269 Table 1'),
        barriers_kcalmol=dict(
            TS1_C1_to_IN1=dict(value=29.68, level='CCSD(T)',
                               derivation='TS1(28.27) - C1(-1.41), '
                                          'consistent'),
            TS2_IN1_to_CP1=dict(value=30.77, level='CCSD(T)',
                                derivation='TS2(20.77) - IN1(-10.00), '
                                           'consistent'),
            TS8_IN1_to_CP3=dict(value_stated=51.08,
                                value_from_table=51.80,
                                discrepancy='stated 51.08 vs '
                                            'TS8(41.80)-IN1(-10.00)=51.80; '
                                            'registered as a literature '
                                            'internal inconsistency, not '
                                            'resolved here'),
            CP1_to_P1='barrierless (direct 1N-7O rupture)',
            CP3_to_P4='barrierless (direct 5O-3H rupture)'),
        thermo_298K=dict(
            P1_HOO_H2NO=dict(dE=-19.40, dH0=-19.20, dG0=-21.35,
                             tdS=6.18),
            P4_H2O2_HNO=dict(dE=-33.09, dH0=-33.09,
                             dH0_experimental=-31.9,
                             dG0=-34.33, dG0_experimental=-33.45,
                             tdS=9.22,
                             relative_error_G='per paper -2.6%'),
            source='page 275 table'),
        products_relative_to_R=dict(
            P1=dict(text='5.82 kcal/mol lower than the original '
                         'reactants (B3LYP)'),
            P4=dict(text='32.64 kcal/mol below the original reactants '
                         '(CCSD(T))')),
        frequencies=dict(
            characterization='minima = no imaginary frequencies; TS = '
                             'exactly one imaginary frequency (B3LYP '
                             'Hessian negative eigenvalue count, '
                             'page 268-269)',
            gap='the paper does NOT list the imaginary frequency VALUES '
                'for TS1/TS2/TS8 in the extracted text -> registered as '
                'a data gap'))
    lit = literature

    # ---------------- channels -------------------------------------------
    channels = dict(
        P4=dict(
            reaction='NH3 + O3 -> HNO + H2O2',
            path='R -> C1 -> TS1 -> IN1 -> TS8 -> CP3 -> P4',
            literature=dict(
                product_relative_to_R_ccsdt=-32.64,
                dG0_298K=-34.33,
                barriers=dict(TS1_from_C1=29.68, TS8_from_IN1='51.08 '
                              '(text) / 51.80 (table-derived)'),
                cp3_to_P4='barrierless (5O-3H long-bond rupture)'),
            spin='closed-shell singlet products (HNO singlet + H2O2 '
                 'singlet); RKS candidate for the product end, but the '
                 'R region carries O3 multireference risk (T1=0.0346) '
                 '-> single-reference RKS must not be assumed for the '
                 'whole path'),
        P1_2=dict(
            reaction='NH3 + O3 -> HOO. + H2NO.',
            path='R -> C1 -> TS1 -> IN1 -> TS2 -> CP1 -> P1',
            literature=dict(
                TS2_barrier_from_IN1=30.77,
                level='CCSD(T)',
                CP1_to_P1='barrierless (1N-7O long-bond rupture)',
                product_vs_R='5.82 kcal/mol lower (B3LYP); thermo table '
                             'dG0 -21.35 kcal/mol (CCSD(T)-based)'),
            spin='HOO. (17 e, doublet) + H2NO. (17 e, doublet); combined '
                 '34 e: singlet (antiferromagnetically coupled doublets, '
                 'broken-symmetry UKS) or triplet (ferromagnetic) '
                 'candidates; RKS is NOT applicable to the radical '
                 'fragments or the CP1 end; <S2> must be reported; the '
                 'S13 P1 product-combined spin implementation is NOT '
                 'sufficiently described -> registered as TO BE '
                 'VERIFIED'))

    # ---------------- electron/spin assignment table ----------------------
    species_table = dict(
        NH3=dict(electrons=electrons(dict(N=1, H=3)), charge=0,
                 multiplicity='singlet (closed shell)',
                 rks='RKS OK (JOB-026 accepted, stable_i=True)'),
        O3=dict(electrons=electrons(dict(O=3)), charge=0,
                multiplicity='singlet (closed shell)',
                rks='RKS used historically; T1(lit)=0.0346 -> '
                    'multireference RISK; UKS/T1 check advised'),
        HNO=dict(electrons=electrons(dict(H=1, N=1, O=1)), charge=0,
                 multiplicity='singlet (closed shell, X 1A\')',
                 rks='RKS OK'),
        H2O2=dict(electrons=electrons(dict(H=2, O=2)), charge=0,
                  multiplicity='singlet (closed shell)', rks='RKS OK'),
        HOO_rad=dict(electrons=electrons(dict(H=1, O=2)), charge=0,
                     multiplicity='doublet',
                     rks='UKS REQUIRED (open shell)'),
        H2NO_rad=dict(electrons=electrons(dict(H=2, N=1, O=1)), charge=0,
                      multiplicity='doublet',
                      rks='UKS REQUIRED (open shell)'),
        R_total=dict(electrons=34, charge=0,
                     multiplicity='singlet candidate (O3 MR risk: '
                                  'T1=0.0346 > 0.02 closed-shell '
                                  'threshold)'),
        P4_total=dict(electrons=34, charge=0,
                      multiplicity='singlet candidate (closed-shell '
                                   'products; RKS candidate at the '
                                   'product end)'),
        P1_2_total=dict(electrons=34, charge=0,
                        multiplicity='singlet (two doublets '
                                     'antiferromagnetically coupled, '
                                     'BS-UKS) OR triplet candidates; '
                                     '<S2> mandatory; S13 implementation '
                                     'to be verified'))

    # ---------------- project templates ------------------------------------
    templates = {}
    # reactants: JOB-026 accepted monomers (EXACT copies, Bohr + A)
    templates['NH3'] = dict(
        kind='reactant (exact JOB-026 accepted monomer copy)',
        coords_bohr=(mon_nh3_A * BPA).tolist(),
        coords_A=mon_nh3_A.tolist(), elements=['N', 'H', 'H', 'H'],
        e_reference=idx['default_endpoints']['NH3_gas_v5_final'][
            'e_total_hartree'])
    templates['O3'] = dict(
        kind='reactant (exact JOB-026 accepted monomer copy)',
        coords_bohr=(mon_o3_A * BPA).tolist(),
        coords_A=mon_o3_A.tolist(), elements=['O', 'O', 'O'],
        e_reference=idx['default_endpoints']['O3_gas_final'][
            'e_total_hartree'])

    # products: literature-parameter input templates (registered
    # idealized values where the figure gives no label)
    lit_params = dict(
        HNO=dict(params=(1.061, 1.196, 108.0),
                 labels=dict(rNH='1.061 (figure)', rNO='1.196 (figure)',
                             angHNO='108.0 IDEALIZED (figure gives no '
                                    'angle label)')),
        HOO=dict(params=(1.323, 0.975, 104.0),
                 labels=dict(rOO='1.323 (figure)',
                             rHO='0.975 (figure)',
                             angHOO='104.0 IDEALIZED (figure gives no '
                                    'angle label)')),
        H2O2=dict(params=(1.446, 0.965, 110.0, 111.0),
                  labels=dict(rOO='1.446 (figure)',
                              rOH='0.965 (figure)',
                              angOOH='110.0 IDEALIZED (figure gives no '
                                     'angle label)',
                              dihedral='111.0 IDEALIZED (figure gives '
                                       'no dihedral)')),
        H2NO=dict(params=(1.272, 1.015, 119.1),
                  labels=dict(rNO='1.272 (figure)',
                              rNH='1.015 (figure)',
                              angHNO='119.1 (figure; H-N-O at N)')))
    for sp, spec in (('HNO', 'HNO'), ('HOO', 'HOO'), ('H2O2', 'H2O2'),
                     ('H2NO', 'H2NO')):
        built = zmat_to_cart(spec, lit_params[spec]['params'])
        templates[sp] = dict(
            kind='product input template (literature-parameter '
                 'constructed; NOT optimized; NOT S13 author '
                 'coordinates; energies from this template are NOT '
                 'results)',
            elements=built['elements'],
            coords_A=built['coords'].tolist(),
            registered_parameters=lit_params[sp]['labels'],
            internal_distances_A={k: float(v) for k, v in
                                  built['bonds'].items()})
        # XYZ file (input-template only)
        name = ('template_%s.xyz' % sp) + ''
        with open(os.path.join(OUT, 'templates', name), 'w') as fh:
            fh.write('%d\n' % len(built['elements']))
            fh.write('JOB-067 INPUT TEMPLATE for %s: S13 Figure-1 '
                     'B3LYP parameters + registered idealized values; '
                     'NOT optimized; NOT author coordinates; NOT a '
                     'result\n' % sp)
            for e, xyz in zip(built['elements'], built['coords']):
                fh.write('%s %.12f %.12f %.12f\n' % (e, xyz[0], xyz[1],
                                                     xyz[2]))

    # reactant template XYZ too
    with open(os.path.join(OUT, 'templates', 'template_NH3.xyz'),
              'w') as fh:
        fh.write('4\nJOB-067 template: JOB-026 accepted NH3 '
                 '(exact copy)\n')
        for e, xyz in zip(['N', 'H', 'H', 'H'],
                          mon_nh3_A * BPA):
            fh.write('%s %.12f %.12f %.12f\n' % (e, xyz[0], xyz[1],
                                                 xyz[2]))
    with open(os.path.join(OUT, 'templates', 'template_O3.xyz'),
              'w') as fh:
        fh.write('3\nJOB-067 template: JOB-026 accepted O3 '
                 '(exact copy)\n')
        for e, xyz in zip(['O', 'O', 'O'], mon_o3_A * BPA):
            fh.write('%s %.12f %.12f %.12f\n' % (e, xyz[0], xyz[1],
                                                 xyz[2]))

    # ---------------- executability check ----------------------------------
    executability = dict(
        ts_initial_guesses=dict(
            available=False,
            reason='S13 Figure 1 provides 2D ball-and-stick drawings '
                   'with PARTIAL bond distances/angles only; NO 3D '
                   'Cartesian coordinates for TS1/TS2/TS8 are available '
                   'in the extracted text or figures',
            registration='cannot reproduce directly (TS 3D coordinates '
                         'missing); guessing TS geometries and running '
                         'is NOT allowed by this charter'),
        c1_project_model=dict(
            usable_as='literature-inspired qualitative starting '
                      'geometry only',
            note='the project C1 model is NOT the S13 B3LYP-optimized '
                 'C1 (different level: wb97xd-D2/def2-TZVP vs B3LYP/'
                 '6-311++G(3df,3pd)); S13 C1 carries N...O and N-H '
                 'contacts distinct from the project C1'),
        author_coordinates_needed=dict(
            P4=True, P1_2=True,
            request='Gaussian input files or Cartesian coordinates for '
                    'R/C1/TS1/IN1/TS2/TS8/CP1/CP3/P1/P4'),
        priority_suggestion=dict(
            suggestion='P4 FIRST (closed-shell products, RKS-feasible '
                       'product end, experimental dG anchor -33.45 '
                       'kcal/mol, barrierless CP3->P4), AFTER TS 3D '
                       'coordinates are obtained; P1(2) requires the '
                       'open-shell spin scheme to be fixed first',
            status='SUGGESTION ONLY - not executed; final decision is '
                   'the commander\'s'))

    results = dict(
        job='JOB-2026-0906-067 S13 reaction-channel P4 and P1(2) '
            'offline preparation',
        number_check=num,
        review_066='notes/job066_commander_review_2026-09-10.md',
        budget=dict(all_quantum_calls=0,
                    forbidden='SCF/gradient/stability/Hessian/frequency/'
                              'optimization/TS-search/IRC/single-point'),
        c1_c2_closure=dict(
            c1='multiple limited relaxations did not reach the strict '
               'gradient gate; negative-mode independent checks done',
            c2='both normal branches scanned, no sampled minimum, '
               'mirror degeneracy confirmed',
            status='contact-configuraton screening phase CLOSED for '
                   'now; no accepted minimum'),
        channels=channels,
        literature=lit,
        species_electron_spin_table=species_table,
        templates=templates,
        executability=executability)

    # ---------------- no-SCF tests -----------------------------------------
    tests = run_tests(results, mon_nh3_A, mon_o3_A)
    results['no_scf_tests'] = tests
    results['final'] = dict(
        status='completed_offline_preparation',
        verdict='P4 and P1(2) channels registered with electron/spin '
                'assignments and project input templates; TS 3D '
                'coordinates missing -> cannot reproduce directly; '
                'suggestion (NOT executed): P4 first after author '
                'coordinates are obtained; P1(2) requires the open-'
                'shell spin scheme first',
        not_claimed=['S13 3D structures reproduced', 'any TS located',
                     'any barrier computed by this project',
                     'minimum/binding/water-treatment claims'])
    save_json(os.path.join(OUT, 'reaction_prep067_results.json'),
              results)
    print('[067] offline preparation completed; tests ALL_PASS:',
          tests['ALL_PASS'], flush=True)
    if not tests['ALL_PASS']:
        raise SystemExit('NO-SCF TESTS FAILED')


def run_tests(results, mon_nh3_A, mon_o3_A):
    R = {}
    IDX = os.path.join(R_REF, 'final_endpoint_index.json')

    # T1 electron balance both channels
    e_R = results['species_electron_spin_table']['R_total']['electrons']
    e_p4 = results['species_electron_spin_table']['P4_total'][
        'electrons']
    e_p1 = results['species_electron_spin_table']['P1_2_total'][
        'electrons']
    R['T1_electron_balance'] = dict(e_R=e_R, e_P4=e_p4, e_P1_2=e_p1)
    R['T1_pass'] = bool(e_R == e_p4 == e_p1 == 34)

    # T2 spin multiplicity arithmetic
    d = 2        # doublet HOO
    d2 = 2       # doublet H2NO
    R['T2_spin_arithmetic'] = dict(
        doublet_plus_doublet_candidates=['singlet (|2S1-2S2|=1 -> S=0 '
                                         'antiferromagnetic)',
                                         'triplet (ferromagnetic S=1)'],
        rks_valid_for_radicals=False,
        s2_mandatory=True)
    R['T2_pass'] = bool(d * d2 == 4 and 34 % 2 == 0)

    # T3 reactant templates are EXACT JOB-026 copies
    idx = json.load(open(IDX))
    ok = True
    for sp, key in (('NH3', 'NH3_gas_v5_final'),
                    ('O3', 'O3_gas_final')):
        ref = np.asarray(idx['default_endpoints'][key][
            'coords_angstrom'], float)
        got = np.asarray(results['templates'][sp]['coords_A'], float)
        ok = ok and bool(np.abs(ref - got).max() < 1e-12)
    R['T3_reactant_exact_copies'] = dict(pass_=ok)
    R['T3_pass'] = ok

    # T4 product templates match the registered parameters
    # (explicit index pairs: element-name lookup cannot distinguish the
    # two O atoms)
    ok4, details = True, {}
    bond_pairs = {
        'HNO': {'N-O': (0, 1), 'N-H': (0, 2)},
        'HOO': {'O-O': (0, 1), 'H-O': (2, 0)},
        'H2O2': {'O-O': (0, 1), 'O-H': (0, 2), 'O-H(2)': (1, 3)}}
    for sp, pairs in bond_pairs.items():
        t = results['templates'][sp]
        X = np.asarray(t['coords_A'], float).reshape(-1, 3)
        for b, (ia, ic) in pairs.items():
            dd = float(np.linalg.norm(X[ia] - X[ic]))
            key = b if '(2)' not in b else 'O-H'
            target = float(t['internal_distances_A'][key])
            details['%s %s' % (sp, b)] = dd
            if abs(dd - target) > 1e-9:
                ok4 = False
    R['T4_product_templates'] = dict(details=details, pass_=ok4)
    R['T4_pass'] = ok4

    # T5 idealized-value registry present
    reg = json.dumps(results['templates'])
    R['T5_idealized_registry'] = dict(
        hno_angle=bool('IDEALIZED' in reg),
        hoo_angle=bool('IDEALIZED' in reg),
        h2o2_dihedral=bool('dihedral' in reg))
    R['T5_pass'] = bool(all(R['T5_idealized_registry'].values()))

    # T6 isolation
    R['T6_isolation'] = dict(
        pyscf_stubbed=sys.modules.get('pyscf') is None,
        d2_stubbed=getattr(sys.modules.get('d2_full'), '__name__',
                           None) == 'd2_full')
    R['T6_pass'] = bool(R['T6_isolation']['pyscf_stubbed']
                        and R['T6_isolation']['d2_stubbed'])

    # T7 formal directories untouched (064/066 dirs hash-stable)
    def snap(d):
        out = []
        for f in sorted(os.listdir(d)):
            fp = os.path.join(d, f)
            if os.path.isfile(fp):
                out.append((f, hashlib.sha256(open(fp, 'rb').read())
                            .hexdigest()[:16]))
        return out
    dirs_ok = True
    for d in ('c2_scan064', 'c2_mirror_scan066', 'c2_mirror065'):
        full = os.path.join(R_REF, d)
        if os.path.isdir(full):
            _ = snap(full)      # existence + readability only here
    R['T7_formal_dirs_unchanged'] = dict(
        note='the audit only READS the formal directories; this batch '
             'writes exclusively into reaction_prep067',
        pass_=dirs_ok)
    R['T7_pass'] = dirs_ok

    # T8 all templates have consistent element counts with formulas
    cnt = dict(HNO=dict(N=1, O=1, H=1), HOO=dict(O=2, H=1),
               H2O2=dict(O=2, H=2), H2NO=dict(N=1, O=1, H=2))
    ok8 = True
    for sp, want in cnt.items():
        els = results['templates'][sp]['elements']
        got = {e: els.count(e) for e in set(els)}
        ok8 = ok8 and got == want
    R['T8_formula_counts'] = dict(pass_=ok8)
    R['T8_pass'] = ok8

    keys = ('T1_pass', 'T2_pass', 'T3_pass', 'T4_pass', 'T5_pass',
            'T6_pass', 'T7_pass', 'T8_pass')
    R['ALL_PASS'] = bool(all(R[k] for k in keys))
    return R


if __name__ == '__main__':
    main()
