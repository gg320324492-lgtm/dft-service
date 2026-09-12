#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-069 step A: OFFLINE preparation (ZERO evaluations).

P4 product frequency acceptance under the PROJECT METHOD.  Per the 24h
plan condition branch: JOB-068 accepted HNO only (H2O2 budget-exhausted
NOT accepted), therefore ONLY HNO enters this batch; NO P4 product
energy and NO dE_project is computed (would require BOTH products).

Content per molecule (HNO):
  1. anchor evaluation (fresh SCF + full gradient at the persisted
     JOB-068 recheck geometry; gate vs that persisted record);
  2. analytical DFT Hessian (grid_response=True) - pure DFT part;
  3. explicit project D2 Hessian (central FD of the analytic D2
     gradient, d2_full.d2_hess); D2 FD gradient calls counted
     SEPARATELY, never mixed into the SCF ledger;
  4. combined Hessian = DFT + D2, cross-checked against the attached
     kernel output to machine precision;
  5. translation/rotation projection (PySCF thermo conventions),
     ALL internal eigenvalues kept (negatives preserved, never
     deleted/abs-ed);
  6. cross-check of frequencies vs pyscf thermo.harmonic_analysis with
     the SAME matrix (frequencies ONLY - PySCF thermo() chemistry
     numbers are NOT used anywhere, known unit problems);
  7. thermochemistry from the project custom function
     run_baseline.thermochemistry, extracted VERBATIM via ast (the
     module itself is NOT imported: its module-level berny import is
     unavailable in this environment and the file is NOT modified);
     T=298.15 K, P=101325 Pa, sigma(HNO)=1 (Cs), mult=1.

Ledger (hard, no borrowing): hno_anchor 1 + hno_hess_dft 1 = 2 SCF
attempts.  Frequency/thermo postprocessing = 0 SCF.  Any real
exception -> save and STOP the whole batch (no restart, no ledger
clearing).  NO TS/IRC/P1 work in this batch.
"""
import os, json, hashlib, ast

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
R_REF = ROOT + '/run_artifacts/02_nh3o3_reference'
SRC68 = R_REF + '/p4_products068'
OUT = R_REF + '/p4_freq069'
JOB_NO = '069'
RB = ROOT + '/protocols/run_baseline.py'

RC_HNO = SRC68 + '/hno/eval_hno_recheck_recheck_hno.json'
STAB_HNO = SRC68 + '/stability_HNO_record.json'
RES68 = SRC68 + '/p4_products068_results.json'

METHOD = dict(
    xc='wb97xd (project -D2 attached)',
    xc_description='wb97xd (libxc) + project explicit D2 once',
    basis='def2-TZVP', grid_level=8, grid_response=True,
    scf_tol=[1e-12, 1e-9], charge=0, spin=0, rks=True,
    solvent='none (gas phase)',
    consistency='SCF/gradient/Hessian settings identical to JOB-026 '
                'monomer references and JOB-068 accepted endpoint')
CAPS = {'hno_anchor': 1, 'hno_hess_dft': 1}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for blk in iter(lambda: fh.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()[:16]


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


def extract_thermo_provenance():
    """Locate the verbatim thermochemistry function + its constants in
    run_baseline.py WITHOUT importing the module (no berny dependency,
    file untouched).  Returns the exact source segments to exec."""
    src = open(RB, encoding='utf-8').read()
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    segs, names = [], []
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id in ('TEMP', 'PRESS', 'HA2JMOL',
                                           'AMU'):
            segs.append(node.lineno)
            names.append(node.targets[0].id)
        elif isinstance(node, ast.FunctionDef) \
                and node.name == 'thermochemistry':
            segs.append(node.lineno)
            names.append('thermochemistry')
    need = ['TEMP', 'PRESS', 'HA2JMOL', 'AMU', 'thermochemistry']
    missing = [n for n in need if n not in names]
    if missing:
        raise RuntimeError('run_baseline.py extraction missing: %s'
                           % missing)
    segment = ''.join(''.join(lines[lo - 1:hi]) for lo, hi in
                      [(s, _end(tree, s)) for s in segs])
    return segment, sha256_file(RB)


def _end(tree, lineno):
    for node in tree.body:
        if getattr(node, 'lineno', None) == lineno:
            return getattr(node, 'end_lineno', lineno)
    return lineno


def main():
    num = check_number()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(OUT + '/hno', exist_ok=True)

    checks = dict(number_check=bool(num['pass_']))

    # ---- accepted endpoint provenance (JOB-068) ----
    rc = json.load(open(RC_HNO))
    stab = json.load(open(STAB_HNO))
    res68 = json.load(open(RES68))
    acc = res68['final']['accepted_molecules']['HNO']
    h2o2 = res68['final']['accepted_molecules']['H2O2']
    # the persisted runtime config must match THIS batch's method on
    # every runtime key (the record additionally carries the outcome
    # flag d2_attached; METHOD's rks/consistency are descriptive)
    expected_runtime = dict(
        xc=METHOD['xc'], d2_attached=True,
        grid_level=METHOD['grid_level'], scf_tol=METHOD['scf_tol'],
        basis=METHOD['basis'], charge=METHOD['charge'],
        spin=METHOD['spin'], grid_response=METHOD['grid_response'],
        solvent=METHOD['solvent'])
    checks['source_recheck_gates_pass'] = bool(
        rc['gate']['gates_pass'])
    checks['source_config_matches_method'] = bool(
        rc['config'] == expected_runtime)
    checks['source_stable_i_true'] = bool(
        stab.get('stable_i') is True
        and acc.get('stable_i') is True)
    checks['h2o2_not_accepted_gate'] = bool(h2o2.get('accepted') is False)
    if not all(checks.values()):
        raise RuntimeError('HARD STOP: source checks failed: %s'
                           % checks)

    coords_A = rc['coords_actual_angstrom']
    inv = 1.0 / 0.52917721092
    coords_bohr = [[x * inv, y * inv, z * inv] for x, y, z in coords_A]

    segment, rb_sha = extract_thermo_provenance()
    with open(OUT + '/hno/run_baseline_thermochemistry_verbatim.py',
              'w', encoding='utf-8') as fh:
        fh.write('# Verbatim ast-extraction from run_baseline.py '
                 '(sha256-16=%s); module NOT imported; file NOT '
                 'modified.\n%s' % (rb_sha, segment))

    manifest = dict(
        job='JOB-2026-0906-069: P4 product frequency acceptance '
            '(HNO ONLY) under the project method model',
        number_check=num,
        policy='notes/simulation_continuation_policy_2026-09-10.md + '
               'notes/autopilot_24h_plan_2026-09-11.md',
        model_status=dict(
            status_is='project method model endpoint frequency '
                      'acceptance',
            status_is_not='NOT an S13 3D structure or TS reproduction; '
                          'NOT author coordinates; NOT S13 CCSD(T); '
                          'frequency support of a local minimum does '
                          'NOT prove reaction-path reliability'),
        single_product_rule=dict(
            both_pass_required_for_p4_energy=False,
            hno_accepted=True, h2o2_accepted=False,
            consequence='ONLY HNO processed; E_products and dE_project '
                        'NOT computed (would need BOTH products); '
                        'JOB-026 monomer sum kept for context only'),
        accepted_endpoint=dict(
            source_job='JOB-2026-0906-068',
            recheck_record=os.path.relpath(RC_HNO, ROOT),
            recheck_record_sha256=sha256_file(RC_HNO),
            stability_record=os.path.relpath(STAB_HNO, ROOT),
            stability_record_sha256=sha256_file(STAB_HNO),
            results68=os.path.relpath(RES68, ROOT),
            e_recheck_Eh=rc['e_total'],
            grad_max=rc['grad_max'],
            coords_actual_angstrom=coords_A,
            coords_start_bohr=coords_bohr,
            elements=['N', 'O', 'H'],
            stable_i=True),
        method=METHOD,
        caps=CAPS,
        caps_total=sum(CAPS.values()),
        d2_accounting=dict(
            rule='D2 finite-difference gradient calls are classical '
                 '(no SCF); counted separately, NEVER in the SCF '
                 'ledger',
            expected_calls_explicit_d2_hess=18,   # 2*3*3
            expected_calls_combined_kernel=18),
        frequency=dict(
            units='wavenumbers cm^-1; negative value = imaginary '
                  'frequency (sign convention: sign(l)*sqrt(|l|))',
            mass_convention='atom_mass_list(isotope_avg=True) (PySCF)',
            projection='translation/rotation subspace rank 6 (PySCF '
                       'thermo._get_TR conventions); internal '
                       'eigenproblem in the orthonormal complement; '
                       'ALL internal modes kept, negatives preserved',
            au2wn_formula='((HARTREE2J/(ATOMIC_MASS*BOHR_SI**2))**0.5'
                          '/(2*pi)/LIGHT_SPEED_SI*1e-2) (thermo.py '
                          'L92-93)',
            cross_check='pyscf.hessian.thermo.harmonic_analysis with '
                        'the SAME Hessian matrix and masses; used for '
                        'FREQUENCIES ONLY'),
        thermochemistry=dict(
            source='protocols/run_baseline.py thermochemistry() '
                   '(project custom RRHO; PySCF thermo() chemistry '
                   'numbers NOT used anywhere)',
            source_sha256_16=rb_sha,
            extraction='verbatim ast segments -> '
                       'hno/run_baseline_thermochemistry_verbatim.py',
            T_K=298.15, P_Pa=101325.0,
            sigma=1, symmetry='Cs (bent planar) -> symmetry number 1',
            mult=1,
            e_elec_source='this batch anchor e_total (wb97xd+D2 total)',
            negative_mode_rule='if any internal mode < 0: minimum '
                               'support NOT registered and thermo '
                               'output flagged not_for_use'),
        decision=dict(
            zero_negative='register: supported as a local minimum '
                          'WITHIN this method and check scope',
            has_negative='register: NOT a local minimum in this scope '
                         '(or curvature pending); no minimum claim'),
        stop_rules=dict(
            real_exception='save and STOP the whole batch (no restart, '
                           'no ledger clearing, no borrowing)',
            anchor_gate='anchor must reproduce the persisted JOB-068 '
                        'recheck record: dE<=1e-8, dgrad<=1e-7, '
                        'dcoords<=1e-9 A, config match, converged, '
                        'finite',
            combined_hess_gate='|hess(attached kernel) - '
                               '(hess_dft + hess_d2)|_max <= 1e-10 '
                               'Eh/Bohr^2'),
        gates=dict(dE=1e-8, dgrad=1e-7, dcoords_A=1e-9,
                   combined_hess=1e-10))

    man_path = OUT + '/input_manifest.json'
    tmp = man_path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, man_path)
    print('[069] manifest written:', man_path)
    print('[069] checks:', checks)
    print('[069] number_check:', num)


if __name__ == '__main__':
    main()
