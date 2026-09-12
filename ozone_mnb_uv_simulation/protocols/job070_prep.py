#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-070 step A: OFFLINE preparation (ZERO evaluations).

P1(2) radical product endpoints (HOO., H2NO.) under the PROJECT METHOD:
wb97xd + project explicit D2 once, def2-TZVP, grid level 8,
grid_response=True, SCF 1e-12/1e-9, gas, charge 0, DOUBLETT (spin=1,
UKS).  Inputs = the JOB-067 templates (NOT author coordinates).  This
is the project method model - NOT an S13 3D reproduction; UKS +
<S²> is mandatory for the radical END; RKS is NOT applicable here;
the S13 T1 risks for the R/TS regions remain.

Per radical (hard, no borrowing): opt 15 (ALL new SCF+gradient
evaluations INCLUDING line-search trials) + recheck 1 (only on
unprojected max|g| <= 1e-5) + internal stability 1 (on a FRESH mf
CONVERGED at the accepted geometry - the JOB-068 flow-anomaly fix).
If one radical fails to converge within its cap: record truthfully,
KEEP its state, and continue with the OTHER radical (independent
tasks).  P1 separated-fragment energy computed ONLY if BOTH pass.

Total caps 34 attempts (this window's remaining global budget after
JOB-068 (26) + JOB-069 (2) is ~44 for the ~72 window cap: 34 fits).
"""
import os, json, hashlib

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
R_REF = ROOT + '/run_artifacts/02_nh3o3_reference'
T67 = R_REF + '/reaction_prep067/templates'
OUT = R_REF + '/p1_radicals070'
JOB_NO = '070'
IDX = R_REF + '/final_endpoint_index.json'
TEMPLATES = {
    'HOO': T67 + '/template_HOO.xyz',
    'H2NO': T67 + '/template_H2NO.xyz',
}
ELEMS = {'HOO': ['O', 'O', 'H'], 'H2NO': ['N', 'O', 'H', 'H']}
CAPS = {'hoo_opt': 15, 'hoo_recheck': 1, 'hoo_stab': 1,
        'h2no_opt': 15, 'h2no_recheck': 1, 'h2no_stab': 1}
# S13 P1(2) literature reference values (JOB-067 registration; for
# the QUALITATIVE comparison only - never merged into one numeric
# table with project results)
S13_P1 = dict(
    level='CCSD(T), S13 Table 1 / thermo table (page 269/275)',
    product_vs_R_ccsdt_kcalmol=-5.82,
    product_vs_R_b3lyp_kcalmol=-19.4,
    dG0_298K_kcalmol=-21.35,
    note='Table 1 relative energy is electronic (no ZPE); the thermo '
         'table dG0 includes thermal corrections; the project '
         'comparison value is an ELECTRONIC energy under the project '
         'method -> qualitative comparison ONLY')


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


def read_xyz(path):
    lines = open(path).read().splitlines()
    n = int(lines[0].split()[0])
    els, xyz = [], []
    for ln in lines[2:2 + n]:
        p = ln.split()
        els.append(p[0])
        xyz.append([float(p[1]), float(p[2]), float(p[3])])
    return els, [list(map(float, v)) for v in xyz]


def main():
    num = check_number()
    os.makedirs(OUT, exist_ok=True)

    molecules = {}
    checks = dict(number_check=bool(num['pass_']))
    for sp in ('HOO', 'H2NO'):
        els, coords_A = read_xyz(TEMPLATES[sp])
        ok_elem = bool(els == ELEMS[sp])
        checks['%s_template_elements' % sp] = ok_elem
        checks['%s_template_hash' % sp] = sha256_file(TEMPLATES[sp])
        molecules[sp] = dict(
            elements=els,
            coords_start_A=coords_A,
            coords_start_bohr=[[x * (1.0 / 0.52917721092)
                                for x in v] for v in coords_A],
            start_hash=hashlib.sha256(
                repr(coords_A).encode()).hexdigest()[:16],
            template_source=os.path.relpath(TEMPLATES[sp], ROOT),
            template_sha256=sha256_file(TEMPLATES[sp]),
            natoms=len(els),
            n_electrons=(17 if sp == 'HOO' else 17),
            provenance='JOB-067 input template (S13 Figure-1 B3LYP '
                       'parameters + registered idealized values; NOT '
                       'author coordinates; NOT an optimized structure)')

    idx = json.load(open(IDX))
    e_nh3 = idx['default_endpoints']['NH3_gas_v5_final'][
        'e_total_hartree']
    e_o3 = idx['default_endpoints']['O3_gas_final']['e_total_hartree']

    manifest = dict(
        job='JOB-2026-0906-070: P1(2) radical product endpoints '
            '(HOO., H2NO.) under the project method model',
        number_check=num,
        policy='notes/simulation_continuation_policy_2026-09-10.md + '
               'notes/autopilot_24h_plan_2026-09-11.md',
        model_status=dict(
            status_is='project method model radical endpoint '
                      'preparation',
            status_is_not='NOT an S13 3D structure or TS reproduction; '
                          'NOT author coordinates; NO TS1/TS2/CP1 '
                          'construction; UKS+<S2> is mandatory at the '
                          'radical end and RKS is NOT applicable; '
                          'stable_i does NOT prove reaction-path '
                          'reliability'),
        molecules=molecules,
        method=dict(
            xc='wb97xd (project -D2 attached)',
            xc_description='wb97xd (libxc) + project explicit D2 once',
            basis='def2-TZVP', grid_level=8, grid_response=True,
            scf_tol=[1e-12, 1e-9], charge=0, spin=1, uks=True,
            solvent='none (gas phase)',
            consistency='SCF/gradient settings identical to JOB-026 '
                        'monomer references and JOB-068 accepted '
                        'endpoints'),
        electron_state=dict(
            each_radical='17 electrons, charge 0, spin=1 (doublet, '
                         'UKS)',
            s2_reference_doublet=0.75,
            s2_rule='record mf.spin_square() after every kernel; spin '
                    'contamination delta = <S2> - 0.75 reported; '
                    'RKS convergence is NOT sufficient evidence',
            stability='internal orbital stability ONLY, on a FRESH mf '
                      'CONVERGED at the accepted geometry (JOB-068 '
                      'flow-anomaly fix), 1 attempt, after a PASSING '
                      'recheck'),
        stop_trigger=dict(gmax_le=1e-5,
                          note='unprojected Cartesian max|g|; only '
                               'this triggers the independent '
                               'recheck; failure to reach it within '
                               'cap = stop that radical, recorded '
                               'truthfully; the OTHER radical '
                               'continues (independent tasks)'),
        caps=CAPS,
        caps_total=sum(CAPS.values()),
        p1_fragment_energy=dict(
            formula='dE_P1frag = [E(HOO.) + E(H2NO.)] - [E(NH3) + '
                    'E(O3)]',
            e_nh3_job026=e_nh3, e_o3_job026=e_o3,
            e_monomer_sum=e_nh3 + e_o3,
            computed_only_if='BOTH radicals pass their rechecks - '
                             'otherwise NO incomplete P1 energy',
            scope='project-method electronic energy ONLY (no ZPE/'
                  'thermal/CP); NOT the S13 CCSD(T) value; NOT called '
                  'a P1 overall reaction energy; NOT aqueous free '
                  'energy; NOT water-treatment efficiency',
            qualitative_comparison=S13_P1),
        stop_rules=dict(
            real_exception='save and STOP the whole batch (no restart, '
                           'no ledger clearing, no borrowing)',
            not_converged='truthful registration, keep the state, '
                          'continue with the independent radical'),
        gates=dict(dE=1e-8, dgrad=1e-7, dcoords_A=1e-9))

    man_path = OUT + '/input_manifest.json'
    tmp = man_path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, man_path)
    print('[070] manifest written:', man_path)
    print('[070] checks:', checks)
    print('[070] number_check:', num)


if __name__ == '__main__':
    main()
