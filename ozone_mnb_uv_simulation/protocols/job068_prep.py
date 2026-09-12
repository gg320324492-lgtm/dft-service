#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-068 step A: OFFLINE preparation (ZERO evaluations).

P4 product endpoints (HNO, H2O2) optimized under the PROJECT METHOD:
wb97xd + project explicit D2 once, def2-TZVP, grid level 8,
grid_response=True, SCF 1e-12/1e-9, gas, charge 0, singlet (RKS
candidate).  Inputs = the JOB-067 templates (S13 Figure-1 B3LYP
parameters + registered idealized values; NOT author coordinates, NOT
optimized structures).  Registered: this is the PROJECT METHOD MODEL -
not an S13 3D reproduction; RKS is a candidate for the P4 product END
only; the S13 T1 risks for R (0.0346) and TS1 (0.0452) remain.

Budget (hard, no borrowing): HNO opt 15 + H2O2 opt 15 + recheck 1 each
(only on meeting unprojected max|g|<=1e-5) + internal stability 1 each
= 34 attempts total.  Any real exception -> save and STOP the whole
batch (no restart, no ledger clearing, no cross-molecule borrowing).
NO frequency/thermochemistry/barrier/TS/IRC/P1(2)/water/UV work.
"""
import os, json, hashlib
import numpy as np

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
R_REF = ROOT + '/run_artifacts/02_nh3o3_reference'
T67 = R_REF + '/reaction_prep067/templates'
OUT = R_REF + '/p4_products068'
JOB_NO = '068'
IDX = R_REF + '/final_endpoint_index.json'
TEMPLATES = {
    'HNO': T67 + '/template_HNO.xyz',
    'H2O2': T67 + '/template_H2O2.xyz',
}
ELEMS = {'HNO': ['N', 'O', 'H'], 'H2O2': ['O', 'O', 'H', 'H']}
CAPS = {'hno_opt': 15, 'hno_recheck': 1, 'hno_stab': 1,
        'h2o2_opt': 15, 'h2o2_recheck': 1, 'h2o2_stab': 1}


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
    return els, np.asarray(xyz, float)


def main():
    num = check_number()
    os.makedirs(OUT, exist_ok=True)

    molecules = {}
    checks = dict(number_check=bool(num['pass_']))
    for sp in ('HNO', 'H2O2'):
        els, coords_A = read_xyz(TEMPLATES[sp])
        ok_elem = bool(els == ELEMS[sp])
        checks['%s_template_elements' % sp] = ok_elem
        checks['%s_template_hash' % sp] = sha256_file(TEMPLATES[sp])
        molecules[sp] = dict(
            elements=els,
            coords_start_A=coords_A.tolist(),
            coords_start_bohr=(coords_A * (1.0 / 0.52917721092)).tolist(),
            start_hash=hashlib.sha256(
                np.asarray(coords_A * (1.0 / 0.52917721092),
                           float).tobytes()).hexdigest()[:16],
            template_source=os.path.relpath(TEMPLATES[sp], ROOT),
            template_sha256=sha256_file(TEMPLATES[sp]),
            natoms=len(els),
            n_dof=3 * len(els),
            provenance='JOB-067 input template (S13 Figure-1 B3LYP '
                       'parameters + registered idealized values; NOT '
                       'author coordinates; NOT an optimized structure)')

    # monomer references for the product reference energy
    idx = json.load(open(IDX))
    e_nh3 = idx['default_endpoints']['NH3_gas_v5_final'][
        'e_total_hartree']
    e_o3 = idx['default_endpoints']['O3_gas_final']['e_total_hartree']
    checks['monomer_reference_present'] = True

    manifest = dict(
        job='JOB-2026-0906-068: P4 product endpoints (HNO, H2O2) '
            'optimized under the project method model',
        number_check=num,
        policy='notes/simulation_continuation_policy_2026-09-10.md '
               '(author data = later calibration evidence, NOT a '
               'blocker; project method model clearly labelled)',
        model_status=dict(
            status_is='project method model endpoint preparation',
            status_is_not='NOT an S13 3D structure or TS reproduction; '
                          'NOT author coordinates',
            rks_scope='RKS is a candidate for the P4 product END ONLY; '
                      'this does NOT prove the reactant R or TS regions '
                      'suit single-reference RKS; the S13 T1 risks '
                      '(R 0.0346, TS1 0.0452) remain'),
        molecules=molecules,
        method=dict(
            xc='wb97xd (libxc) + project explicit D2 once',
            basis='def2-TZVP', grid_level=8, grid_response=True,
            scf_tol=[1e-12, 1e-9], charge=0, spin=0, rks=True,
            solvent='none (gas phase)',
            consistency='SCF/gradient settings identical to JOB-026 '
                        'monomer references'),
        stop_trigger=dict(gmax_le=1e-5,
                          note='unprojected Cartesian max|g|; only this '
                               'triggers the independent recheck; '
                               'failure to reach it within cap = stop '
                               'that molecule, recorded truthfully'),
        caps=CAPS,
        caps_total=sum(CAPS.values()),
        reference_energy=dict(
            formula='dE_project = [E(HNO) + E(H2O2)] - [E(NH3) + E(O3)]',
            e_nh3_job026=e_nh3, e_o3_job026=e_o3,
            e_monomer_sum=e_nh3 + e_o3,
            scope='project-method electronic energy ONLY; NOT S13 '
                  'CCSD(T); no TS; no CP correction unless separately '
                  'authorized; NOT aqueous free energy; NOT water-'
                  'treatment efficiency; computed ONLY if BOTH products '
                  'pass their rechecks - otherwise only the accepted '
                  'molecule is reported, no incomplete P4 energy'),
        stability=dict(when='only after a PASSING independent recheck',
                       scope='internal orbital stability ONLY '
                             '(stable_i=True does NOT mean full '
                             'reaction-path reliability); wrapper '
                             'writes the verbose log DIRECTLY to a '
                             'file and parses the installed 4-tuple '
                             '(mo_i, mo_e, stable_i, stable_e)'),
        stop_rules=dict(
            real_exception='save and STOP the whole batch (no restart, '
                           'no ledger clearing, no cross-molecule '
                           'borrowing)',
            budget_exhausted='stop that molecule truthfully',
            one_product_passed='report only the accepted molecule; NO '
                               'incomplete P4 energy splicing'),
        checks=checks, all_checks_pass=all(checks.values()))
    tmp = OUT + '/input_manifest.json.tmp'
    with open(tmp, 'w') as fh:
        json.dump(manifest, fh, indent=2, default=str)
    os.replace(tmp, OUT + '/input_manifest.json')

    print('[068] number check:', json.dumps(num))
    print('[068] checks:', json.dumps(checks, default=str))
    if not manifest['all_checks_pass']:
        print('[068] PRECHECK FAILED')
        raise SystemExit(1)
    print('[068] PREP OK (zero evaluations)', flush=True)


if __name__ == '__main__':
    main()
