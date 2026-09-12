#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-048 step A: OFFLINE preparation (ZERO evaluations).

(a) batch-number check: 048 must be the next unoccupied number;
(b) charter section 3 - source verification of the 045 +0.02 Bohr
    actually-evaluated displacement point:
        * file belongs to the 045 budget ledger (attempt done);
        * direction provenance: q hash consistent with the 045 manifest;
        * actual energy below the 045 centre;
        * coordinates cross-checked against the 045 manifest geometry AND
          against the line reconstruction R0 + 0.02 q;
        * element order / coordinate units / full gradient / method config
          / source-file hash saved into the batch input manifest.
(c) charter section 2 - zero-eval correction computations for 047:
        * Hartree->kcal/mol conversion regenerated from raw values
          (047 published "~1e-7 Eh ~ 6e-8 kcal/mol", wrong by ~1e3);
        * the 047 "four-point quartic fit" reframed as a FINITE-POINT FIT
          ON THE FIXED STRAIGHT LINE R(s) = R0 + s q: explicit coefficient
          definitions, fit-predicted positions vs actually-evaluated
          geometries.

No SCF, no gradient, no displacement generation: pure bookkeeping.
"""
import os, sys, json, hashlib, glob, re
import numpy as np

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
S45 = ROOT + '/run_artifacts/02_nh3o3_reference/c1_negmode_scan045'
S47 = ROOT + '/run_artifacts/02_nh3o3_reference/c1_bidir_opt047'
OUT = ROOT + '/run_artifacts/02_nh3o3_reference/c1_fulldof_opt048'
JOBS = ROOT + '/jobs'
JOB_NO = '048'
CHARTER_Q_HASH = '59599f2a2c0be260'
SYMS = ['N', 'H', 'H', 'H', 'O', 'O', 'O']
ANG_PER_BOHR = 0.52917721092
BPA = 1.0 / ANG_PER_BOHR

SRC_FILE = S45 + '/eval_eval_displacement_+0.02.json'


def save_json_atomic(path, obj):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(obj, fh, indent=2, default=str)
    os.replace(tmp, path)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for blk in iter(lambda: fh.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def sha_arr(a):
    return hashlib.sha256(np.asarray(a, float).tobytes()).hexdigest()[:16]


def kcal_per_eh():
    from pyscf.data import nist
    return float(nist.HARTREE2J * nist.AVOGADRO / 4184.0)


# ---------------------------------------------------------------- (a) number
def check_number():
    used = set()
    for f in glob.glob(JOBS + '/JOB-*.md'):
        m = re.match(r'JOB-2026-0906-(\d{3})', os.path.basename(f))
        if m:
            used.add(m.group(1))
    nxt = '%03d' % (max(int(u) for u in used) + 1)
    return dict(used_max=max(used), next_free=nxt,
                pass_=bool(nxt == JOB_NO and JOB_NO not in used))


# ------------------------------------------------- (b) source verification
def verify_source():
    src = json.load(open(SRC_FILE))
    man45 = json.load(open(S45 + '/input_manifest.json'))
    bud45 = json.load(open(S45 + '/budget_scan045.json'))
    c45 = json.load(open(S45 + '/eval_eval_centre_centre.json'))

    checks = {}

    # (1) belongs to the 045 ledger as a completed attempt
    att = [a for a in bud45['attempts']
           if a.get('stage') == 'eval_displacement_+0.02']
    checks['ledger_attempt_found'] = bool(len(att) == 1)
    a = att[0] if att else {}
    checks['ledger_attempt_done'] = bool(a.get('status') == 'done')
    checks['ledger_energy_matches'] = bool(
        abs(float(a.get('e_total', 0)) - float(src['e_total'])) == 0.0)
    checks['ledger_gradmax_matches'] = bool(
        float(a.get('grad_max', 0)) == float(src['grad_max']))

    # (2) direction provenance
    checks['q_hash_consistent'] = bool(
        man45['direction']['q_hash'] == CHARTER_Q_HASH)
    q = np.asarray(man45['direction']['q_vector'], float).reshape(7, 3)

    # (3) energy below the 045 centre
    dE = float(src['e_total']) - float(c45['e_total'])
    checks['energy_below_045_centre'] = bool(dE < 0.0)

    # (4) coordinates: record vs 045 manifest t=+0.02 geometry
    g02 = [g for g in man45['displacement_geometries']
           if g['t_Bohr'] == 0.02][0]
    C_man = np.asarray(g02['coords_angstrom'], float)
    C_rec = np.asarray(src['coords_actual_angstrom'], float)
    dC_man = float(np.abs(C_rec - C_man).max())
    checks['coords_vs_manifest_maxdiff_A'] = dC_man
    checks['coords_vs_manifest_le_1e-9'] = bool(dC_man <= 1e-9)

    # (5) coordinates: line reconstruction R0 + 0.02 q (Bohr space)
    R0 = np.asarray(c45['coords_actual_angstrom'], float) * BPA
    R_line = R0 + 0.02 * q
    dC_line = float(np.abs(R_line - C_rec * BPA).max())
    checks['coords_vs_line_reconstruction_maxdiff_Bohr'] = dC_line
    checks['coords_vs_line_le_1e-9'] = bool(dC_line <= 1e-9)

    # (6) record self-consistency
    checks['coords_sha_matches'] = bool(
        src.get('coords_actual_sha') == sha_arr(C_rec))
    checks['grad_max_matches'] = bool(
        abs(float(src['grad_max'])
            - float(np.abs(np.asarray(src['grad'], float)).max())) == 0.0)
    checks['converged_finite'] = bool(src['converged'] and src['all_finite'])
    checks['config_complete'] = bool(
        src.get('config', {}).get('grid_response') is True
        and src['config'].get('grid_level') == 8
        and src['config'].get('d2_attached') is True)

    all_pass = all(bool(v) for k, v in checks.items()
                   if isinstance(v, bool))
    return dict(
        source_path_wsl=SRC_FILE,
        source_path_win='e:' + SRC_FILE.replace('/mnt/e', ''),
        source_sha256=sha256_file(SRC_FILE),
        element_order=SYMS,
        coordinate_units=dict(record='Angstrom (coords_actual_angstrom)',
                              optimizer='Bohr (21 Cartesian variables)',
                              ang_per_bohr_constant=ANG_PER_BOHR),
        e_total=float(src['e_total']),
        e_d2_Eh=float(src['e_d2_Eh']),
        e_dft_part_Eh=float(src['e_dft_part_Eh']),
        grad=src['grad'], grad_max=float(src['grad_max']),
        grad_sha=sha_arr(np.asarray(src['grad'], float)),
        coords_actual_angstrom=src['coords_actual_angstrom'],
        coords_actual_sha_record=src.get('coords_actual_sha'),
        coords_actual_sha_recomputed=sha_arr(C_rec),
        config=src['config'],
        displacement_t_Bohr=0.02, q_hash=CHARTER_Q_HASH,
        dE_vs_045_centre_Eh=dE,
        dE_vs_045_centre_kcal=dE * kcal_per_eh(),
        checks=checks, all_checks_pass=all_pass)


# ------------------------------- (c) zero-eval correction computations
def correction_numbers():
    K = kcal_per_eh()
    e_c = json.load(open(S45 + '/eval_eval_centre_centre.json'))
    e_p01 = json.load(open(S45 + '/eval_eval_displacement_+0.01.json'))
    e_m01 = json.load(open(S45 + '/eval_eval_displacement_-0.01.json'))
    e_p02 = json.load(open(S45 + '/eval_eval_displacement_+0.02.json'))
    e_m02 = json.load(open(S45 + '/eval_eval_displacement_-0.02.json'))
    e_p10 = json.load(open(S47 + '/eval_disp_disp_p10.json'))
    e_m10 = json.load(open(S47 + '/eval_disp_disp_m10.json'))
    E0 = float(e_c['e_total'])
    raw = {  # all from the raw record files (no hardcoded energies)
        'E_centre_045_s0': E0,
        'E_p01_045': float(e_p01['e_total']),
        'E_m01_045': float(e_m01['e_total']),
        'E_p02_045': float(e_p02['e_total']),
        'E_m02_045': float(e_m02['e_total']),
        'E_p10_047': float(e_p10['e_total']),
        'E_m10_047': float(e_m10['e_total']),
    }
    dE = {k: raw[k] - E0 for k in raw if k != 'E_centre_045_s0'}
    dE_kcal = {k: v * K for k, v in dE.items()}

    # odd/even decomposition on the fixed line R(s) = R0 + s q
    odd = {'h0.01': 0.5 * (dE['E_p01_045'] - dE['E_m01_045']),
           'h0.02': 0.5 * (dE['E_p02_045'] - dE['E_m02_045']),
           'h0.10': 0.5 * (dE['E_p10_047'] - dE['E_m10_047'])}
    even = {'h0.01': 0.5 * (dE['E_p01_045'] + dE['E_m01_045']),
            'h0.02': 0.5 * (dE['E_p02_045'] + dE['E_m02_045']),
            'h0.10': 0.5 * (dE['E_p10_047'] + dE['E_m10_047'])}
    a0 = float(np.dot(np.asarray(e_c['grad'], float).reshape(-1),
                      np.asarray(json.load(open(
                          S45 + '/input_manifest.json'))['direction']
                      ['q_vector'], float).reshape(-1)))

    # finite-difference curvatures k_FD(h) = 2*even(h)/h^2 (single-pair)
    k_fd = {h: 2.0 * even[h] / float(h[1:]) ** 2 for h in even}

    def even_fit(s1, e1, s2, e2):
        # e(s) = k2 s^2 + k4 s^4, exact 2-point solve
        A = np.array([[s1 ** 2, s1 ** 4], [s2 ** 2, s2 ** 4]], float)
        k2, k4 = np.linalg.solve(A, np.array([e1, e2], float))
        return float(k2), float(k4)

    def minimum_pred(k2, k4):
        if k2 < 0.0 < k4:
            s2 = -k2 / (2.0 * k4)
            dep = k2 * s2 + k4 * s2 ** 2
            return dict(s_pred_Bohr=float(np.sqrt(s2)),
                        depth_pred_Eh=float(dep),
                        depth_pred_kcal=float(dep * K))
        return None

    fits = {}
    k2, k4 = even_fit(0.01, even['h0.01'], 0.10, even['h0.10'])
    fits['A_exact_even_h0.01_h0.10'] = dict(
        definition='E_even(s) = k2*s^2 + k4*s^4 fitted exactly to the two '
                   'even-part values from the ±0.01 (045) and ±0.10 (047) '
                   'evaluated pairs, relative to the 045 centre',
        k2_Eh_Bohr2=k2, k4_Eh_Bohr4=k4,
        k2_kcal=k2 * K, k4_kcal=k4 * K,
        predicted_even_part_minimum=minimum_pred(k2, k4))
    k2b, k4b = even_fit(0.01, even['h0.01'], 0.02, even['h0.02'])
    fits['Aprime_exact_even_h0.01_h0.02'] = dict(
        definition='same definition, points ±0.01 and ±0.02 (both 045)',
        k2_Eh_Bohr2=k2b, k4_Eh_Bohr4=k4b,
        predicted_even_part_minimum=minimum_pred(k2b, k4b))
    # 3-even-point least squares (ill-conditioned; likely source of the
    # 047 published k4 ~ +2.2e-2)
    s3 = np.array([0.01, 0.02, 0.10])
    y3 = np.array([even['h0.01'], even['h0.02'], even['h0.10']])
    M = np.column_stack([s3 ** 2, s3 ** 4])
    sol, res_, rank_, sv = np.linalg.lstsq(M, y3, rcond=None)
    fits['B_lsq_even_3points'] = dict(
        definition='least squares of E_even(s) = k2*s^2 + k4*s^4 through '
                   'the three even values (h=0.01, 0.02, 0.10); solved by '
                   'SVD lstsq (moderate condition number ~2.5e3)',
        k2_Eh_Bohr2=float(sol[0]), k4_Eh_Bohr4=float(sol[1]),
        singular_values=[float(v) for v in sv],
        condition_number=float(sv[0] / sv[-1]) if sv[-1] > 0 else None,
        predicted_even_part_minimum=minimum_pred(float(sol[0]),
                                                 float(sol[1])))
    # 047 published values (from the 047 report section 8.1)
    k2p, k4p = -7.19e-5, 2.22e-2
    fits['published_047'] = dict(
        k2_Eh_Bohr2=k2p, k4_Eh_Bohr4=k4p,
        s_pred_Bohr_reported=0.057, depth_pred_Eh_reported=5.8e-8,
        s_pred_from_published_pair_Bohr=float(np.sqrt(
            -k2p / (2.0 * k4p))),
        depth_from_published_pair_Eh=float(-k2p ** 2 / (4.0 * k4p)),
        note='the published s*~0.057 Bohr does NOT follow from the '
             'published (k2,k4) pair (which gives s*~0.040 Bohr); the '
             'pair is therefore internally inconsistent and is withdrawn')

    conv = dict(
        kcal_per_Eh=K,
        factor_source='pyscf.data.nist HARTREE2J * AVOGADRO / 4184',
        published_047_statement='~1e-7 Eh ~ 6e-8 kcal/mol',
        published_047_corrected='1e-7 Eh = 6.2751e-5 kcal/mol '
                                '(published value wrong by ~1e3)',
        raw_energies_Eh=raw,
        dE_vs_045_centre_Eh=dE,
        dE_vs_045_centre_kcal_per_mol=dE_kcal,
        odd_part_Eh=odd, even_part_Eh=even,
        centre_slope_a0_Eh_per_Bohr=a0,
        fd_curvature_single_pair_Eh_Bohr2=k_fd,
        fits_on_fixed_line=fits)
    return conv


def main():
    num = check_number()
    src = verify_source()
    conv = correction_numbers()

    save_json_atomic(OUT + '/correction_evidence.json',
                     dict(job='JOB-2026-0906-048 zero-eval correction of 047',
                          nature='offline recomputation from raw records; '
                                 'no new evaluations; no new displacements',
                          conversion_and_fits=conv))
    man = dict(
        job='JOB-2026-0906-048 C1 limited full-DOF optimization from the '
            '045 +0.02 Bohr lowering point',
        number_check=num,
        caps=dict(start_repro=1, opt=19, recheck=1, total=21),
        caps_note='failures count within their category; no borrowing; '
                  'no auto-resume; stability/Hessian/frequency/CP = 0',
        source=src,
        start_gate=dict(dE_max=1e-8, dgrad_max=1e-7, dcoords_max_A=1e-9,
                        extra='SCF converged, config identical, results '
                              'finite; on PASS the reproduction record is '
                              'reused for the optimizer first request at '
                              'the same coordinates'),
        optimizer=dict(method='scipy L-BFGS-B', gtol=1e-6, ftol=1e-13,
                       maxcor=10, maxiter=19, n_variables=21,
                       variables='Cartesian, Bohr', constraints='none',
                       projection='none', per_step_registration='none',
                       state='freshly initialized (displacement-geometry '
                             'start, no warm start)'),
        acceptance=dict(unprojected_max_grad=1e-5,
                        recheck='at most one independent new-object '
                                'recheck; register stationary-point '
                                'candidate only on PASS'),
        zero_eval_correction=dict(
            note='notes/job048_047_correction.md',
            evidence='correction_evidence.json'))
    save_json_atomic(OUT + '/input_manifest.json', man)

    print('[048] number check:', json.dumps(num))
    print('[048] source all_checks_pass =', src['all_checks_pass'])
    for k, v in src['checks'].items():
        print('      %-42s %s' % (k, v))
    print('[048] dE(+0.02 vs 045 centre) = %.6e Eh = %.6e kcal/mol'
          % (src['dE_vs_045_centre_Eh'], src['dE_vs_045_centre_kcal']))
    print('[048] kcal/Eh = %.10f' % conv['kcal_per_Eh'])
    for h, v in conv['even_part_Eh'].items():
        print('[048] even(%s) = %+.6e Eh ; odd = %+.6e ; k_FD = %+.6e'
              % (h, v, conv['odd_part_Eh'][h],
                 conv['fd_curvature_single_pair_Eh_Bohr2'][h]))
    for name, f in conv['fits_on_fixed_line'].items():
        print('[048] fit %-28s k2=%+.6e k4=%+.6e pred=%s'
              % (name, f['k2_Eh_Bohr2'], f['k4_Eh_Bohr4'],
                 json.dumps(f.get('predicted_even_part_minimum'))))
    if not (num['pass_'] and src['all_checks_pass']):
        print('[048] PRECHECK FAILED -> STOP (no SCF will be started)')
        raise SystemExit(1)
    print('[048] PREP OK (zero evaluations consumed)', flush=True)


if __name__ == '__main__':
    main()
