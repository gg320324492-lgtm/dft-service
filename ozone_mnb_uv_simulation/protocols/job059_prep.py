#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-059 step A: OFFLINE preparation (ZERO evaluations).

058 count corrections + directional two-step FD check setup for the NEW
058 negative mode (not the old 052/053 directions).

(a) merged per-attempt index of ALL THREE 058 ledgers (original, launch
    probe, formal resume) - original statuses unchanged; the registered
    probe attempt COUNTS (a missing log does not prove zero cost);
    withdrawn claims: "46 attempts covers all costs" and "fully
    budget-compliant" (cumulative is at least 47 attempts);
(b) machine-readable correction note added to the 058 resume directory
    (additive file; existing 058 evidence untouched): corrected counts,
    scipy cross-check characterisation (same-matrix eigenvalue
    post-processing check), no same-mode/same-basin claims from
    H-dominance, plus the chkfile provenance defect (centre_scf.chk was
    overwritten by the later fd evals; the recorded sha256_after_run
    belongs to the LAST fd point, the authoritative centre wavefunction
    record is centre_mo_*.npy + the in-run stability);
(c) direction = 058 NEW FD Hessian lowest internal mode (mode 0,
    -47.27 cm^-1) from resume01_results.json; verified already-Cartesian
    under the recorded nominal masses (||M^1/2 d|| = 1); q = d/||d|| with
    the largest-|component| entry positive; NO further mass division;
(d) centre = the ACTUAL 058 resume-01 centre geometry (x_bohr),
    cross-verified against the 057 designated recheck record;
(e) four displacement geometries R0 +/- 0.001 q and R0 +/- 0.002 q
    (Bohr, full precision), pair/along-q/orthogonality checks, fragment
    sanity; config identical to 058.
Reads existing records only.
"""
import os, json, hashlib
import numpy as np

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
R58R = ROOT + '/run_artifacts/02_nh3o3_reference/c1_fdhess058_resume01'
R58 = ROOT + '/run_artifacts/02_nh3o3_reference/c1_fdhess058'
R57 = ROOT + '/run_artifacts/02_nh3o3_reference/c1_fdopt057'
OUT = ROOT + '/run_artifacts/02_nh3o3_reference/c1_negdir059'
JOB_NO = '059'
SYMS = ['N', 'H', 'H', 'H', 'O', 'O', 'O']
STEPS = (0.001, -0.001, 0.002, -0.002)


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


def merged_attempt_index():
    """All three 058 ledgers, per attempt, original statuses unchanged."""
    srcs = [
        ('original_058', R58 + '/budget_fdhess058.json'),
        ('launch_probe', R58R + '/budget_resume01_launchprobe_archived.json'),
        ('formal_resume', R58R + '/budget_resume01.json'),
    ]
    index, n = [], 0
    for label, path in srcs:
        led = json.load(open(path))
        for a in led['attempts']:
            n += 1
            index.append(dict(
                merged_idx=n, source_ledger=label, source_file=os.path.relpath(path, ROOT),
                attempt_in_ledger=a['attempt'], category=a['category'],
                status=a['status'], started=a.get('started'),
                error=a.get('error'), tag=(a.get('note') or {}).get('tag')))
    done_scf = sum(1 for a in index
                   if a['category'] in ('centre', 'fd')
                   and a['status'] == 'done')
    scf_started = sum(1 for a in index if a['category'] in ('centre', 'fd'))
    stab = [a for a in index if a['category'] == 'stability']
    corr = dict(
        job='JOB-058 count correction (issued with JOB-059 prep)',
        review='notes/job058_commander_review_2026-09-10.md',
        merged_attempt_index=index,
        separated_counts=dict(
            attempts_total=len(index),
            attempts_total_by_source={
                k: sum(1 for a in index if a['source_ledger'] == k)
                for k in ('original_058', 'launch_probe', 'formal_resume')},
            scf_started_including_unfinished=scf_started,
            completed_scf_full_gradient=done_scf,
            stability_calls=len(stab),
            stability_by_status={s: sum(1 for a in stab
                                        if a['status'] == s)
                                 for s in ('done', 'error')},
            unfinished_or_uncertain=[
                '1 registered probe centre attempt (status pending; '
                'terminated mid-SCF; partial checkpoint '
                'launchprobe_centre_scf_partial.chk exists) -> cost '
                'uncertain, COUNTED in the totals; the ~30 s wall-clock '
                'bound of that single foreground probe is NOT '
                'generalizable',
                'possible extra computation from the reaped nohup launch: '
                'unverifiable from remaining logs; missing logs do NOT '
                'prove zero cost; listed separately as uncertain '
                'historical cost']),
        withdrawn_claims=[
            'the statement that 46 cumulative attempts cover all costs '
            '(the registered probe makes it at least 47)',
            'the statement that the whole batch is fully budget-compliant '
            '("全部核算合规")'],
        preserved='original ledgers, error/pending attempts, checkpoint '
                  'and formal results all preserved unchanged (nothing '
                  'cleared, deleted or overwritten)',
        scipy_cross_check_clarification='the 3.8858e-16 cross-check in '
            'the 058 modes compares eigenvalues of the SAME internal '
            'matrix via a second eigensolver (post-processing validation '
            'only); it is NOT a cm^-1 frequency error and NOT an '
            'independent electronic-structure verification; '
            'mass-weighted eigenvalue unit Eh/(Bohr^2*amu)',
        mode_identity_caveat='H-atom dominance of the 051 and 057 '
            'negative modes does NOT establish that they are the same '
            'soft coordinate, nor that the two candidate geometries lie '
            'in the same potential basin',
        chkfile_provenance_defect=dict(
            finding='in the 058 resume production_eval the runtime '
                    'chkfile wrapper pointed EVERY evaluation (centre '
                    'and all 42 fd points) at the same centre_scf.chk '
                    'path, so the file was overwritten by later points',
            consequence='the chkfile_sha256_after_run recorded in '
                        'resume01_results.json belongs to the LAST fd '
                        'point (fd_m20), not to the centre',
            authoritative_centre_wavefunction='centre_mo_coeff/energy/'
                'occ.npy (saved immediately after the centre eval, '
                'before stability) and the in-run stability record; the '
                'centre gate (dE=5.68e-13 vs the original centre) is '
                'unaffected; no scientific result invalidated',
            fixed_in_059='chkfile path is per-tag (no overwrite)'),
    )
    return corr


def main():
    num = check_number()

    # ---- (a) count corrections + additive correction note ---------------
    corr = merged_attempt_index()
    save_json_atomic(R58R + '/job058_count_correction.json', corr)

    # ---- (d) centre: the ACTUAL 058 resume-01 centre --------------------
    # execution coords live in the hash-pinned 058 ORIGINAL manifest
    # (endpoint_eval records carry only Angstrom coords); they are
    # cross-verified against the resume centre record AND the 057 recheck
    man58 = json.load(open(R58 + '/input_manifest.json'))
    cen = json.load(open(R58R + '/eval_centre.json'))
    recheck = json.load(open(R57 + '/eval_recheck_recheck.json'))
    R0 = np.asarray(man58['centre']['x_bohr'], float).reshape(7, 3)
    checks = {}
    BPA = 1.0 / 0.52917721092
    checks['R0_vs_resume_record_maxdiff_Bohr'] = float(np.abs(
        R0.reshape(-1)
        - np.asarray(cen['coords_actual_angstrom'], float)
        .reshape(-1) * BPA).max())
    checks['R0_vs_resume_record_le_1e-9'] = bool(
        checks['R0_vs_resume_record_maxdiff_Bohr'] <= 1e-9)
    checks['R0_vs_057recheck_x_bohr_maxdiff_Bohr'] = float(np.abs(
        R0.reshape(-1)
        - np.asarray(recheck['x_bohr'], float).reshape(-1)).max())
    checks['R0_cross_le_1e-9_Bohr'] = bool(
        checks['R0_vs_057recheck_x_bohr_maxdiff_Bohr'] <= 1e-9)
    checks['R0_vs_057recheck_dE'] = float(abs(
        cen['e_total'] - recheck['e_total']))
    checks['R0_matches_058_manifest_center'] = bool(
        man58['centre']['e_total'] == cen['e_total'])
    checks['centre_config_058_complete'] = bool(
        cen['config'].get('grid_response') is True
        and cen['config'].get('grid_level') == 8
        and cen['config'].get('d2_attached') is True)

    # ---- (c) direction: 058 NEW FD Hessian lowest internal mode ---------
    r58 = json.load(open(R58R + '/resume01_results.json'))
    m0 = r58['modes']['modes'][0]
    lam0 = float(r58['modes']['eigenvalues_Eh_Bohr2_amu'][0])
    d = np.asarray(m0['cart_norm_mode'], float).reshape(-1)
    masses = np.asarray(r58['modes']['masses'], float)  # [14,1,1,1,16,16,16]
    sqrt_m = np.repeat(np.sqrt(masses), 3)
    mw_norm = float(np.linalg.norm(d * sqrt_m))
    checks['mode_is_cartesian_converted'] = bool(abs(mw_norm - 1.0) < 1e-8)
    checks['mode0_is_negative'] = bool(m0['freq_cm1'] < 0)
    checks['mode0_freq_matches_registration'] = bool(
        abs(m0['freq_cm1'] - (-47.26667776143749)) < 1e-6)
    d_norm = float(np.linalg.norm(d))
    q = d / d_norm
    imax = int(np.argmax(np.abs(q)))
    if q[imax] < 0:
        q = -q
        sign_note = 'flipped so the largest-|component| entry is positive'
    else:
        sign_note = 'largest-|component| entry already positive'
    checks['q_norm_1'] = bool(abs(float(np.linalg.norm(q)) - 1.0) < 1e-12)
    checks['q_largest_entry_positive'] = bool(q[imax] > 0)
    # kH from the SAVED 058 matrices: kH = q^T H q (no mass weights)
    H_sym = np.load(R58R + '/h_fd_sym.npy')
    H_raw = np.load(R58R + '/h_fd_raw.npy')
    kH_sym = float(q @ H_sym @ q)
    kH_raw = float(q @ H_raw @ q)
    # consistency: d^T H_sym d should reproduce the mode-0 eigenvalue
    dHd_sym = float(d @ H_sym @ d)
    checks['kH_eigenvalue_consistency'] = bool(
        abs(dHd_sym - lam0) <= 1e-8 * max(1.0, abs(lam0)))

    # ---- (e) four displacement geometries -------------------------------
    geoms = []
    for t in STEPS:
        R = R0 + (t * q).reshape(7, 3)
        delta = (R - R0).reshape(-1)
        geoms.append(dict(
            tag_full='disp_p%03d' % int(round(abs(t) * 1000))
            if t > 0 else 'disp_m%03d' % int(round(abs(t) * 1000)),
            t_Bohr=t, h_Bohr=abs(t),
            coords_bohr=R.tolist(), coords_bohr_sha=sha_arr(R),
            step_norm_Bohr=float(np.linalg.norm(delta)),
            displacement_along_q_Bohr=float(delta @ q),
            orthogonal_residual_max_Bohr=float(
                np.linalg.norm(delta - (delta @ q) * q)),
            element_order=SYMS,
            unit='Bohr (full-precision floats)'))
    pair_ok = True
    for h in (0.001, 0.002):
        gp = [g for g in geoms if g['t_Bohr'] == h][0]
        gm = [g for g in geoms if g['t_Bohr'] == -h][0]
        diff = (np.asarray(gp['coords_bohr'], float)
                - np.asarray(gm['coords_bohr'], float)).reshape(-1)
        want = 2 * h * q
        if float(np.abs(diff - want).max()) > 1e-15:
            pair_ok = False
    checks['pairs_differ_by_2hq'] = bool(pair_ok)
    checks['disp_along_q_exact'] = bool(all(
        abs(g['displacement_along_q_Bohr'] - g['t_Bohr']) < 1e-15
        for g in geoms))
    checks['disp_orthogonal_residual_zero'] = bool(all(
        g['orthogonal_residual_max_Bohr'] < 1e-15 for g in geoms))
    # fragment sanity at the largest |t|
    R2 = np.asarray([g for g in geoms if abs(g['t_Bohr']) == 0.002][0]
                    ['coords_bohr'], float).reshape(7, 3)
    dNH = [float(np.linalg.norm(R2[i] - R2[0])) for i in (1, 2, 3)]
    dOO = float(np.linalg.norm(R2[5] - R2[4]))
    dmin = float(min(np.linalg.norm(R2[i] - R2[j])
                     for i in range(7) for j in range(i + 1, 7)))
    checks['fragments_intact'] = bool(min(dNH) > 1.5 and dOO > 2.0
                                      and dmin > 1.5)

    all_pass = all(bool(v) for v in checks.values() if isinstance(v, bool))
    man = dict(
        job='JOB-2026-0906-059: 058 count corrections + directional '
            'two-step FD check of the NEW 058 negative mode at the 057 '
            'candidate geometry',
        number_check=num,
        review_058='notes/job058_commander_review_2026-09-10.md',
        count_correction=dict(
            file_in_058_dir='job058_count_correction.json',
            attempts_total_at_least=47,
            completed_scf_full_gradient=44,
            stability_calls=2,
            withdrawn='46-covers-all-costs / fully budget-compliant',
            note='this batch adds at most 5 attempts (cap below); '
                 'failures count'),
        caps=dict(centre=1, displacement=4, total=5,
                  note='total cap 5 SCF+full-gradient attempts: centre 1 '
                       '+ four displacements 1 each; failures count; no '
                       'borrowing; any real execution exception -> save '
                       'and STOP the whole batch; no launch probes, no '
                       'separate recovery budget'),
        centre=dict(
            path=R58R + '/eval_centre.json',
            sha256=sha256_file(R58R + '/eval_centre.json'),
            e_total=float(cen['e_total']),
            grad_max=float(cen['grad_max']),
            x_bohr=R0.tolist(),
            x_bohr_source='058 ORIGINAL input_manifest.json centre block '
                          '(bitwise execution coords; hash-pinned; the '
                          'endpoint_eval record carries only Angstrom '
                          'coords)',
            manifest58_sha256=sha256_file(R58 + '/input_manifest.json'),
            coords_actual_angstrom=cen['coords_actual_angstrom'],
            grad=cen['grad'], config=cen['config'],
            cross_check_ref=dict(
                path=R57 + '/eval_recheck_recheck.json',
                sha256=sha256_file(R57 + '/eval_recheck_recheck.json'),
                e_total=float(recheck['e_total']),
                x_bohr=recheck['x_bohr'],
                grad=recheck['grad'],
                coords_actual_angstrom=recheck['coords_actual_angstrom'])),
        centre_gate=dict(
            refs=['058 resume-01 eval_centre.json (actual centre)',
                  '057 designated recheck record (cross-verification)'],
            dE_max=1e-8, dgrad_max=1e-7, dcoords_max_A=1e-9,
            extra='config identical; SCF converged; finite'),
        direction=dict(
            source='058 NEW FD Hessian (resume01_results.json modes[0], '
                   '-47.27 cm^-1); NOT the 052/053 directions',
            d_raw=d.tolist(),
            euclidean_norm_before_normalisation=d_norm,
            mw_norm_check=float(mw_norm),
            masses=masses.tolist(),
            masses_note='NOMINAL mass numbers as recorded by 058/052 '
                        'post-processing; used only for the '
                        'already-Cartesian verification, NOT for q',
            q=q.tolist(), q_sha=sha_arr(q),
            sign_convention=sign_note,
            mode0=dict(freq_cm1=float(m0['freq_cm1']),
                       eigenvalue=lam0,
                       external_overlap=float(m0['external_overlap'])),
            kH=dict(qTHq_H_sym=kH_sym, qTHq_H_raw=kH_raw,
                    dHd_sym_consistency=dHd_sym,
                    source_files=['h_fd_sym.npy', 'h_fd_raw.npy'],
                    unit='Eh/Bohr^2',
                    note='kH = q^T H q with Euclidean-normalised '
                         'Cartesian q; the mass-weighted mode eigenvalue '
                         'lam0 corresponds to d^T H d (checked)')),
        displacement_geometries=geoms,
        formulas=dict(
            a0='a0 = g0 . q (centre slope, non-zero centre gradient kept)',
            slope='(E(+h) - E(-h)) / (2h)',
            kE='[E(+h) + E(-h) - 2 E0] / h^2',
            kg='[g(+h) - g(-h)] . q / (2h)',
            kH='q^T H q from the SAVED 058 matrices (no recomputation)',
            reporting='absolute and relative differences + step-size '
                      'sensitivity between h=0.001 and h=0.002; single-'
                      'side energy lowering is NOT evidence of curvature '
                      'sign; if |dE| approaches the numerical resolution '
                      '(SCF conv 1e-12), report as-is, no auto extra '
                      'points'),
        checks=checks, all_checks_pass=all_pass,
        analysis_limits='a negative FD curvature is reported as "this '
                        'direction/step check supports negative '
                        'curvature" (no transition-state claim, no mode '
                        'following, no optimisation, no Hessian '
                        'recomputation, no thermochemistry, no minimum '
                        'registration); -47.27 cm^-1 remains a '
                        'single-step matrix prediction (frequency NOT '
                        'converged)')
    save_json_atomic(OUT + '/input_manifest.json', man)

    print('[059] number check:', json.dumps(num))
    print('[059] merged 058 attempts:', len(corr['merged_attempt_index']),
          '| correction note written to 058 dir (additive)')
    print('[059] checks:', json.dumps(checks, default=str))
    if not (num['pass_'] and all_pass):
        print('[059] PRECHECK FAILED -> STOP (zero-evaluation stage)')
        raise SystemExit(1)
    print('[059] PREP OK (zero evaluations consumed)', flush=True)


if __name__ == '__main__':
    main()
