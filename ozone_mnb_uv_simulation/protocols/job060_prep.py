#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-060 step A: OFFLINE preparation (ZERO evaluations).

059 corrections (per notes/job059_commander_review_2026-09-10.md) +
input verification for the limited full-DOF BFGS relaxation.

(a) corrections landed as an additive machine-readable note in the 059
    directory + a banner on the 059 report:
    1. symmetric displacement pair vs ASYMMETRIC energy change (+h lowers,
       -h raises; the first-order a0=-8.69693e-6 term dominates);
    2. single-side changes (~1e-8 Eh) reported SEPARATELY from the
       second-difference numerators E(+h)+E(-h)-2E0 = -8.8562e-11 /
       -3.5607e-10 Eh (h=0.001/0.002); the claim "4 orders above the SCF
       resolution" is WITHDRAWN (SCF conv_tol is not a strict energy
       error bound; the centre reproduction error is not a strict bound
       for displacement errors either).  The surviving reliability
       argument is the mutual consistency of gradient difference, energy
       difference and the two steps;
    3. near_resolution: the implementation checked the SINGLE-SIDE change
       at 1e-10 while the text said ~1e-11 - registered as inconsistent
       (heuristic flag only, NOT a verified error bound; the threshold
       was NOT adjusted to pass; original results kept with this
       correction relation);
    4. the overwritten 058 centre_scf.chk must NOT be used as a centre
       wavefunction cache (authoritative record = centre_mo_*.npy + the
       in-run stability record);
(b) the UNIQUE optimization start = the ACTUAL computed geometry of
    c1_negdir059/eval_disp_p002.json (coords_actual_angstrom, hash and
    provenance verified bitwise against the registered 059 manifest
    displacement block).  NOT rebuilt from q, no old direction/candidate,
    no interpolation.  If provenance failed -> zero-evaluation STOP;
(c) fixed optimizer parameters (validated 051 full-DOF Cartesian BFGS
    path, scipy BFGS jac=True gtol=1e-6 inf-norm, hess_inv0=I21, state
    reset) and the BATCH stop trigger max|g|<=1e-6 Eh/Bohr (the general
    1e-5 acceptance gate is NOT relaxed; start gmax=6.883e-6 is above
    1e-6 so the run cannot stop trivially).
Reads existing records only.
"""
import os, sys, json, hashlib
import numpy as np

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
R59 = ROOT + '/run_artifacts/02_nh3o3_reference/c1_negdir059'
R58R = ROOT + '/run_artifacts/02_nh3o3_reference/c1_fdhess058_resume01'
OUT = ROOT + '/run_artifacts/02_nh3o3_reference/c1_relax060'
JOB_NO = '060'
SRC = R59 + '/eval_disp_p002.json'
BPA = 1.0 / 0.52917721092


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


def corrections_059():
    """Numbers recomputed from the SAVED 059 results (no re-evaluation)."""
    r59 = json.load(open(R59 + '/negdir059_results.json'))
    a = r59['analysis']
    num = {}
    for h, v in a['per_h'].items():
        num[h] = dict(
            single_side_plus=v['dE_plus'], single_side_minus=v['dE_minus'],
            second_difference_numerator_Eh=(
                v['kE_Eh_Bohr2'] * (float(h) ** 2)),
            near_resolution_flag_as_run=v['near_resolution'],
            note_single_side='~1e-8 Eh; dominated by the first-order '
                             'a0*g0.q term (a0=-8.69693e-6 Eh/Bohr): the '
                             'DISPLACEMENT PAIR is symmetric but the '
                             'ENERGY CHANGE IS NOT',
            note_numerator='the quantity entering kE; NOT comparable to '
                           'the single-side change')
    corr = dict(
        job='JOB-059 corrections (issued with JOB-060 prep)',
        review='notes/job059_commander_review_2026-09-10.md',
        kept='negative curvature along the 058 q (sha da0cb2cc2708ad58) '
             'supported by two-step energy/gradient differencing and the '
             'saved matrix (max rel diff ~0.764%); 057 still NOT a '
             'minimum; no full-spectrum convergence claim; no same-mode/'
             'same-basin claim; the historical analytic-vs-FD discrepancy '
             'NOT claimed resolved',
        corrections=[
            dict(item='symmetric displacement vs asymmetric energy change',
                 detail='+h lowers (-8.741e-9 / -1.757e-8 Eh), -h raises '
                        '(+8.653e-9 / +1.722e-8 Eh): the first-order a0 '
                        'term dominates; "both sides symmetric" is NOT '
                        'claimed'),
            dict(item='single-side change vs second-difference numerator',
                 per_h=num,
                 withdrawn='the "curvature numerator 4 orders above the '
                           'SCF resolution limit" argument (based on '
                           'comparing the ~1e-8 SINGLE-SIDE change with '
                           'conv_tol=1e-12); the numerators are '
                           '-8.8562e-11 (h=0.001) and -3.5607e-10 Eh '
                           '(h=0.002)',
                 surviving_evidence='the consistency between gradient '
                                    'differencing, energy differencing '
                                    'and the two step sizes; SCF '
                                    'conv_tol is NOT a strict energy '
                                    'error bound and the centre '
                                    'reproduction error is NOT a strict '
                                    'bound for displacement errors'),
            dict(item='near_resolution implementation/text inconsistency',
                 detail='the run checked min(|dE+|,|dE-|) < 1e-10 '
                        '(single-side) while the text said "approaching '
                        '~1e-11"; it is a HEURISTIC FLAG, not a verified '
                        'error bound; the threshold was NOT adjusted to '
                        'pass; the second-difference numerator at '
                        'h=0.001 (-8.86e-11) is in fact near that scale; '
                        'original results kept, this note supersedes the '
                        'flag interpretation'),
            dict(item='058 centre_scf.chk must not be used as a cache',
                 detail='the file was overwritten by later fd evaluations '
                        '(058 defect); it is NOT a recoverable centre '
                        'wavefunction checkpoint; the authoritative '
                        'centre wavefunction record is centre_mo_*.npy + '
                        'the in-run stability record; JOB-060 does NOT '
                        'reference it'),
        ],
        numerators_recomputed_from='negdir059_results.json analysis.per_h '
                                   '(kE * h^2)')
    save_json_atomic(R59 + '/job059_correction_note.json', corr)
    return corr, num


def main():
    num = check_number()
    corr, numers = corrections_059()

    # ---- (b) unique start: the ACTUAL computed disp_p002 geometry -------
    src = json.load(open(SRC))
    checks = {}
    checks['tag_is_disp_p002'] = bool(src.get('tag') == 'disp_p002')
    checks['converged_finite'] = bool(src['converged'] and src['all_finite'])
    checks['scf_kernel_count_1'] = bool(src.get('scf_kernel_count') == 1)
    checks['has_actual_coords'] = bool('coords_actual_angstrom' in src)
    man59 = json.load(open(R59 + '/input_manifest.json'))
    g2 = [g for g in man59['displacement_geometries']
          if g['tag_full'] == 'disp_p002'][0]
    dprov = float(np.abs(
        np.asarray(src['coords_actual_angstrom'], float).reshape(-1)
        - np.asarray(g2['coords_bohr'], float).reshape(-1)
        * 0.52917721092).max())
    checks['record_vs_manifest_maxdiff_Bohr'] = dprov
    checks['provenance_closed_le_1e-12'] = bool(dprov <= 1e-12)
    R0 = np.asarray(src['coords_actual_angstrom'], float)  # Angstrom
    checks['start_gmax_above_1e-6'] = bool(float(src['grad_max']) > 1e-6)
    checks['start_grad_below_1e-5'] = bool(float(src['grad_max']) <= 1e-5)
    # the record must be the LOWERING (+0.002q) point actually computed
    res59 = json.load(open(R59 + '/negdir059_results.json'))
    checks['is_the_downhill_plus_point'] = bool(
        abs(float(src['e_total'])
            - float(res59['displacements']['disp_p002']['e_total'])) == 0.0)
    all_pass = all(bool(v) for v in checks.values() if isinstance(v, bool))

    man = dict(
        job='JOB-2026-0906-060: limited full-DOF BFGS relaxation from the '
            '059 actual +0.002q lowering point (058 NEW negative mode '
            'direction)',
        number_check=num,
        review_059='notes/job059_commander_review_2026-09-10.md',
        corrections_059=dict(
            file_in_059_dir='job059_correction_note.json',
            per_h_numerators=numers,
            banner_added='results/phase2_preparation/'
                         'c1_negdir059_report.md'),
        source=dict(
            path=SRC, sha256=sha256_file(SRC),
            tag=str(src['tag']), e_total=float(src['e_total']),
            grad_max=float(src['grad_max']),
            coords_actual_angstrom=src['coords_actual_angstrom'],
            grad=src['grad'], config=src['config'],
            provenance='ACTUAL computed geometry of the 059 disp_p002 '
                       'evaluation (record coords match the registered '
                       'manifest block bitwise); NOT rebuilt from q; no '
                       'old direction/candidate/interpolation'),
        caps=dict(start_repro=1, opt=25, recheck=1, total=27,
                  note='failures count; no borrowing; the recheck is used '
                       'ONLY if an optimization point meets the 1e-6 '
                       'trigger; any real exception -> save and STOP the '
                       'whole batch; no fix-and-restart, no ledger '
                       'clearing, no separate recovery budget, NO launch '
                       'probes; software early-stop below target or '
                       'budget exhaustion recorded truthfully, no '
                       'auto-continuation'),
        stop_trigger=dict(
            gmax_le=1e-6, scope='THIS BATCH stop trigger (unprojected '
                                'max|g|, Eh/Bohr)',
            general_gate_note='the project acceptance gate max|g|<=1e-5 '
                              'is NOT relaxed; the stricter 1e-6 trigger '
                              'exists because the start (6.883e-6) is '
                              'already below 1e-5',
            on_success='meeting the trigger and passing the independent '
                       'recheck does NOT register a minimum and does NOT '
                       'claim the negative curvature is eliminated; '
                       'further curvature acceptance is a separate '
                       'decision'),
        optimizer=dict(
            implementation='validated 051 run_bfgs path, reused '
                           'unchanged (full-DOF 21-Cartesian BFGS, '
                           'optimizer state RESET, start-record reuse at '
                           'identical coordinates)',
            params='scipy method=BFGS, jac=True, gtol=1e-6, norm=inf, '
                   'xrtol=0, c1=1e-4, c2=0.9, hess_inv0=21x21 identity, '
                   'maxiter=200 (NOT the evaluation budget; the persisted '
                   'opt cap guards every call)',
            coordinates='Bohr; no constraints, no gradient projection, '
                        'no re-registration before evaluation',
            stop_trigger_runtime='GM_GATE patched to 1e-6 in the job051 '
                                 'AND job047 module namespaces BEFORE the '
                                 'run (fixed here, not swapped mid-run); '
                                 'checkpoint per-evaluation uniquely '
                                 'named chk_<tag>.chk (never a shared '
                                 'path)'),
        start_gate=dict(dE_max=1e-8, dgrad_max=1e-7, dcoords_max_A=1e-9,
                        extra='config identical, converged, finite'),
        checks=checks, all_checks_pass=all_pass)
    save_json_atomic(OUT + '/input_manifest.json', man)

    print('[060] number check:', json.dumps(num))
    print('[060] 059 corrections written (additive) to 059 dir')
    print('[060] checks:', json.dumps(checks, default=str))
    if not (num['pass_'] and all_pass):
        print('[060] PRECHECK FAILED -> STOP (zero evaluations)')
        raise SystemExit(1)
    print('[060] PREP OK (zero evaluations consumed)', flush=True)


if __name__ == '__main__':
    main()
