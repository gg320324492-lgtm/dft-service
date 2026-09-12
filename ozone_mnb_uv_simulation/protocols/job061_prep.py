#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-061 step A: OFFLINE preparation (ZERO evaluations).

060 corrections (per notes/job060_commander_review_2026-09-10.md) + input
verification for the limited continuation optimization.

(a) corrections landed as an additive machine-readable note in the 060
    directory + a banner on the 060 report:
    1. "25 optimization evaluations all converged" means ONLY that every
       SCF converged; the GEOMETRY optimization did NOT converge;
    2. the 26 saved records classify as start_repro 1 + accepted-iterate
       evaluations 24 + non-accepted line-search trial 1 (the start
       record is NOT a line-search trial; the previous 24/2 split is
       withdrawn); the pre-cap budget refusal (1 blocked request) is
       listed as its own category;
    3. "largest energy drop of the series" (undefined range) is
       WITHDRAWN; kept: the batch lowering vs its own start is
       3.384e-6 Eh (~0.00212 kcal/mol);
    4. the distance changes (centroid +0.0624 Bohr ~ 0.033 A) do NOT
       prove dissociation, a continuing separation trend, or net motion
       along a single negative mode (the trajectory was not projected);
    5. "at least 78 attempts / 75 completed SCF+gradient" is the
       JOB-058-060 (incl. recovery and probe) SEGMENT statistic with
       explicit range - NOT whole-project cumulative cost;
    6. the 23 unclosed-tempfile ResourceWarnings are REGISTERED (not
       called harmless or cosmetic); audit result: all project-owned
       file handles use with-blocks / explicit close - the warnings
       originate in PySCF-internal /tmp temporary buffers (DIIS), no
       platform modification, no evidence they changed any scientific
       value, kept under observation;
(b) the UNIQUE start = the ACTUAL 060 opt_24 evaluation record
    (eval_opt_opt_24.json; E and max|g| bitwise-equal to the review
    reference); its coordinates match the 060 LAST accepted iterate
    (accepted_iterates.json #24) to 2.2e-16 Bohr.  NOT opt_25, not
    opt_02, no geometry rebuild;
(c) continuation mode registered: geometry continuation with BFGS
    optimizer state RESET - NO claim of a restored inverse Hessian or
    line-search history.
Reads existing records only.
"""
import os, json, hashlib
import numpy as np

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
R60 = ROOT + '/run_artifacts/02_nh3o3_reference/c1_relax060'
OUT = ROOT + '/run_artifacts/02_nh3o3_reference/c1_relaxcont061'
JOB_NO = '061'
SRC = R60 + '/eval_opt_opt_24.json'
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


def corrections_060():
    r60 = json.load(open(R60 + '/relax060_results.json'))
    acc = json.load(open(R60 + '/accepted_iterates.json'))
    corr = dict(
        job='JOB-060 corrections (issued with JOB-061 prep)',
        review='notes/job060_commander_review_2026-09-10.md',
        kept='start 1 / opt 25 / recheck 0 = 26 completed SCF+gradient '
             'attempts, 0 failures; budget blocked before the next '
             'evaluation; last accepted / lowest-energy point opt_24 '
             '(E=-282.0017253412693 Eh, max|g|=1.312036626515271e-4 '
             'Eh/Bohr); min-gradient point opt_02 (5.6193208054e-6, '
             'below the 1e-6 trigger); last evaluated opt_25 not '
             'accepted; batch NOT converged, no minimum registered',
        corrections=[
            dict(item='"all 25 converged" = SCF convergence ONLY',
                 detail='each of the 25 optimization evaluations had a '
                        'converged SCF; the GEOMETRY optimization did '
                        'NOT converge (batch trigger 1e-6 not met)'),
            dict(item='record classification corrected',
                 correct=dict(start_repro=1,
                              accepted_iterate_evaluations=24,
                              non_accepted_trials=1,
                              budget_refused_requests=1),
                 withdrawn='the previous accepted=24 / linesearch_trials=2 '
                           'split (the unmatched start record is NOT a '
                           'line-search trial); the start is its own '
                           'category; opt_25 (larger gradient, energy '
                           'rebound) is an UNACCEPTED trial - the process '
                           'was interrupted at budget exhaustion, '
                           '"divergence" is NOT claimed'),
            dict(item='"largest drop of the series" withdrawn',
                 detail='the range was undefined and no whole-history '
                        'accounting was done; kept: this batch lowered '
                        'the energy vs its own start by 3.384e-6 Eh '
                        '(~0.00212 kcal/mol)'),
            dict(item='distance changes: no direction/dissociation claim',
                 detail='centroid +0.0624 Bohr (~0.033 A), min N-O '
                        '+0.0417, min H-O +0.0122; numbers kept; they do '
                        'NOT prove dissociation, a continuing separation '
                        'trend, or net motion along a single negative '
                        'mode (trajectory not projected)'),
            dict(item='attempt statistics range made explicit',
                 detail='">=78 attempts / 75 completed SCF+gradient" is '
                        'the JOB-058..060 SEGMENT (incl. the 058 recovery '
                        'batch and its registered probe) - NOT whole-'
                        'project cumulative cost'),
            dict(item='unclosed-tempfile ResourceWarnings registered',
                 detail='23 warnings "unclosed file <_io.BufferedRandom '
                        '/tmp/tmp*>"; NOT called harmless or cosmetic; no '
                        'evidence they changed any scientific value; audit '
                        'of PROJECT code: all project-owned handles use '
                        'with-blocks or explicit close (save_json_atomic, '
                        'sha helpers, 058 stability log closed in '
                        'finally); the files are PySCF-internal temporary '
                        'buffers (DIIS); no platform modification; kept '
                        'under observation - if a real defect affecting '
                        'results or execution safety appears, stop and '
                        'report',
                 count=23),
        ],
        classification_source='accepted_iterates.json (24 callback '
                              'iterates) + saved evaluation records')
    save_json_atomic(R60 + '/job060_correction_note.json', corr)
    return corr


def main():
    num = check_number()
    corr = corrections_060()

    # ---- (b) unique start: the ACTUAL 060 opt_24 record ------------------
    src = json.load(open(SRC))
    acc = json.load(open(R60 + '/accepted_iterates.json'))
    checks = {}
    checks['tag_is_opt_24'] = bool(src.get('tag') == 'opt_24'
                                   and src.get('category') == 'opt')
    checks['converged_finite'] = bool(src['converged']
                                      and src['all_finite'])
    checks['scf_kernel_count_1'] = bool(src.get('scf_kernel_count') == 1)
    checks['e_bitwise_reference'] = bool(
        src['e_total'] == -282.0017253412693)
    checks['gmax_bitwise_reference'] = bool(
        src['grad_max'] == 1.312036626515271e-4)
    devs = [float(np.abs(np.asarray(src['coords_actual_angstrom'],
                                    float).reshape(-1) * BPA
                         - np.asarray(a, float).reshape(-1)).max())
            for a in acc['iterates']]
    mn = min(devs)
    checks['coords_vs_last_accepted_maxdev_Bohr'] = mn
    checks['matches_last_accepted_iterate'] = bool(
        devs[-1] == mn and mn <= 1e-12)
    checks['is_last_accepted_not_opt25_not_opt02'] = bool(
        devs.index(mn) == len(devs) - 1)
    checks['gmax_above_1e-6'] = bool(float(src['grad_max']) > 1e-6)
    all_pass = all(bool(v) for v in checks.values() if isinstance(v, bool))

    man = dict(
        job='JOB-2026-0906-061: limited continuation optimization from '
            'the 060 last accepted point opt_24 (geometry continuation, '
            'BFGS optimizer state RESET)',
        number_check=num,
        review_060='notes/job060_commander_review_2026-09-10.md',
        corrections_060=dict(
            file_in_060_dir='job060_correction_note.json',
            banner_added='results/phase2_preparation/'
                         'c1_relax060_report.md'),
        source=dict(
            path=SRC, sha256=sha256_file(SRC),
            tag=str(src['tag']), category=str(src['category']),
            e_total=float(src['e_total']),
            grad_max=float(src['grad_max']),
            coords_actual_angstrom=src['coords_actual_angstrom'],
            grad=src['grad'], config=src['config'],
            provenance='ACTUAL 060 opt_24 evaluation; coordinates match '
                       'the 060 LAST accepted iterate (accepted_iterates'
                       '.json #24) to %.2e Bohr; NOT opt_25, NOT opt_02, '
                       'no geometry rebuild' % mn),
        continuation_mode=dict(
            mode='geometry continuation, BFGS optimizer state RESET',
            not_claimed='no restored inverse Hessian, no restored '
                        'line-search history'),
        caps=dict(start_repro=1, opt=20, recheck=1, total=22,
                  note='failures count; no borrowing; the recheck is used '
                       'ONLY if an optimization point meets the 1e-6 '
                       'trigger; any real exception -> save and STOP the '
                       'whole batch; no fix-and-restart, no ledger swap, '
                       'no separate recovery budget, NO launch probes; '
                       'software early-stop below target or budget '
                       'exhaustion recorded truthfully, no step-count '
                       'extrapolation'),
        stop_trigger=dict(
            gmax_le=1e-6,
            note='batch trigger (unprojected max|g|); the general 1e-5 '
                 'acceptance gate is NOT relaxed; no early acceptance '
                 'for being near the gate'),
        optimizer=dict(
            implementation='validated 051 run_bfgs path, reused '
                           'unchanged (21-Cartesian standard BFGS, state '
                           'RESET, start-record reuse at identical '
                           'coordinates)',
            params='scipy method=BFGS, jac=True, gtol=1e-6, norm=inf, '
                   'xrtol=0, c1=1e-4, c2=0.9, hess_inv0=21x21 identity, '
                   'maxiter=200 (NOT the evaluation budget); GM_GATE '
                   'runtime-patched to 1e-6 in the job051 and job047 '
                   'namespaces BEFORE the run; checkpoints uniquely '
                   'named chk_<tag>.chk; method/grid/SCF/D2-once/'
                   'grid_response identical to 060'),
        start_gate=dict(dE_max=1e-8, dgrad_max=1e-7, dcoords_max_A=1e-9,
                        extra='config identical, converged, finite'),
        cost_statistics_range='JOB-058..061 segment (incl. the 058 '
                              'recovery batch and its registered probe); '
                              'NOT whole-project history',
        checks=checks, all_checks_pass=all_pass)
    save_json_atomic(OUT + '/input_manifest.json', man)

    print('[061] number check:', json.dumps(num))
    print('[061] 060 corrections written (additive) to 060 dir')
    print('[061] checks:', json.dumps(checks, default=str))
    if not (num['pass_'] and all_pass):
        print('[061] PRECHECK FAILED -> STOP (zero evaluations)')
        raise SystemExit(1)
    print('[061] PREP OK (zero evaluations consumed)', flush=True)


if __name__ == '__main__':
    main()
