#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-058 resume-01 step A: OFFLINE preparation (ZERO new
evaluations).

(a) preserve original-failure evidence: hash the original 058 ledger /
    abort results / exec log / eval_centre.json / manifest / scripts
    AS-IS (the original stability attempt stays 'error' - never rewritten
    to done, never cleared);
(b) register the correct original counts:
      1. original centre SCF+full-gradient COMPLETED once and persisted;
      2. original internal-stability call occurred ONCE and RETURNED,
         but the 4-tuple was wrongly unpacked as 3 items -> results NOT
         saved (error preserved);
      3. fd displacements executed 0 times (no eval_fd_* files exist);
(c) leftover-process audit registered (Windows python.exe = project MCP
    servers + a local http.server only; WSL = unattended-upgrade only;
    no job058 compute process, no extra files in the abort window);
(d) resume manifest: NEW budget centre 1 + stability 1 + fd 42 (new
    SCF+gradient cap 43; stability listed separately); cumulative with
    the original batch: SCF+gradient <= 44, stability <= 2 (including
    the original failed call), cross-category attempts <= 46.  This is a
    NEW authorization, not a reuse or reset of the original budget;
    fd total stays 42; failures count; no borrowing between categories.
Reads existing records only.
"""
import os, json, hashlib, glob

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
S58 = ROOT + '/run_artifacts/02_nh3o3_reference/c1_fdhess058'
OUT = ROOT + '/run_artifacts/02_nh3o3_reference/c1_fdhess058_resume01'
AUTH = 'notes/job058_resume_authorization_2026-09-10.md'


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


ORIGINAL_FILES = [
    'budget_fdhess058.json', 'fdhess058_results.json', 'eval_centre.json',
    'exec_log.txt', 'input_manifest.json', 'mock_test_results.json',
]
ORIGINAL_SCRIPTS = [
    'protocols/job058_exec.py', 'protocols/job058_prep.py',
    'protocols/job058_test_mock.py',
]


def preserve_original_evidence():
    ev = {}
    for f in ORIGINAL_FILES:
        p = os.path.join(S58, f)
        ev[f] = dict(sha256=sha256_file(p), size=os.path.getsize(p))
    for f in ORIGINAL_SCRIPTS:
        p = os.path.join(ROOT, f)
        ev[f] = dict(sha256=sha256_file(p), size=os.path.getsize(p))
    return ev


def original_counts():
    """Read the original ledger/results AS-IS (read-only, never edited)."""
    led = json.load(open(S58 + '/budget_fdhess058.json'))
    res = json.load(open(S58 + '/fdhess058_results.json'))
    cen = json.load(open(S58 + '/eval_centre.json'))
    atts = [(a['category'], a['status']) for a in led['attempts']]
    counts = {c: sum(1 for k, s in atts if k == c)
              for c in ('centre', 'stability', 'fd')}
    done = {c: sum(1 for k, s in atts if k == c and s == 'done')
            for c in ('centre', 'stability', 'fd')}
    leftover_fd = glob.glob(os.path.join(S58, 'eval_fd_*.json'))
    reg = dict(
        attempts_category_status=atts,
        attempts_total=len(atts),
        original_registration=[
            'centre SCF+full-gradient COMPLETED 1x and the record is '
            'COMPLETE (eval_centre.json, converged/finite, scf_kernel_'
            'count=1)',
            'internal-stability call occurred 1x and RETURNED; the return '
            'was wrongly unpacked as 3 items ("too many values to unpack '
            '(expected 3)") -> results NOT saved; the error stays in the '
            'ledger',
            'fd displacements executed 0x (no eval_fd_* files, ledger has '
            'no fd attempts)',
        ],
        counts=counts, done_counts=done,
        centre_record_complete=dict(
            converged=bool(cen['converged']), finite=bool(cen['all_finite']),
            scf_kernel_count=int(cen['scf_kernel_count']),
            e_total=cen['e_total'], grad_max=cen['grad_max'],
            seconds=float(cen['seconds'])),
        abort_error=res.get('abort', {}).get('error'),
        final_status=res.get('final', {}).get('status'),
        leftover_eval_fd_files=leftover_fd,
        fd_undone_note='missing displacements are NOT interpolated; the '
                       'un-run 42 points are executed fresh under the NEW '
                       'resume budget')
    return reg


def main():
    ev = preserve_original_evidence()
    reg = original_counts()

    sanity = dict(
        original_ledger_intact=bool(
            reg['counts'] == dict(centre=1, stability=1, fd=0)
            and reg['done_counts']['centre'] == 1
            and reg['final_status'] == 'aborted'
            and 'too many values to unpack' in str(reg['abort_error'])),
        no_fd_leftover=bool(reg['leftover_eval_fd_files'] == []),
        original_centre_source_hash_still_valid=None,   # filled below
    )
    import sys
    sys.path.insert(0, ROOT + '/protocols')
    man58 = json.load(open(S58 + '/input_manifest.json'))
    cur = sha256_file(man58['centre']['path_wsl'])
    sanity['original_centre_source_hash_still_valid'] = bool(
        cur == man58['centre']['sha256'])
    sanity['all_pass'] = bool(sanity['original_ledger_intact']
                              and sanity['no_fd_leftover']
                              and sanity[
                                  'original_centre_source_hash_still_valid'])

    credential = dict(
        job='JOB-2026-0906-058 resume-01 recovery credential',
        authorization=AUTH,
        authorization_scope='stability interface fix (4-tuple contract) + '
                            'centre rebuild 1 + internal stability 1 + the '
                            'preregistered 42 displacements; NEW budget, '
                            'NOT a reuse/reset of the original',
        original_attempt=dict(category='stability', status='error',
                              error='too many values to unpack (expected 3)',
                              note='the stability CALL happened and returned '
                                   '4 items; the wrapper unpacked 3 -> '
                                   'post-evaluation wrapper interface '
                                   'exception; results not saved; preserved '
                                   'as-is, never rewritten to done'),
        registration=[
            dict(item='original centre SCF+full-gradient completed 1x',
                 evidence=dict(e_total=reg['centre_record_complete'][
                                   'e_total'],
                               grad_max=reg['centre_record_complete'][
                                   'grad_max'],
                               file='eval_centre.json',
                               sha256=ev['eval_centre.json']['sha256'])),
            dict(item='original stability call 1x, returned, unpack failed, '
                      'results not saved',
                 evidence=dict(error=reg['abort_error'],
                               file='fdhess058_results.json + exec_log.txt',
                               sha256=ev['fdhess058_results.json'][
                                   'sha256'])),
            dict(item='fd completed 0x',
                 evidence=dict(counts=reg['counts'],
                               leftover_eval_fd_files=[])),
            dict(item='leftover-process audit: no job058 compute process '
                      'and no extra computation left by the previous nohup '
                      'launch (audit at prep time; Windows python.exe = '
                      'project MCP servers x4 + local http.server only; '
                      'WSL = unattended-upgrade only; abort-window files = '
                      'the 6 known original files only)',
                 evidence=dict(no_active_jobs=True,
                               note='session exit is not proof of zero '
                                    'compute; the audit checked processes '
                                    'AND abort-window file creation')),
        ],
        original_ledger_preserved=reg['attempts_category_status'],
        sanity=sanity)
    save_json_atomic(OUT + '/recovery_credential.json', credential)
    save_json_atomic(OUT + '/original_evidence_hashes.json',
                     dict(files=ev, original_state=reg))

    man = dict(
        job='JOB-2026-0906-058 resume-01: stability 4-tuple interface fix + '
            'centre rebuild + internal stability + the preregistered 42 '
            'displacements (full-gradient central-difference Hessian of the '
            '057 candidate)',
        authorization=AUTH,
        caps_new=dict(centre=1, stability=1, fd=42, scf_grad_total_new=43,
                      note='failures count; no borrowing between '
                           'categories; on exception or non-convergence: '
                           'save and STOP; fd total stays 42 (all 42 points '
                           'still to run, none were done)'),
        cumulative=dict(
            scf_grad_max_total=44, stability_max_total=2,
            attempts_max_total=46,
            note='includes the original centre (1 done) and the original '
                 'failed stability call (1); NOT a reset'),
        original_scene=dict(
            dir_rel='run_artifacts/02_nh3o3_reference/c1_fdhess058',
            evidence_file='original_evidence_hashes.json',
            credential_file='recovery_credential.json',
            manifest_read_only=dict(
                path=S58 + '/input_manifest.json',
                sha256=ev['input_manifest.json']['sha256']),
            centre_record_read_only=dict(
                path=S58 + '/eval_centre.json',
                sha256=ev['eval_centre.json']['sha256']),
            note='the original directory is NEVER written by resume-01; '
                 'the original manifest (42 displacement geometries, '
                 'h=0.001 Bohr, order, configs) is reused READ-ONLY '
                 'verbatim - no re-selection of step sizes'),
        stability_fix=dict(
            actual_contract='(mo_i, mo_e, stable_i, stable_e)',
            conventions_source='052 verified handling (installed return '
                               'order recorded, raw reprs kept, stable_e '
                               'None means NOT checked)',
            log_rule='stability verbose log written DIRECTLY to a file '
                     'handle (no StringIO); restoring stdout/verbose and '
                     'closing the file happen in finally, so the log '
                     'survives contract errors and post-processing '
                     'exceptions',
            raw_rule='return status + raw reprs saved to '
                     'stability_raw_result.json BEFORE any summarising; '
                     'mo arrays saved as .npy with clear provenance; only '
                     'an explicit stable_i is True opens the 42-point '
                     'stage (False/None/interface error -> save and STOP, '
                     'no orbital following, no UKS switch)'),
        centre_gate=dict(
            refs=['original 058 eval_centre.json (rebuilt object must '
                  'reproduce it)',
                  '057 eval_recheck_recheck.json (the designated recheck '
                  'record, via the manifest centre block)'],
            dE_max=1e-8, dgrad_max=1e-7, dcoords_max_A=1e-9, gmax_le=1e-5,
            extra='SCF converged, finite, config identical'),
        sanity=sanity)
    save_json_atomic(OUT + '/input_manifest_resume01.json', man)

    print('[058r] original ledger preserved:', reg['attempts_category_status'])
    print('[058r] original registration: centre done 1 | stability error 1 '
          '(returned, unpack failed, not saved) | fd 0')
    print('[058r] sanity:', json.dumps(sanity))
    if not sanity['all_pass']:
        print('[058r] PREP SANITY FAILED -> STOP (zero new evaluations)')
        raise SystemExit(1)
    print('[058r] PREP OK (zero new evaluations consumed)', flush=True)


if __name__ == '__main__':
    main()
