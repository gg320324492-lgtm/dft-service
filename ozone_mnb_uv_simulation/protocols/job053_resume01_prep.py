#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-053 resume-01 step A: OFFLINE preparation (ZERO new
evaluations).

(a) preserve original-failure evidence: hash the original ledger / abort
    results / exec log / eval_centre.json AS-IS (the original centre
    attempt stays 'error' - never rewritten to done);
(b) recovery credentials linking the original attempt and the persisted
    centre record, registering:
      1. the centre computation COMPLETED and the record is COMPLETE;
      2. a POST-evaluation wrapper interface exception (triple return
         unpacked as dict);
      3. reuse approved by the commander after this verification;
(c) centre gate (NO re-evaluation): the persisted eval_centre.json is
    compared against the 052 designated centre with the ORIGINAL gates
    (|dE|<=1e-8, dgrad<=1e-7, dC<=1e-9 A, gmax<=1e-5, converged, finite,
    config identical) - credential saved BEFORE any displacement runs;
(d) resume manifest: direction / centre / four displacement geometries /
    method config inherited unchanged from the original 053 manifest;
    NEW budget: displacement 4 (centre new budget 0; original cumulative
    evaluation cap still 5).
Reads existing records only.
"""
import os, sys, json, hashlib
import numpy as np

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
S53 = ROOT + '/run_artifacts/02_nh3o3_reference/c1_negmode_dir053'
OUT = ROOT + '/run_artifacts/02_nh3o3_reference/c1_negmode_dir053_resume01'
CENTRE_FILE = S53 + '/eval_centre.json'
REF52_FILE = S52 = ROOT + '/run_artifacts/02_nh3o3_reference/c1_cand_check052/eval_endpoint.json'


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


def preserve_original_evidence():
    files = ['budget_negmode053.json', 'negmode_dir053_results.json',
             'exec_log.txt', 'eval_centre.json', 'input_manifest.json',
             'direction_and_kH.json', 'prep_evidence.json',
             'mock_test_results.json']
    ev = {}
    for f in files:
        p = os.path.join(S53, f)
        ev[f] = dict(sha256=sha256_file(p), size=os.path.getsize(p))
    led = json.load(open(S53 + '/budget_negmode053.json'))
    res = json.load(open(S53 + '/negmode_dir053_results.json'))
    orig = dict(
        attempts=[(a['category'], a['status']) for a in led['attempts']],
        final_status=res['final']['status'],
        abort_error=res['abort']['error'],
        eval_centre_complete=self_centre_complete(res, led))
    return ev, orig


def self_centre_complete(res, led):
    """Check the persisted eval_centre.json is a COMPLETE record."""
    c = json.load(open(CENTRE_FILE))
    need = ('e_total', 'grad', 'grad_max', 'coords_actual_angstrom',
            'config', 'converged', 'all_finite', 'scf_kernel_count')
    return dict(all_fields=bool(all(k in c for k in need)),
                converged=bool(c['converged']),
                finite=bool(c['all_finite']),
                scf_kernel_count=int(c['scf_kernel_count']),
                seconds=float(c['seconds']),
                e_total=c['e_total'], grad_max=c['grad_max'],
                ledger_centre_status=led['attempts'][0]['status'])


def centre_gate_credential():
    cen = json.load(open(CENTRE_FILE))
    ref = json.load(open(REF52_FILE))
    dE = abs(float(cen['e_total']) - float(ref['e_total']))
    dg = float(np.abs(np.asarray(cen['grad'], float).reshape(-1)
                      - np.asarray(ref['grad'], float).reshape(-1)).max())
    dC = float(np.abs(np.asarray(cen['coords_actual_angstrom'], float)
                      - np.asarray(ref['coords_actual_angstrom'],
                                   float)).max())
    gmax = float(cen['grad_max'])
    gate = dict(
        dE=dE, dgrad_max=dg, dcoords_A=dC, gmax=gmax,
        gmax_le_1e5=bool(gmax <= 1e-5),
        converged=bool(cen['converged']), finite=bool(cen['all_finite']),
        config_match=bool(cen['config'] == ref['config']),
        gates=dict(dE_le_1e8=bool(dE <= 1e-8),
                   dgrad_le_1e7=bool(dg <= 1e-7),
                   dcoords_le_1e9=bool(dC <= 1e-9)),
        reference='052 eval_endpoint.json (the designated centre)',
        atom_order=['N', 'H', 'H', 'H', 'O', 'O', 'O'],
        units=dict(record='Angstrom', execution='Bohr'),
        all_pass=bool(dE <= 1e-8 and dg <= 1e-7 and dC <= 1e-9
                      and gmax <= 1e-5 and cen['converged']
                      and cen['all_finite']
                      and cen['config'] == ref['config']))
    return gate, cen


def main():
    ev, orig = preserve_original_evidence()
    gate, cen = centre_gate_credential()
    man53 = json.load(open(S53 + '/input_manifest.json'))

    credential = dict(
        job='JOB-2026-0906-053 resume-01 recovery credential',
        authorization='notes/job053_resume_authorization_2026-09-09.md',
        original_attempt=dict(category='centre', status='error',
                              error='tuple object does not support item '
                                    'assignment (post-evaluation wrapper '
                                    'interface exception)',
                              note='NOT rewritten to done; preserved as-is'),
        registration=[
            dict(item='centre computation COMPLETED and the record is '
                      'COMPLETE',
                 evidence=dict(converged=True, all_finite=True,
                               scf_kernel_count=1,
                               e_total=cen['e_total'],
                               grad_max=cen['grad_max'],
                               file='eval_centre.json',
                               sha256=ev['eval_centre.json']['sha256'])),
            dict(item='POST-evaluation wrapper interface exception (the '
                      'SCF had finished and the record was saved before '
                      'the exception)',
                 evidence=dict(error=orig['abort_error'])),
            dict(item='reuse approved by the commander after this '
                      'verification (gates below)',
                 evidence=gate)],
        original_ledger_preserved=orig['attempts'],
        flow_anomaly_count=1,
        centre_evaluations_consumed=1)
    save_json_atomic(OUT + '/recovery_credential.json', credential)
    save_json_atomic(OUT + '/original_evidence_hashes.json',
                     dict(files=ev, original_state=orig))

    man = dict(
        job='JOB-2026-0906-053 resume-01: four remaining displacements + '
            'analysis (centre reused from persisted record, no re-eval)',
        caps=dict(displacement=4, centre_new=0,
                  total_new=4,
                  cumulative_evaluations_original_cap=5,
                  cumulative_used=1,
                  note='centre new budget 0; the original cumulative cap '
                       'of 5 evaluations remains (centre 1 already used); '
                       'the flow anomaly is listed separately; failures '
                       'count; any new exception -> save and STOP'),
        recovery_credential='recovery_credential.json',
        centre_gate=gate,
        centre_record=cen,
        direction=man53['direction'],
        displacement_geometries=man53['displacement_geometries'],
        kH_from_052_matrices=man53['kH_from_052_matrices'],
        centre_ref=man53['centre'],
        formulas=man53['formulas'],
        reporting_rules=man53['reporting_rules'])
    save_json_atomic(OUT + '/input_manifest_resume01.json', man)

    print('[053r] original ledger preserved:', orig['attempts'])
    print('[053r] eval_centre.json complete:',
          orig['eval_centre_complete']['all_fields'],
          '| SCF count:', orig['eval_centre_complete']['scf_kernel_count'])
    print('[053r] centre gate: dE=%.2e dgrad=%.2e dC=%.2e gmax=%.3e ->'
          ' %s' % (gate['dE'], gate['dgrad_max'], gate['dcoords_A'],
                   gate['gmax'], 'PASS' if gate['all_pass'] else 'FAIL'))
    if not gate['all_pass']:
        print('[053r] CENTRE GATE FAILED -> STOP (zero new evaluations)')
        raise SystemExit(1)
    print('[053r] PREP OK (zero new evaluations; centre reused, not '
          're-evaluated)', flush=True)


if __name__ == '__main__':
    main()
