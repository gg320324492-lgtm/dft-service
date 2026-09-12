#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-068 resume: HNO stability (fixed) + H2O2 full flow +
batch completion.

FLOW-ANOMALY DISCLOSURE (registered, not hidden): the first run aborted
AFTER the HNO optimization (9/15 evals, threshold met) and PASSING
recheck, because the stability invocation was applied to a freshly
constructed (never-kernel'ed) mf object whose scalar mo_occ crashes
numpy 2.5 (`where(mo_occ==2)` on a 0-d array).  No SCF was consumed by
the crash; the hno_stab slot is untouched (0/1).  This resume:
  1. archives the aborted results file (never overwritten);
  2. REUSES every completed evaluation (HNO opt 9 + recheck 1) without
     re-running them (no budget reset, no double spending);
  3. fixes the defect: the stability object is a FRESH mf CONVERGED at
     the accepted geometry (kernel() inside the hno_stab attempt);
  4. then runs the full H2O2 flow (opt 15 + recheck 1 + stab 1).
The anomaly is disclosed in the results and the final report.
"""
import os, sys, json, glob, time, shutil, traceback
import numpy as np

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
HERE = ROOT + '/protocols'
sys.path.insert(0, HERE)
import job047_exec as ex
import job068_exec as E

OUT = E.OUT
MANIFEST = E.MANIFEST
RESULTS = E.RESULTS
LEDGER = E.LEDGER
ABORTED_RESULTS = OUT + '/p4_products068_results_aborted01.json'
ABORTED_LOG = OUT + '/exec_log_aborted01.txt'


def main():
    man = json.load(open(MANIFEST))
    if not man['all_checks_pass']:
        raise RuntimeError('HARD STOP: manifest prechecks failed')

    # ---- archive the aborted-scene artifacts ----------------------------
    if os.path.exists(RESULTS) and not os.path.exists(ABORTED_RESULTS):
        shutil.copy(RESULTS, ABORTED_RESULTS)
    log_src = os.path.join(OUT, 'exec_log_aborted01.txt')
    prev_log = OUT + '/exec_log_first_run.txt'
    if os.path.exists(log_src) and not os.path.exists(prev_log):
        shutil.copy(log_src, prev_log)

    ledger = ex.Ledger(LEDGER, caps=man['caps'])

    # ---- verify the persisted HNO state ---------------------------------
    mdir = os.path.join(OUT, 'hno')
    opt_files = sorted(glob.glob(os.path.join(
        mdir, 'eval_hno_opt_*.json')))
    import glob as _glob
    rc_files = sorted(_glob.glob(os.path.join(
        mdir, 'eval_hno_recheck_*.json')))
    rc_file = rc_files[0] if rc_files else os.path.join(
        mdir, 'eval_recheck_HNO.json')
    if not (opt_files and os.path.exists(rc_file)):
        raise RuntimeError('HARD STOP: persisted HNO state missing - '
                           'cannot resume')
    recs = [json.load(open(f)) for f in opt_files]
    n_opt_done = ledger.count('hno_opt')
    if n_opt_done != len(recs):
        raise RuntimeError('HARD STOP: ledger/files mismatch (%d vs %d)'
                           % (n_opt_done, len(recs)))
    meeting = [r for r in recs if float(r['grad_max']) <= 1e-5]
    if not meeting:
        raise RuntimeError('HARD STOP: no persisted meeting point')
    mrec = meeting[-1]
    rc = json.load(open(rc_file))
    if not rc.get('gate', {}).get('gates_pass', False):
        raise RuntimeError('HARD STOP: persisted recheck did not pass')
    print('[068r] persisted HNO state verified: %d opt evals (meeting '
          '%s gmax=%.3e), recheck PASS (dE=%.2e) - reused WITHOUT '
          're-running' % (len(recs), mrec['tag'], mrec['grad_max'],
                          rc['gate']['dE']), flush=True)

    flow_anomaly = dict(
        event='first-run stability invocation aborted (flow anomaly, '
              'listed separately)',
        cause='stability applied to a freshly constructed mf that had '
              'never been through kernel(): scalar mo_occ crashes '
              'numpy 2.5 (`where(mo_occ==2)` on a 0-d array)',
        cost='zero SCF consumed by the crash; hno_stab slot untouched '
             '(0/1); HNO opt 9/15 + recheck 1/1 completed and reused',
        fix='the stability object is now a fresh mf CONVERGED at the '
            'accepted geometry (kernel() inside the hno_stab attempt)',
        archived=dict(aborted_results=ABORTED_RESULTS,
                      first_log=prev_log))

    results = dict(
        job='JOB-2026-0906-068: P4 product endpoints (HNO, H2O2) '
            'project-method optimization (resumed after the stability-'
            'preparation flow anomaly)',
        model_status=man['model_status'],
        method=man['method'],
        stop_trigger=man['stop_trigger'],
        reference_energy=man['reference_energy'],
        flow_anomaly=flow_anomaly,
        hno_reused=dict(opt_evals=len(recs),
                        meeting_tag=mrec['tag'],
                        meeting_e=mrec['e_total'],
                        meeting_gmax=mrec['grad_max'],
                        recheck_e=rc['e_total'],
                        recheck_gate=rc['gate'],
                        note='reused WITHOUT re-running (no budget '
                             'reset; no double spending)'))
    status = 'aborted'
    try:
        # ---- 1) HNO stability (fixed) on a fresh CONVERGED object ------
        bnd_h = E.P4Backend(man['molecules']['HNO']['elements'])
        R_chk = np.asarray(rc['coords_actual_angstrom'],
                           float) * (1.0 / 0.52917721092)
        a_st = ledger.pre_eval('hno_stab', dict(tag='stab_HNO'))
        try:
            mol_h, mf_h = bnd_h.new_mf(R_chk, OUT, 'stab_HNO')
            mf_h.kernel()
            stab_h = E.stability_wrapper(mf_h, OUT, 'HNO')
            ex.save_json_atomic(os.path.join(
                OUT, 'stability_HNO_record.json'), dict(
                    stable_i=stab_h['stable_i'],
                    stable_e=stab_h['stable_e'],
                    raw_log=stab_h['raw_log'],
                    installed_order=stab_h['installed_order'],
                    geometry='the PASSING recheck geometry',
                    object='fresh mf CONVERGED at that geometry'))
            ledger.post_eval(a_st, dict(stable_i=stab_h['stable_i'],
                                        stable_e=stab_h['stable_e']))
        except Exception as e:                          # noqa: BLE001
            ledger.fail(a_st, e)
            raise
        hno_stab_i = stab_h['stable_i']
        print('[068r] HNO stability: stable_i=%s' % hno_stab_i,
              flush=True)

        # ---- 2) H2O2 full flow ----------------------------------------
        sp = 'H2O2'
        mdir2 = os.path.join(OUT, 'h2o2')
        os.makedirs(mdir2, exist_ok=True)
        elems = man['molecules'][sp]['elements']
        x0 = np.asarray(man['molecules'][sp]['coords_start_bohr'],
                        float).reshape(-1)
        bnd2 = E.P4Backend(elems)
        flow = E.run_opt_general(x0, 4, mdir2, ledger, bnd2, 'h2o2_opt',
                                 opt_cap=15, threshold=1e-5)
        ex.save_json_atomic(os.path.join(
            mdir2, 'accepted_iterates_H2O2.json'),
            dict(n=len(flow['accepted_iterates']),
                 iterates=flow['accepted_iterates']))
        print('[068r] H2O2 opt: %s (%d evals)'
              % (flow['outcome'], flow['n_new_evals']), flush=True)
        h2_pass = False
        if flow['meeting_points']:
            mp = flow['meeting_points'][-1]
            mrec2 = json.load(open('%s/eval_h2o2_opt_%s.json'
                                   % (mdir2, mp['tag'])))
            R_chk2 = np.asarray(mrec2['coords_actual_angstrom'],
                                float) * (1.0 / 0.52917721092)
            a_rc = ledger.pre_eval('h2o2_recheck', dict(tag='recheck'))
            try:
                rec_r2 = ex.eval_point('recheck', R_chk2, mdir2, ledger,
                                       bnd2, 'h2o2_recheck',
                                       gate_fn=ex.centre_gate_fn(mrec2))
                g2 = rec_r2['gate']
                h2_pass = bool(g2['gates_pass'])
                print('[068r] H2O2 recheck %s: dE=%.2e dgrad=%.2e'
                      % ('PASS' if h2_pass else 'FAIL', g2['dE'],
                         g2['dgrad_max']), flush=True)
                if h2_pass:
                    a_st2 = ledger.pre_eval('h2o2_stab',
                                            dict(tag='stab_H2O2'))
                    try:
                        mol2, mf2 = bnd2.new_mf(R_chk2, mdir2,
                                                'stab_H2O2')
                        mf2.kernel()
                        stab2 = E.stability_wrapper(mf2, mdir2, 'H2O2')
                        ex.save_json_atomic(os.path.join(
                            OUT, 'stability_H2O2_record.json'), dict(
                                stable_i=stab2['stable_i'],
                                stable_e=stab2['stable_e'],
                                raw_log=stab2['raw_log'],
                                installed_order=stab2['installed_order'],
                                geometry='the PASSING recheck geometry',
                                object='fresh mf CONVERGED at that '
                                       'geometry'))
                        ledger.post_eval(a_st2, dict(
                            stable_i=stab2['stable_i'],
                            stable_e=stab2['stable_e']))
                        print('[068r] H2O2 stability: stable_i=%s'
                              % stab2['stable_i'], flush=True)
                    except Exception as e:              # noqa: BLE001
                        ledger.fail(a_st2, e)
                        raise
            except RuntimeError as e:
                stop = os.path.join(mdir2, 'STOPPED_gate_fail_recheck'
                                            '.json')
                gate = json.load(open(stop))['gate'] \
                    if os.path.exists(stop) else dict(raw=str(e)[:300])
                mol_out_rc = dict(gate=gate, hard_stop=str(e)[:300])
                ex.save_json_atomic(os.path.join(
                    mdir2, 'recheck_failed_record.json'), mol_out_rc)
                raise RuntimeError('H2O2 recheck FAILED: %s'
                                   % json.dumps(gate)[:200])
        else:
            raise RuntimeError('H2O2: no point met the batch threshold '
                               '(budget exhausted within the mirror '
                               'opt) -> recorded truthfully')

        # ---- 3) product reference energy -------------------------------
        e_hno = float(rc['e_total'])
        e_h2o2 = float(rec_r2['e_total'])
        e_mono = (man['reference_energy']['e_nh3_job026']
                  + man['reference_energy']['e_o3_job026'])
        status = 'completed'
        results['product_reference'] = dict(
            p4_products_complete=True,
            e_hno_recheck=e_hno, e_h2o2_recheck=e_h2o2,
            e_products=e_hno + e_h2o2,
            dE_project=(e_hno + e_h2o2) - e_mono,
            e_monomer_sum=e_mono,
            formula='dE_project = [E(HNO)+E(H2O2)] - [E(NH3)+E(O3)]',
            scope='project-method electronic energy ONLY; not S13 '
                  'CCSD(T); no TS; no CP correction; not aqueous free '
                  'energy; not water-treatment efficiency')
        results['stability'] = dict(
            HNO_stable_i=hno_stab_i, H2O2_stable_i=stab2['stable_i'],
            scope='internal orbital stability ONLY; stable_i=True does '
                  'NOT mean full reaction-path reliability')
        results['final'] = dict(
            status=status,
            accounting=dict(caps=man['caps'],
                            attempts_by_cat={c: ledger.count(c) for c
                                             in man['caps']},
                            total=len(ledger.data['attempts']),
                            flow_anomalies=1),
            scope_limits='project method model ONLY; not an S13 '
                         'reproduction; electronic energy only (no '
                         'ZPE/thermal/CP); not aqueous free energy; '
                         'not water-treatment efficiency')
    except Exception as exc:                                # noqa: BLE001
        results['abort'] = dict(error=str(exc),
                                traceback=traceback.format_exc()[-1500:])
        status = 'aborted'
        print('[068r] ABORT: %s' % exc, flush=True)
    results['final_status'] = status
    results['budget'] = ledger.data
    ex.save_json_atomic(RESULTS, results)
    print('[068r] DONE: %s' % status, flush=True)


if __name__ == '__main__':
    main()
