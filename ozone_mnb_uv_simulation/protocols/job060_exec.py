#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-060: limited full-DOF BFGS relaxation from the ACTUAL
059 +0.002q lowering point (eval_disp_p002.json, the unique start).

Budget (hard, no borrowing): start_repro 1 + opt 25 (ALL new evaluations
INCLUDING line-search trials) + recheck 1 = 27.  The recheck is used ONLY
if an optimization point meets the BATCH trigger max|g|<=1e-6 Eh/Bohr
(runtime GM_GATE patch, fixed before the run; the general 1e-5 acceptance
gate is NOT relaxed).  Meeting the trigger and passing the recheck does
NOT register a minimum and does NOT claim the negative curvature is
eliminated.  ON ANY REAL EXCEPTION: save and STOP the whole batch - no
fix-and-restart, no ledger clearing, no separate budget, NO launch
probes.  Checkpoints are per-evaluation uniquely named (chk_<tag>.chk);
the overwritten 058 centre_scf.chk is NOT referenced anywhere.
"""
import os, sys, json, glob, time, traceback
import numpy as np

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
HERE = ROOT + '/protocols'
sys.path.insert(0, HERE)
import job047_exec as ex
import job051_exec as j51          # validated run_bfgs / LoggingBackend
import job057_exec as j57          # contact_geometry (zero-eval analysis)

R59 = ROOT + '/run_artifacts/02_nh3o3_reference/c1_negdir059'
OUT = ROOT + '/run_artifacts/02_nh3o3_reference/c1_relax060'
LEDGER = OUT + '/budget_relax060.json'
RESULTS = OUT + '/relax060_results.json'
MANIFEST = OUT + '/input_manifest.json'
BPA = 1.0 / ex.ANG_PER_BOHR
CAPS = dict(start_repro=1, opt=25, recheck=1)
TRIGGER = 1e-6                     # batch stop trigger (NOT the general gate)


def sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for blk in iter(lambda: fh.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def sha_arr(a):
    import hashlib
    return hashlib.sha256(np.asarray(a, float).tobytes()).hexdigest()[:16]


def main():
    man = json.load(open(MANIFEST))
    if sha256_file(man['source']['path']) != man['source']['sha256']:
        raise RuntimeError('HARD STOP: 059 disp_p002 source hash changed')
    if not man['all_checks_pass']:
        raise RuntimeError('HARD STOP: source prechecks failed')
    src = json.load(open(man['source']['path']))
    print('[060] source OK: %s (sha256 %.16s..., tag=%s, gmax=%.3e)'
          % (os.path.basename(man['source']['path']),
             man['source']['sha256'], src['tag'], src['grad_max']),
          flush=True)

    # FIXED before the run: batch trigger via runtime GM_GATE patch
    ex.GM_GATE = TRIGGER
    j51.GM_GATE = TRIGGER

    results = dict(
        job='JOB-2026-0906-060 limited full-DOF BFGS relaxation from the '
            '059 actual +0.002q lowering point',
        doing='start reproduction (gate vs the 059 disp_p002 record) + '
              'full-DOF 21-Cartesian standard BFGS (state reset, start '
              'reuse at identical coordinates) to the BATCH trigger '
              'max|g|<=1e-6; conditional independent recheck on success; '
              'no stability/Hessian/frequency/thermochemistry; meeting '
              'the trigger does NOT register a minimum and does NOT '
              'claim the negative curvature is eliminated',
        review_059='notes/job059_commander_review_2026-09-10.md',
        corrections_059='job059_correction_note.json in the 059 dir',
        source=dict(path=man['source']['path'],
                    sha256=man['source']['sha256'],
                    e_total=float(src['e_total']),
                    grad_max=float(src['grad_max']),
                    identity=dict(actual_059_disp_p002_point=True)),
        stop_target=dict(gate=TRIGGER,
                         note='BATCH trigger; general 1e-5 acceptance '
                              'gate unchanged and NOT relaxed; recheck '
                              'only on success'),
        optimizer_setup=j51._setup_note(200)
        + ' | GM_GATE runtime-patched to 1e-6 (batch trigger) in the '
          'job051 and job047 namespaces before the run')
    ledger = ex.Ledger(LEDGER, caps=CAPS)
    backend = j51.LoggingBackend(ex.RealBackend(OUT))
    status = 'aborted'
    try:
        R_start = (np.asarray(src['coords_actual_angstrom'], float) * BPA)
        rec_s = ex.eval_point('start', R_start, OUT, ledger, backend,
                              'start_repro',
                              gate_fn=ex.centre_gate_fn(src))
        g = rec_s['gate']
        print('[060] start gate %s: dE=%.2e dgrad=%.2e dC=%.2e (start '
              'gmax=%.3e; the 1e-5 general gate is NOT the trigger here)'
              % ('PASS' if g['gates_pass'] else 'FAIL', g['dE'],
                 g['dgrad_max'], g['dcoords_A'], rec_s['grad_max']),
              flush=True)
        results['start_reproduction'] = dict(
            gate=g, e_total=rec_s['e_total'],
            grad_max=rec_s['grad_max'])
        opt = j51.run_bfgs(rec_s, OUT, ledger, backend, opt_cap=25)
        results['optimization'] = opt
        ex.save_json_atomic(OUT + '/accepted_iterates.json',
                            dict(n=len(opt['accepted_iterates']),
                                 iterates=opt['accepted_iterates'],
                                 note='optimizer callback coordinates '
                                      '(accepted iterates); line-search '
                                      'trials are evaluations NOT in '
                                      'this list'))
        print('[060] optimization outcome: %s (new evals %d, reuse %s, '
              'blocked %d)'
              % (opt['outcome'], opt['n_new_evals'], opt['reuse_start'],
                 len(opt['blocked_requests'])), flush=True)
        if opt.get('warnings'):
            print('[060] optimizer warnings: %s' % opt['warnings'],
                  flush=True)
        if opt['meeting_points']:
            mp = opt['meeting_points'][0]
            meeting_rec = json.load(open(
                '%s/eval_%s_%s.json' % (OUT, mp['category'], mp['tag'])))
            try:
                rec_r = ex.eval_point('recheck', np.asarray(
                    meeting_rec['coords_actual_angstrom'], float) * BPA,
                    OUT, ledger, backend, 'recheck',
                    gate_fn=ex.recheck_gate_fn(meeting_rec))
                results['recheck'] = dict(gate=rec_r['gate'],
                                          e_total=rec_r['e_total'],
                                          grad_max=rec_r['grad_max'])
                if rec_r['gate']['gates_pass']:
                    status = 'trigger_1e-6_recheck_pass'
                    print('[060] recheck PASS at %.3e -> batch trigger '
                          'met and independently confirmed; NOT a '
                          'minimum registration; negative-mode '
                          'elimination NOT claimed'
                          % rec_r['grad_max'], flush=True)
                else:
                    status = 'recheck_failed'
                    print('[060] recheck FAILED -> trigger not confirmed',
                          flush=True)
            except RuntimeError as e:
                stop = OUT + '/STOPPED_gate_fail_recheck.json'
                gate = json.load(open(stop))['gate'] \
                    if os.path.exists(stop) else dict(raw=str(e)[:300])
                results['recheck'] = dict(gate=gate, hard_stop=str(e)[:300])
                status = 'recheck_failed'
                print('[060] recheck FAILED -> trigger not confirmed',
                      flush=True)
        else:
            status = ('start_meets_trigger'
                      if opt['outcome'] == 'start_already_meets_gate'
                      else 'not_converged_to_1e-6')
            print('[060] no saved optimization point met max|g|<=1e-6'
                  ' -> %s' % status, flush=True)
    except Exception as exc:                                # noqa: BLE001
        results['abort'] = dict(error=str(exc),
                                traceback=traceback.format_exc()[-1500:])
        status = 'aborted'
        print('[060] ABORT: %s' % exc, flush=True)

    cats = {c: ledger.count(c) for c in ('start_repro', 'opt', 'recheck')}
    fails = {c: sum(1 for a in ledger.data['attempts']
                    if a['category'] == c and a.get('status') == 'error')
             for c in cats}
    saved = []
    for pat in ('eval_start_repro_start.json', 'eval_opt_opt_*.json',
                'eval_recheck_recheck.json'):
        for f in sorted(glob.glob(OUT + '/' + pat)):
            saved.append(json.load(open(f)))
    low = min(saved, key=lambda r: float(r['e_total'])) if saved else None
    mng = min(saved, key=lambda r: float(r['grad_max'])) if saved else None
    last_acc = None
    try:
        accf = json.load(open(OUT + '/accepted_iterates.json'))
        if accf['iterates']:
            last_acc = dict(iterate_index=len(accf['iterates']),
                            x_bohr=accf['iterates'][-1],
                            note='last ACCEPTED optimizer iterate '
                                 '(callback); may or may not coincide '
                                 'with a saved evaluation')
    except Exception:                                       # noqa: BLE001
        last_acc = None
    # accepted vs line-search trial classification: nearest accepted
    # iterate within 1e-12 Bohr (bitwise matching is too strict across
    # scipy's line-search/callback code paths)
    acc_list = []
    try:
        accf = json.load(open(OUT + '/accepted_iterates.json'))
        acc_list = [np.asarray(a, float) for a in accf['iterates']]
    except Exception:                                       # noqa: BLE001
        acc_list = []
    for rec in saved:
        x = np.asarray(rec['coords_actual_angstrom'], float) \
            .reshape(-1) * BPA
        dmin = min((float(np.abs(x - a).max()) for a in acc_list),
                   default=float('inf'))
        rec['iterate_class'] = ('accepted_iterate' if dmin <= 1e-12
                                else 'linesearch_trial')
        rec['accepted_match_maxdev_Bohr'] = (None if dmin == float('inf')
                                             else dmin)
    n_acc = sum(1 for r in saved if r['iterate_class'] == 'accepted_iterate')
    n_trial = sum(1 for r in saved
                  if r['iterate_class'] == 'linesearch_trial')
    geom_cmp = None
    if saved:
        ref = j57.contact_geometry(np.asarray(
            src['coords_actual_angstrom'], float).reshape(-1) * BPA)
        cand = {}
        for label, rec in (('lowest_energy', low),
                           ('min_gradient', mng)):
            if rec is None:
                continue
            x = (np.asarray(rec['coords_actual_angstrom'], float) * BPA)
            cg = j57.contact_geometry(x)
            cand[label] = dict(
                tag=rec.get('tag'), category=rec.get('category'),
                iterate_class=rec['iterate_class'],
                geometry=cg,
                delta_vs_start={k: float(cg[k] - ref[k])
                                for k in ('NH3_O3_centroid_dist_Bohr',
                                          'min_N_O_Bohr', 'min_H_O_Bohr')},
                grad_max=float(rec['grad_max']),
                e_total=float(rec['e_total']))
        geom_cmp = dict(start=ref, candidates=cand,
                        note='geometry OBSERVATION only; no "same basin", '
                             'no "lateral relaxation", no "negative mode '
                             'eliminated" presupposed; energy lowering is '
                             'NOT a binding free energy')
    results['final'] = dict(
        status=status,
        budget=dict(caps=CAPS, attempts=cats, failures=fails,
                    total=sum(cats.values()),
                    cumulative_note='project cumulative through 059: at '
                                    'least 52 attempts (49 completed '
                                    'SCF+grad, 2 stability calls, probe '
                                    'counted, uncertain costs listed); '
                                    'this batch adds at most 27'),
        evaluation_split=dict(accepted_iterates=n_acc,
                              linesearch_trials=n_trial,
                              note='accepted optimizer iterates and '
                                   'line-search trial evaluations are '
                                   'reported separately'),
        lowest_energy_point=None if low is None else dict(
            tag=low.get('tag'), category=low.get('category'),
            e_total=float(low['e_total']), grad_max=float(low['grad_max']),
            iterate_class=low.get('iterate_class')),
        min_gradient_point=None if mng is None else dict(
            tag=mng.get('tag'), category=mng.get('category'),
            e_total=float(mng['e_total']),
            grad_max=float(mng['grad_max']),
            iterate_class=mng.get('iterate_class')),
        last_accepted_iterate=last_acc,
        geometry_comparison=geom_cmp,
        bookkeeping_note='optimizer requests / completed evaluations / '
                         'accepted iterates / budget-blocked requests are '
                         'reported separately',
        not_claimed='minimum registration / negative-mode elimination / '
                    'same basin / frequency convergence / transition '
                    'state')
    results['budget'] = ledger.data
    ex.save_json_atomic(RESULTS, results)
    print('DONE: start_repro=%d/1 opt=%d/25 recheck=%d/1 status=%s'
          % (cats['start_repro'], cats['opt'], cats['recheck'], status),
          flush=True)


if __name__ == '__main__':
    main()
