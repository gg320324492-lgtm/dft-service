#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-061: limited continuation optimization from the 060 LAST
accepted point opt_24 (geometry continuation, BFGS optimizer state RESET;
no restored inverse Hessian or line-search history).

Budget (hard, no borrowing): start_repro 1 + opt 20 (ALL new evaluations
INCLUDING line-search trials) + recheck 1 = 22.  Recheck used ONLY if an
optimization point meets the BATCH trigger max|g|<=1e-6 Eh/Bohr (runtime
GM_GATE patch fixed before the run; the general 1e-5 acceptance gate is
NOT relaxed, no early acceptance near the gate).  Meeting the trigger and
passing the recheck does NOT register a minimum and does NOT claim the
negative curvature is eliminated.  ON ANY REAL EXCEPTION: save and STOP
the whole batch - no fix-and-restart, no ledger swap, no separate budget,
NO launch probes.  Record classification: start_repro / accepted-iterate
evaluations / non-accepted trials / budget-refused requests (the start is
NOT a line-search trial).
"""
import os, sys, json, glob, time, traceback
import numpy as np

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
HERE = ROOT + '/protocols'
sys.path.insert(0, HERE)
import job047_exec as ex
import job051_exec as j51          # validated run_bfgs / LoggingBackend
import job057_exec as j57          # contact_geometry (zero-eval analysis)

R60 = ROOT + '/run_artifacts/02_nh3o3_reference/c1_relax060'
OUT = ROOT + '/run_artifacts/02_nh3o3_reference/c1_relaxcont061'
LEDGER = OUT + '/budget_relaxcont061.json'
RESULTS = OUT + '/relaxcont061_results.json'
MANIFEST = OUT + '/input_manifest.json'
BPA = 1.0 / ex.ANG_PER_BOHR
CAPS = dict(start_repro=1, opt=20, recheck=1)
TRIGGER = 1e-6                     # batch stop trigger (NOT the general gate)


def sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for blk in iter(lambda: fh.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def main():
    man = json.load(open(MANIFEST))
    if sha256_file(man['source']['path']) != man['source']['sha256']:
        raise RuntimeError('HARD STOP: 060 opt_24 source hash changed')
    if not man['all_checks_pass']:
        raise RuntimeError('HARD STOP: source prechecks failed')
    src = json.load(open(man['source']['path']))
    print('[061] source OK: %s (sha256 %.16s..., tag=%s, gmax=%.6e)'
          % (os.path.basename(man['source']['path']),
             man['source']['sha256'], src['tag'], src['grad_max']),
          flush=True)

    # FIXED before the run: batch trigger via runtime GM_GATE patch
    ex.GM_GATE = TRIGGER
    j51.GM_GATE = TRIGGER

    results = dict(
        job='JOB-2026-0906-061 limited continuation optimization from '
            'the 060 last accepted point opt_24',
        doing='start reproduction (gate vs the opt_24 record) + '
              'full-DOF 21-Cartesian standard BFGS (geometry '
              'continuation, optimizer state RESET, start reuse at '
              'identical coordinates) to the BATCH trigger max|g|<=1e-6; '
              'conditional independent recheck on success; no stability/'
              'Hessian/frequency/thermochemistry; meeting the trigger '
              'does NOT register a minimum and does NOT claim the '
              'negative curvature is eliminated',
        review_060='notes/job060_commander_review_2026-09-10.md',
        corrections_060='job060_correction_note.json in the 060 dir',
        source=dict(path=man['source']['path'],
                    sha256=man['source']['sha256'],
                    e_total=float(src['e_total']),
                    grad_max=float(src['grad_max']),
                    identity=dict(actual_060_opt_24_point=True)),
        continuation_mode=man['continuation_mode'],
        stop_target=dict(gate=TRIGGER,
                         note='BATCH trigger; general 1e-5 acceptance '
                              'gate unchanged; recheck only on success'),
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
        print('[061] start gate %s: dE=%.2e dgrad=%.2e dC=%.2e (start '
              'gmax=%.3e)'
              % ('PASS' if g['gates_pass'] else 'FAIL', g['dE'],
                 g['dgrad_max'], g['dcoords_A'], rec_s['grad_max']),
              flush=True)
        results['start_reproduction'] = dict(
            gate=g, e_total=rec_s['e_total'],
            grad_max=rec_s['grad_max'])
        opt = j51.run_bfgs(rec_s, OUT, ledger, backend, opt_cap=20)
        results['optimization'] = opt
        ex.save_json_atomic(OUT + '/accepted_iterates.json',
                            dict(n=len(opt['accepted_iterates']),
                                 iterates=opt['accepted_iterates'],
                                 note='optimizer callback coordinates '
                                      '(accepted iterates); NO inverse-'
                                      'Hessian or line-search history is '
                                      'claimed restored'))
        print('[061] optimization outcome: %s (new evals %d, reuse %s, '
              'blocked %d)'
              % (opt['outcome'], opt['n_new_evals'], opt['reuse_start'],
                 len(opt['blocked_requests'])), flush=True)
        # resource warnings saved and registered (NOT called harmless)
        if opt.get('warnings'):
            ex.save_json_atomic(
                OUT + '/resource_warnings.json',
                dict(count=len(opt['warnings']),
                     warnings=opt['warnings'],
                     audit='project-owned handles use with-blocks / '
                           'explicit close; warnings originate in '
                           'PySCF-internal /tmp temporary buffers (DIIS); '
                           'no platform modification; no evidence of '
                           'impact on scientific values; under '
                           'observation',
                     note='NOT called harmless; registered per the 060 '
                          'review'))
            print('[061] optimizer warnings: %d (saved to '
                  'resource_warnings.json)' % len(opt['warnings']),
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
                    print('[061] recheck PASS at %.3e -> batch trigger '
                          'met and independently confirmed; NOT a '
                          'minimum registration; negative-mode '
                          'elimination NOT claimed'
                          % rec_r['grad_max'], flush=True)
                else:
                    status = 'recheck_failed'
                    print('[061] recheck FAILED -> trigger not confirmed',
                          flush=True)
            except RuntimeError as e:
                stop = OUT + '/STOPPED_gate_fail_recheck.json'
                gate = json.load(open(stop))['gate'] \
                    if os.path.exists(stop) else dict(raw=str(e)[:300])
                results['recheck'] = dict(gate=gate, hard_stop=str(e)[:300])
                status = 'recheck_failed'
                print('[061] recheck FAILED -> trigger not confirmed',
                      flush=True)
        else:
            status = ('start_meets_trigger'
                      if opt['outcome'] == 'start_already_meets_gate'
                      else 'not_converged_to_1e-6')
            print('[061] no saved optimization point met max|g|<=1e-6'
                  ' -> %s' % status, flush=True)
    except Exception as exc:                                # noqa: BLE001
        results['abort'] = dict(error=str(exc),
                                traceback=traceback.format_exc()[-1500:])
        status = 'aborted'
        print('[061] ABORT: %s' % exc, flush=True)

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
    last_ev = saved[-1] if saved else None
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

    # four-way classification: start_repro / accepted / non-accepted
    # trial / (budget-refused requests counted separately from ledger)
    acc_list = []
    try:
        accf = json.load(open(OUT + '/accepted_iterates.json'))
        acc_list = [np.asarray(a, float) for a in accf['iterates']]
    except Exception:                                       # noqa: BLE001
        acc_list = []
    cls = dict(start_repro=0, accepted_iterate=0, non_accepted_trial=0)
    for rec in saved:
        if rec.get('category') == 'start_repro':
            rec['iterate_class'] = 'start_repro'
            cls['start_repro'] += 1
            continue
        x = np.asarray(rec['coords_actual_angstrom'], float) \
            .reshape(-1) * BPA
        dmin = min((float(np.abs(x - a).max()) for a in acc_list),
                   default=float('inf'))
        rec['iterate_class'] = ('accepted_iterate' if dmin <= 1e-12
                                else 'non_accepted_trial')
        cls[rec['iterate_class']] += 1
    n_blocked = len(opt.get('blocked_requests', [])) \
        if 'opt' in results else 0
    geom_cmp = None
    if saved:
        ref = j57.contact_geometry(np.asarray(
            src['coords_actual_angstrom'], float).reshape(-1) * BPA)
        cand = {}
        for label, rec in (('lowest_energy', low),
                           ('min_gradient', mng),
                           ('last_evaluated', last_ev)):
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
                        note='distance numbers ONLY; no dissociation / '
                             'separation-trend / single-negative-mode '
                             'direction claim (trajectory not projected); '
                             'no "same basin" or "negative mode '
                             'eliminated"')
    results['final'] = dict(
        status=status,
        budget=dict(caps=CAPS, attempts=cats, failures=fails,
                    total=sum(cats.values()),
                    cost_statistics_range='JOB-058..061 segment (incl. '
                                          'the 058 recovery batch and its '
                                          'registered probe); NOT '
                                          'whole-project history; '
                                          'cumulative through 060: >=78 '
                                          'attempts / 75 completed '
                                          'SCF+gradient; this batch adds '
                                          'at most 22'),
        record_classification=dict(four_way=cls,
                                   budget_refused_requests=n_blocked,
                                   note='start reproduction is its own '
                                        'category (NOT a line-search '
                                        'trial)'),
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
        last_evaluated_point=None if last_ev is None else dict(
            tag=last_ev.get('tag'), category=last_ev.get('category'),
            e_total=float(last_ev['e_total']),
            grad_max=float(last_ev['grad_max']),
            iterate_class=last_ev.get('iterate_class'),
            accepted=bool(last_ev.get('iterate_class')
                          == 'accepted_iterate')),
        geometry_comparison=geom_cmp,
        bookkeeping_note='optimizer requests / completed evaluations / '
                         'accepted iterates / budget-blocked requests are '
                         'reported separately',
        not_claimed='minimum registration / negative-mode elimination / '
                    'same basin / dissociation / frequency convergence / '
                    'transition state')
    results['budget'] = ledger.data
    ex.save_json_atomic(RESULTS, results)
    print('DONE: start_repro=%d/1 opt=%d/20 recheck=%d/1 status=%s'
          % (cats['start_repro'], cats['opt'], cats['recheck'], status),
          flush=True)


if __name__ == '__main__':
    main()
