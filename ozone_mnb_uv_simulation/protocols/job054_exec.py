#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-054: full-DOF relaxation from the 053 energy-lowered
+0.02 Bohr point with standard BFGS.

Budget (no borrowing): start_repro 1, opt 25, recheck 1, total <= 27.

Charter adjustment: optimizer controlled stop at max|g|<=1e-6 (this
batch's stricter target); the project acceptance gate 1e-5 is UNCHANGED
but reaching 1e-5 does NOT stop early. The start itself may exceed 1e-5
and is NOT rejected for that.

Reuses the validated 047 chain (Ledger/eval_point/gates/RealBackend)
identically; only the source, run dir, caps and the stop threshold differ.
"""
import os, sys, json, hashlib, time, traceback, warnings
import numpy as np

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
HERE = ROOT + '/protocols'
sys.path.insert(0, HERE)
import job047_exec as ex
from scipy.optimize import minimize as _scipy_minimize

OUT = ROOT + '/run_artifacts/02_nh3o3_reference/c1_negmode_relax054'
LEDGER = OUT + '/budget_relax054.json'
RESULTS = OUT + '/relax054_results.json'
MANIFEST = OUT + '/input_manifest.json'

BPA = 1.0 / ex.ANG_PER_BOHR
CAPS = dict(start_repro=1, opt=25, recheck=1)
OPT_CAP = 25
MAXITER = 200
GM_STOP = 1e-6             # this batch's controlled-stop target
GM_GATE = ex.GM_GATE      # 1e-5, the acceptance gate (unchanged)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for blk in iter(lambda: fh.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def sha_arr(a):
    return hashlib.sha256(np.asarray(a, float).tobytes()).hexdigest()[:16]


class LoggingBackend(ex.RealBackend):
    def full_eval(self, R_bohr, out_dir, tag):
        R = np.asarray(R_bohr, float)
        print('[054] eval %s: SCF+gradient start' % tag, flush=True)
        t0 = time.time()
        rec = super().full_eval(R, out_dir, tag)
        rec['x_bohr'] = R.tolist()
        rec['x_bohr_sha'] = sha_arr(R)
        rec['seconds'] = round(time.time() - t0, 1)
        print('[054] eval %s: E=%.9f grad_max=%.3e conv=%s (%.0fs)'
              % (tag, rec['e_total'], rec['grad_max'], rec['converged'],
                 rec['seconds']), flush=True)
        return rec


class StopReached(Exception):
    def __init__(self, tag, gmax):
        super().__init__('stop target met at %s (max|g|=%.3e)'
                         % (tag, gmax))
        self.tag = tag
        self.gmax = gmax


def run_bfgs(start_rec, out_dir, ledger, backend,
             opt_cap=OPT_CAP, maxiter=MAXITER):
    x0 = (np.asarray(start_rec['coords_actual_angstrom'], float)
          / ex.ANG_PER_BOHR).reshape(-1)
    start_sha = sha_arr(x0)
    state = dict(n_new=0, records=[], reuse_used=False, accepted=[],
                 requests=[], blocked_requests=[])

    # the start may exceed 1e-5 and is NOT rejected for that; but if it
    # already meets the 1e-6 stop target, no optimization is started
    if float(start_rec['grad_max']) <= GM_STOP:
        return dict(outcome='start_already_meets_stop_target',
                    start_gmax=float(start_rec['grad_max']),
                    reuse_start=True, n_new_evals=0, records=[],
                    accepted_iterates=[], requests=[], blocked_requests=[],
                    stop_target_points=[dict(tag=start_rec['tag'],
                                             category=start_rec['category'],
                                             grad_max=float(
                                                 start_rec['grad_max']),
                                             e_total=float(
                                                 start_rec['e_total']))],
                    acceptance_gate_points=[dict(tag=start_rec['tag'],
                                                 category=start_rec[
                                                     'category'],
                                                 grad_max=float(
                                                     start_rec['grad_max']),
                                                 e_total=float(
                                                     start_rec['e_total']))],
                    warnings=[],
                    optimizer_setup=_setup_note(maxiter))

    # NOTE: the start may exceed 1e-5; we DO NOT early-return for that.
    # The stop target is 1e-6 (GM_STOP). Reaching 1e-5 does NOT stop.
    def fun(x):
        x = np.asarray(x, float).reshape(-1)
        state['requests'].append(x.tolist())
        if state['n_new'] == 0 and sha_arr(x) == start_sha:
            state['reuse_used'] = True
            return (float(start_rec['e_total']),
                    np.asarray(start_rec['grad'], float).reshape(-1))
        if state['n_new'] >= opt_cap:
            state['blocked_requests'].append(x.tolist())
            raise ex.BudgetExhausted('opt attempt cap %d reached' % opt_cap)
        tag = 'opt_%02d' % (state['n_new'] + 1)
        rec = ex.eval_point(tag, x.reshape(7, 3), out_dir, ledger,
                            backend, 'opt')
        state['n_new'] += 1
        state['records'].append(rec)
        if float(rec['grad_max']) <= GM_STOP:
            raise StopReached(tag, float(rec['grad_max']))
        return (float(rec['e_total']),
                np.asarray(rec['grad'], float).reshape(-1))

    def cb(xk):
        xk = getattr(xk, 'x', xk)
        state['accepted'].append(np.asarray(xk, float).tolist())

    opts = dict(gtol=1e-6, norm=np.inf, xrtol=0.0, c1=1e-4, c2=0.9,
                maxiter=maxiter, hess_inv0=np.eye(21))
    wlist = []
    try:
        with warnings.catch_warnings(record=True) as _w:
            warnings.simplefilter('always')
            wlist = _w
            res = _scipy_minimize(fun, x0, method='BFGS', jac=True,
                                 callback=cb, options=opts)
        outcome = 'optimizer_returned'
        res_info = dict(success=bool(res.success),
                        status=int(getattr(res, 'status', -1)),
                        message=str(res.message),
                        nit=int(getattr(res, 'nit', -1)),
                        nfev=int(getattr(res, 'nfev', -1)),
                        final_jac_gmax=float(np.abs(np.asarray(
                            res.jac, float)).max()))
        try:
            hinv = np.asarray(res.hess_inv, float)
            if hinv.shape == (21, 21):
                ex.save_json_atomic(
                    out_dir + '/bfgs_hess_inv_final.json',
                    dict(note='BFGS inverse-Hessian approximation at '
                              'normal return; ALGORITHMIC quantity for '
                              'this optimizer state only; NOT a computed '
                              'molecular Hessian and NOT usable for '
                              'frequency analysis',
                         shape=[21, 21], matrix=hinv.tolist()))
                res_info['hess_inv_saved'] = 'bfgs_hess_inv_final.json'
        except Exception as e:                          # noqa: BLE001
            res_info['hess_inv_saved'] = 'failed: %s' % e
    except StopReached as e:
        outcome = 'stop_target_reached'
        res_info = dict(tag=e.tag, gmax=e.gmax,
                        message='controlled stop: saved point met '
                                'max|g|<=1e-6 (batch stop target)')
    except ex.BudgetExhausted as e:
        outcome = 'budget_exhausted'
        res_info = dict(message=str(e))
    wmsgs = [str(w.message) for w in wlist]
    # meeting points: those <= GM_STOP AND those <= GM_GATE (1e-5)
    stop_pts = [dict(tag=r['tag'], category=r['category'],
                     grad_max=float(r['grad_max']),
                     e_total=float(r['e_total']))
                for r in state['records']
                if float(r['grad_max']) <= GM_STOP]
    gate_pts = [dict(tag=r['tag'], category=r['category'],
                     grad_max=float(r['grad_max']),
                     e_total=float(r['e_total']))
                for r in state['records']
                if float(r['grad_max']) <= GM_GATE]
    return dict(outcome=outcome, optimizer=res_info,
                reuse_start=state['reuse_used'],
                n_new_evals=state['n_new'],
                accepted_iterates=state['accepted'],
                requests=state['requests'],
                blocked_requests=state['blocked_requests'],
                stop_target_points=stop_pts,
                acceptance_gate_points=gate_pts,
                warnings=wmsgs,
                optimizer_setup=_setup_note(maxiter))


def _setup_note(maxiter):
    return ('scipy method=BFGS, jac=True, gtol=1e-6, norm=inf, xrtol=0, '
            'c1=1e-4, c2=0.9, hess_inv0=21x21 identity (Bohr^2/Eh, '
            'algorithmic model, not a computed Hessian), maxiter=%d; '
            '21 Cartesian variables in Bohr; D2 attached once, grid level '
            '8, original SCF tolerances, grid_response=True; geometry '
            'continuation with optimizer state RESET; maxiter is NOT the '
            'evaluation budget (persisted opt cap guards every call); '
            'STOP TARGET max|g|<=1e-6 (acceptance gate 1e-5 unchanged, '
            'reaching 1e-5 does NOT stop early)' % maxiter)


def main():
    man = json.load(open(MANIFEST))
    src_path = man['source']['source_path_wsl']
    sha = sha256_file(src_path)
    if sha != man['source']['source_sha256']:
        raise RuntimeError('HARD STOP: 053 p02 source file hash changed')
    if not man['all_checks_pass']:
        raise RuntimeError('HARD STOP: source prechecks failed')
    src = json.load(open(src_path))
    print('[054] source OK: %s (sha256 %.16s...)'
          % (os.path.basename(src_path), sha), flush=True)

    results = dict(
        job='JOB-2026-0906-054 full-DOF relaxation from the 053 '
            'energy-lowered +0.02 Bohr point',
        doing='standard BFGS full-DOF relaxation from the 053 actual '
              '+0.02 point (energy lowered vs the 052 candidate) to '
              'explore a further-relaxed contact geometry; optimizer '
              'stop target max|g|<=1e-6 (acceptance gate 1e-5 unchanged); '
              'independent recheck required for any candidate registration',
        source=dict(path=src_path, sha256=sha,
                    e_total=float(src['e_total']),
                    grad_max=float(src['grad_max']),
                    identity=dict(actual_053_p02_point=True),
                    continuation_mode='geometry continuation, optimizer '
                                      'state reset'),
        stop_target=dict(gm_stop=1e-6, gm_acceptance_unchanged=1e-5,
                         note='reaching 1e-5 does NOT stop early'))
    ledger = ex.Ledger(LEDGER, caps=CAPS)
    backend = LoggingBackend(OUT)
    status = 'aborted'
    try:
        R_start = (np.asarray(src['coords_actual_angstrom'], float) * BPA)
        # start gate: centre_gate_fn does NOT include a gmax check, so a
        # start above 1e-5 is accepted (charter requirement)
        rec_s = ex.eval_point('start', R_start, OUT, ledger, backend,
                              'start_repro',
                              gate_fn=ex.centre_gate_fn(src))
        g = rec_s['gate']
        print('[054] start gate %s: dE=%.2e dgrad=%.2e dC=%.2e (gmax=%.3e '
              'is NOT a gate criterion)'
              % ('PASS' if g['gates_pass'] else 'FAIL', g['dE'],
                 g['dgrad_max'], g['dcoords_A'], rec_s['grad_max']),
              flush=True)
        results['start_reproduction'] = dict(
            gate=g, e_total=rec_s['e_total'],
            grad_max=rec_s['grad_max'],
            start_grad_above_acceptance=bool(rec_s['grad_max'] > GM_GATE),
            start_grad_above_stop=bool(rec_s['grad_max'] > GM_STOP))
        opt = run_bfgs(rec_s, OUT, ledger, backend)
        results['optimization'] = opt
        print('[054] optimization outcome: %s (new evals %d, '
              'reuse_start %s, blocked %d, stop_pts %d, gate_pts %d)'
              % (opt['outcome'], opt['n_new_evals'], opt['reuse_start'],
                 len(opt['blocked_requests']),
                 len(opt['stop_target_points']),
                 len(opt['acceptance_gate_points'])), flush=True)
        if opt.get('warnings'):
            print('[054] optimizer warnings: %s' % opt['warnings'],
                  flush=True)
        # recheck only if the STOP TARGET was met (<=1e-6)
        if opt['stop_target_points']:
            mp = opt['stop_target_points'][0]
            meeting_rec = json.load(open(
                '%s/eval_%s_%s.json' % (OUT, mp['category'], mp['tag'])))
            try:
                rec_r = ex.eval_point('recheck', np.asarray(
                    meeting_rec['coords_actual_angstrom'], float) * BPA,
                    OUT, ledger, backend, 'recheck',
                    gate_fn=ex.recheck_gate_fn(meeting_rec))
                results['recheck'] = dict(gate=rec_r['gate'],
                                          e_total=rec_r['e_total'],
                                          grad_max=rec_r['grad_max'],
                                          recheck_below_1e6=bool(
                                              rec_r['grad_max']
                                              <= GM_STOP))
                if rec_r['gate']['gates_pass']:
                    status = 'further_relaxed_candidate'
                    print('[054] recheck PASS -> registered: further '
                          'relaxed candidate, curvature pending '
                          'verification', flush=True)
                else:
                    status = 'recheck_failed'
                    print('[054] recheck FAILED (acceptance gate) -> not '
                          'registered', flush=True)
            except RuntimeError as e:
                stop = OUT + '/STOPPED_gate_fail_recheck.json'
                gate = json.load(open(stop))['gate'] \
                    if os.path.exists(stop) else dict(raw=str(e)[:300])
                results['recheck'] = dict(gate=gate,
                                          hard_stop=str(e)[:300])
                status = 'recheck_failed'
                print('[054] recheck FAILED -> not registered', flush=True)
        else:
            status = 'stop_target_not_reached'
            print('[054] no saved point met max|g|<=1e-6 -> not '
                  'converged to the batch stop target', flush=True)
    except Exception as exc:                                # noqa: BLE001
        results['abort'] = dict(error=str(exc),
                                traceback=traceback.format_exc()[-1500:])
        status = 'aborted'
        print('[054] ABORT: %s' % exc, flush=True)

    cats = {c: ledger.count(c) for c in ('start_repro', 'opt', 'recheck')}
    fails = {c: sum(1 for a in ledger.data['attempts']
                    if a['category'] == c and a.get('status') == 'error')
             for c in cats}
    saved = []
    import glob
    for pat in ('eval_start_repro_start.json', 'eval_opt_opt_*.json',
                'eval_recheck_recheck.json'):
        for f in sorted(glob.glob(OUT + '/' + pat)):
            saved.append(json.load(open(f)))
    low = min(saved, key=lambda r: float(r['e_total'])) if saved else None
    mng = min(saved, key=lambda r: float(r['grad_max'])) if saved else None
    la = max(saved, key=lambda r: float(r['e_total'])) if saved else None
    results['final'] = dict(
        status=status,
        budget=dict(caps=CAPS, attempts=cats, failures=fails,
                    total=sum(cats.values())),
        lowest_energy_point=None if low is None else dict(
            tag=low.get('tag'), category=low.get('category'),
            e_total=float(low['e_total']), grad_max=float(low['grad_max'])),
        min_gradient_point=None if mng is None else dict(
            tag=mng.get('tag'), category=mng.get('category'),
            e_total=float(mng['e_total']),
            grad_max=float(mng['grad_max'])),
        last_evaluated_point=None if la is None else dict(
            tag=la.get('tag'), category=la.get('category'),
            e_total=float(la['e_total']), grad_max=float(la['grad_max'])),
        bookkeeping_note='optimizer requests / completed evaluations / '
                         'accepted iterates / budget-blocked requests are '
                         'reported separately; a blocked request is not a '
                         'line-search rejection',
        note='single limited relaxation; no step-count forecast; no '
             'auto-continuation; curvature verification NOT performed')
    results['budget'] = ledger.data
    ex.save_json_atomic(RESULTS, results)
    print('DONE: start_repro=%d/1 opt=%d/25 recheck=%d/1 status=%s'
          % (cats['start_repro'], cats['opt'], cats['recheck'], status),
          flush=True)


if __name__ == '__main__':
    main()
