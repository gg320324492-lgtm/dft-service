#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-050: standard-BFGS limited optimization from the 048 last
accepted point (opt_19).

Budget (per-category, no borrowing): start_repro 1, opt 20, recheck 1
(total cap 22).  No stability / Hessian / frequencies / CP / other species.

Reuses the validated JOB-047 evaluation & persistence components unchanged
(Ledger, eval_point, gates, RealBackend); ONLY the optimizer call and the
source are adapted:
  * scipy method='BFGS', jac=True, gtol=1e-6, norm=inf, xrtol=0,
    c1=1e-4, c2=0.9, hess_inv0 = 21x21 identity (Bohr^2/Eh; algorithmic
    initial model, NOT a computed molecular Hessian), maxiter=200
    (maxiter is NOT the evaluation budget; the persisted pre-call guard
    opt cap 20 is the real limit);
  * 21 Cartesian variables in Bohr; unconstrained, no projection, no
    per-step registration; fresh optimizer state (no claim of resuming
    048 L-BFGS history);
  * line-search trial points are all ledger-counted; accepted iterates
    recorded separately via callback;
  * controlled stop as soon as any fully saved point reaches unprojected
    max|g| <= 1e-5 -> at most one independent new-object recheck.
"""
import os, sys, json, glob, hashlib, time, traceback, warnings
import numpy as np

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
HERE = ROOT + '/protocols'
sys.path.insert(0, HERE)
import job047_exec as ex          # validated chain (Ledger / eval_point /
                                  # gates / RealBackend / exceptions)
from scipy.optimize import minimize as _scipy_minimize

OUT = ROOT + '/run_artifacts/02_nh3o3_reference/c1_bfgs_opt050'
LEDGER = OUT + '/budget_bfgs050.json'
RESULTS = OUT + '/bfgs_opt050_results.json'
MANIFEST = OUT + '/input_manifest.json'

BPA = 1.0 / ex.ANG_PER_BOHR
CAPS = dict(start_repro=1, opt=20, recheck=1)
OPT_CAP = 20
MAXITER = 200
GM_GATE = ex.GM_GATE              # 1e-5, unprojected max|g|


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for blk in iter(lambda: fh.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def sha_arr(a):
    return hashlib.sha256(np.asarray(a, float).tobytes()).hexdigest()[:16]


class LoggingBackend:
    """Per-eval progress + exact trial coordinates persisted in records."""

    def __init__(self, inner):
        self.inner = inner

    def full_eval(self, R_bohr, out_dir, tag):
        R = np.asarray(R_bohr, float)
        print('[050] eval %s: SCF+gradient start' % tag, flush=True)
        t0 = time.time()
        rec = self.inner.full_eval(R, out_dir, tag)
        rec['x_bohr'] = R.tolist()
        rec['x_bohr_sha'] = sha_arr(R)
        rec['seconds'] = round(time.time() - t0, 1)
        print('[050] eval %s: E=%.9f grad_max=%.3e conv=%s (%.0fs)'
              % (tag, rec['e_total'], rec['grad_max'], rec['converged'],
                 rec['seconds']), flush=True)
        return rec


def run_bfgs(start_rec, out_dir, ledger, backend,
             opt_cap=OPT_CAP, maxiter=MAXITER):
    x0 = (np.asarray(start_rec['coords_actual_angstrom'], float)
          / ex.ANG_PER_BOHR).reshape(-1)
    start_sha = sha_arr(x0)
    state = dict(n_new=0, records=[], reuse_used=False, accepted=[],
                 trial_sequence=[])
    if float(start_rec['grad_max']) <= GM_GATE:
        return dict(outcome='start_already_meets_gate',
                    start_gmax=float(start_rec['grad_max']),
                    reuse_start=True, n_new_evals=0, records=[],
                    accepted_iterates=[], trial_sequence=[],
                    meeting_points=[dict(tag=start_rec['tag'],
                                         category=start_rec['category'],
                                         grad_max=float(start_rec['grad_max']))],
                    warnings=[], optimizer=dict(message='start meets gate'),
                    optimizer_setup=_setup_note(maxiter))

    def fun(x):
        x = np.asarray(x, float).reshape(-1)
        state['trial_sequence'].append(x.tolist())
        if state['n_new'] == 0 and sha_arr(x) == start_sha:
            state['reuse_used'] = True
            return (float(start_rec['e_total']),
                    np.asarray(start_rec['grad'], float).reshape(-1))
        if state['n_new'] >= opt_cap:
            raise ex.BudgetExhausted('opt attempt cap %d reached' % opt_cap)
        tag = 'opt_%02d' % (state['n_new'] + 1)
        rec = ex.eval_point(tag, x.reshape(7, 3), out_dir, ledger,
                            backend, 'opt')
        state['n_new'] += 1
        state['records'].append(rec)
        if float(rec['grad_max']) <= GM_GATE:
            raise ex.ThresholdReached(tag, float(rec['grad_max']))
        return float(rec['e_total']), np.asarray(rec['grad'],
                                                 float).reshape(-1)

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
        except Exception as e:                                  # noqa: BLE001
            res_info['hess_inv_saved'] = 'failed: %s' % e
    except ex.ThresholdReached as e:
        outcome = 'threshold_reached'
        res_info = dict(tag=e.tag, gmax=e.gmax,
                        message='controlled stop: saved point met '
                                'max|g|<=1e-5')
    except ex.BudgetExhausted as e:
        outcome = 'budget_exhausted'
        res_info = dict(message=str(e))
    wmsgs = [str(w.message) for w in wlist]
    meeting = [dict(tag=r['tag'], category=r['category'],
                    grad_max=float(r['grad_max']),
                    e_total=float(r['e_total']))
               for r in state['records'] if float(r['grad_max']) <= GM_GATE]
    return dict(outcome=outcome, optimizer=res_info,
                reuse_start=state['reuse_used'],
                n_new_evals=state['n_new'],
                accepted_iterates=state['accepted'],
                trial_sequence=state['trial_sequence'],
                meeting_points=meeting, warnings=wmsgs,
                optimizer_setup=_setup_note(maxiter))


def _setup_note(maxiter):
    return ('scipy method=BFGS, jac=True, gtol=1e-6, norm=inf, xrtol=0, '
            'c1=1e-4, c2=0.9, hess_inv0=21x21 identity (Bohr^2/Eh, '
            'algorithmic model, not a computed Hessian), maxiter=%d; '
            '21 Cartesian variables in Bohr; unconstrained, no '
            'projection, no per-step registration; fresh optimizer '
            'state; maxiter is NOT the evaluation budget (persisted '
            'opt cap guards every call)' % maxiter)


def main():
    man = json.load(open(MANIFEST))
    src_path = man['source']['source_path_wsl']
    sha = sha256_file(src_path)
    if sha != man['source']['source_sha256']:
        raise RuntimeError('HARD STOP: 048 opt_19 source file hash changed')
    if not man['source']['all_checks_pass']:
        raise RuntimeError('HARD STOP: source prechecks failed')
    src = json.load(open(src_path))
    print('[050] source OK: %s (sha256 %.16s...)'
          % (os.path.basename(src_path), sha), flush=True)

    results = dict(
        job='JOB-2026-0906-050 standard-BFGS limited optimization from '
            'the 048 last accepted point (opt_19)',
        doing='from the 048 last accepted point opt_19, run ONE limited '
              'standard-BFGS optimization (21 Cartesian variables, '
              'identity initial inverse Hessian) to test whether changing '
              'the curvature model changes gradient progress; acceptance '
              'gate unchanged (unprojected max|g|<=1e-5 + independent '
              'recheck)',
        source=dict(path=src_path, sha256=sha,
                    e_total=float(src['e_total']),
                    grad_max=float(src['grad_max']),
                    identity=dict(last_accepted=True, last_evaluated=True,
                                  lowest_energy_048=True)))
    ledger = ex.Ledger(LEDGER, caps=CAPS)
    backend = LoggingBackend(ex.RealBackend(OUT))
    status = 'aborted'
    try:
        R_start = (np.asarray(src['coords_actual_angstrom'], float) * BPA)
        rec_s = ex.eval_point('start', R_start, OUT, ledger, backend,
                              'start_repro',
                              gate_fn=ex.centre_gate_fn(src))
        g = rec_s['gate']
        print('[050] start gate %s: dE=%.2e dgrad=%.2e dC=%.2e'
              % ('PASS' if g['gates_pass'] else 'FAIL', g['dE'],
                 g['dgrad_max'], g['dcoords_A']), flush=True)
        results['start_reproduction'] = dict(
            gate=g, e_total=rec_s['e_total'],
            grad_max=rec_s['grad_max'],
            start_grad_meets_gate=bool(rec_s['grad_max'] <= GM_GATE))
        opt = run_bfgs(rec_s, OUT, ledger, backend)
        results['optimization'] = opt
        print('[050] optimization outcome: %s (new evals %d, '
              'reuse_start %s)' % (opt['outcome'], opt['n_new_evals'],
                                   opt['reuse_start']), flush=True)
        if opt.get('warnings'):
            print('[050] optimizer warnings: %s' % opt['warnings'],
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
                    status = 'stationary_point_candidate'
                    print('[050] recheck PASS -> registered: stationary '
                          'point candidate, stability & frequencies '
                          'pending', flush=True)
                else:
                    status = 'recheck_failed'
            except RuntimeError as e:
                stop = OUT + '/STOPPED_gate_fail_recheck.json'
                gate = json.load(open(stop))['gate'] \
                    if os.path.exists(stop) else dict(raw=str(e)[:300])
                results['recheck'] = dict(gate=gate, hard_stop=str(e)[:300])
                status = 'recheck_failed'
                print('[050] recheck FAILED -> not registered', flush=True)
        else:
            status = 'not_converged'
            print('[050] no saved point met max|g|<=1e-5 -> not converged',
                  flush=True)
    except Exception as exc:                                    # noqa: BLE001
        results['abort'] = dict(error=str(exc),
                                traceback=traceback.format_exc()[-1500:])
        status = 'aborted'
        print('[050] ABORT: %s' % exc, flush=True)

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
    results['final'] = dict(
        status=status,
        budget=dict(caps=CAPS, attempts=cats, failures=fails,
                    total=sum(cats.values())),
        lowest_energy_point=None if low is None else dict(
            tag=low.get('tag'), category=low.get('category'),
            e_total=float(low['e_total']), grad_max=float(low['grad_max'])),
        min_gradient_point=None if mng is None else dict(
            tag=mng.get('tag'), category=mng.get('category'),
            e_total=float(mng['e_total']), grad_max=float(mng['grad_max'])),
        note='single-run comparison limited to this start/settings/budget; '
             'no general optimizer ranking; stability & frequency '
             'acceptance NOT performed; no auto-continuation')
    results['budget'] = ledger.data
    ex.save_json_atomic(RESULTS, results)
    print('DONE: start_repro=%d/1 opt=%d/20 recheck=%d/1 status=%s'
          % (cats['start_repro'], cats['opt'], cats['recheck'], status),
          flush=True)


if __name__ == '__main__':
    main()
