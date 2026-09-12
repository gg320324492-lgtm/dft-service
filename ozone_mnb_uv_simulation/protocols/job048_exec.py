#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-048: limited full-DOF optimization from the actually
evaluated 045 +0.02 Bohr lowering point.

Budget (per-category, no borrowing): start_repro 1, opt 19, recheck 1
(total cap 21).  No stability / Hessian / frequencies / CP / other species.

Flow (reuses the validated JOB-047 execution components unchanged):
  (1) source sentinel: 045 eval_eval_displacement_+0.02.json must still
      match the sha256 recorded by job048_prep.py;
  (2) start reproduction gate (independent new object): |dE|<=1e-8 Eh,
      per-component |dgrad|<=1e-7 Eh/Bohr, |dcoords|<=1e-9 Angstrom,
      SCF converged, config identical, finite results;
  (3) ONE L-BFGS-B full-Cartesian optimization (21 variables, Bohr,
      gtol=1e-6, ftol=1e-13, maxcor=10, maxiter=19), unconstrained, no
      projection, no per-step registration, freshly initialized optimizer
      state (displacement-geometry start); the optimizer first request at
      the start coordinates reuses the reproduction record (no new SCF);
  (4) controlled stop as soon as any fully saved point reaches unprojected
      max|g| <= 1e-5 -> at most one independent new-object recheck ->
      register "stationary point candidate, stability & frequencies
      pending" only on PASS.
"""
import os, sys, json, glob, hashlib, time, traceback
import numpy as np

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
HERE = ROOT + '/protocols'
sys.path.insert(0, HERE)
import job047_exec as ex          # validated chain: Ledger, eval_point,
                                  # centre_gate_fn, run_optimization,
                                  # recheck_gate_fn, RealBackend

OUT = ROOT + '/run_artifacts/02_nh3o3_reference/c1_fulldof_opt048'
LEDGER = OUT + '/budget_fulldof048.json'
RESULTS = OUT + '/fulldof_opt048_results.json'
MANIFEST = OUT + '/input_manifest.json'

BPA = 1.0 / ex.ANG_PER_BOHR
CAPS = dict(start_repro=1, opt=19, recheck=1)
OPT_CAP = 19
MAXITER = 19
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
    """Thin wrapper: per-eval progress lines + exact trial coordinates
    (x_bohr) persisted inside every evaluation record."""

    def __init__(self, inner):
        self.inner = inner

    def full_eval(self, R_bohr, out_dir, tag):
        R = np.asarray(R_bohr, float)
        print('[048] eval %s: SCF+gradient start' % tag, flush=True)
        t0 = time.time()
        rec = self.inner.full_eval(R, out_dir, tag)
        rec['x_bohr'] = R.tolist()
        rec['x_bohr_sha'] = sha_arr(R)
        rec['seconds'] = round(time.time() - t0, 1)
        print('[048] eval %s: E=%.9f grad_max=%.3e conv=%s (%.0fs)'
              % (tag, rec['e_total'], rec['grad_max'], rec['converged'],
                 rec['seconds']), flush=True)
        return rec


def summarize_points(records):
    """endpoint / lowest-energy / smallest-gradient among saved points."""
    out = {}
    if records:
        out['lowest_energy_point'] = min(
            records, key=lambda r: float(r['e_total']))
        out['min_gradient_point'] = min(
            records, key=lambda r: float(r['grad_max']))
        out['last_evaluated_point'] = records[-1]
    return out


def main():
    man = json.load(open(MANIFEST))
    src_path = man['source']['source_path_wsl']
    sha = sha256_file(src_path)
    if sha != man['source']['source_sha256']:
        raise RuntimeError('HARD STOP: 045 source file hash changed')
    if not man['source']['all_checks_pass']:
        raise RuntimeError('HARD STOP: source prechecks failed')
    src = json.load(open(src_path))
    print('[048] source OK: %s (sha256 %.16s...)'
          % (os.path.basename(src_path), sha), flush=True)

    results = dict(
        job='JOB-2026-0906-048 limited full-DOF optimization from the '
            '045 +0.02 Bohr lowering point',
        doing='start from the actually evaluated 045 +0.02 Bohr point, '
              'relax ALL 21 Cartesian degrees of freedom with one '
              'budget-capped L-BFGS-B run, accept a stationary-point '
              'candidate only through an independent recheck',
        source=dict(path=src_path, sha256=sha,
                    t_Bohr=man['source']['displacement_t_Bohr'],
                    q_hash=man['source']['q_hash'],
                    e_total=float(src['e_total']),
                    grad_max=float(src['grad_max']),
                    dE_vs_045_centre_Eh=man['source']['dE_vs_045_centre_Eh']))
    ledger = ex.Ledger(LEDGER, caps=CAPS)
    backend = LoggingBackend(ex.RealBackend(OUT))
    status = 'aborted'

    try:
        # ---- (2) start reproduction gate
        R_start = (np.asarray(src['coords_actual_angstrom'], float) * BPA)
        rec_s = ex.eval_point('start', R_start, OUT, ledger, backend,
                              'start_repro',
                              gate_fn=ex.centre_gate_fn(src))
        g = rec_s['gate']
        print('[048] start gate %s: dE=%.2e dgrad=%.2e dC=%.2e'
              % ('PASS' if g['gates_pass'] else 'FAIL', g['dE'],
                 g['dgrad_max'], g['dcoords_A']), flush=True)
        results['start_reproduction'] = dict(
            gate=g, e_total=rec_s['e_total'],
            grad_max=rec_s['grad_max'],
            start_grad_meets_gate=bool(rec_s['grad_max'] <= GM_GATE))

        # ---- (3) one limited full-DOF optimization
        opt = ex.run_optimization(rec_s, OUT, ledger, backend,
                                  opt_cap=OPT_CAP, maxiter=MAXITER)
        results['optimization'] = opt
        print('[048] optimization outcome: %s (new evals %d, '
              'reuse_start %s)' % (opt['outcome'], opt['n_new_evals'],
                                   opt['reuse_start']), flush=True)

        # ---- (4) conditional independent recheck
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
                    print('[048] recheck PASS -> registered: stationary '
                          'point candidate, stability & frequencies '
                          'pending', flush=True)
                else:
                    status = 'recheck_failed'
            except RuntimeError as e:
                stop = OUT + '/STOPPED_gate_fail_recheck.json'
                gate = json.load(open(stop))['gate'] \
                    if os.path.exists(stop) else dict(raw=str(e))
                results['recheck'] = dict(gate=gate, hard_stop=str(e)[:300])
                status = 'recheck_failed'
                print('[048] recheck FAILED -> not registered', flush=True)
        else:
            status = 'not_converged'
            print('[048] no saved point met max|g|<=1e-5 -> not converged'
                  , flush=True)
    except Exception as exc:                       # noqa: BLE001
        results['abort'] = dict(error=str(exc),
                                traceback=traceback.format_exc()[-1500:])
        status = 'aborted'
        print('[048] ABORT: %s' % exc, flush=True)

    # ---- per-charter section 7 accounting
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
        what_done=dict(source_verification='offline (0 evals)',
                       start_repro_attempts=cats['start_repro'],
                       optimization_new_evals=cats['opt'],
                       recheck_attempts=cats['recheck'],
                       total_attempts=sum(cats.values())),
        budget=dict(caps=CAPS, attempts=cats, failures=fails,
                    ledger=LEDGER),
        endpoint=results.get('optimization', {}).get('optimizer'),
        lowest_energy_point=None if low is None else dict(
            tag=low.get('tag'), category=low.get('category'),
            e_total=float(low['e_total']),
            grad_max=float(low['grad_max'])),
        min_gradient_point=None if mng is None else dict(
            tag=mng.get('tag'), category=mng.get('category'),
            e_total=float(mng['e_total']),
            grad_max=float(mng['grad_max'])),
        note='stability and frequency acceptance NOT yet performed; '
             'no further optimization is auto-started after this batch')
    results['budget'] = ledger.data
    ex.save_json_atomic(RESULTS, results)
    print('DONE: start_repro=%d/1 opt=%d/19 recheck=%d/1 status=%s'
          % (cats['start_repro'], cats['opt'], cats['recheck'], status),
          flush=True)


if __name__ == '__main__':
    main()
