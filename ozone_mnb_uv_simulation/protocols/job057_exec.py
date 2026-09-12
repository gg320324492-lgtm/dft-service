#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-057: full-DOF standard-BFGS optimization from the 056
t=0.050 Bohr one-dimensional low point.

Budget (no borrowing): start_repro 1, opt 25, recheck 1, total <= 27.
Stop target = the ORIGINAL acceptance gate max|g|<=1e-5; independent
recheck on success; on budget exhaustion / precision loss / line-search
failure / exception: save and STOP (no recheck, no param change, no other
optimizer).

Reuses the validated 051 components unchanged (run_bfgs / LoggingBackend /
ex.Ledger / eval_point / gates / RealBackend); only the source, run dir,
opt cap and the post-run offline comparison are adapted.
"""
import os, sys, json, glob, time, traceback
import numpy as np

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
HERE = ROOT + '/protocols'
sys.path.insert(0, HERE)
import job047_exec as ex
import job051_exec as j51          # validated run_bfgs (gate 1e-5) etc.

OUT = ROOT + '/run_artifacts/02_nh3o3_reference/c1_fdopt057'
LEDGER = OUT + '/budget_fdopt057.json'
RESULTS = OUT + '/fdopt057_results.json'
MANIFEST = OUT + '/input_manifest.json'

BPA = 1.0 / ex.ANG_PER_BOHR
CAPS = dict(start_repro=1, opt=25, recheck=1)
OPT_CAP = 25
GM_GATE = ex.GM_GATE              # 1e-5


def sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for blk in iter(lambda: fh.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def contact_geometry(x_bohr):
    """NH3...O3 contact descriptors (zero-eval geometric analysis)."""
    R = np.asarray(x_bohr, float).reshape(7, 3)
    cenN = R[:4].mean(axis=0)
    cenO = R[4:].mean(axis=0)
    dNO = [(float(np.linalg.norm(R[0] - R[o])), 'N-O%d' % (o - 4))
           for o in (4, 5, 6)]
    dHO = [(float(np.linalg.norm(R[h] - R[o])), 'H%d-O%d' % (h - 4, o - 4))
           for h in (1, 2, 3) for o in (4, 5, 6)]
    dNO.sort()
    dHO.sort()
    v = cenO - cenN
    return dict(NH3_O3_centroid_dist_Bohr=float(np.linalg.norm(v)),
                min_N_O_Bohr=dNO[0][0], min_N_O_pair=dNO[0][1],
                min_H_O_Bohr=dHO[0][0], min_H_O_pair=dHO[0][1],
                centroid_axis_unit=(v / np.linalg.norm(v)).tolist())


def main():
    man = json.load(open(MANIFEST))
    src_path = man['source']['source_path_wsl']
    if sha256_file(src_path) != man['source']['source_sha256']:
        raise RuntimeError('HARD STOP: 056 t=050 source hash changed')
    if not man['all_checks_pass']:
        raise RuntimeError('HARD STOP: source prechecks failed')
    src = json.load(open(src_path))
    print('[057] source OK: %s (sha256 %.16s...)'
          % (os.path.basename(src_path), man['source']['source_sha256']),
          flush=True)

    results = dict(
        job='JOB-2026-0906-057 full-DOF BFGS optimization from the 056 '
            't=0.050 one-dimensional low point',
        doing='full-DOF standard-BFGS relaxation (21 Cartesian, optimizer '
              'state reset) to test whether the transverse gradients can '
              'descend to the original max|g|<=1e-5 gate and form an '
              'independently recheckable stationary-point candidate',
        source=dict(path=src_path, sha256=man['source']['source_sha256'],
                    e_total=float(src['e_total']),
                    grad_max=float(src['grad_max']),
                    identity=dict(actual_056_t050_point=True),
                    continuation_mode='geometry continuation, BFGS '
                                      'optimizer state RESET'),
        environment=man['environment'],
        stop_target=dict(gate=1e-5, note='ORIGINAL acceptance gate; '
                                         'recheck only on success'))
    ledger = ex.Ledger(LEDGER, caps=CAPS)
    backend = j51.LoggingBackend(ex.RealBackend(OUT))
    status = 'aborted'
    try:
        R_start = (np.asarray(src['coords_actual_angstrom'], float) * BPA)
        rec_s = ex.eval_point('start', R_start, OUT, ledger, backend,
                              'start_repro',
                              gate_fn=ex.centre_gate_fn(src))
        g = rec_s['gate']
        print('[057] start gate %s: dE=%.2e dgrad=%.2e dC=%.2e (start '
              'gmax=%.3e above 1e-5 is NOT a rejection)'
              % ('PASS' if g['gates_pass'] else 'FAIL', g['dE'],
                 g['dgrad_max'], g['dcoords_A'], rec_s['grad_max']),
              flush=True)
        results['start_reproduction'] = dict(
            gate=g, e_total=rec_s['e_total'],
            grad_max=rec_s['grad_max'])
        opt = j51.run_bfgs(rec_s, OUT, ledger, backend, opt_cap=OPT_CAP)
        results['optimization'] = opt
        print('[057] optimization outcome: %s (new evals %d, reuse %s, '
              'blocked %d)'
              % (opt['outcome'], opt['n_new_evals'], opt['reuse_start'],
                 len(opt['blocked_requests'])), flush=True)
        if opt.get('warnings'):
            print('[057] optimizer warnings: %s' % opt['warnings'],
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
                    print('[057] recheck PASS -> registered: stationary '
                          'point candidate, stability & frequencies '
                          'pending', flush=True)
                else:
                    status = 'recheck_failed'
                    print('[057] recheck FAILED -> not registered',
                          flush=True)
            except RuntimeError as e:
                stop = OUT + '/STOPPED_gate_fail_recheck.json'
                gate = json.load(open(stop))['gate'] \
                    if os.path.exists(stop) else dict(raw=str(e)[:300])
                results['recheck'] = dict(gate=gate, hard_stop=str(e)[:300])
                status = 'recheck_failed'
                print('[057] recheck FAILED -> not registered', flush=True)
        else:
            status = 'not_converged'
            print('[057] no saved point met max|g|<=1e-5 -> not converged',
                  flush=True)
    except Exception as exc:                                # noqa: BLE001
        results['abort'] = dict(error=str(exc),
                                traceback=traceback.format_exc()[-1500:])
        status = 'aborted'
        print('[057] ABORT: %s' % exc, flush=True)

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
    # offline comparison vs the 056 t=0.050 start (zero-eval geometry)
    geom_cmp = None
    if saved:
        ref = contact_geometry(src['x_bohr'])
        cand = {}
        for label, rec in (('lowest_energy', low), ('min_gradient', mng)):
            x = (np.asarray(rec['coords_actual_angstrom'], float) * BPA)
            cand[label] = dict(tag=rec.get('tag'),
                               geometry=contact_geometry(x),
                               delta_vs_start={k: cand_delta(
                                   ref, contact_geometry(x), k)
                               for k in ('NH3_O3_centroid_dist_Bohr',
                                         'min_N_O_Bohr', 'min_H_O_Bohr')},
                               grad_max=float(rec['grad_max']),
                               e_total=float(rec['e_total']))
        geom_cmp = dict(start=ref, candidates=cand,
                        note='geometry observation only; energy lowering '
                             'is NOT a binding free energy; a 1-D low '
                             'point or optimizer success is NOT a stable '
                             'complex')
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
        geometry_comparison=geom_cmp,
        bookkeeping_note='optimizer requests / completed evaluations / '
                         'accepted iterates / budget-blocked requests are '
                         'reported separately',
        note='single limited optimization; no step-count forecast; no '
             'auto-continuation; stability & curvature verification NOT '
             'performed')
    results['budget'] = ledger.data
    ex.save_json_atomic(RESULTS, results)
    print('DONE: start_repro=%d/1 opt=%d/25 recheck=%d/1 status=%s'
          % (cats['start_repro'], cats['opt'], cats['recheck'], status),
          flush=True)


def cand_delta(ref, cand, k):
    return float(cand[k] - ref[k])


if __name__ == '__main__':
    main()
