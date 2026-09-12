#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-064: C2 coplanar-model six-point fixed-orientation scan.

Budget (hard): 6 SCF+full-gradient attempts, one per preregistered
geometry (delta = -0.20..+0.60 A on both H...O contacts).  No centre
reproduction.  Config = the JOB-031 formal scan configuration (wb97xd +
project explicit D2 once, def2-TZVP, grid level 8, grid_response=True,
SCF 1e-12/1e-9, gas).  Each record is saved atomically BEFORE the
attempt is marked done.  ANY SCF/gradient/finite/config error -> save
and STOP (no auto-advance, no restart, no separate ledger).  Analysis:
dE vs the delta=0 point, max|g|, geometry hashes, distance-energy curve,
minimum-in-range and radial-derivative sign-change checks - scoped
strictly to the coplanar project model.
"""
import os, sys, json, time, traceback
import numpy as np

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
HERE = ROOT + '/protocols'
sys.path.insert(0, HERE)
import job047_exec as ex
import job052_exec as j52          # endpoint_eval (JOB-031 config path)

OUT = ROOT + '/run_artifacts/02_nh3o3_reference/c2_scan064'
MANIFEST = OUT + '/input_manifest.json'
RESULTS = OUT + '/c2_scan064_results.json'
LEDGER = OUT + '/budget_c2scan064.json'
SYMS = ['N', 'H', 'H', 'H', 'O', 'O', 'O']


class ScanLedger(ex.Ledger):
    def __init__(self, path):
        ex.Ledger.__init__(self, path, caps=dict(scan=6))
        self.data['note'] = ('JOB-064 hard budget: six SCF+full-gradient '
                             'attempts, one per preregistered point; '
                             'failures count; no borrowing; any error -> '
                             'save and STOP, no auto-advance, no '
                             'restart, no separate ledger')
        self._save()


def quantum_eval(coords_bohr, out_dir, tag):
    """JOB-031 configuration: 052 endpoint_eval on a fresh mol."""
    from pyscf import gto
    Rm = np.asarray(coords_bohr, float).reshape(7, 3)
    mol = gto.M(atom=[(s, (float(r[0]), float(r[1]), float(r[2])))
                      for s, r in zip(SYMS, Rm)],
                basis='def2-TZVP', charge=0, spin=0, verbose=0,
                max_memory=4000, unit='Bohr')
    rec, mf, _ = j52.endpoint_eval(mol, out_dir, tag)
    rb = float(np.abs(np.asarray(mol.atom_coords(unit='Bohr'), float)
                      .reshape(-1) - Rm.reshape(-1)).max())
    if rb > 0.0:
        raise RuntimeError('unit readback mismatch %e' % rb)
    return rec


def sha_arr(a):
    import hashlib
    return hashlib.sha256(np.asarray(a, float).tobytes()).hexdigest()[:16]


def analyse(man, recs):
    """dE vs the delta=0 point, radial slopes, minimum-in-range and
    derivative sign-change checks."""
    pts = []
    for g in man['geometries']:
        r = recs[g['tag']]
        X = np.asarray(r['coords_actual_angstrom'], float).reshape(7, 3)
        pts.append(dict(delta=g['delta_A'],
                        d_HO_actual=[float(np.linalg.norm(X[1] - X[4])),
                                     float(np.linalg.norm(X[3] - X[6]))],
                        e_total=float(r['e_total']),
                        grad_max=float(r['grad_max']),
                        geom_sha=g['coords_sha']))
    e0 = [p for p in pts if abs(p['delta']) < 1e-12][0]['e_total']
    for p in pts:
        p['dE_vs_delta0'] = p['e_total'] - e0
    # secant slopes between adjacent points (non-uniform grid)
    slopes = []
    for a, b in zip(pts[:-1], pts[1:]):
        slopes.append(dict(delta_mid=(a['delta'] + b['delta']) / 2,
                           slope_Eh_per_A=(b['e_total'] - a['e_total'])
                           / (b['delta'] - a['delta'])))
    sign_changes = sum(1 for s1, s2 in zip(slopes[:-1], slopes[1:])
                       if s1['slope_Eh_per_A'] * s2['slope_Eh_per_A']
                       < 0)
    e_min = min(p['e_total'] for p in pts)
    interior_min = [p['delta'] for p in pts
                    if p['e_total'] == e_min
                    and pts.index(p) not in (0, len(pts) - 1)]
    return dict(points=pts, secant_slopes=slopes,
                slope_sign_changes=sign_changes,
                lowest_energy_delta=min((p['delta'] for p in pts
                                         if p['e_total'] == e_min)),
                minimum_within_sampled_range=bool(
                    interior_min and slopes[0]['slope_Eh_per_A'] < 0
                    and slopes[-1]['slope_Eh_per_A'] > 0),
                reference='delta = 0.00 point')


def render_chart(pts, out):
    W, H = 900, 560
    dmin = min(p['delta'] for p in pts)
    dmax = max(p['delta'] for p in pts)
    emin = min(p['dE_vs_delta0'] for p in pts)
    emax = max(max(p['dE_vs_delta0'] for p in pts), 1e-12)
    pad = 0.05
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" '
             'height="%d" font-family="monospace" font-size="12">'
             '<rect width="100%%" height="100%%" fill="white"/>'
             '<text x="16" y="24" font-size="15" font-weight="bold">'
             'JOB-064 C2 coplanar-model scan: dE vs delta (ref = '
             'delta 0)</text>' % (W, H)]
    x0, y0, w, h = 70, 50, 760, 420
    parts.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="black"/>'
                 '<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="black"/>'
                 % (x0, y0 + h, x0 + w, y0 + h, x0, y0, x0, y0 + h))

    def xy(d, e):
        return (x0 + (d - dmin) / (dmax - dmin + 1e-30) * w,
                y0 + h - (e - emin + pad * (emax - emin))
                / (emax - emin + 2 * pad * (emax - emin) + 1e-30) * h)
    for p in pts:
        px, py = xy(p['delta'], p['dE_vs_delta0'])
        parts.append('<circle cx="%.1f" cy="%.1f" r="4" fill="#1f77b4"/>'
                     '<text x="%.1f" y="%.1f" text-anchor="middle">'
                     '%+.6f</text>'
                     % (px, py, px, py - 12, p['dE_vs_delta0']))
        parts.append('<text x="%.1f" y="%d" text-anchor="middle">%+.2f'
                     '</text>' % (px, y0 + h + 18, p['delta']))
    parts.append('<text x="%d" y="%d">delta (A)</text>'
                 '<text x="16" y="%d" transform="rotate(-90 16 %d)">'
                 'dE vs delta0 (Eh)</text>'
                 % (x0 + w - 60, y0 + h + 42, y0 + h // 2,
                    y0 + h // 2))
    parts.append('<text x="16" y="%d" font-size="11">fixed orientation '
                 '(coplanar project model); batches/points are the six '
                 'preregistered deltas</text></svg>'
                 % (H - 12))
    with open(os.path.join(out, 'scan_curve.svg'), 'w') as fh:
        fh.write('\n'.join(parts))


def main():
    man = json.load(open(MANIFEST))
    if not man['all_checks_pass']:
        raise RuntimeError('HARD STOP: manifest prechecks failed')
    print('[064s] manifest OK: six preregistered coplanar-model '
          'geometries, branch=%s' % man['scan']['branch'], flush=True)

    results = dict(
        job='JOB-2026-0906-064: C2 coplanar-assumption model six-point '
            'fixed-orientation distance scan',
        model_assumptions=man['model_assumptions'],
        quantum_config=man['quantum_config'],
        caps=man['caps'])
    led = ScanLedger(LEDGER)
    status = 'aborted'
    recs = {}
    try:
        for g in man['geometries']:
            tag = g['tag']
            att = led.pre_eval('scan', dict(tag=tag, delta=g['delta_A']))
            try:
                rec = quantum_eval(np.asarray(g['coords_bohr'], float),
                                   OUT, tag)
                rec['delta_A'] = g['delta_A']
                rec['coords_sha_target'] = g['coords_sha']
                if not (rec['converged'] and rec['all_finite']):
                    raise RuntimeError('%s SCF not converged or non-'
                                       'finite -> HARD STOP' % tag)
                ex.save_json_atomic(os.path.join(OUT, 'eval_%s.json'
                                                 % tag), rec)
                led.post_eval(att, dict(e_total=rec['e_total'],
                                        grad_max=rec['grad_max']))
                recs[tag] = rec
                print('[064s] %s: E=%.9f gmax=%.3e'
                      % (tag, rec['e_total'], rec['grad_max']),
                      flush=True)
            except Exception as e:                      # noqa: BLE001
                led.fail(att, e)
                raise
        ana = analyse(man, recs)
        results['analysis'] = ana
        render_chart(ana['points'], OUT)
        status = 'completed'
        mw = ana['minimum_within_sampled_range']
        results['final'] = dict(
            status=status,
            registration=(
                'fixed-orientation scan under the COPLANAR PROJECT '
                'MODEL: an energy minimum within the sampled delta '
                'range IS indicated by the sampled slopes'
                if mw else
                'fixed-orientation scan under the COPLANAR PROJECT '
                'MODEL: NO energy minimum indicated within the sampled '
                'delta range (monotonic sampled slopes)'),
            accounting=dict(attempts=led.count('scan'),
                            cap=6,
                            failures=sum(1 for a in led.data['attempts']
                                         if a.get('status') == 'error')),
            scope_limits='coplanar project model ONLY; does NOT '
                         'reproduce the literature 3D structure; no '
                         'minimum registration; no claim about '
                         'barriers, binding strength or water-treatment '
                         'relevance; single fixed orientation')
    except Exception as exc:                                # noqa: BLE001
        results['abort'] = dict(error=str(exc),
                                traceback=traceback.format_exc()[-1500:])
        status = 'aborted'
        print('[064s] ABORT: %s' % exc, flush=True)
    results['final_status'] = status
    results['budget'] = led.data
    ex.save_json_atomic(RESULTS, results)
    print('[064s] DONE: %s' % status, flush=True)


if __name__ == '__main__':
    main()
