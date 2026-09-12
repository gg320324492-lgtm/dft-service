#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-066: C2 mirror-branch (negative-normal 3H) six-point
fixed-orientation distance scan.

Budget (hard): 6 SCF+full-gradient attempts, one per mirror geometry
from the 066 manifest.  Config identical to 064/JOB-031.  Each record
saved atomically BEFORE the attempt is marked done.  ANY anomaly -> save
and STOP.  Analysis uses the mirror delta=0 as reference and reports the
064 positive-normal curve BESIDE it (the two branches are NEVER joined
into one continuous path).
"""
import os, sys, json, time, traceback
import numpy as np

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
HERE = ROOT + '/protocols'
sys.path.insert(0, HERE)
import job047_exec as ex
import job052_exec as j52

R64 = ROOT + '/run_artifacts/02_nh3o3_reference/c2_scan064'
OUT = ROOT + '/run_artifacts/02_nh3o3_reference/c2_mirror_scan066'
MANIFEST = OUT + '/input_manifest.json'
RESULTS = OUT + '/c2_mirror_scan066_results.json'
LEDGER = OUT + '/budget_mirror066.json'
SYMS = ['N', 'H', 'H', 'H', 'O', 'O', 'O']


class ScanLedger(ex.Ledger):
    def __init__(self, path):
        ex.Ledger.__init__(self, path, caps=dict(scan=6))
        self.data['note'] = ('JOB-066 hard budget: six SCF+full-gradient '
                             'attempts on the MIRROR branch; failures '
                             'count; any anomaly -> save and STOP, no '
                             'skip, no restart, no extra budget')
        self._save()


def quantum_eval(coords_bohr, out_dir, tag):
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


def analyse(man, recs, pos64):
    pts = []
    for g in man['geometries']:
        r = recs[g['tag']]
        X = np.asarray(r['coords_actual_angstrom'], float).reshape(7, 3)
        pts.append(dict(delta=g['delta_A'],
                        d_HO_actual=[float(np.linalg.norm(X[1] - X[4])),
                                     float(np.linalg.norm(X[3] - X[6]))],
                        e_total=float(r['e_total']),
                        grad_max=float(r['grad_max']),
                        h3_signed_height_A=g['h3_signed_height_A'],
                        geom_sha=g['coords_sha']))
    e0 = [p for p in pts if abs(p['delta']) < 1e-12][0]['e_total']
    for p in pts:
        p['dE_vs_mirror_delta0'] = p['e_total'] - e0
    slopes = []
    for a, b in zip(pts[:-1], pts[1:]):
        slopes.append(dict(delta_mid=(a['delta'] + b['delta']) / 2,
                           slope_Eh_per_A=(b['e_total'] - a['e_total'])
                           / (b['delta'] - a['delta'])))
    sign_changes = sum(1 for s1, s2 in zip(slopes[:-1], slopes[1:])
                       if s1['slope_Eh_per_A'] * s2['slope_Eh_per_A']
                       < 0)
    # side-by-side with the 064 positive-normal branch (separate lists,
    # never joined)
    side_by_side = []
    for p in pts:
        q = [x for x in pos64 if abs(x['delta'] - p['delta']) < 1e-12]
        side_by_side.append(dict(
            delta=p['delta'],
            mirror_dE=p['dE_vs_mirror_delta0'],
            positive_dE=q[0]['dE_vs_delta0'] if q else None,
            dE_difference=p['dE_vs_mirror_delta0']
            - (q[0]['dE_vs_delta0'] if q else None)))
    return dict(points=pts, secant_slopes=slopes,
                slope_sign_changes=sign_changes,
                minimum_within_sampled_range=bool(
                    slopes[0]['slope_Eh_per_A'] < 0
                    and slopes[-1]['slope_Eh_per_A'] > 0),
                reference='mirror-branch delta = 0.00 point',
                side_by_side_with_064_positive=side_by_side,
                branches_note='positive-normal and mirror curves are '
                              'reported SEPARATELY; never joined')


def render_chart(ana, out):
    W, H = 900, 560
    pts = ana['points']
    dmin, dmax = -0.25, 0.65
    vals = ([p['dE_vs_mirror_delta0'] for p in pts]
            + [s['positive_dE'] for s in ana[
                'side_by_side_with_064_positive']
                if s['positive_dE'] is not None])
    vmin, vmax = min(vals) - 0.005, max(vals) + 0.005
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" '
             'height="%d" font-family="monospace" font-size="12">'
             '<rect width="100%%" height="100%%" fill="white"/>'
             '<text x="16" y="24" font-size="15" font-weight="bold">'
             'JOB-066 mirror branch vs JOB-064 positive branch (dE, '
             'separate curves)</text>' % (W, H)]
    x0, y0, w, h = 70, 50, 780, 430

    def xy(d, e):
        return (x0 + (d - dmin) / (dmax - dmin) * w,
                y0 + h - (e - vmin) / (vmax - vmin) * h)
    parts.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="black"/>'
                 '<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="black"/>'
                 % (x0, y0 + h, x0 + w, y0 + h, x0, y0, x0, y0 + h))
    # mirror branch polyline
    pts_m = [(p['delta'], p['dE_vs_mirror_delta0']) for p in pts]
    s = ' '.join('%.1f,%.1f' % xy(d, e) for d, e in pts_m)
    parts.append('<polyline points="%s" fill="none" stroke="#d62728" '
                 'stroke-width="1.8"/>' % s)
    for d, e in pts_m:
        px, py = xy(d, e)
        parts.append('<circle cx="%.1f" cy="%.1f" r="4" '
                     'fill="#d62728"/>' % (px, py))
    # 064 positive branch (separate series)
    pos = [(s['delta'], s['positive_dE'])
           for s in ana['side_by_side_with_064_positive']
           if s['positive_dE'] is not None]
    s2 = ' '.join('%.1f,%.1f' % xy(d, e) for d, e in pos)
    parts.append('<polyline points="%s" fill="none" stroke="#1f77b4" '
                 'stroke-width="1.8" stroke-dasharray="6,3"/>' % s2)
    for d, e in pos:
        px, py = xy(d, e)
        parts.append('<rect x="%.1f" y="%.1f" width="7" height="7" '
                     'fill="none" stroke="#1f77b4"/>' % (px - 3, py - 3))
    for p in pts:
        px, _ = xy(p['delta'], 0)
        parts.append('<text x="%.1f" y="%d" text-anchor="middle">%+.2f'
                     '</text>' % (px, y0 + h + 18, p['delta']))
    parts.append('<circle cx="640" cy="60" r="4" fill="#d62728"/>'
                 '<text x="652" y="64">mirror branch (066, this batch)'
                 '</text>'
                 '<rect x="636" y="76" width="8" height="8" '
                 'fill="none" stroke="#1f77b4"/>'
                 '<text x="652" y="84">positive branch (064) - '
                 'separate series</text>'
                 '<text x="16" y="%d" font-size="11">branches are '
                 'NEVER joined into one continuous path</text></svg>'
                 % (H - 12))
    with open(os.path.join(out, 'mirror_vs_positive.svg'), 'w') as fh:
        fh.write('\n'.join(parts))


def main():
    man = json.load(open(MANIFEST))
    if not man['all_checks_pass']:
        raise RuntimeError('HARD STOP: manifest prechecks failed')
    pos64 = json.load(open(os.path.join(
        R64, 'c2_scan064_results.json')))['analysis']['points']
    print('[066s] manifest OK: six mirror geometries (negative-normal '
          '3H), deltas %s' % man['deltas_A'], flush=True)

    results = dict(
        job='JOB-2026-0906-066: C2 mirror-branch six-point '
            'fixed-orientation distance scan',
        branch='COPLANAR PROJECT MODEL, NEGATIVE-normal 3H mirror '
               'branch (independent per JOB-065)',
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
                rec['coords_sha_mirror_target'] = g['coords_sha']
                rec['h3_signed_height_A'] = g['h3_signed_height_A']
                if not (rec['converged'] and rec['all_finite']):
                    raise RuntimeError('%s SCF not converged or non-'
                                       'finite -> HARD STOP' % tag)
                ex.save_json_atomic(os.path.join(OUT, 'eval_%s.json'
                                                 % tag), rec)
                led.post_eval(att, dict(e_total=rec['e_total'],
                                        grad_max=rec['grad_max']))
                recs[tag] = rec
                print('[066s] %s: E=%.9f gmax=%.3e'
                      % (tag, rec['e_total'], rec['grad_max']),
                      flush=True)
            except Exception as e:                      # noqa: BLE001
                led.fail(att, e)
                raise
        ana = analyse(man, recs, pos64)
        results['analysis'] = ana
        render_chart(ana, OUT)
        status = 'completed'
        mw = ana['minimum_within_sampled_range']
        results['final'] = dict(
            status=status,
            registration=(
                'MIRROR-branch fixed-orientation scan under the '
                'coplanar project model: an energy minimum within the '
                'sampled range IS indicated by the sampled slopes'
                if mw else
                'MIRROR-branch fixed-orientation scan under the '
                'coplanar project model: NO energy minimum indicated '
                'within the sampled range (monotonic sampled slopes)'),
            scope_limits='negative-normal 3H mirror branch of the '
                         'coplanar project model ONLY; not a literature '
                         '3D reproduction; no minimum registration; no '
                         'barrier/binding/water-treatment claims; the '
                         'two branches are never merged into one path')
    except Exception as exc:                                # noqa: BLE001
        results['abort'] = dict(error=str(exc),
                                traceback=traceback.format_exc()[-1500:])
        status = 'aborted'
        print('[066s] ABORT: %s' % exc, flush=True)
    results['final_status'] = status
    results['budget'] = led.data
    ex.save_json_atomic(RESULTS, results)
    print('[066s] DONE: %s' % status, flush=True)


if __name__ == '__main__':
    main()
