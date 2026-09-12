#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-056: refine the 0.04-0.06 Bohr sign-change interval with
three new points (t=0.045/0.050/0.055) on the 055 line.

Budget: refine 3 (failures counted; no borrowing; no retry; no auto extra
points).  The 055 centre is REUSED - NO centre SCF.  ON ANY EXCEPTION OR
NON-CONVERGENCE: save and STOP.

Analysis handover returns a SINGLE DICT merging the new points with the
055 saved t=0/0.04/0.06 records into ONE line table.  Secant slopes are
never presented as endpoint analytic derivatives.  Even an exact 1-D low
point is NOT a full-DOF minimum (transverse gradients unchecked).
"""
import os, sys, json, time, traceback
import numpy as np

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
HERE = ROOT + '/protocols'
sys.path.insert(0, HERE)
import job047_exec as ex

OUT = ROOT + '/run_artifacts/02_nh3o3_reference/c1_refine056'
LEDGER = OUT + '/budget_refine056.json'
RESULTS = OUT + '/refine056_results.json'
MANIFEST = OUT + '/input_manifest.json'

BPA = 1.0 / ex.ANG_PER_BOHR
CAPS = dict(refine=3)


def sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for blk in iter(lambda: fh.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


class Ledger:
    CAPS = dict(CAPS)

    def __init__(self, path):
        self.path = path
        if os.path.exists(path):
            self.data = json.load(open(path))
            for a in self.data.get('attempts', []):
                if a.get('status') == 'error':
                    raise RuntimeError('ledger has error attempt -> HARD '
                                       'STOP (no silent resume): '
                                       + json.dumps(a)[:300])
        else:
            self.data = dict(caps=dict(self.CAPS),
                             note='failures count; no borrowing; no retry; '
                                  'no auto extra points; on exception or '
                                  'non-convergence: save and STOP; the 055 '
                                  'centre is reused (no centre SCF)',
                             attempts=[])
            self._save()

    def _save(self):
        ex.save_json_atomic(self.path, self.data)

    def count(self, cat):
        return sum(1 for a in self.data['attempts']
                   if a['category'] == cat)

    def pre(self, cat, note=None):
        if self.count(cat) >= self.CAPS[cat]:
            raise RuntimeError('cap reached for %s (no auto extra points)'
                               % cat)
        att = dict(attempt=len(self.data['attempts']) + 1, category=cat,
                   note=note or {}, status='pending',
                   started=time.strftime('%F %T'))
        self.data['attempts'].append(att)
        self._save()
        return att

    def post(self, att, fields):
        att.update(fields)
        att['status'] = 'done'
        att['finished'] = time.strftime('%F %T')
        self._save()

    def fail(self, att, err):
        att['status'] = 'error'
        att['error'] = str(err)[:500]
        self._save()


def _eval_point(R_bohr, out_dir, tag):
    """Production evaluation via the validated 047 RealBackend chain.
    Returns (record, unit_readback_maxdev_Bohr)."""
    backend = ex.RealBackend(out_dir)
    R = np.asarray(R_bohr, float).reshape(7, 3)
    mol = backend.make_mol(R)
    readback = float(np.abs(np.asarray(mol.atom_coords(unit='Bohr'),
                                       float).reshape(-1)
                            - R.reshape(-1)).max())
    if readback > 0.0:
        raise RuntimeError('unit readback mismatch %e' % readback)
    rec = backend.full_eval(R, out_dir, tag)
    rec['tag'] = tag
    rec['x_bohr'] = R.tolist()
    rec['unit_readback_maxdev_Bohr'] = readback
    ex.save_json_atomic(os.path.join(out_dir, 'eval_%s.json' % tag), rec)
    return rec, readback


def tag_for(t):
    return 'refine_t%03d' % int(round(t * 1000))


def run_refine(man, led, eval_fn, out_dir):
    """Actual execution entry: the three new points in FIXED order
    (t=0.045, 0.050, 0.055).  NO centre SCF (the 055 centre is reused).
    Returns a SINGLE DICT {records, analysis}."""
    q = np.asarray(man['direction']['q'], float).reshape(-1)
    R0 = np.asarray(man['centre']['x_bohr'], float).reshape(7, 3)
    recs = []
    for g in man['new_geometries']:
        t = g['t_Bohr']
        R = np.asarray(g['coords_bohr'], float).reshape(7, 3)
        att = led.pre('refine', dict(tag=tag_for(t), t_Bohr=t))
        try:
            rec, rb = eval_fn(R, out_dir, tag_for(t))
            if rb > 0.0:
                raise RuntimeError('unit readback mismatch %e' % rb)
            rec['t_Bohr'] = t
            led.post(att, dict(e_total=rec['e_total'],
                               grad_max=rec['grad_max'],
                               seconds=rec.get('seconds')))
            recs.append(rec)
            print('[056] %s (t=%.3f): E=%.9f gmax=%.3e'
                  % (tag_for(t), t, rec['e_total'], rec['grad_max']),
                  flush=True)
        except Exception as e:                              # noqa: BLE001
            led.fail(att, e)
            raise
    return dict(records=recs, analysis=analyse(man, recs, q))


def analyse(man, new_recs, q):
    """Merge new points with the 055 saved t=0/0.04/0.06 records into ONE
    line table.  Single-dict handover."""
    E0 = float(man['centre']['e_total'])
    g0 = np.asarray(man['centre']['grad'], float).reshape(-1)

    def mk(t, e, g, tag, src):
        g = np.asarray(g, float).reshape(-1)
        return dict(t_Bohr=float(t), tag=tag, source=src,
                    e_total=float(e), dE_vs_055centre=float(e) - E0,
                    grad_max=float(np.abs(g).max()), a_Eh_Bohr=float(g @ q),
                    kind='analytic_directional_derivative')
    pts = [mk(0.0, man['centre']['e_total'], g0, 'centre_055',
              'reused 055 centre')]
    r040 = man['reused_055_records']['t040']
    pts.append(mk(r040['t_Bohr'], r040['e_total'], r040['grad'],
                  'scan_t040_055', 'reused 055 record'))
    for r in sorted(new_recs, key=lambda r: r['t_Bohr']):
        pts.append(mk(r['t_Bohr'], r['e_total'], r['grad'], r['tag'],
                      'new 056 evaluation'))
    r060 = man['reused_055_records']['t060']
    pts.append(mk(r060['t_Bohr'], r060['e_total'], r060['grad'],
                  'scan_t060_055', 'reused 055 record'))
    pts.sort(key=lambda p: p['t_Bohr'])
    # adjacent-point secant slopes (NOT endpoint derivatives)
    secants = []
    for i in range(1, len(pts)):
        t0, t1 = pts[i - 1]['t_Bohr'], pts[i]['t_Bohr']
        secants.append(dict(t_from=t0, t_to=t1,
                            slope_Eh_Bohr=(pts[i]['dE_vs_055centre']
                                           - pts[i - 1]['dE_vs_055centre'])
                            / (t1 - t0),
                            note='adjacent-point energy secant slope; not '
                                 'the derivative of either endpoint'))
    a_vals = [p['a_Eh_Bohr'] for p in pts]
    t_vals = [p['t_Bohr'] for p in pts]
    sign_changes = [(t_vals[i - 1], t_vals[i]) for i in range(1, len(a_vals))
                    if (a_vals[i - 1] < 0 <= a_vals[i])
                    or (a_vals[i - 1] > 0 >= a_vals[i])]
    e_min_i = int(np.argmin([p['dE_vs_055centre'] for p in pts]))
    min_location = ('centre' if e_min_i == 0 else
                    ('interior' if 0 < e_min_i < len(pts) - 1
                     else 'boundary'))
    if sign_changes:
        registration = ('a narrower fixed-line sign-change interval: %s'
                        % ['%.3f-%.3f Bohr' % sc for sc in sign_changes])
    else:
        registration = ('no sign change of a(t) on the refined sampling '
                        '[0, %.3f]; lowest sampled energy at t=%.3f '
                        '(location: %s); reported as-is, no auto extra '
                        'points' % (t_vals[-1], pts[e_min_i]['t_Bohr'],
                                    min_location))
    return dict(points=pts, secant_slopes=secants,
                a0_055centre_Eh_Bohr=float(g0 @ q),
                sign_changes=sign_changes,
                lowest_sample_point=dict(index=e_min_i,
                                         t_Bohr=pts[e_min_i]['t_Bohr'],
                                         location=min_location,
                                         dE_vs_055centre=pts[e_min_i][
                                             'dE_vs_055centre']),
                lowest_interior=bool(min_location == 'interior'),
                registration=registration,
                limits='full-DOF minimum NOT claimed: transverse gradients '
                       'require independent checks; secant slopes are not '
                       'endpoint derivatives; no reaction-mechanism or '
                       'water-treatment conclusions')


def make_svg(pts, path):
    W, H, m = 900, 420, 70

    def series(vals):
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-18:
            hi = lo + 1e-18
        return lo, hi

    def coords(t, v, tlo, thi, vlo, vhi, ytop):
        x = m + (t - tlo) / (thi - tlo) * (W - 2 * m)
        y = ytop + m + (vhi - v) / (vhi - vlo) * (H / 2 - 2 * m)
        return x, y
    t_vals = [p['t_Bohr'] for p in pts]
    e_vals = [p['dE_vs_055centre'] for p in pts]
    a_vals = [p['a_Eh_Bohr'] for p in pts]
    tlo, thi = min(t_vals), max(t_vals)
    elo, ehi = series(e_vals)
    alo, ahi = series([min(min(a_vals), 0.0), max(max(a_vals), 0.0)])
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" '
             'height="%d">' % (W, H),
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="%d" y="28" font-size="15">JOB-056 refinement of '
             'the 0.04-0.06 Bohr sign-change interval (055 line)</text>'
             % m]
    for row, (vals, label, ytop, zero) in enumerate(
            ((e_vals, 'dE vs 055 centre (Eh)', m, None),
             (a_vals, 'a(t) = g.q (Eh/Bohr)', H / 2, 0.0))):
        vlo, vhi = series([min(vals + ([zero] if zero is not None else [])),
                           max(vals + ([zero] if zero is not None else []))])
        poly = ' '.join('%0.1f,%0.1f'
                        % coords(t, v, tlo, thi, vlo, vhi, ytop)
                        for t, v in zip(t_vals, vals))
        parts.append('<polyline points="%s" fill="none" stroke="%s" '
                     'stroke-width="2"/>'
                     % (poly, '#1a56a0' if row == 0 else '#a0261a'))
        if zero is not None:
            xa, ya = coords(tlo, 0.0, tlo, thi, vlo, vhi, ytop)
            xb, yb = coords(thi, 0.0, tlo, thi, vlo, vhi, ytop)
            parts.append('<line x1="%0.1f" y1="%0.1f" x2="%0.1f" y2="%0.1f" '
                         'stroke="#999" stroke-dasharray="4,3"/>'
                         % (xa, ya, xb, yb))
        for p, v in zip(pts, vals):
            x, y = coords(p['t_Bohr'], v, tlo, thi, vlo, vhi, ytop)
            new = p['source'].startswith('new')
            parts.append('<circle cx="%0.1f" cy="%0.1f" r="%d" fill="%s" '
                         '%s/>' % (x, y, 5 if new else 3,
                                   '#1a56a0' if row == 0 else '#a0261a',
                                   'stroke="black"' if new else ''))
        parts.append('<text x="%d" y="%0.0f" font-size="13">%s</text>'
                     % (m, ytop + m - 20, label))
        for t in t_vals:
            x, _ = coords(t, vlo, tlo, thi, vlo, vhi, ytop)
            parts.append('<text x="%0.1f" y="%0.0f" font-size="10" '
                         'text-anchor="middle">%.3f</text>'
                         % (x, ytop + H / 2 - m + 18, t))
    parts.append('</svg>')
    with open(path, 'w') as fh:
        fh.write('\n'.join(parts))


def main():
    man = json.load(open(MANIFEST))
    for key, p in (('centre', man['reused_055_records']['t040']['path']),):
        pass
    if not man['all_checks_pass']:
        raise RuntimeError('HARD STOP: prechecks failed')
    # hash verification of the REUSED 055 records (read-only)
    for key in ('t040', 't060'):
        rec = man['reused_055_records'][key]
        if sha256_file(rec['path']) != rec['sha256']:
            raise RuntimeError('HARD STOP: 055 record %s hash changed'
                               % key)
    print('[056] reused 055 records verified (hashes OK)', flush=True)

    results = dict(
        job='JOB-2026-0906-056 refinement of the 0.04-0.06 Bohr '
            'sign-change interval (three new points on the 055 line)',
        doing='three new SCF+gradient evaluations at t=0.045/0.050/0.055 '
              'Bohr on the 055 line (centre reused, no centre SCF), then a '
              'merged line analysis',
        caps=man['caps'])
    led = Ledger(LEDGER)
    status = 'aborted'
    try:
        flow = run_refine(man, led, _eval_point, OUT)
        results['new_points'] = flow['records']
        an = flow['analysis']
        results['analysis'] = an
        for p in an['points']:
            print('[056] t=%.3f (%s): dE=%+.4e gmax=%.3e a=%+.4e'
                  % (p['t_Bohr'], p['source'], p['dE_vs_055centre'],
                     p['grad_max'], p['a_Eh_Bohr']), flush=True)
        make_svg(an['points'], OUT + '/refine056_plot.svg')
        status = 'completed'
        results['final'] = dict(status=status,
                                registration=an['registration'],
                                lowest_sample_point=an['lowest_sample_point'],
                                budget=led.data,
                                limits=an['limits'])
        ex.save_json_atomic(RESULTS, results)
        print('[056] DONE: %s' % an['registration'], flush=True)
    except Exception as exc:                                # noqa: BLE001
        results['abort'] = dict(error=str(exc),
                                traceback=traceback.format_exc()[-1500:])
        results.setdefault('final', dict(status='aborted',
                                         budget=led.data))
        ex.save_json_atomic(RESULTS, results)
        print('[056] ABORT: %s' % exc, flush=True)
        raise


if __name__ == '__main__':
    main()
