#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-055: four-point energy-gradient scan along the fixed q
from the 054 opt_25 endpoint.

Budget (no borrowing): centre reproduction 1, scan 4, total 5.  Everything
else = 0.  ON ANY EXCEPTION OR NON-CONVERGENCE: save and STOP - no retry,
no auto extra points, no offline repair continuation.

The centre is an unconverged scan start: max|g| > 1e-5 is NOT a gate
failure.  The new line is centred at R* (054 opt_25) and is NOT merged
with the 053 R0+tq path.

Analysis handover returns a SINGLE DICT (053 lesson: no tuple/dict
handover errors).
"""
import os, sys, json, time, traceback
import numpy as np

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
HERE = ROOT + '/protocols'
sys.path.insert(0, HERE)
import job047_exec as ex

OUT = ROOT + '/run_artifacts/02_nh3o3_reference/c1_dir_scan055'
LEDGER = OUT + '/budget_scan055.json'
RESULTS = OUT + '/scan055_results.json'
MANIFEST = OUT + '/input_manifest.json'

BPA = 1.0 / ex.ANG_PER_BOHR
CAPS = dict(centre=1, scan=4)


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
                             note='failures count within their category; no '
                                  'borrowing; on exception or non-convergence: '
                                  'save and STOP, no retry, no auto extra '
                                  'points',
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
    """Production evaluation via the validated 047 RealBackend chain
    (lazy pyscf import).  Returns (record, unit_readback_maxdev_Bohr)."""
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
    return 'centre' if t == 0.0 else 'scan_t%03d' % int(round(t * 1000))


def run_scan(man, led, eval_fn, out_dir, centre_gate_fn=None):
    """Actual execution entry: centre reproduction (gated) then the four
    scan points in fixed order, then the analysis handover.

    Returns a SINGLE DICT:
      {centre_record, records, analysis}   (analysis itself is a dict)
    """
    q = np.asarray(man['direction']['q'], float).reshape(-1)
    R0 = np.asarray(man['centre']['x_bohr'], float).reshape(7, 3)
    att = led.pre('centre', dict(tag='centre'))
    try:
        centre_rec, rb = eval_fn(R0, out_dir, tag_for(0.0))
        if rb > 0.0:
            raise RuntimeError('unit readback mismatch %e' % rb)
        if centre_gate_fn is not None:
            gate = centre_gate_fn(centre_rec)
            centre_rec['gate'] = gate
            ex.save_json_atomic(os.path.join(
                out_dir, 'eval_%s.json' % tag_for(0.0)), centre_rec)
            if not gate['gates_pass']:
                raise RuntimeError('HARD STOP: centre gate failed: %s'
                                   % gate)
        led.post(att, dict(e_total=centre_rec['e_total'],
                           grad_max=centre_rec['grad_max'],
                           seconds=centre_rec.get('seconds')))
        print('[055] centre: E=%.9f gmax=%.3e (gate %s)'
              % (centre_rec['e_total'], centre_rec['grad_max'],
                 'PASS' if (centre_gate_fn is None
                            or centre_rec['gate']['gates_pass'])
                 else 'FAIL'), flush=True)
    except Exception as e:                                  # noqa: BLE001
        led.fail(att, e)
        raise
    recs = []
    for g in man['scan_geometries']:
        t = g['t_Bohr']
        R = np.asarray(g['coords_bohr'], float).reshape(7, 3)
        att = led.pre('scan', dict(tag=tag_for(t), t_Bohr=t))
        try:
            rec, rb = eval_fn(R, out_dir, tag_for(t))
            if rb > 0.0:
                raise RuntimeError('unit readback mismatch %e' % rb)
            rec['t_Bohr'] = t
            led.post(att, dict(e_total=rec['e_total'],
                               grad_max=rec['grad_max'],
                               seconds=rec.get('seconds')))
            recs.append(rec)
            print('[055] scan %s (t=%.2f): E=%.9f gmax=%.3e'
                  % (tag_for(t), t, rec['e_total'], rec['grad_max']),
                  flush=True)
        except Exception as e:                              # noqa: BLE001
            led.fail(att, e)
            raise
    return dict(centre_record=centre_rec, records=recs,
                analysis=analyse(centre_rec, recs, q))


def analyse(centre_rec, recs, q):
    """Offline analysis handover (single dict)."""
    E0 = float(centre_rec['e_total'])
    g0 = np.asarray(centre_rec['grad'], float).reshape(-1)
    a0 = float(g0 @ q)
    pts = [dict(t_Bohr=0.0, tag=centre_rec.get('tag', 'centre'),
                dE_vs_centre=0.0,
                grad_max=float(centre_rec['grad_max']), a_Eh_Bohr=a0,
                kind='analytic_directional_derivative')]
    for r in recs:
        g = np.asarray(r['grad'], float).reshape(-1)
        pts.append(dict(t_Bohr=float(r['t_Bohr']), tag=r.get('tag'),
                        dE_vs_centre=float(r['e_total']) - E0,
                        grad_max=float(r['grad_max']),
                        a_Eh_Bohr=float(g @ q),
                        kind='analytic_directional_derivative'))
    # adjacent-point secant slopes (NOT derivatives of the endpoints)
    secants = []
    for i in range(1, len(pts)):
        t0, t1 = pts[i - 1]['t_Bohr'], pts[i]['t_Bohr']
        secants.append(dict(t_from=t0, t_to=t1,
                            slope_Eh_Bohr=(pts[i]['dE_vs_centre']
                                           - pts[i - 1]['dE_vs_centre'])
                            / (t1 - t0),
                            note='adjacent-point energy secant slope; not '
                                 'the derivative of either endpoint'))
    # diagnostics
    a_vals = [p['a_Eh_Bohr'] for p in pts]
    t_vals = [p['t_Bohr'] for p in pts]
    sign_changes = [(t_vals[i - 1], t_vals[i]) for i in range(1, len(a_vals))
                    if (a_vals[i - 1] < 0 <= a_vals[i])
                    or (a_vals[i - 1] > 0 >= a_vals[i])]
    e_min_i = int(np.argmin([p['dE_vs_centre'] for p in pts]))
    min_location = ('centre' if e_min_i == 0 else
                    ('interior' if 0 < e_min_i < len(pts) - 1
                     else 'boundary'))
    if sign_changes:
        registration = ('a sign-change interval on this fixed line pending '
                        'refinement (interval(s) %s)'
                        % ['%.2f-%.2f Bohr' % sc for sc in sign_changes])
    else:
        registration = ('no sign change of a(t) on the sampled interval '
                        '[0, %.2f Bohr]; lowest sampled energy at t=%.2f '
                        '(location: %s); reported as-is, no extrapolation'
                        % (t_vals[-1], pts[e_min_i]['t_Bohr'], min_location))
    return dict(points=pts, secant_slopes=secants, a0_Eh_Bohr=a0,
                sign_changes=sign_changes,
                lowest_sample_point=dict(index=e_min_i,
                                         t_Bohr=pts[e_min_i]['t_Bohr'],
                                         location=min_location,
                                         dE_vs_centre=pts[e_min_i][
                                             'dE_vs_centre']),
                lowest_interior=bool(min_location == 'interior'),
                energy_derivative_consistency='per-point a(t) signs vs '
                                              'secant slopes listed '
                                              'separately above',
                registration=registration,
                not_claimed='full-DOF minimum / curvature removal / '
                            'frequency or minimum prediction from an old '
                            'Hessian / basin statements')


def make_svg(pts, path):
    """Dependency-free SVG scan plot: dE vs t (top) and a(t) vs t (bottom,
    with a zero line)."""
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
    e_vals = [p['dE_vs_centre'] for p in pts]
    a_vals = [p['a_Eh_Bohr'] for p in pts]
    tlo, thi = min(t_vals), max(t_vals)
    elo, ehi = series(e_vals)
    alo, ahi = series([min(min(a_vals), 0.0), max(max(a_vals), 0.0)])
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" '
             'height="%d">' % (W, H),
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="%d" y="28" font-size="15">JOB-055 fixed-direction '
             'scan from the 054 opt_25 endpoint (R(t)=R*+t q)</text>' % m]
    for row, (vals, label, ytop, zero) in enumerate(
            ((e_vals, 'dE vs centre (Eh)', m, None),
             (a_vals, 'a(t) = g(R(t)).q (Eh/Bohr)', H / 2, 0.0))):
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
        for t, v in zip(t_vals, vals):
            x, y = coords(t, v, tlo, thi, vlo, vhi, ytop)
            parts.append('<circle cx="%0.1f" cy="%0.1f" r="4" fill="%s"/>'
                         % (x, y, '#1a56a0' if row == 0 else '#a0261a'))
            parts.append('<text x="%0.1f" y="%0.1f" font-size="10" '
                         'text-anchor="middle">%+.2e</text>'
                         % (x, y - 8, v))
        parts.append('<text x="%d" y="%0.0f" font-size="13">%s</text>'
                     % (m, ytop + m - 20, label))
        for t in t_vals:
            x, _ = coords(t, vlo, tlo, thi, vlo, vhi, ytop)
            parts.append('<text x="%0.1f" y="%0.0f" font-size="11" '
                         'text-anchor="middle">t=%.2f</text>'
                         % (x, ytop + H / 2 - m + 18, t))
    parts.append('</svg>')
    with open(path, 'w') as fh:
        fh.write('\n'.join(parts))


def main():
    man = json.load(open(MANIFEST))
    if sha256_file(man['centre']['path_wsl']) \
            != man['centre']['sha256']:
        raise RuntimeError('HARD STOP: 054 opt_25 source hash changed')
    if not man['all_checks_pass']:
        raise RuntimeError('HARD STOP: prechecks failed')
    print('[055] source OK: %s (sha256 %.16s...)'
          % (os.path.basename(man['centre']['path_wsl']),
             man['centre']['sha256']), flush=True)

    results = dict(
        job='JOB-2026-0906-055 four-point energy-gradient scan along the '
            'fixed q from the 054 opt_25 endpoint',
        doing='centre reproduction + R(t)=R*+t q (t=0.02..0.08 Bohr) '
              'evaluations to test the sign change of the direction '
              'derivative and discrete energy low points on this NEW line '
              '(not an extension of the 053 path)',
        centre_ref=dict(e_total=man['centre']['e_total'],
                        grad_max=man['centre']['grad_max']))
    led = Ledger(LEDGER)
    status = 'aborted'
    try:
        centre_gate_fn = ex.centre_gate_fn(man['centre'])
        flow = run_scan(man, led, _eval_point, OUT,
                        centre_gate_fn=centre_gate_fn)
        results['centre'] = dict(record=flow['centre_record'],
                                 gate=flow['centre_record'].get('gate'))
        results['scan_points'] = flow['records']
        an = flow['analysis']
        results['analysis'] = an
        for p in an['points']:
            print('[055] t=%.2f: dE=%+.4e gmax=%.3e a=%+.4e'
                  % (p['t_Bohr'], p['dE_vs_centre'], p['grad_max'],
                     p['a_Eh_Bohr']), flush=True)
        make_svg(an['points'], OUT + '/scan055_plot.svg')
        status = 'completed'
        results['final'] = dict(status=status,
                                registration=an['registration'],
                                budget=led.data,
                                not_claimed=an['not_claimed'])
        ex.save_json_atomic(RESULTS, results)
        print('[055] DONE: %s' % an['registration'], flush=True)
    except Exception as exc:                                # noqa: BLE001
        results['abort'] = dict(error=str(exc),
                                traceback=traceback.format_exc()[-1500:])
        results.setdefault('final', dict(status='aborted',
                                         budget=led.data))
        ex.save_json_atomic(RESULTS, results)
        print('[055] ABORT: %s' % exc, flush=True)
        raise


if __name__ == '__main__':
    main()
