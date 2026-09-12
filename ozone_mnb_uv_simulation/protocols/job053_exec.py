#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-053: two-step independent FD check of the negative-mode
direction at the 052 candidate geometry.

Budget (no borrowing): centre 1, displacement 4, total 5.  Everything else
(optimization, stability, Hessian, frequency recompute, CP,
thermochemistry, high-level, other species) = 0.

Reuses the validated endpoint evaluation path (identical construction to
052's endpoint_eval) and the 045-style direction-difference design with
NEW inputs and fields.  ON ANY REAL EXECUTION EXCEPTION: save and STOP -
no self-resume (per the 052 commander review).

Formulas (this batch's centre E0, g0):
  a0    = g0 . q
  aE(h) = [E(+h) - E(-h)] / (2h)
  kE(h) = [E(+h) + E(-h) - 2 E0] / h^2
  kg(h) = [g(+h) - g(-h)] . q / (2h)
  kH    = q^T H q from the 052 saved matrices (DFT / D2 listed).
"""
import os, sys, json, hashlib, time, traceback
import numpy as np

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
HERE = ROOT + '/protocols'
sys.path.insert(0, HERE)
import job047_exec as ex
import job052_exec as j52         # validated endpoint_eval component

OUT = ROOT + '/run_artifacts/02_nh3o3_reference/c1_negmode_dir053'
LEDGER = OUT + '/budget_negmode053.json'
RESULTS = OUT + '/negmode_dir053_results.json'
MANIFEST = OUT + '/input_manifest.json'
SYMS = ['N', 'H', 'H', 'H', 'O', 'O', 'O']
BPA = 1.0 / ex.ANG_PER_BOHR


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for blk in iter(lambda: fh.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


class Ledger:
    CAPS = dict(centre=1, displacement=4)

    def __init__(self, path):
        self.path = path
        self.data = dict(caps=dict(self.CAPS),
                         note='failures count within their category; no '
                              'borrowing; on exception: save and STOP, no '
                              'self-resume',
                         attempts=[])
        self._save()

    def _save(self):
        ex.save_json_atomic(self.path, self.data)

    def count(self, cat):
        return sum(1 for a in self.data['attempts']
                   if a['category'] == cat)

    def pre(self, cat, note=None):
        if self.count(cat) >= self.CAPS[cat]:
            raise RuntimeError('cap reached for %s' % cat)
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


# ------------------------------------------------------------------ eval
def _build_mol(R_bohr):
    """Production mol factory: real PySCF mol from the full-precision
    float coordinates (no decimal formatting)."""
    from pyscf import gto
    R = np.asarray(R_bohr, float).reshape(7, 3)
    return gto.M(atom=[(s, (float(r[0]), float(r[1]), float(r[2])))
                       for s, r in zip(SYMS, R)],
                 basis='def2-TZVP', charge=0, spin=0, verbose=0,
                 max_memory=4000, unit='Bohr')


def _eval_point(R_bohr, out_dir, tag, mol_factory=_build_mol):
    """Production evaluation via the validated 052 endpoint_eval path
    (contract: returns (rec, mf, mol) and persists eval_<tag>.json
    internally).  Returns (record, unit_readback_maxdev_Bohr)."""
    R = np.asarray(R_bohr, float).reshape(7, 3)
    mol = mol_factory(R)
    readback = float(np.abs(np.asarray(mol.atom_coords(unit='Bohr'),
                                       float).reshape(-1)
                            - R.reshape(-1)).max())
    if readback > 0.0:
        raise RuntimeError('unit readback mismatch %e' % readback)
    rec = j52.endpoint_eval(mol, out_dir, tag)[0]
    rec['x_bohr'] = R.tolist()
    rec['unit_readback_maxdev_Bohr'] = readback
    ex.save_json_atomic(os.path.join(out_dir,
                                     'eval_%s.json' % tag), rec)
    return rec, readback


def tag_for(t):
    return ('disp_p%02d' % int(round(abs(t) * 100))) if t > 0 \
        else ('disp_m%02d' % int(round(abs(t) * 100)))


def run_displacements(geoms, out_dir, ledger, eval_fn=_eval_point):
    """Fixed order +/-0.01, +/-0.02 Bohr."""
    out = []
    for g in geoms:
        t = g['t_Bohr']
        tag = tag_for(t)
        R = np.asarray(g['coords_bohr'], float).reshape(7, 3)
        att = ledger.pre('displacement', dict(tag=tag, t_Bohr=t))
        try:
            rec, readback = eval_fn(R, out_dir, tag)
            if readback > 0.0:
                raise RuntimeError('unit readback mismatch %e' % readback)
            rec['t_Bohr'] = t
            ledger.post(att, dict(e_total=rec['e_total'],
                                  grad_max=rec['grad_max'],
                                  unit_readback_maxdev_Bohr=readback,
                                  seconds=rec.get('seconds')))
            out.append(rec)
            print('[053] disp %s (t=%+.2f): E=%.9f gmax=%.3e'
                  % (tag, t, rec['e_total'], rec['grad_max']), flush=True)
        except Exception as e:                              # noqa: BLE001
            ledger.fail(att, e)
            raise
    return out


def run_resume(man, led, eval_fn, out_dir, gate_pass=True):
    """Resume flow (resume-01): the ACCEPTED persisted centre record is
    reused with ZERO centre evaluations; only the four displacement
    evaluations run.  Raises before any displacement if the centre gate
    credential did not pass.  Returns a single dict."""
    if not gate_pass:
        raise RuntimeError('centre gate credential failed -> displacements '
                           'not entered')
    centre_rec = man['centre_record']
    recs = run_displacements(man['displacement_geometries'], out_dir, led,
                             eval_fn=eval_fn)
    rd = {tag_for(r['t_Bohr']): r for r in recs}
    q = np.asarray(man['direction']['q'], float).reshape(-1)
    comp = compute_comparisons(centre_rec, rd, q,
                               man['kH_from_052_matrices'])
    return dict(comparisons=comp, displacements=recs,
                centre_record=centre_rec)


def compute_comparisons(rec_c, recs, q, kh):
    E0 = float(rec_c['e_total'])
    g0 = np.asarray(rec_c['grad'], float).reshape(-1)
    a0 = float(g0 @ q)
    per_h = {}
    for h in (0.01, 0.02):
        p = recs[tag_for(h)]
        m = recs[tag_for(-h)]
        Ep, Em = float(p['e_total']), float(m['e_total'])
        gp = np.asarray(p['grad'], float).reshape(-1)
        gm = np.asarray(m['grad'], float).reshape(-1)
        per_h['%g' % h] = dict(
            h_Bohr=h,
            dE_plus=Ep - E0, dE_minus=Em - E0,
            aE_Eh_Bohr=(Ep - Em) / (2 * h),
            kE_Eh_Bohr2=(Ep + Em - 2 * E0) / h ** 2,
            kg_Eh_Bohr2=float((gp - gm) @ q / (2 * h)))
    for k in ('0.01', '0.02'):
        v = per_h[k]
        kh_tot = kh['1e-3' if k == '0.01' else '5e-4'][
            'kH_total_Eh_Bohr2']
        v['kE_minus_kH'] = v['kE_Eh_Bohr2'] - kh_tot
        v['kg_minus_kH'] = v['kg_Eh_Bohr2'] - kh_tot
    per_h['0.01']['kE_step_sensitivity'] = per_h['0.02']['kE_Eh_Bohr2'] \
        - per_h['0.01']['kE_Eh_Bohr2']
    per_h['0.01']['kg_step_sensitivity'] = per_h['0.02']['kg_Eh_Bohr2'] \
        - per_h['0.01']['kg_Eh_Bohr2']
    return dict(a0_Eh_Bohr=a0, per_h=per_h,
                note='nonzero a0 retained; per-step/per-estimator '
                     'differences listed separately (never merged)')


def main():
    man = json.load(open(MANIFEST))
    src_c = man['centre']
    if sha256_file(man['centre_path_wsl']) != man['centre_sha256']:
        raise RuntimeError('HARD STOP: 052 endpoint source hash changed')
    q = np.asarray(man['direction']['q'], float).reshape(-1)
    R0 = np.asarray(man['R0_bohr'], float).reshape(7, 3)
    print('[053] source OK: %s (sha256 %.16s...)'
          % (os.path.basename(man['centre_path_wsl']),
             man['centre_sha256']), flush=True)

    results = dict(
        job='JOB-2026-0906-053 two-step independent FD check of the '
            'negative-mode direction at the 052 candidate geometry',
        doing='centre reproduction + +/-0.01/+/-0.02 Bohr displacements '
              'along the 052 mode-0 direction (actual energies and full '
              'gradients) to test the sign and magnitude of the predicted '
              'negative curvature',
        centre_ref=dict(e_total=src_c['e_total'],
                        grad_max=src_c['grad_max']))
    led = Ledger(LEDGER)
    status = 'aborted'
    try:
        # ---------------- centre reproduction ----------------
        att = led.pre('centre', dict(tag='centre'))
        try:
            rec_c, rb = _eval_point(R0, OUT, 'centre')
        except Exception as e:                          # noqa: BLE001
            led.fail(att, e)
            raise
        dE = abs(rec_c['e_total'] - float(src_c['e_total']))
        dg = float(np.abs(np.asarray(rec_c['grad'], float).reshape(-1)
                          - np.asarray(src_c['grad'], float)
                          .reshape(-1)).max())
        dC = float(np.abs(np.asarray(rec_c['coords_actual_angstrom'],
                                     float)
                          - np.asarray(src_c['coords_actual_angstrom'],
                                       float)).max())
        gate = dict(dE=dE, dgrad_max=dg, dcoords_A=dC,
                    gmax=float(rec_c['grad_max']),
                    gmax_le_1e5=bool(rec_c['grad_max'] <= 1e-5),
                    converged=bool(rec_c['converged']),
                    finite=bool(rec_c['all_finite']),
                    config_match=bool(rec_c['config'] == src_c['config']))
        gate['gates_pass'] = bool(
            dE <= 1e-8 and dg <= 1e-7 and dC <= 1e-9
            and rec_c['grad_max'] <= 1e-5 and rec_c['converged']
            and rec_c['all_finite'] and gate['config_match'])
        led.post(att, dict(e_total=rec_c['e_total'],
                           grad_max=rec_c['grad_max'], gate=gate,
                           seconds=rec_c['seconds']))
        results['centre'] = dict(record=rec_c, gate=gate)
        print('[053] centre gate %s: dE=%.2e dgrad=%.2e dC=%.2e gmax=%.3e'
              % ('PASS' if gate['gates_pass'] else 'FAIL', dE, dg, dC,
                 rec_c['grad_max']), flush=True)
        ex.save_json_atomic(RESULTS, results)
        if not gate['gates_pass']:
            results['final'] = dict(status='centre_gate_failed',
                                    budget=led.data)
            ex.save_json_atomic(RESULTS, results)
            raise RuntimeError('HARD STOP: centre gate failed')

        # ---------------- four displacements (fixed order) --------------
        recs = {}
        for rec in run_displacements(man['displacement_geometries'], OUT,
                                     led):
            recs[tag_for(rec['t_Bohr'])] = rec
        results['displacements'] = {
            k: {kk: vv for kk, vv in r.items()} for k, r in recs.items()}

        # ---------------- comparisons -----------------------------------
        kh = man['kH_from_052_matrices']
        comp = compute_comparisons(rec_c, recs, q, kh)
        results['comparisons'] = comp
        results['kH_from_052'] = kh
        for k in ('0.01', '0.02'):
            v = comp['per_h'][k]
            print('[053] h=%s: dE+=%+.3e dE-=%+.3e aE=%+.3e kE=%+.4e '
                  'kg=%+.4e  (kE-kH=%+.3e, kg-kH=%+.3e)'
                  % (k, v['dE_plus'], v['dE_minus'], v['aE_Eh_Bohr'],
                     v['kE_Eh_Bohr2'], v['kg_Eh_Bohr2'],
                     v['kE_minus_kH'], v['kg_minus_kH']), flush=True)

        # ---------------- registration ----------------------------------
        kE_neg = all(comp['per_h'][k]['kE_Eh_Bohr2'] < 0
                     for k in ('0.01', '0.02'))
        kg_neg = all(comp['per_h'][k]['kg_Eh_Bohr2'] < 0
                     for k in ('0.01', '0.02'))
        if kE_neg and kg_neg:
            verdict = ('the local negative curvature along this direction '
                       'is supported by independent energy AND gradient '
                       'finite differences at both steps; magnitude '
                       'consistency with kH is listed separately')
            reg = 'direction local negative curvature: independent FD support'
        else:
            verdict = ('energy and gradient FD do NOT consistently support '
                       'the negative sign -> registered as PENDING; not '
                       'attributed to noise, no extra points, no mode '
                       'following')
            reg = 'direction curvature sign: pending'
        results['registration'] = dict(verdict=verdict, registered=reg,
                                       kE_negative_both_steps=kE_neg,
                                       kg_negative_both_steps=kg_neg)
        status = 'completed'
        results['final'] = dict(
            status=status, registration=reg,
            budget=led.data,
            not_claimed='verified vibrational frequency / transition '
                        'state / minimum acceptance / whole-Hessian '
                        'correctness / reaction mechanism')
        ex.save_json_atomic(RESULTS, results)
        print('[053] DONE: %s' % reg, flush=True)
    except Exception as exc:                                # noqa: BLE001
        results['abort'] = dict(error=str(exc),
                                traceback=traceback.format_exc()[-1500:])
        results.setdefault('final', dict(status='aborted',
                                         budget=led.data))
        ex.save_json_atomic(RESULTS, results)
        print('[053] ABORT: %s' % exc, flush=True)
        raise


if __name__ == '__main__':
    main()
