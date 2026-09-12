#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-059: directional two-step FD check of the NEW 058
negative mode at the 057 candidate geometry (+ the 058 count corrections
landed by job059_prep.py at zero evaluations).

Budget (hard): total 5 SCF+full-gradient attempts = centre 1 + four
displacements 1 each.  Failures count; no borrowing between categories.
Gate: centre must reproduce the ACTUAL 058 resume-01 centre record AND
the 057 designated recheck record (dE<=1e-8, dgrad<=1e-7 per-component
max, dC<=1e-9 A, config identical, converged, finite) before any
displacement runs.  Each evaluation is saved atomically BEFORE its
attempt is marked done.  ON ANY REAL EXECUTION EXCEPTION: save and STOP
the whole batch - no fix-and-restart, no new launch probes, no separate
recovery budget.  chkfile paths are PER-TAG (058 overwrite defect fixed).

Analysis per step h in {0.001, 0.002} (053 conventions):
  a0    = g0 . q                       (non-zero centre gradient KEPT)
  slope = (E(+h) - E(-h)) / (2h)
  kE    = [E(+h) + E(-h) - 2 E0] / h^2
  kg    = [g(+h) - g(-h)] . q / (2h)
  kH    = q^T H q from the SAVED 058 matrices (no recomputation)
"""
import os, sys, json, time, traceback
import numpy as np

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
HERE = ROOT + '/protocols'
sys.path.insert(0, HERE)
import job047_exec as ex
import job058_exec as j58          # Ledger pattern (error -> HARD STOP)
import job052_exec as j52          # endpoint_eval (production eval path)

R58R = ROOT + '/run_artifacts/02_nh3o3_reference/c1_fdhess058_resume01'
OUT = ROOT + '/run_artifacts/02_nh3o3_reference/c1_negdir059'
LEDGER = OUT + '/budget_negdir059.json'
RESULTS = OUT + '/negdir059_results.json'
MANIFEST = OUT + '/input_manifest.json'
SYMS = ['N', 'H', 'H', 'H', 'O', 'O', 'O']


class Ledger(j58.Ledger):
    CAPS = dict(centre=1, displacement=4)

    def __init__(self, path):
        j58.Ledger.__init__(self, path)
        self.data['caps'] = dict(self.CAPS, total=5)
        self.data['note'] = ('JOB-059 hard budget: centre 1 + four '
                             'displacements 1 each = 5 SCF+full-gradient '
                             'attempts total; failures count; no '
                             'borrowing; any real execution exception -> '
                             'save and STOP the whole batch (no '
                             'fix-and-restart, no launch probes, no '
                             'separate recovery budget)')
        self.data['context'] = dict(
            prior_058_counts='at least 47 attempts (see '
                             'job058_count_correction.json in the 058 '
                             'resume dir); this ledger adds at most 5')
        self._save()


def production_eval(R, out_dir, tag):
    """Same validated path as 058 resume (052 endpoint_eval on a fresh
    mol, config identical to 058); chkfile PER-TAG so nothing is
    overwritten (058 defect fixed)."""
    from pyscf import gto
    Rm = np.asarray(R, float).reshape(7, 3)
    mol = gto.M(atom=[(s, (float(r[0]), float(r[1]), float(r[2])))
                      for s, r in zip(SYMS, Rm)],
                basis='def2-TZVP', charge=0, spin=0, verbose=0,
                max_memory=4000, unit='Bohr')
    chk = os.path.join(out_dir, 'chk_%s.chk' % tag)
    orig_make = j52.d2_full.make_mf_d2

    def make_with_chk(m, solvent=None):
        _mf = orig_make(m, solvent=solvent)
        _mf.chkfile = chk
        return _mf

    j52.d2_full.make_mf_d2 = make_with_chk
    try:
        rec, mf, _ = j52.endpoint_eval(mol, out_dir, tag)
    finally:
        j52.d2_full.make_mf_d2 = orig_make
    rb = float(np.abs(np.asarray(mol.atom_coords(unit='Bohr'), float)
                      .reshape(-1) - Rm.reshape(-1)).max())
    return rec, rb, mf


def run_batch(man, led, eval_fn, out_dir):
    """Centre (gated against BOTH references) then the four preregistered
    displacements.  Returns a SINGLE DICT."""
    R0 = np.asarray(man['centre']['x_bohr'], float).reshape(7, 3)
    q = np.asarray(man['direction']['q'], float).reshape(-1)

    att = led.pre('centre', dict(tag='centre'))
    try:
        rec, rb, mf = eval_fn(R0, out_dir, 'centre')
        if rb > 0.0:
            raise RuntimeError('unit readback mismatch %e' % rb)
        if not (rec.get('converged') and rec.get('all_finite')):
            raise RuntimeError('centre SCF not converged or non-finite '
                               '-> HARD STOP')
        led.post(att, dict(e_total=rec['e_total'],
                           grad_max=rec['grad_max'],
                           seconds=rec.get('seconds')))
    except Exception as e:                                  # noqa: BLE001
        led.fail(att, e)
        raise

    def gate_vs(ref_e, ref_grad, ref_coords, ref_cfg, label):
        dE = abs(rec['e_total'] - float(ref_e))
        dg = float(np.abs(np.asarray(rec['grad'], float).reshape(-1)
                          - np.asarray(ref_grad, float).reshape(-1)).max())
        dC = float(np.abs(np.asarray(rec['coords_actual_angstrom'], float)
                          - np.asarray(ref_coords, float)).max())
        g = dict(dE=dE, dgrad_max=dg, dcoords_A=dC,
                 gmax=float(rec['grad_max']),
                 gmax_le_1e5=bool(rec['grad_max'] <= 1e-5),
                 converged=bool(rec['converged']),
                 finite=bool(rec['all_finite']),
                 config_match=bool(rec['config'] == ref_cfg))
        g['gates_pass'] = bool(dE <= 1e-8 and dg <= 1e-7 and dC <= 1e-9
                               and g['gmax_le_1e5'] and g['converged']
                               and g['finite'] and g['config_match'])
        g['reference'] = label
        return g

    gate58 = gate_vs(man['centre']['e_total'], man['centre']['grad'],
                     man['centre']['coords_actual_angstrom'],
                     man['centre']['config'],
                     '058 resume-01 actual centre')
    cc = man['centre']['cross_check_ref']
    gate57 = gate_vs(cc['e_total'], cc['grad'],
                     cc['coords_actual_angstrom'],
                     man['centre']['config'],
                     '057 designated recheck (cross-verification)')
    gate = dict(vs_058_resume_centre=gate58,
                vs_057_recheck=gate57,
                gates_pass=bool(gate58['gates_pass']
                                and gate57['gates_pass']))
    print('[059] centre gate 058resume %s: dE=%.2e dgrad=%.2e dC=%.2e '
          'gmax=%.3e' % ('PASS' if gate58['gates_pass'] else 'FAIL',
                         gate58['dE'], gate58['dgrad_max'],
                         gate58['dcoords_A'], gate58['gmax']), flush=True)
    print('[059] centre gate 057recheck %s: dE=%.2e dgrad=%.2e dC=%.2e'
          % ('PASS' if gate57['gates_pass'] else 'FAIL', gate57['dE'],
             gate57['dgrad_max'], gate57['dcoords_A']), flush=True)
    if not gate['gates_pass']:
        raise RuntimeError('HARD STOP: centre gate failed')

    disp_recs = {}
    for g in man['displacement_geometries']:
        tag = g['tag_full']
        R = np.asarray(g['coords_bohr'], float).reshape(7, 3)
        att = led.pre('displacement', dict(tag=tag, t_Bohr=g['t_Bohr']))
        try:
            drec, rb, _ = eval_fn(R, out_dir, tag)
            if rb > 0.0:
                raise RuntimeError('unit readback mismatch %e' % rb)
            if not (drec.get('converged') and drec.get('all_finite')):
                raise RuntimeError('displacement %s SCF not converged or '
                                   'non-finite -> HARD STOP' % tag)
            drec['t_Bohr'] = g['t_Bohr']
            led.post(att, dict(e_total=drec['e_total'],
                               grad_max=drec['grad_max'],
                               seconds=drec.get('seconds')))
            disp_recs[tag] = drec
            print('[059] disp %s: E=%.12f gmax=%.3e'
                  % (tag, drec['e_total'], drec['grad_max']), flush=True)
        except Exception as e:                              # noqa: BLE001
            led.fail(att, e)
            raise
    return dict(centre_record=rec, gate=gate, disp_records=disp_recs)


def analyse(man, centre_rec, disp_recs, H_sym, H_raw):
    """Per-step a0 / slope / kE / kg + kH from the SAVED 058 matrices.
    Non-zero centre gradient is kept; single-side lowering is not a
    curvature-sign criterion; near-resolution energy differences are
    flagged, no auto extra points."""
    q = np.asarray(man['direction']['q'], float).reshape(-1)
    g0 = np.asarray(centre_rec['grad'], float).reshape(-1)
    E0 = float(centre_rec['e_total'])
    a0 = float(g0 @ q)
    kH_sym = float(q @ np.asarray(H_sym, float) @ q)
    kH_raw = float(q @ np.asarray(H_raw, float) @ q)
    lam0 = float(man['direction']['mode0']['eigenvalue'])
    d = np.asarray(man['direction']['d_raw'], float).reshape(-1)
    d_norm_sq = float(d @ d)
    per_h, result = {}, dict(
        E0=E0, gmax0=float(centre_rec['grad_max']), a0=a0,
        a0_note='non-zero centre gradient along q (the candidate is a '
                'threshold-stopped point, not an exact stationary '
                'point); kept in all comparisons',
        kH=dict(qTHq_H_sym=kH_sym, qTHq_H_raw=kH_raw, unit='Eh/Bohr^2',
                source='SAVED 058 h_fd_sym.npy / h_fd_raw.npy (no '
                       'recomputation)',
                mode0_eigenvalue_lam0=lam0, d_norm_sq=d_norm_sq,
                consistency_note='lam0 = d^T H d and kH_sym = q^T H q = '
                                 'lam0/d_norm_sq by construction; both '
                                 'reported'),
        scf_resolution_note='SCF conv_tol 1e-12 / conv_tol_grad 1e-9; '
                            'energy differences approaching ~1e-11 Eh '
                            'are near the numerical resolution and are '
                            'flagged, not auto-extended')
    for h in (0.001, 0.002):
        tag_p = 'disp_p%03d' % int(round(h * 1000))
        tag_m = 'disp_m%03d' % int(round(h * 1000))
        Ep = float(disp_recs[tag_p]['e_total'])
        Em = float(disp_recs[tag_m]['e_total'])
        gp = np.asarray(disp_recs[tag_p]['grad'], float).reshape(-1)
        gm = np.asarray(disp_recs[tag_m]['grad'], float).reshape(-1)
        dEp, dEm = Ep - E0, Em - E0
        kE = (Ep + Em - 2.0 * E0) / h ** 2
        kg = float((gp - gm) @ q) / (2.0 * h)
        slope = (Ep - Em) / (2.0 * h)
        per_h['%g' % h] = dict(
            h_Bohr=h, dE_plus=dEp, dE_minus=dEm,
            near_resolution=bool(min(abs(dEp), abs(dEm)) < 1e-10),
            energy_slope_Eh_Bohr=slope,
            kE_Eh_Bohr2=kE, kg_Eh_Bohr2=kg,
            kE_minus_kg=kE - kg,
            kE_minus_kH_sym=kE - kH_sym, kg_minus_kH_sym=kg - kH_sym,
            rel_diff_kE_kg=float(abs(kE - kg)
                                 / max(abs(kE), abs(kg), 1e-30)),
            rel_diff_kE_kH=float(abs(kE - kH_sym)
                                 / max(abs(kE), abs(kH_sym), 1e-30)),
            rel_diff_kg_kH=float(abs(kg - kH_sym)
                                 / max(abs(kg), abs(kH_sym), 1e-30)))
    k1, k2 = per_h['0.001'], per_h['0.002']
    result['per_h'] = per_h
    result['step_sensitivity'] = dict(
        kE_abs_diff=k1['kE_Eh_Bohr2'] - k2['kE_Eh_Bohr2'],
        kE_rel_diff=float(abs(k1['kE_Eh_Bohr2'] - k2['kE_Eh_Bohr2'])
                          / max(abs(k1['kE_Eh_Bohr2']), 1e-30)),
        kg_abs_diff=k1['kg_Eh_Bohr2'] - k2['kg_Eh_Bohr2'],
        kg_rel_diff=float(abs(k1['kg_Eh_Bohr2'] - k2['kg_Eh_Bohr2'])
                          / max(abs(k1['kg_Eh_Bohr2']), 1e-30)))
    signs = [k1['kE_Eh_Bohr2'] < 0, k1['kg_Eh_Bohr2'] < 0,
             k2['kE_Eh_Bohr2'] < 0, k2['kg_Eh_Bohr2'] < 0,
             kH_sym < 0]
    result['sign_summary'] = dict(
        kE_negative_both_steps=bool(k1['kE_Eh_Bohr2'] < 0
                                    and k2['kE_Eh_Bohr2'] < 0),
        kg_negative_both_steps=bool(k1['kg_Eh_Bohr2'] < 0
                                    and k2['kg_Eh_Bohr2'] < 0),
        kH_negative=bool(kH_sym < 0),
        all_routes_negative=bool(all(signs)))
    return result


def sha_file(path):
    h = __import__('hashlib').sha256()
    with open(path, 'rb') as fh:
        for blk in iter(lambda: fh.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def main():
    man = json.load(open(MANIFEST))
    if not man['all_checks_pass']:
        raise RuntimeError('HARD STOP: prechecks failed')
    if sha_file(man['centre']['path']) != man['centre']['sha256']:
        raise RuntimeError('HARD STOP: 058 resume centre record changed')
    if sha_file(man['centre']['cross_check_ref']['path']) \
            != man['centre']['cross_check_ref']['sha256']:
        raise RuntimeError('HARD STOP: 057 recheck record changed')
    H_sym = np.load(R58R + '/h_fd_sym.npy')
    H_raw = np.load(R58R + '/h_fd_raw.npy')
    print('[059] manifest OK; direction = 058 NEW FD mode 0 '
          '(q_sha %.16s...); 058 centre + 057 recheck records hash-'
          'verified' % man['direction']['q_sha'], flush=True)

    results = dict(
        job='JOB-2026-0906-059: 058 count corrections + directional '
            'two-step FD check of the NEW 058 negative mode',
        doing='centre reproduction (dual gate) + R0 +/- 0.001q and '
              'R0 +/- 0.002q (1 SCF+full gradient each, 5 total); '
              'a0/slope/kE/kg per step vs kH=q^T H q from the SAVED 058 '
              'matrices; no optimisation, no Hessian recomputation, no '
              'thermochemistry',
        review_058='notes/job058_commander_review_2026-09-10.md',
        count_correction='see job058_count_correction.json in the 058 '
                         'resume dir (>=47 attempts; compliance claims '
                         'withdrawn); this batch adds at most 5')
    led = Ledger(LEDGER)
    status = 'aborted'
    try:
        flow = run_batch(man, led, production_eval, OUT)
        results['centre'] = dict(record=flow['centre_record'],
                                 gate=flow['gate'])
        results['displacements'] = flow['disp_records']
        ex.save_json_atomic(RESULTS, results)
        ana = analyse(man, flow['centre_record'], flow['disp_records'],
                      H_sym, H_raw)
        results['analysis'] = ana
        status = 'completed'
        s = ana['sign_summary']
        reg = ('negative curvature along the NEW 058 mode: independent '
               'two-step FD support (kE and kg negative at both steps)'
               if s['all_routes_negative'] else
               'curvature sign along the NEW 058 mode: NOT consistently '
               'supported by independent FD (see per-route signs)')
        results['final'] = dict(
            status=status, registration=reg,
            sign_summary=s,
            magnitude=','.join(
                'h=%s: kE=%+.4e kg=%+.4e kH_sym=%+.4e'
                % (h, ana['per_h'][h]['kE_Eh_Bohr2'],
                   ana['per_h'][h]['kg_Eh_Bohr2'], ana['kH'][
                       'qTHq_H_sym'])
                for h in ('0.001', '0.002')),
            accounting=dict(
                new_attempts=dict(centre=led.count('centre'),
                                  displacement=led.count('displacement')),
                new_caps=dict(centre=1, displacement=4, total=5),
                new_failures=sum(1 for a in led.data['attempts']
                                 if a['status'] == 'error'),
                cumulative_058_corrected='>=47 attempts (44 completed '
                                         'SCF+grad, 2 stability calls, '
                                         'probe counted, uncertain '
                                         'costs listed separately) + '
                                         'this batch',
                not_claimed='minimum acceptance / frequency convergence '
                            '/ same soft coordinate or basin / '
                            'transition state'),
            limits=man['analysis_limits'])
        ex.save_json_atomic(RESULTS, results)
        for h in ('0.001', '0.002'):
            v = ana['per_h'][h]
            print('[059] h=%s: dE+=%+.3e dE-=%+.3e aE=%+.3e kE=%+.4e '
                  'kg=%+.4e (kH_sym=%+.4e)'
                  % (h, v['dE_plus'], v['dE_minus'], v['energy_slope_Eh_Bohr'],
                     v['kE_Eh_Bohr2'], v['kg_Eh_Bohr2'],
                     ana['kH']['qTHq_H_sym']), flush=True)
        print('[059] DONE: %s' % reg, flush=True)
    except Exception as exc:                                # noqa: BLE001
        results['abort'] = dict(error=str(exc),
                                traceback=traceback.format_exc()[-1500:])
        results.setdefault('final', dict(status='aborted',
                                         budget=led.data))
        ex.save_json_atomic(RESULTS, results)
        print('[059] ABORT: %s' % exc, flush=True)
        raise


if __name__ == '__main__':
    main()
