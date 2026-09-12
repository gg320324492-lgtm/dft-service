#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unit-consistency audit, REVISED (JOB-2026-0905-009 / JOB-2026-0906-002).

Changes vs the first version (Phase A, batch 002):
  * recovered q keeps FULL precision (rounding only for display);
  * data are matched by (candidate, grid_level, amplitude, sign) -- never by
    record order; the central point is matched by candidate AND grid level;
    no silent fallback to a central of another grid level;
  * the undefined lowercase `b` is removed (B is used consistently);
  * old reported values are READ and compared explicitly; a missing old value
    is reported as missing/not_checked and is never counted as a pass;
  * SCF convergence is read from the recorded field; a missing field is
    treated as NOT converged (never assumed);
  * central-point count, displaced-point count and verification coverage are
    reported explicitly.

Correction direction (do not reverse): corrected = old * B^2 for k_E,
corrected = old * B for k_g and s_E.
"""
import os
import sys
import json
import hashlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import curvature_audit_lib as cal

ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, 'run_artifacts', '01_pure_water_o3_h2o')
FV = os.path.join(ART, 'final_minima_validation')
IS = os.path.join(ART, 'internal_subspace_revision')
DC = os.path.join(ART, 'directional_confirmation')
GR = os.path.join(ART, 'grid_response_diagnostic')
RAW = os.path.join(ART, 'unit_consistency_revision')
os.makedirs(RAW, exist_ok=True)

B = 0.52917721092
MASSES = np.array([15.999, 15.999, 15.999, 15.999, 1.008, 1.008])
TARGETS = ['c14_plus', 'c06_plus']
AMPS_A = [0.002, 0.005, 0.01]
CONSTRUCTIONS = {'c14_plus': 'analytic', 'c06_plus': 'fd'}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:16]


def load(p, default=None):
    if not os.path.exists(p):
        return default
    with open(p) as fh:
        return json.load(fh)


def converged_ok(rec):
    """SCF convergence read from the recorded field; missing field => False."""
    return bool(rec.get('scf_converged', False))


def _old_value(old_summary, tid, gl, amp, field):
    """Read the pre-fix reported value for (candidate, grid, amplitude).

    Two on-disk shapes are supported:
      * directional_summary.json: candidates.<tid>.tables.<grid>.rows[]
      * grid_response_summary.json: candidates.<tid>.verdict.<field>[i]
        (list ordered by the nominal amplitudes)
    Anything not found is reported as missing (never assumed)."""
    if old_summary is None:
        return None
    cand = (old_summary.get('candidates') or {}).get(tid) or {}
    tables = cand.get('tables') or {}
    rows = (tables.get(str(gl)) or tables.get(gl) or {}).get('rows') or []
    for row in rows:
        if abs(row.get('amp_a', -1) - amp) < 1e-12 and field in row:
            return row[field]
    # second shape: verdict.<field>[i]
    verdict = cand.get('verdict') or {}
    lst = verdict.get(field)
    if isinstance(lst, list) and amp in AMPS_A:
        i = AMPS_A.index(amp)
        if i < len(lst):
            return lst[i]
    # third shape (grid_response summary): q2_explains.<tid>.<field>[i]
    q2 = (old_summary.get('q2_explains') or {}).get(tid) or {}
    lst = q2.get(field)
    if isinstance(lst, list) and amp in AMPS_A:
        i = AMPS_A.index(amp)
        if i < len(lst):
            return lst[i]
    return None


def audit_batch(records, batch_name, old_summary):
    isx = load(os.path.join(IS, 'internal_subspace_analysis.json'), {})
    out = {}
    for tid in TARGETS:
        construction = CONSTRUCTIONS[tid]
        d = np.asarray(isx['candidate_reanalysis'][tid][construction]
                       ['lowest_internal_direction_cart'], float)
        d3 = d.reshape(6, 3)
        m_max = float(np.linalg.norm(d3, axis=1).max())
        lam_mw = float(isx['candidate_reanalysis'][tid][construction]
                       ['lowest_internal_eigenvalue'])
        nu_claim = float(np.sign(lam_mw) * np.sqrt(abs(lam_mw)) * 5139.5)

        cand = [r for r in records if r.get('candidate') == tid]
        centrals = [r for r in cand if r.get('kind') == 'central']
        disps = [r for r in cand if r.get('kind') != 'central']
        cen_by_grid = {r.get('grid_level'): r for r in centrals}

        points = []
        for r in disps:
            gl = r.get('grid_level')
            cen = cen_by_grid.get(gl)
            if cen is None:
                raise RuntimeError(
                    'no central point of the same grid level (%s) for %s'
                    % (gl, r['key']))
            delta_A = (np.asarray(r['coords_angstrom'], float) -
                       np.asarray(cen['coords_angstrom'], float))
            per_atom_A = np.linalg.norm(delta_A, axis=1)
            max_disp_A = float(per_atom_A.max())
            q_actual = (max_disp_A / B) / m_max          # FULL precision
            u = delta_A.reshape(-1)
            cosang = float(np.dot(u, d.reshape(-1)) /
                           (np.linalg.norm(u) * np.linalg.norm(d)))
            dev_deg = float(np.degrees(np.arccos(max(-1.0, min(1.0, cosang)))))
            partner = next((r2 for r2 in disps
                            if r2.get('amp_a') == r.get('amp_a')
                            and r2.get('grid_level') == gl
                            and r2.get('sign') == -r.get('sign')), None)
            sym = None
            if partner is not None:
                pd = (np.asarray(partner['coords_angstrom'], float) -
                      np.asarray(cen['coords_angstrom'], float))
                sym = float(np.linalg.norm(pd, axis=1).max() / max_disp_A)
            if r.get('sign') < 0:
                dev_deg = 180.0 - dev_deg          # opposite direction by design
            points.append(dict(
                key=r['key'], grid_level=gl, amp_nominal_a=r['amp_a'],
                sign=r['sign'], max_atom_disp_actual_A=max_disp_A,
                q_actual_bohr=q_actual, amplification=max_disp_A / r['amp_a'],
                direction_deviation_deg=dev_deg, plus_minus_symmetry=sym,
                scf_converged=converged_ok(r),
                e_total=r.get('e_total', r.get('e_scf_total')),
                gradient=r.get('gradient', r.get('grads', {}).get(
                    'grid_response_False', {}).get('gradient')),
                gradient_true=r.get('grads', {}).get('grid_response_True', {}).get('gradient')))

        analysis_F, analysis_T = [], []
        nonconv = [p for p in points if not p['scf_converged']]
        for gl in sorted({p['grid_level'] for p in points}):
            cen = cen_by_grid[gl]
            if not converged_ok(cen):
                continue
            e0 = cen.get('e_total', cen.get('e_scf_total'))
            g0F = np.asarray(cen.get('gradient', cen.get('grads', {}).get(
                'grid_response_False', {}).get('gradient')), float)
            g0T_v = cen.get('grads', {}).get('grid_response_True', {}).get('gradient')
            g0T = np.asarray(g0T_v, float) if g0T_v is not None else None
            s_gF = float(d @ g0F)
            s_gT = float(d @ g0T) if g0T is not None else None
            for amp in AMPS_A:
                plus = next((p for p in points if p['grid_level'] == gl and
                             p['amp_nominal_a'] == amp and p['sign'] == 1), None)
                minus = next((p for p in points if p['grid_level'] == gl and
                              p['amp_nominal_a'] == amp and p['sign'] == -1), None)
                if plus is None or minus is None:
                    continue
                if not (plus['scf_converged'] and minus['scf_converged']):
                    analysis_F.append(dict(grid_level=gl, amp_nominal_a=amp,
                                           excluded='non-converged SCF point'))
                    continue
                q = plus['q_actual_bohr']
                if abs(q - minus['q_actual_bohr']) > 1e-12:
                    raise RuntimeError('asymmetric q for %s amp %s' % (tid, amp))
                k_E = (plus['e_total'] + minus['e_total'] - 2 * e0) / q ** 2
                gF_p = np.asarray(plus['gradient'], float)
                gF_m = np.asarray(minus['gradient'], float)
                k_gF = float(d @ (gF_p - gF_m)) / (2 * q)
                s_E = float(plus['e_total'] - minus['e_total']) / (2 * q)
                old_kE = _old_value(old_summary, tid, gl, amp, 'k_E')
                old_kgF = _old_value(old_summary, tid, gl, amp, 'k_gF')
                if old_kgF is None:
                    old_kgF = _old_value(old_summary, tid, gl, amp, 'k_g')
                old_sE = _old_value(old_summary, tid, gl, amp, 's_E')
                x_kE = (dict(status='missing') if old_kE is None else
                        dict(old=old_kE, expected=B ** 2 * old_kE,
                             recomputed=k_E,
                             abs_diff=abs(k_E - B ** 2 * old_kE),
                             status=('pass' if abs(k_E - B ** 2 * old_kE) <=
                                     1e-8 * max(abs(k_E), 1e-30) + 1e-12
                                     else 'fail')))
                x_kg = (dict(status='missing') if old_kgF is None else
                        dict(old=old_kgF, expected=B * old_kgF,
                             recomputed=k_gF,
                             abs_diff=abs(k_gF - B * old_kgF),
                             status=('pass' if abs(k_gF - B * old_kgF) <=
                                     1e-8 * max(abs(k_gF), 1e-30) + 1e-12
                                     else 'fail')))
                x_sE = (dict(status='missing') if old_sE is None else
                        dict(old=old_sE, expected=B * old_sE, recomputed=s_E,
                             abs_diff=abs(s_E - B * old_sE),
                             status=('pass' if abs(s_E - B * old_sE) <=
                                     1e-8 * max(abs(s_E), 1e-30) + 1e-12
                                     else 'fail')))
                analysis_F.append(dict(
                    grid_level=gl, amp_nominal_a=amp,
                    amp_actual_a=plus['max_atom_disp_actual_A'],
                    q_actual_bohr=q, q_old_bohr=q * B,
                    k_E=k_E, k_gF=k_gF, s_E=s_E, s_gF=s_gF, s_gT=s_gT,
                    k_gap_F=k_gF - k_E, slope_gap_F=s_gF - s_E,
                    xcheck_k_E=x_kE, xcheck_k_gF=x_kg, xcheck_s_E=x_sE))
                if g0T is not None:
                    gT_p = np.asarray(plus['gradient_true'], float)
                    gT_m = np.asarray(minus['gradient_true'], float)
                    k_gT = float(d @ (gT_p - gT_m)) / (2 * q)
                    analysis_T.append(dict(
                        grid_level=gl, amp_nominal_a=amp,
                        amp_actual_a=plus['max_atom_disp_actual_A'],
                        q_actual_bohr=q, k_gT=k_gT, k_E=k_E, gap_T=k_gT - k_E))
        out[tid] = dict(
            construction=construction, lam_mw=lam_mw, nu_claim_cm1=nu_claim,
            direction_max_atom_norm=m_max,
            counts=dict(central=len(centrals), displaced=len(disps),
                        displaced_nonconverged=len(nonconv),
                        coverage_pct=round(100.0 * (len(disps) - len(nonconv)) /
                                           max(len(disps), 1), 1)),
            grid_levels=sorted({p['grid_level'] for p in points}),
            points=points, analysis_false=analysis_F, analysis_true=analysis_T)
    return out


def main():
    tl = load(os.path.join(DC, 'directional_task_list.json'), {})
    dc_records = []
    for t in tl.get('tasks', []):
        k = '%s_L%d_%s' % (t['candidate'], t['grid_level'],
                           'central' if t['kind'] == 'central'
                           else 'amp%g_%+d' % (t['amp_a'], t['sign']))
        r = load(os.path.join(DC, 'point_%s.json' % k))
        if r:
            dc_records.append(r)
    gr_records = load(os.path.join(GR, 'grid_response_points.json'), {}).get('points', [])
    RES = os.path.join(ROOT, 'results', '01_water_matrices', 'pure_water_o3_h2o')
    dc_old = load(os.path.join(RES, 'directional_confirmation',
                               'directional_summary.json'), {})
    gr_old = load(os.path.join(RES, 'grid_response_diagnostic',
                               'grid_response_summary.json'), {})

    dc_aud = audit_batch(dc_records, 'directional_confirmation', dc_old)
    gr_aud = audit_batch(gr_records, 'grid_response_diagnostic', gr_old)

    out = dict(
        job='JOB-2026-0905-009 / JOB-2026-0906-002',
        step='unit_consistency_revision_offline_analysis (revised)',
        defect=dict(description='Bohr-convention q applied to Angstrom '
                                'coordinates -> actual amplitude = 1.8897*nominal',
                    b=B, amplification=1 / B,
                    correction_direction='corrected = old * b^2 (k_E); '
                                         'corrected = old * b (k_g, s_E)'),
        directional_confirmation=dc_aud,
        grid_response_diagnostic=gr_aud,
        source_hashes=dict(
            directional_points=sorted(sha256(os.path.join(DC, f))
                                      for f in os.listdir(DC)
                                      if f.startswith('point_')),
            grid_response_points=sorted(sha256(os.path.join(GR, f))
                                        for f in os.listdir(GR)
                                        if f.startswith('point_'))))
    path = os.path.join(RAW, 'unit_consistency_analysis_revised.json')
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=2)
    print('SAVED ->', path)
    for batch, aud in (('directional', dc_aud), ('grid_response', gr_aud)):
        for tid in TARGETS:
            a = aud[tid]
            print('%s/%s central=%d displaced=%d coverage=%.0f%% grids=%s'
                  % (batch, tid, a['counts']['central'],
                     a['counts']['displaced'], a['counts']['coverage_pct'],
                     a['grid_levels']))
            for row in a['analysis_false']:
                if 'excluded' in row:
                    continue
                print('   L%s nominal=%.3f actual=%.5fA q=%.5f k_E=%+.5e '
                      'k_gF=%+.5e  xcheck: k_E=%s k_g=%s s_E=%s'
                      % (row['grid_level'], row['amp_nominal_a'],
                         row['amp_actual_a'], row['q_actual_bohr'],
                         row['k_E'], row['k_gF'],
                         row['xcheck_k_E'].get('status'),
                         row['xcheck_k_gF'].get('status'),
                         row['xcheck_s_E'].get('status')))


if __name__ == '__main__':
    main()
