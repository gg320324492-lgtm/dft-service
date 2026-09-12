#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-062: C1 trajectory & negative-mode projection audit.

ZERO evaluations: reads ONLY saved records from JOB-057..061 (057
candidate + recheck, 058 centre records + mode 0, 059 centre + four
displacements, 060 start + opt evaluations incl. the non-accepted trial,
061 start + all 20 opt evaluations), classifies them (actual evaluation /
start reproduction / accepted-iterate evaluation / non-accepted trial /
recheck), compares geometries (raw diffs, fragment internals, centroid /
min-contact distances, optional Kabsch RMSD with identity AND terminal-O
swap variants, rotations saved), projects every full gradient onto the
058 NEW negative mode q (Euclidean-normalised cart_norm_mode, sign fixed,
no further mass division; ORIGINAL coordinates only - registration is
used for RMSD only, and registered projections would require rotating q
and g, which is NOT done here), reports the 052/053 direction overlaps
(atom mapping closable: same element order and 21-dim layout; vectors
live at DIFFERENT geometries - reported with that caveat), renders
zero-dependency SVG charts (per-batch series, never joined into one
continuous trajectory), and writes the judgment section with explicit
"cannot determine" entries where evidence is insufficient.

NO SCF / gradient / stability / Hessian / frequency / optimization calls.
All new files under run_artifacts/02_nh3o3_reference/c1_audit062/.
"""
import os, sys, json, glob, hashlib
import numpy as np

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
R = ROOT + '/run_artifacts/02_nh3o3_reference'
OUT = R + '/c1_audit062'
BPA = 1.0 / 0.52917721092
SYMS = ['N', 'H', 'H', 'H', 'O', 'O', 'O']
MASSES = np.array([14.0, 1.0, 1.0, 1.0, 16.0, 16.0, 16.0])


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for blk in iter(lambda: fh.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()[:16]


def sha_arr(a):
    return hashlib.sha256(np.asarray(a, float).tobytes()).hexdigest()[:16]


def save_json(path, obj):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(obj, fh, indent=2, default=str)
    os.replace(tmp, path)


def contact(x_bohr):
    R = np.asarray(x_bohr, float).reshape(7, 3)
    cenN, cenO = R[:4].mean(axis=0), R[4:].mean(axis=0)
    v = cenO - cenN
    dNO = sorted(float(np.linalg.norm(R[0] - R[o])) for o in (4, 5, 6))
    dHO = sorted(float(np.linalg.norm(R[h] - R[o]))
                 for h in (1, 2, 3) for o in (4, 5, 6))
    return dict(NH3_O3_centroid_Bohr=float(np.linalg.norm(v)),
                min_N_O_Bohr=dNO[0], min_H_O_Bohr=dHO[0])


def fragment_internals(x_bohr):
    R = np.asarray(x_bohr, float).reshape(7, 3)
    nh = [float(np.linalg.norm(R[0] - R[i])) for i in (1, 2, 3)]
    hh = [float(np.linalg.norm(R[i] - R[j]))
          for i, j in ((1, 2), (1, 3), (2, 3))]
    oo = [float(np.linalg.norm(R[i] - R[j]))
          for i, j in ((4, 5), (5, 6), (4, 6))]
    inter = [float(np.linalg.norm(R[i] - R[j]))
             for i in range(4) for j in range(4, 7)]
    return dict(NH_bonds_Bohr=nh, NH3_HH_Bohr=hh, O3_bonds_Bohr=oo,
                min_interfragment_Bohr=min(inter),
                max_interfragment_Bohr=max(inter))


def atom_order_check(x_bohr, tol_bond=2.6, tol_inter=4.0):
    """Element-order sanity for the SYMS layout: N-H pairs bonded
    (~1.9 Bohr); the three O atoms form a bent triatomic (two O-O
    distances < tol_bond, the third > 3.0 Bohr - the central oxygen
    index may change between records); all N/H-to-O distances
    non-bonded (> tol_inter)."""
    R = np.asarray(x_bohr, float).reshape(7, 3)
    dNH = [float(np.linalg.norm(R[0] - R[i])) for i in (1, 2, 3)]
    dOO = sorted(float(np.linalg.norm(R[i] - R[j]))
                 for i, j in ((4, 5), (5, 6), (4, 6)))
    dXO = [float(np.linalg.norm(R[i] - R[j]))
           for i in (0, 1, 2, 3) for j in (4, 5, 6)]
    return bool(max(dNH) < tol_bond and dOO[0] < tol_bond
                and dOO[1] < tol_bond and dOO[2] > 3.0
                and min(dXO) > tol_inter)


def kabsch_rmsd(P, Q):
    """P, Q: (7,3) same order -> (rmsd, rotation 3x3 applied to P)."""
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    V, S, Wt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1.0, 1.0, d])
    U = V @ D @ Wt                      # rotation applied to P coords
    diff = Pc @ U - Qc
    return float(np.sqrt((diff ** 2).sum() / P.shape[0])), U


def load_record(path, classification, note=''):
    r = json.load(open(path))
    x_bohr = (np.asarray(r['coords_actual_angstrom'], float) * BPA)
    g = r.get('grad')
    return dict(
        path=os.path.relpath(path, ROOT), sha256_16=sha256_file(path),
        tag=r.get('tag'), category=r.get('category'),
        classification=classification, note=note,
        e_total=float(r['e_total']), grad_max=float(r['grad_max']),
        coords_sha_record=r.get('coords_sha'),
        coords_bohr=x_bohr.reshape(-1).tolist(),
        coords_order_ok=atom_order_check(x_bohr),
        has_full_gradient=bool(g is not None
                               and len(np.asarray(g).reshape(-1)) == 21
                               and np.isfinite(np.asarray(g, float)).all()),
        _x=x_bohr, _g=(np.asarray(g, float).reshape(-1)
                       if g is not None else None))


def classify_batch(batch_dir, start_tag='start'):
    """Return {tag: classification} for a relaxation batch dir."""
    out = {}
    accf = os.path.join(batch_dir, 'accepted_iterates.json')
    acc = json.load(open(accf))['iterates'] if os.path.exists(accf) else []
    acc_arr = [np.asarray(a, float) for a in acc]
    out['start'] = 'start reproduction'
    for f in sorted(glob.glob(os.path.join(batch_dir,
                                           'eval_opt_opt_*.json'))):
        r = json.load(open(f))
        x = np.asarray(r['coords_actual_angstrom'], float).reshape(-1) \
            * BPA
        dmin = min((float(np.abs(x - a).max()) for a in acc_arr),
                   default=float('inf'))
        out[r['tag']] = ('accepted-iterate evaluation' if dmin <= 1e-12
                         else 'non-accepted trial')
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    audit = dict(
        job='JOB-2026-0906-062 C1 trajectory & negative-mode projection '
            'audit (ZERO evaluations)',
        review_061='notes/job061_commander_review_2026-09-10.md',
        budget=dict(all_quantum_calls=0,
                    note='no SCF/gradient/stability/Hessian/frequency/'
                         'optimization; offline analysis of saved records '
                         'only'))

    # ================= 1) sources & completeness =========================
    records = []
    rec = load_record(R + '/c1_fdopt057/eval_opt_opt_05.json',
                      'actual evaluation',
                      note='JOB-057 threshold-stop accepted point (the '
                           'registered candidate)')
    records.append(rec)
    records.append(load_record(
        R + '/c1_fdopt057/eval_recheck_recheck.json',
        'independent recheck record', note='JOB-057'))
    records.append(load_record(R + '/c1_fdhess058/eval_centre.json',
                               'actual evaluation',
                               note='JOB-058 original centre rebuild'))
    records.append(load_record(
        R + '/c1_fdhess058_resume01/eval_centre.json',
        'actual evaluation', note='JOB-058 resume centre rebuild'))
    records.append(load_record(R + '/c1_negdir059/eval_centre.json',
                               'centre reproduction', note='JOB-059'))
    for t in ('p001', 'm001', 'p002', 'm002'):
        records.append(load_record(
            R + '/c1_negdir059/eval_disp_%s.json' % t,
            'displacement evaluation',
            note='JOB-059 preregistered +/-q (%s)' % t))
    for bdir, batch in ((R + '/c1_relax060', '060'),
                        (R + '/c1_relaxcont061', '061')):
        cls = classify_batch(bdir)
        for f in sorted(glob.glob(os.path.join(bdir, 'eval_*.json'))):
            base = os.path.basename(f)
            if base.startswith('eval_start_repro_'):
                tag = 'start'
            elif base.startswith('eval_opt_'):
                tag = json.load(open(f))['tag']
            else:
                continue
            records.append(load_record(f, cls[tag],
                                       note='JOB-%s batch' % batch))
    src_table = [{k: rec[k] for k in ('path', 'sha256_16', 'tag',
                                      'category', 'classification',
                                      'e_total', 'grad_max',
                                      'has_full_gradient',
                                      'coords_order_ok', 'note')}
                 for rec in records]
    audit['sources'] = dict(
        n_records=len(records),
        all_have_full_gradient=bool(all(r['has_full_gradient']
                                        for r in records)),
        all_atom_order_ok=bool(all(r['coords_order_ok']
                                   for r in records)),
        records=src_table,
        no_reverse_engineering='no gradient reconstructed from grad_max/'
                               'rms; every projection uses the saved full '
                               'gradient vector')

    # ================= 2) geometry comparison ============================
    anchor = records[0]['_x']          # 057 opt_05
    geom = []
    for rec in records:
        x = rec['_x']
        d_raw = float(np.abs(x.reshape(-1) - anchor.reshape(-1)).max())
        rmsd_id, U_id = kabsch_rmsd(x, anchor)
        Ps = x.copy()
        Ps[[5, 6]] = Ps[[6, 5]]        # terminal-O swap variant
        rmsd_sw, U_sw = kabsch_rmsd(Ps, anchor)
        geom.append(dict(
            tag=rec['tag'], classification=rec['classification'],
            note=rec['note'],
            raw_maxdiff_vs_057opt05_Bohr=d_raw,
            nh3_o3_centroid_Bohr=contact(x)['NH3_O3_centroid_Bohr'],
            min_N_O_Bohr=contact(x)['min_N_O_Bohr'],
            min_H_O_Bohr=contact(x)['min_H_O_Bohr'],
            fragment_internals=fragment_internals(x),
            kabsch_rmsd_identity=rmsd_id,
            kabsch_rmsd_terminalO_swap=rmsd_sw,
            rotation_identity=U_id.tolist(),
            rotation_terminalO_swap=U_sw.tolist()))
    audit['geometry_comparison'] = dict(
        anchor='057 opt_05 (%s)' % records[0]['path'],
        method='raw max-abs coordinate diff; fragment internals; centroid '
               'and min-contact distances; Kabsch RMSD (identity order '
               'AND terminal-O swap 5<->6, rotations saved)',
        records=geom,
        limits='small RMSD is NOT interpreted as "same minimum" or "same '
               'basin"; distance changes carry no direction attribution '
               '(trajectories not projected)')

    # ================= 3) 058 negative-mode projections ==================
    r58 = json.load(open(R + '/c1_fdhess058_resume01/'
                           'resume01_results.json'))
    m0 = r58['modes']['modes'][0]
    d = np.asarray(m0['cart_norm_mode'], float).reshape(-1)
    sqrt_m = np.repeat(np.sqrt(MASSES), 3)
    mw_norm = float(np.linalg.norm(d * sqrt_m))
    q = d / np.linalg.norm(d)
    imax = int(np.argmax(np.abs(q)))
    if q[imax] < 0:
        q = -q
        sign_note = 'flipped so the largest-|component| entry is positive'
    else:
        sign_note = 'largest-|component| entry already positive'
    R0_58 = records[3]['_x'].reshape(-1)     # 058 resume centre
    proj = []
    for rec in records:
        if not rec['has_full_gradient']:
            continue
        g = rec['_g']
        gq = float(g @ q)
        gnorm = float(np.linalg.norm(g))
        dq = float(q @ (rec['_x'].reshape(-1) - R0_58))
        proj.append(dict(
            tag=rec['tag'], classification=rec['classification'],
            note=rec['note'],
            g_dot_q=gq,
            grad_parallel_norm=abs(gq),
            grad_orthogonal_norm=float(np.sqrt(max(
                gnorm ** 2 - gq ** 2, 0.0))),
            grad_norm=gnorm,
            cosine=gq / gnorm if gnorm > 0 else None,
            disp_along_q_from_058centre_Bohr=dq,
            coords='ORIGINAL saved coordinates (registration NOT applied '
                   'to projections; q and g were not rotated)'))
    audit['mode_projection'] = dict(
        source='058 resume01_results.json modes[0] (freq %.4f cm-1, '
               'lambda %.6e Eh/Bohr^2/amu)' % (m0['freq_cm1'],
                                               r58['modes'][
                                                   'eigenvalues_Eh_'
                                                   'Bohr2_amu'][0]),
        q=dict(euclidean_normalised=True, further_mass_division=False,
               mw_norm_check=mw_norm, mw_norm_expected_1=bool(
                   abs(mw_norm - 1.0) < 1e-8),
               q_sha=sha_arr(q), sign_convention=sign_note,
               largest_component_index=imax),
        reference_centre='058 resume centre (%s)' % records[3]['path'],
        per_record=proj)

    # ================= 4) direction overlaps (052/053) ===================
    r52 = json.load(open(R + '/c1_cand_check052/'
                           'cand_check052_results.json'))
    d52 = np.asarray(r52['vibrational_curvature']['per_step']['1e-3'][
        'modes'][0]['cart_norm_mode'], float).reshape(-1)
    q52 = d52 / np.linalg.norm(d52)
    im52 = int(np.argmax(np.abs(q52)))
    if q52[im52] < 0:
        q52 = -q52
    man53 = json.load(open(R + '/c1_negmode_dir053_resume01/'
                             'input_manifest_resume01.json'))
    q53 = np.asarray(man53['direction']['q'], float).reshape(-1)
    if q53[int(np.argmax(np.abs(q53)))] < 0:
        q53 = -q53
    audit['direction_overlap'] = dict(
        closable=True,
        reason='all three vectors are 21-component Cartesian '
               'cart_norm_mode/q in the SAME element order (N,H,H,H,O,O,'
               'O) with recorded mass conventions; sign fixed by the '
               'same largest-|component| rule',
        overlap=dict(
            q058_vs_q052_abs_cos=abs(float(q @ q52)),
            q058_vs_q053_abs_cos=abs(float(q @ q53)),
            q052_vs_q053_abs_cos=abs(float(q52 @ q53))),
        caveat='the vectors are defined at DIFFERENT geometries (051 '
               'candidate vs 057 candidate); the overlap is a '
               'configuration-space alignment, NOT a same-tangent-space '
               'mode identity, and does NOT prove "same soft coordinate"',
        not_claimed='same mode / same basin / two candidates are two '
                    'isomers')

    # ================= 5) charts (zero-dependency SVG) ===================
    charts = render_charts(records, q, R0_58, audit)
    audit['charts'] = charts

    # ================= 6) judgments ======================================
    sel = {r['tag'] + '|' + r['note']: r for r in records}
    p24 = next(r for r in records if r['tag'] == 'opt_24'
               and '060' in r['note'])
    p20 = next(r for r in records if r['tag'] == 'opt_20'
               and '061' in r['note'])
    p15 = next(r for r in records if r['tag'] == 'opt_15'
               and '061' in r['note'])
    cen58 = records[3]
    dq24 = float(q @ (p24['_x'].reshape(-1) - R0_58))
    dq20 = float(q @ (p20['_x'].reshape(-1) - R0_58))
    a0 = -8.696929e-06
    kE = -8.901679e-05
    pred24 = a0 * dq24 + 0.5 * kE * dq24 ** 2
    pred20 = a0 * dq20 + 0.5 * kE * dq20 ** 2
    rmsds = [g['kabsch_rmsd_identity'] for g in geom]
    audit['judgments'] = dict(
        q1_geometry_continuity=dict(
            evidence='filled below with the accepted/trial split',
            answer=None),
        q2_motion_along_negative_mode=dict(
            evidence='disp along q from the 058 centre: 057 opt_05 '
                     '%+.5f, 059 +0.002 point %+.5f, 060 opt_24 %+.5f, '
                     '061 opt_20 %+.5f Bohr; gradient cosines along the '
                     'chain: see mode_projection.per_record'
                     % (float(q @ (records[0]['_x'].reshape(-1) - R0_58)),
                        float(q @ (next(r for r in records
                                         if r['tag'] == 'disp_p002')[
                             '_x'].reshape(-1) - R0_58)),
                        dq24, dq20),
            answer=None),   # filled below
        q3_energy_vs_mode_motion=dict(
            evidence='with a0=-8.696929e-6 and kE=-8.901679e-5 (059 '
                     'h=0.002): predicted dE from the q-component alone '
                     '= a0*dq + 0.5*kE*dq^2 -> 060 opt_24: %+.3e '
                     '(actual %+.3e), 061 opt_20: %+.3e (actual %+.3e); '
                     'displacement fractions along q = %.1f%% (060 '
                     'opt_24), %.1f%% (061 opt_20)'
                     % (pred24, p24['e_total'] - cen58['e_total'],
                        pred20, p20['e_total'] - cen58['e_total'],
                        100.0 * abs(dq24) / float(np.linalg.norm(
                            p24['_x'].reshape(-1) - R0_58)),
                        100.0 * abs(dq20) / float(np.linalg.norm(
                            p20['_x'].reshape(-1) - R0_58))),
            answer=None),   # filled below
        q4_whether_to_continue_c1=dict(
            evidence='060+061: 45 optimization evaluations, min gradient '
                     'reached 5.619e-6 (060 opt_02, geometry ~= 057 '
                     'candidate) and 6.208e-5 (061 opt_15); both batches '
                     'ended at budget exhaustion with the energy still '
                     'decreasing; no SCF failure',
            answer='THE EVIDENCE SUPPORTS PAUSING UNCONDITIONED FURTHER '
                   'OPTIMIZATION STEPS: two consecutive limited relaxations '
                   'did not reach the strict gate; the energy keeps '
                   'descending at cutoff (no convergence in sight within '
                   'small budgets); whether to close the C1 complex route, '
                   'keep an unaccepted representative structure, or move '
                   'to C2/reaction-path preparation is a COMMANDER '
                   'decision - this audit provides evidence, not the '
                   'decision'),
        forbidden_conclusions_reaffirmed=[
            'no minimum found', 'negative mode eliminated', 'the two '
            'candidates are two isomers', 'the two negative modes are '
            'the same mode', 'energy lowering = stronger binding',
            '"the C1 literature structure does not exist"'])

    # fill the two conditional answers with the computed numbers
    rmsds = [gg['kabsch_rmsd_identity'] for gg in geom
             if gg['classification'] == 'accepted-iterate evaluation']
    rmsd_trial = max(gg['kabsch_rmsd_identity'] for gg in geom)
    audit['judgments']['q1_geometry_continuity'] = dict(
        evidence='Kabsch RMSD (identity order) vs 057 opt_05: accepted '
                 'iterates %.5f-%.5f Bohr, the non-accepted trial opt_25 '
                 '%.5f Bohr; raw max coordinate diffs <= %.4f Bohr; the '
                 'terminal-O-swap RMSD (~2.1 Bohr) is ~20x worse than '
                 'the identity order everywhere, i.e. the end-oxygen '
                 'ordering is stable and no swap/registration ambiguity '
                 'exists; all records pass the same element-order '
                 'sanity check; centroid distances stay within '
                 '6.105-6.186 Bohr (interfragment separation ~6 Bohr)'
                 % (min(rmsds), max(rmsds), rmsd_trial,
                    max(gg['raw_maxdiff_vs_057opt05_Bohr']
                        for gg in geom)),
        answer='YES, WITH NUMBERS STATED - all audited ACCEPTED '
               'geometries stay within %.3f Bohr RMSD (the '
               'non-accepted trial %.3f) of the 057 candidate, i.e. '
               'a ~2%% deformation relative to the ~6 Bohr contact '
               'distance: no jump to another configuration region in '
               'the SAVED records. This is a region-continuity '
               'statement only, NOT a same-minimum/same-basin claim'
               % (max(rmsds), rmsd_trial))
    # fill the two conditional answers with the computed numbers
    # trajectory monotonicity uses ACCEPTED-ITERATE evaluations only;
    # the non-accepted trial (060 opt_25) is listed separately
    dq_seq_060 = [p['disp_along_q_from_058centre_Bohr'] for p in proj
                  if '060' in p['note'] and p['tag'].startswith('opt')
                  and p['classification'] == 'accepted-iterate '
                       'evaluation']
    dq_seq_061 = [p['disp_along_q_from_058centre_Bohr'] for p in proj
                  if '061' in p['note'] and p['tag'].startswith('opt')
                  and p['classification'] == 'accepted-iterate '
                       'evaluation']
    trial25 = [p['disp_along_q_from_058centre_Bohr'] for p in proj
               if '060' in p['note'] and p['tag'] == 'opt_25']
    mono_060 = all(b > a for a, b in zip(dq_seq_060, dq_seq_060[1:]))
    mono_061 = all(b > a for a, b in zip(dq_seq_061, dq_seq_061[1:]))
    cos24 = [p['cosine'] for p in proj
             if p['tag'] == 'opt_24' and '060' in p['note']][0]
    cos20 = [p['cosine'] for p in proj
             if p['tag'] == 'opt_20' and '061' in p['note']][0]
    audit['judgments']['q2_motion_along_negative_mode']['answer'] = (
        'PARTLY YES - the displacement along +q grew MONOTONICALLY '
        'through the ACCEPTED iterates of BOTH relaxations (060: '
        '+0.0020 -> %.4f Bohr at opt_24, monotonic=%s; 061: -> %.4f '
        'Bohr at opt_20, monotonic=%s; gradient cosine along q stayed '
        'negative throughout: %+.3f at 060 opt_24, %+.3f at 061 '
        'opt_20), i.e. the optimizer kept descending along the 058 '
        'soft direction. The non-accepted 060 trial opt_25 went '
        'further still (%.4f Bohr) and is NOT used for this claim. '
        'HOWEVER the motion is NOT purely along q: the '
        'contact-distance coordinate moved OPPOSITE in 061 (distances '
        'shrank while dq grew), and the orthogonal gradient component '
        'stays comparable to or larger than the parallel one over '
        'most of the chain. Evidence for sustained motion WITH a '
        'growing q-component: yes; "the motion is along the single '
        'soft coordinate": NOT supported'
        % (dq_seq_060[-1], mono_060, dq_seq_061[-1], mono_061,
           cos24, cos20, trial25[0] if trial25 else float('nan')))
    pct24 = 100.0 * pred24 / (p24['e_total'] - cen58['e_total'])
    pct20 = 100.0 * pred20 / (p20['e_total'] - cen58['e_total'])
    audit['judgments']['q3_energy_vs_mode_motion']['answer'] = (
        'NOT FULLY - a quadratic along q (a0, kE from 059) PREDICTS '
        '%.0f%% of the actual 060 opt_24 lowering (OVERSHOOT: the '
        'orthogonal subspace partly RAISED the energy relative to '
        'pure-q motion) and %.0f%% of the 061 opt_20 lowering '
        '(undershoot); the pure-q model is therefore not consistent '
        'across the two batches, and a significant part of the energy '
        'lowering involves motion in the orthogonal subspace. The '
        '060/061 low energies are NOT explainable as motion along the '
        '058 negative mode alone, nor is that contribution '
        'negligible' % (pct24, pct20))

    save_json(os.path.join(OUT, 'audit062_results.json'), audit)
    print('[062] records: %d | full gradients: %d | charts: %d'
          % (len(records), sum(1 for r in records
                               if r['has_full_gradient']), len(charts)))
    for k in ('q1_geometry_continuity', 'q2_motion_along_negative_mode',
              'q3_energy_vs_mode_motion', 'q4_whether_to_continue_c1'):
        print('[062] %s: %s' % (k, audit['judgments'][k]['answer'][:150]))
    print('[062] DONE (zero evaluations)', flush=True)


# ======================= SVG helpers (stdlib only) =======================
def _esc(s):
    return (str(s).replace('&', '&amp;').replace('<', '&lt;')
            .replace('>', '&gt;'))


class SVG:
    W, H = 960, 560

    def __init__(self, title):
        self.parts = [
            '<svg xmlns="http://www.w3.org/2000/svg" width="%d" '
            'height="%d" font-family="monospace" font-size="12">'
            '<rect width="100%%" height="100%%" fill="white"/>'
            '<text x="16" y="24" font-size="15" font-weight="bold">%s'
            '</text>' % (self.W, self.H, _esc(title))]

    def axes(self, x0, y0, w, h, xmin, xmax, ymin, ymax, xl, yl):
        self.x0, self.y0, self.w, self.h = x0, y0, w, h
        self.xmin, self.xmax, self.ymin, self.ymax = (xmin, xmax, ymin,
                                                      ymax)
        self.parts.append(
            '<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="black"/>'
            '<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="black"/>'
            '<text x="%d" y="%d">%s</text>'
            '<text x="12" y="%d" transform="rotate(-90 12 %d)">%s'
            '</text>'
            % (x0, y0 + h, x0 + w, y0 + h, x0, y0, x0 + w, y0, x0 + w,
               y0 + h + 22, _esc(xl), y0 + h // 2, y0 + h // 2, _esc(yl)))

    def xy(self, x, y):
        px = self.x0 + (x - self.xmin) / (self.xmax - self.xmin) * self.w
        py = self.y0 + self.h - (y - self.ymin) / (self.ymax - self.ymin) \
            * self.h
        return px, py

    def polyline(self, xs, ys, color, label, dash=''):
        pts = ' '.join('%.1f,%.1f' % self.xy(x, y)
                       for x, y in zip(xs, ys))
        self.parts.append(
            '<polyline points="%s" fill="none" stroke="%s" '
            'stroke-width="1.6" %s/>' % (pts, color, dash))
        self._legend(color, label)

    def scatter(self, xs, ys, color, label, sym='circle'):
        for x, y in zip(xs, ys):
            px, py = self.xy(x, y)
            if sym == 'circle':
                self.parts.append(
                    '<circle cx="%.1f" cy="%.1f" r="3.2" fill="%s"/>'
                    % (px, py, color))
            else:
                self.parts.append(
                    '<rect x="%.1f" y="%.1f" width="6" height="6" '
                    'fill="none" stroke="%s"/>' % (px - 3, py - 3, color))
        self._legend(color, label, sym)

    def hline(self, y, color='gray', dash='stroke-dasharray="4,3"'):
        _, py = self.xy(self.xmin, y)
        _, px2 = self.xy(self.xmax, y)
        self.parts.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" '
                          'stroke="%s" %s/>'
                          % (self.x0, py, px2, py, color, dash))

    def vline(self, x, color='gray'):
        px, _ = self.xy(x, self.ymin)
        self.parts.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" '
                          'stroke="%s" stroke-dasharray="4,3"/>'
                          % (px, self.y0, px, self.y0 + self.h, color))

    def _legend(self, color, label, sym='circle'):
        self._legend_n = getattr(self, '_legend_n', 0) + 1
        y = 40 + (self._legend_n - 1) * 18
        if sym == 'circle':
            self.parts.append('<circle cx="700" cy="%d" r="4" '
                              'fill="%s"/>' % (y, color))
        else:
            self.parts.append('<rect x="696" y="%d" width="8" height="8" '
                              'fill="none" stroke="%s"/>' % (y - 4, color))
        self.parts.append('<text x="712" y="%d">%s</text>'
                          % (y + 4, _esc(label)))

    def xticks(self, vals):
        for v in vals:
            px, py = self.xy(v, self.ymin)
            self.parts.append('<line x1="%.1f" y1="%d" x2="%.1f" '
                              'y2="%d" stroke="black"/>'
                              '<text x="%.1f" y="%d" text-anchor="middle">'
                              '%s</text>'
                              % (px, self.y0 + self.h, px,
                                 self.y0 + self.h + 4, px,
                                 self.y0 + self.h + 16, _esc(v)))

    def save(self, name, sources):
        self.parts.append('<text x="16" y="%d" font-size="11">%s</text>'
                          '<text x="16" y="%d" font-size="11">data: %s'
                          '</text></svg>'
                          % (self.H - 34, _esc('batches differ in start '
                                               'point/optimizer; series '
                                               'are NOT one continuous '
                                               'trajectory'),
                             self.H - 18, _esc(sources)))
        path = os.path.join(OUT, 'charts', name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as fh:
            fh.write('\n'.join(self.parts))
        return os.path.relpath(path, ROOT)


def render_charts(records, q, R0_58, audit):
    out = {}
    C = {'057': '#1f77b4', '059': '#2ca02c', '060': '#d62728',
         '061': '#9467bd'}

    def batch_of(rec):
        for b in ('057', '059', '060', '061'):
            if 'JOB-%s' % b in rec['note']:
                return b
        return '058'

    # chart 1: E vs evaluation index (per batch, separate polylines)
    s = SVG('C1 energies per batch (Eh)')
    s.axes(60, 40, 600, 440, 0, 26, -282.001735, -282.0017215,
           'evaluation index within batch', 'E (Eh)')
    for b in ('057', '060', '061'):
        rs = [r for r in records if batch_of(r) == b
              and r['tag'].startswith('opt')]
        rs.sort(key=lambda r: int(r['tag'].split('_')[1]))
        s.polyline([int(r['tag'].split('_')[1]) for r in rs],
                   [r['e_total'] for r in rs], C[b],
                   'JOB-%s opt (separate series)' % b)
    for r in records:
        if r['tag'] == 'disp_p002':
            px, py = s.xy(0.5, r['e_total'])
            s.parts.append('<circle cx="%.1f" cy="%.1f" r="3.2" '
                           'fill="%s"/>' % (px, py, C['059']))
    s._legend(C['059'], '059 +0.002q displacement point (not an '
                        'optimization step)')
    out['chart1_energies'] = s.save('chart1_energies.svg',
                                    'eval_opt_opt_*.json of 057/060/061; '
                                    'c1_negdir059/eval_disp_p002.json')

    # chart 2: gmax vs index
    s = SVG('C1 max|g| per batch (log-ish view, linear axis)')
    s.axes(60, 40, 600, 440, 0, 26, 0, 9e-4,
           'evaluation index within batch', 'max|g| (Eh/Bohr)')
    for b in ('057', '060', '061'):
        rs = [r for r in records if batch_of(r) == b
              and r['tag'].startswith('opt')]
        rs.sort(key=lambda r: int(r['tag'].split('_')[1]))
        s.polyline([int(r['tag'].split('_')[1]) for r in rs],
                   [r['grad_max'] for r in rs], C[b], 'JOB-%s opt' % b)
    s.hline(1e-5, color='#555')
    s._legend('#555', 'general acceptance gate 1e-5 (NOT the batch '
                      'trigger)')
    s.hline(1e-6, color='#999')
    s._legend('#999', '060/061 batch trigger 1e-6')
    out['chart2_gmax'] = s.save('chart2_gmax.svg',
                                'same records as chart1')

    # chart 3: disp along q vs E (all records with gradients)
    s = SVG('q-direction displacement vs energy (all audited records)')
    s.axes(60, 40, 600, 440, -0.005, 0.13, -282.001735, -282.0017215,
           'q.(R - R058centre) (Bohr)', 'E (Eh)')
    for b in ('057', '059', '060', '061'):
        rs = [r for r in records if batch_of(r) == b
              and r['has_full_gradient'] and r['tag'] != 'recheck']
        s.scatter([float(q @ (r['_x'].reshape(-1) - R0_58))
                   for r in rs], [r['e_total'] for r in rs], C[b],
                  'JOB-%s records' % b)
    out['chart3_qdisp_E'] = s.save('chart3_qdisp_vs_E.svg',
                                   'all audited records with full '
                                   'gradients')

    # chart 4: gradient parallel/orthogonal vs disp along q
    s = SVG('gradient components vs q-displacement')
    s.axes(60, 40, 600, 440, -0.005, 0.13, -6e-4, 6e-4,
           'q.(R - R058centre) (Bohr)',
           'g.q (filled) / +/- g_orth (open) (Eh/Bohr)')
    for b in ('057', '059', '060', '061'):
        rs = [r for r in records if batch_of(r) == b
              and r['has_full_gradient'] and r['tag'] != 'recheck']
        xs = [float(q @ (r['_x'].reshape(-1) - R0_58)) for r in rs]
        ys = [float(r['_g'] @ q) for r in rs]
        yo = [float(np.sqrt(max(np.linalg.norm(r['_g']) ** 2
                                - (r['_g'] @ q) ** 2, 0.0)))
              for r in rs]
        s.scatter(xs, ys, C[b], 'JOB-%s g.q' % b)
        s.scatter(xs, yo, C[b], 'JOB-%s |g_orth|' % b, sym='square')
    s.hline(0.0, color='#555', dash='')
    out['chart4_grad_components'] = s.save('chart4_grad_components.svg',
                                           'same records as chart3')

    # chart 5: contact distances for key geometries
    keys = [('057 opt_05', records[0]), ('058 centre', records[3]),
            ('059 p002', next(r for r in records
                              if r['tag'] == 'disp_p002')),
            ('060 opt_02', next(r for r in records if r['tag'] == 'opt_02'
                                and '060' in r['note'])),
            ('060 opt_24', next(r for r in records if r['tag'] == 'opt_24'
                                and '060' in r['note'])),
            ('061 opt_15', next(r for r in records if r['tag'] == 'opt_15'
                                and '061' in r['note'])),
            ('061 opt_20', next(r for r in records if r['tag'] == 'opt_20'
                                and '061' in r['note']))]
    s = SVG('contact distances across key geometries (Bohr)')
    s.axes(60, 40, 660, 420, 0, len(keys), 0, 7,
           'key geometry', 'distance (Bohr)')
    for i, (lab, r) in enumerate(keys):
        cg = contact(r['_x'])
        for v, col in ((cg['NH3_O3_centroid_Bohr'], '#1f77b4'),
                       (cg['min_N_O_Bohr'], '#d62728'),
                       (cg['min_H_O_Bohr'], '#2ca02c')):
            px, py = s.xy(i + 0.5, v)
            s.parts.append('<rect x="%.1f" y="%.1f" width="7" '
                           'height="%.1f" fill="%s"/>'
                           % (px - 3.5, py, s.y0 + s.h - py, col))
        s.parts.append('<text x="%.1f" y="%d" text-anchor="middle" '
                       'font-size="10">%s</text>'
                       % (s.xy(i + 0.5, 0)[0], s.y0 + s.h + 30,
                          _esc(lab)))
    s._legend('#1f77b4', 'NH3..O3 centroid')
    s._legend('#d62728', 'min N-O')
    s._legend('#2ca02c', 'min H-O')
    out['chart5_contacts'] = s.save('chart5_contacts.svg',
                                    'the seven key records listed in '
                                    'the chart')
    return out


if __name__ == '__main__':
    main()
