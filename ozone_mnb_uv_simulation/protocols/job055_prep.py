#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-055 step A: OFFLINE preparation (ZERO evaluations).

(a) batch-number check: 055 next unoccupied;
(b) 054 corrections (machine-readable, traceable; original JSON and its
    hash snapshot preserved, never rewritten):
      - last_evaluated_point = opt_25 (verified from the ledger completion
        order and the accepted-iterate callback, NOT from energy extremes);
      - displacements of opt_25 vs the 053 centre and vs the p02 start
        RECOMPUTED from full-precision coords, reported SEPARATELY as
        along-q / perpendicular-to-q / total (no mixing; the net-displacement
        direction ratio is NOT used for energy attribution);
(c) centre R* = 054 eval_opt_opt_25.json (hash + reference E/gmax verified,
    x_bohr saved coords used);
(d) direction q from the 053 direction_and_kH.json (already Cartesian,
    Euclidean-unit 21-vector; sign/order preserved; NO mass conversion);
(e) scan geometries R(t) = R* + t q, t = 0.02/0.04/0.06/0.08 Bohr, with
    per-point norm/projection/max-atom-displacement/fragment checks; the
    new line is centred at R* and is NOT merged with the 053 R0+tq path.
No SCF / gradient: reads existing records only.
"""
import os, json, hashlib, glob, re
import numpy as np

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
R54 = ROOT + '/run_artifacts/02_nh3o3_reference/c1_negmode_relax054'
R53 = ROOT + '/run_artifacts/02_nh3o3_reference/c1_negmode_dir053'
R53R = ROOT + '/run_artifacts/02_nh3o3_reference/c1_negmode_dir053_resume01'
OUT = ROOT + '/run_artifacts/02_nh3o3_reference/c1_dir_scan055'
JOBS = ROOT + '/jobs'
JOB_NO = '055'
CENTRE_FILE = R54 + '/eval_opt_opt_25.json'
Q_FILE = R53 + '/direction_and_kH.json'
SYMS = ['N', 'H', 'H', 'H', 'O', 'O', 'O']
BPA = 1.0 / 0.52917721092
STEPS = (0.02, 0.04, 0.06, 0.08)


def save_json_atomic(path, obj):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(obj, fh, indent=2, default=str)
    os.replace(tmp, path)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for blk in iter(lambda: fh.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def sha_arr(a):
    return hashlib.sha256(np.asarray(a, float).tobytes()).hexdigest()[:16]


def check_number():
    used = set()
    for f in glob.glob(JOBS + '/JOB-*.md'):
        m = re.match(r'JOB-2026-0906-(\d{3})', os.path.basename(f))
        if m:
            used.add(m.group(1))
    nxt = '%03d' % (max(int(u) for u in used) + 1)
    return dict(used_max=max(used), next_free=nxt,
                pass_=bool(nxt == JOB_NO and JOB_NO not in used))


def correction_index_054():
    """Machine-readable correction of the 054 last_evaluated_point field."""
    res_path = R54 + '/relax054_results.json'
    res = json.load(open(res_path))
    led = json.load(open(R54 + '/budget_relax054.json'))
    opt_atts = [a for a in led['attempts'] if a['category'] == 'opt']
    # time ordering from COMPLETION records ('finished'), not energy
    ordered = sorted(opt_atts, key=lambda a: a['finished'])
    last_tag = ordered[-1]['note']['tag']
    # accepted iterates from the callback
    acc = res['optimization']['accepted_iterates']
    n_acc = len(acc)
    # the last accepted iterate must match the last completed eval's coords
    last_eval = json.load(open('%s/eval_opt_%s.json' % (R54, last_tag)))
    x_last = np.asarray(last_eval['coords_actual_angstrom'], float) * BPA
    acc_match = float(np.abs(np.asarray(acc[-1], float) - x_last.reshape(-1))
                      .max())
    # full-precision displacement recomputation (SEPARATE components)
    q = np.asarray(json.load(open(Q_FILE))['direction']['q'], float)
    c53 = json.load(open(R53 + '/eval_centre.json'))
    p02 = json.load(open(R53R + '/eval_disp_p02.json'))
    x25 = np.asarray(res and last_eval['coords_actual_angstrom'],
                     float) * BPA

    def decomp(x_ref):
        d = (x25 - x_ref).reshape(-1)
        a = float(d @ q)
        t = float(np.linalg.norm(d))
        return dict(along_q_Bohr=a, perpendicular_Bohr=float(
            np.sqrt(max(t * t - a * a, 0.0))), total_Bohr=t)
    disp = dict(vs_053_centre=decomp(np.asarray(
        c53['coords_actual_angstrom'], float) * BPA),
        vs_p02_start=decomp(np.asarray(
            p02['coords_actual_angstrom'], float) * BPA))
    idx = dict(
        what='machine-readable correction of relax054_results.json '
             'last_evaluated_point (original JSON preserved as-is; its '
             'hash snapshot below)',
        original_file_sha256=sha256_file(res_path),
        original_wrong_field=dict(last_evaluated_point=res['final'][
            'last_evaluated_point']),
        corrected=dict(last_evaluated_point=last_tag,
                       last_accepted_point=last_tag,
                       evidence=dict(
                           rule='time ordering from ledger completion '
                                'records (finished timestamps); energy '
                                'extremes NEVER used for ordering',
                           n_opt_attempts=len(opt_atts),
                           n_accepted_iterates_callback=n_acc,
                           last_accepted_vs_opt_coords_maxdiff_Bohr=acc_match,
                           accepted_iterates_count_matches=bool(
                               n_acc == len(opt_atts)))),
        displacement_recompute=dict(
            method='full-precision coords; along-q / perpendicular / total '
                   'reported SEPARATELY; the net-displacement direction '
                   'ratio is NOT used for energy attribution',
            values=disp),
        note_054_report='the 054 report table listed the along-q and total '
                        'components but not the perpendicular component '
                        'separately; this index supersedes')
    save_json_atomic(R54 + '/correction_index_055.json', idx)
    return idx, acc_match, disp


def verify_centre_and_q():
    cen = json.load(open(CENTRE_FILE))
    qj = json.load(open(Q_FILE))
    checks = {}
    checks['e_matches'] = bool(float(cen['e_total']) == -282.0017216061232)
    checks['gmax_matches'] = bool(float(cen['grad_max'])
                                  == 1.8750279317234203e-5)
    checks['converged_finite'] = bool(cen['converged']
                                      and cen['all_finite'])
    checks['x_bohr_saved'] = 'x_bohr' in cen
    checks['config_complete'] = bool(
        cen['config'].get('grid_response') is True
        and cen['config'].get('grid_level') == 8
        and cen['config'].get('d2_attached') is True)
    q = np.asarray(qj['direction']['q'], float).reshape(-1)
    checks['q_norm_1'] = bool(abs(float(np.linalg.norm(q)) - 1.0) < 1e-12)
    checks['q_hash_match'] = bool(qj['direction']['q_sha'] == sha_arr(q))
    checks['q_is_cartesian_converted'] = bool(
        abs(float(qj['direction']['mass_weighted_norm_of_saved_d'])
            - 1.0) < 1e-8)
    R0 = np.asarray(cen['coords_actual_angstrom'], float) * BPA
    geoms = []
    for t in STEPS:
        R = R0 + t * q.reshape(7, 3)
        d = (R - R0).reshape(-1)
        proj = float(d @ q)
        # fragment geometry sanity
        Rm = R.reshape(7, 3)
        dNH = [float(np.linalg.norm(Rm[i] - Rm[0])) for i in (1, 2, 3)]
        dOO = float(np.linalg.norm(Rm[5] - Rm[4]))
        dmin = float(min(np.linalg.norm(Rm[i] - Rm[j])
                         for i in range(7) for j in range(i + 1, 7)))
        geoms.append(dict(
            t_Bohr=t, coords_bohr=R.tolist(), coords_bohr_sha=sha_arr(R),
            step_norm_Bohr=float(np.linalg.norm(d)),
            displacement_along_q_Bohr=proj,
            max_atom_displacement_Bohr=float(np.abs(
                (R - R0)).max()),
            NH_bonds_Bohr=dNH, O3_bond_Bohr=dOO,
            min_interatomic_distance_Bohr=dmin,
            element_order=SYMS,
            unit='Bohr (full-precision floats, no decimal formatting)'))
    checks['fragments_intact_all'] = bool(all(
        min(g['NH_bonds_Bohr']) > 1.5 and g['O3_bond_Bohr'] > 2.0
        and g['min_interatomic_distance_Bohr'] > 1.5 for g in geoms))
    checks['norms_match_t'] = bool(all(
        abs(g['step_norm_Bohr'] - g['t_Bohr']) < 1e-12
        and abs(g['displacement_along_q_Bohr'] - g['t_Bohr']) < 1e-12
        for g in geoms))
    all_pass = all(bool(v) for v in checks.values() if isinstance(v, bool))
    return cen, q, R0, geoms, checks, all_pass, qj['direction']['q_sha']


def main():
    num = check_number()
    idx, acc_match, disp = correction_index_054()
    cen, q, R0, geoms, checks, ok, q_sha = verify_centre_and_q()
    man = dict(
        job='JOB-2026-0906-055 four-point energy-gradient scan along the '
            'fixed q from the 054 opt_25 endpoint',
        number_check=num,
        caps=dict(centre=1, scan=4, total=5,
                  note='failures count within their category; no borrowing; '
                       'on exception or non-convergence: save and STOP, no '
                       'retry, no auto extra points, no offline repair'),
        correction_index_054='correction_index_055.json (in the 054 dir)',
        centre=dict(path_wsl=CENTRE_FILE, sha256=sha256_file(CENTRE_FILE),
                    e_total=float(cen['e_total']),
                    grad_max=float(cen['grad_max']),
                    x_bohr=cen['x_bohr'],
                    coords_actual_angstrom=cen['coords_actual_angstrom'],
                    grad=cen['grad'], config=cen['config'],
                    note='unconverged scan start: max|g| > 1e-5 is NOT a '
                         'gate failure'),
        centre_gate=dict(dE_max=1e-8, dgrad_max=1e-7, dcoords_max_A=1e-9,
                         extra='SCF converged, finite, config identical'),
        direction=dict(source='053 direction_and_kH.json (q_sha %s)'
                       % q_sha,
                       q=q.tolist(), q_sha=q_sha,
                       already_cartesian=True,
                       note='no mass conversion; sign and element order '
                            'preserved; the new line is centred at R* '
                            '(054 opt_25) and is NOT merged with the 053 '
                            'R0+tq path'),
        scan_geometries=geoms,
        checks=checks, all_checks_pass=ok,
        analysis_spec=dict(per_point=['dE_vs_centre', 'grad_max',
                                      'a_t = g(R(t)).q  (Eh/Bohr)'],
                           rules=['analytic directional derivative a(t) is '
                                  'DISTINCT from adjacent-point secant '
                                  'slopes (a secant is not the derivative '
                                  'of an endpoint)',
                                  'sign change of a(t) -> register "a '
                                  'sign-change interval on this fixed line '
                                  'pending refinement" ONLY',
                                  'no sign change or boundary minimum -> '
                                  'report as-is, NO extrapolation',
                                  'no old-Hessian frequencies or minimum '
                                  'predictions at the new centre']))
    save_json_atomic(OUT + '/input_manifest.json', man)
    print('[055] number check:', json.dumps(num))
    print('[055] correction index: last_eval=%s accepted=%d '
          'acc_vs_coords=%.2e' % (idx['corrected']['last_evaluated_point'],
                                  idx['corrected']['evidence'][
                                      'n_accepted_iterates_callback'],
                                  acc_match))
    print('[055] opt_25 displacements (full precision):')
    for k, v in disp.items():
        print('      %s: along=%+.6e perp=%.6e total=%.6e Bohr'
              % (k, v['along_q_Bohr'], v['perpendicular_Bohr'],
                 v['total_Bohr']))
    print('[055] centre/q checks:', json.dumps(
        {k: v for k, v in checks.items()}, default=str))
    if not (num['pass_'] and ok):
        print('[055] PRECHECK FAILED -> STOP (zero-evaluation stage)')
        raise SystemExit(1)
    print('[055] PREP OK (zero evaluations consumed)', flush=True)


if __name__ == '__main__':
    main()
