#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-049: trajectory convergence evaluation (ZERO evaluations).

Offline analysis of the already saved JOB-048 full trajectory:
  A. accepted-iterate <-> evaluated-point mapping (bit-exact x matching),
     trial points without accepted record;
  B. per-eval table: E, max|g|, ||g||; accepted steps, adjacent dE,
     gradient-vector changes; budget stop vs normal return recorded
     separately;
  C. translation/rotation/internal gradient decomposition (validated
     torque_decomp component) at start / min-grad / last accepted / 041
     centre; NH3 and O3 fragment forces, internal deformation (Kabsch) and
     inter-fragment relative motion;
  D. 1-D secant curvature along accepted steps (bounded interpretation);
  E. 043 saved Hessian as a LIMITED reference: directional curvature along
     the 045 mode (sanity) and along the 048 start->opt_19 net displacement
     (evaluated at the 041-attempt-5 geometry, NOT at 048 points);
  F. read-only scipy L-BFGS-B interface facts (no preconditioner / H0
     interface; maxls semantics).

No SCF, no gradient, no optimization: reads existing records only.
"""
import os, sys, json, glob
import numpy as np

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
sys.path.insert(0, ROOT + '/protocols')
S48 = ROOT + '/run_artifacts/02_nh3o3_reference/c1_fulldof_opt048'
S45 = ROOT + '/run_artifacts/02_nh3o3_reference/c1_negmode_scan045'
S43 = ROOT + '/run_artifacts/02_nh3o3_reference/c1_nondiag_freq'
S41 = ROOT + '/run_artifacts/02_nh3o3_reference/c1_cart_exec_v3'
OUT = ROOT + '/run_artifacts/02_nh3o3_reference/c1_traj_eval049'

from torque_decomp import decompose_force

MASSES = np.array([14.003074, 1.007825, 1.007825, 1.007825,
                   15.994915, 15.994915, 15.994915])   # amu (043 convention)
FRAG_NH3, FRAG_O3 = range(0, 4), range(4, 7)


def save_json_atomic(path, obj):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(obj, fh, indent=2, default=str)
    os.replace(tmp, path)


def load_evals():
    recs = {}
    for f in glob.glob(S48 + '/eval_*.json'):
        r = json.load(open(f))
        recs[r['tag']] = r
    order = ['start'] + ['opt_%02d' % i for i in range(1, 20)]
    return [(t, recs[t]) for t in order if t in recs]


def kabsch(P, Q):
    """RMSD + rotation angle after optimal rigid alignment P->Q (no
    translation removal for fragments? COM removed for both)."""
    P = np.asarray(P, float); Q = np.asarray(Q, float)
    Pc = P - P.mean(0); Qc = Q - Q.mean(0)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    diff = (R @ Pc.T).T - Qc
    rmsd = float(np.sqrt((diff ** 2).sum() / len(P)))
    ang = float(np.degrees(np.arccos(np.clip(
        (np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))
    return rmsd, ang


def main():
    res = {}
    evals = load_evals()
    r48 = json.load(open(S48 + '/fulldof_opt048_results.json'))
    acc = [np.asarray(a, float) for a in r48['optimization']['accepted_iterates']]

    # ---------- A. mapping accepted <-> evaluated ----------
    mapping, trial_only = [], []
    x_by_tag = {t: np.asarray(r['x_bohr'], float).reshape(-1)
                for t, r in evals}
    acc_tags = []
    for k, xa in enumerate(acc):
        hit = sorted(t for t, x in x_by_tag.items() if np.array_equal(x, xa))
        mapping.append(dict(accepted_index=k, matched_tag=hit,
                            n_matches=len(hit)))
        if len(hit) == 1:
            acc_tags.append(hit[0])
    for t in x_by_tag:
        if t != 'start' and t not in acc_tags:
            trial_only.append(t)
    res['mapping'] = dict(
        n_accepted_saved=len(acc), n_accepted_matched=len(acc_tags),
        accepted_tags_in_order=acc_tags, trial_only_tags=trial_only,
        n_eval_files=len(evals),
        mapping_exact_bitwise=all(
            len(m['matched_tag']) == 1 for m in mapping),
        note='accepted iterates matched to evaluated records by bit-exact '
             '21-component coordinate equality')

    # ---------- B. trajectory tables ----------
    def row(t, r):
        g = np.asarray(r['grad'], float)
        return dict(tag=t, e_total=float(r['e_total']),
                    grad_max=float(r['grad_max']),
                    grad_norm=float(np.linalg.norm(g)))

    table = [row(t, r) for t, r in evals]
    E = {d['tag']: d['e_total'] for d in table}
    G = {t: np.asarray(r['grad'], float).reshape(-1) for t, r in evals}
    X = x_by_tag

    seq = []
    prev = 'start'
    for t in acc_tags:
        d = row(t, None) if False else dict(tag=t, e_total=E[t],
                                            grad_max=float(np.abs(G[t]).max()),
                                            grad_norm=float(np.linalg.norm(G[t])))
        if prev is not None:
            dx = X[t] - X[prev]
            s = float(np.linalg.norm(dx)); u = dx / s
            dE = E[t] - E[prev]
            dg = G[t] - G[prev]
            gproj_prev = float(G[prev] @ u)
            d.update(step_Bohr=s, dE=dE,
                     dE_minus_gproj_over_s2=2.0 * (dE - gproj_prev * s) / s ** 2,
                     gproj_along_step_prev=gproj_prev,
                     dgrad_norm=float(np.linalg.norm(dg)),
                     grad_cos=float(G[prev] @ G[t]
                                    / (np.linalg.norm(G[prev])
                                       * np.linalg.norm(G[t]))))
        seq.append(d)
        prev = t
    res['accepted_sequence'] = seq
    res['trial_only_detail'] = []
    for t in trial_only:
        idx = [i for i, (tt, _) in enumerate(evals) if tt == t][0]
        prev_acc = [a for a in acc_tags
                    if [tt for tt, _ in evals].index(a) < idx]
        pa = prev_acc[-1] if prev_acc else 'start'
        dx = X[t] - X[pa]; s = float(np.linalg.norm(dx))
        res['trial_only_detail'].append(dict(
            tag=t, dE_vs_prev_accepted=E[t] - E[pa],
            step_from_prev_accepted_Bohr=s,
            gproj_along_step=float(G[pa] @ dx / s),
            higher_energy_than_prev_accepted=bool(E[t] > E[pa])))
    res['stop_reason'] = dict(
        job048='budget: opt attempt cap 19 reached (BudgetExhausted '
               'exception path, controlled); optimizer did NOT return '
               'normally -> no nit available, none fabricated',
        budget_vs_normal='category recorded: 048=budget-exception stop; '
                         '041 attempt5 was itself a budget stop of its '
                         'own batch; no normal-return optimizer stop has '
                         'been observed for this surface yet')

    # ---------- C. decomposition ----------
    decomp = {}
    src45 = json.load(open(S45 + '/eval_eval_displacement_+0.02.json'))
    man45 = json.load(open(S45 + '/input_manifest.json'))
    pts = dict(start=src45, opt_04=evals[[t for t, _ in evals].index('opt_04')][1],
               opt_19=evals[[t for t, _ in evals].index('opt_19')][1])
    e41 = json.load(open(S41 + '/eval_records.json'))['50571026adb2204d']
    pts['centre_041_att5'] = e41
    X0 = np.asarray(src45['coords_actual_angstrom'], float) / 0.52917721092
    for name, rec in pts.items():
        if 'coords_actual_angstrom' in rec:
            C = np.asarray(rec['coords_actual_angstrom'], float)
        elif 'coords_angstrom' in rec:
            C = np.asarray(rec['coords_angstrom'], float)
        elif 'coords_bohr' in rec:
            C = np.asarray(rec['coords_bohr'], float) * 0.52917721092
        else:
            raise KeyError('no coordinates in record %s' % name)
        Cb = C / 0.52917721092
        g = np.asarray(rec.get('grad', rec.get('grad_full')), float)
        g = g.reshape(7, 3)
        dd = decompose_force(g, Cb, MASSES, unit='Bohr')
        gf = g
        # per-fragment net force and torque about each fragment COM
        # (relative fragment translation / libration / internal distortion)
        frag_torque = {}
        for fn, idx in (('NH3', FRAG_NH3), ('O3', FRAG_O3)):
            cf = Cb[list(idx)]
            comf = np.average(cf, axis=0, weights=MASSES[list(idx)])
            T = np.cross(cf - comf, gf[list(idx)]).sum(0)
            frag_torque[fn] = dict(
                net_force_norm=float(np.linalg.norm(gf[list(idx)].sum(0))),
                torque_norm_Eh=float(np.linalg.norm(T)))
        dd['fragment_net_force'] = frag_torque
        decomp[name] = dd
    # fragment geometry changes vs 048 start
    frag = {}
    for name in ('opt_04', 'opt_19'):
        Cb = X[name].reshape(7, 3)
        dC = dict()
        for fn, idx in (('NH3', FRAG_NH3), ('O3', FRAG_O3)):
            rmsd, ang = kabsch(X0.reshape(7, 3)[list(idx)], Cb[list(idx)])
            dC[fn] = dict(internal_rmsd_Bohr=rmsd, rigid_rotation_deg=ang)
        com = lambda c, idx: np.average(c[list(idx)], axis=0)
        dC['interfragment_COM_distance_Bohr'] = float(np.linalg.norm(
            com(Cb, FRAG_O3) - com(Cb, FRAG_NH3)))
        dC['interfragment_COM_distance_start_Bohr'] = float(np.linalg.norm(
            com(X0.reshape(7, 3), FRAG_O3) - com(X0.reshape(7, 3), FRAG_NH3)))
        dC['net_displacement_Bohr'] = float(np.linalg.norm(Cb - X0.reshape(7, 3)))
        q = np.asarray(man45['direction']['q_vector'], float)
        dv = (Cb - X0.reshape(7, 3)).reshape(-1)
        dC['net_displacement_along_q_Bohr'] = float(dv @ q / np.linalg.norm(q))
        frag[name] = dC
    res['decomposition'] = decomp
    res['fragment_changes_vs_start'] = frag

    # ---------- D/E. 043 Hessian limited reference ----------
    h43 = json.load(open(S43 + '/nondiag_results.json'))
    Hd = np.asarray(h43['dft_hessian']['matrix'], float).reshape(21, 21)
    H2 = np.asarray(h43['d2_hessian']['matrix'], float).reshape(21, 21) \
        if 'd2_hessian' in h43 else None
    H = Hd + H2 if H2 is not None else Hd
    res['hessian_043_reference'] = dict(
        shape_reported=h43['dft_hessian']['shape'],
        unit=h43['dft_hessian']['unit'],
        method=h43['dft_hessian']['method'],
        includes_d2=H2 is not None,
        geometry='041 attempt 5 centre (NOT the 048 points)',
        limit='bounded reference only; 045 showed the analytic Hessian '
              'curvature differs from FD by ~3x along the 045 mode; not a '
              'validated local model for 048 endpoints')
    def dir_curv(u, project_tr):
        u = np.asarray(u, float).reshape(-1)
        u = u / np.linalg.norm(u)
        if project_tr:
            Cb = np.asarray(e41['coords_bohr'], float).reshape(7, 3)
            rc = np.average(Cb, axis=0, weights=MASSES)
            rows = []
            for a in range(3):
                v = np.zeros((7, 3)); v[:, a] = 1.0
                rows.append(v.reshape(-1))
            for a in range(3):
                e = np.zeros(3); e[a] = 1.0
                rows.append(np.cross(Cb - rc, np.tile(e, (7, 1))).reshape(-1))
            M = np.vstack(rows)
            Q, _ = np.linalg.qr(M.T)
            u = u - Q @ (Q.T @ u)
            u = u / np.linalg.norm(u)
        return float(u @ H @ u), float(np.linalg.norm(u))
    q = np.asarray(man45['direction']['q_vector'], float)
    kq, _ = dir_curv(q, False)
    dnet = (X['opt_19'] - X['start'])
    knet_raw, nu1 = dir_curv(dnet, False)
    knet_tr, nu2 = dir_curv(dnet, True)
    knet_dft_only = float((lambda u: (u / np.linalg.norm(u)) @ Hd @
                           (u / np.linalg.norm(u)))(dnet))
    res['hessian_043_reference'].update(
        k_along_045_mode_combined_Eh_Bohr2=kq,
        k_along_045_mode_dft_only_Eh_Bohr2=float(
            (lambda u: u @ Hd @ u)(q / np.linalg.norm(q))),
        k_along_045_sanity_note='combined value expected ~ -2.086e-4 '
                                '(043/045 k_H = dft + d2); mismatch would '
                                'indicate a handling error',
        k_along_048_net_displacement_cart_combined_Eh_Bohr2=knet_raw,
        k_along_048_net_displacement_cart_dft_only_Eh_Bohr2=knet_dft_only,
        k_along_048_net_displacement_trprojected_combined_Eh_Bohr2=knet_tr,
        trprojected_norm=nu2)

    # ---------- F. scipy L-BFGS-B interface facts (read-only) ----------
    import scipy.optimize as so
    import inspect
    src = inspect.getsource(so.optimize._lbfgsb._minimize_lbfgsb) \
        if hasattr(so.optimize, '_lbfgsb') else ''
    facts = dict(options=['gtol (pgtol)', 'ftol (factr-derived)',
                          'maxcor', 'maxiter', 'maxfun', 'maxls',
                          'epsilon (FD interval for numeric jac only)'],
                 no_initial_hessian_interface=True,
                 no_preconditioner_interface=True,
                 maxls_semantics='max line-search steps PER ITERATION; '
                                 'NOT a geometry step-size cap; raising it '
                                 'does not limit steps',
                 note='read from the installed scipy 1.18.0 source; '
                      'documented here to prevent misuse')
    try:
        import re
        m = re.findall(r"'(\w+)':", src[:4000])
        if m:
            facts['options_seen_in_source'] = sorted(set(m))
    except Exception:
        pass
    res['scipy_lbfgsb_interface'] = facts

    res['budget'] = dict(evaluations_this_batch=0,
                         note='offline analysis only; reads existing '
                              'records; no SCF/gradient/opt/Hessian run')
    save_json_atomic(OUT + '/traj_analysis.json', res)

    # console summary
    print('accepted tags:', acc_tags)
    print('trial-only:', trial_only, '| mapping exact:', res['mapping']['mapping_exact_bitwise'])
    print('%-8s %14s %10s %10s %9s %10s %9s' % ('tag', 'E', 'gmax', '|g|', 'step', 'dE', 'k_eff'))
    for d in seq:
        print('%-8s %14.9f %9.2e %9.2e %9.5f %10.3e %9.3e' % (
            d['tag'], d['e_total'], d['grad_max'], d['grad_norm'],
            d.get('step_Bohr', float('nan')), d.get('dE', float('nan')),
            d.get('dE_minus_gproj_over_s2', float('nan'))))
    for d in res['trial_only_detail']:
        print('TRIAL %-6s dE=%+.3e step=%.5f gproj=%+.3e higherE=%s'
              % (d['tag'], d['dE_vs_prev_accepted'],
                 d['step_from_prev_accepted_Bohr'],
                 d['gproj_along_step'], d['higher_energy_than_prev_accepted']))
    for name in ('start', 'opt_04', 'opt_19', 'centre_041_att5'):
        dd = decomp[name]
        print('%-16s |g|=%.3e T=%.2e R=%.2e INT=%.3e (frac %.4f) '
              'NH3: F=%.2e T=%.2e | O3: F=%.2e T=%.2e'
              % (name, dd['g_norm'], dd['g_trans_norm'], dd['g_rot_norm'],
                 dd['g_int_norm'], dd['int_norm_fraction'],
                 dd['fragment_net_force']['NH3']['net_force_norm'],
                 dd['fragment_net_force']['NH3']['torque_norm_Eh'],
                 dd['fragment_net_force']['O3']['net_force_norm'],
                 dd['fragment_net_force']['O3']['torque_norm_Eh']))
    for name, fc in frag.items():
        print(name, json.dumps(fc))
    print('043 combined k(q045)=%.4e k(net,raw)=%.4e k(net,trproj)=%.4e'
          % (kq, knet_raw, knet_tr))


if __name__ == '__main__':
    main()
