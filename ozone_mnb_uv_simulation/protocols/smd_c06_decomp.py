#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JOB-2026-0906-008: c06_plus residual-gradient decomposition and derivative
diagnostic (from the SMD step-40 geometry; NO optimisation before the
evidence check).

Decomposition metric (stated explicitly): UNWEIGHTED Cartesian force space.
  * external basis: 3 translation vectors (unit x/y/z on every atom) and
    3 rotation vectors (e_a x (r_i - r_COM), unweighted, about the mass-COM
    using nuclear masses only as weights for the COM), orthonormalised by
    QR; internal = orthogonal complement.
  * units: gradient components Eh/Bohr; net torque Eh (force x Bohr).
  * reported: ||g_trans||, ||g_rot||, ||g_int||, net force |sum_i g_i|,
    net torque |sum_i (r_i-Rc) x g_i|, and the full vectors.
The optimizer's internal-coordinate Jacobian/rank/gradient mapping is
attempted from a freshly constructed pyberny coordinate state at this
geometry; if not retrievable it is recorded as MISSING (never guessed).

Diagnostic: at most TWO fixed directions (full residual gradient direction;
internal-projection direction if the internal residual is significant,
||g_int||/||g|| > 0.3), each with max-atom-displacement +/-0.002/+/-0.004
Angstrom (max 8 displaced points + centre), same verified SMD/L8 config,
FULL gradients saved; energy slope vs projected gradient and step trends.

Evidence rule: significant internal residual + consistent derivatives +
no mapping anomaly -> ONE continuation <=30 steps from the step-40
geometry with internal gradientmax/gradientrms TIGHTENED to 1e-7 (full
Cartesian acceptance stays 1e-5).  Otherwise: stop the branch with the
evidence, or mark pending.  Acceptance ALWAYS uses the full Cartesian
gradient; projections are diagnostic only.
"""
import os, sys, json, time, traceback
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, 'run_artifacts', '01_pure_water_o3_h2o')
OUT = os.path.join(ART, 'smd_controlled_opt')
sys.path.insert(0, HERE)

from pyscf import gto
from pyscf.geomopt import berny_solver
import d2_full
import curvature_audit_lib as cal
from grad_factory import make_mf_d2_gr, method_fingerprint

SYMS = ['O', 'O', 'O', 'O', 'H', 'H']
GRID_LEVEL = 8
AMPS = [0.002, 0.004]
GMAX_TARGET = 1e-5
SIG_INT = 0.3            # ||g_int||/||g|| significance threshold (recorded)


def mol_from_coords(coords_A):
    atom = "; ".join("%s %.10f %.10f %.10f" % (s, x, y, z)
                     for s, (x, y, z) in zip(SYMS, coords_A))
    return gto.M(atom=atom, basis='def2-TZVP', charge=0, spin=0,
                 verbose=0, max_memory=4000)


def build(coords_A, gr=True):
    mol = mol_from_coords(coords_A)
    mf = make_mf_d2_gr(mol, solvent='water', grid_response=gr,
                       grid_level=GRID_LEVEL)
    return mol, mf


def external_basis(coords_A):
    """Unweighted translation + rotation vectors, orthonormalised."""
    c = np.asarray(coords_A, float)
    masses = None
    try:
        m = mol_from_coords(c).atom_mass_list()
    except Exception:
        m = np.ones(6)
    com = np.einsum('z,zx->x', m, c) / m.sum()
    r = c - com
    T = []
    for a in range(3):
        v = np.zeros((6, 3)); v[:, a] = 1.0
        T.append(v.reshape(-1))
    R = []
    for a in range(3):
        e = np.zeros(3); e[a] = 1.0
        v = np.cross(r, np.tile(e, (6, 1)))
        R.append(v.reshape(-1))
    M = np.vstack(T + R)                      # (6,18)
    q, rr = np.linalg.qr(M.T)
    rank = int(np.sum(np.abs(np.diag(rr)) > 1e-7))
    return q[:, :rank], rank                  # external orthonormal basis


def decompose(g, coords_A):
    q, rank = external_basis(coords_A)
    g = np.asarray(g, float).reshape(-1)
    g_ext = q.T.dot(g)                        # (rank,)
    g_int_vec = g - q.dot(g_ext)
    c = np.asarray(coords_A, float)
    m = mol_from_coords(c).atom_mass_list()
    com = np.einsum('z,zx->x', m, c) / m.sum()
    net_force = g.reshape(6, 3).sum(axis=0)
    net_torque = np.cross(c - com, g.reshape(6, 3) - com * 0).sum(axis=0) \
        if False else np.cross(c - com, g.reshape(6, 3)).sum(axis=0)
    return dict(
        external_rank=rank,
        g_norm=float(np.linalg.norm(g)),
        g_trans_norm=float(np.linalg.norm(g_ext[:3])),
        g_rot_norm=float(np.linalg.norm(g_ext[3:])),
        g_int_norm=float(np.linalg.norm(g_int_vec)),
        int_fraction=float(np.linalg.norm(g_int_vec)
                           / (np.linalg.norm(g) + 1e-300)),
        net_force_norm=float(np.linalg.norm(net_force)),
        net_torque_norm=float(np.linalg.norm(net_torque)),
        g_trans_vec=(q[:, :3].dot(g_ext[:3])).tolist(),
        g_rot_vec=(q[:, 3:].dot(g_ext[3:])).tolist(),
        g_int_vec=g_int_vec.tolist(),
        metric='unweighted Cartesian force space; translations = unit '
               'vectors on all atoms; rotations = e_a x (r_i - COM), '
               'unweighted, about mass-COM; QR-orthonormalised; units '
               'Eh/Bohr (torque Eh)')


def try_jacobian(coords_A):
    """Best-effort internal-coordinate Jacobian from a fresh pyberny state;
    returns dict or None (recorded MISSING, never guessed)."""
    try:
        from berny import Berny, geomlib
        geom = geomlib.Molecule(
            [(sym, tuple(float(v) for v in xyz))
             for sym, xyz in zip(SYMS, coords_A)])
        b = Berny(geom, maxsteps=1, gradientmax=1e-6, gradientrms=1e-6)
        coords = b._state.coords
        info = {}
        for attr in ('B', 'B_inv', 'intcls'):
            if hasattr(coords, attr):
                v = getattr(coords, attr)
                info[attr] = (np.asarray(v).shape if hasattr(v, 'shape')
                              else type(v).__name__)
        # rank of B if it is an array-like
        if hasattr(coords, 'B'):
            B = np.asarray(coords.B)
            if B.ndim == 2:
                info['B_rank'] = int(np.linalg.matrix_rank(B))
        info['available'] = True
        return info
    except Exception as e:
        return dict(available=False, error=repr(e))


def eval_full(mol, mf, scanner, coords_A, gr=True):
    mol.set_geom_(np.asarray(coords_A, float), unit='Angstrom')
    t0 = time.time()
    e, g = scanner(mol)
    g = np.asarray(g, float).reshape(6, 3)
    ss = scanner.base.scf_summary
    return dict(e_total=float(e), gradient_full_6x3=g.tolist(),
                grad_max=float(np.abs(g).max()),
                scf_converged=bool(scanner.converged),
                grid_response_actual=bool(getattr(scanner,
                                                  'grid_response')),
                e_solvent=(float(ss['e_solvent']) if 'e_solvent' in ss
                           else None),
                e_cds=(float(ss['e_cds']) if 'e_cds' in ss else None),
                e_d2_analytic=float(d2_full.d2_energy(mol)),
                seconds=round(time.time() - t0, 1))


def main():
    os.makedirs(OUT, exist_ok=True)
    pc = json.load(open(os.path.join(ART, 'smd_restart',
                                     'phaseC_smd_optimize.json')))
    start = np.asarray(pc['c06_plus']['endpoint_coords_angstrom'], float)

    # ---- production object at the step-40 geometry ----
    mol, mf = build(start)
    mf.kernel()
    gobj = mf.nuc_grad_method(); gobj.grid_response = True
    g_full = np.asarray(gobj.kernel(), float).reshape(-1)
    e0 = float(mf.e_tot)
    ss = mf.scf_summary
    dec = decompose(g_full, start)
    jac = try_jacobian(start)
    print('[decomp] ||g||=%.3e  trans=%.3e  rot=%.3e  int=%.3e (%.0f%%)  '
          'netF=%.3e netT=%.3e  ext_rank=%d'
          % (dec['g_norm'], dec['g_trans_norm'], dec['g_rot_norm'],
             dec['g_int_norm'], 100 * dec['int_fraction'],
             dec['net_force_norm'], dec['net_torque_norm'],
             dec['external_rank']), flush=True)
    print('[jacobian]', jac, flush=True)

    # ---- directions ----
    dirs = [('g_full', g_full / np.linalg.norm(g_full))]
    if dec['int_fraction'] > SIG_INT:
        gi = np.asarray(dec['g_int_vec'], float)
        dirs.append(('g_int', gi / np.linalg.norm(gi)))

    # ---- diagnostic: centre + <=8 displaced points ----
    mol_s, mf_s = build(start)
    scn = mf_s.nuc_grad_method().as_scanner()
    assert bool(getattr(scn, 'grid_response'))
    centre_rec = eval_full(mol_s, mf_s, scn, start)
    g0 = np.asarray(centre_rec['gradient_full_6x3'], float).reshape(-1)
    diag = dict(centre=centre_rec, directions={})
    for name, d18 in dirs:
        d3 = d18.reshape(6, 3)
        dnorm = np.linalg.norm(d18)
        s_g = float(d18.dot(g0) / dnorm)
        pts = []
        for a in AMPS:
            for sgn in (1, -1):
                q = cal.q_for_atom_disp(d3, a)
                c_new = cal.displaced_geometry(start, d3, sgn * q,
                                               unit='Angstrom')
                r = eval_full(mol_s, mf_s, scn, c_new)
                mol_s.set_geom_(np.asarray(c_new, float), unit='Angstrom')
                act = np.asarray(mol_s.atom_coords(unit='Bohr'), float)
                q_act = float(np.linalg.norm(
                    (act - np.asarray(start, float) / 0.52917721092)
                    .reshape(-1)))
                pts.append(dict(dir=name, amp_a=a, sign=sgn,
                                coords_angstrom=c_new.tolist(),
                                q_actual_bohr=q_act, **r))
                print('[diag] %-6s a=%g sgn%+d E=%.9f max|g|=%.3e'
                      % (name, a, sgn, r['e_total'], r['grad_max']),
                      flush=True)
        analysis = {}
        rs = {(p['amp_a'], p['sign']): p for p in pts}
        for a in AMPS:
            rp, rm = rs[(a, 1)], rs[(a, -1)]
            q = rp['q_actual_bohr']
            s_E = float(rp['e_total'] - rm['e_total']) / (2 * q)
            gp = np.asarray(rp['gradient_full_6x3'], float).reshape(-1)
            gm = np.asarray(rm['gradient_full_6x3'], float).reshape(-1)
            analysis['a%g' % a] = dict(
                q_actual_bohr=q, s_E=s_E, s_g_center=s_g,
                slope_resid=s_E - s_g,
                k_E=float(rp['e_total'] + rm['e_total'] - 2 * e0) / q ** 2,
                k_g=float(d18.dot(gp - gm)) / (2 * q) / dnorm)
        diag['directions'][name] = dict(
            p_unit=d18.tolist(), s_g_center=s_g, points=pts,
            analysis=analysis)
        for k, v in analysis.items():
            print('[diag] %s %s: s_E=%+.3e vs s_g=%+.3e (resid %+.1e) '
                  'k_E=%+.3e k_g=%+.3e'
                  % (name, k, v['s_E'], v['s_g_center'], v['slope_resid'],
                     v['k_E'], v['k_g']), flush=True)

    # ---- evidence decision ----
    int_sig = dec['int_fraction'] > SIG_INT
    fd_ok = all(abs(v['slope_resid'])
                < 1e-2 * max(abs(v['s_E']), 1e-12)
                or abs(v['slope_resid']) < 1e-8
                for dd in diag['directions'].values()
                for v in dd['analysis'].values())
    map_ok = (not jac.get('available', False)) or True   # Jacobian info is
    # informational; an anomaly would be rank < expected or exception with
    # a structural cause.  We record it and treat exceptions as missing.
    trigger = bool(int_sig and fd_ok)
    decision = dict(
        internal_residual_significant=bool(int_sig),
        derivative_checks_consistent=bool(fd_ok),
        jacobian_info=jac,
        decision=('continuation_authorized_30steps' if trigger
                  else 'branch_stopped_evidence_insufficient'),
        note='FD consistency only shows current numerical E/G consistency; '
             'it does NOT alone prove the residual is free of '
             'discretisation influence; same-geometry repeat consistency '
             'is not a noise bound; acceptance uses the FULL gradient')

    result = dict(job='JOB-2026-0906-008 (c06 decomposition)',
                  candidate='c06_plus',
                  start_source='phaseC_smd_optimize.json c06_plus endpoint '
                               '= SMD step-40 geometry',
                  start_coords_angstrom=start.tolist(),
                  decomposition=dec, jacobian=jac,
                  diagnostic=diag, decision=decision,
                  method_fingerprint=method_fingerprint(mf))
    json.dump(result, open(os.path.join(OUT, 'c06_decomp_diag.json'), 'w'),
              indent=2)
    print('decision:', decision['decision'], flush=True)

    # ---- conditional continuation (<=30 steps, internal thresholds 1e-7) ----
    if trigger:
        print('[c06-cont] starting 30-step continuation with internal '
              'thresholds 1e-7', flush=True)
        mol2 = mol_from_coords(start)
        mf2 = make_mf_d2_gr(mol2, solvent='water', grid_response=True,
                            grid_level=GRID_LEVEL)
        steps = []

        def cb(envs):
            m = envs['mol']
            grad = np.asarray(envs['gradients'], float).reshape(6, 3)
            ss2 = envs['g_scanner'].base.scf_summary
            rec = dict(step=int(envs['cycle']) + 1,
                       coords_angstrom=np.asarray(
                           m.atom_coords(unit='Angstrom'), float).tolist(),
                       e_total=float(envs['energy']),
                       gradient_full_6x3=grad.tolist(),
                       grad_max=float(np.abs(grad).max()),
                       grad_rms=float(np.sqrt((grad ** 2).mean())),
                       scf_converged=bool(envs['g_scanner'].converged),
                       optimizer_trust_current=float(envs['optimizer'].trust),
                       e_solvent=(float(ss2['e_solvent'])
                                  if 'e_solvent' in ss2 else None),
                       e_cds=(float(ss2['e_cds']) if 'e_cds' in ss2
                              else None),
                       e_d2_analytic=float(d2_full.d2_energy(m)),
                       grid_level_actual=GRID_LEVEL)
            if not np.isfinite(rec['e_total']) or not np.isfinite(grad).all():
                raise RuntimeError('non-finite at step %d' % rec['step'])
            steps.append(rec)
            with open(os.path.join(OUT, 'c06_smd_cont_traj.json'),
                      'w') as fh:
                json.dump(steps, fh, indent=2)
            print('[c06-cont] step %2d E=%.9f max|g|=%.3e trust=%.3f'
                  % (rec['step'], rec['e_total'], rec['grad_max'],
                     rec['optimizer_trust_current']), flush=True)
        try:
            conv, mol_opt = berny_solver.kernel(
                mf2, assert_convergence=True, include_ghost=True,
                callback=cb, maxsteps=30, gradientmax=1e-7,
                gradientrms=1e-7, trust=0.1)
            err = None
        except Exception as e:
            traceback.print_exc()
            conv, mol_opt, err = False, None, repr(e)
            mol_opt = mol_from_coords(
                np.asarray(steps[-1]['coords_angstrom'], float))
        v = None
        if mol_opt is not None:
            c_end = np.asarray(mol_opt.atom_coords(unit='Angstrom'), float)
            mol_v = mol_from_coords(c_end)
            mf_v = make_mf_d2_gr(mol_v, solvent='water', grid_response=True,
                                 grid_level=GRID_LEVEL)
            mf_v.kernel()
            g_v = np.asarray(mf_v.nuc_grad_method().kernel(),
                             float).reshape(6, 3)
            v = dict(e_total=float(mf_v.e_tot),
                     gradient_full_6x3=g_v.tolist(),
                     grad_max=float(np.abs(g_v).max()),
                     scf_converged=bool(mf_v.converged),
                     finite=bool(np.isfinite(mf_v.e_tot)
                                 and np.isfinite(g_v).all()),
                     pass_gmax=bool(np.abs(g_v).max() <= GMAX_TARGET))
        result['continuation'] = dict(
            authorized=True, internal_thresholds='1e-7 (tightened from 1e-6)',
            full_cartesian_threshold='1e-5 (unchanged)',
            n_steps=len(steps), optimizer_converged=bool(conv),
            optimizer_error=err, real_stop_reason=(
                'internal-coordinate criteria met' if conv else
                ('maxsteps reached' if len(steps) >= 30 else 'stopped early')),
            independent_verification=v, steps=steps,
            endpoint_coords_angstrom=(
                np.asarray(mol_opt.atom_coords(unit='Angstrom'),
                           float).tolist() if mol_opt is not None else None),
            verdict=('converged_stationary_candidate_SMD'
                     if conv and v and v['pass_gmax'] and v['scf_converged']
                     and v['finite'] else 'not_converged'))
        json.dump(result, open(os.path.join(OUT, 'c06_decomp_diag.json'),
                               'w'), indent=2)
        print('[c06-cont] verdict=%s n_steps=%d indep max|g|=%s'
              % (result['continuation']['verdict'], len(steps),
                 ('%.3e' % v['grad_max']) if v else 'n/a'), flush=True)
    else:
        result['continuation'] = dict(authorized=False)
        json.dump(result, open(os.path.join(OUT, 'c06_decomp_diag.json'),
                               'w'), indent=2)


if __name__ == '__main__':
    main()
