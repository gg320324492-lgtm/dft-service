#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Endpoint-freq batch Step 4 (JOB-2026-0906-004):
limited independent soft-mode check at the official endpoints.

Per candidate at most TWO Cartesian mode directions: the lowest internal
mode plus any negative / step-size-sensitive mode (from
frequency_analysis.json).  For each direction, displacements with max
per-atom displacement-VECTOR length = +/-0.002, +/-0.004 Angstrom via the
shared unit-fixed production helpers (cal.q_for_atom_disp +
cal.displaced_geometry), evaluated on BOTH grid level 7 and 8.  Each grid
uses its OWN centre (L7 centre from endpoints_fixed.json; L8 centre
computed here).

Analysis per (grid, amplitude) recovers q from the ACTUAL coordinates.
Checks: energy slope vs centre projected gradient; energy curvature vs
gradient-difference curvature; L7 directional curvature vs p^T H p of the
h=0.002 symmetrized Cartesian Hessian; step-size trend and L7/L8 sign
changes.  L8 = same-geometry sensitivity check only (no re-optimisation);
the L8 centre stationary condition is recorded.

Budget: 2 dirs * 2 amps * 2 signs * 2 grids = 16 displaced points per
candidate (32 total) + 2 L8 centres.  No third grid / more steps / reopt.
"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, 'run_artifacts', '01_pure_water_o3_h2o')
OUT = os.path.join(ART, 'endpoint_frequency', 'softmode_points')
sys.path.insert(0, HERE)

from pyscf import gto
import d2_full
import curvature_audit_lib as cal
from grad_factory import make_mf_d2_gr

SYMS = ['O', 'O', 'O', 'O', 'H', 'H']
BOHR_A = 0.52917721092
AMPS_A = [0.002, 0.004]
GRIDS = [7, 8]


def make_driver(coords_A, grid_level):
    atom = "; ".join("%s %.10f %.10f %.10f" % (s, x, y, z)
                     for s, (x, y, z) in zip(SYMS, coords_A))
    mol = gto.M(atom=atom, basis='def2-TZVP', charge=0, spin=0,
                verbose=0, max_memory=4000)
    mf = make_mf_d2_gr(mol, solvent=None, grid_response=True)
    mf.grids.level = grid_level
    g = mf.nuc_grad_method()               # D2-attached, grid_response=True
    scanner = g.as_scanner()
    assert bool(getattr(scanner, 'grid_response'))
    return mol, mf, scanner


def eval_point(mol, scanner, grid_level, coords_A):
    mol.set_geom_(np.asarray(coords_A, float), unit='Angstrom')
    t0 = time.time()
    e, g = scanner(mol)
    g = np.asarray(g, float).reshape(6, 3)
    return dict(e_total=float(e), gradient_full_6x3=g.tolist(),
                grad_max=float(np.abs(g).max()),
                scf_converged=bool(scanner.converged),
                grid_response_actual=bool(getattr(scanner, 'grid_response')),
                grid_level=grid_level,
                seconds=round(time.time() - t0, 1))


def main():
    os.makedirs(OUT, exist_ok=True)
    fx = json.load(open(os.path.join(ART, 'endpoint_frequency',
                                     'endpoints_fixed.json')))
    freq = json.load(open(os.path.join(ART, 'endpoint_frequency',
                                       'frequency_analysis.json')))
    out = dict(job='JOB-2026-0906-004', candidates={})
    for tid in ('c14_plus', 'c06_plus'):
        fc = freq['candidates'][tid]
        ev1 = np.asarray(fc['per_step']['0.002']
                         ['internal_eigenvalues_eh_bohr2_amu1'])
        ev2 = np.asarray(fc['per_step']['0.004']
                         ['internal_eigenvalues_eh_bohr2_amu1'])
        dirs = [0]                                     # lowest mode
        neg = sorted(set(np.where(ev1 < 0)[0].tolist())
                     | set(np.where(ev2 < 0)[0].tolist()))
        flip = [int(i) for i in range(12) if (ev1[i] < 0) != (ev2[i] < 0)]
        for cand_mode in neg + flip:
            if len(dirs) < 2 and cand_mode not in dirs:
                dirs.append(int(cand_mode))
                break
        dirs = dirs[:2]
        print('[%s] directions=%s (negative=%s sign-flip=%s)'
              % (tid, dirs, neg, flip), flush=True)

        coords_A = np.asarray(fx['endpoints'][tid]['coords_angstrom'], float)
        Hsym = np.asarray(fc['per_step']['0.002']['hessian_symmetrized'])
        cand = dict(direction_selection=dict(indices=dirs, negative_modes=neg,
                                             sign_flip_modes=flip),
                    centres={}, directions={})
        cand['centres']['L7'] = dict(
            e_total=fx['endpoints'][tid]['centre_check']['e_total'],
            gradient_full_6x3=fx['endpoints'][tid]['centre_check']
            ['gradient_full_6x3'],
            grad_max=fx['endpoints'][tid]['centre_check']['grad_max'],
            scf_converged=fx['endpoints'][tid]['centre_check']['scf_converged'],
            source='endpoints_fixed.json centre_check (L7)')
        mol8, mf8, scn8 = make_driver(coords_A, 8)
        c8 = eval_point(mol8, scn8, 8, coords_A)
        c8['stationary_check'] = dict(
            max_g=c8['grad_max'], pass_1e5=bool(c8['grad_max'] <= 1e-5),
            note='L8 centre same-geometry stationary condition')
        cand['centres']['L8'] = c8
        print('[%s] L8 centre: E=%.9f max|g|=%.3e conv=%s'
              % (tid, c8['e_total'], c8['grad_max'], c8['scf_converged']),
              flush=True)
        mol7, mf7, scn7 = make_driver(coords_A, 7)
        drivers = {7: (mol7, scn7), 8: (mol8, scn8)}

        for imode in dirs:
            nm = np.asarray(fc['per_step']['0.002']['norm_mode'][imode],
                            float)                     # (6,3) pattern
            p18 = nm.reshape(-1)
            p_unit = p18 / np.linalg.norm(p18)
            hess_proj = float(p_unit.dot(Hsym).dot(p_unit))   # Eh/Bohr^2
            rows = []
            for gl in GRIDS:
                mol, scanner = drivers[gl]
                for amp in AMPS_A:
                    for sgn in (1, -1):
                        q = cal.q_for_atom_disp(nm, amp)
                        c_new = cal.displaced_geometry(coords_A, nm,
                                                       sgn * q,
                                                       unit='Angstrom')
                        key = 'mode%d_L%d_a%g_sgn%+d' % (imode, gl, amp, sgn)
                        r = eval_point(mol, scanner, gl, c_new)
                        # recover actual displacement in Bohr
                        mol.set_geom_(np.asarray(c_new, float),
                                      unit='Angstrom')
                        act = np.asarray(mol.atom_coords(unit='Bohr'), float)
                        delta_B = (act - np.asarray(coords_A, float)
                                   / BOHR_A).reshape(-1)
                        rows.append(dict(
                            key=key, mode=imode, grid_level=gl, amp_a=amp,
                            sign=sgn, coords_angstrom=c_new.tolist(),
                            q_actual_bohr=float(np.linalg.norm(delta_B)),
                            e_total=r['e_total'],
                            gradient_full_6x3=r['gradient_full_6x3'],
                            grad_max=r['grad_max'],
                            scf_converged=r['scf_converged'],
                            grid_response_actual=r['grid_response_actual'],
                            seconds=r['seconds']))
                        print('[%s] %s E=%.9f max|g|=%.3e (%.1f s)'
                              % (tid, key, r['e_total'], r['grad_max'],
                                 r['seconds']), flush=True)
            analysis = {}
            for gl in GRIDS:
                e0 = cand['centres']['L%d' % gl]['e_total']
                g0 = np.asarray(cand['centres']['L%d' % gl]
                                ['gradient_full_6x3'], float).reshape(-1)
                s_g = float(p_unit.dot(g0))
                rs = {(r['amp_a'], r['sign']): r for r in rows
                      if r['grid_level'] == gl}
                for amp in AMPS_A:
                    rp, rm = rs[(amp, 1)], rs[(amp, -1)]
                    q = rp['q_actual_bohr']
                    assert abs(q - rm['q_actual_bohr']) < 1e-9, 'asymmetric q'
                    gp = np.asarray(rp['gradient_full_6x3'],
                                    float).reshape(-1)
                    gm = np.asarray(rm['gradient_full_6x3'],
                                    float).reshape(-1)
                    analysis['L%d_a%g' % (gl, amp)] = dict(
                        q_actual_bohr=q,
                        s_E=float(rp['e_total'] - rm['e_total']) / (2 * q),
                        s_g_center=s_g,
                        k_E=float(rp['e_total'] + rm['e_total'] - 2 * e0)
                        / q ** 2,
                        k_g=float(p_unit.dot(gp - gm)) / (2 * q),
                        hessian_projection_L7=(hess_proj if gl == 7
                                               else None))
            for k, v in analysis.items():
                v['slope_check_resid'] = v['s_E'] - v['s_g_center']
                v['k_E_minus_k_g'] = v['k_E'] - v['k_g']
            cand['directions']['mode%d' % imode] = dict(
                mode_index=imode,
                wn_L7_h002=fc['per_step']['0.002']
                ['freq_wavenumber_signed'][imode],
                wn_L7_h004=fc['per_step']['0.004']
                ['freq_wavenumber_signed'][imode],
                p_unit=p_unit.tolist(),
                hessian_projection_Eh_bohr2=hess_proj,
                points=rows, analysis=analysis)
            with open(os.path.join(OUT, 'softmode_raw.json'), 'w') as fh:
                json.dump(out, fh, indent=2)     # incremental save
        out['candidates'][tid] = cand
        with open(os.path.join(OUT, 'softmode_raw.json'), 'w') as fh:
            json.dump(out, fh, indent=2)
    print('saved -> softmode_raw.json')


if __name__ == '__main__':
    main()
