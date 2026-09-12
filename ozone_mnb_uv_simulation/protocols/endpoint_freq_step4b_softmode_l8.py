#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JOB-2026-0906-005 Phase C step 3: ONE independent lowest-mode direction
check at the NEW c14_plus L8 endpoint, on grid L8.

4 displaced points only: max per-atom displacement +/-0.002, +/-0.004
Angstrom along the lowest internal mode (from
frequency_analysis_l8_c14.json), using the shared unit-fixed production
helpers.  Compares energy curvature, gradient-difference curvature and the
same-direction projection p^T H p of the h=0.002 symmetrized Cartesian
Hessian.  The L8 centre values come from the Phase-B independent
verification (same grid, same geometry).
"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, 'run_artifacts', '01_pure_water_o3_h2o')
sys.path.insert(0, HERE)

import d2_full
import curvature_audit_lib as cal
from grad_factory import make_mf_d2_gr
from pyscf import gto

SYMS = ['O', 'O', 'O', 'O', 'H', 'H']
BOHR_A = 0.52917721092
AMPS_A = [0.002, 0.004]
GRID_LEVEL = 8


def main():
    freq = json.load(open(os.path.join(ART, 'endpoint_frequency',
                                       'frequency_analysis_l8_c14.json')))
    res = json.load(open(os.path.join(ART, 'c14_l8_check',
                                      'l8_optimize_result.json')))
    coords_A = np.asarray(res['endpoint_coords_angstrom'], float)
    centre = res['independent_verification']       # L8 centre (same grid)
    e0 = centre['e_total']
    g0 = np.asarray(centre['gradient_full_6x3'], float).reshape(-1)
    Hsym = np.asarray(freq['per_step']['0.002']['hessian_symmetrized'])

    nm = np.asarray(freq['per_step']['0.002']['norm_mode'][0], float)
    p18 = nm.reshape(-1)
    p_unit = p18 / np.linalg.norm(p18)
    hess_proj = float(p_unit.dot(Hsym).dot(p_unit))
    s_g = float(p_unit.dot(g0))

    mol = gto.M(atom="; ".join("%s %.10f %.10f %.10f"
                               % (s, x, y, z)
                               for s, (x, y, z) in zip(SYMS, coords_A)),
                basis='def2-TZVP', charge=0, spin=0, verbose=0,
                max_memory=4000)
    mf = make_mf_d2_gr(mol, solvent=None, grid_response=True)
    mf.grids.level = GRID_LEVEL
    scanner = mf.nuc_grad_method().as_scanner()
    assert bool(getattr(scanner, 'grid_response'))

    rows = []
    for amp in AMPS_A:
        for sgn in (1, -1):
            q = cal.q_for_atom_disp(nm, amp)
            c_new = cal.displaced_geometry(coords_A, nm, sgn * q,
                                           unit='Angstrom')
            mol.set_geom_(np.asarray(c_new, float), unit='Angstrom')
            t0 = time.time()
            e, g = scanner(mol)
            g = np.asarray(g, float).reshape(-1)
            mol.set_geom_(np.asarray(c_new, float), unit='Angstrom')
            act = np.asarray(mol.atom_coords(unit='Bohr'), float)
            delta_B = (act - coords_A / BOHR_A).reshape(-1)
            rows.append(dict(
                amp_a=amp, sign=sgn, coords_angstrom=c_new.tolist(),
                q_actual_bohr=float(np.linalg.norm(delta_B)),
                e_total=float(e), gradient_full_6x3=g.reshape(6, 3).tolist(),
                grad_max=float(np.abs(g).max()),
                scf_converged=bool(scanner.converged),
                grid_response_actual=bool(getattr(scanner,
                                                  'grid_response')),
                seconds=round(time.time() - t0, 1)))
            print('[c14L8 soft] a=%g sgn%+d  E=%.9f  max|g|=%.3e'
                  % (amp, sgn, e, rows[-1]['grad_max']), flush=True)

    analysis = {}
    rs = {(r['amp_a'], r['sign']): r for r in rows}
    for amp in AMPS_A:
        rp, rm = rs[(amp, 1)], rs[(amp, -1)]
        q = rp['q_actual_bohr']
        assert abs(q - rm['q_actual_bohr']) < 1e-9
        gp = np.asarray(rp['gradient_full_6x3'], float).reshape(-1)
        gm = np.asarray(rm['gradient_full_6x3'], float).reshape(-1)
        analysis['a%g' % amp] = dict(
            q_actual_bohr=q,
            s_E=float(rp['e_total'] - rm['e_total']) / (2 * q),
            s_g_center=s_g,
            k_E=float(rp['e_total'] + rm['e_total'] - 2 * e0) / q ** 2,
            k_g=float(p_unit.dot(gp - gm)) / (2 * q),
            hessian_projection=hess_proj)
    for k, v in analysis.items():
        v['slope_check_resid'] = v['s_E'] - v['s_g_center']
        v['k_E_minus_k_g'] = v['k_E'] - v['k_g']
    out = dict(job='JOB-2026-0906-005 Phase C lowest-mode L8 check',
               mode_index=0,
               wn_L8_h002=freq['per_step']['0.002']['freq_wavenumber_signed'][0],
               wn_L8_h004=freq['per_step']['0.004']['freq_wavenumber_signed'][0],
               p_unit=p_unit.tolist(),
               hessian_projection_Eh_bohr2=hess_proj,
               centre_source='Phase B independent_verification (L8)',
               centre_grad_max=centre['grad_max'],
               points=rows, analysis=analysis)
    with open(os.path.join(ART, 'endpoint_frequency',
                           'softmode_l8_c14.json'), 'w') as fh:
        json.dump(out, fh, indent=2)
    for k, v in analysis.items():
        print('  %s: k_E=%+.4e k_g=%+.4e Hproj=%+.4e slope_resid=%+.1e'
              % (k, v['k_E'], v['k_g'], v['hessian_projection'],
                 v['slope_check_resid']), flush=True)
    print('saved -> softmode_l8_c14.json')


if __name__ == '__main__':
    main()
