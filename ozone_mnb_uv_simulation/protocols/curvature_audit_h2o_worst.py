#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Curvature-audit step B3 (JOB-2026-0905-009): probe the WORST-ELEMENT
direction of the wb97xd analytic-vs-FD Hessian discrepancy on H2O.

Step B/B2 established: matrix-level max|H_analytic - H_FD| = 1.9e-3 (wb97xd)
vs 2.8e-5 (pbe) at h=0.005 Bohr, while the O-H stretch DIRECTIONS are
three-layer consistent to ~4e-4 for both functionals.  The discrepancy is
therefore element-specific.  Here we locate the worst element, build the unit
direction d = (e_i + e_j)/sqrt(2) (or e_i for diagonal), and run the
three-layer probe (k_E, k_g, k_H) along it across amplitudes.
"""
import os
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import curvature_audit_lib as cal

ROOT = os.path.dirname(HERE)
FV = os.path.join(ROOT, 'run_artifacts', '01_pure_water_o3_h2o', 'curvature_audit')
os.makedirs(FV, exist_ok=True)

BASIS = 'def2-TZVP'
AMPS_A = [0.005, 0.01, 0.02, 0.05]
ATOMS = ['O', 'H', 'H']


def fresh_mf(mol, xc):
    from pyscf import dft
    mf = dft.RKS(mol)
    mf.xc = xc
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-9
    mf.grids.level = 5
    mf.kernel()
    assert mf.converged
    return mf


def eg_at(mol, xc, coords_bohr):
    from pyscf import dft
    m = mol.copy()
    m.set_geom_(coords_bohr, unit='Bohr')
    m.build()
    mf = dft.RKS(m)
    mf.xc = xc
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-9
    mf.grids.level = 5
    mf.kernel()
    assert mf.converged
    return float(mf.e_tot), np.asarray(mf.nuc_grad_method().kernel()).reshape(-1)


def gto_mol(coords):
    from pyscf import gto
    s = "; ".join("%s %.10f %.10f %.10f" % (s_, x, y, z)
                  for s_, (x, y, z) in zip(ATOMS, coords))
    return gto.M(atom=s, basis=BASIS, charge=0, spin=0, verbose=0)


def fd_hessian_matrix(mol, xc, step=0.005):
    mf = fresh_mf(mol, xc)
    c0 = mol.atom_coords(unit='Bohr')
    n = mol.natm
    g = np.zeros((n, n, 3, 3))
    for i in range(n):
        for k in range(3):
            for s in (+1, -1):
                c = c0.copy()
                c[i, k] += s * step
                _, gg = eg_at(mol, xc, c)
                g[i, :, k] += s * gg.reshape(n, 3) / (2 * step)
    h = 0.5 * (g + g.transpose(1, 0, 3, 2))
    return h.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n), h


def main():
    coords = np.asarray(json.load(open(os.path.join(
        ROOT, 'run_artifacts', '01_pure_water_o3_h2o', 'gas_freq',
        'h2o.json')))['geometry_angstrom'], float)
    mol = gto_mol(coords)
    masses = np.array([15.999, 15.999, 1.008])

    out = dict(job='JOB-2026-0905-009',
               step='curvature_audit_B3_worst_element', functionals={})
    for xc in ('wb97xd', 'pbe'):
        h_a9, h_a_block = fd_hessian_matrix(mol, xc, step=0.005)
        # h_a_block here is actually the FD Hessian; also get the analytic one
        mf = fresh_mf(mol, xc)
        h_an_block = np.asarray(mf.Hessian().kernel())
        h_an9 = h_an_block.transpose(0, 2, 1, 3).reshape(9, 9)
        e0, g0 = float(mf.e_tot), np.asarray(
            mf.nuc_grad_method().kernel()).reshape(-1)

        D = h_an9 - h_a9
        i, j = np.unravel_index(np.argmax(np.abs(D)), D.shape)
        worst = float(D[i, j])
        d = np.zeros(9)
        d[i] += 1 / np.sqrt(2) if i != j else 1.0
        if i != j:
            d[j] += 1 / np.sqrt(2)
        label = ('H%d-%s / %s-%s' % (i // 3, ATOMS[i % 3], j // 3, ATOMS[j % 3]))
        print('[%s] worst element (%d,%d)=%s  |D|=%.3e' % (xc, i, j, label, worst),
              flush=True)

        rows = []
        d3 = d.reshape(3, 3)
        for a in AMPS_A:
            q = cal.q_for_atom_disp(d3, a)
            dq = (d * q).reshape(3, 3)
            ep, gp = eg_at(mol, xc, mol.atom_coords(unit='Bohr') + dq)
            em, gm = eg_at(mol, xc, mol.atom_coords(unit='Bohr') - dq)
            pr = cal.curvature_three_way(ep, em, e0, gp, gm, g0, d, q)
            pr.update(amp_a=a, q_bohr=q)
            rows.append(pr)
            print('[%s/%s] amp=%.3fA  k_E=%.6e  k_g=%.6e  k_H=%.6e  '
                  'slope_g=%.2e' % (xc, label, a, pr['k_E'], pr['k_g'],
                                    float(d @ h_an9 @ d), pr['slope_g']),
                  flush=True)
        k_E = np.array([r['k_E'] for r in rows])
        k_g = np.array([r['k_g'] for r in rows])
        k_H = float(d @ h_an9 @ d)
        if max(abs(k_g - k_E)) > 5 * abs(k_H - k_E).max() and abs(k_H - k_E[0]) > 1e-5:
            layer = 'gradient layer vs energy layer inconsistent'
        elif abs(k_H - k_E[0]) > 1e-4:
            layer = 'Hessian layer inconsistent'
        else:
            layer = 'consistent at this amplitude (noise floor ~4e-4 @0.005A)'
        out['functionals'][xc] = dict(
            worst_element=dict(index=[int(i), int(j)], label=label,
                               value_hartree_bohr2=worst),
            k_H_directional=k_H, rows=rows, layer_verdict=layer)
        print('[%s/%s] LAYER: %s' % (xc, label, layer), flush=True)

    path = os.path.join(FV, 'h2o_worst_element.json')
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=2)
    print('SAVED ->', path)


if __name__ == '__main__':
    main()
