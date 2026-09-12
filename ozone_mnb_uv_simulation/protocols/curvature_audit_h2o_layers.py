#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Curvature-audit step B (JOB-2026-0905-009): three-layer localization on the
pure-PySCF water minimal case.

Layers, all evaluated along the SAME fixed direction d and the SAME amplitude
definition (max per-atom displacement-vector length):
  energy layer  : k_E = [E(+q)+E(-q)-2E0]/q^2          (2nd energy difference)
  gradient layer: k_g = d.[g(+q)-g(-q)]/(2q)           (gradient difference)
  Hessian layer : k_H = d^T H d                        (analytic Hessian)
Slopes d.g0 and [E(+q)-E(-q)]/(2q) are reported to separate non-stationarity
from curvature.

Localization logic:
  k_E ~ k_g ~ k_H          -> consistent (any residual is q^2 truncation)
  k_E ~ k_g  !=  k_H       -> Hessian layer inconsistent (2nd-derivative path)
  k_E != k_g               -> gradient layer inconsistent with energy layer
Functionals: wb97xd (suspect) and pbe (control).  Directions: the two O-H
stretch coordinates (stiff, large signal).
"""
import os
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import curvature_audit_lib as cal
import run_baseline as rb

ROOT = os.path.dirname(HERE)
FV = os.path.join(ROOT, 'run_artifacts', '01_pure_water_o3_h2o', 'curvature_audit')
os.makedirs(FV, exist_ok=True)

BASIS = 'def2-TZVP'
AMPS_A = [0.005, 0.01, 0.02, 0.05]
MASSES = np.array([15.999, 15.999, 1.008])


def fresh_mf(mol, xc):
    from pyscf import dft
    mf = dft.RKS(mol)
    mf.xc = xc
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-9
    mf.grids.level = 5
    mf.kernel()
    assert mf.converged, xc
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


def main():
    coords = np.asarray(json.load(open(os.path.join(
        ROOT, 'run_artifacts', '01_pure_water_o3_h2o', 'gas_freq',
        'h2o.json')))['geometry_angstrom'], float)
    mol = gto_mol(coords)

    # directions: the two O-H stretch coordinates (unit 3N vectors)
    dirs = {}
    rO, rH1, rH2 = coords
    for name, ih in (('OH1_stretch', 1), ('OH2_stretch', 2)):
        u = coords[ih] - rO
        u /= np.linalg.norm(u)
        d = np.zeros(9)
        d[0:3] = -u
        d[3 * ih:3 * ih + 3] = u
        d /= np.linalg.norm(d)                      # unit 3N vector
        dirs[name] = d

    out = dict(job='JOB-2026-0905-009',
               step='curvature_audit_B_h2o_layer_localization',
               amplitude_convention='max per-atom displacement-VECTOR length',
               functionals={})
    for xc in ('wb97xd', 'pbe'):
        mf = fresh_mf(mol, xc)
        h = np.asarray(mf.Hessian().kernel())
        h = 0.5 * (h + h.transpose(1, 0, 3, 2))
        H9 = h.transpose(0, 2, 1, 3).reshape(9, 9)
        e0, g0 = float(mf.e_tot), np.asarray(mf.nuc_grad_method().kernel()).reshape(-1)
        res = dict(directions={})
        for dname, d in dirs.items():
            q_list = [cal.q_for_atom_disp(d.reshape(3, 3), a) for a in AMPS_A]
            rows = []
            for a, q in zip(AMPS_A, q_list):
                dq = (d * q).reshape(3, 3)          # Cartesian, Bohr
                cp = mol.atom_coords(unit='Bohr') + dq
                cm = mol.atom_coords(unit='Bohr') - dq
                ep, gp = eg_at(mol, xc, cp)
                em, gm = eg_at(mol, xc, cm)
                pr = cal.curvature_three_way(ep, em, e0, gp, gm, g0, d, q)
                pr.update(amp_a=a, q_bohr=q)
                rows.append(pr)
                print('[%s/%s] amp=%.3fA  k_E=%.6e  k_g=%.6e  k_H=%.6e  '
                      'slope_g=%.2e slope_E=%.2e'
                      % (xc, dname, a, pr['k_E'], pr['k_g'],
                         float(d @ H9 @ d), pr['slope_g'], pr['slope_E']),
                      flush=True)
            k_H = float(d @ H9 @ d)
            k_E = [r['k_E'] for r in rows]
            k_g = [r['k_g'] for r in rows]
            if max(abs(np.array(k_g) - np.array(k_E))) > 5 * max(
                    abs(np.array(k_H) - np.array(k_E)).max(), 1e-6) and \
                    max(abs(np.array(k_E) - k_H)) > 1e-4:
                layer = 'gradient layer vs energy layer inconsistent'
            elif max(abs(np.array(k_E) - k_H)) > 5e-5:
                layer = 'Hessian layer inconsistent (k_E ~ k_g != k_H)'
            else:
                layer = 'consistent (residual = q^2 truncation)'
            res['directions'][dname] = dict(
                rows=rows, k_H=k_H, k_E=k_E, k_g=k_g, layer_verdict=layer)
            print('[%s/%s] LAYER: %s' % (xc, dname, layer), flush=True)
        res['settings'] = dict(basis=BASIS, grid_level=5, conv_tol=1e-12,
                               conv_tol_grad=1e-9,
                               amplitude='max per-atom displacement-vector length')
        out['functionals'][xc] = res

    path = os.path.join(FV, 'h2o_layer_localization.json')
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=2)
    print('SAVED ->', path)


def gto_mol(coords):
    from pyscf import gto
    s = "; ".join("%s %.10f %.10f %.10f" % (s_, x, y, z)
                  for s_, (x, y, z) in zip(['O', 'H', 'H'], coords))
    return gto.M(atom=s, basis=BASIS, charge=0, spin=0, verbose=0)


if __name__ == '__main__':
    main()
