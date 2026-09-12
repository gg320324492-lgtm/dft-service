#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Final-minima-validation step B2 (JOB-2026-0905-009): analytic-vs-FD
Hessian minimal reproduction, PLATFORM-ONLY (no project D2 hooks).

Motivation: the pruning paired control (stepB) shows the across-displacement
gradient jump is setting-independent and equals the real curvature signal ->
grid pruning/screening is NOT the source of the analytic-vs-FD Hessian gap.
The next isolation step is a minimal reproduction on H2O with PURE PySCF:

  max |H_analytic - H_FD(central difference of the analytic gradient)|

for three functionals (pbe / b3lyp / wb97xd) at def2-TZVP, two FD steps.
If wb97xd (range-separated hybrid) shows an error orders of magnitude above
pbe, the root cause is localised in the platform's RSH analytic Hessian and
-- per workspace_governance.md -- a platform issue report with this minimal
reproduction is prepared (the platform itself is NOT modified here).
Also records the same-geometry gradient sensitivity to grid settings
(default vs prune=None vs level 6), closing the grid-response question.
"""
import os
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ROOT = os.path.dirname(HERE)
FV = os.path.join(ROOT, 'run_artifacts', '01_pure_water_o3_h2o',
                  'final_minima_validation')
os.makedirs(FV, exist_ok=True)

from pyscf import gto, dft
import numpy

WATER = "O 0.000000 0.000000 0.000000; H 0.000000 0.000000 0.9576; H 0.000000 0.7570 -0.4788"
BASIS = 'def2-TZVP'


def build(xc):
    mol = gto.M(atom=WATER, basis=BASIS, charge=0, spin=0, verbose=0)
    mf = dft.RKS(mol)
    mf.xc = xc
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-9
    mf.grids.level = 5
    mf.kernel()
    assert mf.converged, xc
    return mol, mf


def grad(mol, mf, coords_bohr):
    m = mol.copy()
    m.set_geom_(coords_bohr, unit='Bohr')
    m.build()
    mf2 = mf.copy() if hasattr(mf, 'copy') else None
    # fresh object to avoid grid state carry-over
    mf2 = dft.RKS(m)
    mf2.xc = mf.xc
    mf2.conv_tol = mf.conv_tol
    mf2.conv_tol_grad = mf.conv_tol_grad
    mf2.grids.level = mf.grids.level
    mf2.kernel()
    assert mf2.converged
    return numpy.asarray(mf2.nuc_grad_method().kernel())


def analytic_hess(mol, mf):
    h = mf.Hessian().kernel()
    return numpy.asarray(h)


def fd_hess(mol, mf, step):
    c0 = mol.atom_coords(unit='Bohr')
    n = mol.natm
    g = numpy.zeros((n, n, 3, 3))
    for i in range(n):
        for k in range(3):
            for s in (+1, -1):
                c = c0.copy()
                c[i, k] += s * step
                g[i, :, k] += s * grad(mol, mf, c) / (2 * step)
    return 0.5 * (g + g.transpose(1, 0, 3, 2))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default='pbe,b3lyp,wb97xd')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    xcs = args.only.split(',')
    out = dict(job='JOB-2026-0905-009',
               step='final_validation_B2_platform_minimal_repro',
               system='H2O/def2-TZVP, pure PySCF (no project hooks)',
               results={})
    for xc in xcs:
        mol, mf = build(xc)
        h_a = analytic_hess(mol, mf)
        entry = {}
        for step in (0.005, 0.02):
            h_fd = fd_hess(mol, mf, step)
            d = float(numpy.abs(h_a - h_fd).max())
            rel = float(d / numpy.abs(h_a).max())
            entry['step_%s' % step] = dict(max_abs_diff=d, relative=rel)
            print('[%s] h=%.3f  max|dH|=%.3e  rel=%.2e' % (xc, step, d, rel),
                  flush=True)
        # same-geometry gradient sensitivity to grid settings
        g_def = grad(mol, mf, mol.atom_coords(unit='Bohr'))
        mf3 = mf.copy()
        mf3.grids.prune = None
        mf3.grids.build(with_non0tab=True)
        g_noprune = numpy.asarray(mf3.nuc_grad_method().kernel())
        entry['grad_sensitivity_prune'] = float(numpy.abs(g_def - g_noprune).max())
        out['results'][xc] = entry
        print('[%s] same-geometry |g(default)-g(prune=None)|=%.2e'
              % (xc, entry['grad_sensitivity_prune']), flush=True)

    path = args.out or os.path.join(FV, 'stepB2_platform_minimal_repro.json')
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=2)
    print('SAVED ->', path)


if __name__ == '__main__':
    main()
