#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Final-minima-validation step D (JOB-2026-0905-009): independent soft-mode
verification at the re-optimised endpoints, all at the UNIFIED numerical
settings (grid level 6, SCF 1e-12/1e-9, full-derivative -D2).

For each candidate:
  1. analytic Hessian (from step C, recomputed here if absent) -> lowest-mode
     eigenvector and eigenvalue (mass-weighted);
  2. independent FD Hessian of the total gradient at two ADJACENT steps
     (0.02 and 0.03 Bohr) plus an auxiliary 0.05 Bohr; the lowest-mode
     curvature/frequency is tracked across all constructions -- stability,
     not agreement of the full matrix, is the criterion;
  3. MODE ENERGY SCAN (the arbiter): energies along +/- the lowest mass-weighted
     mode at small fixed displacements (target max atom shift 0.05/0.10/0.20 A),
     parabolic fit -> curvature -> frequency.  Energies are smooth to ~1e-10
     Eh while the disputed curvature signals are ~1e-4-1e-3 Eh at these
     amplitudes, so the scan does NOT rely on the questioned analytic second
     derivatives.  Mass-weighted -> Cartesian conversion is documented inline.

Verdict inputs: does the lowest-mode frequency from (1) FD constructions and
(2) the energy scan agree in SIGN with the analytic Hessian?
Resume-safe per structure.
"""
import os
import sys
import json
import time
import numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import d2_full
import conformer_utils as cu
import run_baseline as rb

ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, 'run_artifacts', '01_pure_water_o3_h2o')
FV = os.path.join(ART, 'final_minima_validation')
os.makedirs(FV, exist_ok=True)

SYMS = ['O', 'O', 'O', 'O', 'H', 'H']
TARGETS = ['c14_plus', 'c06_plus']
FD_STEPS = [0.02, 0.03, 0.05]       # 0.02/0.03 adjacent pair, 0.05 auxiliary
SCAN_SHIFTS_A = [0.05, 0.10, 0.20]  # target max atom displacement, Angstrom
MASS = np.array([15.999, 15.999, 15.999, 15.999, 1.008, 1.008])  # amu, explicit
BOHR_A = 0.52917721092


def make_mf(mol):
    mf = d2_full.make_mf_d2(mol)
    mf.grids.level = 6
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-9
    return mf


def analytic_lowest(mol):
    mf = make_mf(mol)
    mf.kernel()
    h_raw = mf.Hessian().kernel()
    asym = float(np.abs(h_raw - h_raw.transpose(1, 0, 3, 2)).max())
    h = 0.5 * (h_raw + h_raw.transpose(1, 0, 3, 2))
    n = mol.natm
    H = h.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)
    mw = np.repeat(MASS ** -0.5, 3)
    Heff = H * np.outer(mw, mw)
    w, v = np.linalg.eigh(Heff)
    imode = int(np.argmin(w))
    lam = float(w[imode])
    vec = v[:, imode]                      # mass-weighted eigenvector (unit norm)
    # mass-weighted -> Cartesian:  x_cart[Bohr] = mw * vec * q  (q in amu^0.5*Bohr)
    dx_cart = (mw * vec).reshape(n, 3)     # Cartesian shift per unit q, Bohr
    # frequency conversion identical to run_baseline.vib_freqs
    conv = 1.0 / (2 * np.pi * rb.data.nist.LIGHT_SPEED_SI * 100.0)
    nu = float(np.sign(lam) * np.sqrt(abs(lam))
               * np.sqrt(rb.data.nist.HARTREE2J / (rb.AMU * rb.data.nist.BOHR_SI ** 2))
               * conv)
    return dict(asym=asym, lam=lam, nu=nu, dx_cart_bohr=dx_cart, h=h)


def fd_hess(mol, step):
    c0 = mol.atom_coords(unit='Bohr')
    n = mol.natm
    g = np.zeros((n, n, 3, 3))
    for i in range(n):
        for k in range(3):
            for s in (+1, -1):
                c = c0.copy()
                c[i, k] += s * step
                m = mol.copy()
                m.set_geom_(c, unit='Bohr')
                m.build()
                mf = make_mf(m)
                mf.kernel()
                g[i, :, k] += s * mf.nuc_grad_method().kernel().reshape(n, 3) / (2 * step)
    return 0.5 * (g + g.transpose(1, 0, 3, 2))


def lowest_of_h(mol, h):
    n = mol.natm
    H = h.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)
    mw = np.repeat(MASS ** -0.5, 3)
    w = np.linalg.eigvalsh(H * np.outer(mw, mw))
    lam = float(w[0])
    conv = 1.0 / (2 * np.pi * rb.data.nist.LIGHT_SPEED_SI * 100.0)
    nu = float(np.sign(lam) * np.sqrt(abs(lam))
               * np.sqrt(rb.data.nist.HARTREE2J / (rb.AMU * rb.data.nist.BOHR_SI ** 2))
               * conv)
    return lam, nu


def run_target(tid):
    t0 = time.time()
    out_path = os.path.join(FV, 'stepD_softmode_%s.json' % tid)
    if os.path.exists(out_path):
        print('[skip]', tid, flush=True)
        with open(out_path) as fh:
            return json.load(fh)

    coords = np.asarray(json.load(open(
        os.path.join(FV, 'stepC_reopt_%s.json' % tid)))['new_geometry'], float)
    mol = cu.mol_from_coords(coords, SYMS, verbose=0)

    an = analytic_lowest(mol)
    e0 = None
    mf0 = make_mf(mol)
    mf0.kernel()
    e0 = float(mf0.e_tot)

    res = dict(id=tid, analytic=dict(pre_symmetry_asymmetry=an['asym'],
                                     lowest_lambda=an['lam'],
                                     lowest_nu_cm1=an['nu']), fd={}, scan={})

    for step in FD_STEPS:
        h = fd_hess(mol, step)
        lam, nu = lowest_of_h(mol, h)
        res['fd']['step_%s' % step] = dict(lowest_lambda=lam, lowest_nu_cm1=nu)
        print('[%s] FD h=%.2f  lowest lam=%.3e  nu=%.1f cm-1'
              % (tid, step, lam, nu), flush=True)

    # energy scan along the analytic lowest mode
    max_shift_bohr = np.abs(an['dx_cart_bohr']).max()
    for target_a in SCAN_SHIFTS_A:
        q = (target_a / BOHR_A) / max_shift_bohr
        de = {}
        for s in (+1, -1):
            c = mol.atom_coords(unit='Bohr') + s * q * an['dx_cart_bohr']
            m = mol.copy()
            m.set_geom_(c, unit='Bohr')
            m.build()
            mf = make_mf(m)
            mf.kernel()
            de['plus' if s > 0 else 'minus'] = float(mf.e_tot)
        # curvature in Eh/(amu^0.5*Bohr)^2 : E(+q)+E(-q)-2E0 = k q^2  ->  k=dE/q^2
        d2 = (de['plus'] + de['minus'] - 2 * e0) / (2 * q ** 2)
        conv = 1.0 / (2 * np.pi * rb.data.nist.LIGHT_SPEED_SI * 100.0)
        nu = float(np.sign(d2) * np.sqrt(abs(d2))
                   * np.sqrt(rb.data.nist.HARTREE2J / (rb.AMU * rb.data.nist.BOHR_SI ** 2))
                   * conv)
        res['scan']['amp_%sA' % target_a] = dict(
            dE_plus=de['plus'] - e0, dE_minus=de['minus'] - e0,
            curvature=d2, implied_nu_cm1=nu)
        print('[%s] scan amp=%.2fA  dE(+/-)=%.3e/%.3e  curvature=%.3e  nu=%.1f cm-1'
              % (tid, target_a, de['plus'] - e0, de['minus'] - e0, d2, nu), flush=True)

    # consistency verdict
    signs = dict(analytic=np.sign(an['lam']),
                 **{('fd_%s' % k): np.sign(v['lowest_lambda'])
                    for k, v in res['fd'].items()},
                 **{('scan_%s' % k): np.sign(v['curvature'])
                    for k, v in res['scan'].items()})
    res['signs'] = {k: int(v) for k, v in signs.items()}
    res['sign_consistent'] = bool(len(set(signs.values())) == 1)
    res['seconds'] = round(time.time() - t0, 1)
    with open(out_path, 'w') as fh:
        json.dump(res, fh, indent=2)
    print('[%s] SIGN TABLE %s -> consistent=%s  (%.0f s)'
          % (tid, res['signs'], res['sign_consistent'], res['seconds']), flush=True)
    return res


def _init():
    import pyscf.lib as lib
    lib.num_threads(8)


def main():
    done, todo = [], []
    for t in TARGETS:
        if os.path.exists(os.path.join(FV, 'stepD_softmode_%s.json' % t)):
            with open(os.path.join(FV, 'stepD_softmode_%s.json' % t)) as fh:
                done.append(json.load(fh))
        else:
            todo.append(t)
    results = done
    if todo:
        with Pool(len(todo), initializer=_init) as pool:
            results.extend(pool.map(run_target, todo))
    results.sort(key=lambda r: TARGETS.index(r['id']))
    with open(os.path.join(FV, 'stepD_softmode_summary.json'), 'w') as fh:
        json.dump(dict(job='JOB-2026-0905-009',
                       step='final_validation_D_softmode',
                       settings=dict(grid_level=6, scf_tol=1e-12,
                                     scf_tol_grad=1e-9, fd_steps=FD_STEPS,
                                     scan_amplitudes_a=SCAN_SHIFTS_A,
                                     mass_convention='mass-weighted eigenvector; '
                                     'x_cart = M^-1/2 * v * q, q in amu^0.5*Bohr'),
                       targets=results), fh, indent=2)
    print('SAVED stepD summary', flush=True)


if __name__ == '__main__':
    main()
