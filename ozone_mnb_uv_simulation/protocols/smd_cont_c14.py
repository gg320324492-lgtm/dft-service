#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JOB-2026-0906-008: controlled SMD continuation for c14_plus (ONE run,
max 60 steps) from the SMD step-60 saved geometry (NOT the gas start).

Path (Phase-B-verified, unchanged): SMD(water), def2-TZVP, grid level 8,
xc='wb97xd' + project empirical full-derivative -D2, grid_response=True,
SCF 1e-12/1e-9; optimizer pyberny 0.7.0 with CONFIRMED internal-coordinate
thresholds gradientmax=gradientrms=1e-6 and -- NEW this batch, confirmed
from source -- the trust radius parameter `trust` (BernyParams, atomic
units, "maximum RMS of the quadratic step" in internal coordinates;
default 0.3) set to 0.1; Fletcher-parameter dynamic adjustment is built
into pyberny (update_trust, energy_noise=2e-8).  The CURRENT trust radius
is read per step from the live optimizer (`optimizer.trust`) and saved; it
is a step-RMS bound, NOT a hard Cartesian displacement cap.

Pre-SCF configuration assertion from the ACTUAL objects (both the DFT
grid level and the solvent settings read from mf.with_solvent/mf.grids).

Per step saved immediately: actual coords, FULL 6x3 gradient, energy,
SCF state, optimizer state (incl. current trust), solvent components read
from the ACTUAL scanner-current object (g_scanner.base.scf_summary), and
the method fingerprint.

Endpoint: independent freshly built same-config SMD object must reach
SCF-converged, finite, full max|g| <= 1e-5 Eh/Bohr with all optimizer
criteria satisfied; otherwise NOT converged, site preserved, no retry.
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
from grad_factory import make_mf_d2_gr, method_fingerprint

SYMS = ['O', 'O', 'O', 'O', 'H', 'H']
MAXSTEPS = 60
GRID_LEVEL = 8
GMAX_TARGET = 1e-5
TRUST = 0.1


def mol_from_coords(coords_A):
    atom = "; ".join("%s %.10f %.10f %.10f" % (s, x, y, z)
                     for s, (x, y, z) in zip(SYMS, coords_A))
    return gto.M(atom=atom, basis='def2-TZVP', charge=0, spin=0,
                 verbose=0, max_memory=4000)


def pre_scf_config(mf):
    ws = mf.with_solvent
    cfg = dict(dft_grid_level=int(mf.grids.level),
               xc=str(mf.xc), basis=str(mf.mol.basis),
               charge=int(mf.mol.charge), spin=int(mf.mol.spin),
               scf_tol=[float(mf.conv_tol), float(mf.conv_tol_grad)],
               solvent=str(ws.solvent),
               lebedev_order=int(getattr(ws, 'lebedev_order', -1)),
               grid_response=bool(getattr(mf.nuc_grad_method(),
                                          'grid_response')),
               d2_attached=bool(getattr(mf, '_has_full_d2', False)))
    assert cfg['dft_grid_level'] == GRID_LEVEL, cfg
    assert cfg['solvent'] == 'water' and cfg['d2_attached']
    assert cfg['grid_response']
    return cfg


def main():
    os.makedirs(OUT, exist_ok=True)
    pc = json.load(open(os.path.join(ART, 'smd_restart',
                                     'phaseC_smd_optimize.json')))
    src = os.path.join(ART, 'smd_restart', 'phaseC_smd_optimize.json')
    start = np.asarray(pc['c14_plus']['endpoint_coords_angstrom'], float)
    mol = mol_from_coords(start)
    mf = make_mf_d2_gr(mol, solvent='water', grid_response=True,
                       grid_level=GRID_LEVEL)
    cfg = pre_scf_config(mf)                       # assert BEFORE any SCF
    print('[config]', {k: cfg[k] for k in
                       ('dft_grid_level', 'solvent', 'grid_response',
                        'd2_attached')}, flush=True)

    steps = []

    def callback(envs):
        m = envs['mol']
        grad = np.asarray(envs['gradients'], float).reshape(6, 3)
        opt = envs['optimizer']
        trust_now = getattr(opt, 'trust', None)
        # solvent components from the ACTUAL scanner-current object
        ss = envs['g_scanner'].base.scf_summary
        rec = dict(
            step=int(envs['cycle']) + 1,
            coords_angstrom=np.asarray(m.atom_coords(unit='Angstrom'),
                                       float).tolist(),
            e_total=float(envs['energy']),
            gradient_full_6x3=grad.tolist(),
            grad_max=float(np.abs(grad).max()),
            grad_rms=float(np.sqrt((grad ** 2).mean())),
            scf_converged=bool(envs['g_scanner'].converged),
            e_solvent=(float(ss['e_solvent']) if 'e_solvent' in ss
                       else None),
            e_cds=(float(ss['e_cds']) if 'e_cds' in ss else None),
            e_d2_analytic=float(d2_full.d2_energy(m)),
            d2_in_scf_summary=float(ss.get('d2_dispersion', float('nan'))),
            optimizer_trust_current=(float(trust_now)
                                     if trust_now is not None else None),
            grid_level_actual=GRID_LEVEL,
            grid_response_actual=bool(getattr(envs['g_scanner'],
                                              'grid_response')),
            solvent='SMD(water)')
        if not np.isfinite(rec['e_total']) or not np.isfinite(grad).all():
            raise RuntimeError('step %d: non-finite energy/gradient'
                               % rec['step'])
        steps.append(rec)
        with open(os.path.join(OUT, 'c14_smd_cont_traj.json'), 'w') as fh:
            json.dump(steps, fh, indent=2)
        print('[c14-cont] step %2d  E=%.9f  max|g|=%.3e  trust=%s'
              % (rec['step'], rec['e_total'], rec['grad_max'],
                 ('%.3f' % rec['optimizer_trust_current'])
                 if rec['optimizer_trust_current'] is not None else 'n/a'),
              flush=True)

    t0 = time.time()
    try:
        opt_conv, mol_opt = berny_solver.kernel(
            mf, assert_convergence=True, include_ghost=True,
            callback=callback, maxsteps=MAXSTEPS,
            gradientmax=1e-6, gradientrms=1e-6, trust=TRUST)
        opt_error = None
    except Exception as e:
        traceback.print_exc()
        opt_conv, mol_opt, opt_error = False, None, repr(e)
        mol_opt = mol_from_coords(
            np.asarray(steps[-1]['coords_angstrom'], float)) if steps else None

    verify = None
    if mol_opt is not None:
        c_end = np.asarray(mol_opt.atom_coords(unit='Angstrom'), float)
        mol_v = mol_from_coords(c_end)
        mf_v = make_mf_d2_gr(mol_v, solvent='water', grid_response=True,
                             grid_level=GRID_LEVEL)
        pre_scf_config(mf_v)
        mf_v.kernel()
        g_v = np.asarray(mf_v.nuc_grad_method().kernel(), float).reshape(6, 3)
        ss = mf_v.scf_summary
        verify = dict(
            e_total=float(mf_v.e_tot),
            gradient_full_6x3=g_v.tolist(),
            grad_max=float(np.abs(g_v).max()),
            grad_rms=float(np.sqrt((g_v ** 2).mean())),
            scf_converged=bool(mf_v.converged),
            finite=bool(np.isfinite(mf_v.e_tot) and np.isfinite(g_v).all()),
            e_solvent=(float(ss['e_solvent']) if 'e_solvent' in ss else None),
            e_cds=(float(ss['e_cds']) if 'e_cds' in ss else None),
            e_d2_analytic=float(d2_full.d2_energy(mol_v)),
            grid_level_actual=GRID_LEVEL,
            pass_gmax=bool(np.abs(g_v).max() <= GMAX_TARGET),
            dE_vs_last_step=float(mf_v.e_tot - steps[-1]['e_total'])
            if steps else None)

    result = dict(
        job='JOB-2026-0906-008 (c14 continuation)',
        candidate='c14_plus', grid_level=GRID_LEVEL, solvent='SMD(water)',
        start_source=({'file': os.path.relpath(src, ROOT),
                       'record': 'phaseC_smd_optimize.json c14_plus '
                                 'endpoint = SMD step-60 geometry',
                       'cumulative_optimizer_steps_before': 60}),
        start_coords_angstrom=start.tolist(),
        maxsteps=MAXSTEPS, one_run_only=True,
        conv_params=dict(gradientmax=1e-6, gradientrms=1e-6,
                         trust_initial=TRUST,
                         trust_mechanics='BernyParams.trust: initial trust '
                         'radius in atomic units = max RMS of the quadratic '
                         'step in internal coordinates; Fletcher-parameter '
                         'dynamic adjustment (update_trust); NOT a hard '
                         'Cartesian step cap',
                         stepmax='default 1.8e-3', steprms='default 1.2e-3'),
        pre_scf_config=cfg,
        method_fingerprint=method_fingerprint(mf),
        n_steps=len(steps),
        optimizer_converged=bool(opt_conv),
        optimizer_error=opt_error,
        independent_verification=verify,
        steps=steps,
        endpoint_coords_angstrom=(
            np.asarray(mol_opt.atom_coords(unit='Angstrom'), float).tolist()
            if mol_opt is not None else None),
        seconds=round(time.time() - t0, 1))
    v = verify
    ok = bool(opt_conv and v and v['pass_gmax'] and v['scf_converged']
              and v['finite'])
    result['verdict'] = ('converged_stationary_candidate_SMD' if ok else
                         ('maxsteps_reached_not_converged'
                          if len(steps) >= MAXSTEPS
                          else 'stopped_early_not_converged'))
    with open(os.path.join(OUT, 'c14_cont_result.json'), 'w') as fh:
        json.dump(result, fh, indent=2)
    print('verdict=%s  n_steps=%d  indep max|g|=%s'
          % (result['verdict'], len(steps),
             ('%.3e' % v['grad_max']) if v else 'n/a'))


if __name__ == '__main__':
    main()
