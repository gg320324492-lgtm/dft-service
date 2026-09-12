#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage 1 (JOB-2026-0905-009): gas-phase geometry optimisation of the O3.H2O
conformer pool plus the two monomers, all through the audited production path

    make_mf_d2  ->  pyscf.geomopt.berny_solver.optimize

Method: wB97X-D / def2-TZVP (libxc wb97xd DFT part + project full-derivative
-D2), grid level 5, SCF conv 1e-11, optimisation convergence identical to the
Phase-1 baseline (OPT_CONV below).  Every optimised structure is re-evaluated
with a FRESH make_mf_d2 object at its final geometry (strict single point:
total energy, bare DFT part, -D2 part, gradient, <S^2>).

Outputs (per task): run_artifacts/01_pure_water_o3_h2o/gas_opt/<id>.json
         (stage)   : run_artifacts/01_pure_water_o3_h2o/gas_opt/stage1_summary.json
Tasks already present on disk are skipped unless --force is given.
"""
import os
import sys
import json
import time
import math
import argparse
import numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from pyscf import gto, lib, __version__ as PYSCF_VERSION
from pyscf.geomopt import geometric_solver
from pyscf.geomopt.geometric_solver import optimize

import d2_full
import conformers

ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, 'run_artifacts', '01_pure_water_o3_h2o', 'gas_opt')
LOG = os.path.join(ART, 'logs')

BASIS = d2_full.BASIS                       # def2-TZVP
# geomeTRIC (delocalised internal coords) replaces berny, which refuses the
# near-linear O_w-H...O hydrogen-bond chain (CoordinateError).  assert_convergence
# is disabled so a weakly-bound, flat PES that plateaus just above the strict
# baseline gradient still returns its best geometry; the per-task record then
# re-evaluates the gradient with a FRESH make_mf_d2 and flags convergence via
# the project criterion (max |grad| < 5e-5 Ha/Bohr).
# geomeTRIC convergence must be passed as TOP-LEVEL kwargs with its REAL
# key names (PySCF geometric_solver forwards **kwargs straight to geomeTRIC's
# run_optimizer).  The previous 'convergence_params=dict(gmax=...)' form used
# keys geomeTRIC does not recognise, so it silently fell back to the GAU
# default (Max-Grad < 4.5e-4) and every run stopped at max|grad| ~1e-4.
# Correct keys: convergence_gmax/grms/dmax/drms/energy.
#   * gmax=3e-5 (tighter than the project 5e-5 strict criterion)
#   * dmax/drms = 10 A  -> displacement convergence effectively disabled, so
#     geomeTRIC cannot stop early on a tiny step in a flat weak-binding region;
#     only the gradient criterion terminates the optimisation.
OPT_CONV = dict(maxsteps=400, assert_convergence=False,
                convergence_gmax=3e-5, convergence_grms=1.5e-5,
                convergence_dmax=10.0, convergence_drms=10.0,
                convergence_energy=1e-7)
N_PROC = 8
N_THREADS = 4


def _init_worker():
    lib.num_threads(N_THREADS)


def geom_string(coords, syms):
    return "; ".join("%s %.10f %.10f %.10f" % (s, c[0], c[1], c[2])
                     for s, c in zip(syms, coords))


def write_xyz(path, syms, coords, comment=''):
    lines = [str(len(syms)), comment]
    for s, c in zip(syms, coords):
        lines.append('%-2s %18.10f %18.10f %18.10f' % (s, c[0], c[1], c[2]))
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')


def build_tasks():
    tasks = []
    for spec in conformers.SPECS:
        coords = conformers.build(spec)
        tasks.append(dict(id=spec['id'], kind='conformer', syms=['O'] * 4 + ['H', 'H'],
                          coords=coords, coverage=spec['coverage'], rule=spec['rule']))
    # monomers (experimental start geometries, same convention as Phase 1)
    tasks.append(dict(id='h2o', kind='monomer', syms=['O', 'H', 'H'],
                      coords=np.array([[0.0, 0.0, 0.0],
                                       [0.7575, 0.0, 0.5865],
                                       [-0.7575, 0.0, 0.5865]]),
                      coverage='单体参照（相互作用能分母）',
                      rule='实验气相几何 r(O-H)=0.9584 A, 角 104.45 deg'))
    t = math.radians(conformers.ANG_OOO / 2.0)
    r = conformers.R_OO
    tasks.append(dict(id='o3', kind='monomer', syms=['O', 'O', 'O'],
                      coords=np.array([[0.0, 0.0, 0.0],
                                       [r * math.sin(t), r * math.cos(t), 0.0],
                                       [-r * math.sin(t), r * math.cos(t), 0.0]]),
                      coverage='单体参照（相互作用能分母）',
                      rule='实验气相几何 r(O-O)=1.2717 A, 角 117.79 deg'))
    return tasks


def run_task(task):
    tid = task['id']
    os.makedirs(LOG, exist_ok=True)
    log_path = os.path.join(LOG, '%s.log' % tid)
    t_start = time.time()
    rec = dict(id=tid, kind=task['kind'], coverage=task['coverage'], rule=task['rule'],
               basis=BASIS, xc=d2_full.XC, grid_level=d2_full.GRID_LEVEL,
               scf_conv=d2_full.SCF_CONV, opt_conv=dict(OPT_CONV),
               solvent='none', charge=0, multiplicity=1,
               initial_coords_angstrom=[[round(float(x), 8) for x in row]
                                        for row in task['coords']],
               pyscf_version=PYSCF_VERSION)
    try:
        with open(log_path, 'w') as fh:
            fh.write('task %s\n' % tid)
        if task.get('start_coords') is not None:
            start_coords = np.asarray(task['start_coords'], dtype=float)
            rec['initial_coords_angstrom'] = [[round(float(x), 8) for x in row]
                                             for row in start_coords]
        else:
            start_coords = task['coords']
        mol0 = gto.M(atom=geom_string(start_coords, task['syms']), basis=BASIS,
                     charge=0, spin=0, verbose=3)
        mol0.stdout = open(log_path, 'a')
        mf = d2_full.make_mf_d2(mol0)
        opt_conv = task.get('opt_conv', OPT_CONV)
        t0 = time.time()
        mol_opt = optimize(mf, **opt_conv)
        rec['opt_seconds'] = round(time.time() - t0, 1)
        coords_opt = mol_opt.atom_coords(unit='Angstrom')
        rec['optimised_coords_angstrom'] = [[round(float(x), 8) for x in row]
                                            for row in coords_opt]

        # ---- strict single point with a FRESH object at the final geometry
        molf = gto.M(atom=geom_string(coords_opt, task['syms']), basis=BASIS,
                     charge=0, spin=0, verbose=0)
        mff = d2_full.make_mf_d2(molf)
        t0 = time.time()
        e_tot = float(mff.kernel())
        rec['sp_seconds'] = round(time.time() - t0, 1)
        grad = np.asarray(mff.nuc_grad_method().kernel())
        rec['scf_converged'] = bool(mff.converged)
        rec['e_total_hartree'] = e_tot
        rec['e_dft_part_hartree'] = float(mff._d2_parent_cls.energy_tot(mff))
        rec['e_d2_hartree'] = d2_full.d2_energy(molf)
        rec['max_gradient_hartree_bohr'] = float(np.abs(grad).max())
        rec['rms_gradient_hartree_bohr'] = float(np.sqrt((grad ** 2).mean()))
        rec['n_basis_functions'] = int(molf.nao_nr())
        rec['n_grid_points'] = int(mff.grids.weights.size)
        try:
            ss, sz = mff.spin_square()
            rec['spin_s2'] = float(ss)
        except Exception as exc:                       # restricted closed shell
            rec['spin_s2'] = 0.0
            rec['spin_s2_note'] = ('restricted closed-shell determinant: '
                                   '<S^2> = 0 by construction (%s)' % type(exc).__name__)
        rec['optimisation_converged'] = bool(rec['max_gradient_hartree_bohr'] < 5e-5
                                             and rec['scf_converged'])
        rec['status'] = 'converged' if rec['optimisation_converged'] else 'NOT_CONVERGED'
        rec['total_seconds'] = round(time.time() - t_start, 1)
        write_xyz(os.path.join(ART, '%s_opt.xyz' % tid), task['syms'], coords_opt,
                  comment='%s optimised (gas, wB97X-D/def2-TZVP)' % tid)
    except Exception as exc:
        import traceback
        rec['status'] = 'FAILED'
        rec['error'] = '%s: %s' % (type(exc).__name__, exc)
        with open(log_path, 'a') as fh:
            traceback.print_exc(file=fh)
    with open(os.path.join(ART, '%s.json' % tid), 'w') as fh:
        json.dump(rec, fh, indent=2)
    print('[%s] %s  %.1f s' % (tid, rec.get('status'), time.time() - t_start), flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--only', default=None, help='comma separated task ids')
    ap.add_argument('--refine', action='store_true',
                    help='re-seed each task from its existing optimised_coords_angstrom')
    ap.add_argument('--maxsteps', type=int, default=None)
    ap.add_argument('--gmax', type=float, default=None)
    ap.add_argument('--grm', type=float, default=None)
    ap.add_argument('--maxdisp', type=float, default=None)
    ap.add_argument('--rmsdisp', type=float, default=None)
    args = ap.parse_args()

    # per-run convergence override (correct geomeTRIC key names; lets us
    # disable premature displacement convergence and force gradient-driven
    # convergence for flat weak complexes)
    opt_override = {}
    if args.maxsteps is not None:
        opt_override['maxsteps'] = args.maxsteps
    if args.gmax is not None:
        opt_override['convergence_gmax'] = args.gmax
    if args.grm is not None:
        opt_override['convergence_grms'] = args.grm
    if args.maxdisp is not None:
        opt_override['convergence_dmax'] = args.maxdisp
    if args.rmsdisp is not None:
        opt_override['convergence_drms'] = args.rmsdisp
    override_conv = dict(OPT_CONV)
    override_conv.update(opt_override)

    os.makedirs(ART, exist_ok=True)
    os.makedirs(LOG, exist_ok=True)
    tasks = build_tasks()
    if args.only:
        keep = set(args.only.split(','))
        tasks = [t for t in tasks if t['id'] in keep]
    if args.refine:
        for t in tasks:
            prev = os.path.join(ART, '%s.json' % t['id'])
            if os.path.exists(prev):
                d = json.load(open(prev))
                if d.get('optimised_coords_angstrom') is not None:
                    t['start_coords'] = d['optimised_coords_angstrom']
                    print('refine %s: re-seeded from previous optimised geometry'
                          % t['id'], flush=True)
    if not args.force:
        tasks = [t for t in tasks
                 if not os.path.exists(os.path.join(ART, '%s.json' % t['id']))]
    for t in tasks:
        t['opt_conv'] = override_conv
    print('tasks to run: %d -> %s' % (len(tasks), [t['id'] for t in tasks]), flush=True)
    t0 = time.time()
    with Pool(N_PROC, initializer=_init_worker) as pool:
        recs = pool.map(run_task, tasks)
    summary = dict(job='JOB-2026-0905-009', stage='1_gas_optimisation',
                   backend='PySCF %s / WSL2' % PYSCF_VERSION,
                   method='wB97X-D/def2-TZVP (full-derivative -D2 via make_mf_d2)',
                   grid_level=d2_full.GRID_LEVEL, scf_conv=d2_full.SCF_CONV,
                   opt_conv=dict(OPT_CONV), n_proc=N_PROC, n_threads=N_THREADS,
                   wall_seconds=round(time.time() - t0, 1),
                   tasks=recs)
    with open(os.path.join(ART, 'stage1_summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)
    print('STAGE1_DONE wall=%.1f s' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
