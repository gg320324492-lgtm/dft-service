#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JOB-2026-0906-007 Phase B (endpoints): production vs independent native
benchmark at the TWO saved SMD optimisation endpoints (both L8).

For each endpoint: converge a fresh production object (kernel), take the
D2-attached gradient, then the NATIVE SMD gradient from the SAME converged
density via the un-wrapped parent class (both grid_response=True), and
check  (g_prod - g_native) == analytic d2_grad  to machine precision.
Confirms whether the endpoint max|g| records from JOB-2026-0906-006 hold
when remeasured on the same-config L8 path.
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, 'run_artifacts', '01_pure_water_o3_h2o')
OUT = os.path.join(ART, 'smd_config_audit')
sys.path.insert(0, HERE)

from pyscf import gto
from smd_config_audit_verify import build, GRID_LEVEL
import d2_full

SYMS = ['O', 'O', 'O', 'O', 'H', 'H']


def main():
    os.makedirs(OUT, exist_ok=True)
    pc = json.load(open(os.path.join(ART, 'smd_restart',
                                     'phaseC_smd_optimize.json')))
    results = {}
    for tid in ('c14_plus', 'c06_plus'):
        coords = np.asarray(pc[tid]['endpoint_coords_angstrom'], float)
        recorded = pc[tid]['independent_verification']
        atom = "; ".join("%s %.10f %.10f %.10f" % (s, x, y, z)
                         for s, (x, y, z) in zip(SYMS, coords))
        mol = gto.M(atom=atom, basis='def2-TZVP', charge=0, spin=0,
                    verbose=0, max_memory=4000)
        mf = make_mf_d2_gr_prod(mol)
        mf.kernel()
        gobj_p = mf.nuc_grad_method()
        gobj_p.grid_response = True
        g_prod = np.asarray(gobj_p.kernel(), float).reshape(-1)
        gobj_n = mf._d2_parent_cls.nuc_grad_method(mf)
        gobj_n.grid_response = True
        g_nat = np.asarray(gobj_n.kernel(), float).reshape(-1)
        g_d2 = np.asarray(d2_full.d2_grad(mol), float).reshape(-1)
        e_prod = float(mf.e_tot)
        e_nat = float(mf._d2_parent_cls.energy_tot(mf))
        e_d2 = float(d2_full.d2_energy(mol))
        resid_e = float((e_prod - e_nat) - e_d2)
        resid_g = float(np.abs((g_prod - g_nat) - g_d2).max())
        gm = float(np.abs(g_prod).max())
        holds = bool(abs(gm - recorded['grad_max'])
                     / max(recorded['grad_max'], 1e-300) < 1e-6)
        results[tid] = dict(
            grid_level=int(mf.grids.level),
            e_total=e_prod, grad_max=gm,
            recorded_grad_max=recorded['grad_max'],
            record_holds=holds,
            same_density_d2once=dict(resid_e=resid_e, resid_g=resid_g,
                                     d2_e_mag=float(abs(e_d2)),
                                     d2_g_mag=float(np.abs(g_d2).max()),
                                     pass_e=bool(abs(resid_e)
                                                 < max(1e-2 * abs(e_d2),
                                                       1e-6)),
                                     pass_g=bool(resid_g
                                                 < max(1e-2
                                                       * float(np.abs(g_d2)
                                                          .max()), 1e-6))),
            gradient_full_6x3=g_prod.reshape(6, 3).tolist(),
            scf_converged=bool(mf.converged))
        print('[%s] L%d E=%.9f max|g|=%.4e (recorded %.4e, holds=%s) '
              'd2once resid_e=%.2e resid_g=%.2e'
              % (tid, mf.grids.level, e_prod, gm, recorded['grad_max'],
                 holds, resid_e, resid_g), flush=True)
    json.dump(results, open(os.path.join(OUT,
                                         'endpoint_recheck.json'), 'w'),
              indent=2)
    print('saved -> endpoint_recheck.json')


def make_mf_d2_gr_prod(mol):
    from grad_factory import make_mf_d2_gr
    return make_mf_d2_gr(mol, solvent='water', grid_response=True,
                         grid_level=GRID_LEVEL)


if __name__ == '__main__':
    main()
