#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Final-minima-validation step B (JOB-2026-0905-009): numerical-settings
audit + pruning paired control.

Part 1 -- record, item by item, the numerical settings of BOTH sides of the
analytic-vs-finite-difference Hessian comparison (geometry/coordinate units,
functional/basis/-D2 implementation, grid level, pruning, small_rho_cutoff,
SCF energy and orbital-gradient thresholds, grid response to nuclear
displacement).

Part 2 -- PAIRED CONTROL for the pruning hypothesis: at the c14_plus geometry
and the same coordinate displaced by +/-0.005 Bohr, compute the total gradient
under three grid configurations that differ ONLY in pruning/screening:
  (a) project default  (grids.level=5, prune default, small_rho_cutoff default)
  (b) prune disabled   (grids.prune = None)
  (c) screening off    (small_rho_cutoff = 0)
If the across-displacement gradient jump shrinks drastically when pruning is
off, the pruning attribution is CONFIRMED; if not, the attribution is
UNPROVEN and the discrepancy must be attributed otherwise.  The jump is also
reported as the implied Hessian error |jump|/(2h) for direct comparison with
the observed max|dH| ~ 1.7e-3 Eh/Bohr^2.
"""
import os
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import d2_full
import conformer_utils as cu

ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, 'run_artifacts', '01_pure_water_o3_h2o')
FV = os.path.join(ART, 'final_minima_validation')
os.makedirs(FV, exist_ok=True)

SYMS = ['O', 'O', 'O', 'O', 'H', 'H']
STEP = 0.005          # Bohr, same as the disputed FD step
DISP_ATOM, DISP_XYZ = 3, 1        # water O, y? -> use index (3,1); documented below


def grad_at(mol_prototype, coords_bohr, setting):
    m = mol_prototype.copy()
    m.set_geom_(coords_bohr, unit='Bohr')
    m.build()
    mf = d2_full.make_mf_d2(m)
    if setting == 'default':
        pass
    elif setting == 'prune_off':
        mf.grids.prune = None
    elif setting == 'screening_off':
        mf.small_rho_cutoff = 0.0
    elif setting == 'prune_and_screening_off':
        mf.grids.prune = None
        mf.small_rho_cutoff = 0.0
    else:
        raise ValueError(setting)
    mf.kernel()
    assert mf.converged
    g = np.asarray(mf.nuc_grad_method().kernel())
    return float(mf.e_tot), g, int(mf.grids.weights.size)


def main():
    coords = np.asarray(json.load(open(
        os.path.join(ART, 'gas_freq', 'c14_plus.json')))['geometry_angstrom'], float)
    mol0 = cu.mol_from_coords(coords, SYMS, verbose=0)
    c0 = mol0.atom_coords(unit='Bohr')

    # ---------------- Part 1: settings audit (live object)
    mf = d2_full.make_mf_d2(mol0)
    settings = dict(
        geometry_units='input Angstrom -> PySCF Bohr internally (recorded both)',
        functional=d2_full.XC,
        functional_note=("libxc XC_HYB_GGA_XC_wB97X_D = wB97X-D 泛函的不含经验色散部分；"
                         "经验色散由 d2_full 显式补加（全导数，版本参数 s6=1, sR6=1.1, a=6, M=12, "
                         "Grimme-2006 C6/RvdW），ghost 不参与"),
        basis=d2_full.BASIS,
        grid_level=mf.grids.level,
        grids_prune_default=type(mf.grids.prune).__name__,
        small_rho_cutoff_default=mf.small_rho_cutoff,
        scf_energy_conv_tol=mf.conv_tol,
        scf_orbital_grad_conv_tol=mf.conv_tol_grad,
        scf_max_cycle=mf.max_cycle,
        grid_response_handling=(
            'PySCF rks.get_veff 在每个 SCF 周期按当前密度调用 '
            'prune_by_density_(rho, small_rho_cutoff) 重剪枝；grids.build(with_non0tab=True) '
            '在每次 kernel() 重建。即：网格随几何与密度双重变化 —— 单几何上的解析 Hessian '
            '自洽，但跨几何差分混合了不同网格。')),
    print(json.dumps(settings, ensure_ascii=False, indent=1), flush=True)

    # ---------------- Part 2: paired control
    control = {}
    for setting in ('default', 'prune_off', 'screening_off',
                    'prune_and_screening_off'):
        entry = {'setting': setting, 'grids': {}, 'grads': {}}
        gs = {}
        for tag, c in (('ref', c0),
                       ('plus', c0.copy()), ('minus', c0.copy())):
            if tag != 'ref':
                cc = c.copy()
                cc[DISP_ATOM, DISP_XYZ] += (STEP if tag == 'plus' else -STEP)
                c = cc
            e, g, npts = grad_at(mol0, c, setting)
            gs[tag] = g
            entry['grids'][tag] = npts
            entry['grads'][tag] = float(np.abs(g).max())
        dg = gs['plus'] - gs['minus']
        jump = float(np.abs(dg).max())
        entry['max_grad_jump_across_disp'] = jump
        entry['implied_H_error_jump_over_2h'] = jump / (2.0 * STEP)
        control[setting] = entry
        print('[%s] grid pts ref/plus/minus=%s  max|dg|=%.3e  implied|dH|=%.3e'
              % (setting, list(entry['grids'].values()), jump,
                 entry['implied_H_error_jump_over_2h']), flush=True)

    out = dict(
        job='JOB-2026-0905-009',
        step='final_validation_B_settings_audit_and_pruning_control',
        displacement=dict(atom=DISP_ATOM, xyz_index=DISP_XYZ, step_bohr=STEP,
                          note='water O atom, single Cartesian coordinate, both signs'),
        settings_audit=settings,
        paired_control=control,
        interpretation=('compare implied_H_error_jump_over_2h between default and '
                        'prune_off: a drastic drop confirms the pruning '
                        'attribution; an unchanged value means the discrepancy '
                        'is NOT (only) pruning and must be attributed otherwise'),
    )
    path = os.path.join(FV, 'stepB_settings_audit.json')
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print('SAVED ->', path)


if __name__ == '__main__':
    main()
