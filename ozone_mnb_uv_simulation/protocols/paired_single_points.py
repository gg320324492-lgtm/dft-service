#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JOB-2026-0906-018 driver, v2 (JOB-2026-0906-019 recording-mechanism fix).

Changes vs v1 (defects established by the execution audit):
  * all configuration/attribute checks happen BEFORE any computation;
  * an attempt record is persisted BEFORE the SCF starts;
  * SCF / gradient / result-save states are recorded separately in a
    persistent ledger (protocols/sp_ledger.py);
  * on restart the ledger is loaded -> the budget is NOT refreshed, and
    already-saved points are REUSED (after coords+config verification)
    instead of being recomputed;
  * the surface read (which caused the JOB-018 run-2 loss) happens AFTER
    the result is saved and inside its own try/except, so a read error can
    no longer discard a computed result.

Historical note: the original 4 evaluations of JOB-018 (run 3) are already
recorded in single_points.json; THIS v2 driver is the corrected mechanism
for any future rerun/reuse (restarting it reuses the saved points without
new evaluations once their keys are in the ledger).
"""
import os
import sys
import json
import hashlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from pyscf import gto
import d2_full
from grad_factory import make_mf_d2_gr
from sp_ledger import SPLedger, make_key

SYMS = ['O', 'O', 'O', 'O', 'H', 'H']
GRID_LEVEL = 8
ORDER = 41
CAP = 4
OUTDIR = os.path.join(ROOT, 'run_artifacts', '01_pure_water_o3_h2o',
                      'paired_sp')
LEDGER = os.path.join(OUTDIR, 'attempts.json')


def sha(b):
    return hashlib.sha256(b).hexdigest()[:16]


def build_and_check(coords_A, solvent):
    """All configuration/attribute checks BEFORE any computation."""
    mol = gto.M(atom="; ".join("%s %.10f %.10f %.10f" % (s, x, y, z)
                               for s, (x, y, z) in zip(SYMS, coords_A)),
                basis='def2-TZVP', charge=0, spin=0, verbose=0,
                max_memory=4000)
    mf = make_mf_d2_gr(mol, solvent=solvent, grid_response=True,
                       grid_level=GRID_LEVEL)
    if solvent == 'water':
        mf.with_solvent.lebedev_order = ORDER
        cfg = dict(
            surface_discretization_method=str(
                mf.with_solvent.surface_discretization_method),
            lebedev_order=int(mf.with_solvent.lebedev_order),
            grid_level=int(mf.grids.level),
            d2_attached=bool(getattr(mf, '_has_full_d2', False)),
            scf_tol=[float(mf.conv_tol), float(mf.conv_tol_grad)])
        assert cfg['surface_discretization_method'] == 'SWIG', cfg
        assert cfg['lebedev_order'] == ORDER
    else:
        cfg = dict(surface_discretization_method='none (gas phase)',
                   lebedev_order=None, grid_level=int(mf.grids.level),
                   d2_attached=bool(getattr(mf, '_has_full_d2', False)),
                   scf_tol=[float(mf.conv_tol), float(mf.conv_tol_grad)])
    assert cfg['grid_level'] == GRID_LEVEL and cfg['d2_attached']
    return mol, mf, cfg


def compute_core(mol, mf):
    """SCF + gradient.  Returns the result dict WITHOUT surface info."""
    mf.kernel()
    scf_state = dict(converged=bool(mf.converged), e_tot=float(mf.e_tot))
    g = np.asarray(mf.nuc_grad_method().kernel(), float).reshape(6, 3)
    ss = mf.scf_summary
    result = dict(
        e_total=float(mf.e_tot),
        e_d2_in_scf=float(ss.get('d2_dispersion', float('nan'))),
        e_d2_analytic=float(d2_full.d2_energy(mol)),
        e_solvent=(float(ss['e_solvent']) if 'e_solvent' in ss else None),
        e_cds=(float(ss['e_cds']) if 'e_cds' in ss else None),
        gradient_full_6x3=g.tolist(),
        grad_max=float(np.abs(g).max()),
        grad_rms=float(np.sqrt((g ** 2).mean())),
        scf_converged=bool(mf.converged),
        finite=bool(np.isfinite(mf.e_tot) and np.isfinite(g).all()))
    return result, scf_state


def read_surface(mf, solvent):
    """Post-save enhancement; isolated so a read error cannot lose data."""
    if solvent != 'water':
        return dict(surface_points=None)
    surf = getattr(mf.with_solvent, 'surface', None)
    return dict(surface_points=(int(np.asarray(surf['area']).size)
                                if surf else None))


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    ledger = SPLedger(LEDGER, cap=CAP)
    print('ledger: %d prior attempts, cap %d, budget left %d'
          % (ledger.attempts_used(), ledger.cap, ledger.budget_left()),
          flush=True)

    prov = json.load(open(os.path.join(OUTDIR, 'provenance.json')))
    reg = json.load(open(os.path.join(
        ROOT, 'run_artifacts', '01_pure_water_o3_h2o', 'smd_restart',
        'gas_phase_registry.json')))
    coords = {}
    for sid in ('c06_plus', 'c14_plus'):
        c = np.asarray(reg['structures'][sid]['primary_reference']
                       ['coords_angstrom'], float)
        assert sha(c.tobytes()) == prov['candidates'][sid]['registry_coords_sha']
        coords[sid] = c

    plan = [('c06_L7endpoint_gas', coords['c06_plus'], None),
            ('c06_L7endpoint_smd', coords['c06_plus'], 'water'),
            ('c14_L8endpoint_gas', coords['c14_plus'], None),
            ('c14_L8endpoint_smd', coords['c14_plus'], 'water')]

    reused = computed = 0
    for tag, c, sv in plan:
        mol, mf, cfg = build_and_check(c, sv)   # checks BEFORE computation
        c_sha = sha(c.tobytes())
        key = make_key(tag, c_sha,
                       (cfg['surface_discretization_method'],
                        cfg['lebedev_order'], cfg['grid_level'],
                        cfg['d2_attached']))
        prior = ledger.get_saved(key)
        if prior is not None:
            # reuse only after coords+config verification
            assert prior['meta']['coords_sha'] == c_sha
            assert prior['meta']['config'] == cfg
            print('[reuse] %s (saved result verified against coords+config)'
                  % tag, flush=True)
            reused += 1
            continue
        if ledger.budget_left() <= 0:
            print('STOP: budget exhausted (attempts=%d, cap=%d)'
                  % (ledger.attempts_used(), ledger.cap), flush=True)
            break
        idx = ledger.start_attempt(key, dict(tag=tag, coords_sha=c_sha,
                                             config=cfg))
        try:
            result, scf_state = compute_core(mol, mf)
            ledger.update(idx, status='scf_grad_done', scf=scf_state,
                          result=result)
            if not result['finite']:
                raise RuntimeError('non-finite E/g (%s)' % tag)
            if not result['scf_converged']:
                raise RuntimeError('SCF not converged (%s)' % tag)
            # save the result IMMEDIATELY (before any fragile read)
            ledger.save_result(key, dict(meta=dict(tag=tag,
                                                   coords_sha=c_sha,
                                                   config=cfg),
                                         result=result))
            ledger.update(idx, status='saved')
            computed += 1
            print('[%d] %-22s E=%.9f  max|g|=%.3e  d2=%.6e  SAVED'
                  % (computed, tag, result['e_total'], result['grad_max'],
                     result['e_d2_analytic']), flush=True)
            # surface info: post-save enhancement, error-tolerant
            try:
                sinfo = read_surface(mf, sv)
                ledger.update(idx, surface=sinfo)
            except Exception as se:
                ledger.update(idx, surface_read_error=repr(se))
        except Exception as e:
            ledger.mark_error(idx, repr(e))
            print('ERROR on %s -> recorded, stopping (no retry): %s'
                  % (tag, e), flush=True)
            break

    print('DONE: computed=%d reused=%d attempts=%d cap=%d'
          % (computed, reused, ledger.attempts_used(), ledger.cap),
          flush=True)


if __name__ == '__main__':
    main()
