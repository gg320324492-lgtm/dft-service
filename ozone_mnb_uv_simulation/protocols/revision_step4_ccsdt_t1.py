#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Acceptance-revision step #4 (JOB-2026-0905-009): corrected CCSD(T)
diagnostics, recomputed on the REVISED candidate set.

Changes vs the superseded run_stage5_ccsdt.py:
  * the old value 0.0643 is re-labelled as `max_abs_t1` (= max |t1_ia|);
  * the INSTALLED PySCF standard diagnostic is also recorded:
      cc.ccsd.get_t1_diagnostic(t1) = sqrt(|t1|^2 / N_corr_electrons)
    (Lee-type T1, normalised by correlated electron count), together with the
    D1 diagnostic, the correlated-electron count, the frozen-core setting and
    RHF/CCSD convergence flags;
  * `single_reference_ok` is WITHDRAWN -- a single metric cannot establish
    the root cause of multireference character;
  * targets follow the REVISED dedup (c14_plus, c06_plus), not the old ones;
  * monomer single points are computed BOTH at their own optimised geometry
    and at the frozen complex geometry, so both the interaction energy and
    the deformation energy are available downstream;
  * t1 amplitudes are saved (.npy) for reuse; t2 (~1 GB for aug-cc-pVTZ on
    the complex) is NOT persisted -- listed as a known gap, recomputation
    from the archived geometry takes only a few minutes;
  * resume-safe: existing per-structure records are loaded and merged, only
    missing single points are computed.
"""
import os
import sys
import json
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import conformer_utils as cu

ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, 'run_artifacts', '01_pure_water_o3_h2o')
REV = os.path.join(ART, 'acceptance_revision', 'ccsdt_revision')
os.makedirs(REV, exist_ok=True)

BASIS = 'aug-cc-pVTZ'
MAX_MEM_MB = 4000
N_THREADS = 8
TARGETS = ['c14_plus', 'c06_plus']      # revised candidate set (step 2)
SYMS_COMPLEX = ['O', 'O', 'O', 'O', 'H', 'H']


def _mol(syms, coords):
    from pyscf import gto
    s = "; ".join("%s %.10f %.10f %.10f" % (sy, x, y, z)
                  for sy, (x, y, z) in zip(syms, coords))
    return gto.M(atom=s, basis=BASIS, charge=0, spin=0, verbose=0,
                 max_memory=MAX_MEM_MB)


def ccsdt_sp(sp_id, syms, coords, provenance):
    """RHF -> CCSD(T) single point with corrected diagnostics. Resume-safe."""
    path = os.path.join(REV, '%s.json' % sp_id)
    if os.path.exists(path):
        with open(path) as fh:
            print('[skip] %s already present' % sp_id, flush=True)
            return json.load(fh)
    import pyscf
    from pyscf import scf, cc
    t0 = time.time()
    mol = _mol(syms, coords)
    mf = scf.RHF(mol)
    mf.max_memory = MAX_MEM_MB
    mf.chkfile = os.path.join(REV, '%s_rhf.chk' % sp_id)
    mf.kernel()
    mycc = cc.CCSD(mf)
    mycc.max_memory = MAX_MEM_MB
    mycc.frozen = 0                      # record explicitly: no frozen core
    mycc.kernel()
    et = mycc.ccsd_t()
    t1 = np.asarray(mycc.t1)
    max_abs_t1 = float(np.max(np.abs(t1)))
    t1_diag = float(cc.ccsd.get_t1_diagnostic(t1))
    d1_diag = float(cc.ccsd.get_d1_diagnostic(t1))
    n_corr_elec = int(2 * t1.shape[0])   # RHF: t1 (nocc, nvir), nocc = N_elec/2
    np.save(os.path.join(REV, '%s_t1.npy' % sp_id), t1)
    res = dict(
        id=sp_id, provenance=provenance, basis=BASIS,
        pyscf_version=pyscf.__version__,
        geometry_angstrom=[[round(float(x), 8) for x in row] for row in coords],
        symbols=syms,
        n_basis_functions=int(mol.nao_nr()),
        n_correlated_electrons=n_corr_elec,
        frozen_orbitals=mycc.frozen,
        rhf_converged=bool(mf.converged), e_hf=float(mf.e_tot),
        ccsd_converged=bool(mycc.converged), e_ccsd=float(mycc.e_tot),
        e_triples_correction=float(et),
        e_ccsdt_total=float(mycc.e_tot + et),
        max_abs_t1=max_abs_t1,
        t1_diagnostic_pyscf=t1_diag,
        t1_diagnostic_definition='sqrt(|t1|^2 / N_corr_electrons), '
                                 'pyscf.cc.ccsd.get_t1_diagnostic',
        d1_diagnostic_janssen=d1_diag,
        singles_persisted='t1 saved as %s_t1.npy; t2 not persisted (~%.1f GB), '
                          'recomputable from the archived geometry in minutes'
                          % (sp_id, t1.nbytes * (t1.shape[1] ** 2 / t1.size) / 1e9),
        seconds=round(time.time() - t0, 1))
    with open(path, 'w') as fh:
        json.dump(res, fh, indent=2)
    print('[%s] E(CCSD(T))=%.10f  max_abs_t1=%.4f  T1(pyscf)=%.4f  D1=%.4f  %.1f s'
          % (sp_id, res['e_ccsdt_total'], max_abs_t1, t1_diag, d1_diag,
             res['seconds']), flush=True)
    return res


def main():
    # ---------------- geometries
    geo = {}
    for tid in TARGETS:
        c = json.load(open(os.path.join(ART, 'mode_follow', '%s.json' % tid)))['geometry_angstrom']
        geo[tid] = np.asarray(c, float)
    mono = {}
    for m in ('o3', 'h2o'):
        mono[m] = np.asarray(json.load(open(
            os.path.join(ART, 'gas_opt', '%s.json' % m)))['optimised_coords_angstrom'], float)

    tasks = []
    for tid in TARGETS:
        tasks.append(('complex_%s' % tid, SYMS_COMPLEX, geo[tid],
                      dict(role='complex', source=tid)))
        tasks.append(('o3_frozen_%s' % tid, ['O', 'O', 'O'],
                      geo[tid][:3], dict(role='monomer-frozen-at-complex',
                                         source=tid, fragment='o3')))
        tasks.append(('h2o_frozen_%s' % tid, ['O', 'H', 'H'],
                      geo[tid][3:], dict(role='monomer-frozen-at-complex',
                                         source=tid, fragment='h2o')))
    tasks.append(('o3_optimised', ['O', 'O', 'O'], mono['o3'],
                  dict(role='monomer-own-optimum', source='gas_opt/o3.json')))
    tasks.append(('h2o_optimised', ['O', 'H', 'H'], mono['h2o'],
                  dict(role='monomer-own-optimum', source='gas_opt/h2o.json')))

    for sp_id, syms, coords, prov in tasks:
        ccsdt_sp(sp_id, syms, coords, prov)

    # ---------------- merge summary (existing + new, resume-safe)
    summary = dict(job='JOB-2026-0905-009',
                   step='revision_ccsdt_diagnostics',
                   note=('supersedes run_artifacts/01_pure_water_o3_h2o/ccsdt/ccsdt.json; '
                         'single_reference_ok withdrawn'),
                   targets=TARGETS,
                   single_points={})
    for fn in sorted(os.listdir(REV)):
        if fn.endswith('.json') and not fn.startswith('ccsdt_revision_summary'):
            with open(os.path.join(REV, fn)) as fh:
                summary['single_points'][fn[:-5]] = json.load(fh)
    out_path = os.path.join(REV, 'ccsdt_revision_summary.json')
    with open(out_path, 'w') as fh:
        json.dump(summary, fh, indent=2)
    print('SAVED ->', out_path, flush=True)


if __name__ == '__main__':
    main()
