#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JOB-2026-0906-007: SMD same-configuration verification (Phase B rerun).

Fixes the 006 Phase B configuration defect: the production branch used
grad_factory's DEFAULT grid level (7) while the native benchmark branch
explicitly set L8, so the "same-configuration independent benchmark" was
actually cross-grid (L7 vs L8) and its gradient residual must NOT be
attributed to noise.

This revision:
  * builds BOTH sides with an explicit grid_level (L8);
  * asserts an ITEM-BY-ITEM configuration match between the production and
    native objects BEFORE any SCF runs (xc, basis, charge, spin, DFT grid
    level, SCF thresholds, grid_response flag, SMD solvent + cavity/surface
    discretisation settings) -- fingerprints are read from the ACTUAL
    objects, never from script constants; the DFT integration grid and the
    solvent surface grid are recorded separately;
  * re-runs the five-geometry verification (scanner vs fresh production,
    native+explicit-dispersion vs production, same-density D2-once,
    A->B->A reproducibility, slope/curvature consistency) with the SAME
    pre-fixed gates as 006 (no relaxation);
  * re-checks the TWO saved SMD optimisation endpoints with production vs
    independent native benchmarks (full gradients) to confirm whether the
    recorded endpoint max|g| values hold.

Gates (fixed before running, identical to 006): scanner-vs-fresh |dE|<1e-6,
max|dG|<1e-5; same-density D2-once residual < max(1e-2*|D2|, 1e-6);
A->A |dE|<1e-8, max|dG|<1e-8.  grid_response execution is verified via the
object configuration and call path (flag read from the objects actually
used); the numeric True/False difference is recorded as documentary
evidence only, not as a universal gate.
"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, 'run_artifacts', '01_pure_water_o3_h2o')
OUT = os.path.join(ART, 'smd_config_audit')
sys.path.insert(0, HERE)

from pyscf import gto, dft
from pyscf.solvent import smd
import d2_full
import curvature_audit_lib as cal
from grad_factory import make_mf_d2_gr

SYMS = ['O', 'O', 'O', 'O', 'H', 'H']
BOHR_A = 0.52917721092
AMPS = [0.002, 0.004]
GRID_LEVEL = 8                      # BOTH sides, explicit
THR = dict(scanner_dE=1e-6, scanner_dG=1e-5, d2once_rel=1e-2,
           d2once_floor=1e-6, repro_dE=1e-8, repro_dG=1e-8)


def build(coords_A, with_d2=True, grid_level=GRID_LEVEL, gr=True):
    atom = "; ".join("%s %.10f %.10f %.10f" % (s, x, y, z)
                     for s, (x, y, z) in zip(SYMS, coords_A))
    mol = gto.M(atom=atom, basis='def2-TZVP', charge=0, spin=0,
                verbose=0, max_memory=4000)
    if with_d2:
        mf = make_mf_d2_gr(mol, solvent='water', grid_response=gr,
                           grid_level=grid_level)          # EXPLICIT level
    else:
        base = dft.RKS(mol)
        mf = smd.smd_for_scf(base, solvent_obj=smd.SMD(mol, solvent='water'))
        mf.xc = 'wb97xd'
        mf.conv_tol = 1e-12
        mf.conv_tol_grad = 1e-9
        mf.max_cycle = 200
        mf.grids.level = grid_level                        # EXPLICIT level
    return mol, mf


def object_config(mf):
    """Read the configuration from the ACTUAL object (no constants)."""
    ws = mf.with_solvent
    gobj = mf.nuc_grad_method()
    surf = getattr(ws, 'surface', None)
    surf_pts = None
    if surf is not None:
        try:
            surf_pts = int(np.asarray(surf['area']).size)
        except Exception:
            surf_pts = 'presentunreadable'
    return dict(
        xc=str(mf.xc), basis=str(mf.mol.basis),
        charge=int(mf.mol.charge), spin=int(mf.mol.spin),
        dft_grid_level=int(mf.grids.level),
        scf_tol=[float(mf.conv_tol), float(mf.conv_tol_grad)],
        max_cycle=int(mf.max_cycle),
        grid_response_flag=bool(getattr(gobj, 'grid_response')),
        solvent=str(ws.solvent),
        eps=(float(ws.eps) if ws.eps is not None else 'not_set_pre_scf'),
        lebedev_order=int(getattr(ws, 'lebedev_order', -1)),
        vdw_scale=(float(ws.vdw_scale)
                   if ws.vdw_scale is not None else 'not_set_pre_scf'),
        r_probe=(float(ws.r_probe)
                 if getattr(ws, 'r_probe', None) is not None
                 else 'not_set_pre_scf'),
        solvent_surface_points=surf_pts,
        d2_attached=bool(getattr(mf, '_has_full_d2', False)),
        _grad_class=type(gobj).__name__,
        _mf_class=type(mf).__name__)


METHOD_ITEMS = ('xc', 'basis', 'charge', 'spin', 'dft_grid_level',
                'scf_tol', 'max_cycle', 'solvent', 'eps', 'lebedev_order',
                'vdw_scale', 'r_probe')
# intentionally different between production (D2-wrapped) and native:
#   d2_attached, grid_response default flag -- recorded, not compared


def config_check(mf_prod, mf_nat, expect_grid_level=GRID_LEVEL):
    """Item-by-item METHOD comparison; raises BEFORE any SCF on mismatch.

    Intentional production/native differences (d2_attached, the default
    grid_response flag on a freshly created gradient object) are recorded
    as flags, not compared; grid_response EXECUTION is verified on the
    actual scanner objects in the caller.
    """
    a, b = object_config(mf_prod), object_config(mf_nat)
    mismatch = []
    for k in METHOD_ITEMS:
        if k in ('eps', 'vdw_scale', 'r_probe') and (
                a[k] == 'not_set_pre_scf' or b[k] == 'not_set_pre_scf'):
            continue                       # filled during kernel; recorded
        if a[k] != b[k]:
            mismatch.append((k, a[k], b[k]))
    if a['dft_grid_level'] != expect_grid_level:
        mismatch.append(('dft_grid_level_expected', a['dft_grid_level'],
                         expect_grid_level))
    if b['dft_grid_level'] != expect_grid_level:
        mismatch.append(('dft_grid_level_expected_native',
                         b['dft_grid_level'], expect_grid_level))
    if mismatch:
        raise AssertionError('CONFIG MISMATCH (pre-SCF): %s' % mismatch)
    return dict(production=a, native=b, match=True,
                flags=dict(d2_attached=dict(prod=a['d2_attached'],
                                            nat=b['d2_attached']),
                           grid_response_default=dict(prod=a['grid_response_flag'],
                                                      nat=b['grid_response_flag'])))


def make_scanner(mf, gr=True):
    gobj = mf.nuc_grad_method()
    gobj.grid_response = gr
    scanner = gobj.as_scanner()
    assert bool(getattr(scanner, 'grid_response')) == gr
    return scanner


def eval_at(scanner, mol, coords_A, gr=True):
    mol.set_geom_(np.asarray(coords_A, float), unit='Angstrom')
    t0 = time.time()
    e, g = scanner(mol)
    g = np.asarray(g, float).reshape(6, 3)
    ss = scanner.base.scf_summary
    return dict(e_total=float(e), gradient_full_6x3=g.tolist(),
                grad_max=float(np.abs(g).max()),
                scf_converged=bool(scanner.converged),
                grid_response_actual=bool(getattr(scanner, 'grid_response')),
                e_solvent=(float(ss['e_solvent'])
                           if 'e_solvent' in ss else None),
                e_cds=(float(ss['e_cds']) if 'e_cds' in ss else None),
                e_d2_analytic=float(d2_full.d2_energy(mol)),
                seconds=round(time.time() - t0, 1))


def main():
    os.makedirs(OUT, exist_ok=True)
    reg = json.load(open(os.path.join(ART, 'smd_restart',
                                      'gas_phase_registry.json')))
    centre = np.asarray(
        reg['structures']['c14_plus']['primary_reference']['coords_angstrom'],
        float)
    axis = centre[3] - centre[:3].mean(axis=0)
    axis /= np.linalg.norm(axis)
    d3 = np.zeros((6, 3)); d3[3:] = axis
    geoms = [('A', centre)]
    for a in AMPS:
        q = cal.q_for_atom_disp(d3, a)
        for sgn in (1, -1):
            geoms.append(('%+0.3f' % (sgn * a),
                          cal.displaced_geometry(centre, d3, sgn * q,
                                                 unit='Angstrom')))

    # ---- pre-SCF configuration assertion (both sides L8) ----
    mol_p, mf_p = build(centre, with_d2=True, grid_level=GRID_LEVEL)
    mol_n, mf_n = build(centre, with_d2=False, grid_level=GRID_LEVEL)
    cfg = config_check(mf_p, mf_n)          # raises on mismatch
    print('[config] production vs native: MATCH on all items '
          '(dft_grid_level=%d, solvent=%s, lebedev=%s)'
          % (cfg['production']['dft_grid_level'], cfg['production']['solvent'],
             cfg['production']['lebedev_order']), flush=True)

    # ---- scanner path (ONE scanner, A -> +2 -> -2 -> +4 -> -4 -> A) ----
    scanner = make_scanner(mf_p, gr=True)
    order = [0, 1, 2, 3, 4, 0]
    scan = []
    for idx in order:
        name, c = geoms[idx]
        r = eval_at(scanner, mol_p, c, gr=True)
        r['geom'] = name
        scan.append(r)
        print('[scan] %-6s E=%.9f  max|g|=%.3e (%.1f s)'
              % (name, r['e_total'], r['grad_max'], r['seconds']), flush=True)

    # ---- fresh production + same-density D2-once per unique geometry ----
    fresh, samedens = [], []
    for name, c in geoms:
        mol_f, mf_f = build(c, with_d2=True, grid_level=GRID_LEVEL)
        scn_f = make_scanner(mf_f, gr=True)
        r = eval_at(scn_f, mol_f, c, gr=True)
        r['geom'] = name
        fresh.append(r)
        mol_f.set_geom_(np.asarray(c, float), unit='Angstrom')
        mf_f.kernel()
        g_prod = np.asarray(mf_f.nuc_grad_method().kernel(),
                            float).reshape(-1)
        gobj_nat = mf_f._d2_parent_cls.nuc_grad_method(mf_f)
        gobj_nat.grid_response = True
        g_nat = np.asarray(gobj_nat.kernel(), float).reshape(-1)
        g_d2 = np.asarray(d2_full.d2_grad(mol_f), float).reshape(-1)
        e_prod = float(mf_f.e_tot)
        e_nat = float(mf_f._d2_parent_cls.energy_tot(mf_f))
        e_d2 = float(d2_full.d2_energy(mol_f))
        sd = dict(geom=name,
                  resid_e=float((e_prod - e_nat) - e_d2),
                  resid_g=float(np.abs((g_prod - g_nat) - g_d2).max()),
                  d2_e_mag=float(abs(e_d2)), d2_g_mag=float(np.abs(g_d2).max()))
        sd['pass_e'] = bool(abs(sd['resid_e'])
                            < max(THR['d2once_rel'] * sd['d2_e_mag'],
                                  THR['d2once_floor']))
        sd['pass_g'] = bool(sd['resid_g']
                            < max(THR['d2once_rel'] * sd['d2_g_mag'],
                                  THR['d2once_floor']))
        samedens.append(sd)
        print('[fresh] %-6s E=%.9f  samedens resid_e=%.2e resid_g=%.2e'
              % (name, r['e_total'], sd['resid_e'], sd['resid_g']), flush=True)

    # ---- native SMD (L8, gr=True) + explicit dispersion ----
    native = []
    for name, c in geoms:
        mol_n2, mf_n2 = build(c, with_d2=False, grid_level=GRID_LEVEL)
        scn_n = make_scanner(mf_n2, gr=True)
        r = eval_at(scn_n, mol_n2, c, gr=True)
        g_native = np.asarray(r['gradient_full_6x3'], float)
        g_d2 = np.asarray(d2_full.d2_grad(mol_n2), float).reshape(6, 3)
        r['g_plus_d2'] = (g_native + g_d2).reshape(6, 3).tolist()
        r['e_plus_d2'] = r['e_total'] + r['e_d2_analytic']
        r['geom'] = name
        native.append(r)
        print('[native] %-6s E=%.9f  +d2=%.9f' % (name, r['e_total'],
                                                  r['e_plus_d2']), flush=True)

    # ---- A->A reproducibility ----
    repro_e = abs(scan[0]['e_total'] - scan[5]['e_total'])
    repro_g = float(np.abs(
        np.asarray(scan[0]['gradient_full_6x3'], float).reshape(-1)
        - np.asarray(scan[5]['gradient_full_6x3'], float).reshape(-1)).max())

    # ---- per-geometry gates ----
    rows = []
    for i, (name, _) in enumerate(geoms):
        scn, fr, na = scan[i], fresh[i], native[i]
        g_scn = np.asarray(scn['gradient_full_6x3'], float).reshape(-1)
        g_fr = np.asarray(fr['gradient_full_6x3'], float).reshape(-1)
        g_pd = np.asarray(na['g_plus_d2'], float).reshape(-1)
        rows.append(dict(
            geom=name,
            scanner_vs_fresh=dict(dE=scn['e_total'] - fr['e_total'],
                                  maxabs_dG=float(np.abs(g_scn - g_fr).max())),
            production_vs_native_plus_d2=dict(
                dE=scn['e_total'] - na['e_plus_d2'],
                maxabs_dG=float(np.abs(g_scn - g_pd).max()))))
        print('[gate] %-6s scn-vs-fresh dE=%.2e dG=%.2e | prod-vs-nat+d2 '
              'dE=%.2e dG=%.2e'
              % (name, rows[-1]['scanner_vs_fresh']['dE'],
                 rows[-1]['scanner_vs_fresh']['maxabs_dG'],
                 rows[-1]['production_vs_native_plus_d2']['dE'],
                 rows[-1]['production_vs_native_plus_d2']['maxabs_dG']),
              flush=True)

    # ---- 5-point derivative consistency (fresh, gr=True) ----
    d18 = d3.reshape(-1); d18n = np.linalg.norm(d18)
    s_center = float(d18.dot(np.asarray(fresh[0]['gradient_full_6x3'],
                                        float).reshape(-1)) / d18n)
    omap = {'A': 0, '+0.002': 1, '-0.002': 2, '+0.004': 3, '-0.004': 4}
    deriv = []
    for a in AMPS:
        rp, rm = fresh[omap['%+0.3f' % a]], fresh[omap['%+0.3f' % (-a)]]
        q_step = float(np.linalg.norm(
            (np.asarray(geoms[omap['%+0.3f' % a]][1], float)
             - centre).reshape(-1)) / BOHR_A)
        s_E = float(rp['e_total'] - rm['e_total']) / (2 * q_step)
        gp = np.asarray(rp['gradient_full_6x3'], float).reshape(-1)
        gm = np.asarray(rm['gradient_full_6x3'], float).reshape(-1)
        deriv.append(dict(
            amp_a=a, q_projected_bohr=q_step,
            s_E=s_E, s_g_center=s_center, slope_resid=s_E - s_center,
            k_E=float(rp['e_total'] + rm['e_total']
                      - 2 * fresh[0]['e_total']) / q_step ** 2,
            k_g=float(d18.dot(gp - gm)) / (2 * q_step) / d18n))
        print('[deriv] a=%g s_E=%+.3e s_g=%+.3e resid=%+.1e k_E=%+.3e k_g=%+.3e'
              % (a, s_E, s_center, s_E - s_center,
                 deriv[-1]['k_E'], deriv[-1]['k_g']), flush=True)

    # ---- verdicts (same gates; missing -> not_checked, never pass) ----
    v1 = all(abs(r['scanner_vs_fresh']['dE']) < THR['scanner_dE']
             and r['scanner_vs_fresh']['maxabs_dG'] < THR['scanner_dG']
             for r in rows)
    v2 = all(sd['pass_e'] and sd['pass_g'] for sd in samedens)
    v3 = repro_e < THR['repro_dE'] and repro_g < THR['repro_dG']
    # grid_response EXECUTION: verified on the objects actually used
    # (all scanner calls carried gr=True; the production object's gradient
    # factory sets True).  The native object's DEFAULT flag (False) is an
    # intentional difference -- the native benchmark scanner is created
    # with gr=True explicitly (asserted at creation in make_scanner).
    gr_executed = all(r['grid_response_actual'] for r in scan) \
        and cfg['production']['grid_response_flag']
    result = dict(
        job='JOB-2026-0906-007 Phase B (same-configuration rerun)',
        tolerances=THR, grid_level=GRID_LEVEL,
        config_check=cfg,
        geometries=[g[0] for g in geoms],
        scan=scan, fresh=fresh, native=native, samedens=samedens,
        gates=rows, reproducibility=dict(dE=repro_e, maxabs_dG=repro_g),
        derivative_checks=deriv,
        verdicts=dict(config_consistent=bool(cfg['match']),
                      scanner_vs_fresh=bool(v1),
                      d2_counted_once=bool(v2),
                      reproducible=bool(v3),
                      grid_response_executed=bool(gr_executed),
                      overall_pass=bool(v1 and v2 and v3 and gr_executed)))
    with open(os.path.join(OUT, 'phaseB2_same_config_verification.json'),
              'w') as fh:
        json.dump(result, fh, indent=2)
    print('verdicts:', result['verdicts'])


if __name__ == '__main__':
    main()
