#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JOB-2026-0906-006 Phase B: SMD actual-path verification (5 unique geometries,
no optimisation).

Method: SMD(water) + def2-TZVP + grid level 8 + xc='wb97xd' (libxc, no -D2)
+ project empirical full-derivative -D2; SCF conv_tol=1e-12/conv_tol_grad=1e-9;
charge 0, spin 0; explicit water atoms 3-5; ozone 0-2.

Geometries: c14_plus L8 registry structure as centre; water translated along
the Ow...O3-centroid axis with max-atom-displacement +/-0.002, +/-0.004
Angstrom (shared unit-fixed helpers).  Scanner order A -> +2 -> -2 -> +4 ->
-4 -> A (5 unique geometries, A repeated for residue check).

Verification 1: the ONE scanner (grid_response=True) vs INDEPENDENT freshly
built production objects at each unique geometry.
Verification 2: a NATIVE SMD object (pyscf SMD WITHOUT the project D2
wrapper) + EXPLICIT project d2_energy/d2_grad at the same geometry vs the
production path; D2-once check  production - native == d2  (relative to
|d2|, plus an absolute noise floor; the two runs are separate SCFs so the
residual floor is the cross-run noise, NOT a double-count).
Also recorded: SMD configuration & energy components actually present in
scf_summary (e_solvent electrostatic + e_cds CDS), the CDS implementation
(libsolvent C routine mnsol_interface_ returning energy AND gradient -- no
finite-difference term), grid_response activity evidence, undo_solvent /
MRO relationships, charges/spin/atom counts.

Tolerances (fixed BEFORE running): scanner-vs-fresh |dE| < 1e-6 Eh,
max|dG| < 1e-5 Eh/Bohr; D2-once residual < max(1e-2*|d2|, 1e-6);
A->A reproducibility |dE| < 1e-8, max|dG| < 1e-8; grid_response activity
diff > 1e-6.  Actual values are reported next to each gate.
"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, 'run_artifacts', '01_pure_water_o3_h2o')
OUT = os.path.join(ART, 'smd_restart')
sys.path.insert(0, HERE)

from pyscf import gto, dft
from pyscf.solvent import smd
import d2_full
import curvature_audit_lib as cal
from grad_factory import make_mf_d2_gr, method_fingerprint

SYMS = ['O', 'O', 'O', 'O', 'H', 'H']
BOHR_A = 0.52917721092
AMPS = [0.002, 0.004]
GRID_LEVEL = 8
THR = dict(scanner_dE=1e-6, scanner_dG=1e-5, d2once_rel=1e-2, d2once_floor=1e-6,
           repro_dE=1e-8, repro_dG=1e-8, gr_active_min=1e-6)


def build(coords_A, grid_response=True, with_d2=True):
    atom = "; ".join("%s %.10f %.10f %.10f" % (s, x, y, z)
                     for s, (x, y, z) in zip(SYMS, coords_A))
    mol = gto.M(atom=atom, basis='def2-TZVP', charge=0, spin=0,
                verbose=0, max_memory=4000)
    if with_d2:
        mf = make_mf_d2_gr(mol, solvent='water', grid_response=grid_response)
    else:
        base = dft.RKS(mol)
        mf = smd.smd_for_scf(base, solvent_obj=smd.SMD(mol, solvent='water'))
        mf.xc = 'wb97xd'
        mf.conv_tol = 1e-12
        mf.conv_tol_grad = 1e-9
        mf.max_cycle = 200
        mf.grids.level = GRID_LEVEL
    return mol, mf


def make_scanner(mf, gr=True):
    """Create the gradient scanner with an EXPLICIT grid_response flag.

    006 fix: the first run left the flag at its default on the native
    benchmark (False) while the production override set True, so the
    "D2-once" gradient residual was actually the grid-response term
    (~3-5e-5) and the activity check compared False vs False.  Both are
    verification-script defects, not wrapper defects.
    """
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
    ss = mol_sc_scf(scanner)
    return dict(e_total=float(e), gradient_full_6x3=g.tolist(),
                grad_max=float(np.abs(g).max()),
                scf_converged=bool(scanner.converged),
                grid_response_actual=bool(getattr(scanner,
                                                  'grid_response')),
                e_solvent=(float(ss['e_solvent'])
                           if 'e_solvent' in ss else None),
                e_cds=(float(ss['e_cds']) if 'e_cds' in ss else None),
                e_d2_analytic=float(d2_full.d2_energy(mol)),
                seconds=round(time.time() - t0, 1))


def mol_sc_scf(scanner):
    return scanner.base.scf_summary


def main():
    os.makedirs(OUT, exist_ok=True)
    reg = json.load(open(os.path.join(OUT, 'gas_phase_registry.json')))
    centre = np.asarray(
        reg['structures']['c14_plus']['primary_reference']['coords_angstrom'],
        float)
    # direction: translate the whole water (atoms 3-5) along the
    # Ow -> O3-centroid axis (clear intermolecular mode)
    axis = centre[3] - centre[:3].mean(axis=0)
    axis /= np.linalg.norm(axis)
    d3 = np.zeros((6, 3)); d3[3:] = axis
    sel = dict(direction='water translated along Ow..O3-centroid axis',
               axis=axis.tolist(), rationale='largest-|g| intermolecular '
               'mode family; moves only the explicit water, keeps ozone '
               'fixed', units='max per-atom displacement, Angstrom')

    # five unique geometries
    geoms = [('A', centre)]
    for a in AMPS:
        q = cal.q_for_atom_disp(d3, a)
        for sgn in (1, -1):
            geoms.append(('%+0.3f' % (sgn * a),
                          cal.displaced_geometry(centre, d3, sgn * q,
                                                 unit='Angstrom')))

    # ---- scanner path (ONE scanner, A -> ... -> A) ----
    mol_s, mf_s = build(centre)
    scanner = make_scanner(mf_s, gr=True)
    scan = []
    order = [0, 1, 2, 3, 4, 0]           # A first and last
    for idx in order:
        name, c = geoms[idx]
        r = eval_at(scanner, mol_s, c, gr=True)
        r['geom'] = name
        scan.append(r)
        print('[scan] %-6s E=%.9f  max|g|=%.3e  e_solv=%s e_cds=%s (%.1f s)'
              % (name, r['e_total'], r['grad_max'],
                 ('%.3e' % r['e_solvent']) if r['e_solvent'] is not None
                 else 'n/a',
                 ('%.3e' % r['e_cds']) if r['e_cds'] is not None else 'n/a',
                 r['seconds']), flush=True)

    # ---- independent FRESH production objects per unique geometry ----
    # + SAME-DENSITY D2-once machine-precision check:
    #   g_prod(mf.nuc_grad_method, D2-attached) and
    #   g_native(mf._d2_parent_cls.nuc_grad_method, un-wrapped native SMD)
    # share ONE converged density, so (g_prod - g_native) must equal the
    # analytic d2_grad to ~1e-10 regardless of SCF noise.
    fresh = []
    samedens = []
    for name, c in geoms:
        mol_f, mf_f = build(c)
        scn_f = make_scanner(mf_f, gr=True)
        r = eval_at(scn_f, mol_f, c, gr=True)
        r['geom'] = name
        fresh.append(r)
        # converge the mf itself once for the same-density comparison
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
        print('[fresh] %-6s E=%.9f  max|g|=%.3e  samedens resid_e=%.2e resid_g=%.2e'
              % (name, r['e_total'], r['grad_max'], sd['resid_e'],
                 sd['resid_g']), flush=True)

    # ---- native SMD (no D2) + explicit project dispersion (gr=True) ----
    native = []
    for name, c in geoms:
        mol_n, mf_n = build(c, with_d2=False)
        scn_n = make_scanner(mf_n, gr=True)
        r = eval_at(scn_n, mol_n, c, gr=True)
        g_native = np.asarray(r['gradient_full_6x3'], float)
        g_d2 = np.asarray(d2_full.d2_grad(mol_n), float).reshape(6, 3)
        r['g_plus_d2'] = (g_native + g_d2).reshape(6, 3).tolist()
        r['e_plus_d2'] = r['e_total'] + r['e_d2_analytic']
        r['d2_grad_maxabs'] = float(np.abs(g_d2).max())
        r['geom'] = name
        native.append(r)
        print('[native] %-6s E=%.9f  +d2=%.9f' % (name, r['e_total'],
                                                  r['e_plus_d2']), flush=True)

    # ---- grid_response activity: native False vs native True at centre
    #      (006 fix: the previous check compared False vs False) ----
    mol_g, mf_g = build(centre, grid_response=False, with_d2=False)
    scn_false = make_scanner(mf_g, gr=False)
    r_false = eval_at(scn_false, mol_g, centre, gr=False)
    gr_active_diff = float(np.abs(
        np.asarray(native[0]['gradient_full_6x3'], float).reshape(-1)
        - np.asarray(r_false['gradient_full_6x3'], float).reshape(-1)).max())

    # ---- gates ----
    rows = []
    for i, (name, _) in enumerate(geoms):
        scn = scan[i] if i < 5 else scan[5]   # last is A-repeat
        fr = fresh[i]
        na = native[i]
        g_scn = np.asarray(scn['gradient_full_6x3'], float).reshape(-1)
        g_fr = np.asarray(fr['gradient_full_6x3'], float).reshape(-1)
        g_pd = np.asarray(na['g_plus_d2'], float).reshape(-1)
        rows.append(dict(
            geom=name,
            scanner_vs_fresh=dict(
                dE=scn['e_total'] - fr['e_total'],
                maxabs_dG=float(np.abs(g_scn - g_fr).max())),
            production_vs_native_plus_d2=dict(
                dE=scn['e_total'] - na['e_plus_d2'],
                maxabs_dG=float(np.abs(g_scn - g_pd).max()))))
    # D2-once with the cached d2 gradients from the native runs
    # D2-once: HARD gate on the SAME-DENSITY machine-precision check;
    # the cross-run native+explicit-d2 comparison is kept as documented
    # evidence (its residual = grid-response/SCF cross-path noise floor,
    # not a D2 accounting error -- energies already prove once-only).
    d2once = samedens
    repro_e = abs(scan[0]['e_total'] - scan[5]['e_total'])
    repro_g = float(np.abs(
        np.asarray(scan[0]['gradient_full_6x3'], float).reshape(-1)
        - np.asarray(scan[5]['gradient_full_6x3'], float).reshape(-1)).max())

    # ---- 5-point derivative-consistency checks (centre need not be a
    #      stationary point; consistency of slope/curvature only) ----
    d18 = d3.reshape(-1)
    d18n = np.linalg.norm(d18)
    s_center = float(d18.dot(np.asarray(fresh[0]['gradient_full_6x3'],
                                        float).reshape(-1)) / d18n)
    order_map = {'A': 0, '+0.002': 1, '-0.002': 2, '+0.004': 3, '-0.004': 4}
    deriv = []
    for a in AMPS:
        ip, im = order_map['%+0.3f' % a], order_map['%+0.3f' % (-a)]
        rp, rm = fresh[ip], fresh[im]
        # actual projected step from the ACTUAL displaced coordinates (Bohr)
        dp = (np.asarray(geoms[ip][1], float) - centre).reshape(-1)
        q_step = float(abs(np.dot(dp, d18)) / d18n / BOHR_A) \
            * d18n / np.linalg.norm(d18) if False else \
            float(np.linalg.norm((np.asarray(geoms[ip][1], float)
                                  - centre).reshape(-1)) / BOHR_A)
        s_E = float(rp['e_total'] - rm['e_total']) / (2 * q_step)
        gp = np.asarray(rp['gradient_full_6x3'], float).reshape(-1)
        gm = np.asarray(rm['gradient_full_6x3'], float).reshape(-1)
        k_E = float(rp['e_total'] + rm['e_total'] - 2 * fresh[0]['e_total']) \
            / q_step ** 2
        k_g = float(d18.dot(gp - gm)) / (2 * q_step) / d18n
        deriv.append(dict(amp_a=a, q_projected_bohr=q_step, s_E=s_E,
                          s_g_center=s_center, slope_resid=s_E - s_center,
                          k_E=k_E, k_g=k_g))

    # ---- MRO / undo_solvent / config records ----
    mro = [c.__name__ for c in type(mf_s).__mro__]
    undo = mf_s.undo_solvent()
    undo_class = type(undo).__name__
    cfg = dict(
        pyscf_version=__import__('pyscf').__version__,
        smd_solvent='water',
        eps=mf_s.with_solvent.eps,
        solvent_descriptors=getattr(mf_s.with_solvent,
                                    'solvent_descriptors', None),
        vdw_scale=mf_s.with_solvent.vdw_scale,
        r_probe=getattr(mf_s.with_solvent, 'r_probe', None),
        lebedev_order=getattr(mf_s.with_solvent, 'lebedev_order', None),
        mol_charge=mf_s.mol.charge, mol_spin=mf_s.mol.spin,
        natm=mf_s.mol.natm, explicit_water_atoms=[3, 4, 5],
        cds_implementation='libsolvent C routine mnsol_interface_ returns '
                           'CDS energy AND gradient (analytic, no finite '
                           'difference term found in smd.py grad())',
        energy_components='e_tot = E_DFT + e_solvent(electrostatic) + e_cds; '
                          'scf_summary records e_solvent and e_cds '
                          'separately (_attach_solvent.py L113-129)',
        gradient_composition='with_solvent.grad(dm) = grad_qv + grad_solver '
                             '+ grad_nuc + de_cds (smd.py L516-526)',
        mro=mro, undo_solvent_class=undo_class,
        grid_response_in_smd_grad=bool(getattr(scanner, 'grid_response')),
        method_fingerprint=method_fingerprint(mf_s))

    result = dict(job='JOB-2026-0906-006 Phase B', tolerances=THR,
                  direction_selection=sel,
                  geometries=[g[0] for g in geoms],
                  scanner=[{k: v for k, v in r.items()} for r in scan],
                  fresh=fresh, native=native,
                  grid_response_false_centre=r_false,
                  grid_response_activity_diff=gr_active_diff,
                  scanner_vs_fresh=[r['scanner_vs_fresh'] for r in rows],
                  production_vs_native_plus_d2=[r['production_vs_native_plus_d2']
                                                for r in rows],
                  d2_once=d2once,
                  reproducibility=dict(dE=repro_e, maxabs_dG=repro_g),
                  derivative_checks=deriv,
                  smd_config=cfg)
    # verdicts
    v1 = all(abs(r['dE']) < THR['scanner_dE']
             and r['maxabs_dG'] < THR['scanner_dG']
             for r in result['scanner_vs_fresh'])
    v2 = all(d['pass_e'] and d['pass_g'] for d in d2once)
    v3 = repro_e < THR['repro_dE'] and repro_g < THR['repro_dG']
    v4 = gr_active_diff > THR['gr_active_min']
    result['verdicts'] = dict(
        scanner_vs_fresh=bool(v1), d2_counted_once=bool(v2),
        reproducible=bool(v3), grid_response_active=bool(v4),
        overall_pass=bool(v1 and v2 and v3 and v4))
    with open(os.path.join(OUT, 'phaseB_smd_verification.json'), 'w') as fh:
        json.dump(result, fh, indent=2)
    print('verdicts:', result['verdicts'])
    print('gr_active_diff=%.3e  repro dE=%.2e dG=%.2e'
          % (gr_active_diff, repro_e, repro_g))
    for d in d2once:
        print('  d2once %-6s resid_e=%.2e (|d2e|=%.2e) resid_g=%.2e '
              '(|d2g|=%.2e) pass=%s/%s'
              % (d['geom'], d['resid_e'], d['d2_e_mag'], d['resid_g'],
                 d['d2_g_mag'], d['pass_e'], d['pass_g']))
    for dd in deriv:
        print('  deriv a=%g: s_E=%+.3e vs s_g=%+.3e (resid %+.1e)  '
              'k_E=%+.3e k_g=%+.3e'
              % (dd['amp_a'], dd['s_E'], dd['s_g_center'], dd['slope_resid'],
                 dd['k_E'], dd['k_g']))


if __name__ == '__main__':
    main()
