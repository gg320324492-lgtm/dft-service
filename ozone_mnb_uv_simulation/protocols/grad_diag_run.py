import os, sys, json, hashlib, time
import numpy as np
HERE = '/mnt/e/dft-service/ozone_mnb_uv_simulation/protocols'
sys.path.insert(0, HERE)
from sp_ledger import SPLedger
from pyscf import gto
from torque_decomp import decompose_force
from grad_factory import make_mf_d2_gr

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
BATCH = ROOT + '/run_artifacts/02_nh3o3_reference/grad_diag'
LEDGER = BATCH + '/budget_diag.json'
CAPS = dict(total=5, diag=5)
GRID_LEVEL = 8
SYMS = ['N', 'H', 'H', 'H', 'O', 'O', 'O']
BOHR_PER_A = 1.8897261254578281
# reproduction gates (this batch's reproduction gates, NOT stationary
# acceptance standards)
GATE_E = 1e-8          # Eh
GATE_G = 1e-7          # Eh/Bohr (scalar adaptations, registered -- the 034
#                        record stores grad_max/grad_rms but no full vector)

r34 = json.load(open(ROOT + '/run_artifacts/02_nh3o3_reference/'
                            'c1_full_opt_cont/exec_results_formal.json'))
last = r34['trajectory'][-1]
coords_A = np.asarray(last['coords_angstrom'], float)
E_034 = last['e_total']
gmax_034 = last['grad_max']
grms_034 = last['grad_rms']

ledger = SPLedger(LEDGER, cap=CAPS['total'])


def build(coords, unit='Angstrom'):
    """Project method WITH the -D2 attached (identical to the 033/034
    optimization energy/gradient path) -- required for the 1e-8/1e-7
    reproduction gates to be meaningful."""
    atom = "; ".join("%s %.10f %.10f %.10f" % (s, x, y, z)
                     for s, (x, y, z) in zip(SYMS, coords))
    mol = gto.M(atom=atom, basis='def2-TZVP', charge=0, spin=0, verbose=0,
                max_memory=4000, unit=unit)
    mf = make_mf_d2_gr(mol, solvent=None, grid_response=True,
                       grid_level=GRID_LEVEL)
    return mol, mf


def guarded_eval(tag, mol, mf):
    if ledger.attempts_used() >= CAPS['total']:
        raise RuntimeError('TOTAL budget exhausted (5) - rejected before '
                           'evaluation')
    idx = ledger.start_attempt(json.dumps([tag], sort_keys=True),
                               dict(tag=tag, category='diag'))
    t0 = time.time()
    mf.kernel()
    g = np.asarray(mf.nuc_grad_method().kernel(), float).reshape(-1, 3)
    e = float(mf.e_tot)
    if not mf.converged:
        ledger.mark_error(idx, 'SCF not converged')
        raise RuntimeError('HARD STOP: SCF not converged (%s)' % tag)
    if not (np.isfinite(e) and np.isfinite(g).all()):
        ledger.mark_error(idx, 'non-finite')
        raise RuntimeError('HARD STOP: non-finite (%s)' % tag)
    ledger.update(idx, status='done', e_total=e,
                  grad_max=float(np.abs(g).max()),
                  grad_rms=float(np.sqrt((g ** 2).mean())),
                  seconds=round(time.time() - t0, 1))
    print('[%s] E=%.9f max|g|=%.6e (%.1fs)'
          % (tag, e, np.abs(g).max(), time.time() - t0), flush=True)
    return e, g


out = dict(job='JOB-2026-0906-035 gradient consistency diagnosis',
           method='project method WITH -D2 attached (identical to the 033/'
                  '034 optimization path); wb97xd/def2-TZVP/L8/grid_response/'
                  'SCF 1e-12,1e-9; gas',
           reproduction_gates=dict(energy=E_034, gate_E=GATE_E,
                                   gate_grad_scalar=GATE_G,
                                   adaptation='034 record stores grad_max/'
                                   'grad_rms but no full vector -> scalar '
                                   'gates used (registered)'))

# ================= 1) independent center reproduction =================
mol_c, mf_c = build(coords_A, 'Angstrom')
E_c, g_c = guarded_eval('center_reproduction', mol_c, mf_c)
cfg_c = dict(xc='wb97xd (project attachment)', grid_level=8,
             d2_attached=bool(getattr(mf_c, '_has_full_d2', False)),
             scf_tol=[float(mf_c.conv_tol), float(mf_c.conv_tol_grad)])
dE = abs(E_c - E_034)
dgmax = abs(float(np.abs(g_c).max()) - gmax_034)
dgrms = abs(float(np.sqrt((g_c ** 2).mean())) - grms_034)
out['center'] = dict(e_total=E_c, grad_max=float(np.abs(g_c).max()),
                     grad_rms=float(np.sqrt((g_c ** 2).mean())),
                     dE_vs_034=dE, dgrad_max_vs_034=dgmax,
                     dgrad_rms_vs_034=dgrms, config=cfg_c,
                     gates_pass=bool(dE <= GATE_E and dgmax <= GATE_G
                                     and dgrms <= GATE_G))
print('[center] gates: dE=%.2e (<=1e-8), dgrad_max=%.2e, dgrad_rms=%.2e '
      '(<=1e-7) -> %s' % (dE, dgmax, dgrms,
                          'PASS' if out['center']['gates_pass'] else 'FAIL'),
      flush=True)
if not out['center']['gates_pass']:
    json.dump(out, open(BATCH + '/diag_results.json', 'w'), indent=2)
    raise SystemExit('HARD STOP: reproduction gates failed - no direction '
                     'points; locate config/SCF-branch/path difference')

# ================= 2) fixed-direction two-step-size check =================
g_c_full = g_c.reshape(-1)                      # 21 components
q = -g_c_full / np.linalg.norm(g_c_full)        # unit direction, no projection
coords_B = coords_A * BOHR_PER_A
displacements = {}
for h in (0.001, 0.002):
    for sign, tag in ((+1.0, 'plus'), (-1.0, 'minus')):
        R = coords_B + sign * h * q.reshape(7, 3)
        mol, mf = build(R, 'Bohr')
        e, g = guarded_eval('disp_%s_h%.3f' % (tag, h), mol, mf)
        displacements['%s_h%.3f' % (tag, h)] = dict(
            e_total=e, grad_full=g.tolist(),
            g_dot_q=float(g.reshape(-1) @ q), h_Bohr=h)

# comparisons per step size
comp = {}
for h in (0.001, 0.002):
    gp = displacements['plus_h%.3f' % h]
    gm = displacements['minus_h%.3f' % h]
    g0q = float(g_c_full @ q)
    slope = (gp['e_total'] - gm['e_total']) / (2 * h)          # Eh/Bohr
    curv_E = (gp['e_total'] - 2 * E_c + gm['e_total']) / h ** 2  # Eh/Bohr^2
    curv_g = (gp['g_dot_q'] - gm['g_dot_q']) / (2 * h)           # Eh/Bohr^2
    comp['h%.3f' % h] = dict(
        g0_proj_Eh_per_Bohr=g0q,
        central_diff_slope_Eh_per_Bohr=slope,
        abs_diff=abs(g0q - slope),
        rel_diff=abs(g0q - slope) / max(abs(g0q), 1e-300),
        curvature_energy_Eh_per_Bohr2=curv_E,
        curvature_gradientproj_Eh_per_Bohr2=curv_g,
        curvatures_cross_check_consistent=bool(
            abs(curv_E - curv_g) < 0.05 * max(abs(curv_E), abs(curv_g),
                                              1e-30)))
    print('[h=%.3f] g0.q=%.8f slope=%.8f absdiff=%.2e rel=%.2e | curvE=%.4f '
          'curvG=%.4f' % (h, g0q, slope, comp['h%.3f' % h]['abs_diff'],
                          comp['h%.3f' % h]['rel_diff'], curv_E, curv_g),
          flush=True)

# ================= 3) center-gradient decomposition =================
masses = np.array([14.003074, 1.007825, 1.007825, 1.007825,
                   15.994915, 15.994915, 15.994915])
dec = decompose_force(g_c, coords_A, masses, unit='Angstrom', center='com')
recon_err = float(np.linalg.norm(
    g_c.reshape(-1) - (dec['g_trans_vec'] + dec['g_rot_vec'] + dec['g_int_vec'])))
gi = g_c.reshape(-1)
i_max = int(np.argmax(np.abs(gi)))
g_int_at = np.abs(np.asarray(dec['g_int_vec']).reshape(7, 3)).sum(axis=1)
out['center_gradient_decomposition'] = dict(
    metric='unweighted Cartesian force space, Eh/Bohr; external basis '
           'QR-orthonormalised (translations + rotations about COM, Bohr)',
    external_rank=dec['external_rank'],
    g_trans_norm=dec['g_trans_norm'], g_rot_norm=dec['g_rot_norm'],
    g_int_norm=dec['g_int_norm'], g_total_norm=dec['g_total_norm'],
    reconstruction_error=recon_err,
    net_force=dec['net_force'].tolist(),
    T_g=dec['T_g'].tolist(),
    largest_gradient=dict(atom_row=i_max // 3,
                          atom_symbol=SYMS[i_max // 3],
                          component=['x', 'y', 'z'][i_max % 3],
                          value=float(gi[i_max])),
    internal_abs_by_atom=[round(float(x), 8) for x in g_int_at],
    note='decomposing/removing external components does NOT change the '
         'acceptance status (endpoint remains an unconverged optimizer '
         'endpoint)')
out['displacements'] = displacements
out['comparisons'] = comp
json.dump(out, open(BATCH + '/diag_results.json', 'w'), indent=2)
print('GRADIENT DIAGNOSIS DONE: %d evals' % ledger.attempts_used(), flush=True)
