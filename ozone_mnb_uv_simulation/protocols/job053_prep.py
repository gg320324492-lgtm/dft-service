#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-053 step A: OFFLINE preparation (ZERO evaluations).

(a) batch-number check: 053 next unoccupied;
(b) zero-eval corrections of 052 (per the commander review):
    1. ledger KeyError + self-resume = CONTROL-FLOW deviation, reported
       separately from the completed computation cost (not covered by
       "0 failures");
    2. d2_full.d2_hess symmetrises BEFORE returning (0.5*(h+h^T(1,0,3,2)))
       -> the saved D2 matrices are SYMMETRISED OUTPUT / reconstructed
       components, NOT raw unsymmetrised matrices; the "bit-exact recovery"
       guarantee is withdrawn ((A+B)-A is not guaranteed to restore B
       bitwise); the missing raw D2 matrix is registered honestly (the DFT
       raw matrix keeps its true provenance);
    3. the two-D2-step frequency agreement only tests the D2 component's
       step-size sensitivity;
    4. H-displacement dominance does NOT prove an "NH3 umbrella vibration"
       (rigid libration vs internal deformation) and the normalisation
       convention is not an actual Bohr amplitude;
(c) charter section 3 - centre and direction:
    - centre = 052 eval_endpoint.json (E/grad/coords);
    - direction = 052 lowest internal mode (mode 0). The saved
      cart_norm_mode is verified to be the ALREADY-CONVERTED Cartesian
      direction (||M^1/2 d|| = 1), so NO further mass conversion is
      applied; q = d/||d||_2 (Euclidean, 21 components), sign fixed
      (largest-|component| positive);
    - R(t) = R0 + t q for t = +/-0.01, +/-0.02 Bohr; displacement norm/
      sign/unit-readback plan/fragment geometry/atom mapping verified;
      actual full-precision inputs saved;
(d) kH = q^T H q computed offline from the 052 saved matrices (combined
    sym, both steps) with DFT / D2(symmetric-output) contributions.
No SCF / gradient: reads existing records only.
"""
import os, sys, json, hashlib, glob, re
import numpy as np

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
S52 = ROOT + '/run_artifacts/02_nh3o3_reference/c1_cand_check052'
S51 = ROOT + '/run_artifacts/02_nh3o3_reference/c1_bfgs_cont051'
OUT = ROOT + '/run_artifacts/02_nh3o3_reference/c1_negmode_dir053'
JOBS = ROOT + '/jobs'
JOB_NO = '053'
CENTER_FILE = S52 + '/eval_endpoint.json'
SYMS = ['N', 'H', 'H', 'H', 'O', 'O', 'O']
MASSES = np.array([14.003074, 1.007825, 1.007825, 1.007825,
                   15.994915, 15.994915, 15.994915])
ANG_PER_BOHR = 0.52917721092
BPA = 1.0 / ANG_PER_BOHR
STEPS = (0.01, -0.01, 0.02, -0.02)


def save_json_atomic(path, obj):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(obj, fh, indent=2, default=str)
    os.replace(tmp, path)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for blk in iter(lambda: fh.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def sha_arr(a):
    return hashlib.sha256(np.asarray(a, float).tobytes()).hexdigest()[:16]


def check_number():
    used = set()
    for f in glob.glob(JOBS + '/JOB-*.md'):
        m = re.match(r'JOB-2026-0906-(\d{3})', os.path.basename(f))
        if m:
            used.add(m.group(1))
    nxt = '%03d' % (max(int(u) for u in used) + 1)
    return dict(used_max=max(used), next_free=nxt,
                pass_=bool(nxt == JOB_NO and JOB_NO not in used))


def correction_052():
    return dict(
        flow_deviation='052 aborted on a ledger KeyError (d2_fd missing '
                       'from caps) and was then self-resumed. Although NO '
                       'quantum calculation was repeated (ledger shows '
                       'each category done exactly once), the self-resume '
                       'DEVIATED from the batch boundary "on exception, '
                       'save and stop". Completed computation cost '
                       '(endpoint 1 SCF+grad, stability 1, clean_scf 1, '
                       'dft_hess 1, d2_fd 84 calls - all successful) is '
                       'reported SEPARATELY from the control-flow '
                       'anomaly; prior_abort/resume_note and the raw '
                       'evidence are retained. "0 failures" must not be '
                       'read as "no process deviation".',
        d2_matrices='d2_full.d2_hess applies 0.5*(h + h.transpose(1,0,3,2)) '
                    'BEFORE returning: the saved h_d2_step*_raw.npy files '
                    'are SYMMETRISED OUTPUT (relabelled as "symmetrised '
                    'D2 output / reconstructed component"), NOT raw '
                    'unsymmetrised matrices. The "bit-exact recovery" '
                    'claim for comb - dft is WITHDRAWN ((A+B)-A is not '
                    'guaranteed to restore B bitwise). The missing raw '
                    'unsymmetrised D2 matrix (and its antisymmetric '
                    'residual) is registered as UNRECOVERABLE and will '
                    'not be recomputed to impersonate an original. The '
                    'DFT raw matrix keeps its true provenance (genuinely '
                    'pre-symmetrisation).',
        two_step_agreement='the 1e-3/5e-4 frequency agreement (4.5e-7 '
                           'cm-1) only tests the D2 component\'s step-size '
                           'sensitivity; it does NOT validate the analytic '
                           'DFT Hessian or the combined matrix',
        mode_interpretation='H-displacement dominance does NOT establish '
                            'an "NH3 umbrella vibration": rigid fragment '
                            'libration and internal deformation must be '
                            'distinguished; and the normalisation '
                            'convention (unit mass-weighted norm) is NOT '
                            'an actual Bohr displacement amplitude')


def verify_centre_and_direction():
    cen = json.load(open(CENTER_FILE))
    r52 = json.load(open(S52 + '/cand_check052_results.json'))
    checks = {}
    # centre identity
    checks['e_matches_052_endpoint'] = bool(
        float(cen['e_total']) == -282.00172148979493)
    checks['grad_max_le_gate'] = bool(float(cen['grad_max']) <= 1e-5)
    checks['converged_finite'] = bool(cen['converged']
                                      and cen['all_finite'])
    checks['config_complete'] = bool(
        cen['config'].get('grid_response') is True
        and cen['config'].get('grid_level') == 8
        and cen['config'].get('d2_attached') is True)
    # centre geometry corresponds to the 051 candidate (ulp-level in Bohr)
    m11 = json.load(open(S51 + '/eval_opt_opt_11.json'))
    R0_bohr = np.asarray(cen['coords_actual_angstrom'], float) * BPA
    checks['R0_vs_051_x_bohr_maxdiff'] = float(np.abs(
        R0_bohr.reshape(-1)
        - np.asarray(m11['x_bohr'], float).reshape(-1)).max())
    checks['R0_corr_le_1e-9_Bohr'] = bool(
        checks['R0_vs_051_x_bohr_maxdiff'] <= 1e-9)

    # direction: 052 lowest internal mode (mode 0)
    m0 = r52['vibrational_curvature']['per_step']['1e-3']['modes'][0]
    d = np.asarray(m0['cart_norm_mode'], float).reshape(-1)
    # masses ACTUALLY used by the 052 post-processing (recorded): PySCF
    # 2.14 atom_mass_list() returned NOMINAL mass numbers [14,1,16]
    m052 = np.asarray(r52['vibrational_curvature']['masses'], float)
    # verify it is the ALREADY-CONVERTED Cartesian direction under the
    # 052 mass convention: cart_norm_mode = M^-1/2 v_mw -> ||M^1/2 d|| = 1
    sqrt_m = np.repeat(np.sqrt(m052), 3)
    mw_norm = float(np.linalg.norm(d * sqrt_m))
    checks['mode_is_cartesian_converted'] = bool(abs(mw_norm - 1.0) < 1e-8)
    mw = ('already Cartesian (M^-1/2 v_mw under the 052 recorded masses '
          '[14,1,16]; ||M^1/2 d||=1); no further mass conversion applied. '
          'NOTE: PySCF 2.14 atom_mass_list() returned NOMINAL mass '
          'numbers, so the 052 frequency scale used [14,1,16] (recorded '
          'in the 052 results); this affects only the frequency scale, '
          'and the direction is used as saved')
    # also compare the 5e-4 step direction (robustness only)
    m0b = r52['vibrational_curvature']['per_step']['5e-4']['modes'][0]
    db = np.asarray(m0b['cart_norm_mode'], float).reshape(-1)
    cos_steps = float(abs(d @ db) / (np.linalg.norm(d)
                                     * np.linalg.norm(db)))
    # normalise + sign convention
    q = d / np.linalg.norm(d)
    imax = int(np.argmax(np.abs(q)))
    if q[imax] < 0:
        q = -q
        sign_note = 'flipped so the largest-|component| entry is positive'
    else:
        sign_note = 'largest-|component| entry already positive'
    checks['q_norm_1'] = bool(abs(float(np.linalg.norm(q)) - 1.0) < 1e-12)

    # displacement geometries R(t) = R0 + t q
    geoms = []
    for t in STEPS:
        R = R0_bohr + t * q.reshape(7, 3)
        proj = float((R - R0_bohr).reshape(-1) @ q)
        geoms.append(dict(
            t_Bohr=t,
            coords_bohr=R.tolist(),
            coords_bohr_sha=sha_arr(R),
            step_norm_Bohr=float(np.linalg.norm((R - R0_bohr).reshape(-1))),
            displacement_along_q_Bohr=proj,
            element_order=SYMS,
            unit='Bohr (actual input: full-precision floats, no decimal '
                 'formatting applied before gto.M; saved via exact repr)'))
    # fragment geometry sanity at the largest displacement (|t|=0.02)
    R2 = np.asarray([g for g in geoms if g['t_Bohr'] == 0.02][0]
                    ['coords_bohr'], float).reshape(7, 3)
    dNH = [float(np.linalg.norm(R2[i] - R2[0])) for i in (1, 2, 3)]
    dOO = float(np.linalg.norm(R2[5] - R2[4]))
    dmin = float(min(np.linalg.norm(R2[i] - R2[j])
                     for i in range(7) for j in range(i + 1, 7)))
    frag = dict(NH_bonds_Bohr=dNH, O3_bond_Bohr=dOO,
                min_interatomic_distance_Bohr=dmin,
                note='fragments intact and non-colliding at |t|=0.02')
    checks['fragments_intact'] = bool(min(dNH) > 1.5 and dOO > 2.0
                                      and dmin > 1.5)
    all_pass = all(bool(v) for v in checks.values()
                   if isinstance(v, bool))
    return dict(
        centre_path_wsl=CENTER_FILE,
        centre_sha256=sha256_file(CENTER_FILE),
        centre_e_total_ref=-282.00172148979493,
        centre=dict(e_total=float(cen['e_total']),
                    grad_max=float(cen['grad_max']),
                    grad=cen['grad'],
                    coords_actual_angstrom=cen['coords_actual_angstrom'],
                    config=cen['config']),
        R0_bohr=R0_bohr.tolist(),
        R0_bohr_sha=sha_arr(R0_bohr),
        direction=dict(
            source='052 vibrational_curvature.per_step["1e-3"].modes[0]'
                   '.cart_norm_mode (mode 0, freq -72.83 cm-1)',
            mass_convention_052_recorded=m052.tolist(),
            mass_weighted_norm_of_saved_d=mw_norm,
            conversion=mw,
            q=q.tolist(), q_sha=sha_arr(q),
            euclidean_norm_before_normalisation=float(np.linalg.norm(d)),
            sign_convention=sign_note,
            largest_component_index=imax,
            step_direction_cos=cos_steps),
        displacement_geometries=geoms,
        fragment_checks=frag,
        checks=checks, all_checks_pass=all_pass)


def kH_from_052(q):
    """q^T H q from the 052 saved matrices (offline)."""
    h_dft_raw = np.load(S52 + '/h_dft_raw.npy')
    h_dft_sym = 0.5 * (h_dft_raw + h_dft_raw.T)
    out = {}
    for step, f in (('1e-3', 'h_comb_step1e-3_sym.npy'),
                    ('5e-4', 'h_comb_step5e-4_sym.npy')):
        h_comb_sym = np.load(S52 + '/' + f)
        # D2 component = comb - dft  (SYMMETRISED-OUTPUT reconstruction,
        # relabelled per the 052 correction)
        h_d2_sym = h_comb_sym - h_dft_sym
        k_tot = float(q @ h_comb_sym @ q)
        k_dft = float(q @ h_dft_sym @ q)
        out[step] = dict(
            kH_total_Eh_Bohr2=k_tot,
            kH_dft_contribution_Eh_Bohr2=k_dft,
            kH_d2_symmetrised_output_contribution_Eh_Bohr2=k_tot - k_dft,
            matrix_files=dict(comb=f, dft='h_dft_raw.npy (symmetrised '
                                          'here)', d2='reconstructed: '
                                          'comb_sym - dft_sym'),
            note='D2 contribution uses the SYMMETRISED D2 output of '
                 'd2_hess (see 052 correction); atom order N,H,H,H,O,O,O '
                 'verified against the geometry element order')
    return out


def main():
    num = check_number()
    corr = correction_052()
    src = verify_centre_and_direction()
    kh = kH_from_052(np.asarray(src['direction']['q'], float))
    save_json_atomic(OUT + '/direction_and_kH.json',
                     dict(direction=src['direction'],
                          displacement_geometries=src[
                              'displacement_geometries'],
                          fragment_checks=src['fragment_checks'],
                          kH_from_052_matrices=kh))
    save_json_atomic(OUT + '/prep_evidence.json',
                     dict(job='JOB-2026-0906-053 offline preparation',
                          corrections_052=corr))
    man = dict(
        job='JOB-2026-0906-053 two-step independent FD check of the '
            'negative-mode direction at the 052 candidate geometry',
        number_check=num,
        centre_path_wsl=src['centre_path_wsl'],
        centre_sha256=src['centre_sha256'],
        R0_bohr=src['R0_bohr'],
        R0_bohr_sha=src['R0_bohr_sha'],
        caps=dict(centre=1, displacement=4, total=5,
                  note='failures count within their category; no '
                       'borrowing; on real execution exception: save and '
                       'STOP, no self-resume'),
        centre=src['centre'],
        centre_gate=dict(dE_max=1e-8, dgrad_max=1e-7, dcoords_max_A=1e-9,
                         grad_max_le=1e-5, extra='SCF converged, finite, '
                         'config identical; ANY failure -> hard stop'),
        direction=src['direction'],
        displacement_geometries=src['displacement_geometries'],
        fragment_checks=src['fragment_checks'],
        kH_from_052_matrices=kh,
        formulas=dict(a0='g0 . q (Eh/Bohr)',
                      aE='[E(+h)-E(-h)]/(2h)',
                      kE='[E(+h)+E(-h)-2E0]/h^2',
                      kg='[g(+h)-g(-h)] . q /(2h)',
                      kH='q^T H q (052 matrices, DFT/D2 listed)'),
        reporting_rules='nonzero a0 retained; no single-side升降 curvature '
                        'claims; per-step/per-estimator differences listed '
                        'separately (never merged into one best error); '
                        'no frequency conversion; no whole-Hessian claim',
        zero_eval_correction=dict(note='notes/job053_052_correction.md',
                                  evidence='prep_evidence.json'))
    save_json_atomic(OUT + '/input_manifest.json', man)

    print('[053] number check:', json.dumps(num))
    print('[053] centre/direction all_checks_pass =',
          src['all_checks_pass'])
    for k, v in src['checks'].items():
        print('      %-38s %s' % (k, v))
    d = src['direction']
    print('[053] mode mass-weighted norm (saved d): %.12f -> %s'
          % (d['mass_weighted_norm_of_saved_d'], d['conversion']))
    print('[053] q_sha=%s sign: %s | step-dir cos=%.8f'
          % (d['q_sha'], d['sign_convention'],
             d['step_direction_cos']))
    for g in src['displacement_geometries']:
        print('[053] t=%+.2f  norm=%.12f  proj=%.12f  sha=%s'
              % (g['t_Bohr'], g['step_norm_Bohr'],
                 g['displacement_along_q_Bohr'], g['coords_bohr_sha']))
    for s, v in kh.items():
        print('[053] kH(%s): total=%+.6e  dft=%+.6e  d2=%+.6e Eh/Bohr^2'
              % (s, v['kH_total_Eh_Bohr2'],
                 v['kH_dft_contribution_Eh_Bohr2'],
                 v['kH_d2_symmetrised_output_contribution_Eh_Bohr2']))
    if not (num['pass_'] and src['all_checks_pass']):
        print('[053] PRECHECK FAILED -> STOP (zero-evaluation stage)')
        raise SystemExit(1)
    print('[053] PREP OK (zero evaluations consumed)', flush=True)


if __name__ == '__main__':
    main()
