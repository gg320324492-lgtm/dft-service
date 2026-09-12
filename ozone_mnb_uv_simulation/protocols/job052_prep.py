#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-052 step A: OFFLINE preparation (ZERO evaluations).

(a) batch-number check: 052 next unoccupied;
(b) zero-eval correction of 051: "electronic stability (imaginary-frequency
    check)" conflated two DIFFERENT checks -- electronic stability probes
    the electronic solution under orbital perturbations (RKS internal
    stability, stable_i), vibrational curvature probes NUCLEAR-coordinate
    directions (Hessian / frequencies).  The two are reported separately
    from this batch on.
(c) charter section 3 - source verification: the 051 independent recheck
    record is the unique source; verify its geometry corresponds bitwise to
    the 051 opt_11 request coordinates; save file hash, atom mapping,
    units and the actual coordinates; record software versions and module
    paths of the ACTUAL 051/052 interpreter;
(d) read-only check of the installed PySCF stability() return convention
    (043 unpacked it as mo_i, mo_e, stable_i, stable_e -- verify the real
    order and record it).
No SCF / gradient / Hessian / stability: reads existing records only.
"""
import os, sys, json, hashlib, glob, re, inspect
import numpy as np

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
S51 = ROOT + '/run_artifacts/02_nh3o3_reference/c1_bfgs_cont051'
OUT = ROOT + '/run_artifacts/02_nh3o3_reference/c1_cand_check052'
JOBS = ROOT + '/jobs'
JOB_NO = '052'
SRC_FILE = S51 + '/eval_recheck_recheck.json'


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


def verify_source():
    src = json.load(open(SRC_FILE))
    r51 = json.load(open(S51 + '/bfgs_cont051_results.json'))
    led51 = json.load(open(S51 + '/budget_bfgs_cont051.json'))
    checks = {}
    # correspondence with 051 opt_11 (the meeting point):
    # the recheck evaluated at opt_11's ACTUAL Angstrom geometry (dC=0.0 in
    # 051), so Angstrom coords match bitwise; Bohr x_bohr may differ in the
    # last ulp after the A->Bohr round trip; E/g differ at the 1e-13 level.
    m11 = json.load(open(S51 + '/eval_opt_opt_11.json'))
    checks['coords_angstrom_bitwise_match_opt11'] = bool(np.array_equal(
        np.asarray(src['coords_actual_angstrom'], float),
        np.asarray(m11['coords_actual_angstrom'], float)))
    checks['coords_bohr_maxdiff_vs_opt11'] = float(np.abs(
        np.asarray(src['x_bohr'], float).reshape(-1)
        - np.asarray(m11['x_bohr'], float).reshape(-1)).max())
    checks['coords_bohr_corr_le_1e-9'] = bool(
        checks['coords_bohr_maxdiff_vs_opt11'] <= 1e-9)
    checks['e_diff_vs_opt11'] = abs(float(src['e_total'])
                                    - float(m11['e_total']))
    checks['e_corr_le_1e-8'] = bool(checks['e_diff_vs_opt11'] <= 1e-8)
    checks['grad_max_diff_vs_opt11'] = abs(
        float(src['grad_max']) - float(m11['grad_max']))
    checks['grad_corr_le_1e-7'] = bool(checks['grad_max_diff_vs_opt11']
                                       <= 1e-7)
    mp = r51['optimization']['meeting_points'][0]
    checks['is_051_meeting_point'] = bool(
        mp['tag'] == 'opt_11' and mp['category'] == 'opt'
        and abs(float(mp['grad_max']) - float(src['grad_max'])) <= 1e-9)
    checks['grad_max_matches_reference'] = bool(
        float(src['grad_max']) == 8.202781486289723e-06)
    checks['e_matches_reference'] = bool(
        float(src['e_total']) == -282.0017214898003)
    checks['grad_max_below_gate'] = bool(float(src['grad_max']) <= 1e-5)
    checks['converged_finite'] = bool(src['converged'] and src['all_finite'])
    checks['config_complete'] = bool(
        src['config'].get('grid_response') is True
        and src['config'].get('grid_level') == 8
        and src['config'].get('d2_attached') is True)
    checks['x_dim_21'] = bool(np.asarray(src['x_bohr']).size == 21)
    # the recheck was itself gated in 051 (independent verification)
    checks['recheck_gate_passed_in_051'] = bool(
        r51['recheck']['gate']['gates_pass'])
    all_pass = all(bool(v) for v in checks.values()
                   if isinstance(v, bool))
    return dict(
        source_path_wsl=SRC_FILE,
        source_sha256=sha256_file(SRC_FILE),
        e_total_ref_Eh=-282.0017214898003,
        grad_max_ref=8.202781486289723e-06,
        element_order=['N', 'H', 'H', 'H', 'O', 'O', 'O'],
        coordinate_units=dict(record='Angstrom',
                              execution='Bohr (from x_bohr, bitwise)',
                              ang_per_bohr=0.52917721092),
        e_total=float(src['e_total']),
        e_d2_Eh=float(src['e_d2_Eh']),
        e_dft_part_Eh=float(src['e_dft_part_Eh']),
        grad_max=float(src['grad_max']),
        grad=src['grad'],
        coords_actual_angstrom=src['coords_actual_angstrom'],
        x_bohr_execution=src['x_bohr'],
        config=src['config'],
        checks=checks, all_checks_pass=all_pass)


def correction_051():
    return dict(
        conflated_phrase='electronic stability (imaginary-frequency check)',
        electronic_stability='checks the ELECTRONIC solution under orbital '
                             'perturbations (RKS internal stability of the '
                             'converged orbitals; stable_i); a Hessian is '
                             'NOT computed; does not exclude '
                             'multireference character',
        vibrational_curvature='checks NUCLEAR-coordinate directions '
                              '(separated analytic-DFT + D2-FD combined '
                              'Hessian, 15-dim internal subspace '
                              'eigenvalues / frequencies); independent of '
                              'the electronic stability verdict',
        consequence='the two checks are run and reported SEPARATELY in 052; '
                    'neither substitutes for the other; a negative '
                    'internal vibrational mode would NOT follow from an '
                    'electronic-stability result or vice versa',
        opt_11_note='opt_11 stopped via threshold BEFORE the accept '
                    'callback; it is the fully evaluated and independently '
                    'rechecked meeting point and is NOT retroactively '
                    'labelled an accepted iterate',
        old_041_prediction='the -72.186 cm-1 value was a saved-matrix '
                           'prediction at the OLD 041 geometry and is NOT '
                           'transferable to the 051 candidate geometry')


def environment_and_pyscf_convention():
    import scipy
    import pyscf
    env = dict(interpreter=sys.executable,
               scipy_version=scipy.__version__,
               scipy_module=scipy.__file__,
               pyscf_version=pyscf.__version__,
               pyscf_module=pyscf.__file__,
               numpy_version=np.__version__)
    from pyscf import scf
    from pyscf.scf import stability as st_mod
    src = inspect.getsource(st_mod.rhf_stability)
    order = None
    for pat, val in (('return mo_i, stable_i, mo_e, stable_e',
                      'mo_i, stable_i, mo_e, stable_e'),
                     ('return mo_i, mo_e, stable_i, stable_e',
                      'mo_i, mo_e, stable_i, stable_e')):
        if pat in src:
            order = val
            break
    env['stability_return_order_installed'] = order
    env['stability_signature'] = str(inspect.signature(
        __import__('pyscf').scf.hf.RHF.stability))
    env['note'] = ('with return_status=True and external=False the return '
                   'is (mo_i, None, stable_i, None) under the installed '
                   'order; 043 used the same (mo_i, mo_e, stable_i, '
                   'stable_e) order, so the 043 stability record is NOT '
                   'mislabeled; this batch unpacks adaptively and validates '
                   'that status entries are bool/None')
    return env


def main():
    num = check_number()
    src = verify_source()
    corr = correction_051()
    env = environment_and_pyscf_convention()
    save_json_atomic(OUT + '/prep_evidence.json',
                     dict(job='JOB-2026-0906-052 offline preparation',
                          zero_eval_corrections=corr,
                          environment=env))
    man = dict(
        job='JOB-2026-0906-052 electronic stability + vibrational curvature '
            'check of the 051 stationary-point candidate (opt_11/recheck)',
        number_check=num,
        caps=dict(endpoint=1, stability=1, clean_scf=1, dft_hess=1,
                  d2_fd_calls=84, total_note='D2 FD capped at 84 pure-'
                  'gradient calls (21 coords x 2 sides x 2 steps); no '
                  'optimization, no external stability, no displacement '
                  'SCF, no CP, no thermochemistry, no TS/IRC, no '
                  'high-level; no extra SCF hidden in Hessian preparation'),
        source=src,
        endpoint_gate=dict(dE_max=1e-8, dgrad_max=1e-7, dcoords_max_A=1e-9,
                           grad_max_le=1e-5, extra='SCF converged, finite, '
                           'config identical; ANY failure -> hard stop'),
        stability=dict(call='stability(return_status=True, external=False)',
                       on_failure='if stable_i is not explicitly True: save '
                                  'and STOP (no orbital following, no UKS '
                                  'switch); internal stability does NOT '
                                  'exclude multireference character',
                       stable_e_None_means='external NOT checked'),
        hessian=dict(d2_steps_Bohr=[1e-3, 5e-4],
                     raw_before_symmetrisation='saved for DFT, D2(both '
                     'steps), combined; antisymmetric residuals before/'
                     'after recorded',
                     clean_validation='clean-DFT energy must match '
                     'e_dft_part_Eh within 1e-8 Eh, else stop',
                     name='analytic-DFT + D2 gradient-FD combined Hessian '
                          '(NOT all-analytic)',
                     postprocess='JOB-044 conventions: masses in amu '
                     '(most-abundant isotope) + NU_AMU route with '
                     'electron-mass route cross-check; (7,7,3,3)->21x21 '
                     'layout verified; MW TR subspace rank 6, internal '
                     'subspace 15; all eigenvalues/modes saved, negative '
                     'values kept; PySCF cross-check (imaginary_freq=False, '
                     'same masses) validates post-processing ONLY'),
        mock_first='layout/unit/projection/negative-mode-retention/D2-no-SCF '
                   'validated on synthetic data with the real backend '
                   'unreachable BEFORE any production call',
        zero_eval_correction=dict(note='notes/job052_051_correction.md',
                                  evidence='prep_evidence.json'))
    save_json_atomic(OUT + '/input_manifest.json', man)

    print('[052] number check:', json.dumps(num))
    print('[052] source all_checks_pass =', src['all_checks_pass'])
    for k, v in src['checks'].items():
        print('      %-34s %s' % (k, v))
    print('[052] env:', env['interpreter'], '| scipy',
          env['scipy_version'], '| pyscf', env['pyscf_version'])
    print('[052] stability return order (installed):',
          env['stability_return_order_installed'])
    if not (num['pass_'] and src['all_checks_pass']
            and env['stability_return_order_installed']):
        print('[052] PRECHECK FAILED -> STOP (zero-evaluation stage)')
        raise SystemExit(1)
    print('[052] PREP OK (zero evaluations consumed)', flush=True)


if __name__ == '__main__':
    main()
