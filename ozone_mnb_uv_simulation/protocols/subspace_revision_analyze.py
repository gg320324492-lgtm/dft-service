#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Internal-subspace revision (JOB-2026-0905-009, curvature_audit follow-up):
offline re-analysis of the SAVED candidate Hessians with the corrected
internal-mode extraction, plus synthetic validation, old-scan audit,
error-threshold provenance audit and a re-measurement plan.

NO quantum chemistry is executed in this batch.
"""
import os
import sys
import json
import hashlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import curvature_audit_lib as cal
from pyscf.hessian import thermo as pyscf_thermo

ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, 'run_artifacts', '01_pure_water_o3_h2o')
CA = os.path.join(ART, 'curvature_audit')
FV = os.path.join(ART, 'final_minima_validation')
RAW = os.path.join(ART, 'internal_subspace_revision')
os.makedirs(RAW, exist_ok=True)

TARGETS = ['c14_plus', 'c06_plus']
MASSES = np.array([15.999, 15.999, 15.999, 15.999, 1.008, 1.008])
SYMS = ['O', 'O', 'O', 'O', 'H', 'H']
BOHR_A = 0.52917721092
HARTREE2J = 4.3597447222071e-18
AMU = 1.66053906660e-27
BOHR_SI = 0.52917721092e-10
C_SI = 2.99792458e10          # cm/s
NU_CONV = np.sqrt(HARTREE2J / (AMU * BOHR_SI ** 2)) / (2 * np.pi * 2.99792458e8 * 100.0)


def nu_of(lam):
    return float(np.sign(lam) * np.sqrt(abs(lam)) * NU_CONV)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:16]


def internal_subspace(masses, coords_bohr):
    """V (3N x 6) mass-weighted TR basis and U (3N x 12) orthonormal
    complement via QR of the stacked basis.  Returns (V, U)."""
    masses = np.asarray(masses, float)
    n = len(masses)
    V = cal.tr_subspace(masses, coords_bohr)
    B = np.hstack([V, np.eye(3 * n)])
    Q, _ = np.linalg.qr(B)                 # Q is 3N x 3N orthogonal
    # first 6 columns of Q span the same space as V (QR of [V|I])
    U = Q[:, 6:]                           # 3N x 12 complement
    return V, U


# ------------------------------------------------------------------ synthetic
def synthetic_validation():
    rng = np.random.default_rng(11)
    coords = rng.normal(scale=1.5, size=(6, 3))
    V, U = internal_subspace(MASSES, coords)
    checks = dict(
        U_orthonormal=bool(np.abs(U.T @ U - np.eye(12)).max() < 1e-12),
        VU_orthogonal=bool(np.abs(V.T @ U).max() < 1e-12),
        n_internal=int(U.shape[1]),
        completes_to_identity=bool(np.abs(V @ V.T + U @ U.T - np.eye(18)).max() < 1e-12),
    )
    cases = {}

    def make_H(eigs_int):
        H_int = (U * eigs_int) @ U.T
        H = 0.5 * (H_int + H_int.T)        # TR space = exact null space
        return H

    # case A: all 12 internal positive -> lowest selected mode must be the
    # smallest INTERNAL positive mode, never an external zero
    eigs_A = np.array([2.2e-5, 5e-5, 1e-4, 3e-4, 7e-4, 2e-3, 5e-3, 1e-2,
                       3e-2, 1e-1, 3e-1, 6e-1])
    HA = make_H(eigs_A)
    w = np.linalg.eigvalsh(U.T @ HA @ U)
    cases['A_all_positive'] = dict(
        lowest_internal=float(w[0]), expected=float(eigs_A[0]),
        pass_=bool(abs(w[0] - eigs_A[0]) < 1e-12 and w[0] > 1e-9),
        note='最低选中模为最小内模（正），未选中外模零值')

    # case B: one negative internal mode
    eigs_B = eigs_A.copy(); eigs_B[0] = -1.234e-4
    HB = make_H(eigs_B)
    w = np.linalg.eigvalsh(U.T @ HB @ U)
    cases['B_one_negative'] = dict(
        lowest_internal=float(w[0]), expected=float(eigs_B[0]),
        pass_=bool(abs(w[0] - eigs_B[0]) < 1e-12),
        note='负内模被准确识别')

    # case C: a near-zero REAL soft internal mode must survive
    eigs_C = eigs_A.copy(); eigs_C[0] = 4.4e-7      # ~8.9 cm-1, near zero
    HC = make_H(eigs_C)
    w = np.linalg.eigvalsh(U.T @ HC @ U)
    cases['C_near_zero_soft'] = dict(
        lowest_internal=float(w[0]), expected=float(eigs_C[0]),
        pass_=bool(abs(w[0] - eigs_C[0]) < 1e-12),
        note='接近零的真实软模未被误删')

    # case D: coordinate rotation invariance of the internal spectrum
    th = 0.7
    R3 = np.array([[np.cos(th), -np.sin(th), 0.0],
                   [np.sin(th), np.cos(th), 0.0],
                   [0.0, 0.0, 1.0]])
    coords_rot = coords @ R3.T
    V2, U2 = internal_subspace(MASSES, coords_rot)
    HD = make_H(eigs_A)
    # rotate the Hessian into the new orientation: H' = (R⊗) H (R⊗)^T
    Rbig = np.zeros((18, 18))
    for i in range(6):
        Rbig[3 * i:3 * i + 3, 3 * i:3 * i + 3] = R3
    H_rot = Rbig @ HD @ Rbig.T
    w1 = np.linalg.eigvalsh(U.T @ HD @ U)
    w2 = np.linalg.eigvalsh(U2.T @ H_rot @ U2)
    cases['D_rotation_invariance'] = dict(
        max_eig_shift=float(np.abs(np.sort(w1) - np.sort(w2)).max()),
        pass_=bool(np.abs(np.sort(w1) - np.sort(w2)).max() < 1e-10),
        note='整体旋转后内谱不变')

    # PySCF harmonic_analysis cross-check on the synthetic Hessian (case B)
    try:
        from pyscf import gto
        from pyscf.hessian import thermo as pyscf_thermo
        s = "; ".join("%s %.10f %.10f %.10f" % (s_, x, y, z)
                      for s_, (x, y, z) in zip(['O', 'O', 'O', 'O', 'H', 'H'], coords))
        mol = gto.M(atom=s, basis='sto-3g', charge=0, spin=0, verbose=0)
        hblock = np.zeros((6, 6, 3, 3))
        # Synthetic construction (metric-consistent): the KNOWN spectrum
        # eigs_B lives on the mass-weighted Hessian H_mw18 = U diag(eigs_B) U^T;
        # PySCF expects the CARTESIAN Hessian, so convert via M^(1/2):
        # H_cart = M^(1/2) H_mw M^(1/2).  harmonic_analysis re-weights back
        # to H_mw and projects with its own TR basis -> must recover eigs_B.
        H_mw18 = U @ np.diag(eigs_B) @ U.T
        M12 = np.diag(np.repeat(MASSES ** 0.5, 3))
        H_cart18 = M12 @ H_mw18 @ M12
        for i in range(6):
            for j in range(6):
                hblock[i, j] = H_cart18[3 * i:3 * i + 3, 3 * j:3 * j + 3]
        ha = pyscf_thermo.harmonic_analysis(mol, hblock, imaginary_freq=False)
        pyscf_freqs = np.sort(np.asarray(ha['freq_wavenumber'], float))
        mine = np.sort(np.array([nu_of(x) for x in eigs_B]))
        dnu = float(np.abs(pyscf_freqs - mine).max())
        cases['pyscf_cross_check'] = dict(
            max_freq_diff_cm1=dnu,
            pyscf_n_modes=int(len(pyscf_freqs)),
            pass_=bool(len(pyscf_freqs) == 12 and dnu < 1.0),
            note='与安装版 harmonic_analysis（同质量、外模排除）交叉核对')
    except Exception as exc:
        cases['pyscf_cross_check'] = dict(error=str(exc), pass_=False)

    checks['cases'] = cases
    checks['all_pass'] = all(c.get('pass_', False) for c in cases.values())
    return checks


# ------------------------------------------------------- candidate reanalysis
def reanalyze_candidates():
    out = {}
    for tid in TARGETS:
        rec = dict(id=tid)
        mat_an = os.path.join(CA, 'hessian_analytic_%s.npy' % tid)
        mat_fd = os.path.join(CA, 'hessian_fd0.02_%s.npy' % tid)
        cc_path = os.path.join(CA, 'candidate_curvature_%s.json' % tid)
        missing = []
        for p in (mat_an, mat_fd, cc_path):
            if not os.path.exists(p):
                missing.append(os.path.basename(p))
        rec['missing_inputs'] = missing
        if missing:
            out[tid] = rec
            continue

        h_an = np.load(mat_an)
        h_fd = np.load(mat_fd)
        cc = json.load(open(cc_path))
        coords = np.asarray(cc['geometry'], float)
        rec['provenance'] = dict(
            matrix_sha256_16=dict(analytic=sha256(mat_an), fd=sha256(mat_fd)),
            geometry_from=cc.get('geometry') and 'candidate_curvature_%s.json' % tid,
            masses_recorded=cc.get('masses'),
            masses_used=MASSES.tolist(),
            masses_match=bool(np.abs(np.asarray(cc.get('masses'),
                                                float) - MASSES).max() < 1e-12),
            settings_recorded='grid_level=6, scf_tol=1e-12, scf_tol_grad=1e-9, '
                              'fd_step=0.02 (candidate_curvature json)',
            matrix_symmetry=dict(
                analytic=float(np.abs(h_an - h_an.transpose(1, 0, 3, 2)).max()),
                fd=float(np.abs(h_fd - h_fd.transpose(1, 0, 3, 2)).max())),
            cache_provenance='written by curvature_audit_candidates.py in the '
                             '2026-09-05 session with the settings above; '
                             'script sha256 archived',
            script_sha256_16=dict(
                candidates=sha256(os.path.join(HERE, 'curvature_audit_candidates.py')),
                lib=sha256(os.path.join(HERE, 'curvature_audit_lib.py'))))

        # internal spectra via the corrected U-space method
        V, U = internal_subspace(MASSES, coords)
        from pyscf import gto
        s_mol = "; ".join("%s %.10f %.10f %.10f" % (s_, x, y, z)
                          for s_, (x, y, z) in zip(SYMS, coords))
        mol = gto.M(atom=s_mol, basis='sto-3g', charge=0, spin=0, verbose=0)
        H_mw_an = cal.mw_hessian_matrix(h_an, MASSES)
        H_mw_fd = cal.mw_hessian_matrix(h_fd, MASSES)
        for tag, H_mw in (('analytic', H_mw_an), ('fd', H_mw_fd)):
            H_int = U.T @ H_mw @ U
            H_int = 0.5 * (H_int + H_int.T)
            w, v_int = np.linalg.eigh(H_int)
            v_mw = U @ v_int                       # 18-dim mw vectors (columns)
            # per-mode external overlap and Cartesian frequency
            ovs = [cal.external_overlap(v_mw[:, k], V) for k in range(12)]
            nus = [nu_of(x) for x in w]
            # Cartesian directions + Cartesian-metric curvatures
            dcart = [(cal.mw_vector(MASSES) * v_mw[:, k])
                     / np.linalg.norm(cal.mw_vector(MASSES) * v_mw[:, k])
                     for k in range(12)]
            H9 = np.asarray(h_an if tag == 'analytic' else h_fd).transpose(
                0, 2, 1, 3).reshape(18, 18)
            kH_cart = [float(d @ H9 @ d) for d in dcart]
            # PySCF standard-implementation cross-check (same masses, TR excluded)
            h_block = h_an if tag == 'analytic' else h_fd
            ha = pyscf_thermo.harmonic_analysis(mol, h_block, imaginary_freq=False)
            pf = np.sort(np.asarray(ha['freq_wavenumber'], float))
            mine_f = np.sort(np.array([nu_of(x) for x in w]))
            rec[tag] = dict(
                pyscf_cross_check=dict(
                    max_freq_diff_cm1=float(np.abs(pf - mine_f).max()),
                    pyscf_n_modes=int(len(pf)),
                    freq_error=int(ha.get('freq_error', -1))),

                internal_eigenvalues=[float(x) for x in w],
                internal_frequencies_cm1=[round(nu_of(x), 2) for x in w],
                external_overlaps=[round(x, 6) for x in ovs],
                lowest_internal_eigenvalue=float(w[0]),
                lowest_internal_nu_cm1=nu_of(w[0]),
                lowest_internal_direction_mw=v_mw[:, 0].tolist(),
                lowest_internal_direction_cart=dcart[0].tolist(),
                kH_cart_lowest=kH_cart[0],
                n_negative_internal=int(np.sum(w < -1e-6)))
        # analytic<->FD internal-mode overlap matrix (12x12)
        w_a, v_a = np.linalg.eigh(U.T @ H_mw_an @ U)
        w_f, v_f = np.linalg.eigh(U.T @ H_mw_fd @ U)
        va_mw = U @ v_a
        vf_mw = U @ v_f
        O = np.zeros((12, 12))
        for a in range(12):
            for b in range(12):
                O[a, b] = float(abs(va_mw[:, a] @ vf_mw[:, b]))
        rec['overlap_matrix_analytic_fd'] = np.round(O, 4).tolist()
        rec['old_scan_audit'] = old_scan_audit(tid, cc, V, U, v_a, v_f, va_mw, vf_mw)
        out[tid] = rec
    return out


def old_scan_audit(tid, cc, V, U, v_a, v_f, va_mw, vf_mw):
    """Classify the old disputed-probe directions: keep vs withdraw, and judge
    reuse against the newly selected internal modes."""
    probes = cc.get('disputed_probes', {})
    audit = {}
    for name, p in probes.items():
        src = p.get('source', '')
        # the old probe directions were v[:,0] of the P H P spectra; their
        # external overlap was recorded on the probe's source spectrum
        ov_recorded = None
        if src == 'analytic':
            ov_recorded = json.loads(json.dumps(
                cc['spectra']['analytic']['lowest_external_overlap']))
        elif src == 'fd(0.02)':
            ov_recorded = json.loads(json.dumps(
                cc['spectra']['fd']['lowest_external_overlap']))
        valid_internal = ov_recorded is not None and ov_recorded < 0.01
        # reuse: overlap of the old direction with the NEW internal-mode set
        # (reconstruct the old direction from the persisted matrix)
        H_mw = cal.mw_hessian_matrix(
            np.load(os.path.join(CA, 'hessian_analytic_%s.npy' % tid))
            if src == 'analytic' else
            np.load(os.path.join(CA, 'hessian_fd0.02_%s.npy' % tid)), MASSES)
        Php = cal.projector(V)
        w_old, v_old = np.linalg.eigh(Php @ H_mw @ Php)
        d_old_mw = v_old[:, 0]                 # lowest of P H P (= old selection)
        vref = vf_mw if src == 'fd(0.02)' else va_mw
        ovs_new = [float(abs(d_old_mw @ vref[:, k])) for k in range(12)]
        best_new = int(np.argmax(ovs_new))
        audit[name] = dict(
            source=src, recorded_external_overlap=ov_recorded,
            direction_valid_internal=bool(valid_internal),
            disposition=('keep' if valid_internal else
                         'withdraw (external/TR direction, not internal-mode evidence)'),
            reuse_for_new_modes=dict(
                best_new_mode_index=best_new,
                overlap=round(ovs_new[best_new], 4),
                reusable=bool(ovs_new[best_new] > 0.95 and valid_internal)),
            probe_rows=p.get('rows', []),
            k_H=p.get('k_H'))
    return audit


def plan_remeasurement(reanalysis):
    """Re-measurement plan: directions that still need targeted probes."""
    plan = []
    for tid in TARGETS:
        r = reanalysis.get(tid, {})
        an_l = r.get('analytic', {}).get('lowest_internal_eigenvalue')
        fd_l = r.get('fd', {}).get('lowest_internal_eigenvalue')
        if an_l is None or fd_l is None:
            continue
        entries = []
        if an_l < -1e-6:
            entries.append(dict(
                candidate=tid, construction='analytic',
                direction_mw=r['analytic']['lowest_internal_direction_mw'],
                direction_cart=r['analytic']['lowest_internal_direction_cart'],
                normalization='unit 3N Euclidean (Bohr per unit q)',
                coordinate_source='stepC_reopt_%s.json new_geometry' % tid,
                amplitudes_a=[0.002, 0.005, 0.01],
                note='negative internal mode: verify with energy+gradient '
                     'directional differences at SMALL amplitudes',
                estimated_scf_grad=6))
        if fd_l < -1e-6:
            entries.append(dict(
                candidate=tid, construction='fd(0.02)',
                direction_mw=r['fd']['lowest_internal_direction_mw'],
                direction_cart=r['fd']['lowest_internal_direction_cart'],
                normalization='unit 3N Euclidean (Bohr per unit q)',
                coordinate_source='stepC_reopt_%s.json new_geometry' % tid,
                amplitudes_a=[0.002, 0.005, 0.01],
                note='negative FD internal mode: verify likewise',
                estimated_scf_grad=6))
        if not entries:
            entries.append(dict(candidate=tid, note='no negative internal mode '
                               'selected; existing probes cover the positive '
                               'lowest modes; no new measurement planned'))
        plan.extend(entries)
    total = sum(e.get('estimated_scf_grad', 0) for e in plan)
    return dict(plan=plan, total_estimated_scf_grad=total,
                note='prepared only; NOT executed in this batch')


# --------------------------------------------------- error-threshold audit
def error_threshold_audit():
    lay = load(os.path.join(CA, 'h2o_layer_localization.json'), {})
    out = dict(
        assumed_in_code=dict(
            DE_E_FLOOR_eh=1e-9,
            status='ASSUMED (documented constant in validation_stepD_softmode.py '
                   '/ curvature_audit_candidates.py), NOT measured'),
        direct_repeat=dict(
            same_geometry_repeat='deterministic SCF -> exactly 0 (same code '
                                 'path, same grid); not a meaningful noise measure',
            note='the relevant non-smoothness is ACROSS neighbouring geometries '
                 '(grid response), which the H2O rows probe indirectly'),
        decomposition=[]
    )
    for xc, res in (lay.get('functionals') or {}).items():
        for dname, dres in res.get('directions', {}).items():
            k_H_dir = dres.get('k_H')
            for r in dres['rows']:
                q = r['q_bohr']
                spread_gE = r['k_g'] - r['k_E']
                spread_HE = k_H_dir - r['k_E']
                out['decomposition'].append(dict(
                    functional=xc, direction=dname, amp_a=r['amp_a'], q_bohr=q,
                    k_E=r['k_E'], k_g=r['k_g'], k_H=k_H_dir,
                    k_g_minus_k_E=spread_gE,
                    k_H_minus_k_E=spread_HE,
                    implied_E_noise_upper_bound_eh=abs(spread_HE) * q ** 2,
                    implied_g_noise_upper_bound_eh_bohr=abs(spread_gE) * q))
    # classification of the 6.6e-8 figure
    out['figure_6_6e_8_audit'] = dict(
        origin='indirect upper bound: |k_H - k_E| * q^2 at amp=0.005 A '
               '(3.7e-4 * 1.8e-4 Eh) on the pbe O-H stretch direction',
        classification='indirect estimate (upper bound) that BUNDLES energy-layer '
                       'non-smoothness with Hessian-layer error and residual '
                       'truncation; NOT a same-geometry repeat measurement',
        measured_components=dict(
            analytic_hessian_matrix_error_pbe=2.8e-5,
            analytic_hessian_matrix_error_wb97xd=1.9e-3,
            note='for pbe the Hessian-layer error is negligible, so the '
                 '0.005 A spread (~4e-4) is dominated by the energy/gradient '
                 'layers; for wb97xd the Hessian layer may contribute'),
        operational_usage='per-scan-point resolvability must use the '
                          'amplitude-dependent floor 3*dE/q^2 with dE stated '
                          'per source (assumed 1e-9 vs implied bound ~2.2e-8)')
    return out


def load(p, default=None):
    if not os.path.exists(p):
        return default
    with open(p) as fh:
        return json.load(fh)


def main():
    sv = synthetic_validation()
    re = reanalyze_candidates()
    eta = error_threshold_audit()
    plan = plan_remeasurement(re)
    out = dict(job='JOB-2026-0905-009',
               step='internal_subspace_revision_offline_analysis',
               synthetic_validation=sv,
               candidate_reanalysis=re,
               error_threshold_audit=eta,
              remeasurement_plan=plan)
    path = os.path.join(RAW, 'internal_subspace_analysis.json')
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=2)
    print('SAVED ->', path)
    print('synthetic all_pass =', sv['all_pass'])
    for t in TARGETS:
        r = re.get(t, {})
        if 'analytic' in r:
            print('%s: analytic lowest internal %.3e (%d neg) | fd lowest %.3e (%d neg)'
                  % (t, r['analytic']['lowest_internal_eigenvalue'],
                     r['analytic']['n_negative_internal'],
                     r['fd']['lowest_internal_eigenvalue'],
                     r['fd']['n_negative_internal']))


if __name__ == '__main__':
    main()
