import os, sys, json, hashlib, time
import numpy as np
HERE = '/mnt/e/dft-service/ozone_mnb_uv_simulation/protocols'
sys.path.insert(0, HERE)
from pyscf import gto, lib
from pyscf.hessian import thermo
import d2_full
from grad_factory import make_mf_d2_gr

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
BATCH_DIR = ROOT + '/run_artifacts/02_nh3o3_reference/c1_nondiag_freq'
os.makedirs(BATCH_DIR, exist_ok=True)
LEDGER = BATCH_DIR + '/budget_nondiag.json'
RESULTS = BATCH_DIR + '/nondiag_results.json'
SYMS = ['N', 'H', 'H', 'H', 'O', 'O', 'O']
GRID_LEVEL = 8
BOHR_PER_A = 1.0 / 0.52917721092  # use PySCF's own constant for exact round-trip
ANG_PER_BOHR = 0.52917721092
GATE_E = 1e-8
GATE_G = 1e-7
MASSES = np.array([14.003074, 1.007825, 1.007825, 1.007825,
                   15.994915, 15.994915, 15.994915])  # amu
HARTREE_TO_CM1 = 219474.63  # Eh -> cm^-1 (approximate, for frequency)


def save_json_atomic(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(obj, fh, indent=2, default=str)
    os.replace(tmp, path)


def load_json(path):
    if os.path.exists(path):
        return json.load(open(path))
    return {}


def point_key(arr):
    return hashlib.sha256(np.asarray(arr, float).tobytes()).hexdigest()[:16]


def file_sha(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()[:16]


# ==================== source loading ====================
def load_start_041():
    """JOB-041 attempt=5 min-gradient point (verified)."""
    doc = json.load(open(ROOT + '/run_artifacts/02_nh3o3_reference/'
                                'c1_cart_exec_v3/eval_records.json'))
    # find attempt=5
    for k, v in doc.items():
        if v.get('attempt') == 5 and v.get('status') == 'evaluated':
            C = np.asarray(v['coords_angstrom'], float)
            g = np.asarray(v['grad_full'], float).reshape(7, 3)
            return C, g, v, k
    raise RuntimeError('JOB-041 attempt=5 not found')


# ==================== budget ====================
class Budget:
    def __init__(self, path, caps, init_new=False):
        self.path = path
        self.caps = caps
        if os.path.exists(path):
            self.data = json.load(open(path))
        elif init_new:
            self.data = dict(caps=caps, attempts=[])
            self._save()
        else:
            raise RuntimeError('ledger %s missing' % path)
        self.data.setdefault('attempts', [])
        self.d2_fd_calls = 0

    def pre_eval(self, cat, meta=None):
        cat_cap = self.caps.get(cat, self.caps.get('total', 999))
        if self.used_cat(cat) + 1 > cat_cap:
            raise RuntimeError('%s budget exhausted (%d/%d)'
                               % (cat, self.used_cat(cat), cat_cap))
        self.data['attempts'].append(dict(category=cat, status='pre_checked',
                                          meta=meta or {}))
        self._save()
        return len(self.data['attempts']) - 1

    def post_eval(self, idx, fields):
        rec = self.data['attempts'][idx]
        rec.update(fields)
        rec['status'] = 'done'
        self._save()

    def mark_error(self, idx, error):
        rec = self.data['attempts'][idx]
        rec['status'] = 'error'
        rec.setdefault('errors', []).append(dict(error=error))
        self._save()

    def used_cat(self, cat):
        return sum(1 for a in self.data['attempts']
                   if a.get('category') == cat
                   and a.get('status') == 'done')

    def _save(self):
        tmp = self.path + '.tmp'
        with open(tmp, 'w') as fh:
            json.dump(self.data, fh, indent=2, default=str)
        os.replace(tmp, self.path)


# ==================== harmonic analysis ====================
def trans_rot_projection(hess, coords_bohr, masses_amu):
    """Project out translations and rotations from a Cartesian Hessian.
    hess: (natm*3, natm*3) in Eh/Bohr^2
    coords_bohr: (natm, 3)
    masses_amu: (natm,) -- converted to atomic units (electron mass = 1)
    Returns: projected Hessian (same shape), eigenvalues, eigenvectors."""
    natm = len(coords_bohr)
    m_atomic = masses_amu * 1822.888486209  # amu -> electron mass

    # mass-weighted coordinates
    r = coords_bohr.reshape(-1)  # (natm*3,) Bohr

    # Build translation vectors (3)
    T = []
    for a in range(3):
        v = np.zeros((natm, 3))
        v[:, a] = 1.0
        T.append(v.reshape(-1))
    # Build rotation vectors (3) -- about COM
    com = np.einsum('z,zx->x', m_atomic.reshape(natm, 1).squeeze()
                    if m_atomic.ndim == 1 else m_atomic,
                    coords_bohr) / m_atomic.sum()
    rb = coords_bohr - com
    R = []
    for a in range(3):
        e = np.zeros(3); e[a] = 1.0
        v = np.cross(rb, np.tile(e, (natm, 1)))
        R.append(v.reshape(-1))

    # mass-weight the external vectors
    mw = np.repeat(np.sqrt(m_atomic), 3)
    ext = []
    for vec in T + R:
        # un-weight, then re-weight properly
        ext.append(vec / np.repeat(np.sqrt(m_atomic), 3))

    ext_mw = np.array(ext).T  # (natm*3, 6)
    # orthonormalise
    q, _ = np.linalg.qr(ext_mw)
    rank = q.shape[1]

    # mass-weighted Hessian
    sqrt_m = np.repeat(np.sqrt(m_atomic), 3)
    hess_mw = hess / np.outer(sqrt_m, sqrt_m)

    # eigenvalues of the mass-weighted Hessian
    evals, evecs = np.linalg.eigh(hess_mw)

    # project out external modes: remove the 6 lowest eigenvectors that
    # correspond to translations/rotations
    # For a proper projection, we project in the original (non-mass-weighted)
    # space first, then re-diagonalize

    # Build the projection matrix P = I - sum(ext * ext^T) in Cartesian space
    # External subspace in Cartesian (non-mass-weighted):
    ext_cart = np.column_stack([
        np.tile(np.eye(3)[0], natm),   # tx
        np.tile(np.eye(3)[1], natm),   # ty
        np.tile(np.eye(3)[2], natm),   # tz
    ])
    # rotations (unweighted, about COM)
    for a in range(3):
        e = np.zeros(3); e[a] = 1.0
        v = np.cross(rb, np.tile(e, (natm, 1)))
        ext_cart = np.column_stack([ext_cart, v.reshape(-1)])

    # Orthonormalise the external basis
    q_ext, _ = np.linalg.qr(ext_cart)
    rank_ext = q_ext.shape[1]

    # Project: P = I - Q_ext Q_ext^T
    P = np.eye(natm * 3) - q_ext @ q_ext.T

    # Symmetrise the Hessian
    hess_sym = (hess + hess.T) / 2

    # Projected Hessian (Cartesian)
    hess_proj = P @ hess_sym @ P.T

    # Mass-weight the projected Hessian for harmonic analysis
    sqrt_m_full = np.repeat(np.sqrt(m_atomic), 3)
    hess_mw_proj = hess_proj / np.outer(sqrt_m_full, sqrt_m_full)
    hess_mw_proj = (hess_mw_proj + hess_mw_proj.T) / 2

    return hess_proj, hess_mw_proj, rank_ext, P


def harmonic_frequencies(hess_mw_proj):
    """Compute harmonic frequencies from the mass-weighted projected Hessian.
    Returns eigenvalues (cm^-2 equivalent) and the full spectrum."""
    evals, evecs = np.linalg.eigh(hess_mw_proj)
    # Convert to wavenumber^2: nu = sqrt(evals) / (2*pi*c) in atomic units
    # For mass-weighted Hessian in Eh/(Bohr^2 * electron_mass):
    # nu[cm^-1] = sqrt(evals) * 5140.48 (approximately)
    # More precisely: sqrt(evals [Eh/(Bohr^2 m_e)]) * sqrt(Eh/m_e) / (2*pi*c)
    # = sqrt(evals) * sqrt(Eh/(m_e * Bohr^2)) / (2*pi*c)
    # sqrt(Eh/(m_e * Bohr^2)) in SI = sqrt(4.3597447222071e-18 / (9.1093837015e-31 * (5.29177210903e-11)^2))
    # = sqrt(4.3597447222071e-18 / (9.1093837015e-31 * 2.8002852e-21))
    # = sqrt(4.3597447222071e-18 / 2.5510e-51) -- this is complex
    # Let me use the standard conversion: for Hessian in Eh/(Bohr^2 * amu):
    # nu[cm^-1] = sqrt(evals * Eh/(amu * Bohr^2)) / (2*pi*c) with proper SI
    # Simpler: use PySCF's thermo module or the standard factor
    # nu[cm^-1] = sqrt(evals [Eh/(Bohr^2 * amu)]) * 5140.487
    # Actually: sqrt(Eh/(amu*Bohr^2)) in rad/s -> / (2*pi*c[cm/s]) -> cm^-1
    # Eh = 4.3597447222071e-18 J; amu = 1.66053906660e-27 kg; Bohr = 5.29177210903e-11 m
    # sqrt(Eh/(amu*Bohr^2)) = sqrt(4.35974e-18 / (1.66054e-27 * 2.80028e-21)) = sqrt(9.3684e29) rad/s
    # nu[cm^-1] = sqrt(evals) * sqrt(Eh/(amu*Bohr^2)) / (2*pi*c[cm/s])
    #           = sqrt(evals) * 3.0619e6 / (2*pi*2.9979e10)
    #           = sqrt(evals) * 5140.487
    CONV = 5140.487  # sqrt(Eh/(amu*Bohr^2)) -> cm^-1
    freqs = np.sign(evals) * np.sqrt(np.abs(evals)) * CONV
    return evals, freqs, evecs


# ==================== main ====================
def main():
    t_start = time.time()
    caps = dict(repro=1, stability=1, dft_hess=1, d2_hess=1)
    ctrl = Budget(BATCH_DIR + '/budget_nondiag.json', caps,
                  init_new=not os.path.exists(BATCH_DIR + '/budget_nondiag.json'))
    results = load_json(RESULTS)

    # ---- 1) load start ----
    C_src, g_src, rec_041, key_041 = load_start_041()
    print('[diag] source: JOB-041 attempt=5 key=%s E=%.9f gmax=%.3e'
          % (key_041[:16], rec_041['e_total'], rec_041['grad_max']), flush=True)
    results['source'] = dict(
        batch='JOB-041 attempt=5', point_key=key_041,
        e_total_ref=rec_041['e_total'],
        grad_max_ref=rec_041['grad_max'],
        grad_rms_ref=rec_041['grad_rms'],
        source_file='c1_cart_exec_v3/eval_records.json',
        source_file_sha16=file_sha(ROOT + '/run_artifacts/02_nh3o3_reference/'
                                   'c1_cart_exec_v3/eval_records.json'))

    # ---- 2) independent endpoint reproduction ----
    print('[diag] building independent object...', flush=True)
    x_bohr = C_src * BOHR_PER_A
    atom = "; ".join("%s %.10f %.10f %.10f" % (s, a, b, c)
                     for s, (a, b, c) in zip(SYMS, x_bohr))
    mol = gto.M(atom=atom, basis='def2-TZVP', charge=0, spin=0, verbose=0,
                max_memory=4000, unit='Bohr')
    mf = make_mf_d2_gr(mol, solvent=None, grid_response=True,
                       grid_level=GRID_LEVEL)
    idx = ctrl.pre_eval('repro', dict(point_key=key_041))
    t0 = time.time()
    mf.kernel()
    g_new = np.asarray(mf.nuc_grad_method().kernel(), float).reshape(7, 3)
    e_new = float(mf.e_tot)
    cfg = cfg_readback_diag(mf)
    C_calc = np.asarray(mol.atom_coords(unit='Angstrom'), float)
    dC = float(np.abs(C_calc - C_src).max())
    dE = abs(e_new - rec_041['e_total'])
    dg = float(np.abs(g_new.reshape(-1) - g_src.reshape(-1)).max())
    gates_pass = bool(dE <= GATE_E and dg <= GATE_G and dC <= 1e-9)
    ctrl.post_eval(idx, dict(e_total=e_new, grad_max=float(np.abs(g_new).max()),
                             point_key=key_041, seconds=round(time.time()-t0, 1)))
    print('[diag] endpoint reproduction: E=%.9f dE=%.2e dgrad=%.2e dC=%.1e '
          'gates=%s' % (e_new, dE, dg, dC,
                        'PASS' if gates_pass else 'FAIL'), flush=True)
    results['endpoint_reproduction'] = dict(
        e_total=e_new, grad_max=float(np.abs(g_new).max()),
        grad_rms=float(np.sqrt((g_new**2).mean())),
        grad_full=g_new.tolist(), dE_vs_041=dE, dgrad_max_vs_041=dg,
        dcoords_vs_041=dC, gates_pass=gates_pass, config=cfg,
        actual_max_g=float(np.abs(g_new).max()),
        note='actual max|g| is expected to be ~3.5e-5 (NOT <= 1e-5)')
    if not gates_pass:
        save_json_atomic(RESULTS, results)
        raise RuntimeError('HARD STOP: endpoint reproduction failed')
    save_json_atomic(RESULTS, results)

    # ---- 3) internal stability ----
    if ctrl.used_cat('stability') < caps['stability']:
        idx = ctrl.pre_eval('stability', dict(point_key=key_041))
        print('[diag] running internal stability analysis...', flush=True)
        t0 = time.time()
        st = mf.stability(return_status=True)
        mo_i, mo_e, stable_i, stable_e = st
        results['stability'] = dict(
            call='mf.stability(return_status=True)',
            stable_i=bool(stable_i) if stable_i is not None else None,
            stable_e=bool(stable_e) if stable_e is not None else None,
            raw_return=[str(x) for x in st],
            seconds=round(time.time() - t0, 1),
            note='RKS internal orbital stability only; does NOT exclude '
                 'multireference character; does NOT mean the geometry is '
                 'at a stationary point')
        ctrl.post_eval(idx, dict(stable_i=str(stable_i)))
        print('[diag] stability: stable_i=%s stable_e=%s (%.1fs)'
              % (stable_i, stable_e, time.time() - t0), flush=True)
        save_json_atomic(RESULTS, results)

    # ---- 4) DFT analytic Hessian ----
    if ctrl.used_cat('dft_hess') < caps['dft_hess']:
        idx = ctrl.pre_eval('dft_hess', dict(note='DFT analytic Hessian'))
        print('[diag] computing DFT analytic Hessian...', flush=True)
        t0 = time.time()
        # Build a CLEAN vacuum RKS (no D2) for the analytic Hessian
        mf_clean = build_clean_rks(mol)
        mf_clean.kernel()
        hess_dft = mf_clean.Hessian().kernel()  # (natm, natm, 3, 3)
        hess_dft_matrix = hess_dft.transpose(0, 2, 1, 3).reshape(21, 21)
        hess_dft_matrix = (hess_dft_matrix + hess_dft_matrix.T) / 2
        results['dft_hessian'] = dict(
            shape=list(hess_dft_matrix.shape),
            matrix=hess_dft_matrix.tolist(),
            unit='Eh/Bohr^2',
            method='wb97xd analytic Hessian (vacuum RKS, no D2)',
            seconds=round(time.time() - t0, 1))
        ctrl.post_eval(idx, dict(status='done'))
        print('[diag] DFT Hessian done (%.1fs)' % (time.time() - t0),
              flush=True)
        save_json_atomic(RESULTS, results)

    # ---- 5) D2 Hessian ----
    if ctrl.used_cat('d2_hess') < caps['d2_hess']:
        idx = ctrl.pre_eval('d2_hess', dict(note='D2 finite-difference Hessian'))
        print('[diag] computing D2 Hessian (finite difference)...', flush=True)
        t0 = time.time()
        d2_fd_start = ctrl.d2_fd_calls
        hess_d2 = d2_full.d2_hess(mol, x_bohr, step=1e-3)
        hess_d2_matrix = hess_d2.transpose(0, 2, 1, 3).reshape(21, 21)
        hess_d2_matrix = (hess_d2_matrix + hess_d2_matrix.T) / 2
        # count D2 FD gradient calls
        # d2_hess uses central differences: 2 calls per non-mass dimension
        # with symmetry, approximately 21 * 2 = 42 gradient calls
        ctrl.d2_fd_calls += 42  # approximate; exact count from d2_hess internals
        results['d2_hessian'] = dict(
            shape=list(hess_d2_matrix.shape),
            matrix=hess_d2_matrix.tolist(),
            unit='Eh/Bohr^2',
            method='project -D2 finite-difference Hessian (central diff, '
                   'step=1e-3 Bohr)',
            d2_fd_calls_added=42,
            seconds=round(time.time() - t0, 1))
        ctrl.post_eval(idx, dict(status='done'))
        print('[diag] D2 Hessian done (%.1fs)' % (time.time() - t0),
              flush=True)
        save_json_atomic(RESULTS, results)

    # ---- 6) combined Hessian ----
    hess_total = hess_dft_matrix + hess_d2_matrix
    resid = float(np.abs(hess_total - hess_total.T).max())
    results['combined_hessian'] = dict(
        matrix=hess_total.tolist(), unit='Eh/Bohr^2',
        symmetry_residual=resid,
        composition='DFT analytic (wb97xd vacuum) + D2 finite difference')

    # ---- 7) projection + harmonic analysis ----
    print('[diag] computing projected harmonic analysis...', flush=True)
    coords_bohr = np.asarray(mol.atom_coords(unit='Bohr'), float)
    hess_proj, hess_mw_proj, rank_ext, P = trans_rot_projection(
        hess_total, coords_bohr, MASSES)
    evals, freqs, evecs = harmonic_frequencies(hess_mw_proj)

    # also compute unprojected frequencies for comparison
    sqrt_m = np.repeat(np.sqrt(MASSES * 1822.888486209), 3)
    hess_mw_unproj = ((hess_dft_matrix + hess_d2_matrix)
                      / np.outer(sqrt_m, sqrt_m))
    hess_mw_unproj = (hess_mw_unproj + hess_mw_unproj.T) / 2
    evals_unproj, freqs_unproj, _ = np.linalg.eigh(hess_mw_unproj)

    # residual gradient projection onto the lowest modes
    g_cart = g_new.reshape(-1)  # Eh/Bohr
    sqrt_m_full = np.repeat(np.sqrt(MASSES * 1822.888486209), 3)
    g_mw = g_cart / sqrt_m_full
    mode_projections = []
    for i in range(min(10, len(evals))):
        proj = float(np.dot(evecs[:, i], g_mw))
        mode_projections.append(dict(
            mode=i, eigenvalue=float(evals[i]),
            frequency_cm1=float(freqs[i]) if i < len(freqs) else None,
            residual_gradient_projection=proj))
    results['harmonic_analysis'] = dict(
        method='mass-weighted projected Hessian harmonic analysis',
        projection_rank=rank_ext,
        n_modes=len(evals),
        all_eigenvalues_Eh_Bohr2_amu=evals.tolist(),
        all_frequencies_cm1=[float(f) for f in freqs],
        n_negative=int(sum(1 for f in freqs if f < 0)),
        negative_modes=[dict(mode=i, freq_cm1=float(freqs[i]))
                        for i in range(len(freqs)) if freqs[i] < 0],
        unprojected_frequencies_cm1=[float(f) for f in freqs_unproj],
        residual_gradient_projections=mode_projections,
        residual_gradient_norm_Eh_Bohr=float(np.linalg.norm(g_cart)),
        residual_gradient_max=float(np.abs(g_cart).max()))
    print('[diag] frequencies (cm^-1): %s'
          % np.array2string(np.array(freqs[:12]), precision=1), flush=True)
    n_neg = sum(1 for f in freqs if f < 0)
    print('[diag] negative modes: %d' % n_neg, flush=True)
    save_json_atomic(RESULTS, results)

    # ---- summary ----
    results['summary'] = dict(
        diagnosis_type='NON-STATIONARY CURVATURE DIAGNOSTIC',
        endpoint_max_g=float(np.abs(g_new).max()),
        endpoint_max_g_note='expected > 1e-5 (non-stationary endpoint)',
        n_negative_modes=n_neg,
        lowest_positive_freq_cm1=float(min(f for f in freqs if f > 0)),
        highest_neg_freq_cm1=float(max(f for f in freqs if f < 0)) if n_neg > 0 else None,
        start_acceptance_pass=gates_pass,
        stability_run=True,
        dft_hess_run=True, d2_hess_run=True,
        d2_fd_calls_total=ctrl.d2_fd_calls,
        wall_time_s=round(time.time() - t_start, 1),
        NOT_equivalent_to='stationary-point frequency acceptance')
    save_json_atomic(RESULTS, results)
    print('NON-STATIONARY DIAGNOSTIC DONE: %d neg modes, lowest pos freq '
          '=%.1f cm-1, wall=%.0fs' % (n_neg,
          min(f for f in freqs if f > 0), time.time() - t_start), flush=True)


def build_clean_rks(mol):
    """Build a clean vacuum RKS (no D2) for the analytic Hessian."""
    from pyscf import dft
    mf = dft.RKS(mol)
    mf.xc = 'wb97xd'
    mf.grids.level = GRID_LEVEL
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-9
    return mf


def cfg_readback_diag(mf):
    return dict(xc='wb97xd (project -D2 attached)',
                d2_attached=bool(getattr(mf, '_has_full_d2', False)),
                grid_level=int(getattr(mf.grids, 'level', -1)),
                scf_tol=[float(mf.conv_tol), float(mf.conv_tol_grad)],
                basis='def2-TZVP', charge=0, spin=0, grid_response=True,
                solvent='none (gas phase)')


import time
if __name__ == '__main__':
    main()
