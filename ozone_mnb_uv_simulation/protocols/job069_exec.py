#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-069 step C: execution (REAL evaluations, ledgered).

Flow (HNO only - the single JOB-068 accepted P4 product):
  1. anchor attempt (cap 1): fresh SCF + full grid_response gradient at
     the persisted JOB-068 recheck geometry; centre-style gate vs that
     record (dE/dgrad/dcoords/config); the converged (mol, mf) is kept
     for the Hessian - no extra SCF;
  2. hno_hess_dft attempt (cap 1): analytical DFT Hessian
     (hess.grid_response=True, pure DFT part via the pre-attach parent
     kernel), explicit project D2 Hessian (d2_full.d2_hess, classical
     FD of the analytic D2 gradient - D2 FD gradient calls counted
     separately, NEVER in the SCF ledger), combined = sum, then
     cross-checked against the attached-kernel Hessian to <=1e-10;
  3. frequency analysis (0 SCF): PySCF thermo conventions
     (isotope-averaged masses, _get_TR projection rank 6, internal
     eigenproblem, ALL modes kept, negatives preserved), cross-check
     vs pyscf thermo.harmonic_analysis (FREQUENCIES ONLY);
  4. thermochemistry (0 SCF): verbatim ast-extracted project function
     run_baseline.thermochemistry (PySCF thermo() chemistry numbers
     NOT used), sigma=1 (Cs), mult=1, T=298.15 K, P=101325 Pa.

ANY real exception -> save state and STOP the whole batch (ledger
records the failure; no restart, no clearing, no borrowing).
"""
import os, sys, json, time, hashlib, math, traceback
import numpy as np

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
HERE = ROOT + '/protocols'
sys.path.insert(0, HERE)
import job047_exec as ex

OUT = ROOT + '/run_artifacts/02_nh3o3_reference/p4_freq069'
MANIFEST = OUT + '/input_manifest.json'
RESULTS = OUT + '/p4_freq069_results.json'
LEDGER = OUT + '/budget_p4freq069.json'
HDIR = OUT + '/hno'
ANG_PER_BOHR = ex.ANG_PER_BOHR


def sha_arr(a):
    return hashlib.sha256(np.asarray(a, float).tobytes()).hexdigest()[:16]


def au2wn(nist):
    return ((nist.HARTREE2J / (nist.ATOMIC_MASS * nist.BOHR_SI ** 2))
            ** .5 / (2 * math.pi) / nist.LIGHT_SPEED_SI * 1e-2)


# ===================== backend (settings = JOB-068) =====================
class FreqBackend:
    """Project-method backend for the anchor evaluation; keeps the
    converged (mol, mf) for the Hessian attempt."""

    def __init__(self, elements, keep):
        self._gto = None
        self._d2 = None
        self.elements = list(elements)
        self.keep = keep              # dict to store (mol, mf)

    def _lazy(self):
        if self._gto is None:
            from pyscf import gto
            import d2_full
            self._gto = gto
            self._d2 = d2_full

    def make_mol(self, R_bohr):
        self._lazy()
        atom = [(e, (float(r[0]), float(r[1]), float(r[2])))
                for e, r in zip(self.elements,
                                np.asarray(R_bohr, float).reshape(-1, 3))]
        return self._gto.M(atom=atom, basis='def2-TZVP', charge=0,
                           spin=0, verbose=0, max_memory=4000,
                           unit='Bohr')

    def full_eval(self, R_bohr, out_dir, tag):
        self._lazy()
        mol = self.make_mol(R_bohr)
        mf = self._d2.make_mf_d2(mol, solvent=None)
        mf.grids.level = 8
        mf.conv_tol = 1e-12
        mf.conv_tol_grad = 1e-9
        mf.chkfile = os.path.join(out_dir, 'chk_anchor_%s.chk' % tag)
        orig_kernel = mf.kernel

        def counting_kernel(*a, **k):
            mf._exec_scf_kernel_count = getattr(
                mf, '_exec_scf_kernel_count', 0) + 1
            return orig_kernel(*a, **k)

        mf.kernel = counting_kernel
        mf._exec_scf_kernel_count = 0
        mf.kernel()
        e = float(mf.e_tot)
        conv = bool(mf.converged)
        g_obj = mf.nuc_grad_method()
        g_obj.grid_response = True
        if getattr(g_obj, 'grid_response', None) is not True:
            raise RuntimeError('grid_response=True did not reach the '
                               'gradient object')
        g = np.asarray(g_obj.kernel(), float).reshape(-1, 3)
        e_d2 = float(self._d2.d2_energy(mol))
        cfg = dict(xc='wb97xd (project -D2 attached)',
                   d2_attached=bool(getattr(mf, '_has_full_d2', False)),
                   grid_level=int(getattr(mf.grids, 'level', -1)),
                   scf_tol=[float(mf.conv_tol),
                            float(mf.conv_tol_grad)],
                   basis='def2-TZVP', charge=0, spin=0,
                   grid_response=True, solvent='none (gas phase)')
        rec = dict(
            e_total=e, e_d2_Eh=e_d2, e_dft_part_Eh=e - e_d2,
            grad=g.tolist(), grad_max=float(np.abs(g).max()),
            grad_sha=sha_arr(g),
            coords_actual_angstrom=(np.asarray(R_bohr, float)
                                    .reshape(-1, 3)
                                    * ANG_PER_BOHR).tolist(),
            coords_sha=sha_arr(np.asarray(R_bohr, float)),
            config=cfg, converged=conv,
            all_finite=bool(np.isfinite(e) and np.isfinite(g).all()),
            scf_kernel_count=int(getattr(mf,
                                         '_exec_scf_kernel_count', -1)))
        if conv and rec['all_finite']:
            self.keep['mol'] = mol
            self.keep['mf'] = mf
        return rec


def hess_summary(name, h4):
    h33 = np.asarray(h4, float).transpose(0, 2, 1, 3).reshape(
        3 * h4.shape[0], 3 * h4.shape[0])
    return dict(name=name, shape=list(np.asarray(h4, float).shape),
                layout='(natm, natm, 3, 3) = PySCF (p,q,x,y)',
                sym_33_sha16=sha_arr(h33),
                fro_norm=float(np.linalg.norm(h33)),
                max_abs=float(np.abs(h33).max()),
                antisym_max=float(np.abs(h33 - h33.T).max()))


# ============== frequency analysis (PySCF thermo conventions) ==========
def get_TR(mass, coords):
    """Replicated from pyscf thermo._get_TR (as in
    endpoint_freq_step3_frequency.py, provenance registered)."""
    mass_center = np.einsum('z,zx->x', mass, coords) / mass.sum()
    coords = coords - mass_center
    massp = mass ** .5
    Tx = np.einsum('m,x->mx', massp, [1, 0, 0])
    Ty = np.einsum('m,x->mx', massp, [0, 1, 0])
    Tz = np.einsum('m,x->mx', massp, [0, 0, 1])
    im = np.einsum('m,mx,my->xy', mass, coords, coords)
    im = np.eye(3) * im.trace() - im
    w, paxes = np.linalg.eigh(im)
    w = w[::-1]; paxes = paxes[:, ::-1]
    ex_, ey, ez = paxes.T
    cr = coords.dot(paxes)
    cx, cy, cz = cr.T
    Rx = massp[:, None] * (cy[:, None] * ez - cz[:, None] * ey)
    Ry = massp[:, None] * (cz[:, None] * ex_ - cx[:, None] * ez)
    Rz = massp[:, None] * (cx[:, None] * ey - cy[:, None] * ex_)
    return np.vstack([Tx.ravel(), Ty.ravel(), Tz.ravel(),
                      Rx.ravel(), Ry.ravel(), Rz.ravel()])


def frequency_analysis(h4, coords_bohr, masses, AU2WN, elements):
    """TR-projected internal frequencies, negatives preserved."""
    n = len(masses)
    H33 = np.asarray(h4, float).transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)
    H33 = 0.5 * (H33 + H33.T)
    m = np.asarray(masses, float)
    mfull = np.repeat(m, 3)                     # one mass per coordinate
    Hmw = H33 / np.sqrt(np.outer(mfull, mfull))
    coords = np.asarray(coords_bohr, float).reshape(n, 3)
    TR = get_TR(m, coords)                      # (6, 3n)
    q, r = np.linalg.qr(TR.T)
    diag = np.abs(np.diag(r))
    rank6 = bool((diag > 1e-7).all()) and q.shape[1] == 6
    P = np.eye(3 * n) - q.dot(q.T)
    w, v = np.linalg.eigh(P)
    U = v[:, w > 1e-7]
    if U.shape != (3 * n, 3 * n - 6):
        raise RuntimeError('U shape %s != (%d, %d)'
                           % (U.shape, 3 * n, 3 * n - 6))
    if not rank6:
        raise RuntimeError('TR subspace rank != 6')
    ortho_err = float(np.abs(U.T.dot(q)).max())
    Hint = U.T.dot(Hmw).dot(U)
    ev, vec = np.linalg.eigh(Hint)
    modes_cart = U.dot(vec)
    freq_signed = np.sign(ev) * np.sqrt(np.abs(ev)) * AU2WN
    norm_mode = modes_cart.T.reshape(3 * n - 6, n, 3) * (m ** -0.5)[None, :, None]
    red_mass = 1.0 / np.einsum('izr,izr->i', norm_mode, norm_mode)
    # TR-contamination check: full 3n eigenvalues -> 6 should be ~0
    ev_full = np.linalg.eigvalsh(Hmw)
    ev_sorted = np.sort(ev_full)
    tr_contam_wn = np.sign(ev_sorted[:6]) * np.sqrt(
        np.abs(ev_sorted[:6])) * AU2WN
    return dict(
        tr_rank6=rank6, U_shape=list(U.shape),
        orthogonality_err_max=ortho_err,
        mass_amu=[float(x) for x in m],
        mass_convention='atom_mass_list(isotope_avg=True) (PySCF)',
        internal_eigenvalues_Eh_bohr2_amu1=[float(x) for x in ev],
        freq_wavenumber_signed=[float(x) for x in freq_signed],
        n_negative_modes=int((ev < 0).sum()),
        lowest_eigenvalue=float(ev[0]),
        lowest_freq_wn=float(freq_signed[0]),
        norm_mode=norm_mode.tolist(),
        reduced_mass_amu=[float(x) for x in red_mass],
        full_3n_lowest6_wavenumbers=[float(x) for x in tr_contam_wn],
        full_3n_lowest6_max_abs_wn=float(
            np.abs(tr_contam_wn).max()),
        elements=list(elements))


def pyscf_crosscheck(mol, h4, mine_signed):
    """FREQUENCIES ONLY (PySCF thermo() chemistry numbers NOT used)."""
    from pyscf.hessian import thermo as pyscf_thermo
    ha = pyscf_thermo.harmonic_analysis(
        mol, np.asarray(h4, float),
        exclude_trans=True, exclude_rot=True, imaginary_freq=True)
    ref = np.sort(np.asarray(ha['freq_wavenumber'], float))
    mine = np.sort(np.asarray(mine_signed, float))
    if ref.shape != mine.shape:
        raise RuntimeError('cross-check shape mismatch %s vs %s'
                           % (ref.shape, mine.shape))
    return dict(method='pyscf.hessian.thermo.harmonic_analysis '
                       '(same matrix, same isotope-avg masses, '
                       'imaginary_freq=True; FREQUENCIES ONLY)',
                pyscf_freq_wavenumber=[float(x) for x in ref],
                max_abs_wn_diff=float(np.abs(ref - mine).max()))


# ================= verbatim project thermochemistry ====================
def load_project_thermo(segment_path):
    """Exec the verbatim extraction in a namespace that provides the
    names its source references (data/math/np); the constants TEMP/
    PRESS/HA2JMOL/AMU are defined inside the extraction itself."""
    from pyscf import data
    src = open(segment_path, encoding='utf-8').read()
    header = src.splitlines()[0]
    body = '\n'.join(src.splitlines()[1:])
    ns = dict(data=data, math=math, np=np)
    exec(compile(body, segment_path, 'exec'), ns)   # noqa: S102
    return ns['thermochemistry'], header


def main():
    man = json.load(open(MANIFEST))
    if not man['number_check']['pass_']:
        raise RuntimeError('HARD STOP: number check failed')
    print('[069] HNO frequency acceptance (project method model; '
          'HNO ONLY - H2O2 not accepted in JOB-068)', flush=True)

    acc = man['accepted_endpoint']
    gates = man['gates']
    results = dict(job=man['job'], model_status=man['model_status'],
                   method=man['method'],
                   single_product_rule=man['single_product_rule'],
                   accepted_endpoint=acc,
                   d2_accounting=man['d2_accounting'])
    ledger = ex.Ledger(LEDGER, caps=man['caps'])
    keep = {}
    backend = FreqBackend(acc['elements'], keep)
    status = 'aborted'
    try:
        # ---------- 1. anchor ----------
        R0 = np.asarray(acc['coords_start_bohr'], float)
        ref = json.load(open(os.path.join(
            ROOT, acc['recheck_record'])))
        att = ledger.pre_eval('hno_anchor', dict(tag='anchor_hno'))
        try:
            rec = backend.full_eval(R0, HDIR, 'anchor_hno')
            rec['tag'] = 'anchor_hno'
            rec['category'] = 'hno_anchor'
            ex.save_json_atomic(os.path.join(
                HDIR, 'eval_hno_anchor_anchor_hno.json'), rec)
            ledger.post_eval(att, dict(stage='hno_anchor_anchor_hno',
                                       e_total=rec['e_total'],
                                       grad_max=rec['grad_max']))
        except Exception as e:
            ledger.fail(att, e)
            raise
        # centre-style gate vs the persisted recheck record
        ref_g = np.asarray(ref['grad'], float).reshape(-1)
        ref_C = np.asarray(ref['coords_actual_angstrom'], float)
        gate = dict(
            dcoords_A=float(np.abs(np.asarray(
                rec['coords_actual_angstrom'], float) - ref_C).max()),
            dE=abs(float(rec['e_total']) - float(ref['e_total'])),
            dgrad_max=float(np.abs(
                np.asarray(rec['grad'], float).reshape(-1)
                - ref_g).max()),
            converged=bool(rec['converged']),
            finite=bool(rec['all_finite']),
            config_match=bool(rec['config'] == ref['config']))
        gate['gates_pass'] = bool(
            gate['dcoords_A'] <= gates['dcoords_A']
            and gate['dE'] <= gates['dE']
            and gate['dgrad_max'] <= gates['dgrad']
            and gate['converged'] and gate['finite']
            and gate['config_match'])
        rec['gate'] = gate
        ex.save_json_atomic(os.path.join(
            HDIR, 'eval_hno_anchor_anchor_hno.json'), rec)
        results['anchor'] = dict(record=rec, gate=gate)
        if not gate['gates_pass']:
            raise RuntimeError('HARD STOP: anchor gate failed: %s'
                               % json.dumps(gate))
        print('[069] anchor PASS: dE=%.2e dgrad=%.2e dC=%.2e'
              % (gate['dE'], gate['dgrad_max'], gate['dcoords_A']),
              flush=True)

        # ---------- 2. Hessian (one ledgered attempt) ----------
        mol = keep['mol']
        mf = keep['mf']
        att = ledger.pre_eval('hno_hess_dft', dict(tag='hess_hno'))
        try:
            hobj = mf.Hessian()
            hobj.grid_response = True
            if getattr(hobj, 'grid_response', None) is not True:
                raise RuntimeError('grid_response=True did not reach '
                                   'the Hessian object')
            from pyscf.data import nist
            AU2WN = au2wn(nist)
            masses = np.asarray(
                mol.atom_mass_list(isotope_avg=True), float)
            coords_bohr = np.asarray(
                mol.atom_coords(unit='Bohr'), float)
            # explicit D2 Hessian with a call counter (classical, no SCF)
            import d2_full
            counter = {'n': 0}
            orig_d2g = d2_full.d2_grad

            def counting_d2g(m_, coords=None):
                counter['n'] += 1
                return orig_d2g(m_, coords)

            d2_full.d2_grad = counting_d2g
            try:
                t0 = time.time()
                hess_d2 = np.asarray(
                    d2_full.d2_hess(mol, coords_bohr), float)
                t_d2 = time.time() - t0
                n_d2_explicit = counter['n']
            finally:
                d2_full.d2_grad = orig_d2g
            # pure analytical DFT Hessian (pre-attach parent kernel)
            t0 = time.time()
            hess_dft = np.asarray(
                hobj._d2_parent_cls.kernel(hobj), float)
            t_dft = time.time() - t0
            hess_sum = hess_dft + hess_d2
            # attached kernel (should add the same D2 term exactly once)
            counter2 = {'n': 0}
            orig_d2g = d2_full.d2_grad

            def counting_d2g2(m_, coords=None):
                counter2['n'] += 1
                return orig_d2g(m_, coords)

            d2_full.d2_grad = counting_d2g2
            try:
                t0 = time.time()
                hess_kernel = np.asarray(hobj.kernel(), float)
                t_kern = time.time() - t0
                n_d2_kernel = counter2['n']
            finally:
                d2_full.d2_grad = orig_d2g
        except Exception as e:
            ledger.fail(att, e)
            raise
        ledger.post_eval(att, dict(
            stage='hno_hess_dft_hess_hno',
            note='analytical DFT Hessian + explicit D2 Hessian + '
                 'combined check; no additional SCF',
            hess_dft_fro=float(np.linalg.norm(
                hess_dft.transpose(0, 2, 1, 3).reshape(9, 9))),
            t_dft_s=round(t_dft, 1), t_d2_s=round(t_d2, 1),
            t_kernel_s=round(t_kern, 1)))
        dev = float(np.abs(hess_kernel - hess_sum).max())
        hs = dict(
            hess_dft=hess_summary('hess_dft_analytical', hess_dft),
            hess_d2=hess_summary('hess_d2_explicit_FD_of_d2_grad',
                                 hess_d2),
            hess_combined=hess_summary('hess_combined_sum', hess_sum),
            hess_attached_kernel=hess_summary(
                'hess_attached_kernel', hess_kernel),
            combined_check=dict(
                definition='max|hess(attached kernel) - '
                           '(hess_dft + hess_d2)|',
                max_abs_Eh_bohr2=dev,
                gate=gates['combined_hess'],
                pass_=bool(dev <= gates['combined_hess'])),
            d2_fd_gradient_calls=dict(
                explicit_d2_hess=n_d2_explicit,
                combined_kernel=n_d2_kernel,
                expected=man['d2_accounting'][
                    'expected_calls_explicit_d2_hess'],
                note='classical FD of the analytic D2 gradient; '
                     'counted separately, NEVER in the SCF ledger'),
            timings_s=dict(dft=round(t_dft, 1), d2=round(t_d2, 1),
                           kernel=round(t_kern, 1)))
        np.save(os.path.join(HDIR, 'hess_hno_dft.npy'), hess_dft)
        np.save(os.path.join(HDIR, 'hess_hno_d2.npy'), hess_d2)
        np.save(os.path.join(HDIR, 'hess_hno_combined.npy'), hess_sum)
        np.save(os.path.join(HDIR, 'hess_hno_attached_kernel.npy'),
                hess_kernel)
        if not hs['combined_check']['pass_']:
            raise RuntimeError('HARD STOP: combined Hessian check '
                               'failed: %.3e' % dev)
        results['hessian'] = hs
        print('[069] Hessian done: combined check dev=%.2e '
              '(d2 fd calls: %d explicit + %d kernel)'
              % (dev, n_d2_explicit, n_d2_kernel), flush=True)

        # ---------- 3. frequencies (0 SCF) ----------
        fa = frequency_analysis(hess_sum, coords_bohr, masses, AU2WN,
                                acc['elements'])
        fa['pyscf_crosscheck'] = pyscf_crosscheck(
            mol, hess_sum, fa['freq_wavenumber_signed'])
        results['frequency'] = fa
        n_neg = fa['n_negative_modes']
        print('[069] frequencies (cm^-1): %s | negatives=%d | '
              'cross-check maxdiff=%.2e'
              % (['%.2f' % f for f in fa['freq_wavenumber_signed']],
                 n_neg, fa['pyscf_crosscheck']['max_abs_wn_diff']),
              flush=True)

        # ---------- 4. thermochemistry (0 SCF) ----------
        thermo_fn, header = load_project_thermo(
            os.path.join(HDIR,
                         'run_baseline_thermochemistry_verbatim.py'))
        e_elec = float(rec['e_total'])
        th_j = thermo_fn(mol, fa['freq_wavenumber_signed'], e_elec,
                         mult=man['thermochemistry']['mult'],
                         sigma=man['thermochemistry']['sigma'])
        from pyscf.data import nist as _nist
        HA2J = _nist.HARTREE2J * _nist.AVOGADRO
        j2eh = 1.0 / HA2J
        j2kcal = 1.0 / 4184.0
        th = dict(
            source=man['thermochemistry'],
            source_header=header,
            raw_J_per_mol=th_j,
            e_elec_Eh=e_elec,
            conversions=dict(
                J_per_mol_to_Eh=j2eh,
                J_per_mol_to_kcal_mol=j2kcal,
                note='1 kcal = 4184 J exactly'))
        for key in ('ZPE', 'thermal_H_corr', 'thermal_G_corr'):
            th[key + '_Eh'] = float(th_j[key]) * j2eh
            th[key + '_kcal_mol'] = float(th_j[key]) * j2kcal
        for key in ('S_tot', 'S_trans', 'S_rot', 'S_vib', 'S_elec'):
            th[key + '_J_molK'] = float(th_j[key])
        th['negative_mode_note'] = (
            'NEGATIVE internal mode present: harmonic RRHO thermo '
            'output is PROVISIONAL and flagged not_for_use; minimum '
            'NOT registered'
            if n_neg > 0 else
            'all internal modes >= 0: thermo registered for the '
            'project method model only')
        th['not_for_use'] = bool(n_neg > 0)
        results['thermochemistry'] = th

        # ---------- 5. decision ----------
        results['decision'] = dict(
            n_negative_modes=n_neg,
            registration=(
                'supported as a local minimum WITHIN this method '
                '(wb97xd+D2/def2-TZVP/L8/grid_response, gas) and '
                'check scope (TR-projected Cartesian Hessian, '
                'analytical DFT + explicit D2); NOT a claim about '
                'S13 CCSD(T) or the full reaction path'
                if n_neg == 0 else
                'NOT registered as a local minimum in this scope; '
                'curvature status: non-minimum or pending'),
            thermo_registered=bool(n_neg == 0),
            hno_electronic_energy_Eh=e_elec,
            hno_electronic_energy_note='anchor e_total (DFT+D2), '
                                       'gated to the JOB-068 '
                                       'recheck value at %.2e Eh'
                                       % gate['dE'])
        # E_products / dE_project explicitly NOT computed
        results['p4_energy'] = dict(
            e_products=None, dE_project=None,
            reason='H2O2 was NOT accepted in JOB-068 (budget '
                   'exhausted, lowest gmax 6.42e-3); per the charter '
                   'NO incomplete P4 product energy is computed',
            e_monomer_sum_context_only=-281.99812183548585)
        status = 'completed'
    except Exception as e:                      # noqa: BLE001
        results['abort'] = dict(error=str(e)[:500],
                                traceback=traceback.format_exc()[-2000:])
        status = 'aborted'
        print('[069] ABORT: %s' % str(e)[:200], flush=True)
        raise
    finally:
        results['final_status'] = status
        results['finished'] = time.strftime('%F %T')
        try:
            led_data = json.load(open(LEDGER))
            used = {}
            for a in led_data['attempts']:
                used[a['category']] = used.get(a['category'], 0) + 1
            results['accounting'] = dict(
                caps=man['caps'], attempts_used=used,
                total_attempts=sum(used.values()),
                d2_fd_gradient_calls=results.get(
                    'hessian', {}).get(
                    'd2_fd_gradient_calls', None),
                scf_attempt_note='SCF-attempt ledger only; D2 FD '
                                 'calls are classical and listed '
                                 'separately')
        except Exception:
            pass
        ex.save_json_atomic(RESULTS, results)
        print('[069] DONE: %s' % status, flush=True)


if __name__ == '__main__':
    main()
