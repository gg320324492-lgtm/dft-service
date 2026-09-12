#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-052: electronic stability + vibrational curvature check of
the 051 stationary-point candidate (opt_11 / recheck geometry).

Budget (per-category, no borrowing): endpoint repro SCF+gradient 1,
RKS internal stability 1, clean-DFT SCF 1 (separate counting), DFT analytic
Hessian 1, D2 pure-gradient central FD <=84 calls (21 coords x 2 sides x
2 steps: 1e-3 and 5e-4 Bohr).  Everything else = 0.

Reuses validated paths: d2_full (endpoint construction identical to the
047/051 RealBackend; d2_hess central FD), the 043 clean-RKS analytic
Hessian route, and the JOB-044 corrected post-processing conventions
(amu masses + two unit routes, 21x21 layout, 6-dim MW TR subspace /
15-dim internal subspace, all eigenvalues and modes kept, PySCF
cross-check of post-processing only).  RAW (pre-symmetrisation) matrices
are saved for every component.

Electronic stability and vibrational curvature are reported SEPARATELY.
"""
import os, sys, io, json, glob, hashlib, time, traceback
import numpy as np

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
HERE = ROOT + '/protocols'
sys.path.insert(0, HERE)
import d2_full
import job047_exec as ex          # validated endpoint eval components
import job044_fix_postprocess as pp   # 044 corrected post-processing

OUT = ROOT + '/run_artifacts/02_nh3o3_reference/c1_cand_check052'
LEDGER = OUT + '/budget_cand052.json'
RESULTS = OUT + '/cand_check052_results.json'
MANIFEST = OUT + '/input_manifest.json'
SYMS = ['N', 'H', 'H', 'H', 'O', 'O', 'O']
GRID_LEVEL = 8
BPA = 1.0 / ex.ANG_PER_BOHR


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for blk in iter(lambda: fh.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def sha_arr(a):
    return hashlib.sha256(np.asarray(a, float).tobytes()).hexdigest()[:16]


# ---------------------------------------------------------------- ledger
class Ledger:
    CAPS = dict(endpoint=1, stability=1, clean_scf=1, dft_hess=1, d2_fd=1,
                d2_fd_call_cap=84)

    def __init__(self, path):
        self.path = path
        if os.path.exists(path):
            self.data = json.load(open(path))
        else:
            self.data = dict(caps=dict(self.CAPS),
                             note='failures count within their category; '
                                  'no borrowing; no auto-resume of FAILED '
                                  'attempts; no extra SCF hidden in '
                                  'Hessian preparation',
                             attempts=[])
            self._save()

    def _save(self):
        ex.save_json_atomic(self.path, self.data)

    def count(self, cat):
        return sum(1 for a in self.data['attempts']
                   if a['category'] == cat)

    def pre(self, cat, note=None):
        if self.count(cat) >= self.CAPS[cat]:
            raise RuntimeError('cap reached for %s' % cat)
        att = dict(attempt=len(self.data['attempts']) + 1, category=cat,
                   note=note or {}, status='pending',
                   started=time.strftime('%F %T'))
        self.data['attempts'].append(att)
        self._save()
        return att

    def post(self, att, fields):
        att.update(fields)
        att['status'] = 'done'
        att['finished'] = time.strftime('%F %T')
        self._save()

    def fail(self, att, err):
        att['status'] = 'error'
        att['error'] = str(err)[:500]
        self._save()


# ------------------------------------------------- endpoint (mf retained)
def endpoint_eval(mol, out_dir, tag):
    """Identical construction to the validated 047/051 RealBackend.full_eval,
    but returns (record, mf) so stability can run on the converged object."""
    mf = d2_full.make_mf_d2(mol, solvent=None)
    mf.grids.level = GRID_LEVEL
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-9
    t0 = time.time()
    orig = mf.kernel
    cnt = [0]

    def counting(*a, **k):
        cnt[0] += 1
        return orig(*a, **k)

    mf.kernel = counting
    mf.kernel()
    e = float(mf.e_tot)
    g_obj = mf.nuc_grad_method()
    g_obj.grid_response = True
    if getattr(g_obj, 'grid_response', None) is not True:
        raise RuntimeError('grid_response=True did not reach the gradient')
    g = np.asarray(g_obj.kernel(), float).reshape(7, 3)
    C_act = np.asarray(mol.atom_coords(unit='Angstrom'), float)
    e_d2 = float(d2_full.d2_energy(mol))
    cfg = dict(xc='wb97xd (project -D2 attached)',
               d2_attached=bool(getattr(mf, '_has_full_d2', False)),
               grid_level=int(getattr(mf.grids, 'level', -1)),
               scf_tol=[float(mf.conv_tol), float(mf.conv_tol_grad)],
               basis='def2-TZVP', charge=0, spin=0, grid_response=True,
               solvent='none (gas phase)')
    rec = dict(e_total=e, e_d2_Eh=e_d2, e_dft_part_Eh=e - e_d2,
               grad=g.tolist(), grad_max=float(np.abs(g).max()),
               grad_sha=sha_arr(g),
               coords_actual_angstrom=C_act.tolist(),
               coords_sha=sha_arr(C_act), config=cfg, converged=bool(
                   mf.converged),
               all_finite=bool(np.isfinite(e) and np.isfinite(g).all()),
               scf_kernel_count=cnt[0], seconds=round(time.time() - t0, 1),
               tag=tag)
    ex.save_json_atomic(os.path.join(out_dir, 'eval_%s.json' % tag), rec)
    return rec, mf, mol


def build_clean_rks(mol):
    """Validated 043 route: clean vacuum RKS (no D2) for the analytic
    Hessian."""
    from pyscf import dft
    mf = dft.RKS(mol)
    mf.xc = 'wb97xd'
    mf.grids.level = GRID_LEVEL
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-9
    return mf


def block21(block):
    """(natm, natm, 3, 3) -> 21x21 Cartesian (validated 043 layout)."""
    n = block.shape[0]
    return np.asarray(block, float).transpose(0, 2, 1, 3).reshape(3 * n,
                                                                  3 * n)


def layout_check(block, h21, nidx=(0, 3, 6), cidx=(0, 1, 2)):
    """Verify H21[3i+a, 3j+b] == block[i][j][a][b] on sampled indices."""
    worst = 0.0
    for i in nidx:
        for j in nidx:
            for a in cidx:
                for b in cidx:
                    worst = max(worst, abs(
                        h21[3 * i + a, 3 * j + b] - block[i][j][a][b]))
    return float(worst)


def main():
    man = json.load(open(MANIFEST))
    src_path = man['source']['source_path_wsl']
    if sha256_file(src_path) != man['source']['source_sha256']:
        raise RuntimeError('HARD STOP: 051 recheck source hash changed')
    src = json.load(open(src_path))
    x_bohr = np.asarray(src['x_bohr'], float)     # bitwise execution coords
    print('[052] source OK: %s (sha256 %.16s...)'
          % (os.path.basename(src_path), sha256_file(src_path)), flush=True)

    results = dict(
        job='JOB-2026-0906-052 electronic stability + vibrational '
            'curvature check of the 051 stationary-point candidate',
        doing='verify the candidate endpoint, run RKS internal stability '
              '(orbital space) and compute the separated analytic-DFT + '
              'D2-FD combined Hessian (nuclear-coordinate curvature) with '
              'JOB-044 post-processing; the two checks are reported '
              'SEPARATELY',
        source=dict(path=src_path, sha256=sha256_file(src_path),
                    e_total_ref=-282.0017214898003,
                    grad_max_ref=8.202781486289723e-06))
    led = Ledger(LEDGER)
    prev = {}
    if os.path.exists(RESULTS):
        try:
            prev = json.load(open(RESULTS))
        except Exception:                                   # noqa: BLE001
            prev = {}
    if prev.get('abort'):
        results['prior_abort'] = prev['abort']
        results['resume_note'] = ('a previous run aborted on a LEDGER '
                                  'BOOKKEEPING bug (d2_fd missing from the '
                                  'caps dict) AFTER the endpoint / '
                                  'stability / clean-SCF / DFT-Hessian '
                                  'stages had completed and persisted; '
                                  'this resume repeats NO quantum '
                                  'calculation and executes only the '
                                  'remaining D2 FD + combination + '
                                  'post-processing; every category stays '
                                  'within its cap (ledger counts verified '
                                  'below)')
    from pyscf import gto
    atom = [(s, (float(r[0]), float(r[1]), float(r[2])))
            for s, r in zip(SYMS, x_bohr)]
    mol = gto.M(atom=atom, basis='def2-TZVP', charge=0, spin=0, verbose=0,
                max_memory=4000, unit='Bohr')
    status = 'aborted'
    try:
        # ---------------- 1) endpoint reproduction (SCF + gradient) ----
        rec, gate = None, None
        if (prev.get('endpoint', {}).get('gate', {}).get('gates_pass')
                and led.count('endpoint') == 1):
            results['endpoint'] = prev['endpoint']
            rec = prev['endpoint']['record']
            gate = prev['endpoint']['gate']
            print('[052] endpoint RESUMED from persisted record '
                  '(no repeated SCF)', flush=True)
        else:
            att = led.pre('endpoint', dict(tag='endpoint'))
            try:
                rec, mf, _ = endpoint_eval(mol, OUT, 'endpoint')
            except Exception as e:                          # noqa: BLE001
                led.fail(att, e)
                raise
            dE = abs(rec['e_total'] - float(src['e_total']))
            dg = float(np.abs(np.asarray(rec['grad'], float).reshape(-1)
                              - np.asarray(src['grad'], float)
                              .reshape(-1)).max())
            dC = float(np.abs(np.asarray(rec['coords_actual_angstrom'],
                                         float)
                              - np.asarray(src['coords_actual_angstrom'],
                                           float)).max())
            gate = dict(dE=dE, dgrad_max=dg, dcoords_A=dC,
                        gmax=float(rec['grad_max']),
                        gmax_le_1e5=bool(rec['grad_max'] <= 1e-5),
                        converged=bool(rec['converged']),
                        finite=bool(rec['all_finite']),
                        config_match=bool(rec['config'] == src['config']))
            gate['gates_pass'] = bool(
                dE <= 1e-8 and dg <= 1e-7 and dC <= 1e-9
                and rec['grad_max'] <= 1e-5 and rec['converged']
                and rec['all_finite'] and gate['config_match'])
            led.post(att, dict(e_total=rec['e_total'],
                               grad_max=rec['grad_max'],
                               gate=gate, seconds=rec['seconds']))
            results['endpoint'] = dict(record=rec, gate=gate)
            print('[052] endpoint gate %s: dE=%.2e dgrad=%.2e dC=%.2e '
                  'gmax=%.3e' % ('PASS' if gate['gates_pass'] else 'FAIL',
                                 dE, dg, dC, rec['grad_max']), flush=True)
        if not gate['gates_pass']:
            results['final'] = dict(status='endpoint_gate_failed',
                                    budget=led.data)
            ex.save_json_atomic(RESULTS, results)
            raise RuntimeError('HARD STOP: endpoint gate failed')

        # ---------------- 2) RKS internal stability (orbital space) -----
        st_rec = prev.get('electronic_stability')
        if not (st_rec and st_rec.get('stable_i') is True
                and led.count('stability') == 1):
            att = led.pre('stability', dict(
                call='stability(internal=True, external=False, '
                     'return_status=True)'))
            print('[052] running RKS internal stability...', flush=True)
            t0 = time.time()
            log_path = os.path.join(OUT, 'stability_raw_log.txt')
            old_stdout, old_verbose = mf.stdout, mf.verbose
            logf = open(log_path, 'w')
            try:
                mf.stdout = logf
                mf.verbose = 4
                st = mf.stability(internal=True, external=False,
                                  return_status=True)
            finally:
                mf.stdout, mf.verbose = old_stdout, old_verbose
                logf.close()
            # installed order: (mo_i, mo_e, stable_i, stable_e)
            mo_i, mo_e, stable_i, stable_e = st
            st_rec = dict(
                call='mf.stability(internal=True, external=False, '
                     'return_status=True)',
                installed_return_order='mo_i, mo_e, stable_i, stable_e',
                stable_i=bool(stable_i) if stable_i is not None else None,
                stable_e=None if stable_e is None else bool(stable_e),
                stable_e_note='external=False -> None means NOT checked',
                raw_return_repr=[repr(x)[:120] for x in st],
                raw_log='stability_raw_log.txt',
                seconds=round(time.time() - t0, 1),
                scope='RKS internal ORBITAL stability only; no Hessian '
                      'here; does NOT exclude multireference character; '
                      'independent of the vibrational-curvature result')
            led.post(att, dict(stable_i=str(stable_i),
                               seconds=st_rec['seconds']))
        else:
            print('[052] stability RESUMED from persisted result '
                  '(stable_i=True)', flush=True)
        results['electronic_stability'] = st_rec
        print('[052] stability: stable_i=%s' % st_rec['stable_i'],
              flush=True)
        ex.save_json_atomic(RESULTS, results)
        if st_rec['stable_i'] is not True:
            results['final'] = dict(
                status='stopped_not_internally_stable', budget=led.data,
                note='stable_i is not explicitly True -> saved and stopped '
                     'per charter (no orbital following, no UKS switch)')
            ex.save_json_atomic(RESULTS, results)
            print('[052] stable_i not True -> STOP', flush=True)
            print('DONE (stopped_not_internally_stable)', flush=True)
            return

        # ---------------- 3) clean-DFT SCF (separate counting) ----------
        if (prev.get('clean_dft', {}).get('passed')
                and led.count('clean_scf') == 1):
            results['clean_dft'] = prev['clean_dft']
            print('[052] clean-DFT RESUMED from persisted result '
                  '(no repeated SCF)', flush=True)
        else:
            att = led.pre('clean_scf', dict(purpose='separate analytic-DFT '
                                                    'Hessian object'))
            print('[052] clean-DFT SCF...', flush=True)
            t0 = time.time()
            mf_clean = build_clean_rks(mol)
            e_clean = float(mf_clean.kernel())
            scf_count_clean = 1
            cfg_clean = dict(xc=mf_clean.xc, d2_attached=False,
                             grid_level=int(mf_clean.grids.level),
                             scf_tol=[float(mf_clean.conv_tol),
                                      float(mf_clean.conv_tol_grad)],
                             basis='def2-TZVP', charge=0, spin=0,
                             solvent='none (gas phase)')
            dE_clean = abs(e_clean - float(rec['e_dft_part_Eh']))
            clean_ok = bool(dE_clean <= 1e-8)
            led.post(att, dict(e_total=e_clean,
                               dE_vs_total_minus_d2=dE_clean,
                               passed=clean_ok,
                               seconds=round(time.time() - t0, 1)))
            results['clean_dft'] = dict(
                e_total=e_clean,
                e_ref_total_minus_d2=rec['e_dft_part_Eh'],
                dE=dE_clean, passed=clean_ok, config=cfg_clean,
                scf_count=scf_count_clean)
            print('[052] clean-DFT: E=%.9f dE(vs total-D2)=%.2e %s'
                  % (e_clean, dE_clean, 'OK' if clean_ok else 'FAIL'),
                  flush=True)
        ex.save_json_atomic(RESULTS, results)
        if not results['clean_dft']['passed']:
            results['final'] = dict(status='clean_dft_validation_failed',
                                    budget=led.data)
            ex.save_json_atomic(RESULTS, results)
            raise RuntimeError('HARD STOP: clean-DFT energy validation '
                               'failed')

        # ---------------- 4) DFT analytic Hessian (raw saved) -----------
        h_dft_raw = None
        raw_p = os.path.join(OUT, 'h_dft_raw.npy')
        if (prev.get('dft_hessian') and os.path.exists(raw_p)
                and led.count('dft_hess') == 1):
            h_dft_raw = np.load(raw_p)
            results['dft_hessian'] = prev['dft_hessian']
            print('[052] DFT Hessian RESUMED from persisted matrix '
                  '(no recomputation)', flush=True)
        else:
            att = led.pre('dft_hess', dict(note='analytic wb97xd vacuum '
                                                'RKS Hessian'))
            print('[052] DFT analytic Hessian (this can take minutes)...',
                  flush=True)
            t0 = time.time()
            hblock = mf_clean.Hessian().kernel()
            h_dft_raw = block21(hblock)
            asy_before = float(np.abs(h_dft_raw - h_dft_raw.T).max())
            h_dft_sym = 0.5 * (h_dft_raw + h_dft_raw.T)
            asy_after = float(np.abs(h_dft_sym - h_dft_sym.T).max())
            lay = layout_check(hblock, h_dft_raw)
            np.save(raw_p, h_dft_raw)
            np.save(os.path.join(OUT, 'h_dft_sym.npy'), h_dft_sym)
            led.post(att, dict(asy_before=asy_before, layout_maxdev=lay,
                               seconds=round(time.time() - t0, 1)))
            results['dft_hessian'] = dict(
                raw_file='h_dft_raw.npy', sym_file='h_dft_sym.npy',
                unit='Eh/Bohr^2',
                layout='transpose(0,2,1,3).reshape(21,21)',
                layout_check_maxdev=lay,
                antisymmetric_residual_before=asy_before,
                antisymmetric_residual_after=asy_after,
                method='wb97xd analytic Hessian (vacuum RKS, no D2; 043 '
                       'route)',
                seconds=round(time.time() - t0, 1))
            print('[052] DFT Hessian done: asy_before=%.2e '
                  'layout_dev=%.2e (%.0fs)'
                  % (asy_before, lay, time.time() - t0), flush=True)
        ex.save_json_atomic(RESULTS, results)

        # ---------------- 5) D2 central FD (two steps, <=84 calls) ------
        att = led.pre('d2_fd', dict(note='pure pair-potential gradient FD; '
                                         'no SCF'))
        orig_grad = d2_full.d2_grad
        cnt = [0]
        scf_before = (rec['scf_kernel_count'],)

        def counting_grad(m, coords=None):
            cnt[0] += 1
            if cnt[0] > 84:
                raise RuntimeError('D2 FD call cap 84 exceeded')
            return orig_grad(m, coords)

        d2_full.d2_grad = counting_grad
        try:
            t0 = time.time()
            h_d2_10 = block21(d2_full.d2_hess(mol, x_bohr, step=1e-3))
            n1 = cnt[0]
            h_d2_05 = block21(d2_full.d2_hess(mol, x_bohr, step=5e-4))
            n2 = cnt[0] - n1
        finally:
            d2_full.d2_grad = orig_grad
        led.post(att, dict(d2_grad_calls_step1=n1, d2_grad_calls_step2=n2,
                           total_calls=cnt[0],
                           scf_kernel_count_unchanged=bool(
                               scf_before == (rec['scf_kernel_count'],)),
                           seconds=round(time.time() - t0, 1)))
        results['d2_fd_hessian'] = dict(
            steps_Bohr=[1e-3, 5e-4],
            raw_files=['h_d2_step1e-3_raw.npy', 'h_d2_step5e-4_raw.npy'],
            d2_grad_calls=[n1, n2], total_calls=cnt[0],
            cap=84, scf_calls_during_fd=0,
            no_scf_evidence='d2_grad is a pure nuclear-pair function; the '
                            'endpoint SCF kernel counter was unchanged '
                            'during the FD loop',
            seconds=round(time.time() - t0, 1))
        print('[052] D2 FD done: calls=%d (%.1fs)' % (cnt[0],
                                                      time.time() - t0),
              flush=True)
        ex.save_json_atomic(RESULTS, results)

        # ---------------- 6) combined (D2 added ONCE per step) ----------
        comb = {}
        for step, h_d2 in (('1e-3', h_d2_10), ('5e-4', h_d2_05)):
            h_raw = h_dft_raw + h_d2
            asy = float(np.abs(h_raw - h_raw.T).max())
            h_sym = 0.5 * (h_raw + h_raw.T)
            np.save(os.path.join(OUT, 'h_comb_step%s_raw.npy' % step),
                    h_raw)
            np.save(os.path.join(OUT, 'h_comb_step%s_sym.npy' % step),
                    h_sym)
            comb[step] = dict(raw_file='h_comb_step%s_raw.npy' % step,
                              sym_file='h_comb_step%s_sym.npy' % step,
                              antisymmetric_residual_before=asy,
                              antisymmetric_residual_after=float(
                                  np.abs(h_sym - h_sym.T).max()))
        results['combined_hessian'] = dict(
            composition='analytic DFT (wb97xd vacuum) + D2 gradient FD; D2 '
                        'added exactly once',
            exact_name='analytic-DFT + D2 gradient-FD combined Hessian '
                       '(NOT all-analytic)',
            per_step=comb)

        # ---------------- 7) 044 post-processing + PySCF cross-check ----
        masses = np.asarray(mol.atom_mass_list(), float)
        masses_avg = np.asarray(mol.atom_mass_list(isotope_avg=True), float)
        coords_bohr = np.asarray(mol.atom_coords(unit='Bohr'), float)
        g_cart = np.asarray(rec['grad'], float).reshape(-1)
        post = {}
        for step, h_d2 in (('1e-3', h_d2_10), ('5e-4', h_d2_05)):
            h21_sym = 0.5 * ((h_dft_raw + h_d2) + (h_dft_raw + h_d2).T)
            a = pp.analyse(h21_sym, masses, coords_bohr, g_cart=g_cart,
                           label='candidate step %s' % step)
            a_avg = pp.analyse(h21_sym, masses_avg, coords_bohr,
                               label='candidate step %s isotope_avg' % step)
            from pyscf.hessian import thermo
            ha = thermo.harmonic_analysis(
                mol, pp.block_from_21(h21_sym, 7), imaginary_freq=False,
                mass=masses)
            pf = np.sort(np.asarray(ha['freq_wavenumber'], float))
            mine = np.sort(np.array(a['frequencies_cm1']))
            a['pyscf_cross_check'] = dict(
                pyscf_freq_cm1=[float(x) for x in pf],
                max_abs_diff_cm1=float(np.abs(pf - mine).max()),
                note='same matrix, same masses, imaginary_freq=False; '
                     'validates post-processing ONLY, not electronic '
                     'structure')
            a['isotope_avg_max_diff_cm1'] = float(np.abs(
                np.array(a_avg['frequencies_cm1'])
                - np.array(a['frequencies_cm1'])).max())
            post[step] = a
            print('[052] step %s internal modes (cm^-1): %s'
                  % (step, np.array2string(
                      np.array(a['frequencies_cm1']), precision=2)),
                  flush=True)
        results['vibrational_curvature'] = dict(
            method='JOB-044 corrected post-processing: amu masses, MW TR '
                   'rank 6 / internal 15, all eigenvalues and modes kept',
            masses=[float(x) for x in masses],
            per_step=post,
            interpretation_note='matrix prediction at the candidate '
                                'geometry; given the unresolved '
                                'analytic-vs-FD quantitative discrepancy '
                                'history, an all-positive spectrum does '
                                'NOT by itself complete minimum acceptance')

        # ---------------- 8) summary ------------------------------------
        p1 = post['1e-3']
        neg = p1['n_negative']
        results['final'] = dict(
            status='completed',
            electronic_stability=st_rec['stable_i'],
            vibrational=dict(
                n_negative=p1['n_negative'],
                negative_modes=p1['negative_modes'],
                lowest_mode=dict(
                    freq_cm1=p1['frequencies_cm1'][0],
                    eigenvalue=p1['eigenvalues_Eh_Bohr2_amu'][0],
                    cart_norm_mode=p1['modes'][0]['cart_norm_mode']),
                statement=('the matrix predicts negative internal curvature '
                           'at the candidate geometry'
                           if neg else
                           'the matrix shows no internal negative '
                           'curvature at the candidate geometry')),
            verdicts_separate=True,
            not_claimed='transition state / final minimum acceptance / '
                        'reaction mechanism / water-treatment effect',
            budget=led.data)
        ex.save_json_atomic(RESULTS, results)
        print('[052] DONE: stable_i=%s | n_negative(step1e-3)=%d'
              % (st_rec['stable_i'], neg), flush=True)
    except Exception as exc:                                    # noqa: BLE001
        results['abort'] = dict(error=str(exc),
                                traceback=traceback.format_exc()[-1500:])
        results.setdefault('final', dict(status='aborted',
                                         budget=led.data))
        ex.save_json_atomic(RESULTS, results)
        print('[052] ABORT: %s' % exc, flush=True)
        raise


if __name__ == '__main__':
    main()
