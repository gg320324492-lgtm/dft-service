#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-058: full-gradient central-difference Hessian check of
the 057 stationary-point candidate + RKS internal stability.

Budget (no borrowing): centre 1, stability 1 (separate), fd 42
(SCF+gradient total 43).  No analytic Hessian, no extra clean-SCF, no
separate D2 differencing (each full gradient already contains D2 once and
grid_response).  ON ANY EXCEPTION OR NON-CONVERGENCE: save and STOP - no
fix-and-resume without new authorization.

Analysis handover returns a SINGLE DICT.  H_raw is saved FIRST with its
antisymmetric residual, then H_sym.  Missing points -> NO interpolation,
NO mode analysis.  Post-processing uses NOMINAL masses [14,1,1,1,16,16,16]
with the corrected JOB-044 TR projection (rank 6 / internal 15); a
scipy-route eigensolver recomputation cross-checks the post-processing.
"""
import os, sys, json, time, traceback
import numpy as np

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
HERE = ROOT + '/protocols'
sys.path.insert(0, HERE)
import job047_exec as ex
import job052_exec as j52          # endpoint_eval (rec, mf, mol) + Ledger
import job044_fix_postprocess as pp

OUT = ROOT + '/run_artifacts/02_nh3o3_reference/c1_fdhess058'
LEDGER = OUT + '/budget_fdhess058.json'
RESULTS = OUT + '/fdhess058_results.json'
MANIFEST = OUT + '/input_manifest.json'
SYMS = ['N', 'H', 'H', 'H', 'O', 'O', 'O']
BPA = 1.0 / ex.ANG_PER_BOHR
MASSES_NOMINAL = np.array([14.0, 1.0, 1.0, 1.0, 16.0, 16.0, 16.0])
H_STEP = 0.001


class Ledger:
    CAPS = dict(centre=1, stability=1, fd=42)

    def __init__(self, path):
        self.path = path
        if os.path.exists(path):
            self.data = json.load(open(path))
            for a in self.data.get('attempts', []):
                if a.get('status') == 'error':
                    raise RuntimeError('ledger has error attempt -> HARD '
                                       'STOP (no silent resume, no ledger '
                                       're-initialisation): '
                                       + json.dumps(a)[:300])
        else:
            self.data = dict(caps=dict(self.CAPS),
                             note='failures count; no borrowing; no '
                                  're-initialising the ledger to clear '
                                  'failures; on exception or '
                                  'non-convergence: save and STOP',
                             attempts=[])
            self._save()

    def _save(self):
        ex.save_json_atomic(self.path, self.data)

    def count(self, cat):
        return sum(1 for a in self.data['attempts']
                   if a['category'] == cat)

    def pre(self, cat, note=None):
        if self.count(cat) >= self.CAPS[cat]:
            raise RuntimeError('cap reached for %s (no auto extra points)'
                               % cat)
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


def _eval_point(R_bohr, out_dir, tag, eval_fn):
    """Evaluation via the injected eval_fn (production: 052 endpoint_eval
    with a fresh mol).  Returns (record, unit_readback_maxdev_Bohr)."""
    R = np.asarray(R_bohr, float).reshape(7, 3)
    rec, rb = eval_fn(R, out_dir, tag)
    if rb > 0.0:
        raise RuntimeError('unit readback mismatch %e' % rb)
    rec['tag'] = tag
    rec['x_bohr'] = R.tolist()
    rec['unit_readback_maxdev_Bohr'] = rb
    ex.save_json_atomic(os.path.join(out_dir, 'eval_%s.json' % tag), rec)
    return rec, rb


def run_fdhess(man, led, eval_fn, out_dir, stability_fn=None):
    """Actual execution entry.  eval_fn contract: (rec, readback, mf)
    (mf is the converged object, needed for stability; None in mocks).
    Returns a SINGLE DICT {centre_record, gate, stability, fd_records,
    hessian}."""
    R0 = np.asarray(man['centre']['x_bohr'], float).reshape(7, 3)
    att = led.pre('centre', dict(tag='centre'))
    try:
        centre_rec, rb, mf = eval_fn(R0, out_dir, 'centre')
        if rb > 0.0:
            raise RuntimeError('unit readback mismatch %e' % rb)
        if not (centre_rec.get('converged') and centre_rec.get('all_finite')):
            raise RuntimeError('centre SCF not converged or non-finite '
                               '-> HARD STOP')
        led.post(att, dict(e_total=centre_rec['e_total'],
                           grad_max=centre_rec['grad_max'],
                           seconds=centre_rec.get('seconds')))
    except Exception as e:                                  # noqa: BLE001
        led.fail(att, e)
        raise
    # centre gate (gmax<=1e-5 IS a criterion here: converged candidate)
    dE = abs(centre_rec['e_total'] - float(man['centre']['e_total']))
    dg = float(np.abs(np.asarray(centre_rec['grad'], float).reshape(-1)
                      - np.asarray(man['centre']['grad'], float)
                      .reshape(-1)).max())
    dC = float(np.abs(np.asarray(centre_rec['coords_actual_angstrom'],
                                 float)
                      - np.asarray(man['centre']['coords_actual_angstrom'],
                                   float)).max())
    gate = dict(dE=dE, dgrad_max=dg, dcoords_A=dC,
                gmax=float(centre_rec['grad_max']),
                gmax_le_1e5=bool(centre_rec['grad_max'] <= 1e-5),
                converged=bool(centre_rec['converged']),
                finite=bool(centre_rec['all_finite']),
                config_match=bool(centre_rec['config']
                                  == man['centre']['config']))
    gate['gates_pass'] = bool(dE <= 1e-8 and dg <= 1e-7 and dC <= 1e-9
                              and centre_rec['grad_max'] <= 1e-5
                              and centre_rec['converged']
                              and centre_rec['all_finite']
                              and gate['config_match'])
    print('[058] centre gate %s: dE=%.2e dgrad=%.2e dC=%.2e gmax=%.3e'
          % ('PASS' if gate['gates_pass'] else 'FAIL', dE, dg, dC,
             centre_rec['grad_max']), flush=True)
    if not gate['gates_pass']:
        raise RuntimeError('HARD STOP: centre gate failed')

    # ---- RKS internal stability on the converged object (separate cap) --
    stability = None
    if stability_fn is not None:
        att = led.pre('stability', dict(
            call='stability(internal=True, external=False, '
                 'return_status=True)'))
        try:
            stability = stability_fn(mf, out_dir)
            led.post(att, dict(stable_i=stability.get('stable_i')))
        except Exception as e:                          # noqa: BLE001
            led.fail(att, e)
            raise
        if stability.get('stable_i') is not True:
            return dict(centre_record=centre_rec, gate=gate,
                        stability=stability, fd_records=None,
                        hessian=dict(status='stopped_stability_not_true'))

    # ---- 42 central-difference displacements (preregistered order) ------
    recs = {}
    for g in man['displacement_geometries']:
        tag = g['tag_full']
        R = np.asarray(g['coords_bohr'], float).reshape(7, 3)
        att = led.pre('fd', dict(tag=tag, j=g['j'], sign=g['sign']))
        try:
            rec, rb, _ = eval_fn(R, out_dir, tag)
            if rb > 0.0:
                raise RuntimeError('unit readback mismatch %e' % rb)
            if not (rec.get('converged') and rec.get('all_finite')):
                raise RuntimeError('fd %s SCF not converged or non-finite '
                                   '-> HARD STOP (displaced points are used '
                                   'for differencing, non-convergence is a '
                                   'failure)' % tag)
            rec['j'] = g['j']
            rec['sign'] = g['sign']
            led.post(att, dict(e_total=rec['e_total'],
                               grad_max=rec['grad_max'],
                               seconds=rec.get('seconds')))
            recs[tag] = rec
            print('[058] fd %s: E=%.9f gmax=%.3e'
                  % (tag, rec['e_total'], rec['grad_max']), flush=True)
        except Exception as e:                              # noqa: BLE001
            led.fail(att, e)
            raise

    hess = build_hessian(recs, H_STEP, out_dir)
    return dict(centre_record=centre_rec, gate=gate, stability=stability,
                fd_records=recs, hessian=hess)


def build_hessian(recs, h, out_dir):
    """Assemble the FD Hessian columns; save H_raw + antisym residual,
    then H_sym.  Refuse incomplete data."""
    H = np.zeros((21, 21))
    used = 0
    for j in range(21):
        tp = recs.get('fd_p%02d' % j)
        tm = recs.get('fd_m%02d' % j)
        if tp is None or tm is None:
            return dict(status='incomplete_missing_points',
                        missing_column=j,
                        note='NO interpolation performed; NO mode analysis')
        gp = np.asarray(tp['grad'], float).reshape(-1)
        gm = np.asarray(tm['grad'], float).reshape(-1)
        H[:, j] = (gp - gm) / (2 * h)
        used += 1
    antisym = float(np.abs(H - H.T).max())
    H_sym = 0.5 * (H + H.T)
    np.save(os.path.join(out_dir, 'h_fd_raw.npy'), H)
    np.save(os.path.join(out_dir, 'h_fd_sym.npy'), H_sym)
    return dict(status='ok', n_columns=used, h_Bohr=h,
                H_raw=H.tolist(), H_sym=H_sym.tolist(),
                antisymmetric_residual_max=antisym,
                files=['h_fd_raw.npy', 'h_fd_sym.npy'],
                note='H_raw saved first (residual reported); H_sym is the '
                     'symmetrised matrix; the zero residual of H_sym is '
                     'NOT presented as original consistency')


def postprocess(man, hess, centre_rec, out_dir):
    """TR projection + internal modes with NOMINAL masses (JOB-044
    corrected post-processing) + independent recomputation cross-check."""
    if hess.get('status') != 'ok':
        return dict(status=hess['status'])
    H_sym = np.asarray(hess['H_sym'], float)
    coords = np.asarray(man['centre']['x_bohr'], float).reshape(7, 3)
    d = pp.analyse(H_sym, MASSES_NOMINAL, coords, g_cart=man['centre'][
        'grad'], label='058 FD Hessian (nominal masses)')
    # independent post-processing cross-check (scipy eigensolver route)
    from scipy.linalg import eigh as sp_eigh
    mw = pp.mw_mat(H_sym, MASSES_NOMINAL)
    V, U, sv, rank = pp.internal_subspace(MASSES_NOMINAL, coords)
    Hint = U.T @ mw @ U
    Hint = 0.5 * (Hint + Hint.T)
    lam_sp = sp_eigh(Hint, check_finite=True, eigvals_only=True)
    lam_np = np.asarray(d['eigenvalues_Eh_Bohr2_amu'], float)
    d['cross_check'] = dict(
        method='scipy.linalg.eigh on the same Hint vs the numpy route',
        max_abs_diff_Eh_Bohr2=float(np.abs(lam_sp - lam_np).max()),
        note='post-processing validation ONLY (not an independent '
             'electronic-structure verification)')
    d['status'] = 'ok'
    np.save(os.path.join(out_dir, 'h_fd_modes_eigenvalues.npy'), lam_np)
    return d


def main():
    man = json.load(open(MANIFEST))
    if sha_file(man['centre']['path_wsl']) != man['centre']['sha256']:
        raise RuntimeError('HARD STOP: 057 recheck source hash changed')
    if not man['all_checks_pass']:
        raise RuntimeError('HARD STOP: prechecks failed')
    print('[058] source OK: %s (sha256 %.16s...)'
          % (os.path.basename(man['centre']['path_wsl']),
             man['centre']['sha256']), flush=True)

    def production_eval(R, out_dir, tag):
        from pyscf import gto
        Rm = np.asarray(R, float).reshape(7, 3)
        mol = gto.M(atom=[(s, (float(r[0]), float(r[1]), float(r[2])))
                          for s, r in zip(SYMS, Rm)],
                    basis='def2-TZVP', charge=0, spin=0, verbose=0,
                    max_memory=4000, unit='Bohr')
        rec, mf, _ = j52.endpoint_eval(mol, out_dir, tag)
        rb = float(np.abs(np.asarray(mol.atom_coords(unit='Bohr'), float)
                          .reshape(-1) - Rm.reshape(-1)).max())
        return rec, rb, mf

    def stability_fn(mf, out_dir):
        t0 = time.time()
        log_path = os.path.join(out_dir, 'stability_raw_log.txt')
        old_stdout, old_verbose = mf.stdout, mf.verbose
        buf = None
        try:
            import io
            buf = io.StringIO()
            mf.stdout = buf
            mf.verbose = 4
            mo_stab, stable_i, stable_e = mf.stability(
                internal=True, external=False, return_status=True)
        finally:
            mf.stdout, mf.verbose = old_stdout, old_verbose
        log = buf.getvalue() if buf is not None else ''
        with open(log_path, 'w') as fh:
            fh.write(log)
        return dict(stable_i=bool(stable_i) if stable_i is not None
                    else None,
                    stable_e=None,
                    raw_log='stability_raw_log.txt',
                    seconds=round(time.time() - t0, 1),
                    note='stable_e=None only means external stability was '
                         'NOT checked')

    results = dict(
        job='JOB-2026-0906-058 full-gradient central-difference Hessian '
            'check of the 057 stationary-point candidate',
        doing='centre reproduction + RKS internal stability + 42 '
              'central-difference displacements (h=0.001 Bohr) building '
              'the full 21x21 Cartesian Hessian; nominal masses; single '
              'step size (sensitivity NOT verified)',
        centre_ref=dict(e_total=man['centre']['e_total'],
                        grad_max=man['centre']['grad_max']))
    led = Ledger(LEDGER)
    status = 'aborted'
    try:
        flow = run_fdhess(man, led, production_eval, OUT,
                          stability_fn=stability_fn)
        results['centre'] = dict(record=flow['centre_record'],
                                 gate=flow['gate'])
        results['stability'] = flow['stability']
        if flow['stability'] is not None \
                and flow['stability'].get('stable_i') is not True:
            status = 'stopped_stability_not_true'
            results['final'] = dict(status=status, budget=led.data,
                                    note='internal stability not clearly '
                                         'True -> stopped (no orbit '
                                         'following, no UKS switch)')
            ex.save_json_atomic(RESULTS, results)
            print('[058] DONE: %s' % status, flush=True)
            return
        results['fd_records'] = flow['fd_records']
        hess = flow['hessian']
        results['hessian'] = {k: v for k, v in hess.items()
                              if k not in ('H_raw', 'H_sym')}
        ex.save_json_atomic(RESULTS, results)
        if hess.get('status') != 'ok':
            status = 'hessian_incomplete'
            results['final'] = dict(status=status, budget=led.data)
            ex.save_json_atomic(RESULTS, results)
            print('[058] DONE: %s' % status, flush=True)
            return
        modes = postprocess(man, hess, flow['centre_record'], OUT)
        results['modes'] = modes
        if modes.get('status') == 'ok':
            print('[058] TR rank %d / internal %d / negative modes %d'
                  % (modes['tr_rank_svd'], modes['n_internal_modes'],
                     modes['n_negative']), flush=True)
            print('[058] frequencies (cm-1): %s'
                  % ['%.2f' % x for x in modes['frequencies_cm1']],
                  flush=True)
        status = 'completed'
        results['final'] = dict(status=status, budget=led.data,
                                registration=(
                                    'this single-step FD matrix predicts '
                                    'negative internal curvature'
                                    if modes.get('n_negative', 0) > 0 else
                                    'this single-step FD matrix shows no '
                                    'internal negative curvature'),
                                limits=man['analysis_limits'])
        ex.save_json_atomic(RESULTS, results)
        print('[058] DONE: %s' % results['final']['registration'],
              flush=True)
    except Exception as exc:                                # noqa: BLE001
        results['abort'] = dict(error=str(exc),
                                traceback=traceback.format_exc()[-1500:])
        results.setdefault('final', dict(status='aborted',
                                         budget=led.data))
        ex.save_json_atomic(RESULTS, results)
        print('[058] ABORT: %s' % exc, flush=True)
        raise


def sha_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for blk in iter(lambda: fh.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


if __name__ == '__main__':
    main()
