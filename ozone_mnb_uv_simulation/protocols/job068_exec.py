#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-068: P4 product endpoints (HNO, H2O2) optimized under
the project method model.

Budget (hard, no borrowing): HNO opt 15 / H2O2 opt 15 (ALL new SCF+
gradient evaluations INCLUDING line-search trials), recheck 1 each,
internal stability 1 each -> 34 total.  Threshold to trigger the
independent recheck: unprojected Cartesian max|g| <= 1e-5 Eh/Bohr.
Stability runs ONLY after a passing recheck (internal orbital
stability ONLY).  Each record is saved atomically BEFORE the attempt
is marked done; checkpoints are per-tag.  ANY real exception -> save
and STOP the whole batch.  If only one product passes, only that
molecule is reported - NO incomplete P4 energy splicing.
"""
import os, sys, json, time, traceback, warnings
import numpy as np
from scipy.optimize import minimize as _scipy_minimize

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
HERE = ROOT + '/protocols'
sys.path.insert(0, HERE)
import job047_exec as ex

OUT = ROOT + '/run_artifacts/02_nh3o3_reference/p4_products068'
MANIFEST = OUT + '/input_manifest.json'
RESULTS = OUT + '/p4_products068_results.json'
LEDGER = OUT + '/budget_p4products068.json'
BPA = 1.0 / ex.ANG_PER_BOHR


# =========================== generalized backend =========================
class P4Backend:
    """Project-method backend for arbitrary small molecules: wb97xd +
    explicit D2 once, def2-TZVP, L8, grid_response, SCF 1e-12/1e-9,
    gas, per-tag chkfile, kernel counting."""

    def __init__(self, elements):
        from pyscf import gto
        import d2_full
        self._gto = gto
        self._d2 = d2_full
        self.elements = list(elements)

    def make_mol(self, R_bohr):
        atom = [(e, (float(r[0]), float(r[1]), float(r[2])))
                for e, r in zip(self.elements,
                                np.asarray(R_bohr, float)
                                .reshape(-1, 3))]
        return self._gto.M(atom=atom, basis='def2-TZVP', charge=0,
                           spin=0, verbose=0, max_memory=4000,
                           unit='Bohr')

    def new_mf(self, R_bohr, out_dir, tag):
        mol = self.make_mol(R_bohr)
        mf = self._d2.make_mf_d2(mol, solvent=None)
        mf.grids.level = 8
        mf.conv_tol = 1e-12
        mf.conv_tol_grad = 1e-9
        mf.chkfile = os.path.join(out_dir, 'chk_%s.chk' % tag)
        return mol, mf

    def full_eval(self, R_bohr, out_dir, tag):
        mol, mf = self.new_mf(R_bohr, out_dir, tag)
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
        chk = os.path.join(out_dir, 'chk_%s.chk' % tag)
        return dict(
            e_total=e, e_d2_Eh=e_d2, e_dft_part_Eh=e - e_d2,
            grad=g.tolist(), grad_max=float(np.abs(g).max()),
            grad_sha=hashlib_arr(g),
            coords_actual_angstrom=(np.asarray(
                R_bohr, float).reshape(-1, 3)
                * ex.ANG_PER_BOHR).tolist(),
            coords_sha=hashlib_arr(np.asarray(R_bohr, float)),
            config=cfg, converged=conv,
            all_finite=bool(np.isfinite(e) and np.isfinite(g).all()),
            scf_kernel_count=int(getattr(mf,
                                         '_exec_scf_kernel_count', -1)))


def hashlib_arr(a):
    import hashlib
    return hashlib.sha256(np.asarray(a, float).tobytes()).hexdigest()[:16]


# ======================= generalized BFGS runner =========================
def run_opt_general(x0, natoms, out_dir, ledger, backend, category,
                    opt_cap, threshold=1e-5, maxiter=200):
    """Full-DOF Cartesian standard BFGS (scipy, jac=True, gtol=1e-6
    inf-norm, hess_inv0=I), threshold-stop via an evaluation flag.
    The FIRST evaluation (at x0) is a real, counted attempt."""
    x0 = np.asarray(x0, float).reshape(-1)
    state = dict(n_new=0, records=[], requests=[], blocked=[],
                 accepted=[])

    def fun(x):
        x = np.asarray(x, float).reshape(-1)
        state['requests'].append(x.tolist())
        if state['n_new'] >= opt_cap:
            state['blocked'].append(x.tolist())
            raise ex.BudgetExhausted('%s attempt cap %d reached'
                                     % (category, opt_cap))
        tag = '%s_%02d' % (category.split('_')[0], state['n_new'] + 1)
        R = x.reshape(natoms, 3)
        att = ledger.pre_eval(category, dict(tag=tag))
        try:
            rec = backend.full_eval(R, out_dir, tag)
            rec['tag'] = tag
            rec['category'] = category
            ex.save_json_atomic(os.path.join(
                out_dir, 'eval_%s_%s.json' % (category, tag)), rec)
            ledger.post_eval(att, dict(e_total=rec['e_total'],
                                       grad_max=rec['grad_max']))
        except Exception as e:                      # noqa: BLE001
            ledger.fail(att, e)
            raise
        state['n_new'] += 1
        rec['_x_bohr'] = x.tolist()
        state['records'].append(rec)
        if float(rec['grad_max']) <= threshold:
            raise ex.ThresholdReached(tag, float(rec['grad_max']))
        return float(rec['e_total']), np.asarray(rec['grad'],
                                                 float).reshape(-1)

    def cb(xk):
        xk = getattr(xk, 'x', xk)
        state['accepted'].append(np.asarray(xk, float).tolist())

    opts = dict(gtol=1e-6, norm=np.inf, xrtol=0.0, c1=1e-4, c2=0.9,
                maxiter=maxiter, hess_inv0=np.eye(3 * natoms))
    wlist = []
    outcome, res_info = 'optimizer_returned', {}
    try:
        with warnings.catch_warnings(record=True) as _w:
            warnings.simplefilter('always')
            wlist = _w
            res = _scipy_minimize(fun, x0, method='BFGS', jac=True,
                                  callback=cb, options=opts)
        outcome = 'optimizer_returned'
        res_info = dict(success=bool(res.success),
                        message=str(res.message),
                        nit=int(getattr(res, 'nit', -1)),
                        nfev=int(getattr(res, 'nfev', -1)),
                        final_jac_gmax=float(np.abs(np.asarray(
                            res.jac, float)).max()))
    except ex.ThresholdReached as e:
        outcome = 'threshold_reached'
        res_info = dict(tag=e.tag, gmax=e.gmax,
                        message='controlled stop: saved point met the '
                                'batch threshold')
    except ex.BudgetExhausted as e:
        outcome = 'budget_exhausted'
        res_info = dict(message=str(e))
    meeting = [dict(tag=r['tag'], gmax=float(r['grad_max']),
                    e_total=float(r['e_total']))
               for r in state['records']
               if float(r['grad_max']) <= threshold]
    return dict(outcome=outcome, optimizer=res_info,
                n_new_evals=state['n_new'], records=state['records'],
                requests=state['requests'],
                blocked_requests=state['blocked'],
                accepted_iterates=state['accepted'],
                meeting_points=meeting,
                warnings=[str(w.message) for w in wlist])


# =========================== stability wrapper ===========================
def stability_wrapper(mf, out_dir, molname):
    """Fixed internal-stability wrapper (052/058 conventions): the
    verbose log goes DIRECTLY to a file handle; the installed 4-tuple
    (mo_i, mo_e, stable_i, stable_e) is parsed after the finally."""
    import io as _io
    log_path = os.path.join(out_dir,
                            'stability_%s_raw_log.txt' % molname)
    old_stdout, old_verbose = mf.stdout, mf.verbose
    logf = open(log_path, 'w')
    st = None
    try:
        mf.stdout = logf
        mf.verbose = 4
        st = mf.stability(internal=True, external=False,
                          return_status=True)
    finally:
        mf.stdout, mf.verbose = old_stdout, old_verbose
        logf.close()
    if not isinstance(st, tuple) or len(st) != 4:
        raise RuntimeError('stability return contract violation: '
                           'expected (mo_i, mo_e, stable_i, stable_e)')
    mo_i, mo_e, stable_i, stable_e = st
    return dict(stable_i=bool(stable_i) if stable_i is not None
                else None,
                stable_e=None if stable_e is None else bool(stable_e),
                raw_log=os.path.basename(log_path),
                installed_order='mo_i, mo_e, stable_i, stable_e')


def product_reference(mol_results, e_nh3, e_o3):
    both = all(m.get('recheck_pass') for m in mol_results.values())
    out = dict(p4_products_complete=bool(both),
               e_products=None, dE_project=None,
               e_monomer_sum=e_nh3 + e_o3)
    if both:
        e_products = sum(float(m['e_recheck'])
                         for m in mol_results.values())
        out['e_products'] = e_products
        out['dE_project'] = e_products - (e_nh3 + e_o3)
    return out


# =============================== main =====================================
def main():
    man = json.load(open(MANIFEST))
    if not man['all_checks_pass']:
        raise RuntimeError('HARD STOP: manifest prechecks failed')
    print('[068] P4 product endpoints under the project method model '
          '(NOT an S13 reproduction)', flush=True)

    results = dict(
        job='JOB-2026-0906-068: P4 product endpoints (HNO, H2O2) '
            'project-method optimization',
        model_status=man['model_status'],
        method=man['method'],
        stop_trigger=man['stop_trigger'],
        reference_energy=man['reference_energy'])
    ledger = ex.Ledger(LEDGER, caps=man['caps'])
    backend = {}
    mol_out = {}
    status = 'aborted'
    try:
        for sp in ('HNO', 'H2O2'):
            cat_opt = '%s_opt' % sp.lower()
            cat_rc = '%s_recheck' % sp.lower()
            cat_st = '%s_stab' % sp.lower()
            elems = man['molecules'][sp]['elements']
            x0 = np.asarray(man['molecules'][sp]['coords_start_bohr'],
                            float).reshape(-1)
            natoms = man['molecules'][sp]['natoms']
            if sp not in backend:
                backend[sp] = None
            bnd = j68_backend_for(elems)
            mdir = os.path.join(OUT, sp.lower())
            os.makedirs(mdir, exist_ok=True)

            # ---- optimization (first evaluation counts in the cap) --
            flow = run_opt_general(x0, natoms, mdir, ledger, bnd,
                                   cat_opt,
                                   opt_cap=man['caps'][cat_opt],
                                   threshold=1e-5)
            mol_out.setdefault(sp, {})['optimization'] = dict(
                outcome=flow['outcome'], n_new=flow['n_new_evals'],
                requests=flow['requests'],
                blocked=flow['blocked_requests'],
                accepted_iterates=flow['accepted_iterates'],
                optimizer=flow['optimizer'],
                warnings=flow['warnings'])
            ex.save_json_atomic(os.path.join(
                mdir, 'accepted_iterates_%s.json' % sp.lower()),
                dict(n=len(flow['accepted_iterates']),
                     iterates=flow['accepted_iterates']))
            print('[068] %s opt: %s (%d evals)' % (sp, flow['outcome'],
                                                   flow['n_new_evals']),
                  flush=True)

            passed = False
            if flow['meeting_points']:
                mp = flow['meeting_points'][-1]
                mrec = json.load(open('%s/eval_%s_%s.json'
                                      % (mdir, cat_opt, mp['tag'])))
                R_chk = np.asarray(mrec['coords_actual_angstrom'],
                                   float) * BPA
                rec_r = ex.eval_point('recheck_%s' % sp.lower(), R_chk,
                                      mdir, ledger, bnd, cat_rc,
                                      gate_fn=ex.centre_gate_fn(mrec))
                g = rec_r['gate']
                passed = bool(g['gates_pass'])
                mol_out[sp]['recheck'] = dict(gate=g,
                                              e_total=rec_r['e_total'],
                                              grad_max=rec_r['grad_max'],
                                              passed=passed)
                print('[068] %s recheck %s: dE=%.2e dgrad=%.2e dC=%.2e'
                      % (sp, 'PASS' if passed else 'FAIL', g['dE'],
                         g['dgrad_max'], g['dcoords_A']), flush=True)
                if passed:
                    # internal stability on a fresh object at the
                    # accepted geometry
                    mol_r, mf_r = bnd.new_mf(R_chk, mdir,
                                             'stab_%s' % sp.lower())
                    stab = stability_wrapper(mf_r, mdir, sp.lower())
                    a_st = ledger.pre_eval(cat_st, dict(
                        tag='stab_%s' % sp.lower()))
                    ledger.post_eval(a_st, dict(
                        stable_i=stab['stable_i'],
                        stable_e=stab['stable_e']))
                    mol_out[sp]['stability'] = dict(
                        stable_i=stab['stable_i'],
                        stable_e=stab['stable_e'],
                        raw_log=stab['raw_log'],
                        installed_order=stab['installed_order'])
                    print('[068] %s stability: stable_i=%s'
                          % (sp, stab['stable_i']), flush=True)
            else:
                print('[068] %s: no point met max|g|<=1e-5 -> not '
                      'converged, recorded truthfully' % sp, flush=True)
            mol_out[sp]['recheck_pass'] = passed
        status = 'completed'

        # per-molecule accepted energies
        for sp in ('HNO', 'H2O2'):
            m = mol_out.get(sp, {})
            if m.get('recheck', {}).get('passed'):
                m['e_recheck'] = m['recheck']['e_total']

        prod = product_reference({k: mol_out.get(k, {})
                                  for k in ('HNO', 'H2O2')},
                                 man['reference_energy'][
                                     'e_nh3_job026'],
                                 man['reference_energy'][
                                     'e_o3_job026'])
        results['product_reference'] = prod
        results['molecules'] = mol_out
        results['final'] = dict(
            status=status,
            registration=('both P4 products passed independent recheck; '
                          'project-method product reference energy '
                          'computed'
                          if prod['p4_products_complete'] else
                          'NOT both products passed; only accepted '
                          'molecules reported, NO incomplete P4 energy'),
            accounting=dict(caps=man['caps'],
                            attempts_by_cat={c: ledger.count(c)
                                             for c in man['caps']}),
            scope_limits='project method model ONLY; not an S13 '
                         'reproduction; electronic energy only (no '
                         'ZPE/thermal/CP); not aqueous free energy; '
                         'not water-treatment efficiency; stable_i does '
                         'not guarantee full reaction-path reliability')
    except Exception as exc:                                # noqa: BLE001
        results['abort'] = dict(error=str(exc),
                                traceback=traceback.format_exc()[-1500:])
        status = 'aborted'
        print('[068] ABORT: %s' % exc, flush=True)
    results['final_status'] = status
    results['budget'] = ledger.data
    ex.save_json_atomic(RESULTS, results)
    print('[068] DONE: %s' % status, flush=True)


def j68_backend_for(elements):
    return P4Backend(elements)


if __name__ == '__main__':
    main()
