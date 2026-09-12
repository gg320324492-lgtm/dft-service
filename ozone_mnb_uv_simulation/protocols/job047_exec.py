#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-047: bidirectional displacement + single-branch limited
optimization along the 045 negative-mode direction.

Budget (per-category, no borrowing): centre 1, disp 2, opt 20, recheck 1
(total cap 24).  No stability / Hessian / frequencies / CP / other species.

Flow: centre gate vs the 045 centre record -> evaluate R+ = R0 + 0.10 q and
R- = R0 - 0.10 q (actual energies, no Hessian prediction) -> branch
selection -> L-BFGS-B full-Cartesian optimization of ONE branch (start
point reused, no repeated SCF; line-search points all counted; controlled
stop as soon as any fully saved point meets unprojected max|g| <= 1e-5)
-> at most one independent recheck -> register outcome.

Module level imports only stdlib+numpy+scipy.optimize; pyscf is imported
lazily inside RealBackend so the mock test can exercise the identical
pipeline functions.
"""
import os, sys, json, time, hashlib, traceback
import numpy as np
from scipy.optimize import minimize

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
HERE = ROOT + '/protocols'
sys.path.insert(0, HERE)

S45 = ROOT + '/run_artifacts/02_nh3o3_reference/c1_negmode_scan045'
OUT = ROOT + '/run_artifacts/02_nh3o3_reference/c1_bidir_opt047'
LEDGER = OUT + '/budget_bidir047.json'
RESULTS = OUT + '/bidir_opt047_results.json'

SYMS = ['N', 'H', 'H', 'H', 'O', 'O', 'O']
ANG_PER_BOHR = 0.52917721092
GRID_LEVEL = 8
SCF_TOL = 1e-12
SCF_TOL_GRAD = 1e-9
GATE_C = 1e-9
GATE_E = 1e-8
GATE_G = 1e-7
LOWER_THRESH = 1e-8        # side must lower the energy by MORE than this
TIE_THRESH = 1e-8          # |E+ - E-| tie rule -> prefer positive side
GM_GATE = 1e-5             # unprojected max|g| acceptance gate
OPT_CAP = 20
MAXITER = 20
CHARTER_Q_HASH = '59599f2a2c0be260'
TAGS = ('p10', 'm10')


def save_json_atomic(path, obj):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(obj, fh, indent=2, default=str)
    os.replace(tmp, path)


def sha_arr(a):
    return hashlib.sha256(np.asarray(a, float).tobytes()).hexdigest()[:16]


class ThresholdReached(Exception):
    def __init__(self, tag, gmax):
        super().__init__('gradient gate met at %s (max|g|=%.3e)'
                         % (tag, gmax))
        self.tag = tag
        self.gmax = gmax


class BudgetExhausted(Exception):
    pass


# ====================================================================
# ledger: per-category caps, register BEFORE each call
# ====================================================================
class Ledger:
    DEFAULT_CAPS = dict(centre=1, disp=2, opt=20, recheck=1)

    def __init__(self, path, caps=None):
        self.path = path
        if os.path.exists(path):
            self.data = json.load(open(path))
        else:
            self.data = dict(caps=dict(caps or self.DEFAULT_CAPS),
                             note='failures count within their category; '
                                  'no borrowing; no retries; no auto-resume',
                             attempts=[])
            save_json_atomic(path, self.data)
        for a in self.data['attempts']:
            if a.get('status') == 'error':
                raise RuntimeError('ledger has error attempt -> HARD STOP: '
                                   + json.dumps(a)[:300])

    def _save(self):
        save_json_atomic(self.path, self.data)

    def count(self, category):
        return sum(1 for a in self.data['attempts']
                   if a['category'] == category)

    def pre_eval(self, category, note):
        cap = self.data['caps'].get(category)
        if cap is None:
            raise RuntimeError('unknown category %s' % category)
        if self.count(category) >= cap:
            raise RuntimeError('cap reached for %s (%d)' % (category, cap))
        att = dict(attempt=len(self.data['attempts']) + 1,
                   category=category, note=note, status='pending',
                   started=time.strftime('%F %T'))
        self.data['attempts'].append(att)
        self._save()
        return att

    def post_eval(self, att, results):
        att.update(results)
        att['status'] = 'done'
        att['finished'] = time.strftime('%F %T')
        self._save()

    def fail(self, att, err):
        att['status'] = 'error'
        att['error'] = str(err)
        att['traceback'] = traceback.format_exc()[-1500:]
        self._save()


# ====================================================================
# one evaluation = one SCF + one full gradient (grid_response=True)
# ====================================================================
def eval_point(tag, R_bohr, out_dir, ledger, backend, category, gate_fn=None):
    att = ledger.pre_eval(category, dict(tag=tag))
    try:
        rec = backend.full_eval(np.asarray(R_bohr, float), out_dir, tag)
        rec['tag'] = tag
        rec['category'] = category
        save_json_atomic(out_dir + '/eval_%s_%s.json' % (category, tag), rec)
        ledger.post_eval(att, dict(stage='%s_%s' % (category, tag),
                                   e_total=rec['e_total'],
                                   grad_max=rec['grad_max']))
        if gate_fn is not None:
            rec['gate'] = gate_fn(rec)
            save_json_atomic(out_dir + '/eval_%s_%s.json' % (category, tag),
                             rec)
            if not rec['gate']['gates_pass']:
                save_json_atomic(out_dir + '/STOPPED_gate_fail_%s.json'
                                 % tag, dict(reason='gate failed',
                                             gate=rec['gate']))
                raise RuntimeError('HARD STOP: %s gate failed: %s'
                                   % (category, rec['gate']))
        return rec
    except Exception as exc:  # noqa: BLE001
        ledger.fail(att, exc)
        raise


def centre_gate_fn(ref):
    """Centre reproduction gate against the 045 centre record."""
    ref_g = np.asarray(ref['grad'], float).reshape(-1)
    ref_C = np.asarray(ref['coords_actual_angstrom'], float)

    def gate(rec):
        dC = float(np.abs(np.asarray(rec['coords_actual_angstrom'], float)
                          - ref_C).max())
        dE = abs(float(rec['e_total']) - float(ref['e_total']))
        dg = float(np.abs(np.asarray(rec['grad'], float).reshape(-1)
                          - ref_g).max())
        return dict(dcoords_A=dC, dE=dE, dgrad_max=dg,
                    converged=bool(rec['converged']),
                    finite=bool(rec['all_finite']),
                    config_match=bool(rec['config'] == ref['config']),
                    gates_pass=bool(dC <= GATE_C and dE <= GATE_E
                                    and dg <= GATE_G and rec['converged']
                                    and rec['all_finite']
                                    and rec['config'] == ref['config']))
    return gate


# ====================================================================
# branch selection (actual energies, no Hessian prediction)
# ====================================================================
def select_branch(rec_centre, rec_p, rec_m):
    E0 = float(rec_centre['e_total'])
    dEp = float(rec_p['e_total']) - E0
    dEm = float(rec_m['e_total']) - E0
    info = dict(E_centre_batch=E0, E_plus=float(rec_p['e_total']),
                E_minus=float(rec_m['e_total']), dE_plus=dEp, dE_minus=dEm,
                lowering_threshold=-LOWER_THRESH,
                threshold_meaning='side must be LOWER than this batch '
                                  'centre by more than 1e-8 Eh')
    elig = []
    if dEp < -LOWER_THRESH:
        elig.append('p10')
    if dEm < -LOWER_THRESH:
        elig.append('m10')
    info['eligible'] = elig
    if not elig:
        info['decision'] = 'stop_no_side_lowered'
        return None, info
    if len(elig) == 2 and abs(dEp - dEm) <= TIE_THRESH:
        info['decision'] = 'tie_rule_prefer_positive'
        info['near_equal'] = True
        return 'p10', info
    info['near_equal'] = False
    if len(elig) == 2:
        info['decision'] = 'both_lower_pick_lower_energy'
        return ('p10' if dEp < dEm else 'm10'), info
    info['decision'] = 'single_lowering_side'
    return elig[0], info


# ====================================================================
# optimization of ONE branch (L-BFGS-B, 21 Cartesian variables)
# ====================================================================
def run_optimization(start_rec, out_dir, ledger, backend,
                     opt_cap=OPT_CAP, maxiter=MAXITER):
    x0 = (np.asarray(start_rec['coords_actual_angstrom'], float)
          / ANG_PER_BOHR).reshape(-1)
    start_sha = sha_arr(x0.reshape(7, 3))
    state = dict(n_new=0, records=[], reuse_used=False, accepted=[])

    if float(start_rec['grad_max']) <= GM_GATE:
        return dict(outcome='start_already_meets_gate',
                    start_gmax=float(start_rec['grad_max']),
                    reuse_start=True, n_new_evals=0,
                    records=[], accepted=[],
                    meeting_points=[dict(tag=start_rec['tag'],
                                         category=start_rec['category'],
                                         grad_max=float(start_rec['grad_max']))],
                    note='displacement point already meets max|g|<=1e-5; '
                         'optimization NOT started (charter 5)')

    def fun(x):
        x = np.asarray(x, float)
        if state['n_new'] == 0 and sha_arr(x.reshape(7, 3)) == start_sha:
            state['reuse_used'] = True
            return (float(start_rec['e_total']),
                    np.asarray(start_rec['grad'], float).reshape(-1))
        if state['n_new'] >= opt_cap:
            raise BudgetExhausted('opt attempt cap %d reached' % opt_cap)
        tag = 'opt_%02d' % (state['n_new'] + 1)
        rec = eval_point(tag, x.reshape(7, 3), out_dir, ledger, backend,
                         'opt')
        state['n_new'] += 1
        state['records'].append(rec)
        if float(rec['grad_max']) <= GM_GATE:
            raise ThresholdReached(tag, float(rec['grad_max']))
        return float(rec['e_total']), np.asarray(rec['grad'],
                                                 float).reshape(-1)

    def cb(xk):
        xk = getattr(xk, 'x', xk)
        state['accepted'].append(np.asarray(xk, float).tolist())

    optimizer_note = ('L-BFGS-B freshly initialized (no warm start); '
                      'gtol=1e-6 ftol=1e-13 maxcor=10 maxiter=%d; '
                      '21 Cartesian variables, unconstrained, no '
                      'projection, no per-step registration' % maxiter)
    try:
        res = minimize(fun, x0, method='L-BFGS-B', jac=True, callback=cb,
                       options=dict(gtol=1e-6, ftol=1e-13, maxcor=10,
                                    maxiter=maxiter))
        outcome = 'optimizer_returned'
        res_info = dict(success=bool(res.success), message=str(res.message),
                        nit=int(res.nit),
                        nfev=int(getattr(res, 'nfev', -1)),
                        final_jac_gmax=float(np.abs(np.asarray(
                            res.jac, float)).max()))
        xm_sha = sha_arr(np.asarray(res.x, float).reshape(7, 3))
        match = next((r for r in state['records']
                      if sha_arr(np.asarray(r['coords_actual_angstrom'],
                                            float) / ANG_PER_BOHR)
                      == xm_sha), None)
        res_info['final_x_matches_saved_point'] = match is not None
        if match is not None:
            res_info['final_x_tag'] = match['tag']
            if float(match['grad_max']) <= GM_GATE:
                state['records'][-1]['gate_note'] = \
                    'final optimizer point meets the gradient gate'
    except ThresholdReached as e:
        outcome = 'threshold_reached'
        res_info = dict(tag=e.tag, gmax=e.gmax,
                        message='controlled stop: saved point met '
                                'max|g|<=1e-5')
    except BudgetExhausted as e:
        outcome = 'budget_exhausted'
        res_info = dict(message=str(e))

    meeting = [dict(tag=r['tag'], category=r['category'],
                    grad_max=float(r['grad_max']),
                    e_total=float(r['e_total']))
               for r in state['records'] if float(r['grad_max']) <= GM_GATE]
    return dict(outcome=outcome, optimizer=res_info,
                reuse_start=state['reuse_used'],
                n_new_evals=state['n_new'],
                accepted_iterates=state['accepted'],
                meeting_points=meeting,
                optimizer_setup=optimizer_note)


# ====================================================================
# recheck (independent new object, at most one attempt)
# ====================================================================
def recheck_gate_fn(meeting_rec):
    ref_g = np.asarray(meeting_rec['grad'], float).reshape(-1)
    ref_C = np.asarray(meeting_rec['coords_actual_angstrom'], float)

    def gate(rec):
        dC = float(np.abs(np.asarray(rec['coords_actual_angstrom'], float)
                          - ref_C).max())
        dE = abs(float(rec['e_total']) - float(meeting_rec['e_total']))
        dg = float(np.abs(np.asarray(rec['grad'], float).reshape(-1)
                          - ref_g).max())
        return dict(dcoords_A=dC, dE_vs_meeting=dE,
                    dgrad_max_vs_meeting=dg,
                    gmax_recheck=float(rec['grad_max']),
                    converged=bool(rec['converged']),
                    config_match=bool(rec['config']
                                      == meeting_rec['config']),
                    gates_pass=bool(rec['converged']
                                    and rec['config']
                                    == meeting_rec['config']
                                    and float(rec['grad_max']) <= GM_GATE
                                    and dE <= GATE_E and dg <= GATE_G
                                    and dC <= GATE_C))
    return gate


# ====================================================================
# real backend (lazy pyscf import)
# ====================================================================
class RealBackend:
    def __init__(self, out_dir):
        from pyscf import gto
        import d2_full
        self._gto = gto
        self._d2 = d2_full
        self.out_dir = out_dir

    def make_mol(self, R_bohr):
        atom = [(s, (float(r[0]), float(r[1]), float(r[2])))
                for s, r in zip(SYMS, np.asarray(R_bohr, float))]
        return self._gto.M(atom=atom, basis='def2-TZVP', charge=0, spin=0,
                           verbose=0, max_memory=4000, unit='Bohr')

    def full_eval(self, R_bohr, out_dir, tag):
        mol = self.make_mol(R_bohr)
        mf = self._d2.make_mf_d2(mol, solvent=None)
        mf.grids.level = GRID_LEVEL
        mf.conv_tol = SCF_TOL
        mf.conv_tol_grad = SCF_TOL_GRAD
        mf.chkfile = os.path.join(out_dir, 'chk_%s.chk' % tag)
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
        g_obj.grid_response = True          # production setting
        if getattr(g_obj, 'grid_response', None) is not True:
            raise RuntimeError('grid_response=True did not reach the '
                               'gradient object')
        g = np.asarray(g_obj.kernel(), float).reshape(7, 3)
        C_act = np.asarray(mol.atom_coords(unit='Angstrom'), float)
        e_d2 = float(self._d2.d2_energy(mol))
        cfg = dict(xc='wb97xd (project -D2 attached)',
                   d2_attached=bool(getattr(mf, '_has_full_d2', False)),
                   grid_level=int(getattr(mf.grids, 'level', -1)),
                   scf_tol=[float(mf.conv_tol), float(mf.conv_tol_grad)],
                   basis='def2-TZVP', charge=0, spin=0, grid_response=True,
                   solvent='none (gas phase)')
        chk = os.path.join(out_dir, 'chk_%s.chk' % tag)
        chk_info = (dict(path=os.path.basename(chk),
                         size=os.path.getsize(chk))
                    if os.path.exists(chk) else dict(path=chk, exists=False))
        return dict(e_total=e, e_d2_Eh=e_d2, e_dft_part_Eh=e - e_d2,
                    grad=g.tolist(), grad_max=float(np.abs(g).max()),
                    grad_sha=sha_arr(g),
                    coords_actual_angstrom=C_act.tolist(),
                    coords_sha=sha_arr(C_act), config=cfg,
                    chkfile=chk_info,
                    mo_summary=self._mo(mf),
                    converged=conv,
                    all_finite=bool(np.isfinite(e) and np.isfinite(g).all()),
                    scf_kernel_count=int(getattr(
                        mf, '_exec_scf_kernel_count', -1)))

    @staticmethod
    def _mo(mf):
        e = np.asarray(mf.mo_energy, float)
        occ = np.asarray(mf.mo_occ, float)
        nocc = int(round(occ.sum()))
        s = dict(n_mo=int(e.size), nocc=nocc)
        if 0 < nocc <= e.size:
            s['homo'] = float(e[nocc - 1])
            if nocc < e.size:
                s['lumo'] = float(e[nocc])
                s['gap_Eh'] = float(e[nocc] - e[nocc - 1])
        return s


# ====================================================================
# main
# ====================================================================
def main():
    man = json.load(open(OUT + '/input_manifest.json'))
    if man['direction']['q_hash'] != CHARTER_Q_HASH:
        raise RuntimeError('HARD STOP: direction q hash mismatch')
    if not all(g['all_checks_pass'] for g in man['geometries'].values()):
        raise RuntimeError('HARD STOP: displacement input checks failed')
    c45 = json.load(open(S45 + '/eval_eval_centre_centre.json'))

    results = dict(job='JOB-2026-0906-047 bidirectional displacement + '
                       'single-branch optimization',
                   q_hash=CHARTER_Q_HASH,
                   optimizer_setup_note='fixed before run: scipy L-BFGS-B, '
                                        'gtol=1e-6, ftol=1e-13, maxcor=10, '
                                        'maxiter=20, 21 Cartesian variables, '
                                        'unconstrained, no projection, no '
                                        'per-step registration')
    ledger = Ledger(LEDGER)
    backend = RealBackend(OUT)
    BPA = 1.0 / ANG_PER_BOHR

    # 1) centre gate
    R0 = (np.asarray(c45['coords_actual_angstrom'], float) * BPA)
    rec_c = eval_point('centre', R0, OUT, ledger, backend, 'centre',
                       gate_fn=centre_gate_fn(c45))
    print('[047] centre gate PASS: dE=%.2e dgrad=%.2e'
          % (rec_c['gate']['dE'], rec_c['gate']['dgrad_max']), flush=True)
    results['centre'] = dict(e_total=rec_c['e_total'],
                             grad_max=rec_c['grad_max'],
                             gate=rec_c['gate'])

    # 2) two displacement evaluations (actual energies)
    rec_p = eval_point('disp_p10', np.asarray(
        man['geometries']['p10']['coords_bohr']), OUT, ledger, backend,
        'disp')
    rec_m = eval_point('disp_m10', np.asarray(
        man['geometries']['m10']['coords_bohr']), OUT, ledger, backend,
        'disp')
    results['displacements'] = {
        'p10': dict(e_total=rec_p['e_total'], grad_max=rec_p['grad_max']),
        'm10': dict(e_total=rec_m['e_total'], grad_max=rec_m['grad_max'])}
    print('[047] E(R+)=%.9f (dE=%+.3e)  E(R-)=%.9f (dE=%+.3e)'
          % (rec_p['e_total'], rec_p['e_total'] - rec_c['e_total'],
             rec_m['e_total'], rec_m['e_total'] - rec_c['e_total']),
          flush=True)

    # 3) branch selection
    chosen, info = select_branch(rec_c, rec_p, rec_m)
    results['branch_selection'] = info
    save_json_atomic(OUT + '/branch_selection.json', info)
    print('[047] branch: %s (%s)' % (chosen, info['decision']), flush=True)
    if chosen is None:
        results['status'] = 'stopped_no_side_lowered'
        save_json_atomic(RESULTS, results)
        print('[047] no side lowered beyond 1e-8 Eh -> STOP (no '
              'displacement expansion)', flush=True)
        return

    chosen_rec = rec_p if chosen == 'p10' else rec_m
    # 4) optimization (or direct recheck if the start meets the gate)
    opt = run_optimization(chosen_rec, OUT, ledger, backend)
    results['optimization'] = opt
    print('[047] optimization outcome: %s (new evals %d, reuse_start %s)'
          % (opt['outcome'], opt['n_new_evals'], opt['reuse_start']),
          flush=True)

    # 5) recheck if any fully saved point meets the gate
    status = 'not_converged'
    if opt['meeting_points']:
        mp = opt['meeting_points'][0]
        meeting_rec = json.load(open('%s/eval_%s_%s.json'
                                     % (OUT, mp['category'], mp['tag'])))
        rec_r = eval_point('recheck', np.asarray(
            meeting_rec['coords_actual_angstrom'], float) * BPA,
            OUT, ledger, backend, 'recheck',
            gate_fn=recheck_gate_fn(meeting_rec))
        results['recheck'] = dict(gate=rec_r['gate'],
                                  e_total=rec_r['e_total'],
                                  grad_max=rec_r['grad_max'])
        if rec_r['gate']['gates_pass']:
            status = 'stationary_point_candidate'
            print('[047] recheck PASS -> registered: stationary point '
                  'candidate, stability & frequencies pending', flush=True)
        else:
            status = 'recheck_failed'
            print('[047] recheck FAILED -> not registered', flush=True)
    else:
        print('[047] no saved point met max|g|<=1e-5 -> not converged',
              flush=True)
    results['status'] = status
    results['budget'] = ledger.data
    save_json_atomic(RESULTS, results)
    print('DONE: centre=%d/1 disp=%d/2 opt=%d/20 recheck=%d/1'
          % (ledger.count('centre'), ledger.count('disp'),
             ledger.count('opt'), ledger.count('recheck')), flush=True)


if __name__ == '__main__':
    main()
