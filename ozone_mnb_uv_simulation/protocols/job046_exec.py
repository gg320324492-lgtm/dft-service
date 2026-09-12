#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-046: grid-response contribution check along the C1
negative-mode direction.

Budget: SCF x2 (one per fixed geometry), full gradient x4
(grid_response=True/False on the SAME converged SCF object),
analytic D2 gradient x2 (single-counted, NO SCF).  No optimisation,
stability, Hessian, frequency recalculation.

Fixed geometries are read from the 045 actual evaluation records
(+0.01 / -0.01 Bohr); direction q is re-verified against the charter
hash 59599f2a2c0be260.  Module level imports only stdlib+numpy so the
mock test can import this file without pyscf; RealBackend imports
pyscf lazily.
"""
import os, sys, json, time, hashlib, traceback
import numpy as np

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
HERE = ROOT + '/protocols'
sys.path.insert(0, HERE)

S45 = ROOT + '/run_artifacts/02_nh3o3_reference/c1_negmode_scan045'
OUT = ROOT + '/run_artifacts/02_nh3o3_reference/c1_gridresp046'
LEDGER = OUT + '/budget_scan046.json'
RESULTS = OUT + '/grid_response_comparison.json'

SYMS = ['N', 'H', 'H', 'H', 'O', 'O', 'O']
ANG_PER_BOHR = 0.52917721092
GRID_LEVEL = 8            # 045/043 endpoint configuration
SCF_TOL = 1e-12
SCF_TOL_GRAD = 1e-9
GATE_C = 1e-9             # Angstrom
GATE_E = 1e-8             # Eh
GATE_G = 1e-7             # Eh/Bohr
H_STEP = 0.01             # Bohr (charter fixes h = 0.01 for this batch)
CHARTER_Q_HASH = '59599f2a2c0be260'
TAGS = ('p01', 'm01')     # +0.01 then -0.01


def save_json_atomic(path, obj):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(obj, fh, indent=2, default=str)
    os.replace(tmp, path)


def sha_arr(a):
    return hashlib.sha256(np.asarray(a, float).tobytes()).hexdigest()[:16]


# ====================================================================
# ledger: per-category caps, register BEFORE each call, done after save
# ====================================================================
class Ledger:
    def __init__(self, path):
        self.path = path
        if os.path.exists(path):
            self.data = json.load(open(path))
        else:
            self.data = dict(
                caps=dict(scf=2, grad=4, d2_grad=2),
                note='d2_grad is single-counted analytic (no SCF); '
                     'failures count; no retries / no auto-resume',
                attempts=[])
            save_json_atomic(path, self.data)
        for a in self.data['attempts']:
            if a.get('status') == 'error':
                raise RuntimeError('ledger has error attempt -> HARD STOP: '
                                   + json.dumps(a)[:300])

    def _save(self):
        save_json_atomic(self.path, self.data)

    def _count(self, category):
        return sum(1 for a in self.data['attempts']
                   if a['category'] == category)

    def pre_eval(self, category, note):
        cap = self.data['caps'].get(category)
        if cap is None:
            raise RuntimeError('unknown category %s' % category)
        if self._count(category) >= cap:
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
# per-point pipeline (shared by real backend and mock test)
# ====================================================================
def evaluate_point(t_tag, ref, out_dir, ledger, backend):
    """One fixed geometry: SCF once -> gradient(grid_response=True) ->
    reproduction gate vs 045 -> gradient(grid_response=False) on the SAME
    SCF object -> analytic D2 gradient (no SCF)."""
    # ---- 1) SCF (attempt scf) ----
    att = ledger.pre_eval('scf', dict(tag=t_tag))
    try:
        mol = backend.make_mol(np.asarray(ref['coords_actual_angstrom'],
                                          float))
        mf = backend.make_mf(mol, out_dir, t_tag)
        mf.kernel()
        e = float(mf.e_tot)
        conv = bool(mf.converged)
        C_act = np.asarray(backend.mol_coords_angstrom(mol), float)
        cfg = backend.cfg_readback_mf(mf)
        rec = dict(tag=t_tag, e_total=e, converged=conv,
                   coords_actual_angstrom=C_act.tolist(),
                   coords_sha=sha_arr(C_act), config=cfg,
                   chkfile=backend.chkfile_info(mf),
                   mo_summary=backend.mo_summary(mf),
                   all_finite=bool(np.isfinite(e)))
        save_json_atomic(out_dir + '/eval_scf_%s.json' % t_tag, rec)
        ledger.post_eval(att, dict(stage='scf_%s' % t_tag, e_total=e,
                                   converged=conv))
    except Exception as exc:  # noqa: BLE001
        ledger.fail(att, exc)
        raise
    # ---- 2) gradient grid_response=True + reproduction gate ----
    att = ledger.pre_eval('grad', dict(tag=t_tag, setting='on'))
    try:
        g_on, cfg_g = backend.gradient(mf, grid_response=True)
        dC = float(np.abs(C_act - np.asarray(ref['coords_actual_angstrom'],
                                             float)).max())
        dE = abs(e - float(ref['e_total']))
        dg = float(np.abs(g_on.reshape(-1)
                          - np.asarray(ref['grad'], float).reshape(-1)).max())
        gate = dict(
            dcoords_vs_045_A=dC, dE_vs_045=dE, dgrad_max_vs_045=dg,
            scf_converged=conv, finite=bool(np.isfinite(g_on).all()),
            gates_pass=bool(dC <= GATE_C and dE <= GATE_E and dg <= GATE_G
                            and conv and np.isfinite(g_on).all()))
        rec = dict(tag=t_tag, setting='on', grad=g_on.tolist(),
                   grad_max=float(np.abs(g_on).max()), config=cfg_g,
                   reproduction_gate=gate)
        save_json_atomic(out_dir + '/eval_grad_on_%s.json' % t_tag, rec)
        ledger.post_eval(att, dict(stage='grad_on_%s' % t_tag,
                                   grad_max=rec['grad_max'],
                                   gate_pass=gate['gates_pass']))
        if not gate['gates_pass']:
            save_json_atomic(out_dir + '/STOPPED_gate_fail_%s.json' % t_tag,
                             dict(reason='True-gradient reproduction gate '
                                         'failed', gate=gate))
            raise RuntimeError('HARD STOP: True-gradient reproduction gate '
                               'failed at %s: %s' % (t_tag, gate))
    except Exception as exc:  # noqa: BLE001
        ledger.fail(att, exc)
        raise
    # ---- 3) gradient grid_response=False, SAME SCF object, NEW object ----
    att = ledger.pre_eval('grad', dict(tag=t_tag, setting='off'))
    try:
        g_off, cfg_g = backend.gradient(mf, grid_response=False)
        rec = dict(tag=t_tag, setting='off', grad=g_off.tolist(),
                   grad_max=float(np.abs(g_off).max()), config=cfg_g,
                   same_scf_object_as_on=True,
                   all_finite=bool(np.isfinite(g_off).all()))
        save_json_atomic(out_dir + '/eval_grad_off_%s.json' % t_tag, rec)
        ledger.post_eval(att, dict(stage='grad_off_%s' % t_tag,
                                   grad_max=rec['grad_max']))
    except Exception as exc:  # noqa: BLE001
        ledger.fail(att, exc)
        raise
    # ---- 4) analytic D2 gradient (attempt d2_grad, single-counted) ----
    att = ledger.pre_eval('d2_grad', dict(tag=t_tag, note='analytic, no SCF'))
    try:
        n_scf_before = backend.mf_kernel_count(mf)
        g_d2 = backend.d2_grad(mol)
        n_scf_after = backend.mf_kernel_count(mf)
        rec = dict(tag=t_tag, grad=g_d2.tolist(),
                   grad_max=float(np.abs(g_d2).max()),
                   scf_kernel_count_unchanged=bool(n_scf_before
                                                   == n_scf_after),
                   note='analytic project -D2 gradient, no SCF involved')
        save_json_atomic(out_dir + '/eval_d2grad_%s.json' % t_tag, rec)
        ledger.post_eval(att, dict(stage='d2grad_%s' % t_tag,
                                   grad_max=rec['grad_max']))
    except Exception as exc:  # noqa: BLE001
        ledger.fail(att, exc)
        raise
    return dict(tag=t_tag, e_total=e, g_on=g_on, g_off=g_off, g_d2=g_d2,
                C_act=C_act, converged=conv)


# ====================================================================
# comparison (charter section 5)
# ====================================================================
def compare(res, q, out_dir):
    h = H_STEP
    prep = json.load(open(out_dir + '/prep_check_results.json'))
    k_H = prep['h043_check']['k_H_total']
    k_H_dft = prep['h043_check']['k_H_dft']
    k_H_d2 = prep['h043_check']['k_H_d2']
    d45 = json.load(open(S45 + '/direction_derivative_results.json'))
    k_g045 = d45['direction_analysis']['per_h']['0.01']['k_g_Eh_Bohr2']
    k_E045 = d45['direction_analysis']['per_h']['0.01']['k_E_Eh_Bohr2']

    gp, gm = res['p01'], res['m01']

    def k_of(gdict):
        return (float(np.dot(gdict['p01'].reshape(-1)
                             - gdict['m01'].reshape(-1), q)) / (2.0 * h))

    k_on = k_of({t: res[t]['g_on'] for t in TAGS})
    k_off = k_of({t: res[t]['g_off'] for t in TAGS})
    dk_grid = k_on - k_off
    # D2-only curvature along q from the analytic gradients (consistency)
    k_d2 = k_of({t: res[t]['g_d2'] for t in TAGS})
    # DFT-only (D2 subtracted per point) — both settings
    g_dft_on = {t: res[t]['g_on'] - res[t]['g_d2'] for t in TAGS}
    g_dft_off = {t: res[t]['g_off'] - res[t]['g_d2'] for t in TAGS}
    k_on_dft = k_of(g_dft_on)
    k_off_dft = k_of(g_dft_off)
    dk_grid_dft = k_on_dft - k_off_dft

    out = dict(
        h_Bohr=h, direction_q_hash=CHARTER_Q_HASH,
        k_on_total=k_on, k_off_total=k_off, dk_grid=dk_grid,
        k_d2_curvature_from_analytic=k_d2,
        k_on_dft=k_on_dft, k_off_dft=k_off_dft, dk_grid_dft=dk_grid_dft,
        internal_consistency=dict(
            dk_grid_vs_dk_grid_dft_abs=abs(dk_grid - dk_grid_dft),
            note='D2 is grid-response-free; total and DFT-subtracted '
                 'grid-response contributions must agree'),
        ref_045=dict(k_g_001=k_g045, k_E_001=k_E045,
                     k_on_minus_k_g045=k_on - k_g045),
        ref_043=dict(k_H_total=k_H, k_H_dft=k_H_dft, k_H_d2=k_H_d2),
        differences=dict(
            original_gap=k_H - k_on,
            response_off_gap=k_H - k_off,
            grid_response_contribution=dk_grid,
            explained_fraction=(dk_grid / (k_H - k_on)
                                if abs(k_H - k_on) > 0 else None)),
        d2_added_once_per_path=True)
    save_json_atomic(out_dir + '/grid_response_comparison.json', out)

    print('\n=== grid-response curvature comparison (h=0.01 Bohr) ===',
          flush=True)
    print('k_on  (total, resp=True) : %+.6e Eh/Bohr^2' % k_on, flush=True)
    print('k_off (total, resp=False): %+.6e Eh/Bohr^2' % k_off, flush=True)
    print('delta_k_grid = k_on-k_off: %+.6e Eh/Bohr^2' % dk_grid, flush=True)
    print('k_d2 analytic curvature  : %+.6e (Hessian ref %+.6e)'
          % (k_d2, k_H_d2), flush=True)
    print('k_on_dft / k_off_dft     : %+.6e / %+.6e (dk_grid_dft %+.6e)'
          % (k_on_dft, k_off_dft, dk_grid_dft), flush=True)
    print('045 reproduction: k_on - k_g045(0.01) = %+.3e'
          % (k_on - k_g045), flush=True)
    print('k_H - k_on  (original gap)        : %+.6e' % (k_H - k_on),
          flush=True)
    print('k_H - k_off (response-off gap)    : %+.6e' % (k_H - k_off),
          flush=True)
    if out['differences']['explained_fraction'] is not None:
        print('explained fraction dk_grid/(k_H-k_on) = %.3f'
              % out['differences']['explained_fraction'], flush=True)
    return out


# ====================================================================
# real backend (pyscf imported lazily; NEVER imported by the mock test)
# ====================================================================
class RealBackend:
    def __init__(self, out_dir):
        from pyscf import gto
        import d2_full
        self._gto = gto
        self._d2 = d2_full
        self.out_dir = out_dir

    def make_mol(self, C_A):
        R = C_A / ANG_PER_BOHR
        atom = [(s, (float(r[0]), float(r[1]), float(r[2])))
                for s, r in zip(SYMS, R)]
        return self._gto.M(atom=atom, basis='def2-TZVP', charge=0, spin=0,
                           verbose=0, max_memory=4000, unit='Bohr')

    def make_mf(self, mol, out_dir, tag):
        mf = self._d2.make_mf_d2(mol, solvent=None)
        mf.grids.level = GRID_LEVEL
        mf.conv_tol = SCF_TOL
        mf.conv_tol_grad = SCF_TOL_GRAD
        mf.chkfile = os.path.join(out_dir, 'chk_%s.chk' % tag)
        # instance-level SCF call counter (no source modification): lets the
        # d2_grad step prove no SCF was (re)started on the real path too
        orig_kernel = mf.kernel

        def counting_kernel(*a, **k):
            mf._exec_scf_kernel_count = getattr(
                mf, '_exec_scf_kernel_count', 0) + 1
            return orig_kernel(*a, **k)

        mf.kernel = counting_kernel
        mf._exec_scf_kernel_count = 0
        return mf

    def mol_coords_angstrom(self, mol):
        return np.asarray(mol.atom_coords(unit='Angstrom'), float)

    def chkfile_info(self, mf):
        p = mf.chkfile
        if p and os.path.exists(p):
            return dict(path=os.path.basename(p),
                        size=os.path.getsize(p),
                        sha256=hashlib.sha256(open(p, 'rb').read())
                        .hexdigest())
        return dict(path=p, exists=False)

    def mo_summary(self, mf):
        e = np.asarray(mf.mo_energy, float)
        occ = np.asarray(mf.mo_occ, float)
        nocc = int(round(occ.sum()))
        s = dict(n_mo=int(e.size), nocc=nocc)
        if 0 < nocc <= e.size:
            s.update(homo=float(e[nocc - 1]),
                     lumo=float(e[nocc]) if nocc < e.size else None)
            if nocc < e.size:
                s['gap_Eh'] = float(e[nocc] - e[nocc - 1])
        return s

    def gradient(self, mf, grid_response):
        g = mf.nuc_grad_method()          # fresh D2-attached Gradients
        g.grid_response = bool(grid_response)   # explicit set...
        gr = getattr(g, 'grid_response', None)  # ...and readback
        if gr is not bool(grid_response):
            raise RuntimeError('grid_response did not reach gradient object')
        val = np.asarray(g.kernel(), float).reshape(7, 3)
        cfg = dict(grid_response=gr,
                   d2_attached=bool(getattr(g, '_has_full_d2', False)),
                   grad_class=type(g).__name__)
        return val, cfg

    def d2_grad(self, mol):
        return np.asarray(self._d2.d2_grad(mol), float).reshape(7, 3)

    def mf_kernel_count(self, mf):
        return int(getattr(mf, '_exec_scf_kernel_count', 0))

    def cfg_readback_mf(self, mf):
        return dict(xc='wb97xd (project -D2 attached)',
                    d2_attached=bool(getattr(mf, '_has_full_d2', False)),
                    grid_level=int(getattr(mf.grids, 'level', -1)),
                    scf_tol=[float(mf.conv_tol), float(mf.conv_tol_grad)],
                    basis='def2-TZVP', charge=0, spin=0,
                    grid_response='per-gradient (see grad records)',
                    solvent='none (gas phase)')


# ====================================================================
# main
# ====================================================================
def main():
    # q re-verification + prep gate
    man = json.load(open(S45 + '/input_manifest.json'))
    q = np.asarray(man['direction']['q_vector'], float)
    if sha_arr(q) != CHARTER_Q_HASH:
        raise RuntimeError('HARD STOP: direction q hash mismatch')
    prep = json.load(open(OUT + '/prep_check_results.json'))
    if not (prep['q_check']['match']
            and prep['geometry_rebuild']['round_trip_ok']):
        raise RuntimeError('HARD STOP: prep checks not passed')
    results = dict(job='JOB-2026-0906-046 grid-response check',
                   q_hash=CHARTER_Q_HASH, h_Bohr=H_STEP)
    ledger = Ledger(LEDGER)
    backend = RealBackend(OUT)
    res = {}
    for tag in TAGS:
        ref = json.load(open(S45 + '/eval_eval_displacement_%s.json'
                             % {'p01': '+0.01', 'm01': '-0.01'}[tag]))
        t0 = time.time()
        res[tag] = evaluate_point(tag, ref, OUT, ledger, backend)
        print('[046] point %s done (%.1fs): E=%.9f gmax_on=%.3e '
              'gmax_off=%.3e'
              % (tag, time.time() - t0, res[tag]['e_total'],
                 float(np.abs(res[tag]['g_on']).max()),
                 float(np.abs(res[tag]['g_off']).max())), flush=True)
        save_json_atomic(OUT + '/partial_results.json',
                         {k: {kk: (vv.tolist() if isinstance(vv, np.ndarray)
                                   else vv)
                              for kk, vv in v.items()} for k, v in
                          res.items()})
    comp = compare(res, q, OUT)
    results['comparison'] = comp
    results['budget'] = ledger.data
    save_json_atomic(RESULTS, results)
    print('DONE: budget scf=%d/2 grad=%d/4 d2_grad=%d/2'
          % (ledger._count('scf'), ledger._count('grad'),
             ledger._count('d2_grad')), flush=True)


if __name__ == '__main__':
    main()
