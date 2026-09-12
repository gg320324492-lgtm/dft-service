#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-070 step C: execution (REAL evaluations, ledgered).

P1(2) radical endpoints (HOO., H2NO.) under the project method model:
UKS doublets (spin=1, charge 0), wb97xd + project explicit D2 once,
def2-TZVP, L8, grid_response=True, SCF 1e-12/1e-9, gas.

Per radical: opt cap 15 (every new SCF+gradient including line-search
trials) -> threshold unprojected max|g| <= 1e-5 -> independent recheck
(1) -> internal stability (1) on a FRESH mf CONVERGED at the accepted
geometry (JOB-068 flow-anomaly fix; the reconvergence SCF is INSIDE
the stab attempt).  <S²> is recorded after every kernel; spin
contamination delta = <S²> - 0.75.

The two radicals are INDEPENDENT: if one fails to converge within its
cap it is recorded truthfully and the other continues.  A REAL
exception aborts the whole batch.  P1 separated-fragment energy is
computed ONLY if BOTH rechecks pass, and is labelled a project-method
electronic energy for QUALITATIVE comparison with the S13 values -
never called a P1 overall reaction energy.
"""
import os, sys, json, time, hashlib, traceback
import numpy as np

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
HERE = ROOT + '/protocols'
sys.path.insert(0, HERE)
import job047_exec as ex
import job068_exec as j68

OUT = ROOT + '/run_artifacts/02_nh3o3_reference/p1_radicals070'
MANIFEST = OUT + '/input_manifest.json'
RESULTS = OUT + '/p1_radicals070_results.json'
LEDGER = OUT + '/budget_p1radicals070.json'
BPA = 1.0 / ex.ANG_PER_BOHR
S2_REF_DOUBLET = 0.75


def sha_arr(a):
    return hashlib.sha256(np.asarray(a, float).tobytes()).hexdigest()[:16]


# ======================= UKS project backend ==========================
class UKSBackend:
    """wb97xd + explicit D2 once, def2-TZVP, L8, grid_response, UKS
    doublet (spin=1), per-tag chkfile, kernel counting, <S²> record."""

    def __init__(self, elements):
        from pyscf import gto
        import d2_full
        self._gto = gto
        self._d2 = d2_full
        self.elements = list(elements)

    def make_mol(self, R_bohr):
        atom = [(e, (float(r[0]), float(r[1]), float(r[2])))
                for e, r in zip(self.elements,
                                np.asarray(R_bohr, float).reshape(-1, 3))]
        return self._gto.M(atom=atom, basis='def2-TZVP', charge=0,
                           spin=1, verbose=0, max_memory=4000,
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
        s2, mult = mf.spin_square()
        e_d2 = float(self._d2.d2_energy(mol))
        cfg = dict(xc='wb97xd (project -D2 attached)',
                   d2_attached=bool(getattr(mf, '_has_full_d2', False)),
                   grid_level=int(getattr(mf.grids, 'level', -1)),
                   scf_tol=[float(mf.conv_tol),
                            float(mf.conv_tol_grad)],
                   basis='def2-TZVP', charge=0, spin=1, uks=True,
                   grid_response=True, solvent='none (gas phase)')
        return dict(
            e_total=e, e_d2_Eh=e_d2, e_dft_part_Eh=e - e_d2,
            grad=g.tolist(), grad_max=float(np.abs(g).max()),
            grad_sha=sha_arr(g),
            coords_actual_angstrom=(np.asarray(R_bohr, float)
                                    .reshape(-1, 3)
                                    * ex.ANG_PER_BOHR).tolist(),
            coords_sha=sha_arr(np.asarray(R_bohr, float)),
            config=cfg, converged=conv,
            all_finite=bool(np.isfinite(e) and np.isfinite(g).all()),
            scf_kernel_count=int(getattr(mf,
                                         '_exec_scf_kernel_count', -1)),
            s2_total=float(s2), mult_measured=float(mult),
            s2_reference_doublet=S2_REF_DOUBLET,
            s2_delta=float(s2) - S2_REF_DOUBLET)


# ====================== P1 fragment reference =========================
FALLBACK_HA2KCAL = 627.5094740631   # 1 Eh in kcal/mol (CODATA-derived;
#   matches pyscf nist HARTREE2J*AVOGADRO/4184 to <1e-9 rel); used only
#   when pyscf is not importable (mock tests)


def p1_fragment_reference(mol_out, e_monomer_sum):
    """Separated-fragment electronic energy ONLY if BOTH radicals
    passed their rechecks; otherwise NO incomplete P1 energy."""
    both = all(mol_out.get(sp, {}).get('recheck_pass')
               for sp in ('HOO', 'H2NO'))
    frag = dict(p1_fragments_complete=bool(both),
                e_fragments=None, dE_P1frag=None,
                dE_P1frag_kcal_mol=None, e_monomer_sum=e_monomer_sum,
                unit_source=None)
    if not both:
        return frag
    e_f = sum(float(mol_out[sp]['e_recheck'])
              for sp in ('HOO', 'H2NO'))
    dE = e_f - e_monomer_sum
    try:
        from pyscf.data import nist as _nist
        ha2kcal = (_nist.HARTREE2J * _nist.AVOGADRO) / 4184.0
        frag['unit_source'] = 'pyscf nist HARTREE2J*AVOGADRO/4184'
    except Exception:
        ha2kcal = FALLBACK_HA2KCAL
        frag['unit_source'] = 'fallback constant %r' % FALLBACK_HA2KCAL
    frag.update(e_fragments=e_f, dE_P1frag=dE,
                dE_P1frag_kcal_mol=dE * ha2kcal,
                interpretation='project-method ELECTRONIC energy of '
                               'the SEPARATED fragments relative to '
                               'the JOB-026 monomer anchors; '
                               'QUALITATIVE comparison with the S13 '
                               'P1(2) values only (different method, '
                               'different energy definition); NOT '
                               'called a P1 overall reaction energy; '
                               'no ZPE/thermal/CP')
    return frag


# ============================== main ==================================
def main():
    man = json.load(open(MANIFEST))
    if not man['number_check']['pass_']:
        raise RuntimeError('HARD STOP: number check failed')
    if man['method']['spin'] != 1 or man['method']['uks'] is not True:
        raise RuntimeError('HARD STOP: manifest must pin spin=1 UKS')
    print('[070] P1(2) radical endpoints (project method model; UKS '
          'doublets; NOT an S13 reproduction)', flush=True)

    results = dict(job=man['job'], model_status=man['model_status'],
                   method=man['method'],
                   electron_state=man['electron_state'],
                   stop_trigger=man['stop_trigger'],
                   p1_fragment_energy=man['p1_fragment_energy'])
    ledger = ex.Ledger(LEDGER, caps=man['caps'])
    mol_out = {}
    status = 'aborted'
    try:
        for sp in ('HOO', 'H2NO'):
            key = sp.lower()
            cat_opt = '%s_opt' % key
            cat_rc = '%s_recheck' % key
            cat_st = '%s_stab' % key
            elems = man['molecules'][sp]['elements']
            x0 = np.asarray(man['molecules'][sp]['coords_start_bohr'],
                            float).reshape(-1)
            natoms = man['molecules'][sp]['natoms']
            bnd = UKSBackend(elems)
            mdir = os.path.join(OUT, key)
            os.makedirs(mdir, exist_ok=True)

            # ---- optimization (first evaluation counts in the cap) --
            flow = j68.run_opt_general(x0, natoms, mdir, ledger, bnd,
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
                mdir, 'accepted_iterates_%s.json' % key),
                dict(n=len(flow['accepted_iterates']),
                     iterates=flow['accepted_iterates']))
            print('[070] %s opt: %s (%d evals)'
                  % (sp, flow['outcome'], flow['n_new_evals']),
                  flush=True)

            passed = False
            if flow['meeting_points']:
                mp = flow['meeting_points'][-1]
                mrec = json.load(open('%s/eval_%s_%s.json'
                                      % (mdir, cat_opt, mp['tag'])))
                R_chk = np.asarray(mrec['coords_actual_angstrom'],
                                   float) * BPA
                rec_r = ex.eval_point('recheck_%s' % key, R_chk, mdir,
                                      ledger, bnd, cat_rc,
                                      gate_fn=ex.centre_gate_fn(mrec))
                g = rec_r['gate']
                passed = bool(g['gates_pass'])
                mol_out[sp]['recheck'] = dict(
                    gate=g, e_total=rec_r['e_total'],
                    grad_max=rec_r['grad_max'],
                    s2_total=rec_r.get('s2_total'),
                    s2_delta=rec_r.get('s2_delta'), passed=passed)
                print('[070] %s recheck %s: dE=%.2e dgrad=%.2e dC=%.2e '
                      '<S2>=%.4f'
                      % (sp, 'PASS' if passed else 'FAIL', g['dE'],
                         g['dgrad_max'], g['dcoords_A'],
                         rec_r.get('s2_total', float('nan'))),
                      flush=True)
                if passed:
                    # internal stability on a FRESH mf CONVERGED at the
                    # accepted geometry (kernel INSIDE the stab attempt)
                    a_st = ledger.pre_eval(cat_st,
                                           dict(tag='stab_%s' % key))
                    try:
                        mol_r, mf_r = bnd.new_mf(
                            R_chk, mdir, 'stab_%s' % key)
                        mf_r.kernel()
                        s2s, mults = mf_r.spin_square()
                        stab = j68.stability_wrapper(mf_r, mdir, key)
                        rec_s = dict(
                            stable_i=stab['stable_i'],
                            stable_e=stab['stable_e'],
                            raw_log=stab['raw_log'],
                            installed_order=stab['installed_order'],
                            geometry='the PASSING recheck geometry',
                            object='fresh mf CONVERGED at that '
                                   'geometry (JOB-068 flow-anomaly '
                                   'fix)',
                            s2_total=float(s2s),
                            mult_measured=float(mults),
                            s2_reference_doublet=S2_REF_DOUBLET,
                            s2_delta=float(s2s) - S2_REF_DOUBLET)
                        ex.save_json_atomic(os.path.join(
                            mdir, 'stability_%s_record.json' % key),
                            rec_s)
                        ledger.post_eval(a_st, dict(
                            stable_i=stab['stable_i'],
                            stable_e=stab['stable_e'],
                            s2_total=float(s2s)))
                    except Exception as e:
                        ledger.fail(a_st, e)
                        raise
                    mol_out[sp]['stability'] = rec_s
                    print('[070] %s stability: stable_i=%s <S2>=%.4f'
                          % (sp, stab['stable_i'], s2s), flush=True)
            else:
                print('[070] %s: no point met max|g|<=1e-5 within the '
                      'cap -> not converged, recorded truthfully; the '
                      'other radical continues' % sp, flush=True)
            mol_out[sp]['recheck_pass'] = passed

        status = 'completed'

        # ---- per-radical accepted energies + <S2> summary ----------
        for sp in ('HOO', 'H2NO'):
            m = mol_out.get(sp, {})
            if m.get('recheck', {}).get('passed'):
                m['e_recheck'] = m['recheck']['e_total']

        both = all(mol_out.get(sp, {}).get('recheck_pass')
                   for sp in ('HOO', 'H2NO'))
        frag = p1_fragment_reference(
            mol_out, man['p1_fragment_energy']['e_monomer_sum'])
        results['fragment_reference'] = frag
        results['molecules'] = mol_out
        results['final'] = dict(
            status=status,
            registration=(
                'both P1(2) radicals passed independent recheck; '
                'project-method separated-fragment electronic energy '
                'computed for QUALITATIVE S13 comparison'
                if both else
                'NOT both radicals converged; only accepted radicals '
                'reported, NO incomplete P1 fragment energy'),
            accounting=dict(caps=man['caps'],
                            attempts_by_cat={c: ledger.count(c)
                                             for c in man['caps']}),
            scope_limits='project method model ONLY; not an S13 '
                         'reproduction; electronic energy only (no '
                         'ZPE/thermal/CP); not aqueous free energy; '
                         'not water-treatment efficiency; <S2> '
                         'recorded but spin contamination is NOT '
                         'proof of a clean doublet; stable_i does '
                         'not guarantee reaction-path reliability; '
                         'NO TS1/TS2/CP1 construction')
    except Exception as exc:                                # noqa: BLE001
        results['abort'] = dict(error=str(exc)[:500],
                                traceback=traceback.format_exc()[-2000:])
        status = 'aborted'
        print('[070] ABORT: %s' % str(exc)[:200], flush=True)
    results['final_status'] = status
    results['finished'] = time.strftime('%F %T')
    results['budget'] = ledger.data
    ex.save_json_atomic(RESULTS, results)
    print('[070] DONE: %s' % status, flush=True)


if __name__ == '__main__':
    main()
