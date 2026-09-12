#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-058 resume-01: FIXED stability wrapper + centre rebuild +
internal stability + the preregistered 42 displacements.

Fix vs the original 058 run: mf.stability(internal=True, external=False,
return_status=True) returns the 4-tuple (mo_i, mo_e, stable_i, stable_e)
(installed PySCF order, verified in 052); the original wrapper unpacked 3.
The verbose log is written DIRECTLY to a file handle (not StringIO) and
the raw return (status + reprs, then mo arrays) is saved BEFORE any
summarising, so a contract error or a post-processing exception can no
longer lose the stability evidence (052 handling conventions).

Budget (NEW authorization, notes/job058_resume_authorization_2026-09-10.md;
NOT a reuse/reset of the original): centre 1, stability 1 (separate),
fd 42 -> new SCF+gradient cap 43; cumulative with the original batch:
SCF+gradient <= 44, stability <= 2 (incl. the original failed call),
cross-category attempts <= 46.  Failures count; no borrowing; the original
directory stays untouched; ON ANY NEW EXCEPTION: save and STOP - no
fix-and-continue, no interpolation of missing points.
"""
import os, sys, json, time, traceback
import numpy as np

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
HERE = ROOT + '/protocols'
sys.path.insert(0, HERE)
import job047_exec as ex
import job058_exec as j58          # original module: build_hessian /
import job052_exec as j52          #   postprocess reused UNCHANGED; Ledger
import job044_fix_postprocess as pp   #   pattern reused

S58 = ROOT + '/run_artifacts/02_nh3o3_reference/c1_fdhess058'
OUT = ROOT + '/run_artifacts/02_nh3o3_reference/c1_fdhess058_resume01'
LEDGER = OUT + '/budget_resume01.json'
RESULTS = OUT + '/resume01_results.json'
ORIG_LEDGER = S58 + '/budget_fdhess058.json'
ORIG_RESULTS = S58 + '/fdhess058_results.json'
ORIG_CENTRE = S58 + '/eval_centre.json'
ORIG_MANIFEST = S58 + '/input_manifest.json'
EVHASH = OUT + '/original_evidence_hashes.json'
SYMS = ['N', 'H', 'H', 'H', 'O', 'O', 'O']
H_STEP = 0.001


class Ledger(j58.Ledger):
    """Same rules as the original 058 ledger (error attempt on load ->
    HARD STOP, failures count, caps per category) with the NEW resume
    caps.  Prior-batch attempts are referenced read-only in 'prior_batch'
    (never merged into the attempt numbering)."""
    CAPS = dict(centre=1, stability=1, fd=42)

    def __init__(self, path):
        j58.Ledger.__init__(self, path)
        self.data['caps'] = dict(self.CAPS)
        self.data['note'] = ('resume-01 NEW budget: centre 1, stability 1 '
                             '(separate), fd 42 (SCF+gradient new cap 43); '
                             'cumulative incl. the original batch: '
                             'SCF+gradient <= 44, stability <= 2, attempts '
                             '<= 46; failures count; no borrowing; on '
                             'exception or non-convergence: save and STOP')
        self.data['prior_batch'] = dict(
            ledger='original budget_fdhess058.json (read-only, error '
                   'preserved)',
            attempts_summary=[('centre', 'done'), ('stability', 'error')],
            scf_grad_done=1, stability_calls=1)
        self._save()


def stability_fn_fixed(mf, out_dir):
    """FIXED internal-stability wrapper (052 conventions).

    - the verbose log goes DIRECTLY to a file handle; restore/close in
      finally -> the log survives any later exception;
    - the raw return is saved (status JSON first, then mo arrays) BEFORE
      any summarising;
    - contract check: exactly 4 items (mo_i, mo_e, stable_i, stable_e);
      a wrong contract raises AFTER the raw evidence is on disk.
    """
    t0 = time.time()
    log_path = os.path.join(out_dir, 'stability_raw_log.txt')
    raw_path = os.path.join(out_dir, 'stability_raw_result.json')
    mo_i_path = os.path.join(out_dir, 'stability_mo_i.npy')
    mo_e_path = os.path.join(out_dir, 'stability_mo_e.npy')
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
    raw = dict(
        call='mf.stability(internal=True, external=False, '
             'return_status=True)',
        installed_return_order='mo_i, mo_e, stable_i, stable_e',
        contract_expected='4-tuple (mo_i, mo_e, stable_i, stable_e)',
        raw_log='stability_raw_log.txt',
        provenance='the wrapped mf is the centre-rebuild converged object '
                   '(centre_scf.chk + centre_mo_*.npy in this directory); '
                   'log written directly to file during the call',
        seconds=round(time.time() - t0, 1))
    if not isinstance(st, tuple) or len(st) != 4:
        raw.update(contract_ok=False,
                   n_returned=(len(st) if hasattr(st, '__len__')
                               else None),
                   raw_return_repr=[repr(st)[:200]])
        ex.save_json_atomic(raw_path, raw)
        raise RuntimeError(
            'stability return contract violation: expected 4-tuple '
            '(mo_i, mo_e, stable_i, stable_e), got %r with %s items '
            '(raw evidence saved: %s)'
            % (type(st).__name__,
               raw['n_returned'], os.path.basename(raw_path)))
    mo_i, mo_e, stable_i, stable_e = st
    raw.update(contract_ok=True,
               raw_return_repr=[repr(x)[:200] for x in st],
               stable_i_raw=repr(stable_i), stable_e_raw=repr(stable_e))
    ex.save_json_atomic(raw_path, raw)          # status BEFORE array saves
    for val, p in ((mo_i, mo_i_path), (mo_e, mo_e_path)):
        if val is not None:
            np.save(p, np.asarray(val))
    return dict(
        stable_i=bool(stable_i) if stable_i is not None else None,
        stable_e=None if stable_e is None else bool(stable_e),
        stable_i_raw=repr(stable_i),
        stable_e_note='external=False -> None means NOT checked',
        installed_return_order=raw['installed_return_order'],
        raw_result='stability_raw_result.json',
        raw_log='stability_raw_log.txt',
        mo_i_file=(os.path.basename(mo_i_path)
                   if mo_i is not None else None),
        mo_e_file=(os.path.basename(mo_e_path)
                   if mo_e is not None else None),
        seconds=raw['seconds'],
        scope='RKS internal ORBITAL stability of the centre-rebuild '
              'object; does NOT exclude multireference character')


def run_resume(man, led, eval_fn, out_dir, stability_fn,
               orig_centre_rec, checkpoint_rec, checkpoint_hook=None):
    """Actual execution entry (mirrors the original run_fdhess flow; the
    42-point loop, gate arithmetic and Hessian/post-processing handover
    are the ORIGINAL 058 code paths via j58).  eval_fn contract:
    (rec, readback, mf).  Returns a SINGLE DICT."""
    R0 = np.asarray(man['centre']['x_bohr'], float).reshape(7, 3)

    # ---- 1) centre rebuild (SCF + full gradient, 1 allowed) -------------
    att = led.pre('centre', dict(tag='centre_rebuild',
                                 checkpoint=checkpoint_rec))
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

    # ---- 2) centre gate vs BOTH the original 058 centre AND the 057 ----
    #      designated recheck record (manifest centre block)
    def gate_vs(ref_e, ref_grad, ref_coords, ref_cfg, label):
        dE = abs(centre_rec['e_total'] - float(ref_e))
        dg = float(np.abs(np.asarray(centre_rec['grad'], float).reshape(-1)
                          - np.asarray(ref_grad, float).reshape(-1)).max())
        dC = float(np.abs(np.asarray(centre_rec['coords_actual_angstrom'],
                                     float)
                          - np.asarray(ref_coords, float)).max())
        g = dict(dE=dE, dgrad_max=dg, dcoords_A=dC,
                 gmax=float(centre_rec['grad_max']),
                 gmax_le_1e5=bool(centre_rec['grad_max'] <= 1e-5),
                 converged=bool(centre_rec['converged']),
                 finite=bool(centre_rec['all_finite']),
                 config_match=bool(centre_rec['config'] == ref_cfg))
        g['gates_pass'] = bool(dE <= 1e-8 and dg <= 1e-7 and dC <= 1e-9
                               and g['gmax_le_1e5'] and g['converged']
                               and g['finite'] and g['config_match'])
        g['reference'] = label
        return g

    gate58 = gate_vs(orig_centre_rec['e_total'], orig_centre_rec['grad'],
                     orig_centre_rec['coords_actual_angstrom'],
                     orig_centre_rec['config'], '058 original eval_centre')
    gate57 = gate_vs(man['centre']['e_total'], man['centre']['grad'],
                     man['centre']['coords_actual_angstrom'],
                     man['centre']['config'],
                     '057 designated recheck (manifest centre block)')
    gate = dict(vs_058_original=gate58, vs_057_recheck=gate57,
                gates_pass=bool(gate58['gates_pass']
                                and gate57['gates_pass']))
    print('[058r] centre gate 058orig %s: dE=%.2e dgrad=%.2e dC=%.2e '
          'gmax=%.3e' % ('PASS' if gate58['gates_pass'] else 'FAIL',
                         gate58['dE'], gate58['dgrad_max'],
                         gate58['dcoords_A'], gate58['gmax']), flush=True)
    print('[058r] centre gate 057recheck %s: dE=%.2e dgrad=%.2e dC=%.2e'
          % ('PASS' if gate57['gates_pass'] else 'FAIL', gate57['dE'],
             gate57['dgrad_max'], gate57['dcoords_A']), flush=True)
    if not gate['gates_pass']:
        raise RuntimeError('HARD STOP: centre gate failed')

    # wavefunction checkpoint provenance saved BEFORE the stability call
    wavefunction = checkpoint_hook(mf) if checkpoint_hook is not None \
        else None

    # ---- 3) RKS internal stability on the rebuilt object (separate cap) -
    stability = None
    if stability_fn is not None:
        att = led.pre('stability', dict(
            call='stability(internal=True, external=False, '
                 'return_status=True)',
            contract='(mo_i, mo_e, stable_i, stable_e)'))
        try:
            stability = stability_fn(mf, out_dir)
            led.post(att, dict(stable_i=stability.get('stable_i'),
                               raw=stability.get('stable_i_raw'),
                               seconds=stability.get('seconds')))
        except Exception as e:                          # noqa: BLE001
            led.fail(att, e)
            raise
        if stability.get('stable_i') is not True:
            return dict(centre_record=centre_rec, gate=gate,
                        stability=stability, fd_records=None,
                        hessian=dict(
                            status='stopped_stability_not_explicit_true'),
                        stopped=dict(
                            rule='only an explicit stable_i is True opens '
                                 'the 42-point stage; False/None/interface '
                                 'error -> save and STOP (no orbital '
                                 'following, no UKS switch)'))

    # ---- 4) the 42 preregistered displacements (original order/h) -------
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
            print('[058r] fd %s: E=%.9f gmax=%.3e'
                  % (tag, rec['e_total'], rec['grad_max']), flush=True)
        except Exception as e:                              # noqa: BLE001
            led.fail(att, e)
            raise

    hess = j58.build_hessian(recs, H_STEP, out_dir)
    return dict(centre_record=centre_rec, gate=gate, stability=stability,
                wavefunction=wavefunction, fd_records=recs, hessian=hess)


def production_eval(R, out_dir, tag):
    """Identical to the original 058 production_eval (052 endpoint_eval on
    a fresh mol), plus a wavefunction checkpoint: d2_full.make_mf_d2 is
    wrapped AT RUNTIME so mf.chkfile points into the resume directory
    (write-only provenance; it does not touch the SCF math)."""
    from pyscf import gto
    Rm = np.asarray(R, float).reshape(7, 3)
    mol = gto.M(atom=[(s, (float(r[0]), float(r[1]), float(r[2])))
                      for s, r in zip(SYMS, Rm)],
                basis='def2-TZVP', charge=0, spin=0, verbose=0,
                max_memory=4000, unit='Bohr')
    chk = os.path.join(out_dir, 'centre_scf.chk')
    orig_make = j52.d2_full.make_mf_d2

    def make_with_chk(m, solvent=None):
        _mf = orig_make(m, solvent=solvent)
        _mf.chkfile = chk
        return _mf

    j52.d2_full.make_mf_d2 = make_with_chk
    try:
        rec, mf, _ = j52.endpoint_eval(mol, out_dir, tag)
    finally:
        j52.d2_full.make_mf_d2 = orig_make
    rb = float(np.abs(np.asarray(mol.atom_coords(unit='Bohr'), float)
                      .reshape(-1) - Rm.reshape(-1)).max())
    return rec, rb, mf


def checkpoint_provenance(mf):
    """Traceable wavefunction record for the centre object (saved BEFORE
    the stability call)."""
    mo = np.asarray(mf.mo_coeff)
    p = dict(
        chkfile='centre_scf.chk',
        chkfile_note='PySCF SCF checkpoint written by kernel() (MOs, '
                     'densities); sha256 of the final file is registered '
                     'in the results after the run',
        mo_coeff_npy='centre_mo_coeff.npy',
        mo_energy_npy='centre_mo_energy.npy',
        mo_occ_npy='centre_mo_occ.npy',
        e_total=float(mf.e_tot), converged=bool(mf.converged),
        mo_shape=list(mo.shape), nao=int(mo.shape[0]),
        nmo=int(mo.shape[1]),
        note='saved so the stability object is traceable/restartable '
             'without re-running the SCF (the original 058 run had no '
             'checkpoint)')
    np.save(os.path.join(OUT, 'centre_mo_coeff.npy'), mo)
    np.save(os.path.join(OUT, 'centre_mo_energy.npy'),
            np.asarray(mf.mo_energy))
    np.save(os.path.join(OUT, 'centre_mo_occ.npy'), np.asarray(mf.mo_occ))
    return p


def sha_file(path):
    h = __import__('hashlib').sha256()
    with open(path, 'rb') as fh:
        for blk in iter(lambda: fh.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def verify_original_scene(ev):
    """The original scene must be byte-identical to the prep-time
    evidence (read-only guarantee)."""
    bad = []
    for rel, meta in ev['files'].items():
        p = os.path.join(ROOT, rel) if '/' in rel \
            else os.path.join(S58, rel)
        if not os.path.exists(p) or sha_file(p) != meta['sha256']:
            bad.append(rel)
    if bad:
        raise RuntimeError('HARD STOP: original 058 scene changed since '
                           'prep: %s' % bad)


def main():
    man = json.load(open(ORIG_MANIFEST))       # read-only reuse, hash-pinned
    ev = json.load(open(EVHASH))
    verify_original_scene(ev)
    if sha_file(man['centre']['path_wsl']) != man['centre']['sha256']:
        raise RuntimeError('HARD STOP: 057 recheck source hash changed')
    if not man['all_checks_pass']:
        raise RuntimeError('HARD STOP: prechecks failed')
    if sha_file(ORIG_MANIFEST) != ev['files']['input_manifest.json'][
            'sha256']:
        raise RuntimeError('HARD STOP: original manifest hash mismatch')
    orig_centre = json.load(open(ORIG_CENTRE))
    orig_led = json.load(open(ORIG_LEDGER))
    orig_counts = {c: sum(1 for a in orig_led['attempts']
                          if a['category'] == c)
                   for c in ('centre', 'stability', 'fd')}
    if orig_counts != dict(centre=1, stability=1, fd=0):
        raise RuntimeError('HARD STOP: original ledger counts changed: %s'
                           % orig_counts)
    print('[058r] original scene verified (hashes intact, error preserved, '
          'fd=0); original manifest reused read-only', flush=True)

    results = dict(
        job='JOB-2026-0906-058 resume-01: stability 4-tuple interface fix '
            '+ centre rebuild + internal stability + the preregistered 42 '
            'displacements',
        doing='centre rebuild SCF+full gradient (1) gated against the 058 '
              'original centre AND the 057 designated recheck; RKS '
              'internal stability with the FIXED 4-tuple wrapper (log to '
              'file in finally, raw saved before summarising); only an '
              'explicit stable_i True opens the 42 central-difference '
              'displacements (h=0.001 Bohr) building the full 21x21 '
              'Cartesian Hessian; nominal masses; single step size '
              '(sensitivity NOT verified)',
        authorization='notes/job058_resume_authorization_2026-09-10.md',
        original_scene=dict(dir=S58,
                            evidence='original_evidence_hashes.json',
                            original_counts=orig_counts,
                            original_status='aborted (stability unpack '
                                            'error preserved, fd=0)'),
        centre_ref=dict(e_total=man['centre']['e_total'],
                        grad_max=man['centre']['grad_max']))
    led = Ledger(LEDGER)
    status = 'aborted'
    try:
        # centre rebuild first (no stability yet), to save the checkpoint
        # record before the stability call the flow needs mf -> the full
        # flow below reruns nothing: run_resume performs centre, gate,
        # stability, 42x fd, Hessian in one pass; the checkpoint record
        # is captured via the eval_fn side effect files + mf passed to
        # stability_fn; provenance JSON is written inside production path
        # by run_resume wrapper below.
        flow = run_resume(man, led, production_eval, OUT,
                          stability_fn=stability_fn_fixed,
                          orig_centre_rec=orig_centre,
                          checkpoint_rec=dict(
                              chkfile='centre_scf.chk',
                              note='written during the centre rebuild by '
                                   'the runtime chkfile wrapper; final '
                                   'sha256 registered after the run'),
                          checkpoint_hook=checkpoint_provenance)
        wf = flow.get('wavefunction') or {}
        chk_p = os.path.join(OUT, 'centre_scf.chk')
        if os.path.exists(chk_p):
            wf['chkfile_sha256_after_run'] = sha_file(chk_p)
        results['centre'] = dict(record=flow['centre_record'],
                                 gate=flow['gate'],
                                 wavefunction=wf)
        results['stability'] = flow['stability']
        if flow.get('stopped') is not None:
            status = ('stopped_stability_not_explicit_true'
                      if flow['hessian'].get(
                          'status') == 'stopped_stability_not_explicit_true'
                      else 'stopped')
            results['hessian'] = flow['hessian']
            results['final'] = dict(status=status, budget=led.data,
                                    stopped=flow['stopped'],
                                    note='stability result saved with raw '
                                         'evidence; 42-point stage NOT '
                                         'opened')
            ex.save_json_atomic(RESULTS, results)
            print('[058r] DONE: %s' % status, flush=True)
            return
        results['fd_records'] = flow['fd_records']
        hess = flow['hessian']
        results['hessian'] = {k: v for k, v in hess.items()
                              if k not in ('H_raw', 'H_sym')}
        ex.save_json_atomic(RESULTS, results)
        if hess.get('status') != 'ok':
            status = 'hessian_incomplete'
            results['final'] = dict(status=status, budget=led.data,
                                    note='missing displacements are NOT '
                                         'interpolated; NO mode analysis')
            ex.save_json_atomic(RESULTS, results)
            print('[058r] DONE: %s' % status, flush=True)
            return
        modes = j58.postprocess(man, hess, flow['centre_record'], OUT)
        results['modes'] = modes
        if modes.get('status') == 'ok':
            print('[058r] TR rank %d / internal %d / negative modes %d'
                  % (modes['tr_rank_svd'], modes['n_internal_modes'],
                     modes['n_negative']), flush=True)
            print('[058r] frequencies (cm-1): %s'
                  % ['%.2f' % x for x in modes['frequencies_cm1']],
                  flush=True)
        status = 'completed'
        new_scfg = led.count('centre') + led.count('fd')
        results['final'] = dict(
            status=status,
            registration=(
                'this single-step FD matrix predicts negative internal '
                'curvature' if modes.get('n_negative', 0) > 0 else
                'this single-step FD matrix shows no internal negative '
                'curvature'),
            accounting=dict(
                new_attempts=dict(centre=led.count('centre'),
                                  stability=led.count('stability'),
                                  fd=led.count('fd'),
                                  scf_grad_new=new_scfg),
                new_caps=dict(centre=1, stability=1, fd=42),
                cumulative=dict(
                    scf_grad_total_max=1 + new_scfg,
                    scf_grad_cap=44,
                    stability_total=1 + led.count('stability'),
                    stability_cap=2,
                    attempts_total=1 + 1 + led.count('centre')
                    + led.count('stability') + led.count('fd'),
                    attempts_cap=46,
                    note='original: centre 1 done + stability 1 error; '
                         'failures counted; no borrowing; fd total 42'),
                new_failures=sum(1 for a in led.data['attempts']
                                 if a['status'] == 'error')),
            limits=man['analysis_limits'])
        ex.save_json_atomic(RESULTS, results)
        print('[058r] DONE: %s' % results['final']['registration'],
              flush=True)
    except Exception as exc:                                # noqa: BLE001
        results['abort'] = dict(error=str(exc),
                                traceback=traceback.format_exc()[-1500:])
        results.setdefault('final', dict(status='aborted',
                                         budget=led.data))
        ex.save_json_atomic(RESULTS, results)
        print('[058r] ABORT: %s' % exc, flush=True)
        raise


if __name__ == '__main__':
    main()
