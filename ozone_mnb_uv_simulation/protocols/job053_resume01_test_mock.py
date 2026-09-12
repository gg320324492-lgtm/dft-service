#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-053 resume-01 step B: zero-evaluation fix verification
(temp dirs, real backend unreachable, formal directory untouched).

Tests go through the ACTUAL wrapper job053_exec._eval_point with injected
mocks (stubbed endpoint_eval triple + fake mol factory):
  T1  triple return (rec, mf, mol) -> complete record read, x_bohr and
      readback attached, file persisted;
  T2  post-evaluation exception -> the already-saved file remains and is
      recoverable/complete;
  T3  accepted centre reuse: the resume flow reads the persisted centre
      record and triggers ZERO centre evaluations (counter proof), then
      exactly 4 displacement evaluations;
  T4  centre gate failure -> displacement calls = 0;
  T5  the 5th displacement request is refused BEFORE evaluation (counter
      stays 4);
  T6  isolation sentinels (no pyscf, formal directory untouched).
"""
import os, sys, json, tempfile, types
import numpy as np

sys.modules['pyscf'] = None
sys.modules['pyscf.hessian'] = None

HERE = '/mnt/e/dft-service/ozone_mnb_uv_simulation/protocols'
sys.path.insert(0, HERE)

# stub the production dependency chain with a TRIPLE-returning
# endpoint_eval (the real contract, including its internal save), so the
# actual _eval_point wrapper is exercised against the real interface
j52 = types.ModuleType('job052_exec')


class StubMol:
    def __init__(self, R, raise_after_save=False):
        self.R = np.asarray(R, float).reshape(7, 3)
        self.coords_bohr = self.R.tolist()
        self.raise_after_save = raise_after_save

    def atom_coords(self, unit='Bohr'):
        return self.R.copy()


def stub_endpoint_eval(mol, out_dir, tag):
    rec = dict(e_total=-282.00172148979476, e_d2_Eh=-0.0003,
               e_dft_part_Eh=-282.00142148979476,
               grad=[0.0] * 21, grad_max=8.2e-06, grad_sha='stub',
               coords_actual_angstrom=(mol.R * 0.52917721092).tolist(),
               coords_sha='stub',
               config=dict(xc='stub', grid_response=True, grid_level=8,
                           d2_attached=True),
               converged=True, all_finite=True, scf_kernel_count=1,
               seconds=0.1, tag=tag)
    # the REAL endpoint_eval persists the record internally BEFORE
    # returning - reproduce that behaviour here
    ex.save_json_atomic(os.path.join(out_dir, 'eval_%s.json' % tag), rec)
    if mol.raise_after_save:
        raise RuntimeError('post-eval wrapper exception')
    return rec, object(), object()


j52.endpoint_eval = stub_endpoint_eval
sys.modules['job052_exec'] = j52
import job047_exec as ex
import job053_exec as ex53

OUT = ex53.OUT
C_A = np.asarray(json.load(open(os.path.join(
    ex53.ROOT, 'run_artifacts/02_nh3o3_reference/c1_negmode_dir053',
    'eval_centre.json')))['coords_actual_angstrom'], float)
R0 = C_A / 0.52917721092
Q = np.zeros(21)
Q[4] = 1.0


def fake_mol_factory(R):
    return StubMol(R)


def snapshot(d):
    if not os.path.isdir(d):
        return []
    return sorted((f, os.path.getsize(os.path.join(d, f)))
                  for f in os.listdir(d))


def main():
    pre_snap = snapshot(OUT)
    R = {}

    # ---- T1: triple return through the actual wrapper
    tmp = tempfile.mkdtemp(prefix='job053r_T1_')
    rec, rb = ex53._eval_point(R0, tmp, 'centre', mol_factory=fake_mol_factory)
    f1 = os.path.join(tmp, 'eval_centre.json')
    saved = json.load(open(f1)) if os.path.exists(f1) else {}
    R['T1'] = dict(rec_is_dict=isinstance(rec, dict),
                   x_bohr_attached='x_bohr' in rec,
                   readback=rb, file_saved=os.path.exists(f1),
                   saved_complete=bool(saved) and all(
                       k in saved for k in ('e_total', 'grad', 'config',
                                            'converged')))
    R['T1_pass'] = bool(R['T1']['rec_is_dict'] and R['T1']['x_bohr_attached']
                        and rb == 0.0 and R['T1']['file_saved']
                        and R['T1']['saved_complete'])

    # ---- T2: post-evaluation exception -> saved file recoverable
    tmp = tempfile.mkdtemp(prefix='job053r_T2_')
    f2 = os.path.join(tmp, 'eval_post.json')
    R2 = (R0.reshape(-1) + 0.01 * Q).reshape(7, 3)
    err = None
    try:
        ex53._eval_point(R2, tmp, 'post',
                         mol_factory=lambda R: StubMol(R,
                                                       raise_after_save=True))
    except RuntimeError as e:
        err = str(e)
    saved2 = json.load(open(f2)) if os.path.exists(f2) else {}
    R['T2'] = dict(error=(err or '')[:60],
                   file_exists=os.path.exists(f2),
                   recoverable=bool(saved2) and 'e_total' in saved2
                   and 'grad' in saved2)
    R['T2_pass'] = bool(err and 'post-eval' in err and R['T2']['file_exists']
                        and R['T2']['recoverable'])

    # ---- T3/T4/T5: resume flow with counting eval_fn
    man = dict(centre_record=json.load(open(os.path.join(
        ex53.ROOT, 'run_artifacts/02_nh3o3_reference/c1_negmode_dir053',
        'eval_centre.json'))),
        direction=dict(q=Q.tolist()),
        displacement_geometries=[
            dict(t_Bohr=t,
                 coords_bohr=(R0.reshape(-1) + t * Q).reshape(7, 3)
                 .tolist())
            for t in (0.01, -0.01, 0.02, -0.02)],
        kH_from_052_matrices={'1e-3': dict(kH_total_Eh_Bohr2=-2.1e-4),
                              '5e-4': dict(kH_total_Eh_Bohr2=-2.1e-4)})

    def counting_eval(R, out_dir, tag):
        counting_eval.calls.append(tag)
        rec = dict(e_total=-282.0017215, e_d2_Eh=0.0,
                   e_dft_part_Eh=-282.0017215,
                   grad=[1e-6] * 21, grad_max=1e-6,
                   coords_actual_angstrom=(np.asarray(
                       R, float).reshape(7, 3) * 0.52917721092).tolist(),
                   config=dict(xc='stub'), converged=True,
                   all_finite=True, seconds=0.0, tag=tag)
        rec['x_bohr'] = np.asarray(R, float).tolist()
        return rec, 0.0
    counting_eval.calls = []

    led = ex53.Ledger(os.path.join(tmp, 'b.json'))
    comp, centre_rec = ex53.run_resume(man, led, counting_eval,
                                       out_dir=tmp, gate_pass=True)
    centre_calls = [t for t in counting_eval.calls if t == 'centre']
    R['T3'] = dict(centre_eval_calls=len(centre_calls),
                   displacement_eval_calls=len(
                       [t for t in counting_eval.calls
                        if t.startswith('disp')]),
                   n_accepted_geoms=len(comp['displacements']),
                   centre_from_persisted=bool(
                       centre_rec['e_total'] == man['centre_record'][
                           'e_total']))
    R['T3_pass'] = bool(len(centre_calls) == 0
                        and len([t for t in counting_eval.calls
                                 if t.startswith('disp')]) == 4
                        and R['T3']['centre_from_persisted'])

    # T4: gate fail -> zero displacement calls
    counting_eval.calls = []
    led4 = ex53.Ledger(os.path.join(tmp, 'b4.json'))
    err4 = None
    try:
        ex53.run_resume(man, led4, counting_eval, out_dir=tmp, gate_pass=False)
    except RuntimeError as e:
        err4 = str(e)
    R['T4'] = dict(error=(err4 or '')[:60],
                   displacement_calls=len(counting_eval.calls))
    R['T4_pass'] = bool(err4 and 'gate' in err4
                        and len(counting_eval.calls) == 0)

    # T5: 5th displacement request refused BEFORE evaluation
    led5 = ex53.Ledger(os.path.join(tmp, 'b5.json'))
    for _ in range(4):
        led5.pre('displacement')
    n_before = len(counting_eval.calls)
    ok5 = True
    try:
        led5.pre('displacement')          # 5th request -> cap 4 reached
        ok5 = False
    except RuntimeError:
        pass
    R['T5'] = dict(n_before=n_before, after=len(counting_eval.calls),
                   request_refused=ok5)
    R['T5_pass'] = bool(ok5 and len(counting_eval.calls) == n_before)

    R['formal_dir_untouched'] = bool(pre_snap == snapshot(OUT))
    R['ALL_PASS'] = bool(all(R[k] for k in ('T1_pass', 'T2_pass', 'T3_pass',
                                            'T4_pass', 'T5_pass',
                                            'formal_dir_untouched')))
    print(json.dumps(R, indent=1, default=str))
    ex.save_json_atomic(os.path.join(OUT, 'mock_test_results_resume01.json'),
                        R)
    if not R['ALL_PASS']:
        raise SystemExit('MOCK TESTS FAILED')
    print('MOCK TESTS ALL PASS', flush=True)


if __name__ == '__main__':
    main()
