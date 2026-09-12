#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-046 step B: mock-backend tests in a temp directory.

Verifies, on the SAME evaluate_point/compare code path used by the real
run, that:
  T1  the real backend is unreachable in the test process;
  T2  the grid_response switch reaches the gradient object, readback
      works, and the two gradient objects are independent;
  T3  gradient kernels do NOT repeat the SCF; d2_grad does no SCF;
  T4  ledger per-category caps are enforced and all attempts close;
  T5  the comparison pipeline runs and reacts to a nonzero response term;
  T6  the formal output directory is untouched (sentinel snapshot).
"""
import os, sys, json, shutil, tempfile
import numpy as np

# ---- block real backends BEFORE importing the exec module ----
for _m in ('pyscf', 'd2_full', 'grad_factory'):
    sys.modules[_m] = None

HERE = '/mnt/e/dft-service/ozone_mnb_uv_simulation/protocols'
sys.path.insert(0, HERE)
import job046_exec as ex

FORMAL_OUT = ex.OUT


# ---------------------------------------------------------------- mock
class MockGrad:
    def __init__(self, mf):
        self.mf = mf
        self.grid_response = None        # must be set explicitly
        self._has_full_d2 = True

    def kernel(self):
        assert self.mf.kernel_calls >= 1, 'gradient evaluated before SCF'
        resp = getattr(self, 'grid_response', None)
        assert resp is not None, 'grid_response was not explicitly set'
        self.mf.grad_kernel_calls += 1
        base = (np.tile(np.array([1e-5, -2e-5, 3e-5]), 7).reshape(7, 3)
                * (1.0 + 0.1 * self.mf.bias))
        extra = (5e-6 * (1.0 + self.mf.bias)
                 * self.mf.q3) if resp else 0.0
        return base + extra


class MockMF:
    def __init__(self, bias, q3):
        self.kernel_calls = 0
        self.grad_kernel_calls = 0
        self.bias = bias
        self.q3 = q3
        self.converged = True
        self.grids = type('G', (), {'level': 8})()
        self._has_full_d2 = True
        self.conv_tol = 1e-12
        self.conv_tol_grad = 1e-9

    def kernel(self):
        self.kernel_calls += 1
        self.e_tot = -282.0 - 1e-9 * self.bias
        return self.e_tot

    def nuc_grad_method(self):
        return MockGrad(self)            # fresh object per call


class MockBackend:
    def __init__(self, q3):
        self.q3 = q3
        self.mfs = []
        self.d2_calls = 0

    def make_mol(self, C_A):
        self._coords = np.asarray(C_A, float)
        return object()

    def make_mf(self, mol, out_dir, tag):
        mf = MockMF(len(self.mfs), self.q3)
        self.mfs.append(mf)
        return mf

    def mol_coords_angstrom(self, mol):
        return self._coords

    def chkfile_info(self, mf):
        return dict(path='mock.chk', exists=True, size=0)

    def mo_summary(self, mf):
        return dict(n_mo=100, nocc=17, homo=-0.5, lumo=0.1, gap_Eh=0.6)

    def gradient(self, mf, grid_response):
        g = mf.nuc_grad_method()
        g.grid_response = bool(grid_response)
        gr = getattr(g, 'grid_response', None)
        if gr is not bool(grid_response):
            raise RuntimeError('switch did not reach gradient object')
        val = np.asarray(g.kernel(), float).reshape(7, 3)
        return val, dict(grid_response=gr, mock=True)

    def d2_grad(self, mol):
        self.d2_calls += 1
        return np.full((7, 3), 1e-7)

    def mf_kernel_count(self, mf):
        return mf.kernel_calls

    def cfg_readback_mf(self, mf):
        return dict(xc='mock', d2_attached=True, grid_level=8,
                    scf_tol=[1e-12, 1e-9], basis='mock', charge=0, spin=0,
                    grid_response='per-gradient', solvent='none')


def mock_grad_of(bias, resp, q3):
    base = (np.tile(np.array([1e-5, -2e-5, 3e-5]), 7).reshape(7, 3)
            * (1.0 + 0.1 * bias))
    return base + (5e-6 * (1.0 + bias) * q3 if resp else 0.0)


def snapshot(d):
    if not os.path.isdir(d):
        return []
    return sorted((f, os.path.getsize(os.path.join(d, f)))
                  for f in os.listdir(d))


def main():
    tmp = tempfile.mkdtemp(prefix='job046_mock_')
    R = {}
    pre_snap = snapshot(FORMAL_OUT)
    try:
        # T1: real backend unreachable (pyscf blocked via sys.modules)
        try:
            ex.RealBackend(tmp)
            R['T1_real_backend_blocked'] = False
        except Exception as e:  # noqa: BLE001
            R['T1_real_backend_blocked'] = True
            R['T1_error_type'] = type(e).__name__

        q = np.asarray(json.load(open(os.path.join(
            ex.S45, 'input_manifest.json')))['direction']['q_vector'])
        q3 = q.reshape(7, 3)
        C_ref = (np.arange(21, dtype=float).reshape(7, 3) * 0.05 + 1.0)
        refs = {tag: dict(e_total=-282.0 - 1e-9 * i,
                          grad=mock_grad_of(i, True, q3).tolist(),
                          coords_actual_angstrom=C_ref.tolist())
                for i, tag in enumerate(ex.TAGS)}

        ledger = ex.Ledger(os.path.join(tmp, 'budget_test.json'))
        backend = MockBackend(q3)
        res = {tag: ex.evaluate_point(tag, refs[tag], tmp, ledger, backend)
               for tag in ex.TAGS}

        # T2: switch reached + object independence
        g1, c1 = backend.gradient(backend.mfs[0], True)
        g2, c2 = backend.gradient(backend.mfs[0], False)
        g1b, c1b = backend.gradient(backend.mfs[0], True)
        R['T2_switch_reached'] = True    # gradient() asserts readback
        R['T2_on_still_true_after_off'] = bool(c1b['grid_response'] is True)
        R['T2_values_differ'] = bool(np.abs(g1 - g2).max() > 0)

        # T3: no repeated SCF; d2_grad does no SCF
        R['T3_scf_counts_per_mf'] = [mf.kernel_calls for mf in backend.mfs]
        R['T3_no_repeat_scf'] = bool(all(mf.kernel_calls == 1
                                         for mf in backend.mfs))
        R['T3_d2_calls'] = backend.d2_calls
        R['T3_d2_no_scf'] = bool(backend.d2_calls == 2
                                 and all(mf.kernel_calls == 1
                                         for mf in backend.mfs))

        # T4: ledger categories/caps
        cats = [a['category'] for a in ledger.data['attempts']]
        R['T4_categories'] = {c: cats.count(c) for c in sorted(set(cats))}
        R['T4_all_done'] = bool(all(a['status'] == 'done'
                                    for a in ledger.data['attempts']))
        try:
            ledger.pre_eval('scf', dict(note='over cap'))
            R['T4_cap_enforced'] = False
        except RuntimeError:
            R['T4_cap_enforced'] = True

        # T5: comparison pipeline reacts to a nonzero response term
        shutil.copy(os.path.join(FORMAL_OUT, 'prep_check_results.json'),
                    os.path.join(tmp, 'prep_check_results.json'))
        comp = ex.compare(res, q, tmp)
        R['T5_comparison_saved'] = os.path.exists(
            os.path.join(tmp, 'grid_response_comparison.json'))
        R['T5_dk_grid_nonzero'] = bool(abs(comp['dk_grid']) > 0)
        R['T5_dk_grid_mock_value'] = comp['dk_grid']

        # T6: formal dir untouched by the tested pipeline
        R['T6_formal_dir_untouched'] = bool(pre_snap == snapshot(FORMAL_OUT))
        R['temp_dir'] = tmp
        R['ALL_PASS'] = bool(
            R['T1_real_backend_blocked'] and R['T2_switch_reached']
            and R['T2_on_still_true_after_off'] and R['T2_values_differ']
            and R['T3_no_repeat_scf'] and R['T3_d2_no_scf']
            and R['T4_all_done'] and R['T4_cap_enforced']
            and R['T5_comparison_saved'] and R['T5_dk_grid_nonzero']
            and R['T6_formal_dir_untouched'])
    finally:
        pass
    print(json.dumps(R, indent=1, default=str))
    ex.save_json_atomic(os.path.join(
        FORMAL_OUT, 'mock_test_results.json'), R)
    if not R.get('ALL_PASS'):
        raise SystemExit('MOCK TESTS FAILED')
    print('MOCK TESTS ALL PASS', flush=True)


if __name__ == '__main__':
    main()
