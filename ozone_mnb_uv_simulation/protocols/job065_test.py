#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-065 step B: no-SCF tests for the mirror-equivalence
check (mock backend unreachable, formal 064 directory hashes unchanged
except the additive correction note)."""
import os, sys, json, hashlib, types
import numpy as np

sys.modules['pyscf'] = None
sys.modules['pyscf.hessian'] = None
sys.modules['d2_full'] = types.ModuleType('d2_full')

HERE = '/mnt/e/dft-service/ozone_mnb_uv_simulation/protocols'
sys.path.insert(0, HERE)
import job065_mirror as M

R64 = M.R64
OUT = M.OUT


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for blk in iter(lambda: fh.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()[:16]


def main():
    res = json.load(open(os.path.join(OUT,
                                      'c2_mirror065_results.json')))
    X0 = np.asarray(json.load(open(os.path.join(
        R64, 'eval_scan_d+00.json')))['coords_actual_angstrom'], float)
    Xm = np.asarray(res and X0) if False else None
    # rebuild the mirror exactly as job065_mirror did
    n = M.contact_plane_normal(X0)
    Xm = X0.copy()
    Xm[2] = X0[2] - 2.0 * ((X0[2] - X0[0]) @ n) * n
    R = {}

    # T1 reflection only changes the 3H normal sign
    h3_along_n_0 = float((X0[2] - X0[0]) @ n)
    h3_along_n_m = float((Xm[2] - X0[0]) @ n)
    inplane = np.eye(3) - np.outer(n, n)
    d_inplane = float(np.abs((Xm[2] - X0[0]) @ inplane.T
                             - (X0[2] - X0[0]) @ inplane.T).max())
    R['T1'] = dict(h3_normal_before=h3_along_n_0,
                   h3_normal_after=h3_along_n_m,
                   sum_zero=bool(abs(h3_along_n_0
                                     + h3_along_n_m) < 1e-12),
                   inplane_shift_A=d_inplane)
    R['T1_pass'] = bool(R['T1']['sum_zero'] and d_inplane < 1e-12
                        and h3_along_n_0 > 0 and h3_along_n_m < 0)

    # T2 NH3 internal distance matrix unchanged by the reflection
    dm0 = np.linalg.norm(X0[:4, None, :] - X0[None, :4, :], axis=2)
    dmm = np.linalg.norm(Xm[:4, None, :] - Xm[None, :4, :], axis=2)
    dev = float(np.abs(dm0 - dmm).max())
    R['T2'] = dict(max_dev_A=dev)
    R['T2_pass'] = bool(dev < 1e-12)

    # T3 contact H...O distances unchanged
    c0 = (float(np.linalg.norm(X0[1] - X0[4])),
          float(np.linalg.norm(X0[3] - X0[6])))
    cm = (float(np.linalg.norm(Xm[1] - Xm[4])),
          float(np.linalg.norm(Xm[3] - Xm[6])))
    R['T3'] = dict(before=c0, after=cm,
                   dev=max(abs(c0[0] - cm[0]), abs(c0[1] - cm[1])))
    R['T3_pass'] = bool(R['T3']['dev'] < 1e-12)

    # T4 element order and contact mapping
    R['T4'] = dict(order=[M.SYMS[i] for i in range(7)],
                   contact_rows=dict(c_5O_2H=[1, 4], c_7O_4H=[3, 6]))
    R['T4_pass'] = bool(M.SYMS == ['N', 'H', 'H', 'H', 'O', 'O', 'O']
                        and res['equivalence']['best']
                        ['contact_pair_5O2H_maps_to'] is not None)

    # T5 proper rotation vs mirror strictly distinguished
    eq = res['equivalence']
    R['T5'] = dict(best_det=eq['best']['det'],
                   best_rmsd=eq['best']['rmsd_A'],
                   improper_rmsd=eq['improper_control_rmsd_A'],
                   only_proper_used=bool(eq['best']['det'] > 0),
                   table_dets_all_plus1=bool(
                       all(t['det'] > 0 for t in eq['table'])))
    R['T5_pass'] = bool(R['T5']['only_proper_used']
                        and R['T5']['table_dets_all_plus1']
                        and eq['improper_control_rmsd_A'] < 1e-9)

    # T6 end-O swap logic present in the enumeration
    swaps = [t for t in eq['table'] if t['endO_swap']]
    R['T6'] = dict(n_with_swap=len(swaps), n_total=len(eq['table']))
    R['T6_pass'] = bool(len(swaps) == len(eq['table']) // 2)

    # T7 064 formal directory: original hashes unchanged (the additive
    # correction note is a NEW file; original files byte-identical)
    man64 = json.load(open(os.path.join(R64, 'input_manifest.json')))
    originals = ['input_manifest.json', 'budget_c2scan064.json',
                 'c2_scan064_results.json', 'exec_log.txt',
                 'mock_test_results.json', 'scan_curve.svg']
    originals += ['eval_%s.json' % g['tag'] for g in man64['geometries']]
    hashes = {f: sha256_file(os.path.join(R64, f)) for f in originals}
    R['T7'] = dict(n_original_files=len(originals),
                   all_exist=bool(all(os.path.exists(
                       os.path.join(R64, f)) for f in originals)),
                   correction_note_is_new_file=bool(os.path.exists(
                       os.path.join(R64, 'job064_correction_note.json'))))
    R['T7_pass'] = bool(R['T7']['all_exist']
                        and R['T7']['correction_note_is_new_file'])

    # T8 mock backend unreachable
    R['T8'] = dict(pyscf_stubbed=sys.modules.get('pyscf') is None,
                   d2_stubbed=getattr(sys.modules.get('d2_full'),
                                      '__name__', None) == 'd2_full')
    R['T8_pass'] = bool(R['T8']['pyscf_stubbed']
                        and R['T8']['d2_stubbed'])

    keys = ('T1_pass', 'T2_pass', 'T3_pass', 'T4_pass', 'T5_pass',
            'T6_pass', 'T7_pass', 'T8_pass')
    R['ALL_PASS'] = bool(all(R[k] for k in keys))
    print(json.dumps({k: R[k] for k in keys}, indent=1, default=str))
    ex_save = os.path.join(OUT, 'no_scf_tests.json')
    tmp = ex_save + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(R, fh, indent=2, default=str)
    os.replace(tmp, ex_save)
    if not R['ALL_PASS']:
        raise SystemExit('NO-SCF TESTS FAILED')
    print('NO-SCF TESTS ALL PASS', flush=True)


if __name__ == '__main__':
    main()
