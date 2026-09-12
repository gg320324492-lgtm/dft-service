#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-066 step B: zero-evaluation tests (mock backend only,
064 formal directory untouched).

  T1  manifest: six mirror geometries, all negative-normal, caps=6;
  T2  scan flow on a mock backend: 6 attempts in delta order, per-point
      save before done, ledger scan=6;
  T3  SCF failure mid-scan -> error attempt + STOP, remaining points
      not evaluated;
  T4  mirror identification: every point's 3H is on the NEGATIVE
      normal and the in-plane atoms are identical to the 064 sources;
  T5  isolation: pyscf/d2_full stubbed; 064/065 formal dirs untouched.
"""
import os, sys, json, tempfile, types, glob
import numpy as np

sys.modules['pyscf'] = None
sys.modules['pyscf.hessian'] = None
sys.modules['d2_full'] = types.ModuleType('d2_full')

HERE = '/mnt/e/dft-service/ozone_mnb_uv_simulation/protocols'
sys.path.insert(0, HERE)
import job047_exec as ex
import job066_prep as P

OUT = P.OUT
R64 = P.R64
R65 = '/mnt/e/dft-service/ozone_mnb_uv_simulation/run_artifacts' \
      '/02_nh3o3_reference/c2_mirror065'


def snapshot(d):
    if not os.path.isdir(d):
        return []
    return sorted((f, os.path.getsize(os.path.join(d, f)))
                  for f in os.listdir(d))


def mock_eval_factory(fail_tags=()):
    def ev(R, out_dir, tag):
        R = np.asarray(R, float).reshape(7, 3)
        g = R * 0.01
        g[0, 1] += 0.03
        rec = dict(e_total=-281.9 + float(np.cos(R.sum()) * 1e-6),
                   e_d2_Eh=-1e-5, e_dft_part_Eh=-281.9,
                   grad=g.tolist(), grad_max=float(np.abs(g).max()),
                   grad_sha='m',
                   coords_actual_angstrom=(R * 0.52917721092).tolist(),
                   coords_sha='m',
                   config=dict(xc='wb97xd (project -D2 attached)',
                               d2_attached=True, grid_level=8,
                               scf_tol=[1e-12, 1e-9], basis='def2-TZVP',
                               charge=0, spin=0, grid_response=True,
                               solvent='none (gas phase)'),
                   converged=(tag not in fail_tags), all_finite=True,
                   scf_kernel_count=1, tag=tag)
        ex.save_json_atomic(os.path.join(out_dir,
                                         'eval_%s.json' % tag), rec)
        return rec
    return ev


def mock_scan(man, led, ev, out_dir):
    for g in man['geometries']:
        tag = g['tag']
        att = led.pre_eval('scan', dict(tag=tag, delta=g['delta_A']))
        try:
            R = np.asarray(g['coords_bohr'], float).reshape(7, 3)
            rec = ev(R, out_dir, tag)
            rec['delta_A'] = g['delta_A']
            ex.save_json_atomic(os.path.join(out_dir,
                                             'eval_%s.json' % tag), rec)
            if not (rec['converged'] and rec['all_finite']):
                raise RuntimeError('%s SCF not converged -> HARD STOP'
                                   % tag)
            led.post_eval(att, dict(e_total=rec['e_total'],
                                    grad_max=rec['grad_max']))
        except Exception as e:                          # noqa: BLE001
            led.fail(att, e)
            raise
    return True


def main():
    pre64 = snapshot(R64)
    pre65 = snapshot(R65)
    pre66 = snapshot(OUT)
    man = json.load(open(os.path.join(OUT, 'input_manifest.json')))
    R = {}

    # T1 manifest
    R['T1'] = dict(n=len(man['geometries']),
                   all_neg=bool(all(g['negative_normal']
                                    for g in man['geometries'])),
                   caps=man['caps']['scan'],
                   deltas=[g['delta_A'] for g in man['geometries']])
    R['T1_pass'] = bool(R['T1']['n'] == 6 and R['T1']['all_neg']
                        and man['caps']['scan'] == 6)

    # T2 flow
    tmp = tempfile.mkdtemp(prefix='job066_T2_')
    led = ex.Ledger(os.path.join(tmp, 'b.json'), caps=dict(scan=6))
    mock_scan(man, led, mock_eval_factory(), tmp)
    order = [a['note']['tag'] for a in led.data['attempts']]
    R['T2'] = dict(n_scan=led.count('scan'), order=order,
                   all_done=all(a['status'] == 'done'
                                for a in led.data['attempts']))
    R['T2_pass'] = bool(led.count('scan') == 6 and len(order) == 6
                        and order == [g['tag'] for g
                                      in man['geometries']])

    # T3 failure mid-scan
    tmp = tempfile.mkdtemp(prefix='job066_T3_')
    led = ex.Ledger(os.path.join(tmp, 'b.json'), caps=dict(scan=6))
    err = None
    try:
        mock_scan(man, led, mock_eval_factory(
            fail_tags=('mscan_d+10',)), tmp)
    except RuntimeError as e:
        err = str(e)
    atts = led.data['attempts']
    R['T3'] = dict(error=err[:60] if err else None,
                   n_done=sum(1 for a in atts if a['status'] == 'done'),
                   n_err=sum(1 for a in atts if a['status'] == 'error'),
                   failing_row=atts[3]['status'] if len(atts) > 3 else None)
    R['T3_pass'] = bool(err and 'not converged' in err
                        and R['T3']['n_done'] == 3 and R['T3']['n_err'] == 1
                        and R['T3']['failing_row'] == 'error')

    # T4 mirror identification per point
    ok_mirror = True
    for g, g64 in zip(man['geometries'],
                      json.load(open(os.path.join(
                          R64, 'input_manifest.json')))['geometries']):
        X = np.asarray(g['coords_bohr'], float).reshape(7, 3)
        X64 = np.asarray(g64['coords_bohr'], float).reshape(7, 3)
        # signed normal taken from the SOURCE (positive-branch) geometry
        n = P.contact_plane_normal(X64)
        if (X[2] - X[0]) @ n >= 0:
            ok_mirror = False
        if float(np.abs(X[[0, 1, 3, 4, 5, 6]] - X64[[0, 1, 3, 4, 5,
                                                     6]]).max()) > 1e-10:
            ok_mirror = False
    R['T4'] = dict(all_mirror_and_inplane_same=bool(ok_mirror))
    R['T4_pass'] = ok_mirror

    # T5 isolation
    R['T5'] = dict(pyscf_stubbed=sys.modules.get('pyscf') is None,
                   d2_stubbed=getattr(sys.modules.get('d2_full'),
                                      '__name__', None) == 'd2_full',
                   dirs_untouched=bool(pre64 == snapshot(R64)
                                       and pre65 == snapshot(R65)
                                       and pre66 == snapshot(OUT)))
    R['T5_pass'] = bool(R['T5']['pyscf_stubbed']
                        and R['T5']['d2_stubbed']
                        and R['T5']['dirs_untouched'])

    keys = ('T1_pass', 'T2_pass', 'T3_pass', 'T4_pass', 'T5_pass')
    R['ALL_PASS'] = bool(all(R[k] for k in keys))
    print(json.dumps(R, indent=1, default=str)[:1200])
    ex.save_json_atomic(os.path.join(OUT, 'mock_test_results.json'), R)
    if not R['ALL_PASS']:
        raise SystemExit('MOCK TESTS FAILED')
    print('MOCK TESTS ALL PASS', flush=True)


if __name__ == '__main__':
    main()
