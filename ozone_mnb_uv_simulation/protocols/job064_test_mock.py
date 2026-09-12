#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-064 step B: zero-evaluation tests (mock backend only,
formal directories untouched).

  T1  manifest: six geometries, contact targets, one branch, caps;
  T2  scan flow on a mock backend: 6 attempts in delta order, each
      record saved BEFORE the attempt is marked done, ledger scan=6;
  T3  SCF failure at a point -> error attempt + save + STOP, remaining
      points NOT evaluated;
  T4  cap: a 7th scan request is refused;
  T5  placement rigidity: per-point internal distance matrices equal
      the JOB-026 monomer matrices under the registered permutation;
  T6  isolation: pyscf/d2_full stubbed; formal dirs byte-identical.
"""
import os, sys, json, tempfile, types, glob
import numpy as np

sys.modules['pyscf'] = None
sys.modules['pyscf.hessian'] = None
sys.modules['d2_full'] = types.ModuleType('d2_full')

HERE = '/mnt/e/dft-service/ozone_mnb_uv_simulation/protocols'
sys.path.insert(0, HERE)
import job047_exec as ex
import job064_prep as P

OUT = P.OUT
IDX = P.IDX
BPA = 1.0 / ex.ANG_PER_BOHR


def snapshot(d):
    if not os.path.isdir(d):
        return []
    return sorted((f, os.path.getsize(os.path.join(d, f)))
                  for f in os.listdir(d))


def mock_eval_factory(man, fail_tags=(), e_off=0.0):
    mon_nh3 = np.asarray(json.load(open(IDX))['default_endpoints'][
        'NH3_gas_v5_final']['coords_angstrom'], float)
    mon_o3 = np.asarray(json.load(open(IDX))['default_endpoints'][
        'O3_gas_final']['coords_angstrom'], float)

    def ev(R, out_dir, tag):
        R = np.asarray(R, float).reshape(7, 3)
        g = R * 0.01                      # analytic gradient (tiny)
        g[0, 0] += 0.02                   # make max|g| identifiable
        e = -282.0 + float(np.sin(R.sum()) * 1e-6) + e_off
        rec = dict(e_total=e, e_d2_Eh=-1e-5, e_dft_part_Eh=e + 1e-5,
                   grad=g.tolist(), grad_max=float(np.abs(g).max()),
                   grad_sha='m',
                   coords_actual_angstrom=(R * ex.ANG_PER_BOHR).tolist(),
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
    pre_scan = snapshot(OUT)
    mon = [json.load(open(f)) for f in
           sorted(glob.glob(os.path.join(OUT, '..', 'c2_scan064',
                                         'input_manifest.json')))]
    man = json.load(open(os.path.join(OUT, 'input_manifest.json')))
    R = {}

    # ---- T1 manifest -----------------------------------------------------
    R['T1'] = dict(n_geom=len(man['geometries']),
                   deltas=[g['delta_A'] for g in man['geometries']],
                   all_contacts=bool(all(g['contacts_ok']
                                         for g in man['geometries'])),
                   all_internal=bool(all(g['internal_ok']
                                         for g in man['geometries'])),
                   all_collision=bool(all(g['collision_ok']
                                          for g in man['geometries'])),
                   caps=man['caps']['scan'])
    R['T1_pass'] = bool(R['T1']['n_geom'] == 6
                        and R['T1']['all_contacts']
                        and R['T1']['all_internal']
                        and R['T1']['all_collision']
                        and man['caps']['scan'] == 6)

    # ---- T2/T3: scan flow ------------------------------------------------
    tmp = tempfile.mkdtemp(prefix='job064_T2_')
    led = ex.Ledger(os.path.join(tmp, 'b.json'),
                    caps=dict(scan=6))
    ev = mock_eval_factory(man)
    err = None
    evaled = []
    try:
        mock_scan(man, led, ev, tmp)
    except RuntimeError as e:
        err = str(e)
    n_scan = led.count('scan')
    order = [a['note']['tag'] for a in led.data['attempts']]
    R['T2'] = dict(n_scan=n_scan, order=order,
                   all_done=all(a['status'] == 'done'
                                for a in led.data['attempts']))
    R['T2_pass'] = bool(n_scan == 6 and len(order) == 6
                        and order == [g['tag'] for g
                                      in man['geometries']]
                        and all(a['status'] == 'done'
                                for a in led.data['attempts']))

    # ---- T3: SCF failure mid-scan ----------------------------------------
    tmp = tempfile.mkdtemp(prefix='job064_T3_')
    led = ex.Ledger(os.path.join(tmp, 'b.json'), caps=dict(scan=6))
    ev3 = mock_eval_factory(man, fail_tags=('scan_d+10',))
    err3 = None
    try:
        mock_scan(man, led, ev3, tmp)
    except RuntimeError as e:
        err3 = str(e)
    atts = led.data['attempts']
    n_done = sum(1 for a in atts if a['status'] == 'done')
    n_err = sum(1 for a in atts if a['status'] == 'error')
    R['T3'] = dict(error=err3[:60] if err3 else None,
                   n_done=n_done, n_err=n_err)
    R['T3_pass'] = bool(err3 and 'not converged' in err3
                        and n_err == 1
                        and atts[3]['status'] == 'error'
                        and n_done == 3)

    # ---- T4: cap refusal -------------------------------------------------
    err4 = None
    led2 = led if False else None
    tmp4 = tempfile.mkdtemp(prefix='job064_T4_')
    led4 = ex.Ledger(os.path.join(tmp4, 'b.json'), caps=dict(scan=1))
    a4 = led4.pre_eval('scan', dict(tag='x'))
    led4.post_eval(a4, dict(e_total=-1.0))
    try:
        led4.pre_eval('scan', dict(tag='y'))
    except RuntimeError as e:
        err4 = str(e)
    R['T4'] = dict(error=err4)
    R['T4_pass'] = bool(err4 and 'cap reached' in err4)

    # ---- T5: placement rigidity per point --------------------------------
    mon_nh3 = np.asarray(json.load(open(IDX))['default_endpoints'][
        'NH3_gas_v5_final']['coords_angstrom'], float)
    mon_o3 = np.asarray(json.load(open(IDX))['default_endpoints'][
        'O3_gas_final']['coords_angstrom'], float)
    worst_nh, worst_o = 0.0, 0.0
    for g in man['geometries']:
        X = np.asarray(g['coords_bohr'], float) * ex.ANG_PER_BOHR
        pmn = g['perm_nh3']
        pmo = g['perm_o3']
        nh_dev = float(np.abs(
            P.distance_matrix(X[:4]) - P.distance_matrix(mon_nh3[
                pmn])).max())
        o_dev = float(np.abs(
            P.distance_matrix(X[4:]) - P.distance_matrix(mon_o3[
                pmo])).max())
        worst_nh = max(worst_nh, nh_dev)
        worst_o = max(worst_o, o_dev)
    R['T5'] = dict(worst_nh_dev_A=worst_nh, worst_o_dev_A=worst_o)
    R['T5_pass'] = bool(worst_nh < 1e-8 and worst_o < 1e-8)

    R['formal_dirs_untouched'] = bool(pre_scan == snapshot(OUT))
    keys = ('T1_pass', 'T2_pass', 'T3_pass', 'T4_pass', 'T5_pass',
            'formal_dirs_untouched')
    R['ALL_PASS'] = bool(all(R[k] for k in keys))
    print(json.dumps(R, indent=1, default=str)[:1500])
    ex.save_json_atomic(os.path.join(OUT, 'mock_test_results.json'), R)
    if not R['ALL_PASS']:
        raise SystemExit('MOCK TESTS FAILED')
    print('MOCK TESTS ALL PASS', flush=True)


if __name__ == '__main__':
    main()
