#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Part 1B production-path audit (JOB-2026-0904-008) -- runs AFTER the d2_full.py
v2 fix (subclass re-class attach), at production settings (def2-TZVP, grid 5,
SCF conv 1e-11) through the single production entry point make_mf_d2.

Audits
  A) scanner coordinate propagation (gas phase): build the gradient scanner
     exactly as pyscf.geomopt.berny_solver.kernel does
     (mol.copy() -> set_geom_ per step -> g_scanner(mol)); the scanner must
     return at the moved geometry the same total energy/gradient as a freshly
     created make_mf_d2 object there (D2 evaluated at the NEW coordinates).
  A2) Hessian coordinate tracking (6-31G for speed, implementation-level):
     attach at R1 -> SCF -> mutate mol to R2 -> re-SCF -> wrapped Hessian minus
     parent-class-bypass Hessian must equal the analytic D2 Hessian at R2.
  B) D2 counted exactly once, in energy_tot() AND kernel() (return value AND
     e_tot attribute), gas phase AND SMD(water):
     E(total) = E(parent bypass, same converged SCF) + E(D2),
     explicitly != E(bare) and != E(bare) + 2*E(D2).
  C) optimizer end-to-end via pyscf.geomopt.berny_solver.kernel (the production
     path): short optimization of the O3-H2O contact complex; per-step records
     (contact distance, optimizer energy, analytic D2 at recorded coordinates,
     bare-DFT component = energy - D2); endpoint re-verification with a fresh
     make_mf_d2 object: energy agreement with the last trajectory step, and
     wrapped-vs-bypass gradient/Hessian differences = analytic D2 terms.

All numbers are machine-collected into JSON; the markdown report is generated
from the same dict (no manual transcription).
"""
import os, sys, json, math, shutil
import numpy as np
from pyscf import gto
from pyscf.geomopt import berny_solver

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d2_full

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES_DIR = os.path.join(ROOT, 'results', '01_water_matrices', 'pure_water_o3_h2o')
ART_DIR = os.path.join(ROOT, 'run_artifacts', '01_pure_water_o3_h2o')
SHIFT_FAR = 5.0            # Angstrom, water displacement along x (far geometry)
MAXSTEPS_OPT = 10          # short method-audit optimization (NOT a production search)

THR = dict(scanner_energy=1e-11,      # = SCF conv_tol: cross-SCF energy noise bound
           scanner_grad=d2_full.SCF_CONV ** 0.5,  # = PySCF conv_tol_grad: orbital-gradient tolerance
           machine=1e-9,
           scf_noise=1e-8, ratio=1e-6, separation=1e-6, d2_variation=1e-8)


def build(water_x, basis='def2-TZVP'):
    """O3 (C2v, exp geometry) + H2O; water translated by water_x along x."""
    r, ang = 1.2717, 117.79
    t = math.radians(ang / 2.0)
    atoms = [('O', (0.0, 0.0, 0.0)),
             ('O', (r * math.sin(t), r * math.cos(t), 0.0)),
             ('O', (-r * math.sin(t), r * math.cos(t), 0.0)),
             ('O', (water_x, 2.60, 1.05)),
             ('H', (water_x + 0.76, 2.95, 0.55)),
             ('H', (water_x - 0.76, 2.95, 0.55))]
    return gto.M(atom=atoms, basis=basis, charge=0, spin=0, verbose=0)


def contact_distance(mol):
    """min distance from the water oxygen (atom 3) to the three O3 atoms, Angstrom."""
    c = np.asarray(mol.atom_coords(unit='Angstrom'))
    return float(min(np.linalg.norm(c[i] - c[3]) for i in range(3)))


def main():
    res = dict(
        backend='PySCF 2.14.0 / WSL2',
        implementation='attach_d2 v2 (subclass re-class; energy_tot-routed, no kernel wrapper)',
        settings=dict(basis='def2-TZVP', grid_level=d2_full.GRID_LEVEL,
                      scf_conv=d2_full.SCF_CONV, shift_far_A=SHIFT_FAR,
                      opt_maxsteps=MAXSTEPS_OPT, solver='berny_solver.kernel'),
        thresholds=THR)

    mol_R1 = build(0.0)
    mol_R2 = build(SHIFT_FAR)
    e_d2_R1 = d2_full.d2_energy(mol_R1)
    e_d2_R2 = d2_full.d2_energy(mol_R2)
    res['d2_reference'] = dict(e_d2_R1=e_d2_R1, e_d2_R2=e_d2_R2,
                               d2_delta=float(abs(e_d2_R1 - e_d2_R2)))

    # ================= A) scanner coordinate propagation =================
    mf1 = d2_full.make_mf_d2(mol_R1)
    g_scanner = mf1.nuc_grad_method().as_scanner()      # berny_solver construction
    e1, g1 = g_scanner(mol_R1)                          # step at R1
    mol_work = mol_R1.copy()                            # berny_solver: mol = method.mol.copy()
    mol_work.set_geom_(np.asarray(mol_R2.atom_coords(unit='Angstrom')), unit='Angstrom')
    e2s, g2s = g_scanner(mol_work)                      # step at R2 through the SAME scanner

    mf2 = d2_full.make_mf_d2(mol_R2)
    e2f = float(mf2.kernel())
    g2f = np.asarray(mf2.nuc_grad_method().kernel())

    # Attribution: split the scanner-vs-fresh total-gradient difference into
    # the D2 part (must be machine-exact; D2 is a deterministic function of
    # coordinates) and the DFT part (SCF convergence-path noise; the two runs
    # are seeded differently: dm0 from the previous scanner step vs minao).
    g2s_byp = np.asarray(g_scanner._d2_parent_cls.kernel(g_scanner))
    g2f_byp = np.asarray(mf2._d2_parent_cls.nuc_grad_method(mf2).kernel())
    d2_part_scanner = float(np.abs((np.asarray(g2s) - g2s_byp)
                                   - d2_full.d2_grad(mol_work)).max())
    d2_part_fresh = float(np.abs((g2f - g2f_byp)
                                 - d2_full.d2_grad(mol_R2)).max())
    d2_contributes_zero = float(np.abs((np.asarray(g2s) - g2f)
                                       - (g2s_byp - g2f_byp)).max())

    res['A_scanner_propagation'] = dict(
        e_scanner_R1=float(e1),
        e_scanner_R2_moved=float(e2s),
        e_fresh_R2=e2f,
        dE_scanner_vs_fresh_R2=float(e2s - e2f),
        maxdG_scanner_vs_fresh_R2=float(np.abs(np.asarray(g2s) - g2f).max()),
        d2_part_scanner=d2_part_scanner,
        d2_part_fresh=d2_part_fresh,
        d2_contributes_zero_to_diff=d2_contributes_zero,
        dft_path_noise_grad=float(np.abs(g2s_byp - g2f_byp).max()),
        d2_delta_scale=float(abs(e_d2_R1 - e_d2_R2)),
        note=('cross-SCF comparison: scanner seeded dm0 from the previous '
              'step, fresh object from minao; total-gradient difference is '
              'bounded by the PySCF orbital-gradient tolerance '
              'conv_tol_grad = sqrt(conv_tol); the D2 part is attributed by '
              'the parent-class bypass and must vanish from the difference'))
    res['A_scanner_propagation']['pass'] = bool(
        abs(e2s - e2f) < THR['scanner_energy']
        and float(np.abs(np.asarray(g2s) - g2f).max()) < THR['scanner_grad']
        and d2_part_scanner < THR['machine']
        and d2_part_fresh < THR['machine']
        and d2_contributes_zero < THR['machine'])

    # ================= A2) Hessian coordinate tracking (6-31G) =================
    mol_h = build(0.0, basis='6-31G')
    mfh = d2_full.make_mf_d2(mol_h)
    mfh.kernel()                                        # SCF at R1, attach happened at R1
    mol_h.set_geom_(np.asarray(build(SHIFT_FAR, '6-31G').atom_coords(unit='Angstrom')),
                    unit='Angstrom')                    # mutate AFTER attach
    mfh.reset(mol_h)                                    # PySCF idiom: rebind + rebuild grids
    mfh.kernel()                                        # re-SCF at R2 (dm0 seeded)
    e_attr_after = float(mfh.e_tot)
    h_wrp = np.asarray(mfh.Hessian().kernel())
    h_cls = np.asarray(mfh._d2_parent_cls.Hessian(mfh).kernel())
    e_bare_after = float(mfh._d2_parent_cls.energy_tot(mfh))
    h_res = float(np.abs((h_wrp - h_cls) - d2_full.d2_hess(mol_h)).max())
    e_d2_R2_631 = d2_full.d2_energy(mol_h)
    res['A2_hessian_tracking_631G'] = dict(
        basis='6-31G',
        e_tot_attr_minus_bare_over_d2=float((e_attr_after - e_bare_after) / e_d2_R2_631),
        max_hess_wrap_minus_bypass_minus_d2=h_res,
        note=('mutating mol after attach must move the D2 term: the wrapped '
              'Hessian difference against the same-SCF parent-class bypass '
              'must equal the analytic D2 Hessian at the NEW geometry'))
    res['A2_hessian_tracking_631G']['pass'] = bool(
        h_res < THR['machine']
        and abs((e_attr_after - e_bare_after) / e_d2_R2_631 - 1.0) < THR['ratio'])

    # ================= B) single counting, gas + SMD =================
    res['B_single_count'] = {}
    for tag, solvent in (('gas', None), ('smd_water', 'water')):
        mol_b = build(0.0)
        mfb = d2_full.make_mf_d2(mol_b, solvent=solvent)
        e_kernel = float(mfb.kernel())
        e_etot = float(mfb.energy_tot())
        e_attr = float(mfb.e_tot)
        e_bare = float(mfb._d2_parent_cls.energy_tot(mfb))
        e_d2 = d2_full.d2_energy(mol_b)
        rec = dict(
            e_kernel=e_kernel, e_energy_tot=e_etot, e_tot_attribute=e_attr,
            e_bare_bypass=e_bare, e_d2=e_d2,
            ratio_kernel=float((e_kernel - e_bare) / e_d2),
            ratio_energy_tot=float((e_etot - e_bare) / e_d2),
            ratio_e_tot_attr=float((e_attr - e_bare) / e_d2),
            residual_once_kernel=float(abs((e_kernel - e_bare) - e_d2)),
            residual_once_energy_tot=float(abs((e_etot - e_bare) - e_d2)),
            residual_once_e_tot_attr=float(abs((e_attr - e_bare) - e_d2)),
            separation_from_zero=float(abs(e_kernel - e_bare)),
            separation_from_double=float(abs((e_kernel - e_bare) - 2.0 * e_d2)))
        rec['pass'] = bool(
            rec['residual_once_kernel'] < THR['machine']
            and rec['residual_once_energy_tot'] < THR['machine']
            and rec['residual_once_e_tot_attr'] < THR['machine']
            and abs(rec['ratio_kernel'] - 1.0) < THR['ratio']
            and abs(rec['ratio_energy_tot'] - 1.0) < THR['ratio']
            and abs(rec['ratio_e_tot_attr'] - 1.0) < THR['ratio']
            and rec['separation_from_zero'] > THR['separation']
            and rec['separation_from_double'] > THR['separation'])
        res['B_single_count'][tag] = rec

    # ================= C) optimizer end-to-end (berny production path) =================
    mf = d2_full.make_mf_d2(mol_R1)
    mf.verbose = 3
    steps = []

    def recorder(envs):
        mol_step = envs['mol']
        energy = float(envs['energy'])
        e_d2 = d2_full.d2_energy(mol_step)
        steps.append(dict(
            step=len(steps) + 1,
            contact_distance_A=contact_distance(mol_step),
            e_total_optimizer=energy,
            e_d2_analytic=e_d2,
            e_dft_component=energy - e_d2,
            grad_norm=float(np.linalg.norm(np.asarray(envs['gradients'])))))

    opt_conv, mol_opt = berny_solver.kernel(mf, callback=recorder, maxsteps=MAXSTEPS_OPT)

    mf_end = d2_full.make_mf_d2(mol_opt)
    e_end = float(mf_end.kernel())
    g_wrp = np.asarray(mf_end.nuc_grad_method().kernel())
    g_cls = np.asarray(mf_end._d2_parent_cls.nuc_grad_method(mf_end).kernel())
    h_wrp = np.asarray(mf_end.Hessian().kernel())
    h_cls = np.asarray(mf_end._d2_parent_cls.Hessian(mf_end).kernel())
    g_res = float(np.abs((g_wrp - g_cls) - d2_full.d2_grad(mol_opt)).max())
    h_res_c = float(np.abs((h_wrp - h_cls) - d2_full.d2_hess(mol_opt)).max())
    e_d2_end = d2_full.d2_energy(mol_opt)
    d2_values = [s['e_d2_analytic'] for s in steps]

    res['C_optimizer_end_to_end'] = dict(
        optimizer_converged=bool(opt_conv),
        n_steps=len(steps),
        steps=steps,
        endpoint=dict(
            e_fresh_object=e_end,
            e_last_trajectory_step=steps[-1]['e_total_optimizer'] if steps else None,
            dE_fresh_vs_last=float(e_end - steps[-1]['e_total_optimizer']) if steps else None,
            contact_distance_final_A=contact_distance(mol_opt),
            e_d2_endpoint=e_d2_end,
            max_grad_wrap_minus_bypass_minus_d2=g_res,
            max_hess_wrap_minus_bypass_minus_d2=h_res_c))
    res['C_optimizer_end_to_end']['pass'] = bool(
        len(steps) >= 2
        and steps[-1]['e_total_optimizer'] is not None
        and abs(e_end - steps[-1]['e_total_optimizer']) < THR['scf_noise']
        and g_res < THR['machine']
        and h_res_c < THR['machine']
        and (max(d2_values) - min(d2_values)) > THR['d2_variation']
        and abs(steps[0]['e_total_optimizer']
                - res['B_single_count']['gas']['e_kernel']) < THR['scf_noise'])

    # ================= verdict + reports =================
    res['overall_pass'] = bool(
        res['A_scanner_propagation']['pass']
        and res['A2_hessian_tracking_631G']['pass']
        and all(r['pass'] for r in res['B_single_count'].values())
        and res['C_optimizer_end_to_end']['pass'])

    os.makedirs(RES_DIR, exist_ok=True)
    os.makedirs(ART_DIR, exist_ok=True)
    json_path = os.path.join(RES_DIR, 'd2_production_path_audit.json')
    with open(json_path, 'w') as fh:
        json.dump(res, fh, indent=2)
    shutil.copy2(json_path, os.path.join(ART_DIR, 'd2_production_path_audit.json'))
    write_md(res, os.path.join(RES_DIR, 'd2_production_path_audit.md'))
    print('OVERALL_PASS = %s' % res['overall_pass'])
    print('JSON = %s' % json_path)


def fmt(x, spec='%.6e'):
    return spec % x if isinstance(x, float) else str(x)


def write_md(res, path):
    s = res['settings']
    lines = []
    a = lines.append
    a('# 全导数 D2 生产路径审计报告（第 2A 阶段 Part 1B，JOB-2026-0904-008）')
    a('')
    a('生成：2026-09-04　后端：%s' % res['backend'])
    a('实现：%s　脚本：`protocols/audit_d2_production_path.py`　数据：`d2_production_path_audit.json`（本报告数值自动读取）' % res['implementation'])
    a('设置：def2-TZVP，grid level %d，SCF conv %.0e；优化器 `pyscf.geomopt.berny_solver.kernel`（生产路径），maxsteps=%d' % (s['grid_level'], s['scf_conv'], s['opt_maxsteps']))
    a('')
    a('## 结论：**%s（OVERALL_PASS = %s）**' % (
        '全部通过' if res['overall_pass'] else '存在未通过项', res['overall_pass']))
    a('')
    a('| 审计 | 内容 | 判定 |')
    a('| --- | --- | --- |')
    a('| A | scanner 坐标传播（近接触 → 远离，能量+梯度 vs 新建对象） | %s |' % ('通过' if res['A_scanner_propagation']['pass'] else '未通过'))
    a('| A2 | Hessian 坐标追踪（attach 后变异 mol，包装差值 = 解析 D2 Hessian） | %s |' % ('通过' if res['A2_hessian_tracking_631G']['pass'] else '未通过'))
    a('| B | D2 单次计入：kernel()/energy_tot()/e_tot 属性 × 气相/SMD(水) | %s |' % ('通过' if all(r['pass'] for r in res['B_single_count'].values()) else '未通过'))
    a('| C | 优化器端到端（berny 轨迹 + 终点梯度/Hessian/能量一致性） | %s |' % ('通过' if res['C_optimizer_end_to_end']['pass'] else '未通过'))
    a('')
    d2r = res['d2_reference']
    a('D2 参考量级：E(D2, 近接触) = %.6e Eh，E(D2, 远离) = %.6e Eh，|Δ| = %.3e Eh（判别尺度，远超所有通过阈值）' % (d2r['e_d2_R1'], d2r['e_d2_R2'], d2r['d2_delta']))
    a('')
    a('## 审计 A：scanner 坐标传播（与 berny_solver 构造路径逐行一致）')
    a('')
    a('| 量 | 数值 (Eh / Eh·Bohr⁻¹) |')
    a('| --- | --- |')
    A = res['A_scanner_propagation']
    a('| scanner 第 1 步（近接触）总能量 | %.12f |' % A['e_scanner_R1'])
    a('| scanner 第 2 步（远离，同一 scanner）总能量 | %.12f |' % A['e_scanner_R2_moved'])
    a('| 远离构型新建 make_mf_d2 对象总能量 | %.12f |' % A['e_fresh_R2'])
    a('| ΔE（scanner − 新建对象） | %.3e |' % A['dE_scanner_vs_fresh_R2'])
    a('| max|ΔG|（scanner − 新建对象，跨 SCF 总差） | %.3e |' % A['maxdG_scanner_vs_fresh_R2'])
    a('| D2 部分机器精度（scanner 侧 (G−G_bypass)−G_D2） | %.3e |' % A['d2_part_scanner'])
    a('| D2 部分机器精度（fresh 侧 (G−G_bypass)−G_D2） | %.3e |' % A['d2_part_fresh'])
    a('| D2 对 scanner−fresh 差值的贡献（须为零） | %.3e |' % A['d2_contributes_zero_to_diff'])
    a('| DFT 收敛路径噪声（bypass−bypass） | %.3e |' % A['dft_path_noise_grad'])
    a('')
    a('说明：scanner 与新建对象是两次独立 SCF（播种分别为上一步密度与 minao），跨 SCF 总梯度差受 PySCF 轨道梯度收敛容差 conv_tol_grad = √conv_tol 约束；D2 部分经父类旁路逐项归因，对总差值的贡献为零（机器精度），证明差异全部来自 DFT 收敛路径而非 D2 实现。')
    a('')
    a('## 审计 A2：Hessian 坐标追踪（6-31G，实现级判据）')
    a('')
    A2 = res['A2_hessian_tracking_631G']
    a('attach 后变异 mol 并重新 SCF：包装 Hessian − 同-SCF 父类旁路 Hessian = 解析 D2 Hessian（新构型），max 残差 %.3e；e_tot 属性 (E−E_bare)/E(D2) = %.12f' % (A2['max_hess_wrap_minus_bypass_minus_d2'], A2['e_tot_attr_minus_bare_over_d2']))
    a('')
    a('## 审计 B：D2 单次计入（近接触构型，E(D2) = %.6e Eh）' % res['B_single_count']['gas']['e_d2'])
    a('')
    a('| 体系 | 入口 | (E − E_bare)/E(D2) | 单次残差 | 与 0×/2× 的分离度 | 判定 |')
    a('| --- | --- | --- | --- | --- | --- |')
    for tag, name in (('gas', '气相'), ('smd_water', 'SMD(水)')):
        r = res['B_single_count'][tag]
        a('| %s | kernel() 返回值 | %.12f | %.3e | %.3e / %.3e | %s |' % (
            name, r['ratio_kernel'], r['residual_once_kernel'],
            r['separation_from_zero'], r['separation_from_double'],
            '通过' if r['pass'] else '未通过'))
        a('| %s | energy_tot() | %.12f | %.3e | — | %s |' % (
            name, r['ratio_energy_tot'], r['residual_once_energy_tot'],
            '通过' if r['pass'] else '未通过'))
        a('| %s | e_tot 属性 | %.12f | %.3e | — | %s |' % (
            name, r['ratio_e_tot_attr'], r['residual_once_e_tot_attr'],
            '通过' if r['pass'] else '未通过'))
    a('')
    a('## 审计 C：优化器端到端（`pyscf.geomopt.berny_solver.kernel`，%d 步，收敛=%s）' % (
        res['C_optimizer_end_to_end']['n_steps'], res['C_optimizer_end_to_end']['optimizer_converged']))
    a('')
    a('| 步 | O₃–H₂O 接触距离 (Å) | 优化器总能量 (Eh) | 裸 DFT 分量 (Eh) | D2 分量 (Eh) | ‖grad‖ |')
    a('| --- | --- | --- | --- | --- | --- |')
    for st in res['C_optimizer_end_to_end']['steps']:
        a('| %d | %.4f | %.8f | %.8f | %.6e | %.3e |' % (
            st['step'], st['contact_distance_A'], st['e_total_optimizer'],
            st['e_dft_component'], st['e_d2_analytic'], st['grad_norm']))
    a('')
    ep = res['C_optimizer_end_to_end']['endpoint']
    a('终点复核（新建 make_mf_d2 对象）：')
    a('')
    a('| 量 | 数值 |')
    a('| --- | --- |')
    a('| 终点接触距离 (Å) | %.4f |' % ep['contact_distance_final_A'])
    a('| 终点重跑总能量 (Eh) | %.12f |' % ep['e_fresh_object'])
    a('| 轨迹末步总能量 (Eh) | %.12f |' % ep['e_last_trajectory_step'])
    a('| ΔE（重跑 − 末步） | %.3e |' % ep['dE_fresh_vs_last'])
    a('| 终点 D2 分量 (Eh) | %.6e |' % ep['e_d2_endpoint'])
    a('| max|G_wrap − G_bypass − G_D2| | %.3e |' % ep['max_grad_wrap_minus_bypass_minus_d2'])
    a('| max|H_wrap − H_bypass − H_D2| | %.3e |' % ep['max_hess_wrap_minus_bypass_minus_d2'])
    a('')
    a('## 关键说明')
    a('')
    a('- 判据均为"同一收敛 SCF 的包装 − 父类旁路差值 = 解析 D2 项"（机器精度）或与新建对象的直接比较；能量有限差分不作判据（DFT 后端 FD 噪声底，见 Part 1 报告）。')
    a('- v2 实现说明：D2 经 `energy_tot` 进入 SCF 驱动（`hf.kernel` 每步能量经 `self.energy_tot` 实例查找），kernel() 返回值与 `e_tot` 属性自动一致；不再包装 kernel，杜绝双计。')
    a('- v1（实例属性包装）缺陷实证见 `run_artifacts/01_pure_water_o3_h2o/d2_pre_fix_diagnosis.json` 与 `d2_pre_fix_probe.json`：scanner 能量调用执行在原始 mf 对象上（执行对象与几何持有对象分离，第 2 步能量偏差 +3.22 Eh、非收敛）+ kernel() 返回值 2×D2（Part 1 检查 4 的假阳性根源）。')
    a('- 本审计为方法验证：优化轨迹不作为正式构型，不入构象搜索。')
    a('')
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines))


if __name__ == '__main__':
    main()
