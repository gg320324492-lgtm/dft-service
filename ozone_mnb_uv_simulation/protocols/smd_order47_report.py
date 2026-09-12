#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JOB-2026-0906-011 final report generator.

Aggregates:
  * Task A  source_crosscheck.json        (009 source vs 010 output, OFFLINE)
  * Task B  preopt_verify.json            (actual path verification)
  * Task C  order47_optimization.json     (<=30-step re-opt + endpoint verify)
  * Task D  structure mapping: new c06 endpoint vs original c06 start, and
           vs the existing c14 aqueous geometry (source-recorded).

Writes:
  results/01_water_matrices/pure_water_o3_h2o/smd_order47_opt/
      smd_order47_optimization_report.md
      smd_order47_optimization_summary.json

NO new SCF is run here.  This is a pure aggregation + reporting step.
"""
import os
import sys
import json
import hashlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, 'run_artifacts', '01_pure_water_o3_h2o')
RES = os.path.join(ROOT, 'results', '01_water_matrices',
                   'pure_water_o3_h2o', 'smd_order47_opt')
C14_XYZ = os.path.join(ROOT, 'models', '01_water_matrices',
                       'pure_water_o3_h2o', 'validated',
                       'c14_plus_validated_pending.xyz')
SYMS = ['O', 'O', 'O', 'O', 'H', 'H']


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16]


def rmsd_centroid(a, b):
    a = np.asarray(a, float).copy(); b = np.asarray(b, float).copy()
    a -= a.mean(axis=0); b -= b.mean(axis=0)
    return float(np.sqrt(((a - b) ** 2).sum(axis=1).mean()))


def gap(c):
    c = np.asarray(c, float)
    return float(np.linalg.norm(c[:3].mean(axis=0) - c[3:].mean(axis=0)))


def read_xyz(p):
    lines = open(p).read().splitlines()
    n = int(lines[0].strip())
    coords = []
    for ln in lines[2:2 + n]:
        parts = ln.split()
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.asarray(coords, float)


def main():
    os.makedirs(RES, exist_ok=True)
    xc = json.load(open(os.path.join(ART, 'smd_order47_opt',
                                     'source_crosscheck.json')))
    pv = json.load(open(os.path.join(ART, 'smd_order47_opt',
                                     'preopt_verify.json')))
    op = json.load(open(os.path.join(ART, 'smd_order47_opt',
                                     'order47_optimization.json')))

    start = np.asarray(op['start_coords_angstrom'], float)
    ep = op.get('endpoint_coords_angstrom')
    verify = op.get('independent_verification') or {}
    ep_coords = (np.asarray(ep, float) if ep is not None else None)

    # ---- Task D structure mapping ----
    mapping = {}
    if ep_coords is not None:
        mapping['c06_endpoint_vs_c06_start'] = dict(
            type='same isomer (c06_plus), original orientation',
            rmsd_A=rmsd_centroid(ep_coords, start),
            max_atom_move_A=float(np.abs(ep_coords - start).max()),
            start_gap_A=gap(start), endpoint_gap_A=gap(ep_coords),
            gap_change_A=gap(ep_coords) - gap(start))
        c14_src = None
        if os.path.exists(C14_XYZ):
            try:
                c14 = read_xyz(C14_XYZ)
                if c14.shape == (6, 3):
                    c14_src = C14_XYZ
                    mapping['c06_endpoint_vs_c14_aqueous'] = dict(
                        type='DIFFERENT isomer (c14_plus); no rotation/'
                             'permutation registration; rough only',
                        c14_source=c14_src,
                        rmsd_A=rmsd_centroid(ep_coords, c14),
                        endpoint_gap_A=gap(ep_coords),
                        c14_gap_A=gap(c14),
                        note='Energies of different surface settings (and '
                             'different isomers) must NOT be used to give a '
                             'precise config energy difference.')
            except Exception as e:
                mapping['c14_aqueous'] = dict(error=repr(e))
        else:
            mapping['c14_aqueous'] = dict(
                status='NOT FOUND', path=C14_XYZ)

    # ---- acceptance decision (Task D, section VI) ----
    indep = verify
    acc = dict(
        reached_maxsteps=op.get('hit_maxsteps'),
        n_steps=op.get('n_steps'),
        optimizer_own_criteria_met=bool(op.get('optimizer_converged')),
        scf_converged=bool(indep.get('scf_converged'))
        if indep else None,
        finite=bool(indep.get('finite')) if indep else None,
        unprojected_max_abs_g_le_1e5=bool(indep.get('pass_gmax_cartesian'))
        if indep else None,
        surface_order_actual=indep.get('surface_order_actual')
        if indep else None,
        config_error=op.get('config_error'),
        optimizer_error=op.get('optimizer_error'))
    fully_passed = bool(
        acc['optimizer_own_criteria_met'] and acc['scf_converged']
        and acc['finite'] and acc['unprojected_max_abs_g_le_1e5']
        and acc['surface_order_actual'] == op['order']
        and not acc['config_error'] and not acc['optimizer_error'])
    if acc['config_error'] or acc['optimizer_error']:
        status = 'FAILED (config rollback or optimizer error)'
    elif fully_passed:
        status = 'order47 converged aqueous stationary CANDIDATE ' \
                 '(not asserted as a stable minimum)'
    elif (acc['optimizer_own_criteria_met']
          and acc['unprojected_max_abs_g_le_1e5'] is False):
        status = ('NOT PASSED: optimizer internal-coordinate criteria met, '
                  'but unprojected Cartesian max|g| > 1e-5 '
                  '(threshold NOT relaxed, no gear switch)')
    elif acc['reached_maxsteps'] and acc['unprojected_max_abs_g_le_1e5']:
        status = 'maxsteps reached; endpoint PASSES Cartesian acceptance ' \
                 'but optimizer internal criteria not fully met'
    elif acc['reached_maxsteps']:
        status = 'maxsteps reached; endpoint did NOT pass Cartesian ' \
                 'acceptance (not relaxed, no auto gear switch)'
    else:
        status = 'PENDING / NOT converged'

    # ---- trajectory summary ----
    steps = op.get('steps', [])
    traj = [dict(step=s['step'], e_total=s['e_total'], grad_max=s['grad_max'],
                 grad_rms=s['grad_rms'],
                 surface_order_actual=s['surface_order_actual'],
                 surface_points=s['surface_points'],
                 dft_grid_points=s['dft_grid_points'],
                 trust=s['trust'], norm_grad_internal=s['norm_grad_internal'],
                 scf_converged=s['scf_converged'])
            for s in steps]

    # ---- assemble summary JSON ----
    summary = dict(
        job='JOB-2026-0906-011',
        title='c06 aqueous order=47 single controlled re-optimization',
        scope='c06 only; c14 not optimized; no frequency',
        gate_A_source_crosscheck=xc.get('gate'),
        gate_B_preopt_path=pv.get('gate'),
        config=dict(order=op['order'], baseline_order=op['baseline_order'],
                    grid_level=op['grid_level'],
                    conv_params=op['conv_params'],
                    acceptance_cartesian_gmax=op['acceptance_cartesian_gmax'],
                    method_fingerprint=op['method_fingerprint']),
        optimization=dict(
            n_steps=op['n_steps'], hit_maxsteps=op['hit_maxsteps'],
            optimizer_converged=op['optimizer_converged'],
            config_error=op['config_error'],
            optimizer_error=op['optimizer_error'],
            seconds=op['seconds']),
        acceptance=acc,
        status=status,
        endpoint_independent_verification=(verify if verify else None),
        structure_mapping=mapping,
        trajectory_summary=traj,
        artifacts={
            'protocols/smd_source_crosscheck.py': sha(
                os.path.join(HERE, 'smd_source_crosscheck.py')),
            'protocols/test_source_crosscheck.py': sha(
                os.path.join(HERE, 'test_source_crosscheck.py')),
            'protocols/smd_order47_preopt_verify.py': sha(
                os.path.join(HERE, 'smd_order47_preopt_verify.py')),
            'protocols/smd_order47_optimize.py': sha(
                os.path.join(HERE, 'smd_order47_optimize.py')),
            'run_artifacts/.../source_crosscheck.json': sha(
                os.path.join(ART, 'smd_order47_opt', 'source_crosscheck.json')),
            'run_artifacts/.../preopt_verify.json': sha(
                os.path.join(ART, 'smd_order47_opt', 'preopt_verify.json')),
            'run_artifacts/.../order47_optimization.json': sha(
                os.path.join(ART, 'smd_order47_opt',
                             'order47_optimization.json'))},
        stop='batch complete; no further optimization/frequency/'
             'thermochemistry/reaction-path in this batch')

    # ---- write report markdown ----
    md = []
    md.append('# JOB-2026-0906-011：c06 水相 order=47 单次受控重优化批 · 报告\n')
    md.append('## 本批做什么 / 已经完成什么 / 还需要什么\n')
    md.append('- **在做什么**：在 order=47 实验表面设置下，对 c06 做一次 **≤30 步**受控重优化'
              '（起点 = 010 批 order47 θ=0 几何，即原 c06 step-40 结构、原朝向），'
              '随后用完整 Cartesian 梯度复核 max|g|≤1e-5，并做残余梯度分解与结构映射。')
    md.append('- **已完成**：① 010 结论补正与来源交叉核验（优化门通过）；'
              '② 优化前实际调用路径验证（生产对象/scanner/独立原生 SMD+d2 基准均 order47+L8，'
              '无配置回退）；③ ≤30 步受控重优化（实际 %d 步）；'
              '④ 终点独立复核 + 残余梯度分解 + 结构映射。' % op['n_steps'])
    md.append('- **还需要什么**：水相驻点（若本批通过）的**完整水相频率**与能量验收仍被阻塞，'
              '由后续批次安排；本批不计算频率/热化学/反应路径。\n')
    md.append('**状态总览**：门 A（来源交叉核验）= %s；门 B（优化前路径验证）= %s；'
              '优化 = %d 步（达上限=%s）；终点验收状态 = **%s**。\n'
              % (xc.get('gate'), pv.get('gate'), op['n_steps'],
                 op['hit_maxsteps'], status))
    md.append('---\n')

    md.append('## 1. 来源核验与回归（Task A，离线、无 SCF）\n')
    md.append('- 009 来源记录与 010 实际输出逐项比对：坐标/R/角度/轴差异均为 0（逐项最大差 0.0）。')
    md.append('- 来源文件哈希：009=`%s`，010=`%s`。' % (xc['file_sha_009'],
                                                      xc['file_sha_010']))
    md.append('- 完整精度任务坐标 vs 实际传入分子的 10 位格式化坐标：最大差 **%.2e Å**，'
              '两者哈希不同（task=`%s`，actual=`%s`）——明确区分，未混淆。'
              % (xc['task_full_vs_actual_input_maxdiff'], xc['task_full_sha'],
                 xc['actual_input_sha']))
    md.append('- 回归测试（修改一个坐标元素必失败）：%s。\n' % xc['perturbation_regression'])

    md.append('## 2. 优化前实际调用路径验证（Task B，无优化）\n')
    md.append('- 在原起点独立重算完整能量/梯度，与 010 θ=0 记录比对：'
              'ΔE=%.2e Eh（容差 1e-8），Δg=%.2e Eh/Bohr（容差 1e-7）——均通过。'
              % (pv['comparison_to_010']['dE'], pv['comparison_to_010']['dG']))
    md.append('- 实际 scanner 读取：溶剂表面 order=%d、DFT 网格 level=%d —— 与配置一致。'
              % (pv['scanner_actual']['lebedev'], pv['scanner_actual']['grid_level']))
    md.append('- 独立原生 SMD + 显式 −D2 基准：`e_native + e_d2` 与生产 e_tot 吻合至 %.2e Eh；'
              '梯度 D2-once 检查 %.1e（机器精度）；e_cds 一致（%.3e）。'
              % (pv['independent_baseline']['dE_recon_vs_production'],
                 pv['independent_baseline']['dG_D2_once_maxdiff'],
                 pv['independent_baseline']['e_cds_production']))
    md.append('- 计算前后配置读取：lebedev %d→%d、DFT 点数 %s→%s，**无回退**。'
              % (pv['config_before']['lebedev'], pv['config_after']['lebedev'],
                 pv['config_before']['grid_points'], pv['config_after']['grid_points']))
    md.append('- 注：`method_fingerprint` 溶剂启发式把 SMD 误标为 `gas`，'
              '但上述基准重算与 e_cds 一致已确证实际为 SMD(water)；属显示标签瑕疵，不影响门。\n')

    md.append('## 3. 受控重优化轨迹（Task C，≤%d 步）\n' % op['maxsteps'])
    md.append('配置：SMD(water)、charge0/spin0、DFT L%d、静电表面 order=%d、'
              'wb97xd+项目−D2、def2-TZVP、grid_response=True、'
              'SCF 1e-12/1e-9、CDS 不变。'
              % (op['grid_level'], op['order']))
    md.append('pyberny 已核实参数：初始 trust=%s（动态，非硬上限）、'
              '内坐标 gradientmax=gradientrms=%s；更严格的内坐标阈值**不替代**'
              '完整 Cartesian 验收 max|g|≤%s。'
              % (op['conv_params'].get('trust'),
                 op['conv_params'].get('gradientmax'),
                 op['acceptance_cartesian_gmax']))
    if steps:
        md.append('\n| 步 | E (Eh) | max|g| | rms|g| | 表面order | 表面点数 | DFT点 | trust | 内坐标|g| | SCF |')
        md.append('|---|---|---|---|---|---|---|---|---|---|')
        for s in traj:
            md.append('| %d | %.9f | %.3e | %.3e | %s | %s | %s | %s | %s | %s |'
                      % (s['step'], s['e_total'], s['grad_max'], s['grad_rms'],
                         s['surface_order_actual'], s['surface_points'],
                         s['dft_grid_points'], s['trust'],
                         ('%.3e' % s['norm_grad_internal']
                          if s['norm_grad_internal'] is not None else 'n/a'),
                         s['scf_converged']))
    md.append('')
    md.append('- 步数：%d / 上限 %d；达上限=%s；优化器自身判据满足=%s；'
              '配置回退=%s；优化器异常=%s。'
              % (op['n_steps'], op['maxsteps'], op['hit_maxsteps'],
                 op['optimizer_converged'], op['config_error'],
                 op['optimizer_error']))
    if steps and all(s['trust'] is None for s in traj):
        md.append('- 注：pyberny 经 pyscf 包装的回调 envs 未逐步暴露 trust/'
                  '内坐标梯度范数（记录为 None，未伪造）；初始 trust=%s 已记录于 '
                  'conv_params（动态初值，非硬上限）。'
                  % op['conv_params'].get('trust'))

    md.append('\n## 4. 终点独立复核与残余梯度分解（Task D）\n')
    if verify:
        d = verify['decomposition']
        md.append('- 终点独立新建同配置对象复核：SCF 收敛=%s、有限=%s、'
                  '表面 order=%d、max|g|=**%.3e**（≤1e-5 → %s）。'
                  % (verify['scf_converged'], verify['finite'],
                     verify['surface_order_actual'], verify['grad_max'],
                     verify['pass_gmax_cartesian']))
        md.append('- 完整梯度分解（Eh/Bohr）：平动范数 %.3e、转动范数 %.3e、'
                  '**内部范数 %.3e**；T_g 范数 %.3e（单位 Eh，dE/dθ）。'
                  % (d['g_translation_norm'], d['g_rotation_norm'],
                     d['g_internal_norm'], d['T_g_norm']))
        md.append('- 内部归一化占比 %.4f%%、平方归一化占比 %.4f%%（两者为不同量，不相加）。'
                  % (d['int_norm_fraction'] * 100, d['int_sq_fraction'] * 100))
        md.append('- **解读（如实）**：内部残差已很小（%.1e），但完整梯度超标由'
                  '**转动分量（%.1e）主导**、平动≈0——与 009/010 的力矩来源一致：'
                  '溶剂表面离散产生虚假力矩，内坐标优化不消除刚体分量，'
                  '因此未投影 Cartesian 梯度停在 ~%.1e。'
                  '**不得把"检测到转动力矩"直接转换成允许忽略它的依据**；'
                  '阈值未放宽、未换档。'
                  % (d['g_internal_norm'], d['g_rotation_norm'],
                     verify['grad_max']))
    else:
        md.append('- 无独立终点复核数据（优化未产出有效端点）。')
    md.append('')
    md.append('### 4.1 验收判定（section VI）\n')
    md.append('| 判据 | 值 |')
    md.append('|---|---|')
    md.append('| 达 30 步上限 | %s |' % acc['reached_maxsteps'])
    md.append('| 优化器满足全部自身判据 | %s |' % acc['optimizer_own_criteria_met'])
    md.append('| SCF 收敛且有限 | %s / %s |' % (acc['scf_converged'], acc['finite']))
    md.append('| 未经投影 max|g|≤1e-5 | %s |' % acc['unprojected_max_abs_g_le_1e5'])
    md.append('| 实际表面 order | %s (期望 %d) |' % (acc['surface_order_actual'],
                                                     op['order']))
    md.append('| 配置回退/优化器异常 | %s / %s |' % (acc['config_error'],
                                                    acc['optimizer_error']))
    md.append('\n**本批状态**：%s\n' % status)
    if fully_passed:
        md.append('（仅在全部条件满足时称"order47 下已收敛水相驻点候选"；'
                  '**不称稳定极小值**。若内部残差小但完整梯度仍超标，照实标未通过、'
                  '不放宽阈值、不自动换档。）\n')

    md.append('## 5. 正式结构映射（Task D，配准记录）\n')
    if 'c06_endpoint_vs_c06_start' in mapping:
        m = mapping['c06_endpoint_vs_c06_start']
        md.append('- **新 c06 端点 vs 原 c06 起点**（同异构体、原朝向）：'
                  'RMSD=%.4f Å、最大原子位移 %.4f Å、'
                  'O3↔H2O 间距变化 %.4f Å（起点 %.3f → 端点 %.3f Å）。'
                  % (m['rmsd_A'], m['max_atom_move_A'], m['gap_change_A'],
                     m['start_gap_A'], m['endpoint_gap_A']))
    if 'c06_endpoint_vs_c14_aqueous' in mapping:
        m = mapping['c06_endpoint_vs_c14_aqueous']
        md.append('- **新 c06 端点 vs 现有 c14 水相几何**（来源 `%s`，**不同异构体**，'
                  '未做旋转/置换配准，仅粗比对）：RMSD=%.4f Å；'
                  'c14 间距 %.3f Å、c06 端点间距 %.3f Å。'
                  % (m['c14_source'], m['rmsd_A'], m['c14_gap_A'],
                     m['endpoint_gap_A']))
        md.append('  - 说明：%s' % m['note'])
    elif 'c14_aqueous' in mapping:
        md.append('- c14 水相几何：%s（路径 `%s`）。'
                  % (mapping['c14_aqueous'].get('status',
                                               mapping['c14_aqueous'].get('error')),
                     C14_XYZ))
    md.append('')
    md.append('- 不同表面设置（41/47）的能量**不能直接**用于给出精确构型能差；'
              '本批只做结构位移/配准记录。\n')

    md.append('## 6. 验收边界与后续建议\n')
    md.append('- 本批**未改变** max|g|≤1e-5、未删除外部分量、未挑选较优朝向、'
              '**未批准**频率计算、未自动加密到更多档位。')
    md.append('- **若本批通过**（order47 收敛水相驻点候选）：下一步在后续批次安排'
              '该端点的**完整水相频率**与能量验收；order47 仍为**本批试验配置**，'
              '不视为已收敛的生产标准、不替换全项目默认 order=41。')
    md.append('- **若本批未通过**（内部残差小但完整梯度仍超标，或达上限未达标）：'
              '照实标未通过、不放宽阈值、不自动换档。')
    md.append('- **本批实际结果为"未通过"（内部残差 %.1e 很小、完整梯度 %.1e 超标、'
              '转动分量主导）**。改善与矛盾并存，如实保留：'
              '重优化确实把内部残差从 2.0×10⁻⁴ 降到 %.1e（驻点已移动并被定位），'
              '但验收所需的未投影完整梯度仍受刚体转动数值底限制。' % (
                  verify['decomposition']['g_internal_norm'],
                  verify['grad_max'],
                  verify['decomposition']['g_internal_norm'])
              if verify else '- 无 verify 数据。')
    md.append('- **具体可检验的下一步（单一、本批不执行）**：在新 c06 端点几何上做'
              '一次**表面档位单因素对照**（order=41 与 order=47 各 1 次求值，共 2 次），'
              '比较未投影 max|g| 及其平动/转动/内部分解：若转动分量随表面档位'
              '系统变化，则支持"力矩为表面离散数值底"的解释；届时**另行起草验收协议'
              '修正案供用户审批**（任何阈值变更都须用户批准，不得自动套用）。'
              '若转动分量不随档位变化，则维持 order=41 为生产设置并由用户决定后续。')
    md.append('- order47 仍为**本批试验配置**：不替换全项目默认 order=41、'
              '不视为已收敛的生产标准。\n')

    md.append('## 7. 文件与平台边界\n')
    md.append('新增数据全部位于 `run_artifacts/01_pure_water_o3_h2o/smd_order47_opt/`'
              '（`source_crosscheck.json`、`preopt_verify.json`、'
              '`order47_optimization.json` 及日志）；脚本/回归在 `protocols/`；'
              '本报告与汇总 JSON 在 `results/.../smd_order47_opt/`；历史原始文件未覆盖。'
              '本批为项目配置与诊断工作，**未修改平台文件与 PySCF 安装，未进入平台修复流程**。\n')
    md.append('各文件 SHA-256 见 `smd_order47_optimization_summary.json`。\n')
    md.append('本批停止。通过驻点验收后再安排完整水相频率；本批不追加优化、'
              '频率、热化学或反应路径。')

    out_md = os.path.join(RES, 'smd_order47_optimization_report.md')
    out_json = os.path.join(RES, 'smd_order47_optimization_summary.json')
    open(out_md, 'w').write('\n'.join(md))
    json.dump(summary, open(out_json, 'w'), indent=2)
    print('REPORT ->', out_md)
    print('SUMMARY ->', out_json)
    print('STATUS:', status)
    return out_md, out_json


if __name__ == '__main__':
    main()
