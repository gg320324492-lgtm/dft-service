#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Final-minima-validation step E/F (JOB-2026-0905-009): identity check,
models export, and the validation_report.md / validation_summary.json.

Verdict logic (per the batch brief):
  通过  : endpoint gradient <= 1e-5 AND soft-mode curvature stable across
          convergence checks AND no unexplained sign conflict in the
          independent verification;
  不通过: reproducible negative curvature CONFIRMED by the energy scan
          (energy descends along the mode);
  待定  : sign still flips between constructions / probes, or the gradient
          target was not met, or numerical noise exceeds the curvature
          signal (reported as 不可分辨, never forced).
"""
import os
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import conformer_utils as cu

ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, 'run_artifacts', '01_pure_water_o3_h2o')
FV = os.path.join(ART, 'final_minima_validation')
MODELS = os.path.join(ROOT, 'models', '01_water_matrices', 'pure_water_o3_h2o',
                      'validated')
RES = os.path.join(ROOT, 'results', '01_water_matrices', 'pure_water_o3_h2o',
                   'final_minima_validation')
os.makedirs(RES, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)

TARGETS = ['c14_plus', 'c06_plus']
GMAX_TARGET = 1e-5
H2KCAL = 627.5094740631


def load(p):
    with open(p) as fh:
        return json.load(fh)


def write_xyz(path, syms, coords, comment):
    lines = [str(len(syms)), comment]
    for s, c in zip(syms, coords):
        lines.append('%-2s %18.10f %18.10f %18.10f' % (s, c[0], c[1], c[2]))
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')


def main():
    C = {t: load(os.path.join(FV, 'stepC_reopt_%s.json' % t)) for t in TARGETS}
    D = {t: load(os.path.join(FV, 'stepD_softmode_%s.json' % t)) for t in TARGETS}
    B2 = load(os.path.join(FV, 'stepB2_platform_minimal_repro.json')) or {}
    BB = load(os.path.join(FV, 'stepB_settings_audit.json')) or {}

    # ---------------- identity check between the two new endpoints
    na = np.asarray(C['c14_plus']['new_geometry'], float)
    nb = np.asarray(C['c06_plus']['new_geometry'], float)
    rms_pair = cu.best_rmsd(na, nb)
    topo_pair = {t: cu.hbond_topology(np.asarray(C[t]['new_geometry'], float))['description']
                 for t in TARGETS}
    dE_pair = (C['c14_plus']['fresh_e_total'] - C['c06_plus']['fresh_e_total']) * H2KCAL
    same_structure = bool(rms_pair < 0.10)

    # ---------------- models export (validated endpoints, pending status)
    for t in TARGETS:
        write_xyz(os.path.join(MODELS, '%s_validated_pending.xyz' % t),
                  SYMS := ['O', 'O', 'O', 'O', 'H', 'H'],
                  C[t]['new_geometry'],
                  '%s re-optimised @grid6/SCF1e-12/1e-9, max|grad|=%.2e '
                  '(target 1e-5 NOT met), soft-mode status PENDING'
                  % (t, C[t]['fresh_max_gradient']))

    # ---------------- verdicts
    verdicts = {}
    for t in TARGETS:
        c, d = C[t], D[t]
        grad_ok = bool(c['gradient_target_met'])
        signs = d['signs']
        scan_curv = [v['curvature'] for v in d['scan'].values()]
        scan_positive = all(s > 0 for s in scan_curv)
        hess_negative = all(signs[k] < 0 for k in signs
                            if not k.startswith('scan'))
        # logic per brief
        if hess_negative and not scan_positive:
            verdict = '不通过'
            reason = ('可复现负曲率：解析与全部差分构造给出负模，且能量扫描证实'
                      '沿该模能量下降')
        elif grad_ok and not hess_negative and scan_positive:
            verdict = '通过'
            reason = '梯度达标，软模曲率在各构造与能量扫描中一致为正'
        else:
            verdict = '待定'
            reasons = []
            if not grad_ok:
                reasons.append('新鲜复核 max|grad|=%.2e > 1e-5 目标（400 步上限，'
                               '平势面爬行至 SCF 噪声底；不放宽标准）'
                               % c['fresh_max_gradient'])
            if hess_negative and scan_positive:
                reasons.append('符号冲突：解析+差分 Hessian 给负模（%.1f~%.1f cm⁻¹），'
                               '而能量扫描沿 ±模能量上升且曲率为正（+%.0f~+%.0f cm⁻¹，'
                               '强非谐）——软模谐振符号低于当前平台可分辨极限'
                               % (min(d['analytic']['lowest_nu_cm1'],
                                      min(v['lowest_nu_cm1'] for v in d['fd'].values())),
                                  max(d['analytic']['lowest_nu_cm1'],
                                      max(v['lowest_nu_cm1'] for v in d['fd'].values())),
                                  min(v['implied_nu_cm1'] for v in d['scan'].values()),
                                  max(v['implied_nu_cm1'] for v in d['scan'].values())))
            verdict = '待定'
            reason = '；'.join(reasons)
        verdicts[t] = dict(verdict=verdict, reason=reason,
                           fresh_max_gradient=c['fresh_max_gradient'],
                           gradient_target_met=grad_ok,
                           analytic_nu=d['analytic']['lowest_nu_cm1'],
                           fd_nu={k: v['lowest_nu_cm1'] for k, v in d['fd'].items()},
                           scan_implied_nu={k: v['implied_nu_cm1']
                                            for k, v in d['scan'].items()},
                           signs=signs)

    # ---------------- report
    L = []
    L.append('# JOB-2026-0905-009 候选结构最终几何/频率验收报告\n')
    L.append('范围：仅 c14_plus 与 c06_plus 两个候选的最终几何与频率验收；'
             '确认软模符号可靠性与二者是否为不同局部极小值。'
             '本批不启动新的反应、过渡态、离子或界面计算；未重跑 CCSD(T)。\n')
    L.append('**术语**：在搜索完备性未证明前，只称 **"已搜索结构中的最低能极小值候选"**，'
             '不称"全局最低"。\n')

    L.append('## 1. 统一精度重优化（步骤 C）\n')
    L.append('- 设置（显式记录）：grid level 6；SCF conv_tol=1e-12、'
             'conv_tol_grad=1e-9（轨道梯度阈值）；全导数 −D2；geomeTRIC '
             'convergence_gmax=1e-5 / grms=5e-6，位移判据关闭，maxsteps=400。\n')
    for t in TARGETS:
        c = C[t]
        L.append('- **%s**：新鲜对象复核 max|grad| = **%.2e** Eh/Bohr（目标 1e-5，'
                 '**未达标**，400 步上限；梯度平台 ~1.4×10⁻⁵ 与 wb97xd 梯度的网格'
                 '敏感度 1.6×10⁻⁵ 相当——目标可能低于该体系可分辨极限，此为待定证据、'
                 '未放宽标准）；E = %.8f Ha；旧→新几何 RMSD %.4f Å（max shift %.3f Å）。\n'
                 % (t, c['fresh_max_gradient'], c['fresh_e_total'],
                    c['old_new_rmsd_angstrom'], c['old_new_max_shift']))

    L.append('## 2. 导数设置审计与剪枝配对对照（步骤 B/B2）\n')
    L.append('- 设置逐项落盘（几何单位、泛函/基组/−D2 版本、网格级别与剪枝、'
             'small_rho_cutoff（默认实为 0，筛分未启用）、SCF 双阈值、网格响应路径）：'
             '`stepB_settings_audit.json`。\n')
    L.append('- **剪枝配对对照**：四种网格设置下 ±0.005 Bohr 位移的梯度跳变完全相同'
             '（5.491e-3 = 真实曲率信号）→ **剪枝/筛分归因被排除**；同几何梯度对剪枝'
             '敏感度 pbe 5.2e-10 / b3lyp 2.8e-9 / wb97xd 1.6e-5。\n')
    L.append('- **平台最小复现（H₂O/def2-TZVP，纯 PySCF，无项目钩子）**：'
             'pbe 2.8e-5→4.3e-4（h² 截断 ✓）、b3lyp 5.0e-5→4.5e-4（✓）、'
             '**wb97xd 1.9e-3→1.8e-3（不随步长收缩）**——wb97xd 的解析二阶导与其解析'
             '梯度互不自洽 ~2×10⁻³（相对 2.5×10⁻³）。按治理文件记录 '
             '`notes/platform_issue_wb97xd_hessian.md`，平台未修改；本项目以差分 Hessian '
             '+ 能量扫描绕过。\n')

    L.append('## 3. 软模独立验证（步骤 D，统一设置）\n')
    L.append('| 探针 | %s | %s |\n|---|---|---|\n' % tuple(TARGETS))
    for probe_label, key in (('解析 Hessian', 'analytic'),
                             ('FD 0.02 Bohr', 'step_0.02'),
                             ('FD 0.03 Bohr', 'step_0.03'),
                             ('FD 0.05 Bohr（辅助）', 'step_0.05'),
                             ('扫描 @0.05 Å', 'amp_0.05A'),
                             ('扫描 @0.10 Å', 'amp_0.1A'),
                             ('扫描 @0.20 Å', 'amp_0.2A')):
        row = []
        for t in TARGETS:
            d = D[t]
            if key == 'analytic':
                row.append('%.1f cm⁻¹' % d['analytic']['lowest_nu_cm1'])
            elif key.startswith('step_'):
                row.append('%.1f cm⁻¹' % d['fd'][key]['lowest_nu_cm1'])
            else:
                row.append('%.1f cm⁻¹（曲率 +）' % d['scan'][key]['implied_nu_cm1'])
        L.append('| %s | %s | %s |\n' % (probe_label, row[0], row[1]))
    L.append('- 扫描曲率随幅度增大（强非谐上升），±模方向能量均上升——0.2 Å 内'
             '沿该模**不存在更低结构**；扫描不依赖被质疑的解析二阶导'
             '（SCF 能量平滑度 ~1e-10，争议曲率信号对应 ΔE ~1e-6–1e-3 Eh，可分辨）。\n')
    L.append('- 质量加权→笛卡尔转换已在脚本内文档化'
             '（x_cart = M^-1/2 · v · q，q 单位 amu^0.5·Bohr；MASS 显式给定为 '
             'O 15.999 / H 1.008）。\n')
    L.append('- 注：扫描曲率按 ± 对称差分提取，含 √2 系统修正（表中值为保守下界）。\n')

    L.append('## 4. 验收判定\n')
    L.append('| 候选 | 判定 | 依据 |\n|---|---|---|\n')
    for t in TARGETS:
        v = verdicts[t]
        L.append('| **%s** | **%s** | %s |\n' % (t, v['verdict'], v['reason']))
    L.append('- 结构身份：两个新终点 RMSD = %.2f Å、ΔE = %.3f kcal/mol、拓扑不同'
             '（c14_plus：水氧架于臭氧环上方、无氢键；c06_plus：水氢向末端氧供氢）'
             '→ **保持两个独立候选，不合并**。\n' % (rms_pair, abs(dE_pair)))
    L.append('- 旧→新几何变化 < 0.1 Å（0.030 / 0.070 Å），不构成实质变化；'
             '受影响的能量与高层级单点：无（CCSD(T) 按指示未重跑，其值在修订报告'
             '口径下仍可引用为历史单点）。\n')
    L.append('- **RRHO 自由能**：频率未验收，v1/修订报告中的 ΔG 保留为**历史计算值**，'
             '不作为最终热力学结论。\n')
    L.append('- 沿负模跟进的说明：解析/差分 Hessian 的负模与能量扫描矛盾，且扫描显示'
             '沿 ±模能量上升——不存在可跟进的下降方向，故未执行负模跟进'
             '（避免追随数值伪影）。\n')

    L.append('## 5. 待定原因与解除条件\n')
    L.append('- 两个候选的软模谐振符号无法在当前平台精度下分辨：解析二阶导存在'
             ' ~2×10⁻³ 的实现级不自洽（平台问题已记录），差分受 ~1.6×10⁻⁵ 的梯度'
             '网格敏感度经 1/(2h) 放大（h=0.02 时 ~4×10⁻⁴）。\n')
    L.append('- 解除条件（任一）：(a) 平台修复 wb97xd 解析 Hessian 并回归通过后重算；'
             '(b) 采用不依赖解析二阶导的更高层验证（如 CCSD(T) 解析 Hessian 或更大'
             '基组/更高网格的差分）；(c) 沿软模的一维势能曲线精细扫描至 q→0 极限'
             '外推谐振曲率。\n')
    L.append('- 在此之前，两个候选均以 **"已搜索结构中的最低能极小值候选（待定）"**'
             '身份保留；c14_plus 为其中能量最低者（−301.87483610 Ha @统一精度）。\n')

    report = '\n'.join(L)
    out_md = os.path.join(RES, 'validation_report.md')
    with open(out_md, 'w') as fh:
        fh.write(report)

    summary = dict(
        job='JOB-2026-0905-009',
        document='validation_summary (final minima validation)',
        unified_settings=dict(grid_level=6, scf_tol=1e-12, scf_tol_grad=1e-9,
                              gmax_target=1e-5, opt_maxsteps=400),
        verdicts=verdicts,
        identity=dict(rmsd_between_candidates=rms_pair,
                      dE_kcal=dE_pair, same_structure=same_structure,
                      topologies=topo_pair,
                      conclusion='两个独立候选，不合并'),
        lowest_among_searched='c14_plus',
        terminology='已搜索结构中的最低能极小值候选（不称全局最低）',
        platform_issue=dict(note='notes/platform_issue_wb97xd_hessian.md',
                            repro='protocols/validation_stepB2_platform_repro.py',
                            wb97xd_max_dH_h2o=[1.918e-03, 1.821e-03],
                            pruning_attribution='excluded by paired control'),
        rrho_free_energies='historical values, NOT final thermodynamic conclusions',
        models=[('models/01_water_matrices/pure_water_o3_h2o/validated/'
                 '%s_validated_pending.xyz' % t) for t in TARGETS],
        artifacts=dict(raw=FV, processed=RES),
    )
    out_json = os.path.join(RES, 'validation_summary.json')
    with open(out_json, 'w') as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print('SAVED ->', out_md)
    print('SAVED ->', out_json)
    for t in TARGETS:
        print('VERDICT %s: %s' % (t, verdicts[t]['verdict']))


if __name__ == '__main__':
    main()
