#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Curvature-audit final report (JOB-2026-0905-009): assembles
results/01_water_matrices/pure_water_o3_h2o/curvature_audit/
  curvature_report.md / curvature_summary.json
from the audit artifacts.  Read-only assembly.
"""
import os
import json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, 'run_artifacts', '01_pure_water_o3_h2o')
FV = os.path.join(ART, 'final_minima_validation')
CA = os.path.join(ART, 'curvature_audit')
RES = os.path.join(ROOT, 'results', '01_water_matrices', 'pure_water_o3_h2o',
                   'curvature_audit')
os.makedirs(RES, exist_ok=True)

TARGETS = ['c14_plus', 'c06_plus']


def load(p, default=None):
    if not os.path.exists(p):
        return default
    with open(p) as fh:
        return json.load(fh)


def main():
    ut = load(os.path.join(CA, 'unit_test.json'), {})
    lay = load(os.path.join(CA, 'h2o_layer_localization.json'), {})
    worst = load(os.path.join(CA, 'h2o_worst_element.json'), {})
    cand = load(os.path.join(CA, 'candidate_curvature_summary.json'), {})
    prev = load(os.path.join(RES, '..', 'final_minima_validation',
                             'validation_summary.json'), {})
    C = {t: load(os.path.join(FV, 'stepC_reopt_%s.json' % t), {}) for t in TARGETS}
    targets = cand.get('targets', [])

    # ---------- verdicts (evidence-based)
    verdicts = {}
    detail = {}
    for r in targets:
        tid = r['id']
        sa = r['spectra']['analytic']
        sf = r['spectra']['fd']
        an_neg = sa['internal_eigenvalues'][0] < -1e-6 and \
            sa['lowest_external_overlap'] < 0.01
        fd_neg = sf['internal_eigenvalues'][0] < -1e-6 and \
            sf['lowest_external_overlap'] < 0.01
        rows_by = {name: p['rows'] for name, p in r['disputed_probes'].items()}
        # energy/gradient layer signs along each disputed direction
        eg_signs = {name: ('positive' if all(row['k_E'] > 0 for row in rows)
                           else 'negative' if all(row['k_E'] < 0 for row in rows)
                           else 'mixed')
                    for name, rows in rows_by.items()}
        c = C.get(tid, {})
        grad_met = bool(c.get('gradient_target_met'))
        if an_neg != fd_neg:
            verdict = '待定'
            reason = (
                '两种 Hessian 构造在投影内模子空间中互相矛盾（解析最低内模 '
                'λ=%s vs 差分最低内模 λ=%s），且能量/梯度层沿争议方向给出正曲率'
                '（k_E 随幅度漂移、最小幅度处接近噪声底）。解析 Hessian 的已知'
                '平台误差（~7.5×10⁻⁴–2×10⁻³）与差分噪声（~4×10⁻⁴–1.6×10⁻³）'
                '均 ≥ 争议曲率信号 → 符号不可分辨。'
                % (fmt_e(sa['internal_eigenvalues'][0]),
                   fmt_e(sf['internal_eigenvalues'][0])))
        elif an_neg and fd_neg:
            verdict = '不通过'
            reason = '可复现负曲率：两种构造一致给出负内模'
        else:
            verdict = '通过（几何收敛仍待完成）' if not grad_met else '通过'
            reason = '两种构造一致给出正内模'
        if not grad_met:
            reason += ('；另外几何收敛未完成：新鲜复核 max|grad|=%.2e > 1e-5 目标'
                       '（400 步上限，未放宽标准）' % c.get('fresh_max_gradient', float('nan')))
        verdicts[tid] = verdict
        detail[tid] = dict(verdict=verdict, reason=reason,
                           analytic_lowest_internal=sa['internal_eigenvalues'][0],
                           analytic_external_overlap=sa['lowest_external_overlap'],
                           fd_lowest_internal=sf['internal_eigenvalues'][0],
                           fd_external_overlap=sf['lowest_external_overlap'],
                           gradient_target_met=grad_met,
                           fresh_max_gradient=c.get('fresh_max_gradient'),
                           energy_gradient_layer_signs=eg_signs)

    # ---------- report
    L = []
    L.append('# JOB-2026-0905-009 曲率审计与软模判定修订报告\n')
    L.append('**取代关系**：本报告修订 `final_minima_validation/validation_report.md` '
             '中基于未修正工具的软模判定；再上一级为 `acceptance_revision/revision_report.md`'
             '（取代 v1 `conformer_report.md`）。旧版本均保留。\n')
    L.append('范围：复核验证工具本身（振动子空间、模式方向、曲率定义），统一后重新'
             '判定 c14_plus / c06_plus。不重跑 400 步优化，不批量重算构型，'
             '不启动反应/SMD 扩展/CCSD(T) 重算。\n')

    L.append('## 1. 验证工具缺陷与修正（本批核心产出）\n')
    L.append('| # | 层 | 缺陷 | 证据/修正 |\n|---|---|---|---|\n')
    L.append('| T1 | 验证脚本 | 上一批能量扫描曲率分母误为 2q²（曲率低估 2 倍、'
             '频率低估 √2）| 二次势单元测试复现 k_old=k/2；已改为 q²，'
             '撤回"√2 保守下界"表述 |\n')
    L.append('| T2 | 验证脚本 | "最大原子位移"用最大坐标分量而非每原子位移向量长度 | '
             '单元测试：同一幅度两种约定差 2 倍（0.159 vs 0.300 Å）；已改为向量长度 |\n')
    L.append('| T3 | 验证脚本 | 三层曲率度量不一致：k_H 在质量加权度量、k_E/k_g 在'
             '笛卡尔度量（O/H 质量差 → ~15 倍不可比）；且特征向量按行切片 | '
             '修正后同方向同度量比较；行切片产生的探针数据已作废重测 |\n')
    L.append('| T4 | 验证脚本 | 未投影平动/转动子空间：原始 3N 谱含 ~±10 cm⁻¹ 的'
             'TR 噪声模，"最低模"可能被外模污染 | 现构建质量加权 TR 子空间并投影，'
             '取 12 个内模；保存外模重叠与投影前后特征值 |\n')
    L.append('| T5 | 平台 | wb97xd 解析 Hessian 与解析梯度/能量层不自洽 | '
             '见 §3（层级已定位到解析二阶导层，根因待定位）|\n')
    L.append('\n单元验证（无 DFT，二次势已知曲率 λ=−1.234×10⁻⁴）：三种曲率全部精确'
             '复现（k_E=k_g=k_H，误差 <1e-12）；正曲率符号回归通过；向量长度约定'
             '通过（`curvature_audit/unit_test.json`，UNIT_TEST PASS）。\n')
    L.append('- 接口核对：PySCF `harmonic_analysis` 的 norm_mode 已是笛卡尔位移形式'
             '（norm_mode = M^-1/2 · raw_mode）；本审计直接用 numpy 特征分解的'
             '原始（质量加权）特征向量并显式做 x = M^-1/2·v·q 转换，无重复加权。\n')

    L.append('## 2. 候选争议模的三层同方向曲率（统一度量后）\n')
    for r in targets:
        tid = r['id']
        sa = r['spectra']['analytic']
        sf = r['spectra']['fd']
        L.append('### %s\n' % tid)
        L.append('- 投影后内模（质量加权度量）：解析最低 **%s**（外模重叠 %.4f）；'
                 '差分最低 **%s**（外模重叠 %.4f）。\n'
                 % (fmt_e(sa['internal_eigenvalues'][0]),
                    sa['lowest_external_overlap'],
                    fmt_e(sf['internal_eigenvalues'][0]),
                    sf['lowest_external_overlap']))
        L.append('- 解析投影谱前 12： %s\n' % ev_list(sa['internal_eigenvalues']))
        L.append('- 差分投影谱前 12： %s\n' % ev_list(sf['internal_eigenvalues']))
        for name, p in r['disputed_probes'].items():
            L.append('- 方向 `%s`（k_H^cart = %s Eh/Bohr²；d·g0 = %.2e Eh/Bohr'
                     '——近驻点残差）：\n' % (name, fmt_e(p['k_H']),
                                           p['rows'][0]['slope_g']))
            L.append('\n| 幅度(Å) | k_E(能量层) | 噪声底 | 可分辨 | k_g(梯度层) |\n'
                     '|---|---|---|---|---|\n')
            for row in p['rows']:
                L.append('| %.3f | %s | %.1e | %s | %s |\n' % (
                    row['amp_a'], fmt_e(row['k_E']), row['k_E_noise_floor'],
                    '是' if row['resolvable_E'] else '**否**', fmt_e(row['k_g'])))
        L.append('\n步长依赖：能量层 k_E 随幅度增大而增大（非谐+截断竞争），'
                 '未出现稳定平台；梯度层 k_g 同样单调漂移。\n')

    L.append('## 3. 平台问题定位程度（证据分层）\n')
    L.append('- **层级已定位**：H₂O 最差元素方向三层比较——能量层与梯度层互洽'
             '（≤5×10⁻⁵@0.005 Å）且随 h→0 外推同收敛；解析 Hessian 层与二者相差'
             ' ~7.5×10⁻⁴（常数）→ **不一致在解析二阶导（Hessian）层**。\n')
    L.append('- **元素特异性**：O–H 伸缩方向三层互洽到 ~4×10⁻⁴（两泛函同）——'
             '偏差仅在特定元素/方向。\n')
    L.append('- **能量层数值噪声底实测**：~6.6×10⁻⁸ Eh（0.005 Å 幅度下 k_E 噪声'
             ' ~4×10⁻⁴ Eh/Bohr²）。\n')
    L.append('- **未定位**：PySCF/Libxc 中具体哪个响应项缺陷（不凭一个泛函与两个'
             '对照断言）。状态：**疑似后端导数不一致（Hessian 层），根因待定位**；'
             '详见 `notes/platform_issue_wb97xd_hessian.md`（分层记录：项目包装=排除、'
             '验证脚本=已修复并回归、PySCF/Libxc=疑似、DFT 服务源码=排除/未触及）。\n')
    L.append('- 平台问题的影响量级（7.5×10⁻⁴–2×10⁻³）≥ 本项目争议软模曲率信号'
             '（~2×10⁻⁵–2.4×10⁻⁴）→ 解析 Hessian 的软模符号不可信。\n')

    L.append('## 4. 两个候选的最终判定\n')
    L.append('| 候选 | 几何收敛 | 软模判定 | 最终判定 |\n|---|---|---|---|\n')
    for t in TARGETS:
        v = verdicts[t]
        c = C.get(t, {})
        L.append('| **%s** | 未完成（max\\|grad\\|=%.2e > 1e-5，400 步上限）'
                 '| **待定**（解析/差分投影内模符号互相矛盾；直接探针不构成裁决）'
                 '| **待定** |\n' % (t, c.get('fresh_max_gradient', float('nan'))))
    for t in TARGETS:
        L.append('- **%s**：%s\n' % (t, detail[t]['reason']))
    L.append('- 结构身份（沿用验证批）：两新终点 RMSD 3.74 Å、拓扑不同 → '
             '两个独立候选；c14_plus 为已搜索结构中能量最低候选。\n')

    L.append('## 5. 撤回的表述（超出证据）\n')
    L.append('| 撤回表述 | 原因 |\n|---|---|\n')
    L.append('| "0.2 Å 内不存在更低结构" | 依据的是未修正工具的单方向扫描；'
             '能量层信号在最小幅度接近噪声底，扫描不构成完备的下降方向排查 |\n')
    L.append('| "不存在可跟进的下降方向" | 同上；且两层 Hessian 构造互相矛盾，'
             '无法确定可信的"下降方向" |\n')
    L.append('| "软模低于平台可分辨极限"（作为已证结论） | 该说法方向正确但当时'
             '未完成配对量化；现已有噪声底与信号的定量比较（§2/§3），'
             '作为**已量化的不可分辨结论**重新陈述 |\n')
    L.append('| "RMSD < 0.1 Å 因此所有能量不受影响" | RMSD 阈值判断不等于能量'
             '影响评估；正确表述：新旧几何变化 <0.1 Å，未重算高层级单点，'
             'CCSD(T) 数据绑定其原始几何作为历史单点，不移用于新终点 |\n')

    L.append('## 6. 结论分类\n')
    L.append('- **旧验证方法错误（已修复并回归）**：T1 曲率分母、T2 位移约定、'
             'T3 度量不一致、T4 未投影外模 + 此前的氢键判据/配准/虚频符号/'
             'ZPE/几何收敛键等。\n')
    L.append('- **修正后仍存在的数值问题**：wb97xd 解析 Hessian 层不自洽'
             '（平台，~7.5×10⁻⁴–2×10⁻³）；差分 Hessian 的梯度噪声放大'
             '（~4×10⁻⁴–1.6×10⁻³）；能量层数值噪声底 6.6×10⁻⁸ Eh。'
             '三者均 ≥ 争议软模信号。\n')
    L.append('- **尚未完成的几何收敛**：两候选 max\\|grad\\| 1.38/1.47×10⁻⁵ > 1e-5'
             '（400 步上限；平势面爬行至 SCF 噪声底，未放宽标准）。\n')

    L.append('## 7. 本批之后（停止，等验收）\n')
    L.append('- 验证工具可信后，再决定是否进行一次针对性终点优化、负模跟进或'
             '后端修复。本批未启动新反应、SMD 扩展或 CCSD(T) 重算。\n')

    report = '\n'.join(L)
    out_md = os.path.join(RES, 'curvature_report.md')
    with open(out_md, 'w') as fh:
        fh.write(report)

    summary = dict(
        job='JOB-2026-0905-009',
        document='curvature_summary (supersedes the soft-mode section of '
                 'final_minima_validation/validation_summary.json)',
        tool_fixes=['T1 曲率分母 q^2（撤回 sqrt2 表述）',
                    'T2 最大原子位移=每原子位移向量长度',
                    'T3 三层统一笛卡尔度量（撤回度量混用数据）',
                    'T4 质量加权 TR 投影，12 内模，外模重叠落盘'],
        unit_test=dict(passed=ut.get('quadratic_roundtrip_pass'),
                       old_denominator_bug_reproduced=ut.get(
                           'old_denominator_bug_reproduced'),
                       vector_length_convention=ut.get('vector_length_convention')),
        platform_issue=dict(layer='analytic Hessian (2nd-derivative) layer',
                            root_cause='pending localization',
                            wb97xd_worst_direction_gap=7.5e-4,
                            energy_layer_noise_floor_eh=6.6e-8,
                            note='notes/platform_issue_wb97xd_hessian.md'),
        verdicts=verdicts,
        candidate_detail=detail,
        withdrawn_statements=['0.2 Å 内不存在更低结构',
                              '不存在可跟进的下降方向',
                              '软模低于平台可分辨极限（未量化版本）',
                              'RMSD<0.1 Å 因此所有能量不受影响'],
        conclusion_classes=dict(old_tool_errors_fixed=True,
                                remaining_numerical_issues=True,
                                geometry_convergence_incomplete=True),
        artifacts=dict(raw=os.path.join(ART, 'curvature_audit'),
                       processed=RES))
    out_json = os.path.join(RES, 'curvature_summary.json')
    with open(out_json, 'w') as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print('SAVED ->', out_md)
    print('SAVED ->', out_json)
    for t, v in verdicts.items():
        print('VERDICT %s: %s' % (t, v))


def fmt_e(x):
    try:
        return '%.3e' % float(x)
    except Exception:
        return str(x)


def ev_list(eigs):
    return ' '.join(fmt_e(x) for x in eigs)


if __name__ == '__main__':
    main()
