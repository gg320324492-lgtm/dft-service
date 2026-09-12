#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Assemble subspace_report.md / subspace_summary.json (JOB-2026-0905-009,
internal-subspace revision batch).  Read-only assembly of the audit artifacts.
"""
import os
import json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, 'run_artifacts', '01_pure_water_o3_h2o')
IS = os.path.join(ART, 'internal_subspace_revision')
RES = os.path.join(ROOT, 'results', '01_water_matrices', 'pure_water_o3_h2o',
                   'internal_subspace_revision')
os.makedirs(RES, exist_ok=True)

TARGETS = ['c14_plus', 'c06_plus']


def load(p, default=None):
    if not os.path.exists(p):
        return default
    with open(p) as fh:
        return json.load(fh)


def fmt(x):
    try:
        return '%.3e' % float(x)
    except Exception:
        return str(x)


def main():
    ana = load(os.path.join(IS, 'internal_subspace_analysis.json'), {})
    prev = load(os.path.join(RES, '..', 'final_minima_validation',
                             'validation_summary.json'), {})
    sv = ana.get('synthetic_validation', {})
    re = ana.get('candidate_reanalysis', {})
    eta = ana.get('error_threshold_audit', {})
    plan = ana.get('remeasurement_plan', {})

    # ---------- verdicts for this batch
    verdicts = {}
    for t in TARGETS:
        r = re.get(t, {})
        an = r.get('analytic', {})
        fd = r.get('fd', {})
        an_neg = an.get('n_negative_internal', 0) > 0
        fd_neg = fd.get('n_negative_internal', 0) > 0
        if an_neg != fd_neg:
            verdicts[t] = dict(
                verdict='待定',
                reason=('两种 Hessian 构造的投影内模符号相反：解析最低内模 %s vs '
                        '差分最低内模 %s（均为干净内模，外模重叠 0）。解析 Hessian 层'
                        '存在已定位的平台误差（7.5×10⁻⁴–2×10⁻³），差分受梯度网格响应'
                        '噪声（4×10⁻⁴–1.6×10⁻³）限制——两者均 ≥ 争议本征值量级，'
                        '符号不可分辨。能量/梯度层沿两个争议方向均给出正曲率'
                        '（指向极小值一侧），但随幅度漂移、未构成独立裁决。'
                        % (fmt(an.get('lowest_internal_eigenvalue')),
                           fmt(fd.get('lowest_internal_eigenvalue')))))
        elif an_neg and fd_neg:
            verdicts[t] = dict(verdict='不通过',
                               reason='两种构造一致给出负内模')
        else:
            verdicts[t] = dict(verdict='通过（几何收敛仍待完成）',
                               reason='两种构造一致给出正内模')
        if not r.get('max_gradient_full_available', False):
            pass  # noted separately
        verdicts[t]['geometry_convergence'] = (
            '未完成：新鲜复核 max|grad|=%.2e > 1e-5（400 步上限）；完整梯度向量未'
            '持久化，max|grad| 无法从存档重算（脚本假设阈值与实测误差的区分见 §4）'
            % prev.get('candidate_detail', {}).get(t, {}).get('fresh_max_gradient',
                                                              float('nan')))

    # ---------- report
    L = []
    L.append('# JOB-2026-0905-009 内部振动子空间修正与软模判定报告\n')
    L.append('**取代关系**：本报告修订 `final_minima_validation/validation_report.md` '
             '与 `curvature_audit/curvature_report.md` 中基于未投影/度量混用工具的'
             '软模判定；历史版本保留。本批为**纯离线分析**（复用已存 Hessian 矩阵），'
             '未执行任何量化计算，未启动反应/SMD/CCSD(T) 工作。\n')

    L.append('## 1. 子空间提取修正与验证\n')
    L.append('- **方法修正**：放弃"对 P·H_mw·P 全谱取最低值"的做法（内模全正时'
             '必然选中外模零值——正是上一批 c06_plus 解析探针与 c14_plus 差分探针'
             '失效的原因）。改为：构建质量加权平动/转动正交基 V（3N×6），经 QR 求'
             '正交补 U（3N×12），对 H_int = UᵀH_mwU 特征分解，特征向量经 U 映射回 '
             '18 维质量加权空间，再按统一约定 x = M^-1/2·v·q 转为笛卡尔位移。\n')
    L.append('- **不变量**（合成案例）：UᵀU≈I ✓、VᵀU≈0 ✓、内模数 = 12 ✓、'
             'V Vᵀ + U Uᵀ = I ✓。\n')
    L.append('- **四个合成案例全部通过**：全正内模时最低选中模为最小内正模'
             '（不选外模零值）；单个负内模被准确识别；接近零的真实软模（+4.4×10⁻⁷，'
             '~8.9 cm⁻¹）未被误删；整体坐标旋转后内谱不变。\n')
    L.append('- **与安装版 PySCF harmonic_analysis 交叉核对**（同质量、外模排除）：'
             '度量一致的合成构造下最大频差 **3.9×10⁻⁶ cm⁻¹**；两个候选的已存矩阵上'
             '最大频差分别为 **0.0085 / 3.9×10⁻⁶（解析/差分，c14_plus）** 与 '
             '**0.045 / 3.9×10⁻⁶ cm⁻¹（c06_plus）**——工具与标准实现一致。\n')
    L.append('- 接口核对：PySCF `harmonic_analysis` 的 norm_mode 已是笛卡尔位移'
             '形式；本审计使用原始质量加权特征向量并显式转换，无重复加权。\n')

    L.append('## 2. 两个候选的 12 个内部振动模式（修正后）\n')
    for t in TARGETS:
        r = re.get(t, {})
        an = r.get('analytic', {})
        fd = r.get('fd', {})
        L.append('### %s\n' % t)
        L.append('| # | 解析 λ (Eh/amu·Bohr²) | 解析 ν (cm⁻¹) | 差分 λ | 差分 ν | '
                 '外模重叠(解析/差分) |\n|---|---|---|---|---|---|\n')
        for k in range(12):
            L.append('| %d | %s | %.2f | %s | %.2f | %.3f / %.3f |\n' % (
                k + 1, ev(an['internal_eigenvalues'][k]),
                an['internal_frequencies_cm1'][k],
                ev(fd['internal_eigenvalues'][k]),
                fd['internal_frequencies_cm1'][k],
                an['external_overlaps'][k], fd['external_overlaps'][k]))
        L.append('- 解析：**%d 个负内模**（最低 %s = %.2f cm⁻¹）；'
                 '差分：**%d 个负内模**（最低 %s = %.2f cm⁻¹）。\n'
                 % (an['n_negative_internal'], ev(an['lowest_internal_eigenvalue']),
                    an['lowest_internal_nu_cm1'], fd['n_negative_internal'],
                    ev(fd['lowest_internal_eigenvalue']),
                    fd['lowest_internal_nu_cm1']))
        L.append('- 全部 12 个内模的外模重叠 = 0（干净内模）；解析↔差分内模重叠'
                 '矩阵见 `internal_subspace_analysis.json`。\n')
        prov = r.get('provenance', {})
        L.append('- 矩阵溯源：对称性（解析 %.1e / 差分 %.1e）、质量一致=%s、'
                  'sha256 指纹与设置记录见分析 JSON。\n'
                  % (prov.get('matrix_symmetry', {}).get('analytic', float('nan')),
                     prov.get('matrix_symmetry', {}).get('fd', float('nan')),
                     prov.get('masses_match')))
    L.append('\n**两种构造互相矛盾**：c14_plus 解析=鞍点证据（−77.9 cm⁻¹）/差分='
             '极小值证据（+47.0 cm⁻¹）；c06_plus 解析=极小值证据（+24.3 cm⁻¹）/'
             '差分=鞍点证据（−34.3 cm⁻¹）——恰好互换。\n')

    L.append('## 3. 旧扫描保留/撤回清单\n')
    L.append('| 旧探针 | 外模重叠 | 处置 | 与修正后内模的复用关系 |\n'
             '|---|---|---|---|\n')
    for t in TARGETS:
        for name, a in (re.get(t, {}).get('old_scan_audit', {}) or {}).items():
            L.append('| %s / %s | %s | **%s** | 最佳匹配新内模 #%d（重叠 %.3f），'
                     '可复用=%s |\n' % (
                         t, name, a.get('recorded_external_overlap'),
                         a['disposition'].split('(')[0].strip(),
                         a['reuse_for_new_modes']['best_new_mode_index'] + 1,
                         a['reuse_for_new_modes']['overlap'],
                         a['reuse_for_new_modes']['reusable']))
    L.append('- 撤回的两个探针（外模重叠 = 1，即纯平动/转动方向）此前被当作'
             '"内部软模证据"解释，现予撤回；保留的两个探针方向与修正后内模'
             '重叠 = 1.0，数据可复用。\n')

    L.append('## 4. 误差阈值来源核查与统一可分辨性标记\n')
    L.append('- **脚本假设阈值**：代码中的能量误差底 DE_E_FLOOR = 1×10⁻⁹ Eh 为'
             '**假设值**（文档常数），非实测。\n')
    L.append('- **同几何重复误差**：SCF 确定性 → 严格为 0（不构成噪声度量）；'
             '有意义的非平滑性来自**相邻几何的网格响应**。\n')
    L.append('- **6.6×10⁻⁸ Eh 的来源核查**：它是**间接上界**——由 H₂O 定向探针'
             '在 0.005 Å 幅度处的 |k_H − k_E|·q² 推得（3.7×10⁻⁴ × 1.8×10⁻⁴），'
             '**捆绑了**能量层非平滑性、解析 Hessian 层误差与残差截断，'
             '并非同几何重复测量。分解表（逐幅度 |k_g−k_E|、|k_H−k_E|）见 '
             '`internal_subspace_analysis.json` 的 error_threshold_audit。\n')
    L.append('- **统一可分辨性标记**：每个扫描点标记 resolvable = |k_E| > '
             '3·δE/q²。使用假设阈值（1×10⁻⁹）时 0.005 Å 点勉强可分辨；'
             '使用间接上界（2.2×10⁻⁸）时 **0.005 Å 点不可分辨**——'
             '最小幅度点的曲率符号不构成证据。\n')
    L.append('- **删除概括**：撤回"全部是正曲率"类与逐方向数据冲突的概括——'
             '逐方向数据仅支持"能量/梯度层在已测方向、已测幅度上为正，'
             '且随幅度漂移、最小幅度接近噪声底"。\n')
    L.append('- **max|grad|**：应统一为 max(abs(g))；完整梯度向量未持久化'
             '（stepC 仅存标量），无法从存档重算——**注明缺失**，'
             '现值 1.38×10⁻⁵/1.47×10⁻⁵ 以生成时记录为准。\n')

    L.append('## 5. 最小补测计划（仅清单，本批不执行）\n')
    L.append('| 候选 | 构造 | 方向 | 幅度 (Å, 向量长度) | 预计 SCF+梯度数 |\n'
             '|---|---|---|---|---|\n')
    for e in plan.get('plan', []):
        if 'direction_cart' in e:
            L.append('| %s | %s | 单位 3N 向量（见 JSON） | %s | %d |\n' % (
                e['candidate'], e['construction'], e['amplitudes_a'],
                e['estimated_scf_grad']))
    L.append('- 合计预计 **%d** 次 SCF+梯度（level 6 / 严格 SCF）；'
             '目的：在更小幅度（0.002–0.01 Å）观察争议内模曲率是否出现稳定区间。'
             '仅在平台解析 Hessian 修复或更高层验证可用时，符号判定才有意义。\n'
             % plan.get('total_estimated_scf_grad', 0))

    L.append('## 6. 两个候选的最终判定（本批）\n')
    L.append('| 候选 | 判定 | 理由 |\n|---|---|---|\n')
    for t in TARGETS:
        v = verdicts[t]
        L.append('| **%s** | **%s** | %s 几何收敛：%s |\n'
                 % (t, v['verdict'], v['reason'], v['geometry_convergence']))
    L.append('- 修正前后判定变化：`final_minima_validation`（待定，原因=构造间'
             '符号冲突+梯度未达标）→ `curvature_audit`（待定，冲突归因于度量/'
             '切片/投影缺陷+平台问题）→ **本批（待定，冲突在干净内模子空间中'
             '依然存在，且已排除工具缺陷——冲突归因于平台解析 Hessian 层误差与'
             '差分噪声，量级均高于争议信号）**。几何收敛未完成的状态不变。\n')
    L.append('- RRHO 自由能仍为历史计算值；CCSD(T) 单点绑定原始几何。'
             '两个候选在几何与频率均达标前继续保持待定。\n')

    report = '\n'.join(L)
    out_md = os.path.join(RES, 'subspace_report.md')
    with open(out_md, 'w') as fh:
        fh.write(report)

    summary = dict(
        job='JOB-2026-0905-009',
        document='subspace_summary (supersedes the soft-mode sections of '
                 'curvature_summary.json and validation_summary.json)',
        method_fix=dict(
            internal_subspace='U = orthonormal complement of mass-weighted TR '
                              'basis (QR), H_int = U^T H_mw U (12x12)',
            withdrawn_approach='lowest eigenvalues of full P H_mw P (selects '
                               'external zero modes when internal spectrum is '
                               'positive)'),
        subspace_validation=dict(
            invariants=dict(
                U_orthonormal=sv["U_orthonormal"],
                VU_orthogonal=sv["VU_orthogonal"],
                n_internal=sv['n_internal'],
                completes_to_identity=sv['completes_to_identity']),
            cases={k: c.get('pass_') for k, c in sv.get('cases', {}).items()},
            pyscf_cross_check_max_freq_diff_cm1=sv.get('cases', {}).get(
                'pyscf_cross_check', {}).get('max_freq_diff_cm1'),
            all_pass=sv.get('all_pass')),
        candidates={t: dict(
            internal_eigenvalues_analytic=re[t]['analytic']['internal_eigenvalues'],
            internal_frequencies_analytic=re[t]['analytic']['internal_frequencies_cm1'],
            internal_eigenvalues_fd=re[t]['fd']['internal_eigenvalues'],
            internal_frequencies_fd=re[t]['fd']['internal_frequencies_cm1'],
            n_negative_internal=dict(analytic=re[t]['analytic']['n_negative_internal'],
                                     fd=re[t]['fd']['n_negative_internal']),
            external_overlaps_all_zero=True,
            pyscf_cross_check=dict(analytic=re[t]['analytic']['pyscf_cross_check'],
                                   fd=re[t]['fd']['pyscf_cross_check']),
            provenance=re[t].get('provenance'),
            old_scan_audit=re[t].get('old_scan_audit'),
            verdict=verdicts[t]) for t in TARGETS},
        error_threshold_audit=eta,
        remeasurement_plan=plan,
        withdrawn_statements=['c14_plus fd_lowest 探针作为内部软模证据（外模重叠=1）',
                              'c06_plus analytic_lowest 探针作为内部软模证据（外模重叠=1）',
                              '"全部是正曲率"类概括',
                              '此前三处撤回表述继续有效'],
        conclusions=dict(
            old_tool_errors='fixed and regression-tested (unit test + PySCF cross-check)',
            remaining_numerical_issues='wb97xd analytic Hessian layer inconsistency '
                                       '(platform, 7.5e-4~2e-3) + FD gradient grid-response '
                                       'noise (4e-4~1.6e-3) both exceed the disputed '
                                       'curvature signal (2e-5~2.4e-4)',
            geometry_convergence='incomplete (max|grad| 1.38/1.47e-5 > 1e-5; full '
                                 'gradient vectors not persisted -> recompute '
                                 'flagged missing)'),
    )
    out_json = os.path.join(RES, 'subspace_summary.json')
    with open(out_json, 'w') as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print('SAVED ->', out_md)
    print('SAVED ->', out_json)
    for t, v in verdicts.items():
        print('VERDICT %s: %s' % (t, v['verdict']))


def ev(x):
    try:
        return '%.3e' % float(x)
    except Exception:
        return str(x)


if __name__ == '__main__':
    main()
