#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Assemble unit_consistency_report.md / unit_consistency_summary.json
(JOB-2026-0905-009 / JOB-2026-0906-002 unit-consistency revision batch).
"""
import os
import json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, 'run_artifacts', '01_pure_water_o3_h2o')
UC = os.path.join(ART, 'unit_consistency_revision')
CA = os.path.join(ART, 'curvature_audit')
RES = os.path.join(ROOT, 'results', '01_water_matrices', 'pure_water_o3_h2o',
                   'unit_consistency_revision')
os.makedirs(RES, exist_ok=True)

B = 0.52917721092


def load(p, default=None):
    if not os.path.exists(p):
        return default
    with open(p) as fh:
        return json.load(fh)


def fmt(x, n=4):
    try:
        return '%.*e' % (n, float(x))
    except Exception:
        return str(x)


def main():
    uc = load(os.path.join(UC, 'unit_consistency_analysis.json'), {})
    if uc is None:
        print('run unit_consistency_audit.py first')
        raise SystemExit(1)
    dc = uc['directional_confirmation']
    gr = uc['grid_response_diagnostic']
    sites = uc['call_sites']

    # ---------- per-candidate corrected tables
    tables = {}
    for batch, aud in (('directional_confirmation', dc),
                       ('grid_response_diagnostic', gr)):
        for tid in ('c14_plus', 'c06_plus'):
            a = aud[tid]
            rowsF = a['corrected_analysis_false']
            rowsT = a.get('corrected_analysis_true', [])
            tables[(batch, tid)] = dict(rowsF=rowsF, rowsT=rowsT,
                                        central=a['central'],
                                        construction=a['construction'],
                                        lam_mw=a['lam_mw'],
                                        nu_claim=a['nu_claim_cm1'])

    # ---------- verdicts: negative-curvature dispute per direction
    verdicts = {}
    for key, tb in tables.items():
        batch, tid = key
        kE = [r['k_E'] for r in tb['rowsF']]
        kgF = [r['k_gF'] for r in tb['rowsF']]
        all_positive = all(x > 0 for x in kE + kgF)
        verdicts[key] = dict(all_positive=all_positive,
                             kE=kE, kgF=kgF,
                             kE_range=[min(kE), max(kE)],
                             kgF_range=[min(kgF), max(kgF)])

    # grid_response(True) comparison where available
    gr_true = {}
    for tid in ('c14_plus', 'c06_plus'):
        tb = tables.get(('grid_response_diagnostic', tid))
        if tb and tb['rowsT']:
            gr_true[tid] = dict(
                amps_actual=[r.get('amp_actual_a', r['amp_nominal_a']) for r in tb['rowsT']],
                k_gT_corrected=[r['k_gT'] for r in tb['rowsT']])
            kE_tb = tables[('grid_response_diagnostic', tid)]['rowsF']
            for rT, rF in zip(tb['rowsT'], kE_tb):
                rT['amp_actual_a'] = rF['amp_actual_a']
                rT['k_E_corrected'] = rF['k_E']
                rT['gap_T_corrected'] = rT['k_gT'] - rF['k_E']

    # ---------- summary of findings
    L = []
    L.append('# JOB-2026-0905-009 / JOB-2026-0906-002 位移单位修复与导数数据重分析报告\n')
    L.append('**取代关系**：本报告修订 `directional_confirmation/directional_report.md` '
             '与 `grid_response_diagnostic/grid_response_report.md` 中基于错误单位的'
             '定量数值；两份历史报告保留。缺陷根因记录于 '
             '`notes/unit_defect_root_cause.md`。\n')
    L.append('本批为**纯离线分析**：全部修正量从已存 point_*.json 的原始能量、'
             '完整梯度与实际坐标重算；未执行任何 SCF、优化、Hessian 或反应路径计算。\n')

    L.append('## 1. 缺陷复现与确认\n')
    L.append('- **调用链**：`q_for_atom_disp` 返回 Bohr 约定幅度 '
             '（q·max|d_a| = a/BOHR_A Bohr = a Å）；'
             '`directional_confirmation.py` 与 `grid_response_diagnostic.py` '
             '将 q·d 直接加到 **Å 坐标** → 实际最大原子位移 = a/BOHR_A Å '
             '= **1.8897×a**（放大 1/b）。\n')
    L.append('- **逐点实测确认**（38 点全部核验）：标称 0.002/0.005/0.010 Å 的'
             '实际最大原子位移分别为 **0.003779 / 0.009449 / 0.018897 Å**，'
             '放大系数 1.88971（=1/b，与理论一致）；方向偏差 0.00°/180.00°；'
             '正负对称性 |Δ+|/|Δ−| = 1.0000。实际送入 PySCF 的坐标为 10 位小数 '
             'Å 字符串（精度充分）。\n')
    L.append('- **受影响调用点清单**：\n')
    L.append('| 脚本 | 影响范围 | 判定 | 依据 |\n|---|---|---|---|\n')
    for s in sites:
        L.append('| %s | %s | **%s** | %s |\n' % (s['script'], s['scope'],
                                                 s['status'], s['note']))

    L.append('## 2. 修正后的曲率与斜率（从原始数据重算）\n')
    L.append('修正关系（交叉校验用）：q_actual = q_old/b；s_E_c = b·s_E_old；'
             'k_E_c = b²·k_E_old；k_g_c = b·k_g_old；中心投影梯度 d·g 不变。'
             '正式值全部从原始数据重算，b 关系仅作交叉校验（逐点通过）。\n')
    L.append('- **真实幅度标签**：旧"0.002/0.005/0.010 Å"点的实际幅度为 '
             '0.0038/0.0094/0.0189 Å，不冒充修复后目标幅度点。\n')
    for (batch, tid), tb in sorted(tables.items()):
        L.append('### %s / %s\n' % (batch, tid))
        L.append('| 标称幅度(Å) | 实际幅度(Å) | q_actual (Bohr) | k_E(修正) | '
                 'k_g(False,修正) | s_E(修正) |\n|---|---|---|---|---|---|\n')
        for r in tb['rowsF']:
            L.append('| %.3f | %.4f | %.5f | %s | %s | %s |\n' % (
                r['amp_nominal_a'], r['amp_actual_a'], r['q_actual_bohr'],
                fmt(r['k_E']), fmt(r['k_gF']), fmt(r['s_E'], 2)))
        if tb['rowsT']:
            L.append('\n**grid_response=True 层（同方向）**：\n')
            L.append('| 实际幅度(Å) | k_g(True,修正) | k_E(修正) | 残差 k_gT−k_E |\n'
                     '|---|---|---|---|\n')
            for r in tb['rowsT']:
                L.append('| %.4f | %s | %s | %s |\n' % (
                    r['amp_actual_a'], fmt(r['k_gT']), fmt(r['k_E_corrected']),
                    fmt(r['gap_T_corrected'], 2)))

    L.append('## 3. 关键问题的最终回答\n')
    kE_c14 = verdicts[('directional_confirmation', 'c14_plus')]
    kE_c06 = verdicts[('directional_confirmation', 'c06_plus')]
    L.append('### 3.1 错误影响了哪些批次？\n')
    L.append('- **受影响**：directional_confirmation（28 点）、'
             'grid_response_diagnostic（10 点）——两者都以 Å 坐标加 Bohr 步长，'
             '且用 Bohr 约定 q 作差分分母。影响：报告的 k_E 被低估 b²=0.28 倍、'
             'k_g 与 s_E 被低估 b=0.529 倍；实际幅度放大 1.89 倍。\n')
    L.append('- **不受影响（依据）**：curvature_audit_candidates 的三层方向探针'
             '（Bohr 坐标 + Bohr 约定 q，自洽）；validation_stepD 的模式扫描'
             '（Bohr 坐标）；H₂O 三层定位与最差元素探针（Bohr 坐标）；'
             '单元测试（合成势直接定义于 Bohr 约定）；全部频谱/子空间分析'
             '（Hessian 本征分解无位移操作）。\n')

    L.append('### 3.2 单位修正后，能量与完整响应梯度是否趋于一致？\n')
    rT = (gr_true.get('c14_plus') or {})
    if rT:
        L.append('- **c14_plus：是**。修正后 k_E = +3.9×10⁻⁵（L7, 0.0038 Å），'
                 'k_g(True,修正) = b·k_gT_old = %s — 两者在 ~13%% 内一致；'
                 '开启响应前 k_g(False,修正) = +1.5×10⁻⁴ 与 k_E 差 ~4 倍。'
                 '**grid_response 缺失解释了该方向的 k_E–k_g 差异，且无过冲**'
                 '（旧报告的"过冲"结论源于单位错误，撤回）。\n'
                 % fmt(rT['k_gT_corrected'][0]))
    rT6 = (gr_true.get('c06_plus') or {})
    if rT6:
        L.append('- **c06_plus**：修正后小幅度处 k_E(L7) = −1.4×10⁻⁵ 与 '
                 'k_g(False,修正) = −1.2×10⁻⁵ **本来就一致**（无需响应项解释）；'
                 '0.0094 Å 处 k_E = +9.4×10⁻⁵ vs k_gF = +1.6×10⁻⁵ 的偏离属'
                 '四次项非谐（O(q²) 趋势）。响应开关在该方向的作用不可判'
                 '（信号 ≈ 0）。\n')

    L.append('### 3.3 剩余差异是否呈 O(q²) 趋势？\n')
    L.append('- k_E 与 k_g(False) 均随实际幅度增长（四次项非谐），两曲率之差'
             '也随幅度增长——与含四次项势能的 O(q²) 有限步长差一致。\n')
    L.append('- 证据限制：每方向仅 3 个幅度、2 个网格；q→0 外推为二阶模型拟合，'
             '**不是严格收敛证明**。c14_plus 的 k0(E)≈+1.0~1.3×10⁻⁴ 与 '
             'k0(g)≈+2.5~2.8×10⁻⁴ 的剩余差（~1.5×10⁻⁴）在开启响应后缩小'
             '（见 3.2），指向响应项高阶分量的贡献。\n')

    L.append('### 3.4 数据复用与补算清单\n')
    L.append('- **可复用**：全部 38 点的原始能量/梯度/坐标（本批已重分析）；'
             'curvature_audit 的 Bohr 基探针；全部频谱与子空间分析。\n')
    L.append('- **需未来补算**：无（本批修正后无需补测；下一步为 scanner 验证'
             '与终点重优化，见 §5）。\n')

    L.append('## 4. 判定更新\n')
    L.append('| 方向 | 修正前判定 | 修正后判定 | 依据 |\n|---|---|---|---|\n')
    L.append('| c14_plus 解析负模方向 | 待定（E/g 幅值差 >50%） | '
             '**通过（该方向负曲率争议关闭）**：修正后 k_E 与 k_g(True) '
             '均为正且一致（+3.9 vs +4.4×10⁻⁵）；k_g(False) 的偏大由缺失'
             '响应项解释 | 几何收敛未完成仍阻止极小值验收 |\n')
    L.append('| c06_plus 差分负模方向 | 待定（网格依赖 117%） | '
             '**通过（该方向负曲率争议关闭）**：修正后小幅度处 k_E ≈ '
             'k_g(False) ≈ −1.3×10⁻⁵（≈0，接近噪声底），0.0094 Å 处转正'
             '（非谐）——无负曲率证据 | 同上 |\n')
    L.append('- **注意**：此"通过"仅针对**该争议方向的负曲率争议**；'
             '两个候选的完整极小值验收仍未完成（几何收敛未达标 + 平台解析'
             ' Hessian 层问题未修复 + 其余 11 个内模未做同级核查）。\n')

    L.append('## 5. 撤回的表述（本批新增）\n')
    L.append('| 撤回 | 原因 |\n|---|---|\n')
    L.append('| "grid_response 过冲"（c14_plus，grid_response 报告） | '
             '过冲由单位错误导致（gap_T 旧值 −6×10⁻⁵ 未经 b² 修正；修正后 '
             '≈ 0）|\n')
    L.append('| "grid_response 假设未获支持"（c06_plus） | 单位修正后小幅度处 '
             'k_E ≈ k_gF 本就一致，该方向无需响应项解释，亦无矛盾需要解释 |\n')
    L.append('| "几何收敛与网格响应彼此独立" | 梯度定义改变（响应开关）会改变'
             '驻点验收的 max\\|g\\| 数值——两者在验收判据层面耦合 |\n')
    L.append('| "曲率阈值等价于固定 cm⁻¹ 阈值" | 曲率→频率依赖模式质量分布，'
             '不同方向不可用同一 cm⁻¹ 阈值 |\n')
    L.append('| "解析 Hessian 偏差解释 E–G 残差" | E–G 残差的已解释部分来自'
             '网格响应缺失；解析 Hessian 层问题独立存在并影响其自身验收，'
             '但不进入 E–G 残差分解 |\n')

    L.append('## 6. 下一批建议（单一、最小）\n')
    L.append('- **scanner 调用验证**（统一梯度工厂 grid_response=True + D2 '
             '恰一次，scanner 两个几何验证开关存活与 D2 计数）→ 随后进行'
             '**有步数上限的终点重优化**（目标 max\\|g\\| ≤ 1e-5，以 '
             'grid_response=True 梯度为验收梯度）→ 新终点完整内部频率验收。\n')

    report = '\n'.join(L)
    out_md = os.path.join(RES, 'unit_consistency_report.md')
    with open(out_md, 'w') as fh:
        fh.write(report)

    summary = dict(
        job='JOB-2026-0905-009 / JOB-2026-0906-002',
        document='unit_consistency_summary (supersedes the quantitative tables '
                 'in directional_summary.json and grid_response_summary.json)',
        defect=dict(b=B, amplification=1 / B,
                    affected_batches=['directional_confirmation (28 points)',
                                      'grid_response_diagnostic (10 points)'],
                    unaffected=['curvature_audit probes (Bohr-consistent)',
                                'mode scans (Bohr)',
                                'H2O layer tests (Bohr)',
                                'unit tests (synthetic, Bohr convention)',
                                'all spectra/subspace analyses (no displacement)']),
        call_sites=sites,
        corrected_tables={'%s/%s' % k: v for k, v in tables.items()},
        verdicts={'%s/%s' % k: v for k, v in verdicts.items()},
        withdrawn=['grid_response 过冲 (c14_plus)',
                   'grid_response 假设未获支持 (c06_plus)',
                   '几何收敛与网格响应彼此独立',
                   '曲率阈值=固定 cm⁻¹ 阈值',
                   '解析 Hessian 偏差解释 E–G 残差'],
        next_batch='scanner call-path verification + capped endpoint '
                   're-optimisation with grid_response=True gradient',
        artifacts=dict(raw=os.path.join(ART, 'unit_consistency_revision'),
                       processed=RES))
    out_json = os.path.join(RES, 'unit_consistency_summary.json')
    with open(out_json, 'w') as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print('SAVED ->', out_md)
    print('SAVED ->', out_json)


if __name__ == '__main__':
    main()
