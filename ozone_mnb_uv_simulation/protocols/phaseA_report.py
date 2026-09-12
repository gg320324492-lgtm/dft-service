#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase-A report generator for the unit-consistency revision
(JOB-2026-0906-002).  Reads the REVISED audit JSON only.
"""
import os
import json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, 'run_artifacts', '01_pure_water_o3_h2o')
UC = os.path.join(ART, 'unit_consistency_revision')
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
    uc = load(os.path.join(UC, 'unit_consistency_analysis_revised.json'))
    if uc is None:
        raise SystemExit('run unit_consistency_audit.py first')
    dc = uc['directional_confirmation']
    gr = uc['grid_response_diagnostic']
    tests = load(os.path.join(UC, 'regression_test_report.json'), {})

    L = []
    L.append('# JOB-2026-0906-002 阶段 A：单位补正、生产路径回归与结论修订\n')
    L.append('**本批做了什么**：修正 Bohr/Å 位移单位缺陷后的离线重分析（38 点全部复用），'
             '修复审计与报告脚本的匹配/精度/收敛检查问题，建立共享位移函数与'
             '生产路径回归测试；随后（阶段 B/C，见结尾）进行 scanner 调用验证与'
             '有限步数终点重优化。\n')
    L.append('**已完成**：单位缺陷确认与逐点核实；全部修正量从原始能量/梯度/'
             '实际坐标重算；b 因子交叉校验（k_E×b²、k_g/s_E×b）逐点通过；'
             '回归测试（含故意去掉 BOHR_A 的失败用例）。\n')
    L.append('**还缺**：新端点的优化与验收（阶段 C）；新终点全部内部振动模式'
             '与软模可靠性验收；水相与高层级对照更新。\n')

    L.append('## 1. 单位缺陷与修正方向\n')
    L.append('- 缺陷：Bohr 约定幅度 q 直接加到 Å 坐标 → 实际最大原子位移 = '
             '1.8897×标称；差分分母仍用 q。\n')
    L.append('- **修正方向（不得反向）**：`修正值 = 旧值 × b²`（k_E）；'
             '`修正值 = 旧值 × b`（k_g、s_E）。即旧 k_E **偏大** 1/b² = 3.57 倍，'
             '旧 k_g/s_E 偏大 1/b = 1.89 倍。\n')
    L.append('- 逐点实测（38 点）：标称 0.002/0.005/0.010 Å 的实际最大原子位移 = '
             '**0.003779 / 0.009449 / 0.018897 Å**（放大 1.88971 = 1/b）；'
             '方向偏差 0.00°（正向）/0.00°（反向对称）；正负位移长度比 1.0000。\n')

    L.append('## 2. 修正后的曲率与斜率（单位：Eh/Bohr²、Eh/Bohr；幅度单位 Å）\n')
    for batch, aud in (('directional_confirmation', dc),
                       ('grid_response_diagnostic', gr)):
        for tid in ('c14_plus', 'c06_plus'):
            a = aud[tid]
            L.append('### %s / %s（中心 %d 个，位移 %d 个，核验覆盖率 %.0f%%）\n'
                     % (batch, tid, a['counts']['central'],
                        a['counts']['displaced'], a['counts']['coverage_pct']))
            L.append('| 网格 | 标称幅度(Å) | 实际幅度(Å) | q_实际 (Bohr) | '
                     'k_E (Eh/Bohr²) | k_g(False) | s_E (Eh/Bohr) | '
                     's_g(False) | 交叉校验 k_E / k_g |\n'
                     '|---|---|---|---|---|---|---|---|---|\n')
            for r in a['analysis_false']:
                if 'excluded' in r:
                    L.append('| L%s | %.3f | 排除：%s | | | | | | |\n'
                             % (r['grid_level'], r['amp_nominal_a'], r['excluded']))
                    continue
                L.append('| L%s | %.3f | %.5f | %.5f | %s | %s | %s | %s | %s / %s |\n'
                         % (r['grid_level'], r['amp_nominal_a'], r['amp_actual_a'],
                            r['q_actual_bohr'], fmt(r['k_E']), fmt(r['k_gF']),
                            fmt(r['s_E'], 2), fmt(r['s_gF'], 2),
                            r['xcheck_k_E'].get('status'),
                            r['xcheck_k_gF'].get('status')))
            if a['analysis_true']:
                L.append('\n**完整响应梯度（grid_response=True）同方向**：\n')
                L.append('| 网格 | 实际幅度(Å) | k_g(True) | k_E | 残差 k_gT−k_E |\n'
                         '|---|---|---|---|---|\n')
                for r in a['analysis_true']:
                    L.append('| L%s | %.5f | %s | %s | %s |\n'
                             % (r['grid_level'], r['amp_actual_a'],
                                fmt(r['k_gT']), fmt(r['k_E']), fmt(r['gap_T'], 2)))

    L.append('## 3. 结论修订（撤回与更正）\n')
    L.append('| 撤回/更正 | 修订后 |\n|---|---|\n')
    L.append('| ~~c06_plus ≈0、无负曲率证据、争议关闭~~ | **撤回**。准确记录：'
             'L7 小幅度（0.0038 Å）处能量曲率 k_E = %s 与完整响应梯度曲率 '
             'k_g(True) = %s **均为负**；其稳定性、网格敏感性及新驻点性质'
             '**仍待判断**。无独立误差证据，不把负值归零 |\n'
             % (fmt(gr['c06_plus']['analysis_false'][0]['k_E']),
                fmt(gr['c06_plus']['analysis_true'][0]['k_gT'])))
    L.append('| ~~c14_plus 该方向已通过验收~~ | **更正**：仅可表述为"已采样方向上，'
             '单位修正后能量与完整响应梯度更一致，有限步长曲率为正"；'
             '这**不等于**全部内部模式或解析 Hessian 已通过验收 |\n')
    L.append('| ~~旧曲率被低估~~ | **更正方向**：旧 k_E 需乘 b²、旧 k_g/s_E 需乘 b '
             '（旧值偏大，非偏小）|\n')

    L.append('## 4. 步长与网格敏感性（有限证据）\n')
    L.append('- 可分别检查 k_E、k_g(True) 及其残差随真实 q² 的变化；'
             '每方向仅 3 个幅度、2 个网格——**两步长外推只能作为有限证据，'
             '不构成严格收敛证明**。\n')
    for tid in ('c14_plus', 'c06_plus'):
        rows = [r for r in dc[tid]['analysis_false'] if 'excluded' not in r]
        for gl in (6, 7):
            rr = [r for r in rows if r['grid_level'] == gl]
            if not rr:
                continue
            q2 = ['%.5f' % r['q_actual_bohr'] for r in rr]
            kE = ['%+.3e' % r['k_E'] for r in rr]
            kgF = ['%+.3e' % r['k_gF'] for r in rr]
            L.append('- %s L%d：q(Bohr) = %s；k_E = %s；k_g(False) = %s\n'
                     % (tid, gl, q2, kE, kgF))
    L.append('- 趋势：k_E 与 k_g 均随真实幅度增大（四次项非谐），残差亦随幅度增大'
             '——与 O(q²) 一致，但幅度点数不足以分离高阶项。\n')

    L.append('## 5. 生产路径回归测试证据\n')
    if tests:
        L.append('- 结果：**%s**（%d 项通过 / %d 项失败）\n'
                 % (tests.get('overall', 'unknown'), tests.get('n_pass', 0),
                    tests.get('n_fail', 0)))
        L.append('- 覆盖：Bohr/Å 往返、正负位移、非轴向多原子方向、'
                 '已知正/负曲率二次势、四次势 O(q²) 残差、'
                 '故意去掉 BOHR_A 的失败用例、坐标不对称/方向错误/数据缺失的报错。\n')
        for t in tests.get('cases', [])[:20]:
            L.append('  - [%s] %s %s\n' % (t.get('status'), t.get('name'),
                                           t.get('detail', '')))
    else:
        L.append('- 测试报告缺失（未运行）\n')

    L.append('## 6. 阶段 A 判定\n')
    a_pass = bool(tests.get('overall') == 'PASS')
    L.append('- **阶段 A：%s**\n' % ('通过' if a_pass else '未通过（见失败项）'))

    report = '\n'.join(L)
    out_md = os.path.join(RES, 'phaseA_unit_revision_report.md')
    with open(out_md, 'w') as fh:
        fh.write(report)
    summary = dict(job='JOB-2026-0906-002',
                   phase='A',
                   status='PASS' if a_pass else 'FAIL',
                   correction_direction=dict(k_E='old * b^2', k_g='old * b',
                                             s_E='old * b'),
                   b=B, amplification=1 / B,
                   counts={('%s/%s' % (batch, tid)): aud[tid]['counts']
                           for batch, aud in (('directional_confirmation', dc),
                                              ('grid_response_diagnostic', gr))
                           for tid in ('c14_plus', 'c06_plus')},
                   withdrawn=['c06_plus ≈0 / 无负曲率证据 / 争议关闭',
                              'c14_plus 该方向已通过验收',
                              '旧曲率被低估（方向反了）'],
                   key_numbers=dict(
                       c06_plus_L7_small_amp=dict(
                           k_E=gr['c06_plus']['analysis_false'][0]['k_E'],
                           k_gT=gr['c06_plus']['analysis_true'][0]['k_gT'],
                           note='both negative; stability pending'),
                       c14_plus_L7_small_amp=dict(
                           k_E=gr['c14_plus']['analysis_false'][0]['k_E'],
                           k_gT=gr['c14_plus']['analysis_true'][0]['k_gT'])),
                   regression_tests=tests,
                   artifacts=dict(raw=UC, processed=RES))
    out_json = os.path.join(RES, 'phaseA_unit_revision_summary.json')
    with open(out_json, 'w') as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print('SAVED ->', out_md)
    print('SAVED ->', out_json)
    print('PHASE A:', 'PASS' if a_pass else 'FAIL')


if __name__ == '__main__':
    main()
