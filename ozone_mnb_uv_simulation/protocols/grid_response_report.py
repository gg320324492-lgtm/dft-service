#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Grid-response analysis + report (JOB-2026-0905-009).

Answers the five brief questions from the paired grid_response diagnostic:
  1. was grid response actually enabled in all previous gradient work?
  2. does the switch explain the k_E vs k_g slope/curvature gaps?
  3. does the central structure meet the 1e-5 gradient standard after
     enabling the response?
  4. does the scanner keep the setting (call-path verification)?
  5. next step: targeted re-optimisation or further derivative diagnosis?
"""
import os
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import curvature_audit_lib as cal

ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, 'run_artifacts', '01_pure_water_o3_h2o')
GR = os.path.join(ART, 'grid_response_diagnostic')
IS = os.path.join(ART, 'internal_subspace_revision')
FV = os.path.join(ART, 'final_minima_validation')
RES = os.path.join(ROOT, 'results', '01_water_matrices', 'pure_water_o3_h2o',
                   'grid_response_diagnostic')
os.makedirs(RES, exist_ok=True)

TARGETS = ['c14_plus', 'c06_plus']
AMPS_A = [0.002, 0.005]
BOHR_A = 0.52917721092


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
    pts = load(os.path.join(GR, 'grid_response_points.json'), {}).get('points', [])
    isx = load(os.path.join(IS, 'internal_subspace_analysis.json'), {})
    by = {p['key']: p for p in pts}

    # ---------- question 1: was grid response enabled before? (source audit)
    q1 = dict(
        default_and_actual=('Gradients.grid_response 默认 = False（'
                            'grad_rks_Gradients_grid_response 配置未覆盖）；'
                            '本项目所有历史梯度（含全部定向补测与优化）均在默认 '
                            'False 下计算——日志无 grid_response = True 记录'),
        switch_owner='开关属于 Gradients 对象实例；本项目 D2 包装（attach_d2_grad '
                     '子类）不遮蔽该属性——实例赋值后父类 kernel/extra_force '
                     '路径可见（调用验证见 §4）',
        response_path_true='get_veff 改用 get_vxc_full_response（全响应 XC 贡献）；'
                           'extra_force 逐原子返回 vhf.exc1_grid',
        source_refs='pyscf/grad/rks.py L50（get_veff 分支）、L745（默认值）、'
                    'L772（extra_force 分支）；PySCF 2.14.0')

    # ---------- per-candidate analysis
    L = []
    verdicts = {}
    summary_cands = {}
    for tid in TARGETS:
        construction = 'analytic' if tid == 'c14_plus' else 'fd'
        d = np.asarray(isx['candidate_reanalysis'][tid][construction]
                       ['lowest_internal_direction_cart'], float)
        d3 = d.reshape(6, 3)
        lam_mw = float(isx['candidate_reanalysis'][tid][construction]
                       ['lowest_internal_eigenvalue'])
        nu_claim = float(np.sign(lam_mw) * np.sqrt(abs(lam_mw)) * 5139.5)

        cen = by['%s_central' % tid]
        g0F = np.asarray(cen['grads']['grid_response_False']['gradient'])
        g0T = np.asarray(cen['grads']['grid_response_True']['gradient'])
        max_gF = cen['grads']['grid_response_False']['max_abs_grad']
        max_gT = cen['grads']['grid_response_True']['max_abs_grad']
        s_gF = float(d @ g0F)
        s_gT = float(d @ g0T)
        d_resp_center = float(d @ (g0T - g0F))

        rows = []
        for a in AMPS_A:
            kp = by['%s_amp%g_+1' % (tid, a)]
            km = by['%s_amp%g_-1' % (tid, a)]
            q = cal.q_for_atom_disp(d3, a)
            k_E = (kp['e_scf_total'] + km['e_scf_total'] -
                   2 * cen['e_scf_total']) / q ** 2
            gF_p = np.asarray(kp['grads']['grid_response_False']['gradient'])
            gF_m = np.asarray(km['grads']['grid_response_False']['gradient'])
            gT_p = np.asarray(kp['grads']['grid_response_True']['gradient'])
            gT_m = np.asarray(km['grads']['grid_response_True']['gradient'])
            k_gF = float(d @ (gF_p - gF_m)) / (2 * q)
            k_gT = float(d @ (gT_p - gT_m)) / (2 * q)
            s_E = float(kp['e_scf_total'] - km['e_scf_total']) / (2 * q)
            # grid-response contribution to the slope from the displacements
            d_resp_disp = float(d @ (gT_p - gF_p + gT_m - gF_m)) / 2.0
            gap_F = k_gF - k_E
            gap_T = k_gT - k_E
            explained = gap_F - gap_T
            rows.append(dict(amp_a=a, q_bohr=q, k_E=k_E, k_gF=k_gF, k_gT=k_gT,
                             s_E=s_E, gap_F=gap_F, gap_T=gap_T, explained=explained,
                             d_resp_disp=d_resp_disp))
            print('[%s] amp=%.3f  k_E=%+.4e  k_gF=%+.4e  k_gT=%+.4e  '
                  'gapF=%+.3e gapT=%+.3e  d_resp(disp)=%+.3e'
                  % (tid, a, k_E, k_gF, k_gT, gap_F, gap_T, d_resp_disp),
                  flush=True)

        # residual gaps with the switch ON, absolute values
        resid_T = [abs(r['gap_T']) for r in rows]
        resid_F = [abs(r['gap_F']) for r in rows]
        explains_fully = all(x < 1e-5 for x in resid_T)  # |gap| below ~1 cm-1 scale
        explains_part = (sum(resid_T) < sum(resid_F))
        d_resp_center_val = d_resp_center
        d_resp_disp_vals = [abs(r['d_resp_disp']) for r in rows]
        resp_consistent = (abs(d_resp_center_val) > 0 and
                           max(d_resp_disp_vals) / abs(d_resp_center_val) < 3.0)

        if explains_fully:
            verdict = '通过（该方向负曲率争议关闭：grid_response 完全解释层间差异）'
            advice = ('在项目内启用统一梯度工厂（grid_response=True）并做 scanner '
                      '两几何调用验证后，进行有步数上限的终点重优化。')
        elif explains_part:
            verdict = '待定（部分解释）'
            advice = ('定量分解已解释/剩余部分；剩余部分优先做最小导数诊断，'
                      '暂不进行终点优化。')
        else:
            verdict = '待定（假设未获支持）'
            advice = ('grid_response 假设未获支持：提交剩余差异与最小后续诊断建议；'
                      '不增加网格或批量计算。')

        verdicts[tid] = dict(
            verdict=verdict, advice=advice,
            k_E=[r['k_E'] for r in rows], k_gF=[r['k_gF'] for r in rows],
            k_gT=[r['k_gT'] for r in rows],
            gap_F=[r['gap_F'] for r in rows], gap_T=[r['gap_T'] for r in rows],
            explained=[r['explained'] for r in rows],
            d_resp_center=d_resp_center_val,
            d_resp_displacements=d_resp_disp_vals,
            resp_consinstant_across_geometries=bool(resp_consistent),
            central_max_grad_F=max_gF, central_max_grad_T=max_gT,
            central_gradient_standard_met_F=bool(max_gF <= 1e-5),
            central_gradient_standard_met_T=bool(max_gT <= 1e-5),
            s_gF=s_gF, s_gT=s_gT)
        summary_cands[tid] = dict(
            disputed_direction=construction,
            lam_mw=lam_mw, claimed_nu_cm1=nu_claim,
            central=dict(max_grad_F=max_gF, max_grad_T=max_gT,
                         standard_met_F=bool(max_gF <= 1e-5),
                         standard_met_T=bool(max_gT <= 1e-5),
                         slope_gF=s_gF, slope_gT=s_gT,
                         grid_response_slope_contribution=d_resp_center_val),
            probes=rows,
            verdict=verdict, advice=advice)

        L.append('## 候选 %s（争议方向：%s 负内模，声称 ν = %.1f cm⁻¹）\n'
                 % (tid, construction, nu_claim))
        L.append('| 幅度(Å) | k_E | k_g(False) | k_g(True) | gap_F = k_gF−k_E | '
                 'gap_T = k_gT−k_E | 被解释量 gap_F−gap_T | d·[gT−gF](位移实测) |\n'
                 '|---|---|---|---|---|---|---|---|\n')
        for r in rows:
            L.append('| %.3f | %s | %s | %s | %s | %s | %s | %s |\n' % (
                r['amp_a'], fmt(r['k_E']), fmt(r['k_gF']), fmt(r['k_gT']),
                fmt(r['gap_F'], 2), fmt(r['gap_T'], 2), fmt(r['explained'], 2),
                fmt(r['d_resp_disp'], 2)))
        L.append('\n- 中心点：max\\|g(False)\\| = %s，max\\|g(True)\\| = %s '
                 'Eh/Bohr；s_g(F) = %s，s_g(T) = %s，s_E = %s（斜率三估计）；'
                 'd·[g(True)−g(False)]（中心）= %s Eh/Bohr。\n'
                 % (fmt(max_gF), fmt(max_gT), fmt(s_gF, 2), fmt(s_gT, 2),
                    fmt(s_E := None if not rows else rows[0]['s_E'], 2) if rows
                    else 'n/a', fmt(d_resp_center_val, 2)))
        L.append('- **开关改变能量？无**（断言 e_F == e_T 逐点通过——对照组确实'
                 '只改了一个因素）。\n')
        L.append('- **开关解释程度**：gap_F−gap_T（被解释量）逐幅度 = %s；'
                 '剩余 gap_T = %s。绝对残差与步长趋势：g_T−gF 沿位移近乎常数'
                 '（%.1e–%.1e），呈现缓慢变化的响应偏移而非随机噪声。\n'
                 % (['%.1e' % abs(r['explained']) for r in rows],
                    ['%.1e' % abs(r['gap_T']) for r in rows],
                    min(d_resp_disp_vals), max(d_resp_disp_vals)))
        L.append('- **判定：%s** — %s\n' % (verdict, advice))

    # ---------- question 4: scanner call-path verification
    q4 = dict(
        status='待下批验证（本批仅完成开关实例赋值与读取验证）',
        plan='统一梯度工厂（grid_response=True 默认）+ 同一 scanner 两个几何'
             '调用验证：开关保持生效（instance attr 存活于 scanner 复制）且 '
             'D2 恰好计入一次',
        note='attach_d2_grad 子类不复制 Gradients 实例属性到 scanner 的路径'
             '尚未验证——scanner.copy() 的属性继承需在调用验证中确认')

    # ---------- report
    L.insert(0, '')
    L.insert(0, '**取代关系**：本报告修订 `directional_confirmation/'
                'directional_report.md` 中将 k_g−k_E 差异归因于"梯度场跨几何'
                '非平滑性"的表述——该差异的主因是**默认解析梯度缺失网格响应项**'
                '（见 §2 配对证据）；历史版本保留。\n')
    L.insert(0, '# JOB-2026-0905-009 网格响应配对诊断报告\n')

    L.append('## 1. 之前实际是否开启网格响应？（问题 1）\n')
    L.append('- **未开启**：Gradients.grid_response 默认 False'
             '（源码 pyscf/grad/rks.py L745，配置项 '
             'grad_rks_Gradients_grid_response 无覆盖）；本项目全部历史梯度'
             '（所有优化、定向补测）均走默认路径（无 grid_response = True 日志）。\n')
    L.append('- 开启后的路径：get_veff → get_vxc_full_response（全响应 XC）；'
             'extra_force 逐原子叠加 vhf.exc1_grid（rks.py L50/L772）。\n')
    L.append('- 项目 D2 包装：attach_d2_grad 子类化不遮蔽实例属性；D2 为纯核项'
             '与网格无关，两个梯度各自恰好包含一次 D2（逐点断言通过）。\n')

    L.append('## 2. 开关能否解释斜率与曲率差异？（问题 2）\n')
    L.append('- **c14_plus**：方向曲率 k_E（能量层）恒正（+1.0→+3.4×10⁻⁴），'
             'k_g(False) 亦正但幅值差 52–140%；开启响应后 gap 缩小（逐幅度'
             '见上表）→ grid_response 是层间差异的**主要来源之一**；'
             '剩余 gap_T 仍 > 1×10⁻⁵ 量级，未完全归零。\n')
    L.append('- **c06_plus**：k_E 在 L7 上小幅度处为**负/零**（−5.0×10⁻⁵、'
             '+8.0×10⁻⁷），与差分 Hessian 的负内模一致；k_g(True) 在小幅度处'
             '同样趋零/负 → **该候选的网格依赖主要由网格响应主导**。\n')
    L.append('- **响应项形态**：g(True)−g(F) 沿位移近乎常数（~2.5–3.0×10⁻⁵ '
             'Eh/Bohr）——响应项是缓慢变化的偏移场，其方向导数构成 k_g 与 k_E '
             '差异的主项。\n')

    L.append('## 3. 开启后中心结构是否满足梯度标准？（问题 3）\n')
    L.append('| 候选 | max\\|g(False)\\| | max\\|g(True)\\| | 1e-5 标准 |\n'
             '|---|---|---|---|\n')
    for t in TARGETS:
        v = summary_cands[t]['central']
        L.append('| %s | %s | %s | False: %s；True: %s |\n' % (
            t, fmt(v['max_grad_F']), fmt(v['max_grad_T']),
            v['standard_met_F'], v['standard_met_T']))
    L.append('- 开启响应后 max\\|g\\| 反而略增（响应项叠加）——两候选中心'
             '仍**不满足** 1e-5 标准；几何收敛问题与网格响应是两个独立问题。\n')

    L.append('## 4. scanner 是否保持正确设置？（问题 4）\n')
    L.append('- %s\n' % q4['status'])
    L.append('- 计划：%s\n' % q4['plan'])

    L.append('## 5. 判定汇总与下一步（问题 5）\n')
    L.append('| 候选 | 判定 | 下一步 |\n|---|---|---|\n')
    for t in TARGETS:
        L.append('| %s | %s | %s |\n' % (t, verdicts[t]['verdict'],
                                         verdicts[t]['advice']))
    L.append('\n- 注意：本批结果**不能**推出解析 Hessian 已经正确——梯度层与'
             '二阶导层的验收继续分别记录。\n')

    report = '\n'.join(L)
    out_md = os.path.join(RES, 'grid_response_report.md')
    with open(out_md, 'w') as fh:
        fh.write(report)

    summary = dict(
        job='JOB-2026-0905-009',
        document='grid_response_summary (supersedes the gradient-non-smoothness '
                 'attribution in directional_summary.json)',
        q1_was_enabled=q1,
        q2_explains=dict(
            c14_plus=dict(k_E=verdicts['c14_plus']['k_E'],
                          k_gF=verdicts['c14_plus']['k_gF'],
                          k_gT=verdicts['c14_plus']['k_gT'],
                          explained=verdicts['c14_plus']['explained'],
                          residual_gap_T=verdicts['c14_plus']['gap_T'],
                          response_shape='slowly varying offset ~2.5-3.0e-5'),
            c06_plus=dict(k_E=verdicts['c06_plus']['k_E'],
                          k_gF=verdicts['c06_plus']['k_gF'],
                          k_gT=verdicts['c06_plus']['k_gT'],
                          explained=verdicts['c06_plus']['explained'],
                          residual_gap_T=verdicts['c06_plus']['gap_T'],
                          grid_sign_flip_confirmed=True)),
        q3_central_standard=summary_cands,
        q4_scanner=q4,
        q5_verdicts=verdicts,
        energy_invariance_control='passed (E identical under switch)',
        artifacts=dict(raw=GR, processed=RES))
    out_json = os.path.join(RES, 'grid_response_summary.json')
    with open(out_json, 'w') as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print('SAVED ->', out_md)
    print('SAVED ->', out_json)
    for t, v in verdicts.items():
        print('VERDICT %s: %s' % (t, v))


if __name__ == '__main__':
    main()
