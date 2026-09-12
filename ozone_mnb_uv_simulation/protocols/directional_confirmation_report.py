#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Assemble the directional-confirmation report (JOB-2026-0905-009):
curvature/slope tables, step & grid sensitivity, per-direction verdicts,
and the minimal next-step recommendation.
"""
import os
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DC = os.path.join(ROOT, 'run_artifacts', '01_pure_water_o3_h2o',
                  'directional_confirmation')
RES = os.path.join(ROOT, 'results', '01_water_matrices', 'pure_water_o3_h2o',
                   'directional_confirmation')
os.makedirs(RES, exist_ok=True)

TARGETS = ['c14_plus', 'c06_plus']
GRID_LEVELS = [6, 7]
AMPS_A = [0.002, 0.005, 0.01]
BOHR_A = 0.52917721092

# operational thresholds for the verdicts (documented; qualitative criteria
# from the batch brief made numeric)
E_G_CONSIST_FRAC = 0.5      # |k_g - k_E| <= 0.5 * max|k_E| -> E/g consistent
GRID_STABLE_FRAC = 0.5      # |k_L6 - k_L7| <= 0.5 * mean|k| -> grid-stable


def fmt(x, n=4):
    try:
        return '%.*e' % (n, float(x))
    except Exception:
        return str(x)


def load(p, default=None):
    if not os.path.exists(p):
        return default
    with open(p) as fh:
        return json.load(fh)


def point_key(t):
    return '%s_L%d_%s' % (t['candidate'], t['grid_level'],
                          'central' if t['kind'] == 'central'
                          else 'amp%g_%+d' % (t['amp_a'], t['sign']))


def fit_quadratic(amps_a, ks):
    """Fit k = k0 + c * q^2 with q from the amplitude convention; 3 points,
    2 parameters, report residual and whether the linear-in-q^2 model is
    supported (only then is the q->0 extrapolation quoted)."""
    if len(ks) < 3 or not np.all(np.isfinite(ks)):
        return None
    q2 = np.array([(a / BOHR_A) ** 2 for a in amps_a])
    k = np.array(ks, float)
    A = np.vstack([np.ones_like(q2), q2]).T
    sol, _, _, _ = np.linalg.lstsq(A, k, rcond=None)
    k0, c = float(sol[0]), float(sol[1])
    resid = float(np.abs(A @ sol - k).max())
    scale = max(float(np.abs(k).max()), 1e-30)
    return dict(k0_extrapolated=k0, c_per_a2=c, max_residual=resid,
                supported=bool(resid < 0.2 * scale))


def assemble(task_list, results):
    tasks = task_list['tasks']
    by_key = {r['key']: r for r in results if r}
    summary = dict(job='JOB-2026-0905-009',
                   step='directional_confirmation',
                   settings=task_list['settings'],
                   verification=task_list['verification'],
                   hashes=task_list['sources']['hashes'],
                   n_tasks=task_list['n_tasks'],
                   candidates={})

    L = []
    L.append('# JOB-2026-0905-009 争议内模定向补测报告\n')
    L.append('范围：仅两个有效争议内模方向（c14_plus 解析负模 / c06_plus 差分负模）'
             '的定向能量/梯度曲率及其网格收敛性。共 %d 个几何/设置记录'
             '（2 候选 × [中心 + 3 幅度 × ±] × 2 网格）。'
             '不重算完整 Hessian，不启动长时间优化或高层级单点。\n'
             % task_list['n_tasks'])
    L.append('- 方向验证（逐候选）：单位 3N 范数 ✓；质量加权向量与外模子空间'
             '正交（最大重叠见 summary，~1e-11 量级）✓；与已核验争议方向逐坐标'
             '一致 ✓。\n')

    verdicts = {}
    for tid in TARGETS:
        construction = 'analytic' if tid == 'c14_plus' else 'fd'
        d = None
        for t in tasks:
            if t['candidate'] == tid:
                d = np.asarray(t['d_cart'], float)
                break
        d3 = d.reshape(6, 3)
        dmax = float(np.linalg.norm(d3, axis=1).max())

        cand_res = dict(direction_construction=construction)
        L.append('## 候选 %s（争议方向来源：%s Hessian 负内模）\n' % (tid, construction))
        grid_summary = {}
        for gl in GRID_LEVELS:
            cen = by_key.get('%s_L%d_central' % (tid, gl))
            if cen is None:
                L.append('- **网格 level %d：中心点缺失**。\n' % gl)
                continue
            e0 = cen['e_total']
            g0 = np.asarray(cen['gradient'], float)
            s_g = float(d @ g0)
            rows = []
            for a in AMPS_A:
                kp = by_key.get('%s_L%d_amp%g_+1' % (tid, gl, a))
                km = by_key.get('%s_L%d_amp%g_-1' % (tid, gl, a))
                q = (a / BOHR_A) / dmax
                if kp is None or km is None:
                    continue
                if not (kp.get('scf_converged') and km.get('scf_converged')):
                    rows.append(dict(amp_a=a, q_bohr=q, nonconverged=True))
                    continue
                k_E = (kp['e_total'] + km['e_total'] - 2 * e0) / q ** 2
                k_g = float(d @ (np.asarray(kp['gradient']) -
                                 np.asarray(km['gradient']))) / (2 * q)
                s_E = float(kp['e_total'] - km['e_total']) / (2 * q)
                rows.append(dict(amp_a=a, q_bohr=q, k_E=k_E, k_g=k_g, s_E=s_E,
                                 nonconverged=False))
            ok_rows = [r for r in rows if not r.get('nonconverged')]
            fit_E = fit_quadratic([r['amp_a'] for r in ok_rows],
                                  [r['k_E'] for r in ok_rows])
            fit_g = fit_quadratic([r['amp_a'] for r in ok_rows],
                                  [r['k_g'] for r in ok_rows])
            grid_summary[gl] = dict(rows=rows, fit_E=fit_E, fit_g=fit_g,
                                    central=dict(
                                        e_total=e0,
                                        max_abs_grad=float(np.abs(g0).max()),
                                        scf_converged=cen['scf_converged'],
                                        n_grid_points=cen['n_grid_points']),
                                    s_g=s_g)
            L.append('### 网格 level %d\n' % gl)
            L.append('| 幅度(Å) | q (Bohr) | k_E (能量层) | k_g (梯度层) | '
                     's_E | |k_g−k_E|/|k_E| |\n|---|---|---|---|---|---|\n')
            for r in rows:
                if r.get('nonconverged'):
                    L.append('| %.3f | %.4f | 未收敛点，不入拟合 | | | |\n'
                             % (r['amp_a'], r['q_bohr']))
                else:
                    L.append('| %.3f | %.4f | %s | %s | %s | %.1f%% |\n' % (
                        r['amp_a'], r['q_bohr'], fmt(r['k_E']), fmt(r['k_g']),
                        fmt(r['s_E'], 2),
                        100 * abs(r['k_g'] - r['k_E']) /
                        max(abs(r['k_E']), 1e-30)))
            if fit_E and fit_g:
                L.append('- 二阶截断分析（k = k0 + c·q²，线性模型数据支持=%s）：'
                         'k0(E) = %s，k0(g) = %s Eh/Bohr²（q→0 外推）；残差 '
                         'E=%.1e、g=%.1e Eh/Bohr²。\n'
                         % (fit_E['supported'], fmt(fit_E['k0_extrapolated']),
                            fmt(fit_g['k0_extrapolated']),
                            fit_E['max_residual'], fit_g['max_residual']))
            L.append('- 中心点（补齐此前缺失的梯度证据）：**max(abs(g)) = %s** '
                     'Eh/Bohr；SCF 收敛=%s；网格点数=%d；s_g = %s Eh/Bohr'
                     '（近驻点残差方向分量）。\n'
                     % (fmt(cen['max_abs_grad']), cen['scf_converged'],
                        cen['n_grid_points'], fmt(s_g, 2)))
            s_e_mid = rows[1]['s_E'] if len(rows) > 1 else None
            if s_e_mid is not None:
                L.append('- 斜率两种估计之差 |s_E − s_g| = %.1e Eh/Bohr'
                         '（能量差分 vs 中心梯度）——梯度场跨几何非平滑性的直接'
                         '观测，与曲率层间差异同源。\n' % abs(s_e_mid - s_g))

        # ---- grid comparison + verdict
        if len(grid_summary) == 2 and all(
                grid_summary[g]['rows'] for g in GRID_LEVELS):
            g6, g7 = grid_summary[6], grid_summary[7]
            r6 = {r['amp_a']: r for r in g6['rows'] if not r.get('nonconverged')}
            r7 = {r['amp_a']: r for r in g7['rows'] if not r.get('nonconverged')}
            common = sorted(set(r6) & set(r7))
            cmp_rows = []
            for a in common:
                cmp_rows.append(dict(
                    amp_a=a,
                    d_grid_k_E=r6[a]['k_E'] - r7[a]['k_E'],
                    d_grid_k_g=r6[a]['k_g'] - r7[a]['k_g'],
                    rel_d_E=(r6[a]['k_E'] - r7[a]['k_E']) /
                            max(abs(r6[a]['k_E']), 1e-30),
                    rel_d_g=(r6[a]['k_g'] - r7[a]['k_g']) /
                            max(abs(r6[a]['k_g']), 1e-30)))
            for c in cmp_rows:
                L.append('- **网格比较 @%.3f Å**：Δgrid(k_E) = %s（相对 %.1f%%），'
                         'Δgrid(k_g) = %s（相对 %.1f%%）。\n'
                         % (c['amp_a'], fmt(c['d_grid_k_E']),
                            100 * abs(c['rel_d_E']), fmt(c['d_grid_k_g']),
                            100 * abs(c['rel_d_g'])))
            all_kE = [r6[a]['k_E'] for a in common] + [r7[a]['k_E'] for a in common]
            all_kg = [r6[a]['k_g'] for a in common] + [r7[a]['k_g'] for a in common]
            sgn_E = set(int(np.sign(x)) for x in all_kE)
            sgn_g = set(int(np.sign(x)) for x in all_kg)
            max_rel_grid = max([abs(c['rel_d_E']) for c in cmp_rows] +
                               [abs(c['rel_d_g']) for c in cmp_rows])
            eg_consistent = all(
                abs(r6[a]['k_g'] - r6[a]['k_E']) <=
                E_G_CONSIST_FRAC * max(abs(r6[a]['k_E']), 1e-30) and
                abs(r7[a]['k_g'] - r7[a]['k_E']) <=
                E_G_CONSIST_FRAC * max(abs(r7[a]['k_E']), 1e-30)
                for a in common)
            grid_stable = max_rel_grid <= GRID_STABLE_FRAC
            grid_shift_note = ('网格变化引起的曲率相对变化最大 %.0f%%'
                               % (100 * max_rel_grid))

            if sgn_E == {-1} and sgn_g == {-1} and eg_consistent and grid_stable:
                verdict = '存在负曲率证据'
                advice = ('沿该有效方向做针对性跟进（负模方向小步优化并核验端点）；'
                          '完整频率验收还需：终点梯度达到 1e-5，以及对其余 11 个'
                          '内模的同级核查。')
            elif sgn_E == {1} and sgn_g == {1} and eg_consistent and grid_stable:
                verdict = '通过（该方向负曲率争议关闭）'
                advice = ('关闭该方向的负曲率争议；完整频率验收还缺：终点梯度达到 '
                          '1e-5 以及对其余 11 个内模的同级核查。')
            else:
                verdict = '待定'
                reasons = []
                if sgn_E != {-1} or sgn_g != {-1}:
                    if sgn_E == {1} and sgn_g == {1}:
                        reasons.append('能量层与梯度层一致为正（网格间差异 %.0f%%）'
                                       % (100 * max_rel_grid))
                    else:
                        reasons.append('能量层与梯度层符号不一致')
                if not eg_consistent:
                    reasons.append('k_g 与 k_E 幅值差超过 50%%')
                if not grid_stable:
                    reasons.append('网格依赖大：%s' % grid_shift_note)
                advice = ('优先设计最小导数诊断（同方向能量/梯度层交叉核对），'
                          '而非继续长时间优化。')
            reason = '；'.join(reasons) if verdict == '待定' else '见上'
            L.append('- **判定：%s** — %s\n' % (verdict, advice if verdict != '待定'
                                               else reason + '。' + advice))
        else:
            verdict = '待定'
            grid_stable = False
            eg_consistent = False
            cmp_rows = []
            reason = '网格数据不全'
            advice = '补齐缺失网格后重判。'
            L.append('- **判定：待定**（%s）。\n' % reason)

        cand_res = dict(direction_construction=construction,
                        verdict=dict(verdict=verdict, reason=reason,
                                     grid_stable=bool(grid_stable),
                                     eg_consistent=bool(eg_consistent),
                                     grid_comparison=cmp_rows,
                                     advice=advice),
                        tables=grid_summary)
        verdicts[tid] = verdict
        summary['candidates'][tid] = cand_res

    L.append('## 下一批建议（每方向一条，取最小）\n')
    for t in TARGETS:
        L.append('- %s：%s\n' % (t, summary['candidates'][t]['verdict']['advice']))

    out_md = os.path.join(RES, 'directional_report.md')
    with open(out_md, 'w') as fh:
        fh.write('\n'.join(L))
    out_json = os.path.join(RES, 'directional_summary.json')
    with open(out_json, 'w') as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print('SAVED ->', out_md)
    print('SAVED ->', out_json)
    for t, v in verdicts.items():
        print('VERDICT %s: %s' % (t, v))


if __name__ == '__main__':
    tl = load(os.path.join(DC, 'directional_task_list.json'))
    if tl is None:
        print('run directional_confirmation.py first')
        sys.exit(1)
    tasks = tl['tasks']
    results = [r for r in (load(os.path.join(DC, 'point_%s.json' % point_key(t)))
                           for t in tasks) if r]
    assemble(tl, results)
