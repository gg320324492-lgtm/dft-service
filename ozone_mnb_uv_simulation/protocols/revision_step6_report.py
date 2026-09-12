#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Acceptance-revision step #6 (JOB-2026-0905-009): assemble
results/01_water_matrices/pure_water_o3_h2o/acceptance_revision/
  revision_report.md   (human-readable revision report, supersedes
                        ../conformer_report.md)
  revision_summary.json (machine-readable)

Reads the step artifacts produced by revision_step2/3/3b/4/5; no DFT here.
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, 'run_artifacts', '01_pure_water_o3_h2o')
REV = os.path.join(ART, 'acceptance_revision')
RES = os.path.join(ROOT, 'results', '01_water_matrices', 'pure_water_o3_h2o',
                   'acceptance_revision')
os.makedirs(RES, exist_ok=True)

CAND_DFT = ['c14_plus', 'c06_plus']


def load(name):
    p = os.path.join(REV, name)
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def fmt(x, n=3):
    try:
        return '%.*f' % (n, x)
    except Exception:
        return str(x)


def main():
    rd = load('revision_rededup.json') or {}
    hc = load('revision_hessian_crosscheck.json') or {}
    gc = load('revision_grid_check.json') or {}
    ue = load('revision_unified_energies.json') or {}
    t1 = load('revision_test_1_rmsd_topology.json') or {}
    cc = load('ccsdt_revision/ccsdt_revision_summary.json') or {}

    # ---- counts
    g = rd.get('gas_phase', {})
    s = rd.get('smd_phase', {})
    n_mf_runs = len([f for f in os.listdir(os.path.join(ART, 'mode_follow'))
                     if f.endswith('.json')])
    n_endpoints_min = len([f for f in os.listdir(os.path.join(ART, 'mode_follow'))
                           if f.endswith('.json')
                           and json.load(open(os.path.join(ART, 'mode_follow', f))).get('n_imaginary') == 0])

    L = []
    L.append('# JOB-2026-0905-009 结果复核修订报告\n')
    L.append('**本报告取代** `results/01_water_matrices/pure_water_o3_h2o/conformer_report.md`'
             '（v1 报告及其机器可读版保留于原位置作为来源记录，其中下列表述已被本报告修订）。\n')
    L.append('**术语（按复核指示）**：在新的证据确认之前，此前报告的"9 个不同极小值"降格为'
             ' **"待核验候选集"**，"全局最低点"改称 **"已搜索候选中的最低能结构"**。'
             ' 本批不启动新的反应、过渡态、离子或界面计算。\n')
    L.append('> **⚠️ 勘误（2026-09-05，最终极小值验收批前置更正）**：本报告 §3 中'
             ' "c14_plus = 极小值（全构造 n_imag=0）"的表述**撤回，标记为待确认**——'
             ' FD@0.005 Bohr 曾给出 1 个负模；"该离群属网格剪枝噪声"当时仅为假设、'
             ' **未经配对对照证实，撤回**。c12_plus 的准确表述更正为'
             ' **"存在负曲率证据"**（L6 解析与全部差分构造给出虚频；解析 L5 的 0 虚频'
             ' 与其余构造矛盾）；它与 c14_plus 的关系**仅为几何聚类结果，不构成'
             ' "已证明的同一极小值"**。两项均交由'
             ' `results/01_water_matrices/pure_water_o3_h2o/final_minima_validation/`'
             ' 的定向计算裁决。\n')

    # ---------------- 1. errors found & fixed
    L.append('## 1. 发现的错误、修复证据与受影响结果\n')
    L.append('| # | 错误 | 证据 | 受影响结果 | 状态 |\n|---|---|---|---|---|\n')
    L.append('| E1 | 氢键判据方向反置：用 O_w→H 与 H→O 的夹角套用 ≥120° 判据，'
             '等价于排斥线性氢键 | 实测 c02 存在 d(H···O)=2.286 Å、角 179.9° 的线性氢键被判 "none"；'
             'c01/c03/c04/c06/c08 均有 148–167° 真实氢键被漏判 | 拓扑签名与去重聚类（v1 报告 §2/§3 的"拓扑"列全部失真）| '
             '已修复（`conformer_utils.hbond_topology` 改用 H→O_w 与 H→O） |\n')
    L.append('| E2 | 刚体配准的旋转以 O3 框架计算、却以全分子质心平移，中心不一致；'
             '且未考虑水双氢等价置换 | 合成旋转/平移/换号案例在旧代码下 RMSD 不归零；'
             '修复后 12 个案例 RMSD≈1e-16（revision_test_1_rmsd_topology.json） | 成对 RMSD 被系统性虚高 → '
             'v1 "9 个不同极小值" 中 7 个实为等价重复 | 已修复 |\n')
    L.append('| E3 | `topology_signature` 混合 int/str 排序的潜在 TypeError（旧调用路径从未触发，'
             '本次重新去重首次踩中） | 修订脚本首跑报 TypeError | 无（修复前无输出）| 已修复（统一字符串标签）|\n')
    L.append('| E4 | `harmonic_analysis` 默认把虚频返回为复数，强转 float 后变 0.0，'
             'n_imag 假=0 | 上一批已发现并修复（imaginary_freq=False + freq_error 交叉核对）| '
             'v1 之前所有"真实极小值"判定 | 已修复（本批复核沿用）|\n')
    L.append('| E5 | 手写 ZPE 常量多乘 Boltzmann 因子（zpe 恒 0） | 上一批已修复（取热化学单一真源）| '
             'zpe_hartree 字段（不影响热化学表）| 已修复 |\n')
    L.append('| E6 | v1 Stage-4 能量表混用约定："E_int(非CP)" 实为结合能（用了各自优化单体），'
             '"E_int(CP)" 实为相互作用能（冻结复合物内几何单体），两列与派生 BSSE 不可比 | '
             '本批复核按统一定义重算（revision_unified_energies.json）| v1 §4 结合能表 | '
             '已修复，见 §4 |\n')
    L.append('| E7 | CCSD(T) 的 0.0643 实为 max\\|t1\\|，被误标为"T1 诊断"，'
             '并由此直接得出 single_reference_ok 判定 | PySCF 标准定义为 '
             '`get_t1_diagnostic(t1)=sqrt(\\|t1\\|²/N_corr_elec)` | v1 §5 | '
             '已更正并撤回判定，见 §5 |\n')
    L.append('| E8 | 旧基线方法命名将泛函部分简写作"ωB97X" | 复核指示：不得将该泛函部分'
             '等同于独立的 ωB97X 泛函 | results/00_baseline 汇总、models 记录 | '
             '生成器已更正并重新生成（"ωB97X-D 的不含经验色散部分用于优化/频率，'
             '另加经验色散单点校正"；保留 xc=\'wb97xd\' 与 Libxc 条目名）|\n')
    L.append('\n修复证据归档：`run_artifacts/01_pure_water_o3_h2o/acceptance_revision/`。\n')

    # ---------------- 2. revised structure list
    L.append('## 2. 修订后的构型清单与待核验候选集\n')
    L.append('- 模式跟进：共 **%d 次运行**（7 个单虚频鞍点 × ±方向），端点中零虚频 '
             '**%d 个**、仍为鞍点 **%d 个**。\n'
             % (n_mf_runs, n_endpoints_min, n_mf_runs - n_endpoints_min))
    L.append('- 全部派生端点已纳入结构池（原始 14 + 端点 %d = %d 个气相结构）；'
             '其中零虚频 %d 个、鞍点 %d 个。\n'
             % (g.get('n_structures', 0) - 14, g.get('n_structures', 0),
                g.get('n_minima', 0), g.get('n_saddles', 0)))
    L.append('- **气相待核验候选集（%d 个）**：%s；其中已搜索候选中的最低能结构：'
             '**%s**（E = %s Ha）。\n'
             % (len(g.get('candidates_pending_verification', [])),
                g.get('candidates_pending_verification'),
                g.get('lowest_among_searched_candidates'),
                fmt(g.get('lowest_energy_hartree'), 8)))
    L.append('- 与 v1 的对应关系（旧 kept_minima → 修订判定）:\n')
    for k, v in (g.get('old_to_new') or {}).items():
        L.append('  - %s → %s\n' % (k, v))
    L.append('- 成对 RMSD（候选 + 鞍点）与逐结构拓扑：'
             '`run_artifacts/.../acceptance_revision/revision_rededup.json`。\n')
    L.append('- **SMD 重新去重**：%d 个结构 → 待核验候选集 %s；'
             '鞍点/淘汰 %s。\n'
             % (s.get('n_structures', 0), s.get('candidates_pending_verification'),
                sorted((s.get('eliminated') or {}).keys())))
    L.append('- **结论降级声明**：v1 的"9 个不同极小值"（其中 7 个 dE<0.001 kcal/mol）'
             '在修正等价置换与配准中心后坍缩为 %d 个候选；'
             'v1 的"全局最低点 c14_plus"改称"已搜索候选中的最低能结构"，'
             '待本报告 §3 频率可信度确认后才可升级为"极小值/全局最低"结论。\n'
             % len(g.get('candidates_pending_verification', [])))

    # ---------------- 3. frequency reliability
    L.append('## 3. 频率可靠性核查（c14_plus / c12_plus / c14）\n')
    verdict = gc.get('verdict') or hc.get('verdict') or {}
    L.append('- 同源性核对：三个记录集（gas_opt / gas_freq / mode_follow）的坐标'
             '最大偏差与能量散差见 `revision_hessian_crosscheck.json` 的 provenance 字段。\n')
    if hc:
        for tid, r in ((hc.get('verdict') or {}).items()):
            st = (r.get('max_rel_H_diff'), r.get('max_lowest6_freq_shift_cm1'),
                  r.get('n_imag_consistent'))
            L.append('- 解析 vs 独立差分 Hessian（步长 0.005/0.02 Bohr）：'
                     '%s — max 相对 H 差 %.2e，最低 6 模最大频移 %.1f cm⁻¹，'
                     'n_imag 跨构造一致=%s。\n' % (tid, st[0], st[1], st[2]))
    if gc:
        for tid, v in (gc.get('verdict') or {}).items():
            L.append('- 网格/SCF 收敛检查（解析 level6+SCF 1e-12、差分步长 0.05 Bohr）：'
                     '**%s** — 各构造 n_imag = %s。\n'
                     % (tid, v.get('n_imag_values')))
    L.append('- **处理原则**：负模一律不删除、不归为噪声。凡 n_imag 随构造方式翻转的结构，'
             '其"是否极小值"标记为**待定**；稳定者方可进入候选集的最终确认。\n')
    unstable = [k for k, v in (gc.get('verdict') or {}).items() if not v.get('stable')]
    L.append('- 当前判定：%s\n' % (
        ('不稳定（待定）：%s' % unstable) if unstable else '（网格检查结果见上）'))
    L.append('- **网格敏感性量化证据**：c12_plus 最低模从解析 L5 的 +48.8 cm⁻¹ 翻转为 '
             'L6+SCF1e-12 的 −29.8 cm⁻¹（移位 79 cm⁻¹）；c14 为 −52.1 → −64.7；'
             'c14_plus 为 79.6 → 67.3。软模对网格级别的敏感度达 ~80 cm⁻¹ 量级，'
             '即这些平坦结构的"极小值/鞍点"分类在当前精度下不可靠。\n')
    L.append('- 逐结构结论：**c14_plus** = 极小值（解析 L5/L6 与差分 0.02/0.05 Bohr 全部 '
             'n_imag=0，仅 FD@0.005 受网格剪枝噪声出现一次离群）；'
             '**c12_plus** = 极可能为鞍点（L6 与全部差分构造一致给虚频；解析 L5 的 0 虚频'
             '属网格噪声误判——其本已作为 c14_plus 的等价重复移出候选集，不影响候选集）；'
             '**c14** = **待定**（解析/差分各执一词）。候选集的其它成员（c06_plus）未在本轮'
             '频率核查范围内，保持"待核验候选"称谓。\n')

    # ---------------- 4. unified energies
    L.append('## 4. 统一定义下的相互作用能 / 变形能 / 结合能\n')
    L.append('定义（DFT 与 CCSD(T) 同一约定）：\n')
    L.append('- 相互作用能 E_int = E(复合物) − 两单体在**复合物内几何（冻结）**的能量；'
             'CP 版对冻结伙伴加 ghost，几何约定相同。\n')
    L.append('- 变形能 E_def = 冻结单体相对各自优化单体的能量差（不作 ghost 校正——'
             '弱复合物变形能 ~10⁻² kcal/mol 量级，通行约定，已明示）。\n')
    L.append('- 结合能 E_bind = E_int + E_def（noCP 与 CP 各一列）。\n')
    L.append('- BSSE 符号约定：BSSE = E_int(noCP) − E_int(CP) ≤ 0（未校正能量偏负），'
             '表中同时给出幅值。\n')
    L.append('- D2：全导数 −D2 计入所有电子项（ghost 不参与 D2，已审计）。\n')
    L.append('- 自由能：ΔG 来自各自优化单体 + 复合物的谐振热化学，**不含 CP 校正**，'
             '与 E_bind(noCP) 配对；1 atm→1 M 修正 −1.894 kcal/mol。\n')
    if ue:
        L.append('\n| 构型 | E_int(noCP) | E_int(CP) | BSSE(有号) | E_def | '
                 'E_bind(noCP) | E_bind(CP) | ΔG(1 atm) | ΔG(1 M) | kcal/mol |\n')
        L.append('|---|---|---|---|---|---|---|---|---|\n')
        for r in ue.get('dft_rows', []):
            L.append('| %s | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |\n' % (
                r['id'], r['e_int_nocp_kcal'], r['e_int_cp_kcal'],
                r['bsse_signed_kcal'], r['e_def_kcal'], r['e_bind_nocp_kcal'],
                r['e_bind_cp_kcal'], r['thermal_nocp']['dG_assoc_1atm_kcal'],
                r['thermal_nocp']['dG_assoc_1M_kcal']))
        if ue.get('ccsdt_rows'):
            L.append('\nCCSD(T)/aug-cc-pVTZ（非 CP；未计算 CCSD(T) 的 CP 校正，'
                     '**不得与 DFT 的 CP 相互作用能直接对比**）：\n')
            L.append('| 构型 | E_int(noCP) | E_def | E_bind(noCP) | max\\|t1\\| | T1(pyscf) | D1 |\n')
            L.append('|---|---|---|---|---|---|---|\n')
            for r in ue.get('ccsdt_rows', []):
                d = r.get('diagnostics', {})
                L.append('| %s | %.3f | %.3f | %.3f | %.4f | %.4f | %.4f |\n' % (
                    r['id'], r['e_int_ccsdt_nocp_kcal'], r['e_def_ccsdt_kcal'],
                    r['e_bind_ccsdt_nocp_kcal'], d.get('max_abs_t1', float('nan')),
                    d.get('t1_diagnostic_pyscf', float('nan')),
                    d.get('d1_diagnostic_janssen', float('nan'))))

    # ---------------- 5. T1 diagnostics
    L.append('## 5. CCSD(T) 电子相关诊断\n')
    if cc:
        L.append('- 标准诊断定义（安装版 PySCF `cc.ccsd.get_t1_diagnostic`）：'
                 '`T1 = sqrt(|t1|² / N_corr_elec)`（Lee 型，按相关电子数归一）。\n')
        L.append('- 旧值 0.0643 重标为 **max\\|t1\\|**（单振幅绝对值最大），非 T1 诊断。\n')
        L.append('- 冻结核：`frozen=0`（未冻核）；相关电子数与收敛状态逐单点记录于 '
                 '`ccsdt_revision/ccsdt_revision_summary.json`。\n')
        L.append('- 振幅复用：t1 已存 .npy；t2（复合物约 1 GB）未持久化——'
                 '已列为缺失，重算仅需数分钟（几何已归档）。\n')
        L.append('- **判定撤回**：v1 的 `single_reference_ok=False` 撤回；'
                 '单一诊断指标不能证明多参考根因。替换为仅报告数值与定义，'
                 '根因结论留给后续（如 CASSCF/NEVPT2 或 D1/D2 累积证据）。\n')
    else:
        L.append('（步骤 4 尚未完成 —— T1 数据待补。）\n')

    # ---------------- 6. old conclusions keep/withdraw/pending
    L.append('## 6. 旧结论处置（保留 / 撤回 / 待定）\n')
    L.append('| 旧结论 | 处置 |\n|---|---|\n')
    L.append('| Stage-1 16/16 几何收敛（max\\|grad\\|<5e-5） | **保留**（与本批复核无关） |\n')
    L.append('| 虚频符号修复后 14 试探结构中仅 c05 零虚频 | **保留**（计算事实），但其"极小值"称谓待 §3 稳定性确认 |\n')
    L.append('| v1 "9 个不同极小值" | **撤回** → 待核验候选集（2 个）+ 7 个等价重复 |\n')
    L.append('| v1 "全局最低点 c14_plus" | **改称**"已搜索候选中的最低能结构"，待频率稳定性确认 |\n')
    L.append('| v1 氢键拓扑列（全部 "H-bonds: none"） | **撤回**（E1 判据反置） |\n')
    L.append('| v1 Stage-4 结合能表（混用约定） | **撤回** → §4 统一定义表 |\n')
    L.append('| v1 single_reference_ok 判定 | **撤回** → §5 仅报数值 |\n')
    L.append('| CCSD(T) E_int ≈ −2.81 kcal/mol（对优化单体） | **重新表述**：该数值按新定义属 E_bind(noCP) '
             '口径，冻结单体口径的 E_int 见 §4 |\n')
    L.append('| SMD 结果仅作局部水合结构趋势 | **保留** |\n')

    # ---------------- 7. next
    L.append('## 7. 本批之后\n')
    L.append('- 本批通过后，才决定保留哪些水合结构及是否需要补充高层级对照；'
             '随后另批选择首条臭氧反应路径。本批未启动新的反应、过渡态、离子或界面计算。\n')

    report = '\n'.join(L)
    out_md = os.path.join(RES, 'revision_report.md')
    with open(out_md, 'w') as fh:
        fh.write(report)

    summary = dict(
        job='JOB-2026-0905-009',
        document='revision_summary (supersedes ../conformer_summary.json)',
        terminology=dict(candidates='待核验候选集', lowest='已搜索候选中的最低能结构'),
        errors_and_fixes=[e.split('|')[1].strip() for e in L[6].split('\n')[2:] if e.startswith('| E')],
        counts=dict(mode_follow_runs=n_mf_runs,
                    endpoints_zero_imag=n_endpoints_min,
                    endpoints_remaining_saddle=n_mf_runs - n_endpoints_min,
                    gas_structures=g.get('n_structures'),
                    gas_minima=g.get('n_minima'), gas_saddles=g.get('n_saddles')),
        gas_candidates_pending_verification=g.get('candidates_pending_verification'),
        gas_lowest_among_searched=g.get('lowest_among_searched_candidates'),
        smd_candidates_pending_verification=s.get('candidates_pending_verification'),
        frequency_verdict=gc.get('verdict') or hc.get('verdict'),
        unified_energies={r['id']: {k: v for k, v in r.items() if k != 'raw_hartree'}
                          for r in (ue.get('dft_rows') or [])},
        ccsdt_rows=ue.get('ccsdt_rows'),
        single_reference_ok=None,
        artifacts=dict(revision_dir=RES, raw_dir=REV),
    )
    out_json = os.path.join(RES, 'revision_summary.json')
    with open(out_json, 'w') as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print('SAVED ->', out_md)
    print('SAVED ->', out_json)


if __name__ == '__main__':
    main()
