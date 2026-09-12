#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Assemble the machine-readable conformer_summary.json and the human-readable
conformer_report.md for JOB-2026-0905-009 from the Stage 1-5 artifacts.

Reads only; writes to results/01_water_matrices/pure_water_o3_h2o/.
"""
import os
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, 'run_artifacts', '01_pure_water_o3_h2o')
RES = os.path.join(ROOT, 'results', '01_water_matrices', 'pure_water_o3_h2o')
os.makedirs(RES, exist_ok=True)

GAS_OPT = os.path.join(ART, 'gas_opt')
GAS_FREQ = os.path.join(ART, 'gas_freq')
SMD_FREQ = os.path.join(ART, 'smd_freq')

CONF_IDS = ['c%02d' % i for i in range(1, 15)]


def load(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path) as fh:
        return json.load(fh)


def load_gas_opt():
    recs = {}
    if not os.path.isdir(GAS_OPT):
        return recs
    for fn in sorted(os.listdir(GAS_OPT)):
        if not fn.endswith('.json') or fn == 'stage1_summary.json':
            continue
        with open(os.path.join(GAS_OPT, fn)) as fh:
            recs[fn[:-5]] = json.load(fh)
    return recs


def fmt(x, n=4):
    try:
        return round(float(x), n)
    except Exception:
        return x


def build_summary():
    gas = load_gas_opt()
    s2 = load(os.path.join(ART, 'stage2_minima.json'), {})
    s2tab = load(os.path.join(ART, 'stage2_conformer_table.json'), {})
    s3 = load(os.path.join(ART, 'stage3_smd.json'), {})
    bind = load(os.path.join(ART, 'binding.json'), {})
    ccsdt = load(os.path.join(ART, 'ccsdt.json'), {})

    # ---- gas-phase conformer table
    conformers = []
    for cid in CONF_IDS:
        g = gas.get(cid)
        if not g:
            continue
        fr = load(os.path.join(GAS_FREQ, '%s.json' % cid), {})
        tab = s2tab.get(cid, {})
        conformers.append(dict(
            id=cid,
            coverage=g.get('coverage'),
            rule=g.get('rule'),
            status=g.get('status'),
            e_total_hartree=fmt(g.get('e_total_hartree'), 8),
            rel_kcal=tab.get('rel_kcal'),
            n_imag=fr.get('n_imaginary'),
            lowest_freq=fr.get('lowest_frequency_cm1'),
            topology=tab.get('topology') or fr.get('topology'),
            is_minimum=fr.get('is_minimum'),
            kept=tab.get('kept'),
            eliminated_reason=tab.get('eliminated_reason'),
            optimised_xyz=g.get('optimised_coords_angstrom'),
        ))

    # ---- monomers
    monomers = {}
    for m in ('h2o', 'o3'):
        g = gas.get(m, {})
        fr = load(os.path.join(GAS_FREQ, '%s.json' % m), {})
        monomers[m] = dict(e_total_hartree=fmt(g.get('e_total_hartree'), 8),
                           n_imag=fr.get('n_imaginary'),
                           lowest_freq=fr.get('lowest_frequency_cm1'),
                           zpe_kcal_mol=fr.get('zpe_kcal_mol'),
                           topology=fr.get('topology'))

    # ---- SMD comparison
    smd_block = {}
    if s3:
        smd_block = dict(
            solvent_model=s3.get('solvent_model'),
            note=s3.get('note'),
            smd_minima=s3.get('smd_minima'),
            smd_eliminated=s3.get('smd_eliminated'),
            gas_vs_smd_ordering=s3.get('gas_vs_smd_ordering'),
            smd_freqs={cid: load(os.path.join(SMD_FREQ, '%s.json' % cid), {})
                       for cid in (s3.get('smd_minima') or [])})

    # ---- binding
    binding_block = {}
    if bind:
        binding_block = dict(
            standard_state_note=bind.get('standard_state_note'),
            ghost_d2_audit=bind.get('ghost_d2_audit'),
            rows=bind.get('binding'))

    # ---- CCSD(T)
    ccsdt_block = {}
    if ccsdt:
        ccsdt_block = dict(basis=ccsdt.get('basis'),
                           monomer_energies=ccsdt.get('monomer_energies'),
                           interaction=ccsdt.get('interaction'))

    summary = dict(
        job='JOB-2026-0905-009',
        title='O3.H2O local hydration conformers (pure water) - Phase 2A Part 2',
        method='wB97X-D / def2-TZVP, full-derivative -D2 via make_mf_d2; '
               'SMD(H2O) for the solvation comparison',
        conformer_pool=dict(n_conformers=len(CONF_IDS),
                            ids=CONF_IDS,
                            charge=0, multiplicity=1,
                            atom_order='O3 central, O3 term(+x), O3 term(-x), '
                                       'water O, water H1, water H2'),
        gas_phase=dict(
            global_minimum=s2.get('gas_global_minimum'),
            global_energy_hartree=fmt(s2.get('gas_global_energy_hartree'), 8),
            n_distinct_minima=s2.get('n_distinct_minima'),
            window_kcal=s2.get('window_kcal'),
            dedup_kept=s2.get('kept_minima'),
            dedup_eliminated=s2.get('eliminated'),
            conformers=conformers,
            monomers=monomers),
        smd=smd_block,
        binding=binding_block,
        ccsdt=ccsdt_block,
        method_limitations=[
            'Single-reference DFT (RKS); <S^2>=0 by construction, no spin contamination.',
            'Interaction energies are gas-phase; the SMD comparison is a LOCAL '
            'hydration-structure trend only and must NOT be read as an aqueous '
            'binding constant (no free-energy-of-solvation cycle was performed).',
            'geomeTRIC (delocalised internal coordinates) replaces berny because '
            'berny cannot build internal coordinates for the near-linear '
            'O_w-H...O hydrogen-bond chain; convergence assert_convergence=False, '
            'final geometry re-checked with a fresh make_mf_d2 (max|grad|<5e-5 Ha/Bohr).',
            'Boys-Bernardi CP correction uses PySCF ghost atoms (X-O / X-H); the '
            'project -D2 term excludes ghosts (verified) so no spurious dispersion '
            'pairs are introduced by the ghost basis.',
            'Standard states: gas RRHO 1 atm association free energy, and 1 M via '
            'the job-specified -1.894 kcal/mol correction for O3 + H2O -> O3.H2O.',
        ],
        usable_as_next_stage_initial=None,  # filled in main()
    )
    return summary, s2


def fill_next_stage(summary, s2):
    kept = s2.get('kept_minima', []) if s2 else []
    if kept:
        gmin = s2.get('gas_global_minimum')
        summary['usable_as_next_stage_initial'] = dict(
            recommended=gmin,
            reason='global gas-phase minimum (within the 3 kcal/mol distinct-minima set)',
            all_distinct_minima=kept)
    return summary


def write_markdown(summary, s2):
    L = []
    L.append('# O₃·H₂O 局部水合构型搜索报告 (JOB-2026-0905-009)')
    L.append('')
    L.append('**方法**: %s' % summary['method'])
    L.append('')
    L.append('## 1. 构型池')
    cp = summary['conformer_pool']
    L.append('- 初始构型数: %d（电荷 0 / 单重态）' % cp['n_conformers'])
    L.append('- 原子序: %s' % cp['atom_order'])
    L.append('- 覆盖: 末端氧供氢键 / 不同接近方向 / 共面与非共面 / 近远接触距离。')
    L.append('')

    L.append('## 2. 气相构型搜索与去重')
    gp = summary['gas_phase']
    L.append('- 全局最低点: **%s** (E = %.8f Ha)' % (gp['global_minimum'], gp['global_energy_hartree']))
    L.append('- 不同极小值数（3 kcal/mol 窗口）: %s' % gp['n_distinct_minima'])
    L.append('- 去重保留: %s' % gp['dedup_kept'])
    L.append('')
    L.append('| 构型 | 状态 | E_tot (Ha) | Δ(kcal/mol) | n_img | ν_min (cm⁻¹) | 氢键拓扑 | 保留? | 淘汰理由 |')
    L.append('|---|---|---|---|---|---|---|---|---|')
    for c in gp['conformers']:
        L.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            c['id'], c['status'], c['e_total_hartree'], c['rel_kcal'],
            c['n_imag'], c['lowest_freq'], c['topology'], c['kept'],
            c['eliminated_reason'] or ''))
    L.append('')
    L.append('单体: H₂O E=%s Ha (n_img=%s, ν_min=%s); O₃ E=%s Ha (n_img=%s, ν_min=%s).' % (
        gp['monomers']['h2o']['e_total_hartree'], gp['monomers']['h2o']['n_imag'],
        gp['monomers']['h2o']['lowest_freq'], gp['monomers']['o3']['e_total_hartree'],
        gp['monomers']['o3']['n_imag'], gp['monomers']['o3']['lowest_freq']))
    L.append('')

    L.append('## 3. SMD(水) 比较（局部水合结构趋势，非水溶液结合常数）')
    smd = summary.get('smd') or {}
    if smd:
        L.append('- 溶剂模型: %s' % smd.get('solvent_model'))
        L.append('- 保留的 SMD 极小值: %s' % smd.get('smd_minima'))
        L.append('- SMD 二次去重淘汰: %s' % smd.get('smd_eliminated'))
        L.append('')
        L.append('| 构型 | 气相 E (Ha) | SMD E (Ha) | ΔE(SMD−gas) kcal/mol | 气相拓扑 | SMD 拓扑 |')
        L.append('|---|---|---|---|---|---|')
        for row in smd.get('gas_vs_smd_ordering') or []:
            L.append('| %s | %.8f | %.8f | %.3f | %s | %s |' % (
                row['id'], row['gas_e'], row['smd_e'], row['delta_e_smd_minus_gas'],
                row['gas_topology'], row['smd_topology']))
    else:
        L.append('（未运行 / 无输出）')
    L.append('')

    L.append('## 4. 气相结合能与热化学')
    bd = summary.get('binding') or {}
    if bd:
        ga = bd.get('ghost_d2_audit') or {}
        L.append('- Ghost-D2 排除审计: ghosts_excluded_ok=%s (D2 差=%s Ha)' % (
            ga.get('ghosts_excluded_ok'), ga.get('d2_difference')))
        L.append('')
        L.append('| 构型 | E_int(非CP) | E_int(CP) | BSSE | ZPE贡献 | ΔG_assoc(1atm) | ΔG_assoc(1M) |')
        L.append('|---|---|---|---|---|---|---|')
        for r in bd.get('rows') or []:
            L.append('| %s | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |' % (
                r['id'], r['e_int_uncorrected_kcal_mol'], r['e_int_cp_kcal_mol'],
                -r['bsse_kcal_mol'], r['zpe']['value_kcal'],
                r['deltaG_assoc_1atm_kcal_mol'], r['deltaG_assoc_1M_kcal_mol']))
        L.append('')
        L.append('标准态说明: %s' % bd.get('standard_state_note'))
    else:
        L.append('（未运行 / 无输出）')
    L.append('')

    L.append('## 5. CCSD(T)/aug-cc-pVTZ 单点相互作用能')
    cc = summary.get('ccsdt') or {}
    if cc:
        L.append('- 基组: %s' % cc.get('basis'))
        L.append('- 单体能量: O₃=%s Ha, H₂O=%s Ha' % (
            cc.get('monomer_energies', {}).get('o3'),
            cc.get('monomer_energies', {}).get('h2o')))
        L.append('')
        L.append('| 构型 | E_int CCSD(T) (kcal/mol) | T1 诊断 | 单参考? |')
        L.append('|---|---|---|---|')
        for r in cc.get('interaction') or []:
            L.append('| %s | %.3f | %.4f | %s |' % (
                r['id'], r['e_int_ccsdt_kcal_mol'], r['t1_diagnostic'],
                r['single_reference_ok']))
    else:
        L.append('（未运行 / 无输出）')
    L.append('')

    L.append('## 6. 方法局限')
    for m in summary['method_limitations']:
        L.append('- %s' % m)
    L.append('')

    L.append('## 7. 可否作为下一阶段初始结构')
    nx = summary.get('usable_as_next_stage_initial') or {}
    if nx:
        L.append('- 推荐: **%s** — %s' % (nx.get('recommended'), nx.get('reason')))
        L.append('- 全部不同极小值: %s' % nx.get('all_distinct_minima'))
    L.append('')
    return '\n'.join(L)


def main():
    summary, s2 = build_summary()
    summary = fill_next_stage(summary, s2)
    with open(os.path.join(RES, 'conformer_summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)
    md = write_markdown(summary, s2)
    with open(os.path.join(RES, 'conformer_report.md'), 'w') as fh:
        fh.write(md)
    print('REPORT_DONE ->', os.path.join(RES, 'conformer_summary.json'))
    print('REPORT_DONE ->', os.path.join(RES, 'conformer_report.md'))


if __name__ == '__main__':
    main()
