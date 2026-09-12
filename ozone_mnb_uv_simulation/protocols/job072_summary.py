#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-072: cross-batch summary + report data package.

QUANTITATIVE BUDGET: 0 SCF, 0 gradient.  Aggregates JOB-026 (reactant
monomer anchors), 064/066 (C2 scans), 068 (P4 endpoints), 069 (HNO
frequency/thermo), 070 (P1 radical endpoints), 071 (spin prep) into:

  * endpoint energy table (project method) with per-endpoint gradient
    / recheck / stability / frequency status;
  * S13 literature values in STRICTLY SEPARATE columns (never merged
    with project numbers into one numeric table);
  * C1/C2 contact-screening conclusions;
  * P4/P1 executability + missing TS 3D coordinates;
  * SVG charts + CSV data sized for a ~6-page briefing (NO final PPT,
    NO data-cutoff decision for the user).

Every figure/table labels: project model vs S13 literature, method,
energy definition, accepted vs pending.
"""
import os, sys, json, csv, time

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
R_REF = ROOT + '/run_artifacts/02_nh3o3_reference'
OUT = R_REF + '/summary072'
FIG = OUT + '/figures'

P = lambda *a: os.path.join(*a)
RES26 = R_REF + '/final_endpoint_index.json'
RES68 = R_REF + '/p4_products068/p4_products068_results.json'
RES69 = R_REF + '/p4_freq069/p4_freq069_results.json'
RES70 = R_REF + '/p1_radicals070/p1_radicals070_results.json'
RES71 = R_REF + '/p1_spin071/p1_spin_prep071_results.json'
RES67 = R_REF + '/reaction_prep067/reaction_prep067_results.json'
RES64 = R_REF + '/c2_scan064/c2_scan064_results.json'
RES66 = R_REF + '/c2_mirror_scan066/c2_mirror_scan066_results.json'

EH_KCAL = 627.5094740631


def load(path):
    try:
        return json.load(open(path))
    except Exception as e:                       # noqa: BLE001
        return dict(_missing=str(e))


def fmt(x, nd=6):
    return 'n/a' if x is None else ('%.*f' % (nd, x))


def main():
    os.makedirs(FIG, exist_ok=True)
    idx26, r68, r69, r70, r71, r67 = (load(RES26), load(RES68),
                                      load(RES69), load(RES70),
                                      load(RES71), load(RES67))
    e_nh3 = idx26['default_endpoints']['NH3_gas_v5_final'][
        'e_total_hartree']
    e_o3 = idx26['default_endpoints']['O3_gas_final'][
        'e_total_hartree']
    e_mono = e_nh3 + e_o3

    # ---------------- endpoint table (project method) ---------------
    hno = r68['final']['accepted_molecules']['HNO']
    h2o2 = r68['final']['accepted_molecules']['H2O2']
    freq69 = r69.get('frequency', {})
    thermo69 = r69.get('thermochemistry', {})
    m_hoo = r70['molecules'].get('HOO', {})
    m_h2no = r70['molecules'].get('H2NO', {})

    rows = []
    rows.append(dict(
        endpoint='NH3 (reactant monomer)', batch='JOB-026',
        role='reactant fragment', spin='singlet RKS',
        e_electronic_Eh=e_nh3, gmax=None,
        recheck='accepted (JOB-026 chain)',
        stability='n/a (monomer anchor)',
        frequency='n/a (anchor only)', thermo='n/a',
        acceptance='ACCEPTED (historical chain)'))
    rows.append(dict(
        endpoint='O3 (reactant monomer)', batch='JOB-026',
        role='reactant fragment', spin='singlet RKS (T1 risk noted)',
        e_electronic_Eh=e_o3, gmax=None,
        recheck='accepted (JOB-026 chain)',
        stability='n/a (monomer anchor)',
        frequency='n/a (anchor only)', thermo='n/a',
        acceptance='ACCEPTED (historical chain); multireference '
                   'risk registered'))
    rows.append(dict(
        endpoint='HNO (P4 product)', batch='JOB-068/069',
        role='P4 product fragment', spin='singlet RKS (P4 end only)',
        e_electronic_Eh=hno['e_recheck'],
        gmax=hno['meeting_gmax'],
        recheck='PASS (dE=%.1e)' % hno.get(
            'recheck', 0) if isinstance(hno.get('recheck'), float)
        else 'PASS',
        stability='stable_i=True (JOB-068)',
        frequency='1587.84/1726.93/2923.75 cm-1, 0 negative modes '
                  '(JOB-069)',
        thermo='ZPE 8.918 kcal/mol; G_corr -4.413 kcal/mol '
               '(registered)',
        acceptance='ACCEPTED (project method + scope)'))
    rows.append(dict(
        endpoint='H2O2 (P4 product)', batch='JOB-068',
        role='P4 product fragment', spin='singlet RKS (P4 end only)',
        e_electronic_Eh=None, gmax=h2o2.get('lowest_gmax'),
        recheck='not triggered (threshold not met)',
        stability='not triggered',
        frequency='NOT COMPUTED (endpoint not accepted)',
        thermo='NOT COMPUTED',
        acceptance='NOT ACCEPTED (budget exhausted, lowest gmax '
                   '6.42e-3)'))
    rows.append(dict(
        endpoint='HOO radical (P1 fragment)', batch='JOB-070',
        role='P1(2) product fragment',
        spin='doublet UKS (spin=1)',
        e_electronic_Eh=m_hoo.get('e_recheck'),
        gmax=(m_hoo.get('recheck') or {}).get('grad_max'),
        recheck=('PASS' if m_hoo.get('recheck_pass') else
                 'FAIL/not reached'),
        stability=str((m_hoo.get('stability') or {}).get('stable_i')),
        frequency='NOT COMPUTED (out of JOB-070 scope)',
        thermo='NOT COMPUTED',
        acceptance=('ACCEPTED (project method)' if
                    m_hoo.get('recheck_pass') else 'NOT ACCEPTED')))
    rows.append(dict(
        endpoint='H2NO radical (P1 fragment)', batch='JOB-070',
        role='P1(2) product fragment',
        spin='doublet UKS (spin=1)',
        e_electronic_Eh=m_h2no.get('e_recheck'),
        gmax=(m_h2no.get('recheck') or {}).get('grad_max'),
        recheck=('PASS' if m_h2no.get('recheck_pass') else
                 'FAIL/not reached'),
        stability=str((m_h2no.get('stability') or {}).get('stable_i')),
        frequency='NOT COMPUTED (out of JOB-070 scope)',
        thermo='NOT COMPUTED',
        acceptance=('ACCEPTED (project method)' if
                    m_h2no.get('recheck_pass') else 'NOT ACCEPTED')))

    # derived references
    frag70 = r70.get('fragment_reference', {})
    dE_p1 = frag70.get('dE_P1frag')
    derived = dict(
        e_monomer_sum_Eh=e_mono,
        e_products_P4=None,
        dE_P4=None,
        dE_P4_reason='H2O2 endpoint NOT accepted (JOB-068) -> no '
                     'P4 energy may be assembled',
        e_fragments_P1_Eh=frag70.get('e_fragments'),
        dE_P1frag_Eh=dE_p1,
        dE_P1frag_kcal_mol=frag70.get('dE_P1frag_kcal_mol'),
        definitions='all derived values are PROJECT-METHOD '
                    'ELECTRONIC energies (wb97xd+D2/def2-TZVP/L8/'
                    'grid_response, gas, no ZPE/thermal/CP)')

    # ---------------- S13 literature (SEPARATE columns) -------------
    lit = r67.get('literature', {})
    t1 = lit.get('table1_relative_energies_kcalmol', {})
    s13 = dict(
        source='S13 Table 1 (page 269) + thermo table (page 275); '
               'JOB-067 registration',
        P4=dict(b3lyp=t1.get('P4', {}).get('b3lyp'),
                ccsdt=t1.get('P4', {}).get('ccsdt'),
                g3b3=t1.get('P4', {}).get('g3b3'),
                dG0=-34.33, dE_thermo=-33.09),
        P1=dict(b3lyp=t1.get('P1', {}).get('b3lyp'),
                ccsdt=t1.get('P1', {}).get('ccsdt'),
                g3b3=t1.get('P1', {}).get('g3b3'),
                dG0=-21.35, dE_thermo=-19.4),
        barriers=dict(TS1_from_C1=29.68,
                      TS8_stated_51_08_vs_table_51_80=True,
                      TS2_from_IN1=30.77,
                      CP3_to_P4='barrierless',
                      CP1_to_P1='barrierless'),
        t1_diagnostics=dict(P4=t1.get('P4', {}).get('t1'),
                            P1=t1.get('P1', {}).get('t1')),
        rule='S13 numbers are CCSD(T)//B3LYP literature quantities; '
             'they are NEVER placed in the same numeric column as '
             'project results')

    # ---------------- C1/C2 conclusions ------------------------------
    c1c2 = dict(
        C1='multi-round limited relaxations (058-061) never reached '
           'the strict 1e-6 batch trigger; a soft negative-curvature '
           'direction persisted (058-059); NO accepted NH3...O3 '
           'contact minimum exists; unconditional stepping paused '
           '(JOB-062 audit)',
        C2='coplanar project-model fixed-orientation 6-point scans, '
           'both normal (064) and mirror (066) branches: energy '
           'monotonically decreasing over the sampled range, no '
           'sampled minimum; the two branches are exactly degenerate '
           '(mirror symmetry) but remain DISTINCT labelled branches',
        conclusion='contact-configurator search (C1/C2) is closed '
                   'for now; any reopening needs a new construction '
                   'assumption or author coordinates')

    # ---------------- executability + gaps ---------------------------
    exec_tab = dict(
        P4=dict(
            HNO='endpoint DONE (geometry+recheck+stability+'
                'frequency+thermo)',
            H2O2='BLOCKED: optimization did not reach the gradient '
                 'gate within its cap; a continuation batch would be '
                 'a NEW authorized decision (not auto-run)',
            dE_P4='BLOCKED by H2O2'),
        P1=dict(
            fragments=('DONE for both radicals (project method)' if
                       frag70.get('p1_fragments_complete') else
                       'PARTIAL/BLOCKED'),
            complex_CP1='BLOCKED: no 3D coordinates; constructing it '
                        'needs author data or an explicitly '
                        'authorized construction assumption',
            spin_reference='prepared (JOB-071): BS-singlet and '
                           'triplet UKS candidates defined with '
                           '<S2>/spin-density acceptance criteria'),
        missing_3d=['TS1', 'TS2', 'TS8', 'CP1', 'CP3',
                    'full R->P path coordinates'],
        source_of_missing='S13 Figure 1 is 2D (no depth/orientation '
                          'or H-transfer geometry)')

    # ---------------- CSV tables --------------------------------------
    csv_endpoint = OUT + '/endpoints_project_method.csv'
    with open(csv_endpoint, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    csv_s13 = OUT + '/s13_literature_separate.csv'
    with open(csv_s13, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['quantity_kcal_mol', 'S13_B3LYP', 'S13_CCSD(T)',
                    'S13_G3B3', 'S13_dG0_298K', 'method_column_note'])
        w.writerow(['P4 vs R', s13['P4']['b3lyp'],
                    s13['P4']['ccsdt'], s13['P4']['g3b3'],
                    s13['P4']['dG0'],
                    'S13 literature ONLY - no project number here'])
        w.writerow(['P1 vs R', s13['P1']['b3lyp'],
                    s13['P1']['ccsdt'], s13['P1']['g3b3'],
                    s13['P1']['dG0'],
                    'S13 literature ONLY - no project number here'])
        w.writerow(['project dE_P1frag (electronic)', '', '', '', '',
                    'project model: %s kcal/mol (%s)'
                    % (fmt(frag70.get('dE_P1frag_kcal_mol'), 2),
                       'separate definition: electronic, no ZPE/CP'
                       if frag70.get('dE_P1frag_kcal_mol')
                       is not None else 'not computed')])
    csv_gaps = OUT + '/gaps_and_missing_coordinates.csv'
    with open(csv_gaps, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['item', 'status', 'what would unblock it'])
        w.writerow(['TS1/TS2/TS8/CP1/CP3 3D coordinates', 'MISSING',
                    'author coordinates (preferred)'])
        w.writerow(['H2O2 endpoint', 'NOT ACCEPTED',
                    'new authorized continuation batch'])
        w.writerow(['P4 dE', 'BLOCKED', 'accepted H2O2 endpoint'])
        w.writerow(['P1 complex (CP1)', 'BLOCKED',
                    'author coordinates or authorized construction '
                    'assumption'])
        w.writerow(['frequencies for radicals/H2O2', 'NOT COMPUTED',
                    'follow-up acceptance batch after endpoints'])

    # ---------------- SVG figures --------------------------------------
    svgs = {}
    svgs['f1_endpoint_status'] = fig_endpoint_status(rows)
    svgs['f2_method_separation'] = fig_method_separation(s13, frag70)
    svgs['f3_hno_frequencies'] = fig_hno_freq(freq69)
    svgs['f4_contact_search'] = fig_contact_search()
    svgs['f5_gaps'] = fig_gaps(exec_tab)
    for name, svg in svgs.items():
        with open(P(FIG, name + '.svg'), 'w',
                  encoding='utf-8') as fh:
            fh.write(svg)

    results = dict(
        job='JOB-2026-0906-072: cross-batch summary + report data '
            '(ZERO evaluations)',
        quantitative_budget=dict(scf=0, gradients=0, d2_fd_calls=0),
        global_boundaries=dict(
            wall_clock_plan='notes/autopilot_24h_plan_2026-09-11.md',
            attempts_window=dict(
                job068=26, job069=2, job070=r70.get(
                    'budget', {}).get('attempts') and len(
                    r70['budget']['attempts'])),
            data_cutoff='NOT decided here - user owns the briefing '
                        'data cutoff'),
        endpoint_table_project_method=rows,
        derived_references=derived,
        s13_literature_separate=s13,
        c1c2_conclusions=c1c2,
        executability=exec_tab,
        spin_prep=r71.get('job'),
        figures={k: 'figures/%s.svg' % k for k in svgs},
        csv=dict(endpoints=os.path.relpath(csv_endpoint, ROOT),
                 s13=os.path.relpath(csv_s13, ROOT),
                 gaps=os.path.relpath(csv_gaps, ROOT)),
        finished=time.strftime('%F %T'))
    path = OUT + '/summary072_results.json'
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, path)
    print('[072] summary written:', path)
    print('[072] figures:', list(svgs))


# ============================ SVG builders ============================
S14 = ('font-family="Helvetica,Arial,sans-serif" font-size="14"')
S12 = ('font-family="Helvetica,Arial,sans-serif" font-size="12"')
S11 = ('font-family="Helvetica,Arial,sans-serif" font-size="11"')


def _svg(title, body, w=960, h=540):
    return ('<svg xmlns="http://www.w3.org/2000/svg" width="%d" '
            'height="%d" viewBox="0 0 %d %d">'
            '<rect width="%d" height="%d" fill="white"/>'
            '<text x="24" y="34" %s font-weight="bold">%s</text>'
            '<text x="24" y="52" %s fill="#555">项目方法模型 '
            'PROJECT METHOD MODEL — 非S13复现 / NOT an S13 '
            'reproduction</text>%s</svg>'
            % (w, h, w, h, w, h, S14, title, S11, body))


def fig_endpoint_status(rows):
    body, y = [], 84
    for r in rows:
        acc = r['acceptance'].startswith('ACCEPTED')
        color = '#2e7d32' if acc else '#c62828'
        name = r['endpoint']
        e = ('E=%s Eh' % fmt(r['e_electronic_Eh'], 6)
             if r['e_electronic_Eh'] is not None else 'E=n/a')
        g = ('gmax=%s' % ('%.1e' % r['gmax'])
             if r['gmax'] is not None else 'gmax=n/a')
        body.append(
            '<rect x="24" y="%d" width="14" height="14" fill="%s"/>'
            '<text x="48" y="%d" %s>%s</text>'
            '<text x="420" y="%d" %s fill="#333">%s | %s</text>'
            '<text x="700" y="%d" %s fill="%s">%s</text>'
            % (y, color, y + 12, S12, name, y + 12, S11, e, g,
               y + 12, S11, color,
               'ACCEPTED' if acc else 'NOT ACCEPTED'))
        y += 34
    body.append('<text x="24" y="%d" %s fill="#555">green = '
                'accepted within the project method and check '
                'scope; red = not accepted / blocked</text>'
                % (y + 8, S11))
    return _svg('F1 端点验收状态 / Endpoint acceptance status '
                '(project method)', ''.join(body))


def fig_method_separation(s13, frag70):
    bars, labels = [], []
    for key, lab in (('b3lyp', 'S13 B3LYP'),
                     ('ccsdt', 'S13 CCSD(T)'),
                     ('g3b3', 'S13 G3B3')):
        v = s13['P1'].get(key)
        if v is not None:
            bars.append((lab, float(v), '#1565c0'))
    v = frag70.get('dE_P1frag_kcal_mol')
    if v is not None:
        bars.append(('project model (electronic)', float(v),
                     '#2e7d32'))
    body, x = [], 80
    base = 160          # zero baseline; ALL values below are negative
    for lab, val, color in bars:
        hgt = max(min(abs(val) * 8, 220), 2)
        if val < 0:
            body.append(
                '<rect x="%d" y="%d" width="120" height="%d" '
                'fill="%s" opacity="0.85"/>'
                '<text x="%d" y="%d" %s text-anchor="middle">%.2f'
                '</text>'
                '<text x="%d" y="%d" %s text-anchor="middle">%s'
                '</text>'
                % (x, base, hgt, color, x + 60, base + hgt + 18, S12,
                   val, x + 60, base + hgt + 36, S11, lab))
        else:
            body.append(
                '<rect x="%d" y="%d" width="120" height="%d" '
                'fill="%s" opacity="0.85"/>'
                '<text x="%d" y="%d" %s text-anchor="middle">%.2f'
                '</text>'
                '<text x="%d" y="%d" %s text-anchor="middle">%s'
                '</text>'
                % (x, base - hgt, hgt, color, x + 60, base - hgt - 8,
                   S12, val, x + 60, base + 20, S11, lab))
        x += 170
    body.append('<line x1="60" y1="%d" x2="880" y2="%d" '
                'stroke="#999" stroke-width="1"/>'
                '<text x="64" y="%d" %s fill="#555">0 (relative to '
                'R)</text>' % (base, base, base - 6, S11))
    body.append('<text x="24" y="470" %s fill="#555">blue = S13 '
                'literature (relative to R, their definitions); '
                'green = PROJECT-MODEL electronic energy of the '
                'separated fragments</text>' % S11)
    body.append('<text x="24" y="490" %s fill="#555">columns are '
                'deliberately separated: different methods AND '
                'different energy definitions are NEVER mixed into '
                'one numeric column</text>' % S11)
    return _svg('F2 P1(2)能量：S13文献 vs 项目模型（分栏）',
                ''.join(body))


def fig_hno_freq(freq69):
    freqs = freq69.get('freq_wavenumber_signed') or []
    body, x = [], 120
    for f in freqs:
        hgt = f / 3000.0 * 240
        body.append(
            '<rect x="%d" y="%d" width="110" height="%d" '
            'fill="#6a1b9a" opacity="0.85"/>'
            '<text x="%d" y="%d" %s text-anchor="middle">%.2f</text>'
            '<text x="%d" y="%d" %s text-anchor="middle">mode '
            '(cm-1)</text>'
            % (x, 380 - hgt, hgt, x + 55, 370 - hgt, S12, f, x + 55,
               398, S11))
        x += 160
    n_neg = freq69.get('n_negative_modes')
    body.append('<text x="24" y="440" %s fill="#555">HNO internal '
                'modes (TR-projected, isotope-averaged masses): '
                '%d negative modes -> local-minimum support WITHIN '
                'the method and check scope; not an S13 claim'
                '</text>' % (S11, n_neg or 0))
    body.append('<text x="24" y="460" %s fill="#555">PySCF '
                'harmonic_analysis cross-check (same matrix): max '
                'diff %.3f cm-1; thermo: project custom RRHO only'
                '</text>' % (S11, freq69.get('pyscf_crosscheck', {})
                             .get('max_abs_wn_diff', -1)))
    return _svg('F3 HNO频率验收（JOB-069，项目方法）', ''.join(body))


def fig_contact_search():
    body = [
        '<text x="24" y="90" %s>C1 (NH3...O3 接触, 全自由度多轮松弛)'
        '</text>' % S12,
        '<text x="40" y="112" %s fill="#555">优化未达本批1e-6门限；'
        '软负曲率方向持续存在；无已验收极小值（058-062）</text>' % S11,
        '<text x="24" y="150" %s>C2 (共面项目模型固定取向六点扫描)'
        '</text>' % S12,
        '<text x="40" y="172" %s fill="#555">正法向(064)与镜像(066)'
        '分支：能量单调下降、采样范围内无极小；两分支精确简并但保持'
        '独立登记</text>' % S11,
        '<text x="24" y="210" %s>结论</text>' % S12,
        '<text x="40" y="232" %s fill="#c62828">接触构型筛选阶段性'
        '收束：重启需要新构建假设或作者坐标</text>' % S11,
    ]
    return _svg('F4 C1/C2接触筛选结论（历史批次汇总）', ''.join(body))


def fig_gaps(exec_tab):
    body, y = [], 92
    items = [
        ('P4: HNO端点', 'DONE'),
        ('P4: H2O2端点', 'BLOCKED (梯度门限未达)'),
        ('P4: dE(HNO+H2O2 - NH3-O3)', 'BLOCKED (需H2O2)'),
        ('P1: 两个自由基端点',
         'DONE' if exec_tab['P1']['fragments'].startswith('DONE')
         else 'PARTIAL'),
        ('P1: CP1复合物/自旋参照计算', 'BLOCKED (缺三维坐标/假设)'),
        ('TS1/TS2/TS8/CP1/CP3', 'MISSING 3D (作者坐标)'),
        ('IRC/反应路径', '未授权启动'),
    ]
    for name, st in items:
        color = '#2e7d32' if st.startswith('DONE') else '#c62828'
        body.append('<text x="40" y="%d" %s>%s</text>'
                    '<text x="560" y="%d" %s fill="%s">%s</text>'
                    % (y, S12, name, y, S12, color, st))
        y += 36
    body.append('<text x="24" y="%d" %s fill="#555">最合理的首个'
                'TS/IRC前置任务：取得作者坐标后从TS1(P4支)或TS2'
                '(P1支)开始；在此之前不猜测坐标</text>' % (y + 12,
                                                          S11))
    return _svg('F5 可执行性与缺口', ''.join(body))


if __name__ == '__main__':
    main()
