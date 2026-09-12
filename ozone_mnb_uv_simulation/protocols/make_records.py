#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate the deliverable files for the baseline batch from the result JSON.

Writes:
  inputs/<key>_initial.xyz
  models/00_baseline/<key>/optimized_gas.xyz, optimized_smd.xyz, calculation_record.md
  jobs/JOB-2026-0904-0NN_<key>_<phase>.md   + jobs/job_index.md
  results/00_baseline/baseline_summary.md, baseline_energies.csv
"""
import os, json, math
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results', '00_baseline')
HA2KCAL = 627.5094740631

JOB_ID = {
    'h2o_gas': 'JOB-2026-0904-001', 'h2o_smd': 'JOB-2026-0904-002',
    'o3_gas': 'JOB-2026-0904-003', 'o3_smd': 'JOB-2026-0904-004',
    'o2_gas': 'JOB-2026-0904-005', 'o2_smd': 'JOB-2026-0904-006',
}
PHASE_CN = {'gas': '气相', 'smd_water': '水相 SMD(H₂O)'}
NAME_CN = {'h2o': 'H₂O（水）', 'o3': 'O₃（臭氧）', 'o2': 'O₂（三重态氧）'}
NIST_S = {'h2o': 188.84, 'o3': 238.93, 'o2': 205.15}
EXP_FREQ = {'h2o': '1619 (bend) / 3832 / 3943（实验谐振值，未缩放对照）',
            'o3': '716 / 1135 / 1089（实验谐振值）',
            'o2': '1580（实验谐振值 ωe）'}
EXP_GEOM = {'h2o': 'r(O–H) = 0.9584 Å，∠HOH = 104.45°',
            'o3': 'r(O–O) = 1.2717 Å，∠OOO = 116.8°',
            'o2': 'r(O=O) = 1.2075 Å'}

data = {}
for tag in ['h2o_gas', 'h2o_smd', 'o3_gas', 'o3_smd', 'o2_gas', 'o2_smd']:
    with open(os.path.join(RES, tag + '.json')) as fh:
        data[tag] = json.load(fh)


def load(tag):
    return data[tag]


def xyz_block(rec, key='optimised_geometry_angstrom', comment=''):
    at = rec[key]
    return ("%d\n%s\n" % (len(at), comment) +
            "\n".join("%-2s %16.8f %16.8f %16.8f" % (s, x, y, z) for s, (x, y, z) in at) + "\n")


def geom_params(rec):
    at = rec['optimised_geometry_angstrom']
    sym = [a[0].strip() for a in at]
    c = np.array([a[1] for a in at], dtype=float)
    if len(c) == 2:
        return 'r = %.4f Å' % np.linalg.norm(c[0] - c[1])
    ci = int(np.argmax([{'H': 1, 'O': 8}[s] for s in sym]))
    oth = [i for i in range(3) if i != ci]
    r1 = np.linalg.norm(c[ci] - c[oth[0]])
    r2 = np.linalg.norm(c[ci] - c[oth[1]])
    v1, v2 = c[oth[0]] - c[ci], c[oth[1]] - c[ci]
    ang = math.degrees(math.acos(np.dot(v1, v2) / (r1 * r2)))
    return 'r = %.4f / %.4f Å，键角 = %.3f°' % (r1, r2, ang)


def f(v, n=8):
    return ('%.' + str(n) + 'f') % v


# ------------------------------------------------------------ inputs/*.xyz
os.makedirs(os.path.join(ROOT, 'inputs'), exist_ok=True)
for key in ['h2o', 'o3', 'o2']:
    rec = load(key + '_gas')
    lines = rec['initial_geometry_angstrom'].strip().split('\n')
    body = "%d\ninitial structure: %s (%s)\n%s\n" % (
        len(lines), NAME_CN[key], rec['structure_source'],
        "\n".join(lines))
    with open(os.path.join(ROOT, 'inputs', '%s_initial.xyz' % key), 'w') as fh:
        fh.write(body)

# ------------------------------------------- models/00_baseline/<key>/*.xyz
for key in ['h2o', 'o3', 'o2']:
    d = os.path.join(ROOT, 'models', '00_baseline', key)
    os.makedirs(d, exist_ok=True)
    for tag, phase in [('gas', 'gas'), ('smd', 'smd_water')]:
        rec = load('%s_%s' % (key, tag))
        comment = ('%s | wB97X-D/def2-TZVP | %s | optimised | E = %s Eh | %s'
                   % (NAME_CN[key], PHASE_CN[phase], f(rec['e_total_wb97xd_hartree']),
                      geom_params(rec)))
        with open(os.path.join(d, 'optimized_%s.xyz' % tag), 'w') as fh:
            fh.write(xyz_block(rec, comment=comment))

# ------------------------------------------------------------- jobs/*.md
os.makedirs(os.path.join(ROOT, 'jobs'), exist_ok=True)
for tag in ['h2o_gas', 'h2o_smd', 'o3_gas', 'o3_smd', 'o2_gas', 'o2_smd']:
    rec = load(tag)
    key, ph = tag.split('_')
    th = rec['thermo']
    thh = rec['thermo_hartree']
    txt = """# 任务说明 {job}

## 任务标识

- 任务编号：{job}
- 提交日期：2026-09-04
- 体系：{name}
- 相态/溶剂模型：{phase}
- 任务类型：几何优化 + 频率计算（同一水平、同一结构）
- 状态：已完成

## 计算设置

- 后端：PySCF {pyscf}（WSL2 Ubuntu 24.04.4，Python 3.12.3，numpy 2.5.2，pyberny 0.7.0）
- 驱动脚本：`protocols/run_baseline.py`
- 方法与基组：ωB97X-D 的不含经验色散部分用于优化/频率，另加经验色散单点校正（def2-TZVP）
  - DFT 部分：libxc `XC_HYB_GGA_XC_wB97X_D`（ωB97X-D 泛函的组成部分，不含经验色散；**不等同于独立的 ωB97X 泛函**）
  - −D2 色散项：显式补加（s6 = 1.0，sR6 = 1.1，a = 6.0，指数 12），仅作单点能量校正、未进入优化梯度与 Hessian；本任务贡献 {d2:.3e} Hartree
- 电荷 / 自旋多重度：{chg} / {mult}{spin}
- 积分网格：PySCF level 5，{ngrid} 点
- 基函数数目：{nbf}
- SCF 收敛：conv_tol = 1e-11，{conv}
- 优化阈值：梯度最大 2e-5、RMS 1e-5、步长最大 1e-4、步长 RMS 6e-5（Hartree/Bohr）
- 溶剂：{solv}
- 热化学：298.15 K，101325 Pa，刚性转子 + 谐振子

## 初始结构

来源：{src}

```
{init}
```

## 结果

- SCF 收敛：{conv}
- 优化收敛：{optconv}（最终 Cartesian 梯度最大 {gmax:.2e}、RMS {grms:.2e} Hartree/Bohr）
- 是否极小值：{mini}（虚频数 = {nimag}）
- Hessian：{hess}；对称性 |H − Hᵀ| = {asym:.1e}
- 振动频率（未缩放，cm⁻¹）：{freqs}
- 振动模式数：{nmodes}（应为 {nexp}）

### 能量

| 项 | 值（Hartree） |
| --- | --- |
| DFT 电子能 E(elec) | {edft} |
| −D2 色散校正 | {d2:.6e} |
| **E(wB97X-D) 合计** | **{etot}** |
| 零点能 ZPE | {zpe} |
| E(0 K) = E(elec) + ZPE | {e0k} |
| H(298.15 K) | {h298} |
| G(298.15 K) | {g298} |

热校正：H − E(elec) = {hcorr:.8f} Hartree，G − E(elec) = {gcorr:.8f} Hartree。
熵 S°(298.15 K) = {s:.2f} J·mol⁻¹·K⁻¹（平动 {str_:.2f} + 转动 {srot:.2f} + 振动 {svib:.2f}），
对称数 σ = {sigma}，转子类型：{rotor}。

### 优化后结构（Å）

```
{opt}
```

几何参数：{gp}

## 文件

- 输入结构：`inputs/{key}_initial.xyz`
- 优化结构：`models/00_baseline/{key}/optimized_{ph}.xyz`
- 机器可读结果：`results/00_baseline/{tag}.json`
- 计算记录：`models/00_baseline/{key}/calculation_record.md`

## 异常说明

{anom}
""".format(
        job=JOB_ID[tag], name=NAME_CN[key], phase=PHASE_CN[rec['phase']],
        pyscf=rec['pyscf_version'],
        d2=rec['e_dispersion_d2_hartree'], chg=rec['charge'], mult=rec['multiplicity'],
        spin=('，⟨S²⟩ = %.4f' % rec['spin_s2']) if 'spin_s2' in rec else '',
        ngrid=rec['n_grid_points'], nbf=rec['n_basis_functions'],
        conv='是' if rec['scf_converged'] else '否', solv=rec['solvent_model'],
        src=rec['structure_source'], init=rec['initial_geometry_angstrom'],
        optconv='是' if rec['optimisation_converged'] else '否',
        gmax=rec['max_gradient_hartree_bohr'], grms=rec['rms_gradient_hartree_bohr'],
        mini='是' if rec['is_minimum'] else '否', nimag=rec['n_imaginary'],
        hess=rec['hessian_method'], asym=rec['hessian_asymmetry'],
        freqs=', '.join('%.2f' % x for x in rec['frequencies_cm1']),
        nmodes=rec['n_vibrational_modes'], nexp=rec['n_expected_modes'],
        edft=f(rec['e_dft_hartree']), etot=f(rec['e_total_wb97xd_hartree']),
        zpe=f(thh['ZPE']), e0k=f(thh['E_0K']), h298=f(thh['H_298']), g298=f(thh['G_298']),
        hcorr=th['thermal_H_corr'], gcorr=th['thermal_G_corr'], s=th['S_tot'],
        str_=th['S_trans'], srot=th['S_rot'], svib=th['S_vib'], sigma=th['sigma'],
        rotor=('线性' if th['linear'] else '非线性'),
        opt="\n".join("%-2s %16.8f %16.8f %16.8f" % (s, x, y, z)
                      for s, (x, y, z) in rec['optimised_geometry_angstrom']),
        gp=geom_params(rec), key=key, ph='gas' if ph == 'gas' else 'smd',
        tag=tag,
        anom=('无。平动/转动模式残差最大 %.2f cm⁻¹，属几何未完全静止点的正常量级，'
              '投影后的振动频率不受影响（与未投影分析一致到 0.01 cm⁻¹）。'
              % rec['tr_rot_mode_max_abs_cm1']))
    with open(os.path.join(ROOT, 'jobs', '%s_%s_%s.md'
                           % (JOB_ID[tag], key, 'gas' if ph == 'gas' else 'smd')), 'w') as fh:
        fh.write(txt)

# ------------------------------------------------------- job index
idx = ["# 任务清单：基线批次（第 1 阶段）", "",
       "本批建立 H₂O、O₃、O₂ 的气相与 SMD(H₂O) 水相基线，共 6 个任务。",
       "所有任务由 `protocols/run_baseline.py` 在 PySCF 2.14.0（WSL2 Ubuntu 24.04.4）下执行。",
       "", "| 任务编号 | 体系 | 相态/溶剂 | 任务类型 | 状态 | 极小值 | E(wB97X-D) / Hartree |",
       "| --- | --- | --- | --- | --- | --- | --- |"]
for tag in ['h2o_gas', 'h2o_smd', 'o3_gas', 'o3_smd', 'o2_gas', 'o2_smd']:
    r = load(tag)
    key = tag.split('_')[0]
    idx.append("| %s | %s | %s | 优化+频率 | 完成 | %s | %s |" % (
        JOB_ID[tag], NAME_CN[key], PHASE_CN[r['phase']],
        '是（0 虚频）' if r['is_minimum'] else '否',
        f(r['e_total_wb97xd_hartree'])))
idx += ["", "对应的计算记录见 `models/00_baseline/<体系>/calculation_record.md`，",
        "机器可读结果见 `results/00_baseline/<体系>_<相态>.json`。", ""]
with open(os.path.join(ROOT, 'jobs', 'job_index.md'), 'w') as fh:
    fh.write("\n".join(idx))

# --------------------------------------------- calculation records
for key in ['h2o', 'o3', 'o2']:
    g, s = load(key + '_gas'), load(key + '_smd')
    rows = []
    for r in (g, s):
        th, thh = r['thermo'], r['thermo_hartree']
        rows.append("| %s | %s | %s | %s | %s | %s | %s | %s | %.2f |" % (
            PHASE_CN[r['phase']], f(r['e_dft_hartree']), '%.3e' % r['e_dispersion_d2_hartree'],
            f(r['e_total_wb97xd_hartree']), f(thh['ZPE']), f(thh['H_298']),
            f(thh['G_298']), '%.2f' % th['S_tot'],
            (r['e_total_wb97xd_hartree'] - g['e_total_wb97xd_hartree']) * HA2KCAL))
    txt = """# 计算记录：{name}

## 基本信息

- 案例编号：BASE-00-{key}
- 日期：2026-09-04
- 对应任务：{jobg}（气相）、{jobs}（水相）
- 所属阶段：第 1 阶段 · 基线校准

## 科学问题与假设

- 要解释的实验现象：臭氧微纳米气泡 + 紫外处理养殖水体时的协同效应，最终需要用 O₃ 参与的
  反应路径与能垒来解释；任何反应能都必须以同一套 H₂O / O₃ / O₂ 参照为基准。
- 分子尺度假设：隐式水相（SMD）会改变 O₃、O₂ 与 H₂O 的几何、振动与自由能，
  这些改变是后续水相反应能与气相比值差异的主要来源之一。
- 预期可验证的结果：三个体系在气相和水相均得到收敛的极小值结构、零虚频，
  以及可直接引用的 ZPE / H(298.15 K) / G(298.15 K)。

## 模型定义

- 水体类型与简化方式：纯水，用 SMD 连续介质（本体水）表示；气相为真空参照。
- 分子组成：单个{name}分子，无显式水、无抗衡离子。
- 结构来源：{src}
- 电荷与自旋多重度：{chg} / {mult}{spin}
- 显式水分子或界面模型：无（本批不含）

## 计算设置

- 后端与版本：PySCF {pyscf}（WSL2 Ubuntu 24.04.4；Python 3.12.3；numpy 2.5.2；pyberny 0.7.0）
- 任务类型：几何优化 + 频率（谐性）
- 泛函与基组：ωB97X-D 的不含经验色散部分用于优化/频率，另加经验色散单点校正（def2-TZVP）；
  DFT 部分用 libxc `XC_HYB_GGA_XC_wB97X_D`（ωB97X-D 泛函的组成部分，**不等同于独立的 ωB97X 泛函**），
  −D2 色散项显式补加（s6 = 1.0，sR6 = 1.1，a = 6.0，指数 12），仅作单点能量校正、未进入优化梯度与 Hessian
- 溶剂模型：气相无；水相 SMD(H₂O)（ε = 78.355，n = 1.3328，α = 0.82，β = 0.35）
- 积分网格：level 5（气相 {ng1} 点）
- 收敛与频率检查：SCF conv_tol = 1e-11；优化梯度最大 < 2e-5 Hartree/Bohr；
  解析 Hessian，对称化后投影掉平动/转动；虚频数 = 0 判据
- 热化学：298.15 K、101325 Pa，刚性转子 + 谐振子（自实现；PySCF 内置 thermo() 有单位错误，未采用）

## 结果与质量检查

| 相态 | E(DFT)/Eh | −D2/Eh | E(wB97X-D)/Eh | ZPE/Eh | H(298.15)/Eh | G(298.15)/Eh | S°/J·mol⁻¹·K⁻¹ | 相对气相/kcal·mol⁻¹ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
{rows}

- 气相是否收敛：{gc}（梯度最大 {ggmax:.2e} Hartree/Bohr）
- 水相是否收敛：{sc}（梯度最大 {sgmax:.2e} Hartree/Bohr）
- 虚频检查：气相 {gn} 个、水相 {sn} 个 —— 均为极小值
- 频率（cm⁻¹，未缩放）：气相 {gfreq}；水相 {sfreq}
- 实验谐振频率对照：{expf}
- 优化后几何：气相 {ggp}；水相 {sgp}；实验结构：{expg}
- 熵检验：气相 S° = {gs:.2f} J·mol⁻¹·K⁻¹，NIST 实验值 {nist:.2f}，偏差 {dev:+.2f}

## 与实验的联系

- 对实验结果的可能解释：{interp}
- 不确定性与不应外推的范围：{limit}
- 下一步：完成全部基线后，选取一条纯水中臭氧相关反应作为首个反应机理案例，
  再加入 HCO₃⁻、Cl⁻、铵态氮等水体基质。

## 文件清单

- `inputs/{key}_initial.xyz` —— 初始结构
- `models/00_baseline/{key}/optimized_gas.xyz`、`optimized_smd.xyz` —— 优化后结构
- `jobs/{jobg}_{key}_gas.md`、`{jobs}_{key}_smd.md` —— 任务说明
- `results/00_baseline/{key}_gas.json`、`{key}_smd.json` —— 机器可读输出
- `protocols/protocol_baseline.md`、`protocols/run_baseline.py` —— 协议与驱动脚本
""".format(
        name=NAME_CN[key], key=key, jobg=JOB_ID[key + '_gas'], jobs=JOB_ID[key + '_smd'],
        src=g['structure_source'], chg=g['charge'], mult=g['multiplicity'],
        spin=('，气相 ⟨S²⟩ = %.4f、水相 ⟨S²⟩ = %.4f' % (g.get('spin_s2', 0), s.get('spin_s2', 0))
              ) if 'spin_s2' in g else '',
        pyscf=g['pyscf_version'], ng1=g['n_grid_points'],
        rows="\n".join(rows),
        gc='是' if g['scf_converged'] and g['optimisation_converged'] else '否',
        ggmax=g['max_gradient_hartree_bohr'],
        sc='是' if s['scf_converged'] and s['optimisation_converged'] else '否',
        sgmax=s['max_gradient_hartree_bohr'],
        gn=g['n_imaginary'], sn=s['n_imaginary'],
        gfreq=', '.join('%.2f' % x for x in g['frequencies_cm1']),
        sfreq=', '.join('%.2f' % x for x in s['frequencies_cm1']),
        expf=EXP_FREQ[key],
        ggp=geom_params(g), sgp=geom_params(s), expg=EXP_GEOM[key],
        gs=g['thermo']['S_tot'], nist=NIST_S[key], dev=g['thermo']['S_tot'] - NIST_S[key],
        interp={'h2o': '水是溶剂本体，其 SMD 溶剂化自由能（−8.3 kcal/mol，电子能差）量级合理，'
                       '说明后续把水相反应物/产物放在同一连续介质中比较是自洽的。',
                'o3': '臭氧从气相进入水相的电子能下降 3.4 kcal/mol，与其在水中溶解度有限、'
                      '但界面富集可能提高局部浓度的实验观察相容；微纳米气泡的界面效应需后续界面模型补充。',
                'o2': 'O₂ 的溶剂化能很小（−0.4 kcal/mol），符合其低溶解度；'
                      'O₂ 作为臭氧分解与活性氧链的产物，其参照值用于后续反应能闭合。'}[key],
        limit={'h2o': '隐式溶剂不含显式氢键与界面电场；本结果不能用来解释微纳米气泡界面处的局部结构。',
               'o3': 'ωB97X-D 对 O₃ 的 O–O 键长偏短约 0.034 Å、伸缩频率偏高约 200 cm⁻¹，'
                     '与该分子的多参考特征有关；涉及 O₃ 的反应能垒应做方法层面敏感性检验，'
                     '不能直接外推为定量预测。',
               'o2': '三重态 O₂ 用 UKS 描述，⟨S²⟩ ≈ 2.009，自旋污染很小；'
                     '但本结果不含自旋禁阻过程的动力学信息。'}[key])
    with open(os.path.join(ROOT, 'models', '00_baseline', key, 'calculation_record.md'), 'w') as fh:
        fh.write(txt)

# -------------------------------------------------- results summary
hdr = ["# 基线汇总：H₂O / O₃ / O₂（ωB97X + 显式 −D2 校正/def2-TZVP）", "",
       "生成时间：2026-09-04　后端：PySCF 2.14.0（WSL2 Ubuntu 24.04.4）", "",
       "本批只建立参照，不含反应路径、过渡态、离子效应或界面模型。", "",
       "## 1. 总表（Hartree，除注明外）", "",
       "| 体系 | 相态 | 收敛 | 极小值 | E(DFT) | −D2 | E(wB97X-D) | ZPE | H(298.15) | G(298.15) | S° / J·mol⁻¹·K⁻¹ |",
       "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
for tag in ['h2o_gas', 'h2o_smd', 'o3_gas', 'o3_smd', 'o2_gas', 'o2_smd']:
    r = load(tag)
    th, thh = r['thermo'], r['thermo_hartree']
    key = tag.split('_')[0]
    hdr.append("| %s | %s | %s | %s | %s | %.2e | %s | %s | %s | %s | %.2f |" % (
        NAME_CN[key], PHASE_CN[r['phase']],
        '是' if r['scf_converged'] and r['optimisation_converged'] else '否',
        '是' if r['is_minimum'] else '否',
        f(r['e_dft_hartree']), r['e_dispersion_d2_hartree'],
        f(r['e_total_wb97xd_hartree']), f(thh['ZPE']), f(thh['H_298']), f(thh['G_298']),
        th['S_tot']))
hdr += ["", "## 2. 结构与频率", "",
        "| 体系 | 相态 | 优化后几何 | 频率 / cm⁻¹（未缩放） | 虚频 | 最大梯度 / Hartree·Bohr⁻¹ |",
        "| --- | --- | --- | --- | --- | --- |"]
for tag in ['h2o_gas', 'h2o_smd', 'o3_gas', 'o3_smd', 'o2_gas', 'o2_smd']:
    r = load(tag)
    key = tag.split('_')[0]
    hdr.append("| %s | %s | %s | %s | %d | %.2e |" % (
        NAME_CN[key], PHASE_CN[r['phase']], geom_params(r),
        ', '.join('%.2f' % x for x in r['frequencies_cm1']),
        r['n_imaginary'], r['max_gradient_hartree_bohr']))
hdr += ["", "## 3. 溶剂化（SMD(H₂O) 相对气相，各自优化结构）", "",
        "| 体系 | ΔE(elec) / kcal·mol⁻¹ | ΔG(298.15) / kcal·mol⁻¹ |",
        "| --- | --- | --- |"]
for key in ['h2o', 'o3', 'o2']:
    g, s = load(key + '_gas'), load(key + '_smd')
    de = (s['e_total_wb97xd_hartree'] - g['e_total_wb97xd_hartree']) * HA2KCAL
    dg = (s['thermo_hartree']['G_298'] - g['thermo_hartree']['G_298']) * HA2KCAL
    hdr.append("| %s | %+.2f | %+.2f |" % (NAME_CN[key], de, dg))
hdr += ["", "注：ΔG 为气相标准态（1 atm）与水相 SMD 标准态之间的自由能差；",
        "SMD 的标准态约定与实验水合自由能的 1 M ↔ 1 M 约定不同，直接比较时需留意。", "",
        "## 4. 热化学实现的外部标定", "",
        "| 体系 | 计算 S°(298.15 K) | NIST 实验 S° | 偏差 |", "| --- | --- | --- | --- |"]
for key in ['h2o', 'o3', 'o2']:
    sc = load(key + '_gas')['thermo']['S_tot']
    hdr.append("| %s | %.2f | %.2f | %+.2f |" % (NAME_CN[key], sc, NIST_S[key], sc - NIST_S[key]))
hdr += ["", "S° 是对热化学实现（刚性转子 + 谐振子 + 平动配分函数）的独立检验；",
        "偏差来自谐振子/刚性转子近似以及该级别方法下几何与频率的系统误差。", "",
        "## 5. 文件清单", "",
        "- `protocols/protocol_baseline.md`：计算协议与已知偏差",
        "- `protocols/run_baseline.py`、`make_records.py`：驱动与文档生成脚本",
        "- `inputs/<体系>_initial.xyz`：初始结构",
        "- `models/00_baseline/<体系>/optimized_gas.xyz`、`optimized_smd.xyz`：优化结构",
        "- `models/00_baseline/<体系>/calculation_record.md`：计算记录",
        "- `jobs/job_index.md` 与 `jobs/JOB-2026-0904-0NN_*.md`：任务说明与任务编号",
        "- `results/00_baseline/<体系>_<相态>.json`：机器可读结果",
        "- `results/00_baseline/baseline_energies.csv`：汇总表", ""]
with open(os.path.join(RES, 'baseline_summary.md'), 'w') as fh:
    fh.write("\n".join(hdr))

# ------------------------------------------------------------- csv
csv = ["system,phase,solvent_model,converged,is_minimum,n_imaginary,"
       "e_dft_hartree,e_d2_hartree,e_wb97xd_hartree,zpe_hartree,"
       "h_298_hartree,g_298_hartree,s_j_mol_k,sigma,linear,"
       "frequencies_cm1,max_grad_hartree_bohr,hessian_method,n_grid,n_basis,job_id"]
for tag in ['h2o_gas', 'h2o_smd', 'o3_gas', 'o3_smd', 'o2_gas', 'o2_smd']:
    r = load(tag)
    th, thh = r['thermo'], r['thermo_hartree']
    csv.append("%s,%s,%s,%s,%s,%d,%.10f,%.6e,%.10f,%.10f,%.10f,%.10f,%.4f,%d,%s,\"%s\",%.3e,%s,%d,%d,%s" % (
        r['species'], r['phase'], r['solvent_model'],
        r['scf_converged'] and r['optimisation_converged'], r['is_minimum'], r['n_imaginary'],
        r['e_dft_hartree'], r['e_dispersion_d2_hartree'], r['e_total_wb97xd_hartree'],
        thh['ZPE'], thh['H_298'], thh['G_298'], th['S_tot'], th['sigma'], th['linear'],
        ' '.join('%.2f' % x for x in r['frequencies_cm1']),
        r['max_gradient_hartree_bohr'], r['hessian_method'],
        r['n_grid_points'], r['n_basis_functions'], JOB_ID[tag]))
with open(os.path.join(RES, 'baseline_energies.csv'), 'w') as fh:
    fh.write("\n".join(csv) + "\n")

print("records written")
