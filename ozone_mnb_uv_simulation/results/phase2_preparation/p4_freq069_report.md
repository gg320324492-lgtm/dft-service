# JOB-2026-0906-069 报告：P4产物频率验收（仅HNO）— 项目方法模型

日期：2026-09-11。批目录：`run_artifacts/02_nh3o3_reference/p4_freq069/`。
结果：`p4_freq069_results.json`；账本：`budget_p4freq069.json`；mock测试：`mock_test_results.json`（7/7通过）。

## 0. 模型地位（总标注）

> **项目方法模型，非S13原始三维结构复现。** 输入是JOB-068项目方法优化结果（其起点为JOB-067模板，非作者坐标）。本文所有数值不与S13 CCSD(T)数值混表。

## 1. 单产物规则（条件分支执行）

JOB-068验收结果：HNO全链路通过（复核PASS、stable_i=True）；H₂O₂预算耗尽未达门限（最低gmax 6.42×10⁻³，未验收）。按总指令条件分支：**本批只处理HNO**；`E_products` 与 `ΔE_project` **未计算**（需要两个产物都通过；JOB-026单体和 −281.998121835 Eh 仅登记作上下文，不作差值）。

## 2. 方法（运行前固定，与JOB-068/026一致）

wb97xd（libxc）＋项目显式D2一次（d2_full.attach_d2，能量/梯度/Hessian三路一致）；def2-TZVP；grid level 8；grid_response=True（梯度与Hessian对象均验证置位）；SCF 1e-12/1e-9；气相；电荷0；singlet（RKS仅限P4产物端）。

## 3. 执行与账目（SCF attempt账本）

| 类别 | 上限 | 用量 | 结果 |
|---|---|---|---|
| hno_anchor | 1 | 1 | PASS：dE=1.71×10⁻¹³ Eh，dgrad=4.44×10⁻¹⁴，dcoords=2.22×10⁻¹⁶ Å，config逐键匹配 |
| hno_hess_dft | 1 | 1 | 完成：解析DFT Hessian＋显式D2 Hessian＋合成核验 |
| 合计 | 2 | 2 | 0失败、0流程异常 |

D2有限差分梯度调用（经典项，无SCF）**单独统计，不入SCF账本**：显式d2_hess 18次＋attached-kernel路径18次（预期18=2×3原子×3坐标）。

## 4. Hessian三件套与自洽核验

- 解析DFT Hessian：PySCF解析二阶导（CPHF），grid_response=True，纯DFT部分经预附着父类核提取；
- 显式D2 Hessian：`d2_full.d2_hess`＝解析D2梯度的中心有限差分（h=1e-3 Bohr），与梯度构造一致、对称化（转置(1,0,3,2)约定）；
- 合成Hessian＝两者之和；与attached-kernel输出（DFT核＋D2核）最大偏差 **1.28×10⁻¹² Eh/Bohr² ≤ 1e-10 门限，PASS**；
- 矩阵文件：`hno/hess_hno_dft.npy`、`hess_hno_d2.npy`、`hess_hno_combined.npy`、`hess_hno_attached_kernel.npy`（各含sha256登记于结果JSON）。

## 5. 频率验收（投影前约定全部登记）

- 平移/转动子空间秩=6（PySCF thermo `_get_TR` 约定复刻），内禀空间U 9×3，正交误差最大值已记录；
- 质量约定：`atom_mass_list(isotope_avg=True)`（PySCF）；单位：cm⁻¹，符号约定 sign(λ)·√|λ|（负值=虚频，**完整保留，不删不取绝对值**）；
- **HNO内禀频率：1587.84、1726.93、2923.75 cm⁻¹；负模数=0**；
- 全矩阵最低6个本征值（TR污染指示）最大|ν|=43.7 cm⁻¹（L8数值网格下TR子空间非严格零的如实记录；内禀频率来自投影内禀问题）；
- 交叉核验：`pyscf.hessian.thermo.harmonic_analysis`（同一矩阵、同一质量、imaginary_freq=True）最大偏差 **4.62×10⁻² cm⁻¹** —— 仅用于频率核验；**PySCF thermo()的热化学数值全文未使用**（已知单位问题）。

**判定（按总指令措辞）**：负模为0 → 登记"**在本方法（wb97xd+D2/def2-TZVP/L8/grid_response，气相）和本检查范围（TR投影Cartesian Hessian，解析DFT＋显式D2）内支持局部极小值**"。不构成对S13 CCSD(T)或整条反应路径可靠性的声明。

## 6. 热化学准备（项目自定义函数，逐字复用）

来源：`protocols/run_baseline.py thermochemistry()`（项目RRHO实现），经**AST逐字提取**执行（模块未import——其顶层berny依赖不可用；原文件未改动；提取件与来源sha256登记于manifest与结果JSON）。T=298.15 K，P=101325 Pa，σ=1（Cs），mult=1，e_elec=锚点E_total=−130.48419626983627 Eh（与068复核值门控差1.71×10⁻¹³）：

| 量 | Eh | kcal/mol |
|---|---|---|
| ZPE | 0.01421239 | 8.918 |
| 热焓校正 H_corr(298) | 0.01799443 | 11.292 |
| 自由能校正 G_corr(298) | −0.00703225 | −4.413 |

S_tot=220.384 J/(mol·K)（平动/转动/振动/电子分量见结果JSON）。负模=0 → 热化学登记有效（仅限项目方法模型）。

## 7. 边界声明

- 未做CP；非S13 CCSD(T)；≠水相自由能；≠养殖水处理效率；
- H₂O₂未验收 → 不拼接P4热化学/能量；
- stable_i（068）＋零负模（本批）≠反应路径可靠性；TS/IRC仍未授权、未猜测；
- 未修改DFT平台、依赖、PySCF、历史结果；真实计算全部入账本（2/2）。
