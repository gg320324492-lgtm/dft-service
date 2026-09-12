# JOB-2026-0905-009 结果复核修订报告

**本报告取代** `results/01_water_matrices/pure_water_o3_h2o/conformer_report.md`（v1 报告及其机器可读版保留于原位置作为来源记录，其中下列表述已被本报告修订）。

**术语（按复核指示）**：在新的证据确认之前，此前报告的"9 个不同极小值"降格为 **"待核验候选集"**，"全局最低点"改称 **"已搜索候选中的最低能结构"**。 本批不启动新的反应、过渡态、离子或界面计算。

> **⚠️ 勘误（2026-09-05，最终极小值验收批前置更正）**：本报告 §3 中 "c14_plus = 极小值（全构造 n_imag=0）"的表述**撤回，标记为待确认**—— FD@0.005 Bohr 曾给出 1 个负模；"该离群属网格剪枝噪声"当时仅为假设、 **未经配对对照证实，撤回**。c12_plus 的准确表述更正为 **"存在负曲率证据"**（L6 解析与全部差分构造给出虚频；解析 L5 的 0 虚频 与其余构造矛盾）；它与 c14_plus 的关系**仅为几何聚类结果，不构成 "已证明的同一极小值"**。两项均交由 `results/01_water_matrices/pure_water_o3_h2o/final_minima_validation/` 的定向计算裁决。

## 1. 发现的错误、修复证据与受影响结果

| # | 错误 | 证据 | 受影响结果 | 状态 |
|---|---|---|---|---|

| E1 | 氢键判据方向反置：用 O_w→H 与 H→O 的夹角套用 ≥120° 判据，等价于排斥线性氢键 | 实测 c02 存在 d(H···O)=2.286 Å、角 179.9° 的线性氢键被判 "none"；c01/c03/c04/c06/c08 均有 148–167° 真实氢键被漏判 | 拓扑签名与去重聚类（v1 报告 §2/§3 的"拓扑"列全部失真）| 已修复（`conformer_utils.hbond_topology` 改用 H→O_w 与 H→O） |

| E2 | 刚体配准的旋转以 O3 框架计算、却以全分子质心平移，中心不一致；且未考虑水双氢等价置换 | 合成旋转/平移/换号案例在旧代码下 RMSD 不归零；修复后 12 个案例 RMSD≈1e-16（revision_test_1_rmsd_topology.json） | 成对 RMSD 被系统性虚高 → v1 "9 个不同极小值" 中 7 个实为等价重复 | 已修复 |

| E3 | `topology_signature` 混合 int/str 排序的潜在 TypeError（旧调用路径从未触发，本次重新去重首次踩中） | 修订脚本首跑报 TypeError | 无（修复前无输出）| 已修复（统一字符串标签）|

| E4 | `harmonic_analysis` 默认把虚频返回为复数，强转 float 后变 0.0，n_imag 假=0 | 上一批已发现并修复（imaginary_freq=False + freq_error 交叉核对）| v1 之前所有"真实极小值"判定 | 已修复（本批复核沿用）|

| E5 | 手写 ZPE 常量多乘 Boltzmann 因子（zpe 恒 0） | 上一批已修复（取热化学单一真源）| zpe_hartree 字段（不影响热化学表）| 已修复 |

| E6 | v1 Stage-4 能量表混用约定："E_int(非CP)" 实为结合能（用了各自优化单体），"E_int(CP)" 实为相互作用能（冻结复合物内几何单体），两列与派生 BSSE 不可比 | 本批复核按统一定义重算（revision_unified_energies.json）| v1 §4 结合能表 | 已修复，见 §4 |

| E7 | CCSD(T) 的 0.0643 实为 max\|t1\|，被误标为"T1 诊断"，并由此直接得出 single_reference_ok 判定 | PySCF 标准定义为 `get_t1_diagnostic(t1)=sqrt(\|t1\|²/N_corr_elec)` | v1 §5 | 已更正并撤回判定，见 §5 |

| E8 | 旧基线方法命名将泛函部分简写作"ωB97X" | 复核指示：不得将该泛函部分等同于独立的 ωB97X 泛函 | results/00_baseline 汇总、models 记录 | 生成器已更正并重新生成（"ωB97X-D 的不含经验色散部分用于优化/频率，另加经验色散单点校正"；保留 xc='wb97xd' 与 Libxc 条目名）|


修复证据归档：`run_artifacts/01_pure_water_o3_h2o/acceptance_revision/`。

## 2. 修订后的构型清单与待核验候选集

- 模式跟进：共 **14 次运行**（7 个单虚频鞍点 × ±方向），端点中零虚频 **8 个**、仍为鞍点 **6 个**。

- 全部派生端点已纳入结构池（原始 14 + 端点 14 = 28 个气相结构）；其中零虚频 9 个、鞍点 19 个。

- **气相待核验候选集（2 个）**：['c14_plus', 'c06_plus']；其中已搜索候选中的最低能结构：**c14_plus**（E = -301.87483581 Ha）。

- 与 v1 的对应关系（旧 kept_minima → 修订判定）:

  - c14_plus → kept as candidate

  - c12_plus → duplicate of c14_plus (revised RMSD=0.044 A, dE=0.000 kcal/mol, same revised topology)

  - c12_minus → duplicate of c14_plus (revised RMSD=0.014 A, dE=0.000 kcal/mol, same revised topology)

  - c05 → duplicate of c14_plus (revised RMSD=0.061 A, dE=0.000 kcal/mol, same revised topology)

  - c11_minus → duplicate of c14_plus (revised RMSD=0.071 A, dE=0.000 kcal/mol, same revised topology)

  - c11_plus → duplicate of c14_plus (revised RMSD=0.022 A, dE=0.000 kcal/mol, same revised topology)

  - c09_plus → duplicate of c14_plus (revised RMSD=0.083 A, dE=0.000 kcal/mol, same revised topology)

  - c09_minus → duplicate of c14_plus (revised RMSD=0.042 A, dE=0.000 kcal/mol, same revised topology)

  - c06_plus → kept as candidate

- 成对 RMSD（候选 + 鞍点）与逐结构拓扑：`run_artifacts/.../acceptance_revision/revision_rededup.json`。

- **SMD 重新去重**：9 个结构 → 待核验候选集 ['c05', 'c06_plus']；鞍点/淘汰 ['c11_plus', 'c12_minus', 'c12_plus']。

- **结论降级声明**：v1 的"9 个不同极小值"（其中 7 个 dE<0.001 kcal/mol）在修正等价置换与配准中心后坍缩为 2 个候选；v1 的"全局最低点 c14_plus"改称"已搜索候选中的最低能结构"，待本报告 §3 频率可信度确认后才可升级为"极小值/全局最低"结论。

## 3. 频率可靠性核查（c14_plus / c12_plus / c14）

- 同源性核对：三个记录集（gas_opt / gas_freq / mode_follow）的坐标最大偏差与能量散差见 `revision_hessian_crosscheck.json` 的 provenance 字段。

- 解析 vs 独立差分 Hessian（步长 0.005/0.02 Bohr）：c14_plus — max 相对 H 差 2.95e-03，最低 6 模最大频移 69.1 cm⁻¹，n_imag 跨构造一致=False。

- 解析 vs 独立差分 Hessian（步长 0.005/0.02 Bohr）：c12_plus — max 相对 H 差 2.87e-03，最低 6 模最大频移 59.1 cm⁻¹，n_imag 跨构造一致=False。

- 解析 vs 独立差分 Hessian（步长 0.005/0.02 Bohr）：c14 — max 相对 H 差 2.77e-03，最低 6 模最大频移 77.3 cm⁻¹，n_imag 跨构造一致=False。

- 网格/SCF 收敛检查（解析 level6+SCF 1e-12、差分步长 0.05 Bohr）：**c14_plus** — 各构造 n_imag = [0, 0]。

- 网格/SCF 收敛检查（解析 level6+SCF 1e-12、差分步长 0.05 Bohr）：**c12_plus** — 各构造 n_imag = [1, 1]。

- 网格/SCF 收敛检查（解析 level6+SCF 1e-12、差分步长 0.05 Bohr）：**c14** — 各构造 n_imag = [1, 0]。

- **处理原则**：负模一律不删除、不归为噪声。凡 n_imag 随构造方式翻转的结构，其"是否极小值"标记为**待定**；稳定者方可进入候选集的最终确认。

- 当前判定：不稳定（待定）：['c14']

- **网格敏感性量化证据**：c12_plus 最低模从解析 L5 的 +48.8 cm⁻¹ 翻转为 L6+SCF1e-12 的 −29.8 cm⁻¹（移位 79 cm⁻¹）；c14 为 −52.1 → −64.7；c14_plus 为 79.6 → 67.3。软模对网格级别的敏感度达 ~80 cm⁻¹ 量级，即这些平坦结构的"极小值/鞍点"分类在当前精度下不可靠。

- 逐结构结论：**c14_plus** = 极小值（解析 L5/L6 与差分 0.02/0.05 Bohr 全部 n_imag=0，仅 FD@0.005 受网格剪枝噪声出现一次离群）；**c12_plus** = 极可能为鞍点（L6 与全部差分构造一致给虚频；解析 L5 的 0 虚频属网格噪声误判——其本已作为 c14_plus 的等价重复移出候选集，不影响候选集）；**c14** = **待定**（解析/差分各执一词）。候选集的其它成员（c06_plus）未在本轮频率核查范围内，保持"待核验候选"称谓。

## 4. 统一定义下的相互作用能 / 变形能 / 结合能

定义（DFT 与 CCSD(T) 同一约定）：

- 相互作用能 E_int = E(复合物) − 两单体在**复合物内几何（冻结）**的能量；CP 版对冻结伙伴加 ghost，几何约定相同。

- 变形能 E_def = 冻结单体相对各自优化单体的能量差（不作 ghost 校正——弱复合物变形能 ~10⁻² kcal/mol 量级，通行约定，已明示）。

- 结合能 E_bind = E_int + E_def（noCP 与 CP 各一列）。

- BSSE 符号约定：BSSE = E_int(noCP) − E_int(CP) ≤ 0（未校正能量偏负），表中同时给出幅值。

- D2：全导数 −D2 计入所有电子项（ghost 不参与 D2，已审计）。

- 自由能：ΔG 来自各自优化单体 + 复合物的谐振热化学，**不含 CP 校正**，与 E_bind(noCP) 配对；1 atm→1 M 修正 −1.894 kcal/mol。


| 构型 | E_int(noCP) | E_int(CP) | BSSE(有号) | E_def | E_bind(noCP) | E_bind(CP) | ΔG(1 atm) | ΔG(1 M) | kcal/mol |

|---|---|---|---|---|---|---|---|---|

| c14_plus | -2.051 | -1.531 | -0.520 | 0.002 | -2.049 | -1.529 | 4.695 | 2.801 |

| c06_plus | -1.621 | -1.376 | -0.245 | 0.038 | -1.583 | -1.338 | 5.051 | 3.158 |


CCSD(T)/aug-cc-pVTZ（非 CP；未计算 CCSD(T) 的 CP 校正，**不得与 DFT 的 CP 相互作用能直接对比**）：

| 构型 | E_int(noCP) | E_def | E_bind(noCP) | max\|t1\| | T1(pyscf) | D1 |

|---|---|---|---|---|---|---|

| c14_plus | -2.855 | 0.045 | -2.810 | 0.0643 | 0.0189 | 0.0694 |

| c06_plus | -2.784 | 0.057 | -2.727 | 0.0652 | 0.0191 | 0.0707 |

## 5. CCSD(T) 电子相关诊断

- 标准诊断定义（安装版 PySCF `cc.ccsd.get_t1_diagnostic`）：`T1 = sqrt(|t1|² / N_corr_elec)`（Lee 型，按相关电子数归一）。

- 旧值 0.0643 重标为 **max\|t1\|**（单振幅绝对值最大），非 T1 诊断。

- 冻结核：`frozen=0`（未冻核）；相关电子数与收敛状态逐单点记录于 `ccsdt_revision/ccsdt_revision_summary.json`。

- 振幅复用：t1 已存 .npy；t2（复合物约 1 GB）未持久化——已列为缺失，重算仅需数分钟（几何已归档）。

- **判定撤回**：v1 的 `single_reference_ok=False` 撤回；单一诊断指标不能证明多参考根因。替换为仅报告数值与定义，根因结论留给后续（如 CASSCF/NEVPT2 或 D1/D2 累积证据）。

## 6. 旧结论处置（保留 / 撤回 / 待定）

| 旧结论 | 处置 |
|---|---|

| Stage-1 16/16 几何收敛（max\|grad\|<5e-5） | **保留**（与本批复核无关） |

| 虚频符号修复后 14 试探结构中仅 c05 零虚频 | **保留**（计算事实），但其"极小值"称谓待 §3 稳定性确认 |

| v1 "9 个不同极小值" | **撤回** → 待核验候选集（2 个）+ 7 个等价重复 |

| v1 "全局最低点 c14_plus" | **改称**"已搜索候选中的最低能结构"，待频率稳定性确认 |

| v1 氢键拓扑列（全部 "H-bonds: none"） | **撤回**（E1 判据反置） |

| v1 Stage-4 结合能表（混用约定） | **撤回** → §4 统一定义表 |

| v1 single_reference_ok 判定 | **撤回** → §5 仅报数值 |

| CCSD(T) E_int ≈ −2.81 kcal/mol（对优化单体） | **重新表述**：该数值按新定义属 E_bind(noCP) 口径，冻结单体口径的 E_int 见 §4 |

| SMD 结果仅作局部水合结构趋势 | **保留** |

## 7. 本批之后

- 本批通过后，才决定保留哪些水合结构及是否需要补充高层级对照；随后另批选择首条臭氧反应路径。本批未启动新的反应、过渡态、离子或界面计算。
