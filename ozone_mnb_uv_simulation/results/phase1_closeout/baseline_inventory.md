# 第一阶段基线状态清单（v2 · JOB-2026-0906-015 修订）

> **修订记录**：v1 = JOB-2026-0906-014 编制；v2 = 015 批来源核实与范围修正；**v3 = JOB-2026-0906-016 气相基线来源纠正**（按 JOB-004/005/006 实际文件与 gas_phase_registry.json 追溯最新有效证据：c14 主参考改为 JOB-005 已验收 L8 端点+L8 完整频率；c06 主参考=已验收 L7 端点+L7 频率+L8 仅最低方向核查；endpoints_fixed 的 c14 条目与 gas_freq 旧几何频率标为历史阶段；不得把不同几何的优化与频率拼成一条验收证据）。纠正详情：`notes/gas_baseline_source_correction_2026-09-07.md`。主要修订：①全部路径改为完整项目相对路径（不再使用"…"缩写）；②登记单体与气相复合物的**计算当时**方法证据（含色散处理的历史口径）；③"定量比较可用"范围按证据收紧，网格/导数处理差异登记为**精度一致性待核查**；④登记已知应用范围。

## 0. 已知应用范围（JOB-2026-0906-015 登记）

- **已知应用范围**：水产养殖污染物去除或鱼类病毒灭活的机理研究（用户 2026-09-07 补充）。
- **其余具体对象保留待定**：具体化学组分、病毒名称、实验条件均未确定，**暂不自行选定**污染物、病毒或反应机制（见 `results/phase1_closeout/experimental_inputs_gaps.md`）。

## A. 单体基线（JOB-2026-0904-001…006 · 定量可用，但精度一致性待核查）

**计算当时方法（依据历史计算记录，非当前脚本版本）**——三份记录同口径（逐字核实）：
- H₂O：`models/00_baseline/h2o/calculation_record.md`
- O₃：`models/00_baseline/o3/calculation_record.md`
- O₂：`models/00_baseline/o2/calculation_record.md`

| 项 | 当时实际配置 |
|---|---|
| 泛函/基组 | ωB97X-D 的**不含经验色散部分**（libxc `XC_HYB_GGA_XC_wB97X_D`）/ def2-TZVP |
| **色散处理（关键历史口径）** | **−D2 仅作单点能量校正，未进入优化梯度与 Hessian**（记录原文："仅作单点能量校正、未进入优化梯度与 Hessian"） |
| 网格 / SCF | grid level 5（气相 H₂O 90064 点）/ conv_tol 1e-11 |
| 优化/频率 | 解析 Hessian；投影平动/转动后虚频 = 0（H₂O 气相梯度最大 5.63×10⁻⁶、水相 1.28×10⁻⁵ Eh/Bohr） |
| 热化学 | 298.15 K 自实现（内置 thermo() 单位缺陷弃用）；**ZPE/H/G 由不含 −D2 导数的 Hessian 导出**（−D2 仅加在电子能上） |
| 优化几何 | `models/00_baseline/h2o/optimized_gas.xyz`、`optimized_smd.xyz`（o3/o2 同目录结构） |
| 结果 | `results/00_baseline/h2o_gas.json`、`h2o_smd.json`、`o3_gas.json`、`o3_smd.json`、`o2_gas.json`、`o2_smd.json`、`baseline_summary.json`、`baseline_energies.csv`；汇总表 `jobs/job_index.md` |

**单体 −D2 全导数重算**：在 `run_artifacts/` 中**未发现**单体在 D2 全导数路径下的重算记录（仅有路径冒烟测试，如 `run_artifacts/01_pure_water_o3_h2o/phaseB_scanner_verify/scanner_verify_smoke_H2O.json`——属路径验证，非生产重算）。**登记为：无重算记录（如需 D2 全导数口径的单体热化学，需另行立项）。**

**精度一致性待核查（015 修订，取代 v1"可直接用于定量比较"的宽表述）**：
- 单体（grid5、D2 单点口径）↔ 复合物（气相 L7、D2 全导数；水相 L8）之间**任何能量差**——网格档位、色散导数处理、SCF 容差三项均不一致，**不得直接定量比较**；统一需同配置单点复核，且**不同几何/不同导数处理的热化学结果不能靠补一个单点视为全部统一**（H/G/ZPE 的导数口径差异须整批重算才可消除）。
- 同配置内的比较（如单体气相 vs 单体水相，同为 grid5+同 D2 口径）维持可用。

## B. 气相 O₃+H₂O 复合物候选（初始结构 + 气相定量比较可用）—— **v3 来源纠正**

**权威登记**：`run_artifacts/01_pure_water_o3_h2o/smd_restart/gas_phase_registry.json`（JOB-006 Phase A）。纠正详情与逐项核对见 `notes/gas_baseline_source_correction_2026-09-07.md`。

### c14_plus —— 当前主参考 = L8 端点 + L8 完整频率（JOB-005 已验收）
| 证据 | 内容 | 来源（完整路径） |
|---|---|---|
| **L8 端点（主参考）** | grad_max=2.541×10⁻⁶；gas/**L8**/−D2 全导数/SCF 1e-12,1e-9；判定：L7/L8 检查范围内局部极小证据成立（JOB-005） | `run_artifacts/01_pure_water_o3_h2o/c14_l8_check/l8_optimize_result.json`（哈希 `29ecc3617147fefb`） |
| **L8 完整频率（主参考）** | **12/12 内部模式双步长全正，于 L8 端点上** | `run_artifacts/01_pure_water_o3_h2o/endpoint_frequency/frequency_analysis_l8_c14.json`（哈希 `9333a0d95fb47ac8`） |
| L7→L8 结构保持 | RMSD=0.0204 Å | `run_artifacts/01_pure_water_o3_h2o/c14_l8_check/formal_mapping.json` |
| L7 端点+L7 频率（**历史保留**） | 12/12 正——已被 L8 主参考取代 | `run_artifacts/01_pure_water_o3_h2o/endpoint_frequency/endpoints_fixed.json`（c14 条目）、`frequency_analysis_revised.json`（`a4e10cf9a94d3f78`） |

### c06_plus —— 当前主参考 = L7 端点 + L7 频率；L8 核查仅最低方向
| 证据 | 内容 | 来源（完整路径） |
|---|---|---|
| **L7 端点（主参考）** | grad_max=2.142×10⁻⁶；判定：L7 检查范围内局部极小证据成立（JOB-004）；**L8 完整谱未检查（超出当时授权）** | `run_artifacts/01_pure_water_o3_h2o/endpoint_frequency/endpoints_fixed.json` |
| **L7 频率（主参考）** | 12/12 内部模式双步长全正（L7 端点上） | `run_artifacts/01_pure_water_o3_h2o/endpoint_frequency/frequency_analysis_revised.json`（`a4e10cf9a94d3f78`） |
| L8 方向核查范围 | **仅最低方向（L7/L8）；非完整 L8 内部谱** | gas_phase_registry.json `c06_plus.retained_evidence.grid_robustness_scope` |

### 历史阶段结果（不得与上述拼接为验收证据）
| 结果 | 归属原因 | 路径 |
|---|---|---|
| gas_freq 频率（c06/c14_plus） | **旧几何**（与 phaseC 端点实测最大差 0.106 Å）+ grid level 5 + "解析 DFT+解析 −D2 梯度有限差分"口径 | `run_artifacts/01_pure_water_o3_h2o/gas_freq/c06_plus.json`、`c14_plus.json` |
| endpoints_fixed 的 c14 条目 | L7 端点已被 L8 主参考取代（c06 条目仍为主参考） | `run_artifacts/01_pure_water_o3_h2o/endpoint_frequency/endpoints_fixed.json` |

- **可用于**：初始结构（c14→L8 端点；c06→L7 端点）；**同配置内**的气相定量比较。
- **禁止**：把不同几何上的优化与频率拼成一条验收证据；全局极小值声明 = 无（登记规则原文）。
- 精度一致性待核查（承接 v2）：c06（L7）与 c14（L8）主参考网格不同——跨体系气相能量差需同配置复核。
- 端点坐标哈希/索引：`run_artifacts/01_pure_water_o3_h2o/smd_acceptance_evidence/endpoint_index.json`（注意：012 索引引用的 c14 几何为 L7 端点，其 L8 主参考更新见本节）。

## C. 水相端点（**仅供诊断，不得用于定量结论**）

| 端点 | max\|g\|（未投影） | 验收 | 内部残差 | 配置 | 来源（完整路径） |
|---|---|---|---|---|---|
| c06_order41_step40 | 2.580×10⁻⁵ | **未通过** | 1.15×10⁻⁶ | SMD L8，表面档位 missing（来源未记录） | `run_artifacts/01_pure_water_o3_h2o/smd_restart/phaseC_smd_optimize.json` :: c06_plus |
| c06_order47_reopt | 2.503×10⁻⁵ | **未通过** | 1.31×10⁻⁷ | SMD L8 order47（试验档位） | `run_artifacts/01_pure_water_o3_h2o/smd_order47_opt/order47_optimization.json` |
| c14_cont120（累计 120 步） | 1.250×10⁻⁵ | **未通过** | 3.23×10⁻⁷ | SMD L8，档位 missing | `run_artifacts/01_pure_water_o3_h2o/smd_controlled_opt/c14_cont_result.json` |

- 可用于：继续诊断的固定几何（c06_order47 端点 = 013/015 曲率核查输入）；不可用于水相定量能量/频率/自由能结论。
- 统一索引：`run_artifacts/01_pure_water_o3_h2o/smd_acceptance_evidence/endpoint_index.json`。

## D. 方法与验证基础设施（后续批次直接复用）

| 资产 | 验证状态 | 路径 |
|---|---|---|
| −D2 全导数（能量/梯度/Hessian 一致注入） | Kr₂/Ne₂ 探针 + D2-once（1e-15 级） | `protocols/d2_full.py`；`notes/method_validation.md` |
| 生产路径（grid_response=True） | 011 批路径验证（scanner 实读；原生 SMD+d2 基准吻合 1.25e-12） | `protocols/grad_factory.py` |
| method_fingerprint（溶剂实读+档位区分） | 无 SCF 回归 5/5 | `protocols/test_method_fingerprint.py` |
| 力矩/梯度分解（Bohr 约定） | 回归 9/9 | `protocols/torque_decomp.py`、`protocols/test_torque_decomp.py` |
| 来源交叉核验/端点索引/配准 | 扰动必失败回归通过 | `protocols/smd_source_crosscheck.py`、`protocols/test_source_crosscheck.py` |
| 热化学自实现 | 内置 thermo() 单位缺陷规避 | `notes/method_validation.md` |

## E. 诊断性发现（**仅供诊断**）

1. 数值取向依赖与修正 T_g 一致（斜率残差 0.13–1.8%；c06 与 c14 两几何均有旋转证据）。
2. 表面档位敏感性：取向能量幅度 47(8.29e-8) < 31(1.56e-7) < 41(5.57e-7)，非单调，无收敛律。
3. c06_order47 端点软虚内模 −52.3 cm⁻¹；同几何同方向 order41 曲率为正（符号翻转）→ 软模归属待定（013 批，014 批表述补正后口径）。
4. 完整梯度残差为转动型外部分量主导（~5×10⁻⁵），内坐标优化不消除。

## F. 第一阶段尚缺证据（阻塞项）

| 缺口 | 阻塞什么 | 现状 |
|---|---|---|
| 水相完整驻点验收 | 水相频率/自由能/反应能学基线 | 1e-5 判据未过（转动型分量）；阈值未放宽 |
| 软模归属 | 水相基线几何可用性 | 41/47 曲率符号翻转待定；order59 核查见 015 批报告 |
| 单体同配置复核（能量+热化学口径） | 任何单体↔复合物定量能量差 | **未执行**；且热化学统一需整批同口径重算（不能只补单点，见 A） |
| 统一验收标准最终确定 | 第一阶段收尾 | 协议 A/B 已获批（B 为标签）；阈值变更待用户批准 |
