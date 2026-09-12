# 气相基线来源纠正记录（JOB-2026-0906-016 · 2026-09-07）

**依据**：按 JOB-004/005/006 实际文件与 `run_artifacts/01_pure_water_o3_h2o/smd_restart/gas_phase_registry.json`（JOB-006 Phase A 登记）追溯**最新有效证据**，不按文件名或修改时间选择。原始数据一律未改动；本记录只纠正"哪些结果是当前主参考、哪些属历史阶段"。

## 1. 追溯结论（登记文件为准）

### c14_plus —— 主参考 = JOB-005 已验收 **L8** 端点 + **L8 完整频率**
| 证据 | 内容 | 来源 |
|---|---|---|
| L8 端点（**当前主参考**） | grad_max=2.5412×10⁻⁶，E=−301.874837967；pyberny 45/60 步（内坐标 1e-6）；方法：wb97xd+项目−D2 全导数 / def2-TZVP / **gas / L8** / grid_response / SCF 1e-12,1e-9；判定 "local minimum SUPPORTED within L7/L8 checks (JOB-005)" | `run_artifacts/01_pure_water_o3_h2o/c14_l8_check/l8_optimize_result.json`（来源哈希 `29ecc3617147fefb`；登记于 gas_phase_registry.json） |
| L8 完整频率（**当前主参考**） | **12/12 内部模式在双步长（h=0.002/0.004 Bohr）全正，于 L8 端点上** | `run_artifacts/01_pure_water_o3_h2o/endpoint_frequency/frequency_analysis_l8_c14.json`（哈希 `9333a0d95fb47ac8`；JOB-005 Phase C） |
| L7→L8 结构保持 | RMSD=0.0204 Å | `run_artifacts/01_pure_water_o3_h2o/c14_l8_check/formal_mapping.json` |
| L7 端点 + L7 频率 | 12/12 正（双步长）——**保留证据（历史）**，已被 L8 端点+L8 频率作为主参考所取代 | `run_artifacts/01_pure_water_o3_h2o/endpoint_frequency/endpoints_fixed.json`（c14 条目）、`frequency_analysis_revised.json`（哈希 `a4e10cf9a94d3f78`） |

### c06_plus —— 主参考 = 已验收 **L7** 端点 + L7 频率 + **L8 仅最低方向**核查
| 证据 | 内容 | 来源 |
|---|---|---|
| L7 端点（**当前主参考**） | grad_max=2.1417×10⁻⁶；判定 "local minimum SUPPORTED within L7 checks (JOB-004)；**L8 完整谱未检查（超出当时授权）**" | `run_artifacts/01_pure_water_o3_h2o/endpoint_frequency/endpoints_fixed.json`（登记于 gas_phase_registry.json） |
| L7 频率（当前主参考） | 12/12 内部模式双步长全正 | `run_artifacts/01_pure_water_o3_h2o/endpoint_frequency/frequency_analysis_revised.json` |
| L8 方向核查范围 | **仅最低方向（L7/L8 对照）；非完整 L8 内部谱** | gas_phase_registry.json `c06_plus.retained_evidence.grid_robustness_scope` |

### endpoints_fixed.json 与 gas_freq 的归属
- `endpoints_fixed.json`：c06 条目 = **当前主参考**（L7）；**c14 条目 = 历史保留证据**（其 L7 端点已被 L8 端点取代为主参考）。
- `run_artifacts/01_pure_water_o3_h2o/gas_freq/{c06_plus,c14_plus}.json`：**仅属历史阶段**——grid level 5、"解析 DFT + 解析 −D2（解析梯度有限差分）"口径，且其几何为**旧几何**（与 phaseC 端点实测最大差 **0.106 Å**，非同一构型）。**不得与 phaseC 优化拼接成一条验收证据。**
- 登记 `gas_phase_registry.json` 的既有规则重申："不同网格或几何的总能量不得拼合成新的精确构象能/结合能差"；全局极小值声明 = **无**（仅声明检查范围内的局部极小证据）。

## 2. 逐项核对（几何哈希/配置/频率几何/来源关联）

| 核对项 | 结果 |
|---|---|
| c14 L8 端点来源哈希 | `29ecc3617147fefb`（registry 记录一致） |
| c14 L7/L8 频率文件哈希 | `a4e10cf9a94d3f78` / `9333a0d95fb47ac8`（registry 记录一致） |
| gas_freq 几何 vs phaseC 端点 | **不同几何**（max diff 0.106 Å，逐点实测） |
| 原子顺序约定 | O,O,O,O,H,H（0–2 = O₃，中心 O=0；3 = 水 O；4–5 = 水 H；registry 注明"geometry-verified"） |
| 频率计算几何 | L7 频率在 L7 端点上；L8 频率在 **L8 端点上**（frequency_analysis_l8_c14.json, job=JOB-005 Phase C）——几何与优化端点一致，无拼凑 |

## 3. 补正影响范围

- `results/phase1_closeout/baseline_inventory.md` **v2 → v3**：B 节重写（c14 主参考改为 L8 端点+L8 频率；c06 主参考=L7 端点+L7 频率+L8 最低方向核查；endpoints_fixed c14 条目与 gas_freq 标为历史阶段）。
- 012 批端点索引/014 清单中"两候选均已气相验收"的表述**维持成立**，但补充：c14 的当前主参考为 L8（比 012 索引引用的 L7 端点更新）。
- **不影响**：水相端点索引（012）、013/015 曲率诊断（均为水相、与本次气相来源纠正无关）。
- 历史原始文件（gas_freq、endpoints_fixed、frequency_analysis_revised 等）全部保留原文。
