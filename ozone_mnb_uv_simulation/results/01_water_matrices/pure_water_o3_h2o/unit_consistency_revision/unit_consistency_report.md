# JOB-2026-0906-003（验收补正 → scanner 验证 → 有限步数终点优化）· 阶段 A 修订报告

> 控制性任务：JOB-2026-0906-003。本文件修订 JOB-2026-0905-009 / JOB-2026-0906-002 的位移单位修复报告中的**科学结论部分**（§3.2、§4、§5），数据表与缺陷确认（§1–§2）维持不变。编号统一为 003 系列，以解决历史报告/脚本/任务记录编号不一致。

**本批结构**：阶段 A（验收补正与回归）→ 阶段 B（scanner A→B→A 调用验证）→ 阶段 C（c14_plus/c06_plus 各 60 步终点重优化）。阶段 B、C 交付见同目录 `phaseB_scanner_verify/` 与 `phaseC_reoptimization/` 及本批次总报告。

**取代关系**：本报告修订 `directional_confirmation/directional_report.md` 与 `grid_response_diagnostic/grid_response_report.md` 中基于错误单位的定量数值；两份历史报告保留。缺陷根因记录于 `notes/unit_defect_root_cause.md`。

**本批为离线 + 受控 SCF 混合**：阶段 A 纯离线（从已存 point_*.json 重算，无新 SCF）；阶段 B/C 的 SCF 仅用于验证与重优化，不引入新的位移单位问题（共享位移函数已建立，见 §A.4）。

## 1. 缺陷复现与确认

- **调用链**：`q_for_atom_disp` 返回 Bohr 约定幅度 （q·max|d_a| = a/BOHR_A Bohr = a Å）；`directional_confirmation.py` 与 `grid_response_diagnostic.py` 将 q·d 直接加到 **Å 坐标** → 实际最大原子位移 = a/BOHR_A Å = **1.8897×a**（放大 1/b）。

- **逐点实测确认**（38 点全部核验）：标称 0.002/0.005/0.010 Å 的实际最大原子位移分别为 **0.003779 / 0.009449 / 0.018897 Å**，放大系数 1.88971（=1/b，与理论一致）；方向偏差 0.00°/180.00°；正负对称性 |Δ+|/|Δ−| = 1.0000。实际送入 PySCF 的坐标为 10 位小数 Å 字符串（精度充分）。

- **受影响调用点清单**：

| 脚本 | 影响范围 | 判定 | 依据 |
|---|---|---|---|

| protocols/directional_confirmation.py | directional_confirmation 28 points | **AFFECTED** | q*d added to Angstrom coordinates (L117); q used as Bohr denominator in k_E/k_g/s_E |

| protocols/grid_response_diagnostic.py | grid_response_diagnostic 10 points | **AFFECTED** | same pattern (L158-160): q*d on Angstrom coordinates |

| protocols/curvature_audit_candidates.py | curvature_audit 3-layer probes (both candidates) | **NOT AFFECTED** | q*d added to mol.atom_coords(unit=Bohr) -> Bohr displacement, matching the Bohr convention of q |

| protocols/validation_stepD_softmode.py | mode energy scans | **NOT AFFECTED** | q*dx_cart_bohr added to atom_coords(unit=Bohr) |

| protocols/curvature_audit_unit_test.py | synthetic quadratic round-trip | **NOT AFFECTED** | synthetic model defined directly in the Bohr convention; no coordinate-unit translation involved |

| protocols/curvature_audit_h2o_layers.py | H2O three-layer localization | **NOT AFFECTED** | q_for_atom_disp on d reshaped (3,3) added to atom_coords(Bohr) |

| protocols/curvature_audit_h2o_worst.py | H2O worst-element probe | **NOT AFFECTED** | same Bohr-consistent pattern |

## 2. 修正后的曲率与斜率（从原始数据重算）

修正关系（交叉校验用）：q_actual = q_old/b；s_E_c = b·s_E_old；k_E_c = b²·k_E_old；k_g_c = b·k_g_old；中心投影梯度 d·g 不变。正式值全部从原始数据重算，b 关系仅作交叉校验（逐点通过）。

- **真实幅度标签**：旧"0.002/0.005/0.010 Å"点的实际幅度为 0.0038/0.0094/0.0189 Å，不冒充修复后目标幅度点。

### directional_confirmation / c06_plus

| 标称幅度(Å) | 实际幅度(Å) | q_actual (Bohr) | k_E(修正) | k_g(False,修正) | s_E(修正) |
|---|---|---|---|---|---|

| 0.002 | 0.0038 | 0.00718 | 8.0609e-05 | 8.9519e-05 | -3.77e-06 |

| 0.005 | 0.0094 | 0.01794 | 9.3687e-05 | 1.1516e-04 | -3.74e-06 |

| 0.010 | 0.0189 | 0.03588 | 1.4044e-04 | 2.0730e-04 | -3.63e-06 |

| 0.002 | 0.0038 | 0.00718 | -1.4050e-05 | -1.2122e-05 | -3.72e-06 |

| 0.005 | 0.0094 | 0.01794 | 2.2292e-07 | 1.6287e-05 | -3.70e-06 |

| 0.010 | 0.0189 | 0.03588 | 5.1194e-05 | 1.1781e-04 | -3.61e-06 |

### directional_confirmation / c14_plus

| 标称幅度(Å) | 实际幅度(Å) | q_actual (Bohr) | k_E(修正) | k_g(False,修正) | s_E(修正) |
|---|---|---|---|---|---|

| 0.002 | 0.0038 | 0.00997 | 3.0769e-05 | 1.3946e-04 | 1.08e-05 |

| 0.005 | 0.0094 | 0.02492 | 4.2890e-05 | 1.6345e-04 | 1.09e-05 |

| 0.010 | 0.0189 | 0.04983 | 8.6245e-05 | 2.4886e-04 | 1.09e-05 |

| 0.002 | 0.0038 | 0.00997 | 3.9324e-05 | 1.5391e-04 | 1.23e-05 |

| 0.005 | 0.0094 | 0.02492 | 5.1565e-05 | 1.7767e-04 | 1.22e-05 |

| 0.010 | 0.0189 | 0.04983 | 9.5207e-05 | 2.6229e-04 | 1.21e-05 |

### grid_response_diagnostic / c06_plus

| 标称幅度(Å) | 实际幅度(Å) | q_actual (Bohr) | k_E(修正) | k_g(False,修正) | s_E(修正) |
|---|---|---|---|---|---|

| 0.002 | 0.0038 | 0.00718 | -1.4063e-05 | -1.2123e-05 | -3.72e-06 |

| 0.005 | 0.0094 | 0.01794 | 2.2433e-07 | 1.6287e-05 | -3.70e-06 |


**grid_response=True 层（同方向）**：

| 实际幅度(Å) | k_g(True,修正) | k_E(修正) | 残差 k_gT−k_E |
|---|---|---|---|

| 0.0038 | -1.1338e-05 | -1.4063e-05 | 2.72e-06 |

| 0.0094 | 1.7216e-05 | 2.2433e-07 | 1.70e-05 |

### grid_response_diagnostic / c14_plus

| 标称幅度(Å) | 实际幅度(Å) | q_actual (Bohr) | k_E(修正) | k_g(False,修正) | s_E(修正) |
|---|---|---|---|---|---|

| 0.002 | 0.0038 | 0.00997 | 3.9340e-05 | 1.5391e-04 | 1.23e-05 |

| 0.005 | 0.0094 | 0.02492 | 5.1564e-05 | 1.7767e-04 | 1.22e-05 |


**grid_response=True 层（同方向）**：

| 实际幅度(Å) | k_g(True,修正) | k_E(修正) | 残差 k_gT−k_E |
|---|---|---|---|

| 0.0038 | 4.1648e-05 | 3.9340e-05 | 2.31e-06 |

| 0.0094 | 6.6127e-05 | 5.1564e-05 | 1.46e-05 |

## 3. 关键问题的最终回答

### 3.1 错误影响了哪些批次？

- **受影响**：directional_confirmation（28 点）、grid_response_diagnostic（10 点）——两者都以 Å 坐标加 Bohr 步长，且用 Bohr 约定 q 作差分分母。影响：报告的 k_E 被低估 b²=0.28 倍、k_g 与 s_E 被低估 b=0.529 倍；实际幅度放大 1.89 倍。

- **不受影响（依据）**：curvature_audit_candidates 的三层方向探针（Bohr 坐标 + Bohr 约定 q，自洽）；validation_stepD 的模式扫描（Bohr 坐标）；H₂O 三层定位与最差元素探针（Bohr 坐标）；单元测试（合成势直接定义于 Bohr 约定）；全部频谱/子空间分析（Hessian 本征分解无位移操作）。

### 3.2 单位修正后，能量与完整响应梯度是否趋于一致？

- **c14_plus：是**。修正后 k_E = +3.9×10⁻⁵（L7, 0.0038 Å），k_g(True,修正) = b·k_gT_old = 4.1648e-05 — 两者在 ~13% 内一致；开启响应前 k_g(False,修正) = +1.5×10⁻⁴ 与 k_E 差 ~4 倍。**grid_response 缺失解释了该方向的 k_E–k_g 差异，且无过冲**（旧报告的"过冲"结论源于单位错误，撤回）。

- **c06_plus（本批重新判定）**：L7 小幅度（实际 0.0038 Å）处，能量曲率 **k_E = −1.4063×10⁻⁵** 与**完整响应梯度**曲率 **k_g(True,修正) = −1.1338×10⁻⁵ 均为负**（grid_response 诊断直接给出，非仅 k_g(False)）。该负值**不得归零**：单位修正改变了幅度量级，但并未提供使该负曲率消失的独立误差证据（旧"≈0、接近噪声底"的判读不成立——即便幅值小，符号稳定且被两套独立数据一致给出）。其**稳定性、网格敏感性及新驻点性质仍待判断**（见 §4、§5）。0.0094 Å 处 k_E 转正（+2.2×10⁻⁷）属四次项非谐（O(q²) 趋势），不改变小幅度处的负曲率结论。

### 3.3 剩余差异是否呈 O(q²) 趋势？

- k_E 与 k_g(False) 均随实际幅度增长（四次项非谐），两曲率之差也随幅度增长——与含四次项势能的 O(q²) 有限步长差一致。

- 证据限制：每方向仅 3 个幅度、2 个网格；q→0 外推为二阶模型拟合，**不是严格收敛证明**。c14_plus 的 k0(E)≈+1.0~1.3×10⁻⁴ 与 k0(g)≈+2.5~2.8×10⁻⁴ 的剩余差（~1.5×10⁻⁴）在开启响应后缩小（见 3.2），指向响应项高阶分量的贡献。

### 3.4 数据复用与补算清单

- **可复用**：全部 38 点的原始能量/梯度/坐标（本批已重分析）；curvature_audit 的 Bohr 基探针；全部频谱与子空间分析。

- **需未来补算**：无（本批修正后无需补测；下一步为 scanner 验证与终点重优化，见 §5）。

## 4. 判定更新

| 方向 | 修正前判定 | 修正后判定 | 依据 |
|---|---|---|---|

| c14_plus 解析负模方向 | 待定（E/g 幅值差 >50%） | **该方向负曲率争议关闭（仅限该方向）**：修正后采样方向上 k_E 与 k_g(True) 均为正且一致（+3.9 vs +4.4×10⁻⁵）；k_g(False) 的偏大由缺失响应项解释。**但**：这仅说明已采样方向上能量与完整响应梯度更一致、有限步长曲率为正——**不等于全部内部模式或解析 Hessian 已通过验收**（几何收敛未达标 + 平台解析 Hessian 层问题未修复 + 其余 11 个内模未做同级核查） | 见 §5 撤回"全部验收通过"表述 |

| c06_plus 差分负模方向 | 待定（网格依赖 117%） | **撤回"争议关闭 / ≈0 / 无负曲率证据"**：L7 小幅度处 k_E = −1.41×10⁻⁵ 与 k_g(True) = −1.13×10⁻⁵ **均为负**，稳定且被两套独立数据一致给出，不得归零；单位修正不提供使其消失的独立误差证据。该方向的**稳定性、网格敏感性及新驻点性质留作开放问题**，待下一批（新终点频率验收 / 网格收敛研究）判定 | 见 §5 撤回清单 |

- **注意**：两条判定均**不**构成"候选通过极小值验收"。两个候选的完整极小值验收仍未完成（几何收敛未达标 + 平台解析 Hessian 层问题未修复 + 其余 11 个内模未做同级核查）。c14_plus 仅关闭"该方向负曲率争议"；c06_plus 的负值维持为开放问题。

## 5. 撤回的表述（本批新增）

| 撤回 | 原因 |
|---|---|

| "grid_response 过冲"（c14_plus，grid_response 报告） | 过冲由单位错误导致（gap_T 旧值 −6×10⁻⁵ 未经 b² 修正；修正后 ≈ 0）|

| "grid_response 假设未获支持"（c06_plus） | 单位修正后小幅度处 k_E ≈ k_gF 本就一致，该方向无需响应项解释，亦无矛盾需要解释 |

| "几何收敛与网格响应彼此独立" | 梯度定义改变（响应开关）会改变驻点验收的 max\|g\| 数值——两者在验收判据层面耦合 |

| "曲率阈值等价于固定 cm⁻¹ 阈值" | 曲率→频率依赖模式质量分布，不同方向不可用同一 cm⁻¹ 阈值 |

| "解析 Hessian 偏差解释 E–G 残差" | E–G 残差的已解释部分来自网格响应缺失；解析 Hessian 层问题独立存在并影响其自身验收，但不进入 E–G 残差分解 |

| "c06_plus ≈0、无负曲率证据、争议关闭"（本批新增撤回） | L7 小幅度处 k_E = −1.41×10⁻⁵ 与 k_g(True) = −1.13×10⁻⁵ 均为负，符号稳定、两套独立数据一致；单位修正改变幅度量级但未提供使负值消失的独立误差证据，故不得判为 ≈0 或"无负曲率证据"，亦不得宣布争议关闭 |

| "c14_plus 全部验收通过 / 该方向负曲率争议关闭即候选合格" | 修正后仅说明已采样方向上能量与完整响应梯度更一致、有限步长曲率为正；这不等于全部内部模式或解析 Hessian 已通过验收，候选仍待完整频率验收 |

## 6. 下一批建议（单一、最小）

- **scanner 调用验证**（统一梯度工厂 grid_response=True + D2 恰一次，scanner 两个几何验证开关存活与 D2 计数）→ 随后进行**有步数上限的终点重优化**（目标 max\|g\| ≤ 1e-5，以 grid_response=True 梯度为验收梯度）→ 新终点完整内部频率验收。
