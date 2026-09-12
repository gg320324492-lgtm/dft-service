# JOB-2026-0906-003 三阶段批总报告：验收补正 → scanner 验证 → 有限步数终点优化

## 本批做了什么 / 完成了什么 / 还缺什么

- **做了什么**：阶段 A 复核并确认离线验收补正与生产路径回归；阶段 B 用统一梯度入口（grid_response=True + 项目 D2 恰一次）对真实 c14_plus L7 中心几何做 A→B→A 同 scanner 调用验证；阶段 C 对 c14_plus、c06_plus 各执行受 60 步上限约束的气相终点重优化，并以独立新建对象复核。
- **完成了什么**：A 通过（28/0 回归）；B 通过（全部判定为真，残差达机器精度）；C 两个候选均收敛并通过独立 max|g|≤1e-5 复核——c14_plus 46 步、c06_plus 25 步，均未触及 60 步上限。两端点 RMSD=1.59 Å，为不同构型，无去重合并需求。
- **还缺什么**：新端点的**完整内部频率验收**（全部内模 + 软模可靠性）尚未做——在此之前两候选仅为**已收敛驻点候选**，不是最终稳定极小值。平台解析 Hessian 层问题仍未修复（独立事项）。SMD 水相、高层级对照与反应路径研究按计划保持在后续批次。

**各阶段状态**：A = **通过**；B = **通过**；C = **通过**（attempt 2，见 §4）。

---

## 1. 阶段 A：验收补正与回归（通过）

权威交付：`results/.../unit_consistency_revision/phaseA_unit_revision_report.md`（取代关系见 §6）。要点复核确认：

1. **科学结论修订**：撤回"c06_plus ≈0、无负曲率证据、争议关闭"——准确记录 L7 小幅度（实际 0.0038 Å）处能量曲率 k_E = −1.4065×10⁻⁵ 与完整响应梯度曲率 k_g(True) = −1.1339×10⁻⁵ **均为负**，稳定性/网格敏感性/新驻点性质**仍待判断**，不把负值归零。c14_plus 仅表述为"已采样方向上，单位修正后能量与完整响应梯度更一致，有限步长曲率为正"，不等于全部内模或解析 Hessian 通过验收。
2. **audit.py 修复确认**：恢复的 q 保留完整精度（展示才舍入）；按候选/网格/幅度/符号严格匹配；中心按网格匹配、无静默回退；B 常量定义无未定义小写 b；旧值实际读取、缺失=missing/not_checked 不算通过；SCF 收敛读记录字段、缺失默认**未**收敛；中心/位移点数量与核验覆盖率明确区分。
3. **报告生成器**：所有表格含网格级别（L6/L7）与单位列（Eh/Bohr²、Eh/Bohr）。
4. **生产路径回归**（`test_production_path_regression.py`，**28 pass / 0 fail**）：共享无 SCF 副作用位移函数（`curvature_audit_lib.displaced_geometry`）被两个生产脚本与测试共同调用；测试从生成后的实际坐标算模型能量/梯度再走正式差分分析函数；覆盖 Bohr/Å 往返、正负位移、非轴向多原子方向、正/负曲率二次势（负号保持）、四次势 O(q²) 残差（k_g−k_E = c₄q²/12 验证）、**去掉 BOHR_A 后必失败项**、坐标不对称/方向错误/数据缺失正确报错。
5. **防旧缓存混用**：本批 B/C 使用独立批次目录（`phaseB_scanner_verify/`、`phaseC_reoptimization/`），方法指纹（xc/basis/grid_level/scf_tol/D2 参数/grid_response 默认值）写入全部 JSON；历史原始文件未改动；旧幅度数据未标记为新目标幅度。

## 2. 阶段 B：scanner A→B→A 调用验证（通过）

统一梯度入口 `protocols/grad_factory.py`：`make_mf_d2_gr` 使 `nuc_grad_method`/`Gradients` 恒返回 grid_response=True 的梯度对象，D2 经 `attach_d2` 恰好计一次；优化器（berny 经 `nuc_grad_method().as_scanner()`）收到的正是该对象。

真实 c14_plus L7 中心 + 明确小位移（水 O–H 键拉伸 0.05 Å），A→B→A 顺序调用同一 scanner（`scanner_verify_c14_plus_L7_center.json`）：

| 核对项 | 结果 |
|---|---|
| grid_response 实际调用时恒 True | **通过**（每步断言） |
| scanner = 独立 DFT(True) + 解析 D2 | **通过**：dE ≈ 1×10⁻¹³，max\|dG\| ≈ 2.2×10⁻¹⁰ |
| D2 计入恰好一次 | **通过**：(scanner − dft_only) − d2 残差：E ≈ 8×10⁻¹⁴、g ≈ 1.5×10⁻¹⁰（\|D2\|=1.70×10⁻⁴——若双倍计数残差应为 ~1e-4 量级） |
| 返回 A 可复现、无旧几何残留 | **通过**（max\|dG\| = 2.2×10⁻¹⁰） |
| grid_response 实际活跃（非空开关） | **通过**：True/False 梯度差 6.77×10⁻⁵ |
| 独立基准分离 | 全新 mf（独立 SCF）取 `_d2_parent_cls` 未包装 DFT 部分 + 项目解析 `d2_grad`，非同包装路径互比 |

容差预记录：D2-once / scanner 一致性按 |D2| 相对判（0.2×|D2|，双倍计数时残差/|D2|≥1，实际 ~5×10⁻¹⁰）；可复现 1e-7/1e-6。水二聚体冒烟路径同样通过（`scanner_verify_smoke_H2O_dimer.json`）。

排障记录（未发现 scanner/D2 路径错误，两处均为验证脚本自身缺陷）：(1) `make_mf_d2_gr` 曾错误重赋 `_d2_parent_cls`（实例属性查类回退到 RKS_D2），致"独立基准"能量已含 D2——修复为继承原实例属性；(2) 独立基准曾漏设 grid_response=True 致 g_dft 为 DFT_False——修复后残差降至机器精度。

## 3. 方法与优化器参数核对（C 前置）

- 方法未改：**ωB97X-D 的非经验色散 DFT 部分（xc='wb97xd'，libxc 不含 −D2）+ 项目经验色散全导数（Chai–Head-Gordon −D2，参数逐字不变）**、def2-TZVP、grid level 7、SCF conv_tol=1e-12 / conv_tol_grad=1e-9、气相。
- **pyberny 0.7.0 收敛参数核实**（读源码，与 PySCF 文档示例不同）：`gradientmax/gradientrms/stepmax/steprms`（BernyParams，原子单位，全判据 AND，**无能量判据**），默认 0.45e-3/0.15e-3/1.8e-3/1.2e-3；`maxsteps` 独立传参。步长阈值保持默认，未使用未确认参数，未绕过收敛检查。

## 4. 阶段 C：60 步上限终点重优化（通过，attempt 2）

**两次尝试的原因**：attempt 1（gradientmax=1e-5）优化器第 6/13 步即"收敛"，但独立 Cartesian 复核 max|g| = 3.13×10⁻⁵ / 1.38×10⁻⁵ > 1e-5。根因（pyberny 源码 L294）：**收敛判据作用在内坐标梯度 `dot(B_inv.T, g)` 上，非 Cartesian 梯度**——判据表征差异，非梯度路径错误。attempt 2 将内坐标阈值**收紧**至 1e-6（无任何放宽），从原始起点完整重跑；Cartesian 验收与 60 步上限不变。attempt 1 产物保留于 `phaseC_summary_attempt1.json` 等。

**attempt 2 结果**（`phaseC_summary_attempt2.json`；**attempt 2 属追加尝试**——每候选实际执行两次优化：c14_plus attempt 1 = 6 步 / attempt 2 = 46 步；c06_plus attempt 1 = 13 步 / attempt 2 = 25 步。后续批次不得自行追加第三次尝试）：

| 候选 | 步数 | 优化器收敛 | 独立复核 max\|g\| (Eh/Bohr) | rms\|g\| | 判定 |
|---|---|---|---|---|---|
| c14_plus | 46 / 60 | 是 | **2.641×10⁻⁶** | 1.07×10⁻⁶ | **已收敛驻点候选** |
| c06_plus | 25 / 60 | 是 | **2.142×10⁻⁶** | 8.29×10⁻⁷ | **已收敛驻点候选** |

- 独立复核 = 优化终点坐标**独立新建** mol + mf + grid_response=True 梯度：dE vs 优化器末步 ≤ 1.2×10⁻¹²（机器精度）；SCF 全程收敛（两个候选全部逐步 conv=True，无异常几何、无非有限值）。
- 能量降低：c14_plus ΔE = −2.64×10⁻⁶ Eh；c06_plus ΔE = −1.04×10⁻⁶ Eh。
- D2 计数在端点复核一致：解析 `d2_energy` = scf_summary['d2_dispersion']（c14: −1.733×10⁻⁴；c06: −1.603×10⁻⁴），恰好一次。
- 优化器收敛条件（全判据 AND，非单阈值）：gradientmax=gradientrms=1e-6（内坐标）+ stepmax/steprms 默认值；**Cartesian 验收另由独立复核满足**。
- **端点去重**：RMSD = 1.5945 Å（O₃···H₂O 质心间距 2.893 vs 3.663 Å）→ 数值上明显不同构型，两构型均保留。~~按配准与等价原子置换规则核查后确认~~——**撤回**：`rmsd_dedup` 仅做质心平移，**没有旋转配准与等价原子置换**；正式配准去重由 JOB-2026-0906-004 补做（Kabsch + 置换后 RMSD = 3.745 Å，结论不变：不同构型，`formal_dedup.json`）。
- 逐步坐标/能量/完整梯度/SCF 状态/优化器状态：`phaseC_steps_*_attempt2.json`（trajectory 完整保存）。

> **补正（JOB-2026-0906-004）**：`phaseC_reoptimize.py` 逐步记录只含梯度**最大值与 RMS**，未保存逐步完整梯度数组——属历史证据缺项，如实标注，不虚构补齐，也不重跑历史轨迹。逐步完整梯度自本批（004）起恢复保存。

**注意**：即使优化收敛，两候选仅称**已收敛驻点候选**——不是最终稳定极小值。

## 5. 修订与撤回清单（本批）

| 撤回/更正 | 依据 |
|---|---|
| "c06_plus ≈0、无负曲率证据、争议关闭"（阶段 A） | L7 小幅度 k_E、k_g(True) 均为负，无独立误差证据不得归零；性质留作开放问题 |
| "c14_plus 该方向争议关闭即候选合格"的过度引申（阶段 A） | 仅采样方向更一致 + 有限步长曲率为正，不等于全部内模/解析 Hessian 验收 |
| 旧曲率"被低估"方向的表述（阶段 A） | 旧 k_E 乘 b²、旧 k_g/s_E 乘 b 才是修正方向；报告统一按新分析重算 |
| attempt 1 的"优化器收敛"不等于 Cartesian 验收（阶段 C） | pyberny 判据在内坐标表征；Cartesian 验收以独立复核为准 |

未撤回（保留）：单位缺陷影响范围判定（2 批受影响、5 处不受影响）；b 因子交叉校验；O(q²) 非谐趋势的有限证据表述。

## 6. 文件指纹与取代关系

SHA-256（前 16 位）：

| 文件 | hash |
|---|---|
| protocols/grad_factory.py | 54a82f33c2f05718 |
| protocols/phaseB_scanner_verify.py | 6652d4188e9fa7b5 |
| protocols/phaseC_reoptimize.py | e068bba8308dda8b |
| protocols/unit_consistency_audit.py | e42ad5e5553b95f6 |
| protocols/phaseA_report.py | db57e5342662252e |
| protocols/test_production_path_regression.py | 9b8058b3128f3e81 |
| run_artifacts/.../phaseB_scanner_verify/scanner_verify_c14_plus_L7_center.json | 9573527c38de84fb |
| run_artifacts/.../phaseB_scanner_verify/scanner_verify_smoke_H2O_dimer.json | 739f95e50091b8f7 |
| run_artifacts/.../phaseC_reoptimization/phaseC_summary_attempt2.json | 186008c181d6f2a4 |
| run_artifacts/.../phaseC_reoptimization/phaseC_summary_attempt1.json | 5a89cd2c0bcd59c5 |
| results/.../unit_consistency_revision/phaseA_unit_revision_report.md | 4490fb582d6c0606 |
| results/.../unit_consistency_revision/phaseA_unit_revision_summary.json | 6fb42d70e1eafdb9 |

取代关系：本报告（JOB-2026-0906-003）取代 `unit_consistency_report.md`（JOB-2026-0906-002）中的判定性结论；002 报告数据表与缺陷确认仍有效并被 `phaseA_unit_revision_report.md` 继承。历史报告与全部原始 point_*.json 保留未动。attempt 1 的 C 结果被 attempt 2 取代（保留存档）。

## 7. 本批明确回答

1. **错误影响了哪些批次？** directional_confirmation（28 点）与 grid_response_diagnostic（10 点）受 Bohr/Å 位移缺陷影响；curvature_audit 探针、stepD 扫描、H₂O 三层定位、单元测试及全部频谱/子空间分析不受影响（Bohr 坐标自洽）。
2. **单位修正后，能量与完整响应梯度是否趋于一致？** c14_plus 是（k_g(True) ≈ k_E，13% 内一致）；c06_plus 小幅度处两者**均为负**（−1.41/−1.13×10⁻⁵），量值一致但**符号为负**，性质待判断。
3. **剩余差异是否呈 O(q²) 趋势？** 是——多项式测试验证 k_g−k_E = c₄q²/12 精确成立，真实数据两步长显示同向增长；**仅两步长，属有限证据，非严格收敛证明**。
4. **数据复用**：38 点原始能量/梯度/坐标全部可复用（已重分析）；**确需未来补算**：无本批范围补算；后续需新端点频率补算（下一批）。
5. **scanner 验证已通过**，终点优化已完成且通过独立复核。**下一批**：新端点完整内部频率验收与软模可靠性验收 → 通过后恢复水相、高层级对照与反应路径工作。

本批停止。
