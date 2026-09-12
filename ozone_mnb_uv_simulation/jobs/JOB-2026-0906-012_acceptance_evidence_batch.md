# JOB-2026-0906-012：水相验收证据收束与后续协议草案批

## 范围与授权
- **纯离线**：零新增 SCF/优化/Hessian/旋转扫描。
- 全部产物位于项目目录；平台文件与 PySCF 只读；历史原始记录不覆盖。

## 执行状态（全部完成）
| 项 | 状态 |
|---|---|
| method_fingerprint 溶剂识别修复 + 无 SCF 回归 5/5 | 完成 |
| 缓存复用影响排查 + 受影响文件清单 | 完成（无缓存影响；14 文件列出，历史保留、新索引更正） |
| 正式水相端点索引（3 端点） | 完成 |
| Kabsch+等价置换配准（取代 011 粗比对） | 完成 |
| 统一证据表（4 行，缺失标 missing） | 完成 |
| 科学表述修订（011 报告×4 + JSON + 任务记录×3 + 记忆） | 完成（保留取代关系） |
| 验收/诊断协议草案（待审批） | 完成（未实施任何变更） |
| 下一步建议 | 完成（待协议草案获批） |

## 关键结果
- **指纹**：新指纹从 with_solvent 实读（solvent='water'、lebedev_order 等）；回归 T1–T5 通过（气相=gas、SMD=water+档位、41≠47 指纹、包装/scanner 可追溯、可序列化）。旧方案下 order41 与 order47 指纹逐字相同（已证实）；指纹从未用作缓存键 → 无缓存影响。
- **端点索引**：c14=008 批累计 120 步保存终点（max|g|=1.2504e-5）；c06 双端点并存——order41 step-40（2.5803e-5）与 order47 重优化（2.5034e-5）；三者**全部未通过** max|g|≤1e-5；c06_41/c14 的表面档位来源未记录 → missing，不回填。
- **配准**：c06_47 vs c06_41 RMSD=0.0319 Å（恒等置换）；c06_47 vs c14 **RMSD=0.6625 Å**（O3 端基互换+H 互换最优）——取代 011 未配准的 1.5998 Å。
- **证据表**：R1（旧几何×41）/R2（同几何×47，敏感性检查）/R3（47 重优化端点）/R4（c14）；不同几何或方向不得冒充单因素对照；旋转扫描存在于两个水相几何（JOB-013 补正：009 c14_plus 记录即正式 c14 端点，θ=0 坐标吻合 ≤5.6e-17；原文"仅存在于 c06 step-40"系错误表述，已撤回）。
- **表述修订**：四处按 012 要求改写并加取代声明；明确 order47 重优化未使完整梯度通过原标准；不称系统收敛；不自动选 41。
- **协议草案**（notes/acceptance_diagnostic_protocol_draft_2026-09-07.md）：A 维持 max|g|≤1e-5 不宣布通过；B 内部诊断标签（不替代 A）；C 诊断性频率仅说明价值/风险/必要检查。**无任何真正改变验收要求的内容；阈值变更只写"待用户批准"**。
- **下一步建议**：唯一问题=c06 order47 端点是否为内部极小值；最小计算=一次诊断性内部频率批（复用 011 端点与 012 证据，零重算）；分支行动、停止条件、前置审批均已写明；"两点证明"式规则被排除。

## 交付物
- `results/01_water_matrices/pure_water_o3_h2o/smd_acceptance_evidence/smd_acceptance_evidence_report.md` + `smd_acceptance_evidence_summary.json`
- `run_artifacts/01_pure_water_o3_h2o/smd_acceptance_evidence/{endpoint_index, registration, evidence_table}.json`
- `notes/acceptance_diagnostic_protocol_draft_2026-09-07.md`
- `protocols/{grad_factory.py（修改）, test_method_fingerprint.py, smd_acceptance_evidence.py}`

**本批停止，等待审议。** 未运行任何新量化计算；全部水相结果保持"未通过完整驻点验收"状态。
