# JOB-2026-0906-024：NH3/O3 同方法气相单体参考结构批

## 范围与授权
- 任务编号 = 023 之后下一个未占用（已核对）；预算上限 **42 次能量—梯度求值**（实际 **39**）。
- 两个单体（NH₃、O₃）限定：不启动加合物、产物通道、TS、IRC、高层级；不算 Hessian/频率。

## 执行状态（完成；预算内）
| 项 | 结果 |
|---|---|
| 计算前补正 | `notes/job024_premcomputation_corrections_2026-09-07.md`（任务书 v1.1：C/D 移出、13–23 粗估撤回；电子态 4 条收紧；S13 Table 1 视觉核对：R 行 T1=0.0346 属**组合反应物**、T1 在 CCSD 级、ZPE 口径待核实；"(1)"对→"疑似同内容重复"） |
| 计算前检查（无 SCF） | 全通过（D2 N/O/H 覆盖；D2 E-grad FD 一致 6.6e-15；输入往返；NH₃ 非平面初猜；O₃ 历史几何可追溯） |
| 受控优化 + 端点复核 | **NH₃**：E=−56.564219136 Eh、max\|g\|=**7.661×10⁻⁷** ✓；**O₃**：E=−225.433902700 Eh、max\|g\|=**3.514×10⁻⁶** ✓（历史基线几何在新方法下已收敛，一步完成） |
| RKS 稳定性分析 | 各 1 次（pyscf 内稳定分析，~4/13 s）：**None（未发现内不稳定）**；不跟随、不切 UKS；不排除多参考 |
| 账本 | attempts.json 持久化；**粒度缺陷（scanner 级 vs 逐求值）已发现并修正为逐求值** |

## 重大偏差（如实登记）
1. **NH₃ 优化求值超"每单体 ≤20"分配**：根因 = 我方驱动缺陷（对 `nuc_grad_method` 返回对象打 kernel 补丁 → GradScanner `__dict__` 拷贝后梯度冻结于初始几何），15 次求值浪费；其余超额用于收敛收紧（gradientmax 收紧至 3e-5 后达 7.7×10⁻⁷）。**总上限 42 未突破（39）**。
2. NH₃ 端点复核 3 次（v1/v4/v5；v5 有效）。
3. 根因已记录供后续驱动沿用：pyberny/GradScanner 路径禁止 kernel 补丁；使用回调记账 + maxsteps 硬上限（v3–v5 模式）。

## 结果与标签
- 同方法分离单体参考能之和 = **−281.998121836 Eh**（NH₃ −56.564219136 + O₃ −225.433902700）；**无加合物计算，不生成结合能**。
- 判定标签 = **"驻点候选"**（未投影 max|g|≤1e-5 + 优化器自身判据 + SCF 收敛 + 独立新对象复核）；**未算 Hessian/频率——不是已验收极小值；无 ZPE/焓/自由能**。
- 本批为**项目方法计算，不是 S13 原文方法复现**（方法不同，配置已从实际对象读回核对）。

## 未完成项
频率验收（下批）；加合物初猜（须先视觉核对 S13 Figure 1 坐标）；RKS 充分性确认（T1@CCSD 级不在本批）。

## 交付
`results/phase2_preparation/gas_monomer_reference_report.md` + `_summary.json`、`run_artifacts/02_nh3o3_reference/`（precheck、attempts.json、逐步记录、诊断、v3–v5 结果、日志）、`inputs/nh3o3_phase2/*.xyz`、`protocols/{gas_monomer_precheck, gas_monomer_reference, nh3_opt_continue2, nh3_consistency_diag}.py`、`notes/job024_premcomputation_corrections_2026-09-07.md`、任务记录（本文件）。

**本批停止。** 不追加档位/几何/频率；桌面文献原件、历史结果、DFT 平台及 PySCF 安装环境均未修改。
