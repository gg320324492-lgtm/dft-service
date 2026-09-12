# JOB-2026-0906-072：跨批汇总与汇报数据生成批（零求值）

## 范围与授权
- 编号=072。**量化预算：0 SCF、0梯度**。汇总JOB-026/064/066/068/069/070（及071方案）；不生成最终PPT；不决定数据截止点。

## 执行状态
1. 端点能量表（6行：NH₃/O₃/HNO/H₂O₂/HOO·/H₂NO·，含E、gmax、复核、稳定性、频率、热化学、验收状态）。
2. 派生参考登记：dE_P4与dE_P1frag均**未计算**（第二产物端点未验收，不拼接规则）。
3. S13文献严格分栏CSV（B3LYP/CCSD(T)/G3B3/ΔG₀+能垒+T1）；与项目数值零混列。
4. C1/C2结论、P4/P1可执行性、缺失坐标清单（TS1/TS2/TS8/CP1/CP3）。
5. 5张SVG（端点状态/方法分栏/HNO频率/C1C2/缺口）+3份CSV。

## 交付
`results/phase2_preparation/summary072_report.md`、`summary072/`（结果JSON、figures×5、CSV×3、exec_log）、脚本job072_summary.py、PROJECT_STATUS.md同步。

**完成即停止。窗口总预算账目：JOB-068 26 + JOB-069 2 + JOB-070 25 = 53 attempt（上限约72，未追加未授权计算）。**
