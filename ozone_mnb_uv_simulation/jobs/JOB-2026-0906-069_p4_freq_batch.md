# JOB-2026-0906-069：P4产物频率验收批（仅HNO；项目方法模型）

## 范围与授权
- 编号=069（已核对068后下一个未占用）。**预算2 SCF attempt**：hno_anchor 1＋hno_hess_dft 1；频率/热化学后处理0 SCF；D2有限差分梯度调用（18＋18次）单独统计不入SCF账本。任何真实异常→保存现场停止整批。
- 触发条件分支：JOB-068仅HNO通过（H₂O₂预算耗尽未验收）→ 本批只处理HNO；不计算E_products/ΔE_project。
- 方法固定：wb97xd＋项目显式D2一次、def2-TZVP、L8、grid_response=True、SCF 1e-12/1e-9、气相、RKS仅限P4产物端。
- 锚点门：复现JOB-068复核记录 dE≤1e-8、dgrad≤1e-7、dcoords≤1e-9 Å、config逐键匹配。
- 合成Hessian门：|attached-kernel −(DFT+D2)|≤1e-10 Eh/Bohr²。

## 执行状态
1. **prep（零求值）**：来源记录哈希登记（复核记录＋稳定性记录＋068结果）；运行时config子集核验PASS；AST逐字提取run_baseline.thermochemistry（berny缺失故不import模块；原文件未改）。
2. **mock测试7项全过**（manifest/热化学提取逐字性/频率合成正负模保持/账本门/合成核验门/隔离）。
3. **真实执行**：anchor PASS（dE=1.71e-13）→解析DFT Hessian（grid_response置位验证）＋显式D2 Hessian（FD调用计数18）→合成核验dev=1.28e-12 PASS→频率1587.84/1726.93/2923.75 cm⁻¹、负模0→PySCF频率交叉核验0.046 cm⁻¹→项目热化学ZPE 8.918 kcal/mol、G_corr −4.413 kcal/mol→登记局部极小支持（方法与检查范围内）。
4. **账目**：2/2 attempt、0失败、0异常；D2 FD调用36次单独统计。

## 交付
`results/phase2_preparation/p4_freq069_report.md`、`p4_freq069/`（manifest、账本、eval记录、4份npy Hessian、逐字热化学提取件、结果JSON、mock测试、exec_log）、脚本三件（prep/test_mock/exec）、PROJECT_STATUS.md同步。

**完成即停止，不等待用户。不猜测TS、不运行IRC、不做P1复合物、不拼接P4能量；未修改DFT平台、依赖或文献原件。**
