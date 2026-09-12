# JOB-073 端点报告：H₂O₂（续算验收＋零负模频率）— 项目方法模型

日期：2026-09-11。目录：`run_artifacts/02_nh3o3_reference/endpoint_cont073/h2o2/`。
结果：`h2o2_cont073_results.json`、`h2o2_freq073_results.json`；账本：`budget_h2o2_cont073.json`（23上限）、`budget_h2o2_freq073.json`。

> 项目方法模型，非S13原始三维结构复现。方法与JOB-026/068逐项一致：wb97xd＋项目显式D2一次、def2-TZVP、L8、grid_response=True、SCF 1e-12/1e-9、气相、RKS singlet。

## 起点（自动定位，无硬编码）

起点=JOB-068账本中h2o2_opt最后一个done记录**h2o2_15**的实际坐标（E=−151.5646927211，gmax=2.048×10⁻²；溯源哈希=accepted_iterate原始字节精确匹配，config逐键=父批）。起点验收（独立SCF）：**PASS，dE=1.71×10⁻¹³、dgrad=2.75×10⁻¹¹、dcoords=0**。

## 续优化与验收（账目22/23：start1＋opt19＋recheck1＋stab1；reuse生效零重复计费）

- 续优化19/20次新求值达阈值（BFGS状态重置，hess_inv0=I；门限1×10⁻⁵不变）；
- 独立复核**PASS**：dE=0.00e+00（精确复现）、dgrad=6.66×10⁻¹⁴；
- 内部稳定性（验收几何上新收敛mf，`return_status=True`四元组）：**stable_i=True**；
- **E_recheck = −151.5667326897180 Eh**（较068截止再降约2.0×10⁻³ Eh；gmax=3.99×10⁻⁶）。

## 频率验收（JOB-073阶段D）

解析DFT Hessian＋显式D2 Hessian（FD 24次单列）合成核验1.67×10⁻¹³ PASS；TR投影（同位素平均质量、秩6、全模保留）：

**398.64 / 1019.76 / 1350.53 / 1471.89 / 3835.43 / 3835.68 cm⁻¹，负模0**（PySCF同矩阵交叉核验；扭转模398.6为最低，物理合理）。

**登记：在本方法（wb97xd+D2/def2-TZVP/L8/grid_response，气相）和检查范围内支持局部极小值。** stable_i与零负模分别报告，两者均通过。

## 边界

非S13复现；电子能无ZPE/热/CP；stable_i≠路径可靠性；未修改平台/依赖/历史结果。
