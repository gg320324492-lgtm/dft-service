# JOB-2026-0906-073：24小时窗口结果补正与未验收产物续算批

## 范围与授权（用户2026-09-11指令）
- 编号=073。五阶段：A零求值补正与审计→B H₂O₂续算→C H₂NO·续算→D条件式频率→E交付。
- 新文件仅入jobs/notes/results/run_artifacts/；脚本位于`endpoint_cont073/scripts/`。平台/依赖/文献原件只读；原始记录保留，补正走横幅＋机器记录。
- 预算（独立账本，互不借用）：每续算分支23（start1+opt20+recheck1+stab1）；频率每端点3（confirm1+hess_dft1+hess_d2经典1，D2 FD单列）。门限1e-5不变；stable_i须来自return_status=True。

## 执行状态
1. **A（零求值）**：审计双分支全过（h2o2_15/h2no_15账本自动定位、溯源哈希精确匹配、config逐键、chk存在）；072补正三件套（notes文件、双报告横幅、audit073_results.json机器记录）。
2. **mock 7/7过**（含续算reuse零重复计费、分支独立、稳定性四元组、频率资格门控）；期间修复mock自身起点能量不自洽缺陷。
3. **B**：H₂O₂ 22/23——起点PASS（1.71e-13）→opt 19达阈→复核PASS（dE=0）→stable_i=True；E=−151.5667326897180；频率零负模（398.64–3835.68）→局部极小值支持。
4. **C**：H₂NO· 11/23——起点PASS→opt 8达阈→复核PASS→stable_i=True（⟨S²⟩=0.75428）；E=−131.1047092776074；频率**−193.61虚频**→**非极小值（一阶鞍点候选）**，stable_i独立报告；P1片段能不拼接。
5. **D**：HOO·频率补齐（1244.40/1462.40/3670.97，零负模→局部极小值支持，补正升级）；H₂O₂/H₂NO·条件频率如上；PySCF负模复数cast差异如实登记。
6. **E**：三份端点报告＋总报告＋`p4_reference073.json`（ΔE_P4=−33.137 kcal/mol项目方法电子能，两产物验收后按068预登记公式零求值计算；定性对照S13）＋PROJECT_STATUS同步。

## 交付
`results/phase2_preparation/job073_continuation_report.md`＋三份端点报告；`endpoint_cont073/`（audit073_results.json、per-branch manifest/账本/求值记录/accepted_iterates/稳定性记录/结果JSON×2/频率账本×3/Hessian npy×9/summary、p4_reference073.json、mock测试、日志）；notes/job073_072_correction_and_continuation_2026-09-11.md；PROJECT_STATUS.md条目。

**完成即停止。不猜测TS、不跟随H₂NO·负模方向（属新授权）、不启动IRC/CP；未修改DFT平台、依赖或文献原件。**
