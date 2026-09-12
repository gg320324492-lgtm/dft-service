# JOB-2026-0906-068：P4产物端点项目模型优化批（HNO验收；H₂O₂预算耗尽如实停止）

## 范围与授权
- 编号 = 067后下一个未占用（已核对）。**预算34 attempt**：HNO优化15＋复核1＋稳定性1；H₂O₂优化15＋复核1＋稳定性1。分项不可互借；任何真实计算异常→保存现场并停止整个批次，不自动重启、不清空账本、不跨分子补算。
- 触发线：未经投影Cartesian **max|g|≤1×10⁻⁵**（仅此触发独立复核）；不达标则停止该分子并如实登记。
- 方法（运行前固定）：wb97xd＋项目显式D2一次、def2-TZVP、L8、grid_response=True、SCF 1e-12/1e-9、气相、RKS候选（**仅P4产物端**；R/TS区T1风险保留）。
- 输入=JOB-067模板（哈希登记，来源标签保留）。

## 执行状态
1. **prep（零求值）**：模板元素/哈希核验；单体参考能（JOB-026）登记；预算34登记。
2. **mock测试6项全过**：预算表、广义BFGS流程、预算拒收、复核门控稳定性、单产物规则、隔离。
3. **HNO全链路验收**：优化9/15（阈值触发，max|g|=8.205×10⁻⁷）→独立复核PASS（dE=4.26×10⁻¹³）→内部稳定性stable_i=True。E_recheck=−130.4841962698 Eh。
4. **流程异常1次（单列，零SCF消耗）**：稳定性施加于未经kernel的新建mf（标量mo_occ在numpy 2.5崩溃）→现场归档、缺陷修正（稳定性对象=验收几何重新收敛的新mf）、续算复用全部已完成求值（零重复SCF、零预算重置）。异常登记于results.flow_anomaly与报告§4。
5. **H₂O₂预算耗尽未达门限**：15/15求值全部SCF收敛、0失败，但最低gmax仅6.42×10⁻³（h2o2_06/07），未触发复核/稳定性→**未验收，如实停止**。轨迹呈下降-振荡（扭转坐标收敛缓慢）。
6. **账目**：26/34 attempt、0计算失败；**P4产物参考能未计算**（按规则不拼接不完整能量）；JOB-026单体和锚点（−281.998121835 Eh）已登记供后续。

## 交付
`results/phase2_preparation/p4_products068_report.md`、`p4_products068/`（manifest、账本、26份求值记录、accepted_iterates、stability记录×2、`p4_products068_results.json`、归档的`p4_products068_results_aborted01.json`与`exec_log_first_run.txt`、mock测试、exec_log×2）、脚本四件（prep/test_mock/exec/resume_stab）、PROJECT_STATUS.md与长期记忆同步。

**完成即停止，等待下一批决定。不猜测TS、不运行IRC、不做P1(2)自由基优化、不算频率、不做水相或UV激发态；未修改DFT平台、依赖或文献原件。**
