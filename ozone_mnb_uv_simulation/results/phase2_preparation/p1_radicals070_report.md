# JOB-2026-0906-070 报告：P1(2)自由基产物端点（HOO·验收；H₂NO·预算耗尽如实停止）— 项目方法模型

> **【JOB-073阶段A补正横幅，2026-09-11】** §3"HOO·（P1片段）接受"指优化＋独立复核＋内部轨道稳定性（stable_i=True，`return_status=True`返回值）通过；HOO·为**驻点候选，振动曲率未核查（频率未算），不得称局部极小值**。"全链路验收"措辞相应收窄（不含频率/曲率）。频率核查由JOB-073阶段D条件式执行。详见[notes/job073_072_correction_and_continuation_2026-09-11.md](../../notes/job073_072_correction_and_continuation_2026-09-11.md)。原文其余内容不动。

日期：2026-09-11。批目录：`run_artifacts/02_nh3o3_reference/p1_radicals070/`。
结果：`p1_radicals070_results.json`；账本：`budget_p1radicals070.json`；mock测试：`mock_test_results.json`（7/7通过）。

## 0. 模型地位（总标注）

> **项目方法模型，非S13原始三维结构复现。** 输入为JOB-067模板（S13 Figure-1参数＋登记的补项，非作者坐标）。所有数值不与S13 CCSD(T)混表。

## 1. 方法与电子态（运行前固定）

- wb97xd（libxc）＋项目显式D2一次（d2_full.attach_d2）；def2-TZVP；grid level 8；grid_response=True；SCF 1e-12/1e-9；气相；
- 每个自由基：电荷0、**spin=1（doublet，UKS）**；RKS不适用（开壳层端点）；每次kernel后记录 `spin_square()`：⟨S²⟩、测得多重度、污染量Δ=⟨S²⟩−0.75。

## 2. 执行与账目（SCF attempt账本，上限34）

| 类别 | 上限 | 用量 | 结果 |
|---|---|---|---|
| hoo_opt | 15 | 8 | 阈值触发（hoo_08，max\|g\|=8.20×10⁻⁶） |
| hoo_recheck | 1 | 1 | **PASS**：dE=5.68×10⁻¹⁴ Eh，dgrad=1.39×10⁻¹³，dcoords=1.39×10⁻¹⁷ Å |
| hoo_stab | 1 | 1 | **stable_i=True**（新建mf在验收几何收敛后做内部稳定性，068流程缺陷修正内置） |
| h2no_opt | 15 | 15 | 预算耗尽未达门限（最低gmax=2.08×10⁻²，第15次求值；能量截至截止点仍在下降：−130.875→−131.104 Eh，梯度振荡0.02–0.16） |
| h2no_recheck / h2no_stab | 1/1 | 0/0 | 未触发（门控未过） |
| **合计** | 34 | **25** | **0计算失败、0流程异常**；HOO与H₂NO分项互不借用 |

## 3. 验收登记

- **HOO·（P1片段）接受**：E_recheck = **−150.9243782856 Eh**；⟨S²⟩=**0.7540**（理想doublet参考0.75，污染Δ=+0.0040，几乎理想）；stable_i=True。多重度实测1.7547。
- **H₂NO·（P1片段）未验收**：15次求值全部SCF收敛、0失败，但未达max|g|≤1×10⁻⁵门限；未触发复核/稳定性；⟨S²⟩未产生（复核未运行）。轨迹"下降-振荡"（与068 H₂O₂的扭转坐标缓慢收敛定性相似）。
- **P1分离片段电子能未计算**（需两个自由基都通过复核；不拼接不完整能量）。单体和锚点 −281.998121835 Eh 登记供后续。

## 4. 自旋记录说明

⟨S²⟩已按求值逐条保存（eval记录含s2_total/s2_delta字段）；HOO·验收点污染+0.004属"干净doublet"。**注意：⟨S²⟩接近理想≠证明该自由基在整条路径上都适合单参考描述**；S13对R/TS区的T1风险保留。

## 5. 边界声明

- 未构造TS1/TS2/CP1；未猜测任何TS三维坐标；未运行IRC/完整路径；
- H₂NO·未验收 → 若未来获得续算授权，属**新批决策**，本窗口不自动重启；
- 电子能定义：无ZPE/热校正/CP；非S13 CCSD(T)；≠水相自由能；≠养殖水处理效率；
- 未修改DFT平台、依赖、PySCF、历史结果；真实计算全部入账本（25/34）。

## 6. 交付

`results/phase2_preparation/p1_radicals070_report.md`（本文）、`p1_radicals070/`（manifest、账本、24份求值记录=9+15，稳定性attempt不产生eval文件、accepted_iterates×2、稳定性记录×1、chkfiles、结果JSON、mock测试、exec_log）、脚本三件（prep/test_mock/exec）、PROJECT_STATUS.md与计划注记同步。
