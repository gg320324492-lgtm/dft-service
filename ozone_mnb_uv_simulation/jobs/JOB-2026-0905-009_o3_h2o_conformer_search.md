# JOB-2026-0905-009：O3·H2O 局部水合构型搜索（第 2 阶段 A，Part 2）

- **日期**：2026-09-05
- **阶段**：第 2 阶段（水体基质）→ 2A（纯水 O₃·H₂O 局部水合模型），Part 2
- **状态**：执行中

## 已完成内容（前置）

- 第 1 阶段 H₂O、O₃、O₂ 基线计算已归档（ωB97X 优化/频率 + D2 单点校正口径）。
- 第 2A Part 1A：全导数 D2 的能量、梯度、Hessian 公式验证通过。
- 第 2A Part 1B：发现并修复项目内两个生产路径 bug（scanner 未实时使用新坐标；
  kernel() 曾重复计入 D2）。修复后气相与 SMD 下的 scanner、Hessian、kernel、
  energy_tot 和 berny 优化器端到端审计全部通过（`d2_production_path_audit.md`，
  OVERALL_PASS = True）。
- **第 2 阶段起正式计算可标注为"ωB97X-D 优化/频率（全导数 D2）"**。

## 本批目标

建立臭氧在单个水分子附近的稳定接触模式、相对稳定性与溶剂化响应，为之后的显式
水簇、界面模型及反应路径提供可信起始结构。

1. **构型池**：≥12 个 O₃·H₂O 初始构型（总电荷 0、单重态），覆盖水对 O₃ 末端氧供氢键、
   不同水氧接近方向、共面/非共面取向及近远接触距离；保存每个初始 XYZ、编号与构建规则。
2. **气相构型搜索**：统一 `make_mf_d2`（ωB97X-D/def2-TZVP、grid level 5、SCF conv 1e-11），
   全部初始构型几何优化；用结构 RMSD、氢键拓扑与总能量去重；保留全局最低点及气相最低点
   以上 3 kcal/mol 内的全部不同极小值；对保留构型做全导数 D2 频率、确认真实极小值并记录 `<S²>`。
3. **SMD(水) 比较**：从全部保留气相极小值出发做 SMD(水) 优化与频率；再次去重；报告气相与
   水相构型排序、关键距离与氢键拓扑变化；**仅解释为"局部水合结构趋势"**，
   不得将单水分子 + SMD 解释为真实水溶液结合常数。
4. **气相结合能与热化学**：未校正结合能与 Boys–Bernardi CP 校正相互作用能；ghost 原子
   不得引入虚假 D2 原子对；分别报告电子能、ZPE 校正能量与热化学结果；若报告气相缔合
   自由能，明确给出 1 atm 与 1 M 标准态（O₃ + H₂O → O₃·H₂O 的 1 atm → 1 M 修正
   = **−1.894 kcal/mol**）。
5. **高层级敏感性检查**：气相最低的两个不同构型做 CCSD(T)/aug-cc-pVTZ 单点相互作用能，
   同时报告 T1 诊断（或等效单参考可靠性指标）；资源不足则停止并报告需求，
   **不得以较低级方法冒充该对照**。

## 文件写入范围

- 初始结构：`inputs/o3_h2o_hydration/`
- 模型与最终构型：`models/01_water_matrices/pure_water_o3_h2o/`
- 任务记录：`jobs/JOB-2026-0905-009_o3_h2o_conformer_search.md`（本文件）
- 原始产物：`run_artifacts/01_pure_water_o3_h2o/`
- 结果：`results/01_water_matrices/pure_water_o3_h2o/`
- 脚本：`protocols/conformers.py`、`protocols/run_stage{1..5}_*.py`、`protocols/make_conformer_report.py`
- **只改项目目录；不修改 `E:\dft-service` 下任何平台源码、脚本、依赖或测试。**

## 计算设置（沿用第 1 阶段约定）

- 泛函/基组：ωB97X-D（libxc `wb97xd` DFT 部分 + 项目内全导数 −D2）/ def2-TZVP
- 网格 level 5；SCF conv 1e-11；max_cycle 200
- 优化器：`pyscf.geomopt.berny_solver.optimize`（Part 1B 已审计的生产路径），
  `OPT_CONV = gradientmax 2e-5 / gradientrms 1e-5 / stepmax 1e-4 / steprms 6e-5 / maxsteps 200`
- 频率：解析 Hessian（经 `make_mf_d2` 全导数 D2 增强）+ PySCF `harmonic_analysis` 投影取振动态
- 热化学：项目自实现 `run_baseline.thermochemistry`（PySCF 内置 thermo() 有 42.7× 单位 bug，已弃用）；
  T = 298.15 K，P = 101325 Pa（1 atm）
- 资源：WSL2 32 核 / 23 GB；多进程池并行，每进程 4 线程
- 高层级对照：CCSD(T)/aug-cc-pVTZ（冻结芯层，将明确报告）

## 判据与验收

- 全部初始构型的收敛状态与淘汰/保留理由记录在案。
- 气相与 SMD 最终构型：虚频数、`<S²>`、相对能（kcal/mol）、关键几何（接触距离、
  氢键角、O₃ 键长/键角）齐备。
- CP 与非 CP 气相相互作用能、ZPE 校正值与标准态说明齐备（1 atm 与 1 M 双列）。
- CCSD(T) 对照与 T1 诊断齐备；若不可行，明确报告资源需求而非降级。
- 方法局限写明：O₃ 的系统偏差（单参考 DFT 固有，非实现错误）、弱复合物低频模式、
  单水模型适用范围。
- 明确判定最低能构型可否作为下一阶段反应模型的初始结构。
- 产出机器可读 `conformer_summary.json` 与 `conformer_report.md`（数值自动读取）。

## 下一步

- 若本批合格：用最低能水合构型，选一个封闭壳层不饱和键小分子作脂质氧化替代物，
  研究 O₃ 的首条反应路径；之后再逐步加入 HCO₃⁻、Cl⁻ 和铵态氮等水体基质。
