# JOB-2026-0906-003：验收补正 → scanner 验证 → 有限步数终点优化批

**日期**：2026-09-06　**状态**：完成（A/B/C 全部通过）
**总报告**：`results/01_water_matrices/pure_water_o3_h2o/phaseC_reoptimization/JOB-2026-0906-003_three_phase_report.md`
**汇总 JSON**：同目录 `JOB-2026-0906-003_three_phase_summary.json`

## 阶段 A：验收补正与回归 — 通过
- 权威报告 `phaseA_unit_revision_report.md`（网格 L6/L7 列 + 单位）；c06_plus 撤回"≈0/无负曲率/争议关闭"，准确记录 k_E=−1.4065e-5、k_g(True)=−1.1339e-5 **均为负**，性质待判断、不归零。
- 回归 `test_production_path_regression.py`：**28 pass / 0 fail**（含去 BOHR_A 必失败项、负曲率符号保持、四次势 O(q²) 公式验证）。
- audit.py：q 全精度、SCF 字段缺失=False、覆盖率区分、旧值 missing 不算通过、无静默回退。
- 旧 002 报告头部补 003 取代关系；历史原始文件未动。

## 阶段 B：scanner A→B→A 调用验证 — 通过
- 统一梯度入口 `protocols/grad_factory.py`：`make_mf_d2_gr` 恒定 grid_response=True，D2 经 attach_d2 恰一次。
- 真实 c14_plus L7 中心验证（`phaseB_scanner_verify/scanner_verify_c14_plus_L7_center.json`）：
  D2 计一次（残差 E≈8e-14 / g≈1.5e-10，|D2|=1.70e-4）；scanner=独立DFT(True)+D2（dE~1e-13）；
  grid_response 恒 True 且活跃（vs False 差 6.8e-5）；可复现无残留（2.2e-10）。水二聚体冒烟亦通过。
- 排障：两处验证脚本自身缺陷（_d2_parent_cls 重赋回退、独立基准漏设 grid_response），scanner/D2 路径无错误。

## 阶段 C：60 步上限终点重优化 — 通过（attempt 2）
- **pyberny 0.7.0 参数核实**：gradientmax/gradientrms/stepmax/steprms（原子单位、全 AND、无能量判据）；
  **判据作用于内坐标梯度 dot(B_inv.T, g)（berny.py L294），非 Cartesian**。
- attempt 1（gradientmax=1e-5）：优化器 6/13 步"收敛"但 Cartesian 验收失败（3.13e-5 / 1.38e-5 > 1e-5）→ 存档取代。
- attempt 2（内坐标阈值收紧至 1e-6，无放宽，从原始起点重跑，验收与上限不变）：
  - **c14_plus：46 步收敛，独立复核 max|g| = 2.641e-6 ✓**（ΔE=−2.64e-6 Eh）
  - **c06_plus：25 步收敛，独立复核 max|g| = 2.142e-6 ✓**（ΔE=−1.04e-6 Eh）
  - 全程 SCF 收敛、无非有限值；端点 D2 解析值 = scf_summary 分量（恰一次）。
  - 端点去重：RMSD=1.594 Å（O₃···H₂O 间距 2.893 vs 3.663 Å）→ 不同构型，均保留。
- 逐步轨迹：`phaseC_steps_{candidate}_attempt2.json`。

## 边界遵守
未修改平台 dft_service/、根目录 scripts/、依赖与 PySCF 安装；未放宽阈值、未超 60 步、未续跑追加；
本批不含全量 Hessian、新热化学、结合自由能更新、SMD 或反应路径。

## 下一批（按序）
1. 新端点完整内部频率验收 + 软模可靠性验收（在此之前仅称"已收敛驻点候选"）；
2. 通过后恢复水相、高层级对照与反应路径研究。
