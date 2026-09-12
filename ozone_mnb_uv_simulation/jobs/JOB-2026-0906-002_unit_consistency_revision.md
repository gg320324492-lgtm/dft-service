# JOB-2026-0906-002：位移单位修复与已有导数数据重分析

日期：2026-09-06 ｜ 状态：已完成 ｜ 类型：项目脚本修复 + 离线重分析（零量化计算）
前置：JOB-2026-0905-009、JOB-2026-0906-001

## 任务

修正 directional_confirmation / grid_response_diagnostic 两批的 Bohr/Å 位移单位缺陷；离线重分析全部 38 点；回归测试；修订报告。

## 交付

- `results/01_water_matrices/pure_water_o3_h2o/unit_consistency_revision/unit_consistency_report.md`
- `results/.../unit_consistency_revision/unit_consistency_summary.json`
- 回归测试 `protocols/test_unit_consistency.py`（ALL PASS，含修复前缺陷复现项）
- 根因记录 `notes/unit_defect_root_cause.md`

## 结果摘要

- 单位缺陷确认：实际幅度 = 1.8897×标称（38 点逐一核验）；k_E 低估 ×0.28、k_g/s_E 低估 ×0.529。
- 单位修正后：**全部能量层与梯度层曲率沿两个争议方向均为正或 ≈0**——无负曲率证据存活；c14_plus 的 k_E–k_g 层间差异由缺失网格响应项解释（开启后 k_g(True) ≈ k_E，无过冲）；c06_plus 小幅度处 E/G 本就一致。
- 两候选维持**待定**（几何收敛 <1e-5 未达标 + 其余 11 内模未核查 + 平台解析 Hessian 层问题未修复）。
- 下一批：统一梯度工厂（grid_response=True）scanner 验证 → 有步数上限的终点重优化 → 新终点完整内部频率验收。
