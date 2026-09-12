# 全导数 -D2 验证报告（第 2A 阶段 Part 1，Part 1B 后更新为 v2 实现）

生成：2026-09-04（v2 重跑）　后端：PySCF / WSL2　测试体系：O3-H2O 接触构型（6 原子，气相，6-31G 仅用于提速）
实现：`protocols/d2_full.py` v2（子类化 attach_d2，Part 1B 修复版）　脚本：`protocols/verify_d2.py`　原始数据：`d2_verification.json`（本报告数值均自动读取）

## 结论：**全部通过（OVERALL_PASS = True）**

| 检查项 | 步长/说明 | 实测最大偏差 | 阈值 | 判定 |
| --- | --- | --- | --- | --- |
| 0 公式保真 d2_energy == chg_d2 | 逐字同式 | 0.000e+00 | 1e-14 | 通过 |
| 1 解析梯度 vs D2 能量中心差分 | h=1e-4 Bohr | 7.434e-14 | 1e-6 | 通过 |
| 2 Hessian vs 梯度中心差分 | h=1e-3 Bohr | 0.000e+00 | 1e-4 | 通过 |
| 2b Hessian 对称性 transpose(1,0,3,2) | — | 0.000e+00 | 1e-8 | 通过 |
| 3 attach_d2 可加性 E/G/H | 同一 SCF 对比 | 5.684e-14 / 2.142e-09 / 3.168e-09 | 1e-8 | 通过 |
| 4 包装组合精确性 E/G/H（含 2 个位移几何） | 父类旁路（`_d2_parent_cls`），同一收敛 SCF | 8.483e-14 / 5.801e-15 / 5.320e-13 | 1e-8 | 通过 |

## 关键说明

- **Part 1B 修复记录**：v1（实例属性包装）在"实际几何优化与频率调用路径"上被证实存在两个缺陷——scanner 能量调用执行在原始 mf 对象上（执行对象与几何持有对象分离，第 2 步能量偏差 +3.22 Eh 且非收敛）与 kernel() 返回值 2×D2 双计（实证见 `run_artifacts/01_pure_water_o3_h2o/d2_pre_fix_diagnosis.json`、`d2_pre_fix_probe.json`）。实现改为**子类化**（`mf.__class__` 重绑）：D2 经 `energy_tot` 进入 SCF 驱动恰好一次（hf.kernel 每步能量经 `self.energy_tot`），不再包装 kernel；梯度/Hessian 对象同样子类化，D2 项一律在调用时读取 live `self.mol`。Part 1 判据旁路由 `type(mf).method(mf)` 更新为 `mf._d2_parent_cls.method(mf)`，本报告为 v2 实现的重跑结果。
- 能量有限差分的端到端对比受 **DFT 后端自身 FD 噪声底** 限制（~1e-5 Eh/Bohr @ h=1e-4，SCF 误差 eps 的 eps/h 放大），对 D2 实现不具判别力，故不作为通过判据（见 JSON 内 fd_diagnostic_note）。
- 检查 4 取**同一收敛 SCF** 的未包装 DFT 能量/梯度/Hessian（经存档的父类），故 wrapped − unwrapped 必须精确等于解析 D2 项；在参考几何与两个位移几何上残差均为 1e-13~1e-15 量级，可排除 attach_d2 用错几何的整类 bug。
- D2 只依赖核坐标，SCF 密度/Fock 构建未改动。

## 对第 2 阶段的许可

Part 1 判据（本报告）**与** Part 1B 生产路径审计（`d2_production_path_audit.{json,md}`，scanner 坐标传播 / D2 单次计入 / berny 优化器端到端）**均通过**后，才允许开始 Part 2 —— >=12 个 O3-H2O 初始构型的 ωB97X-D2/def2-TZVP（全导数 D2）正式优化与频率。生产运行统一用 `protocols/d2_full.py` 的 `make_mf_d2`/`attach_d2`（v2）。
