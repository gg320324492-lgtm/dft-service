# JOB-2026-0904-008：全导数 D2 生产路径审计（第 2A 阶段 Part 1B）

- **日期**：2026-09-04
- **阶段**：第 2 阶段（水体基质）→ 2A（纯水 O₃·H₂O 局部水合模型），Part 1B
- **状态**：**已完成，全部通过（OVERALL_PASS = True）**

## 已完成内容（前置）

- 第 1 阶段基线 6 作业完成并归档（ωB97X 优化/频率 + D2 单点校正口径）。
- Part 1：`protocols/d2_full.py` 全导数 D2（能量/解析梯度/Hessian FD + attach_d2）实现；
  `protocols/verify_d2.py` 验证 OVERALL_PASS=True（机器精度判据体系）。
- Part 1 遗留问题：验证仅在"单对象调用路径"上通过，未审计**实际几何优化与频率调用路径**。

## 本批目标

1. **坐标传播审计**：同一 `make_mf_d2(mol_R1)` 对象创建实际优化所用的 gradient scanner
   （`mf.nuc_grad_method().as_scanner()`，与 `pyscf.geomopt.berny_solver` 构造方式一致），
   依次传入近接触与远离两个 O₃·H₂O 构型；scanner 返回的总能量/梯度须与"在新构型上重新
   创建的全导数 D2 对象"一致（证明 D2 用新坐标而非 attach 时旧 mol 坐标）。
2. **单次计入审计**：同一收敛构型上 E(total) = E(ωB97X) + E(D2)，`energy_tot()` 与
   `kernel()` 均满足、非遗漏非 2×；气相与 SMD(水) 各一次。
3. **优化器端到端审计**：经 `pyscf.geomopt.berny_solver.kernel`（生产优化路径）短程优化，
   逐步记录 O₃–H₂O 接触距离、总能量、裸 DFT 能量、D2 能量；终点重算梯度与 Hessian
   确认包装一致性；仅方法验证，不入正式构型搜索。
4. 若发现问题只修项目内脚本（`protocols/d2_full.py` 等），不改平台源码；修复后重跑全部审计。
5. 文档一致性：未加限定的"ωB97X-D 优化/频率"继续修正；仅通过本批审计的第 2 阶段计算
   才可标"ωB97X-D 优化/频率（全导数 D2）"。

## 文件写入范围

- 任务记录：`jobs/JOB-2026-0904-008_d2_production_path_audit.md`（本文件）
- 脚本：`protocols/diagnose_d2_scanner.py`（预修复诊断）、`protocols/audit_d2_production_path.py`（正式审计）
- 实现修复：`protocols/d2_full.py`、`protocols/verify_d2.py`（判据旁路更新）
- 原始产物：`run_artifacts/01_pure_water_o3_h2o/`
- 验收报告：`results/01_water_matrices/pure_water_o3_h2o/d2_production_path_audit.{json,md}`
- 方法结论：`notes/method_validation.md`

## 平台源码只读检查记录（不改任何平台文件）

- `pyscf/geomopt/berny_solver.py`：`mol = method.mol.copy()`；scanner = `method.nuc_grad_method().as_scanner()`；
  每步 `mol.set_geom_(...)` 后 `energy, gradients = g_scanner(mol)`（用返回值，不读 `.e_tot`）。
- `pyscf/grad/rhf.py` `SCF_GradScanner.__call__`：`self.reset(mol)` → `mf_scanner(mol)`（SCF scanner）
  → `de = self.kernel(**kwargs)`（经 scanner 类链解析，self=scanner）。
- `pyscf/scf/hf.py`：`SCF_Scanner` 以 `__dict__.update(mf.__dict__)` 复制实例字典；
  `SCF.reset` 只重绑 `self.mol`；`SCF.kernel = scf` → 模块级 `hf.kernel` 每步能量
  **经 `mf.energy_tot(dm, h1e, vhf)` 实例属性查找**。
- `pyscf/solvent/_attach_solvent.py`：`SCFWithSolvent` 覆盖 `get_veff/get_fock/energy_elec`，
  **不覆盖 `energy_tot`/`kernel`**；溶剂能量经 `energy_elec` 流入。

## 预修复诊断结论（2026-09-04，run_artifacts/d2_pre_fix_diagnosis.json + d2_pre_fix_probe.json）

当前实例属性包装方案在生产路径上存在两个真实 bug（6-31G 实证，实现级缺陷与基组无关）：

1. **scanner 路径执行对象与几何持有对象分离**：包装闭包捕获原始 mf 对象的绑定方法与
   attach 时 mol；`SCF_Scanner`/`SCF_GradScanner` 以 `__dict__.update` 复制实例字典后，
   scanner 第 2 步（mol.copy() 变异到远离构型）的能量调用实际在**原始 mf1 对象**上执行
   SCF（mf1.e_tot 被改写为 −298.3750 Eh，非收敛值），scanner 自身 e_tot 保持 0.0、
   converged=False（生产 berny 路径 assert_convergence 将直接 RuntimeError）。
   实测：scanner 第 2 步能量 −298.3751 vs 新建对象 −301.5995（偏差 +3.2244 Eh），
   梯度 max 偏差 4.03 Eh/Bohr（梯度走另一条对象链，返回旧坐标梯度）。
2. **kernel() 返回值 D2 双重计入**：hf.kernel 内部经 `mf.energy_tot`（已被包装，+1×D2）
   得 e_tot，kernel 包装再加 1×D2。实测 (E_kernel − E_bare)/E(D2) = 2.0000000005
   （2× 残差 5.6e-14）；(E_e_tot属性 − E_bare)/E(D2) = 1.0000000003。Part 1 检查 4
   比较"kernel 返回值 − e_tot 属性"，两个相反错误恰好抵消（差值=1×D2），造成假阳性通过。

## 修复方案（项目内，d2_full.py v2）

改为**子类化**（`mf.__class__` 重绑，PySCF 自身 solvent 附着同款机制）：
- 覆盖 `energy_tot`：`parent.energy_tot(self,...) + d2_energy(self.mol)`——D2 经
  `hf.kernel` 内部 `self.energy_tot` 调用**恰好自动进入一次** kernel() 返回值与
  `e_tot` 属性，**不得再包装 kernel**；
- 覆盖 `nuc_grad_method`/`Gradients`/`Hessian`：对返回的梯度/Hessian 对象同样子类化，
  其 `kernel` 末尾加 `d2_grad(self.mol)`/`d2_hess(self.mol)`（调用时取 live `self.mol`）；
- 存档 `mf._d2_parent_cls` 供判据旁路（原 `type(mf).method(mf)` 旁路在子类方案下失义）；
- 幂等保护 `_has_full_d2`。

## 验收标准

- 审计 A：scanner 在新构型返回的总能量/梯度与新建对象一致；Hessian 包装
  差值 = 解析 D2 Hessian（机器精度）。阈值校准：能量用 SCF conv_tol（1e-11），
  梯度用 PySCF 轨道梯度收敛容差 conv_tol_grad = √conv_tol（3.16e-6）——scanner 与
  新建对象是两次独立 SCF（播种不同），跨 SCF 总差以此噪声底为界；并加归因判据：
  D2 部分经父类旁路逐项剥离，对总差值的贡献须为零（机器精度）。
- 审计 B：气相与 SMD 各自满足 E(kernel() 返回值) = E(bypass) + E(D2)、
  E(energy_tot()) = E(bypass) + E(D2)、`e_tot` 属性同式；残差 < 1e-9，且与
  2×E(D2)、0×E(D2) 的偏离 > 1e-6（显式排除遗漏与双计）。
- 审计 C：berny 生产路径优化逐步能量/距离/D2 项合理变化；终点梯度、Hessian 包装差值
  = 解析 D2 项（< 1e-9）；终点重跑能量与轨迹末步一致（< 1e-8）。
- 全部数值自动写入 JSON 与 MD 报告（不手工转录）。

## 实测结果（2026-09-04，audit_d2_production_path.py，OVERALL_PASS = True）

- **A scanner 坐标传播**：ΔE(scanner−新建对象) = 6.8e-12（< 1e-11）；跨 SCF 总梯度差
  9.344e-07（< conv_tol_grad 3.16e-6）；D2 部分残差 scanner 侧 2.2e-16、fresh 侧
  4.4e-16，**D2 对总差值的贡献 4.4e-16（零）**，9.344e-07 全部为 DFT 收敛路径噪声
  （bypass−bypass = 9.344e-07，与总差完全相等）。
- **A2 Hessian 坐标追踪（6-31G）**：attach 后变异 mol 并重新 SCF，包装 − 父类旁路
  = 解析 D2 Hessian（新构型），残差 8.6e-12；e_tot 属性比值 1.0000000000。
- **B 单次计入**：气相与 SMD(水) 各自 kernel() 返回值 / energy_tot() / e_tot 属性
  三个入口 (E−E_bare)/E(D2) = 1.0000000008 / 1.0000000003，残差 ≤ 8.5e-14，
  与 0×/2× 分离度均为 1.04e-4。
- **C 优化器端到端**：berny 生产路径 10 步（maxsteps 截顶，收敛标志 False 属预期，
  仅方法验证）；接触距离 2.462 → 2.851 Å，能量 −301.857417 → −301.873898（−16.5 mEh），
  D2 项随重取向平滑变化（−1.035e-4 → −1.256e-4，极差 2.24e-5）；首步能量与 B 气相
  e_kernel 一致至 1e-13；终点新建对象重跑 ΔE = 1.5e-12，终点梯度/ Hessian 包装差值
  = 解析 D2 项（1.3e-15 / 1.8e-13）。
- 阈值校准说明：审计 A 梯度阈值由初设 1e-8 校准为 conv_tol_grad=3.16e-6（跨 SCF
  噪声底），依据是归因判据证明 D2 部分对差异贡献为零——实现正确性判据不受影响。

## 下一步

- 本批全部通过后：≥12 个 O₃·H₂O 初始构型正式优化（make_mf_d2，def2-TZVP）→ 频率
  （零虚频）→ 去重（RMSD/氢键拓扑/3 kcal/mol）→ SMD 水相比较 → CP/非 CP 结合能
  → CCSD(T) 单点对照。
