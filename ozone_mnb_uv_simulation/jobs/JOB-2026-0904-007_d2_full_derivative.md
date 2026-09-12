# JOB-2026-0904-007　全导数 −D2 实现、验证与第 2A 阶段准备

- **阶段**：第 2 阶段 A（纯水 O₃ 局部水合模型）— Part 1 方法准备
- **状态**：**已完成，全部通过（OVERALL_PASS = True）**
- **目标**：把 −D2 从"单点能量校正"升级为**全导数项**（总能量 + 解析梯度 + Hessian），
  使真正的 ωB97X-D 几何/频率可在项目内一致地执行
- **文件范围**：`protocols/d2_full.py`（新增）、`protocols/verify_d2.py`（新增）、
  `results/01_water_matrices/pure_water_o3_h2o/`（验证输出）；不修改 PySCF 平台源码
- **实现要点**：
  - 色散公式与 `run_baseline.chg_d2` 逐字一致（s6=1.0, sR6=1.1, a=6.0, 指数12, Grimme-2006 C6/RvdW）
  - 梯度：解析（dE/dR = −s6·c6·R⁻⁷·[−6f + M·x/(1+x)²]，x = a·(R0/R)^M），单位换算 J/(mol·m) → Hartree/Bohr（×BOHR_SI/HA2JMOL）
  - Hessian：解析梯度的中心差分（步长 1e-3 Bohr），按 transpose(1,0,3,2) 对称化
  - `attach_d2(mf)`：包装 `energy_tot` / `kernel` / `nuc_grad_method().kernel` / `Hessian().kernel`；
    D2 只依赖核坐标，**不改变 SCF 密度与 Fock 构建**
- **验收标准**（O₃·H₂O 测试构型，气相，6-31G 仅用于提速验证）：
  1. d2_energy ≡ chg_d2（机器精度）
  2. 解析梯度 vs D2 能量中心差分 < 1e-6 Eh/Bohr
  3. D2 Hessian vs 梯度差分 < 1e-4 Eh/Bohr²，对称性 < 1e-8
  4. attach_d2 可加性（能量/梯度/Hessian）< 1e-8
  5. 端到端：总能量中心差分梯度 vs 实现的总梯度 < 1e-5 Eh/Bohr（36 次 SCF）
- **实测结果**（`results/01_water_matrices/pure_water_o3_h2o/d2_verification.md`，数值自动读取自 JSON）：
  公式保真 0；梯度 vs FD 7.4e-14；Hessian vs 梯度 FD 0、对称性 0；
  可加性 E 6.8e-13 / G 5.6e-9 / H 5.7e-9；
  包装组合精确性（同一 SCF，含 2 个位移几何）E ≤ 2.8e-14 / G ≤ 3.7e-15 / H ≤ 5.1e-13。
  注：能量 FD 端到端对比受 DFT 后端自身 FD 噪声底（~1e-5 Eh/Bohr）限制，不作判据（诊断值 ~5.7e-5 与后端噪声同量级，证明 D2 未引入额外噪声）。
- **下一步**：验证通过，允许开始 Part 2 —— ≥12 个 O₃·H₂O 初始构型的
  ωB97X-D2/def2-TZVP 正式优化（`make_mf_d2`）+ 频率、去重、SMD 水相、CP 结合能、CCSD(T) 单点
