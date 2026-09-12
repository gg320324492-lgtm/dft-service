# 方法验证档案：第 1 阶段基线（H₂O / O₃ / O₂）

生成日期：2026-09-04　后端：PySCF 2.14.0 / WSL2 Ubuntu 24.04.4
配套文件：`results/00_baseline/baseline_acceptance.md`、`protocols/audit_and_summary.py`

本文件记录三类方法学结论，供第 2 阶段及后续审计引用。

---

## 1. −D2 色散实现审计

### 现状
PySCF 的 `xc='wb97xd'` 路由到 libxc 的 `XC_HYB_GGA_XC_wB97X_D`，该条目**只包含 ωB97X-D 的 DFT 部分，不含经验 −D2 色散**（已用 Kr₂/Ne₂ 探针验证无 R⁻⁶ 尾巴）。
因此 `protocols/run_baseline.py` 用 `chg_d2(mol)` 显式补加 Chai–Head-Gordon −D2（s6=1.0, sR6=1.1, a=6.0, 指数12, C6/RvdW 取 Grimme 2006）。

### 审计结论（关键）
**−D2 仅参与总能量，未参与优化梯度，也未参与 Hessian/频率。**

证据（`audit_and_summary.py` 在 O₃ 优化几何上的有限差分核查）：
- 总能量 = E(ωB97X) + D2；优化用 `optimize(make_mf(...))`，`make_mf` 不含 D2 钩子 → 几何是 E(ωB97X) 的驻点，不是 E(总) 的驻点。
- Hessian 用 `mf.Hessian().kernel()`，为纯 ωB97X Hessian，不含 D2 曲率。
- 有限差分：沿原子0-x，存储梯度（E_DFT 解析）= −8.5e-7 Eh/Bohr；总能量有限差分梯度 = +8.5e-7 Eh/Bohr；二者之差（D2 对梯度贡献）= 1.7e-6 Eh/Bohr（**非零**）。
- `max|D2 梯度| ≈ 9.75e-6 Eh/Bohr`；`max|D2 Hessian| ≈ 7.6e-5 Eh/Bohr²`。
- 二者量级极小（典型键力常数 ~0.5–1 Eh/Bohr²），对几何/频率影响可忽略（<1e-4 Bohr、远小于 1 cm⁻¹）。

### 命名更正（强制）
- 现有结构/频率**不得称 “ωB97X-D 优化/频率”**，应称 **“ωB97X 几何/频率 + D2 单点校正”**。
- 总单点能量 `E(ωB97X-D) = E(ωB97X) + D2` 仍有效。

### 最小重算方案（第 2A 阶段已实现并验证；基线 6 作业未重跑）
若需严格 ωB97X 优化/频率 + D2 全导数校正（即真正一致的 ωB97X-D），须把 D2 作为 `pyscf` 可微项挂入 mf 对象（如 `dftd3` 或自定义 EnergyTerms / Gradients / Hessian 钩子），使 SCF、梯度、Hessian 三处一致包含 D2，再重跑 6 作业。当前基线作为相对能/势垒参照仍可用，只要统一沿用 “ωB97X 几何/频率 + D2 单点” 口径。

**2026-09-04 更新（第 2A 阶段 Part 1 完成；同日 Part 1B 修正实现）**：全导数 D2 已实现于 `protocols/d2_full.py`。Part 1B 生产路径审计发现初版（实例属性包装 kernel/energy_tot/nuc_grad_method/Hessian）在实际优化调用路径上不成立：scanner 能量调用会执行在原始 mf 对象上（执行对象与几何持有对象分离，实测第 2 步能量偏差 +3.22 Eh 且非收敛），且 `hf.kernel` 内部经 `self.energy_tot` 已注入一次 D2、kernel 包装再加一次，返回值 2×D2 双计。实现已改为**子类化**（`mf.__class__` 重绑，v2）：D2 仅经 `energy_tot` 进入 SCF 驱动（恰好一次，kernel() 返回值与 `e_tot` 属性自动一致，不再包装 kernel），梯度/Hessian 对象同样子类化，D2 项一律调用时读取 live `self.mol`；解析梯度 + 梯度中心差分 Hessian；不改变 SCF 密度。Part 1 机器精度验证与 Part 1B 生产路径审计（scanner 坐标传播 / D2 单次计入气相+SMD / berny 优化器端到端）全部通过后（`results/01_water_matrices/pure_water_o3_h2o/d2_verification.md`、`d2_production_path_audit.md`），第 2 阶段起的正式优化/频率统一用 `make_mf_d2`/`attach_d2`（v2），可称 "ωB97X-D 优化/频率（全导数 D2）"；第 1 阶段基线 6 作业保持原口径不重跑。注意：能量有限差分端到端对比受 DFT 后端自身 FD 噪声底（~1e-5 Eh/Bohr）限制，不作为 D2 判据，精确判据用同一 SCF 的包装/父类旁路（`_d2_parent_cls`）对比。

**2026-09-04 更新（第 2A 阶段 Part 1B 生产路径审计完成，全部通过）**：`protocols/d2_full.py` v2（子类化 attach_d2）已通过生产路径审计（`results/01_water_matrices/pure_water_o3_h2o/d2_production_path_audit.md`，def2-TZVP/grid5 生产设置）：(A) scanner 坐标传播——与 berny_solver 构造路径逐行一致的 gradient scanner 在远离构型的能量与新建对象一致至 6.8e-12，梯度总差 9.3e-7 经父类旁路归因全部为 DFT 收敛路径噪声（D2 贡献 4.4e-16 ≈ 0），D2 项确证使用新坐标；(A2) Hessian 包装差值 = 解析 D2 Hessian（新构型，8.6e-12）；(B) kernel() 返回值 / energy_tot() / e_tot 属性三入口在气相与 SMD(水) 均 E(total) = E(ωB97X) + E(D2) 恰好一次（比值 1.0000000003~1.0000000008，残差 ≤ 8.5e-14，与 0×/2× 分离 1.04e-4）；(C) berny 生产路径 10 步优化轨迹自洽（D2 项随水分子重取向平滑变化），终点新建对象重跑 ΔE = 1.5e-12，终点梯度/Hessian 包装差值 = 解析 D2 项（1.3e-15 / 1.8e-13）。据此，**第 2 阶段起经 `make_mf_d2`（v2）执行的正式优化/频率可标注"ωB97X-D 优化/频率（全导数 D2）"**；第 1 阶段基线 6 作业维持"ωB97X 优化/频率 + D2 单点校正"口径不重跑。跨 SCF 比较的噪声底：能量 ~conv_tol（1e-11）、梯度 ~conv_tol_grad（√conv_tol），判据设计须以父类旁路归因隔离实现误差与收敛路径噪声。

---

## 2. 热化学实现与溶液标准态

### 自实现热化学
PySCF 内置 `pyscf.hessian.thermo.thermo()` 有单位错误（用原子质量而非电子质量，ZPE 偏小 42.7×），已弃用。
热化学由 `run_baseline.py` 自实现：刚性转子 + 谐振子 + 理想气体，SI 单位，T = 298.15 K、P = 101325 Pa（1 atm）。
标定：气相标准熵与 NIST 偏差 H₂O −0.32、O₃ −1.66、O₂ −0.37 J·mol⁻¹·K⁻¹，实现可信。

### 溶液标准态问题（必须标注）
- 气相 G 采用 **1 atm** 理想气体平动标准态。
- 水相 G 的热化学部分使用**同一 1 atm 约定**，**未修正到 1 mol/L**。
- 因此水相 G 与气相 G **不能直接混用**为实验的 1 M↔1 atm 水合自由能。
- 修正：将气相分子从 1 atm 转到 1 M 标准态需 **+1.894 kcal/mol**（RT·ln V_m，V_m = RT/P = 24.47 L/mol @ 298.15 K）。严格 1 M 水合自由能 = 报告 ΔG + 1.894 kcal/mol。
- 修正后 ΔG(1 M)：H₂O −6.57、O₃ −1.60、O₂ +1.49 kcal/mol，与实验水合自由能量级一致（H₂O 接近实验 −6.3），验证修正方向正确。

### 后续规则
比较水相/气相自由能时，要么对水相 G 施加 +1.894 kcal/mol 的 1 M 修正，要么全程统一 1 atm 口径、不与实验 1 M 水合自由能混比。

---

## 3. O₃ 相对于高层级参考的偏差

### 扫描设置
固定键角 118.13°，扫描单 O–O 键长，wB97X-D / HF / CCSD / CCSD(T) 四级理论对比；抛物线拟合极小点键长。
结果文件：`results/00_baseline/o3_ccsdt_scan.txt`。

### 极小点键长
| 方法 | r_min / Å |
| --- | --- |
| wB97X-D（本工作流 DFT 部分） | 1.2393 |
| CCSD | 1.2483 |
| **CCSD(T)** | **1.2731** |
| 实验 | 1.2717 |

### 结论
- CCSD(T) 极小点与实验几乎重合，而 ωB97X-D 仍系统性偏短约 0.034 Å；O₃ 伸缩频率亦偏高约 200 cm⁻¹。
- **该参考扫描支持此偏差是当前泛函层级的系统偏差，且未发现所审计实现导致该偏差的证据**（SCF 层面已确认稳定：RKS 解，破缺对称性 UKS 初猜仍收敛回同一能量与 ⟨S²⟩=0）。固定角度的一维扫描本身不足以构成对根因的完全证明，仅作为支持性证据。
- 判定：**可记录的系统偏差**。作为相对反应能/势垒的参照基线可接受，但涉及 O₃ 的相对能/势垒须做方法敏感性检验（如与 CCSD(T) 或范围分离双杂化对照），不可将 O₃ 几何/频率直接外推为定量预测。

---

## 4. 审计脚本与可复现性
- `protocols/audit_and_summary.py`：读取 6 个结果 JSON 与 `o3_ccsdt_scan.txt`，执行 D2 有限差分审计，自动产出 `results/00_baseline/baseline_summary.json` 与 `baseline_acceptance.md`（所有数值自动读取，无手工转录）。
- 运行：`wsl -d Ubuntu-24.04 -- ~/dftvenv/bin/python /mnt/e/dft-service/ozone_mnb_uv_simulation/protocols/audit_and_summary.py`
