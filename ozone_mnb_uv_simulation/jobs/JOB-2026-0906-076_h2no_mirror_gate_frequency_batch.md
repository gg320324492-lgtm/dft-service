# JOB-2026-0906-076：H₂NO·镜像门修复与条件频率批

## 范围与授权（用户2026-09-12批令）
- 编号=076。修复075镜像门（同层级参照）；minus从074末点续优化至与plus同层；同层级镜像门通过后对plus做条件频率；**minus分支：RMSD≤1e-6 Å→对称性免算Hessian（保留标签）；RMSD>1e-6 Å→minus另做1次锚点和独立Hessian**；P1条件式拼接。
- 预算：minus账本8（start1+opt5+recheck1+stab1）；主账本镜像门SCF 1；频率每端点3最多两端点。门限1e-5与镜像门限（1e-6 Å/1e-8 Eh/1e-7）预登记不放宽。

## 执行状态
1. **A（零求值）九项全过**：075账目拆分（链5+镜像1=6）与FAIL机理（层级参照，非物理不等价）写入；plus终=075 opt_02复核记录、minus起=074 minus opt_15可追溯；同层镜像闭合7.06×10⁻⁹。
2. **B：minus 2/5次达门限**→复核PASS（dE=8.53×10⁻¹⁴、gmax=6.131×10⁻⁶）→stab首试崩溃（manifest缺键，脚本缺陷；1次SCF无效消耗如实登记）→现场保存→resume（新账本1/1，复用全部求值，068先例）→**stable_i=True**（⟨S²⟩=0.754286）。
3. **C：镜像门字面FAIL**（dE=8.82×10⁻⁸、dg=1.67×10⁻⁶、层级dmax=0.067 Å均超门）——SCF已落盘复用（零新增SCF完成评估）；根因=非平面几何L8格点镜像残差+反射面实现不一致（bfp倾斜面vs z框架），定量归因、非物理不等价。
4. **直接同层闭合（零求值补充）**：reflect(minus_final) vs plus_final dmax=**1.17×10⁻⁸ Å**、能量差5.1×10⁻¹³ Eh——镜像等价确证。
5. **D：minus独立频率路径（批令授权的RMSD>1e-6分支）**→锚点PASS（1.7×10⁻¹³）→解析DFT+显式D2 Hessian（FD 24次单列）合成核验5.2×10⁻¹³→**频率265.36/1285.76/1432.74/1680.22/3468.14/3600.13 cm⁻¹零负模→H₂NO·锥形化极小值在本方法和检查范围内成立**（073平面驻点−193.61 cm⁻¹虚频的阱底；plus由镜像对称共享，标签保留）；plus自身频率按字面未执行（镜像门FAIL）。
6. **E：P1解锁并计算**：**ΔE_P1frag=−19.447 kcal/mol**（项目方法电子能；S13定性对照B3LYP −19.4/CCSD(T) −5.82/ΔG₀ −21.35，分栏不混表）。
7. 账目：SCF 11（含1次无效stab）；D2 FD 48次单列；0平台失败；3次执行层脚本缺陷全部现场保存后修复。

## 交付
`results/phase2_preparation/h2no_follow076_report.md`、`jobs/JOB-2026-0906-076_h2no_mirror_gate_frequency_batch.md`、`notes/job076_h2no_mirror_gate_correction_2026-09-12.md`、`h2no_follow076/`（audit、manifest、三本账本、eval记录、稳定性记录、mirror记录、direct_mirror_closure076.json、minus频率全套（Hessian npy+freq JSON+账本）、freq076_summary.json、p1_reference076.json、日志）、075横幅、PROJECT_STATUS同步。

**完成即停止。H₂NO·零负模极小值已登记（minus端点完整验收链）；P1片段能−19.447 kcal/mol（限定口径）；未启动TS/IRC/水相/UV/MNB界面/病毒模型；未修改DFT平台、依赖或文献原件。**
