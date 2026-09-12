# JOB-2026-0906-075：H₂NO·镜像等价分支代表性长续优化批

## 范围与授权（用户2026-09-12批令）
- 编号=075。只延续JOB-074 plus代表分支寻找真实极小值；minus以一次镜像单点核对确认无需重复优化。
- 预算：plus账本34（start1+opt30+recheck1+stab1+anchor1）＋主账本镜像核对1＝总上限35。门限1e-5不放宽；stable_i须return_status=True；负模不称TS（IRC未授权）；镜像失败则不转移、分别报告并停止。
- 方法不变：UKS doublet spin=1、wb97xd＋项目显式D2一次、def2-TZVP、L8、grid_response=True、SCF 1e-12/1e-9。
- 阶段A只允许零求值；核验失败不启动SCF。

## 执行状态
1. **A（零求值）十项全过**：起点=074 plus最后完成求值opt_15（账本定位，非未求值坐标）；溯源/负模一致/镜像闭合（反射RMSD 7.06×10⁻⁹）。
2. **mock 6/6过**（反射对合与梯度变换、reuse、镜像门、资格门控、隔离）。
3. 启动期路径拼接bug一次（零SCF消耗时修复重跑，非计算异常重试）。
4. **B：2/30次新求值即达门限**→复核PASS（dE=5.68×10⁻¹⁴、gmax=6.13×10⁻⁶）→stable_i=True（⟨S²⟩=0.754285）；E=−131.1047349818；账目5/34。
5. **C：镜像核对FAIL（1/1）**：dC=6.66×10⁻²Å/dE=1.04×10⁻⁷/dg=7.69×10⁻⁵均超门——参照系过期（plus多走2步、minus停在074的15步层）；按章程不重试、不转移。
6. **D未触发**（镜像门控）：频率未执行；**E被阻塞**：P1不拼接。总账目6 attempt、0失败。

## 交付
`results/phase2_preparation/h2no_follow075_report.md`、`jobs/JOB-2026-0906-075_h2no_representative_continuation_batch.md`、`h2no_follow075/`（audit075_results.json、manifest、双账本、eval记录、accepted_iterates、稳定性记录、mirror075_results.json、freq_eligibility075.json、p1_reference075.json、mock测试、日志）、notes/job075_h2no_representative_continuation_2026-09-12.md、074补注、PROJECT_STATUS同步。

**完成即停止。极小值未确认（频率被镜像门控阻止，如实登记）；未启动TS/IRC/水相/UV/MNB界面/病毒模型；未修改DFT平台、依赖或文献原件。**
