> **[本批性质]** 本报告是 JOB-2026-0906-058 的 resume-01 交付。058 原批：中心 SCF＋完整梯度 1 次完成并保存；内部稳定性调用已发生且返回（实际契约为四元组），被错误解包为 3 项而中断，结果未保存；42 个位移 0 次运行。原异常现场（账本 error、abort JSON、中心记录、脚本、日志）逐哈希保留于 `c1_fdhess058/`，**未被本批触碰**（交付后再次逐哈希核对：无变化）。恢复授权：`notes/job058_resume_authorization_2026-09-10.md`。

> **[JOB-059 补正（2026-09-10，依据 `notes/job058_commander_review_2026-09-10.md`）]** ①本报告及 `resume01_results.json` 所写"累计46 attempt／44/44、2/2、46/46全部合规"的总评**撤回**：加上已登记且被终止的centre探针（`budget_resume01_launchprobe_archived.json`，pending，终止于中心SCF中，partial checkpoint存在）后**累计至少47 attempt**；探针成本不确定，且不能以"无日志"证明nohup探针零额外成本；已完成44次SCF＋梯度、2次稳定性调用两项计数保留。②scipy交叉核对3.8858×10⁻¹⁶是**同一内部矩阵本征值的第二求解器复算**（仅后处理验证），不是cm⁻¹频率误差，也不是独立电子结构验证。③两候选负模均由H原子主导**不能**证明是同一软坐标或两几何属同一势盆。④`centre_scf.chk`在正式运行中被后续fd求值覆盖（运行时wrapper对全部43次求值用了同一路径），`resume01_results.json`记录的chkfile_sha256_after_run属于最后一个fd点；权威中心波函数记录是`centre_mo_*.npy`（求值后立即保存）及运行中稳定性记录；中心门（dE=5.68×10⁻¹³）不受影响，无科学结论失效。机器可读补正：`run_artifacts/02_nh3o3_reference/c1_fdhess058_resume01/job058_count_correction.json`。原始记录不清零、不删除、不覆盖。

# JOB-2026-0906-058 resume-01 报告：稳定性接口修复 + 完整梯度差分 Hessian 续算

**日期**：2026-09-10 · **新增预算**（本次明确追加授权，非复用/重置原失败额度）：中心重建 SCF＋完整梯度 **1/1** + 内部稳定性 **1/1**（另列）+ 原定 42 个位移 **42/42** = 新增 43 次 SCF＋梯度，**0 次新增失败**。
**累计核算**：SCF＋梯度 **44/44**（原中心 1 + 本批 43）；稳定性调用 **2/2**（含原失败 1 次）；跨类 attempt **46/46**（原批 2 + 本批 44）。fd 总额度 42，失败计入，分项不挪用。
**登记结论**：057 候选几何 RKS 内部轨道稳定性 **stable_i=True**；**该单步长（h=0.001 Bohr）完整梯度差分 Hessian 在 057 候选几何预测 1 个内部负模 −47.27 cm⁻¹**——只登记"预测负内部曲率"，不称过渡态、不跟随负模、不登记极小值。

## 1. 本批在做什么、已完成什么、还需要什么
- **在做什么**：修复稳定性返回契约（实际为 `(mo_i, mo_e, stable_i, stable_e)`，原脚本 `job058_exec.py:288` 误拆 3 项），以 052 已验证约定保证"返回值与日志在后处理前落盘"；重建中心并通过双门；执行内部稳定性；通过后完成 058 预登记的 42 点中心差分与 15 维内部模式分析。
- **已完成**：三脚本（prep／test_mock／exec）零求值阶段全过（模拟测试 10 项，见 §2）；中心双门 PASS；stable_i=True；42/42 位移全部收敛；H_raw/反对称残差先存、H_sym 后存；TR 投影秩 6／内部 15 模式＋scipy 交叉核对。
- **还需要**：两个候选几何（051／057）负模是否同一软坐标、两候选是否同一势盆，由总指挥安排；单步长敏感性未验证（限制保留）。

## 2. 接口修复与零求值验证（不调真后端、不污染正式目录）
- **修复要点**（`protocols/job058_resume01_exec.py::stability_fn_fixed`）：①日志**直接写文件句柄**（不再用 StringIO），stdout/verbose 恢复与文件关闭都在 `finally`——解包或后处理异常不再丢失日志；②返回后先把状态 JSON（含 `installed_return_order`、逐项 `repr`、`stable_i_raw`）原子落盘，再把 mo 数组存 `.npy`，最后才汇总；③契约检查：非 4 元组→先存原始证据再抛错。
- **模拟测试 10 项全过**（`mock_test_results_resume01.json`）：注入假 mf 返回四项，验证 `internal=True / external=False / return_status=True` 实际传入；stable_i=True/False/None 三分支（False/None→停止且 42 点调用数=0）；契约错误（3 项）及后处理异常（mo_i 不可转数组）时日志与已有记录仍保存；中心门失败→稳定性与 fd 调用数均为 0；42 点全流程＋已知对称矩阵复原（maxdiff 9.2×10⁻¹³）；43rd fd 请求拒收；累计 44/2/46 核算断言。

## 3. 中心重建与双门（对 058 原中心 + 057 指定复核）
| 参照 | dE (Eh) | dgrad (Eh/Bohr) | dC (Å) | gmax (Eh/Bohr) | 结论 |
|---|---|---|---|---|---|
| 058 原中心 `eval_centre.json` | 5.68×10⁻¹³ | 5.88×10⁻¹² | 0.0 | 6.847×10⁻⁶ | PASS |
| 057 指定复核 `eval_recheck_recheck.json` | 0.0（逐位同） | 1.86×10⁻¹³ | 0.0 | — | PASS |

- 门限：|ΔE|≤1e-8、dgrad≤1e-7、dC≤1e-9 Å、gmax≤1e-5、SCF 收敛、结果有限、配置一致（wb97xd+D2、grid_level 8、grid_response=True、def2-TZVP）——全部满足。
- **波函数 checkpoint 已留存**（原 058 无）：`centre_scf.chk`（SCF kernel 写入，sha256 `a54c0ebfe72ecdcb…`）+ `centre_mo_coeff/energy/occ.npy`（142×142），稳定性对象可追溯、可免重跑 SCF 复算。

## 4. 内部稳定性（1/1，50.3 s）
- 调用：`mf.stability(internal=True, external=False, return_status=True)`；原始四元组 `(mo_i 数组, None, True, None)` 已逐项留痕（`stability_raw_result.json`），**stable_i=True**——"wavefunction is stable in the internal stability analysis"，内部轨道 Hessian 最低三特征值 +0.393/+0.427/+0.481。
- 限定：RKS 内部轨道稳定，不排除多参考特征；stable_e=None 仅表示未查外部稳定性。仅因 stable_i 明确 True 才进入 42 点（False/None/异常均立即停止——该分支已在模拟测试 T2/T3/T4/T5 验证）。

## 5. 42 点差分、Hessian 与 15 内部模式
- 42/42 位移（21 坐标 ±0.001 Bohr，058 预登记顺序与坐标逐位复用，未重新选步长）全部 SCF 收敛、结果有限；逐点 attempt 先登记后完成。位移点 gmax 介于 5.9×10⁻⁵–6.6×10⁻⁴ Eh/Bohr（软方向位移点仍近驻点，如实记录）。
- 完整梯度已各含 D2 一次与 grid_response，未额外加 D2 矩阵。**H_raw 先存**（`h_fd_raw.npy`，反对称残差 4.42×10⁻⁷ Eh/Bohr²），**H_sym 后存**（`h_fd_sym.npy`）；H_sym 残差为零不作为原始一致性证据。
- 后处理（044 修正约定，名义质量数 [14,1,1,1,16,16,16] 明确标注）：TR 投影 SVD 秩 6／内部 15 维；**15 个模式全部保留**（cm⁻¹）：−47.27、67.06、75.48、121.07、166.18、188.06、787.16、1031.84、1335.07、1341.28、1676.23、1680.82、3533.94、3660.96、3662.01；scipy 路线交叉核对最大差 3.9×10⁻¹⁶（仅后处理验证）。
- **负模成分**：mode 0，λ=−8.45×10⁻⁵ Eh/Bohr²/amu，3 个 H 原子主导（|v|≈0.56/0.56/0.57；N 0.02、O≤0.04），外泛 overlap≈1.5×10⁻¹⁶（纯内部模）——与 051 候选负模（−72.83 cm⁻¹，亦 H 主导）定性同型；量级不可跨矩阵直接比较（方法与矩阵构成不同）。
- **限制（保留）**：单步长，敏感性未验证；名义质量数（非同位素质量）；负模只登记为"该批 FD 矩阵预测负内部曲率"；缺失位移不插值（本批无缺失）。

## 6. 流程异常（单列）与预算合规
- **启动探针 1 次，单列**：第一次 nohup 随 wsl.exe 会话回收（日志文件未建）；第二次 30 s 前台验证被操作者超时终止于中心 SCF 中。无完成求值、无收敛记录，成本不确定（≤30 s 单 kernel），按"不确定历史成本单列"处理：`launch_probe_note.json` + 探针账本/半成品 chk 归档（`budget_resume01_launchprobe_archived.json`、`launchprobe_centre_scf_partial.chk`），**不计入 46 attempt 授权账本**。正式执行由持久后台任务一次跑完。
- 正式账本 `budget_resume01.json`：44 attempts（centre 1 + stability 1 + fd 42）全部 done；原账本 2 attempts（centre done + stability error）原样保留；累计 46/46、44/44、2/2，无挪用、无清零。

## 7. 停止与交付
**完成即停止。** 交付：本报告、[c1_fdhess058_resume01/](run_artifacts/02_nh3o3_reference/c1_fdhess058_resume01/)（`resume01_results.json`、账本、42 份 `eval_fd_*.json`、`eval_centre.json`、`stability_raw_result.json`+日志+mo 数组、checkpoint、`h_fd_raw.npy`/`h_fd_sym.npy`/`h_fd_modes_eigenvalues.npy`、恢复凭证与原现场哈希、mock 测试、exec_log）、脚本 `job058_resume01_prep.py`/`job058_resume01_test_mock.py`/`job058_resume01_exec.py`、PROJECT_STATUS.md 与长期记忆同步。**当前两个候选几何均预测负内部曲率，均不登记极小值；后续核查由总指挥安排。**
