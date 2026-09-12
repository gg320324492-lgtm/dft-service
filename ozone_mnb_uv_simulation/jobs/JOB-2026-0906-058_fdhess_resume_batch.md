> **[JOB-2026-0906-058 resume-01 完成]** 本批含 058 原批中断与 resume-01 续算：原批中心 1 次完成保存、稳定性调用返回后四元组被误拆 3 项中断（结果未保存）、42 位移未运行；resume-01 修复接口后全部完成。**057 候选 stable_i=True；单步长 FD Hessian 预测 1 内部负模 −47.27 cm⁻¹——两候选均预测负内部曲率，均不登记极小值。** 详见 `results/phase2_preparation/c1_fdhess058_resume01_report.md`。

# JOB-2026-0906-058：057 候选完整梯度差分 Hessian 批（含 resume-01）

## 范围与授权
- 编号 = 057 后下一个未占用（prep 已核对）。原批预算：中心 1 + 稳定性 1（另列）+ fd 42 = 43 次 SCF＋梯度。
- **resume-01 授权**（`notes/job058_resume_authorization_2026-09-10.md`，本次明确追加、非复用/重置）：中心重建 SCF＋完整梯度 ≤1、内部稳定性 ≤1、原定 42 位移 ≤42；新增 SCF＋梯度上限 43；与原批累计 SCF＋梯度 ≤44、稳定性 ≤2（含原失败）、跨类 attempt ≤46；fd 总额度 42 不变，原 error 不清零。

## 执行状态
1. **原批（中断）**：中心 SCF＋完整梯度 1 次完成并保存（E=−282.0017219396601、gmax=6.847e-6，双门 PASS）；`mf.stability()` 返回后按 3 项解包失败（实际契约四元组 `(mo_i, mo_e, stable_i, stable_e)`），StringIO 日志随异常丢失、无可验证 checkpoint；fd 0 次。原现场（账本 error/abort JSON/中心记录/脚本/日志）逐哈希保留于 `c1_fdhess058/`。
2. **resume-01 零求值阶段**：prep 保存原现场哈希＋恢复凭证＋恢复 manifest（原 42 位移几何只读复用，不重选步长）；修复 `stability_fn_fixed`（052 约定：日志直接写文件句柄、finally 恢复、状态 JSON＋mo 数组先于汇总落盘、契约检查）；模拟测试 10 项全过（假 mf 四项、kwargs 实传验证、True/False/None 分支、契约错误与后处理异常时日志仍存、门未过 0 调用、43rd fd 拒收、累计 44/2/46 断言、正式目录未触碰）。
3. **流程异常 1 次（单列）**：nohup 随 wsl.exe 会话回收（日志未建）＋30 s 前台验证被超时终止于中心 SCF（无完成求值，成本不确定）→ `launch_probe_note.json`＋探针账本/半成品 chk 归档，不计入授权账本；改用持久后台任务一次跑完。
4. **中心重建双门 PASS**：对 058 原中心 dE=5.68×10⁻¹³、dgrad=5.88×10⁻¹²、dC=0、gmax=6.847×10⁻⁶；对 057 指定复核 dE=0、dgrad=1.86×10⁻¹³。波函数 checkpoint 留存（`centre_scf.chk`＋MO npy，142×142）。
5. **内部稳定性 1/1（50.3 s）**：四元组 `(mo_i, None, True, None)`，**stable_i=True**（最低内部轨道 Hessian 特征值 +0.393/+0.427/+0.481）；原始 repr/日志/mo 数组已落盘；仅因明确 True 才进入 42 点。
6. **42/42 位移全部收敛、0 失败**（h=0.001 Bohr、原预登记顺序；位移点 gmax 5.9×10⁻⁵–6.6×10⁻⁴ 如实记录）；完整梯度各含 D2 一次与 grid_response，未额外加 D2。**H_raw 先存**（反对称残差 4.42×10⁻⁷ Eh/Bohr²）**后存 H_sym**。
7. **15 内部模式**（044 修正后处理、名义质量数 [14,1,1,1,16,16,16] 标注）：TR 秩 6／内部 15；频率 −47.27、67.06、75.48、121.07、166.18、188.06、787.16、1031.84、1335.07、1341.28、1676.23、1680.82、3533.94、3660.96、3662.01 cm⁻¹；scipy 交叉核对 3.9×10⁻¹⁶。**负模 −47.27 cm⁻¹**（λ=−8.45×10⁻⁵），3 个 H 主导（|v|≈0.56），外泛 overlap≈1.5×10⁻¹⁶。单步长敏感性未验证（限制保留）；不称 TS、不跟随负模。

## 账目
| 分项 | 新批上限 | 新批实际 | 失败 | 累计（含原批） |
|---|---|---|---|---|
| 中心 SCF＋完整梯度 | 1 | 1 | 0 | 2/44（原 1 done + 新 1 done） |
| 内部稳定性（另列） | 1 | 1 | 0 | 2/2（原 1 error + 新 1 done） |
| fd 位移 | 42 | 42 | 0 | 42/42 |
| **跨类 attempt** | — | 44 | 0 | **46/46** |

另有启动探针 1 次单列（无完成求值）；原批 stability error 1 条保留不清零。

## 交付
`results/phase2_preparation/c1_fdhess058_resume01_report.md`、`c1_fdhess058_resume01/`（`resume01_results.json`、账本、42 份 `eval_fd_*.json`、`eval_centre.json`、稳定性原始结果/日志/mo 数组、checkpoint、`h_fd_raw.npy`/`h_fd_sym.npy`/`h_fd_modes_eigenvalues.npy`、恢复凭证与原现场哈希、探针归档、mock 测试、exec_log）、脚本 `job058_resume01_prep.py`/`job058_resume01_test_mock.py`/`job058_resume01_exec.py`、本记录、PROJECT_STATUS.md 与长期记忆同步。

**完成即停止。当前无已验收极小值；两候选负模关系待总指挥安排。**
