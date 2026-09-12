# JOB-2026-0906-074：H₂NO·负模方向跟随与真实极小值搜索批

## 范围与授权（用户2026-09-11批令）
- 编号=074。对073发现的H₂NO·−193.61 cm⁻¹负模做独立方向核验+双侧有限优化（0.10 Bohr固定步长），判断能否找到真实极小值。
- 预算（上限39 SCF+完整梯度）：主账本center 1+disp 2；每分支opt 15+recheck 1+stab 1+anchor 1（频率Hessian与D2 FD另行统计）。门限1e-5不放宽；stable_i须return_status=True；负模不自动称TS（TS仅IRC后）；两侧都保留、不强行合并；异常只停对应侧。
- 新文件仅入本项目对应目录（脚本在h2no_follow074/scripts/）；平台/依赖/文献原件只读；073原始记录不覆盖。

## 执行状态
1. **A（零求值）**：模式向量仅从073保存频率记录读取，六项核验全过（维度/单位范数/TR纯度/本征逐位/频率复现/符号规则）；073补正三件套（notes文件+横幅+audit机器记录）。
2. **mock 7/7过**（位移构造对称性、runner reuse零重复、分支隔离、Kabsch RMSD含反射对照、资格门控）。
3. **B**：中心PASS（5.68e-14）；两侧位移能量简并下降（ΔE=−7.1923e-6，宇称），负曲率获独立求值支持。
4. **C**：plus/minus各15/15求值**均未达1e-5**（最低gmax=8.305e-5，E=−131.10473497），如实停止、不外推；0失败0异常。33/39。
5. **去重（零求值）**：±位移点与优化末点均互为精确镜像（反射RMSD≈0/7e-9；正规旋转0.030/0.071 Å）——按065先例登记为独立镜像分支，永不合并；负模为面外/锥形化型软模，方向极平坦。
6. **D/E未触发**：无分支过门限→复核0/2、stable_i 0/2、条件频率未执行、P1片段能不拼接（登记原因）。

## 交付
`results/phase2_preparation/h2no_follow074_report.md`、`jobs/JOB-2026-0906-074_h2no_negative_mode_follow_batch.md`、`h2no_follow074/`（audit074_results.json、input_manifest074+分支manifest、三本账本、全部eval记录、accepted_iterates、dedup074_results.json、follow074_results.json、freq074_summary.json、p1_reference074.json、mock测试、日志）、notes/job074_h2no_negative_mode_follow_2026-09-11.md、073横幅、PROJECT_STATUS同步。

**完成即停止。未找到合格极小值（如实登记）；未启动TS/IRC/水相/UV/MNB界面/病毒模型；未修改DFT平台、依赖或文献原件。**
