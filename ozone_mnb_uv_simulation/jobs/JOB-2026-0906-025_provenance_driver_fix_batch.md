# JOB-2026-0906-025：JOB-024 最终结果来源与驱动修复核验批

## 范围与授权
- 任务编号 = 024 之后下一个未占用（已核对）；**新增量化求值数 = 0**；不运行优化、频率或稳定性分析。

## 执行状态（全部完成）
| 项 | 交付 |
|---|---|
| 1. 最终结果溯源 + 端点索引 | `run_artifacts/02_nh3o3_reference/final_endpoint_index.json` + `notes/job025_provenance_budget_stability_2026-09-08.md`：NH₃ v5（E=−56.564219136/7.661e-7）确认使用**无补丁路径**（现存脚本无 `.kernel =` 补丁；v3/v4 执行版未保留独立原件——"事后状态"登记）；O₃ 复核坐标与输入差 2.2e-16；脚本与结果文件哈希登记；**最终端点索引**建立（默认端点 NH3_gas_v5_final / O3_gas_final；失败/被取代端点保留且 default_read=false） |
| 2. 驱动修复 | `gas_monomer_reference.py` v1 **禁用**（存根指向 v2；执行版历史副本完整保留于 `protocols/deprecated_history/gas_monomer_reference_v1_executed.py`）；**唯一新入口 `gas_monomer_reference_v2.py`**：BudgetedScanner 代理（继承 lib.GradScanner，无 kernel 补丁，扫描器在传入的当前几何求值）+ BudgetController（**求值前**检查并持久化；总/每物种优化/复核三档独立；重启不重置；缺失账本显式报错） |
| 3. 无 SCF 回归 | `protocols/test_gas_driver_v2.py` **5/5 通过**（覆盖真实 berny+GradScanner 适配路径）：T1 连续几何正确变化且 E/g 同几何（3 个不同几何）；T2 求值后记录异常 attempt 不丢；T3 重启预算保留；T4 分项耗尽、总预算有余额仍求值前拒绝；T5 缺失账本显式报错 |
| 4. 稳定性补正 | 源码核实：`mf.stability()` 默认 `return_status=False` 返回 (mo_i, mo_e)；**stable_i/stable_e 仅在 return_status=True 时返回**；我方保存的 st[1]=None = mo_e=None → **"此次保存的返回项不足以判断内部稳定性"**（024 报告/JSON 已逐处标【025 补正】）；后续须 `return_status=True`；不重跑 |
| 5. 预算与根因补正 | 39 次逐项有证据（无"无法确定"项）；**删除"超额被总预算吸收"表述**——NH₃ 优化超额、3 次复核、3 次诊断、4 次续跑分别登记为未获批/超分配项；**"pyberny 过早停止是根因"标注被取代**（信任半径崩溃 = 驱动缺陷的下游结果；v4 的 6.5e-5 为阈值口径差异非缺陷） |

## 频率批执行条件判断

**已具备**：①两个默认端点（NH₃ −56.564219136 / O₃ −225.433902700，均驻点候选）已固定并可索引读取；②修复后的唯一驱动入口 + 分项预算 + 回归测试就绪；③Hessian/频率在 PySCF 现行路径有已验证实现（013 批诊断性 Hessian 流程 + O₃/H₂O 历史频率经验，注意 D2 全导数 Hessian 已在项目路径中）。
**待定项（不阻塞，但须在频率任务书中明确）**：稳定性结论补齐方式（`return_status=True` 重跑一次稳定性，成本极低，可并入频率批首步）；频率批预算须按"Hessian = 1 次解析 Hessian（非梯度求值）+ 每单体 1 次端点确认 SP"展开，并沿用分项账本。

## 边界
本批新增量化求值数 = **0**；历史原始结果、失败记录与代码证据全部保留；未修改桌面文献原件、DFT 平台或 PySCF 安装环境。

**本批停止。** 频率批待总指挥批准后下达。
