# JOB-041 驱动拒绝审计（JOB-2026-0906-042 零求值溯源，2026-09-08）

## 041 全部 21 attempts 逐项分类

| ledger idx | 类别 | 状态 | 起点 key | SCF 启动 | SCF 完成 | 梯度完成 | 拒绝原因 |
|---|---|---|---|---|---|---|---|
| [0] | repro | rejected | a960f4a4… | **否** | 否 | 否 | **驱动缺陷 BEFORE SCF**：`KeyError: 'grad_rms'`——load_start_038 的 ref 标量缺 grad_rms 字段，门计算崩溃于 SCF 启动前；零计算发生 |
| [1] | repro | rejected | a960f4a4… | **是** | **是**（SCF 正常完成） | 是 | **驱动缺陷 AFTER SCF**：`PathGuardError: refusing to write into the formal directory`——run_start_acceptance 的 save_json_atomic 未传 allow_formal_writes=True，PathGuard 误拦正式路径写入；SCF 已完成但结果未持久化（不可恢复） |
| [2] | repro | **done** | a960f4a4… | 是 | 是 | 是 | **成功**（第三次运行）：dE=2.84×10⁻¹³、dgrad_max=2.04×10⁻¹⁴、dgrad_rms=2.38×10⁻¹⁴——全门限通过 |
| [3]–[20] | opt | done（18/18） | 各种 | 是 | 是 | 是 | 正常优化求值（无异常） |

## 拒绝原因详细分类

### rejected [0]：**SCF 启动前驱动缺陷**
- **异常类型**：`KeyError: 'grad_rms'`
- **发生阶段**：起点验收门计算阶段（SCF 启动前）
- **是否启动 SCF**：**否**
- **是否写入结果**：**否**（零计算）
- **根因**：`load_start_038` 的 ref 标量仅含 E 和 grad_max，缺 grad_rms 字段——门计算在构建 gates 字典时访问 `manifest['ref_scalars']['grad_rms']` 触发 KeyError
- **修复**：ref 标量加入 `grad_rms=rec['grad_rms']`

### rejected [1]：**SCF 完成后驱动缺陷（结果丢失）**
- **异常类型**：`PathGuardError: refusing to write into the formal directory: .../c1_cart_exec_v3/eval_records.json`
- **发生阶段**：SCF 完成后、结果持久化阶段
- **是否启动 SCF**：**是**
- **是否完成 SCF**：**是**
- **是否写入结果**：**否**（PathGuard 拒绝了写入，结果不可恢复）
- **根因**：`run_start_acceptance` 内部的 `save_json_atomic(records_path, records)` 未传递 `allow_formal_writes=True`——PathGuard 将正式路径误判为越界写入
- **修复**：`allow_formal_writes` 参数传递到 run_start_acceptance 内部的所有 save_json_atomic 调用

## 计数口径修正

| 量 | 041 报告原口径 | 042 修正口径 |
|---|---|---|
| 预算 attempt | 21/21 | **21/21**（不变） |
| "18/20 优化求值" | 18 次真实求值 | **18 次优化 attempt 中 18 次完成 SCF+梯度**（无 SCF 前失败） |
| "2 次驱动缺陷 rejected" | 2 次 rejected | 2 次 rejected：**[0] SCF 前零计算 + [1] SCF 后结果丢失**——两者性质不同 |
| 总完成 SCF+梯度 | — | **19**（起点验收 1 + 优化 18） |

## 041 报告结论保留与限定

- **保留**："最佳梯度 3.494×10⁻⁵ 低于 pyberny 50 次 ~3×10⁻⁴ 一个数量级"——**数值成立**（18 次完成 SCF+梯度的记录充分支持）；
- **保留**："ΔE 稳定 −2.258 kcal/mol；接触 3.13 Å 取向 ~29°"——**数据成立**；
- **限定**："039/040 问题未复现"——**限定为**"起点来源/单位/记录隔离问题未复现"；2 次驱动拒绝属于**新的驱动缺陷类型**（PathGuard 参数传递 + ref 标量缺失），非 039/040 问题的重现；
- **限定**："登记未收敛（梯度未达标）"——**保留**（3.494×10⁻⁵ > 1×10⁻⁵ 阈值，复核未触发）；不因此宣布数值地板。
