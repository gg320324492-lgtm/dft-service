# JOB-025 溯源、预算与稳定性补正记录（2026-09-08）

## 1. 最终结果溯源（逐项核对）

### NH₃ v5 最终记录
| 项 | 内容 | 证据 |
|---|---|---|
| 实际执行脚本 | `protocols/nh3_opt_continue2.py` **v5 版**（无 kernel 补丁——已核实现文件不含 `.kernel =` 补丁；含回调记账与收紧收敛参数 gradientmax=3e-5） | 文件哈希见 `final_endpoint_index.json.script_provenance` |
| 脚本保存状态 | v5 执行版**即现存盘版**（最后一次编辑先于 v5 运行）；**v3/v4 执行版曾在原文件上就地编辑、未保留独立原件**——登记为"事后状态"，不冒充当时原件（缺项如实列出） | 文件 mtime 与运行时序 |
| 输入与实际计算坐标 | 续跑起点 = v4 最终坐标（脚本构造式衔接）；v4/v5 **预运行输入坐标未独立落盘**（缺口） | `nh3_v5_result.json` |
| 方法配置 | wb97xd+项目−D2/def2-TZVP/L8/grid_response/SCF 1e-12,1e-9（气相）——从实际对象读回（024 记录） | 同上 |
| 优化轨迹 | `nh3_v3_steps.json`（4 步：4.634e-2→1.855e-2→1.624e-3→6.484e-5）、`nh3_v5_steps.json`（3 步：6.484e-5→3.236e-5→7.661e-7） | 文件 |
| 独立复核 | `nh3_v5_result.json :: v3_final`：E=−56.564219135911 Eh、max|g|=7.661×10⁻⁷、SCF 收敛 | 文件 + 端点索引 |
| **链式核验结果** | v3_final → v4 输入、v4_final → v5 输入：**脚本构造式衔接成立，但 v4/v5 预运行输入未独立落盘（缺口登记）**；v3_final 与 v4_final 坐标不同为正常（v4 优化前进了） | `final_endpoint_index.json.chain_verification` |

### O₃ 最终记录
- 来源：`gas_monomer_reference.py` v1（**该路径梯度冻结缺陷对 O₃ 无影响**——O₃ 从历史基线几何起步，第一步即收敛）；`monomer_results.json :: o3.endpoint_recheck`。
- 复核坐标与输入坐标差 **2.2×10⁻¹⁶ Å**（无移动）；E=−225.4339026996 Eh、max|g|=3.514×10⁻⁶。

### NH₃ v5 是否确实使用移除错误绑定的路径？
**是**——`nh3_opt_continue2.py` 现存版不含任何 `.kernel =` 补丁（自动检索确认），且其梯度行为（随几何变化、收敛至 7.7×10⁻⁷）与无补丁路径一致；v1 的冻结梯度现象（恒 4.357×10⁻²）在 v5 中不复现。

### 最终端点索引
`run_artifacts/02_nh3o3_reference/final_endpoint_index.json`：
- **默认端点**：`NH3_gas_v5_final`（−56.564219136 Eh / 7.661×10⁻⁷）、`O3_gas_final`（−225.433902700 / 3.514×10⁻⁶）——均为"驻点候选"（无 Hessian/频率，非已验收极小值）；
- **保留但默认不读**：`NH3_gas_v1_frozen_final`（冻结梯度缺陷端点）、`NH3_gas_v3_endpoint_6.5e-5`（内坐标判据满足但笛卡尔 6.5×10⁻⁵>1e-5）等，`default_read=false`；
- 结果文件与脚本 SHA-256（前 16 位）已登记。

## 2. 预算与根因表述补正

### 39 次求值逐项（全部有证据，无"无法确定"项）
| 项 | 次数 | 证据 |
|---|---|---|
| NH₃ v1 优化（梯度冻结，浪费） | 15 | `gas_monomer_run.log`（run3 段）+ nh3_steps.json |
| NH₃ v1 端点复核 | 1 | 同上 |
| O₃ 优化 + 复核 | 2 | 同上 |
| 诊断（重复 SP + 双向位移） | 3 | `diagnostic_consistency.json`（未获原批令预先安排——为定位梯度冻结所必需，事后补登记） |
| v2 续跑 | 7 | `nh3_continue_run.log` + nh3_continue_steps.json（未获原批令预先安排） |
| v3 续跑 | 5 | `nh3_v3_run.log` + nh3_v3_steps.json（未获预先安排） |
| v4 续跑 | 2 | `nh3_v4_run.log`（未获预先安排） |
| v5 续跑 + 独立复核 | 4 | `nh3_v5_run.log` + nh3_v5_steps.json（未获预先安排） |
| **合计** | **39** | —— |

- **表述补正**：删除"超额被 42 总预算吸收"类措辞——**总上限未突破不使分项超限合规**；NH₃ 优化超额（>20）、3 次复核（原令 ≤1）、诊断与 4 次续跑分别登记为**未获批/超分配项**。
- **账本粒度缺陷**：019 机制的 attempt 在 024 执行中挂于 scanner 创建级（仅 10 条记录 vs 39 次实际求值）——v3–v5 及 v2 驱动已改逐求值记录；035 条历史 attempt 与 39 的差额如实登记。

### 根因表述更正
- 早期判断"pyberny 信任半径崩溃/过早停止是根因"（024 报告 v1 段）→ **已被取代**：信任半径崩溃是**驱动缺陷（梯度冻结于初始几何）的下游结果**（诊断证明 E/g 路径自洽，真实下降方向存在）。
- 与**优化器自身收敛阈值问题**区分：v4 中 pyberny 内坐标判据（4.5×10⁻⁴）已满足而未投影笛卡尔 1e-5 未达——这是**阈值口径差异**，非优化器缺陷；v5 通过收紧收敛参数解决。
- 驱动缺陷与阈值问题**两类问题分开**，不再混写。

## 3. 稳定性记录补正

- **源码核实**（pyscf/scf/stability.py，安装版 2.14.0）：`rhf_stability(mf, internal=True, external=False, return_status=False)` 默认返回 **(mo_i, mo_e)**；**stable_i/stable_e 布尔值仅在 return_status=True 时返回**。
- 我方 024 调用 `mf.stability()` 未带 return_status → 保存的 `st[1]` = mo_e = **None**（未请求外部检查的产物）。
- **补正**：024 报告/JSON 中"None（未发现内不稳定）"→ **"此次保存的返回项不足以判断内部稳定性"**（已逐处标注【025 补正】）；后续稳定性分析必须 `mf.stability(return_status=True)` 并保存 stable_i/stable_e。
- 本批**不重跑**稳定性分析（按批令）；<S²>=0 与内稳定**不等同于电子结构描述充分**。

## 4. 驱动修复（详见 `gas_monomer_reference_v2.py` 头注与回归测试）

- v1 入口**禁用**（存根指向 v2）；执行过的历史副本完整保留于 `protocols/deprecated_history/gas_monomer_reference_v1_executed.py`；
- **唯一新入口** `gas_monomer_reference_v2.py`：`BudgetedScanner`（继承 `pyscf.lib.GradScanner` 的代理）包裹真实扫描器——**每次求值前**检查并持久化预算（总/每物种优化/复核三档独立），**无任何 kernel 补丁**（扫描器在传入的当前几何上求值）；
- **无 SCF 回归 5/5 通过**（`protocols/test_gas_driver_v2.py`，覆盖真实 berny+GradScanner 适配路径）：T1 连续几何正确变化且 E/g 同几何；T2 求值后记录异常 attempt 不丢；T3 重启预算保留；T4 分项耗尽即使总预算有余额仍在求值前拒绝；T5 缺失账本显式报错不静默新开。
