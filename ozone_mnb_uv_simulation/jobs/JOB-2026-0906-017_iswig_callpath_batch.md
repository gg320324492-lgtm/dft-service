# JOB-2026-0906-017：ISWIG 实际调用路径核实与固定方向对照批

## 范围与授权
- 任务编号 = 016 之后下一个未占用（已核对）；前置核实无 SCF；求值上限 5（含失败尝试）。
- 若 ISWIG 能量+梯度路径不能确认一致支持 → 停止求值、提交缺口、不修补安装库。

## 执行状态
| 项 | 结果 |
|---|---|
| A. 调用路径核实（静态 + 运行时跟踪，无 SCF） | **完成 → 前置检查 FAIL** |
| JOB-016 Hessian 表述补正 | 完成（静电解析 / CDS 二阶导=有限差分 / 属性接线不一致） |
| B. 5 次求值 + 对照分析 | **未执行**（前置不成立，按批令停止） |
| C. 报告与建议 | 完成 |

## 关键发现（详细：notes/iswig_callpath_verification_2026-09-07.md）
1. **ISWIG 不可达**：`SMD.build()`（smd.py:430）调 `gen_surface(mol, rad=..., ng=...)` **不传方法**；运行时实证——属性设 ISWIG 后 gen_surface 实际收到 "SWIG"，产物含 SWIG 专属键 `R_in_J/R_sw_J`。
2. **属性无读取点**：全 solvent 包仅 `:385`（_keys）与 `:403`（默认值）出现，无消费。
3. **梯度恒 SWIG**：`get_dF_dA(surface, surface_discretization_method="SWIG")`；调用者 `grad/smd.py:55` 只传 surface。ISWIG 梯度数学分支存在且直调可运行，但 SMD 层无接线。
4. **三处接线不一致**：Hessian 侧读取对象属性 → 设属性将致能量/梯度（SWIG）与 Hessian（ISWIG）互不一致（KeyError）。
5. **CDS 二阶导 = 有限差分**（`hessian/smd.py get_cds` 源码显式警告）→ JOB-016"解析 Hessian 均实现"表述已补正。

## 处置
- **新增求值数 = 0**（预算 5 未动用）；无原始求值记录/对照表（无数据）。
- 缺口属 **PySCF 平台侧**；独立最小案例（运行时跟踪）已具备并可复现（`protocols/iswig_callpath_trace.py`）；**本批不修复平台**，修复须另行立项走既定流程。
- 013/015 SWIG 三档结果维持封存；正式驻点验收**仍未通过**（2.503×10⁻⁵ > 1e-5）。

## 唯一后续建议（暂不执行）
采用**明确标注的受限电子能流程（路线 B）**：在气相已验收几何（c06 L7 / c14 L8，来源已核验）上做水相 SMD 单点，以"气相几何+水相单点"标签仅用于受限电子能分析；决策价值 = 当前唯一不需平台修复、不需新验收假设即可补充第一阶段水相方法学信息的途径。执行前须总指挥批准。

## 交付
`notes/iswig_callpath_verification_2026-09-07.md`、`results/phase1_closeout/iswig_callpath_batch_report.md`、`run_artifacts/.../iswig_callpath/runtime_trace.json`、`protocols/iswig_callpath_trace.py`、016 评估补正、任务记录（本文件）。

**本批停止。** 不追加档位、不算 Hessian、不修复平台、不自行选定污染物或病毒。
