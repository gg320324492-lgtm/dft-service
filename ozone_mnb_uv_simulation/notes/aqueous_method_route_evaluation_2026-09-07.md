# 水相方法路线只读评估（JOB-2026-0906-016 · 2026-09-07）

**性质**：只读评估（零求值）。依据 = 当前安装版本源码逐行核实（PySCF **2.14.0**，`~/dftvenv`，路径 `/root/dftvenv/lib/python3.12/site-packages/pyscf/solvent/`）+ 项目内已有证据。**不凭参数名称推断功能**；本评估不宣称已证明某离散方式"有固有问题"，也不评估替代流程的精度。

## 1. 源码核实的实际支持范围

| 项 | 源码证据 |
|---|---|
| 表面离散选项 | `pyscf/solvent/pcm.py:131`：`gen_surface(mol, ng=302, rad=modified_Bondi, surface_discretization_method="SWIG")`；`:139/:162` SWIG 分支、`:167` **"ISWIG"** 分支、`:172` 其余值抛 `NotImplementedError` —— **实际支持两种：SWIG（默认）与 ISWIG**，本机所有历史计算均为默认 SWIG（实测属性 `surface_discretization_method='SWIG'`） |
| 每原子表面点数 | 由 `lebedev_order` 控制（41=590、47=770、59=1202 点/球，已实测）；`ng=302` 为 gen_surface 的 PCM 路径默认参数 |
| 能量 | SMD 能量（含 CDS）——项目内已大量使用 |
| **梯度** | **解析**：`pyscf/solvent/smd.py:513 def grad(dm)`（经 `pyscf.solvent.grad.smd`：grad_qv/grad_solver/grad_nuc + CDS） |
| **Hessian（017 批表述补正）** | `hess()` 入口存在（smd.py:532）**≠ 全部贡献均为解析二阶导**：静电部分（analytical_hess_nuc/qv/solver）为解析；**CDS 二阶导为有限差分**（`hessian/smd.py get_cds`，源码显式警告 "Using finite difference scheme for CDS contribution"）；且 Hessian 侧读取对象属性 `surface_discretization_method`，而 build()/梯度侧不读取（接线不一致，见 `notes/iswig_callpath_verification_2026-09-07.md`） |
| 其他溶剂模型（仅记录） | `ddcosmo.py`、`ddpcm.py`、`pcm.py`（C-PCM 族）、`cosmors.py`、`smd_experiment.py`（实验性）——切换模型属"改变溶剂模型/参数化"，按项目规则 SMD **不得降级**，不在路线内 |
| 官方文档 | PySCF 官方文档 solvent 章节（http://www.pyscf.org/latest/solvent.html ）与本机源码一致；以上以安装源码为最终权威 |

**三类设置的区分**：
1. **仅数值离散**：`surface_discretization_method`（SWIG↔ISWIG）、`lebedev_order`——不改溶剂模型与参数化；
2. **改变溶剂模型/参数化**：换 `ddcosmo/pcm/cosmors`、改 `rad` 半径表、改溶剂描述符——超出"数值对照"范畴，受"SMD 不得降级"约束；
3. **几何优化与溶剂单点的组合方式**：水相优化 vs "气相已验收几何 + 水相单点"——属流程方案（路线 B），不是模型变更。

## 2. 两条路线评估

### 路线 A：保持 SMD 模型，做有实现依据的数值离散对照

**【017 批更新：路线 A 在当前安装版本不可执行】**——运行时实证与源码核实（`notes/iswig_callpath_verification_2026-09-07.md`）：`SMD.build()` 不把 `surface_discretization_method` 传给 `gen_surface`（恒 SWIG）；`grad/smd.py` 调 `get_dF_dA` 不传方法名（恒 SWIG 分支）；Hessian 侧却读取该属性 → 设属性会造成能量/梯度（SWIG）与 Hessian（ISWIG）互不一致。ISWIG 仅能通过绕过 SMD 层的底层直调实现，超出标准生产路径，**未经授权不得采用**。路线 A 需平台级修复（独立最小案例已具备），另行审批。
- **内容**：唯一改变 = `surface_discretization_method: SWIG → ISWIG`（源码明确支持的两种之一），固定几何（c06_order47_reopt 端点）、固定方向（013 模 0）、其余配置不变，重复 015 式 5 点核查（1 中心 + 4 位移）。
- **依据**：ISWIG 为安装版本内**文档化、可实现**的替代离散方式（pcm.py 源码分支）；此对照直接回答"符号翻转是 SWIG 离散方式的数值行为，还是水相溶剂曲率在该方向上普遍不稳健"。
- **只读核实的注意事项**：设置须在 surface 构建前完成并**从实际对象读回核验**（`surface_discretization_method=='ISWIG'`、点数变化、lebedev_order 不变）；`ng/rad` 默认行为与 SMD 的耦合在执行批内再逐项核对。
- **价值**：若 ISWIG 曲率与 SWIG 同档位不同号 → 符号翻转对离散**方式**也敏感，"数值敏感性主导"判断增强；若 ISWIG 稳定为负或稳定为正 → 为"该方向曲率的稳健性"提供首个跨方式证据。
- **局限**：固定方向核查只能回答该几何/方向/配置的局部行为（014/015 补正口径），不能单独证明完整驻点或物理极小值。

### 路线 B：明确标注的替代建模流程（备选，范围受限）
- **内容**：在**气相已验收几何**（c06 L7 端点、c14 L8 端点）上做**水相溶剂单点**（SMD 能量+梯度，不优化），用于**范围受限的电子能分析**（如水化引起的电子能变化量级、两候选在同一水相处理下的定性比较）。
- **必须声明**：该流程**不能替代**：①溶液相驻点（几何未经水相梯度松弛）；②水相完整频率/热化学；③任何以"水相驻点"名义引用的结果。结果只能以"气相几何 + 水相单点"标签引用。
- **定位**：不解决软模/验收问题；仅在第一阶段收尾需要"定性水相能量排序"时作为补充证据，避免与路线 A 争夺预算。
- **可复用基线**：`endpoints_fixed.json`（c06）、`c14_l8_check/l8_optimize_result.json`（c14）——几何与哈希已核验。

## 3. 结论

- **值得做对照**：路线 A 的 ISWIG 对照有明确实现依据、最小预算（5 求值）、直接针对当前唯一证据缺口（符号翻转的"方式敏感性"），**推荐为下一优先批**（草案见 `notes/phase1_priority_task_draft_2026-09-07.md`）。
- **不做**：不改平台源码/安装环境；不以放宽梯度阈值为默认出路；路线 B 仅在需要受限电子能分析时按其标签使用。
- **仍然开放**：水相完整驻点验收（阈值变更须经用户批准）；软模归属最终判断。
