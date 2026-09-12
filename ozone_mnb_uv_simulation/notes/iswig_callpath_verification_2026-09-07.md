# ISWIG 实际调用路径核实记录（JOB-2026-0906-017 · 2026-09-07）

**结论先行**：**前置检查未通过（FAIL）——ISWIG 无法在当前安装版本的标准 SMD 调用路径上一致地生效；按批令在求值前停止，实际新增求值数 = 0。**

版本与源码：PySCF **2.14.0**（`/root/dftvenv/lib/python3.12/site-packages/pyscf/solvent/`）。证据 = 静态源码逐行 + **运行时跟踪**（项目内临时包装 `pcm.gen_surface` 记录实际参数与分支产物；安装源码未修改；全程无 SCF）。运行时记录：`run_artifacts/01_pure_water_o3_h2o/iswig_callpath/runtime_trace.json`；跟踪脚本：`protocols/iswig_callpath_trace.py`（固化保留，可复跑）。

## 1. 调用关系与逐项核实

### (a) SMD 构建表面时是否把方法传给 gen_surface？——**否**
- 源码：`smd.py:419 def build(ng=None)` → `:430 self.surface = pcm.gen_surface(mol, rad=radii_table, ng=ng)`——**无 `surface_discretization_method` 实参**。
- 运行时实证：将对象属性设为 `'ISWIG'` 后调用 `build()`，被包装的 `gen_surface` 实际收到 **`surface_discretization_method='SWIG'`**（函数默认值），产物表面含 **SWIG 专属键 `R_in_J`/`R_sw_J`**、无 ISWIG 专属键 `R_J`。
- 属性消费点（**018 批表述补正**：取代"全 solvent 包无属性读取点"的粗表述）：经证据支持的具体调用路径为——**`smd.py:430 build()` 与 `grad/smd.py:55` 均不读取对象属性**（前者不传方法给 gen_surface，后者不传方法给 get_dF_dA）；**`hessian/pcm.py` 多处读取对象属性**（`surface_discretization_method = pcmobj.surface_discretization_method`）；`grad/pcm.py` 内该名字是函数参数（默认 "SWIG"），非对象属性读取。

### (b) ISWIG 分支是否实际执行？——**否**（标准路径下）
运行时产物（SWIG 键存在）证明 SWIG 分支执行；ISWIG 分支仅在**绕过 SMD 层直接调用** `pcm.gen_surface(..., surface_discretization_method='ISWIG')` 时可达（诊断性直调验证：产物含 `R_J`，能构建成功）。

### (c) 表面数据是否被能量计算消费？——是（SWIG 表面）
`build()` 后 `pcm.get_F_A/get_D_S` 消费 `surface` 字典（smd.py:431-433）——即能量计算消费的是 build() 实际构建的表面；由于 build 恒走 SWIG，能量路径实际消费的是 **SWIG 表面**。

### (d) 梯度导数实现与 ISWIG 是否一致？——**不一致（接线断裂）**
- `grad/pcm.py`：`def get_dF_dA(surface, surface_discretization_method="SWIG")`——**函数内部两种分支都有**（SWIG 用 `R_in_J/R_sw_J`；ISWIG 用 `R_J`/`charge_exp` 的 erf 公式；直调 ISWIG 可正常运行）。
- **但唯一调用者** `grad/smd.py:55`：`dF, dA = pcm_grad.get_dF_dA(pcmobj.surface)`——**不传方法名** → 生产梯度**恒走 SWIG 分支**。
- 交叉验证：以默认方法处理 ISWIG 表面 → `KeyError('R_in_J')`（分支与表面键强绑定）。
- **Hessian 侧反而读取对象属性**：`hessian/pcm.py` 多处 `get_dF_dA(pcmobj.surface, surface_discretization_method = pcmobj.surface_discretization_method)`——若把属性设为 ISWIG，能量/梯度走 SWIG 而 Hessian 走 ISWIG 分支。**【018 批区分】**"对 SWIG 表面取 `R_J` → KeyError"为**静态推断，未实际触发**；运行时已复现的异常仅一个：默认方法（SWIG）作用于**直调构建的 ISWIG 表面** → `KeyError('R_in_J')`。结论不变：设置该属性会制造内部不一致的配置；异常细节以复现记录为准。

### (e) 有无仅适用 SWIG 的公式或限制？——有
- `gen_surface` 的 SWIG 分支独有 `R_in_J/R_sw_J`（squared-switch 参数化）；ISWIG 分支用 erf 型切换并写入 `R_J`；
- 梯度 `get_dF_dA` 与 Hessian 的二阶导公式均按方法分支实现，但**分发接线**只有 SWIG 一致可达（梯度）；Hessian 分发信属性而 build 不信——三处接线互不一致。

## 2. 门判定与处置

| 前置条件 | 结果 |
|---|---|
| 方法经 build 传入 gen_surface | **未满足**（运行时实证） |
| ISWIG 分支在标准路径实际执行 | **未满足** |
| ISWIG 能量 + 梯度路径一致支持 | **未满足**（梯度恒 SWIG；设属性将致 Hessian 不一致） |

**处置**：按批令"不能确认一致支持 → 停止求值、提交缺口、不自行修补安装库"——**未执行任何 5 点求值**（含失败尝试在内新增求值数 = **0**）。缺口清单：
1. `SMD.build()` → `gen_surface` 缺方法实参；
2. `grad/smd.py` → `get_dF_dA` 缺方法实参；
3. Hessian 侧读取属性而其余侧不读取（属性接线不一致；**018 批定位：属当前 PySCF 依赖的 ISWIG 配置接线问题，历史 SWIG 结果全程一致使用 SWIG，不因此失效**）；
4. 以上属 **PySCF 平台侧问题**（独立最小案例 = 本记录的运行时跟踪，可复现）；**本批不修复平台**（未授权），修复立项须走既定平台修复流程。

## 3. JOB-016 表述补正（已落实）

- 016 路线评估中"SMD 解析梯度与解析 Hessian 均实现"改为：hess() 入口存在**≠全部贡献解析**——静电部分解析；**CDS 二阶导为有限差分**（`hessian/smd.py get_cds`，源码显式警告 "Using finite difference scheme for CDS contribution"）；
- 路线 A 标注"当前安装版本不可执行"（见 `notes/aqueous_method_route_evaluation_2026-09-07.md` 的 017 批更新）。

## 4. 对本批计划的影响

- 第二节（5 次求值）与第三节（SWIG/ISWIG 逐幅度对照）**未执行**——前置条件不成立，无数据可分析；
- 013/015 的 SWIG 三档结果维持封存状态（`notes/diagnostic_round_sealed_2026-09-07.md`）；
- 后续唯一建议见批报告：采用**明确标注的受限电子能流程（路线 B）**作为第一阶段收尾的补充证据路径。
