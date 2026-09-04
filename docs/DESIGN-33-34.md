# 设计稿 #33 / #34 (Phase 3 第二波) — 已于 2026-09-04 实施

状态: **均已实现并真算验证** (下方为原设计, 实施偏差见文末"落地记录")。

## #33 反应能垒 / TS 工作流

**目标**: 从反应物/产物 (或复合物) 出发自动给出 ΔE、ΔG(含热化学校正) 与
TS 候选; 服务"一步反应"的能垒查询。

**接口草案**:
```
POST /dft/reaction
{
  "reactant":   {"xyz_content": "..."} 或 {"smiles": "..."},
  "product":    {...},
  "xc": "B3LYP", "basis": "6-31G(d)", "solvent": "water",
  "ts_search": "qst2" | "none",        # none = 只算两端 ΔG
  "callback_url": "..."
}
→ 编排为多个 gaussian 子任务 (现有 /dft/gaussian 复用, 编排层串):
  1. reactant/product 各 opt freq (#27 一次拿齐) → ΔG_soln
  2. ts_search=qst2: 拼两段几何的复合 gjf (atoms A 块 + 空行 + B 块 +
     "B" 连接行), route 加 opt(qst2) — 走 extra_route/geometric 输入 (#29)
  3. TS 任务失败 (IRC 不连通/未收敛) 如实返回, 带日志路径, 不假装成功
```

**边界与风险 (为什么先讨论)**:
- QST2 成功率对体系极敏感 (需要两端几何原子序一一对应; 重排反应常失败) —
  承诺"全自动找 TS"会误导, 定位为"一键尝试 + 诚实报告"。
- 溶剂 + TS: SMD 对 TS 的适用性有争议, 文档标注。
- 工作量: driver 复合 gjf 拼装 ~120 行 + 编排状态机 ~150 行 + 测试。

**建议**: 先只做第 1 步 (ΔE/ΔG 双端汇总, `ts_search=none`), TS 搜索保持
extra_route 手动逃生舱形态, 等实际课题用到再自动化。

## #34 气液界面分析包

**目标**: 面向气泡课题的 GROMACS 轨迹固定分析流: 密度剖面 (界面位置/厚度)、
水分子氢键数时间序列、离子/表面活性剂的配位数与界面吸附量。

**依赖**: #31 (analyze v1 已交付 RMSD/能量)。实施拆两步:
1. `analyze=["density","hbond"]` 扩展 driver 内 gmx 调用:
   - 密度剖面: `gmx density -d Z` → scichem python 读 xvg 找密度跳变 →
     界面 z 坐标 + 气液界面面积估计
   - 氢键: `gmx hbond` (D-A ≤0.35 nm, 角 <30°) → hbond 数/时间
2. 界面吸附量 (需要索引组定义分子类别) → 需要先确认体系的 .ndx 生成策略
   (topol.top 里的分子名分组 vs 自定义 index), **这是需要课题组定的点**:
   常用分子种类清单、报告哪些量、单位约定。

**需要确认**: 典型盒子尺寸 (影响 gmx 内存/线程参数)、是否固定 NPT、
表面活性剂分子类型 (决定 ndx 模板)。

## 落地记录 (2026-09-04)

**#33 已实施且超出设计**: `POST /dft/reaction` (reactants/products + count +
charge/mult, 各跑 gaussian opt freq → ΔE/ΔG 按计量数聚合)。真算:
- HF + H2O → F- + H3O+ : ΔE=+1641.4 / ΔG=+1662.1 kJ/mol (气相离子对, 物理正确)
- **QST2 真找到 TS** (超出"一键尝试"预期): H+H2 交换反应, G16W 报
  "Search for a saddle point of order 1" + Normal termination,
  barrier = 24.4 kJ/mol (STO-3G 极小基组低估, 实验 ~38.5, 量级正确)。
  踩坑记录: 裸 `qst2` 路由非法 (正确 opt=qst2, 驱动已归一化); "B" 分隔行必须
  在两段分子**之间** (末尾→"Wanted integer", 无→"End of file in ZSymb")。
- try_ts 边界不变: 多物种/计量数≠1 → skipped; TS 确认 (恰 1 虚频) 仍需对
  返回几何单独 freq (log_path/chk 已给)。

**#34 已实施且解决遗留 ①②**: `analyze_options=[rms|energy|density|rdf|hbond]`
+ **`water_model=spce`** — GROMACS 内置金标准 SPC/E 水模型
(gmx solvate + 自包含拓扑; include oplsaa.ff 在 GROMACS 2023.3 Ubuntu 包触发
"Invalid order for directive atomtypes" → 改内联 atomtypes, 参数按官方
spce.itp: q±0.4238/-0.8476, σ=0.3166, ε=0.650, settle 0.1/0.1633)。
真算 (2nm 盒 221 分子):
- O-O RDF 首峰 **0.274 nm** (文献 ~0.28) ✓
- 氢键 **3.24/分子** (SPC/E 300K 文献 3.2-3.6) ✓ — hbond 修复: 本机
  gmx hbond 无 -rtp/-type, 交互喂 "Water\nWater\n" (residuetypes SOL=Water)
- 密度 kg/m³→g/cm³ 单位修正 ✓
- demo GROMOS 路径保留 (链路级), spce 仅纯水 (API 校验 smiles=O)
