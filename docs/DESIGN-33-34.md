# 设计稿 #33 / #34 (Phase 3 第二波, 待确认后实施)

状态: 设计待课题组确认, 未实施。两篇均依赖已交付的能力: #27 (opt freq)、
#28 (extra_route)、#29 (内联 xyz)。

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
