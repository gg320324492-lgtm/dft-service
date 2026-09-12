# 位移单位缺陷：根因、影响范围与修复记录（JOB-2026-0906-002）

日期：2026-09-06 ｜ 类型：**项目脚本缺陷**（不涉及 DFT 平台/PySCF/Libxc）

## 根因

`curvature_audit_lib.q_for_atom_disp(d3, a)` 返回 **Bohr 约定**幅度：
位移向量取 q·d_a（Bohr）时最大原子位移 = a/BOHR_A Bohr = a Å。
但 `directional_confirmation.py` 与 `grid_response_diagnostic.py` 把 q·d
直接加到了 **Ångström 坐标**（stepC new_geometry 以 Å 存储）上——实际最大
原子位移变成 **a/BOHR_A Å = 1.8897·a**（放大 1/b = 1.8897 倍），而差分
分母仍使用 q（Bohr 约定数值）→ 报告的 k_E 被低估 b² = 0.28 倍、
k_g 与 s_E 被低估 b = 0.529 倍。

## 复现命令

```
/root/dftvenv/bin/python protocols/test_unit_consistency.py
```
（[FAIL] pre-fix estimator caught: k_E(prefix) = b²·k_true ≠ k_true——
修复前该测试失败，修复后通过。）

## 影响范围

| 批次 | 点数 | 影响 |
| --- | --- | --- |
| directional_confirmation（28 点） | 全部 | 实际幅度 1.8897×标称；k_E 低估 ×0.28、k_g/s_E 低估 ×0.529 |
| grid_response_diagnostic（10 点） | 全部 | 同上 |
| curvature_audit 三层方向探针 | 无 | Bohr 坐标 + Bohr 约定 q，自洽 |
| validation_stepD 模式扫描 | 无 | Bohr 坐标 + q·dx_cart_bohr，自洽 |
| 全部频谱/子空间分析 | 无 | 无位移操作 |
| 单元测试 | 无 | 合成势直接定义于 Bohr 约定 |

## 修复

两脚本位移行改为 `coords(Å) + q·BOHR_A·d`（等效于 Bohr 坐标 + q·d），
任务清单 settings 记录 unit_fix 字段。回归测试
`protocols/test_unit_consistency.py` 全部通过（含修复前缺陷复现项）。

## 量化影响（修正后重分析，详见 unit_consistency_report.md）

- c14_plus 争议方向：修正后 k_E = +3.1→+9.5×10⁻⁵（L6/L7，随幅度非谐增长）、
  k_g(False,修正) = +1.4→+2.6×10⁻⁴——**均为正**；解析 Hessian 的 −2.4×10⁻⁴
  被否定。
- c06_plus 争议方向：修正后 k_E 与 k_g(False) 在小幅度处 ≈ −1.3×10⁻⁵（≈0）、
  0.0094 Å 处转正（非谐）——**无负曲率证据**。
- grid_response 诊断（c14_plus）：单位修正后响应开关使 k_g 从 1.5×10⁻⁴
  收敛到 k_E 的 3.9×10⁻⁵——**层间差异的主因确认为缺失网格响应项**
  （旧"过冲"结论撤回）；c06_plus 方向 E/G 本就一致（小幅度处），假设未获支持。
- 两候选的极小值/待定状态不变（几何收敛 <1e-5 未达标仍为首要未决项）。
