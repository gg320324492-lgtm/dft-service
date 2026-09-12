# SMD + D2 包装 undo_solvent 缺陷修复（JOB-2026-0906-006 阶段 B）

## 复现命令
```
wsl bash -lc 'cd /mnt/e/dft-service/ozone_mnb_uv_simulation && /root/dftvenv/bin/python protocols/smd_restart_step2_verify.py'
```
（首个 SCF+梯度调用即触发，100% 复现）

## 根因（项目包装层，非平台/PySCF）
- 生产 mf 类链：`SMDRKS_D2_GR → SMDRKS_D2 → SMDRKS → SCFWithSolvent → _Solvation → RKS`。
- PySCF `grad.pcm.make_grad_object` 内部执行 `base_method.undo_solvent().Gradients()`。
- 旧 `d2_full.undo_solvent` 仅剥一层（`mro[1]`），留下 `RKS_D2`；该类由 `attach_d2` 动态创建，其 `Gradients = nuc_grad_method` 闭包引用 **SMDRKS**（attach 时的 parent_cls）→ 调用重入溶剂机制 → `assert isinstance(base_method, _Solvation)` 失败。
- **潜在后果（推测，未实测发生；007 批修订表述）**：若断言未触发，溶剂梯度内部的 vac_grad 会带 D2 包装，而外层 `attach_d2_grad` 又加一次 → SMD 梯度中存在 D2 双重计入的可能。**实测发生的只有断言失败本身**；双重计数从未被观测（断言先失败），也未出现在任何已保存结果中（007 同密度 D2-once 达机器精度）。

## 最小修复（d2_full.py undo_solvent，项目文件）
循环剥离所有项目包装类（名字含 `SMD`/`_D2`/`_GR`）直至干净真空 RKS/UKS。D2 贡献由外层 `attach_d2_grad` 恰好计一次，无损失、无重复。修复后 `undo_solvent()` 返回类 MRO = `[RKS, KohnShamDFT, RHF, SCF, ...]`（已验证）。

## 回归
- `protocols/smd_restart_step2_verify.py` 全程通过（本批阶段 B 即回归：scanner vs 独立对象、原生 SMD+显式 D2 vs 生产路径、D2 计一次、A→B→A 复现）。
- 气相路径不经过 undo_solvent（RKS 无该方法），`test_production_path_regression.py` 不受影响（28/0 维持）。

## 平台版本
PySCF 2.14.0（未修改）；修复仅涉及项目文件 `protocols/d2_full.py`。

## CDS 实现表述修订（007 批）
`libsolvent` C 例程 `mnsol_interface_` 同时返回 CDS 能量与梯度（`get_cds_legacy`）；Python 层梯度路径无有限差分调用。**C 例程内部算法未核验**——"解析"仅指 Python 层以成对能量/梯度接口调用 C 函数，不证明 C 内部为解析推导。
