> **[JOB-047 补正]** 本报告第 3/4 节部分表述已修正："24.9% 解释占比"改为> 绝对量级比（不称消除差额）；删除"剩余 1.04×10⁻⁴"的错误减法，保留两个
> 带符号差额（k_H−k_on=−1.3886×10⁻⁴、k_H−k_off=−1.7345×10⁻⁴）；D2 一致性
> 结论限定当前几何/方向/步长；RSH 仅列待定假说；原始未对称矩阵缺失不作
> 固定方向二次型差异的独立解释；−72.186 cm⁻¹ 继续标为保存矩阵预测值。
> 详见 `notes/job047_046_correction.md`。

# JOB-2026-0906-046：C1 负模方向网格响应贡献核查批

## 范围与授权
- 任务编号 = 045 之后下一个未占用（已核对）。预算：SCF 2、完整梯度 4（grid_response True/False × 2 几何）、纯 D2 解析梯度 2（单列、无 SCF）；优化/稳定性/电子 Hessian/D2 Hessian/频率重算均为 0。实际 **8 attempt 全部完成、0 失败**。

## 执行状态（全部完成）
1. **零求值补正 045**（`notes/job046_045_correction.md` + 045 报告/记录横幅）：k_E/k_g 差异按步长分别报告（h=0.01 差 1.56%、h=0.02 差 6.60%）；"已排除"按证据强度降级为限定结论；对称化不改变固定实方向二次型（数学说明）；保留"负曲率获支持、量级未解决"。
2. **源码只读核查**（PySCF 2.14.0）：梯度 grid_response 默认 False（`grad/rks.py:745`），开启时 get_veff 全响应 + extra_force(exc1_grid) 两条路径生效；**解析 Hessian 无网格响应项**（hessian/rks.py 中 grid_response 仅 NLC 数值二阶导，实测 wb97xd is_nlc=False 不触发）。未改安装源码。
3. **前置核验**：q 哈希三方一致（charter `59599f2a2c0be260`）；几何取 045 实际坐标（往返 ≤4.3e-11 Bohr）；模拟后端测试 11/11（开关到达/对象独立/无重复 SCF/真实后端屏蔽/正式目录哨兵）。
4. **正式执行**：每几何一次 SCF（chkfile+MO 摘要）→ True 梯度 → 复现门（dC=0、dE≤5.7e-13、dgrad≤2.5e-12，全 PASS）→ 同一 SCF 对象新建 False 梯度 → 解析 d2_grad（SCF 计数器证实无 SCF）。
5. **结果**（详见 `results/phase2_preparation/c1_gridresp046_report.md`）：
   - k_on = −6.9723e-5（**复现 045 k_g(0.01) 至 5.9e-13**）；k_off = −3.5133e-5；
   - **Δk_grid = −3.4590e-5**（占开响应量级 49.6%——网格响应把该软模方向 FD 曲率加大约一倍）；
   - 但 Δk_grid 仅解释 k_H−k_on 差额的 **24.9%**，k_off 更远离 k_H（关响应差额 −1.7345e-4）；**判读走"差额仍明显"分支，停止**；
   - **D2 侧嫌疑正面排除**：解析 d2_grad 曲率 +3.49600e-6 与 043 D2 FD Hessian 参考 +3.49605e-6 一致至 4.6e-11 → **矛盾全部在 DFT 部分**（−2.1208e-4 vs −3.8629e-5，~5.5×）；
   - 待定假说：H3 RSH（ωB97xd 为 range-separated hybrid，已实测 is_rsh=True）解析 Hessian 二阶导方法学局限；H2 原始未对称矩阵缺失（沿承）。

## 判读限定
不称过渡态/极小值；不宣布任何方向 Hessian"已验证"；关闭响应的 FD 不自动等同于解析 Hessian；max|g|≤1×10⁻⁵ 驻点门限保留。

## 交付
`results/phase2_preparation/c1_gridresp046_report.md`、`run_artifacts/02_nh3o3_reference/c1_gridresp046/`（prep_check_results.json、mock_test_results.json、budget_scan046.json、eval_scf/grad_on/grad_off/d2grad × p01/m01、grid_response_comparison.json、chk_p01/m01.chk、exec_log.txt）、脚本 `job046_prep_check.py`/`job046_test_mock.py`/`job046_exec.py`、045 补正 `notes/job046_045_correction.md`、任务记录（本文件）。

**本批停止。** 未解释差额（~1.04e-4 Eh/Bohr²，DFT-XC 侧）的归因与后续路线待总指挥决定。
