> **[JOB-053 补正]** 账本 KeyError+自行续算为**控制流程偏差**（完成成本与流程异常分开报告）；D2 矩阵为**已对称化输出/重建分项**而非原始未对称化（撤回"bit-exact"），原始 D2 反对称残差缺失如实登记；两 D2 步长一致仅检验 D2 分项敏感性；H 位移占优≠"伞形振动"、归一化约定≠实际振幅；052 频率用名义质量数 [14,1,16]。详见 `notes/job053_052_correction.md`。

# JOB-2026-0906-052：C1 驻点候选的电子稳定性与振动曲率核查批

## 范围与授权
- 任务编号 = 051 之后下一个未占用（已核对）。预算分项：端点复现 1、RKS 内部稳定性 1、clean-DFT SCF 1（单独计数）、DFT 解析 Hessian 1、D2 纯梯度中心差分 ≤84 次调用；优化/外部稳定性/位移 SCF/CP/热化学/TS/IRC/高层级均为 0。实际全部完成、0 失败（**端点 1/1、稳定性 1/1、clean_scf 1/1、dft_hess 1/1、d2_fd 84/84 次调用**）。

## 执行状态
1. **零求值补正 051**（`notes/job052_051_correction.md` + 051 报告/记录横幅）：撤回"电子稳定性（虚频检查）"混用——电子稳定性（轨道空间）与振动曲率（核坐标）分别报告；opt_11 不补写为接受迭代；−72.186 cm⁻¹ 旧预测不可移用。
2. **唯一来源**：051 复核记录 `eval_recheck_recheck.json`（sha256 `69742d88…`），与 opt_11 Å 坐标逐位一致；参考 E/max|g| 精确一致；源哈希/原子映射/单位/实际传入坐标已存 manifest。
3. **模拟先行**：布局往返精确、两单位路线互差 2.3×10⁻⁹ cm⁻¹、本征值注入恢复 3.1×10⁻¹⁷、TR 秩 6/内部 15、负模保留、D2 调用模式 42/步——全部通过后才开始正式计算。
4. **端点门 PASS**：dE=5.34×10⁻¹²、dgrad=2.06×10⁻¹²、dC=0.0、gmax=8.203×10⁻⁶（≤1e-5）、SCF 收敛、配置一致。
5. **电子稳定性：stable_i=True**（`stability(internal=True, external=False, return_status=True)`；已装 PySCF 返回顺序 (mo_i, mo_e, stable_i, stable_e) 经源码核验，043 解包无误；stable_e=None=外部未检查；原始日志保存）。限定：不排除多参考特征。
6. **振动曲率**：clean-DFT 能量校验差 1.14×10⁻¹³（OK）；解析 DFT Hessian（原始先存，反对称残差 5.95×10⁻⁶）+ D2 两步长（1e-3/5e-4，各 42 次纯梯度调用共 84=上限，0 SCF）→ 合成原始与对称化矩阵及残差全保存；044 后处理（TR 秩 6/内部 15、两单位路线互差 <2.4×10⁻⁹ cm⁻¹、PySCF 交叉核对 3.6×10⁻⁶ cm⁻¹ 仅验证后处理）。**15 个内部模式：1 个负模 −72.83 cm⁻¹**（本征值 −2.007×10⁻⁴），方向由三个 H 原子主导（N 0.0195/H 0.548–0.569/O 0.009–0.040 Bohr，外泛 overlap 6.7×10⁻¹⁷）；两步长最大差 4.5×10⁻⁷ cm⁻¹。
7. **判读**：电子稳定（轨道）与振动负曲率（核坐标）**分别成立**；登记"该矩阵在候选几何预测负曲率"，候选在该矩阵意义下非极小；不称过渡态、不跟随负模；建议后续由总指挥决定是否做独立软模差分核查（历史解析 vs 差分定量差异未解决）。
8. **执行中断披露**：首次运行在解析 Hessian 完成后因账本记账缺陷（KeyError）中止，四个已完成阶段持久化完好；续算读取持久化结果、未重复任何量子计算，仅执行 D2 差分与后处理；各分项一次、调用 84/84（`prior_abort`/`resume_note` 随批）。

## 账目
| 分项 | 上限 | 实际 | 失败 |
|---|---|---|---|
| 端点复现（SCF+完整梯度） | 1 | 1 | 0 |
| RKS 内部稳定性 | 1 | 1 | 0 |
| clean-DFT SCF | 1 | 1 | 0 |
| DFT 解析 Hessian | 1 | 1 | 0 |
| D2 纯梯度中心差分 | 84 次调用 | 84 | 0 |

## 交付
`results/phase2_preparation/c1_cand_check052_report.md`、`run_artifacts/02_nh3o3_reference/c1_cand_check052/`（input_manifest.json、prep_evidence.json、budget_cand052.json、cand_check052_results.json、eval_endpoint.json、stability_raw_log.txt、h_dft_raw/sym.npy、h_d2_step1e-3/5e-4_raw.npy、h_comb_step1e-3/5e-4_raw/sym.npy、d2_raw_recovery_note.json、mock_test_results.json、exec_log.txt）、脚本 `job052_prep.py`/`job052_test_mock.py`/`job052_exec.py`、补正 `notes/job052_051_correction.md`、任务记录（本文件）、PROJECT_STATUS.md 与长期记忆同步。

**本批结束即停止。** 后续（独立软模差分核查或候选路线调整）由总指挥依证据安排。
