# JOB-073 端点报告：HOO·（频率补齐——零负模，局部极小值支持）— 项目方法模型

日期：2026-09-11。目录：`run_artifacts/02_nh3o3_reference/endpoint_cont073/hoo/`。
结果：`hoo_freq073_results.json`；账本：`budget_hoo_freq073.json`（3/3：confirm 1＋hess_dft 1＋hess_d2_classical 1，D2 FD 18次单列）。

> 项目方法模型，非S13原始三维结构复现。方法与JOB-026/070一致：wb97xd＋项目显式D2一次、def2-TZVP、L8、grid_response=True、SCF 1e-12/1e-9、气相、UKS doublet（spin=1）。

## 背景（072补正的落实）

JOB-070中HOO·通过了优化（8/15）、独立复核（dE=5.68×10⁻¹⁴）与内部轨道稳定性（stable_i=True，`return_status=True`），但**无频率**——JOB-072补正将其表述为"驻点候选，振动曲率未核查"。本批阶段D按优先顺序第1位补齐频率。

## 频率验收

新对象端点确认：**PASS，dE=0.00e+00**（精确复现070复核记录）。解析DFT＋显式D2合成核验4.57×10⁻¹³ PASS。TR投影频率（同位素平均质量、秩6、全模保留）：

**1244.40 / 1462.40 / 3670.97 cm⁻¹，负模0**（PySCF同矩阵交叉核验差2.8×10⁻⁴ cm⁻¹）。

**登记（补正升级）：HOO·现为"在本方法（wb97xd+D2/def2-TZVP/L8/grid_response，气相）和检查范围内支持局部极小值"——零负模频率依据补齐，取代此前"驻点候选、曲率待查"表述。** stable_i=True与零负模分别报告，两者均通过。⟨S²⟩=0.7540保持登记。

E=−150.9243782856 Eh（JOB-070复核值，本批未改动）。

## 边界

非S13复现；电子能无ZPE/热/CP；stable_i＋零负模≠反应路径可靠性；未修改平台/依赖/历史结果。
