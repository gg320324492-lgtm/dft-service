# 文献清单与分类报告（JOB-2026-0906-023 · 2026-09-07）

**来源**：`C:\Users\pc\Desktop\DFT学习\DFT文献`（只读；原件未移动/删除/改名）。**入库位置**：`references/literature/`（哈希校验复制，含源路径映射）。完整 manifest：`references/literature/manifest.json`。

## 1. 总量与完整性

- **48 个文件**（46 PDF + 1 ZIP + 1 HTML）；总 ~165 MB。
- 完整性：47 份 PDF 可正常打开并提取文本（含 375 页的 Buxton 评述）；ZIP 完整（10 成员）；HTML 为网页存档。
- **无法读取：0**。

## 2. 重复与版本映射

| 类型 | 文件 | 判定 |
|---|---|---|
| 精确重复（SHA-256） | 无 | —— |
| ZIP 容器冗余 | `Taylor_&_Francis_Articles(07Sep2026).zip` 的 10 个成员 = 已入库散装 00–09（**成员哈希与散装逐一相同**） | ZIP 保留原样，**不重复解压** |
| 重下载副本（内容一致） | `1-s2.0-S0045653520325856-main(1).pdf`、`1-s2.0-S0301010417305505-main(1).pdf`、`1-s2.0-S0304389419304030-main(1).pdf`、`513_1_online(1).pdf` | 页数与首页文本**完全一致**（哈希差异为文件级）→ 记为**同一版本重下载**，非不同版本 |
| 版本对应关系 | 以上 4 组按基础文件名关联；不以"(1)"后缀单独判断 | manifest.json `version_groups` |

## 3. 分类清单

### A. 精读（深度提取记录已建）
| 文件 | 身份 | 记录 |
|---|---|---|
| `2013-MechanismandthermodynamicsofmultichannelNH3O3.pdf` | **S13**：NH₃+O₃ 气相多通道机理（PRKM 38:266–282, 2013） | `notes/literature/S13_2013_extraction_record.md`（18 页全文提取件 `S13_2013_fulltext_extract.txt`） |

### B. 仅筛选（书目+首页识别完成；简明记录）
| 文件 | 身份 | 组 |
|---|---|---|
| `02.NH3OHcomplex.pdf` | Corchado 等，JPC 1993, 97:9129–9132：NH₃+OH 复合物与鞍点 | NH₃+OH |
| `1-s2.0-S0301010417305505-main.pdf`（+(1)） | Vahedpour 组：NH₃/NH₂+OH 单/双/三重 PES 比较（Chem. Phys.） | NH₃+OH |
| `152704_1_1.4986151.pdf` | JPCA 2017（DOI 10.1063/1.4986151）：NH₃+OH 高级别速率 200–2500 K | NH₃+OH |
| `d5ea00042d.pdf` | Salo/Chen/Kjaergaard：大气氨氧化中 aminoperoxyl（NH₂OO·）的形成 | NH₂·+O₂ |
| `Int J of Chemical Kinetics - 2022 - Cai...pdf` | **S2**：UV/O₃ 消化动力学（DOI 10.1002/kin.21606） | UV/O₃ 氨氮 |
| `1-s2.0-S0144860911000318-main.pdf` | **S1**：Schroeder 等 2011 海水 RAS 臭氧除氨 | 海水 RAS |
| `ie9702082.pdf` | Kuo 等：NH₃ 臭氧氧化动力学（±H₂O₂） | 氨氮动力学 |
| `acs.est.6c02098.pdf` | Manasfi/von Gunten：臭氧氧化中 NO₂⁻ 氧化与硝化机理 | NO₂⁻/NO₃⁻ |
| `1-s2.0-S0304389419304030-main.pdf`（+(1)） | J. Hazard. Mater.：NH₂OH 活化臭氧化 | NH₂OH |
| `1-s2.0-S0045653520325856-main.pdf`（+(1)） | Chemosphere 综述：NH₂OH 驱动 AOP | NH₂OH |
| `1-s2.0-S1385894725062096-main.pdf` | Chem. Eng. J.：纳米气泡界面 OH⁻ 驱动臭氧活化 | 界面 |
| `ew3c00031.pdf` | ESWR&T：MNB 强化臭氧的气泡行为与界面反应 | 界面 |
| `ee3c00124.pdf` | EES：纳米气泡 ·OH 生成"有无"的评估 | 界面（反向证据） |
| `ie302212p.pdf` | I&EC Res：臭氧微气泡去除水中氨 | 界面+氨氮 |
| `1-s2.0-S0043135424010479-main.pdf` | WR 263 (2024) 122148：高盐条件臭氧纳米气泡 AOP | 界面+高盐 |
| `1-s2.0-S0043135424018827-main.pdf` | **WR 272 (2025) 122982 = S9 Liang 等**：·OH 病毒灭活+DFT 位点 | 病毒分子损伤 |
| `513_1_online.pdf`（+(1)，375 页） | **Buxton 等 1988**：e⁻aq/H/·OH 速率常数批判性评述（JPCRD 17） | 工具书 |
| `cr50023a005.pdf` | Chem. Rev. 1958（Bailey）：O₃-有机反应经典综述 | 背景 |
| `1-s2.0-014919709500040Q-main.pdf` | Prog. Nucl. Energy 1995（Fábián）：水相臭氧分解机理 | ·OH 引发 |
| `jp206518j.pdf` | JPCA 2011：NH₂/C⁰ + O₂(a¹Δg) 理论 | NH₂+O₂ |
| `00–09`（T&F，Ozone: Sci. & Eng. 等 10 篇） | 酚 UV/O₃、O₃ 自分解与界面残留、传质模型、TNT/RDX、马拉硫磷、·OH/O₃ 比 II、分解模型应用、氯酚建模、对氯硝基苯、甲酚异构体 | 方法/实验解释参考 |
| `nanomaterials-12-01958-v2.pdf` | Nanomaterials 综述：臭氧 MNB+纳米颗粒 | 界面背景 |
| `c7ra05727j.pdf`、`jp0c09207.pdf`、`es2c02098.pdf`、`1-s2.0-S2213343722013987-main.pdf` | 染料 DFT、对羟苯甲酸酯臭氧化 DFT、Mn(VII)/TEMPO、JECE 2022 臭氧化机理 | 其他污染物（方法参考） |

### C. 不相关/低相关（登记，不入网络）
| 文件 | 原因 |
|---|---|
| `acp-6-3603-2006.pdf` | MESSy 排放实现技术札记 |
| `es00104a010.pdf` | 1982 Puget Sound 烃通量（非氧化机理） |
| `j100275a027.pdf` | 1986 JPC Ni 晶体（非主题） |
| `es00142a012.pdf` | 1985 缺失值填补（非主题） |
| `es00131a011.pdf` | 首页为参考文献页——正文主题**待核实**（仅筛选） |
| HTML | 冷水鱼养殖臭氧-氨氮反应塔设计（中文**网页资料**，工程设计非论文） |

## 4. 哈希/版本明细
见 `references/literature/manifest.json`（48 条：sha256、大小、格式、页数、首页文本头、复制动作、重复/版本映射、ZIP 检查）。
