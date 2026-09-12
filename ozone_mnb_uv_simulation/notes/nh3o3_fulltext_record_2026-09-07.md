# S13 全文获取记录与资料缺项表（JOB-2026-0906-022 · 2026-09-07）

**目标文献**：Asgharzade S., Vahedpour M., "Mechanism and Thermodynamics of Multichannel 1:1 Ammonia and Ozone Tropospheric Oxidation Reaction", *Progress in Reaction Kinetics and Mechanism* 38(3): 266–282 (2013), DOI 10.3184/146867813X13738207456659。

## 1. 获取尝试（逐渠道；未绕过任何访问限制）

| 渠道 | 结果 |
|---|---|
| Sage 官网 `journals.sagepub.com/doi/full/...` | **付费墙**：仅题名/作者/摘要/参考文献/卷期可见；正文、表格、图、坐标不可见 |
| ResearchGate 作者页（Somaie Asgarzade） | 检索结果显示该文条目标记 **"Full-text available"**（作者上传过全文副本），但自动访问被 RG 限制页拦截（要求登录）；**未取得** |
| Semantic Scholar 论文页 + API | 元数据可读（BibTeX、卷期页码 266–282）；**openAccessPdf 未能确认**（API 一次限流，页面无 OA PDF 链接）；S2 显示其抽取了 **table 1 / fig 1–3 / table 3–4** 的题注——证明正文含结构/能量表，但题注内容未渲染获取 |
| CNPereading（Sage 中国镜像） | 仅摘要 + **完整参考文献列表（41 条）**可见 |
| 其他 | 未使用任何绕过手段（sci-hub 等不采用） |

**附带收获（合法可见的参考文献列表 → 方法提示，静态推断非逐项确认）**：Gaussian 03（ref 12）；Boys–Bernardi（ref 24 → **对复合物做了 BSSE 校正的推断**）；Gonzalez & Schlegel 1990（ref 23 → **使用了 IRC/TS 方法学的推断**）；AIM2000（ref 25 → 电荷/键临界点分析推断）；SSUMES/GPOP（ref 40/41 → **主方程速率理论**推断）。

## 2. 已取得 vs 未取得（逐项登记）

| 资料项 | 状态 | 来源/依据 |
|---|---|---|
| 题名/作者/卷期/DOI/页码（266–282） | ✅ 已取得 | Sage/CNPereading/S2 元数据 |
| 摘要（通道定义 P1–P6、C1/C2、定性结论） | ✅ 已取得 | 三平台摘要一致 |
| 优化结构（原子坐标、键长/键角：反应物/加合物/C1/C2/TS/P1–P6） | ❌ **未取得**（付费墙） | 全文获取尝试记录 |
| 通道-络合物对应关系（C1/C2 → P1–P6 映射） | ❌ 未取得（摘要仅"variety of transformations"） | 同上 |
| 相对能量/ZPE 校正数值 | ❌ 未取得 | 同上 |
| 过渡态清单与连接关系 | ❌ 未取得（IRC 方法由参考文献**推断**） | 同上 |
| 速率常数数值 | ❌ 未取得（仅"k 仅对 P1(2)/P4、200–2500 K"） | 同上 |
| 方法细节 | ◐ **部分**：摘要给出 B3LYP/CCSD(T)/G3B3/6-311++G(3df,3pd)；参考文献给出方法学线索（BSSE/IRC/AIM/主方程，**推断性质**） | 摘要 + 参考文献 |
| 电子态信息 | ❌ 未取得（摘要无逐物种自旋声明） | —— |

## 3. 结论

**"按原文方法复现"当前不可行**（无结构、无能量数值、无 TS 清单）。可行的仅是：**用项目方法开展对照**（明确标注"项目方法对照，非原文复现"），且最小可执行单元收缩到**证据充分的部分**——单体与加合物层面（见 `notes/nh3o3_minimal_task_draft_2026-09-07.md`）。

**后续获取建议**（不绕过限制）：① 由用户/老师通过机构订阅下载 Sage 全文（正文含 Table 1–4、Fig 1–3，大概率含全部结构与相对能量表）；② 或联系通讯作者（vahed@znu.ac.ir）索取 reprint 与补充材料；③ RG 作者页副本需人工登录获取。取得后本批缺项表逐项销账，复现草案再升级。
