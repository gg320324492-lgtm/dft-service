// 组会汇报 6 页 — 臭氧MNB/UV协同水处理 DFT 阶段汇报
// 上下标统一用 run 级 superscript/subscript（fx() 标记），避免 Unicode 上下标字符的字体回退脱节
const pptxgen = require("pptxgenjs");

const p = new pptxgen();
p.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
p.author = "DFT 模拟工作组";
p.title = "臭氧微纳米气泡/UV协同处理养殖水体 — DFT 模拟阶段汇报";

// ---------- palette ----------
const DARK = "0B3B47", DARK2 = "0F4A59", PRIMARY = "0E5E6F", ACCENT = "E8590C";
const BG = "FFFFFF", TINT = "EAF3F5", TINT2 = "F5FAFB";
const TEXT = "1E293B", MUTED = "5B7080", HAIR = "C9DAE0";
const LIGHT = "E8F1F3", MUTED_D = "9FBCC4";

const F = "微软雅黑";
const W = 13.33, M = 0.5;

const bu = () => ({ code: "2022", indent: 12 });
const ASSETS = "results/discussion/assets";
const SRC = "run_artifacts/02_nh3o3_reference";

// fx("NO_{2}^{-}") -> rich-text runs：_{..}=下标 ^{..}=上标
function fx(s, base = {}) {
  const runs = []; const re = /([_^])\{([^}]*)\}/g; let last = 0, m;
  while ((m = re.exec(s))) {
    if (m.index > last) runs.push({ text: s.slice(last, m.index), options: { ...base } });
    runs.push({ text: m[2], options: { ...base, [m[1] === "^" ? "superscript" : "subscript"]: true } });
    last = m.index + m[0].length;
  }
  if (last < s.length) runs.push({ text: s.slice(last), options: { ...base } });
  return runs;
}

// ---------- helpers ----------
function header(s, num, title, dark = false) {
  s.addText(num + " / 06", { x: W - 1.6, y: 0.34, w: 1.1, h: 0.32, align: "right",
    fontFace: F, fontSize: 12, color: dark ? MUTED_D : MUTED, margin: 0 });
  s.addText(title, { x: M, y: 0.34, w: W - 2.3, h: 0.62, fontFace: F, fontSize: 27,
    bold: true, color: dark ? "FFFFFF" : PRIMARY, margin: 0, valign: "middle" });
}
function sourceLine(s, txt, dark = false) {
  s.addShape(p.shapes.LINE, { x: M, y: 7.02, w: W - 2 * M, h: 0,
    line: { color: dark ? "1D5A6B" : HAIR, width: 0.75 } });
  s.addText("数据来源：" + txt, { x: M, y: 7.08, w: W - 2 * M, h: 0.3, fontFace: F,
    fontSize: 12, color: dark ? MUTED_D : MUTED, margin: 0 });
}
function chipRow(s, x, y, w, h, chip, chipColor, lead, desc, fs = 13) {
  s.addShape(p.shapes.ROUNDED_RECTANGLE, { x, y: y + 0.05, w: 0.62, h: 0.34,
    fill: { color: chipColor }, rectRadius: 0.06 });
  s.addText(chip, { x, y: y + 0.05, w: 0.62, h: 0.34, align: "center", valign: "middle",
    fontFace: F, fontSize: 12, bold: true, color: "FFFFFF", margin: 0 });
  s.addText([
    { text: lead, options: { bold: true, color: TEXT, breakLine: true } },
    ...fx(desc, { color: MUTED, fontSize: fs - 1.5 }),
  ], { x: x + 0.78, y, w: w - 0.78, h, fontFace: F, fontSize: fs, margin: 0, valign: "top", paraSpaceAfter: 3 });
}

// =================================================================
// Slide 1 — 封面：研究问题与总体路线（深色）
// =================================================================
let s = p.addSlide();
s.background = { color: DARK };
s.addText("组会汇报 · 2026-09-10", { x: M, y: 0.55, w: 6, h: 0.35, fontFace: F,
  fontSize: 14, color: ACCENT, bold: true, margin: 0 });
s.addText([
  { text: "臭氧微纳米气泡 / UV 协同处理养殖水体", options: { breakLine: true } },
  { text: "氨氮去除的分子机制 —— DFT 模拟阶段汇报", options: {} },
], { x: M, y: 1.0, w: W - 2 * M, h: 1.55, fontFace: F, fontSize: 31, bold: true,
  color: "FFFFFF", margin: 0, paraSpaceAfter: 6 });

s.addText("研究目标链条：为实验观察提供可检验的分子解释", { x: M, y: 2.85, w: 9, h: 0.35,
  fontFace: F, fontSize: 13.5, color: MUTED_D, margin: 0 });
const chain = [
  "O₃ / UV 活化",
  "活性氧 ROS",
  fx("NH_{3} / NH_{4}^{+} 氧化"),
  fx("NH_{2}OH · NO_{2}^{-} · NO_{3}^{-} 中间体"),
];
const cw = [2.35, 2.35, 2.75, 3.6];
let cx = M;
chain.forEach((t, i) => {
  s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: cx, y: 3.3, w: cw[i], h: 0.62,
    fill: { color: DARK2 }, line: { color: "2E6B7C", width: 1 }, rectRadius: 0.09 });
  s.addText(t, { x: cx + 0.08, y: 3.3, w: cw[i] - 0.16, h: 0.62, align: "center", valign: "middle",
    fontFace: F, fontSize: 13.5, bold: true, color: LIGHT, margin: 0 });
  if (i < 3) s.addText("→", { x: cx + cw[i] - 0.06, y: 3.3, w: 0.42, h: 0.62,
    align: "center", valign: "middle", fontFace: F, fontSize: 17, color: ACCENT, margin: 0 });
  cx += cw[i] + 0.31;
});

s.addText("对照导师批准的四阶段计划", { x: M, y: 4.45, w: 9, h: 0.35, fontFace: F,
  fontSize: 13.5, color: MUTED_D, margin: 0 });
const phases = [
  ["① 臭氧水合基线", [{ text: "已建立（水相仍有未验收项）", options: { fontSize: 11.5, color: MUTED_D } }], false],
  ["② 目标物反应模型", fx("当前：NH_{3}+O_{3} 气相接触构型", { fontSize: 11.5, color: "FFE3D1" }), true],
  ["③ 反应路径与能垒", [{ text: "TS / IRC 尚未开展", options: { fontSize: 11.5, color: MUTED_D } }], false],
  ["④ UV 激发态 · 气液界面", [{ text: "按实验条件论证后开展", options: { fontSize: 11.5, color: MUTED_D } }], false],
];
const pw = [2.95, 3.1, 2.75, 3.05];
cx = M;
phases.forEach(([t, dRuns, cur]) => {
  s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: cx, y: 4.9, w: pw[0], h: 1.05,
    fill: { color: cur ? ACCENT : DARK2 }, line: { color: cur ? ACCENT : "2E6B7C", width: 1 }, rectRadius: 0.09 });
  s.addText([
    { text: t, options: { bold: true, fontSize: 13.5, color: "FFFFFF", breakLine: true } },
    ...dRuns,
  ], { x: cx + 0.16, y: 4.9, w: pw[0] - 0.32, h: 1.05, fontFace: F,
    margin: 0, valign: "middle", paraSpaceAfter: 3 });
  cx += pw[0] + 0.17;
});

s.addText([
  { text: "当前阶段：", options: { bold: true, color: ACCENT } },
  ...fx("第二阶段早期 —— NH_{3}+O_{3} 气相接触构型搜索与性质核查（已完成 JOB-001–059 批次）；水相、UV 激发态、气泡界面与完整反应路径尚未完成。", { color: LIGHT }),
], { x: M, y: 6.25, w: W - 2 * M, h: 0.62, fontFace: F, fontSize: 13.5, margin: 0, valign: "top" });
s.addText("本汇报全部数值取自项目已登记计算记录（results/ 与 run_artifacts/），不含未登记的实验结果。", {
  x: M, y: 7.05, w: W - 2 * M, h: 0.3, fontFace: F, fontSize: 12, color: MUTED_D, margin: 0 });
s.addNotes("开场：课题背景是臭氧微纳米气泡与紫外协同处理养殖水体，DFT 模拟负责给实验现象提供分子层面的机理解释。目标链条从 O₃/UV 活化产生活性氧，到把氨氮（NH₃/NH₄⁺）氧化为 NH₂OH、NO₂⁻、NO₃⁻ 等含氮中间体。强调四阶段定位：导师已批准总体方向，目前处于第二阶段早期，主线是 NH₃＋O₃ 气相接触构型；第一阶段水相基线已建立但仍有未验收项；第三、四阶段尚未开展。声明：本汇报只用已登记的计算证据，不含未登记的实验结果。");

// =================================================================
// Slide 2 — 计算体系与可靠性基础（浅色）
// =================================================================
s = p.addSlide();
s.background = { color: BG };
header(s, "02", "计算体系与可靠性基础");

s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: M, y: 1.22, w: 5.15, h: 4.6,
  fill: { color: TINT }, rectRadius: 0.1 });
s.addText("对照计算方法（已多轮核验）", { x: M + 0.25, y: 1.42, w: 4.6, h: 0.4,
  fontFace: F, fontSize: 15.5, bold: true, color: PRIMARY, margin: 0 });
const meth = [
  ["泛函", [{ text: "wB97X-D ＋ 项目独立实现的 D2 色散校正", options: {} }]],
  ["基组 / 网格", [{ text: "def2-TZVP ｜ 积分网格 L8", options: {} }]],
  ["SCF 收敛", fx("能量 1×10^{-12} ｜ 密度 1×10^{-9}")],
  ["梯度", [{ text: "解析完整梯度，逐求值落盘持久化", options: {} }]],
  ["驻点验收门限", fx("max|g| ≤ 1×10^{-5} Eh/Bohr（未经投影）")],
  ["过程管控", [{ text: "输入来源追踪 · 单位核对 · 逐求值预算 · 测试隔离", options: {} }]],
];
meth.forEach(([k, vRuns], i) => {
  const y = 1.95 + i * 0.63;
  s.addText(k, { x: M + 0.25, y, w: 1.5, h: 0.55, fontFace: F, fontSize: 12.5, bold: true,
    color: PRIMARY, margin: 0, valign: "top" });
  s.addText(vRuns, { x: M + 1.8, y, w: 3.1, h: 0.58, fontFace: F, fontSize: 12.5,
    color: TEXT, margin: 0, valign: "top" });
});

s.addText("单体基线（频率验收：均为 0 虚频，可作反应能参考点）", { x: 5.95, y: 1.22, w: 6.9, h: 0.38,
  fontFace: F, fontSize: 15.5, bold: true, color: PRIMARY, margin: 0 });
const th = { fill: { color: PRIMARY }, color: "FFFFFF", bold: true, fontSize: 12.5 };
const rows = [
  [{ text: "单体", options: th }, { text: "相 / 溶剂模型", options: th }, { text: "E_total (Eh)", options: th }, { text: "max|g| (μEh/Bohr)", options: th }, { text: "虚频", options: th }],
  ["H₂O", "气相", "−76.437668", "5.6", "0"],
  ["H₂O", "SMD 水", "−76.450895", "12.8 *", "0"],
  ["O₃", "气相", "−225.433903", "1.4", "0"],
  ["O₃", "SMD 水", "−225.439360", "1.8", "0"],
  ["O₂", "气相", "−150.340070", "0.011", "0"],
  ["O₂", "SMD 水", "−150.340720", "0.020", "0"],
];
s.addTable(rows, { x: 5.95, y: 1.68, w: 6.88, colW: [1.0, 1.55, 1.95, 1.68, 0.7],
  fontFace: F, fontSize: 12.5, color: TEXT, align: "center", valign: "middle",
  border: { pt: 0.75, color: HAIR }, rowH: 0.4, fill: { color: "FFFFFF" } });
s.addText(fx("* H₂O（SMD 水）梯度略高于 1×10^{-5} 驻点门限；单体均以 0 虚频频率结果验收登记。", { color: MUTED }), {
  x: 5.95, y: 4.58, w: 6.88, h: 0.26, fontFace: F, fontSize: 12, margin: 0 });

s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: 5.95, y: 4.85, w: 6.88, h: 0.97,
  fill: { color: TINT2 }, line: { color: HAIR, width: 1 }, rectRadius: 0.08 });
s.addText([
  { text: "NH₃ 气相单体（JOB-026，第二阶段参考）：", options: { bold: true, color: TEXT, breakLine: true } },
  ...fx("E = −56.564219 Eh；与 O₃ 单体之和 −281.998122 Eh —— 即接触构型能量的零点参考。", { color: MUTED }),
], { x: 6.15, y: 4.98, w: 6.5, h: 0.75, fontFace: F, fontSize: 12.5, margin: 0, paraSpaceAfter: 3 });

s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: M, y: 6.0, w: W - 2 * M, h: 0.82,
  fill: { color: TINT }, rectRadius: 0.08 });
s.addText([
  { text: "可靠性基础：", options: { bold: true, color: PRIMARY } },
  { text: "48 篇文献入库（JOB-023）｜ S13 机论文全文解析 ｜ 来源/单位/预算/梯度持久化机制多轮核验。注意：以上为项目对照方法，未宣称复现 S13 的 B3LYP / CCSD(T) 数值。", options: { color: TEXT } },
], { x: M + 0.25, y: 6.0, w: W - 2 * M - 0.5, h: 0.82, fontFace: F, fontSize: 13,
  margin: 0, valign: "middle" });
sourceLine(s, "results/00_baseline/baseline_energies.csv（JOB-2026-0904-001~006）；results/phase1_closeout/phase1_results_summary.md");
s.addNotes("先讲方法：所有计算统一用 wB97X-D 加项目独立实现的 D2 色散、def2-TZVP 基组、L8 网格、SCF 收敛到 1e-12/1e-9，每个求值点的解析完整梯度都落盘保存，驻点验收门限是未经投影 max|g| ≤ 1e-5 Eh/Bohr。右表是第一阶段验收的单体基线：H₂O、O₃、O₂ 气相和 SMD 水相全部 0 虚频，可作为后续反应能的参考点。NH₃ 气相单体 JOB-026 完成了结构、独立梯度、内部电子稳定性和频率检查，与 O₃ 单体之和 −281.998122 Eh 就是接触构型能量的零点。可靠性机制（文献入库、S13 解析、来源追踪等）是计算可靠性工作，不算机理成果；方法口径是我们项目的对照方法，不宣称复现文献 S13 的 B3LYP/CCSD(T) 数值。");

// =================================================================
// Slide 3 — NH₃···O₃ 接触构型探索（浅色，大图 + 关键数字）
// =================================================================
s = p.addSlide();
s.background = { color: BG };
header(s, "03", "NH₃···O₃ 气相接触构型：从距离扫描到驻点候选");

const figW = 7.45, figH = figW / (1611 / 747);
s.addImage({ path: `${SRC}/c1_scan_formal/energy_radial_gradient.png`,
  x: M, y: 1.3, w: figW, h: figH });
s.addText("JOB-031：固定单体几何、C1 取向的距离扫描（wB97X-D+D2 / def2-TZVP / L8，气相）——非 CP 相互作用能在 2.6–4.2 Å 全为正（+0.4 ~ +11 kcal/mol），该固定取向线上排斥主导；吸引接触构型需全自由度优化寻找。", {
  x: M, y: 1.3 + figH + 0.08, w: figW, h: 1.0, fontFace: F, fontSize: 12.5, color: MUTED,
  margin: 0, valign: "top" });

const rx = 8.25, rw = W - 8.25 - 0.5;
s.addText([
  { text: "2", options: { fontSize: 40, bold: true, color: ACCENT } },
  { text: " 个驻点候选 ", options: { fontSize: 16, bold: true, color: TEXT } },
  { text: "· 0 个已验收极小值", options: { fontSize: 16, bold: true, color: MUTED } },
], { x: rx, y: 1.28, w: rw, h: 0.7, fontFace: F, margin: 0, valign: "middle" });
s.addShape(p.shapes.LINE, { x: rx, y: 2.05, w: rw, h: 0, line: { color: HAIR, width: 0.75 } });

const rows3 = [
  ["全自由度优化", "找到低于分离单体约 2.26 kcal/mol 的接触构型（电子能差 −3.60×10^{-3} Eh）；未经 CP 校正、含片段变形——不是结合自由能。"],
  ["JOB-051 · 首个驻点候选", "opt_11：max|g| = 8.203×10^{-6} ≤ 1×10^{-5}，独立新对象复核 PASS（E = −282.00172149 Eh）。"],
  ["JOB-057 · 第二个候选", "opt_05：max|g| = 6.847×10^{-6}，复核 PASS（E = −282.00172194 Eh，较 051 低 4.50×10^{-7} Eh）；几何几乎不变，横向松弛。"],
];
rows3.forEach(([lead, desc], i) => {
  chipRow(s, rx, 2.22 + i * 1.18, rw, 1.12, "0" + (i + 1), PRIMARY, lead, desc, 13);
});
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: rx, y: 5.82, w: rw, h: 1.0,
  fill: { color: TINT }, rectRadius: 0.08 });
s.addText([
  { text: "表达边界：", options: { bold: true, color: ACCENT } },
  { text: "“驻点候选”仅指电子能＋梯度意义下过门限并复核的几何；不称稳定复合物、结合自由能或水处理效果。", options: { color: TEXT } },
], { x: rx + 0.18, y: 5.82, w: rw - 0.36, h: 1.0, fontFace: F, fontSize: 12.5,
  margin: 0, valign: "middle" });
sourceLine(s, `${SRC}/c1_scan_formal（JOB-031）；c1_bfgs_cont051（JOB-051）；c1_fdopt057（JOB-057）报告与 JSON`);
s.addNotes("这一页讲第二阶段主线的推进过程。左图是 JOB-031 的 C1 取向距离扫描：固定单体几何，非 CP 相互作用能在 2.6–4.2 Å 全部为正，说明这个固定取向线上是排斥主导，不能沿它找到极小值，必须做全自由度优化。全自由度优化确实找到低于分离单体约 2.26 kcal/mol 的接触构型，但强调三个限定：未经 CP 校正、含片段变形、不是结合自由能。之后是漫长的收敛攻坚：JOB-048 到 050 多次优化未达门限，JOB-051 续算首次得到 max|g| = 8.203×10⁻⁶ 过门限几何并经独立复核，登记首个驻点候选；JOB-057 从一维低点出发仅 5 步优化得到第二个候选 6.847×10⁻⁶，能量比 051 低 4.5×10⁻⁷ Eh。最后的边界声明要念：候选不等于稳定复合物，更不涉及水处理效果。");

// =================================================================
// Slide 4 — 软模与候选性质核查（浅色，图 + 证据链）
// =================================================================
s = p.addSlide();
s.background = { color: BG };
header(s, "04", "驻点候选性质核查：软模负曲率的独立验证");

const cW = 5.5, cH = cW / (1259 / 854);
s.addImage({ path: `${ASSETS}/scan055_056_slide.png`, x: M, y: 1.32, w: cW, h: cH });
s.addText([
  ...fx("沿固定方向 q 的直线扫描（055 中心+4 点，056 再细化 3 点）：方向导数 a(t)=g·q 在 t = 0.045–0.050 Bohr 变号，最低采样能量在 t = 0.050。但该点横向梯度大（max|g| ≈ 1.9×10^{-4}）——直线低点 ≠ 全自由度极小值。"),
], { x: M, y: 1.32 + cH + 0.1, w: cW + 0.3, h: 1.55, fontFace: F, fontSize: 12.5,
  color: MUTED, margin: 0, valign: "top" });

const rx4 = 6.6, rw4 = W - rx4 - 0.5;
const rows4 = [
  ["052", "@ 051 候选：内部电子稳定性 stable_i = True；合成 Hessian 预测 1 个内部负模 −72.83 cm^{-1}（H 原子主导）。"],
  ["053", "负模方向双步长独立差分：kE、kg 四个值全部为负 → 负曲率符号获独立求值支持；但与解析矩阵量级差 ~2.8 倍（历史遗留，未解决）。"],
  ["058", "@ 057 候选：42 位移完整 FD Hessian 同样预测 1 个内部负模 −47.27 cm^{-1}（单步长，频率收敛性未验证）；stable_i = True。"],
  ["059", "058 负模方向双步长核查：kE / kg / 保存矩阵五条路线全为负、互差 ≤ 0.76% → 该方向负曲率获独立双步长支持。"],
];
rows4.forEach(([chip, desc], i) => {
  s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: rx4, y: 1.32 + i * 1.06 + 0.02, w: 0.62, h: 0.34,
    fill: { color: PRIMARY }, rectRadius: 0.06 });
  s.addText(chip, { x: rx4, y: 1.32 + i * 1.06 + 0.02, w: 0.62, h: 0.34, align: "center",
    valign: "middle", fontFace: F, fontSize: 12, bold: true, color: "FFFFFF", margin: 0 });
  s.addText(fx(desc), { x: rx4 + 0.78, y: 1.32 + i * 1.06, w: rw4 - 0.78, h: 1.0,
    fontFace: F, fontSize: 12.5, color: TEXT, margin: 0, valign: "top" });
});
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: rx4, y: 5.62, w: rw4, h: 1.2,
  fill: { color: DARK }, rectRadius: 0.08 });
s.addText([
  { text: "登记结论", options: { bold: true, color: ACCENT, fontSize: 13.5, breakLine: true } },
  { text: "两个候选（051 / 057）电子稳定性均为 True、振动曲率核查均未通过 → 均不登记极小值；当前尚无 TS、IRC 或能垒。", options: { color: LIGHT, fontSize: 12.5 } },
], { x: rx4 + 0.2, y: 5.62, w: rw4 - 0.4, h: 1.2, fontFace: F, margin: 0,
  valign: "middle", paraSpaceAfter: 4 });
sourceLine(s, `${SRC}/c1_cand_check052、c1_negmode_dir053、c1_dir_scan055、c1_refine056、c1_fdhess058_resume01、c1_negdir059 各批报告`);
s.addNotes("这一页是当前最核心也最谨慎的部分：两个驻点候选到底是不是极小值。右上是 052 的性质核查：电子内部稳定性通过，但合成 Hessian 在 051 候选几何预测一个 −72.83 cm⁻¹ 的内部负模。053 沿该负模方向做双步长独立差分，kE、kg 全为负，说明负曲率符号是真的；但量级与解析矩阵差约 2.8 倍，这个差异沿袭自 045/046，尚未解决。左图是 055/056 的固定方向扫描：方向导数在 0.045–0.050 Bohr 变号，插值零点约 0.0484，最低采样能量在 t=0.050——但该点横向梯度大，直线低点不等于全自由度极小值。057 从这个低点全自由度优化得到第二个候选后，058 用 42 个位移做完整 FD Hessian，同样预测一个 −47.27 cm⁻¹ 负模；059 再用双步长独立核查，五条路线全负、互差 ≤0.76%。登记结论：两个候选电子稳定性都通过、振动曲率都没通过，都不登记极小值；TS、IRC、能垒都还没有。");

// =================================================================
// Slide 5 — 实验准备的真实状态（浅色）
// =================================================================
s = p.addSlide();
s.background = { color: BG };
header(s, "05", "实验准备的真实状态：框架已有，执行未登记");

s.addText("已登记（计算侧整理）", { x: M, y: 1.25, w: 5.9, h: 0.38, fontFace: F,
  fontSize: 15.5, bold: true, color: PRIMARY, margin: 0 });
s.addText([
  { text: "目标污染物方向：养殖水体氨氮（NH₃ / NH₄⁺）", options: { bullet: bu(), breakLine: true } },
  { text: "已形成需向实验端确认的参数清单", options: { bullet: bu(), breakLine: true } },
  { text: "已提出建议对照组与检测指标框架", options: { bullet: bu() } },
], { x: M, y: 1.68, w: 5.9, h: 1.35, fontFace: F, fontSize: 13.5, color: TEXT,
  margin: 0, paraSpaceAfter: 7, valign: "top" });

s.addText("尚未登记 / 尚未确认", { x: M, y: 3.18, w: 5.9, h: 0.38, fontFace: F,
  fontSize: 15.5, bold: true, color: ACCENT, margin: 0 });
s.addText([
  { text: "正式养殖实验执行数据（项目记录中无）", options: { bullet: bu(), breakLine: true } },
  { text: "水体类型与组成 · pH · 温度 · 氨氮浓度", options: { bullet: bu(), breakLine: true } },
  { text: "UV 波长 / 功率 · 臭氧剂量 · 微纳米气泡条件", options: { bullet: bu(), breakLine: true } },
  { text: "产物与活性物种检测结果", options: { bullet: bu() } },
], { x: M, y: 3.61, w: 5.9, h: 1.8, fontFace: F, fontSize: 13.5, color: TEXT,
  margin: 0, paraSpaceAfter: 7, valign: "top" });

s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: M, y: 5.6, w: 5.9, h: 1.22,
  fill: { color: TINT }, rectRadius: 0.08 });
s.addText([
  { text: "汇报统一表述：", options: { bold: true, color: PRIMARY } },
  { text: "“实验方案框架已提出，具体条件待确认，正式实验尚未在本项目记录中登记。”", options: { color: TEXT, italic: true } },
], { x: M + 0.2, y: 5.6, w: 5.5, h: 1.22, fontFace: F, fontSize: 13, margin: 0, valign: "middle" });

const ex = 6.9, ew = W - ex - 0.5;
s.addText("建议第一轮：小规模水样摸底（合成水或已有养殖水样）", { x: ex, y: 1.25, w: ew, h: 0.4,
  fontFace: F, fontSize: 15.5, bold: true, color: PRIMARY, margin: 0 });
const groups = ["空白", "O₃", "UV", "O₃+UV", "O₃-MNB", "O₃-MNB+UV"];
const gw = (ew - 0.5) / 3;
groups.forEach((grp, i) => {
  const gx = ex + (i % 3) * (gw + 0.25), gy = 1.78 + Math.floor(i / 3) * 0.78;
  const hot = i >= 4;
  s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: gx, y: gy, w: gw, h: 0.6,
    fill: { color: hot ? ACCENT : TINT2 }, line: { color: hot ? ACCENT : HAIR, width: 1 },
    rectRadius: 0.08 });
  s.addText(grp, { x: gx, y: gy, w: gw, h: 0.6, align: "center", valign: "middle",
    fontFace: F, fontSize: 13.5, bold: true, color: hot ? "FFFFFF" : PRIMARY, margin: 0 });
});
s.addText("6 组对照覆盖臭氧、紫外、微纳米气泡三个因素的单独与联合作用（橙色为 MNB 组）", {
  x: ex, y: 3.28, w: ew, h: 0.35, fontFace: F, fontSize: 12, color: MUTED, margin: 0 });

s.addText("必测指标", { x: ex, y: 3.85, w: ew, h: 0.35, fontFace: F, fontSize: 13.5, bold: true, color: PRIMARY, margin: 0 });
s.addText(fx("NH_{4}^{+} / NH_{3} · NO_{2}^{-} · NO_{3}^{-} · pH · 温度 · 溶解臭氧（含时间序列采样）"), {
  x: ex, y: 4.22, w: ew, h: 0.62, fontFace: F, fontSize: 13, color: TEXT, margin: 0, valign: "top" });
s.addText("可选证据", { x: ex, y: 4.92, w: ew, h: 0.35, fontFace: F, fontSize: 13.5, bold: true, color: PRIMARY, margin: 0 });
s.addText("EPR / ROS 间接证据 · 过氧化物 —— 用于活性物种路径佐证", {
  x: ex, y: 5.29, w: ew, h: 0.6, fontFace: F, fontSize: 13, color: TEXT, margin: 0, valign: "top" });
s.addShape(p.shapes.LINE, { x: ex, y: 5.95, w: ew, h: 0, line: { color: HAIR, width: 0.75 } });
s.addText("注：先做可控的水样验证，不把活鱼 / 病毒试验作为当前第一步。", {
  x: ex, y: 6.05, w: ew, h: 0.6, fontFace: F, fontSize: 12.5, color: MUTED, margin: 0, valign: "top" });
sourceLine(s, "notes/presentation_brief_6slides_2026-09-09.md；实验参数清单与对照建议（项目已登记记录）");
s.addNotes("实验线的真实状态要如实讲。已登记的只有三件事：目标污染物方向确定为养殖水体氨氮、需要向实验端确认的参数清单、以及建议对照组和检测指标框架。正式养殖实验执行数据在项目记录里没有；水体类型、pH、温度、氨氮浓度、UV 波长功率、臭氧剂量、微纳米气泡条件、产物检测这些关键条件都还没确认，所以汇报口径统一为：方案框架已提出、具体条件待确认、正式实验尚未登记。右侧是我们建议的第一轮小规模摸底：六组对照——空白、O₃、UV、O₃+UV、O₃-MNB、O₃-MNB+UV——把臭氧、紫外、微纳米气泡三个因素的单独和联合作用拆开；必测氨氮、亚硝酸盐、硝酸盐、pH、温度、溶解臭氧并做时间序列；EPR/ROS 作为可选佐证。明确第一步不做活鱼和病毒试验，先做合成水或已有养殖水样的可控验证。");

// =================================================================
// Slide 6 — 后续计划：双线并行（深色收尾）
// =================================================================
s = p.addSlide();
s.background = { color: DARK };
header(s, "06", "后续计划：计算 — 实验双线并行", true);

const colW6 = 6.0, colL = M, colR = M + colW6 + 0.33;
s.addText("计算线（DFT）", { x: colL, y: 1.12, w: colW6, h: 0.4, fontFace: F,
  fontSize: 16, bold: true, color: ACCENT, margin: 0 });
s.addText("实验线（水样摸底）", { x: colR, y: 1.12, w: colW6, h: 0.4, fontFace: F,
  fontSize: 16, bold: true, color: "4FB3C4", margin: 0 });
const calc = [
  "细化 0.045–0.050 Bohr 变号区间：按 059 审阅路线转受限松弛",
  "核查两候选负模关系 / 是否同一势盆（不自动跟随）",
  "形成可用结论后选首条优先反应路径：TS · 频率 · IRC 连接验证",
  "水相扩展、UV 激发态、气液界面 —— 结合实验条件论证后立项",
];
const exp = [
  "向实验端收集水体、pH、UV、臭氧等关键条件",
  "条件确定即可开始 6 组对照小规模摸底（不必等 DFT 完成）",
  fx("建立氨氮去除 + NO_{2}^{-} / NO_{3}^{-} 产物时间序列"),
  "与 DFT 机制结论对照讨论，互相约束边界",
];
[calc, exp].forEach((arr, c) => {
  const x0 = c === 0 ? colL : colR;
  arr.forEach((t, i) => {
    const y = 1.58 + i * 0.78;
    s.addShape(p.shapes.OVAL, { x: x0, y: y + 0.03, w: 0.4, h: 0.4,
      fill: { color: c === 0 ? ACCENT : "175D6E" } });
    s.addText(String(i + 1), { x: x0, y: y + 0.03, w: 0.4, h: 0.4, align: "center",
      valign: "middle", fontFace: F, fontSize: 13, bold: true, color: "FFFFFF", margin: 0 });
    s.addText(t, { x: x0 + 0.58, y: y - 0.04, w: colW6 - 0.58, h: 0.72, fontFace: F,
      fontSize: 13, color: LIGHT, margin: 0, valign: "top" });
  });
});

s.addShape(p.shapes.LINE, { x: M, y: 4.78, w: W - 2 * M, h: 0, line: { color: "1D5A6B", width: 0.75 } });
s.addText([
  { text: "“机制被实验支持”至少需要：", options: { bold: true, color: ACCENT } },
  ...fx("DFT 给出一个可复核的关键反应步骤或候选路径 ｜ 实验有氨氮去除与 NO_{2}^{-}/NO_{3}^{-} 时间序列 ｜ 有 O₃、UV、MNB 及 ROS 对照。", { color: LIGHT }),
], { x: M, y: 4.9, w: W - 2 * M, h: 0.52, fontFace: F, fontSize: 12.5, margin: 0, valign: "top" });

s.addText("本次汇报三句话", { x: M, y: 5.52, w: 4, h: 0.35, fontFace: F, fontSize: 13.5,
  bold: true, color: ACCENT, margin: 0 });
const concl = [
  fx("平台与单体基线已就绪；NH_{3}+O_{3} 推进到“驻点候选性质核查”，尚无已验收极小值与能垒。"),
  "软模负曲率获独立差分支持，但解析与差分量级差异未解决 —— 结论保持受限的气相模型结论。",
  "实验可先做小规模可控水样摸底；正式养殖实验尚未登记，汇报中不写“已完成”。",
];
concl.forEach((t, i) => {
  s.addText([
    { text: `${i + 1}. `, options: { bold: true, color: ACCENT } },
    ...(typeof t === "string" ? [{ text: t, options: { color: LIGHT } }] : t),
  ], { x: M, y: 5.92 + i * 0.35, w: W - 2 * M, h: 0.33, fontFace: F, fontSize: 12.5, margin: 0, valign: "top" });
});
sourceLine(s, "PROJECT_STATUS.md（2026-09-10）；notes/job059_commander_review_2026-09-10.md", true);
s.addNotes("收尾讲两条线的后续。计算线：按 059 总指挥审阅，下一步对 0.045–0.050 Bohr 变号区间转受限松弛，不再重复全 Hessian；之后核查两个候选的负模是否同一软坐标、是否同一势盆；形成可用结论后再选首条优先反应路径做 TS、频率、IRC；水相、UV 激发态和气液界面要结合实验条件论证后另行立项。实验线：先收集关键条件，条件一确定就可以开始六组对照摸底——不必等 DFT 完成全部反应网络，摸底用于建立边界和观测现象；然后建立氨氮去除和产物时间序列，与 DFT 对照讨论。强调判断标准：只有 DFT 可复核步骤 + 实验时间序列 + 完整对照三者齐备，才能说机制被实验支持。最后用三句话收尾：基线就绪、候选核查中、尚无极小值能垒；负曲率独立支持但量级差异未解决；实验先摸底、正式实验未登记。");

// ---------- write ----------
p.writeFile({ fileName: "results/discussion/组会汇报_DFT阶段_臭氧MNB-UV_2026-09-10.pptx" })
  .then(() => console.log("PPTX written"));
