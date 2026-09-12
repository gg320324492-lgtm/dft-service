// 将组会汇报讲解稿 Markdown 转为 Word（copywriting/script 场景规范）
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Footer, PageNumber,
} = require(path.join("C:/Users/pc/AppData/Roaming/npm/node_modules", "docx"));

const SRC = "results/discussion/\u7ec4\u4f1a\u6c47\u62a5\u8bb2\u89e3\u7a3f_2026-09-10.md";
const OUT = "C:/Users/pc/Desktop/\u7ec4\u4f1a\u6c47\u62a5\u8bb2\u89e3\u7a3f_2026-09-10.docx";

const HEI = "\u9ed1\u4f53";          // SimHei
const YAHEI = "\u5fae\u8f6f\u96c5\u9ed1"; // Microsoft YaHei
const NAVY = "002060";

// ---------- 上/下标字符 → 原生 run ----------
const SUP = { "\u2070": "0", "\u00b9": "1", "\u00b2": "2", "\u00b3": "3", "\u2074": "4",
  "\u2075": "5", "\u2076": "6", "\u2077": "7", "\u2078": "8", "\u2079": "9",
  "\u207a": "+", "\u207b": "\u2212" };
const SUB = { "\u2080": "0", "\u2081": "1", "\u2082": "2", "\u2083": "3", "\u2084": "4",
  "\u2085": "5", "\u2086": "6", "\u2087": "7", "\u2088": "8", "\u2089": "9" };

function segments(text) {
  // 返回 [{text, mode: n|sup|sub}]，＋→+，合并同模式
  const out = [];
  let buf = "", mode = "n";
  const flush = () => { if (buf) out.push({ text: buf, mode }); buf = ""; };
  for (const ch of text) {
    let m = "n", v = ch;
    if (ch === "\uff0b") v = "+";
    if (SUP[ch] !== undefined) { m = "sup"; v = SUP[ch]; }
    else if (SUB[ch] !== undefined) { m = "sub"; v = SUB[ch]; }
    if (m !== mode) { flush(); mode = m; }
    buf += v;
  }
  flush();
  return out;
}

function runs(text, base) {
  const rs = [];
  for (const seg of segments(text)) {
    rs.push(new TextRun(Object.assign({}, base, { text: seg.text },
      seg.mode === "sup" ? { superScript: true } : {},
      seg.mode === "sub" ? { subScript: true } : {})));
  }
  return rs;
}

// ---------- markdown 行解析 ----------
function mdRuns(line, base) {
  // **bold** 分段
  const parts = line.split("**");
  const rs = [];
  parts.forEach((p, i) => {
    if (!p) return;
    rs.push(...runs(p, i % 2 === 1 ? Object.assign({}, base, { bold: true }) : base));
  });
  return rs;
}

const body = (t, extra) => new Paragraph(Object.assign({
  spacing: { before: 120, after: 120, line: 400 },
  children: mdRuns(t, { font: YAHEI, size: 24, color: "000000" }),
}, extra || {}));
const lead = (t) => new Paragraph({
  spacing: { before: 200, after: 80, line: 400 },
  children: mdRuns(t, { font: YAHEI, size: 24, bold: true, color: NAVY }),
});
const note = (t) => new Paragraph({
  spacing: { before: 80, after: 80, line: 320 },
  children: mdRuns(t, { font: YAHEI, size: 20, color: "666666" }),
});
const h1 = (t) => new Paragraph({
  heading: HeadingLevel.HEADING_1,
  spacing: { before: 360, after: 160, line: 360 },
  children: runs(t, { font: HEI, size: 28, bold: true, color: NAVY }),
});

const lines = fs.readFileSync(SRC, "utf8").split(/\r?\n/);
const children = [];
let titleDone = false;

for (const raw of lines) {
  const line = raw.trim();
  if (!line) continue;
  if (line === "---") continue;
  if (line.startsWith("# ") && !titleDone) {
    titleDone = true;
    children.push(new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 0, after: 120, line: 400 },
      children: runs(line.slice(2), { font: HEI, size: 36, bold: true, color: "000000" }),
    }));
    continue;
  }
  if (line.startsWith("## ")) {
    children.push(h1(line.slice(3)));
    continue;
  }
  if (line.startsWith("- ")) {
    children.push(new Paragraph({
      spacing: { before: 60, after: 60, line: 400 },
      indent: { left: 360 },
      children: mdRuns("\u2022 " + line.slice(2), { font: YAHEI, size: 24, color: "000000" }),
    }));
    continue;
  }
  if (/^(日期：|配套：)/.test(line)) {
    children.push(new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 0, after: 160, line: 320 },
      children: runs(line, { font: YAHEI, size: 21, color: "666666" }),
    }));
    continue;
  }
  if (line.startsWith("说明：")) { children.push(note(line)); continue; }
  if (line.startsWith("**这页在讲什么**") || line.startsWith("**讲解词**")) {
    children.push(lead(line));
    continue;
  }
  if (line.startsWith("**Q")) {
    children.push(new Paragraph({
      spacing: { before: 220, after: 60, line: 400 },
      children: mdRuns(line, { font: YAHEI, size: 24, bold: true, color: NAVY }),
    }));
    continue;
  }
  children.push(body(line));
}

const doc = new Document({
  styles: {
    default: { document: { run: { font: YAHEI, size: 24 } } },
  },
  sections: [{
    properties: {
      page: { margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } },
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({
            children: ["\u2014 ", PageNumber.CURRENT, " \u2014"],
            font: YAHEI, size: 18, color: "999999",
          })],
        })],
      }),
    },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(OUT, buf);
  // 自动同步项目目录副本（桌面为主副本）
  const mirror = "results/discussion/\u7ec4\u4f1a\u6c47\u62a5\u8bb2\u89e3\u7a3f_2026-09-10.docx";
  fs.copyFileSync(OUT, mirror);
  console.log("saved:", OUT, "+ project copy updated");
});
