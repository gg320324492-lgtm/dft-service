# -*- coding: utf-8 -*-
# 组会汇报 6 页 — 克隆杜同贺 2026-6-18 模板版式 + 每页"通俗地说"大白话注释条
import copy
import re as _re
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.oxml.ns import qn
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

TPL = r'C:\Users\pc\WPSDrive\757644119\WPS云盘\综合文件夹\微纳米气泡\汇报\2026.6.18汇报\2026-6-18-杜同贺.pptx'
OUT = r'results/discussion/组会汇报_DFT阶段_臭氧MNB-UV_2026-09-10.pptx'
ASSETS = r'results/discussion/assets'
SRC = r'run_artifacts/02_nh3o3_reference'

NAVY = '002060'
COVER_NAVY = '000066'
RED = 'C00000'
BLACK = '000000'
BAND_BLUE = '5B9BD5'
LIGHT_BLUE = 'DCE9F5'
PRIMARY = '002060'
HAIR = 'C9DAE0'
MUTED = '5B7080'
TINT2 = 'F5FAFB'

prs = Presentation(TPL)

# ---------- 克隆工具 ----------
def clone_slide(src):
    dst = prs.slides.add_slide(src.slide_layout)
    for sh in list(dst.shapes):
        sh._element.getparent().remove(sh._element)
    idmap = {}
    for rId, rel in src.part.rels.items():
        if rel.reltype.endswith('/slideLayout') or rel.reltype.endswith('/notesSlide'):
            continue
        if rel.is_external:
            idmap[rId] = dst.part.rels.get_or_add_ext_rel(rel.reltype, rel.target_ref)
        else:
            idmap[rId] = dst.part.relate_to(rel.target_part, rel.reltype)
    for sh in src.shapes:
        el = copy.deepcopy(sh._element)
        for e in el.iter():
            for a in ('embed', 'link', 'id'):
                k = qn('r:' + a)
                v = e.get(k)
                if v and v in idmap:
                    e.set(k, idmap[v])
        dst.shapes._spTree.append(el)
    return dst

def rm_shapes(slide, keep_names):
    for sh in list(slide.shapes):
        if sh.name not in keep_names:
            sh._element.getparent().remove(sh._element)

def find(slide, name):
    for sh in slide.shapes:
        if sh.name == name:
            return sh

def style_run(r, size=None, bold=True, color=BLACK, latin='Times New Roman', ea='微软雅黑', baseline=None):
    if size is not None:
        r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = RGBColor.from_string(color)
    r.font.name = latin
    rPr = r._r.get_or_add_rPr()
    if baseline is not None:
        rPr.set('baseline', str(baseline))
    latin_el = rPr.find(qn('a:latin'))
    ea_el = rPr.find(qn('a:ea'))
    if ea_el is None:
        ea_el = rPr.makeelement(qn('a:ea'), {})
        if latin_el is not None:
            latin_el.addnext(ea_el)
        else:
            rPr.append(ea_el)
    ea_el.set('typeface', ea)

def fx_parse(s):
    pat = _re.compile(r'([_^])\{([^}]*)\}')
    out, last = [], 0
    for m in pat.finditer(s):
        if m.start() > last:
            out.append((s[last:m.start()], None))
        out.append((m.group(2), m.group(1)))
        last = m.end()
    if last < len(s):
        out.append((s[last:], None))
    return out

def add_runs(para, text, **kw):
    for t, kind in fx_parse(text):
        r = para.add_run()
        r.text = t
        style_run(r, baseline=(-25000 if kind == '_' else 30000 if kind == '^' else None), **kw)

def add_textbox(slide, x, y, w, h, paras, wrap=True, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = wrap
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    first = True
    for p in paras:
        para = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        para.alignment = p.get('align', PP_ALIGN.LEFT)
        para.space_after = Pt(p.get('space_after', 4))
        if p.get('line'):
            para.line_spacing = p['line']
        for t, kw in p['runs']:
            add_runs(para, t, **kw)
    return tb

def retext(shape, paras):
    tf = shape.text_frame
    for p in list(tf.paragraphs[1:]):
        p._p.getparent().remove(p._p)
    for r in list(tf.paragraphs[0].runs):
        r._r.getparent().remove(r._r)
    first = True
    for p in paras:
        para = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        para.alignment = p.get('align', para.alignment)
        para.space_after = Pt(p.get('space_after', 4))
        if p.get('line'):
            para.line_spacing = p['line']
        for t, kw in p['runs']:
            add_runs(para, t, **kw)

def plain_band(slide, x, y, w, text, h=0.66):
    """大白话注释条：浅蓝底圆角矩形 + 红色引导词 + 藏青内容"""
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    box.fill.solid()
    box.fill.fore_color.rgb = RGBColor.from_string(LIGHT_BLUE)
    box.line.fill.background()
    box.shadow.inherit = False
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.12)
    tf.margin_top = tf.margin_bottom = Inches(0.05)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    para = tf.paragraphs[0]
    add_runs(para, text, size=12, bold=True, color=NAVY)

BODY = dict(size=12, bold=True, color=BLACK)
SUBHEAD = dict(size=14, bold=True, color=NAVY)
TITLE = dict(size=20, bold=True, color=NAVY)

# ---------- 克隆 6 页 ----------
tpl_cover = prs.slides[0]
tpl_text = prs.slides[4]
tpl_fig = prs.slides[13]

c1 = clone_slide(tpl_cover)
c1b = clone_slide(tpl_text)
c2 = clone_slide(tpl_text)
c1c = clone_slide(tpl_text)
c1d = clone_slide(tpl_text)
c3 = clone_slide(tpl_fig)
c4 = clone_slide(tpl_fig)
c1e = clone_slide(tpl_text)
c5 = clone_slide(tpl_text)
c6 = clone_slide(tpl_fig)

for i in range(24, -1, -1):
    sld = list(prs.slides._sldIdLst)[i]
    prs.part.drop_rel(sld.get(qn('r:id')))
    prs.slides._sldIdLst.remove(sld)

# ============ P1 封面 ============
retext(find(c1, 'Rectangle 2'), [
    {'runs': [('臭氧微纳米气泡/UV协同处理养殖水体：', dict(size=28, bold=True, color=COVER_NAVY))],
     'align': PP_ALIGN.CENTER, 'space_after': 10},
    {'runs': [('氨氮去除分子机制的DFT模拟阶段汇报', dict(size=28, bold=True, color=COVER_NAVY))],
     'align': PP_ALIGN.CENTER},
])
retext(find(c1, '文本框 2'), [
    {'runs': [('杜同贺', dict(size=18, bold=True))], 'align': PP_ALIGN.CENTER, 'space_after': 6},
    {'runs': [('天津大学 环境学院', dict(size=16, bold=True))], 'align': PP_ALIGN.CENTER, 'space_after': 6},
    {'runs': [('2026年9月10日', dict(size=16, bold=True))], 'align': PP_ALIGN.CENTER},
])
c1.notes_slide.notes_text_frame.text = (
    '开场（30 秒）：各位老师、同学好，今天汇报臭氧微纳米气泡＋紫外协同处理养殖水体课题里的理论计算部分——'
    '氨氮去除分子机制的 DFT 模拟阶段汇报。汇报共 10 页：研究背景 → 计算体系与可靠性 → 四阶段计划（①基线、②主攻、③④后续） → 构型与核查进展 → 实验对接 → 后续计划。'
    '计算体系与可靠性 → NH₃＋O₃ 接触构型进展 → 候选性质核查 → 实验对接状态 → 后续计划。'
    '本汇报只用已登记的计算证据，不含未登记的实验结果。')

# ============ P1b 研究背景与总体路线（独立一页，含四阶段） ============
rm_shapes(c1b, {'矩形 10', '图片 2', '直接连接符 1', 'Text Box 2'})
retext(find(c1b, 'Text Box 2'), [
    {'runs': [('研究背景与总体路线：为什么做分子模拟', TITLE)]},
])
bg_lines = [
    '背景：养殖水体氨氮超标，对鱼有毒性，也是水产养殖水质管理的核心指标之一。',
    '工艺思路：以臭氧微纳米气泡与紫外联用的方式去除氨氮。',
    '关键问题：它为什么有效？臭氧和氨氮到底是怎么反应的？',
    '实验的局限：只能测得去除率，难以直接观测分子层面的反应过程。',
]
add_textbox(c1b, 0.35, 1.15, 9.3, 1.6,
            [{'runs': [('• ' + t, BODY)], 'space_after': 8} for t in bg_lines])

add_textbox(c1b, 0.35, 3.0, 4, 0.28, [{'runs': [('研究目标链条', dict(size=13, bold=True, color=NAVY))]}])
chain2 = [('O₃ / UV 活化', 1.85), ('活性氧 ROS', 1.6), ('NH₃ / NH₄⁺ 氧化', 2.05), ('NH₂OH · NO₂⁻ · NO₃⁻', 2.9)]
cx2 = 0.35
for i, (t2, w2) in enumerate(chain2):
    chip2 = c1b.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(cx2), Inches(3.4), Inches(w2), Inches(0.55))
    chip2.fill.solid()
    chip2.fill.fore_color.rgb = RGBColor.from_string(BAND_BLUE)
    chip2.line.fill.background()
    chip2.shadow.inherit = False
    tf2 = chip2.text_frame
    tf2.margin_left = tf2.margin_right = tf2.margin_top = tf2.margin_bottom = 0
    para2 = tf2.paragraphs[0]
    para2.alignment = PP_ALIGN.CENTER
    add_runs(para2, t2, size=12.5, bold=True, color='FFFFFF')
    cx2 += w2
    if i < 3:
        add_textbox(c1b, cx2 - 0.02, 3.44, 0.28, 0.35,
                    [{'runs': [('→', dict(size=13, bold=True, color=NAVY))], 'align': PP_ALIGN.CENTER}])
        cx2 += 0.28

strip2 = c1b.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.35), Inches(4.12), Inches(9.3), Inches(0.62))
strip2.fill.solid()
strip2.fill.fore_color.rgb = RGBColor.from_string(LIGHT_BLUE)
strip2.line.fill.background()
strip2.shadow.inherit = False
tfs2 = strip2.text_frame
tfs2.word_wrap = True
tfs2.margin_left = tfs2.margin_right = Inches(0.12)
tfs2.margin_top = tfs2.margin_bottom = Inches(0.03)
tfs2.vertical_anchor = MSO_ANCHOR.MIDDLE
paraS2 = tfs2.paragraphs[0]
paraS2.alignment = PP_ALIGN.CENTER
add_runs(paraS2, '臭氧溶入水体并经紫外分解产生活性氧 → ·OH 等强氧化性物种将氨氮逐步氧化 → 依次生成 NH₂OH、NO₂⁻、NO₃⁻，可由水质检测直接监测',
         size=11.5, bold=True, color=NAVY)
plain_band(c1b, 0.35, 4.86, 9.3,
           '实验测得处理效果，分子模拟解释背后机理——说明臭氧与氨氮在分子层面如何反应，'
           '为工艺优化提供依据。', h=0.6)
c1b.notes_slide.notes_text_frame.text = (
    '这一页回答三个问题（大白话）：① 为什么值得做——氨氮对鱼有毒、是水质管理核心指标，'
    '工艺上要用"臭氧微纳米气泡＋紫外"联用处理，但它为什么有效、分子层面怎么反应，实验只能看到"去掉了多少"；'
    '② 课题定位——在电脑中构建分子级模拟，用 DFT 看清 O₃/UV 产生活性氧、氧化氨氮的第一步，为实验提供机理依据；'
    '③ 走到哪了——对照导师批准的四阶段计划逐条过：①臭氧水合基线已建立（水相仍有未验收项）；'
    '②目标物反应模型是当前主线（NH₃＋O₃ 气相接触构型）；③反应路径与能垒尚未开展；'
    '④UV 激发态与气液界面待实验条件论证后立项。所以项目处于第二阶段早期。四阶段的具体任务，下一页逐条展开。')

# ============ P1c 四阶段计划：每个阶段的任务是干什么的（独立一页） ============
rm_shapes(c1c, {'矩形 10', '图片 2', '直接连接符 1', 'Text Box 2'})
retext(find(c1c, 'Text Box 2'), [
    {'runs': [('四阶段计划（1/3）：① 臭氧水合基线——单体基线数据', TITLE)]},
])
# ---- 第 3 页（四阶段 1/2）：阶段① + 单体基线数据表 ----
add_textbox(c1c, 0.35, 1.1, 4.0, 0.32, [
    {'runs': [('① 臭氧水合基线', dict(size=13.5, bold=True, color=PRIMARY)),
              ('　已建立', dict(size=11.5, bold=True, color=MUTED))]},
])
add_textbox(c1c, 0.35, 1.52, 4.0, 2.2, [
    {'runs': [('任务：', dict(size=12, bold=True, color=NAVY)),
              ('固定计算方法与验收标准，算准 H₂O、O₃、O₂ 在气相和 SMD 水相的结构、能量与频率。', BODY)], 'space_after': 5},
    {'runs': [('目的：', dict(size=12, bold=True, color=NAVY)),
              ('为后续所有反应计算提供参照点与统一方法口径。', BODY)], 'space_after': 5},
    {'runs': [('成果：', dict(size=12, bold=True, color=NAVY)),
              ('已完成 6 套单体基线计算，结构、能量与频率全部验收（0 虚频，即结构稳定）。', BODY)]},
])
add_textbox(c1c, 4.6, 1.1, 5.23, 0.3, [
    {'runs': [('6 套单体基线数据（JOB-001~006）', dict(size=13, bold=True, color=NAVY))]},
])
mt = c1c.shapes.add_table(7, 5, Inches(4.6), Inches(1.5), Inches(5.23), Inches(2.38)).table
for c, w in enumerate((0.75, 1.05, 1.35, 1.58, 0.5)):
    mt.columns[c].width = Inches(w)
mrows = [
    ['单体', '相/溶剂', 'E_total (Eh)', '频率 (cm⁻¹)', '虚频'],
    ['H₂O', '气相', '−76.437668', '1619.6/3875.8/3985.1', '0'],
    ['H₂O', 'SMD 水', '−76.450895', '1595.0/3848.1/3930.8', '0'],
    ['O₃', '气相', '−225.433903', '784.6/1329.1/1340.3', '0'],
    ['O₃', 'SMD 水', '−225.439360', '786.2/1267.8/1348.3', '0'],
    ['O₂', '气相', '−150.340070', '1715.8', '0'],
    ['O₂', 'SMD 水', '−150.340720', '1717.7', '0'],
]
for ri, row in enumerate(mrows):
    mt.rows[ri].height = Inches(0.34)
    for ci, val in enumerate(row):
        cell = mt.cell(ri, ci)
        cell.margin_left = cell.margin_right = Inches(0.03)
        cell.margin_top = cell.margin_bottom = Inches(0.01)
        para = cell.text_frame.paragraphs[0]
        para.alignment = PP_ALIGN.CENTER
        add_runs(para, val, size=10.5, bold=True, color=('FFFFFF' if ri == 0 else BLACK))
add_textbox(c1c, 0.35, 4.15, 4.0, 1.2, [
    {'runs': [('说明：', dict(size=12, bold=True, color=NAVY)),
              ('H₂O（SMD 水）梯度略高于 1×10⁻⁵ 门限，单体统一以 0 虚频频率结果登记验收；水相驻点、热化学为遗留未验收项。',
               BODY)]},
])
add_textbox(c1c, 0.35, 5.15, 9.3, 0.35, [
    {'runs': [('这些能量就是后续反应能的参照零点——尺子造准了，后面的数才可信。', dict(size=12, bold=True, color=NAVY))]},
])
c1c.notes_slide.notes_text_frame.text = (
    '本页讲第一阶段（已完成）：任务是把尺子造准——固定计算方法和验收标准，'
    '算准水、臭氧、氧气在气相和模拟水相的结构、能量、频率。'
    '右表 6 套单体基线全部 0 虚频（结构稳定），能量即后续反应能的参照零点；'
    'H₂O 水相梯度略高于门限，按频率结果登记验收；水相驻点、热化学为遗留未验收项。')

# ---- 第 4 页（四阶段 2/2）：阶段②③④ ----
rm_shapes(c1d, {'矩形 10', '图片 2', '直接连接符 1', 'Text Box 2'})
retext(find(c1d, 'Text Box 2'), [
    {'runs': [('四阶段计划（2/3）：② 目标物反应模型——当前主攻', TITLE)]},
])
add_textbox(c1d, 0.35, 1.06, 9.3, 0.32, [
    {'runs': [('当前主攻：② 目标物反应模型', dict(size=14, bold=True, color=RED))]},
])
add_textbox(c1d, 0.35, 1.5, 9.3, 0.7, [
    {'runs': [('任务：', dict(size=12, bold=True, color=NAVY)),
              ('研究氨氮与 O₃、活性氧的复合物/中间体——当前是 NH₃ 与 O₃ 的稳定接触构型。', BODY)], 'space_after': 3},
    {'runs': [('目的：', dict(size=12, bold=True, color=NAVY)),
              ('搞清反应起点：两个分子如何结合、结合强度多大。', BODY)]},
])
add_textbox(c1d, 0.35, 2.5, 9.3, 1.8, [
    {'runs': [('已完成：', dict(size=12, bold=True, color=NAVY)),
              ('NH₃/O₃ 单体验收（JOB-026）；C1 取向扫描（JOB-031）；两轮全自由度优化（JOB-048/050/054）。', BODY)], 'space_after': 4},
    {'runs': [('已获 2 个驻点候选：', dict(size=12, bold=True, color=NAVY)),
              ('JOB-051（8.203×10⁻⁶）与 JOB-057（6.847×10⁻⁶），均过梯度门限并独立复核。', BODY)], 'space_after': 4},
    {'runs': [('仍在攻坚：', dict(size=12, bold=True, color=NAVY)),
              ('稳定性检验还没通过——详见下一页性质核查。', BODY)]},
])
hl = c1d.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.35), Inches(3.9), Inches(9.3), Inches(0.75))
hl.fill.solid()
hl.fill.fore_color.rgb = RGBColor.from_string(LIGHT_BLUE)
hl.line.fill.background()
hl.shadow.inherit = False
tfh = hl.text_frame
tfh.word_wrap = True
tfh.margin_left = tfh.margin_right = Inches(0.12)
tfh.margin_top = tfh.margin_bottom = Inches(0.04)
tfh.vertical_anchor = MSO_ANCHOR.MIDDLE
phl = tfh.paragraphs[0]
add_runs(phl, '为什么是主攻：', size=12, bold=True, color=RED)
add_runs(phl, '接触构型是能垒计算的初态——起点找不准，后面的能垒都无从谈起。', size=12, bold=True, color=NAVY)
plain_band(c1d, 0.35, 4.86, 9.3, '② 是当前主攻——把反应起点找稳，后面的路径与能垒才有意义；③④的任务安排见下一页。', h=0.6)
c1d.notes_slide.notes_text_frame.text = (
    '本页讲②③④三个阶段的任务与安排（大白话）：'
    '②目标物反应模型是当前主攻——让氨氮和臭氧找到稳定接触方式，已有 2 个驻点候选、稳定性核查中；'
    '③反应路径与能垒——等②出结论就做 TS/频率/IRC，能垒直接回答反应容易不容易发生；'
    '④UV 激发态与气液界面——解释紫外增效与气泡界面强化，须结合实验条件论证后立项。'
    '三个阶段都以第 3 页基线数据为参照，逐级递进。')
c1c.notes_slide.notes_text_frame.text = (
    '这一页逐条讲清四个阶段"各自要干什么、为什么要这个阶段"（大白话）：'
    '①臭氧水合基线——先把尺子造好：方法和验收标准固定下来，水、臭氧、氧气的结构和能量算准，'
    '后面所有反应能都以此为参照（已建立，水相仍有未验收项）；'
    '②目标物反应模型（当前阶段）——搞清反应起点：氨氮和臭氧怎么结合、结合多强，'
    '目前已获 2 个驻点候选、正在核查稳定性；'
    '③反应路径与能垒——算出"走哪条路、哪步最慢、要多大能量"，'
    '这直接对应工艺条件（臭氧剂量、紫外功率、处理时间）的选择，等②有结论后启动；'
    '④UV 激发态与气液界面——解释紫外增强效果与气泡界面提高利用率的原因，'
    '要结合实验条件论证后立项，不用基态小分子结果硬套。')

# ---- 第 7 页（四阶段 3/3）：③④ 任务与安排 ----
rm_shapes(c1e, {'矩形 10', '图片 2', '直接连接符 1', 'Text Box 2'})
retext(find(c1e, 'Text Box 2'), [
    {'runs': [('四阶段计划（3/3）：③ 反应路径与能垒 · ④ UV 激发态·气液界面', TITLE)]},
])
add_textbox(c1e, 0.35, 1.15, 4.4, 0.3, [
    {'runs': [('③ 反应路径与能垒　', dict(size=13.5, bold=True, color=PRIMARY)),
              ('尚未开展', dict(size=11.5, bold=True, color=MUTED))]},
])
add_textbox(c1e, 0.35, 1.58, 4.4, 2.6, [
    {'runs': [('任务：对优先反应路径做过渡态（TS）、频率与 IRC 验证，算出反应能和能垒。', BODY)], 'space_after': 8},
    {'runs': [('目的：回答反应走哪条路、哪一步最慢、需要多大能量——直接对应工艺条件选择。', BODY)], 'space_after': 8},
    {'runs': [('解读：能垒直接回答反应容易不容易发生，是工艺参数选择的理论依据。', dict(size=11.5, color=MUTED))]},
])
add_textbox(c1e, 5.15, 1.15, 4.4, 0.3, [
    {'runs': [('④ UV 激发态·气液界面　', dict(size=13.5, bold=True, color=PRIMARY)),
              ('尚未开展', dict(size=11.5, bold=True, color=MUTED))]},
])
add_textbox(c1e, 5.15, 1.58, 4.4, 2.6, [
    {'runs': [('任务：研究 UV 激发态（TD-DFT/光化学）与气液界面（微纳米气泡表面）的分子模型。', BODY)], 'space_after': 8},
    {'runs': [('目的：解释紫外增强效果与气泡界面提高利用率的原因——对应两个工艺问题。', BODY)], 'space_after': 8},
    {'runs': [('解读：须结合实验条件论证后立项，不用基态小分子结果硬套。', dict(size=11.5, color=MUTED))]},
])
plain_band(c1e, 0.35, 4.86, 9.3, '③④的产出都要以②的可用结论和实验条件输入为前提——逐级递进，不盲目抢跑。', h=0.55)
c1e.notes_slide.notes_text_frame.text = (
    '本页讲③④两个阶段的任务与安排（大白话）：'
    '③反应路径与能垒——等②出结论就做 TS/频率/IRC，能垒直接回答反应容易不容易发生；'
    '④UV 激发态与气液界面——解释紫外增效与气泡界面强化，须结合实验条件论证后立项。'
    '③④的产出都以②的可用结论和实验条件输入为前提，逐级递进。')

# ============ P2 计算体系与可靠性基础 ============
rm_shapes(c2, {'矩形 10', '图片 2', '直接连接符 1', 'Text Box 2'})
retext(find(c2, 'Text Box 2'), [
    {'runs': [('计算体系与可靠性基础：', TITLE)]},
])
add_textbox(c2, 0.35, 1.06, 4.35, 0.32,
            [{'runs': [('对照计算方法（已多轮核验）', SUBHEAD)]}])
meth = [
    '计算引擎：开源量子化学软件 PySCF 2.14.0（libxc 积分库）',
    '溶剂化模型：SMD 水相（PySCF 内置 Minnesota 溶剂库）',
    '泛函：wB97X-D + D2 色散校正（按原始文献参数实现，并经数值验证）',
    '基组 / 网格：def2-TZVP ｜ L8',
    'SCF 收敛：能量 1×10^{-12} ｜ 密度 1×10^{-9}',
    '梯度：解析完整梯度，逐求值落盘保存',
    '驻点验收门限：max|g| ≤ 1×10^{-5} Eh/Bohr',
    '交叉验证储备：Gaussian 16W（商业金标准，已部署）——本项目登记计算均由 PySCF 完成',
]
add_textbox(c2, 0.35, 1.46, 4.35, 2.9,
            [{'runs': [('• ' + t, BODY)], 'space_after': 5} for t in meth])

plain_band(c2, 0.35, 4.88, 9.35,
           '这一步相当于在电脑中搭建计算实验室：先把水、臭氧、氧气等小分子的结构与能量核算准确，'
           '后续反应研究才能建立在可靠基础之上。')
c2.notes_slide.notes_text_frame.text = (
    '先讲方法（大白话）：化学计算的引擎是国际开源量子化学软件 PySCF 2.14.0——量子化学领域的主流开源程序，'
    '正确性由开源社区和大量文献背书；积分库是 libxc，水相溶剂化用 PySCF 内置的 SMD 模型（Minnesota 溶剂库），'
    '也都是开源成熟实现。服务里另部署了 Gaussian 16W（商业"金标准"）作为今后的交叉验证参照，'
    '本项目已登记计算均由 PySCF 完成。方法统一用 wB97X-D 加项目独立实现的 D2 色散、def2-TZVP 基组、L8 网格、SCF 收敛到 1e-12/1e-9，'
    '每个求值点的解析完整梯度都落盘保存，驻点验收门限是未经投影 max|g| ≤ 1e-5 Eh/Bohr。'
    '右表是第一阶段验收的单体基线：H₂O、O₃、O₂ 气相和 SMD 水相全部 0 虚频；'
    'H₂O SMD 水梯度 1.28×10⁻⁵ 略高于门限，单体以频率结果验收登记。'
    'NH₃ 气相单体 JOB-026 完成结构、独立梯度、内部电子稳定性和频率检查，'
    '与 O₃ 单体之和 −281.998122 Eh 就是接触构型能量零点（好比"称重前先把秤校准"）。'
    '可靠性机制是计算可靠性工作，不算机理成果；方法口径是项目对照方法，不宣称复现 S13 的 B3LYP/CCSD(T) 数值。')

# ============ P3 构型探索 ============
rm_shapes(c3, {'矩形 10', '图片 2', '文本框 5'})
retext(find(c3, '文本框 5'), [
    {'runs': [('NH₃···O₃气相接触构型：从距离扫描到驻点候选', TITLE)]},
])
pic = c3.shapes.add_picture(f'{SRC}/c1_scan_formal/energy_radial_gradient.png',
                            Inches(0.3), Inches(1.1), Inches(5.7))
pic.height = int(Inches(5.7) / (1611 / 747))
add_textbox(c3, 0.3, 1.1 + 5.7 / (1611 / 747) + 0.06, 5.7, 0.62, [
    {'runs': [('JOB-031 固定单体几何 C1 取向扫描（气相）：非 CP 相互作用能在 2.6–4.2 Å 全为正'
               '（+0.4 ~ +11 kcal/mol）——该取向排斥主导，吸引构型需全自由度优化。', BODY)]}])
add_textbox(c3, 6.25, 1.06, 3.5, 0.38, [
    {'runs': [('2 个驻点候选 ', dict(size=15, bold=True, color=RED)),
              ('· 0 个已验收极小值', dict(size=12, bold=True, color=BLACK))]}])
bullets3 = [
    ('① ', '全自由度优化找到低于分离单体约 2.26 kcal/mol 的接触构型；未经 CP 校正、含片段变形——不是结合自由能。'),
    ('② ', 'JOB-051 首个驻点候选：max|g| = 8.203×10^{-6}，独立复核 PASS。'),
    ('③ ', 'JOB-057 第二候选：max|g| = 6.847×10^{-6}，复核 PASS，能量更低。'),
    ('★ ', '边界：驻点候选仅为过门限并经复核的几何，不代表稳定复合物，更不涉及结合自由能。'),
]
add_textbox(c3, 6.25, 1.55, 3.5, 3.2,
            [{'runs': [(b, dict(size=12, bold=True, color=(RED if b == '★ ' else NAVY))),
                       (t, BODY)], 'space_after': 7} for b, t in bullets3])
plain_band(c3, 0.3, 4.88, 9.7,
           '让一个氨氮分子与一个臭氧分子在电脑中逐渐靠近，考察二者能否形成稳定结合。'
           '目前已获得两种候选结合方式，但尚未证明其为能够稳定存在的结构。')
c3.notes_slide.notes_text_frame.text = (
    '这一页讲第二阶段主线的推进（大白话：先看这两个分子"合不合得来"）。'
    '左图是 JOB-031 固定取向的距离扫描：非 CP 相互作用能全为正，说明这个姿势是"顶着的"，合不拢，必须让分子自由转动寻找姿势。'
    '全自由度优化找到低于分离单体约 2.26 kcal/mol 的接触构型，但三个限定：未经 CP 校正、含片段变形、不是结合自由能。'
    'JOB-051 续算首次得到过梯度门限的几何并经独立复核，登记首个驻点候选；'
    'JOB-057 仅 5 步优化得到第二个候选，能量更低。'
    '边界声明要念：候选不等于稳定复合物，更不涉及水处理效果。')

# ============ P4 性质核查 ============
rm_shapes(c4, {'矩形 10', '图片 2', '文本框 5'})
retext(find(c4, '文本框 5'), [
    {'runs': [('驻点候选性质核查：软模负曲率的独立验证', TITLE)]},
])
cw = 4.1
pic = c4.shapes.add_picture(f'{ASSETS}/scan055_056_slide.png', Inches(0.3), Inches(1.1), Inches(cw))
pic.height = int(Inches(cw) / (1259 / 854))
add_textbox(c4, 0.3, 1.1 + cw / (1259 / 854) + 0.05, cw, 0.95, [
    {'runs': [('固定方向 q 直线扫描（055 中心+4 点，056 细化 3 点）：方向导数 a(t)=g·q 在 '
               '0.045–0.050 Bohr 变号；直线低点 ≠ 全自由度极小值。', BODY)]}])
bullets4 = [
    ('052：', '@ 051 候选：电子稳定性 True；合成 Hessian 预测负模 −72.83 cm^{-1}。'),
    ('053：', '双步长差分全为负 → 负曲率获独立支持；与解析矩阵量级差 ~2.8 倍（未解决）。'),
    ('058：', '@ 057 候选：42 位移 FD Hessian 亦预测负模 −47.27 cm^{-1}（单步长）。'),
    ('059：', '双步长核查：五条路线全为负、互差 ≤ 0.76% → 独立支持。'),
    ('登记：', '两候选电子稳定性均 True、振动曲率均未通过 → 均不登记极小值；尚无 TS、IRC、能垒。'),
]
add_textbox(c4, 4.65, 1.1, 4.85, 3.6,
            [{'runs': [(b, dict(size=12, bold=True, color=(RED if b == '登记：' else NAVY))),
                       (t, BODY)], 'space_after': 7} for b, t in bullets4])
plain_band(c4, 0.3, 4.88, 9.7,
           '候选结构需通过振动测试：真正稳定的结构受轻微扰动后会回到原位；'
           '这两个候选受扰后沿同一方向偏移，说明尚不稳定——该判断已用两种独立方法交叉确认。')
c4.notes_slide.notes_text_frame.text = (
    '这一页最核心也最谨慎（大白话：给候选结构做"体检"）。'
    '052：电子内部稳定性通过，但合成 Hessian 在 051 候选几何预测 −72.83 cm⁻¹ 内部负模（有个方向是"下坡"的）。'
    '053 沿该方向双步长独立差分 kE、kg 全为负，负曲率符号是真的，但量级与解析矩阵差约 2.8 倍（045/046 遗留，未解决）。'
    '左图是 055/056 固定方向扫描：方向导数在 0.045–0.050 Bohr 变号，插值零点约 0.0484，'
    '但直线低点横向梯度大，不等于全自由度极小值。058 对第二候选同样预测 −47.27 cm⁻¹ 负模；'
    '059 双步长独立核查五条路线全负、互差 ≤0.76%。登记结论：两个候选电子稳定性通过、振动曲率未通过，'
    '均不登记极小值；TS、IRC、能垒都还没有。')

# ============ P5 实验准备的真实状态 ============
rm_shapes(c5, {'矩形 10', '图片 2', '直接连接符 1', 'Text Box 2'})
retext(find(c5, 'Text Box 2'), [
    {'runs': [('实验准备的真实状态：框架已有，执行未登记', TITLE)]},
])
add_textbox(c5, 0.35, 1.06, 4.4, 2.6, [
    {'runs': [('已登记（计算侧整理）', SUBHEAD)], 'space_after': 4},
    {'runs': [('• 目标污染物方向：养殖水体氨氮', BODY)], 'space_after': 3},
    {'runs': [('• 需向实验端确认的参数清单', BODY)], 'space_after': 3},
    {'runs': [('• 建议对照组与检测指标框架', BODY)], 'space_after': 8},
    {'runs': [('尚未登记 / 尚未确认', dict(size=14, bold=True, color=RED))], 'space_after': 4},
    {'runs': [('• 正式养殖实验执行数据（项目记录中无）', BODY)], 'space_after': 3},
    {'runs': [('• 水体组成 · pH · 温度 · 氨氮浓度', BODY)], 'space_after': 3},
    {'runs': [('• UV 波长/功率 · 臭氧剂量 · MNB 条件 · 产物检测', BODY)], 'space_after': 3},
])
add_textbox(c5, 5.05, 1.06, 4.6, 0.32,
            [{'runs': [('建议第一轮：小规模水样摸底（6 组对照）', SUBHEAD)]}])
groups = [('空白', False), ('O₃', False), ('UV', False),
          ('O₃+UV', False), ('O₃-MNB', True), ('O₃-MNB+UV', True)]
for i, (gname, hot) in enumerate(groups):
    gx = 5.05 + (i % 3) * 1.6
    gy = 1.46 + (i // 3) * 0.6
    chip = c5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(gx), Inches(gy), Inches(1.45), Inches(0.46))
    chip.fill.solid()
    chip.fill.fore_color.rgb = RGBColor.from_string(NAVY if hot else BAND_BLUE)
    chip.line.fill.background()
    chip.shadow.inherit = False
    tf = chip.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    para = tf.paragraphs[0]
    para.alignment = PP_ALIGN.CENTER
    add_runs(para, gname, size=12, bold=True, color='FFFFFF')
add_textbox(c5, 5.05, 2.72, 4.6, 1.7, [
    {'runs': [('必测指标：', dict(size=12, bold=True, color=NAVY)),
              ('NH_{4}^{+}/NH_{3} · NO_{2}^{-} · NO_{3}^{-} · pH · 温度 · 溶解臭氧（含时间序列）', BODY)], 'space_after': 7},
    {'runs': [('可选证据：', dict(size=12, bold=True, color=NAVY)),
              ('EPR / ROS 间接证据 · 过氧化物', BODY)], 'space_after': 7},
    {'runs': [('第一步不做活鱼/病毒试验，先做合成水或已有养殖水样的可控验证。', BODY)]},
])
add_textbox(c5, 0.35, 4.22, 9.35, 0.4, [
    {'runs': [('汇报统一表述：', dict(size=13, bold=True, color=RED)),
              ('实验方案框架已提出，具体条件待确认，正式实验尚未在本项目记录中登记。', BODY)]}])
plain_band(c5, 0.35, 4.88, 9.35,
           '计算侧已完成实验方案的初步设计；但水样成分、pH、紫外功率、臭氧剂量等关键条件尚未确定，'
           '正式实验尚未启动——条件与对照组确定后即可开展。')
c5.notes_slide.notes_text_frame.text = (
    '实验线的真实状态要如实讲（大白话：计算是"军师"，实验是"主力"，军师先把作战图准备好）。'
    '已登记的只有三件事：目标污染物方向确定为养殖水体氨氮、需要向实验端确认的参数清单、建议对照组和检测指标框架。'
    '正式养殖实验执行数据在项目记录里没有；水体类型、pH、温度、氨氮浓度、UV 波长功率、臭氧剂量、'
    '微纳米气泡条件、产物检测都还没确认，汇报口径统一为：方案框架已提出、具体条件待确认、正式实验尚未登记。'
    '右侧是建议的第一轮小规模摸底：六组对照把臭氧、紫外、微纳米气泡三个因素的单独和联合作用拆开；'
    '必测氨氮、亚硝酸盐、硝酸盐、pH、温度、溶解臭氧并做时间序列；EPR/ROS 作为可选佐证。'
    '第一步不做活鱼和病毒试验，先做合成水或已有养殖水样的可控验证。')

# ============ P6 后续计划 ============
rm_shapes(c6, {'矩形 10', '图片 2', '文本框 5'})
retext(find(c6, '文本框 5'), [
    {'runs': [('后续计划：计算 — 实验双线并行', TITLE)]},
])
add_textbox(c6, 0.35, 1.02, 4.5, 0.32, [{'runs': [('计算线（DFT）', dict(size=14, bold=True, color=RED))]}])
calc = [
    '① 细化 0.045–0.050 Bohr 变号区间：按 059 审阅路线转受限松弛',
    '② 核查两候选负模关系 / 是否同一势盆（不自动跟随）',
    '③ 形成可用结论后选首条优先反应路径：TS · 频率 · IRC',
    '④ 水相扩展、UV 激发态、气液界面 —— 结合实验条件论证后立项',
]
add_textbox(c6, 0.35, 1.4, 4.5, 2.2,
            [{'runs': [(t, BODY)], 'space_after': 7} for t in calc])
add_textbox(c6, 5.15, 1.02, 4.5, 0.32, [{'runs': [('实验线（水样摸底）', dict(size=14, bold=True, color=NAVY))]}])
exp = [
    '① 向实验端收集水体、pH、UV、臭氧等关键条件',
    '② 条件确定即开始 6 组对照小规模摸底（不必等 DFT 完成）',
    '③ 建立氨氮去除 + NO_{2}^{-}/NO_{3}^{-} 产物时间序列',
    '④ 与 DFT 机制结论对照讨论，互相约束边界',
]
add_textbox(c6, 5.15, 1.4, 4.5, 2.2,
            [{'runs': [(t, BODY)], 'space_after': 7} for t in exp])
add_textbox(c6, 0.35, 3.7, 9.35, 0.4, [
    {'runs': [('“机制被实验支持”至少需要：', dict(size=12, bold=True, color=RED)),
              ('DFT 可复核的关键反应步骤 ｜ 氨氮去除与 NO_{2}^{-}/NO_{3}^{-} 时间序列 ｜ O₃、UV、MNB 及 ROS 对照。', BODY)]}])
concl = [
    '1. 平台与单体基线已就绪；NH₃+O₃ 推进到“驻点候选性质核查”，尚无已验收极小值与能垒。',
    '2. 软模负曲率获独立差分支持，但解析与差分量级差异未解决 —— 结论保持受限的气相模型结论。',
    '3. 实验可先做小规模可控水样摸底；正式养殖实验尚未登记，汇报中不写“已完成”。',
]
add_textbox(c6, 0.35, 4.18, 9.35, 0.75,
            [{'runs': [(t, BODY)], 'space_after': 3} for t in concl])
plain_band(c6, 0.35, 5.0, 9.35,
           '计算继续明确反应路径与能垒，实验尽早以真实水样开展验证；'
           '二者互相校准，方可可靠确立反应机理。', h=0.5)
c6.notes_slide.notes_text_frame.text = (
    '收尾讲两条线的后续（大白话：电脑和实验两条腿走路，谁也不等谁）。'
    '计算线：按 059 总指挥审阅，下一步对 0.045–0.050 Bohr 变号区间转受限松弛，不再重复全 Hessian；'
    '之后核查两个候选的负模是否同一软坐标、是否同一势盆；形成可用结论后再选首条优先反应路径做 TS、频率、IRC；'
    '水相、UV 激发态和气液界面结合实验条件论证后另行立项。实验线：先收集关键条件，条件一确定就开始六组对照摸底；'
    '然后建立氨氮去除和产物时间序列，与 DFT 对照讨论。'
    '判断标准：DFT 可复核步骤 + 实验时间序列 + 完整对照三者齐备，才能说机制被实验支持。'
    '三句话收尾：基线就绪、候选核查中、尚无极小值能垒；负曲率独立支持但量级差异未解决；实验先摸底、正式实验未登记。')

prs.save(OUT)
# 自动替换桌面副本（此后每次重建均同步桌面最新版）
import shutil
shutil.copyfile(OUT, r'C:\Users\pc\Desktop\组会汇报_DFT阶段_臭氧MNB-UV_2026-09-10.pptx')
print('saved:', OUT, 'slides:', len(prs.slides), '+ Desktop copy updated')
