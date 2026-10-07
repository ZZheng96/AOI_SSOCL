// AOITree 答辩 PPT 生成脚本（PptxGenJS 3.12，CommonJS，兼容 Node 12）
const path = require('path');
const PptxGenJS = require('pptxgenjs');

const IMG = path.join(__dirname, '..', '截图');
const OUT = path.join(__dirname, '..', 'AOITree答辩.pptx');

// ---------- tokens ----------
const C = {
  bg: 'FBFAF7', ink: '1F2329', muted: '646A73', faint: '9AA0A8', line: 'D9DCE1',
  surface: 'F1F2F4', accent: '1F4E9A', accentSoft: 'E3EAF5', amber: 'B7791F', amberSoft: 'F6EBD9',
  positive: '2E7D32', caution: 'B26A00', risk: 'C62828',
};
const F = { cn: 'Microsoft YaHei', num: 'Arial' };
const S = { deckTitle: 38, claim: 22, sectionLabel: 11, body: 13, small: 11, annotation: 10, source: 9 };
const W = 13.333, H = 7.5, MX = 0.6, CW = W - 2 * MX;

const pptx = new PptxGenJS();
pptx.layout = 'LAYOUT_WIDE';
pptx.title = 'AOITree 答辩';

let pageNo = 0;
const TOTAL = 20;

// ---------- primitives ----------
function token(slide, text, o) {
  slide.addText(String(text), Object.assign({
    margin: 0, wrap: false, vert: 'horz', fit: 'shrink', fontFace: F.num,
    fontSize: S.small, color: C.ink, valign: 'middle',
  }, o));
}

function txt(slide, text, o) {
  slide.addText(text, Object.assign({
    margin: 0, fontFace: F.cn, fontSize: S.body, color: C.ink, valign: 'top',
    paraSpaceAfter: 4, lineSpacingMultiple: 1.15,
  }, o));
}

function base(section, claim, source, notes) {
  const s = pptx.addSlide();
  pageNo += 1;
  s.background = { color: C.bg };
  // 页眉：章节标签 + 结论句标题 + 细线
  token(s, section, { x: MX, y: 0.38, w: 6, h: 0.28, fontFace: F.cn, fontSize: S.sectionLabel, color: C.accent, bold: true });
  token(s, 'AOITree', { x: W - MX - 2, y: 0.38, w: 2, h: 0.28, fontSize: S.sectionLabel, color: C.faint, align: 'right' });
  txt(s, claim, { x: MX, y: 0.66, w: CW, h: 0.55, fontSize: S.claim, bold: true, valign: 'middle', paraSpaceAfter: 0 });
  s.addShape(pptx.ShapeType.line, { x: MX, y: 1.27, w: CW, h: 0, line: { color: C.line, width: 0.75 } });
  // 页脚
  if (source) token(s, '来源：' + source, { x: MX, y: 7.0, w: 10.5, h: 0.24, fontFace: F.cn, fontSize: S.source, color: C.faint });
  token(s, pageNo + ' / ' + TOTAL, { x: W - MX - 1, y: 7.0, w: 1, h: 0.24, fontSize: S.source, color: C.faint, align: 'right' });
  if (notes) s.addNotes(notes);
  return s;
}
// 扁平表格：仅横线，表头浅底
function table(slide, header, rows, o) {
  const fs = o.fontSize || 12;
  const bLine = { type: 'solid', color: C.line, pt: 0.75 };
  const none = { type: 'none' };
  const hl = o.highlightRow; // 高亮行索引（数据行）
  const data = [header.map(function (h) {
    return { text: h, options: { bold: true, color: C.muted, fill: { color: C.surface }, fontSize: fs - 1,
      border: [{ type: 'solid', color: C.ink, pt: 1 }, none, bLine, none] } };
  })];
  rows.forEach(function (r, ri) {
    data.push(r.map(function (cell, ci) {
      const isObj = typeof cell === 'object';
      const opt = Object.assign({ color: C.ink, fontSize: fs, border: [none, none, bLine, none] },
        ri === hl ? { fill: { color: C.accentSoft }, bold: true } : {},
        (o.boldFirst && ci === 0) ? { bold: true } : {},
        isObj ? cell.options : {});
      return { text: isObj ? cell.text : cell, options: opt };
    }));
  });
  slide.addTable(data, {
    x: o.x, y: o.y, w: o.w, colW: o.colW, rowH: o.rowH || 0.42, fontFace: F.cn, valign: 'middle',
    margin: [3, 6, 3, 6], autoPage: false,
  });
}

// 横向条形：label | bar | value
function hbar(slide, items, o) {
  const max = o.max || 1, lw = o.labelW || 1.8, vw = o.valueW || 0.8, bh = o.barH || 0.26, gap = o.gap || 0.5;
  const bw = o.w - lw - vw - 0.2;
  items.forEach(function (it, i) {
    const y = o.y + i * gap;
    token(slide, it.label, { x: o.x, y: y, w: lw - 0.1, h: bh, fontFace: F.cn, fontSize: o.fontSize || 12, color: it.bold ? C.ink : C.muted, bold: !!it.bold });
    slide.addShape(pptx.ShapeType.rect, { x: o.x + lw, y: y, w: bw, h: bh, fill: { color: C.surface }, line: { color: C.surface, width: 0 } });
    slide.addShape(pptx.ShapeType.rect, { x: o.x + lw, y: y, w: Math.max(0.02, bw * Math.min(it.value, max) / max), h: bh, fill: { color: it.color || C.accent }, line: { color: it.color || C.accent, width: 0 } });
    token(slide, it.text || String(it.value), { x: o.x + lw + bw + 0.12, y: y, w: vw, h: bh, fontSize: o.fontSize || 12, bold: !!it.bold, color: it.bold ? C.ink : C.muted });
  });
  if (o.ref !== undefined) { // 参考线
    const rx = o.x + lw + bw * o.ref / max;
    slide.addShape(pptx.ShapeType.line, { x: rx, y: o.y - 0.12, w: 0, h: items.length * gap, line: { color: C.risk, width: 1, dashType: 'dash' } });
    token(slide, o.refLabel, { x: rx - 0.8, y: o.y - 0.38, w: 1.6, h: 0.22, fontFace: F.cn, fontSize: S.annotation, color: C.risk, align: 'center' });
  }
}

function img(slide, file, x, y, w, h, frame) {
  slide.addImage({ path: path.join(IMG, file), x: x, y: y, w: w, h: h });
  if (frame !== false) slide.addShape(pptx.ShapeType.rect, { x: x, y: y, w: w, h: h, fill: { color: 'FFFFFF', transparency: 100 }, line: { color: C.line, width: 0.75 } });
}
// 按原图比例放进框内（居中）
function imgFit(slide, file, iw, ih, x, y, bw, bh, frame) {
  const r = Math.min(bw / iw, bh / ih), w = iw * r, h = ih * r;
  img(slide, file, x + (bw - w) / 2, y + (bh - h) / 2, w, h, frame);
}

// 结论条（页底一句话讲法）
function takeaway(slide, text, y) {
  slide.addShape(pptx.ShapeType.rect, { x: MX, y: y, w: 0.06, h: 0.42, fill: { color: C.amber }, line: { color: C.amber, width: 0 } });
  txt(slide, text, { x: MX + 0.18, y: y, w: CW - 0.2, h: 0.42, fontSize: 13.5, bold: true, valign: 'middle', paraSpaceAfter: 0 });
}

function secHead(slide, text, x, y, w, color) {
  token(slide, text, { x: x, y: y, w: w, h: 0.3, fontFace: F.cn, fontSize: 13, bold: true, color: color || C.accent });
}
// ================= slides =================
// 1 封面
(function () {
  const s = pptx.addSlide(); pageNo += 1;
  s.background = { color: C.bg };
  s.addShape(pptx.ShapeType.rect, { x: 0, y: 0, w: 0.18, h: H, fill: { color: C.accent }, line: { color: C.accent, width: 0 } });
  token(s, '工业外观 AI 质检系统 · 答辩', { x: 1.0, y: 1.55, w: 8, h: 0.35, fontFace: F.cn, fontSize: 14, color: C.accent, bold: true });
  token(s, 'AOITree', { x: 1.0, y: 2.0, w: 8, h: 1.0, fontSize: 54, bold: true });
  txt(s, '少样本即可上线 · 错了能教 · 教坏能退', { x: 1.0, y: 3.1, w: 10, h: 0.6, fontSize: 26, bold: true, color: C.ink });
  s.addShape(pptx.ShapeType.line, { x: 1.0, y: 4.0, w: 5.5, h: 0, line: { color: C.amber, width: 2 } });
  txt(s, [
    { text: '队伍：【待补：队名】', options: { breakLine: true } },
    { text: '成员：【待补：成员与分工】', options: { breakLine: true } },
    { text: '赛道：【待补：企业赛题“火眼工坊” / 开放赛题】' },
  ], { x: 1.0, y: 4.3, w: 8, h: 1.2, fontSize: 14, color: C.muted, paraSpaceAfter: 6 });
  token(s, '2026', { x: 1.0, y: 6.6, w: 2, h: 0.3, fontSize: 12, color: C.faint });
  s.addNotes('开场：各位评委好，我们的作品是 AOITree，一套工业外观 AI 质检系统。一句话概括：少样本即可上线、错了能教、教坏能退。');
})();

// 2 一句话定位
(function () {
  const s = base('00 定位', 'AOITree：少样本即可上线、错了能教、教坏能退', '答辩文档 §0；项目文档 3.2',
    '三个关键词对应三个能力。上线：100 正常 + 30 缺陷，十几秒准备；能教：操作员点一下误检/漏检，系统当场吸收；能退：每次更新先过锚定集考试，不及格自动回滚。');
  const cols = [
    { k: '少样本即可上线', big: '100 + 30', unit: '张', d: '100 张正常图 + 30 张缺陷图\n冻结大模型只做统计拟合，约 14s 准备完成' },
    { k: '错了能教', big: '1 次点击', unit: '', d: '操作员标记“误检 / 漏检 / 确认正确”\n老 AOI 的机器复判也能自动生成反馈' },
    { k: '教坏能退', big: '2 次回滚', unit: '', d: '每次更新先过锚定集“考试”\n错标压力测试中门控自动回滚 2 次' },
  ];
  const cw = (CW - 0.8) / 3;
  cols.forEach(function (c, i) {
    const x = MX + i * (cw + 0.4);
    s.addShape(pptx.ShapeType.line, { x: x, y: 1.85, w: cw, h: 0, line: { color: C.ink, width: 1.5 } });
    token(s, '0' + (i + 1), { x: x, y: 2.0, w: 1, h: 0.3, fontSize: 13, color: C.amber, bold: true });
    token(s, c.k, { x: x, y: 2.35, w: cw, h: 0.45, fontFace: F.cn, fontSize: 20, bold: true });
    token(s, c.big, { x: x, y: 3.0, w: cw, h: 0.9, fontFace: F.cn, fontSize: 40, bold: true, color: C.accent });
    txt(s, c.d, { x: x, y: 4.1, w: cw, h: 1.2, fontSize: 13, color: C.muted });
  });
  takeaway(s, '评委关心的不是“又一个高分模型”，而是“新产线拿来能不能用、用错了怎么办”。', 5.9);
})();

// 3 痛点
(function () {
  const s = base('01 问题', '产线要的不是更高的静态分数，而是能上线、能纠错、能追溯', '答辩文档 §1',
    '这五个痛点来自产线实际：换型频繁、缺陷杂、上线后只能等工程师调参、在线学习怕学坏、算法和软件两张皮。右列是我们对应的设计。');
  table(s, ['产线现状', '带来的问题', 'AOITree 的回应'], [
    ['换型频繁，新品类缺陷样本少', '传统深度模型要大量标注、训练数小时', '冻结通用视觉大模型，只做统计拟合；100+30 张、约 14s 准备'],
    ['缺陷种类杂：划痕、色偏、缺件、错序、尺寸', '单一算法总有盲区', '多位“专家”各管一类，分数统一后融合'],
    ['上线后误检/漏检，只能等工程师调参', '现场问题无法快速闭环', '操作员反馈即学（SSOCL），机器复判也能当反馈'],
    ['模型自我更新怕“越学越差”', '不敢在产线开启在线学习', '锚定集门控 + 版本快照 + 自动回滚，全程留痕'],
    ['算法和软件两张皮', '演示能跑，产线难用', '桌面端 + 后端服务 + 数据库一体，覆盖建单到追溯'],
  ], { x: MX, y: 1.6, w: CW, colW: [3.6, 3.2, 5.33], rowH: 0.72, fontSize: 13, boldFirst: true });
  takeaway(s, '我们的每一处设计，都围绕“能不能用”和“用错了怎么办”这两件事。', 6.25);
})();

// 4 系统全景
(function () {
  const s = base('02 系统全景', '一套系统覆盖“准备 → 投产 → 复盘”，算法与软件一体', '答辩文档 §2；项目文档 图 1',
    '前端 PySide6 桌面端，后端 FastAPI + SQLite，引擎是 DINOv2 冻结特征 + 多专家槽位 + SSOCL 受控持续学习。下面是一个工单的完整旅程。');
  imgFit(s, 'fig1_总体架构.png', 1308, 806, MX, 1.5, 7.4, 4.56);
  const x = MX + 7.75, w = CW - 7.75;
  const items = [
    ['前端', 'PySide6 桌面端，按准备期 / 投产期 / 复盘期组织页面'],
    ['后端', 'FastAPI 服务（8017）+ SQLite：检测、反馈、版本、审计'],
    ['引擎', 'DINOv2 冻结特征 + 多专家槽位 + SSOCL 受控持续学习'],
  ];
  items.forEach(function (it, i) {
    const y = 1.55 + i * 0.95;
    token(s, it[0], { x: x, y: y, w: w, h: 0.3, fontFace: F.cn, fontSize: 14, bold: true, color: C.accent });
    txt(s, it[1], { x: x, y: y + 0.33, w: w, h: 0.55, fontSize: 12.5, color: C.ink });
  });
  s.addShape(pptx.ShapeType.line, { x: x, y: 4.4, w: w, h: 0, line: { color: C.line, width: 0.75 } });
  secHead(s, '一个工单的旅程', x, 4.5, w, C.ink);
  const flow = ['建数据源', '建工单', '准备模型 100+30', '实时检测', '标注反馈', '学习提升', '回队复检', '报表与追溯'];
  txt(s, flow.map(function (f, i) { return { text: (i + 1) + '  ' + f, options: { breakLine: i < flow.length - 1 } }; }),
    { x: x, y: 4.85, w: w, h: 1.9, fontSize: 11.5, color: C.muted, paraSpaceAfter: 1, lineSpacingMultiple: 1.0 });
})();
// 5 检测流程
(function () {
  const s = base('02 系统全景', '一张图片的旅程：先统一尺子，再投票，拿不准就交给人', '答辩文档 §2；isolation.py / 融合与判定模块',
    '按顺序讲：隔离检查保证测试图不进训练；DINOv2 冻结提特征；各专家打分出热力图；CDF 校准统一尺子；共识融合；双阈值三档判定；输出可追溯。下方是闭环：反馈 → 候选版本 → 考试 → 上线或回滚。');
  const steps = [
    ['隔离检查', '测试图不许\n进入训练'],
    ['切块提特征', 'DINOv2\n主干冻结'],
    ['专家打分', '7+1 个槽位\n各出热力图'],
    ['CDF 校准', '不同量纲\n换成同一把尺'],
    ['共识融合', '等权保底\n+ 一致度加权'],
    ['双阈值判定', '正常 / 灰区 / 异常'],
    ['输出', '判定 · 缺陷框 · 类型\n主导专家 · 追溯'],
  ];
  const n = steps.length, gap = 0.16, bw = (CW - gap * (n - 1)) / n, y0 = 1.75;
  steps.forEach(function (st, i) {
    const x = MX + i * (bw + gap);
    const bc = (i >= 3 && i <= 5) ? C.amber : C.accent;
    s.addShape(pptx.ShapeType.rect, { x: x, y: y0, w: bw, h: 0.07, fill: { color: bc }, line: { width: 0, color: bc } });
    token(s, String(i + 1).padStart(2, '0'), { x: x, y: y0 + 0.2, w: bw, h: 0.3, fontSize: 13, bold: true, color: C.faint });
    token(s, st[0], { x: x, y: y0 + 0.55, w: bw, h: 0.4, fontFace: F.cn, fontSize: 15, bold: true });
    txt(s, st[1], { x: x, y: y0 + 1.0, w: bw, h: 0.9, fontSize: 11.5, color: C.muted });
  });
  // 闭环
  const ly = 4.35;
  s.addShape(pptx.ShapeType.rect, { x: MX, y: ly, w: CW, h: 1.45, fill: { color: C.surface }, line: { color: C.surface, width: 0 } });
  secHead(s, '学习闭环（SSOCL）', MX + 0.25, ly + 0.15, 4, C.ink);
  const loop = ['操作员 / 机器复判反馈', '生成候选版本', '锚定集考试', '通过 → 上线', '不通过 → 回滚'];
  const lw = (CW - 0.5) / loop.length;
  loop.forEach(function (l, i) {
    const x = MX + 0.25 + i * lw;
    const col = i === 3 ? C.positive : (i === 4 ? C.risk : C.ink);
    token(s, l, { x: x, y: ly + 0.7, w: lw - 0.35, h: 0.4, fontFace: F.cn, fontSize: 14, bold: true, color: col });
    if (i < 2) token(s, '→', { x: x + lw - 0.35, y: ly + 0.7, w: 0.3, h: 0.4, fontSize: 16, color: C.faint, align: 'center' });
    if (i === 2) token(s, '⇉', { x: x + lw - 0.35, y: ly + 0.7, w: 0.3, h: 0.4, fontSize: 16, color: C.faint, align: 'center' });
  });
  takeaway(s, '灰区（拿不准）的样本交给人看，同时也是主动学习的选样来源。', 6.2);
})();

// 6 原理：冻结主干
(function () {
  const s = base('03 原理 · 有效性', '小样本下“不训练”反而更稳更快：冻结大模型是上线的根基', 'U11 对照实验（同数据同协议）；答辩文档 §3.1',
    '比喻：DINOv2 像看过上亿张图的老师傅，我们不重新培训他，只让他看 100 张正常图，记住这条线的正常样子。检测时每一块去记忆库找最近邻，离得越远越可疑。U11 实验：冻结 0.6032，微调掉到 0.4914，从零训练 0.4235。');
  const lx = MX, lw = 5.6;
  secHead(s, '比喻', lx, 1.6, lw);
  txt(s, '请一位“看过上亿张图的老师傅”，不重新培训他，只让他看 100 张正常图，记住这条线的“正常样子”。', { x: lx, y: 1.95, w: lw, h: 0.95, fontSize: 14 });
  secHead(s, '原理', lx, 3.05, lw);
  txt(s, [
    { text: '1  正常图切块 → DINOv2 提特征 → 存进记忆库（coreset）', options: { breakLine: true } },
    { text: '2  新图每一块去记忆库找最近邻', options: { breakLine: true } },
    { text: '3  离得越远越可疑（sem 槽位，借鉴 PatchCore）' },
  ], { x: lx, y: 3.4, w: lw, h: 1.3, fontSize: 13.5, paraSpaceAfter: 6 });
  // 右：柱状对比
  const rx = MX + 6.3, rw = CW - 6.3;
  secHead(s, 'U11 主干策略对照 · AUROC', rx, 1.6, rw, C.ink);
  hbar(s, [
    { label: '冻结 DINOv2（本方案）', value: 0.6032, text: '0.6032', bold: true, color: C.accent },
    { label: '微调', value: 0.4914, text: '0.4914', color: C.faint },
    { label: '从零训练', value: 0.4235, text: '0.4235', color: C.faint },
  ], { x: rx, y: 2.2, w: rw, max: 0.7, labelW: 2.3, valueW: 0.8, barH: 0.42, gap: 0.75, fontSize: 13 });
  s.addShape(pptx.ShapeType.line, { x: rx, y: 4.6, w: rw, h: 0, line: { color: C.line, width: 0.75 } });
  token(s, 'fit 约 14s', { x: rx, y: 4.75, w: 2.6, h: 0.5, fontSize: 24, bold: true, color: C.accent });
  txt(s, '免去约 256s 训练\n微调易过拟合，从零训练数据不够', { x: rx + 2.6, y: 4.75, w: rw - 2.6, h: 0.6, fontSize: 12, color: C.muted, valign: 'middle' });
  takeaway(s, '不教老师傅新本事，只让他记住“正常长什么样”——这就是“少样本即可上线”。', 6.2);
})();
// 7 多专家会诊
(function () {
  const s = base('03 原理 · 有效性', '五类异常机理不同，会诊比单兵可靠：每位专家各管一类病', '答辩文档 §3.2；赛题五类异常定义',
    '为什么不用一个大模型包打天下：色偏对语义特征不敏感，错序对纹理特征不敏感。所以按缺陷机理分工，sem/disc 管一般外观，blob 管微小斑点，trad 管色偏纹理，layout 管缺件尺寸错序，inp 管局部结构，tpl 在有金样板时启用，E_open 兜底没见过的缺陷。');
  const hi = { options: { color: C.accent, bold: true, fontFace: F.num } };
  function k(t) { return { text: t, options: hi.options }; }
  table(s, ['专家（槽位）', '擅长', '一句话原理'], [
    [k('sem'), '一般外观缺陷', '和正常块比距离（借鉴 PatchCore）'],
    [k('disc'), '一般外观缺陷', '用合成的“假缺陷”训练一个判别头（借鉴 RealNet）'],
    [k('blob'), '微小斑点、污染', '全分辨率高斯差分找斑点'],
    [k('trad'), '色偏、纹理', '颜色直方图、纹理、频域等手工特征'],
    [k('layout'), '缺件、尺寸、错序', '元件个数 / 尺寸 / 顺序统计比对，不需要模板'],
    [k('inp'), '局部结构异常', '图像内部自相似：“这块和周围不像”'],
    [k('tpl'), '强配准品类（PCB）', '和金样板做差，仅在有模板时启用'],
    [{ text: 'E_open', options: { color: C.amber, bold: true, fontFace: F.num } }, '没人认识的新缺陷', '兜底哨兵：“有异常但说不清是什么”'],
  ], { x: MX, y: 1.55, w: CW, colW: [2.2, 3.0, 6.93], rowH: 0.5, fontSize: 13 });
  takeaway(s, '色偏对语义特征不敏感，错序对纹理特征不敏感——所以按机理分工，而不是一个模型包打天下。', 6.3);
})();

// 8 校准 + 融合 + 双阈值
(function () {
  const s = base('03 原理 · 有效性', '先统一尺子，再投票：分数可比、权重有底、拿不准的交给人', '答辩文档 §3.3',
    '第一步 CDF 校准：用正常样本分布把每个专家分数换成“比多少比例的正常样本更异常”，截断到 [-1,2]。注意这是相对排名尺子，不能当缺陷概率。第二步融合：一半权重平均分保底，一半按一致度分配，β=0.5。第三步双阈值：Q0.95 以下正常，Q0.99 以上异常，中间灰区交给人。');
  const cw = (CW - 0.8) / 3;
  const cols = [
    { n: '01', t: 'CDF 校准', sub: '统一尺子', body: [
      '问题：各专家量纲不同（0~1 与 0~1000），直接相加无意义',
      '做法：用正常样本分布，换成“比多少比例的正常样本更异常”，截断到 [-1, 2]',
      '注意：这是相对排名尺子，不能当成“缺陷概率”',
    ] },
    { n: '02', t: '共识融合', sub: '等权保底 + 一致度加权', formula: 'wᵢ = (1−β)/N + β · aᵢ / Σa', body: [
      'β = 0.5：一半权重平均分，防止某位专家被误判为“没用”',
      '另一半按各专家在本品类的一致度 / 区分度分配',
    ] },
    { n: '03', t: '双阈值判定', sub: '三档输出', body: [] },
  ];
  cols.forEach(function (c, i) {
    const x = MX + i * (cw + 0.4);
    s.addShape(pptx.ShapeType.line, { x: x, y: 1.6, w: cw, h: 0, line: { color: C.ink, width: 1.5 } });
    token(s, c.n, { x: x, y: 1.72, w: 0.6, h: 0.32, fontSize: 13, bold: true, color: C.amber });
    token(s, c.t, { x: x + 0.55, y: 1.72, w: cw - 0.55, h: 0.32, fontFace: F.cn, fontSize: 17, bold: true });
    token(s, c.sub, { x: x, y: 2.12, w: cw, h: 0.28, fontFace: F.cn, fontSize: 12, color: C.muted });
    let by = 2.6;
    if (c.formula) {
      s.addShape(pptx.ShapeType.rect, { x: x, y: by, w: cw, h: 0.6, fill: { color: C.accentSoft }, line: { color: C.accentSoft, width: 0 } });
      token(s, c.formula, { x: x + 0.1, y: by, w: cw - 0.2, h: 0.6, fontFace: 'Cambria Math', fontSize: 18, color: C.accent, align: 'center' });
      by += 0.8;
    }
    if (c.body.length) txt(s, c.body.map(function (b, j) { return { text: b, options: { bullet: { indent: 12 }, breakLine: j < c.body.length - 1 } }; }),
      { x: x, y: by, w: cw, h: 3.0, fontSize: 12.5, paraSpaceAfter: 8 });
  });
  // 第三列：阈值带
  const x3 = MX + 2 * (cw + 0.4), by = 2.65, bh = 0.5;
  const segs = [[0.55, C.positive, '正常'], [0.2, C.caution, '灰区'], [0.25, C.risk, '异常']];
  let cx = x3;
  segs.forEach(function (sg) {
    const w = cw * sg[0];
    s.addShape(pptx.ShapeType.rect, { x: cx, y: by, w: w, h: bh, fill: { color: sg[1] }, line: { color: 'FFFFFF', width: 1 } });
    token(s, sg[2], { x: cx, y: by, w: w, h: bh, fontFace: F.cn, fontSize: 13, bold: true, color: 'FFFFFF', align: 'center' });
    cx += w;
  });
  token(s, 'τg = Q0.95', { x: x3 + cw * 0.55 - 0.6, y: by + bh + 0.05, w: 1.2, h: 0.25, fontSize: 10.5, color: C.muted, align: 'center' });
  token(s, 'τh = Q0.99', { x: x3 + cw * 0.75 - 0.6 + 0.25, y: by - 0.3, w: 1.2, h: 0.25, fontSize: 10.5, color: C.muted, align: 'center' });
  txt(s, [
    { text: '低于 τg：判正常，直接放行', options: { bullet: { indent: 12 }, breakLine: true } },
    { text: '高于 τh：判异常，出缺陷框与类型', options: { bullet: { indent: 12 }, breakLine: true } },
    { text: '中间：灰区，交给人复核，也是主动学习的选样来源', options: { bullet: { indent: 12 } } },
  ], { x: x3, y: 3.55, w: cw, h: 2.2, fontSize: 12.5, paraSpaceAfter: 8 });
  takeaway(s, '阈值取自正常样本分位数（Q0.95 / Q0.99），不需要缺陷样本也能定档。', 6.2);
})();

// 9 SSOCL + 数据隔离
(function () {
  const s = base('03 原理 · 有效性', '受控持续学习：错题本 + 月考 + 退学机制；考试不许偷看答案', '答辩文档 §3.4–3.5；isolation.py；PCB_Dual 接口',
    'SSOCL 三件事：错题本收反馈，月考是锚定集回归，退学机制是快照回滚。机器复判让产线已有的老 AOI 也能当老师。数据隔离：V4 曾出现 0.9983 的虚高，排查发现评估数据混进了拟合，我们主动废掉并建立代码级断言。');
  const lw = 7.4;
  const rows = [
    ['错题本', '反馈分三类：判错 → 进缺陷样例库建立近邻拦截；判对 → 加固正常模式；无反馈 → 满足一致性和置信条件才缓存'],
    ['月考', '每次更新先生成候选版本，在锚定集上回归；AUROC 或缺陷召回下降即拒绝'],
    ['退学机制', '已上线版本有快照，随时回滚；所有更新留审计，可查“这个参数见过哪些数据”'],
    ['机器复判', 'PCB_Dual 传统 CV 引擎独立判断；AI 与 CV 不一致时自动生成误检 / 漏检反馈——老 AOI 也能当老师'],
  ];
  rows.forEach(function (r, i) {
    const y = 1.6 + i * 1.08;
    s.addShape(pptx.ShapeType.line, { x: MX, y: y, w: lw, h: 0, line: { color: i === 0 ? C.ink : C.line, width: i === 0 ? 1.5 : 0.75 } });
    token(s, r[0], { x: MX, y: y + 0.15, w: 1.5, h: 0.35, fontFace: F.cn, fontSize: 16, bold: true, color: C.accent });
    txt(s, r[1], { x: MX + 1.6, y: y + 0.15, w: lw - 1.6, h: 0.85, fontSize: 12.5 });
  });
  // 右：数据隔离
  const rx = MX + lw + 0.45, rw = CW - lw - 0.45;
  s.addShape(pptx.ShapeType.rect, { x: rx, y: 1.6, w: rw, h: 4.3, fill: { color: C.amberSoft }, line: { color: C.amberSoft, width: 0 } });
  secHead(s, '数据隔离：考试不许偷看答案', rx + 0.25, 1.78, rw - 0.5, C.ink);
  token(s, '0.9983', { x: rx + 0.25, y: 2.3, w: 2.2, h: 0.7, fontSize: 34, bold: true, color: C.faint });
  s.addShape(pptx.ShapeType.line, { x: rx + 0.25, y: 2.66, w: 2.0, h: 0, line: { color: C.risk, width: 2 } });
  token(s, 'V4 虚高分，已主动废弃', { x: rx + 2.4, y: 2.45, w: rw - 2.6, h: 0.4, fontFace: F.cn, fontSize: 12, color: C.risk, bold: true });
  txt(s, [
    { text: '来由：排查发现评估数据混入了拟合流程', options: { bullet: { indent: 12 }, breakLine: true } },
    { text: '代码级断言：测试域数据一旦进入 fit 直接报错', options: { bullet: { indent: 12 }, breakLine: true } },
    { text: '所有结果按“val 选参、test 只验不选”重跑', options: { bullet: { indent: 12 }, breakLine: true } },
    { text: '隔离测试 2/2 通过', options: { bullet: { indent: 12 } } },
  ], { x: rx + 0.25, y: 3.2, w: rw - 0.5, h: 2.5, fontSize: 12.5, paraSpaceAfter: 8 });
  takeaway(s, '我们主动把自己最漂亮的分数废掉了——这是数字可信的底线。', 6.2);
})();
// 10 创新点
(function () {
  const s = base('04 创新性', '零件多有出处，创新在“组合方式”和“做成能在产线闭环的系统”', '答辩文档 §4；各项证据见 §5',
    '先说清楚：单个零件大多有出处（PatchCore、RealNet、AHL、DINOv2），我们不宣称发明了新网络。创新在组合方式和工程闭环，六项各有证据。');
  table(s, ['#', '创新点', '和常见做法的区别', '证据'], [
    ['1', '槽位制多专家 + CDF 统一尺子', '常见方案单模型单分数；我们可插拔多专家，按缺陷机理分工', '五类异常检出率 0.938~1.000'],
    ['2', '场景分层冷启动（L0 / L1a / L1b / L3）', '常见方案只有一种数据假设；我们按“手里有什么数据”自动选档', '仅正常图也能应急上线；有模板自动启用 tpl'],
    ['3', '域差感知软路由 + E_open 哨兵', '常见方案对“没见过的缺陷”沉默；我们会报“疑似体系外缺陷”', 'UI 不完备预警横幅'],
    ['4', 'SSOCL 受控持续学习', '常见在线学习无门控；我们“先考试再上岗，不及格回滚”', 'bottle 错标压力测试触发 2 次回滚'],
    ['5', '机器复判当老师', '反馈只能靠人；我们让传统 AOI 的分歧自动变成反馈', 'PCB_Dual 机器复判接口已落地'],
    ['6', '实时工程闭环', '算法 demo 与软件分离；我们一体化，从建单到追溯', 'GPU 模型段 p50 58ms / mean 102ms'],
  ], { x: MX, y: 1.55, w: CW, colW: [0.5, 3.4, 5.0, 3.23], rowH: 0.62, fontSize: 12.5 });
  takeaway(s, '不宣称发明新网络；我们解决的是“拿到产线能用、用错能改、改坏能退”。', 6.3);
})();

// 11 少样本 + 公开集
(function () {
  const s = base('05 实测证据', '静态精度与主流方法同档；真实工业数据分低，我们如实登记', '五品类 100+30 协议；无泄漏口径 bench；MVTec / MPDD 主报项目文档 sanity 全量口径',
    '左边：赛题默认 100 正常 + 30 缺陷，五个品类平均，我们 0.9441，PatchCore 0.9514，差 0.007，同一水平；但我们多了反馈学习、类型归因、灰区和回滚。右边：标准集都在 0.96 以上；data_local 和 GYU 是真实工业 / 跨域，静态分很低，我们不藏，用它说明为什么必须有在线学习。');
  const lw = 5.3;
  secHead(s, '少样本冷启动 · 五品类平均 AUROC', MX, 1.55, lw, C.ink);
  token(s, 'bottle / capsule / cable / transistor / screw；横轴从 0.80 起', { x: MX, y: 1.88, w: lw, h: 0.25, fontFace: F.cn, fontSize: S.annotation, color: C.faint });
  const base0 = 0.8;
  hbar(s, [
    { label: 'PatchCore', value: 0.9514 - base0, text: '0.9514', color: C.faint },
    { label: 'AOITree', value: 0.9441 - base0, text: '0.9441', bold: true, color: C.accent },
    { label: 'EfficientAD-S', value: 0.8748 - base0, text: '0.8748', color: C.faint },
  ], { x: MX, y: 2.35, w: lw, max: 0.2, labelW: 1.55, valueW: 0.8, barH: 0.4, gap: 0.68, fontSize: 13 });
  txt(s, [
    { text: 'PatchCore、AOITree 无需训练（AOITree fit 约 14s）；EfficientAD-S 需训练', options: { breakLine: true } },
    { text: '差 0.007，但多了反馈学习、类型归因、灰区和回滚', options: { bold: true, color: C.ink } },
  ], { x: MX, y: 4.5, w: lw, h: 1.0, fontSize: 12, color: C.muted, paraSpaceAfter: 6 });
  // 右：公开集
  const rx = MX + lw + 0.5, rw = CW - lw - 0.5;
  secHead(s, '公开 / 工业数据集 · AUROC（无泄漏口径）', rx, 1.55, rw, C.ink);
  const g = { options: { color: C.positive, bold: true } }, r = { options: { color: C.risk, bold: true } };
  function v(t, o) { return { text: t, options: o.options }; }
  table(s, ['数据集', '主报', '速度工作点', '说明'], [
    ['MVTec AD', v('0.9793', g), '0.9811', '15/15 品类 ≥ 0.9'],
    ['BTAD', v('0.9609', g), '0.9679', '3/3 ≥ 0.9'],
    ['MPDD', v('0.9851', g), '0.9908', '6/6 ≥ 0.9'],
    ['data_local（真实工业）', v('0.4035', r), '0.3438', '强域差，静态基线如实登记'],
    ['GYU-DET（跨域）', v('0.5594', r), '0.6110', '跨域，靠在线学习提升'],
  ], { x: rx, y: 2.0, w: rw, colW: [2.2, 1.05, 1.25, rw - 4.5], rowH: 0.5, fontSize: 12 });
  token(s, '注：MVTec / MPDD 全槽位 bench 汇总口径为 0.9701 / 0.9797', { x: rx, y: 5.15, w: rw, h: 0.25, fontFace: F.cn, fontSize: S.annotation, color: C.faint });
  takeaway(s, '静态分数不是我们的主战场；低分场景正好说明“为什么必须有在线学习”。', 6.2);
})();
// 12 在线学习
(function () {
  const s = base('05 实测证据', '错了能教：反馈后跨域与真实工业数据明显回升', 'GYU-DET 10 轮在线学习协议；U62；data_local 单轮 / 多轮（大锚定）',
    '主报 GYU-DET 10 轮 300 条反馈：0.545 到 0.710，过程不是单调的，有波动，disc_ft 应用 18 次、拒绝 4 次，说明门控在工作。U62 单轮调优到 0.818 作为补充。data_local 的 gold_finger 从 0.122 到约 0.72；早期小锚定下 0.91 的数字我们主动废弃了。');
  const lw = 5.6;
  const items = [
    { label: 'GYU-DET 10 轮', a: 0.545, b: 0.710, note: '300 条反馈；非单调；disc_ft 应用 18 / 拒绝 4' },
    { label: 'GYU-DET U62', a: 0.636, b: 0.818, note: '单轮调优，峰值 0.820' },
    { label: 'gold_finger', a: 0.122, b: 0.72, note: '单轮 0.608 / 多轮约 0.72（大锚定）；小锚定 0.91 已废弃' },
    { label: 'solder_smt', a: 0.235, b: 0.531, note: 'data_local 真实工业' },
  ];
  secHead(s, '学习前 → 学习后 · AUROC', MX, 1.55, lw, C.ink);
  const lab = 1.6, bw = lw - lab - 1.4;
  items.forEach(function (it, i) {
    const y = 2.05 + i * 1.0;
    token(s, it.label, { x: MX, y: y, w: lab - 0.1, h: 0.3, fontFace: F.cn, fontSize: 13, bold: true });
    s.addShape(pptx.ShapeType.rect, { x: MX + lab, y: y + 0.04, w: bw, h: 0.22, fill: { color: C.surface }, line: { color: C.surface, width: 0 } });
    s.addShape(pptx.ShapeType.rect, { x: MX + lab, y: y + 0.04, w: bw * it.b, h: 0.22, fill: { color: C.accent }, line: { color: C.accent, width: 0 } });
    s.addShape(pptx.ShapeType.rect, { x: MX + lab, y: y + 0.04, w: bw * it.a, h: 0.22, fill: { color: C.faint }, line: { color: C.faint, width: 0 } });
    token(s, it.a.toFixed(3) + ' → ' + (it.label === 'gold_finger' ? '≈0.72' : it.b.toFixed(3)), { x: MX + lab + bw + 0.1, y: y, w: 1.3, h: 0.3, fontSize: 12.5, bold: true, color: C.accent });
    txt(s, it.note, { x: MX + lab, y: y + 0.35, w: lw - lab, h: 0.5, fontSize: 10.5, color: C.muted });
  });
  token(s, '■ 学习前   ', { x: MX + lab, y: 6.05, w: 1.2, h: 0.22, fontFace: F.cn, fontSize: S.annotation, color: C.faint });
  token(s, '■ 学习后', { x: MX + lab + 1.1, y: 6.05, w: 1.2, h: 0.22, fontFace: F.cn, fontSize: S.annotation, color: C.accent });
  const rx = MX + lw + 0.4, rw = CW - lw - 0.4;
  secHead(s, '学习曲线（项目文档图）', rx, 1.55, rw, C.ink);
  imgFit(s, 'fig2_学习曲线.png', 1635, 650, rx, 2.0, rw, 3.2);
  txt(s, '曲线有波动：门控会拒绝不达标的更新，不承诺“每一轮都更好”。', { x: rx, y: 5.35, w: rw, h: 0.6, fontSize: 12, color: C.muted });
})();

// 13 门控回滚
(function () {
  const s = base('05 实测证据', '教坏了能退：故意给错标，门控自动回滚 2 次', 'bottle 双臂学习协议（结论 PASS）；数据隔离测试',
    '这是专门设计的压力测试：同一品类分两臂，一臂给正确反馈，一臂故意给错误标注。正确臂流内错误从 1 降到 0，锚定 AUROC 保持 1.0；错标臂门控触发 2 次回滚。边界要主动说：单品类、单种子、锚定集只有 4 张，证明机制有效，不证明统计普遍性。');
  const cw = (CW - 0.5) / 2;
  const arms = [
    { t: '正确反馈臂', d: '给正确标注', big: '1 → 0', unit: '流内错误', color: C.positive,
      r: ['锚定 AUROC 保持 1.0', '有 1 次因 AUC 不达标被拒（门控同样在把关）'] },
    { t: '错标压力臂', d: '故意给错误标注', big: '2 次', unit: '自动回滚', color: C.risk,
      r: ['回滚原因：缺陷召回下降', '版本快照可回退，更新全程留审计记录'] },
  ];
  arms.forEach(function (a, i) {
    const x = MX + i * (cw + 0.5);
    s.addShape(pptx.ShapeType.rect, { x: x, y: 1.6, w: cw, h: 0.08, fill: { color: a.color }, line: { color: a.color, width: 0 } });
    token(s, a.t, { x: x, y: 1.85, w: cw, h: 0.4, fontFace: F.cn, fontSize: 19, bold: true });
    token(s, a.d, { x: x, y: 2.3, w: cw, h: 0.3, fontFace: F.cn, fontSize: 12.5, color: C.muted });
    token(s, a.big, { x: x, y: 2.8, w: 2.6, h: 0.9, fontFace: F.cn, fontSize: 44, bold: true, color: a.color });
    token(s, a.unit, { x: x + 2.7, y: 3.1, w: cw - 2.7, h: 0.4, fontFace: F.cn, fontSize: 15, bold: true, color: C.ink });
    txt(s, a.r.map(function (b, j) { return { text: b, options: { bullet: { indent: 12 }, breakLine: j < a.r.length - 1 } }; }),
      { x: x, y: 3.9, w: cw, h: 1.0, fontSize: 13, paraSpaceAfter: 6 });
  });
  s.addShape(pptx.ShapeType.rect, { x: MX, y: 5.05, w: CW, h: 0.75, fill: { color: C.surface }, line: { color: C.surface, width: 0 } });
  txt(s, [
    { text: '边界：', options: { bold: true, color: C.caution } },
    { text: '单品类、单随机种子、锚定集仅 4 张——证明机制有效，不证明统计意义上的普遍性。另：数据隔离测试 2/2 通过。' },
  ], { x: MX + 0.25, y: 5.05, w: CW - 0.5, h: 0.75, fontSize: 12.5, valign: 'middle', paraSpaceAfter: 0 });
  takeaway(s, '在线学习敢开，是因为每一次更新都要先考试，不及格就退回。', 6.2);
})();
// 14 速度
(function () {
  const s = base('05 实测证据', '模型段均值 102ms、p95 183ms，满足 <200ms；端到端远低于 1s', 'GTX 1660S 速度模式 2500² 基准；RTX 2070 本轮经 API 复测；10min×4 并发长稳',
    '赛题口径是模型运行时间小于 200ms、单图小于 1s。速度模式均值 101.6、p95 183.2 达标；max 263 出现在少量大图上，我们如实标出。端到端均值 162.5。无 GPU 时 CPU 降配 e2e 351ms 也在 1s 内。长稳 10 分钟 4 并发 4539 次 0 错误。突发并发 max 1740ms，所以产线建议串行或限流。');
  const lw = 6.0;
  secHead(s, '模型段延迟 · GTX 1660S · 速度模式 2500²', MX, 1.55, lw, C.ink);
  hbar(s, [
    { label: 'p50', value: 58, text: '58 ms', color: C.accent },
    { label: 'mean', value: 101.6, text: '101.6 ms', color: C.accent, bold: true },
    { label: 'p95', value: 183.2, text: '183.2 ms', color: C.accent, bold: true },
    { label: 'max', value: 263.1, text: '263.1 ms', color: C.caution },
  ], { x: MX, y: 2.45, w: lw, max: 300, labelW: 0.8, valueW: 1.0, barH: 0.36, gap: 0.62, fontSize: 13, ref: 200, refLabel: '赛题 200ms' });
  txt(s, 'max 超线出现在少量大图；突发 4 并发客户端 max 1740ms → 演示与产线按串行 / 限流部署', { x: MX, y: 5.0, w: lw, h: 0.7, fontSize: 12, color: C.caution });
  const rx = MX + lw + 0.5, rw = CW - lw - 0.5;
  table(s, ['口径', '硬件', '结果'], [
    ['端到端（2500²）', 'GTX 1660S', 'mean 162.5 / p95 241.3 ms'],
    ['大图零拷贝 4096×3000', 'GTX 1660S', 'e2e mean 131 / p95 172 ms'],
    ['CPU 降配（无 GPU）', 'CPU', '模型段 256.5 / e2e 351.3 ms'],
    ['本轮复测（经 API）', 'RTX 2070', '模型 121~147 / e2e 155~251ms'],
    ['长稳', '—', '10min × 4 并发 4539 次 0 错误'],
  ], { x: rx, y: 1.6, w: rw, colW: [1.95, 1.05, rw - 3.0], rowH: 0.58, fontSize: 11 });
  token(s, '首图冷启动约 7s，演示前需预热', { x: rx, y: 5.25, w: rw, h: 0.28, fontFace: F.cn, fontSize: S.annotation + 1, color: C.muted });
  takeaway(s, '单图 1s 红线留足余量；1660S 级显卡即可实时，无 GPU 也能降配运行。', 6.2);
})();

// 15 五类异常
(function () {
  const s = base('05 实测证据', '赛题五类异常全覆盖：检出 ≥ 0.938，类型归因仍有提升空间', '五类异常覆盖评测；逻辑错误样本仅 16 张',
    '检出率五类都在 0.938 以上，尺寸、缺件、色彩是 1.0。类型归因 top-1 在 0.56 到 0.86 之间，逻辑错误最弱 0.562，但样本只有 16 张，统计上不稳，我们列为局限、持续积累。');
  const rows = [
    ['常见外观', 0.958, 0.836], ['尺寸偏差', 1.0, 0.787], ['缺件少件', 1.0, 0.857], ['色彩变化', 1.0, 0.75], ['逻辑错误*', 0.938, 0.562],
  ];
  const colX = [MX, MX + 2.2, MX + 7.3], bw = 3.6;
  token(s, '缺陷类型', { x: colX[0], y: 1.6, w: 2, h: 0.3, fontFace: F.cn, fontSize: 12, bold: true, color: C.muted });
  token(s, '检出率', { x: colX[1], y: 1.6, w: 2, h: 0.3, fontFace: F.cn, fontSize: 12, bold: true, color: C.muted });
  token(s, '类型归因 top-1', { x: colX[2], y: 1.6, w: 3, h: 0.3, fontFace: F.cn, fontSize: 12, bold: true, color: C.muted });
  s.addShape(pptx.ShapeType.line, { x: MX, y: 1.95, w: CW, h: 0, line: { color: C.ink, width: 1 } });
  rows.forEach(function (r, i) {
    const y = 2.2 + i * 0.75;
    token(s, r[0], { x: colX[0], y: y, w: 2, h: 0.36, fontFace: F.cn, fontSize: 15, bold: true });
    [[1, r[1], C.positive], [2, r[2], r[2] < 0.7 ? C.caution : C.accent]].forEach(function (p) {
      const x = colX[p[0]];
      s.addShape(pptx.ShapeType.rect, { x: x, y: y + 0.05, w: bw, h: 0.26, fill: { color: C.surface }, line: { color: C.surface, width: 0 } });
      s.addShape(pptx.ShapeType.rect, { x: x, y: y + 0.05, w: bw * p[1], h: 0.26, fill: { color: p[2] }, line: { color: p[2], width: 0 } });
      token(s, p[1].toFixed(3), { x: x + bw + 0.12, y: y, w: 0.9, h: 0.36, fontSize: 14, bold: true, color: p[2] });
    });
    s.addShape(pptx.ShapeType.line, { x: MX, y: y + 0.58, w: CW, h: 0, line: { color: C.line, width: 0.5 } });
  });
  token(s, '* 逻辑错误样本仅 16 张，统计不稳，已列为局限', { x: MX, y: 5.95, w: 8, h: 0.25, fontFace: F.cn, fontSize: S.annotation, color: C.faint });
  takeaway(s, '“有没有缺陷”已经可靠；“是哪一类缺陷”随样本积累继续提升。', 6.3);
})();
// 16/17 演示路径：2x2 截图 + 旁注
function demoGrid(s, cells) {
  const colW = (CW - 0.4) / 2, ih = 2.45, iw = 3.8, tw = colW - iw - 0.2;
  cells.forEach(function (c, i) {
    const x = MX + (i % 2) * (colW + 0.4), y = 1.5 + Math.floor(i / 2) * (ih + 0.25);
    imgFit(s, c.f, c.iw || 1188, c.ih || 768, x, y, iw, ih);
    const tx = x + iw + 0.2;
    token(s, c.n, { x: tx, y: y, w: 0.6, h: 0.4, fontSize: 20, bold: true, color: C.amber });
    token(s, c.t, { x: tx, y: y + 0.45, w: tw, h: 0.36, fontFace: F.cn, fontSize: 15, bold: true });
    txt(s, c.d, { x: tx, y: y + 0.9, w: tw, h: ih - 0.9, fontSize: 11.5, color: C.muted });
  });
}
(function () {
  const s = base('06 系统演示', '演示路径①：从开机到模型上线，四步走完', '实际运行截图（bottle 品类）',
    '按产线一天的顺序演示。工作台是开机首屏，有工单和品类状态机；数据管理能自动识别 MVTec、YOLO 和命名规则；模型管理里 100+30 一键准备，约 14 秒完成并给出体检结论；评估看板看上线前指标与阈值。');
  demoGrid(s, [
    { n: '1', f: '02_工作台.jpg', t: '工作台', d: '开机首屏：工单、品类状态机、下一步引导' },
    { n: '2', f: '01_数据管理.jpg', t: '数据管理', d: '自动识别 MVTec / YOLO / 命名规则，确认后入库' },
    { n: '3', f: '04_模型管理.jpg', t: '模型管理', d: '100+30 一键准备，约 14s 完成，附体检结论' },
    { n: '4', f: '05_评估看板.jpg', t: '评估看板', d: '上线前查看指标与双阈值' },
  ]);
})();
(function () {
  const s = base('06 系统演示', '演示路径②：检测 → 反馈 → 学习 → 追溯，闭环跑通', '实际运行截图；演示前先预热首图（约 7s），检测按串行执行',
    '实时监控逐图给出判定、热力图、缺陷框、延迟和主导专家；操作员在标注反馈里点误检或漏检；学习提升生成新版本，先过锚定集考试，可回滚；统计报表提供日报、审计和 CSV 导出。演示前先跑一张图预热。');
  demoGrid(s, [
    { n: '5', f: '06_实时监控.jpg', t: '实时监控', d: '逐图判定、热力图、缺陷框、延迟、主导专家' },
    { n: '6', f: '07_标注反馈.jpg', t: '标注反馈', d: '操作员点选误检 / 漏检，无需懂算法' },
    { n: '7', f: '12_学习提升后.png', iw: 1591, ih: 978, t: '学习提升', d: '生成新版本 → 锚定集考试 → 可回滚' },
    { n: '8', f: '09_统计报表.jpg', t: '统计报表', d: '日报、审计记录、CSV 导出' },
  ]);
})();

// 18 局限
(function () {
  const s = base('07 局限与边界', '主动说清边界：哪些已验证，哪些还没做到', '答辩文档 §7',
    '局限主动讲，比被评委问出来好。最大的是真实工业强域差，静态只有 0.40，靠在线学习拉到 0.53~0.72；光照位移扰动敏感，需要产线约束；在线学习非单调，门控降风险但不是硬保证；24 小时稳定性、RTSP、MES 真实协议还没实测。');
  table(s, ['局限', '现状', '应对'], [
    ['真实工业强域差', 'data_local 静态 AUROC 仅 0.40', '在线学习后 0.53~0.72；有金样板时启用 tpl'],
    ['模板依赖覆盖度', '闭集模板 1.0；仅隔离正常图作模板 0.27', '模板只作可选增强，不混报'],
    ['光照 / 位移扰动敏感', 'gold_finger 扰动下漏检率最高 0.93', 'TTA 可降误报但超 1s，仅用于复检；产线需光照 / 对位约束'],
    ['类型归因偏弱', '逻辑错误 top-1 仅 0.562', '样本少（16 张），持续积累'],
    ['定位精度', 'patch 级（14~28px）', '满足框选，不做像素级分割'],
    ['在线学习非单调', 'MPDD bracket_brown 登记 −0.34 负增益', '门控降风险但非硬保证；锚定集补缺陷样本'],
    ['长时稳定性', '10min 压测通过，24h 未实跑', '工具已就绪'],
    ['部署形态', 'RTSP、MES 真实协议、RTX 2060 未实测', '接口与模拟网关已完成'],
  ], { x: MX, y: 1.55, w: CW, colW: [2.4, 4.4, CW - 6.8], rowH: 0.5, fontSize: 12, boldFirst: true });
  takeaway(s, '所有数字都带口径；虚高结果（如 0.9983、小锚定 0.91）已主动废弃。', 6.3);
})();

// 19 评委问答
(function () {
  const s = base('08 评委问答', '预判评委八问：每个回答都有数据支撑', '答辩文档 §8',
    '这一页备查，按评委提问跳转作答。详细答案见答辩文档第 8 节。');
  const qa = [
    ['和 PatchCore 有什么区别？', '静态精度同档（0.9441 vs 0.9514）；多了三件产线能力：\n多专家归因、反馈即学、门控回滚。'],
    ['data_local 为何只有 0.40？', '真实工业强域差，如实登记；gold_finger 在线学习 0.12 → 约 0.72，小锚定 0.91 已废弃。'],
    ['在线学习会学坏吗？', '有风险，所以有门控：错标实验自动回滚 2 次；\n负增益案例已如实登记。'],
    ['有没有偷用目标数据？', '没有。目标品类只用 100+30；测试集进入 fit 会被代码断言拦截。'],
    ['速度怎么算？max 超 200ms？', '模型段 mean 102 / p95 183ms 达标；max 263ms 为少量大图；端到端 162ms ≪ 1s。'],
    ['操作员要懂算法吗？', '不需要。三个按钮：误检 / 漏检 / 确认正确；灰区样本自动推送复核。'],
    ['没见过的新缺陷怎么办？', 'E_open 哨兵兜底，报“疑似体系外缺陷”；样本积累后孵化新专家。'],
    ['落地成本？', '主干冻结，可训练参数 <0.5M；1660S 即可实时，无 GPU 降配 e2e 351ms。'],
  ];
  const cw = (CW - 0.5) / 2, bh = 1.3;
  qa.forEach(function (q, i) {
    const x = MX + Math.floor(i / 4) * (cw + 0.5), y = 1.5 + (i % 4) * bh;
    token(s, 'Q' + (i + 1), { x: x, y: y, w: 0.5, h: 0.34, fontSize: 14, bold: true, color: C.amber });
    token(s, q[0], { x: x + 0.55, y: y, w: cw - 0.55, h: 0.34, fontFace: F.cn, fontSize: 14, bold: true });
    txt(s, q[1], { x: x + 0.55, y: y + 0.4, w: cw - 0.55, h: 0.75, fontSize: 12, color: C.muted });
    if (i % 4 < 3) s.addShape(pptx.ShapeType.line, { x: x, y: y + bh - 0.1, w: cw, h: 0, line: { color: C.line, width: 0.5 } });
  });
})();

// 20 结束
(function () {
  const s = pptx.addSlide(); pageNo += 1;
  s.background = { color: C.bg };
  s.addShape(pptx.ShapeType.rect, { x: 0, y: 0, w: 0.18, h: H, fill: { color: C.accent }, line: { color: C.accent, width: 0 } });
  token(s, '总结', { x: 1.0, y: 1.2, w: 4, h: 0.35, fontFace: F.cn, fontSize: 14, bold: true, color: C.accent });
  const pts = [
    ['少样本即可上线', '冻结大模型 + 多专家，100+30 张、约 14s 准备，公开集 AUROC 0.96~0.98'],
    ['错了能教', '操作员一次点击即反馈，跨域 GYU-DET 0.545 → 0.710'],
    ['教坏能退', '锚定集门控 + 版本快照，错标压力下自动回滚 2 次'],
  ];
  pts.forEach(function (p, i) {
    const y = 1.85 + i * 1.05;
    token(s, '0' + (i + 1), { x: 1.0, y: y, w: 0.7, h: 0.5, fontSize: 24, bold: true, color: C.amber });
    token(s, p[0], { x: 1.8, y: y, w: 4, h: 0.5, fontFace: F.cn, fontSize: 22, bold: true });
    token(s, p[1], { x: 5.6, y: y, w: 7.0, h: 0.5, fontFace: F.cn, fontSize: 14, color: C.muted });
  });
  s.addShape(pptx.ShapeType.line, { x: 1.0, y: 5.25, w: 5.5, h: 0, line: { color: C.amber, width: 2 } });
  token(s, '谢谢各位评委，欢迎提问', { x: 1.0, y: 5.5, w: 10, h: 0.7, fontFace: F.cn, fontSize: 30, bold: true });
  token(s, 'AOITree', { x: 1.0, y: 6.6, w: 3, h: 0.3, fontSize: 12, color: C.faint });
  s.addNotes('总结三句话：少样本即可上线、错了能教、教坏能退。谢谢各位评委，欢迎提问。');
})();

pptx.writeFile({ fileName: OUT }).then(function (f) { console.log('written: ' + f + ' pages=' + pageNo); });
