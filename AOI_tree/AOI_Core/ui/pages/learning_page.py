"""学习效果页（W-learn 2026-08-29 重构，docs/重构设计_学习与统计页.md）。

按"学习证据三层"组织，替代旧「离线回放为主 + 在线曲线为辅」结构：
- 学过就会（已学缺陷记住）：翻案曲线（从反馈页迁入）+ 离线回放的已反馈重测
- 举一反三（未见缺陷泛化）：离线多轮回放的锚定集 AUROC 曲线（含不学习基线）
- 线上生效（真实产线改善）：误判率趋势（FP/FN 率按日）+ 权重演化 + 滚动判对率

Tab「线上学习证据」（默认）：真实反馈驱动的证据——翻案曲线 / 误判率趋势 /
权重随反馈演化 / 滚动判对率（注明选择偏差：操作员专挑可疑图反馈，
学习越好反馈越少、剩下的多是难例，判对率可能反而下降，解读看误判率趋势）。

Tab「离线回放验证」：模拟预演（反馈真值来自数据标注，algo 诚实边界），
多轮持续学习（algo simulate_learning_rounds，跨轮累积突破旧单轮流 ~53 条
反馈上限与早期饱和）；参数折叠为预设档位。贡献档案三层全展示：
单变量（槽位×缺陷类型）/ 冗余度（槽位相关矩阵）/ 消融 + 融合对照
（fused vs 最强单槽——回答"融合是否被弱槽稀释"，U31 教训）。
"""
from __future__ import annotations

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ui.api_client import ApiClient
from ui.pages.common import (SLOT_CN, attach_level_badge, make_banner,
                             make_card, run_async, warn)
from ui.theme import DANGER, PRIMARY, SUCCESS, TEXT_SUB, WARNING
from ui.widgets.charts import SERIES_COLORS, BarChart, LineChart
from ui.widgets.kpi_card import KpiCard
from ui.widgets.task_progress import TaskMonitor, TaskProgressBar


def _fmt(v, nd: int = 3) -> str:
    """指标容错格式化：None / 非数值显示 '-'。"""
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return "-"


def _pct(v) -> str:
    try:
        return f"{float(v) * 100:.1f}%"
    except (TypeError, ValueError):
        return "-"


# 离线回放预设档位（参数不向最终用户裸露）：(名称, round_size, n_rounds,
# n_eval, eval_per_sample)
REPLAY_PRESETS = [
    ("快速验证（3 轮 × 15 条反馈）", 15, 3, 40, 5),
    ("标准（4 轮 × 25 条反馈）", 25, 4, 60, 5),
    ("深度（6 轮 × 30 条反馈）", 30, 6, 80, 8),
]


class LearningPage(QWidget):
    """学习效果页：线上学习证据 + 离线回放验证。"""

    def __init__(self, client: ApiClient, get_category,
                 task_monitor: TaskMonitor | None = None,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self._client = client
        self._get_category = get_category
        self._monitor = task_monitor

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(8)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_online_tab(), "线上学习证据")
        self.tabs.addTab(self._build_offline_tab(), "离线回放验证")
        self.tabs.currentChanged.connect(self._on_tab_changed)
        root.addWidget(self.tabs, 1)

        self.progress = TaskProgressBar("学习任务")
        root.addWidget(self.progress)
        if self._monitor is not None:
            self._monitor.task_updated.connect(self._on_task)
        self._client.error_occurred.connect(lambda m: warn(self, m))
        self.reload()

    # ══════════════════ Tab1：线上学习证据 ══════════════════
    def _build_online_tab(self) -> QWidget:
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(0, 8, 0, 0)
        lay.setSpacing(8)

        bar = QHBoxLayout()
        bar.addWidget(make_banner(
            "线上学习证据：真实反馈驱动——学过就会（翻案）、线上生效（误判率）、"
            "权重随反馈演化"))
        attach_level_badge(self, bar)
        bar.addStretch(1)
        bar.addWidget(QLabel("品类"))
        self.combo_online_cat = QComboBox()
        self.combo_online_cat.setMinimumWidth(140)
        self.combo_online_cat.currentIndexChanged.connect(
            lambda _i: self.reload_online())
        bar.addWidget(self.combo_online_cat)
        btn_refresh = QPushButton("刷新")
        btn_refresh.setProperty("flat", True)
        btn_refresh.clicked.connect(self.reload_online)
        bar.addWidget(btn_refresh)
        lay.addLayout(bar)

        kpi_card = make_card()
        kpi_row = QHBoxLayout(kpi_card)
        self.kpi_flip = KpiCard("翻案率", "-", accent=SUCCESS)
        self.kpi_flip.setToolTip(
            "学习前判错、学习后判对的反馈占比（学过就会的直接证据）。\n"
            "口径注意：分母为全部有学习前判定记录的反馈（含本就判对的），"
            "反馈增多会稀释走低——翻案率下降不等于学习失效。")
        self.kpi_mis = KpiCard("错检数", "-", accent=DANGER)
        self.kpi_mistake = KpiCard("最新日误判率", "-", accent=WARNING)
        self.kpi_mistake.setToolTip("最近一个有复核的日期，（误报+漏检）/复核数")
        self.kpi_weight = KpiCard("权重学习", "-", accent=PRIMARY)
        self.kpi_weight.setToolTip("融合权重写回通过次数 / 门控否决次数")
        for k in (self.kpi_flip, self.kpi_mis, self.kpi_mistake, self.kpi_weight):
            kpi_row.addWidget(k)
        lay.addWidget(kpi_card)

        grid = QGridLayout()
        grid.setSpacing(8)
        # 翻案曲线（学过就会）
        flip_card = make_card()
        f_lay = QVBoxLayout(flip_card)
        f_head = QLabel("翻案曲线（灰=学习前判定分，蓝=当前引擎重判定分，绿点=已翻案）")
        f_head.setProperty("heading", True)
        f_lay.addWidget(f_head)
        # 前端反馈 2026-09-13 #12：补"怎么看"读法
        f_read = QLabel("怎么看：绿点=学习前判错、学习后判对的样本；绿点持续出现"
                        "=反馈正在被学会。长期无绿点说明学习未生效，应检查反馈质量")
        f_read.setProperty("subtext", True)
        f_read.setWordWrap(True)
        f_lay.addWidget(f_read)
        self.flip_chart = LineChart()
        f_lay.addWidget(self.flip_chart, 1)
        self.lbl_flip_hint = QLabel("暂无反馈记录")
        self.lbl_flip_hint.setProperty("subtext", True)
        self.lbl_flip_hint.setWordWrap(True)
        f_lay.addWidget(self.lbl_flip_hint)
        grid.addWidget(flip_card, 0, 0)

        # 误判率趋势（线上生效）
        trend_card = make_card()
        t_lay = QVBoxLayout(trend_card)
        t_head = QLabel("误判率趋势（按日，复核口径；学习生效应持续下降）")
        t_head.setProperty("heading", True)
        t_lay.addWidget(t_head)
        self.trend_chart = LineChart()
        t_lay.addWidget(self.trend_chart, 1)
        self.lbl_trend_hint = QLabel("暂无复核记录")
        self.lbl_trend_hint.setProperty("subtext", True)
        t_lay.addWidget(self.lbl_trend_hint)
        grid.addWidget(trend_card, 0, 1)

        # 权重演化
        wh_card = make_card()
        w_lay = QVBoxLayout(wh_card)
        w_head = QLabel("融合权重随反馈演化（x=带真值反馈数）")
        w_head.setProperty("heading", True)
        w_lay.addWidget(w_head)
        self.wh_chart = LineChart()
        w_lay.addWidget(self.wh_chart, 1)
        self.lbl_wh_hint = QLabel("反馈累积后展示权重演化")
        self.lbl_wh_hint.setProperty("subtext", True)
        self.lbl_wh_hint.setWordWrap(True)
        w_lay.addWidget(self.lbl_wh_hint)
        grid.addWidget(wh_card, 1, 0)

        # 滚动判对率（保留旧指标 + 选择偏差提示）
        on_card = make_card()
        o_lay = QVBoxLayout(on_card)
        o_head = QLabel("滚动判对率（x=反馈序号，y=最近 10 条判对率）")
        o_head.setProperty("heading", True)
        o_lay.addWidget(o_head)
        self.online_chart = LineChart()
        o_lay.addWidget(self.online_chart, 1)
        o_hint = QLabel("注意选择偏差：操作员专挑可疑图反馈，学习越好反馈越少、"
                        "剩下的多是难例，判对率可能反而下降——学习是否生效"
                        "请以误判率趋势与翻案曲线为准")
        o_hint.setProperty("subtext", True)
        o_hint.setWordWrap(True)
        o_lay.addWidget(o_hint)
        grid.addWidget(on_card, 1, 1)

        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        lay.addLayout(grid, 1)
        run_async(self, self._client.list_categories, self._fill_online_cats)
        return tab

    def _fill_online_cats(self, cats) -> None:
        cats = [str(c) for c in (cats or []) if c]
        cur = str(self.combo_online_cat.currentData() or "") or self._category()
        self.combo_online_cat.blockSignals(True)
        self.combo_online_cat.clear()
        for c in cats:
            self.combo_online_cat.addItem(c, c)
        idx = self.combo_online_cat.findData(cur)
        self.combo_online_cat.setCurrentIndex(idx if idx >= 0 else 0)
        self.combo_online_cat.blockSignals(False)

    def reload_online(self) -> None:
        cat = str(self.combo_online_cat.currentData() or "") or self._category()
        if not cat:
            return
        run_async(self, lambda: self._client.flip_curve(cat),
                  self._fill_flip_curve)
        run_async(self, lambda: self._client.misjudged(
            category=cat, page=1, page_size=1), self._fill_misjudged)
        run_async(self, lambda: self._client.stats_mistake_trend(
            category=cat, range_name="30d"), self._fill_trend)
        run_async(self, lambda: self._client.weight_history(cat),
                  self._fill_weight_history)
        run_async(self, lambda: self._client.online_curve(cat),
                  self._fill_online_curve)

    def _fill_misjudged(self, data) -> None:
        total = int((data or {}).get("total", 0)) if isinstance(data, dict) else 0
        self.kpi_mis.set_value(str(total))

    def _fill_flip_curve(self, data) -> None:
        self.flip_chart.clear()
        if not isinstance(data, dict) or "_status" in data:
            self.kpi_flip.set_value("-")
            return
        if data.get("task_id"):
            self.kpi_flip.set_value("…")
            self.progress.show()
            self.progress.update_task({
                "task_id": data.get("task_id"), "status": "running",
                "progress": 0,
                "message": "翻案曲线后台重打分中（反馈逐条过引擎）…"})
            return
        points = [p for p in (data.get("points") or []) if isinstance(p, dict)]
        summary = data.get("summary") or {}
        rate = summary.get("flip_rate")
        self.kpi_flip.set_value(
            _pct(rate), f"已翻案 {summary.get('flipped_count', 0)}"
                        f"/{summary.get('total', 0)}")
        if not points:
            self.lbl_flip_hint.setText("暂无反馈记录")
            return
        ks = [p.get("k") for p in points]
        self.flip_chart.add_series("学习前判定", ks,
                                   [p.get("pre_score") for p in points],
                                   color="#9CA3AF")
        self.flip_chart.add_series("当前引擎判定", ks,
                                   [p.get("post_score") for p in points],
                                   color=PRIMARY)
        fx = [p.get("k") for p in points if p.get("flipped")]
        fy = [p.get("post_score") for p in points if p.get("flipped")]
        if fx:
            self.flip_chart.plot.plot(
                fx, fy, pen=None, symbol="o", symbolSize=11,
                symbolBrush=pg.mkBrush(SUCCESS), symbolPen=pg.mkPen(SUCCESS),
                name="已翻案")
        self.flip_chart.set_labels(y_title="异常分数")
        legacy = int(summary.get("legacy_count", 0) or 0)
        self.lbl_flip_hint.setText(
            f"反馈 {summary.get('total', 0)} 条"
            + (f"（其中 {legacy} 条老反馈无学习前判定记录）" if legacy else "")
            + "。口径：分母为全部有学习前判定记录的反馈（含本就判对的），"
              "反馈增多会稀释走低——翻案率下降不等于学习失效")

    def _fill_trend(self, data) -> None:
        items = (data or {}).get("items", []) if isinstance(data, dict) else []
        self.trend_chart.clear()
        reviewed_days = [it for it in items if int(it.get("reviewed", 0)) > 0]
        if not items or not reviewed_days:
            self.kpi_mistake.set_value("-")
            self.lbl_trend_hint.setText("暂无复核记录：提交反馈/复核后生成趋势")
            return
        last = reviewed_days[-1]
        self.kpi_mistake.set_value(
            _pct(last.get("mistake_rate")),
            f"{last.get('date', '')[5:]} 复核{last.get('reviewed', 0)}条")
        xs = list(range(len(items)))
        self.trend_chart.add_series(
            "误报率", xs, [float(it.get("fp_rate", 0) or 0) for it in items],
            color=WARNING)
        self.trend_chart.add_series(
            "漏检率", xs, [float(it.get("fn_rate", 0) or 0) for it in items],
            color=DANGER)
        labels = [str(it.get("date", ""))[5:] for it in items]
        self.trend_chart.set_labels(labels, "比率")
        self.lbl_trend_hint.setText(
            "误报率=判异常实为正常/判异常数；漏检率=判正常实为缺陷/实缺陷数"
            "（当日无复核按 0 计）")

    def _fill_weight_history(self, data) -> None:
        data = data if isinstance(data, dict) else {}
        points = [p for p in (data.get("points") or []) if isinstance(p, dict)]
        n_apply = sum(1 for p in points if p.get("action") == "apply")
        n_reject = sum(1 for p in points
                       if str(p.get("action", "")).startswith("reject"))
        self.kpi_weight.set_value(f"{n_apply} / {n_reject}", "通过 / 否决")
        self.wh_chart.clear()
        if not points:
            self.lbl_wh_hint.setText(
                "暂无权重演化记录：反馈累积后展示"
                + ("" if data.get("enabled") else "（品类未准备）"))
            return
        slots = sorted({s for p in points
                        for s in (p.get("weights") or {})})
        if not slots:
            return
        # 末点可能缺失当前态：补引擎实时权重
        cur = data.get("current")
        n_fb_now = int(data.get("n_labeled_fb", 0) or 0)
        if isinstance(cur, dict) and cur and points:
            last = points[-1]
            if int(last.get("n_fb", 0)) < n_fb_now:
                points = points + [{"n_fb": n_fb_now, "action": "current",
                                    "weights": cur}]
        xs = [int(p.get("n_fb", 0)) for p in points]
        for i, s in enumerate(slots):
            ys = [float((p.get("weights") or {}).get(s, 0.0)) for p in points]
            self.wh_chart.add_series(SLOT_CN.get(s, s), xs, ys,
                                     color=SERIES_COLORS[i % len(SERIES_COLORS)],
                                     symbol=False)
        # 否决点标记（红叉）
        rx = [int(p.get("n_fb", 0)) for p in points
              if str(p.get("action", "")).startswith("reject")]
        if rx and slots:
            s0 = slots[0]
            ry = [float((p.get("weights") or {}).get(s0, 0.0)) for p in points
                  if str(p.get("action", "")).startswith("reject")]
            self.wh_chart.plot.plot(
                rx, ry, pen=None, symbol="x", symbolSize=10,
                symbolBrush=pg.mkBrush(DANGER), symbolPen=pg.mkPen(DANGER, width=2),
                name="门控否决")
        self.wh_chart.set_labels([str(x) for x in xs], "权重")
        self.lbl_wh_hint.setText(
            f"共 {len(points)} 个权重快照（随快照版本持久化）")

    def _fill_online_curve(self, data) -> None:
        data = data if isinstance(data, dict) else {}
        points = data.get("points") or []
        self.online_chart.clear()
        if points:
            ks = [float(p.get("k", i + 1)) for i, p in enumerate(points)]
            acc = [float(p.get("rolling_acc", 0) or 0) for p in points]
            self.online_chart.add_series("滚动判对率", ks, acc, color=PRIMARY)
            self.online_chart.set_labels([str(int(k)) for k in ks], "判对率")
        else:
            self.online_chart.set_labels([], "判对率")

    # ══════════════════ Tab2：离线回放验证 ══════════════════
    def _build_offline_tab(self) -> QWidget:
        tab = QWidget()
        off = QVBoxLayout(tab)
        off.setContentsMargins(0, 8, 0, 0)
        off.setSpacing(8)

        top = QHBoxLayout()
        top.addWidget(make_banner(
            "离线回放 = 模拟预演：反馈真值来自数据标注（非真实操作员），"
            "用于预演学习能力；真实产线学习证据见「线上学习证据」"))
        tip = QLabel("回放在独立副本上进行，与产线并行、无需暂停产线")
        tip.setProperty("subtext", True)
        top.addWidget(tip)
        top.addStretch(1)
        top.addWidget(QLabel("品类"))
        self.combo_offline_cat = QComboBox()
        self.combo_offline_cat.setMinimumWidth(140)
        self.combo_offline_cat.setToolTip(
            "回放/贡献档案针对该品类；「跟随工单」= 当前工单单品类推导"
            "（多品类工单为空，需要在此指定）")
        self.combo_offline_cat.addItem("跟随工单", "")
        self.combo_offline_cat.currentIndexChanged.connect(
            lambda _i: self.reload())
        top.addWidget(self.combo_offline_cat)
        top.addWidget(QLabel("档位"))
        self.combo_preset = QComboBox()
        for name, _rs, _nr, _ne, _eps in REPLAY_PRESETS:
            self.combo_preset.addItem(name)
        self.combo_preset.setCurrentIndex(1)
        self.combo_preset.setToolTip(
            "回放规模预设：多轮持续学习（轮次×每轮反馈数）。\n"
            "轮池取自品类锚定集之外的有标注图，不足时自动收紧轮数")
        top.addWidget(self.combo_preset)
        self.btn_run = QPushButton("开始回放")
        self.btn_run.setProperty("primary", True)
        self.btn_run.clicked.connect(self._on_run_curve)
        top.addWidget(self.btn_run)
        self.btn_contrib = QPushButton("生成贡献档案")
        self.btn_contrib.clicked.connect(self._on_run_contribution)
        top.addWidget(self.btn_contrib)
        btn_refresh = QPushButton("刷新")
        btn_refresh.setProperty("flat", True)
        btn_refresh.clicked.connect(self.reload)
        top.addWidget(btn_refresh)
        off.addLayout(top)

        kpi_card = make_card()
        kpi_row = QHBoxLayout(kpi_card)
        self.kpi_initial = KpiCard("初始 AUROC", "-")
        self.kpi_final = KpiCard("最终 AUROC", "-", accent=PRIMARY)
        self.kpi_gain = KpiCard("AUROC 增益", "-", accent=SUCCESS)
        self.kpi_learned = KpiCard("已反馈重测准确率", "-", accent=PRIMARY)
        self.kpi_learned.setToolTip("学习后对已反馈样本的重测判对率（学过就会）")
        self.kpi_fliprate = KpiCard("错→对转化率", "-", accent=SUCCESS)
        self.kpi_fliprate.setToolTip("学习前判错的样本，学习后判对的比例")
        self.kpi_nfb = KpiCard("反馈条数", "-")
        for k in (self.kpi_initial, self.kpi_final, self.kpi_gain,
                  self.kpi_learned, self.kpi_fliprate, self.kpi_nfb):
            kpi_row.addWidget(k)
        off.addWidget(kpi_card)

        curve_card = make_card()
        c_lay = QVBoxLayout(curve_card)
        c_head = QLabel("学习曲线（x=累计反馈数；举一反三=锚定集 AUROC，"
                        "学过就会=已反馈重测）")
        c_head.setProperty("heading", True)
        c_lay.addWidget(c_head)
        # 前端反馈 2026-09-13 #12：补"怎么看"读法
        c_read = QLabel(
            "怎么看：「举一反三」曲线随反馈累积上升=学习推广到了没见过的样本"
            "（核心指标）；「学过就会」接近 1.0 属正常（重测的是已学样本）。"
            "两条线长期持平不动=反馈没带来增益，应检查反馈是否准确")
        c_read.setProperty("subtext", True)
        c_read.setWordWrap(True)
        c_lay.addWidget(c_read)
        self.curve_chart = LineChart()
        self.curve_chart.set_labels([], "比率")
        c_lay.addWidget(self.curve_chart, 1)
        self.lbl_curve_hint = QLabel(
            "暂无学习曲线记录：选择品类后点击「开始回放」生成")
        self.lbl_curve_hint.setProperty("subtext", True)
        self.lbl_curve_hint.setWordWrap(True)
        c_lay.addWidget(self.lbl_curve_hint)
        self.lbl_evidence = QLabel("")
        self.lbl_evidence.setProperty("subtext", True)
        self.lbl_evidence.setWordWrap(True)
        c_lay.addWidget(self.lbl_evidence)
        off.addWidget(curve_card, 3)

        # 贡献档案区：消融图 | 单变量表 | 冗余矩阵
        splitter = QSplitter(Qt.Horizontal)
        ab_card = make_card()
        a_lay = QVBoxLayout(ab_card)
        a_head = QLabel("槽位消融（单槽位 / 剔除该槽位 AUROC）")
        a_head.setProperty("heading", True)
        a_lay.addWidget(a_head)
        # 前端反馈 2026-09-13 #12：补"怎么看"读法
        a_read = QLabel(
            "怎么看：每个检测模块两根柱——「单用」=它独自的判别力，「剔除」=去掉"
            "它之后整体还剩多少。剔除后明显下降=该模块有独立贡献不能裁；"
            "剔除后几乎不降=冗余，可裁剪提速")
        a_read.setProperty("subtext", True)
        a_read.setWordWrap(True)
        a_lay.addWidget(a_read)
        self.ablation_chart = BarChart()
        a_lay.addWidget(self.ablation_chart, 1)
        self.lbl_ablation_hint = QLabel(
            "暂无贡献档案：点击「生成贡献档案」计算")
        self.lbl_ablation_hint.setProperty("subtext", True)
        self.lbl_ablation_hint.setWordWrap(True)
        a_lay.addWidget(self.lbl_ablation_hint)
        splitter.addWidget(ab_card)

        uni_card = make_card()
        u_lay = QVBoxLayout(uni_card)
        u_head = QLabel("单槽位 × 缺陷类型 AUROC（哪类缺陷靠哪个槽位）")
        u_head.setProperty("heading", True)
        u_lay.addWidget(u_head)
        self.table_uni = QTableWidget(0, 0)
        self.table_uni.setEditTriggers(QTableWidget.NoEditTriggers)
        u_lay.addWidget(self.table_uni, 1)
        splitter.addWidget(uni_card)

        red_card = make_card()
        r_lay = QVBoxLayout(red_card)
        r_head = QLabel("槽位相关矩阵（Spearman；高相关=冗余可裁剪提速）")
        r_head.setProperty("heading", True)
        r_lay.addWidget(r_head)
        self.table_red = QTableWidget(0, 0)
        self.table_red.setEditTriggers(QTableWidget.NoEditTriggers)
        r_lay.addWidget(self.table_red, 1)
        splitter.addWidget(red_card)
        splitter.setSizes([430, 330, 300])
        off.addWidget(splitter, 2)

        run_async(self, self._client.list_categories, self._fill_offline_cats)
        return tab

    def _fill_offline_cats(self, cats) -> None:
        cats = [str(c) for c in (cats or []) if c]
        cur = str(self.combo_offline_cat.currentData() or "")
        self.combo_offline_cat.blockSignals(True)
        self.combo_offline_cat.clear()
        self.combo_offline_cat.addItem("跟随工单", "")
        for c in cats:
            self.combo_offline_cat.addItem(c, c)
        idx = self.combo_offline_cat.findData(cur)
        if idx >= 0:
            self.combo_offline_cat.setCurrentIndex(idx)
        self.combo_offline_cat.blockSignals(False)

    # ══════════════════ 数据加载 ══════════════════
    def _category(self) -> str:
        return self._get_category() if self._get_category else ""

    def _offline_cat(self) -> str:
        return str(self.combo_offline_cat.currentData() or "") or self._category()

    def show_category(self, cat: str) -> None:
        """从模型页跳转进入：定位品类并加载（离线 Tab）。"""
        cat = str(cat or "").strip()
        if not cat:
            return
        self.tabs.setCurrentIndex(1)
        combo = self.combo_offline_cat
        if combo.findData(cat) < 0:
            combo.blockSignals(True)
            combo.addItem(cat, cat)
            combo.blockSignals(False)
        idx = combo.findData(cat)
        if combo.currentIndex() != idx:
            combo.setCurrentIndex(idx)
        else:
            self.reload()
        # 线上证据 Tab 同步定位
        oidx = self.combo_online_cat.findData(cat)
        if oidx >= 0:
            self.combo_online_cat.setCurrentIndex(oidx)

    def _on_tab_changed(self, idx: int) -> None:
        if idx == 0:
            self.reload_online()
        else:
            self.reload()

    def reload(self) -> None:
        cat = self._offline_cat()
        if not cat:
            self._fill_curve(None)
            self._fill_contribution(None)
            return
        run_async(self, lambda: self._client.learning_curve_latest(cat),
                  self._fill_curve)
        run_async(self, lambda: self._client.learning_contribution_latest(cat),
                  self._fill_contribution)

    def _fill_curve(self, data) -> None:
        data = data if isinstance(data, dict) else {}
        mode = str(data.get("mode") or "stream")
        self.kpi_initial.set_value(_fmt(data.get("initial_auroc")))
        self.kpi_final.set_value(_fmt(data.get("final_auroc")))
        gain = data.get("auroc_gain")
        self.kpi_gain.set_value(("+" if isinstance(gain, (int, float))
                                 and gain >= 0 else "") + _fmt(gain))
        self.kpi_gain.set_accent(
            SUCCESS if isinstance(gain, (int, float)) and gain >= 0
            else DANGER)
        n_fb = data.get("n_feedback")
        self.kpi_nfb.set_value(str(n_fb) if n_fb is not None else "-")

        steps = data.get("steps") or []
        auroc = data.get("auroc") or []
        self.curve_chart.clear()
        if mode == "rounds":
            self.kpi_learned.set_value(_pct(data.get("final_learned_acc")))
            self.kpi_fliprate.set_value(_pct(data.get("final_flip_rate")))
            if steps and auroc:
                xs = [float(s) for s in steps]
                self.curve_chart.add_series(
                    "锚定集 AUROC（举一反三）", xs,
                    [float(v) for v in auroc], color=PRIMARY)
                # 不学习基线：模型不变 → AUROC 恒为初始值（定义性基线）
                init = data.get("initial_auroc")
                if isinstance(init, (int, float)):
                    self.curve_chart.plot.plot(
                        xs, [float(init)] * len(xs),
                        pen=pg.mkPen(QColor(TEXT_SUB), width=2,
                                     style=Qt.DashLine),
                        name="不学习基线")
                n_fed = data.get("n_fed") or []
                laccs = data.get("learned_accs") or []
                flips = data.get("flip_rates") or []
                if n_fed and laccs:
                    fx = [float(v) for v in n_fed]
                    self.curve_chart.add_series(
                        "已反馈重测准确率（学过就会）", fx,
                        [float(v) for v in laccs], color=SUCCESS)
                if n_fed and flips:
                    self.curve_chart.add_series(
                        "错→对转化率", [float(v) for v in n_fed],
                        [float(v) for v in flips], color=WARNING)
                self.curve_chart.set_labels([str(s) for s in steps], "比率")
                desc = str(data.get("dataset_desc") or "")
                hint_txt = (
                    f"记录 #{data.get('eval_run_id', '-')}　{desc}。"
                    "多轮持续学习回放：每轮新图反馈、跨轮累积；"
                    "锚定集只评估不反馈（test 只验不选）。"
                    "注：错→对转化率口径＝累计『学习前判错』样本中当前被判对"
                    "的比例；分母随轮次单调累积、后期新错判更难翻案，曲线后段"
                    "下行属口径特性而非学习退化——看学习效果请结合锚定集 "
                    "AUROC 曲线与 F1")
                # 末轮锚定集混淆推导指标（algo 复用末次评估零额外前向）
                fm = data.get("final_metrics")
                if isinstance(fm, dict) and fm:
                    hint_txt += (
                        f"。末轮锚定集（阈值口径）：准确率 "
                        f"{_pct(fm.get('accuracy'))}，检出率(缺陷召回) "
                        f"{_pct(fm.get('recall_defect'))}，漏检率(FNR) "
                        f"{_pct(fm.get('fnr'))}，误报率(FPR) "
                        f"{_pct(fm.get('fpr'))}")
                self.lbl_curve_hint.setText(hint_txt)
            else:
                self.curve_chart.set_labels([], "比率")
                self.lbl_curve_hint.setText(
                    "暂无学习曲线记录：选择品类后点击「开始回放」生成")
        else:
            # 旧版单轮流记录（历史 EvalRun）
            self.kpi_learned.set_value(_fmt(data.get("a_recheck_auroc")),
                                       "A 再检 AUROC")
            self.kpi_fliprate.set_value("-")
            f1 = data.get("f1") or []
            if steps and auroc:
                self.curve_chart.add_series("AUROC", list(steps),
                                            [float(v) for v in auroc],
                                            color=PRIMARY)
                if f1:
                    self.curve_chart.add_series("F1", list(steps),
                                                [float(v) for v in f1],
                                                color=SUCCESS)
                self.curve_chart.set_labels([str(s) for s in steps],
                                            "AUROC / F1")
                self.lbl_curve_hint.setText(
                    f"记录 #{data.get('eval_run_id', '-')}（旧版单轮流回放）；"
                    "新一轮回放将使用多轮模式")
            else:
                self.curve_chart.set_labels([], "AUROC / F1")
                self.lbl_curve_hint.setText(
                    "暂无学习曲线记录：选择品类后点击「开始回放」生成")
        self._fill_evidence(data)

    def _fill_evidence(self, data: dict) -> None:
        """学习机制触发：algo 杠杆留痕（拦截/权重学习/阈值重估/头与判别器微调）。"""
        if not isinstance(data, dict) or not data.get("steps"):
            self.lbl_evidence.setText("")
            return
        parts: list[str] = []
        itc = data.get("intercept")
        if isinstance(itc, dict) and itc:
            parts.append(
                f"样例库拦截：{itc.get('n_samples', 0)}个缺陷样例"
                f"（含框选{itc.get('n_box_samples', 0)}），回放中命中"
                f"{itc.get('n_hits_total', 0)}次；锚定集缺陷侧"
                f"{itc.get('eval_boost_gt0_defect', 0)}/{itc.get('eval_defect_n', 0)}"
                "获得拦截加分")
        wl = data.get("weight_learn")
        if isinstance(wl, dict) and (wl.get("n_apply") or wl.get("n_reject")):
            parts.append(f"槽位权重学习：通过{wl.get('n_apply', 0)}次"
                         f"/否决{wl.get('n_reject', 0)}次")
        tr = data.get("thresh_recal")
        if isinstance(tr, dict) and (tr.get("n_recal") or tr.get("n_reject")):
            parts.append(f"阈值重估：{tr.get('n_recal', 0)}次"
                         f"（否决{tr.get('n_reject', 0)}）")
        ft_parts = []
        for key, name in (("head_ft", "异常头微调"), ("disc_ft", "判别器微调")):
            ft = data.get(key)
            if isinstance(ft, dict) and (ft.get("n_ft") or ft.get("n_reject")):
                ft_parts.append(f"{name}{ft.get('n_ft', 0)}次"
                                f"（否决{ft.get('n_reject', 0)}）")
        if ft_parts:
            parts.append("、".join(ft_parts))
        if data.get("box_supervised"):
            parts.append("含框选监督反馈")
        self.lbl_evidence.setText(
            "学习机制触发：" + "；".join(parts) if parts else
            "学习机制触发：本次回放未触发任何学习杠杆（反馈过少或全被门控否决）")

    def _fill_contribution(self, data) -> None:
        data = data if isinstance(data, dict) else {}
        ablation = data.get("ablation") if isinstance(
            data.get("ablation"), dict) else {}
        loo = ablation.get("leave_one_out") or {}
        single = ablation.get("single") or {}
        full = ablation.get("full")
        slots = sorted(set(loo) | set(single))
        if slots:
            self.ablation_chart.set_data(
                [SLOT_CN.get(s, s) for s in slots], {
                    "单槽位 AUROC": [float(single.get(s) or 0) for s in slots],
                    "剔除该槽位 AUROC": [float(loo.get(s) or 0) for s in slots],
                })
            # 融合对照：fused vs 最强单槽（回答"融合是否被弱槽稀释"，U31）
            best_slot, best_val = "", 0.0
            for s in slots:
                v = float(single.get(s) or 0)
                if v > best_val:
                    best_slot, best_val = s, v
            fusion_note = ""
            if isinstance(full, (int, float)) and best_slot:
                if best_val > full + 0.01:
                    fusion_note = (f"　⚠ 最强单槽 {SLOT_CN.get(best_slot, best_slot)}"
                                   f"={best_val:.3f} 高于融合 {full:.3f}，"
                                   "融合被弱槽稀释，关注权重学习收敛")
                else:
                    fusion_note = (f"　融合 {full:.3f} ≥ 最强单槽 "
                                   f"{SLOT_CN.get(best_slot, best_slot)}={best_val:.3f}，"
                                   "融合有效")
            self.lbl_ablation_hint.setText(
                f"全量融合 AUROC={_fmt(full)}　"
                "剔除后掉分越多说明该槽位越关键（记录 #"
                f"{data.get('eval_run_id', '-')}）{fusion_note}")
        else:
            self.ablation_chart.set_data([], {})
            self.lbl_ablation_hint.setText(
                "暂无贡献档案：点击「生成贡献档案」计算")
        self._fill_univariate(data.get("univariate"))
        self._fill_redundancy(data.get("redundancy"))

    def _fill_univariate(self, uni) -> None:
        uni = uni if isinstance(uni, dict) else {}
        overall = uni.get("overall") or {}
        by_type = uni.get("by_type") or {}
        slots = sorted(overall, key=lambda s: -float(overall.get(s) or 0))
        types = sorted(by_type)
        self.table_uni.clear()
        if not slots:
            self.table_uni.setRowCount(0)
            self.table_uni.setColumnCount(0)
            return
        self.table_uni.setColumnCount(1 + len(types))
        self.table_uni.setHorizontalHeaderLabels(["槽位 \\ 类型"] + types)
        self.table_uni.setRowCount(len(slots))
        for r, s in enumerate(slots):
            self.table_uni.setItem(r, 0, QTableWidgetItem(
                f"{SLOT_CN.get(s, s)}（总 {_fmt(overall.get(s))}）"))
            for c, t in enumerate(types):
                v = (by_type.get(t) or {}).get(s)
                it = QTableWidgetItem(_fmt(v))
                it.setTextAlignment(Qt.AlignCenter)
                # 该列最佳标绿（哪类缺陷靠哪个槽位一眼可见）
                col_vals = [float((by_type.get(t) or {}).get(x) or 0)
                            for x in slots]
                if isinstance(v, (int, float)) and col_vals \
                        and float(v) >= max(col_vals) and max(col_vals) > 0:
                    it.setForeground(QColor(SUCCESS))
                self.table_uni.setItem(r, c + 1, it)

    def _fill_redundancy(self, red) -> None:
        red = red if isinstance(red, dict) else {}
        names = [str(n) for n in (red.get("names") or [])]
        corr = red.get("corr") or []
        self.table_red.clear()
        if not names or not corr:
            self.table_red.setRowCount(0)
            self.table_red.setColumnCount(0)
            return
        n = len(names)
        self.table_red.setColumnCount(n)
        self.table_red.setRowCount(n)
        self.table_red.setHorizontalHeaderLabels(
            [SLOT_CN.get(x, x) for x in names])
        self.table_red.setVerticalHeaderLabels(
            [SLOT_CN.get(x, x) for x in names])
        for i in range(n):
            for j in range(n):
                try:
                    v = float(corr[i][j])
                except (TypeError, ValueError, IndexError):
                    v = 0.0
                it = QTableWidgetItem(f"{v:.2f}")
                it.setTextAlignment(Qt.AlignCenter)
                if i != j:
                    # 相关性强度底色：>0.8 深红（强冗余）→ 浅黄（中相关）
                    a = min(1.0, abs(v))
                    if a >= 0.8:
                        it.setBackground(QColor(239, 68, 68, 90))
                    elif a >= 0.5:
                        it.setBackground(QColor(245, 158, 11, 60))
                self.table_red.setItem(i, j, it)

    # ══════════════════ 动作 ══════════════════
    def _on_run_curve(self) -> None:
        cat = self._offline_cat()
        if not cat:
            warn(self, "请先在下拉选择回放品类（多品类工单需手动指定）")
            return
        idx = max(0, self.combo_preset.currentIndex())
        _name, rs, nr, ne, eps = REPLAY_PRESETS[idx]
        self.btn_run.setEnabled(False)

        def _started(res) -> None:
            if res and res.get("task_id"):
                self.progress.show()
                self.progress.update_task({
                    "status": "running", "progress": 0,
                    "message": "多轮学习回放中（离线副本，与产线并行、无需暂停）…"})
            else:
                self.btn_run.setEnabled(True)
        run_async(self, lambda: self._client.learning_curve(
            cat, mode="rounds", round_size=rs, n_rounds=nr,
            n_eval=ne, eval_per_sample=eps), _started)

    def _on_run_contribution(self) -> None:
        cat = self._offline_cat()
        if not cat:
            warn(self, "请先在下拉选择品类（多品类工单需手动指定）")
            return
        self.btn_contrib.setEnabled(False)

        def _started(res) -> None:
            if res and res.get("task_id"):
                self.progress.show()
                self.progress.update_task({
                    "status": "running", "progress": 0,
                    "message": "槽位贡献档案计算中…"})
            else:
                self.btn_contrib.setEnabled(True)
        run_async(self, lambda: self._client.learning_contribution(cat),
                  _started)

    def _on_task(self, task: dict) -> None:
        ttype = str(task.get("task_type", ""))
        if ttype not in ("learning_curve", "contribution", "flip_curve"):
            return
        self.progress.update_task(task)
        status = str(task.get("status"))
        if status in ("done", "success", "failed", "error", "cancelled"):
            self.btn_run.setEnabled(True)
            self.btn_contrib.setEnabled(True)
        if status in ("done", "success"):
            if ttype == "flip_curve":
                # 直接用任务结果渲染：不能重新 GET——反馈数仍 >50 会再次
                # 转后台造成死循环（反馈页既有教训）
                res = task.get("result")
                if isinstance(res, dict):
                    self._fill_flip_curve(res)
            else:
                self.reload()
