"""学习效果页（W-learn 2026-08-29 重构，docs/重构设计_学习与统计页.md）。

按"学习证据三层"组织，替代旧「离线回放为主 + 在线曲线为辅」结构：
- 学过就会（已学缺陷记住）：翻案曲线（从反馈页迁入）+ 离线回放的已反馈重测
- 举一反三（未见缺陷泛化）：离线多轮回放的锚定集 AUROC 曲线（含不学习基线）
- 线上生效（真实产线改善）：误判率趋势（FP/FN 率按批次）+ 权重演化 + 滚动判对率

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
    QCheckBox,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ui.api_client import ApiClient
from ui.pages.common import (SLOT_CN, attach_level_badge, make_banner,
                             make_card, make_card_title, make_scroll_area,
                             run_async, wrap_card_heads, warn)
from ui.theme import DANGER, PRIMARY, SUCCESS, TEXT_SUB, WARNING
from ui.widgets.charts import SERIES_COLORS, BarChart, LineChart
from ui.widgets.kpi_card import KpiCard
from ui.widgets.task_progress import TaskMonitor, TaskProgressBar

# 「线上学习证据」四张图的统一分带高度：四张图卡结构一致 ⇒ 绘图区等高且对齐
_TITLE_BAND = 58      # 标题带（主标题 + 灰色小字说明）
_CHART_H = 232        # 绘图区最小高度
_CHART_H_MAX = 300    # 绘图区最大高度（留出说明行的位置，避免说明被挤出视野）
_STATUS_BAND = 20     # 底部状态行（“暂无反馈记录”一类）


def _fmt(v, nd: int = 3) -> str:
    """指标容错格式化：None / 非数值显示 '-'。"""
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return "-"


def _num(v) -> float:
    """图表取值：缺失/非数值返回 NaN（图表不画），不伪造成 0。"""
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


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
        self.tabs.addTab(self._build_health_tab(), "槽位健康与更新审计")
        self.tabs.currentChanged.connect(self._on_tab_changed)
        root.addWidget(self.tabs, 1)

        # 卡片标题允许折行（长图题在窄卡片上不再被截断，前端反馈 2026-10-14）
        wrap_card_heads(self)
        # 离线 Tab 的图题固定高度：让下方"贡献档案"区一眼可见
        for ch in (self.curve_chart, self.ablation_chart):
            ch.setFixedHeight(170)
        for tb in (self.tbl_slots, self.tbl_audit, self.table_uni,
                   self.table_red, self.table_trad_groups):
            tb.setMinimumHeight(200)

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
            "线上学习证据：仅统计已产生的真实证据，不把 candidate_queued 算作已学习或已激活\n"
            "四图分别回答：已反馈样本是否翻案、整体误判率是否改善、权重如何演化、反馈流判对率如何变化"))
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
        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("权重证据来源"))
        self.combo_weight_source = QComboBox()
        self.combo_weight_source.addItem("生产模型（当前激活）", "production")
        self.combo_weight_source.addItem("最新已完成候选（独立学习证据）", "candidate")
        self.combo_weight_source.currentIndexChanged.connect(
            lambda _i: self._reload_weight_history())
        source_row.addWidget(self.combo_weight_source)
        self.lbl_weight_source = QLabel("仅权重图切换来源；其余三图始终为生产/真实反馈口径")
        self.lbl_weight_source.setWordWrap(True)
        self.lbl_weight_source.setProperty("subtext", True)
        source_row.addWidget(self.lbl_weight_source, 1)
        lay.addLayout(source_row)

        kpi_card = make_card()
        kpi_row = QHBoxLayout(kpi_card)
        self.kpi_flip = KpiCard("翻案率", "-", accent=SUCCESS)
        self.kpi_flip.setToolTip(
            "学习前判错、学习后判对的反馈占比（学过就会的直接证据）。\n"
            "口径注意：分母为全部有学习前判定记录的反馈（含本就判对的），"
            "反馈增多会稀释走低——翻案率下降不等于学习失效。")
        self.kpi_mis = KpiCard("错检数", "-", accent=DANGER)
        self.kpi_mistake = KpiCard("最新批次误判率", "-", accent=WARNING)
        self.kpi_mistake.setToolTip("最近一个有复核的批次，（误报+漏检）/复核数")
        self.kpi_weight = KpiCard("权重学习", "-", accent=PRIMARY)
        self.kpi_weight.setToolTip("融合权重写回通过次数 / 门控否决次数")
        for k in (self.kpi_flip, self.kpi_mis, self.kpi_mistake, self.kpi_weight):
            kpi_row.addWidget(k)
        lay.addWidget(kpi_card)

        grid = QGridLayout()
        grid.setSpacing(8)
        # 2026-10-14（前端反馈）：四张图必须"横轴水平对齐、纵轴比例一致"，
        # 不能出现某张图被单独拉伸。做法：
        #   第 0 行 = 四张图卡，结构完全一致（标题带定高 / 图定高 / 状态行定高），
        #            因此四张图的绘图区落在同一条带里、尺寸完全相同；
        #   第 1 行 = 各图正下方的"怎么看"灰色小字说明（文字移出图卡，不再挤占图的高度）；
        #   多余高度给第 1 行（文字顶部对齐），页面底部不会留空块。
        guides: list[str] = []

        def _add_chart_card(col: int, title: str, chart, status: QLabel,
                            guide: str) -> None:
            card = make_card()
            c_lay = QVBoxLayout(card)
            head = make_card_title(title)
            head.setFixedHeight(_TITLE_BAND)
            c_lay.addWidget(head)
            chart.setMinimumHeight(_CHART_H)
            chart.setMaximumHeight(_CHART_H_MAX)   # 上限：保证说明行始终在视野内
            c_lay.addWidget(chart)
            status.setProperty("subtext", True)
            status.setFixedHeight(_STATUS_BAND)
            # 状态行不参与列宽计算：长文本不顶宽本列（靠悬浮 tooltip 看全文）
            status.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
            c_lay.addWidget(status)
            grid.addWidget(card, 0, col)
            guides.append(guide)

        # 翻案曲线（学过就会）
        self.flip_chart = LineChart()
        self.lbl_flip_hint = QLabel("暂无反馈记录")
        _add_chart_card(
            0, "翻案曲线（灰=学习前判定分，蓝=当前引擎重判定分，绿点=已翻案）",
            self.flip_chart, self.lbl_flip_hint,
            "翻案曲线：绿点=学习前判错、学习后判对的样本；绿点持续出现=反馈正在被"
            "学会。长期无绿点说明学习未生效，应检查反馈质量")

        # 误判率趋势（按真实检测批次）
        self.trend_chart = LineChart()
        self.lbl_trend_hint = QLabel("暂无复核记录")
        _add_chart_card(
            1, "误判率趋势（按批次，复核口径；学习生效应持续下降）",
            self.trend_chart, self.lbl_trend_hint,
            "误判率趋势：学习生效应持续下降；与翻案曲线互相印证（一个看错转对、"
            "一个看整体错判率）")

        # 权重演化
        self.wh_chart = LineChart()
        self.lbl_wh_hint = QLabel("反馈累积后展示权重演化")
        _add_chart_card(
            2, "融合权重随反馈演化（x=带真值反馈数）",
            self.wh_chart, self.lbl_wh_hint,
            "权重演化：看各检测模块的融合权重如何随反馈数变化——上升的模块就是"
            "系统正在依赖的信号")

        # 滚动判对率（保留旧指标 + 选择偏差提示）
        self.online_chart = LineChart()
        _add_chart_card(
            3, "滚动判对率（x=反馈序号，y=最近 10 条判对率）",
            self.online_chart, QLabel(""),
            "滚动判对率：注意选择偏差——操作员专挑可疑图反馈，学习越好反馈越少、"
            "剩下的多是难例，判对率可能反而下降；学习是否生效请以误判率趋势与"
            "翻案曲线为准")

        for col, guide in enumerate(guides):
            lb = QLabel("· " + guide)
            lb.setProperty("subtext", True)
            lb.setWordWrap(True)
            lb.setAlignment(Qt.AlignTop)
            grid.addWidget(lb, 1, col)
        for _c in range(4):
            grid.setColumnStretch(_c, 1)
        # 多余高度按 1:1 分配：图卡行涨到上限后，剩余给说明行（文字顶部对齐），
        # 保证四张图始终对齐、说明始终可见、页面不留空块
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        lay.addLayout(grid)
        run_async(self, self._client.list_categories, self._fill_online_cats)
        return make_scroll_area(tab)

    # ══════════════════ Tab3：槽位健康与更新审计（AC-025/026） ══════════════════
    def _build_health_tab(self) -> QWidget:
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(0, 8, 0, 0)
        lay.setSpacing(8)
        bar = QHBoxLayout()
        bar.addWidget(make_banner(
            "槽位健康：各评分槽位的参照/校准样本量/体检 AUC/当前权重；"
            "更新审计：每轮学习前后锚定集误报率/灰区率、门控结论（零额外前向）"))
        bar.addStretch(1)
        btn = QPushButton("刷新")
        btn.setProperty("flat", True)
        btn.clicked.connect(self.reload_health)
        bar.addWidget(btn)
        lay.addLayout(bar)

        s_card = make_card()
        s_lay = QVBoxLayout(s_card)
        s_head = QLabel("槽位能力与状态")
        s_head.setProperty("heading", True)
        s_lay.addWidget(s_head)
        s_note = QLabel("校准分=正常分布分位，不是缺陷概率；槽位是专家评分、状态与更新单元，不等于特征组")
        s_note.setProperty("subtext", True)
        s_note.setWordWrap(True)
        s_lay.addWidget(s_note)
        self.tbl_slots = QTableWidget(0, 8)
        self.tbl_slots.setHorizontalHeaderLabels(
            ["槽位", "参照", "进融合", "校准样本", "体检AUC", "权重", "小样本", "状态"])
        self.tbl_slots.horizontalHeader().setStretchLastSection(True)
        s_lay.addWidget(self.tbl_slots, 1)
        self.lbl_health = QLabel("-")
        self.lbl_health.setProperty("subtext", True)
        self.lbl_health.setWordWrap(True)
        s_lay.addWidget(self.lbl_health)
        lay.addWidget(s_card, 1)

        a_card = make_card()
        a_lay = QVBoxLayout(a_card)
        a_head = QLabel("更新准入审计")
        a_head.setProperty("heading", True)
        a_lay.addWidget(a_head)
        a_note = QLabel("每轮记录快照 ID、操作、漂移、前后误报率/灰区率与结论；candidate_queued 仅是候选排队，不代表已学习或已激活")
        a_note.setProperty("subtext", True)
        a_note.setWordWrap(True)
        a_lay.addWidget(a_note)
        self.tbl_audit = QTableWidget(0, 7)
        self.tbl_audit.setHorizontalHeaderLabels(
            ["快照", "操作", "漂移", "误报率 前→后", "灰区率 前→后", "结论", "原因"])
        self.tbl_audit.horizontalHeader().setStretchLastSection(True)
        a_lay.addWidget(self.tbl_audit, 1)
        self.lbl_audit = QLabel("-")
        self.lbl_audit.setProperty("subtext", True)
        self.lbl_audit.setWordWrap(True)
        a_lay.addWidget(self.lbl_audit)
        lay.addWidget(a_card, 1)
        return make_scroll_area(tab)

    def reload_health(self) -> None:
        cat = str(self.combo_online_cat.currentData() or "") or self._category()
        if not cat:
            return
        run_async(self, lambda: self._client.slot_health(cat), self._fill_health)

    def _fill_health(self, data) -> None:
        data = data if isinstance(data, dict) else {}
        self.tbl_slots.setRowCount(0)
        self.tbl_audit.setRowCount(0)
        if not data.get("enabled"):
            self.lbl_health.setText("品类未准备或无权限（需工程师及以上）")
            self.lbl_audit.setText("-")
            return
        status_cn = {"ok": "正常", "weak": "偏弱", "reverse": "反向", "unverified": "未体检"}
        for name, r in (data.get("slots") or {}).items():
            row = self.tbl_slots.rowCount()
            self.tbl_slots.insertRow(row)
            vals = [SLOT_CN.get(name, name), str(r.get("reference", "")),
                    "是" if r.get("in_fusion") else "否（哨兵）", str(r.get("cal_n", 0)),
                    _fmt(r.get("sanity_auc")), _fmt(r.get("weight")),
                    "是" if r.get("small_sample") else "否",
                    status_cn.get(r.get("status"), str(r.get("status")))]
            for c, v in enumerate(vals):
                self.tbl_slots.setItem(row, c, QTableWidgetItem(v))
        ra = data.get("router_admission") or {}
        ds = data.get("domain_shift") or {}
        self.lbl_health.setText(
            f"{data.get('version', '')}  路由准入：{ra.get('decision', '-')}"
            f"（permitted 仅表示允许人工切换）；域差：{ds.get('status', '-')}")
        au = data.get("update_audit") or {}
        ent = list(au.get("entries") or [])[-50:]
        dec_cn = {"accepted": "通过", "rolled_back": "回滚", "rejected": "否决",
                  "applied_prior": "先验应用"}
        for e in reversed(ent):
            row = self.tbl_audit.rowCount()
            self.tbl_audit.insertRow(row)
            fpr = e.get("anchor_fpr") or {}
            gray = e.get("anchor_gray_rate") or {}
            vals = [str(e.get("snapshot_id", "")), str(e.get("op", "")),
                    _fmt(e.get("drift"), 4),
                    f"{_pct(fpr.get('before'))}→{_pct(fpr.get('after'))}",
                    f"{_pct(gray.get('before'))}→{_pct(gray.get('after'))}",
                    dec_cn.get(e.get("status"), str(e.get("status", "-"))),
                    str(e.get("rollback_reason", "") or "")]
            for c, v in enumerate(vals):
                self.tbl_audit.setItem(row, c, QTableWidgetItem(v))
        cnt = au.get("counts") or {}
        self.lbl_audit.setText(
            "通过 {a} / 回滚 {r} / 否决 {j} / 先验 {p}；门控前否决 {g} 次（不产生审计条目）"
            .format(a=cnt.get("accepted", 0), r=cnt.get("rolled_back", 0),
                    j=cnt.get("rejected", 0), p=cnt.get("applied_prior", 0),
                    g=au.get("pre_gate_rejects", 0)))

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
        self._reload_weight_history()
        run_async(self, lambda: self._client.online_curve(cat),
                  self._fill_online_curve)

    def _reload_weight_history(self) -> None:
        cat = str(self.combo_online_cat.currentData() or "") or self._category()
        if not cat:
            return
        source = self.combo_weight_source.currentData()
        self.wh_chart.clear()
        self.kpi_weight.set_value("-", "候选通过 / 否决" if source == "candidate"
                                  else "生产通过 / 否决")
        self.lbl_wh_hint.setText("正在读取权重证据…")
        self._weight_request = getattr(self, "_weight_request", 0) + 1
        request = self._weight_request

        def fill(data):
            if request == self._weight_request:
                self._fill_weight_history(data)

        fetch = (self._client.candidate_weight_history if source == "candidate"
                 else self._client.weight_history)
        run_async(self, lambda: fetch(cat), fill)

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
        caliber = ("口径：分母为全部有学习前判定记录的反馈（含本就判对的），"
                   "反馈增多会稀释走低——翻案率下降不等于学习失效")
        self.lbl_flip_hint.setText(
            f"反馈 {summary.get('total', 0)} 条"
            + (f"（其中 {legacy} 条老反馈无学习前判定记录）" if legacy else ""))
        self.lbl_flip_hint.setToolTip(caliber)

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
            f"{last.get('batch_name') or last.get('date', '')} 复核{last.get('reviewed', 0)}条")
        xs = list(range(len(items)))
        # 后端对无复核日回 0.0；按 reviewed==0 置缺失，避免画成"0 误报"
        def _rate(it, k):
            return _num(it.get(k)) if int(it.get("reviewed") or 0) > 0 \
                else float("nan")
        self.trend_chart.add_series(
            "误报率", xs, [_rate(it, "fp_rate") for it in items],
            color=WARNING)
        def _recall(it):
            value = it.get("recall_defect")
            if value is not None:
                return _num(value)
            fn_rate = it.get("fn_rate")
            return (1.0 - float(fn_rate)) if fn_rate is not None else float("nan")
        self.trend_chart.add_series(
            "检出率", xs, [_recall(it) for it in items],
            color=DANGER)
        labels = [str(it.get("batch_name") or it.get("date", "")) for it in items]
        self.trend_chart.set_labels(labels, "比率")
        self.lbl_trend_hint.setText(
            "误报率=FP/(FP+TN)；检出率=TP/(TP+FN)"
            "（无标签或分母为 0 的点留空，不按 0 计）")
        self.lbl_trend_hint.setToolTip(self.lbl_trend_hint.text())

    def _fill_weight_history(self, data) -> None:
        self.wh_chart.clear()
        if data is None:
            self.kpi_weight.set_value("-")
            self.lbl_wh_hint.setText("权重演化读取失败：后端不可用或请求未完成")
            return
        if not isinstance(data, dict):
            self.kpi_weight.set_value("-")
            self.lbl_wh_hint.setText("权重演化返回无效，未判定为无历史")
            return
        points = [p for p in (data.get("points") or []) if isinstance(p, dict)]
        candidate = data.get("source") == "candidate_snapshot"
        version = str(data.get("version") or "未找到")
        scope = "候选" if candidate else "生产"
        source_text = f"{scope} {version}"
        if candidate:
            source_text += (f"；生产仍为 {data.get('active_version') or '未激活'}；"
                            + ("该候选现已激活" if data.get("is_active") else "候选未激活")
                            + "，不是生产提升证据")
        self.lbl_weight_source.setText(
            source_text + "；仅权重图切换来源，其余三图仍为生产/真实反馈口径")
        n_apply = sum(1 for p in points if p.get("action") == "apply")
        n_reject = sum(1 for p in points
                       if str(p.get("action", "")).startswith("reject"))
        self.kpi_weight.set_value(f"{n_apply} / {n_reject}", f"{scope}通过 / 否决")
        if not points:
            self.lbl_wh_hint.setText(
                "暂无权重演化记录：反馈累积后展示"
                + ("" if data.get("enabled") else "（品类未准备）"))
            return
        slots = sorted({s for p in points
                        for s in (p.get("weights") or {})})
        if not slots:
            self.lbl_wh_hint.setText("已有权重快照但未返回可绘制槽位")
            return
        xs = [int(p.get("n_fb", 0)) for p in points]
        for i, s in enumerate(slots):
            ys = [_num((p.get("weights") or {}).get(s)) for p in points]
            self.wh_chart.add_series(SLOT_CN.get(s, s), xs, ys,
                                     color=SERIES_COLORS[i % len(SERIES_COLORS)],
                                     symbol=True)
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
        self.wh_chart.set_labels(y_title="权重")
        self.wh_chart.plot.getAxis("bottom").setTicks(
            [list(dict.fromkeys((x, str(x)) for x in xs))])
        initial_only = all(p.get("action") == "initial" for p in points)
        hint = (f"{source_text}；真实历史 {len(points)} 点"
                + ("（仅初始权重，尚无权重更新）" if initial_only else ""))
        self.lbl_wh_hint.setText(hint)
        self.lbl_wh_hint.setToolTip(hint)

    def _fill_online_curve(self, data) -> None:
        data = data if isinstance(data, dict) else {}
        points = data.get("points") or []
        self.online_chart.clear()
        if points:
            ks = [float(p.get("k", i + 1)) for i, p in enumerate(points)]
            acc = [_num(p.get("rolling_acc")) for p in points]
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
        # 2026-10-14（前端反馈）：这行原来塞了长横幅 + 长提示 + 两个下拉 + 三个按钮，
        # 窄屏会被挤出边界。横幅改短、长说明移到 tooltip，并去掉重复的提示行。
        _off_banner = make_banner("离线回放验证（模拟预演：反馈真值来自数据标注）")
        _off_banner.setToolTip(
            "离线回放 = 模拟预演：反馈真值来自数据标注（非真实操作员），用于预演学习能力。\n"
            "回放在独立副本上进行，与产线并行、无需暂停产线。\n"
            "真实产线学习证据见「线上学习证据」页签")
        top.addWidget(_off_banner)
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
        c_lay.addWidget(make_card_title(
            "学习曲线（x=累计反馈数/轮次；y=指标值：锚定集 AUROC、已反馈重测准确率等）"))
        curve_axis_note = QLabel(
            "离线回放数据来自标注数据的独立副本：多轮模式 x=累计反馈数（曲线也显示轮次），"
            "旧版单轮 x=历史评估步；y 为 AUROC/F1 或重测指标，不是在线产线指标。")
        curve_axis_note.setProperty("subtext", True)
        curve_axis_note.setWordWrap(True)
        c_lay.addWidget(curve_axis_note)
        # 前端反馈 2026-09-13 #12：补"怎么看"读法
        c_read = QLabel(
            "读图：x 轴是累计反馈数（多轮模式显示轮次），y 轴是各线对应指标值；"
            "「举一反三」=锚定集 AUROC，随反馈累积上升表示推广到未见样本；"
            "「学过就会」=已反馈样本重测准确率，接近 1.0 属正常。"
            "两条线长期持平不动=反馈没带来增益，应检查反馈是否准确")
        c_read.setProperty("subtext", True)
        c_read.setWordWrap(True)
        c_lay.addWidget(c_read)
        self.curve_chart = LineChart()
        self.curve_chart.set_labels([], "比率")
        c_lay.addWidget(self.curve_chart)
        self.lbl_curve_hint = QLabel(
            "暂无学习曲线记录：选择品类后点击「开始回放」生成")
        self.lbl_curve_hint.setProperty("subtext", True)
        self.lbl_curve_hint.setWordWrap(True)
        c_lay.addWidget(self.lbl_curve_hint)
        self.lbl_evidence = QLabel("")
        self.lbl_evidence.setProperty("subtext", True)
        self.lbl_evidence.setWordWrap(True)
        c_lay.addWidget(self.lbl_evidence)
        # 学习曲线卡固定高度（图 170px），把余下空间全部让给下方"贡献档案"区，
        # 避免曲线图占满一屏、看不到下面还有内容（前端反馈 2026-10-14）
        off.addWidget(curve_card)

        # 主动实验区：以下档案由用户点击“生成贡献档案”计算，不是持续学习自动产物。
        # 槽位是评分/状态/更新单元；传统特征组只是 trad 输入，二者不等价。
        explain = QLabel(
            "主动实验（手动生成贡献档案）：槽位消融、单槽位×缺陷类型、槽位相关矩阵用于解释评分槽位的独立性/冗余；"
            "传统特征组贡献只分析 trad 输入组。它们不会因持续学习自动产生，也不会直接改生产。")
        explain.setProperty("subtext", True)
        explain.setWordWrap(True)
        off.addWidget(explain)
        ab_card = make_card()
        a_lay = QVBoxLayout(ab_card)
        a_lay.addWidget(make_card_title("槽位消融（单槽位 / 剔除该槽位 AUROC）"))
        # 前端反馈 2026-09-13 #12：补"怎么看"读法
        a_read = QLabel(
            "怎么看：每个检测模块两根柱——「单用」=它独自的判别力，「剔除」=去掉"
            "它之后整体还剩多少。剔除后明显下降=该模块有独立贡献不能裁；"
            "剔除后几乎不降=冗余，可裁剪提速")
        a_read.setProperty("subtext", True)
        a_read.setWordWrap(True)
        a_lay.addWidget(a_read)
        self.ablation_chart = BarChart()
        a_lay.addWidget(self.ablation_chart)
        self.lbl_ablation_hint = QLabel(
            "暂无贡献档案：点击「生成贡献档案」计算")
        self.lbl_ablation_hint.setProperty("subtext", True)
        self.lbl_ablation_hint.setWordWrap(True)
        a_lay.addWidget(self.lbl_ablation_hint)

        # 单槽位 × 缺陷类型 AUROC
        uni_card = make_card()
        u_lay = QVBoxLayout(uni_card)
        u_lay.addWidget(make_card_title(
            "单槽位 × 缺陷类型 AUROC（哪类缺陷靠哪个槽位）"))
        self.table_uni = QTableWidget(0, 0)
        self.table_uni.setEditTriggers(QTableWidget.NoEditTriggers)
        u_lay.addWidget(self.table_uni, 1)

        # 槽位相关矩阵
        red_card = make_card()
        r_lay = QVBoxLayout(red_card)
        r_lay.addWidget(make_card_title(
            "槽位相关矩阵（Spearman；高相关=冗余可裁剪提速）"))
        self.table_red = QTableWidget(0, 0)
        self.table_red.setEditTriggers(QTableWidget.NoEditTriggers)
        r_lay.addWidget(self.table_red, 1)

        # 传统特征组贡献
        tg_card = make_card()
        g_lay = QVBoxLayout(tg_card)
        g_lay.addWidget(make_card_title(
            "传统特征组贡献（AC-012：单组/留一/排列/冗余/耗时；非缺陷概率）"))
        self.lbl_trad_groups = QLabel("暂无组级档案")
        self.lbl_trad_groups.setProperty("subtext", True)
        self.lbl_trad_groups.setWordWrap(True)
        g_lay.addWidget(self.lbl_trad_groups)
        self.table_trad_groups = QTableWidget(0, 0)
        self.table_trad_groups.setEditTriggers(QTableWidget.NoEditTriggers)
        g_lay.addWidget(self.table_trad_groups, 1)

        # 2026-10-14（前端反馈）：原来是一行四列的 QSplitter（合计最小宽约 1420px），
        # 窄屏/中档窗口必然出现明显的水平滚动；改为**两行两列网格**，列宽自适应，
        # 最大化与默认窗口下都不再需要横向滚动。
        g2 = QGridLayout()
        g2.setSpacing(8)
        g2.addWidget(ab_card, 0, 0)
        g2.addWidget(uni_card, 0, 1)
        g2.addWidget(red_card, 1, 0)
        g2.addWidget(tg_card, 1, 1)
        for _c in range(2):
            g2.setColumnStretch(_c, 1)
        g2.setRowStretch(0, 1)
        g2.setRowStretch(1, 1)
        off.addLayout(g2, 1)
        off.addWidget(self._build_feature_set_card())

        run_async(self, self._client.list_categories, self._fill_offline_cats)
        return make_scroll_area(tab)

    # ══════════════════ 当前特征集（AC-016） ══════════════════
    _FS_REPORT_CN = {
        "current": ("档案最新", SUCCESS),
        "stale": ("档案过期（特征集/模型已变更）", WARNING),
        "not_generated": ("档案未生成", WARNING),
        "insufficient": ("数据不足（需测试集同时含正常与缺陷）", DANGER),
    }

    def _build_feature_set_card(self) -> QWidget:
        card = make_card()
        lay = QVBoxLayout(card)
        head = make_card_title("当前特征集（AC-016：启停只提交候选，不直接改生产）")
        lay.addWidget(head)
        self.lbl_fs_status = QLabel("未加载")
        self.lbl_fs_status.setProperty("subtext", True)
        self.lbl_fs_status.setWordWrap(True)
        lay.addWidget(self.lbl_fs_status)
        self._fs_grid = QGridLayout()
        lay.addLayout(self._fs_grid)
        self._fs_checks: dict[str, QCheckBox] = {}
        row = QHBoxLayout()
        row.addWidget(QLabel("组合评分"))
        self.combo_fs_scoring = QComboBox()
        for s, t in (("legacy", "legacy（整体）"), ("grouped", "grouped（分组）")):
            self.combo_fs_scoring.addItem(t, s)
        row.addWidget(self.combo_fs_scoring)
        row.addStretch(1)
        self.btn_fs_draft = QPushButton("提交候选")
        self.btn_fs_draft.clicked.connect(self._on_fs_draft)
        row.addWidget(self.btn_fs_draft)
        self.btn_fs_report = QPushButton("生成特征档案")
        self.btn_fs_report.clicked.connect(self._on_fs_report)
        row.addWidget(self.btn_fs_report)
        self.btn_fs_rebuild = QPushButton("表示重建")
        self.btn_fs_rebuild.setToolTip(
            "AC-015：重提特征→重建库→重校准，产出未激活新版本，须过 G2 门控后激活")
        self.btn_fs_rebuild.clicked.connect(self._on_fs_rebuild)
        self.btn_fs_rebuild.setEnabled(False)
        row.addWidget(self.btn_fs_rebuild)
        self.btn_fs_admission = QPushButton("难例准入试算")
        self.btn_fs_admission.setToolTip(
            "AC-017：用漏检/误报/冲突/漂移/open 难例池独立试算最近提交的候选，"
            "评估增益/冗余/回归/G1/G2；只产出证据，不注册不激活")
        self.btn_fs_admission.clicked.connect(self._on_fs_admission)
        self.btn_fs_admission.setEnabled(False)
        row.addWidget(self.btn_fs_admission)
        lay.addLayout(row)
        self._fs_data: dict = {}
        self._fs_last_rid: str = ""
        return card

    def _fill_feature_set(self, data) -> None:
        data = data if isinstance(data, dict) else {}
        self._fs_data = data
        while self._fs_grid.count():
            w = self._fs_grid.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        self._fs_checks = {}
        status = str(data.get("status") or "")
        can_edit = bool(data.get("can_edit"))
        if status != "available":
            msg = {"not_applicable": "当前模型无传统特征槽位，不适用"}.get(
                status, "未加载模型或品类" if not data else f"状态：{status}")
            self.lbl_fs_status.setText(msg)
            self.btn_fs_draft.setEnabled(False)
            self.btn_fs_report.setEnabled(False)
            self.btn_fs_rebuild.setEnabled(False)
            self.btn_fs_admission.setEnabled(False)
            return
        rep = data.get("report") or {}
        r_txt, r_col = self._FS_REPORT_CN.get(
            str(rep.get("status")), (str(rep.get("status") or "-"), TEXT_SUB))
        if rep.get("running_task_id"):
            r_txt += "（生成中）"
        role_txt = "" if can_edit else "；仅工程师可修改配置/生成档案"
        rebuild_txt = ""
        if data.get("rebuild_required"):
            drift = ",".join((data.get("feature_representation") or {}).get("drift") or [])
            rebuild_txt = (f"<br><span style='color:{DANGER}'>特征表示已过期（{drift}）："
                           "在线学习/巩固已暂停，需表示重建</span>")
        if data.get("pending_rebuild_version"):
            rebuild_txt += (f"<br><span style='color:{WARNING}'>已重建 "
                            f"v{data['pending_rebuild_version']}，待激活门控（G2）</span>")
        self.lbl_fs_status.setText(
            f"模型 v{data.get('version')}｜启用 {len(data.get('enabled_groups') or [])}"
            f"/{len(data.get('groups') or [])} 组，{data.get('enabled_dim')}"
            f"/{data.get('total_dim')} 维｜评分 {data.get('scoring')}｜"
            f"<span style='color:{r_col}'>{r_txt}</span>{role_txt}{rebuild_txt}")
        self.lbl_fs_status.setTextFormat(Qt.RichText)
        for i, g in enumerate(data.get("groups") or []):
            cb = QCheckBox(f"{g.get('name') or g.get('key')}（{g.get('dim')}维）")
            cb.setChecked(bool(g.get("enabled")))
            cb.setToolTip(f"{g.get('key')}｜{g.get('category')}｜权重 "
                          f"{_fmt(g.get('weight'), 2)}")
            cb.setEnabled(can_edit)
            self._fs_grid.addWidget(cb, i // 5, i % 5)
            self._fs_checks[str(g.get("key"))] = cb
        idx = self.combo_fs_scoring.findData(str(data.get("scoring") or "legacy"))
        if idx >= 0:
            self.combo_fs_scoring.setCurrentIndex(idx)
        self.combo_fs_scoring.setEnabled(can_edit)
        self.btn_fs_draft.setEnabled(can_edit)
        self.btn_fs_report.setEnabled(
            can_edit and rep.get("status") != "insufficient"
            and not rep.get("running_task_id"))
        self.btn_fs_rebuild.setEnabled(can_edit)

    def _on_fs_rebuild(self) -> None:
        cat = self._offline_cat()
        if not cat:
            return
        self.btn_fs_rebuild.setEnabled(False)

        def _started(res) -> None:
            if isinstance(res, dict) and res.get("task_id"):
                self.lbl_fs_status.setText(
                    f"表示重建已提交（父版本 v{res.get('parent_version')}），"
                    "新版本不自动激活，需过 G2 门控后在版本页激活")
                self.progress.show()
                self.progress.update_task({
                    "status": "running", "progress": 0, "message": "表示重建中…"})
            else:
                self.btn_fs_rebuild.setEnabled(True)
        run_async(self, lambda: self._client.feature_set_rebuild(cat), _started)

    def _on_fs_admission(self) -> None:
        rid = self._fs_last_rid
        if not rid:
            warn(self, "请先提交候选特征集")
            return
        self.btn_fs_admission.setEnabled(False)

        def _started(res) -> None:
            if isinstance(res, dict) and res.get("task_id"):
                self.lbl_fs_status.setText(
                    f"候选 {rid[:12]} 难例准入试算已提交（只产出证据，不激活；"
                    "结果见 feature_revisions/admission_*.json）")
                self.progress.show()
                self.progress.update_task({
                    "status": "running", "progress": 0, "message": "难例准入试算中…"})
            self.btn_fs_admission.setEnabled(True)
        run_async(self, lambda: self._client.feature_admission(rid), _started)

    def _on_fs_draft(self) -> None:
        cat = self._offline_cat()
        groups = [k for k, cb in self._fs_checks.items() if cb.isChecked()]
        if not cat or not groups:
            warn(self, "请至少保留一个特征组")
            return
        scoring = str(self.combo_fs_scoring.currentData() or "legacy")

        def _done(res) -> None:
            if isinstance(res, dict) and res.get("revision_id"):
                self._fs_last_rid = str(res["revision_id"])
                self.btn_fs_admission.setEnabled(True)
                self.lbl_fs_status.setText(
                    f"已提交候选 {res['revision_id'][:12]}（draft，未改生产；"
                    "需经验证门禁后激活）")
        run_async(self, lambda: self._client.feature_set_draft(
            cat, groups, scoring), _done)

    def _on_fs_report(self) -> None:
        cat = self._offline_cat()
        if not cat:
            return
        self.btn_fs_report.setEnabled(False)

        def _started(res) -> None:
            if isinstance(res, dict) and res.get("task_id"):
                self.progress.show()
                self.progress.update_task({
                    "status": "running", "progress": 0,
                    "message": "特征档案生成中…"})
            else:
                self.btn_fs_report.setEnabled(True)
        run_async(self, lambda: self._client.feature_set_report(cat), _started)

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
        elif idx == 2:
            self.reload_health()
        else:
            self.reload()

    def reload(self) -> None:
        cat = self._offline_cat()
        if not cat:
            self._fill_curve(None)
            self._fill_contribution(None)
            self._fill_feature_set(None)
            return
        run_async(self, lambda: self._client.learning_curve_latest(cat),
                  self._fill_curve)
        run_async(self, lambda: self._client.feature_set(cat),
                  self._fill_feature_set)
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
                # 后端当前只在末轮锚定集提供混淆矩阵计数；仅绘制末轮真实值，
                # 不用错误总率反推，也不把缺失轮次填成 0。
                fm = data.get("final_metrics")
                if steps and isinstance(fm, dict):
                    recall = _num(fm.get("recall_defect"))
                    fpr = _num(fm.get("fpr"))
                    self.curve_chart.add_series(
                        "末轮检出率 TP/(TP+FN)", [float(v) for v in steps],
                        [float("nan")] * max(len(steps) - 1, 0) + [recall],
                        color=SUCCESS)
                    self.curve_chart.add_series(
                        "末轮误报率 FP/(FP+TN)", [float(v) for v in steps],
                        [float("nan")] * max(len(steps) - 1, 0) + [fpr],
                        color=DANGER)
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
                        f"{_pct(fm.get('accuracy'))}，检出率 TP/(TP+FN) "
                        f"{_pct(fm.get('recall_defect'))}，误报率 FP/(FP+TN) "
                        f"{_pct(fm.get('fpr'))}；分母为 0 时显示未知（-），不以 1-错误总率替代。")
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
                    "单槽位 AUROC": [_num(single.get(s)) for s in slots],
                    "剔除该槽位 AUROC": [_num(loo.get(s)) for s in slots],
                })
            # 融合对照：fused vs 最强单槽（回答"融合是否被弱槽稀释"，U31）
            best_slot, best_val = "", 0.0
            for s in slots:
                v = _num(single.get(s))
                if v == v and v > best_val:  # NaN 跳过
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
        self._fill_trad_groups(data.get("trad_groups"))

    TRAD_GROUP_COLS = (("组", None), ("启用", "enabled"), ("单组AUROC", "single_auroc"),
                       ("95%CI", "single_ci95"), ("方向", "direction"),
                       ("固定留一↓", "fixed_loo_drop"), ("重建留一↓", "rebuild_loo_drop"),
                       ("排列↓", "permutation_drop"), ("最相关组", "redundancy"),
                       ("耗时ms", "extract_ms"))

    def _fill_trad_groups(self, tg) -> None:
        tg = tg if isinstance(tg, dict) else {}
        groups = tg.get("groups") or {}
        self.table_trad_groups.clear()
        if tg.get("status") != "available" or not groups:
            self.table_trad_groups.setRowCount(0)
            self.table_trad_groups.setColumnCount(0)
            self.lbl_trad_groups.setText(
                f"组级档案：{tg.get('status', '未生成')} {tg.get('reason') or ''}")
            return
        meta = tg.get("meta") or {}
        clusters = [c for c in (tg.get("cluster_loo") or []) if len(c.get("groups") or []) > 1]
        cl_txt = "；".join(f"{'+'.join(c['groups'])} 簇留一↓{_fmt(c.get('drop'))}"
                           for c in clusters) or "无高相关簇"
        self.lbl_trad_groups.setText(
            f"trad 全组 AUROC={_fmt(tg.get('full_auroc'))} CI={tg.get('full_ci95')}　"
            f"版本 v{meta.get('model_version', '-')} 品类 {meta.get('category') or '-'} "
            f"视图 {tg.get('view')}　正常/缺陷 {meta.get('n_normal')}/{meta.get('n_defect')}　"
            f"{cl_txt}")
        cols = self.TRAD_GROUP_COLS
        keys = sorted(groups, key=lambda k: -float(groups[k].get("single_auroc") or 0))
        self.table_trad_groups.setColumnCount(len(cols))
        self.table_trad_groups.setHorizontalHeaderLabels([c[0] for c in cols])
        self.table_trad_groups.setRowCount(len(keys))
        for r, k in enumerate(keys):
            g = groups[k]
            for c, (_, f) in enumerate(cols):
                if f is None:
                    txt = f"{g.get('name', k)}（{k}）"
                elif f == "enabled":
                    txt = "是" if g.get(f) else "否"
                elif f == "single_ci95":
                    ci = g.get(f)
                    txt = f"[{ci[0]:.3f},{ci[1]:.3f}]" if ci else "-"
                elif f == "redundancy":
                    red = g.get(f)
                    txt = f"{red[0]} {_fmt(red[1])}" if red else "-"
                elif f == "direction":
                    txt = {"reversed": "反向", "normal": "正向"}.get(g.get(f), "-")
                else:
                    txt = _fmt(g.get(f))
                it = QTableWidgetItem(txt)
                if f == "direction" and g.get(f) == "reversed":
                    it.setForeground(QColor(WARNING))
                self.table_trad_groups.setItem(r, c, it)

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
            self.btn_fs_report.setEnabled(bool(self._fs_data.get("can_edit")))
        if status in ("done", "success"):
            if ttype == "flip_curve":
                # 直接用任务结果渲染：不能重新 GET——反馈数仍 >50 会再次
                # 转后台造成死循环（反馈页既有教训）
                res = task.get("result")
                if isinstance(res, dict):
                    self._fill_flip_curve(res)
            else:
                self.reload()
