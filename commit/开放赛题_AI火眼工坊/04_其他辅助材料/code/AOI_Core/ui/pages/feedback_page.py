"""标注反馈页：反馈摘要（工单口径）+ 各品类在线学习状态 + 反馈记录 + 待复核队列。

上：左右结构——左 KPI 卡片 2x2（反馈数/错检占比/待复核/已复核，当前工单口径），
   右各品类在线学习状态（当前版本 / 未巩固反馈数 / 引擎内部；巩固统一在监控页执行）
下：Tab（反馈记录 | 待复核 | 错检追溯）
    反馈记录：类型/品类筛选的反馈流水（服务端分页，统一 PagerBar；双击行追溯）
    待复核（M5 灰区闭环 + v4 复判工单全量复核）：仅产线路径检测进队列
    （试检旁路不进），表格含系统判定列 + 判正常/判缺陷一键复核（即学）
    错检追溯（M7a）：被判错（误检/漏检/新缺陷）的检测清单，
    双击行查看单帧追溯（判定现场→学习后重判定，体现错检对学习的提升）
"""
from __future__ import annotations

import json

import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
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
from ui.pages.common import (
    PagerBar,
    attach_level_badge,
    load_thumb_async,
    make_banner,
    make_card,
    make_thumb_cell,
    notify,
    run_async,
    slot_text,
    version_text,
    warn,
)
from ui.pages.model_page import ORIGIN_NAMES
from ui.theme import DANGER, PRIMARY, SUCCESS, TEXT_SUB, WARNING
from ui.widgets.charts import DonutChart
from ui.widgets.feedback_dialog import FeedbackDialog
from ui.widgets.image_viewer import ImageViewer
from ui.widgets.kpi_card import KpiCard
from ui.widgets.task_progress import TaskMonitor, TaskProgressBar

FEEDBACK_TYPE_NAMES = {
    "false_positive": "误检",
    "false_negative": "漏检",
    "confirmed": "确认正确",
    "new_defect": "新缺陷",
    "review": "复核",
    "uncertain": "无法确认",
}

DECISION_CN = {"normal": "正常", "gray": "灰区", "anomaly": "异常"}

# 反馈记录 Tab 类型筛选（值=后端 feedback_type）
FB_TYPE_FILTERS = [("全部类型", ""), ("误检", "false_positive"),
                   ("漏检", "false_negative"), ("新缺陷", "new_defect"),
                   ("确认正确", "confirmed"), ("人工复核", "review")]


class TraceDialog(QDialog):
    """单帧追溯详情（M7b）：大图 + 判定现场 + 当前引擎重判定 + 反馈时间线。"""

    def __init__(self, client: ApiClient, trace: dict,
                 parent: QWidget | None = None):
        super().__init__(parent)
        trace = trace if isinstance(trace, dict) else {}
        det = trace.get("detection") if isinstance(trace.get("detection"), dict) else {}
        model = trace.get("model") if isinstance(trace.get("model"), dict) else None
        feedbacks = [f for f in (trace.get("feedbacks") or [])
                     if isinstance(f, dict)]
        post = trace.get("post") if isinstance(trace.get("post"), dict) else None

        self.setWindowTitle(f"追溯详情 · 检测 #{det.get('id', '-')}")
        self.resize(980, 620)
        root = QHBoxLayout(self)

        # ── 左：大图（叠加图优先）──
        self.viewer = ImageViewer()
        root.addWidget(self.viewer, 3)
        path = det.get("overlay_path") or det.get("image_path") or ""
        if path:
            self.viewer.set_image_url(client.file_url(path))

        # ── 右：信息区 + 反馈时间线 ──
        right = QVBoxLayout()
        tiles = det.get("n_tiles")
        if isinstance(tiles, str):
            try:
                tiles = json.loads(tiles)
            except Exception:  # noqa: BLE001
                tiles = {}
        tiles = tiles if isinstance(tiles, dict) else {}
        slots = tiles.get("slots") if isinstance(tiles.get("slots"), dict) else {}
        slots_txt = slot_text(slots)
        decision = det.get("decision") or tiles.get("decision")
        try:
            score_txt = f"{float(det.get('final_score')):.3f}"
        except (TypeError, ValueError):
            score_txt = "-"
        created = str(det.get("created_at") or "").replace("T", " ")[:19]
        if model:
            origin = str(model.get("origin") or "")
            model_txt = (f"{version_text(model.get('version'))}"
                         f"（{ORIGIN_NAMES.get(origin, origin or '-')}）")
        else:
            model_txt = "未记录（检测落库未写 model_id）"
        info = QLabel(
            f"检测时间：{created or '-'}\n"
            f"品类：{det.get('category', '-')}\n"
            f"当时模型：{model_txt}\n"
            f"当时判定：{DECISION_CN.get(str(decision), decision or '-')}"
            f"　分数 {score_txt}　评分依据：{slots_txt}")
        info.setWordWrap(True)
        right.addWidget(info)

        # 当前引擎重判定 + 翻案标注
        self.lbl_post = QLabel()
        self.lbl_post.setWordWrap(True)
        if post:
            try:
                post_score = f"{float(post.get('score')):.3f}"
            except (TypeError, ValueError):
                post_score = "-"
            post_dec = str(post.get("decision") or "")
            verdict, color = self._flip_verdict(post_dec, feedbacks)
            self.lbl_post.setText(
                f"当前引擎重判定：{DECISION_CN.get(post_dec, post_dec or '-')}"
                f"　分数 {post_score}　{verdict}")
            self.lbl_post.setStyleSheet(f"color:{color}; font-weight:bold;")
        else:
            self.lbl_post.setText(
                "当前引擎重判定：暂不可用（无反馈前判定记录或引擎未加载该品类）")
            self.lbl_post.setProperty("subtext", True)
        right.addWidget(self.lbl_post)

        head = QLabel("反馈时间线")
        head.setProperty("heading", True)
        right.addWidget(head)
        self.table_fb = QTableWidget(0, 5)
        self.table_fb.setHorizontalHeaderLabels(
            ["时间", "反馈类型", "人工标签", "框选", "反馈前分数"])
        self.table_fb.horizontalHeader().setStretchLastSection(True)
        self.table_fb.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table_fb.setAlternatingRowColors(True)
        if feedbacks:
            self.table_fb.setRowCount(len(feedbacks))
            for r, fb in enumerate(feedbacks):
                at = str(fb.get("created_at") or "").replace("T", " ")[:19]
                self.table_fb.setItem(r, 0, QTableWidgetItem(at or "-"))
                ftype = str(fb.get("feedback_type") or "")
                ftype_txt = FEEDBACK_TYPE_NAMES.get(ftype, ftype or "-")
                if fb.get("invalidated"):
                    ftype_txt += "（已作废）"
                self.table_fb.setItem(r, 1, QTableWidgetItem(ftype_txt))
                self.table_fb.setItem(r, 2, QTableWidgetItem(
                    {0: "正常", 1: "异常"}.get(fb.get("operator_label"), "-")))
                self.table_fb.setItem(r, 3, QTableWidgetItem(
                    "有" if fb.get("box") else "无"))
                pre = fb.get("pre") if isinstance(fb.get("pre"), dict) else None
                pre_score = pre.get("score") if pre else None
                self.table_fb.setItem(r, 4, QTableWidgetItem(
                    f"{float(pre_score):.3f}"
                    if isinstance(pre_score, (int, float)) else "-（老反馈）"))
        else:
            self.table_fb.setRowCount(1)
            self.table_fb.setSpan(0, 0, 1, 5)
            self.table_fb.setItem(0, 0, QTableWidgetItem("该检测暂无反馈记录"))
        right.addWidget(self.table_fb, 1)

        btn_close = QPushButton("关闭")
        btn_close.setProperty("primary", True)
        btn_close.clicked.connect(self.accept)
        right.addWidget(btn_close, 0, Qt.AlignRight)
        root.addLayout(right, 2)

    @staticmethod
    def _flip_verdict(post_decision: str, feedbacks: list) -> tuple[str, str]:
        """当前重判定 vs 最近一条人工结论 → (翻案标注文本, 颜色)。"""
        label = None
        for fb in reversed(feedbacks):
            if fb.get("operator_label") in (0, 1):
                label = fb["operator_label"]
                break
        if label is None:
            return "（无人工结论可比对）", TEXT_SUB
        expect = "anomaly" if label == 1 else "normal"
        if post_decision == expect:
            return "翻案 ✓（已纠正为人工结论）", SUCCESS
        return "未翻案 ✗（仍与人工结论不一致）", DANGER


class FeedbackPage(QWidget):
    """标注反馈页。"""

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

        top = QHBoxLayout()
        top.addWidget(make_banner("反馈驱动：操作员标注 -> 引擎在线即学 -> 巩固生成新版本"))
        attach_level_badge(self, top)   # U-workorder：数据条件徽标
        top.addStretch(1)
        btn_refresh = QPushButton("刷新")
        btn_refresh.clicked.connect(self.reload)
        top.addWidget(btn_refresh)
        root.addLayout(top)

        # M14c：不完备报告横幅（open 哨兵持续偏高 → 体系外缺陷预警）
        self.banner_incomplete = make_banner("")
        self.banner_incomplete.setStyleSheet(
            "background:#7F1D1D; color:white; padding:6px 10px; border-radius:4px;")
        self.banner_incomplete.hide()
        root.addWidget(self.banner_incomplete)

        # ── 上：KPI 卡片 + 各品类在线学习状态（左右结构，2026-08-29 调整：
        #    原为上下堆叠，状态表偏挤；KPI 改 2x2 栅格让出横向空间）──
        upper_row = QHBoxLayout()
        kpi_card = make_card()
        kpi_lay = QVBoxLayout(kpi_card)
        self.lbl_scope = QLabel("统计口径：全部工单")
        self.lbl_scope.setProperty("subtext", True)
        self.lbl_scope.setAlignment(Qt.AlignCenter)
        kpi_lay.addWidget(self.lbl_scope)
        kpi_grid = QGridLayout()
        self.kpi_total = KpiCard("反馈数", "-")
        self.kpi_rate = KpiCard("错检占比", "-", accent=DANGER)
        self.kpi_pending = KpiCard("待复核", "-", accent=WARNING)
        self.kpi_reviewed = KpiCard("已复核", "-", accent=SUCCESS)
        self.kpi_rate.setToolTip("反馈中被判错（误检+漏检）的比例，非产线误判率")
        self.kpi_pending.setToolTip("复核队列当前积压（全部工单品类合计）")
        for i, k in enumerate((self.kpi_total, self.kpi_rate,
                               self.kpi_pending, self.kpi_reviewed)):
            kpi_grid.addWidget(k, i // 2, i % 2)
        kpi_lay.addLayout(kpi_grid)
        upper_row.addWidget(kpi_card, 2)

        # 中：反馈类型占比环形图（2026-08-29 从「追溯与提升」Tab 上移至此——
        # 翻案曲线迁入学习效果页后，环形图是反馈管理侧的留存信息）
        donut_card = make_card()
        d_lay = QVBoxLayout(donut_card)
        d_head = QLabel("反馈类型占比")
        d_head.setProperty("heading", True)
        d_lay.addWidget(d_head)
        self.donut = DonutChart()
        d_lay.addWidget(self.donut, 0, Qt.AlignHCenter)
        upper_row.addWidget(donut_card, 1)

        # 右：各品类在线学习状态
        pending_card = make_card()
        p_lay = QVBoxLayout(pending_card)
        p_head = QLabel("各品类在线学习状态（反馈即学即生效；巩固固化为新版本"
                        "请在监控页「学习提升」执行，避免与产线统计口径混用）")
        p_head.setProperty("heading", True)
        p_lay.addWidget(p_head)
        self.table_pending = QTableWidget(0, 4)
        self.table_pending.setHorizontalHeaderLabels(
            ["品类", "当前版本", "未巩固反馈数", "引擎内部"])
        self.table_pending.horizontalHeader().setStretchLastSection(True)
        self.table_pending.setColumnWidth(2, 220)
        self.table_pending.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table_pending.setMaximumHeight(190)
        p_lay.addWidget(self.table_pending)
        upper_row.addWidget(pending_card, 3)
        root.addLayout(upper_row)

        # ── 下：Tab（反馈记录 | 待复核）──
        tabs = QTabWidget()
        rec_card = make_card()
        r_lay = QVBoxLayout(rec_card)
        r_bar = QHBoxLayout()
        r_head = QLabel("反馈记录")
        r_head.setProperty("heading", True)
        r_bar.addWidget(r_head)
        r_bar.addStretch(1)
        self.combo_fb_type = QComboBox()
        for text, val in FB_TYPE_FILTERS:
            self.combo_fb_type.addItem(text, val)
        r_bar.addWidget(self.combo_fb_type)
        self.combo_fb_cat = QComboBox()
        self.combo_fb_cat.addItem("全部品类", "")
        r_bar.addWidget(self.combo_fb_cat)
        # 两个下拉都建好后才接信号（addItem 会触发 currentIndexChanged）
        self.combo_fb_type.currentIndexChanged.connect(
            lambda _i: self.reload_records())
        self.combo_fb_cat.currentIndexChanged.connect(
            lambda _i: self.reload_records())
        r_lay.addLayout(r_bar)
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(
            ["缩略图", "反馈类型", "操作员标注", "缺陷类型",
             "备注", "学习状态", "时间", "操作"])
        self.table.setColumnWidth(0, 110)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setDefaultSectionSize(66)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setToolTip("双击行查看单帧追溯（判定现场→学习后重判定）")
        self.table.itemDoubleClicked.connect(self._on_record_trace)
        r_lay.addWidget(self.table, 1)
        # 统一分页条：服务端分页，避免反馈堆积后一次加载几百条缩略图
        self.pager_records = PagerBar(page_size=20)
        self.pager_records.page_changed.connect(lambda _p: self.reload_records())
        r_lay.addWidget(self.pager_records)
        tabs.addTab(rec_card, "反馈记录")
        tabs.addTab(self._build_review_tab(), "待复核")
        tabs.addTab(self._build_misjudged_tab(), "错检追溯")
        root.addWidget(tabs, 1)

        self.progress = TaskProgressBar("巩固任务")
        root.addWidget(self.progress)
        if self._monitor is not None:
            self._monitor.task_updated.connect(self._on_task)

        self._client.error_occurred.connect(lambda m: warn(self, m))
        self.reload()

    # ══════════════════ 待复核 Tab（M5 灰区闭环）══════════════
    def _build_review_tab(self) -> QWidget:
        """待复核队列：左表格（时间/品类/系统判定/路径/分数/评分摘要）+ 右预览与判定按钮。"""
        self._review_items: list[dict] = []

        card = make_card()
        lay = QVBoxLayout(card)
        bar = QHBoxLayout()
        head = QLabel("灰区待复核队列（复核结论自动作为反馈即学，价值最高的边界样本）")
        head.setProperty("heading", True)
        bar.addWidget(head)
        bar.addStretch(1)
        bar.addWidget(QLabel("品类"))
        self.combo_review_cat = QComboBox()
        self.combo_review_cat.addItem("全部品类", "")
        self.combo_review_cat.currentIndexChanged.connect(
            lambda _i: self._review_first_page())
        bar.addWidget(self.combo_review_cat)
        btn_refresh = QPushButton("刷新")
        btn_refresh.setProperty("flat", True)
        btn_refresh.clicked.connect(self.reload_review)
        bar.addWidget(btn_refresh)
        # M15b：主动选样清单（demo5 §6 赛题点名"主动学习"）
        btn_suggest = QPushButton("系统推荐复核…")
        btn_suggest.setProperty("primary", True)
        btn_suggest.setToolTip("主动选样：引擎从灰区积累中挑『最不确定×最有代表性』"
                               "的样本请你复核，每张只问一次（反馈预算花在刀刃上）")
        btn_suggest.clicked.connect(self._on_fetch_suggestions)
        bar.addWidget(btn_suggest)
        lay.addLayout(bar)

        # 非模态复核结果提示条（替代 QMessageBox，支持流水化复核）
        self.lbl_review_toast = QLabel("")
        self.lbl_review_toast.setAlignment(Qt.AlignCenter)
        self.lbl_review_toast.setStyleSheet(
            "background:#059669; color:white; padding:5px; border-radius:4px;")
        self.lbl_review_toast.hide()
        lay.addWidget(self.lbl_review_toast)
        self._toast_timer = QTimer(self)
        self._toast_timer.setSingleShot(True)
        self._toast_timer.setInterval(3000)
        self._toast_timer.timeout.connect(self.lbl_review_toast.hide)

        splitter = QSplitter(Qt.Horizontal)
        left = QWidget()
        l_lay = QVBoxLayout(left)
        l_lay.setContentsMargins(0, 0, 0, 0)
        self.table_review = QTableWidget(0, 6)
        self.table_review.setHorizontalHeaderLabels(
            ["时间", "品类", "系统判定", "图片路径", "分数", "评分摘要"])
        self.table_review.horizontalHeader().setStretchLastSection(True)
        self.table_review.setColumnWidth(3, 300)
        self.table_review.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table_review.setSelectionBehavior(QTableWidget.SelectRows)
        self.table_review.setAlternatingRowColors(True)
        self.table_review.itemSelectionChanged.connect(self._on_review_select)
        l_lay.addWidget(self.table_review, 1)
        self.pager_review = PagerBar(page_size=20)
        self.pager_review.page_changed.connect(lambda _p: self.reload_review())
        l_lay.addWidget(self.pager_review)
        splitter.addWidget(left)

        right = QWidget()
        r_lay2 = QVBoxLayout(right)
        r_lay2.setContentsMargins(4, 0, 0, 0)
        self.review_viewer = ImageViewer()
        # 框选模式常开：拖拽框出缺陷区；Esc/单击清除
        self.review_viewer.set_selection_mode(True)
        self._review_region: list | None = None
        self.review_viewer.rect_selected.connect(self._on_review_region)
        self.review_viewer.selection_cleared.connect(self._on_region_cleared)
        r_lay2.addWidget(self.review_viewer, 1)
        self.lbl_review_info = QLabel(
            "选中左侧记录后在此预览并判定（可在图上拖拽框出缺陷区域，"
            "判缺陷时会作为定位反馈喂给拦截通道）")
        self.lbl_review_info.setProperty("subtext", True)
        self.lbl_review_info.setWordWrap(True)
        r_lay2.addWidget(self.lbl_review_info)
        btns = QHBoxLayout()
        self.btn_review_ok = QPushButton("判正常")
        self.btn_review_ok.setProperty("success", True)
        self.btn_review_ok.clicked.connect(lambda: self._on_review_judge(0))
        self.btn_review_ng = QPushButton("判缺陷")
        self.btn_review_ng.setProperty("danger", True)
        self.btn_review_ng.clicked.connect(lambda: self._on_review_judge(1))
        for b in (self.btn_review_ok, self.btn_review_ng):
            b.setEnabled(False)
            btns.addWidget(b)
        r_lay2.addLayout(btns)
        # 流水化快捷键：Enter=判正常，D=判缺陷（仅在待复核 Tab 容器内生效）
        QShortcut(QKeySequence(Qt.Key_Return), card,
                  activated=lambda: self._on_review_judge(0))
        QShortcut(QKeySequence(Qt.Key_Enter), card,
                  activated=lambda: self._on_review_judge(0))
        QShortcut(QKeySequence(Qt.Key_D), card,
                  activated=lambda: self._on_review_judge(1))
        splitter.addWidget(right)
        splitter.setSizes([620, 380])
        lay.addWidget(splitter, 1)

        # 品类筛选下拉候选：全局品类列表
        run_async(self, self._client.list_categories, self._fill_review_cats)
        return card

    def _fill_review_cats(self, cats) -> None:
        if not cats:
            return
        cur = self.combo_review_cat.currentData()
        self.combo_review_cat.blockSignals(True)
        self.combo_review_cat.clear()
        self.combo_review_cat.addItem("全部品类", "")
        for c in cats:
            self.combo_review_cat.addItem(str(c), str(c))
        idx = self.combo_review_cat.findData(cur)
        if idx >= 0:
            self.combo_review_cat.setCurrentIndex(idx)
        self.combo_review_cat.blockSignals(False)

    def _review_first_page(self) -> None:
        self.pager_review.reset()
        self.reload_review()

    def reload_review(self) -> None:
        cat = str(self.combo_review_cat.currentData() or "")
        run_async(self, lambda: self._client.review_queue(
            category=cat, page=self.pager_review.page,
            page_size=self.pager_review.page_size),
            self._fill_review_queue)
        run_async(self, self._client.review_stats, self._fill_review_stats)

    def _fill_review_stats(self, data) -> None:
        if not isinstance(data, dict):
            return
        self.kpi_pending.set_value(str(data.get("pending_total", 0)))
        # 品类下拉并入待复核品类（有积压的品类必须出现在下拉里）
        cats = data.get("categories")
        if cats:
            cur = self.combo_review_cat.currentData()
            self.combo_review_cat.blockSignals(True)
            existing = {self.combo_review_cat.itemData(i)
                        for i in range(self.combo_review_cat.count())}
            for c in cats:
                if c not in existing:
                    self.combo_review_cat.addItem(str(c), str(c))
            idx = self.combo_review_cat.findData(cur)
            if idx >= 0:
                self.combo_review_cat.setCurrentIndex(idx)
            self.combo_review_cat.blockSignals(False)

    def _fill_review_queue(self, data) -> None:
        items = (data or {}).get("items", []) if isinstance(data, dict) else []
        total = int((data or {}).get("total", 0)) if isinstance(data, dict) else 0
        self.pager_review.set_total(total)
        self._review_items = items
        self.btn_review_ok.setEnabled(False)
        self.btn_review_ng.setEnabled(False)
        self.table_review.clearSpans()
        if not items:
            self.table_review.setRowCount(1)
            self.table_review.setSpan(0, 0, 1, 6)
            self.table_review.setItem(0, 0, QTableWidgetItem(
                "暂无待复核记录：产线判为灰区的帧、或开启复判工单的全部产线检测"
                "会自动进入此队列（试检旁路不进队列）"))
            return
        self.table_review.setRowCount(len(items))
        for r, it in enumerate(items):
            created = str(it.get("created_at") or "").replace("T", " ")[:19]
            self.table_review.setItem(r, 0, QTableWidgetItem(created))
            self.table_review.setItem(r, 1, QTableWidgetItem(
                str(it.get("category", ""))))
            dec = str(it.get("decision") or "")
            dec_item = QTableWidgetItem(DECISION_CN.get(dec, dec or "-"))
            dec_color = {"normal": Qt.darkGreen, "gray": Qt.darkYellow,
                         "anomaly": Qt.darkRed}.get(dec)
            if dec_color is not None:
                dec_item.setForeground(dec_color)
            self.table_review.setItem(r, 2, dec_item)
            self.table_review.setItem(r, 3, QTableWidgetItem(
                str(it.get("image_path", ""))))
            try:
                score_text = f"{float(it.get('final_score')):.3f}"
            except (TypeError, ValueError):
                score_text = "-"
            s_item = QTableWidgetItem(score_text)
            s_item.setForeground(Qt.darkYellow)
            self.table_review.setItem(r, 4, s_item)
            slots = it.get("slots")
            slots_text = slot_text(slots) if isinstance(slots, dict) else "-"
            self.table_review.setItem(r, 5, QTableWidgetItem(slots_text))
        # 流水化：复核成功后自动选中下一条（原行位置即下一条）
        pending = getattr(self, "_pending_select_row", None)
        self._pending_select_row = None
        if pending is not None and items:
            sel = min(pending, len(items) - 1)
            self.table_review.selectRow(sel)

    def _on_review_select(self) -> None:
        self._review_override = None          # 表格选择优先于推荐项
        row = self.table_review.currentRow()
        if not (0 <= row < len(self._review_items)):
            self.btn_review_ok.setEnabled(False)
            self.btn_review_ng.setEnabled(False)
            return
        it = self._review_items[row]
        for b in (self.btn_review_ok, self.btn_review_ng):
            b.setEnabled(True)
        path = it.get("overlay_path") or it.get("image_path") or ""
        if path:
            self.review_viewer.set_image_url(self._client.file_url(path))
        self.lbl_review_info.setText(
            f"检测编号 {it.get('id')}　品类 {it.get('category', '-')}\n"
            f"{it.get('image_path', '')}")

    def _on_review_region(self, region: list) -> None:
        self._review_region = region

    def _on_region_cleared(self) -> None:
        self._review_region = None

    def _on_review_judge(self, label: int) -> None:
        # M15b：推荐项优先（从主动选样清单载入的样本不在队列表格里）
        it = getattr(self, "_review_override", None)
        if it is None:
            row = self.table_review.currentRow()
            if not (0 <= row < len(self._review_items)):
                return
            it = self._review_items[row]
        det_id = it.get("id") or it.get("detection_id")
        if det_id is None:
            return
        row = self.table_review.currentRow()  # override 模式可能为 -1
        self.btn_review_ok.setEnabled(False)
        self.btn_review_ng.setEnabled(False)
        verdict = "正常" if label == 0 else "缺陷"
        # 判缺陷且有框选 -> 带 region 喂拦截通道；判正常忽略框选
        region = self._review_region if label == 1 else None
        run_async(self, lambda: self._client.review_submit(
            int(det_id), label, region=region),
            lambda res: self._on_review_done(res, verdict, row, region))

    def _on_review_done(self, res, verdict: str, row: int,
                        region: list | None) -> None:
        if isinstance(res, dict) and res.get("_status") == 409:
            warn(self, "该记录已被复核（可能由其他操作员提交），列表已刷新")
        elif isinstance(res, dict):
            # 非模态提示条（3s 自动消失），支持流水化连续复核
            text = FeedbackDialog._success_text(res).replace("\n", "　")
            if region:
                text += "（含缺陷定位框，已喂拦截通道）"
            self.lbl_review_toast.setText(f"判{verdict} ✓　{text}")
            self.lbl_review_toast.show()
            self._toast_timer.start()
        # 复核出队：刷新队列与统计；自动选中下一条（流水化）。
        # 当前页复核空时自动回退一页（goto 触发 page_changed → reload_review）
        self._review_override = None
        self._pending_select_row = row if row >= 0 else None
        self._review_region = None
        self.review_viewer.clear_selection()
        if self.pager_review.page > 1 and len(self._review_items) <= 1:
            self.pager_review.goto(self.pager_review.page - 1)
        else:
            self.reload_review()

    # ── M15b 主动选样清单（demo5 §6）──────────────────────
    def _on_fetch_suggestions(self) -> None:
        cat = str(self.combo_review_cat.currentData() or "")
        if not cat:
            warn(self, "主动选样需指定品类——请先在右上方选择品类")
            return
        run_async(self, lambda: self._client.get(
            "/api/review/active_suggestions",
            params={"category": cat, "top_n": 10}), self._fill_suggestions)

    def _fill_suggestions(self, data) -> None:
        items = (data or {}).get("items", []) if isinstance(data, dict) else []
        if not items:
            warn(self, "暂无推荐：引擎需要灰区样本积累后才会生成推荐清单"
                       "（每张只问一次，已问过的不再重复）")
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("系统推荐复核清单（主动选样）")
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel("以下样本是引擎认为『最值得你花时间复核』的"
                             "（最不确定 × 最有代表性）；点击「载入复核」后在"
                             "右侧预览区判定即可："))
        table = QTableWidget(0, 5)
        table.setHorizontalHeaderLabels(
            ["图片路径", "价值分", "不确定度", "代表性", "操作"])
        table.horizontalHeader().setStretchLastSection(True)
        table.setColumnWidth(0, 380)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setRowCount(len(items))
        for r, it in enumerate(items):
            table.setItem(r, 0, QTableWidgetItem(str(it.get("image_path", ""))))
            table.setItem(r, 1, QTableWidgetItem(f"{it.get('value', 0):.3f}"))
            table.setItem(r, 2, QTableWidgetItem(f"{it.get('uncertainty', 0):.3f}"))
            table.setItem(r, 3, QTableWidgetItem(f"{it.get('representative', 0):.3f}"))
            btn = QPushButton("载入复核")
            if it.get("detection_id") is None:
                btn.setEnabled(False)
                btn.setToolTip("该图暂无检测记录，无法在线复核")
            btn.clicked.connect(
                lambda _c=False, item=it, d=dlg: self._load_suggestion(item, d))
            table.setCellWidget(r, 4, btn)
        lay.addWidget(table, 1)
        btn_box = QDialogButtonBox(QDialogButtonBox.Close)
        btn_box.button(QDialogButtonBox.Close).setText("关闭")
        btn_box.rejected.connect(dlg.reject)
        lay.addWidget(btn_box)
        dlg.resize(860, 420)
        dlg.exec()

    def _load_suggestion(self, item: dict, dlg: QDialog) -> None:
        """把推荐样本载入右侧预览区，走同一套判正常/判缺陷流程。"""
        dlg.accept()
        self._review_override = item
        path = item.get("image_path") or ""
        if path:
            self.review_viewer.set_image_url(self._client.file_url(path))
        self.lbl_review_info.setText(
            f"【系统推荐】检测编号 {item.get('detection_id')}　"
            f"价值分 {item.get('value', 0):.3f}（不确定 {item.get('uncertainty', 0):.2f} / "
            f"代表性 {item.get('representative', 0):.2f}）\n{path}")
        for b in (self.btn_review_ok, self.btn_review_ng):
            b.setEnabled(True)
        self.table_review.clearSelection()

    def _show_trace(self, trace) -> None:
        if not isinstance(trace, dict) or not trace.get("detection"):
            warn(self, "追溯信息不可用（后端离线或记录不存在）")
            return
        TraceDialog(self._client, trace, self).exec()

    # ══════════════════ 数据加载 ══════════════════
    def _workorder_id(self) -> int | None:
        fn = getattr(self, "_get_workorder_id", None)
        return fn() if callable(fn) else None

    def reload_records(self) -> None:
        """反馈记录 Tab：按类型/品类筛选加载。"""
        ftype = str(self.combo_fb_type.currentData() or "")
        cat = str(self.combo_fb_cat.currentData() or "")
        run_async(self, lambda: self._client.list_feedback(
            feedback_type=ftype, category=cat, page_size=200),
            self._fill_records)

    def reload(self) -> None:
        wid = self._workorder_id()
        self.lbl_scope.setText(
            "统计口径：当前工单" if wid is not None else "统计口径：全部工单（未选工单）")
        run_async(self, lambda: self._client.feedback_summary(workorder_id=wid),
                  self._fill_summary)
        self.reload_records()
        run_async(self, lambda: self._client.get(
            "/api/review/incomplete_report"), self._fill_incomplete)
        self.reload_review()
        self.reload_misjudged()
        run_async(self, self._client.list_categories, self._fill_fb_cats)

    def _fill_fb_cats(self, cats) -> None:
        if not cats:
            return
        cur = self.combo_fb_cat.currentData()
        self.combo_fb_cat.blockSignals(True)
        self.combo_fb_cat.clear()
        self.combo_fb_cat.addItem("全部品类", "")
        for c in cats:
            self.combo_fb_cat.addItem(str(c), str(c))
        idx = self.combo_fb_cat.findData(cur)
        if idx >= 0:
            self.combo_fb_cat.setCurrentIndex(idx)
        self.combo_fb_cat.blockSignals(False)

    def _fill_incomplete(self, data) -> None:
        """M14c：不完备报告横幅——open 信号聚合超阈时显示。"""
        if not isinstance(data, dict) or not data.get("any_alert"):
            self.banner_incomplete.hide()
            return
        alerts = [r for r in data.get("reports", []) if r.get("alert")]
        text = "；".join(
            f"品类 {r['category']} 近 {r['window_n']} 帧未知模式 "
            f"{r['open_hits']} 次（{r['ratio'] * 100:.0f}%）" for r in alerts)
        self.banner_incomplete.setText(
            f"不完备预警：{text}——疑似体系外缺陷信号持续出现，现有特征组无法解释；"
            f"建议复核这些样本（确认后反馈学习），或考虑补充模板图/联系工艺扩充缺陷定义")
        self.banner_incomplete.show()

    def _fill_summary(self, data) -> None:
        if not data:
            return
        s = data.get("summary", {})
        self.kpi_total.set_value(str(s.get("total", 0)))
        self.kpi_reviewed.set_value(str(s.get("reviewed", 0)))
        rate = s.get("misclassification_rate", 0) or 0
        self.kpi_rate.set_value(f"{float(rate) * 100:.1f}%")

        self.donut.set_data({
            "误检": s.get("false_positives", 0),
            "漏检": s.get("false_negatives", 0),
            "确认": s.get("confirmed", 0),
            "新缺陷": s.get("new_defects", 0),
        }, colors={"误检": WARNING, "漏检": DANGER,
                   "确认": SUCCESS, "新缺陷": PRIMARY})

        # 各品类在线学习状态（pending_update_status 新契约）
        pending = data.get("pending", {})
        cats = pending.get("categories", [])
        self.table_pending.clearSpans()
        if not cats:
            self.table_pending.setRowCount(1)
            self.table_pending.setSpan(0, 0, 1, 4)   # 4 列：品类/版本/未巩固/内部
            self.table_pending.setItem(
                0, 0, QTableWidgetItem("暂无已准备品类（准备模型并产生反馈后出现）"))
            return
        self.table_pending.setRowCount(len(cats))
        for r, c in enumerate(cats):
            cat = str(c.get("category", ""))
            cur = c.get("current_version")
            log_len = c.get("feedback_log_len")
            pending_n = (str(log_len) if log_len is not None
                         else f"{c.get('unconsumed_db', 0)}*")
            self.table_pending.setItem(r, 0, QTableWidgetItem(cat))
            self.table_pending.setItem(r, 1, QTableWidgetItem(
                version_text(cur) if cur is not None else "未准备"))
            self.table_pending.setItem(r, 2, QTableWidgetItem(pending_n))
            btn_insight = QPushButton("内部状态…")
            btn_insight.setToolTip("查看在线学习引擎内部：样本库规模 / 缺陷样例库 / "
                                   "专属判别头进度 / 权重更新审核")
            btn_insight.clicked.connect(
                lambda _c=False, name=cat: self._show_insight(name))
            self.table_pending.setCellWidget(r, 3, btn_insight)

    def _show_insight(self, category: str) -> None:
        """M14b：在线学习引擎内部状态对话框。"""
        run_async(self, lambda: self._client.learning_insight(category),
                  lambda d: self._fill_insight(category, d))

    def _fill_insight(self, category: str, d) -> None:
        if not isinstance(d, dict) or not d:
            warn(self, "未取到引擎内部状态")
            return
        dlg = QDialog(self)
        dlg.setWindowTitle(f"在线学习引擎内部状态 — {category}")
        f = QFormLayout(dlg)
        if not d.get("enabled"):
            f.addRow(QLabel("该品类在线学习未激活（未准备或无反馈处理器）"))
        else:
            nb = d.get("normal_bank", {})
            f.addRow("当前版本", QLabel(version_text(d.get("version"))))
            f.addRow("初始权重质量",
                     QLabel(f"验证准确率 = {d.get('train_auroc')}"
                            "（≥0.99 冻结权重学习）"))
            f.addRow("正常锚定库", QLabel(f"{nb.get('core_patches', 0)} 个图块"
                                        + ("（融合已暂停）" if nb.get("fuse_blocked") else "")))
            f.addRow("正常扩展库", QLabel(f"{nb.get('ext_samples', 0)} 张回流样本"))
            db = d.get("defect_bank", {})
            f.addRow("缺陷样例库", QLabel(f"{db.get('samples', 0)} 张"
                                        f"（拦截命中下限 {db.get('intercept_min_hit', 1)}）"))
            inc = d.get("incubate", {})
            pct = int(float(inc.get("progress", 0)) * 100)
            state = ("已就绪（正样本图块 "
                     f"{inc.get('n_pos', 0)} 个）" if inc.get("trained")
                     else f"积累中 {pct}%（{db.get('samples', 0)}/"
                          f"{inc.get('min_samples', 8)} 张后自动训练）")
            f.addRow("专属判别头", QLabel(state))
            f.addRow("主动选样队列", QLabel(f"{d.get('active_queue', 0)} 条"))
            wg = d.get("weight_gate", {})
            f.addRow("权重更新审核",
                     QLabel(f"通过 {wg.get('apply', 0)} 次 / 否决 {wg.get('reject', 0)} 次"))
            st = d.get("stats", {})
            if st:
                f.addRow("学习动作统计", QLabel("，".join(
                    f"{k}={v}" for k, v in st.items())))
        btn = QDialogButtonBox(QDialogButtonBox.Close)
        btn.button(QDialogButtonBox.Close).setText("关闭")
        btn.rejected.connect(dlg.reject)
        f.addRow(btn)
        dlg.resize(560, 360)
        dlg.exec()

    def _fill_records(self, data) -> None:
        items = (data or {}).get("items", []) if isinstance(data, dict) else (data or [])
        pager = getattr(self, "pager_records", None)
        if pager is not None:
            total = (data or {}).get("total", 0) if isinstance(data, dict) else len(items)
            pager.set_total(int(total or 0))
        self._record_items = items
        self.table.clearSpans()
        if not items:
            self.table.setRowCount(1)
            self.table.setSpan(0, 0, 1, 8)
            self.table.setItem(0, 0, QTableWidgetItem(
                "暂无反馈记录，可在监控页对检测结果进行标注"))
            return
        self.table.setRowCount(len(items))
        for r, fb in enumerate(items):
            invalidated = bool(fb.get("invalidated"))
            cell = make_thumb_cell()
            self.table.setCellWidget(r, 0, cell)
            path = fb.get("overlay_path") or fb.get("image_path") or ""
            load_thumb_async(self, self._client,
                             self._client.file_url(path), cell)
            ftype = str(fb.get("feedback_type", ""))
            type_item = QTableWidgetItem(FEEDBACK_TYPE_NAMES.get(ftype, ftype))
            if ftype in ("false_positive", "false_negative"):
                type_item.setForeground(Qt.darkYellow if ftype == "false_positive"
                                        else Qt.darkRed)
            self.table.setItem(r, 1, type_item)
            label = fb.get("operator_label")
            self.table.setItem(r, 2, QTableWidgetItem(
                {0: "正常", 1: "异常"}.get(label, "-")))
            self.table.setItem(r, 3, QTableWidgetItem(str(fb.get("defect_type") or "-")))
            self.table.setItem(r, 4, QTableWidgetItem(str(fb.get("comment") or "-")))
            if invalidated:
                c_item = QTableWidgetItem("已作废")
                c_item.setForeground(Qt.gray)
                c_item.setToolTip("该反馈已被作废（标错撤回）：不计入统计/曲线，"
                                  "对应检测重新进入复核队列")
            else:
                consumed = bool(fb.get("consumed"))
                c_item = QTableWidgetItem("已学习" if consumed else "仅记录")
                c_item.setForeground(Qt.darkGreen if consumed else Qt.darkRed)
                c_item.setToolTip("已学习：该反馈已用于更新模型判断\n"
                                  "仅记录：已保存，尚未进入模型学习")
            self.table.setItem(r, 5, c_item)
            self.table.setItem(r, 6, QTableWidgetItem(str(fb.get("created_at", ""))))
            if invalidated:
                inv = QLabel("—")
                inv.setAlignment(Qt.AlignCenter)
                self.table.setCellWidget(r, 7, inv)
            else:
                btn_inv = QPushButton("作废")
                btn_inv.setProperty("flat", True)
                btn_inv.setToolTip("标错了？作废该反馈：不计入统计/曲线，"
                                   "对应检测重新进入复核队列；"
                                   "已学进模型的影响需回滚版本消除")
                fb_id = fb.get("id")
                btn_inv.clicked.connect(
                    lambda _c=False, fid=fb_id: self._on_invalidate(fid))
                self.table.setCellWidget(r, 7, btn_inv)

    def _on_invalidate(self, feedback_id) -> None:
        """作废反馈（标错撤回），二次确认。"""
        if feedback_id is None:
            return
        from PySide6.QtWidgets import QMessageBox
        ret = QMessageBox.question(
            self, "作废反馈",
            "确定作废这条反馈吗？\n\n作废后：不再计入统计/翻案曲线，对应检测"
            "重新进入复核队列。\n若反馈已被引擎学习，模型内的影响需到"
            "「模型管理」回滚到反馈前版本才能消除。",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if ret != QMessageBox.Yes:
            return

        def _done(res) -> None:
            if isinstance(res, dict):
                notify(self, str(res.get("note") or "已作废"))
                self.reload()

        run_async(self, lambda: self._client.invalidate_feedback(int(feedback_id)),
                  _done)

    def _on_record_trace(self, item) -> None:
        """反馈记录双击 → 单帧追溯（与追溯 Tab 同一 TraceDialog）。"""
        row = item.row()
        items = getattr(self, "_record_items", [])
        if not (0 <= row < len(items)):
            return
        det_id = items[row].get("detection_id")
        if det_id is None:
            return
        run_async(self, lambda: self._client.detection_trace(int(det_id)),
                  self._show_trace)

    # ══════════════════ 错检追溯 Tab（M7a，前端反馈 v7-5）══════════════
    def _build_misjudged_tab(self) -> QWidget:
        """错检集清单：被反馈判错（误检/漏检/新缺陷）的检测记录。

        错检是在线学习价值最高的样本——每条错检反馈即时进入引擎学习，
        双击行可打开单帧追溯，对照「当时判定 → 学习后重判定」看提升效果。
        """
        self._misjudged_items: list[dict] = []

        card = make_card()
        lay = QVBoxLayout(card)
        bar = QHBoxLayout()
        head = QLabel("错检集（被判错的检测：误检/漏检/新缺陷；"
                      "这些样本的反馈已即时学习，双击行看学习后是否翻案）")
        head.setProperty("heading", True)
        bar.addWidget(head)
        bar.addStretch(1)
        bar.addWidget(QLabel("品类"))
        self.combo_misj_cat = QComboBox()
        self.combo_misj_cat.addItem("全部品类", "")
        self.combo_misj_cat.currentIndexChanged.connect(
            lambda _i: self._misjudged_first_page())
        bar.addWidget(self.combo_misj_cat)
        btn_refresh = QPushButton("刷新")
        btn_refresh.setProperty("flat", True)
        btn_refresh.clicked.connect(self.reload_misjudged)
        bar.addWidget(btn_refresh)
        lay.addLayout(bar)

        self.table_misjudged = QTableWidget(0, 7)
        self.table_misjudged.setHorizontalHeaderLabels(
            ["反馈时间", "品类", "图片路径", "系统判定", "当时分数",
             "反馈类型", "操作员标注"])
        self.table_misjudged.horizontalHeader().setStretchLastSection(True)
        self.table_misjudged.setColumnWidth(2, 320)
        self.table_misjudged.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table_misjudged.setSelectionBehavior(QTableWidget.SelectRows)
        self.table_misjudged.setAlternatingRowColors(True)
        self.table_misjudged.setToolTip(
            "双击行查看单帧追溯（判定现场→学习后重判定）")
        self.table_misjudged.itemDoubleClicked.connect(
            self._on_misjudged_trace)
        lay.addWidget(self.table_misjudged, 1)
        self.pager_misjudged = PagerBar(page_size=20)
        self.pager_misjudged.page_changed.connect(
            lambda _p: self.reload_misjudged())
        lay.addWidget(self.pager_misjudged)
        return card

    def _misjudged_first_page(self) -> None:
        self.pager_misjudged.reset()
        self.reload_misjudged()

    def reload_misjudged(self) -> None:
        if not hasattr(self, "table_misjudged"):
            return
        cat = str(self.combo_misj_cat.currentData() or "")
        # 注意：PagerBar.page 是 property（非方法），不能加括号——
        # 加括号抛 TypeError 在 worker 线程被静默吞掉，请求永远发不出去
        run_async(self, lambda: self._client.misjudged(
            category=cat, page=self.pager_misjudged.page,
            page_size=self.pager_misjudged.page_size),
            self._fill_misjudged)

    def _fill_misjudged(self, data) -> None:
        if not isinstance(data, dict):
            return
        items = list(data.get("items") or [])
        self._misjudged_items = items
        total = int(data.get("total") or 0)
        self.pager_misjudged.set_total(total)
        # 品类下拉补全（不动当前选择）
        cur = self.combo_misj_cat.currentData()
        known = {self.combo_misj_cat.itemData(i)
                 for i in range(self.combo_misj_cat.count())}
        self.combo_misj_cat.blockSignals(True)
        for it in items:
            c = str(it.get("category") or "")
            if c and c not in known:
                self.combo_misj_cat.addItem(c, c)
                known.add(c)
        idx = self.combo_misj_cat.findData(cur)
        if idx >= 0:
            self.combo_misj_cat.setCurrentIndex(idx)
        self.combo_misj_cat.blockSignals(False)

        t = self.table_misjudged
        t.setRowCount(0)
        if not items:
            t.setRowCount(1)
            t.setSpan(0, 0, 1, 7)
            t.setItem(0, 0, QTableWidgetItem(
                "暂无错检记录（在监控页/待复核中把系统判错的图反馈后即可追溯）"))
            return
        t.setRowCount(len(items))
        fb_cn = {v: k for k, v in FB_TYPE_FILTERS if v}
        for r, it in enumerate(items):
            score = it.get("final_score")
            score_s = f"{float(score):.3f}" if score is not None else "-"
            dec = DECISION_CN.get(str(it.get("decision") or ""),
                                  str(it.get("decision") or "-"))
            vals = [str(it.get("feedback_time") or "")[:19],
                    str(it.get("category") or ""),
                    str(it.get("image_path") or ""),
                    dec, score_s,
                    fb_cn.get(str(it.get("feedback_type") or ""),
                              str(it.get("feedback_type") or "-")),
                    str(it.get("operator_label") or "")]
            for c, v in enumerate(vals):
                t.setItem(r, c, QTableWidgetItem(v))

    def _on_misjudged_trace(self, item) -> None:
        row = item.row()
        if not (0 <= row < len(self._misjudged_items)):
            return
        det_id = self._misjudged_items[row].get("detection_id")
        if det_id is None:
            return
        run_async(self, lambda: self._client.detection_trace(int(det_id)),
                  self._show_trace)

    # ══════════════════ 动作 ══════════════════
    def _on_task(self, task: dict) -> None:
        ttype = str(task.get("task_type", ""))
        # 翻案曲线已迁入「学习效果」页（learning_page 自行处理 flip_curve
        # 任务），本页不再消费，避免调用不存在的方法
        if ttype == "flip_curve":
            return
        if ttype and "self_learning" not in ttype and "update" not in ttype:
            return
        self.progress.update_task(task)
        if str(task.get("status")) in ("done", "success", "failed", "error"):
            self.reload()
