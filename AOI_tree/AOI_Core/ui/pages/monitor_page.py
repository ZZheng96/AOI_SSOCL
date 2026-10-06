"""实时监控页：产线检测流的实时监视终端 + 操作员干预入口。

数据流视角（2026-08-29 重构）：
  数据源(批次/RTSP/PLC) → 检测组队列 → PipelineService 后台消费 → detections 落库
      → 本页以 1.5s 轮询 /api/detections/live 订阅产线帧
        （target_type ∈ pipeline/stream/plc；试检的 upload/image 不进流）

布局：左栏（上=试检旁路面板，下=检测组数据流队列，可回队插队）
     + 中央 ImageViewer（原图/热力图/叠加图）+ 判定条带 + 分数曲线(含 EMA 平滑)
     + 右结果面板（ScoreGauge、三态判定、槽位分、延迟对比、缺陷框数）
时序语义对应 algo §6：持续帧流 + EMA 平滑抑制单帧抖动；连续异常由后端
AlarmMonitor 滑窗聚合，本页红色横幅呈现（停线检查信号）。
底部：对当前显示帧【标记误检】【标记漏检】【确认正确】（反馈即学）。
"""
from __future__ import annotations

import json
from datetime import datetime

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ui.api_client import ApiClient
from ui.pages.common import (
    SLOT_CN,
    attach_level_badge,
    load_thumb_async,
    make_banner,
    make_card,
    make_thumb_cell,
    notify,
    run_async,
    warn,
)
from ui.theme import DANGER, PRIMARY, SUCCESS, WARNING
from ui.widgets.charts import LineChart
from ui.widgets.feedback_dialog import FeedbackDialog
from ui.widgets.gauge import ScoreGauge
from ui.widgets.image_viewer import ImageViewer, LAYER_NAMES, LAYERS

LAYER_ORDER = ["original", "heatmap", "overlay"]
LIVE_WINDOW = 50          # 判定条带/分数曲线窗口长度
LIVE_POLL_MS = 1500       # 产线流轮询间隔（产线节拍 ~0.5s/图）
EMA_ALPHA = 0.5           # 时序平滑系数（algo §6 smooth_ema 同口径）
# 条带判定色：normal 绿 / gray 黄 / anomaly 红
DECISION_COLORS = {"normal": SUCCESS, "gray": WARNING, "anomaly": DANGER}


def _pulse_flags(frames: list[dict]) -> list[bool]:
    """短脉冲标记（algo §6 pulse_downgrade 同口径）：连续 <2 帧的孤立异常
    在时序上应降级为灰区——条带上以描边样式提示，不改存储判定。"""
    n = len(frames)
    flags = [False] * n
    i = 0
    while i < n:
        if str(frames[i].get("decision")) == "anomaly":
            j = i
            while j < n and str(frames[j].get("decision")) == "anomaly":
                j += 1
            if j - i < 2:          # 脉冲宽度 < min_consecutive(=2)
                for k in range(i, j):
                    flags[k] = True
            i = j
        else:
            i += 1
    return flags


def _ema(values: list[float], alpha: float = EMA_ALPHA) -> list[float]:
    """指数滑动平均（algo/video_pipeline.smooth_ema 同口径，首帧保底）。"""
    out, prev = [], values[0] if values else 0.0
    for v in values:
        prev = alpha * v + (1 - alpha) * prev
        out.append(prev)
    return out


def _row_to_det(row: dict, alarm: dict | None = None) -> dict:
    """Detection 表行（to_dict）→ _show_detection 使用的扁平结构。

    产线帧不落热力图（速度红线），slots/weights/decision/缺陷框等均从
    n_tiles JSON 还原（_persist 已落 threshold/gray_threshold/boxes）。
    """
    nt = row.get("n_tiles") or {}
    if isinstance(nt, str):
        try:
            nt = json.loads(nt)
        except Exception:  # noqa: BLE001
            nt = {}
    is_anomaly = bool(row.get("is_anomaly"))
    return {
        "detection_id": row.get("detection_id", row.get("id")),
        "persisted": bool(row.get("persisted", True)),
        "feedback_supported": bool(row.get("feedback_supported", False)),
        "image_path": row.get("image_path"),
        "heatmap_path": row.get("heatmap_path"),
        "overlay_path": row.get("overlay_path"),
        "final_score": row.get("final_score") or 0.0,
        "is_anomaly": is_anomaly,
        "decision": nt.get("decision") or ("anomaly" if is_anomaly else "normal"),
        "threshold": nt.get("threshold"),
        "gray_threshold": nt.get("gray_threshold"),
        "slots": nt.get("slots") or {},
        "weights": nt.get("weights") or {},
        "triggered_slot": nt.get("triggered_slot") or "",
        "defect_boxes": nt.get("boxes") or [],
        "defect_types": nt.get("types") or [],
        "open_alert": bool(nt.get("open_alert")),
        "align_warn": bool(nt.get("align_warn")),
        "align_offset": nt.get("align_offset"),
        "latency_ms": row.get("latency_ms"),
        # 端到端口径（含解码/持久化）；旧数据为 None，显示端回退推理耗时
        "latency_e2e_ms": row.get("latency_e2e_ms"),
        "created_at": row.get("created_at"),
        "category": row.get("category"),
        "alarm": alarm,
    }


class ImagePickDialog(QDialog):
    """库中选图对话框：缩略图表格，双击或确定选择。"""

    def __init__(self, client: ApiClient, category: str,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self._client = client
        self._category = category
        self.selected_image: dict | None = None
        self.setWindowTitle("从图库选择图片")
        self.resize(720, 480)

        lay = QVBoxLayout(self)
        hint = QLabel(f"当前品类：{category or '全部'}（双击行选择）")
        hint.setProperty("subtext", True)
        lay.addWidget(hint)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["缩略图", "ID", "路径", "标签"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnWidth(0, 110)
        self.table.setColumnWidth(1, 60)
        self.table.setColumnWidth(2, 360)
        self.table.verticalHeader().setDefaultSectionSize(66)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.itemDoubleClicked.connect(lambda _i: self._confirm())
        lay.addWidget(self.table)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("选择")
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        buttons.accepted.connect(self._confirm)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

        run_async(self, lambda: self._client.list_images(
            category=self._category, page_size=50), self._fill)

    def _fill(self, data) -> None:
        items = (data or {}).get("items", [])
        self._items = items
        self.table.clearSpans()
        if not items:
            self.table.setRowCount(1)
            self.table.setSpan(0, 0, 1, 4)
            self.table.setItem(0, 0, QTableWidgetItem("暂无图像数据，请先在数据管理页导入"))
            return
        self.table.setRowCount(len(items))
        for r, img in enumerate(items):
            cell = make_thumb_cell()
            self.table.setCellWidget(r, 0, cell)
            load_thumb_async(self, self._client,
                             self._client.file_url(img.get("path", "")), cell)
            self.table.setItem(r, 1, QTableWidgetItem(str(img.get("id", ""))))
            self.table.setItem(r, 2, QTableWidgetItem(str(img.get("path", ""))))
            self.table.setItem(r, 3, QTableWidgetItem(str(img.get("label", ""))))

    def _confirm(self) -> None:
        row = self.table.currentRow()
        items = getattr(self, "_items", [])
        if 0 <= row < len(items):
            self.selected_image = items[row]
            self.accept()
        else:
            warn(self, "请先选择一行图像")


class MonitorPage(QWidget):
    """实时监控页（产线流订阅模式）。"""

    goto_page = Signal(str)   # 请求主窗口跳转导航页（空转引导去模型管理）

    def __init__(self, client: ApiClient, get_category, get_budget=None,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self._client = client
        self._get_category = get_category
        self._get_budget = get_budget or (lambda: 200.0)
        self._current_detection: dict | None = None
        # 产线实时流状态
        self._frames: list[dict] = []       # 窗口内帧（旧→新，cap LIVE_WINDOW）
        self._last_id = 0                   # 已消费的最大 detection id
        self._polling = False               # 轮询防重入
        self._live_retry_count = 0
        self._live_last_error = ""
        self._live_dropped_count = 0
        self._strip_cells: list[QPushButton] = []
        self._selected_frame: int = -1      # 条带选中下标（-1=跟随最新）

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(8)

        # ── 顶部：页面说明横幅 ──
        top = QHBoxLayout()
        top.addWidget(make_banner("实时异常检测"))
        attach_level_badge(self, top)   # U-workorder：数据条件徽标
        title = QLabel("订阅产线检测流：画面随产线滚动，发现错检随手标记（即学生效）")
        title.setProperty("subtext", True)
        top.addWidget(title)
        top.addStretch(1)
        root.addLayout(top)

        # 连续异常告警横幅（alarm.active 时显示；产线停线检查信号）
        self.lbl_alarm = QLabel("")
        self.lbl_alarm.setAlignment(Qt.AlignCenter)
        self.lbl_alarm.setStyleSheet(
            "background:#DC2626; color:white; font-weight:700; "
            "padding:6px; border-radius:4px;")
        self.lbl_alarm.hide()
        root.addWidget(self.lbl_alarm)

        # ── 工单产线控制（暂停/恢复/学习，不重启不重配）──
        pbox = QGroupBox("产线运行（当前工单）")
        ph = QHBoxLayout(pbox)
        self.lbl_pipeline = QLabel("产线：-")
        self.lbl_pipeline.setProperty("badge", "muted")
        ph.addWidget(self.lbl_pipeline)
        self.chk_auto_resume = QCheckBox("学习完自动恢复产线")
        self.chk_auto_resume.toggled.connect(self._save_auto_resume)
        ph.addWidget(self.chk_auto_resume)
        self.btn_pp_pause = QPushButton("暂停产线")
        self.btn_pp_pause.clicked.connect(lambda: self._pipeline_ctl("pause"))
        self.btn_pp_resume = QPushButton("恢复产线")
        self.btn_pp_resume.clicked.connect(lambda: self._pipeline_ctl("resume"))
        self.btn_pp_stop = QPushButton("停止产线")
        self.btn_pp_stop.setProperty("danger", True)
        self.btn_pp_stop.setToolTip(
            "停止 = 结束本次任务排产：清空回队队列（暂停期间编排的重检批次），"
            "不会被自动恢复，只能手动「恢复产线」重新启动")
        self.btn_pp_stop.clicked.connect(self._on_pipeline_stop)
        self.btn_pp_learn = QPushButton("学习提升")
        self.btn_pp_learn.setToolTip("批量应用该工单积累的反馈并固化；"
                                     "学完按工单配置自动恢复产线")
        self.btn_pp_learn.clicked.connect(self._pipeline_learn)
        self.btn_pp_report = QPushButton("导出报告CSV")
        self.btn_pp_report.setToolTip(
            "下载当前工单检测报告（汇总指标 + 逐图判定明细，CSV）")
        self.btn_pp_report.clicked.connect(self._export_wo_report)
        for b in (self.btn_pp_pause, self.btn_pp_resume, self.btn_pp_stop,
                  self.btn_pp_learn, self.btn_pp_report):
            ph.addWidget(b)
        # 产线空转可见性：待检测 0 / 未准备品类 N 提示与引导
        self.lbl_pipeline_hint = QLabel("")
        self.lbl_pipeline_hint.setProperty("subtext", True)
        self.lbl_pipeline_hint.setWordWrap(True)
        self.lbl_pipeline_hint.hide()
        ph.addWidget(self.lbl_pipeline_hint)
        self.btn_go_prepare = QPushButton("去准备未就绪品类")
        self.btn_go_prepare.setToolTip(
            "跳转「模型管理」为未准备品类准备检测模型，产线随后自动恢复送检")
        self.btn_go_prepare.clicked.connect(
            lambda: self.goto_page.emit("模型管理"))
        self.btn_go_prepare.hide()
        ph.addWidget(self.btn_go_prepare)
        ph.addStretch(1)
        root.addWidget(pbox)

        # ── 中部三栏（左栏 = 试检 + 数据流队列 上下布局，队列高度弹性）──
        splitter = QSplitter(Qt.Horizontal)
        left_col = QWidget()
        lc = QVBoxLayout(left_col)
        lc.setContentsMargins(0, 0, 0, 0)
        lc.setSpacing(8)
        lc.addWidget(self._build_control_panel())
        lc.addWidget(self._build_queue_panel(), 1)
        splitter.addWidget(left_col)
        splitter.addWidget(self._build_viewer_panel())
        splitter.addWidget(self._build_result_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([340, 740, 290])
        root.addWidget(splitter, 1)

        # ── 底部：反馈按钮（对当前显示帧）──
        bottom = QHBoxLayout()
        bottom.addStretch(1)
        self.btn_fp = QPushButton("标记误检")
        self.btn_fp.setProperty("warning", True)
        self.btn_fp.clicked.connect(lambda: self._open_feedback("false_positive"))
        self.btn_fn = QPushButton("标记漏检")
        self.btn_fn.setProperty("danger", True)
        self.btn_fn.clicked.connect(lambda: self._open_feedback("false_negative"))
        self.btn_ok = QPushButton("确认正确")
        self.btn_ok.setProperty("success", True)
        self.btn_ok.clicked.connect(lambda: self._open_feedback("confirmed"))
        for b in (self.btn_fp, self.btn_fn, self.btn_ok):
            b.setEnabled(False)
            bottom.addWidget(b)
        bottom.addStretch(1)
        root.addLayout(bottom)

        self._client.error_occurred.connect(lambda m: warn(self, m))
        # 产线状态/队列：10s 轻量刷新
        self._pipe_timer = QTimer(self)
        self._pipe_timer.setInterval(10_000)
        self._pipe_timer.timeout.connect(self._on_pipe_tick)
        self._pipe_timer.start()
        # 产线实时流：1.5s 增量轮询（仅页面可见时）
        self._live_timer = QTimer(self)
        self._live_timer.setInterval(LIVE_POLL_MS)
        self._live_timer.timeout.connect(self._poll_live)
        self._live_timer.start()

    # ══════════════════ 面板构建 ══════════════════
    def _build_control_panel(self) -> QWidget:
        """试检（旁路调试）：单图检测不进入产线流，用于模型抽查/调参验证。"""
        card = make_card()
        lay = QVBoxLayout(card)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(8)

        head = QLabel("试检（旁路调试）")
        head.setProperty("heading", True)
        lay.addWidget(head)

        btn_upload = QPushButton("上传图片试检")
        btn_upload.setProperty("primary", True)
        btn_upload.clicked.connect(self._on_upload_detect)
        lay.addWidget(btn_upload)
        btn_pick = QPushButton("从图库选图试检…")
        btn_pick.clicked.connect(self._on_pick_detect)
        lay.addWidget(btn_pick)
        tip = QLabel("试检走旁路通道（upload/image），\n"
                     "不进入产线流、判定条带与分数曲线；\n"
                     "产线实时检测由工单队列自动消费。")
        tip.setProperty("subtext", True)
        tip.setWordWrap(True)
        lay.addWidget(tip)
        lay.addStretch(1)

        # 实时流状态
        self.lbl_live = QLabel("实时流：等待产线新帧…")
        self.lbl_live.setProperty("badge", "muted")
        self.lbl_live.setAlignment(Qt.AlignCenter)
        self.lbl_live.setWordWrap(True)
        lay.addWidget(self.lbl_live)
        self.lbl_live_kpi = QLabel("")
        self.lbl_live_kpi.setProperty("subtext", True)
        self.lbl_live_kpi.setAlignment(Qt.AlignCenter)
        self.lbl_live_kpi.setWordWrap(True)
        lay.addWidget(self.lbl_live_kpi)
        return card

    def _build_viewer_panel(self) -> QWidget:
        card = make_card()
        lay = QVBoxLayout(card)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(6)

        bar = QHBoxLayout()
        self.layer_group = QButtonGroup(self)
        self._layer_btns: dict[str, QPushButton] = {}
        for i, layer in enumerate(LAYER_ORDER):
            b = QPushButton(LAYER_NAMES[layer])
            b.setCheckable(True)
            b.setProperty("segment", True)
            if i == 0:
                b.setChecked(True)
            b.clicked.connect(lambda _c=False, l=layer: self.viewer.set_layer(l))
            self.layer_group.addButton(b, i)
            self._layer_btns[layer] = b
            bar.addWidget(b)
        bar.addStretch(1)
        self.chk_follow = QCheckBox("跟随最新帧")
        self.chk_follow.setChecked(True)
        self.chk_follow.setToolTip("勾选时画面随产线最新帧滚动；"
                                   "点击判定条带回看历史帧会自动取消跟随")
        bar.addWidget(self.chk_follow)
        btn_fit = QPushButton("适应窗口")
        btn_fit.setProperty("flat", True)
        btn_fit.clicked.connect(lambda: self.viewer.fit_to_view())
        bar.addWidget(btn_fit)
        lay.addLayout(bar)

        self.viewer = ImageViewer()
        lay.addWidget(self.viewer, 1)

        # ── 时序区：判定条带（近 50 帧 绿/黄/红，点击回看）──
        strip_head = QHBoxLayout()
        t = QLabel("判定条带（近 50 帧，旧→新；点击回看）")
        t.setProperty("subtext", True)
        strip_head.addWidget(t)
        strip_head.addStretch(1)
        self.lbl_strip_stats = QLabel("")
        self.lbl_strip_stats.setProperty("subtext", True)
        strip_head.addWidget(self.lbl_strip_stats)
        lay.addLayout(strip_head)

        self.strip_body = QWidget()
        self.strip_lay = QHBoxLayout(self.strip_body)
        self.strip_lay.setContentsMargins(2, 2, 2, 2)
        self.strip_lay.setSpacing(1)
        self.strip_lay.addStretch(1)
        strip_scroll = QScrollArea()
        strip_scroll.setWidgetResizable(True)
        strip_scroll.setFixedHeight(34)
        strip_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        strip_scroll.setWidget(self.strip_body)
        lay.addWidget(strip_scroll)

        # ── 时序区：分数曲线（原始 + EMA 平滑，algo §6 时序语义）──
        self.score_chart = LineChart()
        self.score_chart.setMinimumHeight(130)
        self.score_chart.setMaximumHeight(170)
        lay.addWidget(self.score_chart)
        return card

    def _build_result_panel(self) -> QWidget:
        card = make_card()
        lay = QVBoxLayout(card)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(6)

        head = QLabel("检测结果")
        head.setProperty("heading", True)
        lay.addWidget(head)

        self.gauge = ScoreGauge()
        self.gauge.setToolTip(
            "综合异常分（0~1）：多个评分依据按融合权重加权后的总评分；\n"
            "指针为判定阈值——低于阈值判正常，超过判异常，\n"
            "接近阈值的进入灰区转人工复核")
        lay.addWidget(self.gauge, 0, Qt.AlignHCenter)

        self.badge_verdict = QLabel("待检测")
        self.badge_verdict.setProperty("badge", "muted")
        f = self.badge_verdict.font()
        f.setPointSizeF(13)
        f.setBold(True)
        self.badge_verdict.setFont(f)
        self.badge_verdict.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.badge_verdict)

        # 灰区入队提示（decision==gray 时显示）
        self.lbl_review_hint = QLabel("已入复核队列（反馈页→待复核）")
        self.lbl_review_hint.setProperty("badge", "warning")
        self.lbl_review_hint.setAlignment(Qt.AlignCenter)
        self.lbl_review_hint.hide()
        lay.addWidget(self.lbl_review_hint)

        # 开放集未知模式徽标（open_alert 时显示：不像任何已知正常，新型缺陷/错料）
        self.badge_open = QLabel("未知模式")
        self.badge_open.setAlignment(Qt.AlignCenter)
        self.badge_open.setStyleSheet(
            "background:#7C3AED; color:white; font-weight:700; "
            "padding:3px 10px; border-radius:4px;")
        self.badge_open.setToolTip(
            "开放集哨兵告警：该样本不像任何已知正常模式（可能是新型缺陷/光照突变/错料）")
        self.badge_open.hide()
        lay.addWidget(self.badge_open)

        # 缺陷类型归因徽标（decision != normal 且有归因结果时显示）
        self.badge_attr = QLabel("")
        self.badge_attr.setAlignment(Qt.AlignCenter)
        self.badge_attr.setStyleSheet(
            "background:#0E7490; color:white; font-weight:700; "
            "padding:3px 10px; border-radius:4px;")
        self.badge_attr.setToolTip("缺陷类型归因：有监督判别器 top-1 类型及置信度")
        self.badge_attr.hide()
        lay.addWidget(self.badge_attr)

        # 对位偏移预警徽标（治具/传送带漂移，提示现场检查定位）
        self.badge_align = QLabel("对位偏移")
        self.badge_align.setAlignment(Qt.AlignCenter)
        self.badge_align.setStyleSheet(
            "background:#B45309; color:white; font-weight:700; "
            "padding:3px 10px; border-radius:4px;")
        self.badge_align.setToolTip(
            "对位偏移预警：当前帧相对参考图平移超阈值，请检查治具/传送带定位")
        self.badge_align.hide()
        lay.addWidget(self.badge_align)

        # 各项评分依据水平条（按 slots 字典动态生成）
        slot_head = QLabel("各项评分依据（黄色=本次主要依据）")
        slot_head.setToolTip(
            "模型从多个维度独立打分（如语义相似/缺陷判别/模板比对等），\n"
            "每条的值是该维度判「异常」的分数（0~1，越高越像缺陷）；\n"
            "上方仪表盘的综合分 = 各项按「融合权重」加权求和，\n"
            "黄色高亮项是本次判定贡献最大的依据")
        slot_head.setProperty("subtext", True)
        lay.addWidget(slot_head)
        self._slots_lay = QVBoxLayout()
        self._slot_bars: dict[str, tuple[QLabel, object]] = {}
        lay.addLayout(self._slots_lay)

        form = QFormLayout()
        self.lbl_latency = QLabel("-")
        form.addRow("检测延迟(端到端)", self.lbl_latency)
        self.lbl_threshold = QLabel("-")
        form.addRow("决策阈值", self.lbl_threshold)
        self.lbl_weights = QLabel("-")
        form.addRow("融合权重", self.lbl_weights)
        self.lbl_boxes = QLabel("-")
        form.addRow("缺陷框", self.lbl_boxes)
        self.lbl_trigger = QLabel("-")
        form.addRow("主要评分依据", self.lbl_trigger)
        lay.addLayout(form)
        lay.addStretch(1)
        return card

    # ══════════════════ 页面生命周期 ══════════════════
    def reload(self) -> None:
        """页面激活时刷新（产线状态 + 数据流队列 + 实时流重订阅）。"""
        self._trial_cat = ""   # 工单可能已切换：多品类试检选择缓存失效
        self.reload_pipeline()
        self._load_queue()
        self._reset_live()

    # ══════════════════ 产线控制 ══════════════════
    def _workorder_id(self) -> int | None:
        fn = getattr(self, "_get_workorder_id", None)
        return fn() if callable(fn) else None

    def reload_pipeline(self) -> None:
        wid = self._workorder_id()
        if wid is None:
            self.lbl_pipeline.setText("产线：-")
            self.lbl_pipeline_hint.hide()
            self.btn_go_prepare.hide()
            return
        # 生成号自增：丢弃旧轮询的迟到响应，防止把刚更新的
        # 「已暂停」覆盖回「运行中」（前端反馈 v5-1 的时序竞争根因）。
        self._pipe_gen = getattr(self, "_pipe_gen", 0) + 1
        gen = self._pipe_gen
        run_async(self, lambda: self._client.get_workorder(wid),
                  lambda wo: self._fill_pipeline_guarded(gen, wo))

    def _on_pipe_tick(self) -> None:
        if self.isVisible() and self._workorder_id() is not None:
            self.reload_pipeline()
            self._load_queue()

    def _fill_pipeline_guarded(self, gen: int, wo) -> None:
        if gen != getattr(self, "_pipe_gen", 0):
            return  # 已有更新的请求发出，这份响应已过期，丢弃
        self._fill_pipeline(wo)

    def _set_pipeline_badge(self, status) -> None:
        # status: "running" / "paused"（人工）/ "auto_paused"（背压自动）/
        # "stopped"（停止任务，回队队列已清空，只能手动恢复），
        # 也兼容旧调用传 bool（True=running）。
        if isinstance(status, bool):
            status = "running" if status else "paused"
        if status == "running":
            text, badge = "产线：运行中", "success"
        elif status == "auto_paused":
            text, badge = "产线：自动暂停（复判积压限流）", "warning"
        elif status == "stopped":
            text, badge = "产线：已停止（回队队列已清空，需手动恢复）", "danger"
        else:
            text, badge = "产线：已暂停", "warning"
        self.lbl_pipeline.setText(text)
        self.lbl_pipeline.setProperty("badge", badge)
        self.lbl_pipeline.style().unpolish(self.lbl_pipeline)
        self.lbl_pipeline.style().polish(self.lbl_pipeline)
        # 按钮状态与产线状态保持一致，避免「按钮行为与状态不一致」
        running = status == "running"
        self.btn_pp_pause.setEnabled(running)
        self.btn_pp_resume.setEnabled(not running)
        self.btn_pp_stop.setEnabled(status != "stopped")

    def _fill_pipeline(self, wo) -> None:
        wo = wo or {}
        st = str(wo.get("pipeline_status") or "")
        running = st == "running"
        self._set_pipeline_badge(st)
        self.chk_auto_resume.blockSignals(True)
        self.chk_auto_resume.setChecked(bool(wo.get("auto_resume")))
        self.chk_auto_resume.blockSignals(False)
        # 复判积压显示（前端反馈 v6-3）：开启人工复判的工单才显示
        review_pending = wo.get("review_pending")
        if review_pending is not None:
            n_rev = int(review_pending)
            self.lbl_pipeline_hint.setText(
                f"待复核 {n_rev} 条"
                + (" · 复判积压限流已启动（产线暂停送检）"
                   if not running and n_rev > 0 else ""))
            self.lbl_pipeline_hint.show()
            self.btn_go_prepare.hide()
            return
        if not running:
            self.lbl_pipeline_hint.hide()
            self.btn_go_prepare.hide()
            return
        sm = wo.get("pipeline_summary") or {}
        n_pending = int(sm.get("n_pending") or 0)
        unprep = [str(c) for c in (sm.get("unprepared_categories") or [])]
        n_blocked = int(sm.get("n_images_blocked") or 0)
        parts = [f"待检测 {n_pending} 张"]
        if unprep:
            parts.append(f"{len(unprep)} 个品类未准备"
                         f"（{n_blocked} 张图整批跳过送检）"
                         f"：{'、'.join(unprep)}")
            self.btn_go_prepare.show()
        else:
            self.btn_go_prepare.hide()
        if n_pending == 0 and not unprep:
            parts.append("产线暂无可检数据，可在「数据管理」回队批次或导入新数据")
        self.lbl_pipeline_hint.setText(" · ".join(parts))
        self.lbl_pipeline_hint.show()

    def _pipeline_ctl(self, action: str) -> None:
        wid = self._workorder_id()
        if wid is None:
            warn(self, "请先在顶栏选择当前工单")
            return
        # 请求在途时禁用控制按钮，防止连点造成状态竞态
        self.btn_pp_pause.setEnabled(False)
        self.btn_pp_resume.setEnabled(False)
        self.btn_pp_stop.setEnabled(False)

        def _ok(r) -> None:
            if action == "pause":
                msg = "产线已暂停（复判/学习/编排数据流不受影响）"
            elif action == "stop":
                n = int((r or {}).get("cleared_requeued") or 0)
                msg = f"产线已停止（已清空回队批次 {n} 个；需手动恢复产线）"
            else:
                msg = "产线已恢复"
            notify(self, msg)
            # 乐观更新：直接用控制响应的状态刷新徽章，不等下一次轮询
            st = str((r or {}).get("pipeline_status") or "")
            if st:
                self._set_pipeline_badge(st)
            # 再拉全量（待检数/未准备品类等提示），由生成号保证不被旧响应覆盖
            self.reload_pipeline()

        def _fail(m) -> None:
            warn(self, str(m))
            self.reload_pipeline()  # 失败时按真实状态恢复按钮

        run_async(self, lambda: self._client.pipeline_control(wid, action),
                  _ok, _fail)

    def _on_pipeline_stop(self) -> None:
        """停止产线（带确认）：清空回队队列，不会被自动恢复。"""
        if self._workorder_id() is None:
            warn(self, "请先在顶栏选择当前工单")
            return
        ret = QMessageBox.question(
            self, "停止产线",
            "停止产线将清空该工单的回队队列（暂停期间编排的重检批次），"
            "且不会被自动恢复，只能手动「恢复产线」重新启动。\n确定停止？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if ret == QMessageBox.Yes:
            self._pipeline_ctl("stop")

    def _export_wo_report(self) -> None:
        """导出当前工单检测报告 CSV（后端生成，前端仅选路径落盘，
        写法参考 stats_page 导出按钮）。"""
        wid = self._workorder_id()
        if wid is None:
            warn(self, "请先在顶栏选择当前工单")
            return
        save_path, _ = QFileDialog.getSaveFileName(
            self, "导出工单检测报告", f"工单检测报告_{wid}.csv",
            "CSV 文件 (*.csv)")
        if not save_path:
            return

        def _done(data) -> None:
            if not data:
                warn(self, "导出失败：后端无响应")
                return
            try:
                with open(save_path, "wb") as f:
                    f.write(data)
            except OSError as e:
                warn(self, f"保存失败：{e}")
                return
            notify(self, f"已保存：{save_path}")

        run_async(self, lambda: self._client.download_workorder_report(wid),
                  _done, lambda m: warn(self, str(m)))

    def _pipeline_learn(self) -> None:
        wid = self._workorder_id()
        if wid is None:
            warn(self, "请先在顶栏选择当前工单")
            return
        self.btn_pp_learn.setEnabled(False)

        def _done(res) -> None:
            self.btn_pp_learn.setEnabled(True)
            notify(self, str((res or {}).get("message") or "学习提升完成"))
            self.reload_pipeline()

        run_async(self, lambda: self._client.workorder_learn(wid),
                  _done, lambda m: (self.btn_pp_learn.setEnabled(True),
                                    warn(self, str(m))))

    def _save_auto_resume(self, checked: bool) -> None:
        wid = self._workorder_id()
        if wid is None:
            return
        run_async(self, lambda: self._client.update_workorder(
            wid, auto_resume=bool(checked)), lambda _r: None, lambda _m: None)

    # ══════════════════ 检测组数据流队列 ══════════════════
    def _build_queue_panel(self) -> QWidget:
        """当前工单的检测组数据流队列（30 图/批）：待检/检测/不良/反馈统计，
        错检批次标记 + 回队/取消（产线暂停期间可编排；运行中也可插队）。
        位于左栏试检面板下方（左右布局），高度随窗口弹性伸展；
        可折叠：点击标题栏收起/展开。"""
        box = QGroupBox("数据流队列（当前工单 · 30图/批）")
        box.setToolTip("错检批次可回队重新检测（允许插队）；"
                       "产线运行中回队 = 立即优先重检该批")
        box.setCheckable(True)
        box.setChecked(True)
        body = QWidget()
        v = QVBoxLayout(body)
        v.setContentsMargins(10, 6, 10, 6)
        v.setSpacing(4)
        self.queue_table = QTableWidget(0, 6)
        self.queue_table.setHorizontalHeaderLabels(
            ["批次", "数据源", "品类", "图片/检测/不良/反馈", "错检批次", "回队"])
        self.queue_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.queue_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.queue_table.verticalHeader().setVisible(False)
        self.queue_table.horizontalHeader().setStretchLastSection(True)
        v.addWidget(self.queue_table, 1)
        row = QHBoxLayout()
        self.lbl_queue_hint = QLabel("选中行后操作")
        self.lbl_queue_hint.setProperty("subtext", True)
        row.addWidget(self.lbl_queue_hint)
        row.addStretch(1)
        btn_reload = QPushButton("刷新队列")
        btn_reload.setProperty("flat", True)
        btn_reload.clicked.connect(self._load_queue)
        row.addWidget(btn_reload)
        self.btn_queue_req = QPushButton("回队（重新检测）")
        self.btn_queue_req.clicked.connect(lambda: self._queue_requeue(True))
        self.btn_queue_unreq = QPushButton("取消回队")
        self.btn_queue_unreq.clicked.connect(lambda: self._queue_requeue(False))
        row.addWidget(self.btn_queue_req)
        row.addWidget(self.btn_queue_unreq)
        v.addLayout(row)
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(body)
        box.toggled.connect(body.setVisible)
        return box

    def _load_queue(self) -> None:
        wid = self._workorder_id()
        if wid is None:
            self.queue_table.setRowCount(0)
            return
        run_async(self, lambda: self._client.workorder_queue(wid),
                  self._fill_queue)

    def _fill_queue(self, data) -> None:
        data = data or {}
        items = data.get("items") or []
        self.queue_table.setRowCount(max(len(items), 1))
        if not items:
            cell = QTableWidgetItem("（当前工单无检测组批次：请先在数据管理为数据源"
                                    "配置分组并挂接到工单）")
            cell.setFlags(Qt.ItemIsEnabled)
            self.queue_table.setItem(0, 0, cell)
            self.queue_table.setSpan(0, 0, 1, 6)
            return
        highlighted = False
        for r, it in enumerate(items):
            cells = [
                str(it.get("name")), str(it.get("source")),
                str(it.get("category")),
                f"{it.get('n_images')}/{it.get('n_detected')}/"
                f"{it.get('n_anomaly')}/{it.get('n_feedback')}",
                "⚠ 错检高发" if it.get("bad_batch") else "—",
                "已回队" if it.get("requeued") else "—",
            ]
            # 正在消费中的批次（首个未检完的批次）高亮
            in_progress = (not highlighted
                           and int(it.get("n_detected") or 0)
                           < int(it.get("n_images") or 0))
            for c, txt in enumerate(cells):
                cell = QTableWidgetItem(txt)
                if it.get("bad_batch"):
                    cell.setForeground(QColor("#E67E22"))
                if in_progress:
                    cell.setBackground(QColor("#DBEAFE"))
                self.queue_table.setItem(r, c, cell)
                if c == 0:
                    cell.setData(Qt.UserRole, it.get("batch_key"))
            highlighted = highlighted or in_progress
        self.queue_table.resizeColumnsToContents()

    def _selected_queue_keys(self) -> list[str]:
        out = []
        for r in range(self.queue_table.rowCount()):
            it = self.queue_table.item(r, 0)
            if it is not None and it.isSelected():
                k = it.data(Qt.UserRole)
                if k:
                    out.append(str(k))
        return out

    def _queue_requeue(self, requeue: bool) -> None:
        wid = self._workorder_id()
        keys = self._selected_queue_keys()
        if wid is None or not keys:
            warn(self, "请先在队列中选择检测批次（选中行）")
            return
        run_async(self, lambda: self._client.queue_requeue(wid, keys, requeue),
                  lambda _r: (notify(self, "已回队：该批图片将优先重新检测"
                                    if requeue else "已取消回队"),
                              self._load_queue()),
                  lambda m: warn(self, str(m)))

    # ══════════════════ 试检（旁路调试）══════════════════
    def _category(self) -> str:
        """试检品类：单品类工单自动推导；多品类工单弹窗选择；无工单提示。

        （前端反馈 2026-09-13 #9：顶栏品类下拉移除后，多品类工单
        current_category() 返回空，试检被一句"请选择品类"卡死——
        改为从当前工单品类集推导/点选。）"""
        cat = self._get_category() if self._get_category else ""
        if cat:
            return cat
        wo_fn = getattr(self, "_get_workorder", None)
        wo = wo_fn() if callable(wo_fn) else None
        if wo is None:
            warn(self, "请先在顶栏选择当前工单（试检品类随工单推导）")
            return ""
        cats = [str(c) for c in (wo.get("categories") or []) if c]
        if not cats:
            warn(self, "当前工单没有品类数据：请检查工单挂接的数据源")
            return ""
        if len(cats) == 1:
            return cats[0]
        # 多品类工单：记住本次选择，避免每张试检图都弹窗
        cached = getattr(self, "_trial_cat", "")
        if cached in cats:
            return cached
        cat, ok = QInputDialog.getItem(
            self, "选择试检品类", "当前工单含多个品类，本次试检用哪个模型？",
            cats, 0, False)
        if not ok or not cat:
            return ""
        self._trial_cat = str(cat)
        return self._trial_cat

    def _on_upload_detect(self) -> None:
        cat = self._category()
        if not cat:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "选择检测图片", "", "图片文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if not path:
            return
        self._set_busy("试检中…")
        run_async(self, lambda: self._client.detect_upload(path, cat),
                  self._on_trial_done)

    def _on_pick_detect(self) -> None:
        cat = self._category()
        if not cat:
            return
        dlg = ImagePickDialog(self._client, cat, self)
        if dlg.exec() != QDialog.Accepted or not dlg.selected_image:
            return
        img = dlg.selected_image
        self._set_busy("试检中…")
        run_async(self, lambda: self._client.detect_image(
            cat, image_id=img.get("id")), self._on_trial_done)

    def _on_trial_done(self, det) -> None:
        """试检结果展示（旁路：不进产线流、判定条带与分数曲线）。"""
        if not det:
            return
        self.lbl_live.setText("试检结果（旁路，不在产线流中）")
        self.lbl_live.setProperty("badge", "info")
        self._repolish(self.lbl_live)
        self._show_detection(det)

    def _set_busy(self, text: str) -> None:
        self.badge_verdict.setText(text)
        self.badge_verdict.setProperty("badge", "info")
        self.badge_verdict.style().unpolish(self.badge_verdict)
        self.badge_verdict.style().polish(self.badge_verdict)

    # ══════════════════ 产线实时流（轮询订阅）══════════════════
    def _reset_live(self) -> None:
        """重订阅实时流：清空窗口并拉取最新 LIVE_WINDOW 帧。"""
        self._frames = []
        self._last_id = 0
        self._live_retry_count = 0
        self._live_last_error = ""
        self._live_dropped_count = 0
        self._selected_frame = -1
        self._poll_live()

    def _poll_live(self) -> None:
        if not self.isVisible() or self._polling:
            return
        self._polling = True

        def _done(data) -> None:
            self._polling = False
            was_error = bool(self._live_last_error)
            self._live_retry_count = 0
            self._live_last_error = ""
            self._on_live_batch(data)
            if was_error and not (data or {}).get("gap"):
                self.lbl_live.setText("实时流连接已恢复")
                self.lbl_live.setProperty("badge", "success")
                self._repolish(self.lbl_live)

        def _err(message) -> None:
            self._polling = False
            self._live_retry_count += 1
            self._live_last_error = str(message)
            self.lbl_live.setText(
                f"实时流连接异常，正在重试（第 {self._live_retry_count} 次）")
            self.lbl_live.setProperty("badge", "danger")
            self._repolish(self.lbl_live)
            self.lbl_live_kpi.setText(f"最近错误：{self._live_last_error}")

        run_async(self, lambda: self._client.detections_live(
            after_id=self._last_id, limit=LIVE_WINDOW,
            workorder_id=self._workorder_id()), _done, _err)

    def _on_live_batch(self, data) -> None:
        if data is None:
            return  # 离线/失败静默（detections_live 为 silent）
        items = data.get("items") or []
        alarm_map = data.get("alarm") or {}
        if data.get("gap") or data.get("reset_required"):
            dropped = int(data.get("dropped_count") or 0)
            self._live_dropped_count += max(0, dropped)
            self.lbl_live.setText(
                f"实时流警告：检测记录丢帧 {dropped} 条，已重置订阅位置")
            self.lbl_live.setProperty("badge", "warning")
            self._repolish(self.lbl_live)
        if not items:
            if not self._frames:
                self.lbl_live.setText("实时流：产线暂无检测（启动产线后自动出现）")
                self.lbl_live.setProperty("badge", "muted")
                self._repolish(self.lbl_live)
            return
        for row in items:
            rid = int(row.get("id") or 0)
            if rid <= self._last_id:
                continue
            self._last_id = rid
            alarm = alarm_map.get(str(row.get("category") or ""))
            self._frames.append(_row_to_det(row, alarm))
        self._frames = self._frames[-LIVE_WINDOW:]
        self._refresh_strip()
        self._refresh_chart()
        # 跟随最新：画面/结果面板滚动到最新帧
        if self.chk_follow.isChecked() or self._selected_frame < 0:
            self._selected_frame = -1
            self._show_detection(self._frames[-1], from_frame=True)
        self._update_live_kpi()

    def _update_live_kpi(self) -> None:
        """左栏实时流 KPI：窗口帧数/异常率/实测吞吐。"""
        n = len(self._frames)
        if not n:
            return
        n_anom = sum(1 for f in self._frames
                     if str(f.get("decision")) == "anomaly")
        n_gray = sum(1 for f in self._frames
                     if str(f.get("decision")) == "gray")
        fps_text = "-"
        try:
            t0 = datetime.fromisoformat(str(self._frames[0]["created_at"]))
            t1 = datetime.fromisoformat(str(self._frames[-1]["created_at"]))
            span = (t1 - t0).total_seconds()
            if span > 0 and n > 1:
                fps_text = f"{(n - 1) / span:.1f} 帧/s"
        except (TypeError, ValueError):
            pass
        if self._live_dropped_count:
            self.lbl_live.setText(
                f"实时流警告：丢帧 {self._live_dropped_count} 条 · 最新 #{self._last_id}")
            self.lbl_live.setProperty("badge", "warning")
        else:
            self.lbl_live.setText(
                f"实时流：窗口 {n} 帧 · 最新 #{self._last_id}")
            self.lbl_live.setProperty("badge", "success")
        self._repolish(self.lbl_live)
        self.lbl_live_kpi.setText(
            f"异常 {n_anom} · 灰区 {n_gray} · 吞吐 {fps_text}"
            f" · 重试 {self._live_retry_count} · 丢帧 {self._live_dropped_count}")
        self.lbl_strip_stats.setText(
            f"异常率 {n_anom / n:.0%}（{n_anom}/{n}）")

    def _refresh_strip(self) -> None:
        """重建判定条带（≤50 格，重建开销可忽略）。

        孤立异常脉冲（连续 <2 帧）按 algo §6 时序语义以灰底描边降级呈现。
        """
        while self.strip_lay.count():
            item = self.strip_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._strip_cells = []
        self._pulses = _pulse_flags(self._frames)
        for i, fr in enumerate(self._frames):
            decision = str(fr.get("decision") or "normal")
            pulse = self._pulses[i]
            color = WARNING if pulse else DECISION_COLORS.get(decision, SUCCESS)
            cell = QPushButton()
            cell.setFixedSize(14, 22)
            cell.setFlat(True)
            tip = (f"#{fr.get('detection_id')} {fr.get('category')} "
                   f"分数 {float(fr.get('final_score') or 0):.3f} "
                   f"判定 {decision}")
            if pulse:
                tip += "（短脉冲：孤立异常，时序降级）"
            cell.setToolTip(tip)
            cell.setStyleSheet(
                f"QPushButton {{ background: {color}; border: none; "
                f"border-radius: 2px; }}")
            cell.clicked.connect(
                lambda _c=False, idx=i: self._on_strip_click(idx))
            self.strip_lay.addWidget(cell)
            self._strip_cells.append(cell)
        self.strip_lay.addStretch(1)
        self._mark_strip_selection()

    def _mark_strip_selection(self) -> None:
        """条带选中态：回看帧加描边，跟随模式无选中。"""
        pulses = getattr(self, "_pulses", [False] * len(self._frames))
        for i, cell in enumerate(self._strip_cells):
            fr = self._frames[i]
            decision = str(fr.get("decision") or "normal")
            color = (WARNING if pulses[i]
                     else DECISION_COLORS.get(decision, SUCCESS))
            border = ("2px solid #111827"
                      if i == self._selected_frame else "none")
            cell.setStyleSheet(
                f"QPushButton {{ background: {color}; border: {border}; "
                f"border-radius: 2px; }}")

    def _on_strip_click(self, idx: int) -> None:
        """点击条带回看历史帧：取消跟随，展示该帧。"""
        if not (0 <= idx < len(self._frames)):
            return
        self.chk_follow.blockSignals(True)
        self.chk_follow.setChecked(False)
        self.chk_follow.blockSignals(False)
        self._selected_frame = idx
        self._mark_strip_selection()
        self._show_detection(self._frames[idx], from_frame=True)

    def _refresh_chart(self) -> None:
        """分数曲线：原始分 + EMA 平滑双序列（algo §6 时序平滑口径）。"""
        self.score_chart.clear()
        if not self._frames:
            return
        scores = [float(f.get("final_score") or 0.0) for f in self._frames]
        x = list(range(len(scores)))
        self.score_chart.add_series("原始分", x, scores,
                                    color=PRIMARY, width=2, symbol=False)
        self.score_chart.add_series("EMA平滑", x, _ema(scores),
                                    color=WARNING, width=2, symbol=False)
        self.score_chart.set_labels(y_title="融合分数")

    # ══════════════════ 结果展示 ══════════════════
    def _update_slot_bars(self, slots: dict, triggered_slot: str = "") -> None:
        """按 slots 字典动态重建评分条（工作点不同评分项集合不同）。"""
        from PySide6.QtWidgets import QProgressBar
        names = sorted(slots)
        if list(self._slot_bars) != names:
            # 评分项集合变化：清空重建
            while self._slots_lay.count():
                item = self._slots_lay.takeAt(0)
                if item.layout():
                    while item.layout().count():
                        sub = item.layout().takeAt(0)
                        if sub.widget():
                            sub.widget().deleteLater()
                elif item.widget():
                    item.widget().deleteLater()
            self._slot_bars = {}
            for name in names:
                row = QHBoxLayout()
                lb = QLabel(SLOT_CN.get(name, name))
                lb.setMinimumWidth(90)
                lb.setToolTip(f"评分依据：{SLOT_CN.get(name, name)}")
                row.addWidget(lb)
                bar = QProgressBar()
                bar.setRange(0, 1000)
                bar.setValue(0)
                bar.setTextVisible(True)
                row.addWidget(bar, 1)
                self._slots_lay.addLayout(row)
                self._slot_bars[name] = (lb, bar)
        for name, (lb, bar) in self._slot_bars.items():
            try:
                fv = float(slots.get(name) or 0.0)
            except (TypeError, ValueError):
                fv = 0.0
            fv = max(0.0, min(1.0, fv))
            bar.setValue(int(fv * 1000))
            bar.setFormat(f"{fv:.3f}")
            hit = bool(triggered_slot) and name == triggered_slot
            color = WARNING if hit else PRIMARY
            bar.setStyleSheet(
                f"QProgressBar::chunk {{ background: {color}; border-radius: 5px; }}")
            lb.setStyleSheet(
                f"color: {WARNING}; font-weight: 700;" if hit else "")

    def _show_detection(self, det, from_frame: bool = False) -> None:
        if not det or not isinstance(det, dict):
            return
        self._current_detection = det

        # 图像三图层（产线帧不落热力图：无热力图时清空旧图层防串图）
        if not det.get("heatmap_path") and not det.get("overlay_path"):
            self.viewer.clear_images()
        path_map = {"original": det.get("image_path"),
                    "heatmap": det.get("heatmap_path"),
                    "overlay": det.get("overlay_path")}
        for layer in LAYERS:
            p = path_map.get(layer)
            if p:
                self.viewer.set_image_url(self._client.file_url(p), layer)
        boxes = det.get("defect_boxes") or []
        self.viewer.set_boxes(boxes)

        # 仪表盘与三态判定（normal 绿 / anomaly 红 / gray 黄"待复核"）
        score = float(det.get("final_score", 0) or 0)
        self.gauge.set_score(score, det.get("threshold"))
        decision = str(det.get("decision") or
                       ("anomaly" if det.get("is_anomaly") else "normal"))
        text, badge = {
            "anomaly": ("异 常", "danger"),
            "gray": ("待复核（灰区）", "warning"),
        }.get(decision, ("正 常", "success"))
        self.badge_verdict.setText(text)
        self.badge_verdict.setProperty("badge", badge)
        self._repolish(self.badge_verdict)
        # 灰区帧自动进入待复核队列（反馈页→待复核 Tab）
        self.lbl_review_hint.setVisible(decision == "gray")

        # 开放集未知模式徽标
        self.badge_open.setVisible(bool(det.get("open_alert")))

        # 缺陷类型归因徽标（top-1 类型 + 置信度，仅非正常时显示）
        types = det.get("defect_types") or []
        if decision != "normal" and types:
            top = types[0]
            conf = float(top.get("confidence") or 0)
            self.badge_attr.setText(
                f"归因：{top.get('type', '未知')}（{conf:.0%}）")
            self.badge_attr.show()
        else:
            self.badge_attr.hide()

        # 对位偏移预警徽标（align_warn 时显示偏移量）
        if det.get("align_warn"):
            off = det.get("align_offset") or [0, 0]
            self.badge_align.setText(f"对位偏移（{off[0]:+.1f}, {off[1]:+.1f}）")
            self.badge_align.show()
        else:
            self.badge_align.hide()

        # 连续异常告警横幅（alarm.active 时保持显示，下一帧消息再更新）
        alarm = det.get("alarm")
        if isinstance(alarm, dict) and alarm.get("active"):
            hits = alarm.get("window_hits", "?")
            self.lbl_alarm.setText(
                f"⚠ 连续异常告警：近期窗口内 {hits} 帧异常（建议停线检查）")
            self.lbl_alarm.show()
        elif isinstance(alarm, dict):
            self.lbl_alarm.hide()

        # 各项评分依据 + 主要依据高亮（slots 缺失时回退 level1/2/3 分）
        slots = det.get("slots") if isinstance(det.get("slots"), dict) else {}
        if not slots:
            for name, key in (("sem", "level1_score"), ("disc", "level2_score"),
                              ("shead", "level3_score")):
                v = det.get(key)
                if v is not None:
                    slots[name] = v
        triggered_slot = str(det.get("triggered_slot") or "")
        self._update_slot_bars(slots, triggered_slot)

        # 延迟与预算对比：主口径端到端（含解码/持久化），推理耗时作副指标
        latency = det.get("latency_e2e_ms")
        if latency is None:
            latency = det.get("total_latency_ms", det.get("latency_ms"))
        budget = float(self._get_budget() or 200.0)
        try:
            lat = float(latency)
            ok = lat < budget
            text = f"{lat:.1f} ms（端到端含解码/持久化，预算 <{budget:.0f} ms）"
            infer = det.get("latency_ms")
            if isinstance(infer, (int, float)):
                text += f"｜推理 {float(infer):.1f} ms"
            self.lbl_latency.setText(text)
            self.lbl_latency.setStyleSheet(
                f"color: {SUCCESS if ok else DANGER}; font-weight: 700;")
        except (TypeError, ValueError):
            self.lbl_latency.setText("-")
            self.lbl_latency.setStyleSheet("")

        # 决策阈值（含灰区下限）与融合权重
        thr = det.get("threshold")
        gray_thr = det.get("gray_threshold")
        thr_text = f"{float(thr):.3f}" if isinstance(thr, (int, float)) else "-"
        if isinstance(gray_thr, (int, float)):
            thr_text += f"（灰区 ≥{float(gray_thr):.3f}）"
        self.lbl_threshold.setText(thr_text)
        weights = det.get("weights") if isinstance(det.get("weights"), dict) else {}
        self.lbl_weights.setText(
            "  ".join(f"{SLOT_CN.get(k, k)}={float(v):.2f}"
                      for k, v in sorted(weights.items())
                      if isinstance(v, (int, float))) or "-")
        self.lbl_boxes.setText(f"{len(boxes)} 个")
        self.lbl_trigger.setText(SLOT_CN.get(triggered_slot, triggered_slot)
                                 or "无（多依据综合判定）")

        # 反馈按钮：无 detection_id 时禁用
        has_id = det.get("detection_id", det.get("id")) is not None
        feedback_supported = det.get("feedback_supported") is True
        for b in (self.btn_fp, self.btn_fn, self.btn_ok):
            b.setEnabled(has_id and feedback_supported)

    def _repolish(self, w: QWidget) -> None:
        w.style().unpolish(w)
        w.style().polish(w)

    # ══════════════════ 反馈 ══════════════════
    def _open_feedback(self, preset: str) -> None:
        if not self._current_detection:
            warn(self, "当前没有可标注的检测记录")
            return
        dlg = FeedbackDialog(self._client, self._current_detection,
                             preset_type=preset, parent=self)
        dlg.exec()
