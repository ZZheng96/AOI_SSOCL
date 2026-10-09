"""数据管理页：图像 / 视频 / 伪异常 三个选项卡。

图像：筛选栏（category/split/label）+ 左侧四维数据树导航（品类→数据集批次
      →划分→标签，与筛选栏联动）+ 缩略图表格（首列缩略图、批次/来源/导入时间
      列、双击看大图、右键标注/删除）+ 导入文件夹 / 导入MVTec / 上传图片
      （均可填批次名/备注）+ 孤儿文件清理。
视频：表格 + 导入按钮。
伪异常：表格（方法/掩码预览/时间）。
"""
from __future__ import annotations

import csv
import os
import re
import time
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QCompleter,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ui.api_client import ApiClient
from ui.pages.common import (
    LAYER_CN,
    TIER_CN,
    attach_level_badge,
    load_thumb_async,
    make_banner,
    make_thumb_cell,
    notify,
    run_async,
    warn,
)
from ui.widgets.image_viewer import ImagePreviewDialog
from ui.widgets.task_progress import TaskMonitor, TaskProgressBar

# 仓库根（AOI_sys）：相对路径统一以此为基准解析，保证工程整体搬移后可移植
AOI_ROOT = Path(__file__).resolve().parents[2]
# 内置兜底默认（取不到配置时回退）；优先级：环境变量 AOI_DATA_ROOT >
# 后端设置中心 data.default_root（configs/default.yaml）> 仓库内 data/
DEFAULT_DATA_ROOT = "data"


def _default_data_root(client: ApiClient | None = None) -> str:
    """数据集默认根目录：环境变量 AOI_DATA_ROOT > 后端 /system/config 的
    data.default_root > 仓库内 data/。相对路径按仓库根解析。"""
    raw = (os.environ.get("AOI_DATA_ROOT") or "").strip()
    if not raw and client is not None:
        # 静默读取：后端不可达只是回退默认值，不弹错误提示
        cfg = client.get("/api/system/config", silent=True) or {}
        for it in (cfg.get("items") or []):
            if it.get("section") == "data" and it.get("key") == "default_root":
                raw = str(it.get("value") or "").strip()
                break
    p = Path(raw) if raw else Path(DEFAULT_DATA_ROOT)
    if not p.is_absolute():
        p = AOI_ROOT / p
    return str(p)

# Image.source → 中文来源名
SOURCE_CN = {
    "manual": "手动导入", "folder": "手动导入", "mvtec": "MVTec",
    "dagm": "DAGM", "upload": "上传", "pseudo": "伪异常",
    "augment": "增强", "augmented": "增强", "video": "视频帧",
    "feedback": "反馈回流", "legacy": "历史数据",
    "adapt": "智能适配",
}

# 表格列数（含批次/来源/导入时间）
_N_COLS = 11


def _fmt_size(path: str) -> str:
    """文件大小友好显示；文件不可访问时返回 '-'。"""
    try:
        n = os.path.getsize(path)
    except OSError:
        return "-"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.1f}MB"
    if n >= 1 << 10:
        return f"{n / (1 << 10):.1f}KB"
    return f"{n}B"


class DatasetAdaptConfirmDialog(QDialog):
    """数据集识别与确认（数据集适配模块）。

    展示后端 probe 识别报告（格式/品类/分组/计数/告警），
    用户确认或修正映射（品类名、未识别分组处理方式、正常图角色）
    后返回导入配置。
    """

    def __init__(self, report: dict, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("数据集识别与确认")
        self.resize(760, 560)
        self._report = report or {}
        self._fmt = str(self._report.get("format") or "dir_rules")
        cats = self._report.get("categories") or []

        v = QVBoxLayout(self)
        v.setSpacing(8)
        info = QLabel(
            f"识别格式：{self._report.get('format_cn', '')}\n"
            f"根目录：{self._report.get('root', '')}\n"
            f"共 {self._report.get('n_images', 0)} 张图片"
            f"｜掩码 {self._report.get('n_masks', 0)} 张（不入库）"
            f"｜其他文件 {self._report.get('n_other', 0)} 个")
        info.setProperty("heading", True)
        info.setWordWrap(True)
        v.addWidget(info)
        desc = QLabel(str(self._report.get("format_desc") or ""))
        desc.setProperty("subtext", True)
        desc.setWordWrap(True)
        v.addWidget(desc)

        f = QFormLayout()
        if self._fmt == "mvtec_like":
            f.addRow("品类", QLabel(f"{len(cats)} 个品类（按目录自动划分，见下表勾选）"))
        else:
            self.edit_category = QLineEdit(
                str(self._report.get("category_hint") or "default"))
            self.edit_category.setToolTip("导入后的品类名（模型按品类准备）")
            f.addRow("品类名", self.edit_category)
        # dir_rules/plain：命名识别为正常的图片（_OK / ok / 良品 / 误报）的角色
        self.combo_okrole = QComboBox()
        self.combo_okrole.addItem("预训练正常图（train，推荐）", "train")
        self.combo_okrole.addItem("模板图（L3 模板比对）", "template")
        self.combo_okrole.setToolTip(
            "命名识别为正常的图片（_OK 后缀 / ok・良品・误报目录）用作"
            "预训练正常图或 L3 模板图；模板目录（模板/template）始终进模板库")
        if self._fmt in ("dir_rules", "plain"):
            f.addRow("正常图角色", self.combo_okrole)
        v.addLayout(f)

        # 分组/品类表格
        self.table = QTableWidget()
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self._overrides: dict[str, QComboBox] = {}
        self._cat_checks: dict[str, QCheckBox] = {}
        if self._fmt == "mvtec_like":
            self._build_mvtec_table(cats)
        elif self._fmt == "yolo_split":
            self._build_yolo_summary(cats)
        else:
            self._build_groups_table(cats)
        v.addWidget(self.table, 1)

        warn_lines = self._report.get("warnings") or []
        if warn_lines:
            w = QLabel("⚠ " + "\n⚠ ".join(str(x) for x in warn_lines))
            w.setProperty("subtext", True)
            w.setWordWrap(True)
            w.setStyleSheet("color:#B45309;")
            v.addWidget(w)

        row = QHBoxLayout()
        row.addStretch(1)
        btn_ok = QPushButton("确认导入")
        btn_ok.setProperty("primary", True)
        btn_cancel = QPushButton("取消")
        row.addWidget(btn_ok)
        row.addWidget(btn_cancel)
        v.addLayout(row)
        btn_ok.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)

    # ── mvtec_like：品类勾选表 ──
    def _build_mvtec_table(self, cats: list) -> None:
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels(
            ["导入", "品类", "训练正常", "检测正常", "检测缺陷",
             "模板图", "未识别", "掩码"])
        self.table.setRowCount(len(cats))
        for r, c in enumerate(cats):
            counts = c.get("counts") or {}
            n_anom = sum(v for k, v in counts.items()
                         if str(k).endswith("_anomaly"))
            chk = QCheckBox()
            chk.setChecked(bool(c.get("n_images")))
            self._cat_checks[str(c.get("name"))] = chk
            cell_w = QWidget()
            h = QHBoxLayout(cell_w)
            h.setContentsMargins(0, 0, 0, 0)
            h.addWidget(chk, 0, Qt.AlignCenter)
            vals = ["", str(c.get("name")),
                    str(counts.get("train_normal", 0)
                        + counts.get("val_normal", 0)),
                    str(counts.get("test_normal", 0)),
                    str(n_anom), str(counts.get("template", 0)),
                    str(counts.get("unknown", 0)),
                    str(c.get("n_masks", 0))]
            for col, val in enumerate(vals):
                if col == 0:
                    self.table.setCellWidget(r, col, cell_w)
                    continue
                item = QTableWidgetItem(val)
                if col == 4 and int(val or 0) > 0:
                    item.setForeground(QColor("#B91C1C"))
                if col in (2, 3, 4, 5, 6, 7):
                    item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(r, col, item)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.resizeColumnsToContents()

    # ── yolo_split：划分计数摘要 ──
    def _build_yolo_summary(self, cats: list) -> None:
        counts = (cats[0].get("counts") or {}) if cats else {}
        self.table.setColumnCount(6)
        self.table.setRowCount(2)
        self.table.setHorizontalHeaderLabels(
            ["划分", "正常", "缺陷", "划分", "正常", "缺陷"])
        cells_map = {(0, 0): ("train", "训练"), (1, 0): ("val", "验证"),
                     (0, 3): ("test", "检测")}
        for (r, c0), (sp, cn) in cells_map.items():
            for j, val in enumerate(
                    [cn, str(counts.get(f"{sp}_normal", 0)),
                     str(counts.get(f"{sp}_anomaly", 0))]):
                item = QTableWidgetItem(val)
                if j:
                    item.setTextAlignment(Qt.AlignCenter)
                    if j == 2 and int(val or 0) > 0:
                        item.setForeground(QColor("#B91C1C"))
                self.table.setItem(r, c0 + j, item)
        self.table.resizeColumnsToContents()

    # ── dir_rules/plain：分组确认表 ──
    def _build_groups_table(self, cats: list) -> None:
        groups: list = []
        for c in cats:
            groups = c.get("groups") or []
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels(
            ["分组（目录）", "正常", "缺陷", "模板", "未识别",
             "样例文件", "未识别图片处理"])
        self.table.setRowCount(len(groups))
        for r, g in enumerate(groups):
            name = str(g.get("name"))
            unknown = int(g.get("unknown", 0) or 0)
            vals = [name, str(g.get("normal", 0)), str(g.get("anomaly", 0)),
                    str(g.get("template", 0)), str(unknown),
                    "\n".join(str(s) for s in (g.get("samples") or [])[:2])]
            for col, val in enumerate(vals):
                item = QTableWidgetItem(val)
                if col in (1, 2, 3, 4):
                    item.setTextAlignment(Qt.AlignCenter)
                if col == 2 and int(val or 0) > 0:
                    item.setForeground(QColor("#B91C1C"))
                if col == 4 and unknown:
                    item.setForeground(QColor("#B45309"))
                self.table.setItem(r, col, item)
            combo = QComboBox()
            combo.addItem("跳过不导入（默认）", "skip")
            combo.addItem("按正常导入", "normal")
            combo.addItem("按缺陷导入", "anomaly")
            combo.setToolTip(
                "该分组内未能自动识别的图片按此方式导入；"
                "已识别（正常/缺陷/模板）的图片不受影响")
            if not unknown:
                combo.setEnabled(False)
            self._overrides[name] = combo
            self.table.setCellWidget(r, 6, combo)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.resizeColumnsToContents()
        self.table.setColumnWidth(5, 220)

    def config(self) -> dict:
        """确认后的导入配置。"""
        out: dict = {"format": self._fmt,
                     "ok_role": self.combo_okrole.currentData()}
        if hasattr(self, "edit_category"):
            out["category"] = self.edit_category.text().strip() or "default"
        if self._fmt == "mvtec_like":
            out["categories"] = [name for name, chk in self._cat_checks.items()
                                 if chk.isChecked()]
        else:
            out["group_overrides"] = {
                name: str(cmb.currentData() or "skip")
                for name, cmb in self._overrides.items() if cmb.isEnabled()}
        return out


class DataPage(QWidget):
    """数据管理页。"""

    def __init__(self, client: ApiClient, get_category,
                 task_monitor: TaskMonitor | None = None,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self._client = client
        self._get_category = get_category
        self._monitor = task_monitor
        self._images: list[dict] = []
        self._ds_names: dict[str, str] = {}   # dataset_id 字符串 → 批次名
        self._ds_layers: dict[str, list] = {}  # M14d：dataset_id → 可支撑层级
        self._datasources: list[dict] = []     # 数据源登记（导入挂源/新建源）
        self._cur_dataset: str = ""           # 当前批次过滤（"" 无 / "none" 未分组）

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(8)

        top = QHBoxLayout()
        top.addWidget(make_banner("数据底座：数据源管理（模态为筛选项，建源即导入）"))
        attach_level_badge(self, top)   # U-workorder：数据条件徽标
        top.addStretch(1)
        root.addLayout(top)

        # 前端反馈 v8：不再分图像/视频 tab，模态作为筛选项
        root.addWidget(self._build_image_tab(), 1)

        # 任务进度条（导入等长任务）
        self.progress = TaskProgressBar("后台任务")
        self.progress.set_client(self._client)   # A15：启用取消按钮
        root.addWidget(self.progress)
        if self._monitor is not None:
            self._monitor.task_updated.connect(self._on_task)

        self._client.error_occurred.connect(lambda m: warn(self, m))
        self.reload()

    # ══════════════════ 图像选项卡 ══════════════════
    def _build_image_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        # 筛选栏（v8：模态作为筛选项；v10：批次下拉按编号过滤，去划分/标签）
        bar = QHBoxLayout()
        bar.addWidget(QLabel("模态"))
        self.combo_modality = QComboBox()
        self.combo_modality.addItem("全部", "")
        self.combo_modality.addItem("图片", "image")
        self.combo_modality.addItem("视频", "video")
        self.combo_modality.currentIndexChanged.connect(
            lambda _i: self._on_modality_changed())
        bar.addWidget(self.combo_modality)
        # 数据源下拉（v10：模态→数据源→品类→批次，选源后品类/批次联动；v10 支持输入搜索）
        bar.addWidget(QLabel("数据源"))
        self.combo_source = QComboBox()
        self.combo_source.setEditable(True)
        self.combo_source.setInsertPolicy(QComboBox.NoInsert)
        self.combo_source.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.combo_source.setMinimumContentsLength(6)
        self.combo_source.addItem("全部", "")
        self.combo_source.currentIndexChanged.connect(
            lambda _i: self._on_source_changed())
        bar.addWidget(self.combo_source)
        self._enable_combo_search(self.combo_source, self._on_source_changed)
        bar.addWidget(QLabel("品类"))
        self.combo_category = QComboBox()
        self.combo_category.setEditable(True)
        self.combo_category.setInsertPolicy(QComboBox.NoInsert)
        self.combo_category.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.combo_category.setMinimumContentsLength(6)
        self.combo_category.addItem("全部", "")
        self.combo_category.currentIndexChanged.connect(
            lambda _i: self._on_category_changed())
        bar.addWidget(self.combo_category)
        self._enable_combo_search(self.combo_category, self._on_category_changed)
        # 批次下拉：只显示编号数字（与表格批次列一致，v10）
        bar.addWidget(QLabel("批次"))
        self.combo_batch = QComboBox()
        self.combo_batch.addItem("全部", "")
        self.combo_batch.setToolTip("检测组按批次浏览：选品类后可按批次编号过滤（数字）")
        self.combo_batch.currentIndexChanged.connect(
            lambda _i: self._on_batch_changed())
        bar.addWidget(self.combo_batch)
        bar.addStretch(1)

        btn_new_source = QPushButton("新建数据源")
        btn_new_source.setProperty("primary", True)
        btn_new_source.setToolTip("登记数据源（名称＋模态＋标注档位＋品类/模板声明＋数据流），"
                                  "一步式建源并导入数据挂到该源")
        btn_new_source.clicked.connect(self._on_create_datasource)
        bar.addWidget(btn_new_source)
        lay.addLayout(bar)

        # 主体：左侧数据树导航 + 右侧缩略图表格
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_tree_panel())

        right = QWidget()
        r_lay = QVBoxLayout(right)
        r_lay.setContentsMargins(0, 0, 0, 0)
        r_lay.setSpacing(4)

        # 右侧堆叠：模态=图片/全部 → 图片表格；模态=视频 → 视频表格（v10 无独立视频页）
        self.stack_data = QStackedWidget()

        # ── 页0：图片缩略图表格 ──
        img_page = QWidget()
        ip_lay = QVBoxLayout(img_page)
        ip_lay.setContentsMargins(0, 0, 0, 0)
        ip_lay.setSpacing(4)
        # 缩略图表格（序号列显示累计编号：第 1 页 1-50、第 2 页 51-100）
        self.table = QTableWidget(0, _N_COLS)
        self.table.setHorizontalHeaderLabels(
            ["序号", "缩略图", "ID", "品类", "批次", "来源", "划分", "标签",
             "缺陷类型", "导入时间", "路径"])
        self.table.setColumnWidth(0, 55)
        self.table.setColumnWidth(1, 110)
        self.table.setColumnWidth(2, 55)
        self.table.setColumnWidth(3, 120)   # 品类列固定宽：长品类名自动换行
        self.table.setWordWrap(True)   # 长文本（如品类名）显示不下自动换行
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setDefaultSectionSize(66)
        self.table.verticalHeader().setVisible(False)  # 行号并入「序号」列（累计编号），避免与自带行号重复
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._on_table_menu)
        self.table.itemDoubleClicked.connect(self._on_preview_image)
        ip_lay.addWidget(self.table, 1)

        # 分页栏（第一页/上一页/下一页/最后一页）：一次只拉一页缩略图，
        # 不再整批 500 张并发下载
        self._page = 1
        self._page_size = 50
        self._cur_ids: list | None = None   # 前端反馈 v9：按图片 id 列表过滤（预训练组/检测组/批次）
        self._cur_group_label: str = ""     # 当前逻辑分组/批次标签（树节点名，表格"批次"列优先显示）
        self._batch_ids_map: dict[str, list[int]] = {}   # 批次编号 → 图片 ids（查询栏批次下拉，v10）
        self._total = 0
        self._total_pages = 1
        self._img_req = 0   # 图片查询请求序号：翻页竞态时丢弃过期响应
        pg = QHBoxLayout()
        self.lbl_img_count = QLabel("共 0 条")
        self.lbl_img_count.setProperty("subtext", True)
        pg.addWidget(self.lbl_img_count)
        pg.addStretch(1)
        self.btn_img_first = QPushButton("第一页")
        self.btn_img_first.setProperty("flat", True)
        self.btn_img_first.clicked.connect(lambda: self._page_goto(1))
        self.btn_img_prev = QPushButton("上一页")
        self.btn_img_prev.setProperty("flat", True)
        self.btn_img_prev.clicked.connect(lambda: self._page_goto(self._page - 1))
        self.lbl_img_page = QLabel("第 1 / 1 页")
        self.lbl_img_page.setProperty("subtext", True)
        self.btn_img_next = QPushButton("下一页")
        self.btn_img_next.setProperty("flat", True)
        self.btn_img_next.clicked.connect(lambda: self._page_goto(self._page + 1))
        self.btn_img_last = QPushButton("最后一页")
        self.btn_img_last.setProperty("flat", True)
        self.btn_img_last.clicked.connect(lambda: self._page_goto(self._total_pages))
        pg.addWidget(self.btn_img_first)
        pg.addWidget(self.btn_img_prev)
        pg.addWidget(self.lbl_img_page)
        pg.addWidget(self.btn_img_next)
        pg.addWidget(self.btn_img_last)
        ip_lay.addLayout(pg)
        self.stack_data.addWidget(img_page)

        # ── 页1：视频表格（v10 并入图像页，模态=视频时展示）──
        self.stack_data.addWidget(self._build_video_panel())

        r_lay.addWidget(self.stack_data, 1)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([260, 800])
        lay.addWidget(splitter, 1)
        # 前端反馈 v5：数据流队列（当前工单待检批次，可回队/取消）
        lay.addWidget(self._build_queue_panel())
        return w

    # ══════════════════ 数据流队列（前端反馈 v5）══════════════════
    def _build_queue_panel(self) -> QWidget:
        box = QGroupBox("数据流队列（当前工单）— 暂停产线后可编排；错检批次可回队重新检测")
        v = QVBoxLayout(box)
        self.queue_table = QTableWidget(0, 6)
        self.queue_table.setHorizontalHeaderLabels(
            ["批次", "数据源", "品类", "图片/检测/不良/反馈", "错检批次", "回队"])
        self.queue_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.queue_table.verticalHeader().setVisible(False)
        self.queue_table.horizontalHeader().setStretchLastSection(True)
        v.addWidget(self.queue_table, 1)
        row = QHBoxLayout()
        self.btn_queue_reload = QPushButton("刷新队列")
        self.btn_queue_reload.clicked.connect(self._load_queue)
        self.btn_queue_req = QPushButton("回队（重新检测）")
        self.btn_queue_req.clicked.connect(lambda: self._queue_requeue(True))
        self.btn_queue_unreq = QPushButton("取消回队")
        self.btn_queue_unreq.clicked.connect(lambda: self._queue_requeue(False))
        row.addWidget(self.btn_queue_reload)
        row.addWidget(self.btn_queue_req)
        row.addWidget(self.btn_queue_unreq)
        row.addStretch(1)
        v.addLayout(row)
        return box

    def _workorder_id(self) -> int | None:
        fn = getattr(self, "_get_workorder_id", None)
        return fn() if callable(fn) else None

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
            cell = QTableWidgetItem("（当前工单未挂数据源或尚无批次；"
                                    "把数据源挂到工单后此处显示待检队列）")
            cell.setFlags(Qt.ItemIsEnabled)
            self.queue_table.setItem(0, 0, cell)
            self.queue_table.setSpan(0, 0, 1, 6)
            return
        for r, it in enumerate(items):
            cells = [
                str(it.get("name")), str(it.get("source")),
                str(it.get("category")),
                f"{it.get('n_images')}/{it.get('n_detected')}/"
                f"{it.get('n_anomaly')}/{it.get('n_feedback')}",
                "⚠ 错检高发" if it.get("bad_batch") else "—",
                "已回队" if it.get("requeued") else "—",
            ]
            for c, txt in enumerate(cells):
                cell = QTableWidgetItem(txt)
                if it.get("bad_batch"):
                    cell.setForeground(QColor("#E67E22"))
                self.queue_table.setItem(r, c, cell)
                if c == 0:
                    cell.setData(Qt.UserRole, it.get("batch_key"))
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
                  lambda _r: self._load_queue(), lambda m: warn(self, str(m)))

    # ── 数据树导航（品类 → 数据集批次 → 划分 → 标签）──
    def _build_tree_panel(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 8, 0)
        lay.setSpacing(4)
        head = QLabel("数据导航（数据源→品类→批次→划分→标签）")
        head.setProperty("heading", True)
        lay.addWidget(head)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setMinimumWidth(230)
        self.tree.itemSelectionChanged.connect(self._on_tree_selected)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._on_tree_menu)
        lay.addWidget(self.tree, 1)
        return w

    def reload_tree(self) -> None:
        """前端反馈 v9：树 = 数据源 → 品类 → 预训练组/检测组（检测组按批展示）。"""
        run_async(self, self._client.list_datasets, self._on_ds_layers)

    def _on_modality_changed(self) -> None:
        """模态变化：视频 → 视频表格页；图片/全部 → 图片表格页（v10 无独立视频页）。"""
        # 数据源下拉按模态过滤（v10 联动）
        self._refresh_source_options()
        src_id = self.combo_source.currentData() or ""
        if self.combo_modality.currentData() == "video":
            self.stack_data.setCurrentIndex(1)
            self.reload_videos()
            self.reload_tree()
        else:
            self.stack_data.setCurrentIndex(0)
            self._refresh_category_options(src_id)
            cat = self.combo_category.currentData() or ""
            self._refresh_batch_options(cat)
            self.reload_tree()
            self._on_query()

    def _on_category_changed(self) -> None:
        """品类变化：刷新批次下拉 + 刷新表格（v10 业务联动，下拉即查）。"""
        if self.combo_category.currentIndex() < 0:
            return   # 输入搜索中未选中项，不联动
        cat = self.combo_category.currentData() or ""
        self._refresh_batch_options(cat)
        if self.combo_modality.currentData() == "video":
            self.reload_videos()
        else:
            self._on_query()

    def _on_batch_changed(self) -> None:
        """批次变化：刷新表格（v10 业务联动，批次列/下拉同看编号数字）。"""
        self._on_query()

    def _enable_combo_search(self, combo, on_pick) -> None:
        """下拉支持输入搜索：QCompleter contains 过滤 + 回车定位选中（v10）。

        输入过程中 currentIndex 变 -1 不触发联动（由 _on_*_changed 守卫）；
        点选下拉项由 currentIndexChanged 联动；回车定位到匹配项后再触发。
        """
        comp = QCompleter(combo.model(), combo)
        comp.setFilterMode(Qt.MatchContains)
        comp.setCaseSensitivity(Qt.CaseInsensitive)
        comp.setCompletionMode(QCompleter.PopupCompletion)
        combo.setCompleter(comp)
        try:
            combo.lineEdit().returnPressed.connect(
                lambda: self._pick_combo_text(combo, on_pick))
        except Exception:  # noqa: BLE001 非 editable 时不处理
            pass

    @staticmethod
    def _pick_combo_text(combo, on_pick) -> None:
        """回车定位：按输入文本找匹配项并选中（触发联动）；找不到保持现状。"""
        txt = combo.currentText().strip()
        if not txt:
            return
        idx = combo.findText(txt, Qt.MatchExactly)
        if idx < 0:
            idx = combo.findText(txt, Qt.MatchContains)
        if idx >= 0 and idx != combo.currentIndex():
            combo.setCurrentIndex(idx)   # 触发 currentIndexChanged → on_pick
        elif idx == combo.currentIndex():
            on_pick()

    def _on_source_changed(self) -> None:
        """数据源变化：品类下拉刷新为该源品类 → 批次刷新 → 表格刷新（v10 联动）。"""
        if self.combo_source.currentIndex() < 0:
            return   # 输入搜索中未选中项，不联动
        src_id = self.combo_source.currentData() or ""
        self._refresh_category_options(src_id)
        cat = self.combo_category.currentData() or ""
        self._refresh_batch_options(cat)
        if self.combo_modality.currentData() == "video":
            self.reload_videos()
        else:
            self._on_query()

    def _refresh_category_options(self, src_id: str) -> None:
        """按选中数据源刷新「品类」下拉：选源 → 该源品类；全部 → 全局品类（v10）。"""
        if not hasattr(self, "combo_category"):
            return
        self.combo_category.blockSignals(True)
        self.combo_category.clear()
        self.combo_category.addItem("全部", "")
        if not src_id:
            for c in getattr(self, "_global_categories", []):
                self.combo_category.addItem(c, c)
            self.combo_category.blockSignals(False)
            return
        src = next((d for d in self._datasources
                    if str(d.get("id")) == str(src_id)), None)
        if src is None:
            self.combo_category.blockSignals(False)
            return
        g = self._client.datasource_groups(int(src_id)) or {}
        for cat in sorted((g.get("categories") or {}), key=str):
            self.combo_category.addItem(cat, cat)
        self.combo_category.blockSignals(False)

    def _refresh_batch_options(self, cat: str) -> None:
        """按品类刷新查询栏「批次」下拉：聚合匹配模态+数据源 该品类检测组批次编号。

        v10：批次编号与表格批次列一致（纯数字），选中后按该批 ids 过滤。
        品类为空或模态=视频时下拉只有「全部」。
        """
        self.combo_batch.blockSignals(True)
        self.combo_batch.clear()
        self.combo_batch.addItem("全部", "")
        self._batch_ids_map = {}
        if not cat or self.combo_modality.currentData() == "video":
            self.combo_batch.blockSignals(False)
            return

        def _load():
            mod = self.combo_modality.currentData() or ""
            src_id = self.combo_source.currentData() or ""
            merged: dict[int, list[int]] = {}
            for src in self._datasources:
                if mod and src.get("modality") != mod:
                    continue   # 模态联动：只聚合匹配模态数据源的批次
                if src_id and str(src.get("id")) != str(src_id):
                    continue   # 数据源联动：只聚合选中数据源的批次
                g = self._client.datasource_groups(src.get("id")) or {}
                cg = (g.get("categories") or {}).get(cat)
                if not cg:
                    continue
                for bi, ids in enumerate(cg.get("detect_batches") or [], 1):
                    merged.setdefault(bi, []).extend(ids or [])
            return {k: list(dict.fromkeys(v)) for k, v in merged.items()}

        def _fill(m) -> None:
            if self.combo_batch is None:
                return
            self.combo_batch.blockSignals(True)
            self.combo_batch.clear()
            self.combo_batch.addItem("全部", "")
            self._batch_ids_map = {str(k): v for k, v in (m or {}).items()}
            for k in sorted(self._batch_ids_map, key=int):
                self.combo_batch.addItem(k, k)
            self.combo_batch.setCurrentIndex(0)
            self.combo_batch.blockSignals(False)

        def _fail(err) -> None:
            # 失败也要恢复信号，避免后续批次下拉信号被屏蔽
            self.combo_batch.blockSignals(False)
            warn(self, f"加载批次失败：{err}")

        run_async(self, _load, _fill, _fail)

    def _on_ds_layers(self, data) -> None:
        """缓存批次→可支撑层级映射后建源树。"""
        self._ds_layers = {}
        items = []
        if isinstance(data, dict):
            items = data.get("items", []) or []
            for it in items:
                self._ds_layers[str(it.get("id"))] = it.get("layers") or []
        self._build_source_tree()

    def _build_source_tree(self) -> None:
        """源树骨架：全部图像 → 各数据源节点（品类 → 预训练组/检测组）。"""
        self.tree.blockSignals(True)
        self.tree.clear()
        self._source_mode = True
        root_item = QTreeWidgetItem(["全部图像"])
        root_item.setData(0, Qt.UserRole, ("", "", "", ""))
        self.tree.addTopLevelItem(root_item)
        for src in self._datasources:
            name = str(src.get("name") or "")
            g = QTreeWidgetItem([name])
            g.setData(0, Qt.UserRole, ("", "", "", ""))
            g.setData(0, Qt.UserRole + 1, name)
            root_item.addChild(g)
            self._load_source_groups(src, g)
        root_item.setExpanded(True)
        self.tree.blockSignals(False)

    def _load_source_groups(self, src: dict, node) -> None:
        """异步加载单数据源分组：品类 → 预训练组 / 检测组（批次数）。"""
        def _fill(data) -> None:
            # 竞态防护：树重建后旧节点已被 GC 销毁，用 shiboken 安全判断
            try:
                if node is None or node.treeWidget() is None:
                    return   # 节点已被重建/移除
            except RuntimeError:
                return   # Internal C++ object already deleted
            data = data or {}
            cats = data.get("categories") or {}
            if not cats:
                ph = QTreeWidgetItem(["（暂无数据，右键可编辑/删除数据源）"])
                ph.setFlags(Qt.NoItemFlags)
                node.addChild(ph)
                return
            total = 0
            for cat in sorted(cats, key=str):
                g = cats[cat]
                cat_item = QTreeWidgetItem([str(cat)])
                cat_item.setData(0, Qt.UserRole, (str(cat), "", "", ""))
                node.addChild(cat_item)
                pc = g.get("pretrain_counts") or {}
                pre_ids = g.get("pretrain") or []
                pre = QTreeWidgetItem(
                    [f"预训练组（{pc.get('normal', 0)} 正常 + "
                     f"{pc.get('anomaly', 0)} 异常）"])
                pre.setData(0, Qt.UserRole, (str(cat), "", "", ""))
                pre.setData(0, Qt.UserRole + 3,
                            ("pretrain", str(cat), list(pre_ids)))
                cat_item.addChild(pre)
                batches = g.get("detect_batches") or []
                det = QTreeWidgetItem(
                    [f"检测组（{g.get('detect_count', 0)} 图 · "
                     f"{len(batches)} 批）"])
                det.setData(0, Qt.UserRole, (str(cat), "", "", ""))
                det.setData(0, Qt.UserRole + 3,
                            ("detect", str(cat),
                             [i for b in batches for i in b]))
                cat_item.addChild(det)
                # 具体检测批次子节点：可选中查看该批图片（前端反馈 v9）
                for bi, b in enumerate(batches, 1):
                    b_item = QTreeWidgetItem([f"批次 {bi}（{len(b)} 图）"])
                    b_item.setData(0, Qt.UserRole, (str(cat), "", "", ""))
                    b_item.setData(0, Qt.UserRole + 3,
                                   ("batch", str(cat), list(b)))
                    det.addChild(b_item)
                total += int(g.get("total", 0) or 0)
            node.setText(0, f"{str(src.get('name'))}（{total} 图）")
        run_async(self, lambda: self._client.datasource_groups(src.get("id")),
                  _fill)

    def _tree_filter(self) -> tuple[str, str, str, str]:
        """当前树节点对应的 (品类, 批次id, 划分, 标签) 过滤条件。"""
        item = self.tree.currentItem()
        data = item.data(0, Qt.UserRole) if item is not None else None
        if isinstance(data, (tuple, list)) and len(data) == 4:
            return tuple(str(v) for v in data)
        return ("", "", "", "")

    def _fill_tree(self, data) -> None:
        # 数据树：数据源顶层（右键可管理）；与工单对话框对齐：
        # 对话框列所有数据源，树也展示所有数据源（v10 无"未挂源"概念）
        sgroups = data.get("source_groups") if isinstance(data, dict) else None
        self._source_mode = False
        if isinstance(sgroups, dict) and sgroups:
            # 前端反馈 v8：模态作为筛选项（按数据源模态过滤）；
            # _datasources 未加载（异步）时跳过筛选，避免把所有源误删
            mod = str(self.combo_modality.currentData() or "")
            if mod and self._datasources:
                src_mod = {str(d.get("name")): str(d.get("modality"))
                           for d in self._datasources}
                sgroups = {k: v for k, v in sgroups.items()
                           if src_mod.get(k) == mod}
            if sgroups:
                self._source_mode = True
                self._fill_tree_groups(sgroups, is_source=True)
                return
        datasets = data.get("datasets") if isinstance(data, dict) else None
        if isinstance(datasets, dict) and datasets:
            self._fill_tree_datasets(datasets)
        else:
            # 老后端兜底：三维 tree（品类→划分→标签），批次维置空
            tree = data.get("tree") if isinstance(data, dict) else None
            self._fill_tree_legacy(tree if isinstance(tree, dict) else {})

    def _fill_tree_groups(self, groups: dict, is_source: bool = False) -> None:
        """三层数据树：数据集/数据源(顶层) → 品类 → 批次 → 划分/标签。

        数据集节点对应 4 元组过滤条件为空（点击展开下钻品类），
        品类/批次节点沿用 (cat, ds_key, split, label) 四元组。
        is_source=True 时顶层为数据源节点（右键可编辑/删除数据源）。
        """
        cur = self._tree_filter()
        self.tree.blockSignals(True)
        self.tree.clear()
        self._ds_names = {}
        src_names = {str(d.get("name")) for d in self._datasources}
        total = 0
        for gname in sorted(groups, key=str):
            cats_map = groups.get(gname)
            if not isinstance(cats_map, dict):
                continue
            g_item = QTreeWidgetItem([str(gname)])
            g_item.setData(0, Qt.UserRole, ("", "", "", ""))
            # 仅真正的数据源顶层节点可右键编辑/删除
            if is_source and str(gname) in src_names:
                g_item.setData(0, Qt.UserRole + 1, str(gname))
            if not cats_map:
                # 无数据源（如残留空壳源）：占位子节点，仍可右键管理
                ph = QTreeWidgetItem(["（暂无数据，右键可编辑/删除数据源）"])
                ph.setFlags(Qt.NoItemFlags)
                g_item.addChild(ph)
                self.tree.addTopLevelItem(g_item)
                continue
            g_total = 0
            for cat in sorted(cats_map, key=str):
                ds_map = cats_map.get(cat)
                if not isinstance(ds_map, dict):
                    continue
                cat_item = QTreeWidgetItem([str(cat)])
                cat_item.setData(0, Qt.UserRole, (str(cat), "", "", ""))
                cat_count = 0
                # 批次排序：数字 id 倒序（新批次在前），"none" 垫底
                keys = sorted(ds_map, key=lambda k: (
                    k == "none", -int(k) if str(k).isdigit() else 0, str(k)))
                for ds_key in keys:
                    node = ds_map.get(ds_key)
                    if not isinstance(node, dict):
                        continue
                    name = str(node.get("name") or "未分组")
                    splits = node.get("splits")
                    if not isinstance(splits, dict):
                        continue
                    self._ds_names[str(ds_key)] = name
                    ds_count = sum(int(labels.get(lb, 0) or 0)
                                   for labels in splits.values()
                                   if isinstance(labels, dict)
                                   for lb in labels)
                    cat_count += ds_count
                    layers = self._ds_layers.get(str(ds_key), [])
                    # U-workorder：批次可支撑条件中文化（不露 L 代号）
                    badge = (f"｜可支撑：{'/'.join(LAYER_CN.get(x, x) for x in layers)}"
                             if layers else "")
                    ds_item = QTreeWidgetItem([f"{name} ({ds_count}){badge}"])
                    if layers:
                        ds_item.setToolTip(0, "这份数据可以支撑的建模方式：\n"
                                           "· 只有正常图 → 零样本\n"
                                           "· 有正常+缺陷标签 → 少样本\n"
                                           "· 有缺陷框标注 → 框级反馈\n"
                                           "· 有模板图 → 模板比对")
                    ds_item.setData(0, Qt.UserRole,
                                    (str(cat), str(ds_key), "", ""))
                    cat_item.addChild(ds_item)
                    self._fill_split_nodes(ds_item, splits,
                                           (str(cat), str(ds_key)))
                cat_item.setText(0, f"{cat} ({cat_count})")
                g_total += cat_count
                g_item.addChild(cat_item)
            g_item.setText(0, f"{gname} ({g_total})")
            total += g_total
            self.tree.addTopLevelItem(g_item)
        root_item = QTreeWidgetItem([f"全部图像 ({total})"])
        root_item.setData(0, Qt.UserRole, ("", "", "", ""))
        self.tree.insertTopLevelItem(0, root_item)
        root_item.setExpanded(True)
        target = self._find_tree_item(cur) or root_item
        self.tree.setCurrentItem(target)
        self.tree.blockSignals(False)

    def _fill_tree_datasets(self, datasets: dict) -> None:
        cur = self._tree_filter()  # 刷新后尽量恢复原选中节点
        self.tree.blockSignals(True)
        self.tree.clear()
        self._ds_names = {}
        total = 0
        for cat in sorted(datasets, key=str):
            ds_map = datasets.get(cat)
            if not isinstance(ds_map, dict):
                continue
            cat_item = QTreeWidgetItem([str(cat)])
            cat_item.setData(0, Qt.UserRole, (str(cat), "", "", ""))
            cat_count = 0
            # 批次排序：数字 id 倒序（新批次在前），"none" 垫底
            keys = sorted(ds_map, key=lambda k: (
                k == "none", -int(k) if str(k).isdigit() else 0, str(k)))
            for ds_key in keys:
                node = ds_map.get(ds_key)
                if not isinstance(node, dict):
                    continue
                name = str(node.get("name") or "未分组")
                splits = node.get("splits")
                if not isinstance(splits, dict):
                    continue
                self._ds_names[str(ds_key)] = name
                ds_count = sum(int(labels.get(lb, 0) or 0)
                               for labels in splits.values()
                               if isinstance(labels, dict)
                               for lb in labels)
                cat_count += ds_count
                layers = self._ds_layers.get(str(ds_key), [])
                # U-workorder：批次可支撑条件中文化（不露 L 代号）
                badge = (f"｜可支撑：{'/'.join(LAYER_CN.get(x, x) for x in layers)}"
                         if layers else "")
                ds_item = QTreeWidgetItem([f"{name} ({ds_count}){badge}"])
                if layers:
                    ds_item.setToolTip(0, "这份数据可以支撑的建模方式：\n"
                                       "· 只有正常图 → 零样本\n"
                                       "· 有正常+缺陷标签 → 少样本\n"
                                       "· 有缺陷框标注 → 框级反馈\n"
                                       "· 有模板图 → 模板比对")
                ds_item.setData(0, Qt.UserRole,
                                (str(cat), str(ds_key), "", ""))
                cat_item.addChild(ds_item)
                self._fill_split_nodes(ds_item, splits,
                                       (str(cat), str(ds_key)))
            cat_item.setText(0, f"{cat} ({cat_count})")
            total += cat_count
            self.tree.addTopLevelItem(cat_item)
        root_item = QTreeWidgetItem([f"全部图像 ({total})"])
        root_item.setData(0, Qt.UserRole, ("", "", "", ""))
        self.tree.insertTopLevelItem(0, root_item)
        root_item.setExpanded(True)
        # 恢复选中（找不到同名节点则回到根节点）
        target = self._find_tree_item(cur) or root_item
        self.tree.setCurrentItem(target)
        self.tree.blockSignals(False)

    def _fill_tree_legacy(self, tree: dict) -> None:
        cur = self._tree_filter()
        self.tree.blockSignals(True)
        self.tree.clear()
        self._ds_names = {}
        total = 0
        for cat in sorted(tree, key=str):
            splits = tree.get(cat)
            if not isinstance(splits, dict):
                continue
            cat_count = sum(int(labels.get(lb, 0) or 0)
                            for labels in splits.values()
                            if isinstance(labels, dict)
                            for lb in labels)
            total += cat_count
            cat_item = QTreeWidgetItem([f"{cat} ({cat_count})"])
            cat_item.setData(0, Qt.UserRole, (str(cat), "", "", ""))
            self.tree.addTopLevelItem(cat_item)
            self._fill_split_nodes(cat_item, splits, (str(cat), ""))
        root_item = QTreeWidgetItem([f"全部图像 ({total})"])
        root_item.setData(0, Qt.UserRole, ("", "", "", ""))
        self.tree.insertTopLevelItem(0, root_item)
        root_item.setExpanded(True)
        target = self._find_tree_item(cur) or root_item
        self.tree.setCurrentItem(target)
        self.tree.blockSignals(False)

    def _fill_split_nodes(self, parent_item, splits: dict,
                          prefix: tuple[str, str]) -> None:
        """在父节点下挂 划分→标签 两级子节点，data 为四元组。"""
        cat, ds_key = prefix
        for split in sorted(splits, key=str):
            labels = splits.get(split)
            if not isinstance(labels, dict):
                continue
            sp_count = sum(int(v or 0) for v in labels.values())
            sp_item = QTreeWidgetItem([f"{split} ({sp_count})"])
            sp_item.setData(0, Qt.UserRole, (cat, ds_key, str(split), ""))
            parent_item.addChild(sp_item)
            for label in sorted(labels, key=str):
                lb_item = QTreeWidgetItem(
                    [f"{label} ({int(labels[label] or 0)})"])
                lb_item.setData(0, Qt.UserRole,
                                (cat, ds_key, str(split), str(label)))
                sp_item.addChild(lb_item)

    def _find_tree_item(self, flt: tuple[str, str, str, str]):
        """按过滤条件在树中查找节点。"""
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            stack = [item] + [item.child(j) for j in range(item.childCount())]
            while stack:
                node = stack.pop()
                if node.data(0, Qt.UserRole) == flt:
                    return node
                stack.extend(node.child(j) for j in range(node.childCount()))
        return None

    @staticmethod
    def _set_combo_data(combo: QComboBox, value: str) -> None:
        """把下拉框设为指定 data 值；选项不存在时动态补充。"""
        combo.blockSignals(True)
        idx = combo.findData(value)
        if idx < 0 and value:
            combo.addItem(value, value)
            idx = combo.findData(value)
        combo.setCurrentIndex(idx if idx >= 0 else 0)
        combo.blockSignals(False)

    def _on_tree_selected(self) -> None:
        """点击树节点：同步顶部筛选下拉并按节点过滤右侧表格。"""
        item = self.tree.currentItem()
        cat, ds_id, split, label = self._tree_filter()
        gk = item.data(0, Qt.UserRole + 3) if item is not None else None
        if isinstance(gk, (tuple, list)) and len(gk) == 3 and \
                gk[0] in ("pretrain", "detect", "batch"):
            # 预训练组 / 检测组 / 检测批次：按图片 id 列表过滤（前端反馈 v9）
            self._cur_ids = [int(i) for i in (gk[2] or [])]
            self._cur_dataset = ""
            # 批次列只显示数字：批次节点取编号，预训练/检测组显示组名
            if gk[0] == "batch":
                m = re.search(r"\d+", item.text(0))
                self._cur_group_label = m.group(0) if m else item.text(0)
            else:
                self._cur_group_label = "预训练组" if gk[0] == "pretrain" else "检测组"
            self._set_combo_data(self.combo_category, str(gk[1]))
            self._set_combo_data(self.combo_batch, "")
            self._page = 1
        else:
            self._cur_ids = None
            self._cur_group_label = ""
            self._cur_dataset = ds_id
            self._set_combo_data(self.combo_category, cat)
            self._set_combo_data(self.combo_batch, "")
            # 树点击品类：_set_combo_data 屏蔽了信号，需手动刷新批次下拉（v10 联动）
            self._refresh_batch_options(cat)
        # 模态=视频时树点击仍展示视频页
        if self.combo_modality.currentData() == "video":
            self.stack_data.setCurrentIndex(1)
            self.reload_videos()
        else:
            self.reload_images()

    def _on_query(self) -> None:
        """查询按钮：只按筛选栏条件查询（清除树节点的批次过滤）。"""
        self._cur_dataset = ""
        self._cur_ids = None
        # 批次列标签与查询栏批次下拉对齐（选中编号则显示编号，否则显示导入批次 id）
        self._cur_group_label = self.combo_batch.currentData() or ""
        self._page = 1
        self.reload_images()

    def reload_images(self) -> None:
        cat = self.combo_category.currentData() or ""
        ds = self._cur_dataset
        src_id = self.combo_source.currentData() or ""
        page = max(1, self._page)
        size = self._page_size
        self._img_req += 1
        rid = self._img_req
        # 查询栏批次下拉：编号 → ids（聚合跨数据源该品类检测组批次，v10）
        batch_no = self.combo_batch.currentData() or ""
        ids = self._cur_ids
        if not ids and batch_no:
            ids = self._batch_ids_map.get(batch_no)
            if ids is None:
                # 选中了不存在的批次编号（如越界）：直接空结果，提示无匹配
                self._fill_images({"total": 0, "items": []}, rid)
                return
        if ids:
            # 预训练组/检测组/检测批次/查询栏批次：按图片 id 列表过滤
            ids_str = ",".join(str(i) for i in ids)
            run_async(self, lambda: self._client.list_images(
                ids=ids_str, page=page, page_size=size),
                lambda data, _rid=rid: self._fill_images(data, _rid))
        elif ds and ds != "none":
            # 批次过滤：后端按 dataset_id 过滤并返回准确 total（分页可用）
            run_async(self, lambda: self._client.list_images(
                category=cat, dataset_id=int(ds), page=page, page_size=size,
                datasource_id=src_id),
                lambda data, _rid=rid: self._fill_images(data, _rid))
        elif ds == "none":
            # 未分组：取品类图片后筛 dataset_id 为空的（未分组数据极少，单页）
            run_async(
                self,
                lambda: self._client.list_images(
                    category=cat, page=1, page_size=500,
                    datasource_id=src_id),
                lambda data, _rid=rid: self._fill_images({
                    "total": len([i for i in (data or {}).get("items", [])
                                  if i.get("dataset_id") is None]),
                    "items": [i for i in (data or {}).get("items", [])
                              if i.get("dataset_id") is None]}, _rid))
        else:
            run_async(self, lambda: self._client.list_images(
                category=cat, page=page, page_size=size,
                datasource_id=src_id),
                lambda data, _rid=rid: self._fill_images(data, _rid))

    def _page_goto(self, page: int) -> None:
        """分页跳转：clamp 到 [1, total_pages] 后重新查询。"""
        page = max(1, min(page, self._total_pages))
        if page == self._page:
            return
        self._page = page
        self.reload_images()

    def _dataset_id(self, img: dict) -> str:
        """批次列数字：导入批次 id；无批次显示 -。"""
        dsid = img.get("dataset_id")
        return str(dsid) if dsid is not None else "-"

    def _clear_thumb_cells(self) -> None:
        """移除并销毁缩略图列的全部单元格控件。
        Qt 的 setRowCount/clearSpans 不会删除 setCellWidget 放进去的控件
        ——它们仍是 viewport 的子控件，会以"幽灵缩略图"浮在空态文本上
        （2026-10-09 用户反馈：清空后仍残留一张缩略图）。"""
        for r in range(self.table.rowCount()):
            w = self.table.cellWidget(r, 1)
            if w is not None:
                self.table.removeCellWidget(r, 1)
                w.deleteLater()

    def _fill_images(self, data, rid: int | None = None) -> None:
        if rid is not None and rid != self._img_req:
            return   # 过期响应（快速翻页时旧页数据晚到），丢弃
        if isinstance(data, dict):
            items = data.get("items") or []
            total = int(data.get("total") or 0)
        else:
            items = data or []
            total = len(items)
        self._total = total
        self._images = items
        self._total_pages = max(1, (total + self._page_size - 1) // self._page_size)
        if self._page > self._total_pages:   # 数据减少后越界回退
            self._page = self._total_pages
        self.lbl_img_count.setText(f"共 {total} 条")
        self.lbl_img_page.setText(f"第 {self._page} / {self._total_pages} 页")
        for b, cond in ((self.btn_img_prev, self._page > 1),
                        (self.btn_img_next, self._page < self._total_pages),
                        (self.btn_img_first, self._page > 1),
                        (self.btn_img_last, self._page < self._total_pages)):
            b.setEnabled(cond)
        self._clear_thumb_cells()
        self.table.clearSpans()
        if not items:
            self.table.setRowCount(1)
            self.table.setSpan(0, 0, 1, _N_COLS)
            # 区分「库为空」与「有筛选条件但无匹配结果」
            cat = self.combo_category.currentData() or ""
            batch = self.combo_batch.currentData() or ""
            if cat or batch or self._cur_ids or self._cur_group_label:
                msg = "无匹配结果：请调整品类/批次筛选条件"
            else:
                msg = "暂无图像数据，可点击右上角按钮导入"
            self.table.setItem(0, 0, QTableWidgetItem(msg))
            return
        self.table.setRowCount(len(items))
        base = (self._page - 1) * self._page_size   # 累计序号偏移
        for r, img in enumerate(items):
            self.table.setItem(
                r, 0, QTableWidgetItem(str(base + r + 1)))
            cell = make_thumb_cell()
            self.table.setCellWidget(r, 1, cell)
            load_thumb_async(self, self._client,
                             self._client.file_url(img.get("path", "")), cell)
            self.table.setItem(r, 2, QTableWidgetItem(str(img.get("id", ""))))
            self.table.setItem(r, 3, QTableWidgetItem(str(img.get("category", ""))))
            # 批次列：逻辑组/批次显示组名或批次号；普通浏览显示导入批次 id（tooltip 保留全名）
            batch_name = (self._cur_group_label
                          if self._cur_group_label else self._dataset_id(img))
            b_item = QTableWidgetItem(batch_name)
            dsid = img.get("dataset_id")
            if dsid is not None:
                b_item.setToolTip(
                    self._ds_names.get(str(dsid), f"批次{dsid}"))
            self.table.setItem(r, 4, b_item)
            source = str(img.get("source") or "")
            self.table.setItem(r, 5, QTableWidgetItem(
                SOURCE_CN.get(source, source or "-")))
            self.table.setItem(r, 6, QTableWidgetItem(str(img.get("split", ""))))
            label_item = QTableWidgetItem(str(img.get("label", "")))
            if img.get("label") == "anomaly":
                label_item.setForeground(Qt.red)
            self.table.setItem(r, 7, label_item)
            self.table.setItem(r, 8, QTableWidgetItem(str(img.get("defect_type") or "-")))
            created = str(img.get("created_at") or "")[:19].replace("T", " ")
            self.table.setItem(r, 9, QTableWidgetItem(created))
            self.table.setItem(r, 10, QTableWidgetItem(str(img.get("path", ""))))
        # 换行后行高自适应内容（缩略图列最小高度由 minimumSectionSize 保证）
        self.table.verticalHeader().setMinimumSectionSize(60)
        self.table.resizeRowsToContents()

    def _on_preview_image(self, _item) -> None:
        row = self.table.currentRow()
        if not (0 <= row < len(self._images)):
            return
        img = self._images[row]
        dlg = ImagePreviewDialog(f"图片 #{img.get('id')}", self)
        dlg.show_url(self._client.file_url(img.get("path", "")))
        dlg.exec()

    def _on_table_menu(self, pos) -> None:
        row = self.table.currentRow()
        if not (0 <= row < len(self._images)):
            return
        menu = QMenu(self)
        act_annotate = menu.addAction("标注…")
        # M14a：L3 模板图快捷标记（split=template，prepare L3 时自动收集）
        img = self._images[row]
        if img.get("split") == "template":
            act_tpl = menu.addAction("取消模板图标记")
        else:
            act_tpl = menu.addAction("设为模板图（L3 用）")
        act_del = menu.addAction("删除该图像…")
        act_view = menu.addAction("查看大图")
        chosen = menu.exec(self.table.viewport().mapToGlobal(pos))
        if chosen is act_annotate:
            self._annotate_image(self._images[row])
        elif chosen is act_tpl:
            new_split = "train" if img.get("split") == "template" else "template"
            def _done_tpl(_res) -> None:
                self.reload_images()
                self.reload_tree()
            run_async(self, lambda: self._client.annotate_image(
                img.get("id"), split=new_split), _done_tpl)
        elif chosen is act_del:
            self._delete_image(self._images[row])
        elif chosen is act_view:
            self._on_preview_image(None)

    # ── 标注流转 ──
    def _build_annotate_dialog(self, img: dict) -> tuple[QDialog, QComboBox,
                                                        QComboBox, QLineEdit]:
        dlg = QDialog(self)
        dlg.setWindowTitle(f"标注图像 #{img.get('id')}")
        f = QFormLayout(dlg)
        combo_label = QComboBox()
        combo_label.addItem("保持不变", None)
        for lb in ("normal", "anomaly", "unknown"):
            combo_label.addItem(lb, lb)
        if img.get("label"):
            idx = combo_label.findData(img.get("label"))
            if idx >= 0:
                combo_label.setCurrentIndex(idx)
        f.addRow("标签", combo_label)
        combo_split = QComboBox()
        combo_split.addItem("保持不变", None)
        for sp in ("train", "val", "test", "feedback", "template", "unlabeled"):
            combo_split.addItem(sp, sp)
        if img.get("split"):
            idx = combo_split.findData(img.get("split"))
            if idx < 0:
                combo_split.addItem(str(img.get("split")), img.get("split"))
                idx = combo_split.findData(img.get("split"))
            combo_split.setCurrentIndex(idx)
        f.addRow("划分", combo_split)
        edit_defect = QLineEdit(str(img.get("defect_type") or ""))
        edit_defect.setPlaceholderText("缺陷类型（留空则不修改）")
        f.addRow("缺陷类型", edit_defect)
        row = QHBoxLayout()
        btn_ok = QPushButton("确定")
        btn_ok.setProperty("primary", True)
        btn_cancel = QPushButton("取消")
        row.addStretch(1)
        row.addWidget(btn_ok)
        row.addWidget(btn_cancel)
        f.addRow(row)
        btn_ok.clicked.connect(dlg.accept)
        btn_cancel.clicked.connect(dlg.reject)
        return dlg, combo_label, combo_split, edit_defect

    def _annotate_image(self, img: dict) -> None:
        dlg, combo_label, combo_split, edit_defect = \
            self._build_annotate_dialog(img)
        if dlg.exec() != QDialog.Accepted:
            return
        label = combo_label.currentData()
        split = combo_split.currentData()
        defect = edit_defect.text().strip() or None
        if label is None and split is None and defect is None:
            QMessageBox.information(self, "标注", "未做任何修改")
            return

        def _done(_res) -> None:
            self.reload_images()
            self.reload_tree()
        run_async(self, lambda: self._client.annotate_image(
            img.get("id"), label=label, split=split, defect_type=defect), _done)

    def _delete_image(self, img: dict) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle("确认删除")
        v = QVBoxLayout(dlg)
        v.addWidget(QLabel(f"确定删除图像 #{img.get('id')} 吗？"))
        chk_file = QCheckBox("同时删除磁盘文件")
        v.addWidget(chk_file)
        row = QHBoxLayout()
        btn_ok = QPushButton("删除")
        btn_ok.setProperty("primary", True)
        btn_cancel = QPushButton("取消")
        row.addStretch(1)
        row.addWidget(btn_ok)
        row.addWidget(btn_cancel)
        v.addLayout(row)
        btn_ok.clicked.connect(dlg.accept)
        btn_cancel.clicked.connect(dlg.reject)
        if dlg.exec() != QDialog.Accepted:
            return
        # 对话框销毁后 chk_file 已失效，必须先取布尔值再进后台线程
        delete_file = chk_file.isChecked()

        def _done(_res) -> None:
            self.reload_images()
            self.reload_tree()
        run_async(self, lambda: self._client.delete_image(
            img.get("id"), delete_file=delete_file),
            _done, lambda m: warn(self, f"删除失败：{m}"))

    # ── 批次节点右键：删除批次 / 导出清单 ──
    def _on_tree_menu(self, pos) -> None:
        item = self.tree.itemAt(pos)
        data = item.data(0, Qt.UserRole) if item is not None else None
        if not (isinstance(data, (tuple, list)) and len(data) == 4):
            return
        cat, ds_id, split, label = (str(v) for v in data)
        # 数据源节点（source 模式顶层）：编辑/删除数据源（前端反馈 v3：CRUD）
        if getattr(self, "_source_mode", False) and not ds_id and \
                item.data(0, Qt.UserRole + 1):
            self._source_menu(str(item.data(0, Qt.UserRole + 1)), pos)
            return
        if not ds_id or split or label:
            return  # 仅批次节点有批次操作
        if ds_id == "none":
            warn(self, "未分组图片没有批次实体，不支持批次操作")
            return
        name = self._ds_names.get(ds_id, f"批次{ds_id}")
        menu = QMenu(self)
        act_lineage = menu.addAction("批次详情与数据谱系…")
        act_export = menu.addAction("导出清单（CSV）…")
        act_del = menu.addAction("删除批次…")
        chosen = menu.exec(self.tree.viewport().mapToGlobal(pos))
        if chosen is act_del:
            self._delete_dataset(int(ds_id), name, item.text(0))
        elif chosen is act_export:
            self._export_dataset(int(ds_id), name)
        elif chosen is act_lineage:
            self._show_lineage(int(ds_id), name)

    def _show_lineage(self, ds_id: int, name: str) -> None:
        """M16f 数据谱系（review R6）：批次 → 参与训练的模型版本反查。"""

        def _done(data) -> None:
            data = data or {}
            used = data.get("used_by_models") or []
            ds = data.get("dataset") or {}
            lines = [
                f"批次：{name}（#{ds_id}）",
                f"品类：{ds.get('category', '-')}　"
                f"图片数：{data.get('total', 0)}　"
                f"来源：{ds.get('source_type', '-')}",
            ]
            if used:
                lines.append("参与训练的模型版本：")
                for m in used:
                    tag = "（当前激活）" if m.get("is_active") else ""
                    lines.append(f"  · {m.get('name', '-')}{tag}")
            else:
                lines.append("尚未参与任何模型版本的训练"
                             "（模型管理 → 准备模型后此处可见谱系）")
            QMessageBox.information(self, "批次详情与数据谱系",
                                    "\n".join(lines))
        run_async(self, lambda: self._client.dataset_detail(
            ds_id, page=1, page_size=1), _done)

    def _delete_dataset(self, ds_id: int, name: str, node_text: str) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle("删除批次")
        v = QVBoxLayout(dlg)
        v.addWidget(QLabel(
            f"确定删除批次「{name}」吗？\n"
            f"将删除该批次下全部图片记录（{node_text}）。"))
        chk_files = QCheckBox("同时删除磁盘文件")
        chk_keep = QCheckBox("保留检测记录（仅解除与图片的关联）")
        chk_keep.setChecked(True)
        v.addWidget(chk_files)
        v.addWidget(chk_keep)
        row = QHBoxLayout()
        btn_ok = QPushButton("删除批次")
        btn_ok.setProperty("primary", True)
        btn_cancel = QPushButton("取消")
        row.addStretch(1)
        row.addWidget(btn_ok)
        row.addWidget(btn_cancel)
        v.addLayout(row)
        btn_ok.clicked.connect(dlg.accept)
        btn_cancel.clicked.connect(dlg.reject)
        if dlg.exec() != QDialog.Accepted:
            return
        # 对话框销毁后复选框已失效，必须先取布尔值再进后台线程
        delete_files = chk_files.isChecked()
        keep_detections = chk_keep.isChecked()

        def _done(res) -> None:
            res = res or {}
            QMessageBox.information(
                self, "删除完成",
                f"已删除图片记录 {res.get('deleted_images', 0)} 条，"
                f"磁盘文件 {res.get('deleted_files', 0)} 个，"
                f"关联检测记录 {res.get('affected_detections', 0)} 条")
            self.reload_images()
            self.reload_tree()
        run_async(self, lambda: self._client.delete_dataset(
            ds_id, delete_files=delete_files,
            keep_detections=keep_detections), _done)

    def _export_dataset(self, ds_id: int, name: str) -> None:
        out, _ = QFileDialog.getSaveFileName(
            self, "导出批次图片清单", f"{name}.csv", "CSV 文件 (*.csv)")
        if not out:
            return

        def _collect():
            rows: list[tuple] = []
            page = 1
            while True:
                data = self._client.dataset_detail(ds_id, page=page,
                                                   page_size=500)
                items = (data or {}).get("items", [])
                rows.extend(
                    (i.get("id"), i.get("path"), i.get("split"),
                     i.get("label"), i.get("defect_type") or "")
                    for i in items)
                total = int((data or {}).get("total", 0) or 0)
                if not items or page * 500 >= total:
                    break
                page += 1
            with open(out, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(["id", "path", "split", "label", "defect_type"])
                writer.writerows(rows)
            return {"count": len(rows)}

        def _done(res) -> None:
            QMessageBox.information(
                self, "导出完成",
                f"已导出 {(res or {}).get('count', 0)} 条到：\n{out}")
        run_async(self, _collect, _done)

    # ── 孤儿文件清理 ──
    def _on_orphan_scan(self) -> None:
        self.progress.show()
        self.progress.update_task({"status": "running", "progress": 0,
                                   "message": "孤儿文件扫描中…"})

        def _work():
            res = self._client.orphan_scan()
            tid = (res or {}).get("task_id")
            if tid is None:
                return None
            for _ in range(600):  # 最长约 5 分钟
                t = self._client.get_task(tid) or {}
                status = str(t.get("status", ""))
                if status in ("done", "success"):
                    return t.get("result") or {}
                if status in ("failed", "error"):
                    raise RuntimeError(t.get("message") or "孤儿扫描失败")
                time.sleep(0.5)
            raise RuntimeError("孤儿扫描超时")

        run_async(self, _work, self._show_orphans,
                  on_fail=lambda m: warn(self, f"孤儿扫描失败：{m}"))

    def _show_orphans(self, result) -> None:
        self.progress.update_task({"status": "done", "progress": 1,
                                   "message": "孤儿文件扫描完成"})
        if result is None:
            return  # 后端离线等：错误信号已弹窗
        orphans = list(result.get("orphans", []))
        if not orphans:
            notify(self, f"未发现孤儿文件（扫描 {result.get('n_scanned', 0)} 个文件，"
                         f"DB 引用 {result.get('n_referenced', 0)} 条）")
            return
        dlg, lst = self._build_orphan_dialog(orphans)
        if dlg.exec() != QDialog.Accepted:
            return
        paths = [lst.item(i).data(Qt.UserRole) for i in range(lst.count())
                 if lst.item(i).checkState() == Qt.Checked]
        if not paths:
            return

        def _done(res) -> None:
            res = res or {}
            QMessageBox.information(
                self, "清理完成",
                f"已删除 {res.get('deleted', 0)} 个文件；"
                f"不存在 {res.get('missing', 0)} 个；"
                f"跳过 {len(res.get('skipped', []))} 个")
        run_async(self, lambda: self._client.orphan_clean(paths), _done)

    def _build_orphan_dialog(self, orphans: list) -> tuple[QDialog, QListWidget]:
        dlg = QDialog(self)
        dlg.setWindowTitle(f"孤儿文件清单（{len(orphans)} 个）")
        dlg.resize(720, 480)
        v = QVBoxLayout(dlg)
        v.addWidget(QLabel("以下文件在数据库中无引用，勾选后可删除："))
        lst = QListWidget()
        for p in orphans:
            item = QListWidgetItem(f"{p}    ({_fmt_size(p)})")
            item.setData(Qt.UserRole, p)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            lst.addItem(item)
        v.addWidget(lst, 1)
        row = QHBoxLayout()
        btn_all = QPushButton("全选")
        btn_none = QPushButton("全不选")
        btn_ok = QPushButton("删除选中")
        btn_ok.setProperty("primary", True)
        btn_cancel = QPushButton("取消")
        row.addWidget(btn_all)
        row.addWidget(btn_none)
        row.addStretch(1)
        row.addWidget(btn_ok)
        row.addWidget(btn_cancel)
        v.addLayout(row)
        btn_all.clicked.connect(lambda: self._orphan_set_all(lst, Qt.Checked))
        btn_none.clicked.connect(lambda: self._orphan_set_all(lst, Qt.Unchecked))
        btn_ok.clicked.connect(dlg.accept)
        btn_cancel.clicked.connect(dlg.reject)
        return dlg, lst

    @staticmethod
    def _orphan_set_all(lst: QListWidget, state) -> None:
        for i in range(lst.count()):
            lst.item(i).setCheckState(state)

    # ── 数据源：新建 / 编辑 / 删除 ──
    def _source_menu(self, src_name: str, pos) -> None:
        """数据源节点右键菜单：更新数据 / 编辑 / 删除（v8 增删改查）。"""
        menu = QMenu(self)
        act_update = menu.addAction("更新数据…")
        act_edit = menu.addAction("编辑数据源…")
        act_del = menu.addAction("删除数据源…")
        chosen = menu.exec(self.tree.viewport().mapToGlobal(pos))
        if chosen is act_update:
            self._ask_update_source(src_name)
        elif chosen is act_edit:
            self._edit_datasource(src_name)
        elif chosen is act_del:
            self._delete_datasource(src_name)

    def _ask_update_source(self, src_name: str) -> None:
        """更新数据源注册的数据内容：追加导入（可选清空重建）。

        用户要求"业务弹窗最多一次"：导入字段内联本对话框，确定后直接提交。
        """
        src = self._find_source(src_name)
        if not src:
            warn(self, f"数据源「{src_name}」不存在或已删除")
            return
        dlg = QDialog(self)
        dlg.setWindowTitle(f"更新数据：{src_name}")
        f = QFormLayout(dlg)
        combo, get_vals, sync = self._add_import_field_group(
            f, on_change=lambda m: btn_ok.setText("更新并导入" if m else "开始更新"))
        chk_rebuild = QCheckBox("清空该源现有数据后重建（危险）")
        f.addRow("", chk_rebuild)
        row = QHBoxLayout()
        btn_ok = QPushButton("开始更新")
        btn_ok.setProperty("primary", True)
        btn_cancel = QPushButton("取消")
        row.addStretch(1)
        row.addWidget(btn_ok)
        row.addWidget(btn_cancel)
        f.addRow(row)
        sync()
        btn_ok.clicked.connect(dlg.accept)
        btn_cancel.clicked.connect(dlg.reject)
        if dlg.exec() != QDialog.Accepted:
            return
        # 立即捕获（GC 竞态防护），闭包只引用局部变量
        method, folder, root, dsname = get_vals()
        rebuild = chk_rebuild.isChecked()
        src_id = int(src["id"])
        if rebuild:
            self._rebuild_source(src_id, src_name)
        self._import_into_source(src_id, method,
                                 folder=folder, root=root, dsname=dsname)

    def _rebuild_source(self, datasource_id: int, src_name: str) -> None:
        """清空该数据源下的全部批次与图片（保留源登记）。"""
        run_async(self, lambda: self._client.clear_datasource_data(datasource_id),
                  lambda r: (notify(self, f"已清空数据源数据：{src_name}"),
                             self.reload()), lambda m: warn(self, str(m)))

    def _find_source(self, src_name: str) -> dict | None:
        return next((d for d in self._datasources
                     if str(d.get("name")) == src_name), None)

    def _edit_datasource(self, src_name: str) -> None:
        """编辑数据源：名称/标注档位/备注/采样可改，模态锁定。"""
        src = self._find_source(src_name)
        if not src:
            warn(self, f"数据源「{src_name}」不存在或已删除")
            return
        dlg = QDialog(self)
        dlg.setWindowTitle(f"编辑数据源：{src_name}")
        dlg.setMinimumWidth(460)
        f = QFormLayout(dlg)
        edit_name = QLineEdit(str(src.get("name") or ""))
        f.addRow("名称", edit_name)
        modality = str(src.get("modality") or "image")
        f.addRow("模态（锁定）",
                 QLabel("图片" if modality == "image" else "视频"))
        combo_tier = QComboBox()
        for code in ("L0", "L1a", "L1b"):
            combo_tier.addItem(TIER_CN[code], code)
        cur_tier = str(src.get("label_tier") or "L0")
        idx = combo_tier.findData(cur_tier)
        combo_tier.setCurrentIndex(idx if idx >= 0 else 0)
        combo_tier.setToolTip("声明这份数据已具备的标注水平；导入数据后体检并自动校正")
        f.addRow("标注档位", combo_tier)
        # 前端反馈 v9：预训练组/检测组配置（调整即动态重算）
        spin_pn = QSpinBox()
        spin_pn.setRange(0, 9999)
        spin_pn.setValue(int(src.get("pretrain_normal") or 100))
        spin_pn.setToolTip("每个品类从 train/valid 取前 N 张正常图进预训练组")
        f.addRow("预训练正常图", spin_pn)
        spin_pa = QSpinBox()
        spin_pa.setRange(0, 9999)
        spin_pa.setValue(int(src.get("pretrain_anomaly") or 30))
        spin_pa.setToolTip("每个品类取前 M 张异常图进预训练组（不足则取尽）")
        f.addRow("预训练异常图", spin_pa)
        spin_bs = QSpinBox()
        spin_bs.setRange(1, 9999)
        spin_bs.setValue(int(src.get("batch_size") or 30))
        spin_bs.setToolTip("检测组按此数量分批次（产线按批消费/回队）")
        f.addRow("检测批次图数", spin_bs)
        chk_category = QCheckBox("数据已分品类（导入时按目录自动识别）")
        chk_category.setChecked(bool(src.get("per_category")))
        f.addRow("品类", chk_category)
        chk_template = QCheckBox("提供模板图（A_tpl.png 命名自动识别）")
        chk_template.setChecked(bool(src.get("has_template")))
        f.addRow("模板", chk_template)
        spin_interval = QSpinBox()
        spin_interval.setRange(0, 9999)
        spin_interval.setSpecialValueText("不抽帧（全帧）")
        spin_interval.setToolTip("视频采样：每隔 N 帧取 1 帧（仅视频模态生效）")
        interval = (src.get("sampling") or {}).get("frame_interval") or 0
        spin_interval.setValue(int(interval))
        spin_interval.setVisible(modality == "video")   # v6：图片源隐藏
        f.addRow("抽帧间隔", spin_interval)
        # 用 labelForField 取表单行标签（PySide6 的 addRow 返回值不可靠）
        lbl_interval = f.labelForField(spin_interval)
        if lbl_interval is not None:
            lbl_interval.setVisible(modality == "video")
        edit_note = QLineEdit(str(src.get("note") or ""))
        f.addRow("备注", edit_note)
        lbl_tpl = QLabel("模板图命名规则：检测图 A.png ↔ 模板图 A_tpl.png\n"
                         "同目录同名后缀配对；可分品类子目录，同品类可多张模板图")
        lbl_tpl.setProperty("subtext", True)
        lbl_tpl.setWordWrap(True)
        f.addRow("模板图", lbl_tpl)
        row = QHBoxLayout()
        btn_plan = QPushButton("另存为分组方案…")
        btn_plan.setToolTip("把当前 预训练组/检测组 分配保存为可复用方案")
        btn_plan.clicked.connect(lambda: self._save_plan_dialog(src.get("id")))
        row.addWidget(btn_plan)
        row.addStretch(1)
        btn_ok = QPushButton("保存")
        btn_ok.setProperty("primary", True)
        btn_cancel = QPushButton("取消")
        row.addWidget(btn_ok)
        row.addWidget(btn_cancel)
        f.addRow(row)
        btn_ok.clicked.connect(dlg.accept)
        btn_cancel.clicked.connect(dlg.reject)
        if dlg.exec() != QDialog.Accepted:
            return
        name = edit_name.text().strip()
        if not name:
            warn(self, "请填写数据源名称")
            return
        label_tier = str(combo_tier.currentData() or "L0")
        chk_category_val = chk_category.isChecked()
        chk_template_val = chk_template.isChecked()
        note_val = edit_note.text().strip()
        pn_val = spin_pn.value()
        pa_val = spin_pa.value()
        bs_val = spin_bs.value()
        sampling: dict = {}
        if modality == "video" and spin_interval.value() > 0:
            sampling = {"frame_interval": spin_interval.value()}

        def _done(res) -> None:
            if isinstance(res, dict) and res.get("id"):
                notify(self, f"数据源已更新：{name}")
                run_async(self, self._client.list_datasources,
                          self._fill_datasources)
                self.reload()
            else:
                warn(self, "数据源更新失败（可能名称重复）")
        run_async(self, lambda: self._client.update_datasource(
            src.get("id"), name=name, label_tier=label_tier,
            per_category=chk_category_val,
            has_template=chk_template_val,
            sampling=sampling, note=note_val,
            pretrain_normal=pn_val, pretrain_anomaly=pa_val,
            batch_size=bs_val), _done)

    def _save_plan_dialog(self, datasource_id: int) -> None:
        """把当前分组分配另存为可复用方案（前端反馈 v9）。"""
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "另存分组方案", "方案名称：")
        if not ok or not name.strip():
            return
        run_async(self, lambda: self._client.save_plan(datasource_id, name.strip()),
                  lambda r: notify(self, f"方案已保存：{(r or {}).get('name')}"),
                  lambda m: warn(self, str(m)))

    def _delete_datasource(self, src_name: str) -> None:
        """删除数据源：解除工单挂接，并级联删除其下批次/图片（v10 无未挂源）。"""
        src = self._find_source(src_name)
        if not src:
            warn(self, f"数据源「{src_name}」不存在或已删除")
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("删除数据源")
        v = QVBoxLayout(dlg)
        v.addWidget(QLabel(
            f"确定删除数据源「{src_name}」吗？\n"
            f"将解除其与工单的挂接，并删除该数据源下的全部批次和图片记录。"))
        row = QHBoxLayout()
        btn_ok = QPushButton("删除数据源")
        btn_ok.setProperty("danger", True)
        btn_cancel = QPushButton("取消")
        row.addStretch(1)
        row.addWidget(btn_ok)
        row.addWidget(btn_cancel)
        v.addLayout(row)
        btn_ok.clicked.connect(dlg.accept)
        btn_cancel.clicked.connect(dlg.reject)
        if dlg.exec() != QDialog.Accepted:
            return

        def _done(res) -> None:
            if isinstance(res, dict) and res.get("deleted"):
                notify(self, f"数据源已删除：{src_name}")
                # 问题2 修复（2026-10-09）：删除后立即重置筛选/选择状态并
                # 清空右侧表格——否则残留的批次/数据源过滤会让旧缩略图
                # 在 reload 完成前继续显示
                self._cur_ids = None
                self._cur_dataset = ""
                self._cur_group_label = ""
                self._page = 1
                self._images = []
                self._clear_thumb_cells()
                self.table.clearSpans()
                self.table.setRowCount(0)
                run_async(self, self._client.list_datasources,
                          self._fill_datasources)
                self.reload()
            else:
                warn(self, "删除失败，请重试")
        run_async(self, lambda: self._client.delete_datasource(
            src.get("id")), _done)

    def _fill_datasources(self, items) -> None:
        self._datasources = [d for d in (items or []) if isinstance(d, dict)]
        self._refresh_source_options()

    def _refresh_source_options(self) -> None:
        """查询栏「数据源」下拉：按模态过滤，首项「全部」（v10）。"""
        if not hasattr(self, "combo_source"):
            return
        cur = self.combo_source.currentData()
        self.combo_source.blockSignals(True)
        self.combo_source.clear()
        self.combo_source.addItem("全部", "")
        mod = self.combo_modality.currentData() or ""
        for d in self._datasources:
            if mod and d.get("modality") != mod:
                continue
            self.combo_source.addItem(str(d.get("name")), d.get("id"))
        idx = self.combo_source.findData(cur)
        self.combo_source.setCurrentIndex(idx if idx >= 0 else 0)
        self.combo_source.blockSignals(False)

    def _fill_source_combo(self, combo: QComboBox, modality: str = "") -> None:
        """数据源下拉：按模态过滤（空=全部）。

        v10：无"不挂数据源"选项——数据必须归属数据源，没有数据源时占位提示。
        """
        combo.clear()
        srcs = [d for d in self._datasources
                if not modality or d.get("modality") == modality]
        if not srcs:
            combo.addItem("（暂无数据源，请先新建数据源）", None)
            return
        for d in srcs:
            combo.addItem(str(d.get("name")), d.get("id"))
        combo.setCurrentIndex(0)

    def _add_import_field_group(self, f, on_change=None):
        """在 QFormLayout 中加「导入数据」下拉 + 内联路径字段（浏览按钮）。

        用户要求：① 业务弹窗最多一次——导入字段内联进本对话框，
        确定后直接提交，不再弹第二个导入对话框；② 文件路径字段必须
        支持点按钮打开文件浏览器。

        返回 (combo_import, get_values, sync)；get_values() 在 exec 返回后
        调用返回 (method, folder, root, dsname)；sync 用于按钮文案联动。
        """
        combo = QComboBox()
        combo.addItem("（暂不导入，稍后用「更新数据」）", "")
        combo.addItem("智能识别适配（推荐）", "adapt")
        combo.addItem("文件夹导入", "folder")
        combo.addItem("标准数据集", "mvtec")
        combo.addItem("上传图片", "upload")
        combo.setToolTip("一步式导入：确定后立即导入并挂到该数据源")
        f.addRow("导入数据", combo)

        lbl_folder = QLabel("文件夹路径")
        edit_folder = QLineEdit()
        edit_folder.setReadOnly(True)
        edit_folder.setPlaceholderText("点击「浏览…」选择图像文件夹")
        btn_folder = QPushButton("浏览…")
        h_folder = QHBoxLayout()
        h_folder.addWidget(edit_folder, 1)
        h_folder.addWidget(btn_folder)
        f.addRow(lbl_folder, h_folder)

        # 默认根目录读配置（环境变量/后端设置中心），取不到回退内置默认
        default_root = _default_data_root(self._client)
        lbl_root = QLabel("数据集根目录")
        edit_root = QLineEdit(default_root)
        edit_root.setPlaceholderText("如：D:/datasets/我的数据集")
        btn_root = QPushButton("浏览…")
        h_root = QHBoxLayout()
        h_root.addWidget(edit_root, 1)
        h_root.addWidget(btn_root)
        f.addRow(lbl_root, h_root)
        lbl_dsname = QLabel("数据集名称")
        edit_dsname = QLineEdit(Path(default_root).name)
        edit_dsname.setPlaceholderText("树顶层分组名，如 my_dataset")
        f.addRow(lbl_dsname, edit_dsname)

        lbl_hint = QLabel("")
        lbl_hint.setProperty("subtext", True)
        lbl_hint.setWordWrap(True)
        f.addRow("", lbl_hint)

        _HINTS = {
            "": "暂不导入：可稍后在数据源上点「更新数据」再导入。",
            "adapt": "自动识别目录结构与命名约定（MVTec/BTAD/GYU-DET/OK 后缀/"
                     "模板命名等，支持 data_origin / data_local / test_images）；"
                     "识别后弹窗确认再导入。",
            "folder": "导入 {源根}/{品类}/ 结构的图像文件夹（自动分品类/标正常）。",
            "mvtec": "导入 MVTec/MPDD/BTAD 标准目录（{根}/{品类}/{train|test}/{正常|缺陷}）。",
            "upload": "确定后打开图片文件多选（上传到该数据源）。",
        }

        def _sync() -> None:
            m = str(combo.currentData() or "")
            show_f, show_m = m == "folder", m in ("mvtec", "adapt")
            for w in (lbl_folder, edit_folder, btn_folder):
                w.setVisible(show_f)
            for w in (lbl_root, edit_root, btn_root, lbl_dsname, edit_dsname):
                w.setVisible(show_m)
            lbl_hint.setText(_HINTS.get(m, ""))
            lbl_hint.setVisible(bool(lbl_hint.text()))
            if on_change:
                on_change(m)

        def _pick(dest: QLineEdit) -> None:
            path = QFileDialog.getExistingDirectory(self, "选择文件夹", dest.text() or "")
            if path:
                dest.setText(path)
        btn_folder.clicked.connect(lambda: _pick(edit_folder))
        btn_root.clicked.connect(lambda: _pick(edit_root))
        combo.currentIndexChanged.connect(lambda _i: _sync())

        def _get_values():
            return (str(combo.currentData() or ""),
                    edit_folder.text().strip(),
                    edit_root.text().strip(),
                    edit_dsname.text().strip())
        return combo, _get_values, _sync

    def _on_create_datasource(self) -> None:
        """新建数据源：名称 + 模态 + 标注档位 + 视频抽帧间隔 + 模板提示。"""
        dlg = QDialog(self)
        dlg.setWindowTitle("新建数据源")
        dlg.setMinimumWidth(460)
        f = QFormLayout(dlg)
        edit_name = QLineEdit()
        edit_name.setPlaceholderText("如：3号线贴片机拍照图 / 产线A检测视频")
        f.addRow("名称", edit_name)
        combo_modality = QComboBox()
        combo_modality.addItem("图片", "image")
        combo_modality.addItem("视频", "video")
        f.addRow("模态", combo_modality)
        # 前端反馈 v10：数据流类型 本地目录|RTSP流|视频文件（后端已支持，
        # 此处补 UI 入口；视频文件按后端口径存 source_type=rtsp +
        # stream_config.path，由产线 _rtsp_loop 循环拉流消费）
        combo_srctype = QComboBox()
        combo_srctype.addItem("本地目录（导入/文件夹）", "local")
        combo_srctype.addItem("RTSP 实时流", "rtsp")
        combo_srctype.addItem("视频文件（循环模拟实时流）", "video")
        combo_srctype.setToolTip("数据流类型：本地目录 / RTSP 实时拉流 / "
                                 "本地视频文件循环模拟实时流（产线自动消费）")
        f.addRow("源类型", combo_srctype)
        edit_rtsp = QLineEdit()
        edit_rtsp.setPlaceholderText("rtsp://user:pass@host:554/stream")
        edit_rtsp.setToolTip("RTSP 拉流地址（选择 RTSP 实时流时必填）")
        f.addRow("RTSP 地址", edit_rtsp)
        lbl_rtsp = f.labelForField(edit_rtsp)
        # 视频文件路径（与后端 stream_config.path 口径一致）
        lbl_vpath = QLabel("视频文件")
        edit_vpath = QLineEdit()
        edit_vpath.setReadOnly(True)
        edit_vpath.setPlaceholderText("点击「浏览…」选择本地视频文件")
        btn_vpath = QPushButton("浏览…")
        h_vpath = QHBoxLayout()
        h_vpath.addWidget(edit_vpath, 1)
        h_vpath.addWidget(btn_vpath)
        f.addRow(lbl_vpath, h_vpath)
        # 流公共配置（与 _rtsp_loop 消费键严格一致：category/interval/loop）
        edit_scat = QLineEdit()
        edit_scat.setPlaceholderText("检测品类（可选，默认 default）")
        f.addRow("流品类", edit_scat)
        lbl_scat = f.labelForField(edit_scat)
        spin_sgap = QSpinBox()
        spin_sgap.setRange(0, 3600)
        spin_sgap.setSpecialValueText("默认（产线节拍）")
        spin_sgap.setToolTip("实时流抽帧间隔（秒）：每隔 N 秒取 1 帧送检；"
                             "0 = 跟随产线节拍")
        f.addRow("流抽帧间隔(秒)", spin_sgap)
        lbl_sgap = f.labelForField(spin_sgap)
        chk_loop = QCheckBox("循环播放（播完回到开头，模拟连续实时流）")
        chk_loop.setChecked(True)
        f.addRow("循环", chk_loop)
        lbl_loop = f.labelForField(chk_loop)

        def _pick_vpath() -> None:
            p, _f = QFileDialog.getOpenFileName(
                self, "选择视频文件", "",
                "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*)")
            if p:
                edit_vpath.setText(p)
        btn_vpath.clicked.connect(_pick_vpath)
        combo_tier = QComboBox()
        for code in ("L0", "L1a", "L1b"):
            combo_tier.addItem(TIER_CN[code], code)
        combo_tier.setToolTip("声明这份数据已具备的标注水平（前端反馈 v4："
                              "条件下沉到数据源）；导入数据后系统会体检并自动校正")
        f.addRow("标注档位", combo_tier)
        # 前端反馈 v9：预训练组/检测组配置（默认 100+30 / 批 30）
        spin_pn = QSpinBox()
        spin_pn.setRange(0, 9999)
        spin_pn.setValue(100)
        spin_pn.setToolTip("每个品类从 train/valid 取前 N 张正常图进预训练组")
        f.addRow("预训练正常图", spin_pn)
        spin_pa = QSpinBox()
        spin_pa.setRange(0, 9999)
        spin_pa.setValue(30)
        spin_pa.setToolTip("每个品类取前 M 张异常图进预训练组（不足则取尽）")
        f.addRow("预训练异常图", spin_pa)
        spin_bs = QSpinBox()
        spin_bs.setRange(1, 9999)
        spin_bs.setValue(30)
        spin_bs.setToolTip("检测组按此数量分批次（产线按批消费/回队）")
        f.addRow("检测批次图数", spin_bs)
        combo_plan = QComboBox()
        combo_plan.addItem("（不使用方案，按 N/M 自动分配）", None)
        combo_plan.setToolTip("复用已保存的分组方案：预训练组固定为方案清单")
        f.addRow("复用分组方案", combo_plan)

        def _fill_plans(data) -> None:
            for p in (data or {}).get("items", []) or []:
                combo_plan.addItem(
                    f"{p.get('name')}（预训练 {p.get('n_pretrain') or 0} 图）",
                    p.get("id"))
        run_async(self, self._client.list_plans, _fill_plans)
        # 前端反馈 v6：分品类 / 提供模板 声明开关
        chk_category = QCheckBox("数据已分品类（导入时按目录自动识别）")
        chk_category.setToolTip("声明这份数据按品类组织（{源根}/{品类}/ 目录结构）；"
                                "工单聚合时视为「品类独立」")
        f.addRow("品类", chk_category)
        chk_template = QCheckBox("提供模板图（A_tpl.png 命名自动识别）")
        chk_template.setToolTip("声明该源提供模板比对能力；模板图按命名规则 "
                                "A_tpl.png 或 template 目录识别")
        f.addRow("模板", chk_template)
        spin_interval = QSpinBox()
        spin_interval.setRange(0, 9999)
        spin_interval.setSpecialValueText("不抽帧（全帧）")
        spin_interval.setToolTip("视频采样：每隔 N 帧取 1 帧（仅视频模态生效）")
        f.addRow("抽帧间隔", spin_interval)
        # 用 labelForField 取表单行标签（PySide6 的 addRow 返回值不可靠）
        lbl_interval = f.labelForField(spin_interval)

        def _sync_modality(_i: int) -> None:
            is_video = combo_modality.currentData() == "video"
            # 前端反馈 v6/v7：图片模态下隐藏抽帧间隔行（label + 控件）
            spin_interval.setVisible(is_video)
            if lbl_interval is not None:
                lbl_interval.setVisible(is_video)
            spin_interval.setEnabled(is_video)
        combo_modality.currentIndexChanged.connect(_sync_modality)
        _sync_modality(0)

        def _sync_srctype(_i=None) -> None:
            st = combo_srctype.currentData()
            is_rtsp, is_video = st == "rtsp", st == "video"
            is_stream = is_rtsp or is_video
            for w, show in ((lbl_rtsp, is_rtsp), (edit_rtsp, is_rtsp),
                            (lbl_vpath, is_video), (edit_vpath, is_video),
                            (btn_vpath, is_video),
                            (lbl_scat, is_stream), (edit_scat, is_stream),
                            (lbl_sgap, is_stream), (spin_sgap, is_stream),
                            (lbl_loop, is_video), (chk_loop, is_video)):
                if w is not None:
                    w.setVisible(show)
            edit_rtsp.setEnabled(is_rtsp)
            if is_stream:
                # 实时流（RTSP/视频文件）本质是视频模态，联动为 video
                idx = combo_modality.findData("video")
                if idx >= 0:
                    combo_modality.setCurrentIndex(idx)
        combo_srctype.currentIndexChanged.connect(_sync_srctype)
        _sync_srctype()
        edit_note = QLineEdit()
        edit_note.setPlaceholderText("备注（可选）")
        f.addRow("备注", edit_note)
        lbl_tpl = QLabel("模板图命名规则：检测图 A.png ↔ 模板图 A_tpl.png\n"
                         "同目录同名后缀配对；可分品类子目录，同品类可多张模板图")
        lbl_tpl.setProperty("subtext", True)
        lbl_tpl.setWordWrap(True)
        f.addRow("模板图", lbl_tpl)
        # 一步式建源即导入（字段内联进本对话框，业务弹窗只此一次）
        combo_import, get_import_vals, sync_import = self._add_import_field_group(
            f, on_change=lambda m: btn_ok.setText("创建并导入" if m else "创建"))
        row = QHBoxLayout()
        btn_ok = QPushButton("创建")
        btn_ok.setProperty("primary", True)
        btn_cancel = QPushButton("取消")
        row.addStretch(1)
        row.addWidget(btn_ok)
        row.addWidget(btn_cancel)
        f.addRow(row)
        sync_import()   # btn_ok 定义后首次同步（按钮文案/字段可见性）
        btn_ok.clicked.connect(dlg.accept)
        btn_cancel.clicked.connect(dlg.reject)
        if dlg.exec() != QDialog.Accepted:
            return
        # exec 返回后对话框随时可能被 GC 销毁：立即把控件值全部捕获到
        # 局部变量，异步闭包只引用局部变量，绝不引用对话框控件——
        # 否则闭包内访问 chk_category.isChecked() 等会在对话框销毁后
        # 抛 "Internal C++ object already deleted"，创建静默失败
        # （竞态：测试时对话框未回收而通过，实际使用偶发——用户实测
        #  选标准数据集"没反应"）。
        name = edit_name.text().strip()
        modality = str(combo_modality.currentData() or "image")
        source_type = str(combo_srctype.currentData() or "local")
        rtsp_url = edit_rtsp.text().strip()
        vpath_val = edit_vpath.text().strip()
        scat_val = edit_scat.text().strip()
        sgap_val = spin_sgap.value()
        loop_val = chk_loop.isChecked()
        label_tier = str(combo_tier.currentData() or "L0")
        import_method, folder_val, root_val, dsname_val = get_import_vals()
        chk_category_val = chk_category.isChecked()
        chk_template_val = chk_template.isChecked()
        note_val = edit_note.text().strip()
        pn_val = spin_pn.value()
        pa_val = spin_pa.value()
        bs_val = spin_bs.value()
        plan_id_val = combo_plan.currentData()
        if not name:
            warn(self, "请填写数据源名称")
            return
        if source_type == "rtsp" and not rtsp_url:
            warn(self, "请填写 RTSP 拉流地址")
            return
        if source_type == "video" and not vpath_val:
            warn(self, "请选择视频文件")
            return
        sampling: dict = {}
        if modality == "video" and spin_interval.value() > 0:
            sampling = {"frame_interval": spin_interval.value()}
        # stream_config 键与后端 _rtsp_loop 消费口径严格一致：
        # rtsp -> {url}；视频文件 -> {path, loop}；公共可选 category/interval
        stream_config = None
        if source_type == "rtsp":
            stream_config = {"url": rtsp_url}
        elif source_type == "video":
            # 后端 source_type 仅 local/rtsp 两值：视频文件走 rtsp+path
            source_type = "rtsp"
            stream_config = {"path": vpath_val, "loop": loop_val}
        if stream_config is not None:
            if scat_val:
                stream_config["category"] = scat_val
            if sgap_val > 0:
                stream_config["interval"] = sgap_val
            import_method = ""   # 实时流不做本地导入（流由产线自动消费）

        def _done(res) -> None:
            if isinstance(res, dict) and res.get("id"):
                notify(self, f"数据源已创建：{name}")
                if import_method:
                    # 先刷新数据源列表再导入：否则导入的下拉里还没有新源，
                    # 批次会挂到"不挂数据源"（源 0 张 → 工单条件 L0）
                    def _after_fill(_items) -> None:
                        self._fill_datasources(_items)
                        self._import_into_source(
                            int(res["id"]), import_method,
                            folder=folder_val, root=root_val, dsname=dsname_val)
                    run_async(self, self._client.list_datasources, _after_fill)
                else:
                    run_async(self, self._client.list_datasources,
                              self._fill_datasources)
            else:
                warn(self, "数据源创建失败（可能名称重复）")
        run_async(self, lambda: self._client.create_datasource(
            name, modality=modality, label_tier=label_tier,
            per_category=chk_category_val, has_template=chk_template_val,
            sampling=sampling, note=note_val,
            pretrain_normal=pn_val, pretrain_anomaly=pa_val,
            batch_size=bs_val, plan_id=plan_id_val,
            source_type=source_type, stream_config=stream_config), _done)

    def _import_into_source(self, datasource_id, method: str,
                            folder: str = "", root: str = "",
                            dsname: str = "") -> None:
        """一步式导入到数据源（不再弹第二个对话框，字段已内联在调用方）。

        用户要求"业务弹窗最多一次"：路径来自内联字段；上传走文件浏览器
        （系统级选择，非业务弹窗链）。
        """
        if method == "adapt":
            root = (root or "").strip()
            if not root:
                warn(self, "请先填写/选择数据集根目录")
                return
            self._start_adapt_import(datasource_id, root, dsname)
        elif method == "mvtec":
            root = (root or "").strip()
            if not root:
                warn(self, "请先填写/选择数据集根目录")
                return
            run_async(self, lambda: self._client.import_mvtec(
                root, name="", note="",
                dataset_name=(dsname or Path(root).name),
                datasource_id=datasource_id), self._on_task_started)
        elif method == "upload":
            self._pick_upload_to_source(datasource_id)
        elif method == "folder":
            folder = (folder or "").strip()
            if not folder:
                warn(self, "请先选择图像文件夹")
                return
            cat = self.combo_category.currentData() or "default"
            run_async(self, lambda: self._client.import_folder(
                folder, cat, split="train", label="normal", name="", note="",
                datasource_id=datasource_id), self._on_task_started)

    # ── 智能识别适配（数据集适配模块：识别 -> 确认 -> 导入）──
    def _start_adapt_import(self, datasource_id: int, root: str,
                            dsname: str = "") -> None:
        """第一步：后台识别数据集内容（probe 只读扫描），完成后弹确认框。"""
        self.progress.show()
        self.progress.update_task({"status": "running", "progress": 0,
                                   "message": f"正在识别数据集内容：{root}"})

        def _work():
            return self._client.probe_dataset(root)

        def _ok(report) -> None:
            self.progress.update_task({"status": "done", "progress": 1,
                                       "message": "识别完成"})
            if not isinstance(report, dict) or not report.get("format"):
                warn(self, "识别失败：后端返回为空")
                return
            if not report.get("n_images"):
                warn(self, "该目录下未识别到图片，请检查路径")
                return
            self._show_adapt_confirm(report, datasource_id, dsname)

        def _fail(msg) -> None:
            self.progress.update_task({"status": "failed",
                                       "message": "识别失败"})
            warn(self, f"数据集识别失败：{msg}")

        run_async(self, _work, _ok, _fail)

    def _show_adapt_confirm(self, report: dict, datasource_id: int,
                            dsname: str) -> None:
        """第二步：识别结果确认弹窗，确认后按映射提交后台导入。"""
        dlg = DatasetAdaptConfirmDialog(report, self)
        if dlg.exec() != QDialog.Accepted:
            return
        cfg = dlg.config()
        root = str(report.get("root") or "")
        fmt = str(cfg.get("format") or report.get("format") or "dir_rules")
        category = str(cfg.get("category") or report.get("category_hint")
                       or "default")
        run_async(self, lambda: self._client.adapt_import(
            root, fmt, category=category,
            ok_role=str(cfg.get("ok_role") or "train"),
            group_overrides=cfg.get("group_overrides") or None,
            categories=cfg.get("categories"),
            dataset_name=(dsname or Path(root).name),
            datasource_id=datasource_id), self._on_task_started,
            lambda m: warn(self, f"适配导入提交失败：{m}"))
        # 识别结论回写数据源声明（工单条件聚合用）：多品类 / 含模板图
        self._sync_source_flags_after_adapt(report, cfg, datasource_id)

    def _sync_source_flags_after_adapt(self, report: dict, cfg: dict,
                                       datasource_id: int) -> None:
        """把识别结论（分品类 / 有模板图 / 标注档位）同步到数据源声明。

        label_tier 按识别结果提升（只升不降）：有掩码 -> L1b，有缺陷图 -> L1a。
        用户手动声明了更高档位时保持；识别发现更丰富标注时自动校正。
        """
        try:
            fmt = str(report.get("format") or "")
            cats = report.get("categories") or []
            per_category = fmt == "mvtec_like" and len(cats) > 1
            has_template = False
            n_anomaly = 0
            n_masks = int(report.get("n_masks", 0) or 0)
            if fmt == "mvtec_like":
                for c in cats:
                    counts = c.get("counts") or {}
                    has_template = has_template or bool(counts.get("template", 0))
                    n_anomaly += sum(v for k, v in counts.items()
                                     if str(k).endswith("_anomaly"))
                    n_masks += int(c.get("n_masks", 0) or 0)
            else:
                for c in cats:
                    for g in (c.get("groups") or []):
                        has_template = has_template or bool(g.get("template", 0))
                        n_anomaly += int(g.get("anomaly", 0) or 0)
                if str(cfg.get("ok_role")) == "template":
                    has_template = True
            src = next((d for d in self._datasources
                        if str(d.get("id")) == str(datasource_id)), None)
            if src is None:
                return
            # 推荐档位只升不降（L0 < L1a < L1b）
            recommend = "L1b" if n_masks else ("L1a" if n_anomaly else "L0")
            cur_tier = str(src.get("label_tier") or "L0")
            order = {"L0": 0, "L1a": 1, "L1b": 2}
            new_tier = recommend if order.get(recommend, 0) > order.get(cur_tier, 0) \
                else cur_tier
            if (bool(src.get("per_category")) != per_category or
                    bool(src.get("has_template")) != has_template or
                    new_tier != cur_tier):
                run_async(self, lambda: self._client.update_datasource(
                    datasource_id, per_category=per_category,
                    has_template=has_template, label_tier=new_tier),
                    lambda _r: run_async(self, self._client.list_datasources,
                                         self._fill_datasources))
        except Exception:  # noqa: BLE001 声明同步失败不影响导入
            pass

    def _pick_upload_to_source(self, datasource_id) -> None:
        """上传图片到数据源（文件浏览器多选；品类取当前筛选）。"""
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择要上传的图片", "",
            "图片文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if not paths:
            return
        cat = self.combo_category.currentData() or "default"

        def _upload_all():
            for p in paths:
                self._client.upload_image(p, cat, name="", note="",
                                          datasource_id=datasource_id)
            return {"count": len(paths)}

        run_async(self, _upload_all,
                  lambda r: (notify(self, f"已上传 {(r or {}).get('count', 0)} 张"),
                             self.reload_images(), self.reload_tree()),
                  lambda m: warn(self, str(m)))

    # ── 导入动作 ──
    def _ask_import_meta(self, title: str, modality: str = "image",
                         with_source: bool = True,
                         with_batch: bool = True,
                         ) -> tuple[str, str, str, int | None] | None:
        """导入对话框公共部分：品类 + 批次名 + 备注 + 数据源（批次名/备注可空）。"""
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        f = QFormLayout(dlg)
        edit_cat = QLineEdit(
            self.combo_category.currentData() or self._get_category() or "")
        f.addRow("品类", edit_cat)
        edit_name = QLineEdit()
        edit_name.setPlaceholderText("批次名（留空则自动生成：来源_时间戳）")
        edit_note = QLineEdit()
        edit_note.setPlaceholderText("备注（可选）")
        if with_batch:
            f.addRow("批次名", edit_name)
            f.addRow("备注", edit_note)
        combo_source = QComboBox()
        if with_source:
            self._fill_source_combo(combo_source, modality)
            combo_source.setToolTip("导入的数据挂到该数据源下（工单按数据源统计）")
            f.addRow("数据源", combo_source)
        row = QHBoxLayout()
        btn_ok = QPushButton("确定")
        btn_ok.setProperty("primary", True)
        btn_cancel = QPushButton("取消")
        row.addStretch(1)
        row.addWidget(btn_ok)
        row.addWidget(btn_cancel)
        f.addRow(row)
        btn_ok.clicked.connect(dlg.accept)
        btn_cancel.clicked.connect(dlg.reject)
        if dlg.exec() != QDialog.Accepted:
            return None
        cat = edit_cat.text().strip()
        if not cat:
            return None
        ds_id = combo_source.currentData() if with_source else None
        if with_source and ds_id is None:
            warn(self, "请先新建数据源，再把数据导入到该数据源")
            return None
        return cat, edit_name.text().strip(), edit_note.text().strip(), ds_id

    def _on_import_data(self) -> None:
        """统一导入入口：导入方式 + 路径字段（浏览按钮）内联，一次弹窗直接提交。"""
        dlg = QDialog(self)
        dlg.setWindowTitle("导入数据")
        dlg.setMinimumWidth(420)
        f = QFormLayout(dlg)
        combo_source = QComboBox()
        self._fill_source_combo(combo_source, "")
        f.addRow("导入到数据源", combo_source)
        combo, get_vals, sync = self._add_import_field_group(
            f, on_change=lambda m: (btn_ok.setText("开始导入"),
                                    btn_ok.setEnabled(bool(m))))
        row = QHBoxLayout()
        btn_ok = QPushButton("开始导入")
        btn_ok.setProperty("primary", True)
        btn_cancel = QPushButton("取消")
        row.addStretch(1)
        row.addWidget(btn_ok)
        row.addWidget(btn_cancel)
        f.addRow(row)
        sync()
        btn_ok.clicked.connect(dlg.accept)
        btn_cancel.clicked.connect(dlg.reject)
        if dlg.exec() != QDialog.Accepted:
            return
        ds_id = combo_source.currentData()
        if ds_id is None:
            # v10：数据必须归属数据源，无源时引导新建（不产生未挂源数据）
            warn(self, "请先新建数据源，再把数据导入到该数据源")
            return
        method, folder, root, dsname = get_vals()
        if not method:
            warn(self, "请先选择导入方式")
            return
        self._import_into_source(ds_id, method,
                                 folder=folder, root=root, dsname=dsname)

    def _on_task_started(self, data) -> None:
        if data and data.get("task_id"):
            self.progress.show()
            self.progress.update_task({"status": "running", "progress": 0,
                                       "message": f"任务 {data['task_id']} 已提交"})

    def _on_task(self, task: dict) -> None:
        self.progress.update_task(task)
        if str(task.get("status")) in ("done", "success"):
            self.reload_images()
            self.reload_tree()
            self._load_queue()

    # ══════════════════ 视频面板（v10 并入图像页，模态=视频时展示）══════════════════
    def _build_video_panel(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        self.table_video = QTableWidget(0, 7)
        self.table_video.setHorizontalHeaderLabels(
            ["ID", "路径", "品类", "帧率", "帧数", "时长(s)", "分辨率"])
        self.table_video.horizontalHeader().setStretchLastSection(True)
        self.table_video.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table_video.setAlternatingRowColors(True)
        self.table_video.verticalHeader().setVisible(False)
        lay.addWidget(self.table_video, 1)
        return w

    def reload_videos(self) -> None:
        run_async(self, lambda: self._client.list_videos(page_size=100),
                  self._fill_videos)

    def _fill_videos(self, data) -> None:
        items = (data or {}).get("items", []) if isinstance(data, dict) else (data or [])
        self.table_video.clearSpans()
        if not items:
            self.table_video.setRowCount(1)
            self.table_video.setSpan(0, 0, 1, 7)
            self.table_video.setItem(0, 0, QTableWidgetItem("暂无视频数据"))
            return
        self.table_video.setRowCount(len(items))
        for r, v in enumerate(items):
            vals = [v.get("id"), v.get("path"), v.get("category"),
                    v.get("fps"), v.get("n_frames"), v.get("duration_s"),
                    f"{v.get('width', '?')}×{v.get('height', '?')}"]
            for c, val in enumerate(vals):
                self.table_video.setItem(r, c, QTableWidgetItem(str(val or "-")))

    def _on_import_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mov *.mkv)")
        if not path:
            return
        meta = self._ask_import_meta("导入视频", modality="video",
                                     with_batch=False)
        if not meta:
            return
        cat, _name, _note, ds_id = meta

        def _done(_res) -> None:
            notify(self, "视频已导入")
            self.reload()
        run_async(self, lambda: self._client.import_video(
            path, cat, datasource_id=ds_id), _done)

    # ══════════════════ 通用 ══════════════════
    def reload(self, categories: list[str] | None = None) -> None:
        """页面激活/品类列表更新时刷新。"""
        if categories:
            self._global_categories = list(categories)
            cur = self.combo_category.currentData() or ""
            self.combo_category.blockSignals(True)
            self.combo_category.clear()
            self.combo_category.addItem("全部", "")
            for c in categories:
                self.combo_category.addItem(c, c)
            idx = self.combo_category.findData(cur)
            if idx >= 0:
                self.combo_category.setCurrentIndex(idx)
            self.combo_category.blockSignals(False)
            # 品类列表重建后刷新批次下拉（依赖数据源分组）
            self._refresh_batch_options(self.combo_category.currentData() or "")
        run_async(self, self._client.list_datasources, self._fill_datasources)
        # 模态=视频：展示视频页；否则图片页（v10 无独立视频页）
        if self.combo_modality.currentData() == "video":
            self.stack_data.setCurrentIndex(1)
            self.reload_videos()
        else:
            self.stack_data.setCurrentIndex(0)
            self.reload_images()
        self.reload_tree()
        self._load_queue()   # 前端反馈 v5：数据流队列
        return
