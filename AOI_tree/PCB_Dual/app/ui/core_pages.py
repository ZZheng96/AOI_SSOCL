"""嵌入树干 AOI_Core 的 PySide6 页面（数据/增强/模型/评估/反馈复核/学习/统计）。

Core 的 ui 包只经 HTTP/WS 访问后端（ApiClient / TaskMonitor），不 import
backend / algo / torch，因此可在 PCB_Dual 进程内直接加载：把 AOI_Core 根目录
追加到 sys.path 末尾，再 ``import ui.*``。
PCB_Dual 没有顶层 ``ui`` 包（它的是 ``app.ui``），不会撞名。

页面在首次切到本 Tab 且 Core 健康时才构建；Core 未就绪显示占位 + 重试。
"""
from __future__ import annotations

import logging
import sys

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.config import get_settings

logger = logging.getLogger(__name__)

# (导航标题, 模块, 类名, 构造方式)
_PAGES = [
    ("数据管理", "ui.pages.data_page", "DataPage", "cat_tm"),
    ("数据增强", "ui.pages.augment_page", "AugmentPage", "cat_tm"),
    ("模型管理", "ui.pages.model_page", "ModelPage", "cat_tm"),
    ("评估看板", "ui.pages.eval_page", "EvalPage", "cat_budget_tm"),
    ("标注复核", "ui.pages.feedback_page", "FeedbackPage", "cat_tm"),
    ("学习效果", "ui.pages.learning_page", "LearningPage", "cat_tm"),
    ("存档统计", "ui.pages.stats_page", "StatsPage", "cat_budget"),
]


def _ensure_core_on_path() -> None:
    core_dir = str(get_settings().aoi_core_dir)
    if core_dir not in sys.path:
        sys.path.append(core_dir)


class CoreHubPage(QWidget):
    """「特征学习（AOI_Core）」Tab：左侧子导航 + 右侧 Core 页面栈。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._built = False
        self._client = None
        self._task_monitor = None
        self._pages: list[QWidget] = []
        self._categories: list[str] = []

        self._root = QVBoxLayout(self)
        self._root.setContentsMargins(0, 0, 0, 0)

        self._placeholder = QWidget()
        ph = QVBoxLayout(self._placeholder)
        ph.addStretch(1)
        self._lbl_status = QLabel("AOI_Core 树干服务连接中…")
        self._lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_status.setWordWrap(True)
        ph.addWidget(self._lbl_status)
        self._btn_retry = QPushButton("重试连接")
        self._btn_retry.clicked.connect(self.ensure_built)
        ph.addWidget(self._btn_retry, 0, Qt.AlignmentFlag.AlignCenter)
        ph.addStretch(1)
        self._root.addWidget(self._placeholder)

    # ── 生命周期 ──
    def ensure_built(self) -> None:
        """Tab 激活时调用：已构建则刷新当前页，否则探活后构建。"""
        if self._built:
            self._reload_current()
            return
        from app.core.aoi_core_launcher import core_healthy

        url = get_settings().aoi_core_url
        if not core_healthy(url, timeout=1.0):
            self._lbl_status.setText(
                f"AOI_Core 未就绪（{url}）。\n启动时会自动拉起，冷启动约 10s；"
                f"日志见 storage/logs/aoi_core_child.log")
            QTimer.singleShot(3000, self._auto_retry)
            return
        try:
            self._build(url)
        except Exception as exc:  # noqa: BLE001
            logger.exception("加载 AOI_Core 页面失败")
            self._lbl_status.setText(f"加载 AOI_Core 页面失败：{exc}")

    def _auto_retry(self) -> None:
        if not self._built and self.isVisible():
            self.ensure_built()

    def shutdown(self) -> None:
        if self._task_monitor is not None:
            try:
                self._task_monitor.stop()
            except Exception:  # noqa: BLE001
                pass

    # ── 构建 ──
    def _build(self, url: str) -> None:
        import importlib

        _ensure_core_on_path()
        from ui.api_client import ApiClient
        from ui.theme import LIGHT_QSS
        from ui.widgets.task_progress import TaskMonitor

        self._client = ApiClient(url, self)
        self._task_monitor = TaskMonitor(self._client.ws_url("/ws/tasks"), self)
        self._task_monitor.start()

        body = QWidget()
        body.setObjectName("CoreHub")
        # Core 主题只作用于本 Tab 子树，不影响 PCB_Dual 其它页
        body.setStyleSheet(LIGHT_QSS)
        lay = QHBoxLayout(body)
        lay.setContentsMargins(0, 0, 0, 0)

        left = QVBoxLayout()
        left.addWidget(QLabel("品类"))
        self.combo_cat = QComboBox()
        self.combo_cat.setMinimumWidth(150)
        self.combo_cat.currentIndexChanged.connect(lambda _i: self._reload_current())
        left.addWidget(self.combo_cat)
        self.nav = QListWidget()
        self.nav.setFixedWidth(150)
        left.addWidget(self.nav, 1)
        lay.addLayout(left)

        self.stack = QStackedWidget()
        lay.addWidget(self.stack, 1)

        get_cat = self.current_category
        get_budget = lambda: 200.0  # noqa: E731
        tm = self._task_monitor
        by_title: dict[str, QWidget] = {}
        for title, mod_name, cls_name, kind in _PAGES:
            try:
                cls = getattr(importlib.import_module(mod_name), cls_name)
                if kind == "cat_tm":
                    page = cls(self._client, get_cat, tm)
                elif kind == "cat_budget_tm":
                    page = cls(self._client, get_cat, get_budget, tm)
                else:
                    page = cls(self._client, get_cat, get_budget)
                page._get_level = lambda: None
                page._get_workorder = lambda: None
                page._get_workorder_id = lambda: None
            except Exception as exc:  # noqa: BLE001
                logger.exception("Core 页面 %s 加载失败", title)
                page = QLabel(f"{title} 加载失败：{exc}")
                page.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.stack.addWidget(page)
            self.nav.addItem(title)
            self._pages.append(page)
            by_title[title] = page

        model_page, learning_page = by_title.get("模型管理"), by_title.get("学习效果")
        if hasattr(model_page, "open_learning") and hasattr(learning_page, "show_category"):
            def _open_learning(cat: str) -> None:
                learning_page.show_category(cat)
                self.goto("学习效果")
            model_page.open_learning.connect(_open_learning)

        self.nav.currentRowChanged.connect(self._on_nav)
        self._root.removeWidget(self._placeholder)
        self._placeholder.deleteLater()
        self._root.addWidget(body)
        self._built = True
        self._load_categories()
        self.nav.setCurrentRow(0)

    def _load_categories(self) -> None:
        try:
            cats = self._client.list_categories()
        except Exception as exc:  # noqa: BLE001
            logger.warning("拉取 Core 品类失败：%s", exc)
            cats = []
        self._categories = [str(c) for c in cats]
        keep = self.combo_cat.currentData()
        self.combo_cat.blockSignals(True)
        self.combo_cat.clear()
        self.combo_cat.addItem("（全部品类）", "")
        for c in self._categories:
            self.combo_cat.addItem(c, c)
        idx = self.combo_cat.findData(keep) if keep else 0
        self.combo_cat.setCurrentIndex(max(idx, 0))
        self.combo_cat.blockSignals(False)

    # ── 导航 ──
    def current_category(self) -> str:
        combo = getattr(self, "combo_cat", None)
        return str(combo.currentData() or "") if combo is not None else ""

    def goto(self, title: str) -> None:
        for i, (t, *_rest) in enumerate(_PAGES):
            if t == title:
                self.nav.setCurrentRow(i)
                return

    def _on_nav(self, row: int) -> None:
        if 0 <= row < self.stack.count():
            self.stack.setCurrentIndex(row)
            self._reload_current()

    def _reload_current(self) -> None:
        if not self._built:
            return
        page = self.stack.currentWidget()
        fn = getattr(page, "reload", None)
        if not callable(fn):
            return
        try:
            if type(page).__name__ == "DataPage":
                fn(self._categories)
            else:
                fn()
        except TypeError:
            fn()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Core 页面刷新失败：%s", exc)
