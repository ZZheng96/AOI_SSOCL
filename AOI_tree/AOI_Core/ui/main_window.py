"""MainWindow：AOI 实时在线 AI 质检系统主窗口。

左侧导航（QListWidget 10 项，M16b 按用户操作流分组：工作台首屏 →
准备期 → 投产期 → 复盘期 → 系统）+ 右侧 QStackedWidget；
顶栏：系统标题 / 当前品类下拉（全局共享）/ 设备徽标 / 延迟预算标签 /
后端连接指示灯（定时 ping /api/health）。
后端未启动时构造不崩溃，所有加载均为后台线程 + 失败容错。
"""
from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QGuiApplication
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QStackedWidget,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from ui.api_client import ApiClient
from ui.pages.augment_page import AugmentPage
from ui.pages.common import make_level_badge, run_async, update_level_badge
from ui.pages.dashboard_page import DashboardPage
from ui.pages.data_page import DataPage
from ui.pages.eval_page import EvalPage
from ui.pages.feedback_page import FeedbackPage
from ui.pages.learning_page import LearningPage
from ui.pages.model_page import ModelPage
from ui.pages.monitor_page import MonitorPage
from ui.pages.settings_page import SettingsPage
from ui.pages.stats_page import StatsPage
from ui.theme import DANGER, SUCCESS, TEXT_SUB
from ui.widgets.task_progress import TaskMonitor

# 导航项：(标题, QStyle 标准图标) —— M16b 按用户操作流重排：
# 工作台（首屏总览）→ 准备期（数据/增强/模型/验收）→ 投产期（监控/反馈）
# → 复盘期（学习/报表）→ 系统设置。行索引与页面栈一一对应。
NAV_ITEMS = [
    ("工作台", QStyle.SP_DesktopIcon),
    ("数据管理", QStyle.SP_DirHomeIcon),
    ("数据增强", QStyle.SP_FileDialogNewFolder),
    ("模型管理", QStyle.SP_ComputerIcon),
    ("评估看板", QStyle.SP_DialogApplyButton),
    ("实时监控", QStyle.SP_MediaPlay),
    ("标注反馈", QStyle.SP_MessageBoxInformation),
    ("学习效果", QStyle.SP_FileDialogInfoView),
    ("存档统计", QStyle.SP_FileDialogDetailedView),
    ("系统设置", QStyle.SP_FileDialogListView),
]

# M16b 导航分组标题：{插入位置（该 NAV_ITEMS 索引之前）: 分组名}
NAV_GROUP_HEADERS = {
    1: "── 准备期：数据与上线 ──",
    5: "── 投产期：日常检测 ──",
    7: "── 复盘期：学习与报表 ──",
    9: "── 系统 ──",
}


# M15c 角色→可见导航（用户操作流 §1 角色分工）：
# operator（产线操作员）只保留日常总览/送检/反馈/报表；engineer/admin 全量。
ROLE_NAV_VISIBLE = {
    "operator": {"工作台", "实时监控", "标注反馈", "存档统计"},
}
ROLE_NAMES = {"operator": "操作员", "engineer": "工程师", "admin": "管理员"}


def role_allows_nav(role: str, title: str) -> bool:
    """该角色是否可访问某导航页（A14 页面级管控判定，供 goto/_apply_role 复用）。

    未在 ROLE_NAV_VISIBLE 登记的角色（engineer/admin）= 全量可见。
    """
    visible = ROLE_NAV_VISIBLE.get(role)
    return visible is None or title in visible


class MainWindow(QMainWindow):
    """系统主窗口。"""

    def __init__(self, base_url: str = "http://127.0.0.1:8017"):
        super().__init__()
        self.setWindowTitle("AOI 实时在线 AI 质检系统")
        # 窗口尺寸自适应屏幕（前端反馈：1440×900 在小屏放不下）：
        # 取可用区 94%，上限 1440×900，下限 1180×740；无屏信息时回退 1280×800
        geo = QGuiApplication.primaryScreen().availableGeometry() \
            if QGuiApplication.primaryScreen() is not None else None
        if geo is not None and geo.width() > 0:
            self.resize(max(1180, min(1440, int(geo.width() * 0.94))),
                        max(740, min(900, int(geo.height() * 0.94))))
        else:
            self.resize(1280, 800)
        self.setMinimumSize(1120, 700)

        self.client = ApiClient(base_url, self)
        self._budget_ms = 200.0
        self._categories: list[str] = []
        # U-workorder v2：全局以工单为主轴（工单 -> 品类联动 -> 条件徽标）
        self._workorders: list[dict] = []

        # 全局任务进度监视器（/ws/tasks）
        self.task_monitor = TaskMonitor(self.client.ws_url("/ws/tasks"), self)
        self.task_monitor.error.connect(
            lambda m: self.statusBar().showMessage(m, 5000))
        self.task_monitor.start()

        # ── 中央：左导航 + 右页面栈 ──
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_topbar())

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        root.addLayout(body, 1)

        self.nav = QListWidget()
        self.nav.setObjectName("nav")
        self.nav.setFixedWidth(190)
        style = self.style()
        # M16b：导航含分组标题行（不可选），row→页面栈索引经 _row2page 映射
        self._row2page: list[int | None] = []
        self._title2row: dict[str, int] = {}
        for page_idx, (title, icon_id) in enumerate(NAV_ITEMS):
            header = NAV_GROUP_HEADERS.get(page_idx)
            if header:
                h = QListWidgetItem(header)
                h.setFlags(Qt.NoItemFlags)  # 分组标题不可选
                h.setForeground(QColor(TEXT_SUB))
                self.nav.addItem(h)
                self._row2page.append(None)
            item = QListWidgetItem(style.standardIcon(icon_id), title)
            item.setSizeHint(item.sizeHint().expandedTo(
                item.sizeHint().__class__(190, 42)))
            self.nav.addItem(item)
            self._title2row[title] = self.nav.count() - 1
            self._row2page.append(page_idx)
        body.addWidget(self.nav)

        self.stack = QStackedWidget()
        body.addWidget(self.stack, 1)
        self._build_pages()

        self.nav.currentRowChanged.connect(self._on_nav_changed)
        self.nav.setCurrentRow(0)

        # ── 后端连接指示灯：定时 ping ──
        self._ping_timer = QTimer(self)
        self._ping_timer.setInterval(3000)
        self._ping_timer.timeout.connect(self._ping_backend)
        self._ping_timer.start()
        self._ping_backend()

        # 产线状态徽标轻量轮询（10s；仅刷新徽标，不触发页面 reload）
        self._pipe_status_timer = QTimer(self)
        self._pipe_status_timer.setInterval(10_000)
        self._pipe_status_timer.timeout.connect(self._poll_pipeline_badge)
        self._pipe_status_timer.start()

        self.statusBar().showMessage("就绪")

    # ══════════════════ 顶栏 ══════════════════
    def _build_topbar(self) -> QWidget:
        bar = QFrame()
        bar.setStyleSheet(
            "QFrame { background: #FFFFFF; border-bottom: 1px solid #E5E7EB; }")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(16, 8, 16, 8)
        lay.setSpacing(12)

        title = QLabel("AOI 实时在线 AI 质检")
        font = title.font()
        font.setPointSizeF(13)
        font.setBold(True)
        title.setFont(font)
        title.setStyleSheet("color: #2F6FED;")
        lay.addWidget(title)

        sub = QLabel("少样本即可上线 · 反馈即学即用")
        sub.setProperty("subtext", True)
        lay.addWidget(sub)
        lay.addStretch(1)

        # U-workorder v2：先选工单（统计/检测口径），品类随工单联动
        lay.addWidget(QLabel("当前工单"))
        self.combo_workorder = QComboBox()
        self.combo_workorder.setMinimumWidth(160)
        self.combo_workorder.addItem("加载中…", None)
        self.combo_workorder.currentIndexChanged.connect(
            self._on_workorder_changed)
        lay.addWidget(self.combo_workorder)

        # U-workorder v2：当前工单数据条件徽标
        self.badge_level = make_level_badge()
        lay.addWidget(self.badge_level)

        self.badge_device = QLabel("设备 -")
        self.badge_device.setProperty("badge", "muted")
        self.badge_device.setToolTip(
            "推理计算设备：CUDA=使用 NVIDIA 显卡加速（检测快），\n"
            "CPU=无显卡降速运行。此处显示型号便于确认显卡是否被识别")
        lay.addWidget(self.badge_device)

        # 产线状态徽标（前端反馈 2026-09-13 #7）：跟随当前工单，
        # 10s 轻量轮询刷新（监控页手动启停后顶栏同步可见）
        self.badge_pipeline = QLabel("产线 -")
        self.badge_pipeline.setProperty("badge", "muted")
        self.badge_pipeline.setToolTip(
            "当前工单的产线状态（运行/暂停/停止）；\n"
            "产线一律在「实时监控」页手动启停")
        lay.addWidget(self.badge_pipeline)

        # M15c：当前角色徽标（system/info.role，鉴权关闭时 admin）
        self.badge_role = QLabel("角色 -")
        self.badge_role.setProperty("badge", "info")
        self.badge_role.setToolTip("当前 API Key 角色：operator 操作员 / "
                                   "engineer 工程师 / admin 管理员")
        lay.addWidget(self.badge_role)

        self.lbl_budget = QLabel("延迟预算 <200ms")
        self.lbl_budget.setProperty("badge", "info")
        self.lbl_budget.setToolTip("单张图片检测的耗时上限：超过会标红提示，"
                                   "说明当前硬件下可考虑降配模式")
        lay.addWidget(self.lbl_budget)

        self.badge_conn = QLabel("● 后端未连接")
        self.badge_conn.setProperty("badge", "danger")
        lay.addWidget(self.badge_conn)
        return bar

    def _repolish(self, w: QWidget) -> None:
        w.style().unpolish(w)
        w.style().polish(w)

    # ══════════════════ 页面 ══════════════════
    def _build_pages(self) -> None:
        get_cat = self.current_category
        get_budget = lambda: self._budget_ms  # noqa: E731
        # 页面栈顺序与 NAV_ITEMS 一一对应（M16b 流程化重排）
        self.page_dashboard = DashboardPage(self.client, get_cat)
        self.page_data = DataPage(self.client, get_cat, self.task_monitor)
        self.page_augment = AugmentPage(self.client, get_cat, self.task_monitor)
        self.page_model = ModelPage(self.client, get_cat, self.task_monitor)
        self.page_eval = EvalPage(self.client, get_cat, get_budget,
                                  self.task_monitor)
        self.page_monitor = MonitorPage(self.client, get_cat, get_budget)
        self.page_feedback = FeedbackPage(self.client, get_cat, self.task_monitor)
        self.page_learning = LearningPage(self.client, get_cat,
                                          self.task_monitor)
        self.page_stats = StatsPage(self.client, get_cat, get_budget)
        self.page_settings = SettingsPage(self.client)
        for p in (self.page_dashboard, self.page_data, self.page_augment,
                  self.page_model, self.page_eval, self.page_monitor,
                  self.page_feedback, self.page_learning, self.page_stats,
                  self.page_settings):
            self.stack.addWidget(p)
            # U-workorder v2：各页数据条件徽标 = 当前工单条件
            p._get_level = self.current_workorder
            # 需要工单口径的页面（工作台/统计等）可取当前工单 dict/id
            p._get_workorder = self.current_workorder
            p._get_workorder_id = self.current_workorder_id
        # M16b/M16c：工作台与设置页"下一步/直达"按钮 -> 跳转导航页
        self.page_dashboard.goto_page.connect(self.goto)
        self.page_settings.goto_page.connect(self.goto)
        # 走查 2026-08-29：监控页空转引导「去准备未就绪品类」-> 模型管理
        self.page_monitor.goto_page.connect(self.goto)
        # U-workorder v2：工作台表格选中行 -> 切换当前工单
        self.page_dashboard.workorder_selected.connect(self.select_workorder)
        # 前端反馈 2026-09-13 #3：新建工单成功 -> 顶栏下拉自动选中新工单
        self.page_dashboard.workorder_created.connect(
            self.select_workorder_after_reload)
        # U-workorder v2：工单变更后刷新工单下拉（品类/徽标随之联动）
        self.page_settings.levels_changed.connect(self._load_workorders)
        self.page_dashboard.levels_changed.connect(self._load_workorders)
        # 走查 2026-08-29：设置保存后重拉 system_info——延迟预算等热更新
        # 到顶栏徽标与各页 get_budget 回调（此前只在首次连接时拉一次，
        # 设置页改完预算顶栏/看板全程不刷新）
        self.page_settings.config_saved.connect(
            lambda: run_async(self, self.client.system_info, self._on_sysinfo))
        # 模型页「学习曲线」按钮 -> 携带选中品类跳转（走查 2026-08-29 修复：
        # 多品类工单全局品类为空，不带品类学习页只显示空态）
        self.page_model.open_learning.connect(self._on_open_learning)

    def _on_open_learning(self, cat: str) -> None:
        """模型管理 -> 学习曲线：先定位品类（触发该品类数据加载），
        再切页（页面激活 reload 时下拉已选中该品类，同一品类幂等）。"""
        self.page_learning.show_category(cat)
        self.goto("学习效果")

    def goto(self, title: str) -> None:
        """按导航标题跳转页面（M16b 流程引导入口）。
        A14：被当前角色裁剪的页面拒绝跳转（原仅隐藏导航项，goto 可绕过）。"""
        if not role_allows_nav(getattr(self, "_role", "admin"), title):
            return
        row = self._title2row.get(title)
        if row is not None:
            self.nav.setCurrentRow(row)

    def _on_nav_changed(self, row: int) -> None:
        if row < 0 or row >= len(self._row2page):
            return
        page_idx = self._row2page[row]
        if page_idx is None:  # 分组标题行不可选，防御
            return
        # A14 页面级管控：被当前角色裁剪的页面即使 nav 行被程序化选中
        # 也拒绝切换（统一闸口，双保险于 goto/nav setHidden）
        title = NAV_ITEMS[page_idx][0] if page_idx < len(NAV_ITEMS) else ""
        if not role_allows_nav(getattr(self, "_role", "admin"), title):
            return
        self.stack.setCurrentIndex(page_idx)
        page = self.stack.currentWidget()
        reload_fn = getattr(page, "reload", None)
        if callable(reload_fn):
            try:
                if page is self.page_data:
                    reload_fn(self._categories)
                else:
                    reload_fn()
            except TypeError:
                reload_fn()

    # ══════════════════ 工单 / 品类（全局共享） ══════════════════
    def current_category(self) -> str:
        """当前工单的品类：单品类工单返回该品类；多品类/无工单返回空（全局）。
        （前端反馈 v3：移除顶栏品类下拉，品类随工单推导。）"""
        wo = self.current_workorder()
        if not wo:
            return ""
        cats = [str(c) for c in (wo.get("categories") or [])]
        return cats[0] if len(cats) == 1 else ""

    def current_workorder(self) -> dict | None:
        """当前选中工单 dict（无工单时 None）。"""
        wid = self.combo_workorder.currentData()
        if wid is None:
            return None
        for w in self._workorders:
            if w.get("id") == wid:
                return w
        return None

    def current_workorder_id(self) -> int | None:
        wo = self.current_workorder()
        return wo.get("id") if wo else None

    def _reload_current_page(self) -> None:
        page = self.stack.currentWidget()
        reload_fn = getattr(page, "reload", None)
        if callable(reload_fn):
            try:
                if page is self.page_data:
                    reload_fn(self._categories)
                else:
                    reload_fn()
            except TypeError:
                reload_fn()

    def _on_workorder_changed(self, _idx: int) -> None:
        """工单切换 -> 品类集合更新 + 条件徽标 + 产线徽标 + 当前页刷新。"""
        wo = self.current_workorder()
        if wo is None:
            self._load_categories()  # 无工单时回退全品类
        else:
            cats = [str(c) for c in (wo.get("categories") or [])]
            self._fill_categories(cats)
        update_level_badge(self.badge_level, self.current_workorder)
        self._set_pipeline_badge(
            str(wo.get("pipeline_status") or "") if wo else "")
        self._refresh_page_level_badges()
        self._reload_current_page()

    # ══════════════════ 顶栏产线状态徽标 ══════════════════
    _PIPELINE_BADGE = {
        "running": ("产线：运行中", "success"),
        "paused": ("产线：已暂停", "warning"),
        "auto_paused": ("产线：自动暂停（复判积压）", "warning"),
        "stopped": ("产线：已停止", "danger"),
    }

    def _set_pipeline_badge(self, status: str) -> None:
        text, badge = self._PIPELINE_BADGE.get(status, ("产线：-", "muted"))
        self.badge_pipeline.setText(text)
        self.badge_pipeline.setProperty("badge", badge)
        self._repolish(self.badge_pipeline)

    def _poll_pipeline_badge(self) -> None:
        """10s 轻量刷新产线徽标（监控页手动启停后顶栏同步）。"""
        wid = self.current_workorder_id()
        if wid is None:
            self._set_pipeline_badge("")
            return

        def _fill(wo) -> None:
            # 轮询返回时工单可能已切换：只在接受仍是同一工单时更新
            if isinstance(wo, dict) and wo.get("id") == self.current_workorder_id():
                self._set_pipeline_badge(str(wo.get("pipeline_status") or ""))
                # 缓存同步：监控页改了状态，下次 _on_workorder_changed 用新值
                for w in self._workorders:
                    if w.get("id") == wo.get("id"):
                        w["pipeline_status"] = wo.get("pipeline_status")
                        break

        run_async(self, lambda: self.client.get_workorder(wid), _fill,
                  lambda _e: None)

    def _refresh_page_level_badges(self) -> None:
        """刷新所有页面的数据条件徽标（页面存在 update_level_badge 时）。"""
        for i in range(self.stack.count()):
            page = self.stack.widget(i)
            fn = getattr(page, "update_level_badge", None)
            if callable(fn):
                fn()

    def _fill_categories(self, cats: list[str]) -> None:
        """更新当前品类集合（顶栏下拉已移除，仅维护列表并同步数据页）。"""
        self._categories = list(cats)
        # 数据页品类筛选同步
        self.page_data.reload(self._categories)

    def _load_workorders(self) -> None:
        """加载工单列表并刷新工单下拉（随后联动品类/徽标）。"""
        def _fill(items: list) -> None:
            items = [w for w in (items or []) if isinstance(w, dict)]
            # 新建工单后优先选中新工单（前端反馈 2026-09-13 #3：
            # 已有工单时此前保持旧选中，用户感受不到"自动选中"）
            pending = getattr(self, "_pending_select_wo", None)
            self._pending_select_wo = None
            prev = pending if pending is not None \
                else self.combo_workorder.currentData()
            self._workorders = items
            self.combo_workorder.blockSignals(True)
            self.combo_workorder.clear()
            if not items:
                self.combo_workorder.addItem("（未建工单）", None)
            else:
                for w in items:
                    self.combo_workorder.addItem(str(w.get("name") or "-"),
                                                 w.get("id"))
                idx = self.combo_workorder.findData(prev)
                self.combo_workorder.setCurrentIndex(idx if idx >= 0 else 0)
            self.combo_workorder.blockSignals(False)
            self._on_workorder_changed(0)
        run_async(self, self.client.list_workorders, _fill)

    def select_workorder(self, workorder_id: int) -> None:
        """外部（工作台表格）请求切换当前工单。"""
        idx = self.combo_workorder.findData(workorder_id)
        if idx >= 0:
            self.combo_workorder.setCurrentIndex(idx)

    def select_workorder_after_reload(self, workorder_id: int) -> None:
        """新建工单后：重拉列表并选中新工单（列表未到先挂起，_fill 兜底）。"""
        self._pending_select_wo = int(workorder_id)
        self._load_workorders()

    def _load_categories(self) -> None:
        """无工单场景的回退：全品类下拉。"""
        def _fill(items: list) -> None:
            items = [c for c in items if isinstance(c, dict)]
            cats = [str(c.get("category")) for c in items]
            self._fill_categories(cats)
            update_level_badge(self.badge_level, self.current_workorder)
            self._refresh_page_level_badges()
        run_async(self, self.client.categories_full, _fill)

    # ══════════════════ 连接状态 ══════════════════
    def _ping_backend(self) -> None:
        run_async(self, self.client.ping, self._on_ping, lambda _e: None)

    def _on_ping(self, health) -> None:
        if health and isinstance(health, dict) and health.get("status") == "ok":
            self.badge_conn.setText("● 后端已连接")
            self.badge_conn.setProperty("badge", "success")
            device = str(health.get("device", "-")).lower()
            gpu = str(health.get("gpu_name", "") or "")
            if device == "cuda":
                self.badge_device.setText(f"GPU {gpu[:18]}".strip())
                self.badge_device.setProperty("badge", "success")
            else:
                self.badge_device.setText(device.upper() or "CPU")
                self.badge_device.setProperty("badge", "muted")
            self._repolish(self.badge_device)
            # 首次连接成功后加载工单（品类随工单联动）与系统信息。
            # 工单为空也要只加载一次：否则每 3s ping 都会因 _workorders 为空
            # 重复 _load_workorders → _on_workorder_changed → 数据页 reload 循环
            # （"后台一直在滚动"的来源之一）。
            if not getattr(self, "_wo_fetched", False):
                self._wo_fetched = True
                self._load_workorders()
                run_async(self, self.client.system_info, self._on_sysinfo)
        else:
            self.badge_conn.setText("● 后端未连接")
            self.badge_conn.setProperty("badge", "danger")
        self._repolish(self.badge_conn)

    def _on_sysinfo(self, info) -> None:
        if not info:
            return
        budget = info.get("latency_budget_ms")
        if budget:
            self._budget_ms = float(budget)
            self.lbl_budget.setText(f"延迟预算 <{self._budget_ms:.0f}ms")
        self._apply_role(str(info.get("role") or "admin"))

    def _apply_role(self, role: str) -> None:
        """M15c：按角色裁剪导航（幂等）。"""
        if getattr(self, "_role", None) == role:
            return
        self._role = role
        self.badge_role.setText(f"角色 {ROLE_NAMES.get(role, role)}")
        self._repolish(self.badge_role)
        # A14：与 goto 共用 role_allows_nav 判定，避免两处口径漂移
        for title, _icon in NAV_ITEMS:
            row = self._title2row[title]
            self.nav.item(row).setHidden(not role_allows_nav(role, title))
        # 当前页被隐藏时退回第一可见项（跳过分组标题行）
        cur = self.nav.currentRow()
        if (0 <= cur < self.nav.count()
                and (self.nav.item(cur).isHidden()
                     or self._row2page[cur] is None)):
            for row in range(self.nav.count()):
                if (self._row2page[row] is not None
                        and not self.nav.item(row).isHidden()):
                    self.nav.setCurrentRow(row)
                    break

    # ══════════════════ 关闭清理 ══════════════════
    def closeEvent(self, event) -> None:  # noqa: N802
        self._ping_timer.stop()
        self._pipe_status_timer.stop()
        self.task_monitor.stop()
        super().closeEvent(event)
