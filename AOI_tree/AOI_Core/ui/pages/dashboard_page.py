"""工作台首页（W-workorder v2，前端反馈 2026-08-27）：工单为统计基础。

- KPI 行：检测数 / 不良数 / 不良率 / 待复核——按「当前工单 + 时间范围」
  （全部/近7天/今天）统计；未选工单时为全局口径并在副文本注明
- 待办卡片：待复核队列 / 不完备预警 / 待巩固反馈（点击直达处理页）
- 工单运营表：list_workorders(range) 驱动；单击行切换主窗口当前工单，
  双击「数据条件」列弹工单编辑（三档＋两开关＋体检），可新建工单
- 能力开关状态条：webhook / PLC 触发 / 自动归档 / API 鉴权 / API 文档
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ui.api_client import ApiClient
from ui.pages.common import (attach_level_badge, make_banner, make_card,
                             notify, run_async, warn)
from ui.theme import DANGER, PRIMARY, TEXT_SUB, WARNING
from ui.widgets.kpi_card import KpiCard
from ui.widgets.workorder_dialogs import (WorkOrderCreateDialog,
                                          WorkOrderEditDialog)

_RANGE_CN = {"all": "全部", "7d": "近 7 天", "today": "今天"}


class DashboardPage(QWidget):
    """工作台首页（工单口径）。"""

    goto_page = Signal(str)          # 请求主窗口跳转到指定导航页（标题）
    levels_changed = Signal()        # 工单变更 -> 主窗口刷新工单下拉
    workorder_selected = Signal(int) # 表格选中行 -> 主窗口切换当前工单
    workorder_created = Signal(int)  # 新建工单成功 -> 主窗口选中新工单

    def __init__(self, client: ApiClient, get_category=None,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self._client = client
        self._get_category = get_category   # 兼容旧注入（统计已按工单）
        self._wo_items: list[dict] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(8)

        top = QHBoxLayout()
        top.addWidget(make_banner(
            "工作台：以工单为统计基础（先建工单、挂数据源，再检测统计）"))
        attach_level_badge(self, top)   # 当前工单数据条件徽标
        top.addStretch(1)
        top.addWidget(QLabel("时间范围"))
        self.combo_range = QComboBox()
        for name, code in (("全部", "all"), ("近 7 天", "7d"), ("今天", "today")):
            self.combo_range.addItem(name, code)
        self.combo_range.setToolTip("KPI 与工单表统计的时间范围")
        self.combo_range.currentIndexChanged.connect(lambda _i: self.reload())
        top.addWidget(self.combo_range)
        btn_new = QPushButton("新建工单")
        btn_new.setProperty("primary", True)
        btn_new.clicked.connect(self._on_create_workorder)
        top.addWidget(btn_new)
        btn_refresh = QPushButton("刷新")
        btn_refresh.clicked.connect(self.reload)
        top.addWidget(btn_refresh)
        root.addLayout(top)

        # ── KPI 卡片行（当前工单 + 时间范围口径） ──
        kpi_row = QHBoxLayout()
        self.kpi_inspected = KpiCard("检测数", "-", sub_text="当前工单：-")
        self.kpi_anomaly = KpiCard("不良数", "-", accent=DANGER,
                                   sub_text="系统判定口径（异常 + 灰区判异常）")
        self.kpi_rate = KpiCard("不良率", "-", accent=WARNING,
                                sub_text="系统判定口径，含未复核/重复检测；"
                                         "复核后真实口径见统计报表")
        self.kpi_pending = KpiCard("待复核", "-", accent=WARNING,
                                   sub_text="当前工单灰区待人工裁定")
        for k in (self.kpi_inspected, self.kpi_anomaly,
                  self.kpi_rate, self.kpi_pending):
            kpi_row.addWidget(k)
        root.addLayout(kpi_row)

        # ── 待办卡片行 ──
        todo_row = QHBoxLayout()
        self.todo_review = self._make_todo_card(
            "待复核队列", "实时监控灰区帧等待人工裁定", "去复核", "标注反馈")
        self.todo_incomplete = self._make_todo_card(
            "不完备预警", "体系外缺陷信号持续出现的品类", "查看", "标注反馈")
        self.todo_learn = self._make_todo_card(
            "待巩固反馈", "已反馈但未巩固进模型的条目", "去巩固", "模型管理")
        for c in (self.todo_review, self.todo_incomplete, self.todo_learn):
            todo_row.addWidget(c[0])
        root.addLayout(todo_row)

        # ── 工单运营表 ──
        st_card = make_card()
        st_lay = QVBoxLayout(st_card)
        st_head = QLabel(
            "工单运营状态（单击行切换当前工单；双击「数据条件」可调整工单）")
        st_head.setProperty("heading", True)
        st_lay.addWidget(st_head)
        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels(
            ["工单", "数据条件", "数据源", "品类", "图片数",
             "体检", "检测", "不良", "不良率"])
        h1 = self.table.horizontalHeaderItem(1)
        if h1 is not None:
            h1.setToolTip("工单声明的数据条件（三档＋两开关，决定检测方式）：\n"
                          "标注档位：仅正常图 / 图像级标注 / 缺陷位置标注\n"
                          "叠加开关：品类独立 / 模板比对\n双击此列单元格可直接修改")
        h5 = self.table.horizontalHeaderItem(5)
        if h5 is not None:
            h5.setToolTip("声明条件 vs 数据实际支撑的体检结果；"
                          "有提醒时橙色显示，悬停查看明细")
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.cellClicked.connect(self._on_cell_clicked)
        self.table.cellDoubleClicked.connect(self._on_cell_double_clicked)
        st_lay.addWidget(self.table, 1)
        # 工单增删改查（前端反馈 2026-09-13 #1）：增=右上「新建工单」；
        # 改=双击「数据条件」或此处按钮；删=删除所选（检测记录保留）
        op_row = QHBoxLayout()
        op_row.addStretch(1)
        btn_edit = QPushButton("编辑所选工单")
        btn_edit.setToolTip("修改所选工单的名称/备注/复判开关/挂接数据源")
        btn_edit.clicked.connect(self._on_edit_selected)
        op_row.addWidget(btn_edit)
        btn_del = QPushButton("删除所选工单")
        btn_del.setProperty("danger", True)
        btn_del.setToolTip("删除所选工单（仅删工单与挂接关系，"
                           "数据源/图片/检测记录保留）")
        btn_del.clicked.connect(self._on_delete_selected)
        op_row.addWidget(btn_del)
        st_lay.addLayout(op_row)
        root.addWidget(st_card, 1)

        # ── 能力开关状态条 ──
        feat_card = make_card()
        f_lay = QVBoxLayout(feat_card)
        f_head = QLabel("产线对接能力状态（仅展示，在设置中心/配置文件修改）")
        f_head.setProperty("heading", True)
        f_lay.addWidget(f_head)
        f_row = QHBoxLayout()
        self._feature_badges: dict[str, QLabel] = {}
        for key, name, tip in (
                ("webhook", "Webhook 告警",
                 "检测/告警事件推送到 MES/上位机。在设置中心「Webhook 地址」配置，空=禁用"),
                ("plc_trigger", "PLC 触发",
                 "传统 AOI/光电信号落图目录触发检测。在设置中心「PLC 触发目录」配置，空=禁用"),
                ("auto_archive", "自动归档",
                 "命中决策的缺陷帧自动归档留存。在设置中心「缺陷自动归档」开关"),
                ("auth", "API 鉴权",
                 "外部接口访问控制。安全密钥属高危项，需手改配置文件 security 段")):
            lb = QLabel(f"{name} -")
            lb.setProperty("badge", "muted")
            lb.setToolTip(tip)
            self._feature_badges[key] = lb
            f_row.addWidget(lb)
        f_row.addStretch(1)
        btn_goto_settings = QPushButton("去设置中心")
        btn_goto_settings.clicked.connect(
            lambda: self.goto_page.emit("系统设置"))
        f_row.addWidget(btn_goto_settings)
        f_lay.addLayout(f_row)
        root.addWidget(feat_card)

        self._client.error_occurred.connect(lambda m: warn(self, m))

    # ══════════════════ 待办卡片 ══════════════════
    def _make_todo_card(self, title: str, desc: str, btn_text: str,
                        page: str) -> tuple:
        """返回 (card, value_label)。卡片含数量大字 + 描述 + 直达按钮。"""
        card = make_card()
        lay = QVBoxLayout(card)
        head = QLabel(title)
        head.setProperty("heading", True)
        lay.addWidget(head)
        val = QLabel("-")
        f = val.font()
        f.setPointSizeF(20)
        f.setBold(True)
        val.setFont(f)
        lay.addWidget(val)
        d = QLabel(desc)
        d.setProperty("subtext", True)
        lay.addWidget(d)
        btn = QPushButton(btn_text)
        btn.clicked.connect(lambda _=False, p=page: self.goto_page.emit(p))
        lay.addWidget(btn, 0, Qt.AlignRight)
        return card, val

    # ══════════════════ 当前上下文 ══════════════════
    def _range_name(self) -> str:
        return str(self.combo_range.currentData() or "all")

    def _current_wo(self) -> dict | None:
        fn = getattr(self, "_get_workorder", None)
        return fn() if callable(fn) else None

    def _current_wo_id(self) -> int | None:
        fn = getattr(self, "_get_workorder_id", None)
        return fn() if callable(fn) else None

    def _scope_text(self) -> str:
        wo = self._current_wo()
        scope = f"当前工单：{wo.get('name')}" if wo else "未选工单（全局口径）"
        return f"{scope}｜{_RANGE_CN.get(self._range_name(), '')}"

    # ══════════════════ 数据加载 ══════════════════
    def reload(self) -> None:
        range_name = self._range_name()
        wid = self._current_wo_id()
        run_async(self, lambda: self._client.stats_overview(
            workorder_id=wid, range_name=range_name), self._fill_kpi)
        run_async(self, self._client.review_stats, self._fill_review)
        run_async(self, lambda: self._client.list_feedback(
            consumed=False, page=1, page_size=1), self._fill_unconsumed)
        run_async(self, self._client.incomplete_report,
                  self._fill_incomplete)
        run_async(self, lambda: self._client.list_workorders(range_name),
                  self._fill_workorders)
        run_async(self, self._client.system_info, self._fill_features)

    def _fill_kpi(self, data) -> None:
        if not data:
            return
        scope = data.get("scope_stats") or {}
        n_ins = int(scope.get("n_inspected", 0) or 0)
        n_an = int(scope.get("n_anomaly", 0) or 0)
        rate = float(scope.get("rate", 0.0) or 0.0)
        ctx = self._scope_text()
        self.kpi_inspected.set_value(str(n_ins), ctx)
        self.kpi_anomaly.set_value(str(n_an), ctx)
        self.kpi_rate.set_value(
            f"{rate * 100:.1f}%",
            f"{n_an} / {n_ins}" if n_ins else "该范围暂无检测")

    def _fill_review(self, data) -> None:
        if not data:
            return
        # 待复核按当前工单品类集合计（无工单时为全局）
        by_cat = data.get("pending_by_category") or {}
        wo = self._current_wo()
        if wo:
            cats = set(str(c) for c in (wo.get("categories") or []))
            n = sum(int(v or 0) for c, v in by_cat.items() if str(c) in cats)
        else:
            n = int(data.get("pending_total", 0) or 0)
        self.kpi_pending.set_value(str(n), self._scope_text())
        total = int(data.get("pending_total", 0) or 0)
        self.todo_review[1].setText(str(total))
        self.todo_review[1].setStyleSheet(
            f"color: {WARNING if total else TEXT_SUB};")

    def _fill_unconsumed(self, data) -> None:
        if not data:
            return
        n = int(data.get("total", 0) or 0)
        self.todo_learn[1].setText(str(n))
        self.todo_learn[1].setStyleSheet(
            f"color: {PRIMARY if n else TEXT_SUB};")

    def _fill_incomplete(self, data) -> None:
        if not data:
            return
        warns = [r for r in (data.get("reports") or []) if r.get("warn")]
        self.todo_incomplete[1].setText(str(len(warns)))
        color = DANGER if warns else TEXT_SUB
        self.todo_incomplete[1].setStyleSheet(f"color: {color};")
        if warns:
            cats = "、".join(str(r.get("category", "")) for r in warns)
            self.todo_incomplete[1].setToolTip(f"预警品类：{cats}")

    # ══════════════════ 工单表 ══════════════════
    def _fill_workorders(self, items) -> None:
        items = [w for w in (items or []) if isinstance(w, dict)]
        self._wo_items = items
        cur_wid = self._current_wo_id()
        if not items:
            self.table.clearSpans()
            self.table.setRowCount(1)
            self.table.setSpan(0, 0, 1, 9)
            self.table.setItem(0, 0, QTableWidgetItem(
                "暂无工单：点击右上角「新建工单」开始"
                "（先建工单并挂接数据源，统计以工单为基础）"))
            return
        self.table.clearSpans()
        self.table.setRowCount(len(items))
        for r, wo in enumerate(items):
            stats = wo.get("stats") or {}
            check = wo.get("check") or {}
            warns = check.get("warnings") or []
            srcs = "、".join(str(d.get("name"))
                             for d in (wo.get("datasources") or [])) or "-"
            cats = "、".join(str(c) for c in (wo.get("categories") or [])) or "-"
            n_ins = int(stats.get("inspected", 0) or 0)
            n_an = int(stats.get("anomaly", 0) or 0)
            rate = float(stats.get("rate", 0.0) or 0.0)
            vals = [str(wo.get("name") or "-"),
                    str(wo.get("conditions_cn") or "-"),
                    srcs, cats,
                    str(int(wo.get("n_images", 0) or 0)),
                    "正常" if check.get("matched") else f"{len(warns)} 项提醒",
                    str(n_ins), str(n_an),
                    f"{rate * 100:.1f}%" if n_ins else "-"]
            for c, val in enumerate(vals):
                cell = QTableWidgetItem(val)
                cell.setData(Qt.UserRole, wo.get("id"))
                if c == 0 and wo.get("id") == cur_wid:
                    f = cell.font()
                    f.setBold(True)
                    cell.setFont(f)
                    cell.setForeground(QColor(PRIMARY))
                    cell.setToolTip("当前工单（单击行可切换）")
                elif c == 0:
                    cell.setToolTip("单击切换为当前工单")
                if c == 1:
                    cell.setToolTip("双击修改该工单的数据条件（三档＋两开关）")
                if c == 5 and not check.get("matched"):
                    cell.setForeground(QColor(WARNING))
                    cell.setToolTip("体检提醒：\n" + "\n".join(
                        str(w) for w in warns[:5]))
                self.table.setItem(r, c, cell)
        self.table.resizeColumnsToContents()

    def _on_cell_clicked(self, row: int, _col: int) -> None:
        """单击行 -> 切换主窗口当前工单（统计随之联动）。"""
        item = self.table.item(row, 0)
        wid = item.data(Qt.UserRole) if item else None
        if wid is not None:
            self.workorder_selected.emit(int(wid))

    def _on_cell_double_clicked(self, row: int, col: int) -> None:
        """双击「数据条件」列 -> 弹窗修改该工单（三档＋两开关＋体检）。"""
        if col != 1 or row >= len(self._wo_items):
            return
        dlg = WorkOrderEditDialog(self._client, self._wo_items[row], self)
        dlg.levels_changed.connect(self.reload)
        dlg.levels_changed.connect(self.levels_changed.emit)
        dlg.exec()

    def _on_create_workorder(self) -> None:
        dlg = WorkOrderCreateDialog(self._client, self)
        dlg.levels_changed.connect(self.reload)
        dlg.levels_changed.connect(self.levels_changed.emit)
        dlg.created.connect(self.workorder_created.emit)
        dlg.exec()

    def _selected_wo(self) -> dict | None:
        """表格当前行对应的工单 dict。"""
        row = self.table.currentRow()
        item = self.table.item(row, 0) if row >= 0 else None
        wid = item.data(Qt.UserRole) if item is not None else None
        if wid is None:
            return None
        return next((w for w in self._wo_items if w.get("id") == wid), None)

    def _on_edit_selected(self) -> None:
        wo = self._selected_wo()
        if wo is None:
            warn(self, "请先单击选中要编辑的工单行")
            return
        dlg = WorkOrderEditDialog(self._client, wo, self)
        dlg.levels_changed.connect(self.reload)
        dlg.levels_changed.connect(self.levels_changed.emit)
        dlg.exec()

    def _on_delete_selected(self) -> None:
        """删除所选工单（后端仅删工单与挂接关系，检测/数据保留）。"""
        wo = self._selected_wo()
        if wo is None:
            warn(self, "请先单击选中要删除的工单行")
            return
        wid = int(wo.get("id"))
        if QMessageBox.question(
                self, "删除工单",
                f"确定删除工单「{wo.get('name')}」？\n"
                "仅删除工单及其数据源挂接关系；数据源、图片、"
                "检测记录与模型均保留。") != QMessageBox.Yes:
            return

        def _done(res) -> None:
            if isinstance(res, dict) and res.get("deleted"):
                notify(self, f"工单已删除：{wo.get('name')}")
                self.reload()
                self.levels_changed.emit()
            else:
                warn(self, "删除失败，请重试")

        run_async(self, lambda: self._client.delete_workorder(wid), _done)

    # ══════════════════ 能力开关 ══════════════════
    def _fill_features(self, info) -> None:
        if not info:
            return
        feats = info.get("features") or {}
        names = {"webhook": "Webhook 告警", "plc_trigger": "PLC 触发",
                 "auto_archive": "自动归档", "auth": "API 鉴权"}
        for key, lb in self._feature_badges.items():
            on = bool(feats.get(key))
            lb.setText(f"{names[key]} {'开' if on else '关'}")
            lb.setProperty("badge", "success" if on else "muted")
            lb.style().unpolish(lb)
            lb.style().polish(lb)
