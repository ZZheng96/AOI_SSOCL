"""模型管理页：按「数据源 → 品类」组织模型状态（与数据管理页结构镜像）。

前端反馈 2026-08-29 重构要点：
- 删除引擎架构示意图（原图把三槽位并行画成串联，有误导且不可交互）、
  "分数校准"卡（功能已下线只剩说明文字）、"扫描模型"按钮（运维操作）；
- 清除用户困惑信息：内部代号 / 槽位 / 问题三 / CDF 校准等术语一律不出现；
- 结构镜像数据管理页：数据源 → 品类 → 模型状态；有当前工单时只列工单
  挂接的数据源，品类节点直接显示「当前版本 · 上线阶段」；
- 同品类不罗列全部版本：版本历史折叠收纳（激活/回滚/A-B 对比在展开区内）；
- 准备模型接数据源「预训练组」（datasource_id），数据条件由工单/体检
  自动判定，不再让用户手选 L0-L3 代号；
- 品类上线状态机（readiness：数据未导入→可训练→已训练→已验收→已投产）
  作为页面主线，引导用户下一步动作。
"""
from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
    QFormLayout, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit,
    QPushButton, QRadioButton, QSplitter, QStackedWidget, QTableWidget,
    QTableWidgetItem, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from ..api_client import ApiClient, ApiWorker
from ..widgets.task_progress import TaskMonitor, TaskProgressBar
from .common import (LAYER_CN, attach_level_badge, make_banner, make_card,
                     notify, run_async, warn)


# ══════════════════ 激活质量门控结果对话框 ══════════════════

class ActivationGateDialog(QDialog):
    """激活前自动验证结果展示：通过/拒绝时列出当前版本与新版本对比数据，
    用户可取消或强制激活。"""

    def __init__(self, category: str, passed: bool, gate_report: dict,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self._force = False
        self.setWindowTitle("激活前自动验证")
        self.setMinimumWidth(520)
        lay = QVBoxLayout(self)
        pass_txt = ("验证通过：新版本在验证图上的表现不劣于当前版本，已激活"
                    if passed else
                    "已阻止激活：新版本在验证图上的准确率明显低于当前版本")
        title = QLabel(pass_txt)
        title.setProperty("heading", True)
        title.setWordWrap(True)
        lay.addWidget(title)
        checks = (gate_report or {}).get("checks") or []
        if checks:
            tbl = QTableWidget(len(checks), 5)
            tbl.setHorizontalHeaderLabels(
                ["检查项", "当前版本", "新版本", "差距", "结论"])
            tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            tbl.verticalHeader().setVisible(False)
            tbl.setMaximumHeight(min(200, 40 + 28 * len(checks)))
            for r, c in enumerate(checks):
                vals = [c.get("metric", ""), c.get("current", ""),
                        c.get("candidate", ""), c.get("delta", ""),
                        "通过" if c.get("pass") else "未通过"]
                for col, v in enumerate(vals):
                    tbl.setItem(r, col, QTableWidgetItem(str(v)))
            lay.addWidget(tbl)
        msg = str((gate_report or {}).get("message") or "")
        if msg:
            ml = QLabel(msg)
            ml.setWordWrap(True)
            ml.setProperty("subtext", True)
            lay.addWidget(ml)
        if passed:
            bb = QDialogButtonBox(QDialogButtonBox.Ok)
            bb.accepted.connect(self.accept)
        else:
            bb = QDialogButtonBox(QDialogButtonBox.Cancel)
            force_btn = bb.addButton(
                "仍要强制激活", QDialogButtonBox.DestructiveRole)
            force_btn.clicked.connect(self._on_force)
            bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _on_force(self) -> None:
        self._force = True
        self.accept()

    @property
    def force_requested(self) -> bool:
        return self._force


def handle_activation_result(page: QWidget, client: ApiClient, category: str,
                             res: dict | None, on_reload) -> bool:
    """统一处理激活响应：门控通过/拒绝弹对比对话框；409 支持强制激活。
    返回 True 表示已处理（含弹窗），False 表示请求失败（错误信号已弹出）。"""
    if res is None:
        return False
    if isinstance(res, dict) and res.get("_status") == 409:
        detail = res.get("detail")
        report = detail.get("gate_report") if isinstance(detail, dict) else {}
        dlg = ActivationGateDialog(category, False, report or {}, page)
        if dlg.exec() and dlg.force_requested:
            run_async(
                page,
                lambda: client.activate_model(
                    res.get("_model_id", 0), force=True),
                on_reload)
        return True
    if isinstance(res, dict) and res.get("active_version") is not None:
        report = res.get("gate_report") or {}
        if report:
            ActivationGateDialog(category, True, report, page).exec()
        else:
            notify(page, "激活成功", f"已激活 v{res.get('active_version')}")
        on_reload(res)
        return True
    return False


# ══════════════════ 模型管理页 ══════════════════

ORIGIN_NAMES = {
    "engine_fit": "初始准备",
    "self_learned_engine": "在线学习",
    "consolidate": "反馈学习",
    "uploaded": "导入",
}

# 品类上线阶段 → 徽标配色（文字色, 底色）
_STAGE_STYLE = {
    "active":       ("#1a7f37", "#e6f4ea"),
    "validated":    ("#0969da", "#ddf4ff"),
    "trained":      ("#9a6700", "#fff8c5"),
    "ready":        ("#9a6700", "#fff8c5"),
    "insufficient": ("#57606a", "#f6f8fa"),
    "empty":        ("#57606a", "#f6f8fa"),
}


def _version_num(m: dict) -> int | None:
    """版本号字符串 'v3' → 3；解析失败返回 None。"""
    try:
        return int(str(m.get("version") or "").lstrip("vV"))
    except (TypeError, ValueError):
        return None


# 后端数据体检警告中的内部术语 → 用户语言（前端反馈 2026-08-29：
# 用户界面不出现 CDF/判别头/槽位/条件代号等困惑信息）
_WARN_JARGON = [
    ("CDF 校准粒度较粗，AUROC 可能系统性低估约 0.1（M9 验收实测）；",
     "样本较少时评估分数会偏低、不稳定；"),
    ("CDF 校准", "分数校准"),
    ("disc/shead 判别头", "判别模块"),
    ("判别头", "判别模块"),
    ("tpl 模板差分槽位", "模板比对"),
    ("L0 零样本", "仅正常图模式"),
    ("L1a", "图像级标注"),
    ("L1b", "缺陷位置标注"),
    ("L3", "模板比对"),
]


def _user_warning(w) -> str:
    s = str(w)
    for src, dst in _WARN_JARGON:
        s = s.replace(src, dst)
    return s


class ModelPage(QWidget):
    """模型管理页：左侧「数据源→品类」导航树（当前工单口径），
    右侧品类模型面板（状态头 + 当前状态 + 主操作 + 折叠版本历史）。"""

    # 走查 2026-08-29 修复：携带当前选中品类，避免跳转后学习页品类断链
    open_learning = Signal(str)

    def __init__(self, client: ApiClient, get_category,
                 task_monitor: TaskMonitor, parent: QWidget | None = None):
        super().__init__(parent)
        self._client = client
        self._get_category = get_category
        self._monitor = task_monitor
        self._get_workorder = lambda: None   # main_window 启动后注入
        self._models: list[dict] = []
        self._readiness: dict[str, dict] = {}
        self._sl_by_cat: dict[str, dict] = {}
        self._cur: dict | None = None        # 当前选中品类上下文
        self._model_buttons: dict[int, QRadioButton] = {}
        self._sl_running = False
        self._rollback_target_ver: int | None = None
        self._learn_task_id: str | None = None
        if self._monitor is not None:
            self._monitor.task_updated.connect(self._on_task)
        # B3 修复（2026-08-30 首用质检）：激活/回滚等请求失败必须可见——
        # 其余页面均已接 error_occurred，本页遗漏导致激活 400 完全静默
        self._client.error_occurred.connect(lambda m: warn(self, m))

        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(make_banner(
            "模型管理：按数据源组织品类模型，准备 → 验收 → 激活 → 反馈学习"))
        attach_level_badge(self, top)
        lay.addLayout(top)
        tip = QLabel("左侧选择「数据源 → 品类」查看模型状态；版本历史默认折叠")
        tip.setProperty("subtext", True)
        lay.addWidget(tip)

        splitter = QSplitter(Qt.Horizontal)
        lay.addWidget(splitter, 1)

        # ── 左：数据源 → 品类 导航树 ──
        nav_card = make_card()
        nav_lay = QVBoxLayout(nav_card)
        nav_lay.addWidget(self._mk_head("数据源与品类"))
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.itemSelectionChanged.connect(self._on_tree_select)
        nav_lay.addWidget(self.tree, 1)
        splitter.addWidget(nav_card)

        # ── 右：品类模型面板（空态 / 面板）──
        self._stack = QStackedWidget()
        empty = QLabel("← 请在左侧选择数据源下的品类")
        empty.setAlignment(Qt.AlignCenter)
        empty.setProperty("subtext", True)
        self._stack.addWidget(empty)
        self._stack.addWidget(self._build_panel())
        splitter.addWidget(self._stack)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([260, 780])

    # ── 右侧品类面板 ─────────────────────────────
    def _build_panel(self) -> QWidget:
        panel = QWidget()
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)

        # 状态头：品类名 + 阶段徽标 + 下一步提示
        head = make_card()
        hl = QVBoxLayout(head)
        row = QHBoxLayout()
        self.lbl_cat = QLabel("-")
        self.lbl_cat.setProperty("heading", True)
        row.addWidget(self.lbl_cat)
        self.lbl_stage = QLabel("")
        self.lbl_stage.setVisible(False)
        row.addWidget(self.lbl_stage)
        row.addStretch(1)
        hl.addLayout(row)
        self.lbl_next = QLabel("")
        self.lbl_next.setWordWrap(True)
        self.lbl_next.setProperty("subtext", True)
        hl.addWidget(self.lbl_next)
        lay.addWidget(head)

        # 当前状态卡
        status = make_card()
        st = QFormLayout(status)
        st.addRow(self._mk_head("当前状态"))
        self.lbl_version = QLabel("—")
        st.addRow("当前版本：", self.lbl_version)
        self.lbl_auroc = QLabel("—")
        st.addRow("历史最佳验证 AUC：", self.lbl_auroc)
        self.lbl_pending = QLabel("—")
        st.addRow("待学习反馈：", self.lbl_pending)
        self.lbl_source = QLabel("—")
        self.lbl_source.setWordWrap(True)
        st.addRow("数据来源：", self.lbl_source)
        lay.addWidget(status)

        # 操作卡
        ops = make_card()
        op = QVBoxLayout(ops)
        op.addWidget(self._mk_head("操作"))
        self.lbl_op_hint = QLabel("")
        self.lbl_op_hint.setWordWrap(True)
        self.lbl_op_hint.setProperty("subtext", True)
        op.addWidget(self.lbl_op_hint)
        btn_row = QHBoxLayout()
        self.btn_prepare = QPushButton("准备模型")
        self.btn_prepare.setProperty("primary", True)
        self.btn_prepare.clicked.connect(self._on_prepare)
        btn_row.addWidget(self.btn_prepare)
        btn_row.addSpacing(12)
        btn_row.addWidget(QLabel("备注："))
        self.edit_note = QLineEdit()
        self.edit_note.setPlaceholderText(
            "本次学习的说明（可选），如：换型后首批反馈")
        btn_row.addWidget(self.edit_note, 1)
        self.btn_learn = QPushButton("学习反馈并更新模型")
        self.btn_learn.setProperty("primary", True)
        self.btn_learn.clicked.connect(self._on_self_learning)
        btn_row.addWidget(self.btn_learn)
        self.btn_curve = QPushButton("学习效果")
        self.btn_curve.setToolTip("查看该品类的学习效果与引擎内部状态"
                                  "（跳转学习效果页并定位品类）")
        self.btn_curve.clicked.connect(self._emit_open_learning)
        btn_row.addWidget(self.btn_curve)
        # 快照导出/导入（跨机器/跨品类复用）：导出当前激活版本为 zip；
        # 导入 zip 到指定品类（不自动激活，需在版本历史中显式激活）
        self.btn_export = QPushButton("导出当前版本")
        self.btn_export.setToolTip("把当前激活版本打包为 zip 下载，"
                                   "可迁移到其他电脑或其他品类复用")
        self.btn_export.clicked.connect(self._on_export)
        btn_row.addWidget(self.btn_export)
        self.btn_import = QPushButton("导入快照")
        self.btn_import.setToolTip("从 zip 导入模型快照到指定品类"
                                   "（导入后不自动激活，需验收后手动激活）")
        self.btn_import.clicked.connect(self._on_import)
        btn_row.addWidget(self.btn_import)
        btn_row.addStretch(1)
        op.addLayout(btn_row)
        self.bar_learn = TaskProgressBar()
        self.bar_learn.setVisible(False)
        op.addWidget(self.bar_learn)
        lay.addWidget(ops)

        # 版本历史卡（默认折叠）
        hist = make_card()
        hs = QVBoxLayout(hist)
        hrow = QHBoxLayout()
        self.btn_hist_toggle = QPushButton("▸ 版本历史")
        self.btn_hist_toggle.setFlat(True)
        self.btn_hist_toggle.clicked.connect(self._toggle_history)
        hrow.addWidget(self.btn_hist_toggle)
        hrow.addStretch(1)
        self.btn_activate = QPushButton("激活选中版本")
        self.btn_activate.setEnabled(False)
        self.btn_activate.setToolTip("将选中的历史版本激活为线上版本"
                                     "（激活前自动验证）")
        self.btn_activate.clicked.connect(self._on_activate)
        hrow.addWidget(self.btn_activate)
        self.btn_rollback = QPushButton("回滚")
        self.btn_rollback.setEnabled(False)
        self.btn_rollback.clicked.connect(self._on_rollback)
        hrow.addWidget(self.btn_rollback)
        hs.addLayout(hrow)
        self.hist_body = QWidget()
        hb = QVBoxLayout(self.hist_body)
        hb.setContentsMargins(0, 0, 0, 0)
        self.tbl = QTableWidget(0, 5)
        self.tbl.setHorizontalHeaderLabels(
            ["选择", "版本", "来源", "验证指标", "操作"])
        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl.verticalHeader().setVisible(False)
        hb.addWidget(self.tbl)
        self.hist_body.setVisible(False)
        hs.addWidget(self.hist_body)
        lay.addWidget(hist)
        lay.addStretch(1)
        return panel

    @staticmethod
    def _mk_head(text: str) -> QLabel:
        lb = QLabel(text)
        lb.setProperty("heading", True)
        return lb

    # ── 数据加载 ─────────────────────────────────
    def update_level_badge(self) -> None:
        """顶栏工单切换时同步数据条件徽标。"""
        from .common import update_level_badge as _upd
        _upd(self.badge_level, self._get_workorder)

    def reload(self) -> None:
        self._build_tree()
        run_async(self, self._client.models_readiness, self._fill_readiness)
        run_async(self, self._client.list_models, self._fill_models)
        run_async(self, self._client.self_learning_status, self._fill_sl)

    def _emit_open_learning(self) -> None:
        """「学习曲线」按钮：带当前选中品类跳转（多品类工单全局品类为空，
        不带品类则学习页只能显示空态）。"""
        cat = str((self._cur or {}).get("category")
                  or (self._get_category() if self._get_category else "")
                  or "")
        self.open_learning.emit(cat)

    # ── 左侧树：数据源 → 品类 ────────────────────
    def _build_tree(self) -> None:
        """树骨架：当前工单的数据源 →（异步）品类节点；无工单列全部数据源。"""
        self.tree.blockSignals(True)
        self.tree.clear()
        self.tree.blockSignals(False)
        wo = self._get_workorder() if callable(self._get_workorder) else None
        if wo is not None:
            srcs = wo.get("datasources") or []
            if not srcs:
                ph = QTreeWidgetItem(["当前工单未挂数据源，请到「工单管理」挂接"])
                ph.setFlags(Qt.NoItemFlags)
                self.tree.addTopLevelItem(ph)
                return
            self._add_source_nodes(srcs)
        else:
            run_async(self, self._client.list_datasources,
                      self._fill_all_sources)

    def _fill_all_sources(self, items: list) -> None:
        self.tree.blockSignals(True)
        self.tree.clear()
        self.tree.blockSignals(False)
        self._add_source_nodes(
            [{"id": s.get("id"), "name": s.get("name")}
             for s in (items or [])])

    def _add_source_nodes(self, srcs: list[dict]) -> None:
        for src in srcs:
            node = QTreeWidgetItem([str(src.get("name") or "-")])
            node.setData(0, Qt.UserRole,
                         ("source", src.get("id"), str(src.get("name") or "")))
            loading = QTreeWidgetItem(["加载中…"])
            loading.setFlags(Qt.NoItemFlags)
            node.addChild(loading)
            self.tree.addTopLevelItem(node)
            node.setExpanded(True)
            run_async(self,
                      lambda sid=src.get("id"):
                          self._client.datasource_groups(sid),
                      lambda data, n=node, s=src:
                          self._fill_source_groups(data, n, s))

    def _fill_source_groups(self, data, node: QTreeWidgetItem,
                            src: dict) -> None:
        """填品类子节点（竞态防护：树重建后旧节点已销毁则丢弃）。"""
        try:
            if node is None or node.treeWidget() is None:
                return
        except RuntimeError:
            return
        node.takeChildren()   # 移除“加载中”
        cats = (data or {}).get("categories") or {}
        if not cats:
            ph = QTreeWidgetItem(["（暂无数据，请到「数据管理」导入）"])
            ph.setFlags(Qt.NoItemFlags)
            node.addChild(ph)
            return
        total = 0
        for cat in sorted(cats, key=str):
            g = cats[cat] or {}
            item = QTreeWidgetItem([f"{cat}（{self._node_status(cat)}）"])
            item.setData(0, Qt.UserRole,
                         ("category", src.get("id"),
                          str(src.get("name") or ""), str(cat)))
            item.setData(0, Qt.UserRole + 1, g)
            node.addChild(item)
            total += int(g.get("total", 0) or 0)
        node.setText(0, f"{src.get('name')}（{total} 图）")

    def _node_status(self, cat: str) -> str:
        """树品类节点的状态短句：'v3 · 已投产' 或阶段名。"""
        it = self._readiness.get(cat) or {}
        av = it.get("active_version")
        if av is not None:
            ver = str(av)
            if not ver.lower().startswith("v"):
                ver = f"v{ver}"
            return f"{ver} · {it.get('stage_cn', '已投产')}"
        return str(it.get("stage_cn") or "未准备")

    def _refresh_tree_status(self) -> None:
        """readiness/models 到达后刷新品类节点状态文本。"""
        root = self.tree.invisibleRootItem()
        for i in range(root.childCount()):
            src = root.child(i)
            for j in range(src.childCount()):
                item = src.child(j)
                data = item.data(0, Qt.UserRole)
                if isinstance(data, tuple) and data[0] == "category":
                    item.setText(0, f"{data[3]}（{self._node_status(data[3])}）")

    # ── 异步数据回调 ─────────────────────────────
    def _fill_readiness(self, data: dict | None) -> None:
        items = (data or {}).get("items") or []
        self._readiness = {str(i.get("category")): i
                           for i in items if i.get("category")}
        self._refresh_tree_status()
        self._refresh_panel()

    def _fill_models(self, items: list | None) -> None:
        self._models = [m for m in (items or [])
                        if isinstance(m, dict) and not self._is_demo_legacy(m)]
        self._refresh_tree_status()
        self._refresh_panel()

    @staticmethod
    def _is_demo_legacy(m: dict) -> bool:
        """过滤旧引擎遗留（历史数据未清理前的显示兜底，不参与版本号排序）。"""
        try:
            ver = int(m.get("version") or 0)
        except (TypeError, ValueError):
            ver = 0
        if ver < 100:
            return False
        metrics = m.get("metrics")
        if isinstance(metrics, str):
            try:
                import json as _json
                metrics = _json.loads(metrics or "{}")
            except Exception:  # noqa: BLE001
                metrics = {}
        metrics = metrics if isinstance(metrics, dict) else {}
        return ver >= 100 and metrics.get("engine") in (None, "", "demo4")

    def _fill_sl(self, data: dict | None) -> None:
        cats = (data or {}).get("categories") or []
        self._sl_by_cat = {str(c.get("category")): c
                           for c in cats if c.get("category")}
        self._refresh_panel()

    # ── 品类选择与面板渲染 ───────────────────────
    def _on_tree_select(self) -> None:
        item = self.tree.currentItem()
        data = item.data(0, Qt.UserRole) if item is not None else None
        if isinstance(data, tuple) and len(data) == 4 \
                and data[0] == "category":
            g = item.data(0, Qt.UserRole + 1) or {}
            self._cur = {"category": data[3], "datasource_id": data[1],
                         "datasource_name": data[2], "groups": g}
            self._stack.setCurrentIndex(1)
            self._refresh_panel()

    def _refresh_panel(self) -> None:
        if not self._cur or self._stack.currentIndex() != 1:
            return
        cat = self._cur["category"]
        models = [m for m in self._models if m.get("category") == cat]
        it = self._readiness.get(cat) or {}
        sl = self._sl_by_cat.get(cat) or {}
        stage = str(it.get("stage") or
                    ("ready" if not models else "trained"))

        # 状态头
        self.lbl_cat.setText(cat)
        stage_cn = str(it.get("stage_cn") or "")
        if stage_cn:
            fg, bg = _STAGE_STYLE.get(stage, ("#57606a", "#f6f8fa"))
            self.lbl_stage.setText(stage_cn)
            self.lbl_stage.setStyleSheet(
                f"color:{fg}; background:{bg}; border-radius:8px;"
                f" padding:2px 10px; font-weight:600;")
            self.lbl_stage.setVisible(True)
        else:
            self.lbl_stage.setVisible(False)
        self.lbl_next.setText(str(it.get("next_action") or ""))

        # 当前状态
        active = next((m for m in models if m.get("is_active")), None)
        ver = active.get("version") if active else it.get("active_version")
        self.lbl_version.setText(str(ver) if ver is not None else "未激活")
        # 导出按钮仅在有激活版本时可用（导出对象=当前激活版本）
        self.btn_export.setEnabled(active is not None
                                   or it.get("active_version") is not None)
        best = it.get("best_auroc")
        self.lbl_auroc.setText(
            f"{float(best):.4f}" if best is not None else "—")
        pend = sl.get("unconsumed_db")
        self.lbl_pending.setText(f"{pend} 条" if pend is not None else "0 条")
        g = self._cur.get("groups") or {}
        pc = g.get("pretrain_counts") or {}
        if pc:
            src_txt = (f"数据源「{self._cur['datasource_name']}」 · "
                       f"预训练组 {pc.get('normal', 0)} 正常 + "
                       f"{pc.get('anomaly', 0)} 异常 · 检测组 "
                       f"{g.get('detect_count', 0)} 图")
        else:
            src_txt = f"数据源「{self._cur['datasource_name']}」"
        self.lbl_source.setText(src_txt)

        # 操作区按阶段引导
        has_models = bool(models)
        if stage in ("empty", "insufficient"):
            self.lbl_op_hint.setText(
                "该品类数据未导入或不足，请先到「数据管理」为数据源导入数据")
            self.btn_prepare.setVisible(False)
        elif stage == "ready":
            self.lbl_op_hint.setText(
                "数据已就绪：点击「准备模型」，样本将取自该数据源的预训练组")
            self.btn_prepare.setText("准备模型")
            self.btn_prepare.setVisible(True)
        else:
            self.lbl_op_hint.setText(
                "可「学习反馈并更新模型」生成新版本；或重新准备（换型/数据大改时）")
            self.btn_prepare.setText("重新准备模型")
            self.btn_prepare.setVisible(True)
        self.btn_learn.setVisible(has_models)

        self._fill_history(models)

    # ── 版本历史（折叠区）────────────────────────
    def _toggle_history(self) -> None:
        vis = not self.hist_body.isVisible()
        self.hist_body.setVisible(vis)
        self.btn_hist_toggle.setText(
            ("▾" if vis else "▸") + self.btn_hist_toggle.text()[1:])

    def _fill_history(self, models: list[dict]) -> None:
        models = sorted(models, key=lambda m: (_version_num(m) or 0),
                        reverse=True)
        self._model_buttons = {}
        self.tbl.setRowCount(len(models))
        self.btn_hist_toggle.setText(
            f"{'▾' if self.hist_body.isVisible() else '▸'} "
            f"版本历史（{len(models)} 个版本）")
        for row, m in enumerate(models):
            rb = QRadioButton()
            rb.toggled.connect(self._toggle_ops)
            self.tbl.setCellWidget(row, 0, rb)
            self._model_buttons[m["id"]] = rb
            ver_item = QTableWidgetItem(str(m.get("version") or "-"))
            if m.get("is_active"):
                ver_item.setForeground(QColor("#1a7f37"))
            self.tbl.setItem(row, 1, ver_item)
            self.tbl.setItem(row, 2, QTableWidgetItem(
                ORIGIN_NAMES.get(m.get("origin"), m.get("origin") or "-")))
            metrics = m.get("metrics") or {}
            txt = " ".join(f"{k}={v}" for k, v in
                           (("AUC", metrics.get("auroc")),
                            ("AP", metrics.get("ap")),
                            ("F1", metrics.get("f1"))) if v is not None)
            self.tbl.setItem(row, 3, QTableWidgetItem(txt or "—"))
            btn_ab = QPushButton("A/B 对比")
            btn_ab.clicked.connect(
                lambda _=False, mm=m: self._on_ab(mm))
            self.tbl.setCellWidget(row, 4, btn_ab)
            if m.get("is_active"):
                rb.setChecked(True)
        # 回滚目标 = 次新版本号
        vers = sorted({_version_num(m) for m in models} - {None})
        if len(vers) >= 2:
            self._rollback_target_ver = vers[-2]
            self.btn_rollback.setText(f"回滚到 v{vers[-2]}")
            self.btn_rollback.setEnabled(True)
            self.btn_rollback.setToolTip(
                f"放弃当前版本，回到上一版本 v{vers[-2]}")
        else:
            self._rollback_target_ver = None
            self.btn_rollback.setText("回滚")
            self.btn_rollback.setEnabled(False)
            self.btn_rollback.setToolTip("只有一个版本，无可回滚目标")

    def _toggle_ops(self) -> None:
        mid = self._selected_model_id()
        if mid is None:
            self.btn_activate.setEnabled(False)
            return
        m = next((x for x in self._models if x.get("id") == mid), None)
        self.btn_activate.setEnabled(bool(m) and not m.get("is_active"))

    def _selected_model_id(self) -> int | None:
        for mid, rb in self._model_buttons.items():
            if rb.isChecked():
                return mid
        return None

    # ── 操作：准备 / 学习 / 激活 / 回滚 / A-B ─────
    def _on_prepare(self) -> None:
        if not self._cur:
            return
        wo = self._get_workorder() if callable(self._get_workorder) else None
        dlg = PrepareModelDialog(
            self._client, self._cur["category"],
            datasource={"id": self._cur["datasource_id"],
                        "name": self._cur["datasource_name"]},
            workorder=wo, parent=self)
        dlg.succeeded.connect(self.reload)
        dlg.exec()

    def _on_self_learning(self) -> None:
        if not self._cur:
            return
        cat = self._cur["category"]
        if not notify(self, "确认学习更新",
                      f"把积累的反馈固化到「{cat}」模型并生成新版本？",
                      ask=True):
            return
        self._sl_running = True
        self.btn_learn.setEnabled(False)
        self.bar_learn.setVisible(True)
        self.bar_learn.show()
        self.bar_learn.update_task({"progress": 0, "message": "提交学习更新…",
                                    "status": "running"})

        def _apply(res) -> None:
            self._sl_running = False
            self.btn_learn.setEnabled(True)
            if not isinstance(res, dict):
                self.bar_learn.setVisible(False)
                return
            if res.get("task_id"):
                self._learn_task_id = str(res["task_id"])
            else:
                # 后端直返（无有效反馈可学习等）
                self.bar_learn.setVisible(False)
                notify(self, "学习更新",
                       str(res.get("note") or res.get("message") or "完成"))
                self.reload()

        run_async(self, lambda: self._client.self_learning_update(
            cat, note=self.edit_note.text().strip()), _apply)

    def _on_task(self, t: dict) -> None:
        if self._learn_task_id is None:
            return
        if str(t.get("id")) != self._learn_task_id:
            return
        status = str(t.get("status"))
        if status in ("done", "success", "failed", "error"):
            self._learn_task_id = None
            self._sl_running = False
            self.btn_learn.setEnabled(True)
            self.bar_learn.setVisible(False)
            if status in ("done", "success"):
                cat = (self._cur or {}).get("category", "")
                notify(self, "学习更新", f"「{cat}」已生成新版本")
            else:
                warn(self, "学习更新失败", str(t.get("message") or t.get("error") or ""))
            self.reload()
        else:
            self.bar_learn.update_task(t)

    def _on_activate(self) -> None:
        if not self._cur:
            return
        mid = self._selected_model_id()
        if mid is None:
            warn(self, "未选中", "请先在版本历史中选择一个版本")
            return
        cat = self._cur["category"]
        run_async(self, lambda: self._client.activate_model(mid, gate=True),
                  lambda res: self._apply_activation(mid, cat, res))

    def _apply_activation(self, mid: int, cat: str,
                          res: dict | None) -> None:
        if isinstance(res, dict) and res.get("_status") == 409:
            res["_model_id"] = mid
        handle_activation_result(self, self._client, cat, res,
                                 lambda _r: self.reload())

    def _on_rollback(self) -> None:
        if not self._cur:
            return
        cat = self._cur["category"]
        tgt = self._rollback_target_ver
        tgt_txt = f"v{tgt}" if tgt is not None else "上一版本"
        if not notify(self, "确认回滚",
                      f"确定把「{cat}」回滚到 {tgt_txt}？\n"
                      "当前版本将保留在版本历史中。",
                      ask=True):
            return
        run_async(self, lambda: self._client.rollback_model(cat),
                  self._fill_rollback_result)

    def _fill_rollback_result(self, res: dict | None) -> None:
        if isinstance(res, dict) and res.get("active_version") is not None:
            notify(self, "回滚完成",
                   f"当前版本：v{res.get('active_version')}")
        self.reload()

    # ── 操作：快照导出 / 导入（跨机器/跨品类复用）─────────────
    def _on_export(self) -> None:
        if not self._cur:
            return
        cat = self._cur["category"]
        path, _ = QFileDialog.getSaveFileName(
            self, "导出当前版本", f"{cat}_snapshot.zip",
            "模型快照包 (*.zip)")
        if not path:
            return
        run_async(self,
                  lambda: self._client.export_snapshot(cat, path),
                  lambda res: self._on_exported(res, path))

    def _on_exported(self, res: str | None, path: str) -> None:
        if res:
            notify(self, f"已导出：{path}")

    def _on_import(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择模型快照包", "", "模型快照包 (*.zip)")
        if not path:
            return
        default_cat = str((self._cur or {}).get("category") or "")
        cat, ok = QInputDialog.getText(
            self, "导入快照",
            "目标品类名（可与原品类不同，用于跨品类复用）：",
            text=default_cat)
        cat = str(cat).strip()
        if not ok or not cat:
            return
        run_async(self,
                  lambda: self._client.import_snapshot(path, cat),
                  self._on_imported)

    def _on_imported(self, res: dict | None) -> None:
        if isinstance(res, dict) and res.get("version"):
            notify(self, f"导入成功：{res.get('category')} {res.get('version')}"
                         "（未激活）。请在版本历史中选中该版本，"
                         "验收后点「激活选中版本」")
            self.reload()

    def _on_ab(self, m: dict) -> None:
        if not self._cur:
            return
        cat = self._cur["category"]
        ver = _version_num(m)
        if ver is None:
            warn(self, "版本异常", f"该模型版本号无法解析：{m.get('version')}")
            return
        models = [x for x in self._models if x.get("category") == cat]
        active = next((x for x in models if x.get("is_active")), None)
        if active is None:
            warn(self, "无法对比", "当前品类没有激活版本可作为基准")
            return
        base = _version_num(active)
        if base is None:
            warn(self, "版本异常",
                 f"激活版本号无法解析：{active.get('version')}")
            return
        if ver == base:
            warn(self, "无意义对比",
                 "选中版本就是当前激活版本，请选择其他版本对比")
            return
        if not notify(self, "A/B 对比确认",
                      f"将提交 A/B 对比任务：\n"
                      f"品类：{cat}\n基准（当前激活）：v{base}\n对比版本：v{ver}",
                      ask=True):
            return
        res = self._client.ab_compare(cat, version_b=ver, version_a=base)
        if isinstance(res, dict) and res.get("task_id"):
            notify(self, "A/B 对比已提交",
                   "请到「评估看板」查看 A/B 对比结果")
        else:
            warn(self, "提交失败", "A/B 对比任务提交失败")


# ══════════════════ 准备模型对话框 ══════════════════

class PrepareModelDialog(QDialog):
    """准备模型对话框：样本取自所选数据源的「预训练组/检测组」。

    前端反馈 2026-08-29 重构：
    - 品类、数据源由模型页树选中带入（只读），不再让用户手选；
    - 数据条件由工单条件/数据体检自动判定，只读显示中文，不再手选代号；
    - 提交传 datasource_id，走数据源预训练组取样本（前端反馈 v9）。
    """

    succeeded = Signal()

    _PROFILE_ITEMS = [
        ("快速模式（单图 <200ms，产线实时）", "fast"),
        ("高精度模式（更准但更慢，适合复检）", "accuracy"),
        ("CPU 模式（无显卡环境）", "cpu"),
    ]

    def __init__(self, client: ApiClient, category: str,
                 datasource: dict | None = None,
                 workorder: dict | None = None,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self._client = client
        self._cat = category
        self._datasource = datasource or {}
        self._workorder = workorder
        self._precheck_data: dict = {}
        self._task_id: str | None = None
        self._result: dict | None = None
        self._worker: ApiWorker | None = None
        self._poll = QTimer(self)
        self._poll.setInterval(800)
        self._poll.timeout.connect(self._poll_task)
        self._preparing = False
        # 数据条件自动判定：工单条件优先，无工单由数据体检建议填充
        self._order_level = self._sync_order_level(workorder)
        self._scenario: str = self._order_level or "L1a"

        self.setWindowTitle(f"准备模型：{category}")
        self.setMinimumWidth(620)
        lay = QVBoxLayout(self)

        title = QLabel(f"为品类「{category}」准备检测模型")
        title.setProperty("heading", True)
        lay.addWidget(title)
        ds_name = self._datasource.get("name")
        self.lbl_source = QLabel(
            f"样本来源：数据源「{ds_name}」的预训练组" if ds_name
            else "样本来源：图库（未挂数据源）")
        self.lbl_source.setWordWrap(True)
        self.lbl_source.setProperty("subtext", True)
        lay.addWidget(self.lbl_source)
        self.lbl_scenario = QLabel("数据条件：判定中…")
        self.lbl_scenario.setWordWrap(True)
        self.lbl_scenario.setProperty("subtext", True)
        lay.addWidget(self.lbl_scenario)

        # 数据体检区
        self.grp_check = make_card()
        ck = QVBoxLayout(self.grp_check)
        ck.addWidget(self._mk_head("数据体检"))
        self.lbl_counts = QLabel("检查中…")
        self.lbl_counts.setProperty("subtext", True)
        ck.addWidget(self.lbl_counts)
        self.lbl_reso = QLabel("")
        self.lbl_reso.setProperty("subtext", True)
        ck.addWidget(self.lbl_reso)
        self.lbl_warn = QLabel("")
        self.lbl_warn.setWordWrap(True)
        self.lbl_warn.setStyleSheet("color:#b26a00;")
        ck.addWidget(self.lbl_warn)
        lay.addWidget(self.grp_check)

        # 选项区：运行模式 + 强制重备
        opt = QHBoxLayout()
        opt.addWidget(QLabel("运行模式："))
        self.cmb_profile = QComboBox()
        for label, code in self._PROFILE_ITEMS:
            self.cmb_profile.addItem(label, code)
        self._profile_touched = False
        self.cmb_profile.currentIndexChanged.connect(
            lambda _i: setattr(self, "_profile_touched", True))
        opt.addWidget(self.cmb_profile)
        self.chk_force = QCheckBox("重新准备（覆盖已有版本，生成新版本号）")
        self.chk_force.setChecked(True)
        opt.addWidget(self.chk_force)
        opt.addStretch(1)
        lay.addLayout(opt)

        # 进度区
        self.bar = TaskProgressBar()
        self.bar.setVisible(False)
        lay.addWidget(self.bar)
        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        lay.addWidget(self.lbl_status)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.btn_ok = bb.button(QDialogButtonBox.Ok)
        self.btn_ok.setText("开始准备")
        self.btn_ok.setProperty("primary", True)
        self.btn_ok.clicked.connect(self._on_start)
        bb.rejected.connect(self._on_cancel)
        lay.addWidget(bb)

        self._load_precheck()

    @staticmethod
    def _mk_head(text: str) -> QLabel:
        lb = QLabel(text)
        lb.setProperty("heading", True)
        return lb

    # ── 数据条件判定 ─────────────────────────────
    @staticmethod
    def _sync_order_level(workorder: dict | None) -> str:
        """从工单聚合条件取数据条件；有模板开关时按模板比对判定。"""
        cond = (workorder or {}).get("conditions") or {}
        if not isinstance(cond, dict):
            return ""
        if cond.get("has_template"):
            return "L3"
        return str(cond.get("label_tier") or "")

    def _scenario_cn(self) -> str:
        """当前判定结果的中文描述。"""
        cond = (self._workorder or {}).get("conditions") or {}
        if cond.get("has_template"):
            return "模板比对"
        return LAYER_CN.get(self._scenario, self._scenario) or "少样本"

    # ── 数据体检 ─────────────────────────────────
    def _load_precheck(self) -> None:
        # 前端反馈 2026-09-13 #4：带数据源 ID 让后端做数据血缘核验
        # （新导入数据与已准备模型是否同源，而非只凭品类名匹配）
        ds_id = (self._datasource or {}).get("id")
        self._worker = run_async(
            self, lambda: self._client.precheck(self._cat, ds_id),
            self._fill_precheck)

    def _fill_precheck(self, data: dict | None) -> None:
        data = data or {}
        self._precheck_data = data
        counts = data.get("counts") or {}
        ng = int(counts.get("anomaly_images", 0) or 0)
        ok = int(counts.get("normal_images", 0) or 0)
        ds_id = self._datasource.get("id")
        if ds_id:
            # 有数据源：以数据源预训练组统计为准（prepare 实际取数口径）
            run_async(self, lambda: self._client.datasource_groups(ds_id),
                      lambda g: self._fill_group_counts(g, ok, ng))
        else:
            self.lbl_counts.setText(
                f"图库标注统计：异常图 {ng} 张 / 正常图 {ok} 张")
        res_list = data.get("resolutions") or []
        native = "?"
        if isinstance(res_list, list) and res_list:
            r0 = res_list[0] or {}
            native = f"{r0.get('width', '?')}×{r0.get('height', '?')}"
        self.lbl_reso.setText(f"主流分辨率：{native}")
        warns = [_user_warning(w) for w in (data.get("warnings") or [])]
        # 工单声明模板比对但品类无模板图：拦截
        if self._scenario == "L3" and not bool(data.get("has_template")):
            warns = list(warns) + [
                "工单声明了「模板比对」，但该品类还没有模板图："
                "请先到「数据管理」为品类设置标准模板图，再准备模型"]
        self.lbl_warn.setText("\n".join(str(w) for w in warns))
        self.lbl_warn.setVisible(bool(warns))
        # 数据条件判定：工单条件优先，无工单用体检建议
        if not self._order_level:
            sugg = str(data.get("suggested_scenario") or "")
            if sugg:
                self._scenario = sugg
        if self._order_level:
            self.lbl_scenario.setText(
                f"数据条件：{self._scenario_cn()}（按工单条件自动判定）")
        else:
            self.lbl_scenario.setText(
                f"数据条件：{self._scenario_cn()}（按数据体检自动判定）")
        # 体检建议运行模式（仅用户未手动改过时联动）
        sugg_p = str(data.get("suggested_profile") or "")
        if sugg_p and not getattr(self, "_profile_touched", False):
            idx = self.cmb_profile.findData(sugg_p)
            if idx >= 0:
                self.cmb_profile.setCurrentIndex(idx)

    def _fill_group_counts(self, data: dict | None, ok: int, ng: int) -> None:
        cats = (data or {}).get("categories") or {}
        g = cats.get(self._cat) or {}
        pc = g.get("pretrain_counts") or {}
        if pc:
            self.lbl_counts.setText(
                f"预训练组：{pc.get('normal', 0)} 正常 + "
                f"{pc.get('anomaly', 0)} 异常 · 检测组 "
                f"{g.get('detect_count', 0)} 图（图库合计：异常 {ng} / "
                f"正常 {ok}）")
        else:
            self.lbl_counts.setText(
                f"图库标注统计：异常图 {ng} 张 / 正常图 {ok} 张")

    # ── 提交准备 ─────────────────────────────────
    def _on_start(self) -> None:
        self._preparing = True
        self.btn_ok.setEnabled(False)
        self.bar.setVisible(True)
        self.bar.show()
        self.bar.update_task({"progress": 0, "message": "提交准备任务…",
                              "status": "running"})
        self.lbl_status.setText("")
        scenario = self._scenario
        profile = self.cmb_profile.currentData()
        force = self.chk_force.isChecked()
        ds_id = self._datasource.get("id")
        res = self._client.prepare_model(
            self._cat, scenario=scenario, profile=profile, force=force,
            datasource_id=ds_id)
        if not isinstance(res, dict) or not res.get("task_id"):
            self._preparing = False
            self.btn_ok.setEnabled(True)
            self.bar.setVisible(False)
            self.lbl_status.setText("准备任务提交失败（后端未返回任务 ID）")
            return
        self._task_id = str(res["task_id"])
        self._poll.start()

    def _poll_task(self) -> None:
        if not self._task_id:
            self._poll.stop()
            return
        t = self._client.get_task(self._task_id)
        if not isinstance(t, dict):
            return
        self.bar.update_task(t)
        status = t.get("status")
        # 2026-09-13 修复：后端任务失败状态为 "failed"（TaskManager 契约），
        # 原代码只认 "error"，失败时无限轮询、弹窗卡死
        if status in ("done", "error", "failed"):
            self._poll.stop()
            self._preparing = False
            self._result = t.get("result") or {}
            if status == "done":
                self._show_result()
            else:
                self.btn_ok.setEnabled(True)
                self.bar.setVisible(False)
                self.lbl_status.setText(
                    f"准备失败：{t.get('message') or '未知错误'}")

    def _show_result(self) -> None:
        r = self._result or {}
        version = str(r.get("version") or "-")
        n = r.get("n_normal")
        a = r.get("n_defect")
        self.bar.setVisible(False)
        self.lbl_status.setText(
            f"准备完成，已生成新版本 {version}\n"
            f"样本量：正常 {n if n is not None else '?'} 张 / "
            f"异常 {a if a is not None else '?'} 张\n"
            f"数据条件：{self._scenario_cn()}\n"
            f"运行模式：{self.cmb_profile.currentText()}\n"
            "下一步：到「评估看板」验收该版本，确认无误后再激活")
        self.btn_ok.setText("完成")
        self.btn_ok.setEnabled(True)
        try:
            self.btn_ok.clicked.disconnect(self._on_start)
        except Exception:  # noqa: BLE001
            pass
        self.btn_ok.clicked.connect(self.accept)
        self.succeeded.emit()

    def _on_cancel(self) -> None:
        if self._preparing and not notify(
                self, "任务进行中",
                "准备任务正在执行，确定关闭？（后台任务仍会继续）", ask=True):
            return
        self._poll.stop()
        self.reject()


# ══════════════════ A/B 对比结果对话框 ══════════════════

def _fmt_metric(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


class ABCompareDialog(QDialog):
    """A/B 对比结果展示：version_a（基准/当前激活） vs version_b（候选）。"""

    def __init__(self, data: dict, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("A/B 对比结果")
        self.setMinimumWidth(640)
        lay = QVBoxLayout(self)
        va, vb = data.get("version_a"), data.get("version_b")
        cat = data.get("category", "")
        title = QLabel(f"品类「{cat}」：v{va}（基准） vs v{vb}（候选）")
        title.setProperty("heading", True)
        lay.addWidget(title)
        keys = sorted(set((data.get("a") or {}).keys())
                      | set((data.get("b") or {}).keys()))
        keys = [k for k in keys if k not in ("version", "category")]
        tbl = QTableWidget(len(keys), 3)
        tbl.setHorizontalHeaderLabels(["指标", f"v{va}（基准）", f"v{vb}（候选）"])
        tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        tbl.verticalHeader().setVisible(False)
        a, b = data.get("a") or {}, data.get("b") or {}
        for r, k in enumerate(keys):
            tbl.setItem(r, 0, QTableWidgetItem(str(k)))
            tbl.setItem(r, 1, QTableWidgetItem(_fmt_metric(a.get(k))))
            tbl.setItem(r, 2, QTableWidgetItem(_fmt_metric(b.get(k))))
        lay.addWidget(tbl)
        verdict = str(data.get("verdict") or "")
        if verdict:
            v = QLabel(verdict)
            v.setWordWrap(True)
            v.setProperty("subtext", True)
            lay.addWidget(v)
        bb = QDialogButtonBox(QDialogButtonBox.Ok)
        bb.accepted.connect(self.accept)
        lay.addWidget(bb)
