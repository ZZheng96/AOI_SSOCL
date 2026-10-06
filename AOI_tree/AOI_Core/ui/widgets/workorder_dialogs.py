"""工单对话框（U-workorder v2，前端反馈 2026-08-27）：

- WorkOrderCreateDialog：新建工单——名称 + 数据条件（三档＋两开关）
  + 挂接数据源 + 可从任务模板导入配置 + 可同时存为模板。
  创建后后端自动体检并校正条件（check.applied 在此提示）。
- WorkOrderEditDialog：修改工单条件/挂接（工作台双击「数据条件」弹出），
  含「体检并自动校正」按钮。
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ui.api_client import ApiClient
from ui.pages.common import (condition_text, notify, run_async, warn)

_MODALITY_CN = {"image": "图片", "video": "视频"}


def _source_label(src: dict) -> str:
    """数据源条目显示文本：名称（模态）＋ 支撑摘要。"""
    cap = src.get("capability") or {}
    bits = []
    if cap.get("n_images"):
        bits.append(f"图 {cap['n_images']}")
    if cap.get("n_anomaly"):
        bits.append(f"缺陷 {cap['n_anomaly']}")
    if cap.get("n_masks"):
        bits.append(f"掩码 {cap['n_masks']}")
    if cap.get("n_videos"):
        bits.append(f"视频 {cap['n_videos']}")
    if cap.get("n_templates"):
        bits.append(f"模板 {cap['n_templates']}")
    cats = cap.get("categories") or []
    if cats:
        bits.append("品类：" + "、".join(str(c) for c in cats[:4])
                    + ("…" if len(cats) > 4 else ""))
    tail = f"｜{'，'.join(bits)}" if bits else "｜暂无数据"
    return (f"{src.get('name')}（{_MODALITY_CN.get(src.get('modality'), src.get('modality'))}）"
            f"{tail}")


def _applied_text(check: dict) -> str:
    """体检自动调整项 -> 中文提示。"""
    applied = (check or {}).get("applied") or {}
    if not applied:
        return ""
    sugg = (check or {}).get("suggestion") or {}
    cur = condition_text(sugg.get("label_tier"),
                         bool(sugg.get("per_category")),
                         bool(sugg.get("has_template")))
    return f"体检已自动校正条件为「{cur}」（可修复数据后再次体检）"


class _SourcePickList(QListWidget):
    """数据源多选列表（checkbox）。"""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMaximumHeight(120)

    def fill(self, sources: list, checked_ids: set | None = None) -> None:
        self.clear()
        checked_ids = checked_ids or set()
        if not sources:
            it = QListWidgetItem("（尚无数据源：先到「数据管理」新建数据源）")
            it.setFlags(Qt.NoItemFlags)
            self.addItem(it)
            return
        for src in sources:
            it = QListWidgetItem(_source_label(src))
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(
                Qt.Checked if src.get("id") in checked_ids else Qt.Unchecked)
            it.setData(Qt.UserRole, src.get("id"))
            self.addItem(it)

    def checked_ids(self) -> list[int]:
        out = []
        for i in range(self.count()):
            it = self.item(i)
            if it.flags() & Qt.ItemIsUserCheckable and \
                    it.checkState() == Qt.Checked:
                out.append(int(it.data(Qt.UserRole)))
        return out


class WorkOrderCreateDialog(QDialog):
    """新建工单对话框。"""

    levels_changed = Signal()
    created = Signal(int)   # 创建成功：携带新工单 id（顶栏自动选中用）

    def __init__(self, client: ApiClient, parent: QWidget | None = None):
        super().__init__(parent)
        self._client = client
        self._templates: list[dict] = []
        self.setWindowTitle("新建工单")
        self.setMinimumWidth(520)
        lay = QVBoxLayout(self)

        tip = QLabel("工单 = 一条产线质检任务：挂接数据源（数据条件由数据源"
                     "属性自动聚合），统计与检测都以工单为基础")
        tip.setProperty("subtext", True)
        tip.setWordWrap(True)
        lay.addWidget(tip)

        form = QFormLayout()
        self.edit_name = QLineEdit()
        self.edit_name.setPlaceholderText("如：3号线 SMT 质检")
        form.addRow("工单名称", self.edit_name)

        self.combo_template = QComboBox()
        self.combo_template.addItem("（不使用）", None)
        self.combo_template.currentIndexChanged.connect(self._on_template)
        form.addRow("复制工单配置", self.combo_template)
        self.combo_template.setToolTip(
            "从已保存的任务模板复制数据源挂接与复判设置（数据条件随源自动算）")
        lay.addLayout(form)

        form2 = QFormLayout()
        self.edit_note = QLineEdit()
        form2.addRow("备注", self.edit_note)
        lay.addLayout(form2)

        lay.addWidget(QLabel("挂接数据源（必选：至少勾选一个数据源）"))
        self.list_sources = _SourcePickList()
        self.list_sources.itemChanged.connect(lambda _it: self._sync_ok())
        lay.addWidget(self.list_sources)

        self.chk_review = QCheckBox("人工复判（开启后全部检测进待复核队列，"
                                    "复核后才计入统计与持续学习）")
        self.chk_review.setToolTip("复判开关：该工单检测结果全部需要人工复核；"
                                   "复核提交后自动生成反馈回流持续学习")
        lay.addWidget(self.chk_review)

        self.chk_save_template = QCheckBox("同时把该配置保存为任务模板")
        lay.addWidget(self.chk_save_template)

        self.lbl_state = QLabel("")
        self.lbl_state.setProperty("subtext", True)
        self.lbl_state.setWordWrap(True)
        lay.addWidget(self.lbl_state)

        row = QHBoxLayout()
        self.btn_ok = QPushButton("创建工单")
        self.btn_ok.setProperty("primary", True)
        self.btn_ok.setEnabled(False)   # 数据源必选：未勾选不可创建
        self.btn_ok.clicked.connect(self._on_create)
        btn_cancel = QPushButton("取消")
        btn_cancel.clicked.connect(self.reject)
        row.addStretch(1)
        row.addWidget(self.btn_ok)
        row.addWidget(btn_cancel)
        lay.addLayout(row)

        run_async(self, self._client.list_workorder_templates,
                  self._fill_templates)
        run_async(self, self._client.list_datasources, self._fill_sources)

    def _fill_sources(self, items) -> None:
        """数据源列表填充；无数据源时给出可操作引导（v10 无未挂源概念：
        数据必须先建数据源、导入到源，工单才可选）。"""
        if items:
            self.list_sources.fill(items or [])
            return

        def _probe(ds_data) -> None:
            self.list_sources.fill([])
            if self.list_sources.count():
                it = self.list_sources.item(0)
                it.setText("当前没有登记的数据源。\n"
                           "请到「数据管理」新建数据源并导入数据后，"
                           "即可在此选择。")
        run_async(self, self._client.list_datasets, _probe)

    def done(self, result: int) -> None:
        """关闭前等待后台线程，避免 'QThread: Destroyed while thread is still
        running' 崩溃（前端反馈 v3：点新建工单闪退）。"""
        for w in list(getattr(self, "_async_workers", [])):
            if w.isRunning():
                w.wait(3000)
        super().done(result)

    def _fill_templates(self, items) -> None:
        self._templates = [t for t in (items or []) if isinstance(t, dict)]
        for t in self._templates:
            cfg = t.get("config") or {}
            n_src = len(cfg.get("datasource_ids") or [])
            rev = "复判" if cfg.get("review_enabled") else "不复判"
            self.combo_template.addItem(
                f"{t.get('name')}（{n_src} 源 · {rev}）", t.get("id"))

    def _on_template(self, _idx: int) -> None:
        tid = self.combo_template.currentData()
        tpl = next((t for t in self._templates if t.get("id") == tid), None)
        if not tpl:
            return
        cfg = tpl.get("config") or {}
        src_ids = {int(d) for d in (cfg.get("datasource_ids") or [])}
        if src_ids:   # 预勾选模板挂接的数据源
            for i in range(self.list_sources.count()):
                it = self.list_sources.item(i)
                if it.flags() & Qt.ItemIsUserCheckable and \
                        int(it.data(Qt.UserRole)) in src_ids:
                    it.setCheckState(Qt.Checked)
        if "review_enabled" in cfg:
            self.chk_review.setChecked(bool(cfg.get("review_enabled")))
        self._sync_ok()

    def _sync_ok(self) -> None:
        """数据源必选：至少勾选一个才允许创建。"""
        self.btn_ok.setEnabled(bool(self.list_sources.checked_ids()))

    def _on_create(self) -> None:
        name = self.edit_name.text().strip()
        if not name:
            self.lbl_state.setText("请填写工单名称")
            return
        if not self.list_sources.checked_ids():
            self.lbl_state.setText("请至少勾选一个数据源（必选）")
            return
        self.btn_ok.setEnabled(False)
        src_ids = self.list_sources.checked_ids()
        payload = dict(
            name=name, review_enabled=self.chk_review.isChecked(),
            note=self.edit_note.text().strip(),
            datasource_ids=src_ids,
            from_template=self.combo_template.currentData())

        def _done(res) -> None:
            if not isinstance(res, dict) or not res.get("id"):
                self.btn_ok.setEnabled(True)
                self.lbl_state.setText("创建失败，请重试")
                return
            if self.chk_save_template.isChecked():
                run_async(self, lambda: self._client.create_workorder_template(
                    name=name, datasource_ids=src_ids,
                    review_enabled=self.chk_review.isChecked(),
                    note=self.edit_note.text().strip()),
                    lambda _r: None, lambda _e: None)
            check = res.get("check") or {}
            warns = check.get("warnings") or []
            msg = f"工单已创建：{name}"
            adjusted = _applied_text(check)
            if adjusted:
                msg += f"；{adjusted}"
            elif warns:
                msg += "；" + "；".join(str(w) for w in warns[:2])
            notify(self, msg)
            self.created.emit(int(res["id"]))
            self.levels_changed.emit()
            self.accept()

        def _fail(msg) -> None:
            self.btn_ok.setEnabled(True)
            self.lbl_state.setText(f"创建失败：{msg}")

        run_async(self, lambda: self._client.create_workorder(**payload),
                  _done, _fail)


class WorkOrderEditDialog(QDialog):
    """修改工单数据条件 / 挂接数据源（工作台双击「数据条件」弹出）。"""

    levels_changed = Signal()

    def __init__(self, client: ApiClient, workorder: dict,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self._client = client
        self._wo = dict(workorder or {})
        wid = self._wo.get("id")
        self.setWindowTitle(f"编辑工单（{self._wo.get('name')}）")
        self.setMinimumWidth(520)
        lay = QVBoxLayout(self)

        tip = QLabel("数据条件由所挂数据源属性自动聚合（标注档位取源最高档、"
                     "品类并集、模板比对任一源有）；可在数据管理页修改数据源。")
        tip.setProperty("subtext", True)
        tip.setWordWrap(True)
        lay.addWidget(tip)

        form = QFormLayout()
        self.edit_name = QLineEdit(str(self._wo.get("name") or ""))
        form.addRow("工单名称", self.edit_name)
        self.edit_note = QLineEdit(str(self._wo.get("note") or ""))
        form.addRow("备注", self.edit_note)
        conds = self._wo.get("conditions") or {}
        self.lbl_cond = QLabel(
            condition_text(conds.get("label_tier"),
                           bool(conds.get("per_category")),
                           bool(conds.get("has_template")))
            or "（未挂数据源）")
        self.lbl_cond.setProperty("badge", "info")
        form.addRow("数据条件（自动聚合）", self.lbl_cond)
        lay.addLayout(form)

        lay.addWidget(QLabel("挂接数据源（可多选）"))
        self.list_sources = _SourcePickList()
        lay.addWidget(self.list_sources)
        attached = {int(d.get("id")) for d in (self._wo.get("datasources") or [])
                    if d.get("id") is not None}
        run_async(self, self._client.list_datasources,
                  lambda items: self.list_sources.fill(items or [], attached))

        self.chk_review = QCheckBox("人工复判（全部检测进待复核队列，"
                                    "复核后才计入统计与持续学习）")
        self.chk_review.setChecked(bool(self._wo.get("review_enabled")))
        lay.addWidget(self.chk_review)

        self.lbl_state = QLabel("")
        self.lbl_state.setProperty("subtext", True)
        self.lbl_state.setWordWrap(True)
        lay.addWidget(self.lbl_state)

        row = QHBoxLayout()
        self.btn_check = QPushButton("体检并自动校正")
        self.btn_check.setToolTip("按数据实际支撑自动调整数据条件")
        self.btn_check.clicked.connect(self._on_check)
        row.addWidget(self.btn_check)
        row.addStretch(1)
        self.btn_ok = QPushButton("保存")
        self.btn_ok.setProperty("primary", True)
        self.btn_ok.clicked.connect(self._on_save)
        btn_cancel = QPushButton("取消")
        btn_cancel.clicked.connect(self.reject)
        row.addWidget(self.btn_ok)
        row.addWidget(btn_cancel)
        lay.addLayout(row)
        self._wid = int(wid)

    def done(self, result: int) -> None:
        """关闭前等待后台线程，避免 'QThread: Destroyed while thread is still
        running' 崩溃。"""
        for w in list(getattr(self, "_async_workers", [])):
            if w.isRunning():
                w.wait(3000)
        super().done(result)

    def _show_check(self, check: dict) -> None:
        warns = (check or {}).get("warnings") or []
        adjusted = _applied_text(check or {})
        parts = []
        if adjusted:
            parts.append(adjusted)
        if warns:
            parts.append("体检提醒：" + "；".join(str(w) for w in warns[:3]))
        self.lbl_state.setText("\n".join(parts))

    def _on_save(self) -> None:
        name = self.edit_name.text().strip()
        if not name:
            self.lbl_state.setText("请填写工单名称")
            return
        self.btn_ok.setEnabled(False)

        def _done(res) -> None:
            self.btn_ok.setEnabled(True)
            if isinstance(res, dict) and res.get("id"):
                notify(self, f"工单已更新：{name}")
                conds = res.get("conditions") or {}
                self.lbl_cond.setText(
                    condition_text(conds.get("label_tier"),
                                   bool(conds.get("per_category")),
                                   bool(conds.get("has_template")))
                    or "（未挂数据源）")
                self._show_check(res.get("check") or {})
                self.levels_changed.emit()
                self.accept()
            else:
                self.lbl_state.setText("保存失败，请重试")

        def _fail(msg) -> None:
            self.btn_ok.setEnabled(True)
            self.lbl_state.setText(f"保存失败：{msg}")

        run_async(self, lambda: self._client.update_workorder(
            self._wid, name=name,
            review_enabled=self.chk_review.isChecked(),
            note=self.edit_note.text().strip(),
            datasource_ids=self.list_sources.checked_ids()), _done, _fail)

    def _on_check(self) -> None:
        self.btn_check.setEnabled(False)

        def _done(res) -> None:
            self.btn_check.setEnabled(True)
            if not isinstance(res, dict):
                self.lbl_state.setText("体检失败，请重试")
                return
            check = res.get("check") or {}
            conds = res.get("conditions") or {}
            self.lbl_cond.setText(
                condition_text(conds.get("label_tier"),
                               bool(conds.get("per_category")),
                               bool(conds.get("has_template")))
                or "（未挂数据源）")
            self._show_check(check)
            if (check.get("applied") or {}) or check.get("matched"):
                notify(self, "体检完成")
                self.levels_changed.emit()

        def _fail(msg) -> None:
            self.btn_check.setEnabled(True)
            self.lbl_state.setText(f"体检失败：{msg}")

        run_async(self, lambda: self._client.check_workorder(
            self._wid, apply=True), _done, _fail)
