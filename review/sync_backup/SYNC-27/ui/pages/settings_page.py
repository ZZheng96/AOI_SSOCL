"""系统设置中心（M16d，review R3）+ 新建品类向导（M16e，review R2）。

设置中心：GET /api/system/config 白名单项表单化展示（按 section 分组，
int/float 用数字框、bool 用勾选框、其余用文本框），保存时只回传改动项；
非白名单高危项（引擎槽位参数/安全密钥）仍走 yaml 手改。

新建品类向导：按操作流"定品类 → 定数据 → 定功能 → 冷启动 → 验收 → 投产"
给出引导清单，每一步一个直达按钮（跳转到对应导航页执行实际操作）。
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ui.api_client import ApiClient
from ui.pages.common import (
    make_banner,
    notify,
    run_async,
    warn,
)

# section 分组中文名
_SECTION_CN = {
    "pipeline": "推理管线",
    "self_learning": "在线学习",
    "evaluation": "评估",
    "notify": "告警通知",
    "archive": "自动归档",
    "plc": "PLC 触发",
    "logging": "日志",
}


class CategoryWizardDialog(QDialog):
    """新建品类向导（M16e）：把操作流开篇步骤做成引导清单。"""

    goto_page = Signal(str)
    levels_changed = Signal()   # U-workorder：工单提交成功后通知刷新品类数据条件

    _STEPS = [
        ("① 建工单", "填工单名并挂接数据源（数据条件由数据源属性自动聚合）；"
         "数据源在「数据管理」页新建（声明标注档位）并导入，"
         "导入后系统会体检并自动校正档位。", None),
        ("② 定数据", "导入预训练/冷启动数据：正常图（train/normal）≥3 张，"
         "缺陷图（anomaly）随意；缺陷不足可用「数据增强」合成。", "数据管理"),
        ("③ 准备模型", "按工单的数据条件自动选择建模方式；"
         "有模板图的品类先在数据管理设模板图。", "模型管理"),
        ("④ 冷启动训练", "模型管理 -> 准备模型（少样本冷启动，分钟级）。", "模型管理"),
        ("⑤ 评估验收", "评估看板 -> 精度验收（test 集 AUROC），达标再上线。", "评估看板"),
        ("⑥ 投产", "模型管理激活版本 -> 实时监控开始日常检测，"
         "异常帧人工反馈即进入在线学习。", "实时监控"),
    ]

    def __init__(self, client: ApiClient, parent: QWidget | None = None):
        super().__init__(parent)
        self._client = client
        self.setWindowTitle("新建品类上线向导")
        self.resize(560, 560)
        lay = QVBoxLayout(self)

        lay.addWidget(make_banner(
            "按操作流走一遍：建工单 -> 定数据 -> 准备模型 -> 冷启动 -> 验收 -> 投产"))

        form = QFormLayout()
        self.edit_name = QLineEdit()
        self.edit_name.setPlaceholderText("工单名，例如：3号线 SMT 质检 / 螺丝-A线")
        form.addRow("工单名", self.edit_name)
        self.chk_review = QCheckBox("人工复判（全部检测进待复核队列，"
                                    "复核后才计入统计与持续学习）")
        self.chk_review.setToolTip("复判开关：检测结果全部需人工复核，"
                                   "复核提交后自动生成反馈回流持续学习")
        form.addRow("复判", self.chk_review)
        self.edit_note = QLineEdit()
        self.edit_note.setPlaceholderText("工单备注（可选）")
        form.addRow("备注", self.edit_note)
        lay.addLayout(form)

        self.lbl_wo_state = QLabel("")
        self.lbl_wo_state.setProperty("subtext", True)
        self.lbl_wo_state.setWordWrap(True)
        lay.addWidget(self.lbl_wo_state)

        row_btn = QHBoxLayout()
        self.btn_submit = QPushButton("提交工单")
        self.btn_submit.setProperty("primary", True)
        self.btn_submit.setToolTip("登记工单与数据条件（导入数据后系统自动体检校正），"
                                   "提交后可直接去导数据")
        self.btn_submit.clicked.connect(self._on_submit)
        row_btn.addWidget(self.btn_submit)
        row_btn.addStretch(1)
        lay.addLayout(row_btn)

        for title, desc, page in self._STEPS:
            box = QGroupBox(title)
            b_lay = QVBoxLayout(box)
            d = QLabel(desc)
            d.setWordWrap(True)
            d.setProperty("subtext", True)
            b_lay.addWidget(d)
            if page:
                btn = QPushButton(f"去「{page}」执行")
                btn.setProperty("flat", True)
                btn.clicked.connect(
                    lambda _=False, p=page: self._goto(p))
                b_lay.addWidget(btn, 0, Qt.AlignRight)
            lay.addWidget(box)

        btns = QDialogButtonBox(QDialogButtonBox.Close)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

    def _on_submit(self) -> None:
        """提交工单（工单名 + 复判开关；数据条件由数据源聚合）到后端。"""
        cat = self.edit_name.text().strip()
        if not cat:
            self.lbl_wo_state.setText("请先填写工单名")
            return
        self.btn_submit.setEnabled(False)

        def _done(res) -> None:
            self.btn_submit.setEnabled(True)
            if isinstance(res, dict) and res.get("id"):
                # 响应为工单 item，体检结果在嵌套 check 字段内
                check = res.get("check") or {}
                cond_cn = str(res.get("conditions_cn") or "（未挂数据源）")
                extra = ""
                applied = check.get("applied") or {}
                if applied:  # 新建即体检：数据源声明档位按支撑自动校正
                    extra = "；体检已自动校正数据源标注档位"
                warns = [str(w) for w in (check.get("warnings") or [])]
                if warns:
                    extra += f"；提醒：{'；'.join(warns)}"
                self.lbl_wo_state.setText(
                    f"✓ 工单已登记：「{cat}」，数据条件「{cond_cn}」{extra}")
                notify(self, f"工单已登记：{cat}（{cond_cn}）")
                self.levels_changed.emit()
            else:
                self.lbl_wo_state.setText("提交失败，请重试")

        def _fail(msg) -> None:
            self.btn_submit.setEnabled(True)
            if "已存在" in str(msg):
                self.lbl_wo_state.setText(
                    f"工单「{cat}」已存在：可在「工作台」表格中双击改条件")
            else:
                self.lbl_wo_state.setText(f"提交失败：{msg}")

        run_async(self, lambda: self._client.create_workorder(
            cat, review_enabled=self.chk_review.isChecked(),
            note=self.edit_note.text().strip()), _done, _fail)

    def _goto(self, page: str) -> None:
        self.goto_page.emit(page)
        self.accept()


class SettingsPage(QWidget):
    """系统设置中心页。"""

    goto_page = Signal(str)  # 请求主窗口跳转导航页（供向导用）
    levels_changed = Signal()  # U-workorder：工单变更 -> 主窗口刷新品类数据条件
    config_saved = Signal()    # 配置保存成功 -> 主窗口重拉 system_info（预算等热更新）

    def __init__(self, client: ApiClient, parent: QWidget | None = None):
        super().__init__(parent)
        self._client = client
        self._editors: dict[str, QWidget] = {}  # "sec.key" → 编辑控件

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 10, 12, 10)
        outer.setSpacing(8)

        top = QHBoxLayout()
        top.addWidget(make_banner(
            "系统设置：白名单参数热修改（写回 yaml，无需重启）；"
            "引擎内部参数/安全密钥等高危项仍需手改配置文件"))
        top.addStretch(1)
        btn_wizard = QPushButton("新建品类向导")
        btn_wizard.setProperty("primary", True)
        btn_wizard.setToolTip("按操作流引导完成新品类上线（定数据/定功能/冷启动/验收）")
        btn_wizard.clicked.connect(self._open_wizard)
        top.addWidget(btn_wizard)
        btn_refresh = QPushButton("刷新")
        btn_refresh.clicked.connect(self.reload)
        top.addWidget(btn_refresh)
        self.btn_save = QPushButton("保存修改")
        self.btn_save.setProperty("success", True)
        self.btn_save.clicked.connect(self._on_save)
        top.addWidget(self.btn_save)
        outer.addLayout(top)

        self.lbl_path = QLabel("配置文件：-")
        self.lbl_path.setProperty("subtext", True)
        outer.addWidget(self.lbl_path)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self._form_host = QWidget()
        self._form_lay = QVBoxLayout(self._form_host)
        self._form_lay.setContentsMargins(0, 0, 0, 0)
        self._form_lay.setSpacing(8)
        scroll.setWidget(self._form_host)
        outer.addWidget(scroll, 1)

        # 系统自检（2026-10-09）：数据残留/引用悬空/页面一致性体检
        outer.addWidget(self._build_selfcheck_section())

        # 操作日志（审计）：自统计报表页迁入——报表页面向生产口径，
        # 审计功能归入设置中心
        outer.addWidget(self._build_logs_section())

        self._client.error_occurred.connect(lambda m: warn(self, m))

    # ══════════════════ 表单构建 ══════════════════
    def reload(self) -> None:
        run_async(self, self._client.get_config, self._fill)
        self.reload_logs()
        if hasattr(self, "table_check"):
            self.run_selfcheck()

    def _fill(self, data) -> None:
        if not data:
            return
        self.lbl_path.setText(f"配置文件：{data.get('config_path', '-')}")
        # 清空旧表单
        while self._form_lay.count():
            item = self._form_lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._editors.clear()

        groups: dict[str, list] = {}
        for it in data.get("items") or []:
            groups.setdefault(str(it.get("section", "")), []).append(it)
        for sec, items in groups.items():
            box = QGroupBox(_SECTION_CN.get(sec, sec))
            form = QFormLayout(box)
            for it in items:
                editor = self._make_editor(it)
                self._editors[f"{it['section']}.{it['key']}"] = editor
                lb = QLabel(str(it.get("name", it["key"])))
                lb.setToolTip(str(it.get("desc", "")))
                form.addRow(lb, editor)
            self._form_lay.addWidget(box)
        self._form_lay.addStretch(1)

    def _make_editor(self, it: dict) -> QWidget:
        typ = str(it.get("type", "str"))
        val = it.get("value")
        if typ == "bool":
            w = QCheckBox("启用")
            w.setChecked(bool(val))
            w.setToolTip(str(it.get("desc", "")))
            return w
        if typ == "int":
            w = QSpinBox()
            w.setRange(-1_000_000, 1_000_000)
            w.setValue(int(val or 0))
            return w
        if typ == "float":
            w = QDoubleSpinBox()
            w.setDecimals(3)
            w.setRange(-1e9, 1e9)
            w.setValue(float(val or 0.0))
            return w
        if typ == "list":
            # list 类型：逗号分隔编辑，保存时还原为真正的列表。
            # 此前落到 QLineEdit 纯文本，把 Python list 的字符串表示写回
            # yaml（events/decisions 被污染成字符串，前端反馈 v7-6）。
            if isinstance(val, (list, tuple)):
                text = ", ".join(str(x) for x in val)
            else:
                text = "" if val is None else str(val)
            w = QLineEdit(text)
            w.setMinimumWidth(280)
            w.setProperty("ctype", "list")
            w.setToolTip(str(it.get("desc", "")) + "（多项用英文逗号分隔）")
            return w
        w = QLineEdit("" if val is None else str(val))
        w.setMinimumWidth(280)
        return w

    def _editor_value(self, editor: QWidget):
        if isinstance(editor, QCheckBox):
            return editor.isChecked()
        if isinstance(editor, (QSpinBox, QDoubleSpinBox)):
            return editor.value()
        if editor.property("ctype") == "list":
            return [x.strip() for x in editor.text().split(",") if x.strip()]
        return editor.text()

    # ══════════════════ 保存 ══════════════════
    def _on_save(self) -> None:
        updates = {k: self._editor_value(w) for k, w in self._editors.items()}
        run_async(self, lambda: self._client.save_config(updates),
                  self._on_saved)

    def _on_saved(self, res) -> None:
        if not res:
            return
        applied = res.get("applied") or []
        rejected = res.get("rejected") or []
        msg = f"已应用 {len(applied)} 项。"
        if rejected:
            msg += f"\n被拒绝（非白名单/权限不足）：{', '.join(rejected)}"
        notify(self, msg)
        if applied:
            self.config_saved.emit()

    # ══════════════════ 系统自检 ══════════════════
    _STATUS_CN = {"ok": "正常", "warn": "警告", "fail": "严重"}

    def _build_selfcheck_section(self) -> QGroupBox:
        box = QGroupBox("系统自检")
        lay = QVBoxLayout(box)
        bar = QHBoxLayout()
        self.lbl_selfcheck = QLabel(
            "检查数据残留、引用悬空与页面一致性；发现问题可一键清理")
        self.lbl_selfcheck.setProperty("subtext", True)
        bar.addWidget(self.lbl_selfcheck, 1)
        self.btn_selfcheck = QPushButton("运行自检")
        self.btn_selfcheck.setProperty("primary", True)
        self.btn_selfcheck.setToolTip(
            "只读体检：孤儿检测记录、悬空反馈、图片文件缺失、"
            "认领卡死、空数据源、统计聚合偏差等")
        self.btn_selfcheck.clicked.connect(self.run_selfcheck)
        bar.addWidget(self.btn_selfcheck)
        self.btn_cleanup = QPushButton("一键清理")
        self.btn_cleanup.setProperty("success", True)
        self.btn_cleanup.setEnabled(False)
        self.btn_cleanup.setToolTip(
            "修复所有「可修复」项：级联删除孤儿/悬空检测与反馈、"
            "删除裂图登记行、释放卡死认领、重算今日统计")
        self.btn_cleanup.clicked.connect(self._on_cleanup)
        bar.addWidget(self.btn_cleanup)
        lay.addLayout(bar)
        self.table_check = QTableWidget(0, 4)
        self.table_check.setHorizontalHeaderLabels(
            ["检查项", "状态", "数量", "说明"])
        self.table_check.horizontalHeader().setStretchLastSection(True)
        self.table_check.verticalHeader().setVisible(False)
        self.table_check.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table_check.setMinimumHeight(150)
        self.table_check.setMaximumHeight(220)
        lay.addWidget(self.table_check)
        return box

    def run_selfcheck(self) -> None:
        self.btn_selfcheck.setEnabled(False)
        self.lbl_selfcheck.setText("自检中……")

        def _done(res) -> None:
            self.btn_selfcheck.setEnabled(True)
            if not res:
                self.lbl_selfcheck.setText("自检失败，请重试")
                return
            self._fill_selfcheck(res)

        def _fail(msg) -> None:
            self.btn_selfcheck.setEnabled(True)
            self.lbl_selfcheck.setText(f"自检失败：{msg}")

        run_async(self, self._client.system_selfcheck, _done, _fail)

    def _fill_selfcheck(self, res: dict) -> None:
        items = res.get("items") or []
        n_bad = int(res.get("n_issues") or 0)
        n_fixable = sum(1 for i in items if i.get("fixable"))
        self.lbl_selfcheck.setText(
            "全部正常，未发现数据残留或不一致" if n_bad == 0 else
            f"发现 {n_bad} 类问题（其中 {n_fixable} 类可一键清理）")
        self.table_check.setRowCount(len(items))
        for r, it in enumerate(items):
            status = str(it.get("status", "ok"))
            vals = [str(it.get("name", "")),
                    self._STATUS_CN.get(status, status),
                    str(it.get("count", 0)),
                    str(it.get("detail", ""))]
            for c, v in enumerate(vals):
                cell = QTableWidgetItem(v)
                if c == 1 and status != "ok":
                    cell.setForeground(Qt.red if status == "fail"
                                       else Qt.darkYellow)
                if c == 3:
                    cell.setToolTip(v)
                self.table_check.setItem(r, c, cell)
        self.btn_cleanup.setEnabled(n_fixable > 0)

    def _on_cleanup(self) -> None:
        ret = QMessageBox.question(
            self, "一键清理",
            "将级联删除孤儿/悬空的检测与反馈记录、删除无数据源归属的批次"
            "及其图片（仅删应用生成的文件，外部原始数据只删登记行）、"
            "删除文件已缺失的图片登记、释放卡死的图片认领，并重算今日统计。"
            "\n该操作不可恢复，确定继续？")
        if ret != QMessageBox.Yes:
            return
        self.btn_cleanup.setEnabled(False)

        def _done(res) -> None:
            if not res:
                warn(self, "清理失败，请重试")
                return
            fixed = res.get("fixed") or {}
            total = sum(int(v) for v in fixed.values())
            notify(self, f"清理完成，共处理 {total} 条记录")
            self.levels_changed.emit()
            self.run_selfcheck()  # 清理后复检，结果落表

        run_async(self, self._client.system_selfcheck_cleanup, _done)

    # ══════════════════ 操作日志（审计） ══════════════════
    _LOG_ACTIONS = [
        "feedback", "self_update", "self_update_consolidate",
        "activate_model", "model_prepare", "accuracy_eval",
        "learning_curve", "import_images", "create_workorder",
        "review_submit", "config_update", "alarm",
    ]

    def _build_logs_section(self) -> QGroupBox:
        box = QGroupBox("操作日志（审计）")
        lay = QVBoxLayout(box)
        bar = QHBoxLayout()
        bar.addWidget(QLabel("动作过滤"))
        self.combo_log_action = QComboBox()
        self.combo_log_action.setEditable(True)
        self.combo_log_action.addItem("全部", "")
        for a in self._LOG_ACTIONS:
            self.combo_log_action.addItem(a, a)
        self.combo_log_action.currentIndexChanged.connect(
            lambda _i: self.reload_logs())
        bar.addWidget(self.combo_log_action)
        btn_refresh = QPushButton("刷新日志")
        btn_refresh.clicked.connect(self.reload_logs)
        bar.addWidget(btn_refresh)
        btn_export = QPushButton("导出 CSV")
        btn_export.setToolTip("导出审计日志（时间/用户/动作/详情，CSV 带 BOM）")
        btn_export.clicked.connect(self._export_logs)
        bar.addWidget(btn_export)
        bar.addStretch(1)
        lay.addLayout(bar)
        self.table_logs = QTableWidget(0, 4)
        self.table_logs.setHorizontalHeaderLabels(["时间", "用户", "动作", "详情"])
        self.table_logs.horizontalHeader().setStretchLastSection(True)
        self.table_logs.verticalHeader().setVisible(False)
        self.table_logs.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table_logs.setMinimumHeight(180)
        self.table_logs.setMaximumHeight(260)
        lay.addWidget(self.table_logs)
        return box

    def _log_action_filter(self) -> str:
        """动作过滤：可编辑框，优先取用户手输文本，否则取下拉数据。"""
        text = self.combo_log_action.currentText().strip()
        data = str(self.combo_log_action.currentData() or "")
        return text if text and text != "全部" else data

    def reload_logs(self) -> None:
        if not hasattr(self, "table_logs"):
            return
        action = self._log_action_filter()
        run_async(self, lambda: self._client.list_logs(limit=200, action=action),
                  self._fill_logs)

    def _fill_logs(self, rows) -> None:
        if rows is None:
            return
        self.table_logs.setRowCount(len(rows))
        for r, it in enumerate(rows):
            vals = [str(it.get("created_at") or "-"),
                    str(it.get("user") or "-"),
                    str(it.get("action") or "-"),
                    str(it.get("detail") or "-")]
            for c, v in enumerate(vals):
                item = QTableWidgetItem(v)
                if c == 3:
                    item.setToolTip(v)
                self.table_logs.setItem(r, c, item)

    def _export_logs(self) -> None:
        import datetime
        default = f"aoi_audit_{datetime.date.today().isoformat()}.csv"
        path, _ = QFileDialog.getSaveFileName(
            self, "导出审计日志", default, "CSV 文件 (*.csv)")
        if not path:
            return
        action = self._log_action_filter()
        url = f"{self._client.base_url}/api/logs/export?limit=2000"
        if action:
            from urllib.parse import quote
            url += f"&action={quote(action)}"

        def _done(data) -> None:
            if not data:
                warn(self, "导出失败：未获取到数据")
                return
            with open(path, "wb") as f:
                f.write(data)
            notify(self, f"审计日志已导出：{path}")

        run_async(self, lambda: self._client.fetch_bytes(url, timeout=60), _done)

    # ══════════════════ 新建品类向导 ══════════════════
    def _open_wizard(self) -> None:
        dlg = CategoryWizardDialog(self._client, self)
        dlg.goto_page.connect(self.goto_page.emit)
        dlg.levels_changed.connect(self.levels_changed.emit)
        dlg.exec()
