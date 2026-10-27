"""开放告警只读证据及人工处置，不对历史样本重打分。"""
import json
import uuid

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                               QTableWidget, QTableWidgetItem, QDialog, QLineEdit,
                               QComboBox, QDialogButtonBox, QFormLayout)
from ui.pages.common import run_async, warn, PagerBar
from ui.widgets.image_viewer import ImageViewer

STATES = {"queued": "待处置", "confirmed_defect": "确认缺陷", "confirmed_normal": "确认正常",
          "template_candidate": "模板候选", "deferred": "暂缓", "ignored": "忽略"}


class OpenAlertQueue(QWidget):
    def __init__(self, client, workorder_id, parent=None):
        super().__init__(parent)
        self.client, self.workorder_id = client, workorder_id
        self.items = []
        self.page = 1
        root = QVBoxLayout(self)
        root.addWidget(QLabel("未知不等于缺陷。原分数/阈值未记录显示未知，不使用当前模型补推；处置不训练或发布。"))
        bar = QHBoxLayout()
        self.filter = QComboBox()
        self.filter.addItem("全部状态", "")
        for value, label in STATES.items():
            self.filter.addItem(label, value)
        self.filter.currentIndexChanged.connect(self._reset)
        bar.addWidget(self.filter)
        refresh = QPushButton("刷新开放告警样本")
        refresh.clicked.connect(self.reload)
        bar.addWidget(refresh)
        root.addLayout(bar)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["检测ID", "品类", "模型版本", "E_open 原分/阈值", "状态", "证据"])
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.cellDoubleClicked.connect(self._open)
        root.addWidget(self.table)
        self.pager = PagerBar(page_size=20)
        self.pager.page_changed.connect(self._page)
        root.addWidget(self.pager)
        root.addWidget(QLabel("双击样本查看原图、处置时间线并提交处置"))

    def _reset(self):
        self.page = 1
        self.reload()

    def _page(self, page):
        self.page = page
        self.reload()

    def reload(self):
        params = {"page": self.page, "page_size": 20, "status": self.filter.currentData() or None,
                  "workorder_id": self.workorder_id()}
        run_async(self, lambda: self.client.get("/api/review/open_alerts", params=params), self.fill)

    def fill(self, data):
        if not isinstance(data, dict):
            return
        self.items = data.get("items", [])
        self.table.setRowCount(len(self.items))
        for row, item in enumerate(self.items):
            values = [item["detection_id"], item["category"], item.get("model_version"),
                      f"{item.get('e_open_score')} / {item.get('e_open_threshold')}",
                      STATES[item["status"]], "旧证据未知" if item["evidence_status"] == "legacy_unknown" else "已记录"]
            for col, value in enumerate(values):
                self.table.setItem(row, col, QTableWidgetItem("未知" if value is None else str(value)))
        self.pager.set_total(data.get("total", 0))

    def _open(self, row, col):
        detection_id = self.items[row]["detection_id"]
        run_async(self, lambda: self.client.get(f"/api/review/open_alerts/{detection_id}"), self._detail)

    def _detail(self, item):
        if not isinstance(item, dict):
            return
        dlg = QDialog(self)
        dlg.setWindowTitle(f"开放告警样本 #{item['detection_id']} · 未知不等于缺陷")
        dlg.resize(760, 650)
        root = QVBoxLayout(dlg)
        viewer = ImageViewer()
        if item.get("image_path"):
            viewer.set_image_url(self.client.file_url(item["image_path"]))
        root.addWidget(viewer, 1)
        evidence = QLabel(json.dumps(item, ensure_ascii=False, indent=2))
        evidence.setWordWrap(True)
        evidence.setMaximumHeight(200)
        root.addWidget(evidence)
        form = QFormLayout()
        status, operator, reason = QComboBox(), QLineEdit("operator"), QLineEdit()
        for value, label in STATES.items():
            if value != "queued":
                status.addItem(label, value)
        form.addRow("处置", status)
        cause = QComboBox()   # AC-034 原因归类；新缺陷仅登记特征准入候选，不自动上线
        for value, label in ((None, "未归类"), ("drift", "漂移"), ("noise", "噪声"),
                             ("registration", "配准"), ("new_defect", "新缺陷（准入候选）"),
                             ("unknown", "未知")):
            cause.addItem(label, value)
        form.addRow("原因归类", cause)
        form.addRow("处置人", operator)
        form.addRow("原因（必填）", reason)
        root.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        root.addWidget(buttons)
        if dlg.exec() != QDialog.Accepted:
            return
        if not operator.text().strip() or not reason.text().strip():
            warn(self, "请填写处置人和原因")
            return
        payload = {"status": status.currentData(), "operator": operator.text(), "reason": reason.text(),
                   "event_id": uuid.uuid4().hex, "expected_revision": item["revision"]}
        if cause.currentData():
            payload["cause"] = cause.currentData()
        run_async(self, lambda: self.client.post(
            f"/api/review/open_alerts/{item['detection_id']}/disposition", json=payload),
            lambda result: self.reload() if result else None)
