"""AC-064 多框复核面板：列出候选/人工区域，逐框确认/拒绝/标争议。

候选框≠人工真值：仅 confirmed 进入框级训练；disputed 隔离不训。
走 GET/POST /detections/{id}/regions（修订链 + 乐观锁 + 事件幂等）。
"""
from __future__ import annotations

import uuid

from PySide6.QtWidgets import (
    QAbstractItemView, QDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
)

STATUS_TEXT = {"candidate": "候选(不训)", "confirmed": "已确认(可训)",
               "rejected": "已拒绝", "disputed": "争议(隔离)"}
COLUMNS = ("区域", "bbox", "来源", "峰值", "状态", "操作人")


class RegionReviewDialog(QDialog):
    def __init__(self, client, detection_id: int, parent=None):
        super().__init__(parent)
        self._client = client
        self._det_id = int(detection_id)
        self._review: dict = {}
        self.setWindowTitle(f"多框复核 · 检测 {self._det_id}")
        self.resize(720, 420)
        lay = QVBoxLayout(self)
        hint = QLabel("候选框不是人工真值，仅“已确认”框进入框级训练；"
                      "推翻他人结论需填写理由，否则标为争议并隔离。")
        hint.setWordWrap(True)
        hint.setProperty("subtext", True)
        lay.addWidget(hint)
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        lay.addWidget(self.table, 1)
        form = QFormLayout()
        self.edit_operator = QLineEdit()
        self.edit_operator.setPlaceholderText("操作员姓名/工号（必填）")
        form.addRow("操作员", self.edit_operator)
        self.edit_reason = QLineEdit()
        self.edit_reason.setPlaceholderText("理由（推翻他人结论/仲裁争议时必填）")
        form.addRow("理由", self.edit_reason)
        lay.addLayout(form)
        row = QHBoxLayout()
        self.buttons = {}
        for op, text in (("confirm", "确认"), ("reject", "拒绝"), ("dispute", "标争议")):
            b = QPushButton(text)
            b.clicked.connect(lambda _c=False, o=op: self.apply(o))
            row.addWidget(b)
            self.buttons[op] = b
        btn_refresh = QPushButton("刷新")
        btn_refresh.clicked.connect(self.refresh)
        row.addWidget(btn_refresh)
        lay.addLayout(row)
        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        lay.addWidget(self.lbl_status)
        self.refresh()

    def refresh(self) -> None:
        data = self._client.get(f"/detections/{self._det_id}/regions")
        if data is None:
            self.lbl_status.setText("加载区域失败")
            return
        self._review = data
        regions = data.get("regions") or []
        self.table.setRowCount(len(regions))
        for i, r in enumerate(regions):
            peak = r.get("score_peak")
            vals = (r.get("region_id"), r.get("bbox"), r.get("source"),
                    "-" if peak is None else f"{float(peak):.3f}",
                    STATUS_TEXT.get(r.get("status"), r.get("status")),
                    r.get("operator") or "-")
            for j, v in enumerate(vals):
                self.table.setItem(i, j, QTableWidgetItem("-" if v is None else str(v)))
        n_conf = sum(1 for r in regions if r.get("status") == "confirmed")
        self.lbl_status.setText(f"共 {len(regions)} 框，可训 {n_conf} 框，"
                                f"修订号 {data.get('revision', 0)}")

    def selected_region_id(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return None
        return (self._review.get("regions") or [])[rows[0].row()].get("region_id")

    def apply(self, op: str) -> bool:
        rid = self.selected_region_id()
        operator = self.edit_operator.text().strip()
        if rid is None or not operator:
            self.lbl_status.setText("请先选择区域并填写操作员")
            return False
        body = {"op": op, "region_id": rid, "operator": operator,
                "event_id": uuid.uuid4().hex,
                "expected_revision": int(self._review.get("revision", 0)),
                "reason": self.edit_reason.text().strip() or None}
        res = self._client.post(f"/detections/{self._det_id}/regions",
                                json=body, conflict_ok=True)
        if res is None or res.get("_status") == 409:
            detail = (res or {}).get("detail", "请查看错误提示")
            if res is not None:
                self.refresh()  # 乐观锁冲突：拉取最新修订号
            self.lbl_status.setText(f"提交失败：{detail}")
            return False
        self.refresh()
        return True
