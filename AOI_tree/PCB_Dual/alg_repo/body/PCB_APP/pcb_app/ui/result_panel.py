"""底部：检测结果（统计 + 缺陷明细表 + 图层显隐）。"""

from __future__ import annotations

from typing import List, Optional, Set

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QFrame, QHBoxLayout, QHeaderView, QLabel,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from . import theme
from .widgets import Card, StatusPill
from ..core.models import DefectBox, RunSummary
from ..metadata import DEFECT_BY_CODE

_COLS = ["瑕疵", "区域", "坐标(x,y)", "尺寸(w×h)", "置信度", "描述"]


class StatChip(QFrame):
    def __init__(self, label: str, color: str, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            f"background: {theme.BG_2}; border: 1px solid {theme.BORDER};"
            f" border-radius: 9px;")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 6, 12, 6)
        lay.setSpacing(8)
        self._num = QLabel("0")
        self._num.setStyleSheet(f"color: {color}; font-size: 18px; font-weight: 700;")
        cap = QLabel(label)
        cap.setStyleSheet(f"color: {theme.TEXT_DIM};")
        lay.addWidget(self._num)
        lay.addWidget(cap)

    def set(self, n):
        self._num.setText(str(n))


class ResultPanel(Card):
    locate = Signal(object)             # DefectBox
    visibility_changed = Signal(object)  # set[str] 或 None

    def __init__(self, parent=None):
        super().__init__("检测结果", parent)

        # 顶部统计条
        top = QWidget()
        tl = QHBoxLayout(top)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.setSpacing(10)
        self.chip_ng = StatChip("NG", theme.NG)
        self.chip_ok = StatChip("OK", theme.OK)
        self.chip_err = StatChip("异常", theme.WARN)
        tl.addWidget(self.chip_ng)
        tl.addWidget(self.chip_ok)
        tl.addWidget(self.chip_err)
        tl.addSpacing(8)
        self.time_lbl = QLabel("耗时 —")
        self.time_lbl.setStyleSheet(f"color: {theme.TEXT_DIM};")
        tl.addWidget(self.time_lbl)
        tl.addStretch(1)

        self._layer_box = QWidget()
        self._layer_l = QHBoxLayout(self._layer_box)
        self._layer_l.setContentsMargins(0, 0, 0, 0)
        self._layer_l.setSpacing(10)
        lb = QLabel("图层:")
        lb.setStyleSheet(f"color: {theme.TEXT_MUTE};")
        self._layer_l.addWidget(lb)
        tl.addWidget(self._layer_box)

        self.save_lbl = QLabel("")
        self.save_lbl.setStyleSheet(f"color: {theme.TEXT_MUTE}; font-size: 11px;")
        tl.addSpacing(8)
        tl.addWidget(self.save_lbl)
        self.add(top)

        # 表格
        self.table = QTableWidget(0, len(_COLS))
        self.table.setHorizontalHeaderLabels(_COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(len(_COLS) - 1, QHeaderView.Stretch)
        for i in range(len(_COLS) - 1):
            hh.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        self.table.setMinimumHeight(120)
        self.table.itemSelectionChanged.connect(self._on_select)
        self.add(self.table)

        self._defects: List[DefectBox] = []
        self._layer_checks = {}

    # ------------------------------------------------------------------ API --
    def clear(self):
        """清空结果：统计归零、表格清空、图层复选移除。"""
        self.chip_ng.set(0)
        self.chip_ok.set(0)
        self.chip_err.set(0)
        self.time_lbl.setText("耗时 —")
        self.save_lbl.setText("")
        self.table.setRowCount(0)
        for chk in self._layer_checks.values():
            chk.setParent(None)
        self._layer_checks.clear()
        self._defects = []

    def show_summary(self, summary: RunSummary):
        self.chip_ng.set(summary.ng_count)
        self.chip_ok.set(summary.ok_count)
        self.chip_err.set(summary.err_count)
        self.time_lbl.setText(f"耗时 {summary.total_ms:.0f} ms")
        if summary.session_dir:
            self.save_lbl.setText(f"已自动保存 → {summary.session_dir}")

        self._defects = summary.all_defects
        self.table.setRowCount(0)
        # 错误项也列出
        for it in summary.items:
            if it.status == "ERROR":
                self._append_error_row(it.defect_name, it.roi_name, it.error or "")
        for d in self._defects:
            self._append_defect_row(d)

        self._rebuild_layers(summary)

    def _append_defect_row(self, d: DefectBox):
        meta = DEFECT_BY_CODE.get(d.defect_code)
        row = self.table.rowCount()
        self.table.insertRow(row)
        name = meta.name if meta else d.label
        vals = [name, d.roi_name, f"{d.x}, {d.y}", f"{d.width}×{d.height}",
                f"{d.confidence:.2f}", d.description]
        for c, v in enumerate(vals):
            item = QTableWidgetItem(str(v))
            if c == 0 and meta:
                item.setForeground(Qt.white)
                item.setData(Qt.UserRole, d)
            self.table.setItem(row, c, item)
        # 存 DefectBox 到首列
        self.table.item(row, 0).setData(Qt.UserRole, d)

    def _append_error_row(self, name: str, roi_name: str, err: str):
        row = self.table.rowCount()
        self.table.insertRow(row)
        vals = [name or "—", roi_name, "—", "—", "ERR", err]
        for c, v in enumerate(vals):
            item = QTableWidgetItem(str(v))
            if c == 4:
                item.setForeground(Qt.yellow)
            self.table.setItem(row, c, item)

    def _on_select(self):
        items = self.table.selectedItems()
        if not items:
            return
        d = self.table.item(items[0].row(), 0).data(Qt.UserRole)
        if isinstance(d, DefectBox):
            self.locate.emit(d)

    def _rebuild_layers(self, summary: RunSummary):
        # 清空旧图层复选
        for chk in self._layer_checks.values():
            chk.setParent(None)
        self._layer_checks.clear()
        codes = []
        for d in self._defects:
            if d.defect_code not in codes:
                codes.append(d.defect_code)
        for code in codes:
            meta = DEFECT_BY_CODE.get(code)
            chk = QCheckBox(meta.name if meta else code)
            chk.setChecked(True)
            chk.toggled.connect(self._emit_visibility)
            self._layer_l.addWidget(chk)
            self._layer_checks[code] = chk

    def _emit_visibility(self):
        if not self._layer_checks:
            self.visibility_changed.emit(None)
            return
        vis = {c for c, chk in self._layer_checks.items() if chk.isChecked()}
        self.visibility_changed.emit(vis)
