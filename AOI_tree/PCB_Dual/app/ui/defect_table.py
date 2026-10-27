"""L2 缺陷表：类型、位置、数量；点行高亮并放大对应框。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import json

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
    QVBoxLayout,
)

from app.detect.scheduler import merge_defect_overlays
from app.detect.types import DetectSummary
from app.ui import theme


@dataclass
class OverlayDefect:
    x: int
    y: int
    w: int
    h: int
    label: str
    color: str = ""
    index: int = 0
    item_id: str = ""
    score: float | None = None
    roi_id: str | None = None


class DefectTable(QWidget):
    defect_selected = Signal(int)  # overlay index; -1 = 清除

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.overlays: list[OverlayDefect] = []
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.table = QTableWidget(0, 10)
        self.table.setHorizontalHeaderLabels(
            ["#", "类型", "位置", "尺寸", "算法来源", "ROI", "状态", "耗时 ms", "分数", "原因 / 原始结果"]
        )
        header = self.table.horizontalHeader()
        header.setMinimumSectionSize(36)
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        for column in (0, 2, 3, 5, 6, 7, 8):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(9, QHeaderView.ResizeMode.Stretch)
        self.table.setWordWrap(True)
        self.table.setTextElideMode(Qt.TextElideMode.ElideNone)
        header.setStretchLastSection(False)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        lay.addWidget(self.table)
        self.table.currentCellChanged.connect(self._on_row)

    def clear(self) -> None:
        self.overlays = []
        self.table.setRowCount(0)

    def bind_summary(self, summary: DetectSummary | None, *, error: str = "") -> list[OverlayDefect]:
        self.table.blockSignals(True)
        self.clear()
        if error:
            self.table.setRowCount(1)
            item = QTableWidgetItem(error)
            item.setForeground(QColor(theme.ERROR))
            self.table.setItem(0, 1, item)
            self.table.blockSignals(False)
            return []
        if summary is None:
            self.table.blockSignals(False)
            return []
        merged = merge_defect_overlays(summary.results)
        self.overlays = [
            OverlayDefect(
                x=int(m["x"]),
                y=int(m["y"]),
                w=int(m["w"]),
                h=int(m["h"]),
                label=str(m.get("label") or ""),
                color=str(m.get("color") or theme.NG),
                index=int(m.get("index") or i + 1),
                item_id=str(m.get("item_id") or ""),
                score=m.get("score"),
                roi_id=m.get("roi_id"),
            )
            for i, m in enumerate(merged)
        ]
        status_rows = [r for r in summary.results if r.status in {"ERROR", "REVIEW", "SKIP"}
                       or (r.status == "NG" and not r.boxes)]
        self.table.setRowCount(len(self.overlays) + len(status_rows))
        for i, d in enumerate(self.overlays):
            num = QTableWidgetItem(str(d.index))
            typ = QTableWidgetItem(d.label)
            pos = QTableWidgetItem(f"{d.x},{d.y}")
            size = QTableWidgetItem(f"{d.w}×{d.h}")
            color = QColor(d.color or theme.NG)
            num.setForeground(color)
            typ.setForeground(color)
            for col, item in enumerate((num, typ, pos, size)):
                item.setData(Qt.ItemDataRole.UserRole, i)
                self.table.setItem(i, col, item)
            sources = merged[i].get("sources", [merged[i]])
            values = [
                " · ".join(dict.fromkeys(s.get("item_id") or "未知" for s in sources)),
                " · ".join(dict.fromkeys(s.get("roi_id") or "未提供" for s in sources)),
                "NG", " / ".join(str(s.get("elapsed_ms", 0)) for s in sources),
                "不可用（未标定置信度）",
                " · ".join(dict.fromkeys(s.get("reason") or "" for s in sources)),
            ]
            details = json.dumps(sources, ensure_ascii=False, indent=2, default=str)
            for col, value in enumerate(values, start=4):
                item = QTableWidgetItem(value)
                item.setToolTip(details)
                self.table.setItem(i, col, item)
        for row, r in enumerate(status_rows, start=len(self.overlays)):
            snapshot = r.metadata.get("request_snapshot") or {}
            rois = snapshot.get("rois") or []
            reason = " · ".join(p for p in (r.error_code, r.error_message or r.skip_reason or r.message) if p)
            values = ["—", r.display_name or r.algorithm, "—", "—", r.item_id,
                      " · ".join(str(roi.get("id") or "未提供") for roi in rois) or "未提供",
                      r.status, str(r.elapsed_ms), "不可用", reason]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(json.dumps(snapshot, ensure_ascii=False, indent=2, default=str))
                self.table.setItem(row, col, item)
        self.table.blockSignals(False)
        return list(self.overlays)

    def select_index(self, index: int) -> None:
        if 0 <= index < self.table.rowCount():
            self.table.blockSignals(True)
            self.table.selectRow(index)
            self.table.blockSignals(False)

    def _on_row(self, row: int, _col: int, _prev_row: int, _prev_col: int) -> None:
        if row < 0 or row >= len(self.overlays):
            self.defect_selected.emit(-1)
            return
        self.defect_selected.emit(row)
