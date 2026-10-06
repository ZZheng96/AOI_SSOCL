"""历史（质量台）：表 + 共用画布/缺陷表；默认过滤调试记录。"""
from __future__ import annotations

import os
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.result.store import ResultStore
from app.ui import theme
from app.ui.defect_table import DefectTable
from app.ui.inspect_canvas import InspectCanvas
from app.ui.result_hud import ResultHud
from app.ui.widgets import Card, ToolbarStrip
from app.utils.cv_io import imread_unicode
from app.detect.types import AlgorithmResult, DefectBox, DetectSummary


def _summary_from_record(record: dict) -> DetectSummary:
    results = []
    for item in record.get("results") or []:
        boxes = []
        for b in item.get("boxes") or []:
            boxes.append(
                DefectBox(
                    x=int(b.get("x") or 0),
                    y=int(b.get("y") or 0),
                    w=int(b.get("w") or 0),
                    h=int(b.get("h") or 0),
                    label=str(b.get("label") or ""),
                    score=b.get("score"),
                    roi_id=b.get("roi_id"),
                    color=str(b.get("color") or ""),
                    item_id=str(b.get("item_id") or item.get("algorithm") or ""),
                )
            )
        results.append(
            AlgorithmResult(
                algorithm=str(item.get("algorithm") or ""),
                ok=bool(item.get("ok", True)),
                message=str(item.get("message") or ""),
                boxes=boxes,
                skipped=bool(item.get("skipped")),
                skip_reason=item.get("skip_reason"),
                item_id=str(item.get("item_id") or item.get("algorithm") or ""),
                display_name=str(item.get("display_name") or ""),
                status=str(item.get("status") or ""),
                defect_type=str(item.get("defect_type") or ""),
                defect_count=int(item.get("defect_count") or len(boxes)),
                elapsed_ms=int(item.get("elapsed_ms") or 0),
                algorithm_version=str(item.get("algorithm_version") or ""),
                error_code=item.get("error_code"),
                error_message=item.get("error_message"),
                manual_verdict=item.get("manual_verdict"),
            )
        )
    overall = str(record.get("overall") or "NG")
    return DetectSummary(
        overall_ok=overall == "OK",
        overall=overall,
        results=results,
        elapsed_ms=int(record.get("elapsed_ms") or 0),
        ng_count=int(record.get("ng_count") or 0),
        template_id=record.get("template_id"),
        template_version=record.get("template_version"),
        source=str(record.get("source") or "inspect"),
        gate_blocked=bool(record.get("gate_blocked")),
        gate_message=str(record.get("gate_message") or ""),
    )


class HistoryPage(QWidget):
    def __init__(self, result_store: ResultStore, parent=None) -> None:
        super().__init__(parent)
        self.result_store = result_store
        self.records: list[dict] = []
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(10)

        filters = ToolbarStrip()
        self.combo_filter = QComboBox()
        self.combo_filter.addItems(["全部", "OK", "NG", "GRAY", "ERROR", "复判"])
        self.combo_template = QComboBox()
        self.combo_template.setMinimumWidth(160)
        self.combo_template.addItem("全部模板", "")
        self.edit_keyword = QLineEdit()
        self.edit_keyword.setPlaceholderText("关键字（图号/模板）")
        self.chk_debug = QCheckBox("调试记录")
        self.btn_refresh = QPushButton("刷新")
        self.btn_refresh.setObjectName("Ghost")
        self.btn_open_dir = QPushButton("打开文件夹")
        self.btn_open_dir.setObjectName("Ghost")
        self.btn_delete = QPushButton("删除")
        self.btn_delete.setObjectName("Danger")
        filters.layout_.addWidget(QLabel("判定"))
        filters.layout_.addWidget(self.combo_filter)
        filters.layout_.addWidget(QLabel("模板"))
        filters.layout_.addWidget(self.combo_template)
        filters.layout_.addWidget(self.edit_keyword, 1)
        filters.layout_.addWidget(self.chk_debug)
        filters.layout_.addWidget(self.btn_refresh)
        filters.layout_.addWidget(self.btn_open_dir)
        filters.layout_.addWidget(self.btn_delete)
        root.addWidget(filters)

        self.lbl_stats = QLabel("")
        self.lbl_stats.setProperty("class", "Hint")
        root.addWidget(self.lbl_stats)

        splitter = QSplitter(Qt.Horizontal)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["时间", "模板", "图号", "判定", "复判", "NG 摘要", "耗时"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setAlternatingRowColors(True)
        splitter.addWidget(self.table)

        right = QWidget()
        rlay = QVBoxLayout(right)
        rlay.setContentsMargins(0, 0, 0, 0)
        rlay.setSpacing(8)
        self.result_hud = ResultHud()
        rlay.addWidget(self.result_hud)
        self.canvas = InspectCanvas("历史图")
        self.canvas.set_placeholder("选择一条历史记录")
        rlay.addWidget(self.canvas, 2)
        card = Card("缺陷")
        self.defect_table = DefectTable()
        card.body.addWidget(self.defect_table, 1)
        rlay.addWidget(card, 1)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        root.addWidget(splitter, 1)

        self.btn_refresh.clicked.connect(self.refresh)
        self.btn_open_dir.clicked.connect(self.open_folder)
        self.btn_delete.clicked.connect(self.delete_record)
        self.combo_filter.currentTextChanged.connect(self.refresh)
        self.combo_template.currentIndexChanged.connect(self.refresh)
        self.edit_keyword.textChanged.connect(self.refresh)
        self.chk_debug.toggled.connect(self.refresh)
        self.table.currentCellChanged.connect(lambda *_: self.show_record())
        self.defect_table.defect_selected.connect(self.canvas.highlight)
        self.canvas.defect_clicked.connect(self.defect_table.select_index)

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self.refresh()

    def _rebuild_template_filter(self, all_records: list[dict]) -> None:
        current = self.combo_template.currentData()
        ids: list[str] = []
        for r in all_records:
            tid = str(r.get("template_id") or r.get("template_name") or "").strip()
            if tid and tid not in ids:
                ids.append(tid)
        self.combo_template.blockSignals(True)
        self.combo_template.clear()
        self.combo_template.addItem("全部模板", "")
        for tid in sorted(ids):
            self.combo_template.addItem(tid, tid)
        if current:
            idx = self.combo_template.findData(current)
            if idx >= 0:
                self.combo_template.setCurrentIndex(idx)
        self.combo_template.blockSignals(False)

    HISTORY_LIMIT = 1000  # 单次最多加载的归档条数，防止 outputs 积压后全量扫描卡死页面

    def refresh(self) -> None:
        all_records = self.result_store.list_records(
            include_debug=self.chk_debug.isChecked(), limit=self.HISTORY_LIMIT)
        self._rebuild_template_filter(all_records)
        filt = self.combo_filter.currentText()
        tpl_filter = self.combo_template.currentData() or ""
        keyword = self.edit_keyword.text().strip().lower()
        records = []
        for r in all_records:
            overall = str(r.get("overall") or "")
            verdict = str(r.get("manual_overall_verdict") or "")
            if filt == "OK" and overall != "OK":
                continue
            if filt == "NG" and overall != "NG":
                continue
            if filt == "GRAY" and overall != "GRAY":
                continue
            if filt == "ERROR" and overall != "ERROR":
                continue
            if filt == "复判" and not verdict:
                continue
            tid = str(r.get("template_id") or r.get("template_name") or "")
            if tpl_filter and tid != tpl_filter:
                continue
            if keyword:
                blob = " ".join(
                    [
                        tid,
                        str(r.get("test_image") or ""),
                        str(r.get("time") or ""),
                    ]
                ).lower()
                if keyword not in blob:
                    continue
            records.append(r)
        self.records = records
        self._fill_table()
        self._fill_stats(all_records)
        self.result_hud.clear()
        self.canvas.set_image(None)
        self.defect_table.clear()

    def _fill_stats(self, all_records: list[dict]) -> None:
        today = datetime.now().strftime("%Y-%m-%d")
        shift = [r for r in all_records if str(r.get("time") or "").startswith(today)]
        ok = sum(1 for r in shift if r.get("overall") == "OK")
        ng = sum(1 for r in shift if r.get("overall") == "NG")
        yield_txt = f"{100 * ok / (ok + ng):.0f}%" if (ok + ng) else "—"
        types: Counter[str] = Counter()
        for r in shift:
            for item in r.get("ng_list") or []:
                types[str(item.get("defect_type") or item.get("display_name") or item.get("algorithm") or "")] += 1
        top = "、".join(f"{k}×{v}" for k, v in types.most_common(3) if k) or "—"
        cap = f"（仅加载最近 {self.HISTORY_LIMIT} 条）" if len(all_records) >= self.HISTORY_LIMIT else ""
        self.lbl_stats.setText(
            f"当班直通率 {yield_txt}  ·  Top 缺陷 {top}  ·  本页 {len(self.records)} 条{cap}")

    def _fill_table(self) -> None:
        self.table.setRowCount(len(self.records))
        colors = {"OK": theme.OK, "NG": theme.NG, "GRAY": theme.REVIEW, "ERROR": theme.ERROR}
        for i, r in enumerate(self.records):
            overall = str(r.get("overall") or "-")
            ver = r.get("template_version")
            tid = str(r.get("template_id") or r.get("template_name") or "-")
            if ver is not None:
                tid = f"{tid} v{ver}"
            ng_names = "、".join(
                str(x.get("defect_type") or x.get("display_name") or x.get("algorithm") or "")
                for x in (r.get("ng_list") or [])
            )
            vals = [
                str(r.get("time") or "-"),
                tid,
                Path(str(r.get("test_image") or "")).name or "-",
                overall,
                str(r.get("manual_overall_verdict") or "-"),
                ng_names or "-",
                f'{r.get("elapsed_ms") or "-"} ms' if r.get("elapsed_ms") else "-",
            ]
            for c, text in enumerate(vals):
                item = QTableWidgetItem(text)
                if c == 3:
                    item.setForeground(QColor(colors.get(overall, theme.TEXT)))
                self.table.setItem(i, c, item)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)

    def _current(self) -> dict | None:
        row = self.table.currentRow()
        if row < 0 or row >= len(self.records):
            return None
        return self.records[row]

    def show_record(self) -> None:
        r = self._current()
        if r is None:
            return
        summary = _summary_from_record(r)
        self.result_hud.bind_summary(
            summary,
            test_path=str(r.get("test_image") or ""),
            template_name=str(r.get("template_id") or r.get("template_name") or ""),
            template_version=r.get("template_version"),
            extra=str(r.get("manual_overall_verdict") or ""),
        )
        overlays = self.defect_table.bind_summary(summary)
        vis = Path(r["_dir"]) / r.get("vis_file", "result_vis.png")
        img = imread_unicode(vis) if vis.exists() else None
        if img is None:
            test = r.get("test_image")
            if test and Path(str(test)).exists():
                img = imread_unicode(test)
        self.canvas.bind_job(img, summary, overlays, filename=str(r.get("test_image") or ""))

    def open_folder(self) -> None:
        r = self._current()
        if not r:
            QMessageBox.information(self, "提示", "请先选择一条记录")
            return
        folder = r["_dir"]
        if sys.platform.startswith("win"):
            os.startfile(folder)  # type: ignore[attr-defined]
        elif sys.platform.startswith("darwin"):
            subprocess.Popen(["open", folder])
        else:
            subprocess.Popen(["xdg-open", folder])

    def delete_record(self) -> None:
        r = self._current()
        if not r:
            QMessageBox.information(self, "提示", "请先选择一条记录")
            return
        ret = QMessageBox.question(self, "确认删除", f"确定删除该记录？\n{r['_dir']}")
        if ret != QMessageBox.StandardButton.Yes:
            return
        self.result_store.delete_record(r["_dir"])
        self.refresh()
