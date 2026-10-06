"""检测画布：原图 + ROI 虚线 + NG 实线编号 + HUD + 定位缩放。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from app.detect.types import DetectSummary
from app.ui.defect_table import OverlayDefect
from app.ui.image_canvas import ImageCanvas, LayerConfig
from app.ui.widgets import Card


class InspectCanvas(QWidget):
    image_dropped = Signal(str)
    defect_clicked = Signal(int)

    def __init__(self, title: str = "检测图", parent=None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        hud = QHBoxLayout()
        hud.setSpacing(8)
        self.lbl_hud = QLabel("")
        self.lbl_hud.setProperty("class", "Hint")
        self.btn_1to1 = QPushButton("1:1")
        self.btn_1to1.setObjectName("Ghost")
        self.btn_fit = QPushButton("自适应")
        self.btn_fit.setObjectName("Ghost")
        hud.addWidget(self.lbl_hud, 1)
        hud.addWidget(self.btn_1to1)
        hud.addWidget(self.btn_fit)
        root.addLayout(hud)

        card = Card(title)
        self.canvas = ImageCanvas("")
        self.canvas.setMinimumSize(320, 240)
        self.canvas.set_interaction_mode("view")
        self.canvas.hud_enabled = True
        self.canvas.setAcceptDrops(True)
        card.body.addWidget(self.canvas, 1)
        root.addWidget(card, 1)

        self.btn_fit.clicked.connect(self.fit)
        self.btn_1to1.clicked.connect(self.zoom_1to1)
        self.canvas.image_dropped.connect(self.image_dropped.emit)
        self.canvas.defect_clicked.connect(self._on_defect_clicked)
        self.canvas.view_changed.connect(self._refresh_hud)
        self._filename = ""

    def set_placeholder(self, text: str) -> None:
        self.canvas.set_placeholder_text(text)

    def set_image(self, img: np.ndarray | None, *, filename: str = "") -> None:
        self._filename = Path(filename).name if filename else ""
        self.canvas.set_hud(self._filename, enabled=True)
        self.canvas.set_image(img)
        self._refresh_hud()

    def set_rois(self, region_sets: dict[str, list[dict]] | None, *, colors: dict[str, str] | None = None) -> None:
        self.canvas.clear_all_layers()
        if not region_sets:
            return
        colors = colors or {}
        default = {"body": "#0f766e", "th": "#15803d", "smt_pads": "#ca8a04", "smt_toe": "#ca8a04", "smt_rim": "#e06c75", "gold": "#2563eb"}
        for layer, shapes in region_sets.items():
            if not shapes:
                continue
            shape_kind = "point" if layer == "gold" else "rect_or_circle"
            self.canvas.set_layer(
                layer,
                LayerConfig(color=colors.get(layer, default.get(layer, "#8a93a3")), shape=shape_kind, editable=False, faded=True, label=""),
                list(shapes),
            )

    def set_defects(self, defects: Iterable[Any], *, highlight_index: int | None = None) -> None:
        self.canvas.set_defects(list(defects or []), highlight_index=highlight_index)

    def highlight(self, index: int | None) -> None:
        self.canvas.set_highlight(index)
        if index is None:
            return
        defects = self.canvas.defects
        if 0 <= index < len(defects):
            d = defects[index]
            self.canvas.zoom_to_rect(float(d.x), float(d.y), float(d.w), float(d.h))

    def bind_job(
        self,
        image: np.ndarray | None,
        summary: DetectSummary | None,
        overlays: list[OverlayDefect] | None = None,
        *,
        filename: str = "",
        region_sets: dict | None = None,
        highlight_index: int | None = None,
    ) -> None:
        self.set_image(image, filename=filename)
        self.set_rois(region_sets)
        self.set_defects(overlays or [], highlight_index=highlight_index)

    def fit(self) -> None:
        if self.canvas.has_image():
            self.canvas.fit_to_view()
            self.canvas.update()
            self._refresh_hud()

    def zoom_1to1(self) -> None:
        self.canvas.set_zoom_1to1()
        self._refresh_hud()

    def has_image(self) -> bool:
        return self.canvas.has_image()

    def _on_defect_clicked(self, index: int) -> None:
        self.highlight(index)
        self.defect_clicked.emit(index)

    def _refresh_hud(self) -> None:
        w, h = self.canvas.image_size()
        zoom = self.canvas.zoom_percent()
        parts = [p for p in (self._filename, f"{w}×{h}" if w and h else "", f"{zoom}%") if p]
        self.lbl_hud.setText("  ·  ".join(parts) if parts else "无图像")
