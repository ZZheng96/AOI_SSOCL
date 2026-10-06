"""通用图像画布：显示 + 缩放/平移 + 拖拽打开 + 可选形状标注（矩形/圆形/点）。

设计目标是同一个组件既能在检测页当只读画布（叠加缺陷框/差异热力图），
又能在标定弹窗里当标注画布（插件焊点画 1 个锡面区域矩形/圆形；贴片焊盘
画多个矩形；金手指点选多个种子点）。用"图层(layer)"抽象把这些差异都
收敛成同一套鼠标交互逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QWidget

from app.ui import theme

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
_POINT_RADIUS_IMG = 6  # 点标注在“图像坐标”下的半径（用于命中测试/绘制近似）


def _bgr_to_qimage(img: np.ndarray) -> QImage:
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    rgb = np.ascontiguousarray(img[:, :, ::-1])
    h, w, _ = rgb.shape
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()


def shape_bounds(shape: dict) -> tuple[float, float, float, float]:
    """返回 (x, y, w, h) 外接矩形（图像坐标）。"""
    kind = shape.get("shape")
    if kind == "circle":
        r = float(shape.get("r", 0))
        return float(shape["cx"]) - r, float(shape["cy"]) - r, 2 * r, 2 * r
    if kind == "point":
        r = _POINT_RADIUS_IMG
        return float(shape["x"]) - r, float(shape["y"]) - r, 2 * r, 2 * r
    return float(shape.get("x", 0)), float(shape.get("y", 0)), float(shape.get("w", 0)), float(shape.get("h", 0))


def shape_contains(shape: dict, x: float, y: float) -> bool:
    kind = shape.get("shape")
    if kind == "circle":
        dx, dy = x - shape["cx"], y - shape["cy"]
        return (dx * dx + dy * dy) ** 0.5 <= max(1.0, float(shape.get("r", 0)))
    if kind == "point":
        dx, dy = x - shape["x"], y - shape["y"]
        return (dx * dx + dy * dy) ** 0.5 <= _POINT_RADIUS_IMG
    bx, by, bw, bh = shape_bounds(shape)
    return bx <= x <= bx + bw and by <= y <= by + bh


def shape_is_valid(shape: dict) -> bool:
    kind = shape.get("shape")
    if kind == "circle":
        return float(shape.get("r", 0)) >= 3
    if kind == "point":
        return True
    return float(shape.get("w", 0)) >= 3 and float(shape.get("h", 0)) >= 3


def make_rect(x0: float, y0: float, x1: float, y1: float) -> dict:
    x, y = min(x0, x1), min(y0, y1)
    w, h = abs(x1 - x0), abs(y1 - y0)
    return {"shape": "rect", "x": x, "y": y, "w": w, "h": h}


def make_circle(cx: float, cy: float, x1: float, y1: float) -> dict:
    r = ((x1 - cx) ** 2 + (y1 - cy) ** 2) ** 0.5
    return {"shape": "circle", "cx": cx, "cy": cy, "r": r}


def make_point(x: float, y: float) -> dict:
    return {"shape": "point", "x": x, "y": y}


@dataclass
class LayerConfig:
    color: str
    shape: str = "rect"  # "rect" | "circle" | "point" | "rect_or_circle"
    multi: bool = True
    label: str = ""
    editable: bool = True
    faded: bool = False


@dataclass
class _Layer:
    config: LayerConfig
    shapes: list[dict] = field(default_factory=list)


class ImageCanvas(QWidget):
    image_dropped = Signal(str)
    shapes_changed = Signal(str)  # layer name
    view_changed = Signal()
    defect_clicked = Signal(int)

    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        self.title = title
        self.setMinimumSize(280, 220)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAcceptDrops(True)

        self._pix: QPixmap | None = None
        self._img_wh = (0, 0)
        self.scale = 1.0
        self.ox = 0.0
        self.oy = 0.0

        self._layers: dict[str, _Layer] = {}
        self._layer_order: list[str] = []
        self.active_layer: str | None = None
        self.selected_index: int | None = None  # 当前选中形状在 active_layer.shapes 中的下标
        self.draw_shape_kind = "rect"  # 当 active layer 是 rect_or_circle 时，当前画的是哪种

        self.placeholder_text: str | None = None

        self.defects: list[Any] = []  # 只读缺陷框叠加（对象需有 x/y/w/h/label，可选 color/index）
        self.diff_overlay: np.ndarray | None = None  # BGR 差异热力图，与底图等大小
        self.diff_alpha = 0.45
        self.readonly = False  # True 时禁止一切标注编辑，只能缩放/平移
        self.highlight_index: int | None = None
        self.hud_filename: str = ""
        self.hud_enabled: bool = False
        # view=仅浏览平移；draw=新建形状；move=选中并拖移；scale=选中后拖拽改尺寸
        # 默认 draw：兼容 CalibrationDialog 等既有标注页
        self.interaction_mode: str = "draw"

        self._drag_mode: str | None = None  # pan | create | move | scale
        self._last_pos: QPointF | None = None
        self._create_start: QPointF | None = None
        self._temp_shape: dict | None = None

    def set_interaction_mode(self, mode: str) -> None:
        mode = mode if mode in {"view", "draw", "move", "scale"} else "view"
        self.interaction_mode = mode
        self._drag_mode = None
        self._create_start = None
        self._temp_shape = None
        cursors = {
            "view": Qt.CursorShape.ArrowCursor,
            "draw": Qt.CursorShape.CrossCursor,
            "move": Qt.CursorShape.SizeAllCursor,
            "scale": Qt.CursorShape.SizeFDiagCursor,
        }
        self.setCursor(cursors.get(mode, Qt.CursorShape.ArrowCursor))
        self.update()

    # ---- 图层管理 ----
    def set_layer(self, name: str, config: LayerConfig, shapes: list[dict] | None = None) -> None:
        if name not in self._layers:
            self._layer_order.append(name)
        self._layers[name] = _Layer(config=config, shapes=list(shapes or []))
        self.update()

    def set_layer_shapes(self, name: str, shapes: list[dict]) -> None:
        if name in self._layers:
            self._layers[name].shapes = list(shapes)
            self.update()

    def get_layer_shapes(self, name: str) -> list[dict]:
        layer = self._layers.get(name)
        return list(layer.shapes) if layer else []

    def set_active_layer(self, name: str | None) -> None:
        self.active_layer = name
        self.selected_index = None
        if name and name in self._layers:
            cfg = self._layers[name].config
            if cfg.shape in ("rect", "circle"):
                self.draw_shape_kind = cfg.shape
        self.update()

    def set_draw_shape_kind(self, kind: str) -> None:
        self.draw_shape_kind = kind

    def clear_layer(self, name: str) -> None:
        if name in self._layers:
            self._layers[name].shapes.clear()
            self.selected_index = None
            self.update()

    def clear_all_layers(self) -> None:
        self._layers.clear()
        self._layer_order.clear()
        self.active_layer = None
        self.selected_index = None
        self.update()

    def delete_selected(self) -> bool:
        if self.active_layer is None or self.selected_index is None:
            return False
        layer = self._layers.get(self.active_layer)
        if layer is None or not (0 <= self.selected_index < len(layer.shapes)):
            return False
        del layer.shapes[self.selected_index]
        self.selected_index = None
        self.shapes_changed.emit(self.active_layer)
        self.update()
        return True

    def keyPressEvent(self, e) -> None:  # noqa: N802
        if e.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.delete_selected()
        else:
            super().keyPressEvent(e)

    def set_placeholder_text(self, text: str) -> None:
        self.placeholder_text = text
        self.update()

    # ---- 只读叠加 ----
    def set_defects(self, defects: list[Any], *, highlight_index: int | None = None) -> None:
        self.defects = defects or []
        self.highlight_index = highlight_index
        self.update()

    def set_highlight(self, index: int | None) -> None:
        self.highlight_index = index
        self.update()

    def set_hud(self, filename: str = "", *, enabled: bool | None = None) -> None:
        self.hud_filename = filename or ""
        if enabled is not None:
            self.hud_enabled = bool(enabled)
        self.update()

    def image_size(self) -> tuple[int, int]:
        return self._img_wh

    def zoom_percent(self) -> int:
        return int(round(self.scale * 100))

    def set_zoom_1to1(self) -> None:
        if self._pix is None:
            return
        w, h = self._img_wh
        self.scale = 1.0
        self.ox = (self.width() - w) / 2.0
        self.oy = (self.height() - h) / 2.0
        self.update()
        self.view_changed.emit()

    def zoom_to_rect(self, x: float, y: float, w: float, h: float, *, margin: float = 0.35) -> None:
        if self._pix is None or w <= 0 or h <= 0:
            return
        pad_w = max(8.0, w * margin)
        pad_h = max(8.0, h * margin)
        vw = max(1.0, float(self.width()))
        vh = max(1.0, float(self.height()))
        self.scale = max(0.05, min(12.0, min(vw / (w + 2 * pad_w), vh / (h + 2 * pad_h))))
        cx = x + w / 2.0
        cy = y + h / 2.0
        self.ox = vw / 2.0 - cx * self.scale
        self.oy = vh / 2.0 - cy * self.scale
        self.update()
        self.view_changed.emit()

    def set_diff_overlay(self, diff_bgr: np.ndarray | None, alpha: float = 0.45) -> None:
        self.diff_overlay = diff_bgr
        self.diff_alpha = alpha
        self.update()

    # ---- 图像 ----
    def set_image(self, img: np.ndarray | None) -> None:
        if img is None:
            self._pix = None
            self._img_wh = (0, 0)
        else:
            self._pix = QPixmap.fromImage(_bgr_to_qimage(img))
            self._img_wh = (img.shape[1], img.shape[0])
            self.fit_to_view()
        self.update()
        self.view_changed.emit()

    def has_image(self) -> bool:
        return self._pix is not None

    def fit_to_view(self) -> None:
        w, h = self._img_wh
        if w == 0 or h == 0:
            return
        m = 16
        sw = (self.width() - 2 * m) / w
        sh = (self.height() - 2 * m) / h
        self.scale = max(0.02, min(sw, sh))
        self.ox = (self.width() - w * self.scale) / 2.0
        self.oy = (self.height() - h * self.scale) / 2.0
        self.view_changed.emit()

    def to_img(self, pt: QPointF) -> QPointF:
        return QPointF((pt.x() - self.ox) / self.scale, (pt.y() - self.oy) / self.scale)

    def to_widget(self, x: float, y: float) -> QPointF:
        return QPointF(x * self.scale + self.ox, y * self.scale + self.oy)

    # ---- 拖放 ----
    def dragEnterEvent(self, e) -> None:  # noqa: N802
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e) -> None:  # noqa: N802
        for url in e.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith(_IMAGE_EXTS):
                self.image_dropped.emit(path)
                break

    # ---- 事件 ----
    def resizeEvent(self, e) -> None:  # noqa: N802
        if self._pix is not None and self.scale <= 0:
            self.fit_to_view()
        super().resizeEvent(e)

    def wheelEvent(self, e) -> None:  # noqa: N802
        if self._pix is None:
            return
        delta = e.angleDelta().y()
        factor = 1.0015 ** delta
        old = self.to_img(e.position())
        self.scale = max(0.02, min(40.0, self.scale * factor))
        self.ox = e.position().x() - old.x() * self.scale
        self.oy = e.position().y() - old.y() * self.scale
        self.update()
        self.view_changed.emit()

    def _active(self) -> _Layer | None:
        return self._layers.get(self.active_layer) if self.active_layer else None

    def mousePressEvent(self, e) -> None:  # noqa: N802
        if self._pix is None:
            return
        self.setFocus()
        self._last_pos = e.position()
        ip = self.to_img(e.position())

        if e.button() == Qt.MiddleButton:
            self._drag_mode = "pan"
            return

        if e.button() != Qt.LeftButton:
            return

        if self.interaction_mode == "view" and self.defects:
            ip = self.to_img(e.position())
            for i, d in enumerate(self.defects):
                x, y, w, h = float(getattr(d, "x", 0)), float(getattr(d, "y", 0)), float(getattr(d, "w", 0)), float(getattr(d, "h", 0))
                if x <= ip.x() <= x + w and y <= ip.y() <= y + h:
                    self.highlight_index = i
                    self.defect_clicked.emit(i)
                    self.update()
                    return

        layer = self._active()
        editable = (
            layer is not None
            and layer.config.editable
            and not self.readonly
            and self.interaction_mode in {"draw", "move", "scale"}
        )

        if editable:
            hit = None
            for i, shp in enumerate(layer.shapes):
                if shape_contains(shp, ip.x(), ip.y()):
                    hit = i

            if self.interaction_mode == "draw":
                kind = layer.config.shape
                if kind == "rect_or_circle":
                    kind = self.draw_shape_kind
                if kind in ("rect", "circle", "point"):
                    self._drag_mode = "create"
                    self._create_start = ip
                    self._temp_shape = make_point(ip.x(), ip.y()) if kind == "point" else None
                    self.selected_index = None
                    self.update()
                    return

            if self.interaction_mode == "move":
                if hit is not None:
                    self.selected_index = hit
                    self._drag_mode = "move"
                    self.update()
                    return
                self.selected_index = None
                self._drag_mode = "pan"
                self.update()
                return

            if self.interaction_mode == "scale":
                if hit is not None:
                    self.selected_index = hit
                    self._drag_mode = "scale"
                    self.update()
                    return
                self.selected_index = None
                self._drag_mode = "pan"
                self.update()
                return

        self.selected_index = None
        self._drag_mode = "pan"
        self.update()

    def mouseMoveEvent(self, e) -> None:  # noqa: N802
        if self._pix is None:
            return
        pos = e.position()
        ip = self.to_img(pos)

        if self._drag_mode == "pan" and self._last_pos is not None:
            self.ox += pos.x() - self._last_pos.x()
            self.oy += pos.y() - self._last_pos.y()
            self._last_pos = pos
            self.update()
        elif self._drag_mode == "create" and self._create_start is not None:
            layer = self._active()
            kind = layer.config.shape if layer else "rect"
            if kind == "rect_or_circle":
                kind = self.draw_shape_kind
            if kind == "circle":
                self._temp_shape = make_circle(self._create_start.x(), self._create_start.y(), ip.x(), ip.y())
            elif kind == "rect":
                self._temp_shape = make_rect(self._create_start.x(), self._create_start.y(), ip.x(), ip.y())
            self.update()
        elif self._drag_mode == "move" and self._last_pos is not None and self.selected_index is not None:
            layer = self._active()
            if layer is not None and 0 <= self.selected_index < len(layer.shapes):
                dx = (pos.x() - self._last_pos.x()) / self.scale
                dy = (pos.y() - self._last_pos.y()) / self.scale
                shp = layer.shapes[self.selected_index]
                if shp.get("shape") == "circle":
                    shp["cx"] += dx
                    shp["cy"] += dy
                elif shp.get("shape") == "point":
                    shp["x"] += dx
                    shp["y"] += dy
                else:
                    shp["x"] += dx
                    shp["y"] += dy
                self._last_pos = pos
                self.update()
        elif self._drag_mode == "scale" and self._last_pos is not None and self.selected_index is not None:
            layer = self._active()
            if layer is not None and 0 <= self.selected_index < len(layer.shapes):
                dx = (pos.x() - self._last_pos.x()) / self.scale
                dy = (pos.y() - self._last_pos.y()) / self.scale
                shp = layer.shapes[self.selected_index]
                if shp.get("shape") == "circle":
                    shp["r"] = max(3.0, float(shp.get("r", 0)) + (dx + dy) * 0.5)
                elif shp.get("shape") == "point":
                    pass
                else:
                    shp["w"] = max(3.0, float(shp.get("w", 0)) + dx)
                    shp["h"] = max(3.0, float(shp.get("h", 0)) + dy)
                self._last_pos = pos
                self.update()

    def mouseReleaseEvent(self, e) -> None:  # noqa: N802
        if self._drag_mode == "create":
            layer = self._active()
            shp = self._temp_shape
            if shp is None and layer is not None:
                kind = layer.config.shape
                if kind == "rect_or_circle":
                    kind = self.draw_shape_kind
                if kind == "point" and self._create_start is not None:
                    shp = make_point(self._create_start.x(), self._create_start.y())
            if layer is not None and shp is not None and shape_is_valid(shp):
                if not layer.config.multi:
                    layer.shapes.clear()
                layer.shapes.append(shp)
                self.selected_index = len(layer.shapes) - 1
                self.shapes_changed.emit(self.active_layer)
            self._temp_shape = None
        elif self._drag_mode in {"move", "scale"}:
            if self.active_layer:
                self.shapes_changed.emit(self.active_layer)
        self._drag_mode = None
        self._create_start = None
        self._last_pos = None
        self.update()

    # ---- 绘制 ----
    def paintEvent(self, _) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.fillRect(self.rect(), QColor(theme.BG_2))

        if self._pix is None:
            self._draw_placeholder(p)
            self._draw_corner_title(p)
            return

        w, h = self._img_wh
        target = QRectF(self.ox, self.oy, w * self.scale, h * self.scale)
        p.drawPixmap(target, self._pix, QRectF(self._pix.rect()))

        if self.diff_overlay is not None:
            diff_pix = QPixmap.fromImage(_bgr_to_qimage(self.diff_overlay))
            p.setOpacity(self.diff_alpha)
            p.drawPixmap(target, diff_pix, QRectF(diff_pix.rect()))
            p.setOpacity(1.0)

        for name in self._layer_order:
            layer = self._layers[name]
            for i, shp in enumerate(layer.shapes):
                selected = name == self.active_layer and i == self.selected_index
                faded = bool(getattr(layer.config, "faded", False)) and name != self.active_layer
                self._draw_shape(p, shp, layer.config.color, selected, layer.config.label, faded=faded)
        if self._temp_shape is not None:
            self._draw_shape(p, self._temp_shape, theme.ACCENT_Hi, True, "", dashed=True)

        for i, d in enumerate(self.defects):
            self._draw_defect(p, d, i)

        self._draw_corner_title(p)
        if self.hud_enabled:
            self._draw_hud(p)

    def _draw_shape(self, p: QPainter, shape: dict, color: str, selected: bool, label: str, dashed: bool = False, faded: bool = False) -> None:
        col = QColor(color)
        if faded:
            col.setAlpha(90)
        pen = QPen(col, 1 if faded else 2, Qt.DashLine if dashed or faded else Qt.SolidLine)
        pen.setCosmetic(True)
        p.setPen(pen)
        fill = QColor(col)
        fill.setAlpha(18 if faded else (40 if selected else 20))
        p.setBrush(QBrush(fill))

        kind = shape.get("shape")
        if kind == "point":
            center = self.to_widget(shape["x"], shape["y"])
            r = max(4.0, _POINT_RADIUS_IMG * self.scale * 0.6)
            p.drawEllipse(center, r, r)
            p.setPen(QPen(QColor("white"), 1))
            p.drawLine(QPointF(center.x() - r, center.y()), QPointF(center.x() + r, center.y()))
            p.drawLine(QPointF(center.x(), center.y() - r), QPointF(center.x(), center.y() + r))
            return

        x, y, rw, rh = shape_bounds(shape)
        tl = self.to_widget(x, y)
        rect = QRectF(tl.x(), tl.y(), rw * self.scale, rh * self.scale)
        if kind == "circle":
            p.drawEllipse(rect)
        else:
            p.drawRect(rect)

        if label:
            p.setBrush(QBrush(col))
            p.setPen(Qt.NoPen)
            f = QFont()
            f.setPointSize(8)
            f.setBold(True)
            p.setFont(f)
            fm = p.fontMetrics()
            tw = fm.horizontalAdvance(label) + 10
            th = fm.height() + 4
            p.drawRoundedRect(QRectF(rect.left(), rect.top() - th - 2, tw, th), 4, 4)
            p.setPen(QColor("white"))
            p.drawText(QRectF(rect.left() + 5, rect.top() - th - 2, tw, th), Qt.AlignVCenter | Qt.AlignLeft, label)

        if selected:
            p.setPen(QPen(QColor("white"), 1, Qt.DotLine))
            p.setBrush(Qt.NoBrush)
            p.drawRect(rect.adjusted(-2, -2, 2, 2))

    def _draw_defect(self, p: QPainter, d: Any, index: int = 0) -> None:
        hex_color = getattr(d, "color", None) or theme.NG
        col = QColor(str(hex_color))
        if not col.isValid():
            col = QColor(theme.NG)
        highlighted = self.highlight_index is not None and index == self.highlight_index
        pen = QPen(col, 3 if highlighted else 2)
        pen.setCosmetic(True)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        x = float(getattr(d, "x", 0))
        y = float(getattr(d, "y", 0))
        w = float(getattr(d, "w", 0))
        h = float(getattr(d, "h", 0))
        tl = self.to_widget(x, y)
        rect = QRectF(tl.x(), tl.y(), w * self.scale, h * self.scale)
        p.drawRect(rect)

        num = getattr(d, "index", None)
        if num is None:
            num = index + 1
        badge = str(num)
        f = QFont()
        f.setPointSize(9)
        f.setBold(True)
        p.setFont(f)
        fm = p.fontMetrics()
        tw = fm.horizontalAdvance(badge) + 8
        th = fm.height() + 2
        p.setBrush(QBrush(col))
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(QRectF(rect.left() - 1, rect.top() - th - 2, tw, th), 4, 4)
        p.setPen(QColor("white"))
        p.drawText(QRectF(rect.left() + 3, rect.top() - th - 2, tw, th), Qt.AlignVCenter | Qt.AlignLeft, badge)

        tag = getattr(d, "label", "") or ""
        if not tag:
            return
        f.setPointSize(8)
        p.setFont(f)
        fm = p.fontMetrics()
        lw = fm.horizontalAdvance(tag) + 10
        lh = fm.height() + 4
        p.setBrush(QBrush(col))
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(QRectF(rect.left(), rect.bottom() + 2, lw, lh), 4, 4)
        p.setPen(QColor("white"))
        p.drawText(QRectF(rect.left() + 5, rect.bottom() + 2, lw, lh), Qt.AlignVCenter | Qt.AlignLeft, tag)

    def _draw_hud(self, p: QPainter) -> None:
        w, h = self._img_wh
        zoom = self.zoom_percent()
        name = self.hud_filename or ""
        parts = [p for p in (name, f"{w}×{h}" if w and h else "", f"{zoom}%") if p]
        text = "  ·  ".join(parts)
        if not text:
            return
        f = QFont()
        f.setPointSize(9)
        p.setFont(f)
        fm = p.fontMetrics()
        tw = fm.horizontalAdvance(text) + 16
        th = fm.height() + 8
        box = QRectF(8, self.height() - th - 8, min(tw, self.width() - 16), th)
        p.setBrush(QBrush(QColor(255, 255, 255, 220)))
        p.setPen(QPen(QColor(theme.BORDER), 1))
        p.drawRoundedRect(box, 6, 6)
        p.setPen(QColor(theme.TEXT_DIM))
        p.drawText(box.adjusted(8, 0, -8, 0), Qt.AlignVCenter | Qt.AlignLeft, text)

    def _draw_placeholder(self, p: QPainter) -> None:
        p.fillRect(self.rect(), QColor(theme.BG_2))
        pen = QPen(QColor(theme.BORDER), 1, Qt.DashLine)
        p.setPen(pen)
        p.setBrush(QColor(theme.BG_1))
        box = self.rect().adjusted(18, 18, -18, -18)
        p.drawRoundedRect(box, 14, 14)

        p.setPen(QColor(theme.TEXT_MUTE))
        f = QFont()
        f.setPointSize(11)
        p.setFont(f)
        text = self.placeholder_text or f"拖入图片 或 点击上方按钮打开{self.title}"
        p.drawText(box, Qt.AlignCenter, text)

    def _draw_corner_title(self, p: QPainter) -> None:
        if not self.title:
            return
        f = QFont()
        f.setPointSize(9)
        f.setBold(True)
        p.setFont(f)
        fm = p.fontMetrics()
        tw = fm.horizontalAdvance(self.title) + 16
        th = fm.height() + 6
        p.setBrush(QBrush(QColor(theme.BG_1)))
        p.setPen(QPen(QColor(theme.BORDER), 1))
        p.drawRoundedRect(QRectF(8, 8, tw, th), 6, 6)
        p.setPen(QColor(theme.TEXT_DIM))
        p.drawText(QRectF(16, 8, tw, th), Qt.AlignVCenter | Qt.AlignLeft, self.title)
