"""图像画布：缩放/平移；标准图编辑焊点框，待检图显示缺陷与配准预览。"""

from __future__ import annotations

from typing import List, Optional, Set, Tuple

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QWidget

from . import theme
from ..core.models import DefectBox, ManualRoi
from ..metadata import DEFECT_BY_ID

_HANDLE_HIT_PX = 8.0
_HANDLE_DRAW_PX = 6.0
_MIN_RECT = 4.0
_MIN_RADIUS = 2.0

_RECT_HANDLES = ("nw", "n", "ne", "e", "se", "s", "sw", "w")
_CIRCLE_HANDLES = ("n", "e", "s", "w")

_CURSOR_FOR_HANDLE = {
    "nw": Qt.SizeFDiagCursor,
    "se": Qt.SizeFDiagCursor,
    "ne": Qt.SizeBDiagCursor,
    "sw": Qt.SizeBDiagCursor,
    "n": Qt.SizeVerCursor,
    "s": Qt.SizeVerCursor,
    "e": Qt.SizeHorCursor,
    "w": Qt.SizeHorCursor,
}

_BRIDGE_COLOR = "#FF8000"


def _bgr_to_qimage(img: np.ndarray) -> QImage:
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    rgb = np.ascontiguousarray(img[:, :, ::-1])
    h, w, _ = rgb.shape
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()


class ImageCanvas(QWidget):
    roi_created = Signal(object)       # ManualRoi：锡面单框新建/替换
    roi_changed = Signal()             # 锡面单框拖动/缩放
    roi_cleared = Signal()             # 锡面单框删除
    bridge_rois_changed = Signal()
    image_dropped = Signal(str)

    def __init__(self, title: str, editable: bool = False, parent=None):
        super().__init__(parent)
        self.title = title
        self.editable = editable
        self.setMinimumSize(320, 260)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAcceptDrops(True)

        self._pix: Optional[QPixmap] = None
        self._img_wh = (0, 0)
        self.scale = 1.0
        self.ox = 0.0
        self.oy = 0.0

        self.roi: Optional[ManualRoi] = None
        self.roi_selected = False
        self.bridge_rois: List[ManualRoi] = []
        self.bridge_selected: int = -1
        self.annotate_bridge: bool = False
        self.preview_roi: Optional[ManualRoi] = None
        self.preview_rois: List[ManualRoi] = []
        self.defects: List[DefectBox] = []
        self.visible_defect_ids: Optional[Set[int]] = None
        self._solder_overlay: Optional[QPixmap] = None

        self.tool = "select"
        self._drag_mode = None   # pan | create | move | resize | bridge_move | bridge_resize
        self._resize_handle: Optional[str] = None
        self._last_pos = None
        self._create_start = None
        self._temp_roi: Optional[ManualRoi] = None

    # ---- 数据接口 ----
    def set_image(self, img: Optional[np.ndarray]):
        if img is None:
            self._pix = None
            self._img_wh = (0, 0)
            self._solder_overlay = None
        else:
            self._pix = QPixmap.fromImage(_bgr_to_qimage(img))
            self._img_wh = (img.shape[1], img.shape[0])
            self._solder_overlay = None
            self.fit_to_view()
        self.update()

    def set_roi(self, roi: Optional[ManualRoi]):
        self.roi = roi
        self.roi_selected = False
        self.update()

    def clear_roi(self):
        self.roi = None
        self.roi_selected = False
        self.update()

    def set_bridge_rois(self, rois: List[ManualRoi]):
        self.bridge_rois = list(rois or [])
        self.bridge_selected = -1
        self.update()

    def clear_bridge_rois(self):
        self.bridge_rois = []
        self.bridge_selected = -1
        self.update()

    def set_annotate_bridge(self, enabled: bool):
        self.annotate_bridge = bool(enabled)
        if enabled:
            self.roi_selected = False
        else:
            self.bridge_selected = -1
        self.update()

    def set_preview_roi(self, roi: Optional[ManualRoi]):
        self.set_preview_rois([roi] if roi is not None else [])

    def set_preview_rois(self, rois: Optional[List[ManualRoi]]):
        self.preview_rois = [r for r in (rois or []) if r is not None]
        self.preview_roi = self.preview_rois[0] if self.preview_rois else None
        self.update()

    def set_solder_mask(self, mask: Optional[np.ndarray]):
        if mask is None or self._img_wh == (0, 0):
            self._solder_overlay = None
            self.update()
            return
        import cv2
        w, h = self._img_wh
        if mask.shape[0] != h or mask.shape[1] != w:
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        _, bin_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        overlay[bin_mask > 0] = (80, 220, 120, 70)
        contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bgr_a = overlay[:, :, [2, 1, 0, 3]].copy()
        cv2.drawContours(bgr_a, contours, -1, (0, 0, 255, 220), 1)
        rgba = np.ascontiguousarray(bgr_a[:, :, [2, 1, 0, 3]])
        qimg = QImage(rgba.data, w, h, 4 * w, QImage.Format_RGBA8888).copy()
        self._solder_overlay = QPixmap.fromImage(qimg)
        self.update()

    def set_defects(self, defects: List[DefectBox]):
        self.defects = defects
        self.update()

    def set_visible_defect_ids(self, ids: Optional[Set[int]]):
        self.visible_defect_ids = ids
        self.update()

    def set_tool(self, tool: str):
        self.tool = tool
        self.setCursor(Qt.CrossCursor if tool in ("rect", "circle") else Qt.ArrowCursor)

    def has_image(self) -> bool:
        return self._pix is not None

    # ---- 视图变换 ----
    def fit_to_view(self):
        w, h = self._img_wh
        if w == 0 or h == 0:
            return
        m = 16
        sw = (self.width() - 2 * m) / w
        sh = (self.height() - 2 * m) / h
        self.scale = max(0.02, min(sw, sh))
        self.ox = (self.width() - w * self.scale) / 2.0
        self.oy = (self.height() - h * self.scale) / 2.0

    def to_img(self, pt: QPointF) -> QPointF:
        return QPointF((pt.x() - self.ox) / self.scale, (pt.y() - self.oy) / self.scale)

    def to_widget(self, x: float, y: float) -> QPointF:
        return QPointF(x * self.scale + self.ox, y * self.scale + self.oy)

    # ---- 拖放 ----
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        for url in e.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")):
                self.image_dropped.emit(path)
                break

    # ---- 手柄 ----
    def _handle_names(self, roi: ManualRoi) -> Tuple[str, ...]:
        return _CIRCLE_HANDLES if roi.shape == "circle" else _RECT_HANDLES

    def _handle_points_img(self, roi: ManualRoi) -> dict:
        x, y, w, h = roi.bounds()
        cx, cy = x + w / 2.0, y + h / 2.0
        pts = {
            "nw": (x, y), "n": (cx, y), "ne": (x + w, y),
            "e": (x + w, cy), "se": (x + w, y + h), "s": (cx, y + h),
            "sw": (x, y + h), "w": (x, cy),
        }
        return {k: pts[k] for k in self._handle_names(roi)}

    def _hit_handle_on(self, roi: ManualRoi, ip: QPointF) -> Optional[str]:
        hit_r = _HANDLE_HIT_PX / max(self.scale, 1e-6)
        best = None
        best_d2 = hit_r * hit_r
        for name, (hx, hy) in self._handle_points_img(roi).items():
            d2 = (ip.x() - hx) ** 2 + (ip.y() - hy) ** 2
            if d2 <= best_d2:
                best_d2 = d2
                best = name
        return best

    def _hit_bridge_at(self, ip: QPointF) -> int:
        """从上到下命中连锡框，返回索引，未命中 -1。"""
        for i in range(len(self.bridge_rois) - 1, -1, -1):
            if self.bridge_rois[i].contains(ip.x(), ip.y()):
                return i
        return -1

    def _active_edit_roi(self) -> Optional[ManualRoi]:
        if self.annotate_bridge and 0 <= self.bridge_selected < len(self.bridge_rois):
            return self.bridge_rois[self.bridge_selected]
        if self.roi_selected:
            return self.roi
        return None

    def _apply_resize(self, roi: ManualRoi, handle: str, ip: QPointF):
        if roi.shape == "circle":
            r = ((ip.x() - roi.cx) ** 2 + (ip.y() - roi.cy) ** 2) ** 0.5
            roi.r = max(_MIN_RADIUS, r)
            return
        x0, y0, x1, y1 = roi.x, roi.y, roi.x + roi.w, roi.y + roi.h
        px, py = ip.x(), ip.y()
        if "w" in handle:
            x0 = min(px, x1 - _MIN_RECT)
        if "e" in handle:
            x1 = max(px, x0 + _MIN_RECT)
        if "n" in handle:
            y0 = min(py, y1 - _MIN_RECT)
        if "s" in handle:
            y1 = max(py, y0 + _MIN_RECT)
        roi.x, roi.y = x0, y0
        roi.w, roi.h = x1 - x0, y1 - y0

    def _update_hover_cursor(self, ip: QPointF):
        if not self.editable or self.tool != "select":
            return
        if self.annotate_bridge and self.bridge_rois:
            idx = self.bridge_selected if self.bridge_selected >= 0 else self._hit_bridge_at(ip)
            if 0 <= idx < len(self.bridge_rois):
                handle = self._hit_handle_on(self.bridge_rois[idx], ip)
                if handle is not None:
                    self.setCursor(_CURSOR_FOR_HANDLE.get(handle, Qt.ArrowCursor))
                    return
                if self.bridge_rois[idx].contains(ip.x(), ip.y()):
                    self.setCursor(Qt.SizeAllCursor)
                    return
        if self.roi is not None:
            handle = self._hit_handle_on(self.roi, ip)
            if handle is not None and self.roi_selected:
                self.setCursor(_CURSOR_FOR_HANDLE.get(handle, Qt.ArrowCursor))
                return
            if self.roi.contains(ip.x(), ip.y()):
                self.setCursor(Qt.SizeAllCursor)
                return
        self.setCursor(Qt.ArrowCursor)

    # ---- 事件 ----
    def resizeEvent(self, e):
        if self._pix is not None and self.scale <= 0:
            self.fit_to_view()
        super().resizeEvent(e)

    def wheelEvent(self, e):
        if self._pix is None:
            return
        delta = e.angleDelta().y()
        factor = 1.0015 ** delta
        old = self.to_img(e.position())
        self.scale = max(0.02, min(40.0, self.scale * factor))
        self.ox = e.position().x() - old.x() * self.scale
        self.oy = e.position().y() - old.y() * self.scale
        self.update()

    def mousePressEvent(self, e):
        if self._pix is None:
            return
        self._last_pos = e.position()
        ip = self.to_img(e.position())

        if e.button() == Qt.MiddleButton:
            self._drag_mode = "pan"
            return

        if e.button() == Qt.LeftButton:
            if self.editable and self.tool in ("rect", "circle"):
                self._drag_mode = "create"
                self._create_start = ip
                self._temp_roi = None
                return

            if self.editable and self.tool == "select":
                if self.annotate_bridge and self.bridge_rois:
                    for i in range(len(self.bridge_rois) - 1, -1, -1):
                        roi = self.bridge_rois[i]
                        handle = self._hit_handle_on(roi, ip)
                        if handle is not None or roi.contains(ip.x(), ip.y()):
                            self.bridge_selected = i
                            self.roi_selected = False
                            if handle is not None:
                                self._drag_mode = "bridge_resize"
                                self._resize_handle = handle
                            else:
                                self._drag_mode = "bridge_move"
                            self.update()
                            return

                if self.roi is not None and not self.annotate_bridge:
                    handle = self._hit_handle_on(self.roi, ip)
                    if handle is not None:
                        self.roi_selected = True
                        self.bridge_selected = -1
                        self._drag_mode = "resize"
                        self._resize_handle = handle
                        self.update()
                        return
                    if self.roi.contains(ip.x(), ip.y()):
                        self.roi_selected = True
                        self.bridge_selected = -1
                        self._drag_mode = "move"
                        self.update()
                        return

            self.roi_selected = False
            self.bridge_selected = -1
            self._drag_mode = "pan"
            self.update()

    def mouseMoveEvent(self, e):
        if self._pix is None:
            return
        pos = e.position()
        ip = self.to_img(pos)

        if self._drag_mode is None:
            self._update_hover_cursor(ip)
            return

        if self._drag_mode == "pan" and self._last_pos is not None:
            self.ox += pos.x() - self._last_pos.x()
            self.oy += pos.y() - self._last_pos.y()
            self._last_pos = pos
            self.update()
        elif self._drag_mode == "create" and self._create_start is not None:
            self._temp_roi = self._make_roi(self._create_start, ip)
            self.update()
        elif self._drag_mode == "move" and self._last_pos is not None and self.roi is not None:
            self._translate_roi(self.roi, pos)
            self._last_pos = pos
            self.roi_changed.emit()
            self.update()
        elif self._drag_mode == "resize" and self._resize_handle and self.roi is not None:
            self._apply_resize(self.roi, self._resize_handle, ip)
            self.roi_changed.emit()
            self.update()
        elif self._drag_mode == "bridge_move" and self._last_pos is not None:
            roi = self._active_edit_roi()
            if roi is not None:
                self._translate_roi(roi, pos)
                self._last_pos = pos
                self.bridge_rois_changed.emit()
                self.update()
        elif self._drag_mode == "bridge_resize" and self._resize_handle:
            roi = self._active_edit_roi()
            if roi is not None:
                self._apply_resize(roi, self._resize_handle, ip)
                self.bridge_rois_changed.emit()
                self.update()

    def _translate_roi(self, roi: ManualRoi, pos: QPointF):
        dx = (pos.x() - self._last_pos.x()) / self.scale
        dy = (pos.y() - self._last_pos.y()) / self.scale
        if roi.shape == "circle":
            roi.cx += dx
            roi.cy += dy
        else:
            roi.x += dx
            roi.y += dy

    def mouseReleaseEvent(self, e):
        if self._drag_mode == "create" and self._temp_roi is not None:
            roi = self._temp_roi
            if roi.is_valid():
                if self.annotate_bridge:
                    self.bridge_rois.append(roi)
                    self.bridge_selected = len(self.bridge_rois) - 1
                    self.roi_selected = False
                    self.bridge_rois_changed.emit()
                else:
                    self.roi = roi
                    self.roi_selected = True
                    self.bridge_selected = -1
                    self.roi_created.emit(roi)
            self._temp_roi = None
        self._drag_mode = None
        self._resize_handle = None
        self._create_start = None
        self._last_pos = None
        self.update()

    def _make_roi(self, p0: QPointF, p1: QPointF) -> ManualRoi:
        if self.tool == "circle":
            r = ((p1.x() - p0.x()) ** 2 + (p1.y() - p0.y()) ** 2) ** 0.5
            return ManualRoi.make_circle(p0.x(), p0.y(), r)
        x0, y0 = min(p0.x(), p1.x()), min(p0.y(), p1.y())
        w, h = abs(p1.x() - p0.x()), abs(p1.y() - p0.y())
        return ManualRoi.make_rect(x0, y0, w, h)

    def delete_selected_roi(self) -> bool:
        if self.annotate_bridge and 0 <= self.bridge_selected < len(self.bridge_rois):
            del self.bridge_rois[self.bridge_selected]
            self.bridge_selected = -1
            self.update()
            self.bridge_rois_changed.emit()
            return True
        if self.roi is not None and self.roi_selected:
            self.roi = None
            self.roi_selected = False
            self.update()
            self.roi_cleared.emit()
            return True
        return False

    # ---- 绘制 ----
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.fillRect(self.rect(), QColor(theme.BG_0))

        if self._pix is None:
            self._draw_placeholder(p)
            self._draw_corner_title(p)
            return

        w, h = self._img_wh
        target = QRectF(self.ox, self.oy, w * self.scale, h * self.scale)
        p.drawPixmap(target, self._pix, QRectF(self._pix.rect()))

        if self._solder_overlay is not None:
            p.drawPixmap(target, self._solder_overlay, QRectF(self._solder_overlay.rect()))

        n_prev = len(self.preview_rois)
        for i, proi in enumerate(self.preview_rois):
            label = (
                f"焊点{i + 1}(配准映射)" if n_prev > 1
                else "预计检测区域(配准映射)"
            )
            self._draw_roi(p, proi, False, color=theme.ACCENT_Hi,
                           label=label, dashed=True)
        if self.roi is not None and not self.annotate_bridge:
            self._draw_roi(p, self.roi, self.roi_selected)

        for i, broi in enumerate(self.bridge_rois):
            self._draw_roi(
                p, broi, i == self.bridge_selected,
                color=_BRIDGE_COLOR, label=f"焊点{i + 1}")

        if self._temp_roi is not None:
            label = f"焊点{len(self.bridge_rois) + 1}" if self.annotate_bridge else "人工框选区域"
            color = _BRIDGE_COLOR if self.annotate_bridge else None
            self._draw_roi(p, self._temp_roi, True, temp=True, color=color, label=label)

        for d in self.defects:
            if self.visible_defect_ids is not None and d.defect_id not in self.visible_defect_ids:
                continue
            self._draw_defect(p, d)

        self._draw_corner_title(p)

    def _draw_roi(self, p: QPainter, roi: ManualRoi, selected: bool, temp: bool = False,
                 color: Optional[str] = None, label: str = "人工框选区域", dashed: bool = False):
        col = QColor(color or theme.ACCENT)
        pen = QPen(col, 2, Qt.DashLine if (temp or dashed) else Qt.SolidLine)
        pen.setCosmetic(True)
        p.setPen(pen)
        fill = QColor(col)
        fill.setAlpha(40 if selected else 20)
        p.setBrush(QBrush(fill))

        x, y, rw, rh = roi.bounds()
        tl = self.to_widget(x, y)
        rect = QRectF(tl.x(), tl.y(), rw * self.scale, rh * self.scale)
        if roi.shape == "circle":
            p.drawEllipse(rect)
        else:
            p.drawRect(rect)

        p.setBrush(QBrush(col))
        p.setPen(Qt.NoPen)
        f = QFont(); f.setPointSize(8); f.setBold(True); p.setFont(f)
        fm = p.fontMetrics()
        tw = fm.horizontalAdvance(label) + 10
        th = fm.height() + 4
        p.drawRoundedRect(QRectF(rect.left(), rect.top() - th - 2, tw, th), 4, 4)
        p.setPen(QColor("white"))
        p.drawText(QRectF(rect.left() + 5, rect.top() - th - 2, tw, th),
                   Qt.AlignVCenter | Qt.AlignLeft, label)

        if selected and not temp:
            p.setPen(QPen(QColor("white"), 1, Qt.DotLine))
            p.setBrush(Qt.NoBrush)
            p.drawRect(rect.adjusted(-2, -2, 2, 2))
            hs = _HANDLE_DRAW_PX
            p.setBrush(QBrush(QColor("white")))
            p.setPen(QPen(col, 1))
            for hx, hy in self._handle_points_img(roi).values():
                wp = self.to_widget(hx, hy)
                p.drawRect(QRectF(wp.x() - hs / 2, wp.y() - hs / 2, hs, hs))

    def _draw_defect(self, p: QPainter, d: DefectBox):
        meta = DEFECT_BY_ID.get(d.defect_id)
        col = QColor(meta.color if meta else theme.NG)
        pen = QPen(col, 2); pen.setCosmetic(True)
        p.setPen(pen); p.setBrush(Qt.NoBrush)
        tl = self.to_widget(d.x, d.y)
        rect = QRectF(tl.x(), tl.y(), d.width * self.scale, d.height * self.scale)
        p.drawRect(rect)
        tag = f"{(meta.name if meta else d.label)} {d.confidence:.2f}"
        f = QFont(); f.setPointSize(8); f.setBold(True); p.setFont(f)
        fm = p.fontMetrics()
        tw = fm.horizontalAdvance(tag) + 10; th = fm.height() + 4
        p.setBrush(QBrush(col)); p.setPen(Qt.NoPen)
        p.drawRoundedRect(QRectF(rect.left(), rect.bottom() + 2, tw, th), 4, 4)
        p.setPen(QColor("white"))
        p.drawText(QRectF(rect.left() + 5, rect.bottom() + 2, tw, th),
                   Qt.AlignVCenter | Qt.AlignLeft, tag)

    def _draw_placeholder(self, p: QPainter):
        p.setPen(QColor(theme.TEXT_MUTE))
        f = QFont(); f.setPointSize(11); p.setFont(f)
        p.drawText(self.rect(), Qt.AlignCenter, f"拖入或打开{self.title}")
        pen = QPen(QColor(theme.BORDER), 1, Qt.DashLine)
        p.setPen(pen); p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(self.rect().adjusted(14, 14, -14, -14), 12, 12)

    def _draw_corner_title(self, p: QPainter):
        f = QFont(); f.setPointSize(9); f.setBold(True); p.setFont(f)
        fm = p.fontMetrics()
        tw = fm.horizontalAdvance(self.title) + 16
        p.setBrush(QBrush(QColor(theme.BG_2))); p.setPen(Qt.NoPen)
        p.drawRoundedRect(QRectF(10, 10, tw, 22), 8, 8)
        p.setPen(QColor(theme.TEXT_DIM))
        p.drawText(QRectF(18, 10, tw, 22), Qt.AlignVCenter | Qt.AlignLeft, self.title)
