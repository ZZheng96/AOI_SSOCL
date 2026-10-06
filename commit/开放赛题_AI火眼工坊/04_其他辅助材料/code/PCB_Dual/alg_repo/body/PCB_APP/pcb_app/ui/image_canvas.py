"""图像画布：显示图片、缩放/平移、绘制并编辑 ROI、叠加缺陷框。

多个画布可共享同一份 ``rois`` 列表（引用），从而实现"模板画框 → 待检图同位置自动出现"。
坐标系：ROI/缺陷均以原图像素为准，绘制时按当前 scale/offset 换算到控件坐标。
"""

from __future__ import annotations

from typing import List, Optional, Set

import numpy as np
from PySide6.QtCore import Qt, QPointF, QRectF, Signal
from PySide6.QtGui import (
    QColor, QFont, QImage, QPainter, QPen, QPixmap, QBrush, QPolygonF)
from PySide6.QtWidgets import QWidget

from . import theme
from ..core.models import ROI, DefectBox, MatchedROI
from ..metadata import DEFECT_BY_CODE

_HANDLE = 7  # 选中框拖动热区


def _bgr_to_qimage(img: np.ndarray) -> QImage:
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    rgb = np.ascontiguousarray(img[:, :, ::-1])
    h, w, _ = rgb.shape
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()


class ImageCanvas(QWidget):
    roi_created = Signal(object)     # ROI
    roi_selected = Signal(object)    # rid 或 None
    roi_changed = Signal()
    roi_geometry_ready = Signal(object)  # ROI：新建或拖动结束，几何已定稿，可触发结构匹配预览
    view_changed = Signal(float, float, float)  # scale, ox, oy（用于双图联动，可选）
    image_dropped = Signal(str)      # 拖入的图片路径

    def __init__(self, title: str, editable: bool = True, parent=None):
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

        self.rois: List[ROI] = []
        self.selected_rid: Optional[int] = None
        self.defects: List[DefectBox] = []
        self.matched_rois: List[MatchedROI] = []
        self.visible_codes: Optional[Set[str]] = None  # None=全部可见
        self.pending_defect_codes: Set[str] = set()  # 新建 ROI 默认绑定的瑕疵（可多选）

        self.tool = "select"     # select | rect | circle
        self._drag_mode = None   # pan | create | move
        self._last_pos = None
        self._create_start = None
        self._temp_roi: Optional[ROI] = None

    # ---------------------------------------------------------- 数据接口 ----
    def set_image(self, img: Optional[np.ndarray]):
        if img is None:
            self._pix = None
            self._img_wh = (0, 0)
        else:
            self._pix = QPixmap.fromImage(_bgr_to_qimage(img))
            self._img_wh = (img.shape[1], img.shape[0])
            self.fit_to_view()
        self.update()

    def set_rois(self, rois: List[ROI]):
        self.rois = rois
        self.update()

    def set_defects(self, defects: List[DefectBox]):
        self.defects = defects
        self.update()

    def set_matched_rois(self, matched_rois: List[MatchedROI]):
        self.matched_rois = matched_rois
        self.update()

    def set_visible_codes(self, codes: Optional[Set[str]]):
        self.visible_codes = codes
        self.update()

    def set_tool(self, tool: str):
        self.tool = tool
        self.setCursor(Qt.CrossCursor if tool in ("rect", "circle") else Qt.ArrowCursor)

    def has_image(self) -> bool:
        return self._pix is not None

    # ---------------------------------------------------------- 视图变换 ----
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
        self._emit_view()

    def _emit_view(self):
        self.view_changed.emit(self.scale, self.ox, self.oy)

    def set_view(self, scale: float, ox: float, oy: float):
        self.scale, self.ox, self.oy = scale, ox, oy
        self.update()

    def to_img(self, pt: QPointF) -> QPointF:
        return QPointF((pt.x() - self.ox) / self.scale, (pt.y() - self.oy) / self.scale)

    def to_widget(self, x: float, y: float) -> QPointF:
        return QPointF(x * self.scale + self.ox, y * self.scale + self.oy)

    # ---------------------------------------------------------- 拖放 ----
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        for url in e.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")):
                self.image_dropped.emit(path)
                break

    # ---------------------------------------------------------- 事件 ----
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
        # 保持光标下的图像点不动
        self.ox = e.position().x() - old.x() * self.scale
        self.oy = e.position().y() - old.y() * self.scale
        self._emit_view()
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
            if self.tool in ("rect", "circle") and self.editable:
                self._drag_mode = "create"
                self._create_start = ip
                self._temp_roi = None
                return
            # select 模式：命中 ROI → 选中并可移动；否则平移
            hit = self._roi_at(ip)
            if hit is not None:
                self.selected_rid = hit.rid
                self.roi_selected.emit(hit.rid)
                self._drag_mode = "move" if self.editable else "pan"
                self.update()
            else:
                self.selected_rid = None
                self.roi_selected.emit(None)
                self._drag_mode = "pan"
                self.update()

    def mouseMoveEvent(self, e):
        if self._pix is None:
            return
        pos = e.position()
        ip = self.to_img(pos)

        if self._drag_mode == "pan" and self._last_pos is not None:
            self.ox += pos.x() - self._last_pos.x()
            self.oy += pos.y() - self._last_pos.y()
            self._last_pos = pos
            self._emit_view()
            self.update()
        elif self._drag_mode == "create" and self._create_start is not None:
            self._temp_roi = self._make_roi(self._create_start, ip)
            self.update()
        elif self._drag_mode == "move" and self._last_pos is not None:
            roi = self._selected_roi()
            if roi is not None:
                dx = (pos.x() - self._last_pos.x()) / self.scale
                dy = (pos.y() - self._last_pos.y()) / self.scale
                roi.x += dx
                roi.y += dy
                self._last_pos = pos
                self.roi_changed.emit()
                self.update()

    def mouseReleaseEvent(self, e):
        if self._drag_mode == "create" and self._temp_roi is not None:
            roi = self._temp_roi
            if roi.w >= 4 and roi.h >= 4:
                roi.defect_codes = sorted(self.pending_defect_codes)
                self.rois.append(roi)
                self.selected_rid = roi.rid
                self.roi_created.emit(roi)
                self.roi_geometry_ready.emit(roi)
            self._temp_roi = None
        elif self._drag_mode == "move":
            roi = self._selected_roi()
            if roi is not None:
                # 拖动结束（几何已定稿）才触发一次结构匹配预览，避免拖动过程中反复搜索。
                self.roi_geometry_ready.emit(roi)
        self._drag_mode = None
        self._create_start = None
        self._last_pos = None
        self.update()

    # ---------------------------------------------------------- ROI 工具 ----
    def _make_roi(self, p0: QPointF, p1: QPointF) -> ROI:
        x0, y0 = min(p0.x(), p1.x()), min(p0.y(), p1.y())
        w, h = abs(p1.x() - p0.x()), abs(p1.y() - p0.y())
        shape = "circle" if self.tool == "circle" else "rect"
        codes = sorted(self.pending_defect_codes)
        if shape == "circle":
            side = max(w, h)
            return ROI("circle", x0, y0, side, side, codes)
        return ROI("rect", x0, y0, w, h, codes)

    def _roi_at(self, ip: QPointF) -> Optional[ROI]:
        for roi in reversed(self.rois):
            if roi.contains(ip.x(), ip.y()):
                return roi
        return None

    def _selected_roi(self) -> Optional[ROI]:
        for roi in self.rois:
            if roi.rid == self.selected_rid:
                return roi
        return None

    # ---------------------------------------------------------- 绘制 ----
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

        # ROI
        for roi in self.rois:
            self._draw_roi(p, roi, roi.rid == self.selected_rid)
        if self._temp_roi is not None:
            self._draw_roi(p, self._temp_roi, True, temp=True)

        # 待检图中通过结构特征实际匹配到的旋转 ROI。
        for matched in self.matched_rois:
            if matched.matched and len(matched.polygon) >= 3:
                self._draw_matched_roi(p, matched)

        # 检测结果不在画布上画框，只在底部结果表中展示。

        self._draw_corner_title(p)

    def _roi_color(self, roi: ROI) -> QColor:
        meta = DEFECT_BY_CODE.get(roi.defect_code or "")  # 取首个绑定类型的主题色
        return QColor(meta.color if meta else theme.ACCENT)

    def _draw_roi(self, p: QPainter, roi: ROI, selected: bool, temp: bool = False):
        col = self._roi_color(roi)
        pen = QPen(col, 2, Qt.DashLine if temp else Qt.SolidLine)
        pen.setCosmetic(True)
        p.setPen(pen)
        fill = QColor(col)
        fill.setAlpha(40 if selected else 20)
        p.setBrush(QBrush(fill))

        tl = self.to_widget(roi.x, roi.y)
        rw, rh = roi.w * self.scale, roi.h * self.scale
        rect = QRectF(tl.x(), tl.y(), rw, rh)
        if roi.shape == "circle":
            p.drawEllipse(rect)
        else:
            p.drawRect(rect)

        # 标签：显示 ROI 名 + 全部已绑定的瑕疵类型（可多个，用 "+" 连接）
        label = roi.name
        names = [DEFECT_BY_CODE[c].name for c in roi.defect_codes if c in DEFECT_BY_CODE]
        if names:
            label += f" · {'+'.join(names)}"
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

    def _draw_matched_roi(self, p: QPainter, matched: MatchedROI):
        col = QColor("#22C55E")
        pen = QPen(col, 2, Qt.DashLine)
        pen.setCosmetic(True)
        p.setPen(pen)
        fill = QColor(col); fill.setAlpha(22)
        p.setBrush(QBrush(fill))
        points = QPolygonF([
            self.to_widget(float(x), float(y)) for x, y in matched.polygon
        ])
        p.drawPolygon(points)

        branch = " · 180°候选" if matched.reversed_180 else ""
        tag = (
            f"{matched.name} 匹配 {matched.score:.2f} "
            f"θ={matched.angle_deg:.1f}°{branch}")
        f = QFont(); f.setPointSize(8); f.setBold(True); p.setFont(f)
        fm = p.fontMetrics()
        tw = fm.horizontalAdvance(tag) + 10
        th = fm.height() + 4
        anchor = points.boundingRect().topLeft()
        p.setBrush(QBrush(col)); p.setPen(Qt.NoPen)
        p.drawRoundedRect(QRectF(anchor.x(), anchor.y() - th - 2, tw, th), 4, 4)
        p.setPen(QColor("white"))
        p.drawText(
            QRectF(anchor.x() + 5, anchor.y() - th - 2, tw, th),
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
