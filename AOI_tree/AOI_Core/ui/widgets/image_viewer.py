"""ImageViewer：图片查看器。

功能：网络/本地字节加载、滚轮缩放、拖拽平移、适应窗口、
三层切换（原图/热力图/叠加图）、缺陷框叠加（红框 2px）、
"漏检框选"橡皮筋选区（输出 [x, y, w, h] 图像坐标）。
"""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, QThread, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPen, QPixmap, QWheelEvent
from PySide6.QtWidgets import (
    QDialog,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QVBoxLayout,
)

LAYERS = ("original", "heatmap", "overlay")
LAYER_NAMES = {"original": "原图", "heatmap": "热力图", "overlay": "叠加图"}


class _DownloadWorker(QThread):
    """后台下载图片字节。"""

    loaded = Signal(str, bytes)  # layer, bytes
    failed = Signal(str)

    def __init__(self, url: str, layer: str, parent=None):
        super().__init__(parent)
        self._url = url
        self._layer = layer

    def run(self) -> None:
        try:
            import requests

            from ..api_client import _load_api_key
            headers = {}
            key = _load_api_key()
            if key:  # M10d：鉴权开启时 /api/file 也需带 Key
                headers["X-API-Key"] = key
            r = requests.get(self._url, timeout=20, headers=headers)
            r.raise_for_status()
            self.loaded.emit(self._layer, r.content)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"图片加载失败：{e}")


class ImageViewer(QGraphicsView):
    """带缩放/平移/图层/缺陷框/框选的图像查看器。"""

    rect_selected = Signal(list)    # [x, y, w, h] 原图像素坐标
    region_selected = Signal(list)  # 兼容旧接口，与 rect_selected 同时发射
    selection_cleared = Signal()    # 选区被清除（Esc/单击/set_selection_mode 关闭）

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = QGraphicsPixmapItem()
        self._pixmap_item.setTransformationMode(Qt.SmoothTransformation)
        self._scene.addItem(self._pixmap_item)

        self._pixmaps: dict[str, QPixmap | None] = {k: None for k in LAYERS}
        self._layer = "original"
        self._box_items: list[QGraphicsRectItem] = []
        self._boxes_visible = True
        self._select_mode = False
        self._sel_origin: QPointF | None = None
        self._sel_item: QGraphicsRectItem | None = None
        self._sel_dragging = False
        self._workers: list[QThread] = []

        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setBackgroundBrush(QColor("#EDF1F7"))
        self.setMinimumHeight(320)
        self.setFocusPolicy(Qt.StrongFocus)  # 接收 Esc 清除选区

    # ── 图像加载 ──────────────────────────────────────────
    def set_image_bytes(self, data: bytes, layer: str = "original") -> None:
        pm = QPixmap()
        pm.loadFromData(data)
        if pm.isNull():
            return
        self._pixmaps[layer] = pm
        if layer == self._layer:
            self._show_current()

    def set_image_url(self, url: str, layer: str = "original") -> None:
        worker = _DownloadWorker(url, layer, self)
        worker.loaded.connect(self._on_loaded)
        worker.finished.connect(lambda w=worker: self._forget_worker(w))
        self._workers.append(worker)
        worker.start()

    def set_image_file(self, path: str, layer: str = "original") -> None:
        pm = QPixmap(path)
        if pm.isNull():
            return
        self._pixmaps[layer] = pm
        if layer == self._layer:
            self._show_current()

    def _on_loaded(self, layer: str, data: bytes) -> None:
        self.set_image_bytes(data, layer)

    def _forget_worker(self, w: QThread) -> None:
        if w in self._workers:
            self._workers.remove(w)
        w.deleteLater()

    def clear_images(self) -> None:
        self._pixmaps = {k: None for k in LAYERS}
        self._pixmap_item.setPixmap(QPixmap())
        self.clear_boxes()
        self.clear_selection()

    # ── 图层切换 ──────────────────────────────────────────
    def set_layer(self, layer: str) -> None:
        if layer not in LAYERS:
            return
        self._layer = layer
        self._show_current()

    def current_layer(self) -> str:
        return self._layer

    def _show_current(self) -> None:
        pm = self._pixmaps.get(self._layer)
        if pm is None or pm.isNull():
            # 当前层无图时回退到原图
            pm = self._pixmaps.get("original")
        if pm is None or pm.isNull():
            self._pixmap_item.setPixmap(QPixmap())
            return
        self._pixmap_item.setPixmap(pm)
        self._scene.setSceneRect(QRectF(pm.rect()))
        self._redraw_boxes()

    def fit_to_view(self) -> None:
        if not self._pixmap_item.pixmap().isNull():
            self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)

    # ── 缺陷框 ────────────────────────────────────────────
    def set_boxes(self, boxes: list) -> None:
        self._boxes = [list(map(float, b)) for b in (boxes or [])]
        self._redraw_boxes()

    def clear_boxes(self) -> None:
        self._boxes = []
        self._redraw_boxes()

    def set_boxes_visible(self, visible: bool) -> None:
        self._boxes_visible = visible
        for it in self._box_items:
            it.setVisible(visible)

    def _redraw_boxes(self) -> None:
        for it in self._box_items:
            self._scene.removeItem(it)
        self._box_items = []
        pm = self._pixmap_item.pixmap()
        if pm.isNull():
            return
        sx = pm.width() / max(1, self._pixmaps.get("original", pm).width()) \
            if self._pixmaps.get("original") is not None else 1.0
        # 缺陷框按原图坐标给出；当前显示图与原图尺寸可能不同，按比例缩放
        orig = self._pixmaps.get("original")
        if orig is not None and not orig.isNull():
            sx = pm.width() / orig.width()
        pen = QPen(QColor("#DC2626"), 2)
        pen.setCosmetic(True)
        for (x, y, w, h) in getattr(self, "_boxes", []):
            item = self._scene.addRect(
                QRectF(x * sx, y * sx, w * sx, h * sx), pen)
            item.setVisible(self._boxes_visible)
            item.setZValue(10)
            self._box_items.append(item)

    # ── 漏检框选 ──────────────────────────────────────────
    def set_selection_mode(self, on: bool) -> None:
        """开启/关闭框选模式：开启后左键拖拽为橡皮筋框选（优先于平移）。"""
        self._select_mode = on
        self.setDragMode(QGraphicsView.NoDrag if on
                         else QGraphicsView.ScrollHandDrag)
        if not on:
            self.clear_selection()

    # 兼容旧接口
    def set_region_select_mode(self, on: bool) -> None:
        self.set_selection_mode(on)

    def clear_selection(self) -> None:
        had = self._sel_item is not None
        if self._sel_item is not None:
            self._scene.removeItem(self._sel_item)
            self._sel_item = None
        self._sel_origin = None
        self._sel_dragging = False
        if had:
            self.selection_cleared.emit()

    # ── 交互事件 ──────────────────────────────────────────
    def wheelEvent(self, event: QWheelEvent) -> None:
        if self._pixmap_item.pixmap().isNull():
            return
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        # Esc 清除当前选区
        if event.key() == Qt.Key_Escape and self._sel_item is not None:
            self.clear_selection()
            return
        super().keyPressEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if (self._select_mode and event.button() == Qt.LeftButton
                and not self._pixmap_item.pixmap().isNull()):
            pos = self.mapToScene(event.position().toPoint())
            if self._pixmap_item.contains(pos):
                # 记录起点，待拖动时再起橡皮筋（区分"单击清除"与"拖拽框选"）
                self._sel_origin = pos
                self._sel_dragging = False
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._select_mode and self._sel_origin is not None:
            pos = self.mapToScene(event.position().toPoint())
            rect = QRectF(self._sel_origin, pos).normalized()
            rect = rect.intersected(self._pixmap_item.boundingRect())
            if self._sel_item is None:
                # 醒目虚线红框
                pen = QPen(QColor("#DC2626"), 2, Qt.DashLine)
                pen.setCosmetic(True)
                self._sel_item = self._scene.addRect(rect, pen)
                self._sel_item.setZValue(20)
            else:
                self._sel_item.setRect(rect)
            self._sel_dragging = True
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if (self._select_mode and event.button() == Qt.LeftButton
                and self._sel_origin is not None):
            if not self._sel_dragging:
                # 单纯点击（未拖动）：清除当前选区
                self.clear_selection()
                return
            rect = self._sel_item.rect() if self._sel_item is not None else QRectF()
            pm = self._pixmap_item.pixmap()
            orig = self._pixmaps.get("original")
            sx = (orig.width() / pm.width()) if (
                orig is not None and not orig.isNull() and pm.width()) else 1.0
            region = [int(rect.x() * sx), int(rect.y() * sx),
                      int(rect.width() * sx), int(rect.height() * sx)]
            self._sel_origin = None
            self._sel_dragging = False
            if region[2] >= 4 and region[3] >= 4:
                self.rect_selected.emit(region)
                self.region_selected.emit(region)
            else:
                self.clear_selection()
            return
        super().mouseReleaseEvent(event)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)


class ImagePreviewDialog(QDialog):
    """大图预览对话框。"""

    def __init__(self, title: str = "图片预览", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(760, 620)
        lay = QVBoxLayout(self)
        self.viewer = ImageViewer(self)
        lay.addWidget(self.viewer)

    def show_url(self, url: str) -> None:
        self.viewer.set_image_url(url)
