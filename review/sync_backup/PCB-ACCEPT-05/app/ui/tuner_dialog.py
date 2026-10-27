"""调参辅助对话框（通用版，复现 reference5/UI算法/zz pcba_tuner 的成熟方案）。

业务定位：在「模板建模」流程里，导入 ROI（pad/toe/rim）→ 分配算法（勾选检测项）
→ 调出本控件辅助调参。与早期"只针对单个算法"的版本不同，本版：

1. **多算法可注册**：从已勾选算法的 param_specs 自动发现 HSV 参数组
   （:func:`app.tools.tuner.discover_hsv_objects`），每个参数组 = 一个"抽色对象"
   （焊锡/阻焊/丝印/元件/toe…），对象可切换，互不影响。
2. **ROI 叠加**：显示模板已标注的 pad/toe/rim 框，抽色/预览以模板区域为参考。
3. **自动抽色**：ROI 内 Otsu + V→S→H 三级搜索 + Youden's J 门控，结果写当前对象。
4. **手动取色**：点击图像采样点 → 生成色板（多段，红色跨边界）。
5. **实时预览**：滑条拖动即刷新 HSV 掩膜叠加 / 通道 / 预处理视图。
6. **写回模板**：把当前对象 HSV 六参写回所有勾选算法中匹配的参数键。
"""
from __future__ import annotations

from typing import Callable, Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup, QComboBox, QDialog, QFileDialog, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QMessageBox, QPushButton, QRadioButton, QSlider,
    QSpinBox, QSplitter, QVBoxLayout, QWidget,
)

from app.tools import tuner as T
from app.utils.cv_io import imread_unicode
from app.utils.image_qt import numpy_to_qpixmap

_MODE_BOX = "框选ROI"
_MODE_PICK = "点击取色"

# 滑条定义：(槽位, 显示名, 最小值, 最大值)
_HSV_SLOTS = [
    ("h_lo", "H 下限", 0, 179),
    ("h_hi", "H 上限", 0, 179),
    ("s_lo", "S 下限", 0, 255),
    ("s_hi", "S 上限", 0, 255),
    ("v_lo", "V 下限", 0, 255),
    ("v_hi", "V 上限", 0, 255),
]

# ROI 框类型 → (显示名, BGR 颜色)
_ROI_COLORS: dict[str, tuple[tuple[int, int, int], str]] = {
    "smt_pads": ((0, 200, 255), "pad"),
    "pad": ((0, 200, 255), "pad"),
    "smt_toe": ((0, 165, 255), "toe"),
    "toe": ((0, 165, 255), "toe"),
    "smt_rim": ((224, 108, 117), "rim"),
    "rim": ((224, 108, 117), "rim"),
}


class _Canvas(QWidget):
    """图像画布：框选 ROI / 点击取色点 / 显示 ROI 框与采样点标记。"""

    roi_selected = Signal(int, int, int, int)
    point_picked = Signal(int, int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self._scale: float = 1.0
        self._ox = self._oy = 0
        self._mode = _MODE_BOX
        self._dragging = False
        self._sel: tuple[int, int, int, int] | None = None
        self._points: list[tuple[int, int]] = []
        self._roi_boxes: list[tuple[str, int, int, int, int]] = []
        self.setMinimumSize(480, 360)
        self.setMouseTracking(True)

    def set_image(self, img: np.ndarray) -> None:
        self._pixmap = numpy_to_qpixmap(img)
        self._fit()
        self.update()

    def set_mode(self, mode: str) -> None:
        self._mode = mode

    def set_roi_boxes(self, boxes: list[tuple[str, int, int, int, int]]) -> None:
        """叠加显示模板 ROI 框：(种类, x, y, w, h)。"""
        self._roi_boxes = list(boxes)

    def _fit(self) -> None:
        if self._pixmap is None:
            return
        w, h = self.width(), self.height()
        pw, ph = self._pixmap.width(), self._pixmap.height()
        self._scale = min((w - 4) / pw, (h - 4) / ph) if pw and ph else 1.0
        self._ox = (w - pw * self._scale) / 2
        self._oy = (h - ph * self._scale) / 2

    def _to_img(self, px: float, py: float) -> tuple[int, int]:
        return int((px - self._ox) / self._scale), int((py - self._oy) / self._scale)

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(30, 30, 34))
        if self._pixmap is not None:
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            p.drawPixmap(int(self._ox), int(self._oy),
                         int(self._pixmap.width() * self._scale),
                         int(self._pixmap.height() * self._scale),
                         self._pixmap)
            # 采样点
            pen = QPen(QColor(0, 200, 255), 2)
            p.setPen(pen)
            for pt in self._points:
                p.drawEllipse(int(self._ox + pt[0] * self._scale) - 5,
                              int(self._oy + pt[1] * self._scale) - 5, 10, 10)
            # 模板 ROI 框（pad/toe/rim 分色）
            for kind, x, y, w, h in self._roi_boxes:
                color, label = _ROI_COLORS.get(kind, ((200, 200, 200), kind))
                pen = QPen(QColor(*color), 2)
                p.setPen(pen)
                sx, sy = self._ox + x * self._scale, self._oy + y * self._scale
                sw, sh = w * self._scale, h * self._scale
                p.drawRect(int(sx), int(sy), int(sw), int(sh))
                p.drawText(int(sx) + 3, int(sy) - 4, label)
            # 当前框选 ROI
            if self._sel is not None:
                x, y, w, h = self._sel
                pen = QPen(QColor(255, 255, 255), 2)
                pen.setStyle(Qt.PenStyle.DashLine)
                p.setPen(pen)
                p.drawRect(int(self._ox + x * self._scale),
                           int(self._oy + y * self._scale),
                           int(w * self._scale), int(h * self._scale))
        p.end()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self._pixmap is None:
            return
        ix, iy = self._to_img(event.position().x(), event.position().y())
        if ix < 0 or iy < 0:
            return
        if self._mode == _MODE_PICK:
            self._points.append((ix, iy))
            self.point_picked.emit(ix, iy)
        else:
            self._dragging = True
            self._sel = (ix, iy, 1, 1)
        self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._dragging and self._sel is not None:
            ix, iy = self._to_img(event.position().x(), event.position().y())
            x, y, _w, _h = self._sel
            self._sel = (min(x, ix), min(y, iy), abs(ix - x) + 1, abs(iy - y) + 1)
            self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._dragging and self._sel is not None:
            self._dragging = False
            x, y, w, h = self._sel
            if w > 4 and h > 4:
                self.roi_selected.emit(x, y, w, h)
        self.update()

    def resizeEvent(self, event) -> None:  # noqa: N802
        self._fit()
        self.update()

    def clear_marks(self) -> None:
        self._points = []
        self._sel = None
        self.update()


class TunerDialog(QDialog):
    """通用调参辅助对话框：色板抽色 / 自动抽色 / 预处理 / 辅助视图。

    参数：
        image_path: 标准图路径
        hsv_objects: 从勾选算法发现的抽色对象列表（无则自动 discover）
        roi_boxes: 模板 region_sets 里的 ROI 框（pad/toe/rim）
        initial_params: 模板当前算法参数（用于初始化各对象 HSV）
        apply_callback: (object_spec, hsv_dict) -> None，写回模板参数
    """

    applied = Signal(dict)

    def __init__(self, image_path: str = "",
                 apply_callback: Optional[Callable[[T.HsvObjectSpec, dict], None]] = None,
                 hsv_objects: Optional[list[T.HsvObjectSpec]] = None,
                 roi_boxes: Optional[list[tuple[str, int, int, int, int]]] = None,
                 initial_params: Optional[dict] = None,
                 parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("调参辅助：色板抽色 / 自动抽色 / 预处理")
        self.resize(1180, 680)
        self.image_bgr: np.ndarray | None = None
        self.image_path = image_path
        self.apply_callback = apply_callback
        self.hsv_objects: list[T.HsvObjectSpec] = list(hsv_objects or [])
        self.roi_boxes = list(roi_boxes or [])
        self.initial_params = dict(initial_params or {})
        self.roi: tuple[int, int, int, int] | None = None
        self.swatch_points: list[tuple[int, int]] = []
        self._current_hsv: dict = {}   # 当前对象 HSV 六值
        self._swatch_pixmap: QPixmap | None = None
        self._sliders: dict[str, QSlider] = {}
        self._spinners: dict[str, QSpinBox] = {}
        self._suppress_hsv = False      # 程序化赋值滑条时抑制重算
        self._build()
        if image_path:
            self._load_image(image_path)
        self._init_objects()

    # ── UI ──────────────────────────────────────────────
    def _build(self) -> None:
        root = QHBoxLayout(self)
        split = QSplitter(Qt.Orientation.Horizontal)

        # 左：画布 + 工具条
        left = QWidget()
        lv = QVBoxLayout(left)
        toolbar = QHBoxLayout()
        self.btn_open = QPushButton("打开图片")
        self.rb_box = QRadioButton(_MODE_BOX)
        self.rb_box.setChecked(True)
        self.rb_pick = QRadioButton(_MODE_PICK)
        self.btn_clear = QPushButton("清除标记")
        toolbar.addWidget(self.btn_open)
        toolbar.addWidget(self.rb_box)
        toolbar.addWidget(self.rb_pick)
        toolbar.addWidget(self.btn_clear)
        toolbar.addStretch(1)
        lv.addLayout(toolbar)
        self.canvas = _Canvas()
        self.canvas.roi_selected.connect(self._on_roi)
        self.canvas.point_picked.connect(self._on_pick)
        self.canvas.set_roi_boxes(self.roi_boxes)
        lv.addWidget(self.canvas, 1)
        split.addWidget(left)

        # 中：抽色对象 + HSV 滑条 + 动作
        mid = QWidget()
        mv = QVBoxLayout(mid)
        box_obj = QGroupBox("抽色对象（来自已勾选算法的参数）")
        obj_lay = QVBoxLayout(box_obj)
        self.obj_group = QButtonGroup(self)
        obj_lay.addWidget(QLabel("（扫描算法参数自动注册）"))
        self._obj_host = QWidget()
        self._obj_host_lay = QVBoxLayout(self._obj_host)
        self._obj_host_lay.setContentsMargins(0, 0, 0, 0)
        obj_lay.addWidget(self._obj_host)
        mv.addWidget(box_obj)

        box_hsv = QGroupBox("HSV 颜色范围")
        hsv_lay = QGridLayout(box_hsv)
        for i, (slot, label, lo, hi) in enumerate(_HSV_SLOTS):
            s = QSlider(Qt.Orientation.Horizontal)
            s.setRange(lo, hi)
            sp = QSpinBox()
            sp.setRange(lo, hi)
            s.valueChanged.connect(sp.setValue)
            sp.valueChanged.connect(s.setValue)
            self._sliders[slot] = s
            self._spinners[slot] = sp
            hsv_lay.addWidget(QLabel(label), i, 0)
            hsv_lay.addWidget(s, i, 1)
            hsv_lay.addWidget(sp, i, 2)
        for slot in ("h_lo", "h_hi", "s_lo", "s_hi", "v_lo", "v_hi"):
            self._sliders[slot].valueChanged.connect(self._on_hsv_changed)
        mv.addWidget(box_hsv, 1)

        act = QHBoxLayout()
        self.btn_auto = QPushButton("自动抽色（当前ROI→当前对象）")
        self.btn_auto.setObjectName("PrimaryButton")
        self.btn_swatch = QPushButton("生成色板（全部取色点）")
        act.addWidget(self.btn_auto, 1)
        act.addWidget(self.btn_swatch, 1)
        mv.addLayout(act)

        mv.addStretch(1)
        btns = QHBoxLayout()
        self.btn_copy = QPushButton("复制参数")
        self.btn_apply = QPushButton("应用到模板")
        self.btn_close = QPushButton("关闭")
        btns.addWidget(self.btn_copy)
        btns.addWidget(self.btn_apply)
        btns.addWidget(self.btn_close)
        mv.addLayout(btns)
        split.addWidget(mid)

        # 右：结果信息 + 视图切换
        right = QWidget()
        rv = QVBoxLayout(right)
        self.lbl_hsv = QLabel("HSV 范围：—")
        self.lbl_hsv.setWordWrap(True)
        self.lbl_hsv.setAlignment(Qt.AlignmentFlag.AlignTop)
        rv.addWidget(self.lbl_hsv)

        self.lbl_swatch = QLabel("色板：—")
        self.lbl_swatch.setWordWrap(True)
        rv.addWidget(self.lbl_swatch)

        rv.addWidget(QLabel("图像预处理预览"))
        self.cmb_pre = QComboBox()
        self.cmb_pre.addItems(["原始", "灰度", "Otsu", "自适应二值", "自动Gamma"])
        rv.addWidget(self.cmb_pre)

        rv.addWidget(QLabel("辅助视图"))
        self.cmb_aux = QComboBox()
        self.cmb_aux.addItems(["无", "HSV掩膜", "H通道", "S通道", "V通道"])
        rv.addWidget(self.cmb_aux)
        rv.addStretch(1)
        split.addWidget(right)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)
        split.setStretchFactor(2, 1)
        root.addWidget(split)

        self.btn_open.clicked.connect(self._open_image)
        self.rb_box.toggled.connect(
            lambda ck: self.canvas.set_mode(_MODE_BOX) if ck else None)
        self.rb_pick.toggled.connect(
            lambda ck: self.canvas.set_mode(_MODE_PICK) if ck else None)
        self.btn_clear.clicked.connect(self.canvas.clear_marks)
        self.btn_auto.clicked.connect(self._auto_extract)
        self.btn_swatch.clicked.connect(self._make_swatch)
        self.cmb_pre.currentIndexChanged.connect(self._refresh_views)
        self.cmb_aux.currentIndexChanged.connect(self._refresh_views)
        self.btn_copy.clicked.connect(self._copy_params)
        self.btn_apply.clicked.connect(self._apply)

    def _init_objects(self) -> None:
        """从算法发现抽色对象；无则给空占位提示。"""
        if not self.hsv_objects:
            self.hsv_objects = T.discover_hsv_objects(self._selected_algorithm_ids())
        # 清空旧对象按钮
        while self._obj_host_lay.count():
            item = self._obj_host_lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        if not self.hsv_objects:
            self._obj_host_lay.addWidget(QLabel("（当前勾选的算法没有 HSV 颜色参数）"))
            self.btn_auto.setEnabled(False)
            self.btn_swatch.setEnabled(False)
            self.btn_apply.setEnabled(False)
            self.lbl_hsv.setText("当前勾选算法未暴露 HSV 参数，无法抽色。\n"
                                 "请到算法调试/参数面板确认该算法是否提供颜色范围参数。")
            return
        for i, spec in enumerate(self.hsv_objects):
            rb = QRadioButton(f"{spec.label}（{spec.id}）")
            rb.setChecked(i == 0)
            self.obj_group.addButton(rb, i)
            self._obj_host_lay.addWidget(rb)
        self.obj_group.idClicked.connect(self._switch_object)
        self._switch_object(0)

    def _selected_algorithm_ids(self) -> list[str]:
        parent = self.parent()
        sel = getattr(parent, "_selected_ids", None)
        if callable(sel):
            return sel()
        return []

    def _switch_object(self, idx: int) -> None:
        if not self.hsv_objects or idx >= len(self.hsv_objects):
            return
        spec = self.hsv_objects[idx]
        # 从模板当前参数读该对象的 HSV（未保存过则对象默认值）
        hsv = T.obj_hsv_from_params(spec, self.initial_params)
        self._current_hsv = hsv
        self._set_sliders(hsv)
        self.lbl_hsv.setText(
            f"对象：{spec.label}（{spec.id}）\n"
            f"H [{hsv['h_lo']}-{hsv['h_hi']}]  S [{hsv['s_lo']}-{hsv['s_hi']}]  "
            f"V [{hsv['v_lo']}-{hsv['v_hi']}]")
        self._refresh_views()

    def _set_sliders(self, hsv: dict) -> None:
        for slot, _label, _lo, _hi in _HSV_SLOTS:
            s, sp = self._sliders[slot], self._spinners[slot]
            val = max(s.minimum(), min(s.maximum(), int(hsv.get(slot, 0))))
            self._suppress_hsv = True
            s.setValue(val)
            sp.setValue(val)
            self._suppress_hsv = False

    def _current_hsv_from_sliders(self) -> dict:
        return {slot: self._sliders[slot].value() for slot, _l, _lo, _hi in _HSV_SLOTS}

    # ── 动作 ──────────────────────────────────────────────
    def _load_image(self, path: str) -> None:
        img = imread_unicode(path)
        if img is None:
            QMessageBox.warning(self, "无法读取", path)
            return
        self.image_bgr = img
        self.image_path = path
        self.canvas.set_image(img)
        self._refresh_views()

    def _open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.webp)")
        if path:
            self._load_image(path)

    def _on_roi(self, x: int, y: int, w: int, h: int) -> None:
        self.roi = (x, y, w, h)
        self.lbl_hsv.setText(f"ROI：({x},{y}) {w}×{h}")

    def _on_pick(self, x: int, y: int) -> None:
        self.swatch_points.append((x, y))

    def _on_hsv_changed(self, _v: int) -> None:
        if self._suppress_hsv:
            return
        self._current_hsv = self._current_hsv_from_sliders()
        self._refresh_hsv_label()
        self._refresh_views()

    def _refresh_hsv_label(self) -> None:
        if not self.hsv_objects:
            return
        spec = self.hsv_objects[self.obj_group.checkedId() if self.obj_group.checkedId() >= 0 else 0]
        h = self._current_hsv
        self.lbl_hsv.setText(
            f"对象：{spec.label}（{spec.id}）\n"
            f"H [{h['h_lo']}-{h['h_hi']}]  S [{h['s_lo']}-{h['s_hi']}]  "
            f"V [{h['v_lo']}-{h['v_hi']}]")

    def _auto_extract(self) -> None:
        if self.image_bgr is None:
            QMessageBox.information(self, "提示", "先打开图片")
            return
        if self.roi is None:
            QMessageBox.information(self, "提示", "先在图上框选 ROI")
            return
        r = T.extract_hsv_range(self.image_bgr, self.roi)
        # 应用到当前对象
        self._current_hsv = {k: r[k] for k in ("h_lo", "h_hi", "s_lo", "s_hi", "v_lo", "v_hi")}
        self._set_sliders(self._current_hsv)
        self._refresh_hsv_label()
        self.cmb_aux.setCurrentText("HSV掩膜")
        self._refresh_views()

    def _make_swatch(self) -> None:
        if self.image_bgr is None or not self.swatch_points:
            QMessageBox.information(self, "提示", "先点击取色点")
            return
        ranges = T.multi_swatch_ranges(self.image_bgr, self.swatch_points)
        if len(ranges) == 1:
            lo, hi, s0, s1, v0, v1 = ranges[0]
            self._current_hsv = {"h_lo": lo, "h_hi": hi, "s_lo": s0, "s_hi": s1,
                                 "v_lo": v0, "v_hi": v1}
            self._set_sliders(self._current_hsv)
            self._refresh_hsv_label()
            self.lbl_swatch.setText("色板：单段")
        else:
            desc = "；".join(
                f"[{lo}-{hi} S{s0}-{s1} V{v0}-{v1}]" for lo, hi, s0, s1, v0, v1 in ranges)
            self.lbl_swatch.setText(f"多段色板 {len(ranges)} 段（红色跨边界）\n{desc}")
        self.cmb_aux.setCurrentText("HSV掩膜")
        self._refresh_views()

    def _refresh_views(self) -> None:
        if self.image_bgr is None:
            return
        pre = self.cmb_pre.currentText()
        aux = self.cmb_aux.currentText()
        img = self.image_bgr
        if pre == "灰度":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif pre == "Otsu":
            img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        elif pre == "自适应二值":
            img = cv2.adaptiveThreshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 31, 5)
        elif pre == "自动Gamma":
            img = T.auto_gamma(img, 0.35)
        if aux == "HSV掩膜" and self._current_hsv:
            mask = T.hsv_mask(self.image_bgr, self._current_hsv)
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            overlay = img.copy()
            overlay[mask > 0] = (0, 220, 255)
            img = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
        elif aux in ("H通道", "S通道", "V通道"):
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) if img.ndim == 3 else img
            idx = {"H通道": 0, "S通道": 1, "V通道": 2}[aux]
            img = hsv[:, :, idx]
        self.canvas.set_image(img)
        self.canvas._points = list(self.swatch_points)
        if self.roi is not None:
            self.canvas._sel = self.roi
        self.canvas.update()

    def _copy_params(self) -> None:
        if not self._current_hsv:
            QMessageBox.information(self, "提示", "先生成颜色范围")
            return
        h = self._current_hsv
        from PySide6.QtWidgets import QApplication
        txt = (f'h_lo={h["h_lo"]} h_hi={h["h_hi"]} s_lo={h["s_lo"]} s_hi={h["s_hi"]} '
               f'v_lo={h["v_lo"]} v_hi={h["v_hi"]}')
        QApplication.clipboard().setText(txt)
        QMessageBox.information(self, "已复制", txt)

    def _apply(self) -> None:
        if not self._current_hsv:
            QMessageBox.information(self, "提示", "先生成颜色范围")
            return
        idx = self.obj_group.checkedId()
        if idx < 0 or idx >= len(self.hsv_objects):
            return
        spec = self.hsv_objects[idx]
        if self.apply_callback is not None:
            self.apply_callback(spec, dict(self._current_hsv))
            self.applied.emit(dict(self._current_hsv))
        else:
            QMessageBox.information(self, "未绑定", "未提供应用回调（可复制参数）")
