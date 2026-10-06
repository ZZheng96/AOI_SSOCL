"""标定弹窗：在标准图上标注外部算法所需的区域/点位。

- 元件本体：可选框选若干矩形/圆形重点区域；不画则整图检测。
- 插件焊点：可选画 1 个矩形/圆形"锡面区域"；不画则要求选择自动轮廓类型
  （椭圆拟合 / 不规则轮廓分割）。
- 贴片锡焊：框选若干焊盘矩形（必需），可选再框选 toe/rim 标注（虚焊用）。
- 金手指：点选若干种子点估计板面颜色范围，可预览覆盖掩膜。
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.calibration.body_roi import get_body_rois, set_body_rois
from app.calibration.gold_seed import get_gold_seed, set_gold_seed
from app.calibration.smt_pads import get_smt_pads, get_smt_rim, get_smt_toe, set_smt_pads, set_smt_rim, set_smt_toe
from app.calibration.th_roi import get_th_roi, set_th_shape_choice
from app.config import GOLD_REPO_DIR
from app.ui.image_canvas import ImageCanvas, LayerConfig
from app.ui import theme


def _rects_to_shapes(rects: list[tuple[int, int, int, int]]) -> list[dict]:
    return [{"shape": "rect", "x": x, "y": y, "w": w, "h": h} for x, y, w, h in rects]


def _shapes_to_rects(shapes: list[dict]) -> list[tuple[int, int, int, int]]:
    return [
        (int(round(s["x"])), int(round(s["y"])), int(round(s["w"])), int(round(s["h"])))
        for s in shapes
        if s.get("shape") == "rect"
    ]


class _BodyRoiTab(QWidget):
    """元件本体重点区域框选（参考「元件本体检测」验证台）。"""

    def __init__(self, template_name: str | None, std_image: np.ndarray | None, parent=None) -> None:
        super().__init__(parent)
        self.template_name = template_name

        root = QVBoxLayout(self)
        root.addWidget(QLabel(
            "元件本体框选：在标准图上框选需要检测的元件区域（可多个）。\n"
            "· 不画框 → 整图检测\n"
            "· 画框 → 每个区域单独裁剪比对后合并结果\n"
            "建议先勾选要检测的本体缺陷，再来此页完成框选。"
        ))

        tool_row = QHBoxLayout()
        self.rb_rect = QRadioButton("矩形")
        self.rb_circle = QRadioButton("圆形")
        self.rb_rect.setChecked(True)
        tool_row.addWidget(QLabel("画笔："))
        tool_row.addWidget(self.rb_rect)
        tool_row.addWidget(self.rb_circle)
        tool_row.addStretch()
        self.btn_clear = QPushButton("清除全部区域（改用整图）")
        tool_row.addWidget(self.btn_clear)
        root.addLayout(tool_row)

        self.canvas = ImageCanvas("标准图")
        self.canvas.set_layer(
            "body_roi",
            LayerConfig(color=theme.ACCENT, shape="rect_or_circle", multi=True, label="ROI"),
        )
        self.canvas.set_active_layer("body_roi")
        self.canvas.set_image(std_image)
        root.addWidget(self.canvas, 1)

        existing = get_body_rois(template_name)
        if existing:
            self.canvas.set_layer_shapes("body_roi", [dict(p) for p in existing])
            if existing[0].get("shape") == "circle":
                self.rb_circle.setChecked(True)

        self.rb_rect.toggled.connect(lambda checked: checked and self.canvas.set_draw_shape_kind("rect"))
        self.rb_circle.toggled.connect(lambda checked: checked and self.canvas.set_draw_shape_kind("circle"))
        self.btn_clear.clicked.connect(lambda: self.canvas.clear_layer("body_roi"))

    def save(self) -> None:
        shapes = self.canvas.get_layer_shapes("body_roi")
        rois = []
        for s in shapes:
            if s.get("shape") == "circle":
                rois.append({"shape": "circle", "cx": s["cx"], "cy": s["cy"], "r": s["r"]})
            else:
                rois.append({"shape": "rect", "x": s["x"], "y": s["y"], "w": s["w"], "h": s["h"]})
        set_body_rois(self.template_name, rois)


class _ThRoiTab(QWidget):
    def __init__(self, template_name: str | None, std_image: np.ndarray | None, parent=None) -> None:
        super().__init__(parent)
        self.template_name = template_name

        root = QVBoxLayout(self)
        root.addWidget(QLabel(
            "锡面区域标定：\n"
            "· 不画框 → 自动轮廓（跑孔洞/少锡/多锡/不出脚）\n"
            "· 画 1 个框 → 人工单焊点（同上盘内四类）\n"
            "· 画 ≥2 个框 → 多焊点连锡模式（算法只跑连锡）"
        ))

        tool_row = QHBoxLayout()
        self.rb_rect = QRadioButton("矩形")
        self.rb_circle = QRadioButton("圆形")
        self.rb_rect.setChecked(True)
        tool_row.addWidget(QLabel("画笔："))
        tool_row.addWidget(self.rb_rect)
        tool_row.addWidget(self.rb_circle)
        tool_row.addStretch()
        self.btn_clear = QPushButton("清除已画区域（改用自动轮廓）")
        tool_row.addWidget(self.btn_clear)
        root.addLayout(tool_row)

        self.canvas = ImageCanvas("标准图")
        self.canvas.set_layer(
            "th_roi",
            LayerConfig(color=theme.ACCENT, shape="rect_or_circle", multi=True, label="焊点"),
        )
        self.canvas.set_active_layer("th_roi")
        self.canvas.set_image(std_image)
        root.addWidget(self.canvas, 1)

        auto_row = QHBoxLayout()
        auto_row.addWidget(QLabel("未画区域时的自动轮廓类型："))
        self.combo_shape = QComboBox()
        self.combo_shape.addItem("椭圆拟合（元件规整、锡面近似圆/椭圆）", "ellipse")
        self.combo_shape.addItem("不规则轮廓分割（锡面形状不规则）", "contour")
        auto_row.addWidget(self.combo_shape, 1)
        root.addLayout(auto_row)

        existing = get_th_roi(template_name)
        idx = self.combo_shape.findData(existing.get("shape_choice", "ellipse"))
        if idx >= 0:
            self.combo_shape.setCurrentIndex(idx)
        pads = existing.get("pads") or []
        if pads:
            self.canvas.set_layer_shapes("th_roi", [dict(p) for p in pads])
            if pads[0].get("shape") == "circle":
                self.rb_circle.setChecked(True)

        self.rb_rect.toggled.connect(lambda checked: checked and self.canvas.set_draw_shape_kind("rect"))
        self.rb_circle.toggled.connect(lambda checked: checked and self.canvas.set_draw_shape_kind("circle"))
        self.btn_clear.clicked.connect(lambda: self.canvas.clear_layer("th_roi"))

    def save(self) -> None:
        from app.calibration.th_roi import set_th_pads

        shapes = self.canvas.get_layer_shapes("th_roi")
        pads = []
        for s in shapes:
            if s.get("shape") == "circle":
                pads.append({"shape": "circle", "cx": s["cx"], "cy": s["cy"], "r": s["r"]})
            else:
                pads.append({"shape": "rect", "x": s["x"], "y": s["y"], "w": s["w"], "h": s["h"]})
        set_th_pads(self.template_name, pads)
        set_th_shape_choice(self.template_name, self.combo_shape.currentData())


class _SmtPadsTab(QWidget):
    _LAYERS = {
        "pads": ("smt_pads", theme.ACCENT, "焊盘"),
        "toe": ("smt_toe", theme.REVIEW, "toe(脚尖)"),
        "rim": ("smt_rim", theme.NG, "rim(焊盘边缘)"),
    }

    def __init__(self, template_name: str | None, std_image: np.ndarray | None, parent=None) -> None:
        super().__init__(parent)
        self.template_name = template_name

        root = QVBoxLayout(self)
        root.addWidget(QLabel(
            "贴片锡焊标定：框选每个焊盘的矩形区域（多锡/少锡/连锡/虚焊都依赖此标定，必需）。\n"
            "虚焊(gull-wing)判定可选再补充 toe（脚尖）/ rim（焊盘边缘）标注框，用于更精确的规则判定。"
        ))

        layer_row = QHBoxLayout()
        layer_row.addWidget(QLabel("当前正在标注的图层："))
        self.combo_layer = QComboBox()
        for key, (_layer_name, _color, label) in self._LAYERS.items():
            self.combo_layer.addItem(label, key)
        layer_row.addWidget(self.combo_layer, 1)
        self.btn_clear = QPushButton("清空当前图层")
        layer_row.addWidget(self.btn_clear)
        root.addLayout(layer_row)

        self.canvas = ImageCanvas("标准图")
        for key, (layer_name, color, label) in self._LAYERS.items():
            self.canvas.set_layer(layer_name, LayerConfig(color=color, shape="rect", multi=True, label=label))
        self.canvas.set_active_layer(self._LAYERS["pads"][0])
        self.canvas.set_image(std_image)
        root.addWidget(self.canvas, 1)

        self.canvas.set_layer_shapes("smt_pads", _rects_to_shapes(get_smt_pads(template_name)))
        self.canvas.set_layer_shapes("smt_toe", _rects_to_shapes(get_smt_toe(template_name)))
        self.canvas.set_layer_shapes("smt_rim", _rects_to_shapes(get_smt_rim(template_name)))

        self.combo_layer.currentIndexChanged.connect(self._on_layer_changed)
        self.btn_clear.clicked.connect(self._on_clear)

    def _current_layer_name(self) -> str:
        key = self.combo_layer.currentData()
        return self._LAYERS[key][0]

    def _on_layer_changed(self, _idx: int) -> None:
        self.canvas.set_active_layer(self._current_layer_name())

    def _on_clear(self) -> None:
        self.canvas.clear_layer(self._current_layer_name())

    def save(self) -> None:
        set_smt_pads(self.template_name, _shapes_to_rects(self.canvas.get_layer_shapes("smt_pads")))
        set_smt_toe(self.template_name, _shapes_to_rects(self.canvas.get_layer_shapes("smt_toe")))
        set_smt_rim(self.template_name, _shapes_to_rects(self.canvas.get_layer_shapes("smt_rim")))


class _GoldSeedTab(QWidget):
    def __init__(self, template_name: str | None, std_image: np.ndarray | None, parent=None) -> None:
        super().__init__(parent)
        self.template_name = template_name
        self.std_image = std_image

        root = QVBoxLayout(self)
        root.addWidget(QLabel(
            "金手指/板面颜色标定：在板面良品颜色上点几个种子点（3~8 个较稳），"
            "用于估计板面 HSV 颜色范围；不标定则用仓库默认沉金色范围。"
        ))

        tol_row = QHBoxLayout()
        tol_row.addWidget(QLabel("容差 tol："))
        self.spin_tol = QDoubleSpinBox()
        self.spin_tol.setRange(0.2, 5.0)
        self.spin_tol.setSingleStep(0.1)
        tol_row.addWidget(self.spin_tol)
        self.btn_preview = QPushButton("重新计算颜色范围并预览")
        tol_row.addWidget(self.btn_preview)
        self.btn_clear = QPushButton("清除种子点")
        tol_row.addWidget(self.btn_clear)
        tol_row.addStretch()
        root.addLayout(tol_row)

        self.canvas = ImageCanvas("标准图（点选种子点）")
        self.canvas.set_layer("gold_seed", LayerConfig(color=theme.ACCENT, shape="point", multi=True, label=""))
        self.canvas.set_active_layer("gold_seed")
        self.canvas.set_image(std_image)
        root.addWidget(self.canvas, 1)

        self.lbl_result = QLabel("尚未计算颜色范围")
        self.lbl_result.setProperty("class", "Hint")
        root.addWidget(self.lbl_result)

        existing = get_gold_seed(template_name)
        self.spin_tol.setValue(float(existing.get("tol", 1.0)))
        seeds = existing.get("seeds") or []
        self.canvas.set_layer_shapes("gold_seed", [{"shape": "point", "x": x, "y": y} for x, y in seeds])
        self._ranges: list = existing.get("ranges") or []
        if self._ranges:
            self.lbl_result.setText(f"已保存 {len(self._ranges)} 段 HSV 范围（来自 {len(seeds)} 个种子点）")

        self.btn_preview.clicked.connect(self._recompute)
        self.btn_clear.clicked.connect(lambda: self.canvas.clear_layer("gold_seed"))

    def _seeds(self) -> list[tuple[int, int]]:
        return [
            (int(round(s["x"])), int(round(s["y"])))
            for s in self.canvas.get_layer_shapes("gold_seed")
            if s.get("shape") == "point"
        ]

    def _recompute(self) -> None:
        seeds = self._seeds()
        if not seeds:
            QMessageBox.information(self, "提示", "请先在标准图上点选至少 1 个种子点")
            return
        if self.std_image is None:
            QMessageBox.warning(self, "提示", "没有标准图，无法计算")
            return
        try:
            repo_dir = str(GOLD_REPO_DIR)
            if repo_dir not in sys.path:
                sys.path.insert(0, repo_dir)
            import seed_extract  # noqa: PLC0415 - 延迟导入外部仓库模块

            tol = float(self.spin_tol.value())
            info = seed_extract.preview_from_seeds(self.std_image, seeds, tol=tol)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "计算失败", f"计算颜色范围失败：{exc}")
            return
        self._ranges = info.get("custom_hsv_ranges") or info.get("ranges") or []
        cover = info.get("cover_ratio", 0.0)
        self.lbl_result.setText(
            f"计算完成：{len(self._ranges)} 段 HSV 范围，板面覆盖占比 {cover:.1%}（{len(seeds)} 个种子点，tol={tol}）"
        )

    def save(self) -> None:
        seeds = self._seeds()
        set_gold_seed(self.template_name, seeds, float(self.spin_tol.value()), self._ranges)


class CalibrationDialog(QDialog):
    def __init__(
        self,
        template_name: str | None,
        std_image: np.ndarray | None,
        relevant_kinds: set[str] | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"标定 - 模板：{template_name or '(未命名)'}")
        self.resize(880, 680)

        root = QVBoxLayout(self)
        if std_image is None:
            root.addWidget(QLabel("请先在检测页打开标准图，再进行标定。"))
            btn_close = QPushButton("关闭")
            btn_close.clicked.connect(self.reject)
            root.addWidget(btn_close)
            return

        relevant_kinds = relevant_kinds or {"body", "th", "smt", "gold"}
        self.tabs = QTabWidget()
        self._body_tab = _BodyRoiTab(template_name, std_image) if "body" in relevant_kinds else None
        self._th_tab = _ThRoiTab(template_name, std_image) if "th" in relevant_kinds else None
        self._smt_tab = _SmtPadsTab(template_name, std_image) if "smt" in relevant_kinds else None
        self._gold_tab = _GoldSeedTab(template_name, std_image) if "gold" in relevant_kinds else None
        if self._body_tab is not None:
            self.tabs.addTab(self._body_tab, "元件本体 · 重点区域框选")
        if self._th_tab is not None:
            self.tabs.addTab(self._th_tab, "插件焊点 · 锡面区域")
        if self._smt_tab is not None:
            self.tabs.addTab(self._smt_tab, "贴片锡焊 · 焊盘/toe/rim")
        if self._gold_tab is not None:
            self.tabs.addTab(self._gold_tab, "金手指 · 板面取色")
        root.addWidget(self.tabs, 1)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.btn_save = QPushButton("保存并关闭")
        self.btn_save.setObjectName("Primary")
        self.btn_cancel = QPushButton("取消")
        btn_row.addWidget(self.btn_cancel)
        btn_row.addWidget(self.btn_save)
        root.addLayout(btn_row)

        self.btn_save.clicked.connect(self._save_all)
        self.btn_cancel.clicked.connect(self.reject)

    def _save_all(self) -> None:
        for tab in (self._body_tab, self._th_tab, self._smt_tab, self._gold_tab):
            if tab is not None:
                tab.save()
        self.accept()
