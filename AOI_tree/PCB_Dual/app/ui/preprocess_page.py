"""预处理调参页：独立于检测页，用来试调"图像预处理链路"（颜色通道/增强/
滤波/形态学）并保存为配方。v1.3 起去掉了"类别"下拉框——检测页已不再区分
类别，这里改成两级：全局默认 / 指定算法专属覆盖（同一套算法列表，跟检测
页右侧勾选的算法一一对应）。
"""
from __future__ import annotations

from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

import numpy as np

from app.detect.catalog import get_catalog
from app.preprocess.engine import (
    COLOR_OPTIONS,
    ENHANCE_OPTIONS,
    FILTER_OPTIONS,
    MORPH_OPTIONS,
    PreprocessParams,
    run_preprocess,
)
from app.recipe.store import RecipeStore
from app.ui.image_canvas import ImageCanvas
from app.ui.widgets import Card, ImageLabel
from app.utils.cv_io import imread_unicode
from app.utils.image_qt import numpy_to_qpixmap

_NO_ALGORITHM = "__global__"


def _all_algorithms() -> list[tuple[str, str]]:
    return [(m.id, m.display_name) for m in get_catalog().items(job_only=True)]


class PreprocessPage(QWidget):
    def __init__(self, recipe_store: RecipeStore, parent=None) -> None:
        super().__init__(parent)
        self.recipe_store = recipe_store
        self.image_bgr: np.ndarray | None = None
        self.image_path = ""
        self._build()
        self._reload_for_algorithm()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(10)

        toolbar = QHBoxLayout()
        self.btn_open = QPushButton("打开图片")
        self.btn_use_detect = QPushButton("使用检测页当前测试图")
        toolbar.addWidget(self.btn_open)
        toolbar.addWidget(self.btn_use_detect)
        toolbar.addStretch()
        root.addLayout(toolbar)

        scope_row = QHBoxLayout()
        scope_row.addWidget(QLabel("应用对象"))
        self.combo_algorithm = QComboBox()
        self.combo_algorithm.addItem("（全局默认，对所有算法生效）", _NO_ALGORITHM)
        for alg_id, label in _all_algorithms():
            self.combo_algorithm.addItem(label, alg_id)
        self.combo_algorithm.setMinimumWidth(220)
        scope_row.addWidget(self.combo_algorithm, 1)
        self.lbl_hit = QLabel("")
        self.lbl_hit.setProperty("class", "Hint")
        scope_row.addWidget(self.lbl_hit, 2)
        root.addLayout(scope_row)

        preview_row = QHBoxLayout()
        preview_row.setSpacing(10)
        card_src = Card("原图（可拖拽图片到此处）")
        self.canvas_src = ImageCanvas("")
        self.canvas_src.setMinimumSize(320, 260)
        self.canvas_src.set_placeholder_text("拖拽图片到此处，或点击上方「打开图片」")
        card_src.add(self.canvas_src)
        preview_row.addWidget(card_src, 1)

        card_out = Card("预处理效果预览")
        self.view_out = ImageLabel("")
        self.view_out.setMinimumSize(320, 260)
        self.view_out.setText("打开图片后自动预览")
        card_out.add(self.view_out)
        preview_row.addWidget(card_out, 1)
        root.addLayout(preview_row, 1)

        card_params = Card("预处理参数")
        params_row = QHBoxLayout()
        self.combo_color = self._labeled_combo(params_row, "颜色通道", COLOR_OPTIONS)
        self.combo_enhance = self._labeled_combo(params_row, "增强", ENHANCE_OPTIONS)
        self.combo_filter = self._labeled_combo(params_row, "滤波", FILTER_OPTIONS)
        self.combo_morph = self._labeled_combo(params_row, "形态学", MORPH_OPTIONS)
        card_params.add_layout(params_row)

        btn_row = QHBoxLayout()
        self.btn_save_global = QPushButton("保存为全局默认")
        self.btn_save_algo = QPushButton("保存为该算法专属")
        self.btn_clear_algo = QPushButton("清除该算法专属覆盖")
        self.btn_reset = QPushButton("控件重置为 NONE")
        self.btn_save_global.setToolTip("影响所有模板：未设算法专属配方的算法都会使用此配方")
        btn_row.addWidget(self.btn_save_global)
        btn_row.addWidget(self.btn_save_algo)
        btn_row.addWidget(self.btn_clear_algo)
        btn_row.addStretch()
        btn_row.addWidget(self.btn_reset)
        card_params.add_layout(btn_row)
        root.addWidget(card_params)

        self.btn_open.clicked.connect(self._open_image)
        self.canvas_src.image_dropped.connect(self._load_image)
        self.combo_algorithm.currentIndexChanged.connect(self._reload_for_algorithm)
        for combo in (self.combo_color, self.combo_enhance, self.combo_filter, self.combo_morph):
            combo.currentTextChanged.connect(self._update_preview)
        self.btn_save_global.clicked.connect(self._save_global)
        self.btn_save_algo.clicked.connect(self._save_algorithm)
        self.btn_clear_algo.clicked.connect(self._clear_algorithm)
        self.btn_reset.clicked.connect(self._reset_controls)

    @staticmethod
    def _labeled_combo(row: QHBoxLayout, label: str, options: list[str]) -> QComboBox:
        row.addWidget(QLabel(label))
        combo = QComboBox()
        combo.addItems(options)
        combo.setMinimumWidth(140)
        row.addWidget(combo)
        return combo

    # ---- 图片输入 ----
    def _open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if path:
            self._load_image(path)

    def _load_image(self, path: str) -> None:
        image = imread_unicode(path)
        if image is None:
            QMessageBox.warning(self, "打开失败", f"无法读取图片：{path}")
            return
        self.set_image_from_detect(image, path)

    def set_image_from_detect(self, image, path: str) -> None:
        if image is None:
            return
        self.image_bgr = image
        self.image_path = path
        self.canvas_src.set_image(image)
        self._update_preview()

    # ---- 算法切换 / 生效层级 ----
    def _current_algorithm(self) -> str | None:
        data = self.combo_algorithm.currentData()
        return None if data == _NO_ALGORITHM else data

    def _reload_for_algorithm(self) -> None:
        algorithm = self._current_algorithm()
        record, hit = self.recipe_store.resolve(None, algorithm)
        params = PreprocessParams.from_dict(record.preprocess)
        self.lbl_hit.setText(f"当前生效：{hit}")
        self.combo_color.blockSignals(True)
        self.combo_enhance.blockSignals(True)
        self.combo_filter.blockSignals(True)
        self.combo_morph.blockSignals(True)
        self.combo_color.setCurrentText(params.color)
        self.combo_enhance.setCurrentText(params.enhance)
        self.combo_filter.setCurrentText(params.filter)
        self.combo_morph.setCurrentText(params.morphology)
        self.combo_color.blockSignals(False)
        self.combo_enhance.blockSignals(False)
        self.combo_filter.blockSignals(False)
        self.combo_morph.blockSignals(False)
        can_algo_scope = algorithm is not None
        self.btn_save_algo.setEnabled(can_algo_scope)
        self.btn_clear_algo.setEnabled(can_algo_scope)
        self._update_preview()

    def _current_params(self) -> PreprocessParams:
        return PreprocessParams(
            color=self.combo_color.currentText(),
            enhance=self.combo_enhance.currentText(),
            filter=self.combo_filter.currentText(),
            morphology=self.combo_morph.currentText(),
        )

    def _update_preview(self) -> None:
        if self.image_bgr is None:
            return
        params = self._current_params()
        try:
            out = run_preprocess(self.image_bgr, params)
        except Exception as exc:  # noqa: BLE001 - 预处理异常直接提示，不让预览崩掉
            self.view_out.setText(f"预处理失败：{exc}")
            return
        pix = numpy_to_qpixmap(out)
        self.view_out.set_image(pix if not pix.isNull() else QPixmap())

    def _reset_controls(self) -> None:
        self.combo_color.setCurrentText("NONE")
        self.combo_enhance.setCurrentText("NONE")
        self.combo_filter.setCurrentText("NONE")
        self.combo_morph.setCurrentText("NONE")

    # ---- 保存 ----
    def _save_global(self) -> None:
        new = self._current_params()
        old_rec = self.recipe_store.get_saved("global_default")
        old = PreprocessParams.from_dict(old_rec.preprocess) if old_rec else PreprocessParams()
        fields = (("颜色通道", "color"), ("增强", "enhance"), ("滤波", "filter"), ("形态学", "morphology"))
        diff = "\n".join(
            f"  {label}：{getattr(old, key)} → {getattr(new, key)}"
            + ("" if getattr(old, key) != getattr(new, key) else "（不变）")
            for label, key in fields
        )
        reply = QMessageBox.question(
            self,
            "确认修改全局默认",
            "全局默认配方会作用于所有模板的检测：\n"
            "凡未设置“算法专属配方”的算法都会立即改用新配方，可能影响现有检测结果。\n\n"
            f"{diff}\n\n确定保存吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self.recipe_store.save("global_default", new)
        QMessageBox.information(
            self, "已保存", "已保存为全局默认预处理配方，对所有未设专属配方的算法生效。"
        )
        self._reload_for_algorithm()

    def _save_algorithm(self) -> None:
        algorithm = self._current_algorithm()
        if algorithm is None:
            return
        self.recipe_store.save("algorithm_default", self._current_params(), algorithm=algorithm)
        QMessageBox.information(self, "已保存", f"已保存为「{self.combo_algorithm.currentText()}」专属配方。")
        self._reload_for_algorithm()

    def _clear_algorithm(self) -> None:
        algorithm = self._current_algorithm()
        if algorithm is None:
            return
        self.recipe_store.clear("algorithm_default", algorithm=algorithm)
        self._reload_for_algorithm()
