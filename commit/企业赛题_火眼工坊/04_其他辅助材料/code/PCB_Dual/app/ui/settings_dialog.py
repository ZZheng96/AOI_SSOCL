"""设置弹窗：尺寸门禁阈值 + 输出目录（P1 质量门禁设置项）。"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from app.config import get_gate_settings, get_outputs_dir, set_gate_settings, set_outputs_dir


class SettingsDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("系统设置")
        self.resize(460, 260)

        root = QVBoxLayout(self)
        form = QFormLayout()

        self.chk_gate = QCheckBox("启用尺寸门禁（标准图/测试图尺寸差异超阈值时拦截）")
        form.addRow(self.chk_gate)

        self.spin_max_diff = QSpinBox()
        self.spin_max_diff.setRange(0, 200)
        self.spin_max_diff.setSuffix(" px")
        form.addRow("最大允许尺寸差异 N", self.spin_max_diff)

        hint = QLabel(
            "标准图与测试图尺寸差异 ≤ N 像素时，系统会自动缩放对齐测试图后继续检测；\n"
            "超过 N 像素时直接拦截，不执行检测。"
        )
        hint.setProperty("class", "Hint")
        hint.setWordWrap(True)
        form.addRow(hint)

        out_row = QHBoxLayout()
        self.edit_outputs = QLineEdit(str(get_outputs_dir()))
        self.btn_browse = QPushButton("浏览…")
        out_row.addWidget(self.edit_outputs, 1)
        out_row.addWidget(self.btn_browse)
        form.addRow("输出目录", out_row)

        root.addLayout(form)
        root.addStretch()

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.btn_save = QPushButton("保存")
        self.btn_save.setObjectName("Primary")
        self.btn_cancel = QPushButton("取消")
        btn_row.addWidget(self.btn_cancel)
        btn_row.addWidget(self.btn_save)
        root.addLayout(btn_row)

        gate = get_gate_settings()
        self.chk_gate.setChecked(gate["gate_enabled"])
        self.spin_max_diff.setValue(gate["size_gate_max_diff_px"])

        self.btn_browse.clicked.connect(self._browse)
        self.btn_save.clicked.connect(self._save)
        self.btn_cancel.clicked.connect(self.reject)

    def _browse(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择输出根目录", self.edit_outputs.text())
        if path:
            self.edit_outputs.setText(path)

    def _save(self) -> None:
        set_gate_settings(self.chk_gate.isChecked(), self.spin_max_diff.value())
        outputs = self.edit_outputs.text().strip()
        if outputs:
            set_outputs_dir(outputs)
        self.accept()
