"""深色专业主题：调色板常量 + 全局 QSS"""

from __future__ import annotations

# ----- 调色板 -----
BG_0 = "#0e1420"      # 最底层背景
BG_1 = "#131a29"      # 面板背景
BG_2 = "#1b2436"      # 卡片/输入背景
BG_3 = "#232f47"      # hover / 分隔
BORDER = "#2b3852"
TEXT = "#e6ecf5"
TEXT_DIM = "#8a97ad"
TEXT_MUTE = "#5d6b82"

ACCENT = "#4c8dff"     # 主强调
ACCENT_Hi = "#6aa1ff"
OK = "#30a46c"
NG = "#e5484d"
REVIEW = "#f5a623"
WARN = "#f5a623"

FONT = "'Microsoft YaHei UI', 'Segoe UI', 'PingFang SC', sans-serif"


QSS = f"""
* {{
    font-family: {FONT};
    font-size: 13px;
    color: {TEXT};
    outline: none;
}}
QWidget#RootWindow {{
    background: {BG_0};
}}
QWidget#TitleBar {{
    background: {BG_1};
    border-bottom: 1px solid {BORDER};
}}
QLabel#AppTitle {{ font-size: 14px; font-weight: 600; color: {TEXT}; }}
QLabel#AppSub {{ color: {TEXT_MUTE}; font-size: 12px; }}

/* 面板容器 */
QFrame.Panel, QWidget.Panel {{
    background: {BG_1};
    border: 1px solid {BORDER};
    border-radius: 12px;
}}
QLabel.PanelTitle {{
    font-size: 13px; font-weight: 600; color: {TEXT};
    padding: 2px 2px 6px 2px;
}}
QLabel.Hint {{ color: {TEXT_MUTE}; font-size: 12px; }}
QLabel.SectionLabel {{ color: {TEXT_DIM}; font-size: 12px; font-weight: 600; }}

/* 普通按钮 */
QPushButton {{
    background: {BG_2};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 7px 14px;
    color: {TEXT};
}}
QPushButton:hover {{ background: {BG_3}; border-color: {ACCENT}; }}
QPushButton:pressed {{ background: {BG_1}; }}
QPushButton:disabled {{ color: {TEXT_MUTE}; border-color: {BORDER}; background: {BG_1}; }}

/* 主行动按钮 */
QPushButton#Primary {{
    background: {ACCENT}; border: 1px solid {ACCENT}; color: white; font-weight: 600;
}}
QPushButton#Primary:hover {{ background: {ACCENT_Hi}; border-color: {ACCENT_Hi}; }}
QPushButton#Primary:disabled {{ background: {BG_2}; border-color: {BORDER}; color: {TEXT_MUTE}; }}

/* 标题栏窗口按钮 */
QPushButton.WinBtn {{
    background: transparent; border: none; border-radius: 6px;
    min-width: 34px; max-width: 34px; min-height: 26px;
    color: {TEXT_DIM}; font-size: 14px;
}}
QPushButton.WinBtn:hover {{ background: {BG_3}; color: {TEXT}; }}
QPushButton#CloseBtn:hover {{ background: {NG}; color: white; }}

/* 工具按钮(画布工具) */
QToolButton {{
    background: {BG_2}; border: 1px solid {BORDER}; border-radius: 8px;
    padding: 6px; color: {TEXT_DIM};
}}
QToolButton:hover {{ background: {BG_3}; color: {TEXT}; }}
QToolButton:checked {{ background: {ACCENT}; border-color: {ACCENT}; color: white; }}

/* 输入类 */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit {{
    background: {BG_2}; border: 1px solid {BORDER}; border-radius: 8px;
    padding: 5px 8px; selection-background-color: {ACCENT};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QPlainTextEdit:focus {{
    border-color: {ACCENT};
}}
QSpinBox, QDoubleSpinBox {{
    padding-right: 8px;
    min-height: 28px;
}}
QFrame#ParamValueBox {{
    background: {BG_2};
    border: 1px solid {BORDER};
    border-radius: 8px;
}}
QFrame#ParamValueBox:hover {{
    border-color: {ACCENT};
}}
QSpinBox#ParamValueSpin, QDoubleSpinBox#ParamValueSpin {{
    background: transparent;
    border: none;
    border-radius: 0;
    padding: 0 6px 0 8px;
    min-height: 0;
    max-height: 26px;
}}
QSpinBox#ParamValueSpin:focus, QDoubleSpinBox#ParamValueSpin:focus {{
    border: none;
}}
QFrame#ParamStepBox {{
    background: {BG_3};
    border: none;
    border-left: 1px solid {BORDER};
    border-top-right-radius: 7px;
    border-bottom-right-radius: 7px;
}}
QToolButton#ParamStepUp, QToolButton#ParamStepDown {{
    background: transparent;
    border: none;
    border-radius: 0;
    color: {TEXT_DIM};
    font-size: 8px;
    font-weight: 700;
    padding: 0;
    margin: 0;
    min-width: 20px;
    max-width: 20px;
    min-height: 13px;
}}
QToolButton#ParamStepUp {{
    border-bottom: 1px solid {BORDER};
}}
QToolButton#ParamStepUp:hover, QToolButton#ParamStepDown:hover {{
    background: {ACCENT};
    color: white;
}}
QToolButton#ParamStepUp:pressed, QToolButton#ParamStepDown:pressed {{
    background: {ACCENT_Hi};
    color: white;
}}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox QAbstractItemView {{
    background: {BG_2}; border: 1px solid {BORDER};
    selection-background-color: {ACCENT}; outline: none;
}}

/* 单选/复选框 */
QCheckBox, QRadioButton {{ spacing: 8px; }}
QCheckBox::indicator {{
    width: 18px; height: 18px; border-radius: 5px;
    border: 1px solid {BORDER}; background: {BG_2};
}}
QCheckBox::indicator:hover {{ border-color: {ACCENT}; }}
QCheckBox::indicator:checked {{
    background: {ACCENT}; border-color: {ACCENT};
    image: none;
}}
QRadioButton::indicator {{
    width: 16px; height: 16px; border-radius: 8px;
    border: 1px solid {BORDER}; background: {BG_2};
}}
QRadioButton::indicator:hover {{ border-color: {ACCENT}; }}
QRadioButton::indicator:checked {{
    background: {ACCENT}; border: 4px solid {BG_2};
}}

/* 列表/表格 */
QListWidget, QTableWidget, QTreeWidget {{
    background: {BG_1}; border: 1px solid {BORDER}; border-radius: 10px;
    alternate-background-color: {BG_2};
}}
QListWidget::item {{ padding: 6px 8px; border-radius: 6px; }}
QListWidget::item:selected {{ background: {ACCENT}; color: white; }}
QListWidget::item:hover {{ background: {BG_3}; }}
QHeaderView::section {{
    background: {BG_2}; color: {TEXT_DIM}; border: none;
    border-bottom: 1px solid {BORDER}; padding: 6px 8px; font-weight: 600;
}}
QTableWidget {{ gridline-color: {BG_3}; }}
QTableWidget::item {{ padding: 4px 6px; }}
QTableWidget::item:selected {{ background: {ACCENT}; color: white; }}

/* 滚动条 */
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {BG_3}; border-radius: 5px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: {BORDER}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {BG_3}; border-radius: 5px; min-width: 30px; }}

/* 参数滑杆 */
QSlider::groove:horizontal {{
    height: 4px; background: {BG_3}; border-radius: 2px;
}}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    width: 14px; height: 14px; margin: -6px 0; border-radius: 7px;
    background: {ACCENT_Hi}; border: 2px solid {BG_0};
}}
QLabel.ParamHintIcon {{
    color: {TEXT_MUTE};
    font-size: 11px;
    padding: 0 2px;
}}
QLabel.ParamHintIcon:hover {{
    color: {ACCENT};
}}

/* 分组框 */
QGroupBox {{
    border: 1px solid {BORDER}; border-radius: 10px; margin-top: 10px;
    padding: 10px 8px 8px 8px; background: {BG_2};
}}
QGroupBox::title {{
    subcontrol-origin: margin; left: 12px; padding: 0 4px;
    color: {TEXT_DIM}; font-weight: 600;
}}

/* 进度条 */
QProgressBar {{
    background: {BG_2}; border: 1px solid {BORDER}; border-radius: 7px;
    text-align: center; height: 14px; color: {TEXT_DIM};
}}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 6px; }}

QToolTip {{
    background: {BG_2}; color: {TEXT}; border: 1px solid {ACCENT};
    border-radius: 6px; padding: 6px 10px; font-size: 12px;
}}

/* 滚动区域（避免系统默认白底） */
QScrollArea {{
    background: transparent;
    border: none;
}}
QScrollArea > QWidget > QWidget {{
    background: {BG_1};
}}
QWidget#ScrollHost {{
    background: {BG_1};
}}
QWidget#ScrollHost QLabel {{
    background: transparent;
    color: {TEXT};
}}

/* 尺寸拖拽角标 */
QSizeGrip {{
    background: transparent;
    width: 16px; height: 16px;
}}

/* 对话框 / 消息框 */
QDialog, QMessageBox {{
    background: {BG_1};
    border: 1px solid {BORDER};
}}
QMessageBox QLabel {{
    color: {TEXT};
    background: transparent;
    font-size: 13px;
    min-width: 300px;
}}
QMessageBox QPushButton {{
    min-width: 78px; padding: 6px 16px;
}}
QDialog QLabel {{ color: {TEXT}; background: transparent; }}
QWidget#StatusBar {{
    background: {BG_1};
    border-top: 1px solid {BORDER};
}}
QSplitter::handle {{ background: transparent; }}
QSplitter::handle:hover {{ background: {BG_3}; }}
"""
