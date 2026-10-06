"""简洁现代工业软件主题：白底黑字，清晰层级，低干扰。"""

from __future__ import annotations

# 背景：工作台浅灰 → 面板纯白 → 控件浅底
BG_0 = "#f4f5f7"
BG_1 = "#ffffff"
BG_2 = "#f0f1f3"
BG_3 = "#e6e8ec"
BG_4 = "#d5d9e0"

BORDER = "#d0d5de"
BORDER_SOFT = "#e4e7ec"
BORDER_FOCUS = "#0f766e"

TEXT = "#14181f"
TEXT_DIM = "#4b5565"
TEXT_MUTE = "#8a93a3"

# 强调色：深青钢，白底上更稳
ACCENT = "#0f766e"
ACCENT_Hi = "#0d9488"
ACCENT_DIM = "#115e59"

OK = "#15803d"
OK_DIM = "#166534"
NG = "#dc2626"
NG_DIM = "#b91c1c"
REVIEW = "#ca8a04"
ERROR = "#ea580c"
RUNNING = "#0891b2"
SKIP = TEXT_MUTE

FONT = "'Microsoft YaHei UI', 'Segoe UI', 'PingFang SC', sans-serif"
FONT_MONO = "'Cascadia Mono', 'Consolas', 'Microsoft YaHei UI', monospace"


QSS = f"""
* {{
    font-family: {FONT};
    font-size: 13px;
    color: {TEXT};
    outline: none;
}}

QMainWindow {{
    background: {BG_0};
}}
QWidget {{
    background: transparent;
    color: {TEXT};
}}
QMainWindow > QWidget {{
    background: {BG_0};
}}

/* ---- 顶栏 ---- */
QWidget#AppHeader {{
    background: {BG_1};
    border-bottom: 1px solid {BORDER};
}}
QLabel#BrandMark {{
    color: {ACCENT};
    font-size: 16px;
    font-weight: 800;
    letter-spacing: 1px;
    background: transparent;
}}
QLabel#BrandTitle {{
    color: {TEXT};
    font-size: 15px;
    font-weight: 700;
    background: transparent;
}}
QLabel#BrandSub {{
    color: {TEXT_MUTE};
    font-size: 11px;
    background: transparent;
}}
QLabel#HeaderMeta {{
    color: {TEXT_DIM};
    font-size: 12px;
    background: transparent;
}}

/* ---- Tab ---- */
QTabWidget::pane {{
    border: 1px solid {BORDER};
    border-radius: 10px;
    background: {BG_1};
    top: -1px;
    margin-top: 2px;
}}
QTabWidget#MainTabs::pane {{
    border-top-left-radius: 0;
}}
QTabBar::tab {{
    background: transparent;
    color: {TEXT_MUTE};
    border: none;
    border-bottom: 2px solid transparent;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    padding: 10px 18px;
    margin-right: 2px;
    min-width: 72px;
    font-weight: 600;
}}
QTabBar::tab:selected {{
    color: {TEXT};
    background: {BG_1};
    border-bottom: 2px solid {ACCENT};
}}
QTabBar::tab:hover:!selected {{
    color: {TEXT_DIM};
    background: {BG_2};
}}

QMenuBar {{
    background: {BG_1};
    border-bottom: 1px solid {BORDER};
    padding: 2px 6px;
    color: {TEXT};
}}
QMenuBar::item {{
    padding: 6px 12px;
    background: transparent;
    border-radius: 6px;
    color: {TEXT};
}}
QMenuBar::item:selected {{
    background: {BG_2};
}}
QMenu {{
    background: {BG_1};
    border: 1px solid {BORDER};
    padding: 6px;
    border-radius: 8px;
}}
QMenu::item {{
    padding: 7px 22px;
    border-radius: 6px;
    color: {TEXT};
}}
QMenu::item:selected {{
    background: {BG_2};
    color: {TEXT};
}}
QMenu::separator {{
    height: 1px;
    background: {BORDER_SOFT};
    margin: 4px 10px;
}}

/* ---- 面板 / 卡片 ---- */
QFrame.Panel, QWidget.Panel, QFrame#Card {{
    background: {BG_1};
    border: 1px solid {BORDER};
    border-radius: 10px;
}}
QFrame#CardElevated {{
    background: {BG_1};
    border: 1px solid {BORDER};
    border-radius: 10px;
}}
QLabel.PanelTitle {{
    font-size: 13px;
    font-weight: 700;
    color: {TEXT};
    padding: 0 0 2px 0;
    background: transparent;
}}
QLabel.Hint {{
    color: {TEXT_MUTE};
    font-size: 12px;
    background: transparent;
}}
QLabel.SectionLabel {{
    color: {TEXT_DIM};
    font-size: 12px;
    font-weight: 700;
    background: transparent;
}}
QLabel#VerdictHero {{
    font-size: 28px;
    font-weight: 800;
    letter-spacing: 1px;
    background: transparent;
    color: {TEXT};
}}

QFrame#ToolbarStrip {{
    background: {BG_1};
    border: 1px solid {BORDER};
    border-radius: 10px;
}}

/* ---- 抽屉 ---- */
QToolButton#DrawerHeader {{
    background: {BG_2};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 9px 12px;
    color: {TEXT};
    font-weight: 700;
    text-align: left;
    min-height: 24px;
}}
QToolButton#DrawerHeader:hover {{
    background: {BG_3};
    border-color: {ACCENT};
}}
QToolButton#DrawerHeader:checked {{
    background: {BG_1};
    border-color: {ACCENT};
    color: {TEXT};
}}

/* ---- 按钮 ---- */
QPushButton {{
    background: {BG_1};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 8px 14px;
    color: {TEXT};
    font-weight: 600;
}}
QPushButton:hover {{
    background: {BG_2};
    border-color: {ACCENT};
}}
QPushButton:pressed {{
    background: {BG_3};
}}
QPushButton:disabled {{
    color: {TEXT_MUTE};
    border-color: {BORDER_SOFT};
    background: {BG_2};
}}

QPushButton#Primary {{
    background: {ACCENT};
    border: 1px solid {ACCENT};
    color: #ffffff;
    font-weight: 700;
    padding: 9px 18px;
}}
QPushButton#Primary:hover {{
    background: {ACCENT_Hi};
    border-color: {ACCENT_Hi};
    color: #ffffff;
}}
QPushButton#Primary:disabled {{
    background: {BG_3};
    border-color: {BORDER};
    color: {TEXT_MUTE};
}}

QPushButton#Danger {{
    background: {NG};
    border: 1px solid {NG};
    color: #ffffff;
    font-weight: 700;
}}
QPushButton#Danger:hover {{
    background: #ef4444;
    color: #ffffff;
}}

QPushButton#Ghost {{
    background: transparent;
    border: 1px solid {BORDER};
    color: {TEXT_DIM};
}}
QPushButton#Ghost:hover {{
    color: {TEXT};
    border-color: {ACCENT};
    background: {BG_2};
}}

QToolButton {{
    background: {BG_1};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 7px 10px;
    color: {TEXT_DIM};
    font-weight: 600;
}}
QToolButton:hover {{
    background: {BG_2};
    color: {TEXT};
    border-color: {ACCENT};
}}
QToolButton:checked {{
    background: {ACCENT};
    border-color: {ACCENT};
    color: #ffffff;
}}

QToolButton#ParamStepBtn {{
    background: {BG_1};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 0;
    font-size: 9px;
    color: {TEXT};
}}
QToolButton#ParamStepBtn:hover {{
    background: {BG_2};
    border-color: {ACCENT};
}}
QToolButton#ParamStepBtn:pressed {{
    background: {ACCENT};
    border-color: {ACCENT};
    color: #ffffff;
}}

/* ---- 输入 ---- */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit {{
    background: {BG_1};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 6px 10px;
    color: {TEXT};
    selection-background-color: {ACCENT};
    selection-color: #ffffff;
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus,
QPlainTextEdit:focus, QTextEdit:focus {{
    border-color: {BORDER_FOCUS};
}}
QComboBox::drop-down {{
    border: none;
    width: 22px;
}}
QComboBox QAbstractItemView {{
    background: {BG_1};
    border: 1px solid {BORDER};
    color: {TEXT};
    selection-background-color: {BG_2};
    selection-color: {TEXT};
    outline: none;
    padding: 4px;
}}

QCheckBox, QRadioButton {{
    spacing: 8px;
    background: transparent;
    color: {TEXT};
}}
QCheckBox::indicator {{
    width: 17px;
    height: 17px;
    border-radius: 4px;
    border: 1px solid {BORDER};
    background: {BG_1};
}}
QCheckBox::indicator:hover {{
    border-color: {ACCENT};
}}
QCheckBox::indicator:checked {{
    background: {ACCENT};
    border-color: {ACCENT};
}}
QRadioButton::indicator {{
    width: 15px;
    height: 15px;
    border-radius: 8px;
    border: 1px solid {BORDER};
    background: {BG_1};
}}
QRadioButton::indicator:hover {{
    border-color: {ACCENT};
}}
QRadioButton::indicator:checked {{
    background: {ACCENT};
    border: 4px solid {BG_1};
}}

QListWidget, QTableWidget, QTreeWidget {{
    background: {BG_1};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 4px;
    color: {TEXT};
    alternate-background-color: {BG_2};
}}
QListWidget::item {{
    padding: 8px 10px;
    border-radius: 6px;
    margin: 1px 2px;
    color: {TEXT};
}}
QListWidget::item:selected {{
    background: {BG_2};
    color: {TEXT};
    border: 1px solid {BORDER};
}}
QListWidget::item:hover:!selected {{
    background: {BG_2};
}}
QHeaderView::section {{
    background: {BG_2};
    color: {TEXT_DIM};
    border: none;
    border-bottom: 1px solid {BORDER};
    padding: 7px 8px;
    font-weight: 700;
}}
QTableWidget {{
    gridline-color: {BORDER_SOFT};
}}
QTableWidget::item:selected {{
    background: {BG_2};
    color: {TEXT};
}}

QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 4px 2px;
}}
QScrollBar::handle:vertical {{
    background: {BG_4};
    border-radius: 5px;
    min-height: 28px;
}}
QScrollBar::handle:vertical:hover {{
    background: {BORDER};
}}
QScrollBar::add-line, QScrollBar::sub-line {{
    height: 0;
    width: 0;
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
    margin: 2px 4px;
}}
QScrollBar::handle:horizontal {{
    background: {BG_4};
    border-radius: 5px;
    min-width: 28px;
}}

QSlider::groove:horizontal {{
    height: 4px;
    background: {BG_3};
    border-radius: 2px;
}}
QSlider::sub-page:horizontal {{
    background: {ACCENT};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    width: 14px;
    height: 14px;
    margin: -6px 0;
    border-radius: 7px;
    background: {ACCENT};
    border: 2px solid {BG_1};
}}
QLabel.ParamHintIcon {{
    color: {TEXT_MUTE};
    font-size: 11px;
    padding: 0 2px;
    background: transparent;
}}
QLabel.ParamHintIcon:hover {{
    color: {ACCENT};
}}

QGroupBox {{
    border: 1px solid {BORDER};
    border-radius: 10px;
    margin-top: 12px;
    padding: 12px 10px 10px 10px;
    background: {BG_1};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: {TEXT_DIM};
    font-weight: 700;
    background: {BG_1};
}}

QProgressBar {{
    background: {BG_2};
    border: 1px solid {BORDER};
    border-radius: 8px;
    text-align: center;
    height: 16px;
    color: {TEXT_DIM};
    font-size: 11px;
}}
QProgressBar::chunk {{
    background: {ACCENT};
    border-radius: 7px;
}}

QToolTip {{
    background: {BG_1};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 7px 11px;
    font-size: 12px;
}}

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

QDialog, QMessageBox {{
    background: {BG_1};
    border: 1px solid {BORDER};
}}
QMessageBox QLabel {{
    color: {TEXT};
    background: transparent;
    font-size: 13px;
    min-width: 280px;
}}
QMessageBox QPushButton {{
    min-width: 78px;
    padding: 7px 16px;
}}
QDialog QLabel {{
    color: {TEXT};
    background: transparent;
}}

QSplitter::handle {{
    background: transparent;
}}
QSplitter::handle:horizontal {{
    width: 6px;
}}
QSplitter::handle:vertical {{
    height: 6px;
}}
QSplitter::handle:hover {{
    background: {BORDER_SOFT};
}}

QStatusBar {{
    background: {BG_1};
    border-top: 1px solid {BORDER};
    color: {TEXT_MUTE};
}}
QStatusBar::item {{
    border: none;
}}
"""
