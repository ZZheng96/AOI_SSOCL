"""浅色现代主题（LIGHT_QSS）：工业质检系统风格。

主色 #2F6FED（蓝），背景 #F5F7FA，卡片白底圆角 8px 浅灰边框；
成功 #22C55E / 危险 #EF4444 / 警告 #F59E0B。
导航栏白底、选中项左侧蓝色指示条 + 浅蓝底；按钮圆角、主按钮蓝底白字 hover 加深；
表格隔行 #F8FAFC、表头浅灰底加粗。
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication

# ── 主题色板 ─────────────────────────────────────────────
PRIMARY = "#2F6FED"        # 主色（蓝）
PRIMARY_DARK = "#2559C7"   # 主色加深（hover）
PRIMARY_LIGHT = "#EAF1FE"  # 浅蓝底（选中）
SUCCESS = "#22C55E"        # 成功（绿）
DANGER = "#EF4444"         # 危险（红）
WARNING = "#F59E0B"        # 警告（橙）
BG = "#F5F7FA"             # 全局背景
CARD_BORDER = "#E5E7EB"    # 卡片浅灰边框
TEXT = "#1F2937"           # 主文字
TEXT_SUB = "#6B7280"       # 次要文字

# QComboBox 下拉箭头图片（QSS 自定义 drop-down 后需显式指定，
# 否则 Fusion 风格下箭头不渲染，用户反馈"下拉标志看不见"）
_ARROW_DOWN = (Path(__file__).resolve().parents[1]
               / "assets" / "arrow_down.png").as_posix()
# 数值输入框（QSpinBox/QDoubleSpinBox）的上下箭头图片：同样必须显式指定，
# 否则两个调节按钮会渲染成空白小方块（前端反馈 2026-10-14）
_ARROW_UP = (Path(__file__).resolve().parents[1]
             / "assets" / "arrow_up.png").as_posix()

LIGHT_QSS = f"""
* {{
    font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
    font-size: 9.5pt;
    color: {TEXT};
    outline: none;
}}
QMainWindow, QDialog, QWidget {{
    background: {BG};
}}
/* ── 卡片 ─────────────────────────────── */
QFrame[card="true"] {{
    background: #FFFFFF;
    border: 1px solid {CARD_BORDER};
    border-radius: 8px;
}}
QLabel[heading="true"] {{
    font-size: 12pt;
    font-weight: 600;
}}
QLabel[subtext="true"] {{
    color: {TEXT_SUB};
}}
/* 徽标（状态小标签） */
QLabel[badge="success"] {{
    background: #DCFCE7; color: #15803D; border-radius: 9px;
    padding: 2px 10px; font-weight: 600;
}}
QLabel[badge="danger"] {{
    background: #FEE2E2; color: #B91C1C; border-radius: 9px;
    padding: 2px 10px; font-weight: 600;
}}
QLabel[badge="warning"] {{
    background: #FEF3C7; color: #B45309; border-radius: 9px;
    padding: 2px 10px; font-weight: 600;
}}
QLabel[badge="info"] {{
    background: {PRIMARY_LIGHT}; color: {PRIMARY}; border-radius: 9px;
    padding: 2px 10px; font-weight: 600;
}}
QLabel[badge="muted"] {{
    background: #F3F4F6; color: {TEXT_SUB}; border-radius: 9px;
    padding: 2px 10px; font-weight: 600;
}}
/* 赛题问题横幅标签 */
QLabel[banner="true"] {{
    background: {PRIMARY_LIGHT}; color: {PRIMARY};
    border: 1px solid #C7DBFB; border-radius: 6px;
    padding: 4px 12px; font-weight: 600;
}}
/* ── 按钮 ─────────────────────────────── */
QPushButton {{
    background: #FFFFFF;
    border: 1px solid #D1D5DB;
    border-radius: 6px;
    padding: 6px 14px;
}}
QPushButton:hover {{ background: #F3F4F6; }}
QPushButton:pressed {{ background: #E5E7EB; }}
QPushButton:disabled {{ color: #9CA3AF; background: #F3F4F6; border-color: #E5E7EB; }}
QPushButton[primary="true"] {{
    background: {PRIMARY};
    color: #FFFFFF;
    border: none;
    font-weight: 600;
}}
QPushButton[primary="true"]:hover {{ background: {PRIMARY_DARK}; }}
QPushButton[primary="true"]:disabled {{ background: #93B4F5; color: #FFFFFF; }}
QPushButton[success="true"] {{
    background: {SUCCESS}; color: #FFFFFF; border: none; font-weight: 600;
}}
QPushButton[success="true"]:hover {{ background: #16A34A; }}
QPushButton[danger="true"] {{
    background: {DANGER}; color: #FFFFFF; border: none; font-weight: 600;
}}
QPushButton[danger="true"]:hover {{ background: #DC2626; }}
QPushButton[warning="true"] {{
    background: {WARNING}; color: #FFFFFF; border: none; font-weight: 600;
}}
QPushButton[warning="true"]:hover {{ background: #D97706; }}
QPushButton[flat="true"] {{
    border: none;
    background: transparent;
    color: {PRIMARY};
}}
QPushButton[flat="true"]:hover {{ background: {PRIMARY_LIGHT}; }}
/* 紧凑按钮（表格单元格 / 小面板内的行内操作，2026-10-14 新增：
   避免小面板里的按钮被拉伸成长条或挤成省略号） */
QPushButton[compact="true"] {{
    padding: 3px 10px;
    font-size: 8.5pt;
    min-height: 20px;
}}
/* 分段按钮（图层切换等，checkable） */
QPushButton[segment="true"] {{
    border: 1px solid #D1D5DB;
    background: #FFFFFF;
    padding: 5px 16px;
}}
QPushButton[segment="true"]:checked {{
    background: {PRIMARY};
    color: #FFFFFF;
    border-color: {PRIMARY};
    font-weight: 600;
}}
/* ── 输入控件 ─────────────────────────── */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit {{
    background: #FFFFFF;
    border: 1px solid #D1D5DB;
    border-radius: 6px;
    padding: 5px 8px;
    selection-background-color: {PRIMARY};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus,
QPlainTextEdit:focus, QTextEdit:focus {{
    border: 1px solid {PRIMARY};
}}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox::down-arrow {{
    image: url({_ARROW_DOWN});
    width: 14px;
    height: 14px;
}}
/* 数值输入框的上下调节按钮（2026-10-14：此前只显示成两个空白小方块） */
QSpinBox, QDoubleSpinBox {{ padding-right: 26px; }}
QSpinBox::up-button, QDoubleSpinBox::up-button {{
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 20px;
    margin: 1px 1px 0 0;
    background: #F9FAFB;
    border-left: 1px solid #D1D5DB;
    border-top-right-radius: 5px;
}}
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 20px;
    margin: 0 1px 1px 0;
    background: #F9FAFB;
    border-left: 1px solid #D1D5DB;
    border-bottom-right-radius: 5px;
}}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
    background: #E5E7EB;
}}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    image: url({_ARROW_UP});
    width: 12px;
    height: 12px;
}}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    image: url({_ARROW_DOWN});
    width: 12px;
    height: 12px;
}}
QComboBox QAbstractItemView {{
    background: #FFFFFF;
    border: 1px solid {CARD_BORDER};
    selection-background-color: {PRIMARY_LIGHT};
    selection-color: {TEXT};
}}
/* ── 导航列表（白底 + 选中左侧蓝色指示条） ──
   2026-10-14：不再用 border-right 画分割线（会让列表视口少 1px 并触发
   水平滚动条），改由主窗口在导航与页面之间插一条独立 1px 分隔线。 */
QListWidget#nav {{
    background: #FFFFFF;
    border: none;
    padding: 8px 0px;
}}
QListWidget#nav::item {{
    padding: 11px 14px 11px 17px;
    margin: 2px 8px 2px 0px;
    border-left: 3px solid transparent;
    border-top-right-radius: 6px;
    border-bottom-right-radius: 6px;
    color: {TEXT};
}}
QListWidget#nav::item:hover {{ background: #F3F4F6; }}
QListWidget#nav::item:selected {{
    background: {PRIMARY_LIGHT};
    border-left: 3px solid {PRIMARY};
    color: {PRIMARY};
    font-weight: 600;
}}
/* ── 表格（隔行 #F8FAFC、表头浅灰底加粗） ── */
QTableWidget, QTableView {{
    background: #FFFFFF;
    border: 1px solid {CARD_BORDER};
    border-radius: 8px;
    gridline-color: transparent;
    alternate-background-color: #F8FAFC;
    selection-background-color: {PRIMARY_LIGHT};
    selection-color: {TEXT};
}}
QHeaderView::section {{
    background: #F1F5F9;
    border: none;
    border-bottom: 1px solid {CARD_BORDER};
    padding: 7px 8px;
    font-weight: 600;
    color: {TEXT_SUB};
}}
QTableWidget::item, QTableView::item {{ padding: 4px 6px; border: none; }}
/* ── 选项卡 ───────────────────────────── */
QTabWidget::pane {{
    border: 1px solid {CARD_BORDER};
    border-radius: 8px;
    background: #FFFFFF;
    top: -1px;
}}
QTabBar::tab {{
    background: transparent;
    padding: 8px 18px;
    margin-right: 2px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    color: {TEXT_SUB};
}}
QTabBar::tab:selected {{
    background: #FFFFFF;
    color: {PRIMARY};
    font-weight: 600;
    border: 1px solid {CARD_BORDER};
    border-bottom: 2px solid #FFFFFF;
}}
QTabBar::tab:hover:!selected {{ color: {TEXT}; }}
/* ── 进度条 ───────────────────────────── */
QProgressBar {{
    background: #EEF2F7;
    border: none;
    border-radius: 5px;
    height: 10px;
    text-align: center;
    color: {TEXT_SUB};
    font-size: 8pt;
}}
QProgressBar::chunk {{ background: {PRIMARY}; border-radius: 5px; }}
/* ── 滚动条 ───────────────────────────── */
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 2px;
}}
QScrollBar::handle:vertical {{
    background: #D1D5DB;
    border-radius: 5px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{ background: #9CA3AF; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
    margin: 2px;
}}
QScrollBar::handle:horizontal {{
    background: #D1D5DB;
    border-radius: 5px;
    min-width: 30px;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
/* ── 其它 ─────────────────────────────── */
QToolTip {{
    background: #FFFFFF;
    border: 1px solid {CARD_BORDER};
    padding: 4px 8px;
}}
QSplitter::handle {{ background: {CARD_BORDER}; }}
QGroupBox {{
    border: 1px solid {CARD_BORDER};
    border-radius: 8px;
    margin-top: 14px;
    padding-top: 10px;
    background: #FFFFFF;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: {TEXT_SUB};
}}
QCheckBox, QRadioButton {{ spacing: 6px; }}
QStatusBar {{ background: #FFFFFF; border-top: 1px solid {CARD_BORDER}; }}
QMessageBox QLabel {{ font-size: 10pt; }}
"""


def apply_light_theme(app: QApplication) -> None:
    """应用浅色现代主题（Fusion 风格 + LIGHT_QSS + 微软雅黑字体）。"""
    app.setStyle("Fusion")
    font = QFont("Microsoft YaHei UI")
    font.setPointSizeF(9.5)
    app.setFont(font)

    palette = app.palette()
    palette.setColor(QPalette.Window, QColor(BG))
    palette.setColor(QPalette.Base, QColor("#FFFFFF"))
    palette.setColor(QPalette.Text, QColor(TEXT))
    palette.setColor(QPalette.WindowText, QColor(TEXT))
    palette.setColor(QPalette.Highlight, QColor(PRIMARY))
    palette.setColor(QPalette.HighlightedText, QColor("#FFFFFF"))
    app.setPalette(palette)
    app.setStyleSheet(LIGHT_QSS)
