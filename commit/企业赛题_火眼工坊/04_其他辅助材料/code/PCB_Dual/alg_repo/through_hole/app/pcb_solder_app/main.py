"""应用入口：创建 QApplication，应用主题，显示主窗口"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("PYTHONUTF8", "1")


def main() -> int:
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import QApplication

    from .ui import theme
    from .ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("PCB Through-Hole Solder Inspection")
    app.setFont(QFont("Microsoft YaHei UI", 10))
    app.setStyle("Fusion")
    app.setStyleSheet(theme.QSS)

    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
