"""T15 UI 离屏冒烟：8 大页面实例化 + 切换 + 在线数据加载容错。

以子进程方式由 run_all 调用（QT_QPA_PLATFORM=offscreen），
也可独立运行：python -m tests.ui.offscreen
"""
from __future__ import annotations

import os
import sys
import time

from ..framework import Ctx, run_test


def _ui_smoke() -> tuple[int, int, list[str]]:
    """返回 (通过页面数, 页面总数, 错误列表)。"""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QMessageBox
    # 离屏环境无交互：屏蔽模态弹窗（后端离线时错误弹窗会阻塞事件循环）
    QMessageBox.warning = staticmethod(lambda *a, **k: None)
    QMessageBox.critical = staticmethod(lambda *a, **k: None)
    QMessageBox.information = staticmethod(lambda *a, **k: None)
    app = QApplication([])
    from ui.main_window import MainWindow
    w = MainWindow()
    w.show()
    app.processEvents()
    time.sleep(1.0)
    app.processEvents()
    errors: list[str] = []
    n_pages = 0
    stack = getattr(w, "stack", None) or getattr(w, "stacked", None)
    nav = getattr(w, "nav", None) or getattr(w, "nav_list", None)
    if stack is None:
        # 兜底：遍历查找 QStackedWidget
        from PySide6.QtWidgets import QStackedWidget
        stack = w.findChild(QStackedWidget)
    total = stack.count() if stack is not None else 0
    for i in range(total):
        try:
            if nav is not None and hasattr(nav, "setCurrentRow"):
                nav.setCurrentRow(i)
            else:
                stack.setCurrentIndex(i)
            app.processEvents()
            time.sleep(0.3)
            app.processEvents()
            n_pages += 1
        except Exception as e:  # noqa: BLE001
            errors.append(f"page{i}: {e}")
    w.close()
    return n_pages, total, errors


def run(ctx: Ctx) -> None:
    n, total, errors = _ui_smoke()
    ctx.check("UI 当前导航页面全部实例化并切换（含特征工程）",
              n == total and total >= 11 and not errors,
              f"pages={n}/{total} errors={errors}")
    ctx.metrics["ui"] = {"pages_loaded": n, "errors": errors}


if __name__ == "__main__":
    run_test("t15", "UI 离屏冒烟", run)
