"""分镜 9 演示驱动：同一张 PCB 图 传统 CV + AI 特征引擎 双检并行展示。

流程：启动内嵌服务 + 主窗口 → 自动检测页 → 选模板 → 导入 1 张 NG PCB 图 →
点击开始检测（同步双检）→ 等待结果 → 停留展示 20 秒 → 退出（不清任何数据）。

进度用 print 输出（启动时重定向到日志文件，便于外部跟踪）。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from app.config import ensure_dirs  # noqa: E402
from app.ui.main_window import MainWindow  # noqa: E402

TPL_ID = "PixPin_2026-07-01_11-21-15_OK"
NG_IMG = ROOT / "alg_repo" / "smt" / "test_images" / "excess_pair" / "pair4" / "PixPin_2026-07-01_11-21-15.png"
LOG_FILE = Path(r"d:\CGAIC\commit\03_项目视频\raw\pcb_log.txt")
READY_FILE = Path(r"d:\CGAIC\commit\03_项目视频\raw\pcb_ready.txt")


def log(msg: str) -> None:
    print(f"[dual_demo] {msg}", flush=True)
    try:
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(f"[dual_demo] {msg}\n")
    except OSError:
        pass


def main() -> int:
    ensure_dirs()
    from server import start_server_background
    start_server_background()

    app = QApplication(sys.argv)
    app.setApplicationName("PCB 缺陷检测系统 v2")
    win = MainWindow()
    win.show()

    page = win.inspect_page
    log("WINDOW_SHOWN")
    try:
        READY_FILE.write_text("ready", encoding="utf-8")
    except OSError:
        pass

    def step_select() -> None:
        win.tabs.setCurrentWidget(page)
        page.select_template(TPL_ID)
        tpl = page.current_template
        log(f"step_select: 模板={tpl.display_name if tpl else None} "
            f"engine_mode={getattr(tpl, 'engine_mode', None)}")
        QTimer.singleShot(3000, step_import)

    def step_import() -> None:
        page._append_paths([NG_IMG])
        log(f"step_import: 队列数={len(page.queue)} "
            f"首项={page.queue[0].name if page.queue else None}")
        QTimer.singleShot(3000, step_run)

    def step_run() -> None:
        log(f"step_run: btn_run.enabled={page.btn_run.isEnabled()}")
        page.btn_run.click()  # 同步双检（~0.5s）
        QTimer.singleShot(1000, step_wait)

    def step_wait() -> None:
        run = page.last_run
        if run is None or run.summary is None:
            QTimer.singleShot(1000, step_wait)
            return
        s = run.summary
        feats = [r for r in s.results if getattr(r, "item_id", "") == "aoi_feature"]
        log(f"step_wait: overall={s.overall} ng={s.ng_count} "
            f"elapsed={s.elapsed_ms}ms aoi_feature={feats[0].message if feats else 'N/A'}")
        QTimer.singleShot(20000, step_exit)  # 停留展示 20 秒

    def step_exit() -> None:
        log("DEMO_DONE")
        app.quit()

    QTimer.singleShot(4000, step_select)  # 窗口显示后停留 4s，给录屏启动留缓冲
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
