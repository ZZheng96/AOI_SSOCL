# -*- coding: utf-8 -*-
"""v2 逐分镜录制驱动（PCB_Dual 双引擎系统，开放赛题独有分镜）。

策略与 AOI 版一致：每分镜独立录 mp4（2s 引导 + 正片 + 1s 余量），
ops 用 QTimer 调度（避免 exec() 弹窗阻塞主线程调用栈），
剪辑裁 ss=2 取正片、逐 clip 烧字幕后拼接。

用法：
    python tests\\_record_shots_v2.py               # 录全部 6 分镜
    python tests\\_record_shots_v2.py 20_pcb_detect # 只录指定分镜

前置：关闭已运行的 PCB 实例（避免同标题窗口冲突）；AOI 窗口最小化。
素材输出：../AOI_sys/docs/视频录制素材/raw/v2/clips/<shot>.mp4（与 AOI 同目录，便于统一拼接）
"""
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FF = (r"C:\Users\imia\AppData\Local\Microsoft\WinGet\Packages"
      r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
      r"\ffmpeg-9.0.1-full_build\bin\ffmpeg.exe")
FP = FF.replace("ffmpeg.exe", "ffprobe.exe")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIPS = os.path.join(os.path.dirname(ROOT), "AOI_sys",
                     "docs", "视频录制素材", "raw", "v2", "clips")
os.makedirs(CLIPS, exist_ok=True)

WIN_TITLE = "PCB 缺陷检测系统"
TPL_ID = "PixPin_2026-07-01_11-21-15_OK"  # SMT 焊点示例（已发布）
NG_IMG = Path(ROOT) / "alg_repo" / "smt" / "test_images" / "excess_pair" / "pair4" / "PixPin_2026-07-01_11-21-15.png"
LEAD = 2.0    # 引导段（剪辑裁掉）
TAIL = 3.0    # 录制总长 = 正片 + TAIL
BASE = "http://127.0.0.1:8021"

import logging
logging.basicConfig(level=logging.ERROR)

import requests  # noqa: E402

# 后端：已被占用则复用，否则拉起
try:
    if requests.get(f"{BASE}/api/health", timeout=1).json().get("status") != "ok":
        raise RuntimeError("bad")
    print("[启动] 复用已运行后端", flush=True)
except Exception:  # noqa: BLE001
    from server import start_server_background
    start_server_background()
    for _ in range(100):
        try:
            if requests.get(f"{BASE}/api/health", timeout=1).json().get("status") == "ok":
                break
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.3)
    print("[启动] 后端就绪", flush=True)

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from app.config import ensure_dirs  # noqa: E402
from app.ui.main_window import MainWindow  # noqa: E402

ISSUES = []

ensure_dirs()
app = QApplication(sys.argv)
app.setApplicationName("PCB 缺陷检测系统 v2")
win = MainWindow()
win.resize(1440, 900)
win.move(30, 20)
win.show()
win.raise_()
win.activateWindow()

page_inspect = win.inspect_page
page_history = win.history_page
page_studio = win.studio_page
page_model = win.model_page
page_detect = win.detect_page
page_prep = win.preprocess_page


def pump(ms):
    end = time.time() + ms / 1000
    while time.time() < end:
        app.processEvents()
        time.sleep(0.02)


def goto(page, settle=1200):
    win.tabs.setCurrentWidget(page)
    pump(settle)


# ═══════════ 页内操作（offset 均为正片秒） ═══════════
def op_select_tpl():
    page_inspect.select_template(TPL_ID)
    tpl = page_inspect.current_template
    if tpl:
        print(f"  [op] 选中模板 {tpl.display_name} v{tpl.version}", flush=True)
    else:
        ISSUES.append("自动检测页选模板失败")


def op_import_ng():
    page_inspect._append_paths([NG_IMG])
    print(f"  [op] 导入 NG 样例图（队列 {len(page_inspect.queue)} 张）", flush=True)


def op_run():
    page_inspect.btn_run.click()
    print("  [op] 开始检测（双引擎同步）", flush=True)


def op_history_refresh():
    page_history.refresh()
    pump(600)
    if page_history.table.rowCount():
        page_history.table.selectRow(0)
        print(f"  [op] 历史记录选中首行（共 {page_history.table.rowCount()} 行）", flush=True)
    else:
        ISSUES.append("历史页无记录")


def op_studio_pick():
    lp = page_studio.list_panel
    for i, tpl in enumerate(lp._templates):
        if tpl.id == TPL_ID:
            lp.list.setCurrentRow(i)
            print(f"  [op] 模板库选中 {tpl.display_name} v{tpl.version}", flush=True)
            return
    ISSUES.append("模板库找不到 SMT 焊点示例")


def op_debug_tpl():
    cb = page_detect.template_picker.combo
    idx = cb.findData(TPL_ID)
    if idx >= 0:
        cb.setCurrentIndex(idx)
        print("  [op] 算法调试页选中 SMT 焊点示例模板", flush=True)
    else:
        ISSUES.append("算法调试页模板下拉无 SMT 焊点示例")


# ═══════════ 分镜表：(name, 页面, 正片时长s, [(offset, op)]) ═══════════
SHOTS = [
    ("20_pcb_detect", page_inspect, 22, [
        (1, op_select_tpl), (5, op_import_ng), (9, op_run)]),
    ("21_pcb_history", page_history, 8, [(1, op_history_refresh)]),
    ("22_pcb_template", page_studio, 16, [(3, op_studio_pick)]),
    ("23_pcb_model", page_model, 10, []),
    ("24_pcb_debug", page_detect, 12, [(2, op_debug_tpl)]),
    ("25_pcb_prep", page_prep, 10, []),
]


def start_rec(name, dur):
    out = os.path.join(CLIPS, f"{name}.mp4")
    logp = os.path.join(CLIPS, f"{name}.log")
    lf = open(logp, "wb")
    cmd = [FF, "-y", "-f", "gdigrab", "-framerate", "30",
           "-i", f"title={WIN_TITLE}", "-t", str(dur + TAIL),
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
           "-pix_fmt", "yuv420p", out]
    proc = subprocess.Popen(cmd, stdout=lf, stderr=lf)
    return proc, out, lf


def verify(name, out, dur):
    try:
        r = subprocess.run([FP, "-v", "error", "-show_entries", "format=duration",
                            "-of", "default=noprint_wrappers=1:nokey=1", out],
                           capture_output=True, text=True, timeout=30)
        d = float(r.stdout.strip())
    except Exception as e:  # noqa: BLE001
        d = -1
        print(f"  [校验] ffprobe 异常 {e}", flush=True)
    ok = d >= dur + TAIL - 1.5
    print(f"  [校验] {name}.mp4 时长 {d:.1f}s（期望 ≥{dur + TAIL - 1.5:.1f}）"
          f" {'✅' if ok else '❌'}", flush=True)
    if not ok:
        ISSUES.append(f"{name} 录制时长异常 {d}")


def _safe_op(name, off, fn):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        ISSUES.append(f"{name} op@{off}s 异常 {type(e).__name__}: {e}")
        print(f"  [op 异常] {type(e).__name__}: {e}", flush=True)


def main():
    only = set(sys.argv[1:])
    pump(1500)
    for name, page, dur, ops in SHOTS:
        if only and name not in only:
            continue
        print(f"\n[shot] {name}（正片 {dur}s）", flush=True)
        goto(page)
        win.raise_()
        win.activateWindow()
        pump(400)
        proc, out, lf = start_rec(name, dur)
        for off, fn in ops:
            QTimer.singleShot(int((LEAD + off) * 1000),
                              lambda fn=fn, off=off: _safe_op(name, off, fn))
        while proc.poll() is None:
            pump(200)
        lf.close()
        verify(name, out, dur)
    print(f"\n{'='*50}", flush=True)
    if ISSUES:
        print(f"完成，{len(ISSUES)} 个问题：", flush=True)
        for it in ISSUES:
            print(f"  ❌ {it}", flush=True)
    else:
        print("全部分镜录制完成 ✅", flush=True)
    sys.exit(1 if ISSUES else 0)


main()
