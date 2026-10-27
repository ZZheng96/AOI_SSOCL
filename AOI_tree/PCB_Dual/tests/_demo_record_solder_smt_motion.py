"""PCB_Dual 连锡产线演示：连续动作录制（合成光标 + 点击波纹）。

- 只用 QWidget.grab() 抓主窗口（最大化）与 Qt 自有弹窗，不录屏。
- 虚拟时钟 15fps：每 tick 抓 1 帧，帧时间 = 序号/15；后端长计算期间以延时摄影方式稀疏抓帧。
- 交互全部为真实控件事件（QTest.mouseClick / showPopup / slider.setValue），光标与波纹为后期叠加绘制。
- 节奏读取 SLOTS_JSON（每段配音起止），输出 frames/*.jpg + manifest.json。
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import sys
import time
from pathlib import Path

from PySide6.QtCore import QPoint, QPointF, Qt, QTimer, QRectF
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QBrush, QFont
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (QApplication, QFileDialog, QMessageBox, QStyle, QStyleOptionSlider)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CATEGORY = "solder_smt"
DATA_ROOT = Path(r"D:\CGAIC\data_local") / CATEGORY
WORK = Path(os.environ.get("PCB_WORK", r"C:\Users\imia\AppData\Local\Temp\cgaic_v5\pcb_work"))
FRAMES = WORK / "frames"
SLOTS_JSON = Path(os.environ.get("SLOTS_JSON", str(WORK / "slots.json")))
FPS = 15
DT = 1.0 / FPS
STAMP = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

PAIR_DIR = ROOT / "alg_repo" / "smt" / "test_images" / "bridge_pair" / "pair4"
STANDARD = PAIR_DIR / "PixPin_2026-07-01_09-22-58_OK.png"
GOOD = STANDARD
BRIDGE = DATA_ROOT / "test" / "bridge" / "PixPin_2026-07-01_09-22-58.png"
PAD_JSON = PAIR_DIR / "pad.json"
PARAMS_JSON = PAIR_DIR / "params.json"
WORKORDER_ID = int(os.environ.get("WORKORDER_ID", "6"))
PAIR_DETECTIONS = (1079, 1077)
for _p in (STANDARD, BRIDGE, PAD_JSON, PARAMS_JSON, SLOTS_JSON):
    if not _p.is_file():
        raise FileNotFoundError(f"缺少资源: {_p}")
META = json.loads(SLOTS_JSON.read_text(encoding="utf-8"))
SLOTS = META["slots"]
TOTAL = float(META["total"])

LOG: list[str] = []


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    LOG.append(line)
    print(line, flush=True)


def pump(ms: int = 30) -> None:
    app = QApplication.instance()
    end = time.time() + ms / 1000
    while True:
        app.processEvents()
        if time.time() >= end:
            break
        time.sleep(0.004)


# ── 文件/消息框自动应答（与旧脚本一致，弹窗由真实 Qt 控件流程触发）──
NEXT = {"file": "", "files": [], "dir": ""}
QFileDialog.getOpenFileName = lambda *a, **k: (str(NEXT["file"]), "")
QFileDialog.getOpenFileNames = lambda *a, **k: ([str(x) for x in NEXT["files"]], "")
QFileDialog.getExistingDirectory = lambda *a, **k: str(NEXT["dir"] or DATA_ROOT)


def _answer(kind):
    def fn(parent, title, text, *a, **k):
        log(f"Qt 消息框[{kind}] {title}: {str(text)[:160]}")
        return QMessageBox.StandardButton.Yes if kind == "question" else QMessageBox.StandardButton.Ok
    return fn


for _k in ("information", "warning", "critical", "question"):
    setattr(QMessageBox, _k, _answer(_k))


def ease(p: float) -> float:
    return 4 * p ** 3 if p < 0.5 else 1 - (-2 * p + 2) ** 3 / 2


class Recorder:
    """虚拟时钟录制器：tick() = 抓 1 帧并前进 1/15 s。"""

    def __init__(self, window):
        self.w = window
        self.n = 0
        self.frames: list[list] = []
        self.cur = QPointF(window.width() * 0.5, window.height() * 0.45)
        self.anchor = QPointF(self.cur)
        self.ripples: list[tuple[QPointF, float]] = []
        FRAMES.mkdir(parents=True, exist_ok=True)
        for old in FRAMES.glob("f*.jpg"):
            old.unlink()

    @property
    def t(self) -> float:
        return self.n * DT

    # 坐标：任意控件（含顶层弹窗）局部点 → 主窗口坐标
    def to_win(self, widget, p: QPoint | None = None) -> QPointF:
        if p is None:
            p = widget.rect().center()
        return QPointF(self.w.mapFromGlobal(widget.mapToGlobal(p)))

    def _overlays(self):
        dialogs, popups = [], []
        for tl in QApplication.topLevelWidgets():
            if tl is self.w or not tl.isVisible() or tl.width() < 8:
                continue
            wt = tl.windowType()
            if wt in (Qt.WindowType.Popup, Qt.WindowType.ToolTip):
                popups.append(tl)
            elif wt in (Qt.WindowType.Dialog, Qt.WindowType.Window, Qt.WindowType.Tool):
                dialogs.append(tl)
        return dialogs, popups

    def tick(self) -> None:
        pump(5)
        pix = self.w.grab()
        p = QPainter(pix)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        dialogs, popups = self._overlays()
        for d in dialogs:
            pos = self.to_win(d, QPoint(0, 0))
            dp = d.grab()
            r = QRectF(pos.x(), pos.y() - 30, d.width(), d.height() + 30)
            p.fillRect(r.adjusted(4, 6, 8, 10), QColor(0, 0, 0, 50))
            p.fillRect(QRectF(r.x(), r.y(), r.width(), 30), QColor(243, 243, 243))
            p.setPen(QColor(30, 30, 30))
            f = QFont("Microsoft YaHei", 9)
            p.setFont(f)
            p.drawText(QRectF(r.x() + 12, r.y(), r.width() - 24, 30),
                       Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, d.windowTitle())
            p.drawPixmap(pos.toPoint(), dp)
            p.setPen(QPen(QColor(150, 150, 150), 1))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(r)
        for pp in popups:
            pos = self.to_win(pp, QPoint(0, 0))
            p.fillRect(QRectF(pos.x() + 3, pos.y() + 4, pp.width(), pp.height()), QColor(0, 0, 0, 45))
            p.drawPixmap(pos.toPoint(), pp.grab())
        # 点击波纹
        keep = []
        for c, t0 in self.ripples:
            age = self.t - t0
            if 0 <= age <= 0.32:
                k = age / 0.32
                rad = 7 + 26 * k
                col = QColor(0, 120, 215, int(200 * (1 - k)))
                p.setPen(QPen(col, 3))
                p.setBrush(QBrush(QColor(0, 120, 215, int(70 * (1 - k)))))
                p.drawEllipse(c, rad, rad)
            if age <= 0.32:
                keep.append((c, t0))
        self.ripples = keep
        # 光标（白箭头黑边）
        s = 1.35
        pts = [(0, 0), (0, 17), (4.2, 13.2), (7, 19.5), (9.8, 18.4), (7.1, 12.2), (12.5, 12.2)]
        path = QPainterPath()
        path.moveTo(self.cur + QPointF(pts[0][0] * s, pts[0][1] * s))
        for x, y in pts[1:]:
            path.lineTo(self.cur + QPointF(x * s, y * s))
        path.closeSubpath()
        p.setPen(QPen(QColor(0, 0, 0), 1.6))
        p.setBrush(QBrush(QColor(255, 255, 255)))
        p.drawPath(path)
        p.end()
        name = f"f{self.n:05d}.jpg"
        pix.save(str(FRAMES / name), "JPG", 92)
        self.frames.append([name, round(self.t, 4)])
        self.n += 1
        if self.n % 75 == 0:
            log(f"[进度] 已录 {self.n} 帧 / 虚拟 {self.t:.1f}s / 目标 {TOTAL:.1f}s")

    # ── 动作原语 ──
    def move(self, target: QPointF, dur: float = 0.45) -> None:
        start = QPointF(self.cur)
        k = max(2, round(dur * FPS))
        for i in range(1, k + 1):
            e = ease(i / k)
            self.cur = start + (target - start) * e
            self.tick()
        self.anchor = QPointF(target)

    def hold(self, dur: float) -> None:
        k = round(dur * FPS)
        t0 = self.t
        for i in range(k):
            env = math.sin(math.pi * (i + 1) / (k + 1))
            tt = self.t - t0
            self.cur = self.anchor + QPointF(9 * math.sin(2 * math.pi * tt / 1.4) * env,
                                             6 * math.sin(2 * math.pi * tt / 0.95) * env)
            self.tick()
        self.cur = QPointF(self.anchor)

    def hold_until(self, t_target: float) -> None:
        if self.t < t_target:
            self.hold(t_target - self.t)

    def ripple(self) -> None:
        self.ripples.append((QPointF(self.cur), self.t))

    def click(self, widget, pos: QPoint | None = None, move_dur: float = 0.45, after: float = 0.3,
              action=None) -> None:
        """移到控件 → 波纹 → 真实点击（或自定义 action）→ 波纹继续约 0.3s。"""
        pos = widget.rect().center() if pos is None else pos
        self.move(self.to_win(widget, pos), move_dur)
        self.ripple()
        self.tick()
        if action is not None:
            action()
        else:
            QTest.mouseClick(widget, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, pos)
        pump(20)
        self.hold(max(0.0, after - DT))

    def wait(self, cond, what: str, max_video: float = 0.8, timeout: float = 90.0) -> bool:
        """后端计算：延时摄影（最多 max_video 秒成片），其余时间只泵事件不抓帧。"""
        deadline = time.monotonic() + timeout
        frames_left = round(max_video * FPS)
        last = 0.0
        t0 = self.t
        wall0 = time.monotonic()
        while time.monotonic() < deadline:
            pump(30)
            if cond():
                log(f"就绪: {what}（墙钟 {time.monotonic() - wall0:.1f}s，成片 {self.t - t0:.2f}s）")
                return True
            if frames_left > 0 and time.monotonic() - last >= 0.2:
                tt = self.t - t0
                self.cur = self.anchor + QPointF(9 * math.sin(2 * math.pi * tt / 1.4),
                                                 6 * math.sin(2 * math.pi * tt / 0.95))
                self.tick()
                frames_left -= 1
                last = time.monotonic()
        log(f"等待超时: {what}")
        return False

    def combo(self, combo, index: int, show_hold: float = 0.5) -> None:
        self.click(combo, after=0.05, action=combo.showPopup)
        pump(60)
        self.hold(show_hold)
        view = combo.view()
        view.scrollTo(combo.model().index(index, 0))
        r = view.visualRect(combo.model().index(index, 0))
        self.move(self.to_win(view.viewport(), r.center()), 0.4)
        self.ripple()
        self.tick()
        combo.setCurrentIndex(index)
        combo.hidePopup()
        pump(30)
        self.hold(0.25)

    def slider(self, slider, values: list[int], dur: float = 1.0) -> None:
        def handle_pos():
            opt = QStyleOptionSlider()
            slider.initStyleOption(opt)
            r = slider.style().subControlRect(QStyle.ComplexControl.CC_Slider, opt,
                                              QStyle.SubControl.SC_SliderHandle, slider)
            return self.to_win(slider, r.center())
        self.move(handle_pos(), 0.45)
        self.ripple()
        k = max(len(values), round(dur * FPS))
        for i in range(k):
            v = values[min(len(values) - 1, int(i * len(values) / k))]
            if slider.value() != v:
                slider.setValue(v)
            pump(10)
            self.cur = handle_pos()
            self.tick()
        self.anchor = QPointF(self.cur)

    def finish(self) -> None:
        man = {"fps": FPS, "total": TOTAL, "window_capture": "QWidget.grab", "desktop_capture": False,
               "window_maximized": True, "frames": self.frames}
        (FRAMES / "manifest.json").write_text(json.dumps(man, ensure_ascii=False), encoding="utf-8")
        (WORK / "record_log.txt").write_text("\n".join(LOG), encoding="utf-8")
        log(f"录制完成: {self.n} 帧 = {self.t:.2f}s -> {FRAMES}")


def main() -> int:
    from app.config import ensure_dirs
    ensure_dirs()
    import main as main_mod
    main_mod._start_backend_if_port_free()
    main_mod._start_aoi_core_async()
    from app.ui.main_window import MainWindow
    from app.template.store import TemplateStore
    from app.recipe.param_store import ParamStore
    from app.detect.adapters.smt_adapter import SMT_SHARED_ID
    from app.calibration.smt_pads import set_smt_pads

    app = QApplication(sys.argv)
    app.setApplicationName("PCB 缺陷检测系统 v2")
    app.setEffectEnabled(Qt.UIEffect.UI_AnimateCombo, False)
    window = MainWindow()
    window.showMaximized()

    # ── 独立演示草稿（不删除/不修改已有模板）──
    store = TemplateStore()
    pad_data = json.loads(PAD_JSON.read_text(encoding="utf-8"))
    pads = []
    for sh in pad_data.get("shapes", []):
        if sh.get("label") == "pad" and sh.get("shape_type") == "rectangle":
            (x0, y0), (x1, y1) = sh["points"]
            pads.append([int(min(x0, x1)), int(min(y0, y1)), int(abs(x1 - x0)), int(abs(y1 - y0))])
    name = f"演示_连锡_动作实录_{STAMP}"
    if store.find_by_standard_path(STANDARD) is None:
        tpl = store.create_from_standard_image(STANDARD, display_name=name, category=CATEGORY)
    else:
        tpl = store.create_draft(display_name=name, category=CATEGORY)
        tpl.standard_image = store._make_image_ref(STANDARD)
    tpl.engine_mode = "traditional"
    tpl.model_category = ""
    tpl.category = CATEGORY
    params = json.loads(PARAMS_JSON.read_text(encoding="utf-8"))
    ps = ParamStore()
    ps.save("template_algorithm", SMT_SHARED_ID, params, template_name=tpl.id)
    ps.save("template_algorithm", "solder_连锡", params, template_name=tpl.id)
    set_smt_pads(tpl.id, pads)
    tpl.set_region_shapes("smt_pads", [{"shape": "rect", "x": x, "y": y, "w": w, "h": h} for x, y, w, h in pads])
    store.save(tpl)
    log(f"演示草稿: {tpl.id} pads={len(pads)}")

    tabs = window.tabs
    studio = window.studio_page
    ip = window.inspect_page
    core = window.model_page
    state: dict = {}

    def wait_plain(cond, timeout, what):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            pump(100)
            if cond():
                return True
        log(f"准备阶段超时: {what}")
        return False

    def learning_idle(child):
        st = child.progress._pct.text()
        return not getattr(child, "_async_workers", []) and not st.startswith(("运行中", "排队中"))

    def monitor_frame_idx(mon):
        for did in PAIR_DETECTIONS:
            for i, fr in enumerate(mon._frames):
                if fr.get("detection_id") == did:
                    return i
        return -1

    def setup():
        """录制前准备（不抓帧）：预热 AOI_Core、工单 6、学习页数据；建模页停在其它模板。"""
        tabs.setCurrentWidget(core)
        pump(300)
        if not wait_plain(lambda: getattr(core, "_built", False), 90, "AOI_Core 构建"):
            raise RuntimeError("AOI_Core 未构建（检查 aoi_core_child.log）")
        if not wait_plain(lambda: core.combo_workorder.findData(WORKORDER_ID) >= 0, 40, "工单列表"):
            raise RuntimeError(f"工单 {WORKORDER_ID} 不存在")
        core.combo_workorder.setCurrentIndex(core.combo_workorder.findData(WORKORDER_ID))
        pump(800)
        log(f"工单: {core.combo_workorder.currentText()} 品类={core.current_category()}")
        core.goto("学习效果")
        pump(800)
        child = core.stack.currentWidget()
        wait_plain(lambda: learning_idle(child), 90, "学习页首轮请求")
        for cb in (child.combo_online_cat, child.combo_offline_cat):
            i = cb.findData(CATEGORY)
            if i >= 0:
                cb.blockSignals(True); cb.setCurrentIndex(i); cb.blockSignals(False)
        child.tabs.blockSignals(True); child.tabs.setCurrentIndex(0); child.tabs.blockSignals(False)
        child.reload(); child.reload_online()
        pump(300)
        wait_plain(lambda: learning_idle(child), 90, "学习页线上数据")
        state["learning"] = child
        core.goto("实时监控")
        pump(800)
        mon = core.stack.currentWidget()
        state["monitor"] = mon
        mon.chk_follow.setChecked(False)
        sp = mon.monitor_splitter
        tot = sum(sp.sizes())
        sp.widget(0).setMinimumHeight(1)
        sp.setSizes([int(tot * 0.64), tot - int(tot * 0.64)])
        if not wait_plain(lambda: monitor_frame_idx(mon) >= 0, 40, "产线待检帧"):
            raise RuntimeError(f"监控帧窗口无 {PAIR_DETECTIONS}: {[f.get('detection_id') for f in mon._frames]}")
        mon._load_queue()
        wait_plain(lambda: mon.queue_table.rowCount() > 0, 20, "数据流队列")
        log(f"监控就绪: 帧={len(mon._frames)} 队列行={mon.queue_table.rowCount()}")
        # 建模页先停在其它模板，录制时由列表点击切到本次草稿
        tabs.setCurrentWidget(window.detect_page)
        pump(400)
        studio.list_panel.refresh(keep_id=None, reload_selection=True)
        pump(300)
        others = [t for t in studio.list_panel._templates if t.id != tpl.id]
        if others:
            studio.list_panel.refresh(keep_id=others[0].id, reload_selection=True)
        pump(500)

    def tab_click(rec, widget, after=0.35):
        bar = tabs.tabBar()
        i = tabs.indexOf(widget)
        rec.click(bar, bar.tabRect(i).center(), after=after)

    def nav_click(rec, title, after=0.3):
        nav = core.nav
        row = next(i for i in range(nav.count()) if nav.item(i).text() == title)
        rec.click(nav.viewport(), nav.visualItemRect(nav.item(row)).center(), after=after)

    def seg_begin(rec, i):
        s = SLOTS[i]
        target = max(0.0, s["start"] - 0.3)
        if rec.t > target + 0.05:
            log(f"[警告] 段{i + 1} 开始晚于配音 {rec.t - target:.2f}s")
        rec.hold_until(target)
        log(f"[段{i + 1}] 虚拟 {rec.t:.2f}s 配音 {s['start']:.2f}-{s['end']:.2f} {s['text']}")

    def run():
        try:
            setup()
        except Exception as exc:
            log(f"准备失败: {exc}")
            (WORK / "record_log.txt").write_text("\n".join(LOG), encoding="utf-8")
            QTimer.singleShot(100, app.quit)
            return
        rec = Recorder(window)
        try:
            record(rec)
        except Exception as exc:
            import traceback
            log(f"录制异常: {exc}\n{traceback.format_exc()}")
        rec.hold_until(TOTAL + 0.15)
        rec.finish()
        window.close()
        QTimer.singleShot(300, app.quit)

    def record(rec: Recorder):
        mon = state["monitor"]
        # ── ① 模板建模：标准图 / 双引擎 / 勾选连锡 ──
        seg_begin(rec, 0)
        tab_click(rec, studio, after=0.15)
        lst = studio.list_panel.list
        row = next(i for i, t in enumerate(studio.list_panel._templates) if t.id == tpl.id)
        lst.scrollToItem(lst.item(row))
        pump(50)
        rec.click(lst.viewport(), lst.visualItemRect(lst.item(row)).center(), after=0.05)
        studio._load_standard_image(studio.current)
        studio.btn_fit.click()
        group = next(n for n, items in studio._algo_groups.items() if any(a == "solder_连锡" for a, _ in items))
        studio.combo_model_category.setCurrentText(CATEGORY)
        studio.combo_major.setCurrentIndex(studio.combo_major.findData(group))
        pump(150)
        chk = studio.algo_checks["solder_连锡"]
        if chk.isChecked():
            chk.setChecked(False)
            pump(100)
        log(f"建模页当前模板: {getattr(studio.current, 'id', None)}")
        rec.hold(0.1)
        rec.combo(studio.combo_engine, studio.combo_engine.findData("dual"), show_hold=0.45)
        rec.click(chk, QPoint(9, chk.height() // 2), move_dur=0.4, after=0.15)
        log(f"engine={studio.combo_engine.currentData()} 连锡勾选={chk.isChecked()}")
        rec.click(studio._tool_buttons["solder_连锡"], move_dur=0.4, after=0.1)

        # ── ② 连锡最小面积 → 试跑 → 保存 → 发布 ──
        seg_begin(rec, 1)
        panel = studio._param_panels["solder_连锡"]
        control = panel.sliders["bridge_min_area"]
        studio.param_scroll.ensureWidgetVisible(control)
        pump(80)
        sl = control.slider
        v0 = sl.value()
        step = max(sl.singleStep(), (sl.maximum() - sl.minimum()) // 30)
        up = [min(sl.maximum(), v0 + step * k) for k in (1, 2, 3, 4)]
        rec.slider(sl, up + [up[2], up[1], up[0], v0], dur=1.0)
        log(f"bridge_min_area 拖动 {v0}->{up[-1]}->{sl.value()}")

        def trial_driver(tries=[0]):
            dlg = next((d for d in QApplication.topLevelWidgets()
                        if d is not window and d.isVisible() and d.windowTitle().startswith("试跑结果")), None)
            if dlg is None:
                tries[0] += 1
                if tries[0] < 1200:
                    QTimer.singleShot(50, trial_driver)
                return
            pump(150)
            rec.hold(0.3)
            from PySide6.QtWidgets import QDialogButtonBox
            box = dlg.findChild(QDialogButtonBox)
            btn = box.button(QDialogButtonBox.StandardButton.Close)
            rec.click(btn, move_dur=0.4, after=0.05)
            log("试跑结果对话框已关闭")

        NEXT["file"] = BRIDGE
        QTimer.singleShot(0, trial_driver)
        rec.click(studio.btn_trial, move_dur=0.4, after=0.0)
        rec.click(studio.btn_save, move_dur=0.4, after=0.1)
        if studio.btn_publish.isEnabled():
            rec.click(studio.btn_publish, move_dur=0.4, after=0.1)
        else:
            log(f"[警告] 发布按钮未启用: {studio.lbl_checklist.text()}")
        saved = store.load(tpl.id)
        log(f"发布状态: {getattr(saved, 'status', None)}")

        # ── ③ 自动检测：批量运行良品 + 待检图 ──
        seg_begin(rec, 2)
        tab_click(rec, ip, after=0.1)
        ip.select_template(tpl.id)
        pump(200)
        NEXT["files"] = [GOOD, BRIDGE]
        rec.click(ip.btn_add_files, after=0.2)
        rec.click(ip.btn_run, after=0.1)
        rec.wait(lambda: ip._worker is None, "批量检测", max_video=0.8, timeout=120)
        brow = -1
        for r in range(ip.list_queue.count()):
            qi = int(ip.list_queue.item(r).data(Qt.ItemDataRole.UserRole))
            if Path(ip.queue[qi]).name == BRIDGE.name:
                brow = r
        if brow >= 0:
            item = ip.list_queue.item(brow)
            rec.click(ip.list_queue.viewport(), ip.list_queue.visualItemRect(item).center(), after=0.3)
        log(f"自动检测状态: {ip.lbl_status.text()}")

        # ── ④ 同位配对：原图 / 差异热力图 / 叠加图 ──
        seg_begin(rec, 3)
        canvas = ip.inspect_canvas
        rec.click(canvas.chk_pair, QPoint(9, canvas.chk_pair.height() // 2), after=0.15)
        ok = rec.wait(lambda: canvas.btn_heatmap.isEnabled(), "同位差异热图", max_video=0.6, timeout=90)
        end4 = SLOTS[4]["start"] - 0.3
        if ok:
            rec.click(canvas.btn_original, move_dur=0.4, after=0.2)
            span = max(0.6, (end4 - rec.t - 1.7) / 2)
            rec.hold(max(0.0, span - 0.9))
            rec.click(canvas.btn_heatmap, move_dur=0.4, after=span - 0.4)
            rec.click(canvas.btn_overlay, move_dur=0.4, after=0.3)

        # ── ⑤ 实时监控：产线待检帧 → 双引擎评分/融合权重/阈值 ──
        seg_begin(rec, 4)
        tab_click(rec, core, after=0.1)
        rec.wait(lambda: monitor_frame_idx(mon) >= 0 and len(mon._strip_cells) == len(mon._frames),
                 "监控帧窗口", max_video=0.4, timeout=40)
        nav_click(rec, "实时监控", after=0.2)
        idx = monitor_frame_idx(mon)
        cell = mon._strip_cells[idx] if 0 <= idx < len(mon._strip_cells) else None
        if cell is not None and cell.isVisible():
            rec.click(cell, after=0.3)
        else:
            rec.click(mon, after=0.3, action=lambda: mon._on_strip_click(idx))
        rec.wait(lambda: (mon._current_detection or {}).get("detection_id") in PAIR_DETECTIONS,
                 "选中帧", max_video=0.3, timeout=20)
        log("监控: verdict={} scores={} weights={} threshold={}".format(
            mon.badge_verdict.text(), mon.lbl_scores.text(), mon.lbl_weights.text(), mon.lbl_threshold.text()))
        for lab in (mon.badge_verdict, mon.lbl_scores, mon.lbl_weights, mon.lbl_threshold):
            rec.move(rec.to_win(lab, QPoint(min(lab.width() // 2, 120), lab.height() // 2)), 0.45)
            rec.hold(0.45)

        # ── ⑥ 错检批次：反馈标注 + 回队入口 ──
        seg_begin(rec, 5)
        rec.click(mon.btn_fp, move_dur=0.4, after=0.1)
        dlg = mon._feedback_dialog
        if dlg is not None:
            pump(150)
            ed = dlg.edit_comment
            rec.click(ed, QPoint(60, 16), move_dur=0.4, after=0.0)
            text = "同批连锡误报，回队复检"
            for k in range(0, len(text), 2):
                ed.insertPlainText(text[k:k + 2])
                rec.tick()
            from PySide6.QtWidgets import QDialogButtonBox
            box = dlg.findChild(QDialogButtonBox)
            rec.move(rec.to_win(box.button(QDialogButtonBox.StandardButton.Ok)), 0.4)
            rec.hold(0.1)
            rec.click(box.button(QDialogButtonBox.StandardButton.Cancel), move_dur=0.4, after=0.05)
            log("反馈对话框：已填写，按取消关闭（不写入真实反馈）")
        qt = mon.queue_table
        pick = next((r for r in range(qt.rowCount()) if qt.item(r, 7) and qt.item(r, 7).text() == "待关注"), 0)
        if qt.rowCount() > 0:
            qt.scrollToItem(qt.item(pick, 4))
            rect = qt.visualItemRect(qt.item(pick, 4))
            rec.click(qt.viewport(), rect.center(), move_dur=0.4, after=0.1)
            log(f"队列选中行 {pick}: {[qt.item(pick, c).text() if qt.item(pick, c) else '' for c in range(9)]}")
        rec.move(rec.to_win(mon.btn_queue_req), 0.4)

        # ── ⑦ 学习效果：线上判对率 / 翻案曲线 ──
        seg_begin(rec, 6)
        nav_click(rec, "学习效果", after=0.15)
        child = core.stack.currentWidget()
        rec.wait(lambda: learning_idle(child), "学习页刷新", max_video=0.4, timeout=60)
        for cb in (child.combo_online_cat, child.combo_offline_cat):
            i = cb.findData(CATEGORY)
            if i >= 0 and cb.currentIndex() != i:
                cb.blockSignals(True); cb.setCurrentIndex(i); cb.blockSignals(False)
        pump(100)
        for name in ("online_chart", "flip_chart"):
            ch = getattr(child, name, None)
            if ch is not None and ch.isVisible():
                rec.move(rec.to_win(ch), 0.55)
                rec.hold(0.7)
        log(f"学习页品类: {child.combo_online_cat.currentData()}")

    QTimer.singleShot(5 * 60 * 1000 + 120000, lambda: (log("全局超时，退出"), app.quit()))
    QTimer.singleShot(1500, run)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
