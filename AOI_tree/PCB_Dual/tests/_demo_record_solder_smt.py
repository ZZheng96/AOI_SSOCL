"""PCB_Dual solder_smt 实际 UI 录制驱动。

录制源只来自 PCB_Dual 主窗口或当前可见 Qt owned 弹窗的 QWidget.grab()，不使用桌面/屏幕捕获。
每个分镜单独写入 practice/solder_smt/shots，并生成 manifest；成片由 ffmpeg 拼接。
"""
from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMessageBox,
    QToolButton,
    QTableWidget,
    QTabWidget,
    QTreeWidget,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATA_LOCAL = Path(os.environ.get("DATA_LOCAL", r"D:\CGAIC\data_local"))
CATEGORY = os.environ.get("CATEGORY", "solder_smt")
DATA_ROOT = DATA_LOCAL / CATEGORY
OUT_ROOT = ROOT.parent.parent / "practice" / CATEGORY
STAMP = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
TEMP_ROOT = OUT_ROOT / "Temp" / f"pcb45_real_{STAMP}"
SHOT_DIR = TEMP_ROOT / "shots"
MANIFEST = SHOT_DIR / "manifest.json"
FPS = 8
SLOT = {r["id"]: r["slot"] for r in json.loads(Path(os.environ.get("SLOTS_JSON", r"C:\Users\imia\AppData\Local\Temp\pcb45v3\slots.json")).read_text(encoding="utf-8"))}
MARGIN = 1.0
WIDTH, HEIGHT = 1920, 1080
NO_VIDEO = os.environ.get("NO_VIDEO", "0") == "1"

REQUIRED = ["train/good", "test/good", "test/bridge", "test/cold_solder", "test/excess", "test/insufficient", "ground_truth"]
for rel in REQUIRED:
    if not (DATA_ROOT / rel).is_dir():
        raise FileNotFoundError(f"缺少数据目录: {DATA_ROOT / rel}")

def first(rel: str, pattern: str = "*") -> Path:
    xs = sorted((DATA_ROOT / rel).glob(pattern))
    if not xs:
        raise FileNotFoundError(f"没有样本: {DATA_ROOT / rel / pattern}")
    return xs[0]

PAIR_DIR = ROOT / "alg_repo" / "smt" / "test_images" / "bridge_pair" / "pair4"
STANDARD = PAIR_DIR / "PixPin_2026-07-01_09-22-58_OK.png"
GOOD = STANDARD
BRIDGE = Path(r"D:\CGAIC\data_local\solder_smt\test\bridge\PixPin_2026-07-01_09-22-58.png")
PAD_JSON = PAIR_DIR / "pad.json"
PARAMS_JSON = PAIR_DIR / "params.json"
DEFECTS = [BRIDGE]
WORKORDER_ID = int(os.environ.get("WORKORDER_ID", "6"))
PAIR_DETECTIONS = (1079, 1077)
for _path in (STANDARD, BRIDGE, PAD_JSON, PARAMS_JSON):
    if not _path.is_file():
        raise FileNotFoundError(f"缺少 bridge 实测资源: {_path}")
TEMP_ROOT.mkdir(parents=True, exist_ok=True)

LOG_LINES: list[str] = []
def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    LOG_LINES.append(line)
    print(line, flush=True)

def pump(ms: int = 100) -> None:
    app = QApplication.instance()
    end = time.time() + ms / 1000
    while time.time() < end:
        app.processEvents()
        time.sleep(0.005)

NEXT = {"file": "", "files": [], "dir": ""}
def open_name(parent=None, caption="", dir="", filter="", *a, **k):
    p = NEXT["file"]
    log(f"真实文件控件应答: {caption} -> {p}")
    return str(p), filter

def open_names(parent=None, caption="", dir="", filter="", *a, **k):
    ps = [str(x) for x in NEXT["files"]]
    log(f"真实文件控件应答: {caption} -> {len(ps)} files")
    return ps, filter

def existing_dir(parent=None, caption="", dir="", *a, **k):
    p = NEXT["dir"] or str(DATA_ROOT)
    log(f"真实目录控件应答: {caption} -> {p}")
    return p

QFileDialog.getOpenFileName = open_name
QFileDialog.getOpenFileNames = open_names
QFileDialog.getExistingDirectory = existing_dir

# 弹窗仍由真实 Qt 控件产生；自动按业务流程应答，不伪造检测结果。
def answer(kind):
    def fn(parent, title, text, *a, **k):
        log(f"真实 Qt 弹窗[{kind}] {title}: {str(text)[:180]}")
        if kind == "question":
            return QMessageBox.StandardButton.Yes
        return QMessageBox.StandardButton.Ok
    return fn
QMessageBox.information = answer("information")
QMessageBox.warning = answer("warning")
QMessageBox.critical = answer("critical")
QMessageBox.question = answer("question")

class ShotRecorder:
    def __init__(self, window):
        self.window = window
        self.writer = None
        self.path = None
        self.frames = 0
        self.shot = None
        self.shots = []
        self.timer = QTimer(window)
        self.timer.timeout.connect(self.capture)

    def start(self, name: str, caption: str):
        self.stop()
        SHOT_DIR.mkdir(parents=True, exist_ok=True)
        self.shot = {"name": name, "caption": caption, "started": dt.datetime.now().isoformat(), "path": str(SHOT_DIR / f"{name}.mp4"), "frames": 0}
        self.path = Path(self.shot["path"])
        if NO_VIDEO:
            self.path = self.path.with_suffix(".png")
            self.shot["path"] = str(self.path)
            self.timer.start(100)
            log(f"截图证据开始: {self.path}")
            return
        self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
        if not self.writer.isOpened():
            self.writer.release()
            self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"avc1"), FPS, (WIDTH, HEIGHT))
        if not self.writer.isOpened():
            raise RuntimeError("无法打开分镜视频编码器")
        self.timer.start(round(1000 / FPS))
        log(f"分镜开始: {name} | {caption}")

    def capture(self):
        if NO_VIDEO:
            target = QApplication.activeModalWidget() or QApplication.activeWindow() or self.window
            if target.isVisible():
                target.grab().save(str(self.path))
            return
        if self.writer is None:
            return
        target = QApplication.activeModalWidget() or QApplication.activeWindow() or self.window
        if not target.isVisible():
            target = self.window
        pix = target.grab()
        if pix.isNull():
            return
        img = pix.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
        w, h = img.width(), img.height()
        arr = np.frombuffer(img.bits(), dtype=np.uint8, count=w * h * 4).reshape((h, w, 4))
        frame = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        frame = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        self.writer.write(frame)
        self.frames += 1
        self.shot["frames"] = self.frames

    def pause(self):
        self.timer.stop()

    def resume(self):
        if self.shot is not None:
            self.timer.start(100 if NO_VIDEO else round(1000 / FPS))

    def stop(self):
        if self.writer is not None or (NO_VIDEO and self.shot is not None):
            self.timer.stop()
            if self.writer is not None:
                self.writer.release()
            self.shot["ended"] = dt.datetime.now().isoformat()
            self.shots.append(self.shot)
            log(f"分镜完成: {self.shot['name']} frames={self.frames}")
            self.writer = None
            self.shot = None
            self.frames = 0

    def finish(self):
        self.stop()
        (SHOT_DIR / "record.log").write_text("\n".join(LOG_LINES), encoding="utf-8")
        self.shots_file = SHOT_DIR / "shots.txt"
        self.shots_file.write_text("\n".join(f"file '{Path(s['path']).as_posix()}'" for s in self.shots), encoding="utf-8")
        # 成片由 pcb45v3_build.py compose 按配音槽位裁剪合成，这里只输出分镜与 manifest。
        manifest = {"category": CATEGORY, "data_root": str(DATA_ROOT), "window_capture": "QWidget.grab", "desktop_capture": False, "window_maximized": True, "standard": str(STANDARD), "defects": [str(x) for x in DEFECTS], "shots": self.shots, "no_video": NO_VIDEO}
        MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"分镜目录: {SHOT_DIR}")
        return SHOT_DIR

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
    window = MainWindow()
    window.showMaximized()
    recorder = ShotRecorder(window)
    log(f"数据根目录: {DATA_ROOT}")
    log(f"真实样本: standard={STANDARD.name}, good={GOOD.name}, defects={[x.name for x in DEFECTS]}")

    store = TemplateStore()
    log("验收模式：不执行清理旧模板逻辑，不删除已有数据")
    pad_data = json.loads(PAD_JSON.read_text(encoding="utf-8"))
    pad_frames = []
    for shape in pad_data.get("shapes", []):
        if shape.get("label") != "pad" or shape.get("shape_type") != "rectangle":
            continue
        (x0, y0), (x1, y1) = shape["points"]
        pad_frames.append([int(min(x0, x1)), int(min(y0, y1)), int(abs(x1 - x0)), int(abs(y1 - y0))])
    if not pad_frames:
        raise RuntimeError("真实 pad.json 中没有焊盘 ROI")
    demo_name = f"演示_连锡_bridge_实录_{STAMP}"
    existing = store.find_by_standard_path(STANDARD)
    if existing is None:
        tpl = store.create_from_standard_image(STANDARD, display_name=demo_name, category=CATEGORY)
    else:
        # 一图一模板的 UI 会选回旧模板；独立草稿引用真实标准图，不修改旧模板或图像字节。
        tpl = store.create_draft(display_name=demo_name, category=CATEGORY)
        tpl.standard_image = store._make_image_ref(STANDARD)
    tpl.engine_mode = "traditional"
    tpl.model_category = ""
    tpl.category = CATEGORY
    measured_params = json.loads(PARAMS_JSON.read_text(encoding="utf-8"))
    param_store = ParamStore()
    param_store.save("template_algorithm", SMT_SHARED_ID, measured_params, template_name=tpl.id)
    param_store.save("template_algorithm", "solder_连锡", measured_params, template_name=tpl.id)
    set_smt_pads(tpl.id, pad_frames)
    tpl.set_region_shapes("smt_pads", [{"shape": "rect", "x": x, "y": y, "w": w, "h": h} for x, y, w, h in pad_frames])
    store.save(tpl)
    log(f"录制前仅准备独立草稿与真实资源，不预先发布: {tpl.display_name} id={tpl.id} pads={len(pad_frames)}")

    def page(widget, name):
        window.tabs.setCurrentWidget(widget)
        log(f"页面: {name}")
        pump(250)

    def verify_studio_image():
        from app.utils.cv_io import imread_unicode
        studio = window.studio_page
        expected = imread_unicode(str(STANDARD))
        if expected is None:
            raise RuntimeError("无法读取 bridge 真实标准图")
        if studio.current is None or studio.current.id != tpl.id:
            raise RuntimeError("模板建模页未选中本次 bridge 独立模板")
        if studio.std_bgr is None or not np.array_equal(studio.std_bgr, expected):
            raise RuntimeError("模板建模 std_bgr 与 STANDARD 像素/尺寸不一致")
        pix = studio.canvas._pix
        if pix is None or pix.isNull():
            raise RuntimeError("模板建模 canvas 无实际图像")
        image = pix.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
        w, h = image.width(), image.height()
        rgba = np.frombuffer(image.bits(), dtype=np.uint8, count=image.sizeInBytes())
        rgba = rgba.reshape(h, image.bytesPerLine())[:, :w * 4].reshape(h, w, 4)
        actual = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        if (studio.canvas._img_wh != (expected.shape[1], expected.shape[0])
                or actual.shape != expected.shape or not np.array_equal(actual, expected)):
            raise RuntimeError("模板建模 canvas 实际 QPixmap 与 STANDARD 像素/尺寸不一致")
        log(f"canvas 真实像素验证通过: template={tpl.id} size={w}x{h}")

    def select_studio_template():
        """通过 UI 列表重绑本次草稿，并从真实文件重新加载画布。"""
        studio = window.studio_page
        panel = studio.list_panel
        panel.refresh(keep_id=tpl.id, reload_selection=True)
        pump(300)
        selected = panel.current_template()
        if selected is None or selected.id != tpl.id or studio.current is None or studio.current.id != tpl.id:
            raise RuntimeError("模板建模页列表/当前模板未绑定本次 bridge 独立草稿")
        # 列表高亮和文件路径不等于画布内容；显式走真实加载路径重建 QPixmap。
        studio._load_standard_image(studio.current)
        studio.btn_fit.click()
        studio.canvas.repaint()
        pump(300)
        verify_studio_image()

    def wait_learning_requests(child, timeout=90):
        deadline = time.monotonic() + timeout
        next_log = 0.0
        while time.monotonic() < deadline:
            pump(100)
            workers = list(getattr(child, "_async_workers", []))
            status = child.progress._pct.text()
            task_pending = status.startswith(("运行中", "排队中"))
            if not workers and not task_pending:
                pump(200)
                if not getattr(child, "_async_workers", []):
                    return
            if time.monotonic() >= next_log:
                log(f"等待学习页真实请求: workers={len(workers)} task={status or '无'}")
                next_log = time.monotonic() + 5
        raise RuntimeError("学习页真实请求/重打分等待超时，保留图表，不判定无数据")

    def prepare_preprocess():
        """预处理页加载真实标准图，四项保持 NONE 并显示原图/预览。"""
        from app.utils.cv_io import imread_unicode
        image = imread_unicode(str(BRIDGE))
        if image is None:
            raise RuntimeError("预处理页无法读取真实连锡图")
        preprocess = window.preprocess_page
        preprocess.set_image_from_detect(image, str(BRIDGE))
        preprocess._reset_controls()
        log(
            "预处理真实图像: "
            f"color={preprocess.combo_color.currentText()} "
            f"enhance={preprocess.combo_enhance.currentText()} "
            f"filter={preprocess.combo_filter.currentText()} "
            f"morph={preprocess.combo_morph.currentText()}"
        )

    def hold_total(seconds):
        """补录到本分镜累计 seconds 秒（按实际写入帧数计，避免卡顿导致素材不足）。"""
        if NO_VIDEO:
            pump(int(seconds * 300))
            return
        target = int(seconds * FPS) + 1
        while recorder.frames < target:
            pump(50)

    def shot(sid, caption):
        recorder.start(sid, caption)
        log(f"[进度] {sid} 槽位 {SLOT[sid]:.2f}s 需录 {SLOT[sid] + MARGIN:.2f}s")

    def wait_until(cond, timeout, what):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            pump(100)
            if cond():
                return True
        log(f"等待超时: {what}")
        return False

    def boot():
        studio = window.studio_page
        # ── 01 预处理 ──
        page(window.preprocess_page, "预处理")
        try:
            prepare_preprocess()
        except Exception as exc:
            log(f"预处理展示阻断: {exc}")
        pump(600)
        shot("01_preprocess", "预处理页载入分割后的待检图")
        # 镜头内逐项切换真实预处理选项（只改控件触发预览，不保存配方）
        pre = window.preprocess_page
        total = SLOT["01_preprocess"] + MARGIN
        steps = [(0.18, pre.combo_color, "GRAYSCALE"),
                 (0.40, pre.combo_enhance, "LINEAR_AUTO"),
                 (0.62, pre.combo_filter, "MEDIAN")]
        for frac, combo, value in steps:
            hold_total(total * frac)
            combo.setCurrentText(value)
            pump(200)
            log(f"预处理切换: {combo.objectName() or 'combo'}={combo.currentText()}")
        hold_total(total)
        recorder.stop()

        # ── 02 建模：选择 ──
        page(window.studio_page, "模板建模")
        try:
            select_studio_template()
            if studio.current.id != tpl.id or studio.std_bgr is None:
                raise RuntimeError("模板或标准图未真实加载")
            if store._file_sha1(Path(studio.current.standard_image.path)) != store._file_sha1(STANDARD):
                raise RuntimeError("建模标准图不是 bridge_pair/pair4 真实标准图")
            log(f"UI 已选择模板与标准图: {studio.current.id} / {studio.lbl_std_path.toolTip()}")
        except Exception as exc:
            log(f"模板建模选择阻断: {exc}")
        shot("02_tpl_select", "连锡标准图、双引擎、品类、勾选连锡")
        try:
            # 槽位约 5.3s：标准图 1.2s → 双引擎 → 品类 → 检测大类 → 勾选连锡
            pump(1200)
            studio.combo_engine.setCurrentIndex(studio.combo_engine.findData("dual"))
            pump(900)
            studio.combo_model_category.setCurrentText(CATEGORY)
            pump(900)
            group = next(name for name, items in studio._algo_groups.items()
                         if any(alg_id == "solder_连锡" for alg_id, _ in items))
            studio.combo_major.setCurrentIndex(studio.combo_major.findData(group))
            pump(800)
            studio.algo_checks["solder_连锡"].setChecked(True)
            pump(300)
            if "solder_连锡" not in studio._selected_ids():
                raise RuntimeError("未勾选 solder_连锡")
            log(f"建模选择: engine={studio.combo_engine.currentData()} model_cat={studio.combo_model_category.currentText()}")
        except Exception as exc:
            log(f"模板建模选择阻断: {exc}")
        hold_total(SLOT["02_tpl_select"] + MARGIN)
        recorder.stop()

        # ── 03 建模：参数 + ROI ──
        shot("03_tpl_param", "连锡最小面积滑块与焊盘 ROI")
        try:
            studio._tool_buttons["solder_连锡"].click()
            pump(500)
            panel = studio._param_panels["solder_连锡"]
            control = panel.sliders["bridge_min_area"]
            studio.param_scroll.ensureWidgetVisible(control)
            slider = control.slider
            before = slider.value()
            after = before + slider.singleStep()
            if after > slider.maximum():
                after = before - slider.singleStep()
            slider.setValue(after)
            pump(600)
            if slider.value() == before:
                raise RuntimeError("连锡参数滑块未发生变化")
            log(f"真实滑块 bridge_min_area: {before} -> {slider.value()} values={panel.current_values()}")
            # 展示调节后恢复 UI 加载值；控件精度可能对样本实测阈值取整，以日志为准。
            slider.setValue(before)
            pump(400)
            roi_idx = studio.combo_calib_target.findData("smt_pads")
            if roi_idx < 0:
                raise RuntimeError("未找到 smt_pads ROI 目标")
            studio.combo_calib_target.setCurrentIndex(roi_idx)
            studio._reload_calib_layer()
            studio.btn_fit.click()
            pump(700)
            shapes = studio.canvas.get_layer_shapes("smt_pads")
            if not shapes or len(shapes) != len(pad_frames):
                raise RuntimeError("真实焊盘 ROI 未显示到画布")
            verify_studio_image()
            log(f"UI smt_pads 真实 ROI 已显示: {len(shapes)}")
        except Exception as exc:
            log(f"模板建模参数阻断: {exc}")
        hold_total(SLOT["03_tpl_param"] + MARGIN)
        recorder.stop()

        # ── 04 建模：试跑、保存、发布（槽位 2.65s，保留开头）──
        shot("04_tpl_publish", "试跑待检图后保存并发布")
        try:
            def close_trial_dialog(attempt=0):
                for dialog in QApplication.topLevelWidgets():
                    if dialog is not window and dialog.isVisible() and dialog.windowTitle().startswith("试跑结果"):
                        dialog.showMaximized()
                        pump(900)
                        dialog.close()
                        return
                if attempt < 100:
                    QTimer.singleShot(100, lambda: close_trial_dialog(attempt + 1))

            NEXT["file"] = BRIDGE
            QTimer.singleShot(200, close_trial_dialog)
            studio.btn_trial.click()
            pump(300)
            studio.btn_save.click()
            pump(500)
            if not studio.btn_publish.isEnabled():
                raise RuntimeError(f"发布检查未通过: {studio.lbl_checklist.text()}")
            studio.btn_publish.click()
            pump(700)
            saved = store.load(tpl.id)
            if saved is None or saved.status != "published":
                raise RuntimeError("真实 UI 发布未成功")
            log(f"UI 保存并发布成功: {saved.id}, engine={saved.engine_mode}, model={saved.model_category}")
        except Exception as exc:
            log(f"模板建模展示阻断（保留真实状态）: {exc}")
        hold_total(SLOT["04_tpl_publish"] + MARGIN)
        recorder.stop()

        # ── 05 算法调试（保留结尾：检测结果）──
        page(window.detect_page, "算法调试")
        shot("05_debug", "算法调试绑定模板，运行待检图")
        try:
            detect = window.detect_page
            detect.template_picker.refresh(keep_id=tpl.id)
            combo = detect.template_picker.combo
            idx = combo.findData(tpl.id)
            if combo.count() <= 1 or idx < 0:
                raise RuntimeError("template_picker 为空或未找到演示模板")
            combo.setCurrentIndex(idx)
            pump(500)
            bound = detect.bound_template
            if bound is None or bound.id != tpl.id:
                raise RuntimeError("bound_template 未绑定目标模板")
            if not Path(bound.standard_image.path).is_file() or detect.std_bgr is None:
                raise RuntimeError("算法调试标准图不存在或未加载")
            if store._file_sha1(Path(bound.standard_image.path)) != store._file_sha1(STANDARD):
                raise RuntimeError("算法调试绑定的不是连锡标准图")
            log(f"调试验证: picker={combo.count()} bound_template={bound.id} standard={detect.std_path}")
            detect.load_test(str(BRIDGE))
            pump(700)
            if detect.test_bgr is None:
                raise RuntimeError("真实连锡缺陷图未加载")
            detect.run_detect()
            pump(2800)
            summary = detect.last_summary
            log(f"真实连锡检测结果（不预设 OK/NG）: overall={getattr(summary, 'overall', '')} ng_count={getattr(summary, 'ng_count', 0)}")
        except Exception as exc:
            log(f"算法调试阻断: {exc}")
        hold_total(SLOT["05_debug"] + MARGIN)
        recorder.stop()

        # ── 06 自动检测：批量运行良品+连锡图，录到判定结果出现 ──
        page(window.inspect_page, "自动检测")
        shot("06_inspect", "自动检测良品+待检图批量判定")
        ip = window.inspect_page
        try:
            ip.select_template(tpl.id)
            pump(300)
            if ip.current_template is None or ip.current_template.id != tpl.id:
                raise RuntimeError("自动检测模板未完成刷新")
            NEXT["files"] = [GOOD, BRIDGE]
            ip._append_paths(NEXT["files"])
            log(f"检测队列: {[x.name for x in NEXT['files']]}")
            pump(500)
            if ip.btn_run.isEnabled():
                ip.btn_run.click()
                log("已点击开始检测，等待真实结果")
                pump(9000)
            else:
                log("阻断: 检测按钮未就绪，保持真实状态")
        except Exception as exc:
            log(f"自动检测阻断: {exc}")
        hold_total(SLOT["06_inspect"] + MARGIN)
        recorder.stop()

        # ── 07 同位配对：连锡图 vs 标准图，原图 → 差异热力图 → 叠加图（PCB 特征引擎真实计算）──
        canvas = ip.inspect_canvas
        pair_ok = False
        try:
            row = -1
            for r in range(ip.list_queue.count()):
                qi = int(ip.list_queue.item(r).data(Qt.ItemDataRole.UserRole))
                if Path(ip.queue[qi]).name == BRIDGE.name:
                    row = r
            if row < 0:
                raise RuntimeError("队列中未找到连锡图")
            ip.list_queue.setCurrentRow(row)
            pump(800)
            log(f"当前图: {ip._current_path().name}")
        except Exception as exc:
            log(f"同位配对准备阻断: {exc}")
        slot7 = SLOT["07_pair_layers"]
        shot("07_pair_layers", "同位配对：原图、差异热力图、叠加图")
        try:
            canvas.chk_pair.setChecked(True)
            log("已勾选同位配对，等待差异热图")
            pair_ok = wait_until(lambda: canvas.btn_heatmap.isEnabled(), 60, "同位差异热图")
            log(f"同位热图: ok={pair_ok} status={ip.lbl_status.text()}")
        except Exception as exc:
            log(f"同位配对阻断: {exc}")
        base = 0.0
        if pair_ok:
            canvas.btn_original.click()
            # 等待段放在镜头开头，合成时裁掉（trim=tail），三层切换各占约 1/3 槽位
            base = 0.0 if NO_VIDEO else recorder.frames / FPS
            log(f"配对等待耗时 {base:.2f}s")
            hold_total(base + slot7 * 0.33)
            for frac, btn, name in ((0.66, canvas.btn_heatmap, "heatmap"), (1.0, canvas.btn_overlay, "overlay")):
                btn.click()
                log(f"配对图层: {name}")
                hold_total(base + slot7 * frac)
        hold_total(base + slot7 + 0.2)
        recorder.stop()

        # ── 08 实时监控：工单产线待检帧 → 人工确认同位 → 叠加图 + 双引擎评分/权重/阈值 ──
        core = window.model_page
        page(window.model_page, "AOI_Core")
        monitor = None
        try:
            core.ensure_built()
            wait_until(lambda: getattr(core, "_built", False), 30, "AOI_Core 构建")
            if not wait_until(lambda: core.combo_workorder.findData(WORKORDER_ID) >= 0, 30, "工单列表加载"):
                raise RuntimeError(f"工单列表未找到真实工单 {WORKORDER_ID}")
            core.combo_workorder.setCurrentIndex(core.combo_workorder.findData(WORKORDER_ID))
            pump(800)
            log(f"工单: {core.combo_workorder.currentText()} id={core.current_workorder_id()} 品类={core.current_category()}")
            core.goto("实时监控")
            pump(500)
            monitor = core.stack.currentWidget()
            monitor.chk_follow.setChecked(False)
            monitor.monitor_splitter.setSizes([680, 260])

            def frame_idx():
                for did in PAIR_DETECTIONS:
                    for i, fr in enumerate(monitor._frames):
                        if fr.get("detection_id") == did:
                            return i
                return -1
            if not wait_until(lambda: frame_idx() >= 0, 30, "产线待检帧"):
                raise RuntimeError(f"实时流窗口内无待检帧 {PAIR_DETECTIONS}，"
                                   f"窗口帧={[f.get('detection_id') for f in monitor._frames]}")
            monitor._on_strip_click(frame_idx())
            pump(800)
            det = monitor._current_detection
            log(f"选中产线帧 #{det.get('detection_id')} {det.get('image_path')} decision={det.get('decision')}")
            # 人工同位确认：文件对话框返回真实标准图，确认弹窗应答 Yes（均为全局 Qt 应答）
            NEXT["file"] = STANDARD
            monitor._confirm_pair()
            if not wait_until(lambda: bool((monitor._current_detection or {}).get("pair_overlay_path")),
                              60, "监控同位叠加图"):
                raise RuntimeError(f"同位叠加图未返回: {monitor.lbl_pair.text()}")
            pump(1500)   # 等 viewer 下载叠加图
            monitor._layer_btns["overlay"].click()
            pump(800)
            log(f"监控叠加图: {monitor._current_detection.get('pair_overlay_path')} | {monitor.lbl_pair.text()}")
        except Exception as exc:
            log(f"实时监控准备阻断: {exc}")
        shot("08_monitor_scores", "实时监控：叠加图、双引擎评分、融合权重、阈值与判定")
        if monitor is not None:
            log("监控评分: "
                f"verdict={monitor.badge_verdict.text()} slots={list(monitor._slot_bars.keys())} "
                f"scores={monitor.lbl_scores.text()} weights={monitor.lbl_weights.text()} "
                f"threshold={monitor.lbl_threshold.text()} trigger={monitor.lbl_trigger.text()}")
        hold_total(SLOT["08_monitor_scores"] + MARGIN)
        recorder.stop()

        # ── 09 监控页下方数据流队列：批次异常/反馈统计与回队入口（只选中，不提交回队）──
        try:
            if monitor is None:
                raise RuntimeError("监控页未就绪")
            sp = monitor.monitor_splitter
            total = sum(sp.sizes())
            sp.widget(0).setMinimumHeight(1)   # 仅放开分割条下限，不隐藏任何控件
            sp.setSizes([total - int(total * 0.55), int(total * 0.55)])
            pump(300)
            log(f"监控分割: sizes={sp.sizes()} top_min={sp.widget(0).minimumSizeHint().height()}")
            monitor.queue_table.clearContents()
            monitor.queue_table.setRowCount(0)
            monitor._load_queue()
            if not wait_until(lambda: monitor.queue_table.rowCount() > 0, 20, "数据流队列加载"):
                raise RuntimeError("数据流队列未返回")
            pump(500)
            rows = []
            for r in range(monitor.queue_table.rowCount()):
                rows.append(" | ".join(monitor.queue_table.item(r, c).text() if monitor.queue_table.item(r, c) else ""
                                       for c in range(monitor.queue_table.columnCount())))
            log(f"数据流队列: {rows}")
            pick = next((r for r in range(monitor.queue_table.rowCount())
                         if monitor.queue_table.item(r, 7) and monitor.queue_table.item(r, 7).text() == "待关注"), 0)
            if monitor.queue_table.item(pick, 0) is not None and monitor.queue_table.item(pick, 0).data(Qt.ItemDataRole.UserRole):
                monitor.queue_table.selectRow(pick)
                log(f"选中批次行 {pick}（不点击回队，不改变生产队列）")
            else:
                log("队列无可选批次，如实展示")
        except Exception as exc:
            log(f"数据流队列阻断: {exc}")
        shot("09_feedback", "数据流队列：批次统计与回队（重新检测）入口")
        try:
            if monitor is not None:
                pump(int(SLOT["09_feedback"] * 550))
                monitor.btn_queue_req.setFocus()   # 聚焦回队按钮，提示入口；不触发 clicked
        except Exception as exc:
            log(f"回队按钮聚焦失败: {exc}")
        hold_total(SLOT["09_feedback"] + MARGIN)
        recorder.stop()

        # ── 10 学习效果：既有线上证据 ──
        try:
            core.goto("学习效果")
            pump(1600)
            child = core.stack.currentWidget()
            # Hub.goto 只调用 reload（离线数据），不会刷新线上四图；先排空构建时的品类请求。
            wait_learning_requests(child)
            for combo in (child.combo_online_cat, child.combo_offline_cat):
                idx = combo.findData(CATEGORY)
                if idx < 0:
                    raise RuntimeError(f"学习页品类列表未找到 {CATEGORY}")
                combo.blockSignals(True)
                combo.setCurrentIndex(idx)
                combo.blockSignals(False)
            child.tabs.blockSignals(True)
            child.tabs.setCurrentIndex(0)
            child.tabs.blockSignals(False)
            child.reload()
            child.reload_online()
            wait_learning_requests(child)
            if core.current_category() != CATEGORY or child.combo_online_cat.currentData() != CATEGORY:
                raise RuntimeError("学习页品类发生变化，不能使用其它品类的学习证据")
            for name in ("flip_chart", "trend_chart", "wh_chart", "online_chart"):
                chart = getattr(child, name)
                points = 0
                for item in chart.plot.listDataItems():
                    x, y = item.getData()
                    if x is not None and y is not None:
                        points += int(np.count_nonzero(np.isfinite(np.asarray(x, dtype=float)) & np.isfinite(np.asarray(y, dtype=float))))
                no_weights = (name == "wh_chart" and points == 0
                              and child.lbl_wh_hint.text().startswith("暂无权重演化记录"))
                chart.parentWidget().setVisible(not no_weights)
                log(f"学习证据[{name}]: 有效真实点数={points}"
                    + ("；后端确认无权重记录，仅隐藏权重图" if no_weights else "；保留真实图表/状态"))
            pump(300)
        except Exception as exc:
            log(f"学习效果阻断: {exc}")
        shot("10_learning", "学习效果既有线上证据（不运行学习/训练）")
        hold_total(SLOT["10_learning"] + MARGIN)
        recorder.finish()
        QTimer.singleShot(300, app.quit)

    QTimer.singleShot(1200, boot)
    return app.exec()

if __name__ == "__main__":
    raise SystemExit(main())
