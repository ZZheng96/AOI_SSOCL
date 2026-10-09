"""PCB_Dual 演示视频录制驱动（分镜见 docs/演示视频脚本.md）。

复用 _ui_full_datalocal 的真实 UI 步骤，额外：弹窗可见化、按钮点击涟漪、字幕、
长等待延时摄影、补充分镜（双检阻断/删除/一图一模板/试跑/调参/升版本/双检检测/Core 子页）。

用法（PCB_Dual 目录下）：
    $env:DEMO_RECORD="1"; e:\\CPIPC\\CGAIC\\.env\\Scripts\\python.exe tests\\_demo_record.py
DEMO_RECORD=0 只操作不录制；DEMO_SLOW 放慢倍数（默认 1.8）。
"""
from __future__ import annotations

import datetime
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

os.environ.setdefault("UI_TEST_SEED", "20261007")
os.environ.setdefault("UI_TEST_PAIRS", "2")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import _ui_full_datalocal as T  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtCore import Qt, QTimer  # noqa: E402
from PySide6.QtGui import QCursor  # noqa: E402
from PySide6.QtWidgets import QAbstractButton, QApplication, QMessageBox  # noqa: E402

RECORD = os.environ.get("DEMO_RECORD", "1") == "1"
SLOW = float(os.environ.get("DEMO_SLOW") or 1.8)
DLG_MS = 1800
FPS = 8
FONT = "C:/Windows/Fonts/msyh.ttc"
OUT_DIR = ROOT / "docs" / "演示视频"

CAP = {"title": "", "sub": "", "speed": 1}
CLICKS: list[tuple[float, float, float]] = []
_raw_pump = T.pump
REC = None  # Recorder

# ---------------- 字幕 ----------------
def title(text: str) -> None:
    CAP["title"], CAP["sub"] = text, ""
    print(f"\n[字幕] {text}", flush=True)


def sub(text: str) -> None:
    CAP["sub"] = text


def hold(ms: int) -> None:
    """按原始时长停留（不乘 SLOW）。"""
    _raw_pump(ms)


def _slow_pump(ms: int = 50) -> None:
    _raw_pump(int(ms * SLOW))


T.pump = _slow_pump

_orig_ck, _orig_warn = T.ck, T.warn


def _ck(name, ok, detail=""):
    _orig_ck(name, ok, detail)
    sub(("[通过] " if ok else "[失败] ") + name)


def _warn(name, detail=""):
    _orig_warn(name, detail)
    sub("[提示] " + name)


T.ck, T.warn = _ck, _warn

# ---------------- 弹窗可见化 ----------------
_ICONS = {"information": QMessageBox.Icon.Information, "warning": QMessageBox.Icon.Warning,
          "critical": QMessageBox.Icon.Critical, "question": QMessageBox.Icon.Question}


def _vis_dlg(kind):
    def _show(parent, ttl, text, *args, **kwargs):
        T.DIALOGS.append(f"[{kind}] {ttl} | {str(text)[:160]}")
        print(f"    [弹窗{kind}] {ttl} | {str(text)[:120]}", flush=True)
        box = QMessageBox(_ICONS[kind], ttl, str(text), parent=parent)
        if kind == "question":
            box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        box.setWindowModality(Qt.WindowModality.NonModal)
        box.show()
        _raw_pump(150)
        b = box.button(QMessageBox.StandardButton.Yes if kind == "question" else QMessageBox.StandardButton.Ok)
        _raw_pump(DLG_MS)
        if b is not None:
            _mark(b)
            _raw_pump(350)
        box.close()
        box.deleteLater()
        _raw_pump(100)
        return QMessageBox.StandardButton.Yes if kind == "question" else QMessageBox.StandardButton.Ok
    return _show


for _k in _ICONS:
    setattr(QMessageBox, _k, _vis_dlg(_k))

# ---------------- 点击示意 ----------------
_orig_click = QAbstractButton.click


def _mark(w) -> None:
    try:
        if not w.isVisible():
            return
        c = w.mapToGlobal(w.rect().center())
        QCursor.setPos(c)
        dpr = w.devicePixelRatioF() or 1.0
        CLICKS.append((c.x() * dpr, c.y() * dpr, time.time()))
    except Exception:
        pass


def _click(self):
    _mark(self)
    _raw_pump(250)
    return _orig_click(self)


QAbstractButton.click = _click

# ---------------- 延时摄影 ----------------
_orig_poll = T.poll_until


def _poll(desc, cond, done, timeout_s=60.0, interval_ms=300):
    if timeout_s >= 300:
        CAP["speed"] = 6
        sub(f"等待：{desc}  [加速 x6]")

        def _done():
            CAP["speed"] = 1
            done()
        return _orig_poll(desc, cond, _done, timeout_s, interval_ms)
    sub(f"等待：{desc}")
    return _orig_poll(desc, cond, done, timeout_s, interval_ms)


T.poll_until = _poll

# ---------------- 录屏 ----------------
class Recorder(threading.Thread):
    def __init__(self, out: Path) -> None:
        super().__init__(daemon=True)
        from PIL import ImageFont
        self.out = out
        self.tmp = Path(tempfile.gettempdir()) / f"demo_{int(time.time())}.mp4"
        self.f_big = ImageFont.truetype(FONT, 34)
        self.f_mid = ImageFont.truetype(FONT, 26)
        self.f_card = ImageFont.truetype(FONT, 56)
        self.stop_ev = threading.Event()
        self.writer = None
        self.size = None
        self.scale = 1.0
        self.frames = 0

    def _open(self, w: int, h: int) -> None:
        for cc in ("avc1", "mp4v"):
            wr = cv2.VideoWriter(str(self.tmp), cv2.VideoWriter_fourcc(*cc), FPS, (w, h))
            if wr.isOpened():
                self.writer = wr
                print(f"[录屏] 编码 {cc} {w}x{h}@{FPS}", flush=True)
                return
        raise RuntimeError("VideoWriter 打不开")

    def card(self, lines: list[str], secs: float = 3.0) -> None:
        from PIL import Image, ImageDraw
        w, h = self.size
        img = Image.new("RGB", (w, h), (18, 24, 38))
        d = ImageDraw.Draw(img)
        y = h // 2 - len(lines) * 45
        for i, t in enumerate(lines):
            f = self.f_card if i == 0 else self.f_mid
            tw = d.textlength(t, font=f)
            d.text(((w - tw) / 2, y), t, font=f, fill=(255, 255, 255) if i == 0 else (180, 200, 230))
            y += 90 if i == 0 else 48
        fr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        for _ in range(int(secs * FPS)):
            self.writer.write(fr)
            self.frames += 1

    def _overlay(self, img):
        from PIL import ImageDraw
        d = ImageDraw.Draw(img, "RGBA")
        now = time.time()
        for x, y, t in list(CLICKS[-6:]):
            age = now - t
            if 0 <= age < 0.9:
                r = 14 + age * 60
                a = int(230 * (1 - age / 0.9))
                cx, cy = x * self.scale, y * self.scale
                d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(255, 60, 60, a), width=5)
                d.ellipse((cx - 7, cy - 7, cx + 7, cy + 7), fill=(255, 60, 60, a))
        w, h = img.size
        ttl, sb = CAP["title"], CAP["sub"]
        if CAP["speed"] > 1:
            d.rectangle((w - 190, 16, w - 16, 64), fill=(200, 120, 0, 220))
            d.text((w - 178, 20), f"加速 x{CAP['speed']}", font=self.f_mid, fill=(255, 255, 255))
        if ttl or sb:
            d.rectangle((0, h - 112, w, h), fill=(10, 14, 24, 200))
            d.text((28, h - 104), ttl, font=self.f_big, fill=(255, 255, 255))
            col = (120, 230, 140) if sb.startswith("[通过]") else (
                (255, 110, 110) if sb.startswith("[失败]") else (255, 214, 102))
            d.text((28, h - 52), sb[:90], font=self.f_mid, fill=col)
        return img

    def grab(self):
        from PIL import ImageGrab
        img = ImageGrab.grab()
        if self.size is None:
            sw, sh = img.size
            self.scale = 1920 / sw
            self.size = (1920, int(sh * self.scale) // 2 * 2)
            self._open(*self.size)
        if img.size != self.size:
            img = img.resize(self.size)
        return img

    def run(self) -> None:
        self.grab()
        self.card(["PCB_Dual 智能检测系统 操作演示",
                   "模板建模 → 自动检测 → 历史 → 调试 → 特征学习（AOI_Core）→ 双检",
                   f"数据：data_local/extra_part（多件检测）  {datetime.date.today()}"])
        k = 0
        while not self.stop_ev.is_set():
            t0 = time.time()
            try:
                img = self._overlay(self.grab())
            except Exception as exc:
                print(f"[录屏] 抓帧异常 {exc}", flush=True)
                time.sleep(0.2)
                continue
            k += 1
            if CAP["speed"] <= 1 or k % CAP["speed"] == 0:
                self.writer.write(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR))
                self.frames += 1
            dt = 1.0 / FPS - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)
        self.card(["演示结束", "演示数据已自动清理（模型版本保留）",
                   "脚本：PCB_Dual/docs/演示视频脚本.md"])
        self.writer.release()
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        shutil.move(str(self.tmp), str(self.out))
        print(f"[录屏] 输出 {self.out} 帧数 {self.frames} 时长 {self.frames / FPS:.0f}s", flush=True)


# ---------------- 分镜步骤 ----------------
CAPTIONS = {
    "s02b_single": "分镜15  单张检测：清空队列 → 导入 1 张 → 开始检测",
    "s03_history": "分镜16  历史：检索 / 查看归档 / 复判记录",
    "s04_detect": "分镜17  算法调试：选模板 + 打开测试图 → 运行 → 结果摘要",
    "s05_preprocess": "分镜18  预处理：配方参数调整与预览",
    "s06_core": "分镜19  特征学习（AOI_Core）：等待 Core 就绪，浏览 7 个子页",
    "s06b_import": "分镜20  数据管理：新建数据源 → 识别确认 → 导入 extra_part",
    "s06c_prepare": "分镜21  模型管理：准备模型（fast 档，强制重训）",
    "s06d_activate": "分镜22  激活：激活前自动验证（门控），必要时强制激活",
    "s07_menus": "分镜27  系统设置（尺寸门禁）与菜单：适配器状态 / 输出目录",
}


def _captioned(name: str, fn):
    def w(window):
        title(CAPTIONS[name])
        return fn(window)
    return w


_ORIG = {n: getattr(T, n) for n in ("s00_boot", "s01_studio", "s02_inspect", *CAPTIONS)}
for _n in CAPTIONS:
    setattr(T, _n, _captioned(_n, _ORIG[_n]))


def d00_boot(window) -> None:
    title("分镜1  启动：6 个功能页；特征学习页在 Core 就绪前显示占位")
    for i in range(window.tabs.count()):
        window.tabs.setCurrentIndex(i)
        sub(f"页面：{window.tabs.tabText(i)}")
        hold(1300)
    _ORIG["s00_boot"](window)


def d01_studio(window) -> None:
    title("分镜2  双检模板（未勾选检测项 / 未画 ROI）：发布检查阻断「无法发布」")
    page = window.studio_page
    window.tabs.setCurrentWidget(page)
    from app.template.store import TemplateStore
    store = TemplateStore()
    for tpl in store.list_templates():
        if tpl.display_name.startswith("UI测试_"):
            store.delete(tpl.id)
    tpl = store.create_from_standard_image(T.probe_std(T.STD_IMG, "dual_block"),
                                           display_name="UI测试_dual演示", category=T.CATEGORY,
                                           algorithm_ids=T.ALGS)
    tpl.engine_mode = "dual"
    tpl.model_category = T.CATEGORY
    store.save(tpl)
    page.list_panel.refresh(keep_id=tpl.id)
    hold(1500)
    page.btn_publish.click()
    T.pump(400)
    T.ck("不完整模板发布被阻断", store.load(tpl.id).status != "published")
    hold(2500)
    title("分镜3  删除模板：二次确认后删除")
    page.btn_delete.click()
    T.pump(400)
    T.ck("模板已删除", not store.exists(tpl.id))
    title("分镜4-7  新建传统模板 → 检测项全选/全不选/精确勾选 → 草稿 → ROI → 发布 v1")
    _ORIG["s01_studio"](window)


T.s00_boot = d00_boot
T.s01_studio = d01_studio
def _reject(dlg) -> None:
    hold(3500)
    dlg.reject()


def d02_inspect(window) -> None:
    page = window.studio_page
    window.tabs.setCurrentWidget(page)
    from app.template.store import TemplateStore
    store = TemplateStore()
    tid = T.STATE["template_id"]
    title("分镜8  一图一模板：再次选同一标准图 → 提示已绑定并打开已有模板")
    T.NEXT_FILE["file"] = T.STD_IMG
    page.btn_std.click()
    T.pump(300)
    T.ck("同图未新建模板", page.current is not None and page.current.id == tid)

    title("分镜9  试跑：选一张多件图试跑（不归档、不写产线历史）")
    T.NEXT_FILE["file"] = T.DEFECT_IMG
    T.wait_modal("试跑结果", _reject, timeout_s=60, desc="试跑结果")
    page.btn_trial.click()
    T.pump(300)

    title("分镜10  调参辅助：色板抽色 / 自动抽色 / 预处理（与模板 ROI 打通）")
    T.wait_modal("调参辅助", _reject, timeout_s=30, desc="调参辅助", optional=True)
    page.btn_tuner.click()
    T.pump(300)

    title("分镜11  已发布模板再保存 → 升版本为 v2 草稿 → 重新发布")
    page.btn_save.click()
    T.pump(300)
    tpl = store.load(tid)
    T.ck("升版本后为草稿 v2", tpl.status == "draft" and int(tpl.version) == 2,
         f"status={tpl.status} v{tpl.version}")
    page.btn_publish.click()
    T.pump(500)
    tpl = store.load(tid)
    T.ck("v2 已发布", tpl.status == "published" and int(tpl.version) == 2, f"v{tpl.version}")
    title("分镜12-14  自动检测：批量（多件 + 良品 + 4 张扰动良品）→ 过滤 → 下一张 NG → 复判")
    _ORIG["s02_inspect"](window)


def d02c_skip(window) -> None:
    T.later(200, lambda: T.s03_history(window))


T.s02_inspect = d02_inspect
T.s02c_metrics = d02c_skip


def d_dual(window) -> None:
    title("分镜23  双检模板（传统 + 特征）：模型已激活 → 发布检查通过 → 发布")
    page = window.studio_page
    window.tabs.setCurrentWidget(page)
    store, tpl = T.make_tpl(T.STD_IMG, "UI测试_dual", T._X0, T._Y0, "dual", model_category=T.CATEGORY)
    page.list_panel.refresh(keep_id=tpl.id)
    T.pump(300)
    cur = page.current
    T.note(f"dual 页面当前模板={getattr(cur, 'id', None)}/{getattr(cur, 'display_name', None)} "
           f"目标={tpl.id} 引擎={page.combo_engine.currentData()} 品类框={page.combo_model_category.currentText()!r} "
           f"勾选={sum(cb.isChecked() for cb in page.algo_checks.values())}")
    T.ck("dual 模板已选中", cur is not None and cur.id == tpl.id)
    sub(f"品类模型选择：{T.CATEGORY}")
    page.combo_model_category.setCurrentText(T.CATEGORY)
    hold(1500)
    page.btn_publish.click()
    T.pump(500)
    T.ck("dual 模板已发布", store.load(tpl.id).status == "published")
    insp = window.inspect_page
    window.tabs.setCurrentWidget(insp)
    T.pump(300)
    idx = insp.picker.combo.findData(tpl.id)
    if idx >= 0:
        insp.picker.combo.setCurrentIndex(idx)
    T.pump(300)
    title("分镜24  双检检测：多件 / 良品 / 尺寸不符图（尺寸门禁 → ERROR）")
    img = T._imread(T.STD_IMG)
    small = T._imwrite(T.TMP / "size_mismatch.jpg", cv2.resize(img, (3000, 2200)), jpeg=95)
    files = [T.DEFECT_IMG, T.STD_IMG, small]
    insp.btn_clear_queue.click()
    T.pump(300)
    T.NEXT_FILE["files"] = files
    insp.btn_add_files.click()
    T.pump(300)
    insp.btn_run.click()
    T.pump(300)

    def fin():
        return insp._worker is None and len(insp._runs_by_path) >= len(files)

    def after():
        tags = {Path(p).name: insp._run_tag(r) for p, r in insp._runs_by_path.items()}
        T.note(f"dual 判定: {tags}")
        exp = {Path(T.DEFECT_IMG).name: ("NG", "GRAY"), Path(T.STD_IMG).name: ("OK", "GRAY"),
               "size_mismatch.jpg": ("ERR",)}
        for n, ok in exp.items():
            T.ck(f"dual {n} → {tags.get(n)}", tags.get(n) in ok, f"期望 {'/'.join(ok)}")
        for r in range(insp.list_queue.count()):
            insp.list_queue.setCurrentRow(r)
            sub("逐张查看：" + insp.list_queue.item(r).text()[:70])
            hold(2500)
        T.later(300, lambda: d_hub(window))

    T.poll_until("双检检测完成", fin, after, timeout_s=120)


def _set_combo(combo, data=None, text=None) -> None:
    idx = combo.findData(data) if data is not None else combo.findText(text)
    if idx >= 0:
        combo.setCurrentIndex(idx)


def d_hub(window) -> None:
    hub = window.model_page
    window.tabs.setCurrentWidget(hub)
    T.pump(300)
    if not getattr(hub, "_built", False):
        T.warn("CoreHub 未构建，跳过分镜25-26")
        T.later(300, lambda: T.s07_menus(window))
        return
    title("分镜25  数据增强 / 模型管理 / 评估看板（benchmark，limit=5）")
    for t in ("数据增强", "模型管理"):
        T._hub_page(hub, t)
        sub(f"子页：{t}")
        hold(2500)
    ev = T._hub_page(hub, "评估看板")
    try:
        _set_combo(ev.combo_category, data=T.CATEGORY)
        if ev.combo_category.currentData() != T.CATEGORY:
            _set_combo(ev.combo_category, text=T.CATEGORY)
        _set_combo(ev.combo_type, data="benchmark")
        T.pump(200)
        ev.spin_limit.setValue(5)  # 切 benchmark 会自动设为 20，放在后面
        hold(1200)
        ev.btn_run.click()
        sub("运行评估：benchmark ×5")
        hold(8000)
        T.ck("评估看板 benchmark 已运行", True, f"type={ev.combo_type.currentData()} limit={ev.spin_limit.value()}")
    except Exception as exc:
        T.ck("评估看板 benchmark", False, str(exc)[:150])

    title("分镜26  标注复核 / 学习效果 / 存档统计")
    for t in ("标注复核", "学习效果"):
        T._hub_page(hub, t)
        sub(f"子页：{t}")
        hold(2500)
    st = T._hub_page(hub, "存档统计")
    try:
        # 存档统计：选首个工单查看统计（无工单则展示空态）
        tree = st.tree
        for gi in range(tree.topLevelItemCount()):
            grp = tree.topLevelItem(gi)
            if grp is not None and grp.childCount():
                tree.setCurrentItem(grp.child(0))
                sub(f"工单档案：{grp.child(0).text(0)}")
                hold(2000)
                break
    except Exception as exc:
        T.warn("存档统计选择工单", str(exc)[:150])
    T.later(300, lambda: T.s07_menus(window))


T.s06e_feature_detect = lambda w: T.later(200, lambda: d_dual(w))

_orig_finish = T.finish


def _finish() -> None:
    title("演示结束")
    if REC is not None:
        REC.stop_ev.set()
        REC.join(60)
        print(f"视频已保存：{REC.out}", flush=True)
    _orig_finish()


T.finish = _finish


def main() -> None:
    global REC
    from app.core.single_instance import acquire_lock
    if not acquire_lock("pcbins_ui"):
        print("已有 PCB_Dual 实例在运行，请先关闭再测试", flush=True)
        os._exit(2)
    from app.config import ensure_dirs
    ensure_dirs()
    import main as main_mod
    main_mod._start_backend_if_port_free()
    main_mod._start_aoi_core_async()

    app = QApplication(sys.argv)
    app.setApplicationName("PCB 缺陷检测系统 v2")
    from app.core.aoi_core_launcher import stop_core
    app.aboutToQuit.connect(stop_core)
    from app.ui.main_window import MainWindow
    window = MainWindow()
    if hasattr(window.model_page, "shutdown"):
        app.aboutToQuit.connect(window.model_page.shutdown)
    window.showMaximized()
    hold(800)
    if RECORD:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        REC = Recorder(OUT_DIR / f"演示_{datetime.datetime.now():%Y%m%d_%H%M%S}.mp4")
        REC.start()
    T.later(1500, lambda: T.s00_boot(window))
    QTimer.singleShot(90 * 60 * 1000, T.finish)
    app.exec()


if __name__ == "__main__":
    main()
