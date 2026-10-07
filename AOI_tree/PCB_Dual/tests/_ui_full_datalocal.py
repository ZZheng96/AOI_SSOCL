"""PCB_Dual 全页面真实 UI 驱动测试（data_local 数据）。

进程内驱动真实 PySide6 控件：真实窗口显示、真实按钮 click、monkeypatch
QMessageBox（自动应答并记录）与 QFileDialog（按测试队列返回 data_local 图片）。
逐页操作：模板建模 → 自动检测 → 历史 → 算法调试 → 预处理 → 特征学习(AOI_Core) → 菜单。

用法（PCB_Dual 目录下）：
    e:\\CPIPC\\CGAIC\\.env\\Scripts\\python.exe tests\\_ui_full_datalocal.py
"""
from __future__ import annotations

import os
import random
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

DATA_LOCAL = Path(r"e:\CPIPC\CGAIC\data_local")
CATEGORY = "extra_part"          # 同尺寸 4096x3000 成对数据（temp 良品 / img 多件）
EXTRA = DATA_LOCAL / CATEGORY
ALGS = ["body_错件", "body_缺件", "body_移位", "body_破损"]   # 多件无专用算法，取 body 大类近似
ROI_HALF = 300                   # 以文件名 (x,y) 为中心框 600x600

from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox  # noqa: E402
from PySide6.QtCore import QTimer  # noqa: E402

# ---------------- 问题收集 ----------------
ISSUES: list[tuple[str, str, str]] = []   # (级别FAIL/WARN, 检查项, 说明)
DIALOGS: list[str] = []                   # 弹窗记录
NEXT_FILE: dict = {"file": "", "files": [], "dir": ""}


def ck(name: str, ok: bool, detail: str = "") -> None:
    ISSUES.append(("FAIL" if not ok else "PASS", name, detail))
    print(f"{'✅' if ok else '❌'} {name}" + (f"  | {detail}" if detail else ""), flush=True)


def warn(name: str, detail: str = "") -> None:
    ISSUES.append(("WARN", name, detail))
    print(f"⚠️ {name}  | {detail}", flush=True)


def note(msg: str) -> None:
    print(f"…… {msg}", flush=True)


# ---------------- monkeypatch ----------------
def _dlg(kind):
    def _show(parent, title, text, *args, **kwargs):
        line = f"[{kind}] {title} | {str(text)[:160]}"
        DIALOGS.append(line)
        print(f"    [弹窗{kind}] {title} | {str(text)[:120]}", flush=True)
        if kind == "question":
            return QMessageBox.StandardButton.Yes
        return QMessageBox.StandardButton.Ok
    return _show


QMessageBox.information = _dlg("information")  # type: ignore[assignment]
QMessageBox.warning = _dlg("warning")  # type: ignore[assignment]
QMessageBox.critical = _dlg("critical")  # type: ignore[assignment]
QMessageBox.question = _dlg("question")  # type: ignore[assignment]


def _get_open_file_name(parent=None, caption="", dir="", filter="", *a, **k):
    path = NEXT_FILE.get("file") or ""
    print(f"    [文件对话框:{caption}] -> {path}", flush=True)
    return path, filter


def _get_open_file_names(parent=None, caption="", dir="", filter="", *a, **k):
    paths = list(NEXT_FILE.get("files") or [])
    print(f"    [文件对话框:{caption}] -> {len(paths)} 个文件", flush=True)
    return paths, filter


def _get_existing_directory(parent=None, caption="", dir="", *a, **k):
    path = NEXT_FILE.get("dir") or ""
    print(f"    [目录对话框:{caption}] -> {path}", flush=True)
    return path


QFileDialog.getOpenFileName = _get_open_file_name  # type: ignore[assignment]
QFileDialog.getOpenFileNames = _get_open_file_names  # type: ignore[assignment]
QFileDialog.getExistingDirectory = _get_existing_directory  # type: ignore[assignment]


# ---------------- Qt 驱动辅助 ----------------
def pump(ms: int = 50) -> None:
    app = QApplication.instance()
    end = time.time() + ms / 1000.0
    while time.time() < end:
        app.processEvents()
        time.sleep(0.005)


STEPS: list[tuple[int, callable]] = []


def later(ms: int, fn) -> None:
    def _wrap():
        try:
            fn()
        except Exception:
            ck(f"步骤异常:{getattr(fn, '__name__', fn)}", False, traceback.format_exc()[-400:])
            finish()
    QTimer.singleShot(ms, _wrap)


def poll_until(desc: str, cond, done, timeout_s: float = 60.0, interval_ms: int = 300) -> None:
    """轮询 cond() 为真后调 done()；超时记 FAIL 并继续。"""
    t0 = time.time()

    def _tick():
        try:
            if cond():
                print(f"    [轮询就绪] {desc} ({time.time()-t0:.1f}s)", flush=True)
                done()
                return
            if time.time() - t0 > timeout_s:
                ck(f"轮询超时:{desc}", False, f">{timeout_s:.0f}s")
                done()
                return
            QTimer.singleShot(interval_ms, _tick)
        except Exception:
            ck(f"轮询异常:{desc}", False, traceback.format_exc()[-400:])
            done()
    QTimer.singleShot(interval_ms, _tick)


# ---------------- 测试数据（随机抽样，seed 可复现） ----------------
SEED = int(os.environ.get("UI_TEST_SEED") or int(time.time()))
RNG = random.Random(SEED)
N_PAIRS = int(os.environ.get("UI_TEST_PAIRS") or 3)


def _imread(p: str | Path):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def _imwrite(p: str | Path, img, jpeg: int | None = None) -> str:
    params = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg)] if jpeg else []
    ok, buf = cv2.imencode(Path(p).suffix or ".jpg", img, params)
    assert ok
    buf.tofile(str(p))
    return str(p)


def _pairs() -> list[tuple[str, str, str, int, int]]:
    """test/extra 多件图 ↔ 同 stem 的 *_temp.jpg 良品，(temp, img, stem, x, y)。"""
    temps = {p.name[: -len("_temp.jpg")]: p for d in ("train/good", "test/good")
             for p in (EXTRA / d).glob("*_temp.jpg")}
    out = []
    for p in sorted((EXTRA / "test/extra").glob("*_img.jpg")):
        stem = p.name[: -len("_img.jpg")]
        if stem in temps:
            parts = stem.split("_")
            out.append((str(temps[stem]), str(p), stem, int(parts[-2]), int(parts[-1])))
    return out


ALL_PAIRS = _pairs()
PAIRS = RNG.sample(ALL_PAIRS, min(N_PAIRS, len(ALL_PAIRS)))
TMP = Path(tempfile.mkdtemp(prefix="ui_test_"))


def synth(src: str, tag: str, dx: float = 0, dy: float = 0, sigma: float = 0,
          bright: float = 0, jpeg: int | None = None, seed: int = 0) -> str:
    """由良品合成"应判 OK"的扰动图（平移/噪声/亮度/JPEG 重压缩）。"""
    img = _imread(src).astype(np.float32)
    if dx or dy:
        m = np.float32([[1, 0, dx], [0, 1, dy]])
        img = cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REPLICATE)
    if sigma:
        img += np.random.default_rng(seed).normal(0, sigma, img.shape).astype(np.float32)
    img += bright
    out = np.clip(img, 0, 255).astype(np.uint8)
    return _imwrite(TMP / f"{Path(src).stem}__{tag}.jpg", out, jpeg=jpeg or 100)


_T0, _I0, _S0, _X0, _Y0 = PAIRS[0]
STD_IMG = _T0
DEFECT_IMG = _I0
EXPECT: dict[str, str] = {
    _I0: "NG",
    _T0: "OK",
    synth(_T0, "jpeg95", jpeg=95): "OK",
    synth(_T0, "noise2", sigma=2, seed=SEED): "OK",
    synth(_T0, "shift2_noise2", dx=2, dy=1, sigma=2, seed=SEED + 1): "OK",
    synth(_T0, "bright5", bright=5): "OK",
}
TEST_IMGS: list[str] = list(EXPECT)
TEMPLATE_NAME = "UI测试_extra_part"
STATE: dict = {"template_id": None, "batch_done": 0}
METRICS: dict = {}   # 报告用的量化指标


def roi_rect(x: int, y: int, w: int, h: int) -> dict:
    x0, y0 = max(1, x - ROI_HALF), max(1, y - ROI_HALF)
    x1, y1 = min(w - 1, x + ROI_HALF), min(h - 1, y + ROI_HALF)
    return {"shape": "rect", "x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}


def apply_roi(tpl, x: int, y: int, std_path: str) -> list[str]:
    """把以 (x,y) 为中心的 600x600 框写入模板所有区域目标。"""
    from app.detect.catalog import get_catalog
    from app.template.regions import region_targets_for_kind
    h, w = _imread(std_path).shape[:2]
    targets = set()
    for it in tpl.enabled_detection_items():
        kind = it.region_kind or get_catalog().region_kind(it.algorithm_id)
        targets.update(region_targets_for_kind(kind if kind != "none" else ""))
    for t in targets:
        tpl.set_region_shapes(t, [roi_rect(x, y, w, h)])
    return sorted(targets)


def probe_std(std: str, name: str) -> str:
    """一图一模板（store 按内容 sha1 去重）：探针模板用无损 PNG 副本，像素一致但哈希不同。"""
    src = Path(std)
    d = TMP / "probe_std"
    d.mkdir(exist_ok=True)
    copy = d / f"{src.stem}__{name}.png"
    img = cv2.imdecode(np.fromfile(str(src), np.uint8), cv2.IMREAD_COLOR)
    cv2.imencode(".png", img)[1].tofile(str(copy))
    return str(copy)


def make_tpl(std: str, name: str, x: int, y: int, mode: str, model_category: str = ""):
    from app.template.store import TemplateStore
    store = TemplateStore()
    tpl = store.create_from_standard_image(probe_std(std, name), display_name=name, category=CATEGORY,
                                           algorithm_ids=ALGS)
    tpl.engine_mode = mode
    if model_category:
        tpl.model_category = model_category
    apply_roi(tpl, x, y, std)
    store.save(tpl)
    return store, store.load(tpl.id)


def judge(summary) -> str:
    if summary is None:
        return "ERROR"
    if getattr(summary, "gate_blocked", False):
        return "BLOCKED"
    return "OK" if summary.overall_ok else "NG"


# ================= 各页面测试步骤 =================
def s00_boot(window) -> None:
    print("\n== S00 启动检查 ==", flush=True)
    ck("窗口已显示", window.isVisible())
    titles = [window.tabs.tabText(i) for i in range(window.tabs.count())]
    ck("6 个页面 Tab", window.tabs.count() == 6, "、".join(titles))
    expect = ["自动检测", "历史", "模板建模", "特征学习（AOI_Core）", "算法调试", "预处理"]
    ck("Tab 标题与顺序", titles == expect, "、".join(titles))
    # 后端健康
    try:
        import requests
        r = requests.get("http://127.0.0.1:8022/api/health", timeout=3)
        ck("后端 8022 /api/health", r.ok, f"status={r.status_code}")
    except Exception as exc:
        ck("后端 8022 /api/health", False, str(exc)[:120])
    status = window.statusBar().currentMessage()
    if "未加载" in status:
        warn("启动时算法适配器有未加载项", status[:150])
    else:
        ck("状态栏无适配器告警", True, status[:60] or "（空）")
    later(300, lambda: s01_studio(window))


def s01_studio(window) -> None:
    print("\n== S01 模板建模页 ==", flush=True)
    page = window.studio_page
    window.tabs.setCurrentWidget(page)
    pump(200)

    # 清理上次测试残留模板
    from app.template.store import TemplateStore
    store = TemplateStore()
    for tpl in store.list_templates():
        if tpl.display_name.startswith("UI测试_"):
            store.delete(tpl.id)
            note(f"清理残留模板 {tpl.display_name}")

    # 1) 选择标准图新建模板
    NEXT_FILE["file"] = STD_IMG
    page.btn_std.click()
    pump(400)
    tpl = page.current
    ck("选择标准图后创建模板", tpl is not None,
       f"id={tpl.id if tpl else None} std={Path(STD_IMG).name}")
    if tpl is None:
        finish()
        return
    STATE["template_id"] = tpl.id
    lbl = page.lbl_std_path.text()
    ck("标准图标签已显示", bool(lbl.strip()), lbl[:80])
    if "standard" in lbl and Path(STD_IMG).stem not in lbl:
        warn("[UX] 标准图标签显示副本名 standard.jpg 而非源文件名",
             f"源={Path(STD_IMG).name} 标签={lbl[:60]}（store 复制为 standard.jpg，用户无法确认选的是哪张）")

    # 2) 表单字段
    page.edit_name.setText(TEMPLATE_NAME)
    engines = [page.combo_engine.itemText(i) for i in range(page.combo_engine.count())]
    ck("引擎模式下拉含传统/特征/双检", len(engines) >= 3, "、".join(engines))
    # 默认 dual：品类模型未准备时发布检查应阻断（预期行为，验证门禁存在）
    from app.engines.feature import get_engine
    cur_ver = None
    try:
        cur_ver = get_engine().current_version(CATEGORY)
    except Exception as exc:
        note(f"current_version 查询异常: {exc}")
    tpl_d = store.load(tpl.id)
    tpl_d.engine_mode = "dual"
    tpl_d.category = CATEGORY
    dual_issues = store.publish_checklist(tpl_d)
    has_gate = any("尚未准备" in i for i in dual_issues)
    if cur_ver:
        note(f"Core 已有 {CATEGORY} 激活模型 {cur_ver}，dual 门禁不触发属正常")
    else:
        ck("dual 模式下品类模型未准备 → 发布检查阻断", has_gate, "；".join(dual_issues)[:160])
    idx_trad = page.combo_engine.findData("traditional")
    if idx_trad >= 0:
        page.combo_engine.setCurrentIndex(idx_trad)
    majors = [page.combo_major.itemText(i) for i in range(page.combo_major.count())]
    ck("检测类别下拉非空", len(majors) > 0, "、".join(majors))

    # 3) 选择 body 大类，只勾 ALGS（多件无专用算法，用 body 近似）
    body_idx = -1
    for i in range(page.combo_major.count()):
        items = page._algo_groups.get(page.combo_major.itemData(i), [])
        if any(str(m[0]).startswith("body_") for m in items):  # items 为 (alg_id, label) 元组
            body_idx = i
            break
    if body_idx < 0 and page.combo_major.count() > 0:
        body_idx = 0
        warn("未找到 body 大类，回退第 1 个大类", majors[0] if majors else "")
    page.combo_major.setCurrentIndex(body_idx)
    pump(200)
    page.btn_items_all.click()
    pump(200)
    n_all = sum(1 for cbw in page.algo_checks.values() if cbw.isChecked())
    page.btn_items_none.click()
    pump(200)
    n_none = sum(1 for cbw in page.algo_checks.values() if cbw.isChecked())
    ck("全选/全不选按钮生效（当前大类）", n_all > 0 and n_all > n_none, f"全选后 {n_all} 项，全不选后 {n_none} 项")
    for aid, cbw in page.algo_checks.items():
        cbw.setChecked(aid in ALGS)
    pump(200)
    checked = sorted(aid for aid, cbw in page.algo_checks.items() if cbw.isChecked())
    missing = [a for a in ALGS if a not in page.algo_checks]
    ck("检测项精确勾选 ALGS", checked == sorted(a for a in ALGS if a in page.algo_checks) and not missing,
       f"勾选={checked} 缺失={missing}")

    # 4) 保存草稿
    page.btn_save.click()
    pump(400)
    tpl = page.current
    ck("保存草稿成功", tpl is not None and store.exists(tpl.id),
       f"status={tpl.status if tpl else None}")

    # 5) 发布前检查：未画 ROI 时应提示阻断；写入 (x,y) 中心 600x600 框后应无阻断
    tpl = store.load(STATE["template_id"])
    ck("保存后引擎模式=traditional", tpl.engine_mode == "traditional", tpl.engine_mode)
    ck("保存后启用算法=ALGS", sorted(i.algorithm_id for i in tpl.enabled_detection_items()) == sorted(ALGS),
       str([i.algorithm_id for i in tpl.enabled_detection_items()]))
    pre = [i for i in store.publish_checklist(tpl) if i.startswith("[阻断]")]
    note(f"未画 ROI 时阻断项: {pre}")
    targets = apply_roi(tpl, _X0, _Y0, STD_IMG)
    store.save(tpl)
    note(f"ROI 中心=({_X0},{_Y0}) 600x600 写入目标: {targets}")
    page.list_panel.refresh(keep_id=tpl.id)
    pump(300)
    blockers2 = [i for i in store.publish_checklist(store.load(STATE["template_id"])) if i.startswith("[阻断]")]
    ck("写入 ROI 后发布检查无阻断", not blockers2, "；".join(blockers2)[:200])
    page.btn_publish.click()
    pump(600)
    tpl = store.load(STATE["template_id"])
    ck("点击发布后模板为已发布", tpl is not None and tpl.status == "published",
       f"status={tpl.status if tpl else None} v{tpl.version if tpl else '-'}")
    pump(400)
    # 发布成功弹窗选 Yes → 应自动跳到自动检测页并选中该模板
    later(500, lambda: s02_inspect(window))


def s02_inspect(window) -> None:
    print("\n== S02 自动检测页 ==", flush=True)
    page = window.inspect_page
    in_inspect = window.tabs.currentWidget() is page
    ck("发布后自动跳转自动检测页", in_inspect,
       f"当前 Tab={window.tabs.tabText(window.tabs.currentIndex())}")
    window.tabs.setCurrentWidget(page)
    pump(400)
    tpl_ok = page.current_template is not None and page.current_template.id == STATE["template_id"]
    if not tpl_ok:
        # 手动选中
        idx = page.picker.combo.findData(STATE["template_id"])
        if idx >= 0:
            page.picker.combo.setCurrentIndex(idx)
            pump(300)
        tpl_ok = page.current_template is not None and page.current_template.id == STATE["template_id"]
    ck("模板选择器已选中发布模板", tpl_ok,
       page.picker.combo.currentText()[:80])
    ck("模板摘要文案已填充", "检测项" in page.lbl_rules.text(), page.lbl_rules.text()[:100].replace("\n", " / "))
    ck("未上料时开始检测按钮禁用", not page.btn_run.isEnabled())

    # ---- 批量检测 ----
    NEXT_FILE["files"] = TEST_IMGS
    page.btn_add_files.click()
    pump(400)
    ck("导入图片后队列计数", len(page.queue) == len(TEST_IMGS),
       f"queue={len(page.queue)}/{len(TEST_IMGS)}")
    ck("队列计数标签", page.lbl_queue_meta.text().endswith(f"/{len(TEST_IMGS)}"),
       page.lbl_queue_meta.text())
    ck("上料后开始检测按钮可用", page.btn_run.isEnabled())

    STATE["batch_done"] = 0
    t0 = time.time()
    page.btn_run.click()
    pump(300)

    def batch_finished():
        return page._worker is None and len(page._runs_by_path) >= len(TEST_IMGS)

    def after_batch():
        elapsed = time.time() - t0
        n = len(page._runs_by_path)
        ck("批量检测完成（全部出结果）", n == len(TEST_IMGS), f"{n}/{len(TEST_IMGS)} 耗时 {elapsed:.1f}s")
        tags = {}
        rows = []
        for p, run in page._runs_by_path.items():
            tag = page._run_tag(run)
            tags[tag] = tags.get(tag, 0) + 1
            if run.error:
                warn(f"检测出错:{Path(p).name}", str(run.error)[:150])
            exp = EXPECT.get(str(p)) or EXPECT.get(str(Path(p)))
            bad = []
            if getattr(run, "summary", None) is not None:
                bad = [r.algorithm for r in run.summary.results if not r.ok and not r.skipped]
            rows.append((Path(p).name, exp, tag, bad))
            note(f"  {Path(p).name}: 期望={exp} 实际={tag} 报NG算法={bad}")
        note(f"判定分布: {tags}")
        ng_exp = [r for r in rows if r[1] == "NG"]
        ok_exp = [r for r in rows if r[1] == "OK"]
        hit = sum(1 for r in ng_exp if r[2] in ("NG", "GRAY"))
        fp = [r for r in ok_exp if r[2] != "OK"]
        per = elapsed / max(1, n)
        METRICS["ui_batch"] = {"recall": f"{hit}/{len(ng_exp)}", "fp": f"{len(fp)}/{len(ok_exp)}",
                               "per_img_s": round(per, 2), "rows": rows}
        ck("UI 批量：多件图检出（召回）", hit == len(ng_exp), f"{hit}/{len(ng_exp)}")
        ck("UI 批量：良品/合成良品无误检", not fp,
           f"误检 {len(fp)}/{len(ok_exp)}: " + "；".join(f"{r[0]}→{r[2]} {r[3]}" for r in fp)[:220])
        ck("UI 批量：单图平均耗时 ≤1s", per <= 1.0, f"{per:.2f}s/张（含 UI 渲染，{n} 张 {elapsed:.1f}s）")
        ck("批量后队列列表带判定标签", "[" in page.list_queue.item(0).text() if page.list_queue.count() else False,
           page.list_queue.item(0).text()[:60] if page.list_queue.count() else "")
        ck("直通率标签已更新", "直通率" in page.lbl_yield.text() and "—" not in page.lbl_yield.text(),
           page.lbl_yield.text())
        snap = page.header_snapshot()
        ck("头部快照统计", snap["total"] == len(TEST_IMGS), str(snap))
        # 归档
        from app.config import get_outputs_dir
        runs = list(page._runs_by_path.values())
        need = [r for r in runs if getattr(r, "summary", None) is not None and not r.summary.gate_blocked]
        missing_f = [r for r in need if not (r.folder and Path(r.folder).exists())]
        n_blocked = len(runs) - len(need)
        ck("检测结果已归档（非门禁拦截的结果均有归档目录）", not missing_f,
           f"应归档 {len(need)}，缺 {len(missing_f)}；门禁拦截不归档 {n_blocked}（设计如此）于 {get_outputs_dir()}")
        # 过滤按钮
        page.btn_filt_ng.click()
        pump(200)
        ng_rows = page.list_queue.count()
        page.btn_filt_err.click()
        pump(200)
        err_rows = page.list_queue.count()
        page.btn_filt_all.click()
        pump(200)
        all_rows = page.list_queue.count()
        ck("队列过滤（NG/ERROR/全部）", all_rows == len(TEST_IMGS) and ng_rows <= all_rows and err_rows <= all_rows,
           f"NG={ng_rows} ERR={err_rows} ALL={all_rows}")
        # 下一张 NG
        page.current_index = -1
        page.btn_next_ng.click()
        pump(200)
        if any(page._run_tag(r) in ("NG", "GRAY") for r in page._runs_by_path.values()):
            run = page._runs_by_path.get(str(page.queue[page.current_index]))
            ck("下一张 NG 定位", run is not None and page._run_tag(run) in ("NG", "GRAY"),
               f"index={page.current_index}")
        else:
            note("本批无 NG，跳过下一张 NG 验证（弹窗应提示）")
        # 复判（对当前张）
        if page.last_run is not None:
            page.btn_fp.click()
            pump(300)
            page.btn_confirm_ng.click()
            pump(300)
            ck("复判按钮执行无异常", True, "复判合格+复判不合格均已点击")
        else:
            ck("复判按钮执行", False, "last_run 为空")
        later(300, lambda: s02b_single(window))

    poll_until("批量检测完成", batch_finished, after_batch, timeout_s=180)


def s02b_single(window) -> None:
    print("\n== S02b 单张检测（Enter 路径）==", flush=True)
    page = window.inspect_page
    page.btn_clear_queue.click()
    pump(300)
    ck("清空队列", len(page.queue) == 0 and page.list_queue.count() == 0)
    NEXT_FILE["files"] = TEST_IMGS[:1]
    page.btn_add_files.click()
    pump(300)
    t0 = time.time()
    page.btn_run.click()  # 单张走 _run_one
    pump(200)

    def one_done():
        return page._worker is None and len(page._runs_by_path) >= 1

    def after_one():
        elapsed = time.time() - t0
        run = page._runs_by_path.get(str(page.queue[0])) if page.queue else None
        ck("单张检测完成", run is not None, f"耗时 {elapsed:.1f}s overall={run.summary.overall if run else '-'}")
        if run and elapsed > 0:
            per = elapsed
            if per > 1.0:
                warn("单张（UI 路径）耗时超 1s", f"{per:.2f}s（含 UI 渲染）")
        later(200, lambda: s02c_metrics(window))

    poll_until("单张检测完成", one_done, after_one, timeout_s=60)


def s02c_metrics(window) -> None:
    """服务层量化：随机样本对上的召回/误检/耗时 + ROI 偏离 + 各大类同图自比误报。"""
    print("\n== S02c 服务层检测量化（随机样本对）==", flush=True)
    from app.inspect.service import InspectService
    from app.detect.catalog import get_catalog
    svc = InspectService()
    rows, times = [], []
    for k, (temp, img, stem, x, y) in enumerate(PAIRS):
        store, tpl = make_tpl(temp, f"PROBE_pair{k}", x, y, "traditional")
        tpl = store.publish(tpl)
        cases = [("多件", img, "NG"), ("自比", temp, "OK"),
                 ("jpeg95", synth(temp, f"p{k}_j95", jpeg=95), "OK"),
                 ("噪声σ2", synth(temp, f"p{k}_n2", sigma=2, seed=SEED + k), "OK"),
                 ("平移2px+σ2", synth(temp, f"p{k}_s2", dx=2, dy=1, sigma=2, seed=SEED + 10 + k), "OK"),
                 ("亮度+5", synth(temp, f"p{k}_b5", bright=5), "OK")]
        for cname, path, exp in cases:
            t0 = time.time()
            r = svc.run_with_template(tpl, path, test_path=path, archive=False)
            dt = time.time() - t0
            times.append(dt)
            got = judge(r.summary)
            bad = [z.algorithm for z in r.summary.results if not z.ok and not z.skipped]
            rows.append((stem, cname, exp, got, bad, dt))
            note(f"  [{stem}] {cname}: 期望={exp} 实际={got} {dt:.2f}s 报NG={bad}")
        store.delete(tpl.id)
        # ROI 偏离：框放到远离 (x,y) 的位置，多件不在框内 → 理论应 OK。
        # 注意 *_img.jpg 与 *_temp.jpg 是独立采集（整板可错位 30~40px、框外还有其它差异），
        # 直接用 img 会把"整板不一致"误当框外隔离失败；故在良品上只贴入多件处的 600x600 实拍块。
        tmat = _imread(temp)
        h, w = tmat.shape[:2]
        comp = tmat.copy()
        ya, yb, xa, xb = max(0, y - ROI_HALF), min(h, y + ROI_HALF), max(0, x - ROI_HALF), min(w, x + ROI_HALF)
        comp[ya:yb, xa:xb] = _imread(img)[ya:yb, xa:xb]
        comp_path = _imwrite(TMP / f"{stem}__far_comp.jpg", comp, jpeg=100)
        fx, fy = (w - x if abs(w - 2 * x) > 1200 else (x + 1500) % w), (h - y if abs(h - 2 * y) > 1200 else (y + 1200) % h)
        fx, fy = min(max(fx, ROI_HALF + 2), w - ROI_HALF - 2), min(max(fy, ROI_HALF + 2), h - ROI_HALF - 2)
        store, tpl = make_tpl(temp, f"PROBE_far{k}", fx, fy, "traditional")
        tpl = store.publish(tpl)
        r = svc.run_with_template(tpl, comp_path, test_path=comp_path, archive=False)
        got = judge(r.summary)
        rows.append((stem, f"ROI偏离({fx},{fy})多件", "OK", got,
                     [z.algorithm for z in r.summary.results if not z.ok and not z.skipped], 0.0))
        note(f"  [{stem}] ROI 偏离多件位置: 期望=OK 实际={got}")
        store.delete(tpl.id)

    ng_rows = [r for r in rows if r[2] == "NG"]
    ok_rows = [r for r in rows if r[2] == "OK"]
    hit = sum(1 for r in ng_rows if r[3] == "NG")
    fp = [r for r in ok_rows if r[3] != "OK"]
    by_case: dict[str, list[int]] = {}
    for r in ok_rows:
        key = r[1].split("(")[0]
        by_case.setdefault(key, [0, 0])
        by_case[key][0] += r[3] != "OK"
        by_case[key][1] += 1
    avg = sum(times) / max(1, len(times))
    METRICS["svc"] = {"recall": f"{hit}/{len(ng_rows)}", "fp": f"{len(fp)}/{len(ok_rows)}",
                      "fp_by_case": {k: f"{a}/{b}" for k, (a, b) in by_case.items()},
                      "avg_s": round(avg, 2), "max_s": round(max(times), 2), "rows": rows}
    ck("服务层：多件召回", hit == len(ng_rows), f"{hit}/{len(ng_rows)}")
    ck("服务层：良品/扰动良品/ROI偏离 无误检", not fp,
       f"误检 {len(fp)}/{len(ok_rows)}，分项 {METRICS['svc']['fp_by_case']}")
    ck("服务层：单图耗时 ≤1s", max(times) <= 1.0, f"均 {avg:.2f}s 最大 {max(times):.2f}s（4096x3000）")

    # 各大类同图自比误报：同一张 temp 自己对自己，任何 NG 都是误报
    temp, _, stem, x, y = PAIRS[0]
    self_fp = {}
    from app.template.store import TemplateStore
    store = TemplateStore()
    for gid, gname, mans in get_catalog().groups():
        ids = [m.id for m in mans]
        if not ids:
            continue
        tpl = store.create_from_standard_image(probe_std(temp, f"grp_{gid}"), display_name=f"PROBE_grp_{gid}",
                                               category=CATEGORY,
                                               algorithm_ids=ids)
        tpl.engine_mode = "traditional"
        apply_roi(tpl, x, y, temp)
        store.save(tpl)
        try:
            tpl = store.publish(store.load(tpl.id))
            r = svc.run_with_template(tpl, temp, test_path=temp, archive=False)
            bad = [z.algorithm for z in r.summary.results if not z.ok and not z.skipped]
            self_fp[gname] = (judge(r.summary), bad)
        except Exception as exc:
            self_fp[gname] = ("SKIP", [str(exc)[:60]])
        store.delete(tpl.id)
        note(f"  同图自比 [{gname}]: {self_fp[gname][0]} 报NG={self_fp[gname][1]}")
    METRICS["self_fp"] = self_fp
    # 只有"模板差分"类（body/board）同图自比必须 OK；其余大类是对测试图的绝对判定
    # （焊锡/起皱/移位/插件），在未按其目标对象标定的通用 600x600 框上判 NG 属域外，记 WARN。
    diff_groups = {gname for gid, gname, _m in get_catalog().groups() if gid in ("body", "board")}
    bad_groups = {g: v[1] for g, v in self_fp.items() if v[0] == "NG"}
    bad_diff = {g: v for g, v in bad_groups.items() if g in diff_groups}
    bad_abs = {g: v for g, v in bad_groups.items() if g not in diff_groups}
    ck("差分类（本体/板面）同图自比无误报", not bad_diff,
       "；".join(f"{g}:{v}" for g, v in bad_diff.items())[:240])
    if bad_abs:
        warn("绝对判定类在通用框上报 NG（域外，非差分误报；需按对象标定 ROI）",
             "；".join(f"{g}:{v}" for g, v in bad_abs.items())[:240])
    later(200, lambda: s03_history(window))


def s03_history(window) -> None:
    print("\n== S03 历史页 ==", flush=True)
    page = window.history_page
    window.tabs.setCurrentWidget(page)
    pump(500)
    rows = page.table.rowCount()
    ck("历史记录列表非空", rows > 0, f"{rows} 条")
    ck("统计文案已填充", "直通率" in page.lbl_stats.text(), page.lbl_stats.text()[:100])
    if rows == 0:
        later(200, lambda: s04_detect(window))
        return
    # 选中第一条 → 右侧应绑定
    page.table.setCurrentCell(0, 0)
    pump(500)
    rec = page._current()
    ck("选中记录后画布/HUD 绑定", rec is not None,
       f"图号={Path(str(rec.get('test_image', ''))).name if rec else '-'} 判定={rec.get('overall') if rec else '-'}")
    # 判定过滤
    page.combo_filter.setCurrentText("NG")
    pump(300)
    ng_rows = page.table.rowCount()
    page.combo_filter.setCurrentText("OK")
    pump(300)
    ok_rows = page.table.rowCount()
    page.combo_filter.setCurrentText("全部")
    pump(300)
    ck("判定过滤 OK/NG", ng_rows + ok_rows <= rows, f"NG={ng_rows} OK={ok_rows} ALL={rows}")
    # 关键字
    page.edit_keyword.setText(CATEGORY)
    pump(300)
    kw_rows = page.table.rowCount()
    ck("关键字过滤", 0 < kw_rows <= rows, f"'{CATEGORY}' -> {kw_rows} 条")
    page.edit_keyword.clear()
    pump(300)
    # 模板过滤
    ck("模板过滤下拉含测试模板", page.combo_template.count() >= 1,
       f"{page.combo_template.count()} 项")
    # 调试记录开关
    page.chk_debug.setChecked(True)
    pump(300)
    page.chk_debug.setChecked(False)
    pump(300)
    ck("调试记录开关切换无异常", True)
    # 删除一条记录（仅删本次测试产生的 data_local 记录，避免误删真实归档）
    del_row = -1
    for r in range(page.table.rowCount()):
        page.table.setCurrentCell(r, 0)
        pump(100)
        rec = page._current()
        if rec and "data_local" in str(rec.get("test_image") or ""):
            del_row = r
            break
    if del_row >= 0:
        before = page.table.rowCount()
        page.btn_delete.click()
        pump(400)
        after = page.table.rowCount()
        ck("删除记录（data_local 测试归档）", after == before - 1, f"{before} -> {after}")
    else:
        note("历史中无 data_local 记录，跳过删除验证")
    later(200, lambda: s04_detect(window))


def s04_detect(window) -> None:
    print("\n== S04 算法调试页 ==", flush=True)
    page = window.detect_page
    window.tabs.setCurrentWidget(page)
    pump(500)
    # 绑定模板（调试页选择器含草稿）
    idx = page.template_picker.combo.findData(STATE["template_id"])
    ck("调试页模板下拉含测试模板", idx >= 0, page.template_picker.combo.currentText()[:60])
    if idx >= 0:
        page.template_picker.combo.setCurrentIndex(idx)
        pump(400)
    ck("模板已绑定", page.bound_template is not None,
       page.bound_template.display_name if page.bound_template else "None")
    # 打开测试图
    NEXT_FILE["file"] = DEFECT_IMG
    page.open_test()
    pump(500)
    ck("测试图已加载", page.test_bgr is not None, Path(DEFECT_IMG).name)

    def summary_ready():
        return page.last_summary is not None

    def after_run():
        s = page.last_summary
        ck("对照跑图出结果", s is not None,
           f"overall={s.overall if s else '-'} 算法行={len(s.results) if s else 0}" if s else "无")
        if s:
            skipped = [r.display_name or r.algorithm for r in s.results if r.skipped]
            errs = [(r.algorithm, r.error_message or r.message) for r in s.results if r.status == "ERROR"]
            ngs = [r.algorithm for r in s.results if r.status == "NG"]
            if skipped:
                warn("调试页有算法被跳过", "、".join(skipped)[:150])
            for alg, msg in errs:
                warn(f"调试页算法执行出错:{alg}", str(msg)[:150])
            ck("调试页：多件缺陷图判 NG", s.overall == "NG", f"overall={s.overall} 报NG={ngs}")
        # 去预处理调参按钮 → 跳转预处理页
        later(300, lambda: s05_preprocess(window))

    if page.last_summary is None:
        page.btn_run.click()
    poll_until("调试页跑图", summary_ready, after_run, timeout_s=90)


def s05_preprocess(window) -> None:
    print("\n== S05 预处理页 ==", flush=True)
    page = window.preprocess_page
    window.tabs.setCurrentWidget(page)
    pump(400)
    NEXT_FILE["file"] = TEST_IMGS[0]
    page.btn_open.click()
    pump(500)
    ck("打开图片", page.image_bgr is not None, Path(TEST_IMGS[0]).name)
    # 依次切换参数，预览应更新
    from PySide6.QtGui import QPixmap
    combos = [page.combo_color, page.combo_enhance, page.combo_filter, page.combo_morph]
    labels = ["颜色通道", "增强", "滤波", "形态学"]
    for c, lb in zip(combos, labels):
        if c.count() > 1:
            c.setCurrentIndex(1)
            pump(300)
            txt = page.view_out.text() if hasattr(page.view_out, "text") else ""
            failed = "预处理失败" in (txt or "")
            ck(f"切换{lb}={c.currentText()} 预览不报错", not failed, txt[:80] if failed else c.currentText())
    # 保存全局（先备份用户真实配方，_cleanup 中还原）
    gpath = Path(page.recipe_store.root) / "global_default.json"
    RECIPE_BACKUP["path"] = gpath
    RECIPE_BACKUP["text"] = gpath.read_text(encoding="utf-8") if gpath.exists() else None
    page.btn_save_global.click()
    pump(300)
    ck("保存全局默认配方", True, "已点击（弹窗见记录）")
    # 算法专属
    if page.combo_algorithm.count() > 1:
        page.combo_algorithm.setCurrentIndex(1)
        pump(300)
        ck("算法专属保存按钮启用", page.btn_save_algo.isEnabled())
        page.btn_save_algo.click()
        pump(300)
        page.btn_clear_algo.click()
        pump(300)
        ck("算法专属配方保存+清除", True)
    page.btn_reset.click()
    pump(200)
    ck("控件重置为 NONE", all(c.currentText() == "NONE" for c in combos))
    # 立即还原全局配方，避免测试写入的配方污染后续页面的检测耗时/结果
    _restore_recipe()
    later(200, lambda: s06_core(window))


def s06_core(window) -> None:
    print("\n== S06 特征学习（AOI_Core）页 ==", flush=True)
    page = window.model_page
    window.tabs.setCurrentWidget(page)
    pump(800)

    def core_ok():
        from app.core.aoi_core_launcher import core_healthy
        return core_healthy("http://127.0.0.1:8017", timeout=1.5)

    def after_core():
        healthy = core_ok()
        ck("AOI_Core 8017 健康", healthy)
        if not healthy:
            warn("Core 不可用，特征学习页只能验证占位/重试 UI", "")
            later(200, lambda: s07_menus(window))
            return
        page.ensure_built()
        pump(1000)
        built = getattr(page, "_built", False)
        ck("CoreHub 页面已构建", built)
        from app.ui.core_pages import _PAGES as CORE_PAGES
        titles = [t for t, *_ in CORE_PAGES]
        ck("CoreHub 子页清单 7 个", len(titles) == 7, "、".join(titles))
        cats = [page.combo_cat.itemText(i) for i in range(page.combo_cat.count())] if hasattr(page, "combo_cat") else []
        note(f"Core 品类下拉: {cats}")

        def visit(i: int) -> None:
            if i >= len(titles):
                later(300, lambda: s06b_import(window))
                return
            t = titles[i]
            try:
                page.goto(t)
                pump(600)
                w = page.stack.currentWidget() if hasattr(page, "stack") else None
                bad = w is None or "加载失败" in (w.text() if hasattr(w, "text") else "")
                ck(f"子页「{t}」加载", not bad, type(w).__name__ if w else "None")
            except Exception:
                ck(f"子页「{t}」加载", False, traceback.format_exc()[-300:])
            later(100, lambda: visit(i + 1))

        visit(0)

    poll_until("AOI_Core 健康", core_ok, after_core, timeout_s=150, interval_ms=1000)


def _hub_page(hub, title: str):
    from app.ui.core_pages import _PAGES as CORE_PAGES
    hub.goto(title)
    pump(400)
    return hub._pages[[t for t, *_ in CORE_PAGES].index(title)]


def wait_modal(title_part: str, fn, timeout_s: float = 60.0, desc: str = "",
               optional: bool = False, stop=lambda: False) -> None:
    """轮询当前模态框标题含 title_part 后调 fn(dlg)；超时记 FAIL（optional 时只记 note）。"""
    t0 = time.time()

    def _tick():
        dlg = QApplication.activeModalWidget()
        if dlg is not None and title_part in (dlg.windowTitle() or ""):
            try:
                fn(dlg)
            except Exception:
                ck(f"模态框操作异常:{desc or title_part}", False, traceback.format_exc()[-400:])
                dlg.reject()
            return
        if stop():
            return
        if time.time() - t0 > timeout_s:
            if optional:
                note(f"未出现模态框「{title_part}」")
            else:
                ck(f"等待模态框超时:{desc or title_part}", False,
                   f">{timeout_s:.0f}s 当前={dlg.windowTitle() if dlg else None}")
            return
        QTimer.singleShot(300, _tick)
    QTimer.singleShot(300, _tick)


def _btn(dlg, text: str):
    from PySide6.QtWidgets import QPushButton
    return next((b for b in dlg.findChildren(QPushButton) if text in b.text()), None)


def s06b_import(window) -> None:
    """特征学习链路 1/4：数据管理页 新建数据源 → 适配导入 extra_part。"""
    print("\n== S06b 特征学习：新建数据源 + 导入 extra_part ==", flush=True)
    from PySide6.QtWidgets import QComboBox, QLineEdit
    hub = window.model_page
    cli = hub._client
    for ds in cli.list_datasources() or []:
        if str(ds.get("name", "")).startswith("UI测试_"):
            cli.delete_datasource(ds["id"])
            note(f"清理残留数据源 {ds['name']}")
    ds_name = f"UI测试_{CATEGORY}_{time.strftime('%H%M%S')}"
    STATE["ds_name"] = ds_name
    dp = _hub_page(hub, "数据管理")
    ck("数据管理页可用", hasattr(dp, "_on_create_datasource"), type(dp).__name__)

    def fill_create(dlg):
        edits = {e.placeholderText(): e for e in dlg.findChildren(QLineEdit)}
        def _set(key, val):
            e = next((w for p, w in edits.items() if key in p), None)
            if e is not None:
                e.setText(val)
            return e is not None
        ok_name = _set("贴片机", ds_name)
        combo = next((c for c in dlg.findChildren(QComboBox) if c.findData("adapt") >= 0), None)
        if combo is not None:
            combo.setCurrentIndex(combo.findData("adapt"))
            pump(200)
        ok_root = _set("D:/datasets", str(EXTRA).replace("\\", "/"))
        ok_ds = _set("树顶层分组名", CATEGORY)
        ck("新建数据源表单填写（名称/适配导入/根目录/分组名）",
           ok_name and combo is not None and ok_root and ok_ds,
           f"name={ok_name} combo={combo is not None} root={ok_root} dsname={ok_ds}")
        b = _btn(dlg, "创建并导入")
        ck("「创建并导入」按钮出现", b is not None)
        (b.click() if b else dlg.reject())
        wait_modal("数据集识别与确认", confirm_adapt, timeout_s=120, desc="适配确认")

    def confirm_adapt(dlg):
        fmt = getattr(dlg, "_fmt", "?")
        cats = getattr(dlg, "_cat_checks", None)
        if cats:
            for name, cb in cats.items():
                cb.setChecked(name == CATEGORY)
            detail = f"fmt={fmt} 品类候选={list(cats)}"
            ok = CATEGORY in cats
        else:
            ec = getattr(dlg, "edit_category", None)
            if ec is not None:
                ec.setText(CATEGORY)
            detail = f"fmt={fmt} 品类={ec.text() if ec else None}"
            ok = ec is not None
        ck("适配识别对话框给出 extra_part 品类", ok, detail)
        STATE["adapt_fmt"] = fmt
        b = _btn(dlg, "确认导入")
        (b.click() if b else dlg.reject())
        STATE["import_t0"] = time.time()
        later(500, wait_import)

    def find_ds():
        return next((d for d in cli.list_datasources() or [] if d.get("name") == ds_name), None)

    last = {"n": -1, "t": time.time()}

    def imported():
        ds = find_ds()
        if not ds:
            return False
        STATE["ds_id"] = ds["id"]
        g = cli.datasource_groups(ds["id"]) or {}
        n = int(((g.get("categories") or {}).get(CATEGORY) or {}).get("total") or 0)
        if n != last["n"]:
            last.update(n=n, t=time.time())
            return False
        return n > 0 and time.time() - last["t"] > 6   # 计数稳定 6s 视为导入完成

    def wait_import():
        def after():
            elapsed = time.time() - STATE.get("import_t0", time.time())
            n_src = sum(1 for _ in EXTRA.rglob("*.jpg"))
            ck("extra_part 导入完成且计数稳定", last["n"] > 0,
               f"导入 {last['n']} 张（源目录 jpg {n_src}），{elapsed:.0f}s，ds_id={STATE.get('ds_id')}")
            METRICS["import"] = {"n": last["n"], "src_jpg": n_src, "s": round(elapsed), "fmt": STATE.get("adapt_fmt")}
            later(300, lambda: s06c_prepare(window))
        poll_until("extra_part 导入", imported, after, timeout_s=300, interval_ms=2000)

    wait_modal("新建数据源", fill_create, timeout_s=20)
    later(0, dp._on_create_datasource)   # exec 阻塞在嵌套事件循环，定时器照常驱动


def s06c_prepare(window) -> None:
    """特征学习链路 2/4：模型管理页 选中品类 → 准备模型(fast)。"""
    print("\n== S06c 特征学习：准备模型 ==", flush=True)
    hub = window.model_page
    mp = _hub_page(hub, "模型管理")
    mp.reload()
    ds_id = STATE.get("ds_id")

    def find_item():
        it = mp.tree.invisibleRootItem()
        stack = [it.child(i) for i in range(it.childCount())]
        while stack:
            x = stack.pop()
            d = x.data(0, 0x0100)  # Qt.UserRole
            if isinstance(d, (tuple, list)) and d and d[0] == "category" and d[1] == ds_id and d[3] == CATEGORY:
                return x
            stack.extend(x.child(i) for i in range(x.childCount()))
        return None

    def selected():
        x = find_item()
        if x is None:
            return False
        mp.tree.setCurrentItem(x)
        pump(300)
        return bool(mp._cur) and mp._cur.get("category") == CATEGORY

    def after_select():
        ck("模型管理树选中 UI测试数据源/extra_part", bool(mp._cur) and mp._cur.get("category") == CATEGORY,
           str(mp._cur)[:120])
        if not mp._cur:
            later(300, lambda: s06d_activate(window))
            return
        pump(1500)
        ck("「准备模型」按钮可见", mp.btn_prepare.isVisible(), mp.btn_prepare.text())
        t0 = time.time()

        def operate(dlg):
            i = dlg.cmb_profile.findData("fast")
            if i >= 0:
                dlg.cmb_profile.setCurrentIndex(i)
            dlg._profile_touched = True
            dlg.chk_force.setChecked(True)
            pump(1500)   # 等 precheck 回填
            note(f"准备对话框预检: {dlg.lbl_status.text()[:160]}")
            dlg.btn_ok.click()

            def finished():
                return dlg._result is not None or "失败" in dlg.lbl_status.text()

            def done():
                el = time.time() - t0
                ok = dlg._result is not None and "准备完成" in dlg.lbl_status.text()
                ck("准备模型（fast）成功生成新版本", ok,
                   f"{el:.0f}s | {dlg.lbl_status.text()[:200].replace(chr(10), ' / ')}")
                METRICS["prepare"] = {"ok": ok, "s": round(el), "result": {k: v for k, v in (dlg._result or {}).items()
                                                                          if k in ("version", "n_normal", "n_defect")}}
                (dlg.accept() if ok else dlg.reject())
                later(800, lambda: s06d_activate(window))
            poll_until("准备模型任务", finished, done, timeout_s=1200, interval_ms=2000)

        wait_modal("准备模型", operate, timeout_s=30)
        later(0, mp.btn_prepare.click)

    poll_until("模型树出现 extra_part 节点", selected, after_select, timeout_s=60, interval_ms=1000)


def s06d_activate(window) -> None:
    """特征学习链路 3/4：激活最新版本（经激活门控）。"""
    print("\n== S06d 特征学习：激活模型 ==", flush=True)
    from app.engines.feature import get_engine
    hub = window.model_page
    mp = _hub_page(hub, "模型管理")

    def cur_ver():
        try:
            return get_engine().current_version(CATEGORY)
        except Exception:
            return None

    t_start = time.time()
    poll_until("版本列表加载", lambda: bool(getattr(mp, "_model_buttons", None)) or time.time() - t_start > 20,
               lambda: _s06d_go(window, mp, cur_ver), timeout_s=30, interval_ms=500)


def _s06d_go(window, mp, cur_ver) -> None:
    btns = getattr(mp, "_model_buttons", {}) or {}
    before = cur_ver()
    note(f"版本列表 {list(btns)}，当前激活 {before}")
    if not btns:
        ck("版本历史有可激活版本", False, "无版本（准备失败？）")
        later(300, lambda: s06e_feature_detect(window))
        return
    mid = max(btns)
    btns[mid].setChecked(True)
    pump(300)
    if not mp.btn_activate.isEnabled():
        note(f"最新模型(id={mid}, 版本={before}) 已是激活态（首次准备自动激活）")
        ck("extra_part 已有激活版本", bool(before), f"版本={before}")
        METRICS["active_version"] = before
        later(300, lambda: s06e_feature_detect(window))
        return
    flags = {"gate": False, "changed_at": None}

    def gate(dlg):
        flags["gate"] = True
        force = _btn(dlg, "仍要强制激活")
        from PySide6.QtWidgets import QLabel, QTextEdit
        txt = " ".join([w.text() for w in dlg.findChildren(QLabel)] +
                       [w.toPlainText() for w in dlg.findChildren(QTextEdit)]).replace("\n", " ")[:300]
        METRICS["activate_gate"] = "拒绝→强制" if force else "通过"
        if force:
            warn("激活门控拒绝新版本（测试中强制激活以继续链路）", txt)
            force.click()
        else:
            ck("激活门控通过", True, txt)
            dlg.accept()

    wait_modal("激活前自动验证", gate, timeout_s=300, desc="激活门控", optional=True,
               stop=lambda: flags.get("done", False))
    mp.btn_activate.click()

    def active():
        v = cur_ver()
        if not v or v == before:
            return False
        if flags["changed_at"] is None:
            flags["changed_at"] = time.time()
        # 通过型门控弹窗在切换后才出现：等它被处理，或切换后 5s 无模态框
        return QApplication.activeModalWidget() is None and (flags["gate"] or time.time() - flags["changed_at"] > 5)

    def after():
        flags["done"] = True
        v = cur_ver()
        ck("extra_part 激活新版本", bool(v) and v != before, f"{before} -> {v}")
        if not flags["gate"]:
            METRICS["activate_gate"] = "无门控报告（直接激活）"
        METRICS["active_version"] = cur_ver()
        later(300, lambda: s06e_feature_detect(window))
    poll_until("激活生效", active, after, timeout_s=300, interval_ms=1500)


def s06e_feature_detect(window) -> None:
    """特征学习链路 4/4：feature / dual 模板实测（模型就绪后门禁应解除）。"""
    print("\n== S06e 特征/双检模板检测 ==", flush=True)
    from app.inspect.service import InspectService
    svc = InspectService()
    import threading as _th
    _ort, _orf = svc.run_with_template, svc._run_feature
    def _tw(name, fn):
        def w(*a, **k):
            t = time.perf_counter(); r = fn(*a, **k)
            note(f"    [diag] {name} {time.perf_counter()-t:.2f}s threads={_th.active_count()}")
            return r
        return w
    svc.run_with_template, svc._run_feature = _tw("trad", _ort), _tw("feat", _orf)
    try:
        import torch, cv2 as _cv
        note(f"    [diag] torch_threads={torch.get_num_threads()} cv_threads={_cv.getNumThreads()} cuda={torch.cuda.is_available()}")
    except Exception as e:
        note(f"    [diag] {e}")
    try:
        from concurrent.futures import ThreadPoolExecutor as _TPE
        _temp = PAIRS[0][0]
        for _m in ("traditional", "dual"):
            _st, _tp = make_tpl(_temp, f"PROBE_diag_{_m}", PAIRS[0][3], PAIRS[0][4], _m, model_category=CATEGORY)
            _tp = _st.publish(_tp)
            for _i in range(2):
                t = time.perf_counter(); _ort(_tp, _temp, test_path=_temp, archive=False)
                a = time.perf_counter() - t
                with _TPE(1) as ex:
                    t = time.perf_counter(); ex.submit(_ort, _tp, _temp, test_path=_temp, archive=False).result()
                b = time.perf_counter() - t
                note(f"    [diag] tpl={_m} main={a:.2f}s worker={b:.2f}s")
            if _m == "traditional":
                import cProfile, pstats, io as _io
                pr = cProfile.Profile(); pr.enable()
                _ort(_tp, _temp, test_path=_temp, archive=False)
                pr.disable(); s = _io.StringIO()
                pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(25)
                note("    [diag-prof]\n" + s.getvalue()[-6000:])
            _st.delete(_tp.id)
    except Exception as e:
        note(f"    [diag] err {e}")
    res = {}
    for mode in ("feature", "dual"):
        store, tpl = make_tpl(_T0, f"PROBE_feat_{mode}", _X0, _Y0, mode, model_category=CATEGORY)
        blk = [i for i in store.publish_checklist(tpl) if i.startswith("[阻断]")]
        ck(f"{mode} 模板：模型就绪后发布检查无阻断", not blk, "；".join(blk)[:200])
        if blk:
            store.delete(tpl.id)
            continue
        tpl = store.publish(tpl)
        store.delete(tpl.id)
        rows = []
        # 每对样本单独建 PROBE 模板（传统分支需按多件位置框 ROI；特征分支整图推理）
        for k, (temp, img, stem, x, y) in enumerate(PAIRS):
            cases = [("多件", img, "NG"), ("自比", temp, "OK"),
                     ("亮度+5", synth(temp, f"f{k}_b5", bright=5), "OK"),
                     ("平移2px+σ2", synth(temp, f"f{k}_s2", dx=2, dy=1, sigma=2, seed=SEED + 20 + k), "OK")]
            st, tp = make_tpl(temp, f"PROBE_feat_{mode}_{k}", x, y, mode, model_category=CATEGORY)
            tp = st.publish(tp)
            for cname, path, exp in cases:
                t0 = time.time()
                try:
                    ds = svc.run_dual(tp, path, test_path=path)
                    got = ds.overall
                    trad = ds.traditional.overall if ds.traditional else "-"
                    f = ds.feature or {}
                    fdesc = (f"{f.get('decision')} score={f.get('score')} thr={f.get('threshold')}"
                             if f else f"无({ds.feature_error[:60]})")
                except Exception as exc:
                    got, trad, fdesc = "ERROR", "-", str(exc)[:120]
                el = time.time() - t0
                split = "train" if "train" in Path(temp).parts else "test"
                rows.append((f"{stem}({split})", cname, exp, got, trad, fdesc, round(el, 2)))
                note(f"  [{mode}][{stem}] {cname}: 期望={exp} 融合={got} 传统={trad} 特征={fdesc} {el:.2f}s")
            st.delete(tp.id)
        ng = [r for r in rows if r[2] == "NG"]
        ok = [r for r in rows if r[2] == "OK"]
        hit = sum(r[3] == "NG" for r in ng)
        fp = [r for r in ok if r[3] != "OK"]
        mx = max(r[6] for r in rows)
        res[mode] = {"recall": f"{hit}/{len(ng)}", "fp": f"{len(fp)}/{len(ok)}", "max_s": mx,
                     "rows": rows}
        ck(f"{mode}：多件召回", hit == len(ng), f"{hit}/{len(ng)}")
        ck(f"{mode}：良品/扰动良品无误检", not fp,
           f"误检 {len(fp)}/{len(ok)}: " + "；".join(f"{r[0]}/{r[1]}→{r[3]}" for r in fp))
        ck(f"{mode}：单图耗时 ≤1s", mx <= 1.0, f"最大 {mx:.2f}s")
    METRICS["feature"] = res
    later(300, lambda: s07_menus(window))


def s07_menus(window) -> None:
    print("\n== S07 菜单与设置对话框 ==", flush=True)
    from app.config import get_gate_settings
    before = get_gate_settings()

    # 系统设置对话框（exec 模态：先排一个定时器去点保存）
    def operate_dialog():
        dlg = QApplication.activeModalWidget()
        if dlg is None:
            QTimer.singleShot(200, operate_dialog)
            return
        from app.ui.settings_dialog import SettingsDialog
        ck("系统设置对话框弹出", isinstance(dlg, SettingsDialog), type(dlg).__name__)
        if isinstance(dlg, SettingsDialog):
            dlg.chk_gate.setChecked(not before["gate_enabled"])
            dlg.spin_max_diff.setValue(33)
            dlg.btn_save.click()
    QTimer.singleShot(400, operate_dialog)
    window._open_settings()
    pump(400)
    after = get_gate_settings()
    ck("设置保存生效（门禁开关翻转）", after["gate_enabled"] != before["gate_enabled"],
       f"{before} -> {after}")
    ck("设置保存生效（阈值=33）", after["size_gate_max_diff_px"] == 33)
    # 还原
    def revert_dialog():
        dlg = QApplication.activeModalWidget()
        if dlg is None:
            QTimer.singleShot(200, revert_dialog)
            return
        from app.ui.settings_dialog import SettingsDialog
        if isinstance(dlg, SettingsDialog):
            dlg.chk_gate.setChecked(before["gate_enabled"])
            dlg.spin_max_diff.setValue(before["size_gate_max_diff_px"])
            dlg.btn_save.click()
    QTimer.singleShot(400, revert_dialog)
    window._open_settings()
    pump(300)

    # 菜单动作（消息框已 patch，只验证不抛异常）
    menus = {a.text(): a for a in window.menuBar().actions()}
    ck("菜单栏含 设置/模板", "设置" in menus and "模板" in menus, "、".join(menus))
    try:
        window._show_adapter_errors()
        ck("查看算法适配器状态", True, DIALOGS[-1][:120] if DIALOGS else "")
    except Exception as exc:
        ck("查看算法适配器状态", False, str(exc)[:150])
    try:
        NEXT_FILE["dir"] = ""  # 取消选择，不改输出目录
        window._choose_outputs()
        ck("设置输出目录（取消路径）", True)
    except Exception as exc:
        ck("设置输出目录", False, str(exc)[:150])
    later(200, finish)


RECIPE_BACKUP: dict = {}


def _restore_recipe() -> None:
    gpath = RECIPE_BACKUP.get("path")
    if gpath is not None:
        if RECIPE_BACKUP.get("text") is not None:
            gpath.write_text(RECIPE_BACKUP["text"], encoding="utf-8")
        else:
            gpath.unlink(missing_ok=True)


def _cleanup() -> None:
    _restore_recipe()
    try:
        from app.template.store import TemplateStore
        store = TemplateStore()
        from app.recipe.param_store import ParamStore
        ta_root = Path(ParamStore().root) / "template_algorithm"
        for tpl in store.list_templates():
            if tpl.display_name.startswith(("UI测试_", "PROBE_")):
                store.delete(tpl.id)
                # store.delete 不清理调参页写入的模板专用参数，这里手动删
                for f in ta_root.glob(f"{tpl.id}__*.json"):
                    f.unlink(missing_ok=True)
    except Exception as exc:
        print(f"清理模板异常: {exc}", flush=True)
    try:
        from app.core.aoi_core_launcher import core_healthy
        if core_healthy("http://127.0.0.1:8017", timeout=1.5):
            import requests
            r = requests.get("http://127.0.0.1:8017/api/datasources", timeout=5)
            data = r.json() if r.ok else []
            items = data.get("items", []) if isinstance(data, dict) else data
            for ds in items or []:
                if str(ds.get("name", "")).startswith("UI测试_"):
                    requests.delete(f"http://127.0.0.1:8017/api/datasources/{ds['id']}", timeout=5)
    except Exception as exc:
        print(f"清理数据源异常: {exc}", flush=True)
    shutil.rmtree(TMP, ignore_errors=True)


def finish() -> None:
    _cleanup()
    print("\n================ 量化指标 ================", flush=True)
    print(f"SEED={SEED} PAIRS={[p[2] for p in PAIRS]}", flush=True)
    def _strip(v):
        if isinstance(v, dict):
            return {a: _strip(b) for a, b in v.items() if a != "rows"}
        return v
    for k, v in METRICS.items():
        print(f"  {k}: {_strip(v)}", flush=True)
    for mode, d in (METRICS.get("feature") or {}).items():
        for r in d.get("rows", []):
            print(f"    [{mode}] {r[0]} | {r[1]} | 期望{r[2]} 融合{r[3]} 传统{r[4]} | 特征 {r[5]} | {r[6]:.2f}s",
                  flush=True)
    for r in (METRICS.get("svc") or {}).get("rows", []):
        print(f"    {r[0]} | {r[1]} | 期望{r[2]} 实际{r[3]} | {r[5]:.2f}s | {r[4]}", flush=True)
    print("\n================ 测试汇总 ================", flush=True)
    fails = [i for i in ISSUES if i[0] == "FAIL"]
    warns = [i for i in ISSUES if i[0] == "WARN"]
    passes = [i for i in ISSUES if i[0] == "PASS"]
    print(f"PASS={len(passes)}  FAIL={len(fails)}  WARN={len(warns)}", flush=True)
    if fails:
        print("\n---- FAIL ----", flush=True)
        for _, name, detail in fails:
            print(f"❌ {name} | {detail}", flush=True)
    if warns:
        print("\n---- WARN ----", flush=True)
        for _, name, detail in warns:
            print(f"⚠️ {name} | {detail}", flush=True)
    print("\n---- 弹窗记录 ----", flush=True)
    for d in DIALOGS:
        print(f"  {d}", flush=True)
    print("==========================================", flush=True)
    os._exit(1 if fails else 0)


# ================= 入口 =================
def main() -> None:
    from app.core.single_instance import acquire_lock
    if not acquire_lock("pcbins_ui"):
        print("已有 PCB_Dual 实例在运行，请先关闭再测试", flush=True)
        os._exit(2)

    from app.config import ensure_dirs
    ensure_dirs()

    # 后端（复用 main.py 的逻辑）
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
    window.show()
    pump(500)

    later(300, lambda: s00_boot(window))
    QTimer.singleShot(40 * 60 * 1000, finish)  # 兜底 40 分钟（含模型准备）
    app.exec()


if __name__ == "__main__":
    main()
