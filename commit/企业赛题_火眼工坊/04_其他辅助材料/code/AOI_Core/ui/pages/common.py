"""页面公共辅助：异步执行、缩略图加载、内联提示横幅、空数据态。"""
from __future__ import annotations

import shiboken6
import threading
from PySide6.QtCore import Qt, QEvent, QThread, QTimer, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QFrame,
                               QHBoxLayout, QLabel, QPushButton, QVBoxLayout,
                               QWidget)

from ui.api_client import ApiClient, ApiWorker

# U-opsflow(2026-08-26)：缩略图并发下载上限（连接池 + 线程双保险）
_THUMB_CONCURRENCY = threading.BoundedSemaphore(8)
# 缩略图全局缓存（url -> bytes）：页面切换/查询/工单联动会反复触发
# reload_images，同一批图若不缓存会每轮重新下载（"数据管理页一直在
# 加载图片"）。FIFO 裁剪上限防内存膨胀。
_THUMB_CACHE: dict[str, bytes] = {}
_THUMB_CACHE_MAX = 3000


def thumb_url(url: str, size: int = 128) -> str:
    """把 /api/file?path= 原图 URL 转成缩略图 URL（幂等：已带 thumb 不改）。

    缩略图列表只拉小 JPEG（最长边 ≤size），不再整图下载原图。
    """
    if not url or "thumb=" in url:
        return url
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}thumb={size}"

# 内联横幅配色（error/info/ok）
_TOAST_BG = {"error": "#FEE2E2", "info": "#DBEAFE", "ok": "#DCFCE7"}
_TOAST_FG = {"error": "#B91C1C", "info": "#1D4ED8", "ok": "#15803D"}


class _Toast(QWidget):
    """页面内提示横幅（非模态，悬浮在顶层窗口顶部，3.5s 自动消失）。

    替代 QMessageBox 弹窗：操作结果（错误/信息/成功）以页面内横幅展示，
    不打断用户操作流。跟随父窗口移动/缩放。
    """

    def __init__(self, top: QWidget, text: str, kind: str = "info"):
        super().__init__(top)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(
            f"background:{_TOAST_BG[kind]}; color:{_TOAST_FG[kind]};"
            "border-radius:6px; padding:8px 12px;")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 6)
        self.lbl = QLabel(text)
        self.lbl.setWordWrap(True)
        lay.addWidget(self.lbl)
        self._kind = kind
        top.installEventFilter(self)
        self.adjustSize()
        self._reposition()
        self.show()
        self.raise_()
        QTimer.singleShot(3500, self._auto_hide)

    def eventFilter(self, obj, ev):
        if obj is self.parentWidget() and ev.type() in (QEvent.Resize, QEvent.Move):
            self._reposition()
        return False

    def _reposition(self):
        p = self.parentWidget()
        if p is None or not shiboken6.isValid(p):
            return
        w = min(max(self.sizeHint().width(), 260), max(200, p.width() - 24))
        self.setFixedWidth(w)
        self.adjustSize()
        self.move(max(12, (p.width() - self.width()) // 2), 10)

    def _auto_hide(self):
        if shiboken6.isValid(self):
            self.hide()
            self.deleteLater()


def show_inline(parent: QWidget, text: str, kind: str = "info") -> None:
    """在顶层窗口顶部显示内联横幅（error/info/ok）。"""
    top = parent.window() if parent is not None else None
    if top is None or not shiboken6.isValid(top):
        return
    _Toast(top, text, kind)


def warn(parent: QWidget, title: str, message: str | None = None) -> None:
    """统一错误提示——页面内红色横幅（替代 QMessageBox 弹窗）。

    兼容两种调用：warn(page, msg) 或 warn(page, title, msg)。"""
    if message is None:
        message = title
    show_inline(parent, message, "error")


def notify(parent: QWidget, title: str, message: str | None = None,
           ask: bool = False) -> bool:
    """统一信息/成功提示——页面内蓝色横幅（替代 QMessageBox.information）。

    兼容两种调用：notify(page, msg) 或 notify(page, title, msg)。
    ask=True 时退化为确认对话框（需用户点确认才继续），返回是否确认。"""
    if message is None:
        message = title
    if ask:
        from PySide6.QtWidgets import QMessageBox
        btn = QMessageBox.question(parent.window(), title, message)
        return btn == QMessageBox.Yes
    show_inline(parent, message, "info")
    return True


def run_async(parent: QWidget, fn, on_ok=None, on_fail=None) -> ApiWorker:
    """在后台线程执行 fn（通常是 API 调用），结果经回调返回 UI 线程。"""
    if not hasattr(parent, "_async_workers"):
        parent._async_workers = []  # type: ignore[attr-defined]
    worker = ApiWorker(fn, parent)
    if on_ok is not None:
        worker.succeeded.connect(on_ok)
    if on_fail is not None:
        worker.failed.connect(on_fail)
    workers = parent._async_workers  # type: ignore[attr-defined]
    workers.append(worker)
    worker.finished.connect(lambda w=worker: _forget(workers, w))
    worker.start()
    return worker


def _forget(workers: list, w: QThread) -> None:
    if w in workers:
        workers.remove(w)
    w.deleteLater()


def make_banner(text: str) -> QLabel:
    """赛题问题横幅标签（浅蓝底圆角）。"""
    lb = QLabel(text)
    lb.setProperty("banner", True)
    return lb


# U-workorder v2（前端反馈 2026-08-27）：数据条件不是平铺五选一，
# 而是「标注档位（三档递进，选高档兼容低档）＋ 两个可叠加开关」。
# UI 一律用中文叫法，不显示 L 代号（tooltip 里可提）。
TIER_CN = {
    "L0": "仅正常图",
    "L1a": "图像级标注",
    "L1b": "缺陷位置标注",
}
TIER_ORDER = ["L0", "L1a", "L1b"]
TIER_DESC = {
    "L0": "手头只有正常图（无缺陷样本）",
    "L1a": "正常图 + 整图 OK/NG 标注（多少张是缺陷），兼容仅正常图",
    "L1b": "还标了缺陷位置（框/掩码），兼容图像级标注与仅正常图",
}
SWITCH_CN = {"per_category": "品类独立", "has_template": "模板比对"}
SWITCH_DESC = {
    "per_category": "数据已按品类分开，每品类独立建模",
    "has_template": "提供标准模板图，可与模板逐位置比对",
}

# 批次「可支撑层级」展示（后端 layers 仍沿用五档词汇，仅用于数据页展示）
LAYER_CN = {
    "L0": "仅正常图",
    "L1a": "图像级标注",
    "L1b": "缺陷位置标注",
    "L2": "品类独立",
    "L3": "模板比对",
}

# 评分依据英文缩写 → 中文（用户可读；2026-08-29 从 monitor_page 上收共用）
SLOT_CN = {
    "sem": "语义相似", "disc": "判别器", "shead": "伪异常判别",
    "blob": "亮度结构", "trad": "传统特征", "layout": "布局逻辑",
    "inp": "内在原型", "tpl": "模板差分", "color": "色彩",
}


def slot_text(slots: dict, limit: int = 0) -> str:
    """{sem:0.92, disc:0.11} → '语义相似=0.92  判别器=0.11'（中文化）。

    limit>0 时只保留分数最高的前 limit 项。
    """
    if not isinstance(slots, dict) or not slots:
        return "-"
    items = [(k, float(v)) for k, v in slots.items()
             if isinstance(v, (int, float))]
    if limit > 0:
        items = sorted(items, key=lambda kv: -kv[1])[:limit]
    else:
        items = sorted(items)
    return "  ".join(f"{SLOT_CN.get(k, k)}={v:.2f}" for k, v in items) or "-"


def version_text(v) -> str:
    """版本号统一显示 'v16'：DB 已带 v 前缀的不重复补（防 vv16）。"""
    s = str(v if v is not None else "").strip()
    if not s:
        return "-"
    return s if s.lower().startswith("v") else f"v{s}"


def condition_text(tier: str | None, per_category: bool = False,
                   has_template: bool = False) -> str:
    """工单数据条件 -> 中文串（如「图像级标注＋品类独立」）。"""
    parts = [TIER_CN.get(str(tier or ""), str(tier or ""))]
    if per_category:
        parts.append(SWITCH_CN["per_category"])
    if has_template:
        parts.append(SWITCH_CN["has_template"])
    return "＋".join(p for p in parts if p)


def level_name(code: str | None) -> str:
    """条件/层级代号 -> 中文名；未知/None 返回空。"""
    s = str(code or "")
    return TIER_CN.get(s) or LAYER_CN.get(s, "")


class ConditionEditor(QWidget):
    """数据条件编辑器：标注档位（三档递进）＋ 两个可叠加开关。

    供「建工单 / 改条件 / 设置向导」复用；UI 只出中文叫法。
    value() -> {"label_tier", "per_category", "has_template"}
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        self.combo_tier = QComboBox()
        for code in TIER_ORDER:
            self.combo_tier.addItem(TIER_CN[code], code)
            self.combo_tier.setItemData(
                self.combo_tier.count() - 1, TIER_DESC[code], Qt.ToolTipRole)
        self.combo_tier.setToolTip(
            "三档递进：选高档兼容低档（如「缺陷位置标注」天然包含图像级标注）")
        form.addRow("标注档位", self.combo_tier)

        sw = QVBoxLayout()
        sw.setContentsMargins(0, 0, 0, 0)
        self.chk_per_category = QCheckBox(SWITCH_CN["per_category"])
        self.chk_per_category.setToolTip(SWITCH_DESC["per_category"])
        self.chk_template = QCheckBox(SWITCH_CN["has_template"])
        self.chk_template.setToolTip(SWITCH_DESC["has_template"])
        sw.addWidget(self.chk_per_category)
        sw.addWidget(self.chk_template)
        form.addRow("叠加开关", sw)

        hint = QLabel("开关可与任意档位叠加；导入数据后系统会体检并自动校正")
        hint.setProperty("subtext", True)
        form.addRow("", hint)

    def value(self) -> dict:
        return {
            "label_tier": str(self.combo_tier.currentData() or "L1a"),
            "per_category": self.chk_per_category.isChecked(),
            "has_template": self.chk_template.isChecked(),
        }

    def set_value(self, tier: str | None = None,
                  per_category: bool | None = None,
                  has_template: bool | None = None) -> None:
        if tier is not None:
            idx = self.combo_tier.findData(tier)
            if idx >= 0:
                self.combo_tier.setCurrentIndex(idx)
        if per_category is not None:
            self.chk_per_category.setChecked(bool(per_category))
        if has_template is not None:
            self.chk_template.setChecked(bool(has_template))


def make_level_badge() -> QLabel:
    """数据条件徽标（显示当前工单的数据条件；未指定时灰色提示）。

    用 update_level_badge(badge, get_level) 刷新。
    """
    lb = QLabel("数据条件：-")
    lb.setProperty("badge", "muted")
    lb.setToolTip("当前工单声明的数据条件（在「工作台」创建/编辑工单时指定）。\n"
                  "条件 = 标注档位（仅正常图 / 图像级标注 / 缺陷位置标注）\n"
                  "＋ 可叠加开关（品类独立 / 模板比对）")
    return lb


def update_level_badge(badge: QLabel, get_level) -> None:
    """按 get_level() 刷新徽标：返回值可为条件中文串、工单 dict 或旧代号。"""
    try:
        val = get_level() if callable(get_level) else None
    except Exception:  # noqa: BLE001
        val = None
    if isinstance(val, dict):  # 工单 dict -> 条件中文串
        # 工单 item 的条件嵌套在 conditions（服务端已算好 conditions_cn）；
        # 兼容直接传裸 conditions dict 的旧用法。
        cn = str(val.get("conditions_cn") or "")
        cond = val.get("conditions")
        if not isinstance(cond, dict):
            cond = val
        val = cn or condition_text(cond.get("label_tier"),
                                   bool(cond.get("per_category")),
                                   bool(cond.get("has_template")))
    elif val is not None:
        val = level_name(val) or str(val)  # 旧代号映射；已是中文串则原样显示
    if val:
        badge.setText(f"数据条件：{val}")
        badge.setProperty("badge", "info")
    else:
        badge.setText("数据条件：未指定")
        badge.setProperty("badge", "warning")
    badge.style().unpolish(badge)
    badge.style().polish(badge)


def attach_level_badge(page: QWidget, layout) -> QLabel:
    """把数据条件徽标挂到页面顶部布局，并注册 page.update_level_badge()。

    数据条件来源 page._get_level（由主窗口注入当前工单条件）。
    """
    badge = make_level_badge()
    layout.addWidget(badge)
    page._level_badge = badge  # type: ignore[attr-defined]

    def _update() -> None:
        update_level_badge(badge, getattr(page, "_get_level", None))
    page.update_level_badge = _update  # type: ignore[attr-defined]
    _update()
    return badge


class PagerBar(QWidget):
    """统一分页条：上一页/下一页 + 页码/总数 + 每页条数。

    所有数据项列表统一接入（配合服务端分页），避免数据堆积后一次加载
    几百条导致列表卡顿、看不过来：

        pager = PagerBar(page_size=20)
        pager.page_changed.connect(lambda _p: self.reload_xxx())
        # 数据返回后：pager.set_total(data.get("total", 0))
        # 筛选条件变化：pager.reset()（回第 1 页，不发信号）再 reload
        # 行内删除/作废后当前页空：pager.goto(pager.page - 1)
    """

    page_changed = Signal(int)

    def __init__(self, page_size: int = 20,
                 page_sizes: tuple = (20, 50, 100),
                 parent: QWidget | None = None):
        super().__init__(parent)
        self._page = 1
        self._total = 0
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.btn_prev = QPushButton("上一页")
        self.btn_prev.setProperty("flat", True)
        self.btn_prev.clicked.connect(lambda: self.goto(self._page - 1))
        lay.addWidget(self.btn_prev)
        self.lbl = QLabel("")
        self.lbl.setProperty("subtext", True)
        self.lbl.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.lbl, 1)
        self.btn_next = QPushButton("下一页")
        self.btn_next.setProperty("flat", True)
        self.btn_next.clicked.connect(lambda: self.goto(self._page + 1))
        lay.addWidget(self.btn_next)
        lay.addWidget(QLabel("每页"))
        self.combo_size = QComboBox()
        for s in page_sizes:
            self.combo_size.addItem(str(s), s)
        idx = self.combo_size.findData(page_size)
        self.combo_size.setCurrentIndex(max(0, idx))
        self.combo_size.currentIndexChanged.connect(self._on_size)
        lay.addWidget(self.combo_size)
        self._refresh()

    @property
    def page(self) -> int:
        return self._page

    @property
    def page_size(self) -> int:
        return int(self.combo_size.currentData())

    def _max_page(self) -> int:
        return max(1, (self._total + self.page_size - 1) // self.page_size)

    def goto(self, page: int) -> None:
        """翻页（越界自动夹取），页码变化时发 page_changed。"""
        page = min(max(1, page), self._max_page())
        if page != self._page:
            self._page = page
            self._refresh()
            self.page_changed.emit(self._page)

    def reset(self) -> None:
        """筛选条件变化时回第 1 页（不发信号，调用方随后自行 reload）。"""
        self._page = 1
        self._refresh()

    def set_total(self, total: int) -> None:
        """数据返回后更新总数；当前页超出新范围时自动回落。"""
        self._total = max(0, int(total or 0))
        if self._page > self._max_page():
            self._page = self._max_page()
        self._refresh()

    def _on_size(self, _i: int) -> None:
        # 每页条数变化必须重拉第一页
        self._page = 1
        self._refresh()
        self.page_changed.emit(1)

    def _refresh(self) -> None:
        self.lbl.setText(
            f"第 {self._page} / {self._max_page()} 页 · 共 {self._total} 条")
        self.btn_prev.setEnabled(self._page > 1)
        self.btn_next.setEnabled(self._page < self._max_page())


def make_card() -> QFrame:
    """白色圆角卡片容器。"""
    card = QFrame()
    card.setProperty("card", True)
    return card


def make_empty_hint(text: str = "暂无数据") -> QLabel:
    """空数据态友好提示标签。"""
    lb = QLabel(text)
    lb.setAlignment(Qt.AlignCenter)
    lb.setProperty("subtext", True)
    lb.setMinimumHeight(60)
    return lb


class _ImageLoader(QThread):
    """后台下载图片字节。"""

    loaded = Signal(bytes)

    def __init__(self, client: ApiClient, url: str, parent=None):
        super().__init__(parent)
        self._client = client
        self._url = url

    def run(self) -> None:
        # U-opsflow(2026-08-26)：页面表格每行一个 loader 并发下载缩略图，
        # 无上限会把连接池打满（urllib3 刷屏）+ 线程堆积。全局信号量限 8 并发。
        with _THUMB_CONCURRENCY:
            data = _THUMB_CACHE.get(self._url)
            if data is None:
                data = self._client.fetch_bytes(self._url)
                if data:
                    # 缓存写回（GIL 下 dict 写原子）；FIFO 裁剪
                    if len(_THUMB_CACHE) >= _THUMB_CACHE_MAX:
                        _THUMB_CACHE.pop(next(iter(_THUMB_CACHE)))
                    _THUMB_CACHE[self._url] = data
            if data:
                self.loaded.emit(data)


def _apply_thumb(label: QLabel, data: bytes,
                 size: tuple[int, int]) -> None:
    """把缩略图字节缩放到 QLabel（label 可能已被销毁，需校验）。"""
    if not shiboken6.isValid(label):
        return
    pm = QPixmap()
    pm.loadFromData(data)
    if pm.isNull():
        label.setText("加载失败")
        return
    label.setPixmap(pm.scaled(size[0], size[1], Qt.KeepAspectRatio,
                              Qt.SmoothTransformation))


def load_thumb_async(parent: QWidget, client: ApiClient, url: str,
                     label: QLabel, size: tuple[int, int] = (96, 64)) -> None:
    """异步加载缩略图到 QLabel（保持比例缩放居中）。

    - 命中全局缓存直接同步显示，不重复起下载线程；
    - URL 自动转缩略图接口（thumb=128），不再整图下载原图。
    """
    url = thumb_url(url)
    if not url:
        label.setText("无图")
        return
    cached = _THUMB_CACHE.get(url)
    if cached is not None:
        _apply_thumb(label, cached, size)
        return
    if not hasattr(parent, "_thumb_loaders"):
        parent._thumb_loaders = []  # type: ignore[attr-defined]

    def _on_loaded(data: bytes, lb=label, sz=size) -> None:
        _apply_thumb(lb, data, sz)

    loader = _ImageLoader(client, url, parent)
    loader.loaded.connect(_on_loaded)
    loaders = parent._thumb_loaders  # type: ignore[attr-defined]
    loaders.append(loader)
    loader.finished.connect(lambda w=loader: _forget(loaders, w))
    loader.start()


def make_thumb_cell(text: str = "加载中…") -> QLabel:
    """表格缩略图单元格标签。"""
    lb = QLabel(text)
    lb.setAlignment(Qt.AlignCenter)
    lb.setMinimumSize(96, 60)
    lb.setStyleSheet("color:#9CA3AF; background:transparent;")
    return lb
