"""数据增强配置页（§15.29：伪异常合成 / 图像预处理 / 训练增强）。

本页只涉及数据增强相关的三类配置（不包含学习率/轮数等训练参数）：
- 伪异常合成：准备模型时用已标注缺陷图移植到正常图（无缺陷图时回退 CutPaste/
  色斑），让判别头学会识别真实缺陷形态；
- 图像预处理：对训练与检测图片统一处理（灰度/CLAHE/中值去噪），两侧同口径；
- 训练增强：仅增加正常样本外观多样性（翻转/旋转/亮度），降低对轻微平移、
  光照差异的误报。

左：基底图选择（normal 图，预览用）
中：三类配置卡片 + 保存
右：伪异常合成预览
"""
from __future__ import annotations

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ui.api_client import ApiClient
from ui.pages.common import (
    _THUMB_CACHE,
    _THUMB_CACHE_MAX,
    attach_level_badge,
    load_thumb_async,
    make_banner,
    make_card,
    notify,
    run_async,
    thumb_url,
    warn,
)

# 伪异常合成方式（与 algo/slots/disc.py 的 synth_pseudo 对应）
METHODS = {
    "defect_transplant": "真实缺陷移植（按缺陷位置标注抠出缺陷区域贴入正常图）",
    "cutpaste": "CutPaste 切块贴回（从图中裁块翻转后贴回）",
    "color_blot": "色斑（椭圆区域颜色偏移）",
}


class _IconLoader(QThread):
    """后台加载列表图标。"""

    loaded = Signal(int, bytes)

    def __init__(self, client: ApiClient, row: int, url: str, parent=None):
        super().__init__(parent)
        self._client = client
        self._row = row
        self._url = url

    def run(self) -> None:
        data = _THUMB_CACHE.get(self._url)
        if data is None:
            data = self._client.fetch_bytes(self._url)
            if data:
                if len(_THUMB_CACHE) >= _THUMB_CACHE_MAX:
                    _THUMB_CACHE.pop(next(iter(_THUMB_CACHE)))
                _THUMB_CACHE[self._url] = data
        if data:
            self.loaded.emit(self._row, data)


class AugmentPage(QWidget):
    """数据增强配置页：伪异常合成 / 图像预处理 / 训练增强。"""

    def __init__(self, client: ApiClient, get_category,
                 task_monitor=None, parent: QWidget | None = None):
        super().__init__(parent)
        self._client = client
        self._get_category = get_category
        self._monitor = task_monitor
        self._images: list[dict] = []
        self._icon_loaders: list[QThread] = []
        self._page = 1
        self._page_size = 100
        self._total = 0
        self._total_pages = 1
        self._img_req = 0
        # 当前配置（pseudo/preprocess/enhance，来自 GET /api/augment/config）
        self._cfg: dict = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(8)

        top = QHBoxLayout()
        top.addWidget(make_banner("数据增强配置"))
        attach_level_badge(self, top)   # U-workorder：数据条件徽标
        tip = QLabel("配置伪异常合成、图像预处理与训练增强；"
                     "保存后下一次「准备模型」时生效")
        tip.setProperty("subtext", True)
        tip.setWordWrap(True)
        top.addWidget(tip)
        top.addStretch(1)
        btn_reload = QPushButton("刷新基底图")
        btn_reload.clicked.connect(self.reload_images)
        top.addWidget(btn_reload)
        root.addLayout(top)

        # 三栏：基底图 / 配置 / 预览
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_base_panel())
        splitter.addWidget(self._build_config_panel())
        splitter.addWidget(self._build_preview_panel())
        splitter.setSizes([280, 460, 420])
        root.addWidget(splitter, 1)

        self._client.error_occurred.connect(lambda m: warn(self, m))
        self.reload_images()
        self._load_config()

    # ══════════════════ 左：基底图选择 ══════════════════
    def _build_base_panel(self) -> QWidget:
        card = make_card()
        lay = QVBoxLayout(card)
        head = QLabel("基底图（normal 图，预览合成效果）")
        head.setProperty("heading", True)
        lay.addWidget(head)
        self.list_images = QListWidget()
        self.list_images.setViewMode(QListWidget.IconMode)
        self.list_images.setIconSize(self.list_images.iconSize().scaled(
            96, 96, Qt.KeepAspectRatio))
        self.list_images.setResizeMode(QListWidget.Adjust)
        self.list_images.setSelectionMode(QListWidget.ExtendedSelection)
        self.list_images.setSpacing(6)
        lay.addWidget(self.list_images, 1)
        pg = QHBoxLayout()
        self.lbl_base_count = QLabel("共 0 条")
        self.lbl_base_count.setProperty("subtext", True)
        pg.addWidget(self.lbl_base_count)
        pg.addStretch(1)
        self.btn_base_prev = QPushButton("上一页")
        self.btn_base_prev.setProperty("flat", True)
        self.btn_base_prev.clicked.connect(lambda: self._base_page(-1))
        self.lbl_base_page = QLabel("第 1 / 1 页")
        self.lbl_base_page.setProperty("subtext", True)
        self.btn_base_next = QPushButton("下一页")
        self.btn_base_next.setProperty("flat", True)
        self.btn_base_next.clicked.connect(lambda: self._base_page(1))
        pg.addWidget(self.btn_base_prev)
        pg.addWidget(self.lbl_base_page)
        pg.addWidget(self.btn_base_next)
        lay.addLayout(pg)
        return card

    def _base_page(self, d: int) -> None:
        self._page = max(1, min(self._page + d, self._total_pages))
        self.reload_images()

    def reload_images(self) -> None:
        cat = self._get_category() if self._get_category else ""
        self._img_req += 1
        rid = self._img_req
        run_async(self, lambda: self._client.list_images(
            category=cat, label="normal", page=self._page,
            page_size=self._page_size),
            lambda data, _rid=rid: self._fill_images(data, _rid))

    def _fill_images(self, data, rid: int | None = None) -> None:
        if rid is not None and rid != self._img_req:
            return
        items = (data or {}).get("items", [])
        self._total = int((data or {}).get("total") or 0)
        self._total_pages = max(
            1, (self._total + self._page_size - 1) // self._page_size)
        self._page = min(max(1, self._page), self._total_pages)
        self.lbl_base_count.setText(f"共 {self._total} 条")
        self.lbl_base_page.setText(f"第 {self._page} / {self._total_pages} 页")
        self.btn_base_prev.setEnabled(self._page > 1)
        self.btn_base_next.setEnabled(self._page < self._total_pages)
        self._images = items
        self.list_images.clear()
        if not items:
            QListWidgetItem("暂无正常样本\n请先在数据页导入", self.list_images)
            return
        for i, img in enumerate(items):
            name = str(img.get("path", "")).replace("\\", "/").split("/")[-1]
            item = QListWidgetItem(f"#{img.get('id')} {name[:18]}")
            item.setData(Qt.UserRole, img)
            self.list_images.addItem(item)
            loader = _IconLoader(
                self._client, i,
                thumb_url(self._client.file_url(img.get("path", ""))), self)
            loader.loaded.connect(self._set_icon)
            loader.finished.connect(
                lambda w=loader: self._forget_loader(w))
            self._icon_loaders.append(loader)
            loader.start()

    def _set_icon(self, row: int, data: bytes) -> None:
        item = self.list_images.item(row)
        if item is None:
            return
        pm = QPixmap()
        pm.loadFromData(data)
        if not pm.isNull():
            item.setIcon(QIcon(pm))

    def _forget_loader(self, w: QThread) -> None:
        if w in self._icon_loaders:
            self._icon_loaders.remove(w)

    def _selected_images(self) -> list[dict]:
        out = []
        for item in self.list_images.selectedItems():
            img = item.data(Qt.UserRole)
            if img:
                out.append(img)
        return out

    # ══════════════════ 中：三类配置 ══════════════════
    def _build_config_panel(self) -> QWidget:
        card = make_card()
        lay = QVBoxLayout(card)
        head = QLabel("数据增强配置（保存后下次准备模型生效）")
        head.setProperty("heading", True)
        lay.addWidget(head)

        self._method_checks: dict[str, QCheckBox] = {}
        self._build_pseudo_card(lay)
        self._build_preprocess_card(lay)
        self._build_enhance_card(lay)

        lay.addStretch(1)
        row = QHBoxLayout()
        row.addStretch(1)
        btn_save = QPushButton("保存配置")
        btn_save.setProperty("primary", True)
        btn_save.clicked.connect(self._on_save_config)
        row.addWidget(btn_save)
        btn_reset = QPushButton("重新读取")
        btn_reset.clicked.connect(self._load_config)
        row.addWidget(btn_reset)
        lay.addLayout(row)
        self.lbl_cfg_state = QLabel("")
        self.lbl_cfg_state.setProperty("subtext", True)
        self.lbl_cfg_state.setWordWrap(True)
        lay.addWidget(self.lbl_cfg_state)
        return card

    # ── 伪异常合成 ──
    def _build_pseudo_card(self, lay) -> None:
        title = QLabel("伪异常合成（准备模型时自动执行，不生成图片文件）")
        title.setProperty("subtext", True)
        title.setWordWrap(True)
        lay.addWidget(title)
        desc = QLabel("按缺陷位置标注把缺陷区域从缺陷图中抠出，随机翻转后贴到"
                      "正常图任意位置，让判别头学会识别真实缺陷形态；本品类"
                      "暂无缺陷标注或无位置标注时该方式自动回退到 CutPaste。")
        desc.setProperty("subtext", True)
        desc.setWordWrap(True)
        lay.addWidget(desc)
        for key, text in METHODS.items():
            chk = QCheckBox(text)
            self._method_checks[key] = chk
            lay.addWidget(chk)
        # 每图伪异常数 / 移植缩放范围 / 羽化
        self.spin_pseudo_per_image = QSpinBox()
        self.spin_pseudo_per_image.setRange(0, 64)
        lay.addLayout(self._kv("每图伪异常数", self.spin_pseudo_per_image))
        self.spin_scale_lo = QDoubleSpinBox()
        self.spin_scale_lo.setRange(0.05, 2.0)
        self.spin_scale_lo.setSingleStep(0.05)
        self.spin_scale_hi = QDoubleSpinBox()
        self.spin_scale_hi.setRange(0.05, 2.0)
        self.spin_scale_hi.setSingleStep(0.05)
        row = QHBoxLayout()
        row.addWidget(QLabel("缺陷块缩放范围"))
        row.addWidget(self.spin_scale_lo)
        row.addWidget(QLabel("~"))
        row.addWidget(self.spin_scale_hi)
        row.addStretch(1)
        lay.addLayout(row)
        self.chk_feather = QCheckBox("边缘羽化（避免硬边拼接假影）")
        lay.addWidget(self.chk_feather)

    # ── 图像预处理 ──
    def _build_preprocess_card(self, lay) -> None:
        title = QLabel("图像预处理（对训练与检测图片统一处理，改动后需重新准备模型）")
        title.setProperty("subtext", True)
        title.setWordWrap(True)
        lay.addWidget(title)
        self.chk_pre_enabled = QCheckBox("启用预处理")
        lay.addWidget(self.chk_pre_enabled)
        self.chk_pre_gray = QCheckBox("灰度化（忽略颜色信息，仅看结构）")
        lay.addWidget(self.chk_pre_gray)
        self.spin_pre_clahe = QDoubleSpinBox()
        self.spin_pre_clahe.setRange(0.0, 8.0)
        self.spin_pre_clahe.setSingleStep(0.5)
        lay.addLayout(self._kv(
            "CLAHE 对比度增强（0=关闭）", self.spin_pre_clahe))
        self.combo_pre_median = QComboBox()
        self.combo_pre_median.addItem("关闭", 0)
        self.combo_pre_median.addItem("中值 3×3", 3)
        self.combo_pre_median.addItem("中值 5×5", 5)
        lay.addLayout(self._kv("中值去噪", self.combo_pre_median))

    # ── 训练增强 ──
    def _build_enhance_card(self, lay) -> None:
        title = QLabel("训练增强（仅增加正常样本外观多样性，降低轻微平移/光照误报）")
        title.setProperty("subtext", True)
        title.setWordWrap(True)
        lay.addWidget(title)
        self.chk_en_flip = QCheckBox("水平翻转")
        lay.addWidget(self.chk_en_flip)
        self.chk_en_rot90 = QCheckBox("旋转 90°")
        lay.addWidget(self.chk_en_rot90)
        self.chk_en_brightness = QCheckBox("亮度 / 对比度扰动")
        lay.addWidget(self.chk_en_brightness)

    @staticmethod
    def _kv(label: str, edit: QWidget) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        row.addWidget(edit)
        row.addStretch(1)
        return row

    # ── 读配置 / 回填 ──
    def _load_config(self) -> None:
        def _fill(data) -> None:
            data = data or {}
            self._cfg = data
            pseudo = data.get("pseudo") or {}
            pre = data.get("preprocess") or {}
            enh = data.get("enhance") or {}
            # 伪异常
            methods = set(pseudo.get("methods") or [])
            for key, chk in self._method_checks.items():
                chk.setChecked(key in methods)
            self.spin_pseudo_per_image.setValue(int(
                pseudo.get("pseudo_per_image") or 8))
            ts = list(pseudo.get("transplant_scale") or [0.35, 0.9])
            self.spin_scale_lo.setValue(float(ts[0]) if ts else 0.35)
            self.spin_scale_hi.setValue(float(ts[1]) if len(ts) > 1 else 0.9)
            self.chk_feather.setChecked(bool(pseudo.get("feather", True)))
            # 预处理
            self.chk_pre_enabled.setChecked(bool(pre.get("enabled")))
            self.chk_pre_gray.setChecked(bool(pre.get("gray")))
            self.spin_pre_clahe.setValue(float(pre.get("clahe") or 0))
            med = int(pre.get("median") or 0)
            idx = self.combo_pre_median.findData(med)
            self.combo_pre_median.setCurrentIndex(max(0, idx))
            # 训练增强
            self.chk_en_flip.setChecked(bool(enh.get("flip")))
            self.chk_en_rot90.setChecked(bool(enh.get("rot90")))
            self.chk_en_brightness.setChecked(bool(enh.get("brightness")))
            note = data.get("note") or ""
            if note:
                self.lbl_cfg_state.setText(note)
        run_async(self, self._client.get_augment_config, _fill,
                  lambda err: warn(self, f"读取配置失败：{err}"))

    # ── 保存 ──
    def _collect_pseudo(self) -> dict:
        methods = [k for k, chk in self._method_checks.items()
                   if chk.isChecked()]
        if not methods:
            methods = ["cutpaste"]
        lo = min(self.spin_scale_lo.value(), self.spin_scale_hi.value())
        hi = max(self.spin_scale_lo.value(), self.spin_scale_hi.value())
        return {
            "methods": methods,
            "pseudo_per_image": self.spin_pseudo_per_image.value(),
            "transplant_scale": [round(lo, 3), round(hi, 3)],
            "feather": self.chk_feather.isChecked(),
        }

    def _collect_preprocess(self) -> dict:
        return {
            "enabled": self.chk_pre_enabled.isChecked(),
            "gray": self.chk_pre_gray.isChecked(),
            "clahe": self.spin_pre_clahe.value(),
            "median": int(self.combo_pre_median.currentData() or 0),
        }

    def _collect_enhance(self) -> dict:
        return {
            "flip": self.chk_en_flip.isChecked(),
            "rot90": self.chk_en_rot90.isChecked(),
            "brightness": self.chk_en_brightness.isChecked(),
        }

    def _on_save_config(self) -> None:
        pseudo = self._collect_pseudo()
        pre = self._collect_preprocess()
        enh = self._collect_enhance()

        def _done(res) -> None:
            notify(self, "数据增强配置已保存，将在下一次「准备模型」时生效")
            self._load_config()
        run_async(self, lambda: self._client.save_augment_config(
            pseudo=pseudo, preprocess=pre, enhance=enh), _done,
            lambda err: warn(self, f"保存失败：{err}"))

    # ══════════════════ 右：伪异常预览 ══════════════════
    def _build_preview_panel(self) -> QWidget:
        card = make_card()
        lay = QVBoxLayout(card)
        head = QLabel("伪异常合成预览")
        head.setProperty("heading", True)
        lay.addWidget(head)
        desc = QLabel("真实缺陷移植需当前品类已有带缺陷位置标注的缺陷图；"
                      "没有时该方式会自动回退为 CutPaste。")
        desc.setProperty("subtext", True)
        desc.setWordWrap(True)
        lay.addWidget(desc)

        row = QHBoxLayout()
        row.addWidget(QLabel("合成方式"))
        self.combo_method = QComboBox()
        for key, desc_text in METHODS.items():
            self.combo_method.addItem(desc_text, key)
        row.addWidget(self.combo_method, 1)
        lay.addLayout(row)

        grid_row = QHBoxLayout()
        self._pv_labels: dict[str, QLabel] = {}
        for i, (key, name) in enumerate(
                (("orig", "原图"), ("pseudo", "伪异常"), ("mask", "掩码"))):
            box = QVBoxLayout()
            cap = QLabel(name)
            cap.setAlignment(Qt.AlignCenter)
            cap.setProperty("subtext", True)
            box.addWidget(cap)
            lb = QLabel("—")
            lb.setAlignment(Qt.AlignCenter)
            lb.setMinimumSize(140, 140)
            lb.setStyleSheet(
                "background:#EDF1F7; border-radius:6px; color:#9CA3AF;")
            box.addWidget(lb)
            grid_row.addLayout(box, 1)
            self._pv_labels[key] = lb
        lay.addLayout(grid_row)

        btn_preview = QPushButton("预览合成")
        btn_preview.setProperty("primary", True)
        btn_preview.clicked.connect(self._on_preview)
        lay.addWidget(btn_preview, 0, Qt.AlignLeft)
        lay.addStretch(1)
        return card

    def _on_preview(self) -> None:
        imgs = self._selected_images()
        if not imgs:
            warn(self, "请先在左侧选择一张基底图")
            return
        method = self.combo_method.currentData() or "defect_transplant"
        img = imgs[0]

        def _fill(res) -> None:
            if not res:
                return
            load_thumb_async(self, self._client,
                             self._client.file_url(img.get("path", "")),
                             self._pv_labels["orig"], (180, 150))
            load_thumb_async(self, self._client,
                             self._client.file_url(res.get("image_path", "")),
                             self._pv_labels["pseudo"], (180, 150))
            load_thumb_async(self, self._client,
                             self._client.file_url(res.get("mask_path", "")),
                             self._pv_labels["mask"], (180, 150))

        run_async(self, lambda: self._client.pseudo_preview(
            img.get("id"), method), _fill)

    # ══════════════════ 通用 ══════════════════
    def reload(self) -> None:
        self.reload_images()
