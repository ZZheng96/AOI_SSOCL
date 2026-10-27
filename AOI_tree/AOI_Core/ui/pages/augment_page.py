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

from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QScrollArea,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
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
    make_card_title,
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

# 批次节点展开时懒加载缩略图的每批上限（防大批次卡顿）
_TREE_ICON_CAP = 200


class _TreeIconLoader(QThread):
    """后台加载树节点图标（携带 QTreeWidgetItem 指针回主线程）。"""

    loaded = Signal(object, bytes)

    def __init__(self, client: ApiClient, item, url: str, parent=None):
        super().__init__(parent)
        self._client = client
        self._item = item
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
            self.loaded.emit(self._item, data)


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
        # 2026-10-14（前端反馈）：这行引导文字不该换行——开 wordWrap 会让 QLabel 的
        # sizeHint 收窄、放得下也折成两行；关掉后保持单行，右侧留白交给 stretch
        tip.setWordWrap(False)
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

    # ══════════════════ 左：基底图选择（树状：数据源→品类→批次）══════════════════
    def _build_base_panel(self) -> QWidget:
        card = make_card()
        lay = QVBoxLayout(card)
        lay.addWidget(make_card_title(
            "基底图（normal 图，预览合成效果；按 数据源→品类→批次 分组）"))
        # 树状管理（2026-10-09）：平铺缩略图墙过于杂乱，改为树；
        # 缩略图在批次节点展开时懒加载
        self.tree_base = QTreeWidget()
        self.tree_base.setHeaderLabel("基底图")
        self.tree_base.setIconSize(self.tree_base.iconSize().scaled(
            48, 48, Qt.KeepAspectRatio))
        self.tree_base.setSelectionMode(QTreeWidget.ExtendedSelection)
        self.tree_base.itemExpanded.connect(self._on_node_expanded)
        lay.addWidget(self.tree_base, 1)
        self.lbl_base_count = QLabel("共 0 条")
        self.lbl_base_count.setProperty("subtext", True)
        lay.addWidget(self.lbl_base_count)
        return card

    def reload_images(self) -> None:
        """加载树骨架：数据源 + 批次 + 正常图一次性拉取后客户端分组。"""
        cat = self._get_category() if self._get_category else ""
        self._img_req += 1
        rid = self._img_req

        def _fetch() -> dict:
            images: list[dict] = []
            page = 1
            while True:   # 分页拉全（cap 5000 防失控）
                data = self._client.list_images(
                    category=cat, label="normal", page=page, page_size=500)
                items = (data or {}).get("items") or []
                images.extend(items)
                total = int((data or {}).get("total") or 0)
                if len(images) >= total or not items or len(images) >= 5000:
                    break
                page += 1
            return {"images": images,
                    "datasets": (self._client.list_datasets() or {}).get("items") or [],
                    "sources": self._client.list_datasources() or []}

        run_async(self, _fetch,
                  lambda data, _rid=rid: self._fill_tree(data, _rid))

    def _fill_tree(self, data, rid: int | None = None) -> None:
        if rid is not None and rid != self._img_req:
            return
        images = data.get("images") or []
        datasets = {int(d.get("id")): d for d in (data.get("datasets") or [])
                    if d.get("id") is not None}
        sources = {int(s.get("id")): s for s in (data.get("sources") or [])
                   if s.get("id") is not None}
        self._images = images
        self._feature_datasets = datasets
        self.lbl_base_count.setText(f"共 {len(images)} 条")
        self.tree_base.clear()
        if not images:
            QTreeWidgetItem(self.tree_base, ["暂无正常样本：请先在数据页导入"])
            return
        # 分组：数据源 → 品类 → 批次 → 图片
        tree: dict = {}
        for img in images:
            ds = datasets.get(img.get("dataset_id") or -1) or {}
            src = sources.get(ds.get("datasource_id") or -1) or {}
            src_name = str(src.get("name") or "未挂数据源")
            cat = str(img.get("category") or "通用")
            batch = str(ds.get("dataset_name") or ds.get("name")
                        or (f"批次{ds.get('id')}" if ds else "未分组"))
            tree.setdefault(src_name, {}).setdefault(cat, {}).setdefault(
                batch, []).append(img)
        for src_name in sorted(tree):
            src_node = QTreeWidgetItem(self.tree_base, [f"📁 {src_name}"])
            src_node.setFlags(src_node.flags() & ~Qt.ItemIsSelectable)
            for cat in sorted(tree[src_name]):
                cat_node = QTreeWidgetItem(src_node, [cat])
                cat_node.setFlags(cat_node.flags() & ~Qt.ItemIsSelectable)
                for batch in sorted(tree[src_name][cat]):
                    imgs = tree[src_name][cat][batch]
                    b_node = QTreeWidgetItem(
                        cat_node, [f"{batch}（{len(imgs)} 张）"])
                    b_node.setFlags(b_node.flags() & ~Qt.ItemIsSelectable)
                    b_node.setData(0, Qt.UserRole + 1, False)  # 未加载图标
                    for img in imgs:
                        name = str(img.get("path", "")).replace(
                            "\\", "/").split("/")[-1]
                        leaf = QTreeWidgetItem(
                            b_node, [f"#{img.get('id')} {name[:18]}"])
                        leaf.setData(0, Qt.UserRole, img)
        self.tree_base.expandToDepth(0)   # 默认只展开数据源层

    def _on_node_expanded(self, node: QTreeWidgetItem) -> None:
        """批次节点展开时懒加载其下图片缩略图（每批上限 _TREE_ICON_CAP）。"""
        if node.data(0, Qt.UserRole + 1):
            return
        node.setData(0, Qt.UserRole + 1, True)
        n = 0
        for i in range(node.childCount()):
            leaf = node.child(i)
            img = leaf.data(0, Qt.UserRole)
            if not img:
                continue
            if n >= _TREE_ICON_CAP:
                break
            n += 1
            loader = _TreeIconLoader(
                self._client, leaf,
                thumb_url(self._client.file_url(img.get("path", ""))), self)
            loader.loaded.connect(self._set_icon)
            loader.finished.connect(
                lambda w=loader: self._forget_loader(w))
            self._icon_loaders.append(loader)
            loader.start()

    def _set_icon(self, item: QTreeWidgetItem, data: bytes) -> None:
        pm = QPixmap()
        pm.loadFromData(data)
        if not pm.isNull():
            item.setIcon(0, QIcon(pm))

    def _forget_loader(self, w: QThread) -> None:
        if w in self._icon_loaders:
            self._icon_loaders.remove(w)

    def _selected_images(self) -> list[dict]:
        out = []
        for item in self.tree_base.selectedItems():
            img = item.data(0, Qt.UserRole)
            if img:
                out.append(img)
        return out

    # ══════════════════ 中：三类配置 ══════════════════
    def _build_config_panel(self) -> QWidget:
        card = make_card()
        lay = QVBoxLayout(card)
        lay.addWidget(make_card_title("数据增强配置（保存后下次准备模型生效）"))

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        content = QWidget()
        options = QVBoxLayout(content)
        options.setContentsMargins(0, 0, 0, 0)
        self._method_checks: dict[str, QCheckBox] = {}
        self._build_pseudo_card(options)
        self._build_preprocess_card(options)
        self._build_enhance_card(options)
        note = QLabel("效果需在同一评估集上对比模型版本，测量检出率与误报率；"
                      "右侧可启动真实单因素同集消融（隔离训练，不激活模型）。传统特征三维投影"
                      "仅用于样本分析，不代表生产模型效果。")
        note.setProperty("subtext", True)
        note.setWordWrap(True)
        options.addWidget(note)
        options.addStretch(1)
        scroll.setWidget(content)
        lay.addWidget(scroll, 1)
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
                      "暂无缺陷标注或无位置标注时该方式自动回退到 CutPaste。"
                      "伪异常训练只影响已启用的 disc 判别头；特征提取器冻结，"
                      "不做特征向量拉远。")
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
        self.spin_pseudo_per_image.setToolTip(
            "控制训练时每张正常图的伪异常数量；0 不合成。过多会增加准备耗时，"
            "也可能偏向合成缺陷。右侧预览始终只生成一张，不受数量控制。")
        lay.addLayout(self._kv("每图伪异常数", self.spin_pseudo_per_image))
        self.spin_scale_lo = QDoubleSpinBox()
        self.spin_scale_lo.setRange(0.05, 2.0)
        self.spin_scale_lo.setSingleStep(0.05)
        self.spin_scale_hi = QDoubleSpinBox()
        self.spin_scale_hi.setRange(0.05, 2.0)
        self.spin_scale_hi.setSingleStep(0.05)
        for spin, value in ((self.spin_scale_lo, 0.1), (self.spin_scale_hi, 0.4)):
            spin.setValue(value)
            spin.setEnabled(False)
            spin.setToolTip("当前算法不读取 transplant_scale：缺陷块长边固定为"
                            "基底短边的 10%~40%，此控件禁用且不提交。")
        row = QHBoxLayout()
        row.addWidget(QLabel("缺陷块缩放范围"))
        row.addWidget(self.spin_scale_lo)
        row.addWidget(QLabel("~"))
        row.addWidget(self.spin_scale_hi)
        row.addStretch(1)
        lay.addLayout(row)
        fixed = QLabel("缺陷块长边固定为基底短边的 10%~40%（缩放配置不生效）")
        fixed.setWordWrap(True)
        fixed.setProperty("subtext", True)
        lay.addWidget(fixed)
        self.chk_feather = QCheckBox("边缘羽化（避免硬边拼接假影）")
        self.chk_feather.setToolTip("仅真实缺陷移植使用；柔化拼接边缘，但可能淡化细小缺陷。"
                                    "无有效移植源回退 CutPaste 时不生效。")
        lay.addWidget(self.chk_feather)

    # ── 图像预处理 ──
    def _build_preprocess_card(self, lay) -> None:
        title = QLabel("图像预处理（对训练与检测图片统一处理，改动后需重新准备模型）")
        title.setProperty("subtext", True)
        title.setWordWrap(True)
        lay.addWidget(title)
        self.chk_pre_enabled = QCheckBox("启用预处理")
        self.chk_pre_enabled.setToolTip("训练与检测必须使用一致配置；修改后需要重新准备模型。")
        lay.addWidget(self.chk_pre_enabled)
        self.chk_pre_gray = QCheckBox("灰度化（忽略颜色信息，仅看结构）")
        self.chk_pre_gray.setToolTip("可减弱无关颜色变化，但会丢失色差、变色等缺陷线索。")
        lay.addWidget(self.chk_pre_gray)
        self.spin_pre_clahe = QDoubleSpinBox()
        self.spin_pre_clahe.setRange(0.0, 8.0)
        self.spin_pre_clahe.setSingleStep(0.5)
        self.spin_pre_clahe.setToolTip("0 关闭；增强局部对比度可能改善光照不均，"
                                     "也可能放大噪点、改变缺陷外观，引起误报。")
        lay.addLayout(self._kv(
            "CLAHE 对比度增强（0=关闭）", self.spin_pre_clahe))
        self.combo_pre_median = QComboBox()
        self.combo_pre_median.addItem("关闭", 0)
        self.combo_pre_median.addItem("中值 3×3", 3)
        self.combo_pre_median.addItem("中值 5×5", 5)
        self.combo_pre_median.setToolTip("用于抑制椒盐噪声；较大核可能抹除细划痕、微小缺陷和边缘。")
        lay.addLayout(self._kv("中值去噪", self.combo_pre_median))

    # ── 训练增强 ──
    def _build_enhance_card(self, lay) -> None:
        title = QLabel("训练增强（仅增加正常样本外观多样性，不保证改善检测）")
        title.setProperty("subtext", True)
        title.setWordWrap(True)
        lay.addWidget(title)
        self.chk_en_flip = QCheckBox("水平翻转")
        self.chk_en_flip.setToolTip("仅适合左右方向等价的正常外观；方向敏感、文字或装配朝向任务可能受损。")
        lay.addWidget(self.chk_en_flip)
        self.chk_en_rot90 = QCheckBox("旋转 90°")
        self.chk_en_rot90.setToolTip("仅适合旋转后仍为正常的样本；可能掩盖方向和装配位置缺陷。")
        lay.addWidget(self.chk_en_rot90)
        self.chk_en_brightness = QCheckBox("亮度 / 对比度扰动")
        self.chk_en_brightness.setToolTip("模拟正常光照变化；可能弱化低对比度缺陷、色差线索，不能保证降低误报。")
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
            count = pseudo.get("pseudo_per_image", 8)
            self.spin_pseudo_per_image.setValue(int(8 if count is None else count))
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
        return {
            "methods": methods,
            "pseudo_per_image": self.spin_pseudo_per_image.value(),
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
    def _on_ablation(self) -> None:
        pseudo = self._collect_pseudo()
        pre = self._collect_preprocess()
        enhance = self._collect_enhance()
        options = dict(category=self._get_category(), factor=self.combo_ablation.currentData(),
                       split=self.combo_ablation_split.currentText(),
                       train_limit=self.spin_ablation_train.value(),
                       eval_limit=self.spin_ablation_eval.value())
        self.btn_ablation.setEnabled(False)
        self.lbl_ablation.setText("正在提交隔离训练任务…")

        def failed(err):
            self.btn_ablation.setEnabled(True)
            self.lbl_ablation.setText(f"提交失败：{err}")

        def done(result):
            if not result or not result.get('task_id'):
                failed('未获得任务 ID')
                return
            self._ablation_task = result['task_id']
            self._ablation_timer.start(1000)
            self._poll_ablation()

        run_async(self, lambda: self._client.augment_ablation(pseudo, pre, enhance, **options), done, failed)

    def _poll_ablation(self):
        if getattr(self, '_ablation_polling', False) or not getattr(self, '_ablation_task', None):
            return
        self._ablation_polling = True

        def fill(task):
            self._ablation_polling = False
            if not task:
                return
            self.lbl_ablation.setText(f"任务 #{task.get('id')}：{task.get('status')} "
                                      f"{100 * (task.get('progress') or 0):.0f}%\n{task.get('message', '')}")
            if task.get('status') in ('done', 'failed', 'cancelled'):
                self._ablation_timer.stop()
                self.btn_ablation.setEnabled(True)
                result = task.get('result') or {}
                if result.get('status') == 'completed':
                    lines = [f"EvalRun #{result['eval_run_id']} · {result.get('n_images')} 张固定留出集",
                             '指标：基线 → 关闭因素（差值）']
                    for key in ('auroc', 'ap', 'f1', 'recall', 'fpr'):
                        a = result['baseline']['metrics'][key]
                        b = result['ablated']['metrics'][key]
                        lines.append(f'{key}: {a:.4f} → {b:.4f} ({b-a:+.4f})')
                    lines.append(result.get('limitations', ''))
                    self.lbl_ablation.setText('\n'.join(lines))
                elif task.get('status') == 'failed':
                    self.lbl_ablation.setText(f"任务失败：{task.get('message', '未知错误')}")
                elif task.get('status') == 'cancelled':
                    self.lbl_ablation.setText('任务已取消；对应 EvalRun 已记录取消状态。')

        def failed(err):
            self._ablation_polling = False
            self._ablation_timer.stop()
            self.btn_ablation.setEnabled(True)
            self.lbl_ablation.setText(f"查询任务失败：{err}")

        run_async(self, lambda: self._client.get_task(str(self._ablation_task)), fill, failed)

    def _build_preview_panel(self) -> QWidget:
        card = make_card()
        lay = QVBoxLayout(card)
        lay.addWidget(make_card_title("真实同集增强消融"))
        desc = QLabel("固定同一批 train 与 test/holdout 样本，分别重新拟合基线和关闭单一因素的模型，"
                      "只展示真实推理指标；不激活或覆盖生产模型。默认少量样本，耗时取决于当前模型配置。")
        desc.setProperty("subtext", True)
        desc.setWordWrap(True)
        lay.addWidget(desc)

        factor_row = QHBoxLayout()
        factor_row.addWidget(QLabel("关闭因素"))
        self.combo_ablation = QComboBox()
        for key, name in (
                ("pseudo.enabled", "伪异常合成"),
                ("preprocess.enabled", "预处理总开关"),
                ("preprocess.gray", "灰度预处理"),
                ("preprocess.clahe", "CLAHE"),
                ("preprocess.median", "中值去噪"),
                ("enhance.flip", "翻转增强"),
                ("enhance.rot90", "90°旋转增强"),
                ("enhance.brightness", "亮度增强")):
            self.combo_ablation.addItem(name, key)
        factor_row.addWidget(self.combo_ablation, 1)
        lay.addLayout(factor_row)

        split_row = QHBoxLayout()
        split_row.addWidget(QLabel("留出集"))
        self.combo_ablation_split = QComboBox()
        self.combo_ablation_split.addItem("test")
        self.combo_ablation_split.addItem("holdout")
        split_row.addWidget(self.combo_ablation_split)
        split_row.addWidget(QLabel("train 正常"))
        self.spin_ablation_train = QSpinBox()
        self.spin_ablation_train.setRange(2, 100)
        self.spin_ablation_train.setValue(8)
        split_row.addWidget(self.spin_ablation_train)
        split_row.addWidget(QLabel("留出集"))
        self.spin_ablation_eval = QSpinBox()
        self.spin_ablation_eval.setRange(2, 100)
        self.spin_ablation_eval.setValue(12)
        split_row.addWidget(self.spin_ablation_eval)
        lay.addLayout(split_row)

        self.btn_ablation = QPushButton("启动真实同集消融")
        self.btn_ablation.setProperty("primary", True)
        self.btn_ablation.clicked.connect(self._on_ablation)
        lay.addWidget(self.btn_ablation)
        self.lbl_ablation = QLabel("请选择基线中已启用的因素。disc 冻结或未参与训练时，伪异常/训练增强消融会明确失败。")
        self.lbl_ablation.setWordWrap(True)
        lay.addWidget(self.lbl_ablation)
        self._ablation_task = None
        self._ablation_polling = False
        self._ablation_timer = QTimer(self)
        self._ablation_timer.timeout.connect(self._poll_ablation)
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

        lay.addWidget(make_card_title("图像预处理对照（独立于伪异常预览）"))
        note = QLabel("使用当前未保存控件值，仅处理左侧第一张选择；最长边 1024，"
                      "先缩图再预处理，仅供外观展示，非性能评估，不执行模型推理。")
        note.setProperty("subtext", True)
        note.setWordWrap(True)
        lay.addWidget(note)
        pre_row = QHBoxLayout()
        self._pre_labels: dict[str, QLabel] = {}
        for key, caption in (("orig", "原图"), ("processed", "预处理后")):
            box = QVBoxLayout()
            cap = QLabel(caption)
            cap.setAlignment(Qt.AlignCenter)
            box.addWidget(cap)
            label = QLabel("—")
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumSize(140, 120)
            label.setStyleSheet("background:#EDF1F7; border-radius:6px; color:#9CA3AF;")
            box.addWidget(label)
            pre_row.addLayout(box, 1)
            self._pre_labels[key] = label
        lay.addLayout(pre_row)
        self.btn_pre_preview = QPushButton("预览当前预处理")
        self.btn_pre_preview.clicked.connect(self._on_preprocess_preview)
        lay.addWidget(self.btn_pre_preview, 0, Qt.AlignLeft)
        btn_space = QPushButton('真实特征空间分析')
        btn_space.clicked.connect(self._on_feature_space)
        lay.addWidget(btn_space, 0, Qt.AlignLeft)
        lay.addStretch(1)
        return card

    def _on_feature_space(self) -> None:
        images = self._selected_images()
        if not images:
            warn(self, '请先在左侧选择一张正常基底图，以确定品类和数据源范围')
            return
        image = images[0]
        scope = {'category': image.get('category')}
        dataset = getattr(self, '_feature_datasets', {}).get(image.get('dataset_id'), {})
        if dataset.get('datasource_id') is not None:
            scope['datasource_id'] = dataset['datasource_id']
        elif image.get('dataset_id') is not None:
            scope['dataset_id'] = image['dataset_id']
        try:
            from ui.widgets.feature_space_dialog import FeatureSpaceDialog
            dialog = FeatureSpaceDialog(self._client, scope, self._collect_pseudo(),
                                        self._collect_preprocess(), self)
        except ImportError as exc:
            warn(self, f'三维绘图依赖缺失，请安装requirements.txt（matplotlib）：{exc}')
            return
        dialog.exec()

    def _on_preprocess_preview(self) -> None:
        imgs = self._selected_images()
        if not imgs:
            warn(self, "请先在左侧选择一张基底图")
            return
        image_id = imgs[0]["id"]
        preprocess = self._collect_preprocess()
        self.btn_pre_preview.setEnabled(False)

        def _fill(res) -> None:
            self.btn_pre_preview.setEnabled(True)
            if not res:
                return
            for key, field in (("orig", "original_path"),
                               ("processed", "image_path")):
                load_thumb_async(self, self._client,
                                 self._client.file_url(res.get(field, "")),
                                 self._pre_labels[key], (200, 150))

        def _failed(err) -> None:
            self.btn_pre_preview.setEnabled(True)
            warn(self, f"预处理预览失败：{err}")

        run_async(self, lambda: self._client.preprocess_preview(
            image_id, preprocess), _fill, _failed)

    def _on_preview(self) -> None:
        imgs = self._selected_images()
        if not imgs:
            warn(self, "请先在左侧选择一张基底图")
            return
        method = self.combo_method.currentData() or "defect_transplant"
        img = imgs[0]
        params = self._collect_pseudo()

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
            img.get("id"), method, params=params), _fill)

    # ══════════════════ 通用 ══════════════════
    def reload(self) -> None:
        self.reload_images()
