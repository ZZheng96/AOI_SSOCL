"""特征工程页：统一展示生产特征事实，并把实验变更限制在 draft 候选。"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from ..api_client import ApiClient
from ..widgets.charts import BarChart
from .common import attach_level_badge, make_banner, make_card, notify, run_async, warn


_TEXT = {
    "available": "可用", "not_applicable": "未启用传统特征", "unknown": "未知",
    "not_generated": "未生成", "disabled": "未启用", "revision": "配置版本",
    "current": "当前有效", "stale": "已过期", "missing": "未生成",
    "insufficient": "样本不足", "running": "生成中", "pending": "排队中",
    "completed": "已完成", "failed": "失败", "draft": "候选草稿",
    "validated": "已验证", "approved": "已批准", "rejected": "已拒绝",
    "evaluated": "已评估", "active": "已生效", "retired": "已退役",
    "bound": "已绑定", "baseline": "基准得分", "ablated": "移除后得分",
    "contribution": "贡献值", "revision_id": "配置版本", "state": "状态",
    "activated": "已激活", "changes_production": "修改生产", "message": "说明",
    "task_id": "任务编号", "status": "状态", "good": "正常", "defect": "缺陷",
    "normal": "正常", "anomaly": "异常", "total": "总数",
    "sem": "语义特征", "disc": "判别特征", "shead": "监督检测头",
    "trad": "传统特征", "tpl": "模板特征", "open": "开放集特征",
    "blob": "斑块特征", "color": "颜色特征", "layout": "布局特征",
    "inp": "重建特征", "version": "版本", "needs_dino": "依赖主干特征",
}


def _display(value) -> str:
    if isinstance(value, dict):
        return "；".join(f"{_TEXT.get(str(k), str(k))}：{_display(v)}" for k, v in value.items()) or "-"
    if isinstance(value, bool):
        return "是" if value else "否"
    if value is None:
        return "未知"
    text = _TEXT.get(str(value), str(value))
    for key in ("baseline", "ablated", "contribution", "revision", "draft"):
        text = text.replace(key, _TEXT[key])
    return text


class FeaturePage(QWidget):
    """Core 内嵌特征分析/治理页；不直接调用 AOI_feature 独立窗口。"""

    def __init__(self, client: ApiClient, get_category, parent=None):
        super().__init__(parent)
        self._client = client
        self._get_category = get_category
        self._fs: dict = {}
        self._groups: list[dict] = []
        self._production_pointer: dict = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(8)
        top = QHBoxLayout()
        top.addWidget(make_banner("特征工程：生产事实只读，候选配置隔离"))
        attach_level_badge(self, top)
        top.addStretch(1)
        root.addLayout(top)

        guide = QLabel(
            "生产特征是当前检测正在使用的只读事实；本页的档案、候选配置版本和重建结果都先留在候选侧。"
            "只有完成评估/准入并显式绑定后，才会影响生产。"
        )
        guide.setWordWrap(True)
        guide.setProperty("subtext", True)
        root.addWidget(guide)

        self.lbl_state = QLabel("请选择品类并加载生产特征")
        self.lbl_state.setProperty("subtext", True)
        root.addWidget(self.lbl_state)

        body = QHBoxLayout()
        left = make_card()
        ll = QVBoxLayout(left)
        lh = QLabel("当前生产特征")
        lh.setProperty("heading", True)
        ll.addWidget(lh)
        form = QFormLayout()
        self.combo_category = QComboBox()
        self.combo_category.currentTextChanged.connect(lambda _v: self.reload())
        form.addRow("品类", self.combo_category)
        ll.addLayout(form)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["特征组", "维度", "启用", "参数", "来源"])
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        ll.addWidget(self.table, 1)
        self.lbl_meta = QLabel("生产配置版本：以生产指针为准\n生产提取器：以当前快照元数据为准")
        self.lbl_meta.setWordWrap(True)
        self.lbl_meta.setProperty("subtext", True)
        ll.addWidget(self.lbl_meta)
        body.addWidget(left, 3)

        right = make_card()
        rl = QVBoxLayout(right)
        rh = QLabel("候选分析与治理")
        rh.setProperty("heading", True)
        rl.addWidget(rh)
        self.lbl_report = QLabel("当前特征贡献档案：-\n（小样本评估证据，不是新的生产配置）")
        self.lbl_report.setWordWrap(True)
        rl.addWidget(self.lbl_report)
        self.lbl_analysis_scope = QLabel(
            "样本分析：按最近一次测试集评价档案展示特征组判别力与留一影响；"
            "不是生产模型的逐图贡献，也不代表因果贡献。")
        self.lbl_analysis_scope.setWordWrap(True)
        self.lbl_analysis_scope.setProperty("subtext", True)
        rl.addWidget(self.lbl_analysis_scope)
        self.group_chart = BarChart()
        self.group_chart.setMinimumHeight(220)
        rl.addWidget(self.group_chart)
        self.lbl_image_scope = QLabel("逐图贡献：尚无档案")
        self.lbl_image_scope.setWordWrap(True)
        rl.addWidget(self.lbl_image_scope)
        self.combo_image = QComboBox()
        self.combo_image.currentIndexChanged.connect(self._render_image_contribution)
        rl.addWidget(self.combo_image)
        self.image_table = QTableWidget(0, 4)
        self.image_table.setHorizontalHeaderLabels(["特征组", "基准得分", "移除后得分", "贡献值"])
        self.image_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.image_table.horizontalHeader().setStretchLastSection(True)
        rl.addWidget(self.image_table)
        refresh_report = QPushButton("刷新分析结果")
        refresh_report.clicked.connect(lambda: self.reload())
        rl.addWidget(refresh_report)
        self.edit_note = QLineEdit()
        self.edit_note.setPlaceholderText("候选说明（不会修改生产）")
        rl.addWidget(self.edit_note)
        row = QHBoxLayout()
        self.btn_report = QPushButton("分析当前特征影响")
        self.btn_report.setToolTip("使用少量测试集样本生成传统特征贡献档案，不修改生产")
        self.btn_report.setProperty("primary", True)
        self.btn_report.clicked.connect(self._report)
        self.btn_draft = QPushButton("登记候选配置版本")
        self.btn_draft.setToolTip("按当前启用特征组和评分方式登记候选版本，不修改生产")
        self.btn_draft.clicked.connect(self._draft)
        row.addWidget(self.btn_report)
        row.addWidget(self.btn_draft)
        rl.addLayout(row)
        self.btn_rebuild = QPushButton("重建候选表示")
        self.btn_rebuild.setToolTip("按当前代码、注册表和预处理重提特征并生成未激活模型版本；不是自动绑定配置版本")
        self.btn_rebuild.clicked.connect(self._rebuild)
        rl.addWidget(self.btn_rebuild)
        self.revisions = QTableWidget(0, 4)
        self.revisions.setHorizontalHeaderLabels(["配置版本", "状态", "维度", "生产绑定"])
        self.revisions.setEditTriggers(QTableWidget.NoEditTriggers)
        self.revisions.horizontalHeader().setStretchLastSection(True)
        rl.addWidget(QLabel("候选配置版本（仅记录，不代表生产已使用）"))
        rl.addWidget(self.revisions, 1)
        body.addWidget(right, 2)
        root.addLayout(body, 1)
        self._client.error_occurred.connect(lambda m: warn(self, m))
        self.reload()

    def _category(self) -> str:
        return self.combo_category.currentText().strip()

    def reload(self, categories=None) -> None:
        if categories is not None:
            self._category_scope = list(categories)
        categories = getattr(self, "_category_scope", None)
        if categories is None:
            categories = [x.get("category") for x in self._client.categories_full()]
        current = self._category()
        self.combo_category.blockSignals(True)
        self.combo_category.clear()
        self.combo_category.addItems([str(x) for x in categories if x])
        if current:
            i = self.combo_category.findText(current)
            if i >= 0:
                self.combo_category.setCurrentIndex(i)
        self.combo_category.blockSignals(False)
        cat = self._category()
        self._load_id = getattr(self, "_load_id", 0) + 1
        load_id = self._load_id
        self._production_pointer = {}
        self._on_feature({})
        self.revisions.setRowCount(0)
        if not cat:
            self.lbl_state.setText("当前范围暂无可选品类")
            return
        run_async(self, lambda: self._client.feature_set(cat),
                  lambda data: self._on_feature(data) if load_id == self._load_id else None)
        run_async(self, self._client.feature_revisions,
                  lambda data: self._on_revisions(data) if load_id == self._load_id else None)

    def _on_feature(self, data) -> None:
        self._fs = data or {}
        self._groups = list(self._fs.get("groups") or [])
        self.table.setRowCount(0)
        slots = [s for s in self._fs.get("production_slots", []) if s.get("key") != "trad"]
        for g in self._groups + slots:
            r = self.table.rowCount()
            self.table.insertRow(r)
            params = g.get("params")
            if not params:
                params = f"权重={g.get('weight', '-')}；尺度={g.get('scale', '-')}"
            vals = [g.get("name") or _display(g.get("key", "")),
                    g.get("dim") if g.get("dim") is not None else "未知",
                    "是" if g.get("enabled") else "否", _display(params),
                    "生产槽位（主干维度）" if g in slots else "生产传统特征组"]
            for c, v in enumerate(vals):
                self.table.setItem(r, c, QTableWidgetItem(str(v)))
        status = self._fs.get("status", "未准备生产模型" if self._fs.get("enabled") is False else "unknown")
        binding = self._fs.get("feature_revision_binding") or {}
        rid = binding.get("revision_id") or "未绑定/未知"
        self.lbl_state.setText(f"生产特征状态：{_display(status)}；实验配置不会直接替换该状态")
        self.lbl_meta.setText(
            f"生产配置版本：{rid}\n生产模型版本：{self._fs.get('version', '-')}\n"
            f"生产签名：{_display(self._fs.get('signature') or '-')}\n"
            f"传统特征维度：{self._fs.get('enabled_dim', '-')} / {self._fs.get('total_dim', '-')}\n"
            "槽位维度按各自主干列示，共享主干不可相加。品类来自当前范围内的数据或保留模型；删除工单不等于删除模型。"
        )
        editable = bool(self._fs.get("can_edit"))
        traditional = status == "available"
        self.btn_report.setEnabled(editable and (traditional or bool(slots)))
        self.btn_report.setText("分析当前特征影响" if traditional else "分析生产槽位影响")
        self.btn_report.setToolTip("少量测试集槽位消融，不是逐图传统贡献，不修改生产" if not traditional
                                   else "使用少量测试集样本生成传统特征贡献档案，不修改生产")
        self.btn_draft.setEnabled(editable and traditional)
        self.btn_rebuild.setEnabled(editable and bool(self._fs.get("enabled")))
        report = self._fs.get("report") or {}
        self.lbl_report.setText(f"传统特征档案：{_display(report.get('status', '未生成'))}\n"
                                f"测试集：{_display(report.get('test_counts', {}))}"
                                + ("\n当前生产未启用传统特征，传统组贡献分析和候选登记不适用。"
                                   if status == "not_applicable" else ""))
        self._render_group_analysis(report)
        tg = report.get("trad_groups") or {}
        self._image_rows = list(tg.get("per_image") or [])
        self.combo_image.blockSignals(True)
        self.combo_image.clear()
        for image_id in sorted({r["image_id"] for r in self._image_rows if "image_id" in r}):
            self.combo_image.addItem(f"图片编号={image_id}", image_id)
        self.combo_image.blockSignals(False)
        self.lbl_image_scope.setText(
            f"档案状态：{_display(report.get('status', '未生成'))}；"
            + _display(tg.get("per_image_semantics", "传统特征距离留一，非生产融合解释")
                       if self._image_rows else tg.get("per_image_reason") or tg.get("reason")
                       or "旧档案或引擎未提供逐图分数，请生成档案后刷新"))
        self._render_image_contribution()

    def _render_image_contribution(self, *_args) -> None:
        image_id = self.combo_image.currentData()
        rows = [r for r in getattr(self, "_image_rows", []) if r.get("image_id") == image_id]
        self.image_table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j, key in enumerate(("feature_group", "baseline", "ablated", "contribution")):
                value = row.get(key)
                text = f"{value:.6g}" if isinstance(value, (float, int)) else str(value or "-")
                self.image_table.setItem(i, j, QTableWidgetItem(text))

    def _render_group_analysis(self, report: dict) -> None:
        """展示 test 样本档案中的特征组证据，不冒充逐图生产模型解释。"""
        groups = (report.get("trad_groups") or {}).get("groups") or {}
        if not isinstance(groups, dict):
            groups = {}
        rows = []
        for key, item in groups.items():
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or key)
            single = item.get("single_auroc")
            drop = item.get("rebuild_loo_drop")
            if isinstance(single, (int, float)) or isinstance(drop, (int, float)):
                rows.append((name, single, drop))
        rows.sort(key=lambda row: max(
            abs(float(row[1])) if isinstance(row[1], (int, float)) else 0.0,
            abs(float(row[2])) if isinstance(row[2], (int, float)) else 0.0), reverse=True)
        rows = rows[:8]
        if not rows:
            ablation = report.get("ablation") or {}
            single = ablation.get("single") or {}
            loo = ablation.get("leave_one_out") or {}
            labels = list(single)
            self.group_chart.set_data([_display(k) for k in labels], {
                "单槽位AUROC（测试集小样本）": [single[k] for k in labels],
                "移除槽位后AUROC（测试集小样本）": [loo.get(k) for k in labels],
            } if labels else {})
            if labels:
                self.lbl_analysis_scope.setText("真实生产槽位小样本消融；非传统特征组，非逐图贡献，不代表因果贡献。")
            return
        labels = [r[0] for r in rows]
        self.group_chart.set_data(labels, {
            "单组 AUROC（测试集样本）": [r[1] for r in rows],
            "重建留一掉分（测试集样本）": [r[2] for r in rows],
        })

    def _on_revisions(self, data) -> None:
        data = data or {}
        pointer = data.get("production_pointer") or {}
        snapshot = pointer.get("model_snapshot") or {}
        self._production_pointer = pointer if snapshot.get("category") == self._category() else {}
        items = data.get("items") or {}
        if isinstance(items, dict):
            items = [{"revision_id": k, **(v if isinstance(v, dict) else {})}
                     for k, v in items.items()]
        self.revisions.setRowCount(0)
        for item in items:
            manifest = item.get("manifest") or {}
            if (manifest.get("category") or (manifest.get("model_snapshot") or {}).get("category")) != self._category():
                continue
            r = self.revisions.rowCount()
            self.revisions.insertRow(r)
            manifest = item.get("manifest") or {}
            vals = [item.get("revision_id") or item.get("id") or "-",
                    _display(item.get("state") or item.get("status") or "-"),
                    manifest.get("dimension", "-"),
                    "是" if (item.get("production_binding") or
                             item.get("revision_id") == self._production_pointer.get("revision_id"))
                    else "否"]
            for c, v in enumerate(vals):
                self.revisions.setItem(r, c, QTableWidgetItem(str(v)))

    def _report(self) -> None:
        cat = self._category()
        if cat:
            run_async(self, lambda: self._client.feature_set_report(cat),
                      lambda r: (notify(self, "已排队生成特征档案", _display(r)), self.reload()))

    def _draft(self) -> None:
        cat = self._category()
        groups = [g.get("key") for g in self._groups if g.get("enabled")]
        if not cat or not groups:
            warn(self, "当前没有可登记的生产特征组")
            return
        run_async(self, lambda: self._client.feature_set_draft(
            cat, groups, self._fs.get("scoring", "legacy"), self.edit_note.text()),
            lambda r: (notify(self, "候选已登记，未修改生产", _display(r)), self.reload()))

    def _rebuild(self) -> None:
        cat = self._category()
        if cat:
            run_async(self, lambda: self._client.feature_set_rebuild(cat),
                      lambda r: notify(self, "表示重建已排队，完成后仍需独立验收", _display(r)))
