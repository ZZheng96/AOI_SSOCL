"""FeedbackDialog：检测记录反馈对话框。

对某条检测记录标记 误检 / 漏检 / 确认 / 新缺陷；
显示原图 + 叠加图缩略；缺陷类型、备注、操作员字段；
确认后通过后台线程提交 POST /api/feedback。
"""
from __future__ import annotations

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from ui.api_client import ApiClient
from ui.widgets.image_viewer import ImageViewer

# 反馈类型 → (中文名, 操作员标注 operator_label)
FEEDBACK_TYPES = {
    "false_positive": ("误检（实为正常）", 0),
    "false_negative": ("漏检（实为异常）", 1),
    "confirmed": ("确认判定正确", -1),   # confirmed 的 -1 表示按检测结果自动推导
    "new_defect": ("新缺陷类型", 1),
    "uncertain": ("无法确认（进系统推荐复核清单）", -1),  # M15a demo5 §5 第三类
}


class _ThumbLoader(QThread):
    """后台加载缩略图字节。"""

    loaded = Signal(bytes)

    def __init__(self, client: ApiClient, url: str, parent=None):
        super().__init__(parent)
        self._client = client
        self._url = url

    def run(self) -> None:
        data = self._client.fetch_bytes(self._url)
        if data:
            self.loaded.emit(data)


class FeedbackDialog(QDialog):
    """单条检测记录的操作员反馈对话框。"""

    submitted = Signal(dict)  # 提交成功（参数为服务端返回）

    def __init__(self, client: ApiClient, detection: dict,
                 preset_type: str = "", parent: QWidget | None = None):
        super().__init__(parent)
        self._client = client
        self._det = detection or {}
        self._loaders: list[QThread] = []
        self._region: list | None = None  # 漏检框选区域 [x,y,w,h] 原图像素坐标
        self.setWindowTitle("检测反馈标注")
        self.setMinimumWidth(560)

        root = QHBoxLayout(self)

        # ── 左：原图 / 叠加图缩略 ──
        img_col = QVBoxLayout()
        self._thumb_orig = self._make_thumb("原图")
        self._thumb_overlay = self._make_thumb("叠加图")
        img_col.addWidget(QLabel("原图"))
        img_col.addWidget(self._thumb_orig)
        img_col.addWidget(QLabel("叠加图"))
        img_col.addWidget(self._thumb_overlay)
        img_col.addStretch(1)
        root.addLayout(img_col)

        # ── 右：反馈表单 ──
        right = QVBoxLayout()
        det = self._det
        info = QLabel(
            f"检测编号：{det.get('detection_id', det.get('id', '-'))}　"
            f"分数：{float(det.get('final_score', 0) or 0):.3f}　"
            f"判定：{'异常' if det.get('is_anomaly') else '正常'}")
        info.setProperty("subtext", True)
        right.addWidget(info)

        # 反馈类型（单选）
        self._type_group = QButtonGroup(self)
        self._radios: dict[str, QRadioButton] = {}
        for i, (key, (label, _)) in enumerate(FEEDBACK_TYPES.items()):
            rb = QRadioButton(label)
            self._type_group.addButton(rb, i)
            self._radios[key] = rb
            right.addWidget(rb)
            if key == (preset_type or "false_positive"):
                rb.setChecked(True)
        if self._type_group.checkedId() < 0:
            next(iter(self._radios.values())).setChecked(True)

        form = QFormLayout()
        self.edit_defect_type = QLineEdit()
        self.edit_defect_type.setPlaceholderText("如：划痕 / 凹陷 / 污染（新缺陷必填）")
        form.addRow("缺陷类型", self.edit_defect_type)
        self.edit_operator = QLineEdit()
        self.edit_operator.setPlaceholderText("操作员姓名/工号")
        form.addRow("操作员", self.edit_operator)
        self.edit_comment = QPlainTextEdit()
        self.edit_comment.setPlaceholderText("补充说明（可选）")
        self.edit_comment.setMaximumHeight(80)
        form.addRow("备注", self.edit_comment)
        right.addLayout(form)

        # ── 漏检区域框选（仅"漏检"类型显示） ──
        self._region_box = QWidget()
        rg = QVBoxLayout(self._region_box)
        rg.setContentsMargins(0, 4, 0, 0)
        rg.setSpacing(4)
        hint = QLabel("可框选漏检区域（可选）：在下方图片上按住左键拖拽框选，"
                      "Esc 或单击图片清除选区")
        hint.setProperty("subtext", True)
        hint.setWordWrap(True)
        rg.addWidget(hint)
        self._sel_viewer = ImageViewer()
        self._sel_viewer.setMinimumHeight(240)
        self._sel_viewer.setMaximumHeight(280)
        self._sel_viewer.set_selection_mode(True)
        self._sel_viewer.rect_selected.connect(self._on_region_selected)
        rg.addWidget(self._sel_viewer)
        row = QHBoxLayout()
        self.lbl_region = QLabel("当前选区：未框选")
        self.lbl_region.setProperty("subtext", True)
        row.addWidget(self.lbl_region, 1)
        btn_clear = QPushButton("清除选区")
        btn_clear.setProperty("flat", True)
        btn_clear.clicked.connect(self._clear_region)
        row.addWidget(btn_clear)
        rg.addLayout(row)
        right.addWidget(self._region_box)
        right.addStretch(1)
        # 框选区构建完成后再连接类型切换（避免初始化期触发未就绪控件）
        for rb in self._radios.values():
            rb.toggled.connect(
                lambda _c=False: self._update_region_visibility())

        # 页面内状态提示（替代 QMessageBox 弹窗：成功/失败直接展示在表单内）
        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setVisible(False)
        right.addWidget(self.lbl_status)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("提交反馈")
        buttons.button(QDialogButtonBox.Ok).setProperty("primary", True)
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        buttons.accepted.connect(self._on_submit)
        buttons.rejected.connect(self.reject)
        right.addWidget(buttons)
        root.addLayout(right, 1)

        self._load_thumbnails()
        self._update_region_visibility()
        self._load_region_image()

    # ── 反馈类型 ─────────────────────────────────────────
    def _set_status(self, text: str, kind: str = "info") -> None:
        """页面内状态标签（ok=绿 / error=红 / info=蓝）。"""
        color = {"ok": "#15803D", "error": "#B91C1C", "info": "#1D4ED8"}[kind]
        bg = {"ok": "#DCFCE7", "error": "#FEE2E2", "info": "#DBEAFE"}[kind]
        self.lbl_status.setText(text)
        self.lbl_status.setStyleSheet(
            f"background:{bg}; color:{color}; border-radius:4px;"
            "padding:6px 8px;")
        self.lbl_status.setVisible(True)

    def _on_submit(self) -> None:
        ftype = self._selected_type()
        defect_type = self.edit_defect_type.text().strip()
        if ftype == "new_defect" and not defect_type:
            self._set_status("标记新缺陷时请先填写缺陷类型", "error")
            return
        _, label = FEEDBACK_TYPES[ftype]
        if ftype == "confirmed" and label < 0:  # 确认正确：操作员标注与系统判定一致
            label = 1 if self._det.get("is_anomaly") else 0
        # uncertain 保持 -1（不定真值，后端登记 label=unknown）
        det_id = self._det.get("detection_id", self._det.get("id"))
        if det_id is None:
            self._set_status("该记录缺少检测编号，无法提交反馈", "error")
            return
        payload = dict(
            detection_id=int(det_id), feedback_type=ftype,
            operator_label=label, defect_type=defect_type,
            comment=self.edit_comment.toPlainText().strip(),
            operator=self.edit_operator.text().strip(),
            # 漏检且已框选时随提交携带区域，否则为 None
            region=self._region if ftype == "false_negative" else None)

        from ui.api_client import ApiWorker
        self._worker = ApiWorker(lambda: self._client.submit_feedback(**payload), self)
        self._worker.succeeded.connect(self._on_done)
        self._worker.failed.connect(
            lambda e: self._set_status(f"提交失败：{e}", "error"))
        self._worker.start()

    def _on_done(self, result) -> None:
        if result is None:
            self._set_status("反馈提交失败（无响应），请重试", "error")
            return
        self._set_status(self._success_text(result), "ok")
        self.submitted.emit(result if isinstance(result, dict) else {})
        QTimer.singleShot(900, self.accept)          # 展示成功状态后自动关闭

    # ── 漏检框选 ─────────────────────────────────────────
    def _update_region_visibility(self) -> None:
        """仅"漏检(false_negative)"类型显示框选区。"""
        self._region_box.setVisible(self._selected_type() == "false_negative")

    def _load_region_image(self) -> None:
        """框选小图：优先叠加图，其次原图（走 /api/file 加载）。"""
        path = self._det.get("overlay_path") or self._det.get("image_path")
        if path:
            self._sel_viewer.set_image_url(self._client.file_url(path))

    def _on_region_selected(self, region: list) -> None:
        self._region = list(region)
        self.lbl_region.setText(f"当前选区：[x={region[0]}, y={region[1]}, "
                                f"w={region[2]}, h={region[3]}]（原图像素）")

    def _clear_region(self) -> None:
        self._region = None
        self._sel_viewer.clear_selection()
        self.lbl_region.setText("当前选区：未框选")

    # ── 内部 ─────────────────────────────────────────────
    def _make_thumb(self, placeholder: str) -> QLabel:
        lb = QLabel(placeholder)
        lb.setFixedSize(220, 150)
        lb.setAlignment(Qt.AlignCenter)
        lb.setStyleSheet("background:#EDF1F7; border-radius:6px; color:#6B7280;")
        return lb

    def _load_thumbnails(self) -> None:
        for path, label in (
                (self._det.get("image_path"), self._thumb_orig),
                (self._det.get("overlay_path"), self._thumb_overlay)):
            if not path:
                label.setText("无图片")
                continue
            url = self._client.file_url(path)
            loader = _ThumbLoader(self._client, url, self)
            loader.loaded.connect(
                lambda data, lb=label: self._set_thumb(lb, data))
            loader.finished.connect(lambda w=loader: self._forget(w))
            self._loaders.append(loader)
            loader.start()

    def _set_thumb(self, label: QLabel, data: bytes) -> None:
        pm = QPixmap()
        pm.loadFromData(data)
        if not pm.isNull():
            label.setPixmap(pm.scaled(
                label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _forget(self, w: QThread) -> None:
        if w in self._loaders:
            self._loaders.remove(w)
        w.deleteLater()

    def _selected_type(self) -> str:
        for key, rb in self._radios.items():
            if rb.isChecked():
                return key
        return "confirmed"

    @staticmethod
    def _success_text(result) -> str:
        """成功提示：附带引擎即学结果摘要（engine_update）；
        响应含 pre（反馈前判定，M7a）时在摘要前插"分数 x.xx→已学习"。
        2026-08-30：改读 handler 真实 action（原 triggered_* 键后端从不
        下发，恒显"已学习"，熔断/待成簇等状态静默）。"""
        result = result if isinstance(result, dict) else {}
        pre_txt = ""
        pre = result.get("pre")
        if isinstance(pre, dict) and isinstance(pre.get("score"), (int, float)):
            pre_txt = f"分数 {float(pre['score']):.2f}→已学习"
        upd = result.get("engine_update")
        if not isinstance(upd, dict):
            note = result.get("note")
            base = f"反馈已提交。{note}" if note else "反馈已提交，感谢标注！"
            return f"{base}（{pre_txt}）" if pre_txt else base
        action_cn = {
            "reflow_accepted": "正常样本已回流，校准分布已更新",
            "reflow_pending": "回流证据累积中（需连续相似样本成簇）",
            "reflow_rejected": "样本与正常库差异大，未回流",
            "reflow_blocked_by_fuse_ratio": "熔断保护中：在线缺陷率高，正常回流暂停",
            "reflow_rolled_back": "更新被回归门控回滚（锚定分布漂移）",
            "reflow_rejected_small": "样本特征不足，未回流",
            "defect_sample_add": "缺陷样本已入库，同类缺陷即时拦截",
            "difficult_queued": "难例已入学习队列",
        }
        parts = []
        action = str(upd.get("action") or "")
        if action in action_cn:
            parts.append(action_cn[action])
        if upd.get("fuse_blocked") and action != "reflow_blocked_by_fuse_ratio":
            parts.append("熔断保护中（在线缺陷率高）")
        ms = upd.get("feedback_ms")
        tail = f"（耗时 {float(ms):.0f}ms）" if isinstance(ms, (int, float)) else ""
        summary = "、".join(parts) if parts else "已学习"
        if pre_txt:
            summary = pre_txt + ("、" + summary if parts else "")
        return f"反馈已提交并在线学习：{summary}{tail}"
