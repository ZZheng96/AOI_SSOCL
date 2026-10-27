"""AC-064 多框复核面板：列出候选/人工区域，逐框确认/拒绝/标争议，原图叠加框 + 绘框新增/重画修正。

候选框≠人工真值：仅 confirmed 进入框级训练；disputed 隔离不训。
走 GET/POST /detections/{id}/regions（修订链 + 乐观锁 + 事件幂等）。
坐标契约：bbox 为原图像素 xyxy（coord_space.size=[W, H]）；显示图可能被缩放，
换算与 payload 构造均为纯函数（scene_rect_to_image / build_box_payload），便于离屏单测。
"""
from __future__ import annotations

import uuid

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPen
from PySide6.QtWidgets import (
    QAbstractItemView, QDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from ui.widgets.image_viewer import ImageViewer

STATUS_TEXT = {"candidate": "候选(不训)", "confirmed": "已确认(可训)",
               "rejected": "已拒绝", "disputed": "争议(隔离)"}
TRAIN_TEXT = {"eligible": "可训", "excluded_candidate": "排除:候选",
              "excluded_rejected": "排除:拒绝", "excluded_disputed": "排除:争议",
              "excluded_visual": "排除:仅可视"}
COLUMNS = ("区域", "bbox", "来源", "峰值", "状态", "操作人", "训练状态", "复核人")
# 叠加框样式：(颜色, 线型)。候选=橙虚线，人工可训=绿实线，拒绝=灰点线，争议=紫点划线
BOX_STYLE = {"eligible": ("#16A34A", "solid"), "excluded_candidate": ("#F59E0B", "dash"),
             "excluded_rejected": ("#9CA3AF", "dot"), "excluded_disputed": ("#9333EA", "dashdot"),
             "excluded_visual": ("#60A5FA", "dot")}
HUMAN_EXCLUDED = ("#2563EB", "solid")    # 人工画过但未进训练（如争议中的人工框）用蓝色实线
_PEN_STYLE = {"solid": Qt.SolidLine, "dash": Qt.DashLine, "dot": Qt.DotLine,
              "dashdot": Qt.DashDotLine}
MIN_BOX = 2   # 原图像素，最小有效边长


def box_style(region: dict):
    """按 train_status（其次 source）给出 (颜色, 线型)；候选框与人工框可区分。"""
    ts = region.get("train_status") or "excluded_candidate"
    if ts == "eligible":
        return BOX_STYLE["eligible"]
    if region.get("source") == "human_box" and ts != "excluded_rejected":
        return HUMAN_EXCLUDED
    return BOX_STYLE.get(ts, BOX_STYLE["excluded_candidate"])


def scene_rect_to_image(rect, disp_wh, image_wh):
    """显示像素矩形 (x0,y0,x1,y1) → 原图像素 xyxy（按显示/原图缩放换算并裁剪到图内）。

    disp_wh=当前显示图尺寸；image_wh=原图尺寸（coord_space.size，缺省同显示图）。"""
    dw, dh = (float(v) for v in disp_wh)
    iw, ih = (float(v) for v in (image_wh or disp_wh))
    sx, sy = iw / max(dw, 1.0), ih / max(dh, 1.0)
    x0, x1 = sorted((float(rect[0]), float(rect[2])))
    y0, y1 = sorted((float(rect[1]), float(rect[3])))
    clip = lambda v, hi: int(round(min(max(v, 0.0), hi)))  # noqa: E731
    return [clip(x0 * sx, iw), clip(y0 * sy, ih), clip(x1 * sx, iw), clip(y1 * sy, ih)]


def image_to_scene_rect(bbox, disp_wh, image_wh):
    """原图像素 xyxy → 显示像素 (x, y, w, h)，用于叠加绘制。"""
    dw, dh = (float(v) for v in disp_wh)
    iw, ih = (float(v) for v in (image_wh or disp_wh))
    sx, sy = dw / max(iw, 1.0), dh / max(ih, 1.0)
    x0, y0, x1, y1 = (float(v) for v in bbox)
    return (x0 * sx, y0 * sy, (x1 - x0) * sx, (y1 - y0) * sy)


def hit_region(regions, x, y):
    """原图坐标 (x, y) 命中的最小框 region_id（无命中返回 None）。"""
    hits = [r for r in regions or [] if r.get("bbox")
            and r["bbox"][0] <= x < r["bbox"][2] and r["bbox"][1] <= y < r["bbox"][3]]
    if not hits:
        return None
    b = lambda r: (r["bbox"][2] - r["bbox"][0]) * (r["bbox"][3] - r["bbox"][1])  # noqa: E731
    return min(hits, key=b).get("region_id")


def build_box_payload(op, bbox, operator, reason, revision, region_id=None,
                      image_wh=None, event_id=None):
    """构造 add/revise 请求体；非法输入返回 (None, 中文错误)。

    几何修改即人工真值断言：add/revise 一律要求 reason，并携带乐观锁 expected_revision。"""
    if op not in ("add", "revise"):
        return None, f"不支持的绘框操作: {op}"
    operator, reason = (operator or "").strip(), (reason or "").strip()
    if not operator:
        return None, "请填写操作员"
    if not reason:
        return None, "新增/修正框须填写理由"
    if op == "revise" and not region_id:
        return None, "请先在表格中选择要修正的框"
    if revision is None:
        return None, "缺少修订号，请刷新后重试"
    if not bbox or len(bbox) != 4:
        return None, "请先在预览图上拖拽画框"
    x0, y0, x1, y1 = (int(v) for v in bbox)
    if x1 - x0 < MIN_BOX or y1 - y0 < MIN_BOX or x0 < 0 or y0 < 0:
        return None, "框太小或越界，请重画"
    if image_wh and (x1 > int(image_wh[0]) or y1 > int(image_wh[1])):
        return None, "框超出原图范围，请重画"
    body = {"op": op, "bbox": [x0, y0, x1, y1], "operator": operator,
            "event_id": event_id or uuid.uuid4().hex,
            "expected_revision": int(revision), "reason": reason}
    if op == "revise":
        body["region_id"] = region_id
    return body, None


class RegionCanvas(ImageViewer):
    """原图预览：按 train_status/source 叠加各区域框、高亮选中框；左键拖拽画框（原图 xyxy）。"""

    box_drawn = Signal(list)       # 原图像素 xyxy
    region_clicked = Signal(str)   # 单击命中的 region_id

    def __init__(self, parent=None):
        super().__init__(parent)
        self._regions: list = []
        self._image_wh = None
        self._highlight = None
        self._region_items: dict = {}
        self.set_selection_mode(True)

    def set_regions(self, regions, image_wh=None, highlight=None) -> None:
        self._regions = list(regions or [])
        self._image_wh = image_wh
        self._highlight = highlight
        self._redraw_boxes()

    def set_highlight(self, region_id) -> None:
        self._highlight = region_id
        self._redraw_boxes()

    def disp_wh(self):
        pm = self._pixmap_item.pixmap()
        return None if pm.isNull() else (pm.width(), pm.height())

    def image_wh(self):
        return self._image_wh or self.disp_wh()

    def region_items(self) -> dict:
        return self._region_items

    def _show_current(self) -> None:
        super()._show_current()
        self.fit_to_view()

    def _redraw_boxes(self) -> None:
        super()._redraw_boxes()
        for it in getattr(self, "_region_items", {}).values():
            self._scene.removeItem(it)
        self._region_items = {}
        disp = self.disp_wh()
        if disp is None:
            return
        for r in getattr(self, "_regions", []):
            if not r.get("bbox"):
                continue
            color, style = box_style(r)
            hl = r.get("region_id") == self._highlight
            pen = QPen(QColor(color), 4 if hl else 2, _PEN_STYLE[style])
            pen.setCosmetic(True)
            item = self._scene.addRect(QRectF(*image_to_scene_rect(r["bbox"], disp, self.image_wh())), pen)
            item.setZValue(15 if hl else 10)
            item.setToolTip(f"{r.get('region_id')} · {r.get('source')} · {r.get('train_status')}")
            self._region_items[r.get("region_id")] = item

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if (self._select_mode and event.button() == Qt.LeftButton
                and self._sel_origin is not None):
            disp = self.disp_wh()
            if not self._sel_dragging:      # 单击：命中框则选中，否则清除选区
                p = self._sel_origin
                x0, y0, _, _ = scene_rect_to_image((p.x(), p.y(), p.x(), p.y()), disp, self.image_wh())
                self.clear_selection()
                rid = hit_region(self._regions, x0, y0)
                if rid is not None:
                    self.region_clicked.emit(rid)
                return
            r = self._sel_item.rect() if self._sel_item is not None else QRectF()
            bbox = scene_rect_to_image((r.left(), r.top(), r.right(), r.bottom()),
                                       disp, self.image_wh())
            self._sel_origin = None
            self._sel_dragging = False
            if bbox[2] - bbox[0] >= MIN_BOX and bbox[3] - bbox[1] >= MIN_BOX:
                self.box_drawn.emit(bbox)
            else:
                self.clear_selection()
            return
        super().mouseReleaseEvent(event)


class RegionReviewDialog(QDialog):
    def __init__(self, client, detection_id: int, parent=None, image_url: str = ""):
        super().__init__(parent)
        self._client = client
        self._det_id = int(detection_id)
        self._review: dict = {}
        self._pending_bbox = None    # 预览上最近一次拖拽得到的原图 xyxy（待提交）
        self.setWindowTitle(f"多框复核 · 检测 {self._det_id}")
        self.resize(1120, 560)
        root = QHBoxLayout(self)
        # 左：原图叠加框 + 绘框
        left = QVBoxLayout()
        self.canvas = RegionCanvas(self)
        self.canvas.setMinimumWidth(420)
        self.canvas.box_drawn.connect(self._on_box_drawn)
        self.canvas.region_clicked.connect(self.select_region)
        left.addWidget(self.canvas, 1)
        legend = QLabel("<span style='color:#F59E0B'>■</span>候选(不训) "
                        "<span style='color:#16A34A'>■</span>人工可训 "
                        "<span style='color:#2563EB'>■</span>人工未训 "
                        "<span style='color:#9CA3AF'>■</span>已拒绝 "
                        "<span style='color:#9333EA'>■</span>争议；左键拖拽画框，单击框选中")
        legend.setWordWrap(True)
        legend.setProperty("subtext", True)
        left.addWidget(legend)
        self.lbl_pending = QLabel("待提交框：无")
        self.lbl_pending.setProperty("subtext", True)
        left.addWidget(self.lbl_pending)
        draw_row = QHBoxLayout()
        self.btn_add = QPushButton("提交为新增框")
        self.btn_add.clicked.connect(lambda _c=False: self.submit_box("add"))
        draw_row.addWidget(self.btn_add)
        self.btn_revise = QPushButton("替换选中框")
        self.btn_revise.clicked.connect(lambda _c=False: self.submit_box("revise"))
        draw_row.addWidget(self.btn_revise)
        left.addLayout(draw_row)
        root.addLayout(left, 1)

        lay = QVBoxLayout()
        hint = QLabel("候选框不是人工真值，仅“已确认”框进入框级训练；"
                      "推翻他人结论需填写理由，否则标为争议并隔离；新增/修正框须填写理由。")
        hint.setWordWrap(True)
        hint.setProperty("subtext", True)
        lay.addWidget(hint)
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.itemSelectionChanged.connect(
            lambda: self.canvas.set_highlight(self.selected_region_id()))
        lay.addWidget(self.table, 1)
        form = QFormLayout()
        self.edit_operator = QLineEdit()
        self.edit_operator.setPlaceholderText("操作员姓名/工号（必填）")
        form.addRow("操作员", self.edit_operator)
        self.edit_reason = QLineEdit()
        self.edit_reason.setPlaceholderText("理由（推翻他人结论/仲裁争议/新增或修正框时必填）")
        form.addRow("理由", self.edit_reason)
        lay.addLayout(form)
        row = QHBoxLayout()
        self.buttons = {}
        for op, text in (("confirm", "确认"), ("reject", "拒绝"), ("dispute", "标争议")):
            b = QPushButton(text)
            b.clicked.connect(lambda _c=False, o=op: self.apply(o))
            row.addWidget(b)
            self.buttons[op] = b
        btn_refresh = QPushButton("刷新")
        btn_refresh.clicked.connect(self.refresh)
        row.addWidget(btn_refresh)
        lay.addLayout(row)
        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        lay.addWidget(self.lbl_status)
        root.addLayout(lay, 1)
        if image_url:
            self.canvas.set_image_url(image_url)
        self.refresh()

    def refresh(self) -> None:
        data = self._client.get(self.api_path())
        if data is None:
            self.lbl_status.setText("加载区域失败")
            return
        keep = self.selected_region_id() if self._review else None
        self._review = data
        regions = data.get("regions") or []
        self.table.blockSignals(True)
        self.table.setRowCount(len(regions))
        for i, r in enumerate(regions):
            peak = r.get("score_peak")
            ts = r.get("train_status")
            vals = (r.get("region_id"), r.get("bbox"), r.get("source"),
                    "-" if peak is None else f"{float(peak):.3f}",
                    STATUS_TEXT.get(r.get("status"), r.get("status")),
                    r.get("operator") or "-", TRAIN_TEXT.get(ts, ts),
                    r.get("reviewer") or "-")
            for j, v in enumerate(vals):
                self.table.setItem(i, j, QTableWidgetItem("-" if v is None else str(v)))
        self.table.blockSignals(False)
        self.canvas.set_regions(regions, data.get("coord_space"), highlight=keep)
        if keep is not None:
            self.select_region(keep)
        n_conf = sum(1 for r in regions if r.get("status") == "confirmed")
        self.lbl_status.setText(f"共 {len(regions)} 框，可训 {n_conf} 框，"
                                f"修订号 {data.get('revision', 0)}")

    def api_path(self) -> str:
        # 后端路由挂在 /api 前缀下（app.include_router(prefix="/api")），ApiClient 直接拼 base_url
        return f"/api/detections/{self._det_id}/regions"

    def select_region(self, region_id) -> None:
        for i, r in enumerate(self._review.get("regions") or []):
            if r.get("region_id") == region_id:
                self.table.selectRow(i)
                return

    def selected_region_id(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return None
        return (self._review.get("regions") or [])[rows[0].row()].get("region_id")

    def _on_box_drawn(self, bbox: list) -> None:
        self._pending_bbox = list(bbox)
        self.lbl_pending.setText(f"待提交框：{self._pending_bbox}（原图像素 xyxy）")

    def box_payload(self, op: str):
        return build_box_payload(op, self._pending_bbox, self.edit_operator.text(),
                                 self.edit_reason.text(), self._review.get("revision"),
                                 region_id=self.selected_region_id(),
                                 image_wh=self._review.get("coord_space"))

    def submit_box(self, op: str) -> bool:
        body, err = self.box_payload(op)
        if body is None:
            self.lbl_status.setText(err)
            return False
        if not self._post(body):
            return False
        self._pending_bbox = None
        self.lbl_pending.setText("待提交框：无")
        self.canvas.clear_selection()
        return True

    def apply(self, op: str) -> bool:
        rid = self.selected_region_id()
        operator = self.edit_operator.text().strip()
        if rid is None or not operator:
            self.lbl_status.setText("请先选择区域并填写操作员")
            return False
        body = {"op": op, "region_id": rid, "operator": operator,
                "event_id": uuid.uuid4().hex,
                "expected_revision": int(self._review.get("revision", 0)),
                "reason": self.edit_reason.text().strip() or None}
        return self._post(body)

    def _post(self, body: dict) -> bool:
        res = self._client.post(self.api_path(), json=body, conflict_ok=True)
        if res is None or res.get("_status") == 409:
            detail = (res or {}).get("detail", "请查看错误提示")
            if res is not None:
                self.refresh()  # 乐观锁冲突：拉取最新修订号
            self.lbl_status.setText(f"提交失败：{detail}")
            return False
        self.refresh()
        return True
