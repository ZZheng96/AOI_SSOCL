"""AC-064 多框复核面板：列出候选/人工区域，逐框确认/拒绝/标争议，原图叠加框 + 绘框新增/重画修正。

候选框≠人工真值：仅 confirmed 进入框级训练；disputed 隔离不训。
走 GET/POST /detections/{id}/regions（修订链 + 乐观锁 + 事件幂等）。
坐标契约：bbox 为原图像素 xyxy（coord_space.size=[W, H]）；显示图可能被缩放，
换算与 payload 构造均为纯函数（scene_rect_to_image / build_box_payload），便于离屏单测。
编辑（AC-064 剩余项）：选中框显示 4 角手柄，拖动微调（drag_corner 纯函数，复用 revise 提交）；
仅人工框可删除（软删除，模型框只能拒绝）；“撤销上一步”撤销本人最近一次操作（反向修订事件）。
拖边（drag_edge）/框内平移（move_box）同样复用 revise；“重做”恢复本人最近一次撤销（redo 事件）。
"""
from __future__ import annotations

import uuid

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPen
from PySide6.QtWidgets import (
    QAbstractItemView, QDialog, QDoubleSpinBox, QFormLayout, QGraphicsItem, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from ui.widgets.image_viewer import ImageViewer

STATUS_TEXT = {"candidate": "候选(不训)", "confirmed": "已确认(可训)",
               "rejected": "已拒绝", "disputed": "争议(隔离)",
               "deleted": "已删除(留档)", "retracted": "已撤销(留档)"}
TRAIN_TEXT = {"eligible": "可训", "excluded_candidate": "排除:候选",
              "excluded_rejected": "排除:拒绝", "excluded_disputed": "排除:争议",
              "excluded_visual": "排除:仅可视", "excluded_deleted": "排除:删除",
              "excluded_retracted": "排除:撤销"}
COLUMNS = ("区域", "bbox", "来源", "峰值", "状态", "操作人", "训练状态", "复核人", "学习消费")


def consumption_text(region: dict) -> str:
    """AC-063 训练消费标签：stale_after_use 优先于 used（已用后变更需重建候选）。"""
    if region.get("stale_after_use"):
        return "已用后变更"
    if region.get("used"):
        return "已用于学习"
    return "未用于学习"


# 叠加框样式：(颜色, 线型)。候选=橙虚线，人工可训=绿实线，拒绝=灰点线，争议=紫点划线
BOX_STYLE = {"eligible": ("#16A34A", "solid"), "excluded_candidate": ("#F59E0B", "dash"),
             "excluded_rejected": ("#9CA3AF", "dot"), "excluded_disputed": ("#9333EA", "dashdot"),
             "excluded_visual": ("#60A5FA", "dot")}
HUMAN_EXCLUDED = ("#2563EB", "solid")    # 人工画过但未进训练（如争议中的人工框）用蓝色实线
_PEN_STYLE = {"solid": Qt.SolidLine, "dash": Qt.DashLine, "dot": Qt.DotLine,
              "dashdot": Qt.DashDotLine}
MIN_BOX = 2   # 原图像素，最小有效边长
HANDLE_PX = 6  # 角点手柄半边长（屏幕像素）
INACTIVE = ("deleted", "retracted")


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


def is_active(region: dict) -> bool:
    """deleted/retracted 区域仅留档：不绘制、不可点选/微调。"""
    return region.get("status") not in INACTIVE


def can_delete(region) -> bool:
    """仅原始来源为 human_box 的有效框可删除；模型候选框（含被人工修订过的）只能拒绝。"""
    if not region or not is_active(region):
        return False
    return (region.get("origin_source") or region.get("source")) == "human_box"


def bbox_corners(bbox) -> dict:
    """原图 xyxy → 4 个角点 {tl, tr, br, bl: (x, y)}。"""
    x0, y0, x1, y1 = (float(v) for v in bbox)
    return {"tl": (x0, y0), "tr": (x1, y0), "br": (x1, y1), "bl": (x0, y1)}


def handle_tolerance(view_scale, disp_wh, image_wh, px=None) -> float:
    """手柄命中半径：屏幕 px → 原图像素（视图缩放 × 显示/原图比例）。"""
    px = HANDLE_PX if px is None else px
    dw = float((disp_wh or (1, 1))[0])
    iw = float((image_wh or disp_wh or (1, 1))[0])
    return px / max(float(view_scale or 1.0), 1e-6) * iw / max(dw, 1.0)


def hit_corner(bbox, x, y, tol):
    """原图坐标 (x, y) 命中的最近角点名（超出 tol 返回 None）。"""
    best, dist = None, None
    for name, (cx, cy) in bbox_corners(bbox).items():
        d = max(abs(x - cx), abs(y - cy))
        if d <= tol and (dist is None or d < dist):
            best, dist = name, d
    return best


def drag_corner(bbox, corner, x, y, image_wh=None, min_size=None):
    """拖动角点 → 新 bbox（对角固定；裁剪到图内；宽高 ≥ min_size，不翻转）。"""
    m = MIN_BOX if min_size is None else int(min_size)
    x0, y0, x1, y1 = (int(round(float(v))) for v in bbox)
    if image_wh:
        x = min(max(float(x), 0.0), float(image_wh[0]))
        y = min(max(float(y), 0.0), float(image_wh[1]))
    x, y = int(round(max(float(x), 0.0))), int(round(max(float(y), 0.0)))
    if corner in ("tl", "bl"):
        x0 = min(x, x1 - m)
    else:
        x1 = max(x, x0 + m)
    if corner in ("tl", "tr"):
        y0 = min(y, y1 - m)
    else:
        y1 = max(y, y0 + m)
    return [max(x0, 0), max(y0, 0), x1, y1]


EDGES = ("l", "t", "r", "b")


def hit_edge(bbox, x, y, tol):
    """原图坐标 (x, y) 命中的边名 l/t/r/b（须在该边跨度内且距边 ≤ tol；角点优先由 hit_corner 判）。"""
    x0, y0, x1, y1 = (float(v) for v in bbox)
    cand = []
    if y0 - tol <= y <= y1 + tol:
        cand += [("l", abs(x - x0)), ("r", abs(x - x1))]
    if x0 - tol <= x <= x1 + tol:
        cand += [("t", abs(y - y0)), ("b", abs(y - y1))]
    cand = [c for c in cand if c[1] <= tol]
    return min(cand, key=lambda c: c[1])[0] if cand else None


def drag_edge(bbox, edge, x, y, image_wh=None, min_size=None):
    """拖动单条边 → 新 bbox（对边固定；裁剪到图内；宽高 ≥ min_size，不翻转）。"""
    m = MIN_BOX if min_size is None else int(min_size)
    x0, y0, x1, y1 = (int(round(float(v))) for v in bbox)
    W, H = (image_wh or (None, None))[:2]
    x = int(round(min(max(float(x), 0.0), float(W) if W else float(x))))
    y = int(round(min(max(float(y), 0.0), float(H) if H else float(y))))
    if edge == "l":
        x0 = min(max(x, 0), x1 - m)
    elif edge == "r":
        x1 = max(x, x0 + m)
    elif edge == "t":
        y0 = min(max(y, 0), y1 - m)
    elif edge == "b":
        y1 = max(y, y0 + m)
    return [x0, y0, x1, y1]


def move_box(bbox, dx, dy, image_wh=None):
    """整框平移 (dx, dy)（原图像素）；尺寸不变，平移量被夹到图内。"""
    x0, y0, x1, y1 = (int(round(float(v))) for v in bbox)
    dx, dy = int(round(float(dx))), int(round(float(dy)))
    dx = max(dx, -x0)
    dy = max(dy, -y0)
    if image_wh:
        dx = min(dx, int(image_wh[0]) - x1)
        dy = min(dy, int(image_wh[1]) - y1)
    return [x0 + dx, y0 + dy, x1 + dx, y1 + dy]


def build_redo_payload(operator, reason, revision, event_id=None):
    """构造 redo 请求体（重做本人最近一次撤销）。"""
    body, err = build_undo_payload(operator, reason, revision, event_id)
    if body is None:
        return None, err.replace("撤销", "重做")
    return {**body, "op": "redo"}, None


def build_delete_payload(region, operator, reason, revision, event_id=None):
    """构造 delete 请求体（仅人工框；须 reason + expected_revision）；非法返回 (None, 错误)。"""
    operator, reason = (operator or "").strip(), (reason or "").strip()
    if not region:
        return None, "请先选择要删除的框"
    if not can_delete(region):
        return None, "模型候选框不可删除，请使用拒绝"
    if not operator:
        return None, "请填写操作员"
    if not reason:
        return None, "删除人工框须填写理由"
    if revision is None:
        return None, "缺少修订号，请刷新后重试"
    return {"op": "delete", "region_id": region.get("region_id"), "operator": operator,
            "event_id": event_id or uuid.uuid4().hex,
            "expected_revision": int(revision), "reason": reason}, None


def build_undo_payload(operator, reason, revision, event_id=None):
    """构造 undo 请求体（撤销本人最近一次操作；reason 可选）。"""
    operator = (operator or "").strip()
    if not operator:
        return None, "请填写操作员（只能撤销本人操作）"
    if revision is None:
        return None, "缺少修订号，请刷新后重试"
    return {"op": "undo", "operator": operator, "event_id": event_id or uuid.uuid4().hex,
            "expected_revision": int(revision), "reason": (reason or "").strip() or None}, None


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


def build_recompute_payload(rel_thresh=None, min_area=None, morph_kernel=None):
    """AC-063 热图复算请求体（与后端 HeatmapRecompute 约束一致）；返回 (body, err)。

    None 表示沿用留存的原参数；复算只读预览，不落库、不改区域。"""
    body = {}
    if rel_thresh is not None:
        v = float(rel_thresh)
        if not 0.0 < v < 1.0:
            return None, "rel_thresh 须在 (0,1) 内"
        body["rel_thresh"] = v
    if min_area is not None:
        v = int(min_area)
        if not 1 <= v <= 1_000_000:
            return None, "min_area 须在 1..1e6"
        body["min_area"] = v
    if morph_kernel is not None:
        v = int(morph_kernel)
        if not 1 <= v <= 31:
            return None, "morph_kernel 须在 1..31"
        body["morph_kernel"] = v
    return body, ""


RECOMPUTE_ERRORS = {404: "检测不存在", 409: "未留存原始热图，无法复算",
                    403: "留存路径越界", 410: "原始热图已按保留策略清理"}


class RegionCanvas(ImageViewer):
    """原图预览：按 train_status/source 叠加各区域框、高亮选中框；左键拖拽画框（原图 xyxy）。"""

    box_drawn = Signal(list)       # 原图像素 xyxy
    region_clicked = Signal(str)   # 单击命中的 region_id
    box_adjusted = Signal(str, list)  # 角点微调完成：(region_id, 原图 xyxy)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._regions: list = []
        self._image_wh = None
        self._highlight = None
        self._region_items: dict = {}
        self._handle_items: dict = {}
        self._drag = None              # 角点拖拽中：(region_id, corner)
        self._preview_boxes: list = []  # AC-063 热图复算只读预览框（原图 xyxy）
        self._preview_items: list = []
        self.set_selection_mode(True)

    def set_preview_boxes(self, boxes) -> None:
        self._preview_boxes = [list(b) for b in (boxes or []) if b and len(b) == 4]
        self._redraw_boxes()

    def preview_items(self) -> list:
        return self._preview_items

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

    def handle_items(self) -> dict:
        return self._handle_items

    def highlighted_region(self):
        return next((r for r in getattr(self, "_regions", [])
                     if r.get("region_id") == getattr(self, "_highlight", None)
                     and is_active(r) and r.get("bbox")), None)

    def scene_to_image(self, x, y):
        """显示像素点 → 原图像素（浮点，不裁剪）。"""
        disp, img = self.disp_wh(), self.image_wh()
        return (float(x) * img[0] / max(disp[0], 1), float(y) * img[1] / max(disp[1], 1))

    def image_to_scene(self, x, y):
        disp, img = self.disp_wh(), self.image_wh()
        return (float(x) * disp[0] / max(img[0], 1), float(y) * disp[1] / max(img[1], 1))

    def _show_current(self) -> None:
        super()._show_current()
        self.fit_to_view()

    def _redraw_boxes(self) -> None:
        super()._redraw_boxes()
        for it in [*getattr(self, "_region_items", {}).values(),
                   *getattr(self, "_handle_items", {}).values(),
                   *getattr(self, "_preview_items", [])]:
            self._scene.removeItem(it)
        self._region_items, self._handle_items, self._preview_items = {}, {}, []
        disp = self.disp_wh()
        if disp is None:
            return
        if getattr(self, "_preview_boxes", None):
            pen = QPen(QColor("#06B6D4"), 2, Qt.DashLine)
            pen.setCosmetic(True)
            for b in self._preview_boxes:
                it = self._scene.addRect(QRectF(*image_to_scene_rect(b, disp, self.image_wh())), pen)
                it.setZValue(8)
                it.setToolTip("热图复算预览（只读，不落库）")
                self._preview_items.append(it)
        for r in getattr(self, "_regions", []):
            if not r.get("bbox") or not is_active(r):   # 已删除/撤销仅表格留档
                continue
            color, style = box_style(r)
            hl = r.get("region_id") == self._highlight
            pen = QPen(QColor(color), 4 if hl else 2, _PEN_STYLE[style])
            pen.setCosmetic(True)
            item = self._scene.addRect(QRectF(*image_to_scene_rect(r["bbox"], disp, self.image_wh())), pen)
            item.setZValue(15 if hl else 10)
            item.setToolTip(f"{r.get('region_id')} · {r.get('source')} · {r.get('train_status')}")
            self._region_items[r.get("region_id")] = item
        hl = self.highlighted_region()
        if hl is not None:                        # 选中框 4 角手柄（屏幕固定尺寸）
            pen = QPen(QColor("#111827"), 1)
            pen.setCosmetic(True)
            for name, (cx, cy) in bbox_corners(hl["bbox"]).items():
                h = self._scene.addRect(QRectF(-HANDLE_PX / 2, -HANDLE_PX / 2, HANDLE_PX, HANDLE_PX),
                                        pen, QBrush(QColor("#FFFFFF")))
                h.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
                h.setPos(*self.image_to_scene(cx, cy))
                h.setZValue(25)
                h.setToolTip(f"拖动角点微调 {hl.get('region_id')}")
                self._handle_items[name] = h

    def _corner_at(self, scene_pos):
        hl = self.highlighted_region()
        if hl is None or self.disp_wh() is None:
            return None
        x, y = self.scene_to_image(scene_pos.x(), scene_pos.y())
        tol = handle_tolerance(self.transform().m11(), self.disp_wh(), self.image_wh())
        c = hit_corner(hl["bbox"], x, y, tol) or hit_edge(hl["bbox"], x, y, tol)
        if c is not None:
            return (hl["region_id"], c)
        b = hl["bbox"]
        if b[0] <= x < b[2] and b[1] <= y < b[3]:   # 选中框内部：整框平移
            return (hl["region_id"], "move", x, y, list(b))
        return None

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self._select_mode and event.button() == Qt.LeftButton:
            drag = self._corner_at(self.mapToScene(event.position().toPoint()))
            if drag is not None:
                self._drag = drag
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag is not None:
            p = self.mapToScene(event.position().toPoint())
            self.move_corner(*self.scene_to_image(p.x(), p.y()))
            return
        super().mouseMoveEvent(event)

    def move_corner(self, x, y) -> None:
        """拖拽中：角点/单边/整框平移，按原图坐标更新本地框（最小尺寸、图内）并重绘。"""
        rid, kind = self._drag[0], self._drag[1]
        for i, r in enumerate(self._regions):
            if r.get("region_id") != rid:
                continue
            if kind == "move":
                ox, oy, b0 = self._drag[2:]
                nb = move_box(b0, x - ox, y - oy, self.image_wh())
            elif kind in EDGES:
                nb = drag_edge(r["bbox"], kind, x, y, self.image_wh())
            else:
                nb = drag_corner(r["bbox"], kind, x, y, self.image_wh())
            self._regions[i] = {**r, "bbox": nb}
        self._redraw_boxes()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._drag is not None and event.button() == Qt.LeftButton:
            drag, self._drag = self._drag, None
            rid = drag[0]
            hl = self.highlighted_region()
            moved = not (drag[1] == "move" and list(hl["bbox"] if hl else []) == drag[4])
            if hl is not None and hl["region_id"] == rid and moved:
                self.box_adjusted.emit(rid, list(hl["bbox"]))
            return
        if (self._select_mode and event.button() == Qt.LeftButton
                and self._sel_origin is not None):
            disp = self.disp_wh()
            if not self._sel_dragging:      # 单击：命中框则选中，否则清除选区
                p = self._sel_origin
                x0, y0, _, _ = scene_rect_to_image((p.x(), p.y(), p.x(), p.y()), disp, self.image_wh())
                self.clear_selection()
                rid = hit_region([r for r in self._regions if is_active(r)], x0, y0)
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
        self._pending_rid = None     # 角点微调对应的 region_id（revise 目标）
        self.setWindowTitle(f"多框复核 · 检测 {self._det_id}")
        self.resize(1120, 560)
        root = QHBoxLayout(self)
        # 左：原图叠加框 + 绘框
        left = QVBoxLayout()
        self.canvas = RegionCanvas(self)
        self.canvas.setMinimumWidth(420)
        self.canvas.box_drawn.connect(self._on_box_drawn)
        self.canvas.region_clicked.connect(self.select_region)
        self.canvas.box_adjusted.connect(self._on_box_adjusted)
        left.addWidget(self.canvas, 1)
        legend = QLabel("<span style='color:#F59E0B'>■</span>候选(不训) "
                        "<span style='color:#16A34A'>■</span>人工可训 "
                        "<span style='color:#2563EB'>■</span>人工未训 "
                        "<span style='color:#9CA3AF'>■</span>已拒绝 "
                        "<span style='color:#9333EA'>■</span>争议；左键拖拽画框，单击框选中，"
                        "拖动选中框角点微调后点“替换选中框”提交")
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
        self.btn_delete = QPushButton("删除人工框")
        self.btn_delete.setToolTip("仅人工新增框可删除（软删除留档）；模型候选框请用“拒绝”")
        self.btn_delete.clicked.connect(lambda _c=False: self.delete_selected())
        draw_row.addWidget(self.btn_delete)
        self.btn_undo = QPushButton("撤销上一步")
        self.btn_undo.setToolTip("撤销本人最近一次操作（追加反向修订；被他人改过则冲突）")
        self.btn_undo.clicked.connect(lambda _c=False: self.undo_last())
        draw_row.addWidget(self.btn_undo)
        self.btn_redo = QPushButton("重做")
        self.btn_redo.setToolTip("重做本人最近一次撤销（撤销后有新操作则失效；被他人改过则冲突）")
        self.btn_redo.clicked.connect(lambda _c=False: self.redo_last())
        draw_row.addWidget(self.btn_redo)
        left.addLayout(draw_row)
        # AC-063：热图复算只读预览（不落库、不改区域，仅叠加青色虚线框）
        rc_row = QHBoxLayout()
        rc_row.addWidget(QLabel("rel_thresh"))
        self.spin_rel = QDoubleSpinBox()
        self.spin_rel.setRange(0.0, 0.99)       # 0 = 沿用原参数
        self.spin_rel.setSingleStep(0.05)
        self.spin_rel.setDecimals(2)
        self.spin_rel.setSpecialValueText("原值")
        rc_row.addWidget(self.spin_rel)
        rc_row.addWidget(QLabel("min_area"))
        self.spin_area = QSpinBox()
        self.spin_area.setRange(0, 1_000_000)   # 0 = 沿用原参数
        self.spin_area.setSpecialValueText("原值")
        rc_row.addWidget(self.spin_area)
        rc_row.addWidget(QLabel("morph"))
        self.spin_morph = QSpinBox()
        self.spin_morph.setRange(0, 31)         # 0 = 沿用原参数
        self.spin_morph.setSpecialValueText("原值")
        rc_row.addWidget(self.spin_morph)
        self.btn_recompute = QPushButton("热图复算(只读预览)")
        self.btn_recompute.setToolTip("用留存原始热图按新阈值复算候选框，仅预览不落库")
        self.btn_recompute.clicked.connect(lambda _c=False: self.recompute_heatmap())
        rc_row.addWidget(self.btn_recompute)
        self.btn_clear_preview = QPushButton("清除预览")
        self.btn_clear_preview.clicked.connect(lambda _c=False: self.clear_preview())
        rc_row.addWidget(self.btn_clear_preview)
        left.addLayout(rc_row)
        self.lbl_recompute = QLabel("")
        self.lbl_recompute.setProperty("subtext", True)
        self.lbl_recompute.setWordWrap(True)
        left.addWidget(self.lbl_recompute)
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
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
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
                    r.get("reviewer") or "-", consumption_text(r))
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

    def selected_region(self):
        rid = self.selected_region_id()
        return next((r for r in self._review.get("regions") or []
                     if r.get("region_id") == rid), None) if rid is not None else None

    def _on_selection_changed(self) -> None:
        rid = self.selected_region_id()
        if self._pending_rid is not None and rid != self._pending_rid:
            # 切换到其他框：放弃未提交的角点微调（画布恢复服务端坐标）
            self._pending_rid, self._pending_bbox = None, None
            self.lbl_pending.setText("待提交框：无")
            self.canvas.set_regions(self._review.get("regions") or [],
                                    self._review.get("coord_space"), highlight=rid)
            return
        self.canvas.set_highlight(rid)

    def _on_box_drawn(self, bbox: list) -> None:
        self._pending_bbox = list(bbox)
        self._pending_rid = None
        self.lbl_pending.setText(f"待提交框：{self._pending_bbox}（原图像素 xyxy）")

    def _on_box_adjusted(self, region_id: str, bbox: list) -> None:
        self.select_region(region_id)
        self._pending_bbox, self._pending_rid = list(bbox), region_id
        self.lbl_pending.setText(f"角点微调 {region_id} → {self._pending_bbox}"
                                 "（填写理由后点“替换选中框”提交）")

    def box_payload(self, op: str):
        return build_box_payload(op, self._pending_bbox, self.edit_operator.text(),
                                 self.edit_reason.text(), self._review.get("revision"),
                                 region_id=self.selected_region_id(),
                                 image_wh=self._review.get("coord_space"))

    def delete_selected(self) -> bool:
        body, err = build_delete_payload(self.selected_region(), self.edit_operator.text(),
                                         self.edit_reason.text(), self._review.get("revision"))
        if body is None:
            self.lbl_status.setText(err)
            return False
        return self._post(body)

    def undo_last(self) -> bool:
        body, err = build_undo_payload(self.edit_operator.text(), self.edit_reason.text(),
                                       self._review.get("revision"))
        if body is None:
            self.lbl_status.setText(err)
            return False
        return self._post(body)

    def redo_last(self) -> bool:
        body, err = build_redo_payload(self.edit_operator.text(), self.edit_reason.text(),
                                       self._review.get("revision"))
        if body is None:
            self.lbl_status.setText(err)
            return False
        return self._post(body)

    def submit_box(self, op: str) -> bool:
        body, err = self.box_payload(op)
        if body is None:
            self.lbl_status.setText(err)
            return False
        if not self._post(body):
            return False
        self._pending_bbox, self._pending_rid = None, None
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

    def recompute_path(self) -> str:
        return f"/api/detections/{self._det_id}/heatmap/recompute"

    def recompute_payload(self):
        return build_recompute_payload(self.spin_rel.value() or None,
                                       self.spin_area.value() or None,
                                       self.spin_morph.value() or None)

    def clear_preview(self) -> None:
        self.canvas.set_preview_boxes([])
        self.lbl_recompute.setText("")

    def recompute_heatmap(self):
        """AC-063：调用只读复算接口，叠加预览框；不修改区域/修订号。"""
        body, err = self.recompute_payload()
        if body is None:
            self.lbl_recompute.setText(f"复算参数无效：{err}")
            return None
        res = self._client.post(self.recompute_path(), json=body, conflict_ok=True)
        if res is None or "_status" in res:
            st = (res or {}).get("_status")
            detail = (res or {}).get("detail") or RECOMPUTE_ERRORS.get(st, "请查看错误提示")
            self.lbl_recompute.setText(f"复算失败：{detail}")
            self.canvas.set_preview_boxes([])
            return None
        boxes = [b for b in (res.get("boxes_image") or []) if b and len(b) == 4]
        self.canvas.set_preview_boxes(boxes)
        p = res.get("params") or {}
        self.lbl_recompute.setText(
            f"复算预览：{len(boxes)} 框，mask 像素 {res.get('mask_pixels')}，"
            f"参数 {p}（只读，未落库）")
        return res

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
