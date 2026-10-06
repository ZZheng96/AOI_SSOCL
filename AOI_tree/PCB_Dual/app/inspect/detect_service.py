"""双检检测服务：传统+特征双检 → 结果落库（Detection + 每日统计）。

AOI_sys DetectionService 精简版（去掉 alarm/notify/archive，P6 再加），
核心复用 InspectService.run_dual 的判定，补充：
- 双检字段持久化（engine_mode/template_id/traditional_overall/feature_decision/dual_json）
- 每日统计（StatsDaily.n_inspected/n_anomaly）
"""
from __future__ import annotations

import time
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from app.db.database import session_scope
from app.db.models import DefectRecord, Detection, Inspect, StatsDaily
from app.inspect.service import InspectService
from app.template.model import InspectionTemplate

# 兼容 v1 单引擎调用方的路径常量
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class DetectionService:
    def __init__(self) -> None:
        self.inspect = InspectService()

    def detect_dual(self, template: InspectionTemplate, image_path: str | Path,
                    *, image_id: Optional[int] = None,
                    workorder_id: Optional[int] = None,
                    target_type: str = "image",
                    persist: bool = True,
                    allow_stub: bool = False,
                    inspect_id: Optional[int] = None,
                    board_barcode: Optional[str] = None) -> dict:
        """双检并落库（板级 Inspect + 缺陷明细 DefectRecord + 对位信息）。
        返回 DualSummary 字典 + detection_id + inspect_id + align。"""
        ds = self.inspect.run_dual(template, str(image_path),
                                   test_path=str(image_path),
                                   allow_stub=allow_stub)
        out = ds.to_dict()
        # 传统算法已完成像素级配准时，直接复用其元数据，避免再次跑整图配准。
        align_info: dict = {}
        if template.engine_mode in ("traditional", "dual"):
            traditional = out.get("traditional") or {}
            results = traditional.get("results") or []
            for result in results:
                alignment = (result.get("metadata") or {}).get("alignment")
                if alignment:
                    affine = alignment.get("affine") or [[1, 0, 0], [0, 1, 0]]
                    align_info = {
                        "ok": bool(alignment.get("ok")),
                        "offset_x": round(float(affine[0][2]), 2),
                        "offset_y": round(float(affine[1][2]), 2),
                        "method": str(alignment.get("method") or ""),
                        "score": float(alignment.get("score") or 0.0),
                        "warn": (
                            abs(float(affine[0][2])) > 10.0
                            or abs(float(affine[1][2])) > 10.0
                        ),
                        "affine": affine,
                    }
                    break
            if not align_info:
                from app.utils.cv_io import imread_unicode
                try:
                    std_img = imread_unicode(template.standard_image.path)
                    test_img = imread_unicode(str(image_path))
                    if std_img is not None and test_img is not None:
                        from app.inspect.align import align
                        align_info = align(std_img, test_img)
                except Exception as exc:  # noqa: BLE001
                    align_info = {"ok": False, "error": str(exc)}
        out["align"] = align_info
        if persist:
            out["detection_id"], out["inspect_id"] = self._persist_dual(
                template, str(image_path), ds.to_dict(),
                image_id=image_id, workorder_id=workorder_id,
                target_type=target_type, inspect_id=inspect_id,
                board_barcode=board_barcode, align_info=align_info)
        return out

    def _persist_dual(self, template: InspectionTemplate, image_path: str,
                      dual: dict, *, image_id: Optional[int] = None,
                      workorder_id: Optional[int] = None,
                      target_type: str = "image",
                      inspect_id: Optional[int] = None,
                      board_barcode: Optional[str] = None,
                      align_info: Optional[dict] = None) -> tuple[int, int]:
        feature = dual.get("feature") or {}
        trad = dual.get("traditional") or {}
        align_info = align_info or {}
        # 双坐标（P2）：模板标定存在才转物理坐标
        from app.core.coords import get_calibration
        cal = get_calibration(template.id)
        cal_active = cal.px_per_mm_x != 1.0 or cal.px_per_mm_y != 1.0
        boxes0 = (dual.get("boxes") or [])[:1]
        with session_scope() as s:
            # 板级 Inspect：无则建（一图即一板，简化口径；产线按板汇聚可传 inspect_id）
            if inspect_id is None:
                insp = Inspect(
                    workorder_id=workorder_id,
                    template_id=template.id,
                    template_version=template.version,
                    board_barcode=board_barcode,
                    total_count=0,
                    ng_count=0,
                    pass_=True,
                    inspect_time=datetime.now(),
                )
                s.add(insp)
                s.flush()
                inspect_id = insp.id
            row = Detection(
                target_type=target_type,
                workorder_id=workorder_id,
                image_id=image_id,
                inspect_id=inspect_id,
                image_path=image_path,
                category=template.model_category or template.category or "",
                engine_mode=dual.get("engine_mode", "dual"),
                template_id=template.id,
                template_version=template.version,
                traditional_overall=trad.get("overall"),
                feature_decision=feature.get("decision"),
                final_score=float(feature.get("score") or 0.0),
                is_anomaly=bool(dual.get("overall") == "NG"),
                latency_ms=float(dual.get("elapsed_ms") or 0),
                n_tiles={
                    "decision": feature.get("decision"),
                    "slot_scores": feature.get("slot_scores"),
                    "weights": feature.get("weights"),
                    "triggered_slot": feature.get("triggered_slot"),
                    "boxes": dual.get("boxes") or [],
                    "traditional": trad.get("results") or [],
                },
                dual_json=dual,
                align_offset_x=align_info.get("offset_x"),
                align_offset_y=align_info.get("offset_y"),
                align_warn=bool(align_info.get("warn")),
                board_x=(cal.px_to_board(
                    float(boxes0[0].get("x", 0)) + float(boxes0[0].get("w", 0)) / 2,
                    float(boxes0[0].get("y", 0)) + float(boxes0[0].get("h", 0)) / 2)[0]
                    if cal_active and boxes0 else None),
                board_y=(cal.px_to_board(
                    float(boxes0[0].get("x", 0)) + float(boxes0[0].get("w", 0)) / 2,
                    float(boxes0[0].get("y", 0)) + float(boxes0[0].get("h", 0)) / 2)[1]
                    if cal_active and boxes0 else None),
            )
            s.add(row)
            s.flush()
            det_id = row.id
            # 缺陷明细 DefectRecord（dual_json.boxes 已合并传统+AOI）
            for b in dual.get("boxes") or []:
                bx, by = None, None
                if cal_active:
                    cx = float(b.get("x") or 0) + float(b.get("w") or 0) / 2
                    cy = float(b.get("y") or 0) + float(b.get("h") or 0) / 2
                    bx, by = cal.px_to_board(cx, cy)
                s.add(DefectRecord(
                    detection_id=det_id,
                    inspect_id=inspect_id,
                    defect_type=str(b.get("label") or "未知"),
                    engine=str(b.get("engine") or "traditional"),
                    x=float(b.get("x") or 0), y=float(b.get("y") or 0),
                    width=float(b.get("w") or 0), height=float(b.get("h") or 0),
                    board_x=bx, board_y=by,
                    image_path=image_path,
                    score=None,
                ))
            # 板级统计更新
            insp = s.get(Inspect, inspect_id)
            if insp is not None:
                insp.total_count += 1
                if row.is_anomaly:
                    insp.ng_count += 1
                    insp.pass_ = False
                insp.finish_time = datetime.now()
            # 不良过滤（P2）：命中规则拦截缺陷明细（原始判定保留在 dual_json）
            try:
                from app.core.ng_filter import apply_ng_filters
                apply_ng_filters(s, inspect_id, template.id)
            except Exception:  # noqa: BLE001 过滤失败不阻断落库
                pass
            stats = s.query(StatsDaily).filter(StatsDaily.date == date.today()).first()
            if stats is None:
                stats = StatsDaily(date=date.today())
                s.add(stats)
                s.flush()
            stats.n_inspected += 1
            stats.n_anomaly += int(row.is_anomaly)
        return det_id, int(inspect_id)


_service: DetectionService | None = None


def get_detection_service() -> DetectionService:
    global _service
    if _service is None:
        _service = DetectionService()
    return _service
