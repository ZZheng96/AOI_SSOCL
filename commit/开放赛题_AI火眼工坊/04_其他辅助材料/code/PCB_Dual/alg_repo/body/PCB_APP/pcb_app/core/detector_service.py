"""检测编排服务：封装 pcb_defect_detector，处理多 ROI、共享预处理、自动保存。

对外只暴露 ``DetectorService``：
  - ``list_algorithms()``               列出可用算法
  - ``run(template, test, tasks, ...)``  执行一批"算法×区域"检测任务，返回 ``RunSummary``

多 ROI 合并优化（对应需求"合并共同步骤"）：
  - 图片只从磁盘/内存加载一次，之后以 ndarray 复用；
  - 算法实例池由底层门面缓存复用（不重复加载）；
  - 对完全相同的 (算法, 参数, 区域) 任务去重，只算一次并复用结果，避免多余耗时。
"""

from __future__ import annotations

import hashlib
import inspect
import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from .. import config as app_config
from ..metadata import DEFECT_BY_CODE
from .models import DefectBox, MatchedROI, ROI, RunItemResult, RunSummary
from .roi_util import apply_roi

ImageInput = Union[str, np.ndarray]

# 结构匹配成功时，移位/极反按匹配到的方向分支二选一，避免同一 ROI 同时报两种结果。
_ORIENTATION_CODES = {"component_shift", "component_reverse_polarity"}


def _hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


class DetectorService:
    def __init__(self):
        app_config.ensure_detector_on_path()
        # 延迟到路径注入后再导入
        from pcb_defect_detector import PCBDefectDetector  # noqa: WPS433
        from pcb_defect_detector.image_io import imread_unicode  # noqa: WPS433

        self._det = PCBDefectDetector()
        self._imread = imread_unicode
        # ROI 匹配器属于引擎公共几何能力；在路径注入完成后再导入。
        from algorithms.component.common import (  # noqa: WPS433
            extract_matched_roi, match_roi_structure)
        self._match_roi_structure = match_roi_structure
        self._extract_matched_roi = extract_matched_roi
        # 旧版 pcb_defect_detector 可能没有 detect(roi=...) 参数
        sig = inspect.signature(self._det.detect)
        self._det_supports_roi = "roi" in sig.parameters
        # 预热算法池（一次性开销移出检测计时）
        self._available = set(self._det.available_algorithms())
        # 每个 ROI 的结构匹配结果缓存：画框/挪框后立即算一次，检测时直接复用，
        # 避免"画框预览"和"点击运行检测"重复搜索同一个 ROI。
        self._roi_match_cache: Dict[int, Tuple[Any, Dict[str, Any]]] = {}
        self._roi_match_lock = threading.Lock()

    @property
    def detector_supports_roi(self) -> bool:
        return self._det_supports_roi

    # ------------------------------------------------------------------ API --
    def list_algorithms(self) -> Dict[str, str]:
        return self._det.list_algorithms()

    def available(self) -> set:
        return set(self._available)

    def load_image(self, path: str) -> Optional[np.ndarray]:
        return self._imread(str(path))

    def run(
        self,
        template: ImageInput,
        test: ImageInput,
        tasks: List[Dict[str, Any]],
        auto_save: bool = True,
        sessions_dir: Optional[Path] = None,
        progress_cb=None,
    ) -> RunSummary:
        """执行检测。

        参数:
            template/test: 图片路径或 ndarray（内部只加载一次）。
            tasks: 任务列表，每项 ``{"defect_code", "config", "roi"(ROI|None), "roi_name"}``。
            progress_cb: 可选回调 ``fn(done:int, total:int, text:str)``。
        """
        t_all = time.perf_counter()
        tpl = template if isinstance(template, np.ndarray) else self._imread(str(template))
        tst = test if isinstance(test, np.ndarray) else self._imread(str(test))
        if tpl is None or tst is None:
            summary = RunSummary()
            summary.items.append(RunItemResult(
                "", "", "整图", "ERROR", error="模板图或待检图加载失败"))
            summary.err_count = 1
            return summary

        summary = RunSummary()
        cache: Dict[str, RunItemResult] = {}
        # 同一缺陷类型只保留最先检出的一条：命中后记入此集合，
        # 该类型后续的其他 ROI/位置直接跳过，不再重复检测。
        ng_defect_seen: set = set()
        match_cache = self._prepare_roi_matches(tpl, tst, tasks)
        summary.matched_rois = [
            self._matched_roi_model(roi, match_cache[roi.rid])
            for roi in self._unique_rois(tasks)
        ]
        total = len(tasks)
        for i, task in enumerate(tasks):
            code = task["defect_code"]
            cfg = task.get("config") or {}
            roi: Optional[ROI] = task.get("roi")
            roi_name = task.get("roi_name") or ("整图" if roi is None else roi.name)
            meta = DEFECT_BY_CODE.get(code)
            defect_name = meta.name if meta else code

            if code in ng_defect_seen:
                if progress_cb:
                    progress_cb(i, total, f"{defect_name} · 已检出，跳过 {roi_name}")
                continue

            if progress_cb:
                progress_cb(i, total, f"{defect_name} · {roi_name}")

            match = match_cache.get(roi.rid) if roi is not None else None
            if (roi is not None and match and match.get("matched")
                    and code in _ORIENTATION_CODES):
                active_code = ("component_reverse_polarity" if match.get("branch_180")
                               else "component_shift")
                if code != active_code:
                    # 结构匹配已确定该 ROI 属于普通方向或180°方向之一：
                    # 移位/极反只保留与该方向对应的一项结果，另一项直接判 OK，不重复检测。
                    item = RunItemResult(code, defect_name, roi_name, "OK", [], 0.0, None)
                    summary.items.append(item)
                    self._tally(summary, item)
                    continue
            key = self._task_key(code, cfg, roi, match)
            if key in cache:
                cached = cache[key]
                item = RunItemResult(code, defect_name, roi_name, cached.status,
                                     list(cached.defects), cached.cost_ms, cached.error)
            else:
                item = self._run_one(
                    code, defect_name, roi_name, tpl, tst, cfg, roi, match)
                cache[key] = item

            summary.items.append(item)
            self._tally(summary, item)
            if item.status == "NG" and item.defects:
                ng_defect_seen.add(code)

        summary.total_ms = (time.perf_counter() - t_all) * 1000.0
        if progress_cb:
            progress_cb(total, total, "完成")

        if auto_save:
            try:
                summary.session_dir = self._auto_save(tpl, tst, tasks, summary, sessions_dir)
            except Exception as exc:  # 保存失败不影响结果返回
                summary.items.append(RunItemResult(
                    "", "", "保存", "ERROR", error=f"自动保存失败: {exc}"))
        return summary

    # -------------------------------------------------------------- internal --
    @staticmethod
    def _unique_rois(tasks) -> List[ROI]:
        out: List[ROI] = []
        seen = set()
        for task in tasks:
            roi = task.get("roi")
            if roi is not None and roi.rid not in seen:
                seen.add(roi.rid)
                out.append(roi)
        return out

    def _prepare_roi_matches(self, tpl, tst, tasks) -> Dict[int, Dict[str, Any]]:
        """每个 ROI 只做一次结构搜索；若界面画框/挪框时已提前算过，直接复用缓存。"""
        return {roi.rid: self._match_for_roi(tpl, tst, roi) for roi in self._unique_rois(tasks)}

    def compute_roi_match(self, template: ImageInput, test: ImageInput, roi: ROI) -> MatchedROI:
        """对单个 ROI 立即做一次结构匹配（整图优先，回退原位），供画框/挪框后的即时预览调用。

        结果按 (图像对象, ROI 几何) 做指纹缓存；随后运行检测时 :meth:`run` 会直接
        复用同一份缓存，不会重复搜索。
        """
        tpl = template if isinstance(template, np.ndarray) else self._imread(str(template))
        tst = test if isinstance(test, np.ndarray) else self._imread(str(test))
        if tpl is None or tst is None:
            return self._matched_roi_model(roi, {"matched": False, "reason": "image_load_failed"})
        match = self._match_for_roi(tpl, tst, roi)
        return self._matched_roi_model(roi, match)

    @staticmethod
    def _roi_fingerprint(tpl, tst, roi: ROI):
        return (id(tpl), tpl.shape, id(tst), tst.shape, roi.shape,
                round(float(roi.x), 1), round(float(roi.y), 1),
                round(float(roi.w), 1), round(float(roi.h), 1))

    def _match_for_roi(self, tpl, tst, roi: ROI) -> Dict[str, Any]:
        fp = self._roi_fingerprint(tpl, tst, roi)
        with self._roi_match_lock:
            cached = self._roi_match_cache.get(roi.rid)
            if cached is not None and cached[0] == fp:
                return cached[1]
        engine_roi = roi.to_engine_roi()
        tpl_crop, _, _, _ = apply_roi(tpl, tst, engine_roi)
        try:
            match = self._match_roi_structure(
                tpl_crop, tst, (roi.x, roi.y, roi.w, roi.h), search_expand=1.0)
        except Exception as exc:
            match = {
                "matched": False, "confidence": 0.0, "score": 0.0,
                "normal_score": 0.0, "reverse_score": 0.0,
                "orientation_confident": False, "reversed_180": False,
                "branch_180": False, "angle_deg": 0.0,
                "residual_angle_deg": 0.0, "dx": 0.0, "dy": 0.0,
                "polygon": [], "template_to_test": None,
                "crop_to_test": None,
                "reason": f"matcher_error:{type(exc).__name__}",
            }
        with self._roi_match_lock:
            self._roi_match_cache[roi.rid] = (fp, match)
        return match

    def forget_roi_match(self, rid: int):
        """ROI 被删除/清空时丢弃其匹配缓存。"""
        with self._roi_match_lock:
            self._roi_match_cache.pop(rid, None)

    @staticmethod
    def _matched_roi_model(roi: ROI, match: Dict[str, Any]) -> MatchedROI:
        polygon = [
            (float(point[0]), float(point[1]))
            for point in (match.get("polygon") or [])
        ]
        return MatchedROI(
            rid=roi.rid, name=roi.name,
            matched=bool(match.get("matched")),
            fallback=not bool(match.get("matched")),
            polygon=polygon,
            confidence=float(match.get("confidence", 0.0)),
            score=float(match.get("score", 0.0)),
            angle_deg=float(match.get("angle_deg", 0.0)),
            residual_angle_deg=float(match.get("residual_angle_deg", 0.0)),
            dx=float(match.get("dx", 0.0)),
            dy=float(match.get("dy", 0.0)),
            reversed_180=bool(match.get("reversed_180")),
            normal_score=float(match.get("normal_score", 0.0)),
            reverse_score=float(match.get("reverse_score", 0.0)),
            reason=str(match.get("reason", "")),
        )

    @staticmethod
    def _map_matched_box(d, transform, image_shape):
        """把摆正裁图内的轴对齐缺陷框映射回待检整图轴对齐外接框。"""
        x0, y0 = float(d.x), float(d.y)
        x1, y1 = x0 + float(d.width), y0 + float(d.height)
        corners = np.array(
            [[[x0, y0]], [[x1, y0]], [[x1, y1]], [[x0, y1]]],
            dtype=np.float32)
        mapped = cv2.transform(corners, transform.astype(np.float32)).reshape(-1, 2)
        min_xy = mapped.min(axis=0)
        max_xy = mapped.max(axis=0)
        ih, iw = image_shape[:2]
        bx0 = max(0, min(iw - 1, int(np.floor(min_xy[0]))))
        by0 = max(0, min(ih - 1, int(np.floor(min_xy[1]))))
        bx1 = max(bx0 + 1, min(iw, int(np.ceil(max_xy[0]))))
        by1 = max(by0 + 1, min(ih, int(np.ceil(max_xy[1]))))
        return bx0, by0, bx1 - bx0, by1 - by0

    def _run_one(self, code, defect_name, roi_name, tpl, tst, cfg, roi,
                 match=None) -> RunItemResult:
        if code not in self._available:
            return RunItemResult(code, defect_name, roi_name, "ERROR",
                                 error=f"算法未加载: {code}")
        engine_roi = roi.to_engine_roi() if roi is not None else None
        off_x = off_y = 0
        tpl_in, tst_in = tpl, tst
        matched_transform = None
        run_cfg = dict(cfg)

        if engine_roi and match and match.get("matched"):
            # 模板仍使用人工 ROI；待检图使用结构匹配得到的区域并摆正。
            tpl_in, _, _, _ = apply_roi(tpl, tst, engine_roi)
            tst_in = self._extract_matched_roi(
                tst, match, (tpl_in.shape[1], tpl_in.shape[0]))
            if tst_in is None:
                match = dict(match)
                match["matched"] = False
            else:
                matched_transform = np.asarray(match["crop_to_test"], np.float64)
                engine_roi = None
                # 只透传标量/布尔量，避免配置中出现 ndarray。
                run_cfg["_roi_match"] = {
                    key: match.get(key)
                    for key in (
                        "matched", "confidence", "score", "normal_score",
                        "reverse_score", "orientation_confident",
                        "reversed_180", "branch_180", "angle_deg",
                        "residual_angle_deg", "dx", "dy", "reason")
                }

        if engine_roi and matched_transform is None and not self._det_supports_roi:
            # 兼容旧版算法包：应用侧裁剪 ROI，坐标稍后偏移回整图
            tpl_in, tst_in, off_x, off_y = apply_roi(tpl, tst, engine_roi)
            engine_roi = None

        if self._det_supports_roi:
            r = self._det.detect(code, tpl_in, tst_in, config=run_cfg, roi=engine_roi)
        else:
            r = self._det.detect(code, tpl_in, tst_in, config=run_cfg)

        if r.has_error:
            return RunItemResult(code, defect_name, roi_name, "ERROR",
                                 cost_ms=r.cost_time * 1000.0, error=r.error)
        boxes: List[DefectBox] = []
        for d in r.defects:
            if matched_transform is not None:
                bx, by, bw, bh = self._map_matched_box(
                    d, matched_transform, tst.shape)
            else:
                bx, by, bw, bh = (
                    int(d.x) + off_x, int(d.y) + off_y,
                    int(d.width), int(d.height))
            boxes.append(DefectBox(
                label=d.label, defect_code=code,
                x=bx, y=by, width=bw, height=bh,
                confidence=float(d.confidence), description=d.description,
                roi_name=roi_name,
            ))
            # 每种缺陷类型每次调用最多保留 1 条，避免同类型重复出现在结果里。
            break
        return RunItemResult(code, defect_name, roi_name, r.status, boxes,
                             r.cost_time * 1000.0, None)

    @staticmethod
    def _tally(summary: RunSummary, item: RunItemResult):
        if item.status == "NG":
            summary.ng_count += 1
        elif item.status == "OK":
            summary.ok_count += 1
        else:
            summary.err_count += 1

    @staticmethod
    def _task_key(code: str, cfg: Dict[str, Any], roi: Optional[ROI],
                  match=None) -> str:
        payload = {
            "code": code,
            "cfg": {k: cfg[k] for k in sorted(cfg)},
            "roi": roi.to_engine_roi() if roi is not None else None,
            "match": ({
                "matched": bool(match.get("matched")),
                "angle": round(float(match.get("angle_deg", 0.0)), 3),
                "dx": round(float(match.get("dx", 0.0)), 3),
                "dy": round(float(match.get("dy", 0.0)), 3),
            } if match else None),
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    # ----------------------------------------------------------- auto save ----
    def render_overlay(self, test_img: np.ndarray, summary: RunSummary) -> np.ndarray:
        """在待检图上叠加所有缺陷框（用于保存与预览）。"""
        canvas = test_img.copy()
        if canvas.ndim == 2:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        for matched in summary.matched_rois:
            if not matched.matched or len(matched.polygon) < 3:
                continue
            points = np.asarray(matched.polygon, np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [points], True, (94, 197, 34), 2, cv2.LINE_AA)
            x, y = points.reshape(-1, 2).min(axis=0)
            tag = f"matched {matched.score:.2f} {matched.angle_deg:.1f}deg"
            if matched.reversed_180:
                tag += " 180"
            cv2.putText(
                canvas, tag, (int(x), max(12, int(y) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (94, 197, 34), 1, cv2.LINE_AA)
        for d in summary.all_defects:
            meta = DEFECT_BY_CODE.get(d.defect_code)
            color = _hex_to_bgr(meta.color) if meta else (0, 0, 255)
            cv2.rectangle(canvas, (d.x, d.y), (d.x + d.width, d.y + d.height), color, 2)
            tag = f"{d.defect_code.replace('component_', '')} {d.confidence:.2f}"
            yt = max(0, d.y - 6)
            cv2.putText(canvas, tag, (d.x, yt), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, color, 1, cv2.LINE_AA)
        return canvas

    def _auto_save(self, tpl, tst, tasks, summary: RunSummary,
                   sessions_dir: Optional[Path]) -> str:
        root = Path(sessions_dir) if sessions_dir else app_config.DEFAULT_SESSIONS_DIR
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        sess = root / stamp
        sess.mkdir(parents=True, exist_ok=True)

        overlay = self.render_overlay(tst, summary)
        cv2.imwrite(str(sess / "result_overlay.png"), overlay)
        cv2.imwrite(str(sess / "template.png"), tpl)
        cv2.imwrite(str(sess / "test.png"), tst)

        report = {
            "time": stamp,
            "summary": {
                "total_ms": round(summary.total_ms, 2),
                "ok": summary.ok_count, "ng": summary.ng_count, "error": summary.err_count,
            },
            "matched_rois": [m.to_dict() for m in summary.matched_rois],
            "tasks": [
                {
                    "defect_code": t["defect_code"],
                    "roi": (t["roi"].to_dict() if t.get("roi") is not None else None),
                    "config": t.get("config") or {},
                }
                for t in tasks
            ],
            "items": [
                {
                    "defect_code": it.defect_code, "defect_name": it.defect_name,
                    "roi_name": it.roi_name, "status": it.status,
                    "cost_ms": round(it.cost_ms, 2), "error": it.error,
                    "defects": [d.to_dict() for d in it.defects],
                }
                for it in summary.items
            ],
        }
        with open(sess / "result.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        return str(sess)
