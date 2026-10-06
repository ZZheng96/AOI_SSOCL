"""PCB 表皮起皱检测算法控件。"""
from __future__ import annotations

import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from core.entities.algorithms import IDetectionAlgorithm
from core.entities.detect import AlgorithmResult, BoundingBox, DetectPartBox, ResultType
from core.utils.alg_config import merge_config

from . import algo_trad


class SurfaceWrinkleAlg(IDetectionAlgorithm):
    """单图起皱检测（免模板）。"""

    algorithm_code = "surface_wrinkle"
    version = 1

    def __init__(self, config_path: str | None = None):
        super().__init__(config_path)

    @classmethod
    def default_config(cls) -> Dict[str, Any]:
        return dict(algo_trad.DEFAULT_CONFIG)

    def run(
        self,
        image: np.ndarray,
        config: Dict[str, Any],
        roi_bbox: Optional[BoundingBox] = None,
        original_template_image: Optional[np.ndarray] = None,
        multiple_roi_images: Optional[List[np.ndarray]] = None,
    ) -> AlgorithmResult:
        t0 = time.perf_counter()
        del original_template_image, multiple_roi_images
        try:
            if image is None or getattr(image, "size", 0) == 0:
                return self._fail_result("empty image", t0)

            cfg = merge_config(self._defaults, config)
            work, off_x, off_y = self._apply_roi(image, roi_bbox)
            raw = algo_trad.run(work, config=cfg)

            parts: List[DetectPartBox] = []
            if raw["status"] == "NG":
                for b in raw.get("boxes") or []:
                    parts.append(
                        DetectPartBox(
                            x=int(b["x"]) + off_x,
                            y=int(b["y"]) + off_y,
                            width=int(b["width"]),
                            height=int(b["height"]),
                            confidence=float(b["confidence"]),
                            box_type="defect",
                            label="wrinkle",
                            metadata={"area_px": b.get("area_px", 0)},
                        )
                    )

            vis = self._draw(
                work,
                raw.get("boxes") or [],
                status=str(raw["status"]),
                score=float(raw["score"]),
            )
            return AlgorithmResult(
                code=0,
                message="ok",
                algorithm_code=self.algorithm_code,
                cost_time=time.perf_counter() - t0,
                result_type=ResultType.PARTS,
                parts=parts,
                metadata={
                    "status": raw["status"],
                    "wrinkle_score": float(raw["score"]),
                    "num_defects": len(parts),
                    "image_shape": [int(image.shape[0]), int(image.shape[1])],
                    "output_image": vis,
                    "roi_bbox": self._bbox_dict(roi_bbox),
                    **(raw.get("extra") or {}),
                },
            )
        except Exception as exc:
            return self._fail_result(str(exc), t0, exc=exc)

    @staticmethod
    def _apply_roi(
        image: np.ndarray,
        roi_bbox: Optional[BoundingBox],
    ) -> Tuple[np.ndarray, int, int]:
        if roi_bbox is None:
            return image, 0, 0
        h, w = image.shape[:2]
        x = max(0, int(roi_bbox.x))
        y = max(0, int(roi_bbox.y))
        x2 = min(w, x + max(0, int(roi_bbox.width)))
        y2 = min(h, y + max(0, int(roi_bbox.height)))
        if x2 - x < 2 or y2 - y < 2:
            return image, 0, 0
        return image[y:y2, x:x2].copy(), x, y

    @staticmethod
    def _draw_scale(h: int, w: int) -> Tuple[int, float, int]:
        short = max(1, min(h, w))
        thick = max(6, min(48, int(round(short * 0.01))))
        font_scale = max(1.0, min(4.0, short / 600.0))
        font_thick = max(1, min(3, int(round(font_scale))))
        return thick, font_scale, font_thick

    @classmethod
    def _draw(
        cls,
        image: np.ndarray,
        boxes: List[Dict[str, Any]],
        status: str = "",
        score: Optional[float] = None,
        color: Tuple[int, int, int] = (0, 140, 255),
    ) -> np.ndarray:
        vis = image.copy()
        if vis.ndim == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        h, w = vis.shape[:2]
        thick, font_scale, font_thick = cls._draw_scale(h, w)
        base_inset = max(thick, 4)

        for b in boxes:
            x, y, bw, bh = int(b["x"]), int(b["y"]), int(b["width"]), int(b["height"])
            coverage = (bw * bh) / float(max(1, w * h))
            inset = max(base_inset, int(min(h, w) * 0.025)) if coverage > 0.75 else base_inset
            x1, y1 = max(inset, x), max(inset, y)
            x2, y2 = min(w - 1 - inset, x + bw), min(h - 1 - inset, y + bh)
            if x2 <= x1 or y2 <= y1:
                continue
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 0), thick + 6, cv2.LINE_AA)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thick, cv2.LINE_AA)
            tag = f"wrinkle {float(b['confidence']):.2f}"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
            pad = max(6, thick)
            tx, ty = x1 + pad, y1 + th + pad
            cv2.rectangle(
                vis,
                (tx - pad // 2, ty - th - pad // 2),
                (tx + tw + pad // 2, ty + pad // 2),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                vis, tag, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, font_thick, cv2.LINE_AA,
            )

        if status:
            banner = f"{status}" + (f"  score={score:.3f}" if score is not None else "")
            (tw, th), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
            pad = max(10, thick)
            cv2.rectangle(vis, (0, 0), (tw + pad * 2, th + pad * 2), (0, 0, 0), -1)
            banner_color = (0, 0, 255) if status == "NG" else (40, 200, 40)
            cv2.putText(
                vis, banner, (pad, th + pad), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, banner_color, font_thick, cv2.LINE_AA,
            )
        return vis

    def _fail_result(
        self,
        message: str,
        t0: float,
        exc: Optional[BaseException] = None,
    ) -> AlgorithmResult:
        meta: Dict[str, Any] = {"error": message}
        if exc is not None:
            meta["traceback"] = traceback.format_exc()
        return AlgorithmResult(
            code=1,
            message=message,
            algorithm_code=self.algorithm_code,
            cost_time=time.perf_counter() - t0,
            result_type=ResultType.ERROR,
            metadata=meta,
        )

    @staticmethod
    def _bbox_dict(bbox: Optional[BoundingBox]) -> Optional[Dict[str, Any]]:
        if bbox is None:
            return None
        return {
            "x": bbox.x, "y": bbox.y,
            "width": bbox.width, "height": bbox.height,
        }
