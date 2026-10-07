"""PCB 表皮起皱检测适配器：包装 `alg_repo/wrinkle` 的
`SurfaceWrinkleAlg`（免模板单图算法，来自 reference5/UI算法/wrinkle_detector）。

免模板：不依赖金样板，直接对输入图做 纹理响应 + 弯曲度筛选 + 占比判定。
"""
from __future__ import annotations

import importlib
import threading
from typing import Any

from app.config import WRINKLE_REPO_DIR
from app.detect.adapters.base import BaseAdapter, ParamSpec
from app.detect.adapters.path_bootstrap import IsolatedImport
from app.detect.contract import ModuleManifest
from app.detect.types import AlgorithmResult, DefectBox, DetectRequest

MANIFEST = ModuleManifest(
    id="wrinkle_起皱",
    display_name="起皱",
    group_id="wrinkle",
    group_name="表皮起皱检测",
    requires_standard=False,
    region_kind="none",
    color="#0ea5e9",
    algorithm_version="1.0.0",
)

_PARAM_SPECS = [
    ParamSpec(key="wrinkle_score_thresh", label="起皱占比阈值", kind="float",
              default=0.25, minimum=0.0, maximum=1.0, step=0.01,
              hint="起皱像素占比超过该值判 NG"),
    ParamSpec(key="lap_thresh", label="纹理响应阈值", kind="float",
              default=5.0, minimum=0.0, maximum=50.0, step=0.5,
              hint="Laplacian 纹理响应阈值"),
    ParamSpec(key="curvature_ratio_min", label="弯曲度下限", kind="float",
              default=0.05, minimum=0.0, maximum=1.0, step=0.01),
    ParamSpec(key="resize_width", label="分析长边", kind="int",
              default=1400, minimum=200, maximum=4000, step=50),
    ParamSpec(key="box_display_mode", label="出框模式", kind="enum",
              default="region", choices=["region", "line"],
              hint="region=零散区域框（闭运算+邻近合并）；line=每一条皱纹单独框"),
    ParamSpec(key="dilate_px", label="闭运算核", kind="int",
              default=40, minimum=0, maximum=200, step=2),
    ParamSpec(key="merge_gap_px", label="邻近合并间隙", kind="int",
              default=50, minimum=0, maximum=500, step=5),
    ParamSpec(key="max_defect_boxes", label="最多输出框数", kind="int",
              default=4, minimum=1, maximum=64, step=1),
]


def _load_wrinkle() -> dict[str, Any]:
    with IsolatedImport([WRINKLE_REPO_DIR / "src"], ["algorithms", "core", "utils"]):
        mod = importlib.import_module("algorithms.surface_wrinkle.surface_wrinkle_alg")
        alg = mod.SurfaceWrinkleAlg()
    return {"instance": alg, "path": WRINKLE_REPO_DIR}


_wrinkle_cache: dict[str, Any] | None = None
_wrinkle_lock = threading.Lock()


class WrinkleAdapter(BaseAdapter):
    algorithm_ids = ["wrinkle_起皱"]
    requires_standard = False

    def __init__(self) -> None:
        global _wrinkle_cache
        if _wrinkle_cache is None:
            with _wrinkle_lock:
                if _wrinkle_cache is None:
                    _wrinkle_cache = _load_wrinkle()

    def manifests(self) -> list[ModuleManifest]:
        return [MANIFEST]

    def param_specs(self, algorithm_id: str) -> list[ParamSpec]:
        return list(_PARAM_SPECS)

    def is_ready(self, algorithm_id, template_name=None, rois=None) -> tuple[bool, str]:
        _ = algorithm_id, template_name, rois
        return True, ""

    def run(self, algorithm_id: str, req: DetectRequest, params: dict[str, Any]) -> AlgorithmResult:
        wrinkle = _wrinkle_cache
        test_bgr = req.image_test_raw if req.image_test_raw is not None else req.image_test
        std_bgr = req.image_std_raw if req.image_std_raw is not None else req.image_std
        del std_bgr  # 免模板
        cfg: dict[str, Any] = dict(wrinkle["instance"].default_config())
        cfg.update(params or {})
        result = wrinkle["instance"].run(test_bgr, cfg)
        ok = result.code == 0 and result.metadata.get("status") != "NG"
        boxes = [
            DefectBox(
                x=int(p.x), y=int(p.y), w=int(p.width), h=int(p.height),
                label="起皱", score=float(getattr(p, "confidence", 0.0) or 0.0),
            )
            for p in (result.parts or [])
        ]
        msg = result.message or ""
        extra = dict(result.metadata or {})
        extra.pop("output_image", None)
        score = extra.get("wrinkle_score", "")
        thr = cfg.get("wrinkle_score_thresh", "")
        score_s = f"{score:.4f}" if isinstance(score, float) else str(score)
        if result.code != 0:
            message = f"[错误] 起皱: {msg}"
        else:
            # 库在正常完成时 message 恒为 "ok"，NG 时须展示判定依据而非该字样
            message = (f"起皱判定={'OK' if ok else 'NG'} 占比={score_s} 阈值={thr}"
                       f"，缺陷数={len(boxes)}")
        return AlgorithmResult(
            algorithm=algorithm_id,
            ok=ok,
            message=message,
            boxes=boxes,
            defect_count=len(boxes),
            elapsed_ms=int(result.cost_time * 1000) if result.cost_time else 0,
            hit_layer="外部算法仓库(alg_repo/wrinkle)",
            status="OK" if ok else "NG",
            skipped=False,
        )
