"""独立 DINO 配对热图接口；不进入生产 Detection/DB/模型指针链。"""
from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from algo.backbone.dino import FrozenDINO
from algo.common.io import load_image
from algo.slots.tpl import TplSlot
from ..core.config import get_settings
from ..pipeline.service import imwrite, render_heatmap_overlay, _save_npy

router = APIRouter()
_pair_lock = threading.Lock()
_pair_backbone = None
_pair_slot = None
_pair_template_sig = None
_LOCAL_WEIGHTS = Path(r"D:/CGAIC/assets/dinov2/dinov2_vits14_pretrain.pth")


class PairHeatmapRequest(BaseModel):
    path: str
    pair_template: str
    same_view_confirmed: bool


def _pair_engine(template_path: str):
    global _pair_backbone, _pair_slot, _pair_template_sig
    sig = (template_path, Path(template_path).stat().st_mtime_ns,
           Path(template_path).stat().st_size)
    if _pair_backbone is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用；独立 DINO 配对接口不自动回退 CPU")
        local_dir = _LOCAL_WEIGHTS.parent
        if (not (local_dir / "hubconf.py").is_file()
                or not _LOCAL_WEIGHTS.is_file()):
            raise RuntimeError(f"本地 DINO vendor 不完整: {local_dir}（需含 hubconf.py 与 {_LOCAL_WEIGHTS.name}）")
        _pair_backbone = FrozenDINO(model_name="vits14", output_layer=9,
                                    grid=32, device="cuda", local_dir=str(local_dir),
                                    allow_download=False)
        _pair_slot = TplSlot({"pair_mode": True, "auto_from_normal": False,
                              "align": False, "topk_ratio": .1,
                              "pix_weight": 0.0, "feat_weight": 1.0},
                             _pair_backbone, device="cuda")
    if _pair_template_sig != sig:
        _pair_slot._pair_cache.clear()
        _pair_template_sig = sig
    return _pair_backbone, _pair_slot


def _bad_path(path: str, label: str) -> Path:
    p = Path(path)
    if not p.is_file():
        raise HTTPException(status_code=400, detail=f"{label} 文件不存在或无效: {path}")
    return p


@router.post("/detect/pair-heatmap")
def api_pair_heatmap(req: PairHeatmapRequest):
    if req.same_view_confirmed is not True:
        raise HTTPException(status_code=400,
                            detail="必须显式确认 same_view_confirmed=true；同位确认不等于算法验证")
    image_path = _bad_path(req.path, "输入图像")
    template_path = _bad_path(req.pair_template, "配对模板")
    try:
        image = load_image(image_path)
        template = load_image(template_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"图像不可读取: {exc}") from exc
    if image.shape[:2] != template.shape[:2]:
        raise HTTPException(status_code=400, detail="输入图像与配对模板尺寸不符")
    started = time.perf_counter()
    try:
        with _pair_lock:
            backbone, slot = _pair_engine(str(template_path))
            feats = backbone.extract_tiles([image])
            raw_score, _tiles, heatmaps = slot.score_image(
                image, feats=feats, pair_template=str(template_path))
            raw = np.asarray(heatmaps[0], dtype=np.float32)
            settings = get_settings()
            out_dir = settings.storage("heatmaps")
            token = uuid.uuid4().hex
            raw_path = out_dir / f"pair_{token}_raw.npy"
            heat_path = out_dir / f"pair_{token}_heat.jpg"
            overlay_path = out_dir / f"pair_{token}_overlay.jpg"
            if not _save_npy(raw_path, raw):
                raise RuntimeError("原始热图写盘失败")
            h, w = image.shape[:2]
            heat_color, overlay_bgr, _boxes = render_heatmap_overlay(
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR), raw, engine_boxes=[])
            if not imwrite(heat_path, heat_color) or not imwrite(overlay_path, overlay_bgr):
                raise RuntimeError("热图产物写盘失败")
            return {
                "status": "ok", "same_view_confirmed": True,
                "aux_pair": {"visual_only": True, "source": "dino_tpl_pair",
                              "calibrated": False, "alignment": False,
                              "coord": "input_image", "raw_score": float(raw_score),
                              "raw_shape": [int(v) for v in raw.shape],
                              "image_size": [int(w), int(h)],
                              "template_path": str(template_path),
                              "score_semantics": "raw DINO/template distance, not probability",
                              "heatmap_normalization": "per-image min-max for display; constant maps are all zero"},
                "heatmap_path": str(heat_path),
                "overlay_path": str(overlay_path),
                "raw_path": str(raw_path),
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            }
    except HTTPException:
        raise
    except RuntimeError as exc:
        if "CUDA 不可用" in str(exc):
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        raise HTTPException(status_code=500, detail=f"独立配对热图失败: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"独立配对热图失败: {exc}") from exc
