"""贴片锡焊检测适配器：包装 `Dev_solder_smt_Alg` 仓库的
`SolderSmtAllAlg`（一次算法覆盖 多锡/少锡/连锡/虚焊 四类缺陷）。

该算法必须提供 `pad_frames`（标准图上的焊盘矩形框），没有标定时直接跳过，
不强行瞎跑。标定通过 :mod:`app.calibration.smt_pads` 按模板名保存。

参数分两层（对应仓库 `docs/params.md` 里"现场真正需要调"的一批，深层内部
参数如 W校正LUT分段、裂纹Hough检测细节等不暴露，保持默认即可）：

- ``SMT_SHARED_ID``（"元件焊锡检测 - 共用参数"）：焊锡颜色提取 HSV、焊盘外扩
  比例、干扰/焊盘最小面积、自动缩放——这些是"干扰排除→焊盘定位→焊锡提取"
  这条共用流水线上的参数，四类缺陷各自调用同一份底层提取逻辑，所以只在这
  一个入口调一次，四类缺陷同时生效，不需要分别调 4 遍。
- 四个缺陷类型（``solder_多锡`` 等）：只保留各自缺陷判定这一步专属的阈值。

调用真实检测时（:func:`resolve_smt_params`），先解出共用层的生效值作为
基线，再叠加该缺陷自己的参数覆盖，合并后一次性传给算法。
"""
from __future__ import annotations

import importlib
import os
import threading
from typing import Any

from app.calibration.smt_pads import get_smt_pads, get_smt_rim, get_smt_toe
from app.calibration.store import load_calib
from app.config import SMT_REPO_DIR
from app.detect.adapters.base import BaseAdapter, ParamSpec
from app.detect.adapters.path_bootstrap import IsolatedImport
from app.detect.contract import ModuleManifest, Roi
from app.detect.roi import rects_of
from app.detect.types import AlgorithmResult, DefectBox, DetectRequest

SMT_SHARED_ID = "solder_共用参数"

DETECT_FLAG_BY_ALG: dict[str, str] = {
    "solder_多锡": "detect_excess",
    "solder_少锡": "detect_insufficient",
    "solder_连锡": "detect_bridge",
    "solder_虚焊": "detect_cold_solder",
}

LABEL_KEY_BY_ALG: dict[str, str] = {
    "solder_多锡": "excess",
    "solder_少锡": "insufficient",
    "solder_连锡": "bridge",
    "solder_虚焊": "cold_solder",
}

LABEL_CN_BY_ALG: dict[str, str] = {
    "solder_多锡": "多锡(包锡)",
    "solder_少锡": "少锡",
    "solder_连锡": "连锡",
    "solder_虚焊": "虚焊",
}

# label_key（算法内部输出的英文标签）-> 中文缺陷名，供"共用参数预览"合并四类结果时使用。
_LABEL_CN_BY_KEY: dict[str, str] = {key: LABEL_CN_BY_ALG[alg] for alg, key in LABEL_KEY_BY_ALG.items()}

# 布尔参数在 UI 上用 enum("true"/"false") 表示，传给外部算法前需要转换回真正的 bool。
_BOOL_KEYS = {"insufficient_use_diff", "enable_auto_resize"}


def _coerce_bools(cfg: dict[str, Any]) -> None:
    for key in _BOOL_KEYS:
        if key in cfg and isinstance(cfg[key], str):
            cfg[key] = cfg[key].strip().lower() == "true"


# 共用层：干扰过滤最小面积 + 焊盘定位外扩 + 焊锡颜色提取 HSV + 自动缩放。
# 四类缺陷调用同一条提取流水线，这里调一次，四类同时生效。
_SHARED_SPECS: list[ParamSpec] = [
    ParamSpec("solder_blue_h_low", "焊锡蓝色 HSV-H 下限", "int", 85, 0, 179, 1,
              "焊锡金属光泽的颜色范围下限。不同批次/光照下焊锡颜色可能偏移，需按实际图片调整。"),
    ParamSpec("solder_blue_h_high", "焊锡蓝色 HSV-H 上限", "int", 115, 0, 179, 1,
              "焊锡颜色范围上限，配合下限框定“蓝色焊锡”的色调区间。"),
    ParamSpec("solder_blue_s_min", "焊锡蓝色 HSV-S 下限", "int", 60, 0, 255, 1,
              "饱和度下限，调低可识别更浅/偏灰的焊锡，但可能误识别背景。"),
    ParamSpec("solder_blue_v_min", "焊锡蓝色 HSV-V 下限", "int", 150, 0, 255, 1,
              "亮度下限，调低可识别较暗的焊锡，但可能引入暗部噪声。"),
    ParamSpec("pad_expand_ratio", "焊盘外扩比例（长边）", "float", 0.30, 0.0, 1.0, 0.01,
              "判定区域相对焊盘框向外扩展的比例。调大：判定范围更大，更容易报多锡/连锡。"),
    ParamSpec("pad_expand_short_ratio", "焊盘外扩比例（短边）", "float", 0.30, 0.0, 1.0, 0.01,
              "短边方向的外扩比例，配合长边外扩比例适配长宽比差异较大的焊盘。"),
    ParamSpec("min_pad_area", "焊盘最小面积（像素）", "int", 20, 0, 2000, 1,
              "小于该面积的候选焊盘框会被忽略，防止把噪点误识别为焊盘。"),
    ParamSpec("min_interference_area", "干扰区最小面积（像素）", "int", 30, 0, 500, 1,
              "阻焊层/丝印/元件干扰区域小于该面积会被忽略，避免过度抠除。"),
    ParamSpec("enable_auto_resize", "启用自动缩放", "enum", "true", choices=["true", "false"],
              hint="大图先等比缩小再检测，提速明显；关闭可能提高少锡检出灵敏度但更慢。"),
    ParamSpec("auto_resize_max_size", "自动缩放目标边长", "int", 400, 100, 1200, 10,
              "触发缩放后，图片长边缩小到该像素值以内。"),
    ParamSpec("auto_resize_threshold", "自动缩放触发阈值", "int", 450, 100, 1600, 10,
              "图片宽或高超过该像素值时才触发自动缩放。"),
]

_PARAM_SPECS: dict[str, list[ParamSpec]] = {
    SMT_SHARED_ID: _SHARED_SPECS,
    "solder_多锡": [
        ParamSpec("excess_thresh", "多锡阈值", "float", 0.15, 0.0, 1.0, 0.01,
                  "焊盘外焊锡占中位焊盘面积的比例超过此值判多锡。调大：更少报；调小：更多报。"),
        ParamSpec("excess_min_area", "多锡碎片最小面积", "int", 5, 0, 200, 1,
                  "小于该像素面积的多锡候选斑块直接忽略，过滤掉噪点碎片。"),
    ],
    "solder_少锡": [
        ParamSpec("insufficient_thresh", "少锡阈值", "float", 0.09, 0.0, 1.0, 0.01,
                  "焊盘框内焊锡覆盖率低于此值判定为少锡。调大：更多报；调小：更少报。"),
        ParamSpec("diff_insufficient_thresh", "差分少锡阈值", "float", 0.05, 0.0, 0.5, 0.01,
                  "有标准图模板对比时使用的差分阈值。调大：更少报；调小：更多报。"),
        ParamSpec("insufficient_use_diff", "少锡启用模板差分辅助判定", "enum", "true",
                  choices=["true", "false"],
                  hint="启用后少锡判定会同时参考标准图差分结果，一般更准确；关闭则只用绝对覆盖率阈值。"),
    ],
    "solder_连锡": [
        ParamSpec("bridge_min_area", "连锡最小面积", "int", 20, 1, 300, 1,
                  "相邻焊盘间连通焊锡小于该像素面积时忽略。调大：更少报；调小：更多报。"),
    ],
    "solder_虚焊": [
        ParamSpec("pin_type", "引脚类型", "enum", "gull-wing", choices=["gull-wing", "terminal"],
                  hint="gull-wing=翼形引脚（走 3 规则判定）；terminal=端子类（走暗区+红胶判定）。"),
        ParamSpec("cold_solder_dark_ratio", "暗区比例阈值", "float", 0.30, 0.0, 1.0, 0.01,
                  "焊盘暗区占比超过此值参与虚焊判定。调大：更少报；调小：更多报。"),
        ParamSpec("cold_solder_v_dark", "暗区亮度阈值", "int", 80, 0, 255, 1,
                  "低于该亮度值视为“暗”。调大：更多报；调小：更少报。"),
        ParamSpec("cold_solder_rim_ratio", "边缘焊锡覆盖率阈值", "float", 0.5, 0.0, 1.0, 0.01,
                  "gull-wing 规则1：检测区内蓝色焊锡覆盖率低于此值判虚焊。"),
        ParamSpec("toe_metal_ratio_thresh", "脚尖金属覆盖率阈值", "float", 0.3, 0.0, 1.0, 0.01,
                  "gull-wing 规则2：脚尖区域金属颜色占比超过此值（配合裂纹检测）判虚焊。"),
        ParamSpec("cold_diff_ratio_thresh", "差分覆盖率阈值", "float", 0.38, 0.0, 1.0, 0.01,
                  "gull-wing 规则3：焊盘区域模板差分覆盖率超过此值（配合裂纹检测）判虚焊。"),
    ],
}

_lock = threading.Lock()
_cache: dict[str, Any] = {}


def _load_smt() -> dict[str, Any]:
    with _lock:
        if _cache:
            return _cache
        src_dir = SMT_REPO_DIR / "src"
        contract_dir = SMT_REPO_DIR / "contract_reference"
        with IsolatedImport([src_dir, contract_dir], ["core", "algorithms"]):
            alg_mod = importlib.import_module(
                "algorithms.solder_smt.composite_smt_all.solder_smt_all_alg"
            )
            _cache["klass"] = alg_mod.SolderSmtAllAlg
            _cache["instance"] = alg_mod.SolderSmtAllAlg()
        return _cache


def resolve_smt_params(param_store, template_name: str | None, algorithm_id: str) -> tuple[dict[str, Any], str]:
    """先解出"共用参数"的生效值作为基线，再叠加该缺陷自己的参数覆盖。

    ``own_resolved`` 会按 ``own_defaults`` 的键过滤一遍——旧版本参数划分
    不同（比如 pad_expand_ratio 以前是挂在具体缺陷下的），如果某个缺陷之前
    保存过现在已经挪到共用层的同名 key，过滤掉它才能保证共用层的值不被
    这种历史遗留数据悄悄覆盖。
    """
    shared_defaults = {p.key: p.default for p in _SHARED_SPECS}
    shared_resolved, shared_hit = param_store.resolve(template_name, None, SMT_SHARED_ID, shared_defaults)
    own_defaults = {p.key: p.default for p in _PARAM_SPECS.get(algorithm_id, [])}
    own_resolved, own_hit = param_store.resolve(template_name, None, algorithm_id, own_defaults)
    own_resolved = {k: v for k, v in own_resolved.items() if k in own_defaults}
    merged = {**shared_resolved, **own_resolved}
    hit = f"共用参数:{shared_hit}；本项:{own_hit}"
    return merged, hit


def _smt_pads(template_name: str | None, rois: list[Roi] | None) -> list[tuple[int, int, int, int]]:
    """焊盘框提取。

    v2 修复：矩形坐标统一用 floor 取整（int()），与算法作者在 test_images
    中用 labelme pad.json 转换的取整方式一致——算法对 pad 边界 1px 极敏感，
    round(81.77)=82 与 floor=81 会导致同一张 NG 图判定完全不同。
    """
    rects: list[tuple[int, int, int, int]] = []
    if rois:
        for roi in rois:
            if roi.layer != "smt_pads" and roi.kind != "smt":
                continue
            shape = roi.shape or {}
            if str(shape.get("shape") or "rect") != "rect":
                continue
            try:
                x, y, w, h = (float(shape[k]) for k in ("x", "y", "w", "h"))
            except (KeyError, TypeError, ValueError):
                continue
            if w > 0 and h > 0:
                rects.append((int(x), int(y), int(w), int(h)))
    if rects:
        return rects
    data = load_calib(template_name)
    for p in data.get("smt_pads") or []:
        try:
            x, y, w, h = (float(v) for v in p)
        except (TypeError, ValueError):
            continue
        if w > 0 and h > 0:
            rects.append((int(x), int(y), int(w), int(h)))
    return rects


def _smt_layer_rects(template_name: str | None, rois: list[Roi] | None, layer: str, fallback) -> list[tuple[int, int, int, int]]:
    if rois:
        found = rects_of(rois, layer=layer)
        if found:
            return found
    return fallback(template_name)


class SmtAdapter(BaseAdapter):
    algorithm_ids = [SMT_SHARED_ID, *DETECT_FLAG_BY_ALG.keys()]

    def __init__(self) -> None:
        _load_smt()

    def manifests(self) -> list[ModuleManifest]:
        items = [
            ModuleManifest(
                id=SMT_SHARED_ID,
                display_name="焊锡共用参数",
                group_id="smt_solder",
                group_name="元件焊锡检测",
                requires_standard=True,
                region_kind="smt",
                color="#ca8a04",
                algorithm_version="1.0.0",
                trigger="manual",
                is_job_item=False,
            )
        ]
        rec = ("电阻", "电容", "二极管", "LED", "IC")
        for alg_id, label in LABEL_CN_BY_ALG.items():
            items.append(
                ModuleManifest(
                    id=alg_id,
                    display_name=label.replace("(包锡)", ""),
                    group_id="smt_solder",
                    group_name="元件焊锡检测",
                    requires_standard=True,
                    region_kind="smt",
                    color="#ca8a04",
                    algorithm_version="1.0.0",
                    shared_param_id=SMT_SHARED_ID,
                    trigger="on_job",
                    default_selected=alg_id in ("solder_多锡", "solder_少锡", "solder_连锡"),
                    recommended_categories=rec,
                )
            )
        return items

    def param_specs(self, algorithm_id: str) -> list[ParamSpec]:
        return list(_PARAM_SPECS.get(algorithm_id, []))

    def is_ready(
        self,
        algorithm_id: str,
        template_name: str | None,
        rois: list[Roi] | None = None,
    ) -> tuple[bool, str]:
        pads = _smt_pads(template_name, rois)
        if not pads:
            return False, "未标定焊盘框（pad_frames），请先在“标定”中框选贴片焊盘区域"
        return True, ""

    def run(self, algorithm_id: str, req: DetectRequest, params: dict[str, Any]) -> AlgorithmResult:
        if algorithm_id == SMT_SHARED_ID:
            return self._run_shared_preview(req, params)

        flag_key = DETECT_FLAG_BY_ALG.get(algorithm_id)
        label_key = LABEL_KEY_BY_ALG.get(algorithm_id)
        label_cn = LABEL_CN_BY_ALG.get(algorithm_id, algorithm_id)
        if flag_key is None:
            return AlgorithmResult(algorithm=algorithm_id, ok=False, message=f"未知的贴片锡焊缺陷类型: {algorithm_id}")

        pads = _smt_pads(req.template_name, list(req.rois or []))
        if not pads:
            reason = "未标定焊盘框（pad_frames）"
            return AlgorithmResult(
                algorithm=algorithm_id, ok=True, skipped=True, skip_reason=reason,
                message=f"[跳过] {label_cn}：{reason}，请先标定",
            )

        std_bgr = req.image_std_raw if req.image_std_raw is not None else req.image_std
        test_bgr = req.image_test_raw if req.image_test_raw is not None else req.image_test

        smt = _load_smt()
        # v2 修复：以算法 default_config() 为底，再叠加 detect 标志与冻结参数，
        # 保证缺省键（auto_resize/morph/扩展 solder 提取参数等）行为与算法
        # 直接调用一致（PCB_Ins 原版缺 default 为底，特定图会漏检）。
        cfg: dict[str, Any] = dict(smt["instance"].default_config())
        cfg.update(
            {
                "detect_excess": False,
                "detect_insufficient": False,
                "detect_bridge": False,
                "detect_cold_solder": False,
                "pad_frames": [list(p) for p in pads],
                "enable_visualization": True,
            }
        )
        cfg[flag_key] = True
        cfg.update(params)
        _coerce_bools(cfg)

        if algorithm_id == "solder_虚焊":
            toe = _smt_layer_rects(req.template_name, list(req.rois or []), "smt_toe", get_smt_toe)
            rim = _smt_layer_rects(req.template_name, list(req.rois or []), "smt_rim", get_smt_rim)
            if toe:
                cfg["toe_frames"] = [list(r) for r in toe]
            if rim:
                cfg["rim_frames"] = [list(r) for r in rim]

        try:
            result = smt["instance"].run(test_bgr, cfg, original_template_image=std_bgr)
        except Exception as exc:  # noqa: BLE001
            return AlgorithmResult(algorithm=algorithm_id, ok=False, message=f"[适配器异常] {label_cn}: {exc}")

        if result.code != 0:
            return AlgorithmResult(
                algorithm=algorithm_id, ok=False, status="ERROR",
                error_code="ALG_ERROR", error_message=result.message,
                message=f"[错误] {label_cn}：{result.message}",
            )

        boxes = [
            DefectBox(x=int(p.x), y=int(p.y), w=int(p.width), h=int(p.height), label=label_cn)
            for p in result.parts
            if getattr(p, "label", None) == label_key
        ]
        ok = len(boxes) == 0
        vis = result.metadata.get("output_image")
        pad_count = result.metadata.get("pad_count", len(pads))
        message = f"{label_cn} 判定={'OK' if ok else 'NG'}，缺陷数={len(boxes)}，焊盘数={pad_count}"
        return AlgorithmResult(algorithm=algorithm_id, ok=ok, message=message, boxes=boxes, diff_image=vis)

    def _run_shared_preview(self, req: DetectRequest, shared_params: dict[str, Any]) -> AlgorithmResult:
        """“共用参数”本身不是一类缺陷，调参时把四类缺陷一起跑一遍做预览，
        方便看清焊盘定位/焊锡颜色提取这层共用设置的实际效果。"""
        pads = _smt_pads(req.template_name, list(req.rois or []))
        if not pads:
            reason = "未标定焊盘框（pad_frames）"
            return AlgorithmResult(
                algorithm=SMT_SHARED_ID, ok=True, skipped=True, skip_reason=reason,
                message=f"[跳过] 共用参数预览：{reason}，请先标定",
            )

        std_bgr = req.image_std_raw if req.image_std_raw is not None else req.image_std
        test_bgr = req.image_test_raw if req.image_test_raw is not None else req.image_test

        smt = _load_smt()
        cfg: dict[str, Any] = dict(smt["instance"].default_config())
        cfg.update(
            {
                "detect_excess": True,
                "detect_insufficient": True,
                "detect_bridge": True,
                "detect_cold_solder": True,
                "pad_frames": [list(p) for p in pads],
                "enable_visualization": True,
            }
        )
        cfg.update(shared_params)
        _coerce_bools(cfg)

        try:
            result = smt["instance"].run(test_bgr, cfg, original_template_image=std_bgr)
        except Exception as exc:  # noqa: BLE001
            return AlgorithmResult(algorithm=SMT_SHARED_ID, ok=False, message=f"[适配器异常] 共用参数预览: {exc}")

        if result.code != 0:
            # 算法内部失败（code!=0）不是"跳过"，必须透出 ERROR
            return AlgorithmResult(
                algorithm=SMT_SHARED_ID, ok=False, status="ERROR",
                error_code="ALG_ERROR", error_message=result.message,
                message=f"[错误] 共用参数预览：{result.message}",
            )

        boxes = [
            DefectBox(
                x=int(p.x), y=int(p.y), w=int(p.width), h=int(p.height),
                label=_LABEL_CN_BY_KEY.get(getattr(p, "label", None), getattr(p, "label", "?")),
            )
            for p in result.parts
        ]
        ok = len(boxes) == 0
        vis = result.metadata.get("output_image")
        pad_count = result.metadata.get("pad_count", len(pads))
        message = f"共用参数预览（四类缺陷合并显示）判定={'OK' if ok else 'NG'}，缺陷数={len(boxes)}，焊盘数={pad_count}"
        return AlgorithmResult(algorithm=SMT_SHARED_ID, ok=ok, message=message, boxes=boxes, diff_image=vis)
