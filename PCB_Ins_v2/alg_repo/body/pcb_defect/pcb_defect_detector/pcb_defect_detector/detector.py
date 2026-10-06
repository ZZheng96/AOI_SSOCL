from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import numpy as np

from .image_io import imread_unicode
from .results import Defect, DetectionResult

# 算法 code -> 中文名，便于展示与文档。与引擎注册表一致。
_ALG_CN_NAME: Dict[str, str] = {
    "component_wrong_part": "错件",
    "component_missing": "缺件",
    "component_shift": "移位",
    "component_tombstone": "立碑",
    "component_flipped": "翻件",
    "component_reverse_polarity": "极反",
    "component_damage": "破损",
}

# 默认算法执行顺序（与原 manual_detect 一致）。
_DEFAULT_ORDER: Sequence[str] = tuple(_ALG_CN_NAME.keys())

ImageInput = Union[str, os.PathLike, np.ndarray]

_ENGINE_DIR = Path(__import__("pcb_defect_detector").__file__).resolve().parent / "_engine"
_ENGINE_LOCK = threading.Lock()
_ENGINE_READY = False


def _ensure_engine_on_path() -> None:
    """把引擎目录加入 sys.path（仅一次），使其 core/algorithms/utils 可被导入。"""
    global _ENGINE_READY
    if _ENGINE_READY:
        return
    with _ENGINE_LOCK:
        if _ENGINE_READY:
            return
        p = str(_ENGINE_DIR)
        if p not in sys.path:
            sys.path.insert(0, p)
        _ENGINE_READY = True


def _load_image(img: ImageInput) -> Optional[np.ndarray]:
    """接受 路径 或 ndarray，返回 BGR ndarray（读盘失败返回 None）。"""
    if img is None:
        return None
    if isinstance(img, np.ndarray):
        return img
    if isinstance(img, (str, os.PathLike)):
        return imread_unicode(str(img))
    raise TypeError(f"不支持的图片输入类型: {type(img).__name__}")


def _result_to_detection_result(algorithm: str, raw: Any) -> DetectionResult:
    """把引擎 AlgorithmResult 转成对外 DetectionResult。"""
    # raw 可能为 None（run 抛了异常 / 方法名错）
    if raw is None:
        return DetectionResult(
            algorithm=algorithm,
            status="ERROR",
            defect_count=0,
            error="算法 run() 返回 None（异常）",
        )

    code = getattr(raw, "code", 1)
    meta: Mapping[str, Any] = getattr(raw, "metadata", {}) or {}
    cost = float(getattr(raw, "cost_time", 0.0) or 0.0)
    out_image = meta.get("output_image")

    if code != 0:
        return DetectionResult(
            algorithm=algorithm,
            status="ERROR",
            defect_count=0,
            cost_time=cost,
            output_image=out_image if isinstance(out_image, np.ndarray) else None,
            error=str(meta.get("error") or getattr(raw, "message", "") or "未知错误"),
            extra={"traceback": meta.get("traceback")} if meta.get("traceback") else {},
        )

    status = str(meta.get("status") or ("NG" if getattr(raw, "parts", None) else "OK"))
    defects: list = []
    for part in getattr(raw, "parts", []) or []:
        pmeta = getattr(part, "metadata", {}) or {}
        defects.append(Defect(
            label=str(getattr(part, "label", "defect") or "defect"),
            x=int(getattr(part, "x", 0) or 0),
            y=int(getattr(part, "y", 0) or 0),
            width=int(getattr(part, "width", 0) or 0),
            height=int(getattr(part, "height", 0) or 0),
            confidence=float(getattr(part, "confidence", 0.0) or 0.0),
            description=str(pmeta.get("description", "")),
            severity=float(pmeta.get("severity", 0.0) or 0.0),
            extra={k: v for k, v in pmeta.items()
                   if k not in ("severity", "description") and not isinstance(v, np.ndarray)},
        ))

    extra = {k: v for k, v in meta.items()
             if k not in ("output_image", "status", "defect_count")
             and not isinstance(v, np.ndarray)}

    return DetectionResult(
        algorithm=algorithm,
        status=status,
        defect_count=len(defects),
        defects=defects,
        cost_time=cost,
        output_image=out_image if isinstance(out_image, np.ndarray) else None,
        extra=extra,
    )


class PCBDefectDetector:
    """PCB 元件缺陷检测器（7 项算法的统一门面）。

    参数:
        engine_dir: 自定义引擎目录（一般无需指定，默认指向包内 _engine）。

    典型用法::

        det = PCBDefectDetector()
        det.list_algorithms()
        r = det.detect("component_missing", "tpl.png", "test.png")
        print(r.status, r.defect_count, r.defects)
        r.save_image("out.png")

        # 一次跑全部 7 项：
        all_r = det.detect_all("tpl.png", "test.png")
    """

    def __init__(self, engine_dir: Optional[Union[str, os.PathLike]] = None):
        global _ENGINE_DIR
        if engine_dir is not None:
            _ENGINE_DIR = Path(engine_dir).resolve()
            # 重置就绪标志，强制下次重新注入新引擎路径
            global _ENGINE_READY
            with _ENGINE_LOCK:
                _ENGINE_READY = False

        _ensure_engine_on_path()

        self._pool: Dict[str, Any] = {}
        self._pool_lock = threading.Lock()
        self._pool_config_path = _ENGINE_DIR / "resources" / "config" / "alg_pool" / "component_algs.json"

    # ---- 公共 API ----

    @staticmethod
    def list_algorithms() -> Dict[str, str]:
        """返回 ``{algorithm_code: 中文名}``。顺序固定为 7 项默认顺序。"""
        return dict(_ALG_CN_NAME)

    def available_algorithms(self) -> list:
        """返回当前检测器实际成功加载的 algorithm_code 列表。"""
        self._ensure_pool()
        return list(self._pool.keys())

    def detect(
        self,
        algorithm: str,
        template: ImageInput,
        test: ImageInput,
        config: Optional[Dict[str, Any]] = None,
    ) -> DetectionResult:
        """对一张测试图执行指定算法的缺陷检测。

        参数:
            algorithm: 算法 code，如 ``"component_missing"``（见 ``list_algorithms``）。
            template:  模板图（金板），路径或 ``np.ndarray``。
            test:      测试图（来料），路径或 ``np.ndarray``。
            config:    可选，覆盖算法默认阈值的参数字典（如 ``{"size_tol_ratio": 0.3}``）。

        返回:
            ``DetectionResult``。任何异常都被吸收为 ``status="ERROR"`` 的结果，
            绝不向外抛异常。
        """
        try:
            self._ensure_pool()
            if algorithm not in self._pool:
                return DetectionResult(
                    algorithm=algorithm,
                    status="ERROR",
                    defect_count=0,
                    error=f"未注册的算法: {algorithm}；可用: {self.available_algorithms()}",
                )

            tpl = _load_image(template)
            tst = _load_image(test)
            if tst is None or tst.size == 0:
                return DetectionResult(algorithm=algorithm, status="ERROR",
                                       defect_count=0, error="测试图为空或读取失败")
            if tpl is None or tpl.size == 0:
                return DetectionResult(algorithm=algorithm, status="ERROR",
                                       defect_count=0, error="模板图为空或读取失败")

            inst = self._pool[algorithm]
            raw = inst.run(
                image=tst,
                config=dict(config) if config else {},
                roi_bbox=None,
                original_template_image=tpl,
            )
            return _result_to_detection_result(algorithm, raw)
        except Exception as exc:  # 兜底，绝不抛出
            return DetectionResult(algorithm=algorithm, status="ERROR",
                                   defect_count=0, error=f"{type(exc).__name__}: {exc}")

    def detect_all(
        self,
        template: ImageInput,
        test: ImageInput,
        config: Optional[Dict[str, Any]] = None,
        algorithms: Optional[Sequence[str]] = None,
    ) -> Dict[str, DetectionResult]:
        """对一张测试图依次执行多个算法（默认全部 7 项）。

        参数:
            algorithms: 指定要跑的 algorithm_code 列表；None 表示全部 7 项按默认顺序。

        返回:
            ``{algorithm_code: DetectionResult}``。单项失败不影响其他项。
        """
        self._ensure_pool()
        codes = list(algorithms) if algorithms else [c for c in _DEFAULT_ORDER if c in self._pool]
        results: Dict[str, DetectionResult] = {}
        for code in codes:
            results[code] = self.detect(code, template, test, config=config)
        return results

    def detect_batch(
        self,
        algorithm: str,
        pairs: Sequence,
        config: Optional[Dict[str, Any]] = None,
    ) -> list:
        """批量检测：``pairs`` 为 ``(template, test)`` 序列，返回 ``DetectionResult`` 列表。"""
        out = []
        for tpl, tst in pairs:
            out.append(self.detect(algorithm, tpl, tst, config=config))
        return out

    # ---- 内部 ----

    def _ensure_pool(self) -> None:
        if self._pool:
            return
        with self._pool_lock:
            if self._pool:
                return
            self._pool = self._build_pool()

    def _build_pool(self) -> Dict[str, Any]:
        _ensure_engine_on_path()
        from core.load_class import import_class  # noqa: WPS433

        if not self._pool_config_path.is_file():
            raise FileNotFoundError(f"找不到算法池配置: {self._pool_config_path}")

        with open(self._pool_config_path, "r", encoding="utf-8") as f:
            regs = json.load(f).get("cv_algorithms", [])

        pool: Dict[str, Any] = {}
        for reg in regs:
            code = reg["algorithm_code"]
            cls = import_class(reg["class_path"])
            if cls is None:
                continue
            cfg_path = reg.get("config_path")
            abs_cfg = None
            if cfg_path:
                p = Path(cfg_path)
                abs_cfg = str(p if p.is_absolute() else _ENGINE_DIR / p)
            try:
                pool[code] = cls(abs_cfg)
            except Exception:
                continue  # 单个算法实例化失败不影响其他
        return pool
