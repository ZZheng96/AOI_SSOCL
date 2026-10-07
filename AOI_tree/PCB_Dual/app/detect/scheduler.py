from __future__ import annotations

import threading
import time
from concurrent.futures import Future, TimeoutError as FutureTimeoutError

import cv2
import numpy as np

from app.detect.catalog import get_catalog
from app.detect.contract import DetectionContext, job_status
from app.detect.gate import run_gate
from app.detect.registry import get_adapter_for
from app.detect.roi import rois_from_region_sets
from app.detect.types import AlgorithmResult, DefectBox, DetectRequest, DetectSummary
from app.preprocess.engine import PreprocessParams, run_preprocess
from app.recipe.param_store import ParamStore
from app.recipe.store import RecipeStore

# P1：算法执行保护。adapter.run 可能持有不可 pickle 的模型/运行时状态，
# 因此不能把现有闭包直接交给 Windows spawn 的独立进程；这里不伪造不可取消的重试。
_ALGO_TIMEOUT_S = 30.0
_ALGO_RETRIES = 2         # 保留既有配置名；超时任务无法安全取消时不重试
_MAX_INFLIGHT_THREADS = 4
_inflight_lock = threading.Lock()
_inflight: dict[str, Future] = {}
_inflight_count = 0


def _run_with_timeout(fn, timeout_s: float, tries: int = 2, *, key: str = ""):
    """执行一次并超时返回；超时后的线程继续运行，期间同一请求不会重入。"""
    del tries  # 线程无法强制终止，重试会制造并发副作用而非真实重试。
    global _inflight_count
    with _inflight_lock:
        previous = _inflight.get(key)
        if previous is not None and not previous.done():
            raise TimeoutError(f"检测项 {key} 上次执行仍在运行")
        if _inflight_count >= _MAX_INFLIGHT_THREADS:
            raise TimeoutError("检测执行线程已达上限，拒绝新增不可取消任务")
        fut = Future()
        _inflight[key] = fut
        _inflight_count += 1

    def forget(done):
        global _inflight_count
        with _inflight_lock:
            if _inflight.get(key) is done:
                del _inflight[key]
                _inflight_count -= 1

    fut.add_done_callback(forget)

    def execute():
        if not fut.set_running_or_notify_cancel():
            return
        try:
            fut.set_result(fn())
        except BaseException as exc:
            fut.set_exception(exc)

    threading.Thread(target=execute, daemon=True, name=f"aoi-algo-{key}").start()
    try:
        return fut.result(timeout=timeout_s)
    except FutureTimeoutError as exc:
        raise TimeoutError(f"检测项 {key} 执行超过 {timeout_s}s") from exc


__all__ = [
    "AlgorithmResult",
    "DefectBox",
    "DetectRequest",
    "DetectSummary",
    "DetectScheduler",
    "merge_defect_overlays",
]


class DetectScheduler:
    """门禁 → 预处理 → 按 Catalog 逐项 run。

    禁止按 algorithm_id 前缀写特例；SMT 共用参数走 manifest.shared_param_id。
    """

    def __init__(self, recipe_store: RecipeStore | None = None, param_store: ParamStore | None = None) -> None:
        self.recipe_store = recipe_store or RecipeStore()
        self.param_store = param_store or ParamStore()

    def run(
        self,
        image_std: np.ndarray | None,
        image_test: np.ndarray,
        category: str | None,
        template_name: str | None,
        algorithm_ids: list[str],
        algorithm_labels: dict[str, str] | None = None,
        param_overrides: dict[str, dict] | None = None,
        *,
        allow_stub: bool = True,
        prefer_frozen_params: bool = False,
        region_sets: dict | None = None,
        template_id: str | None = None,
        template_version: int | None = None,
        trigger: str = "on_job",
        stop_on_error: bool = False,
        source: str = "inspect",
        gate_enabled: bool | None = None,
        size_gate_max_diff_px: int | None = None,
    ) -> DetectSummary:
        catalog = get_catalog()
        algorithm_labels = algorithm_labels or catalog.labels()
        param_overrides = param_overrides or {}
        job_ids = [a for a in algorithm_ids if catalog.is_job_item(a)]
        need_std = catalog.requires_standard(job_ids)
        gate_note = ""
        t_job = time.perf_counter()

        if need_std:
            if image_std is None:
                return DetectSummary(
                    overall_ok=False,
                    overall="ERROR",
                    results=[
                        AlgorithmResult(
                            algorithm="__gate__",
                            ok=False,
                            message="对照算法需要标准图",
                            status="ERROR",
                            error_code="GATE_NO_STANDARD",
                            error_message="对照算法需要标准图",
                        )
                    ],
                    vis_image=self.draw_result(image_test, [], False),
                    gate_blocked=True,
                    gate_message="对照算法需要标准图",
                    template_id=template_id or template_name,
                    template_version=template_version,
                    source=source,
                    elapsed_ms=int((time.perf_counter() - t_job) * 1000),
                )
            gate_outcome = run_gate(
                image_std, image_test,
                gate_enabled=gate_enabled,
                size_gate_max_diff_px=size_gate_max_diff_px,
            )
            if gate_outcome.blocked:
                gate_result = AlgorithmResult(
                    algorithm="__gate__",
                    ok=False,
                    message=gate_outcome.block_message,
                    status="ERROR",
                    error_code="GATE_BLOCKED",
                    error_message=gate_outcome.block_message,
                )
                vis = self.draw_result(image_test, [], False)
                return DetectSummary(
                    overall_ok=False,
                    overall="ERROR",
                    results=[gate_result],
                    vis_image=vis,
                    gate_blocked=True,
                    gate_message=gate_outcome.block_message,
                    template_id=template_id or template_name,
                    template_version=template_version,
                    source=source,
                    elapsed_ms=int((time.perf_counter() - t_job) * 1000),
                )
            image_test = gate_outcome.aligned_test if gate_outcome.aligned_test is not None else image_test
            gate_note = gate_outcome.note
        else:
            image_std = image_test

        results: list[AlgorithmResult] = []
        # 同一次检测中配方相同的算法共用预处理结果（整图预处理是主要耗时）
        pre_cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}

        for alg_id in job_ids:
            manifest = catalog.get(alg_id)
            label = algorithm_labels.get(alg_id, manifest.display_name if manifest else alg_id)
            record, hit = self.recipe_store.resolve(category, alg_id)
            params = PreprocessParams.from_dict(record.preprocess)
            pkey = tuple(sorted(params.to_dict().items()))
            if pkey not in pre_cache:
                std_src = image_std if image_std is not None else image_test
                pre_cache[pkey] = (run_preprocess(std_src, params), run_preprocess(image_test, params))
            std_p, test_p = pre_cache[pkey]

            kind = catalog.region_kind(alg_id)
            item_rois = rois_from_region_sets(region_sets, kind if kind != "none" else None)

            req = DetectRequest(
                image_std=std_p,
                image_test=test_p,
                image_std_raw=image_std,
                image_test_raw=image_test,
                category=category,
                template_name=template_name,
                algorithm=alg_id,
                rois=item_rois,
                template_id=template_id or template_name,
                template_version=template_version,
                trigger=trigger,
                preprocess_meta={"recipe_hit": hit, "preprocess": dict(record.preprocess or {})},
            )
            ctx = DetectionContext(
                item_id=alg_id,
                template_id=template_id or template_name,
                template_version=template_version,
                image_test=test_p,
                image_std=std_p,
                image_test_raw=image_test,
                image_std_raw=image_std,
                rois=item_rois,
                params={},
                preprocess_meta=req.preprocess_meta,
                trigger=trigger,  # type: ignore[arg-type]
                category=category,
                template_name=template_name,
            )

            adapter = get_adapter_for(alg_id)
            t_item = time.perf_counter()
            if adapter is None:
                if allow_stub:
                    result = self._detect_stub(req, label)
                    param_hit = ""
                else:
                    result = AlgorithmResult(
                        algorithm=alg_id,
                        ok=False,
                        message=f"[失败] {label}：算法适配器未加载，产线模式禁止 stub",
                        status="ERROR",
                        error_code="ADAPTER_MISSING",
                        error_message="算法适配器未加载",
                        display_name=label,
                    )
                    param_hit = ""
            else:
                ready, reason = adapter.is_ready(alg_id, template_name, rois=item_rois)
                if not ready:
                    if allow_stub:
                        result = AlgorithmResult(
                            algorithm=alg_id,
                            ok=True,
                            skipped=True,
                            skip_reason=reason,
                            message=f"[跳过] {label}：{reason}",
                            status="SKIP",
                            display_name=label,
                        )
                    else:
                        result = AlgorithmResult(
                            algorithm=alg_id,
                            ok=False,
                            skipped=False,
                            skip_reason=reason,
                            message=f"[失败] {label}：{reason}",
                            status="ERROR",
                            error_code="NOT_READY",
                            error_message=reason,
                            display_name=label,
                        )
                    param_hit = ""
                else:
                    resolved_params, param_hit = self._resolve_params(
                        adapter,
                        alg_id,
                        template_name,
                        category,
                        param_overrides,
                        prefer_frozen_params,
                    )
                    ok_params, param_err = adapter.validate_params(alg_id, resolved_params)
                    if not ok_params:
                        result = AlgorithmResult(
                            algorithm=alg_id,
                            ok=False,
                            message=f"[失败] {label}：参数无效（{param_err}）",
                            status="ERROR",
                            error_code="PARAM_INVALID",
                            error_message=param_err,
                            display_name=label,
                        )
                    else:
                        ctx.params = resolved_params
                        req.params = resolved_params
                        try:
                            result = _run_with_timeout(
                                lambda: adapter.run(alg_id, req, resolved_params),
                                timeout_s=_ALGO_TIMEOUT_S,
                                tries=_ALGO_RETRIES,
                                key=f"{id(req)}:{alg_id}",
                            )
                        except TimeoutError as exc:  # noqa: BLE001
                            result = AlgorithmResult(
                                algorithm=alg_id,
                                ok=False,
                                message=f"[超时] {label}: 算法超过 {_ALGO_TIMEOUT_S}s",
                                status="ERROR",
                                error_code="RUN_TIMEOUT",
                                error_message=str(exc),
                                display_name=label,
                            )
                        except Exception as exc:  # noqa: BLE001
                            result = AlgorithmResult(
                                algorithm=alg_id,
                                ok=False,
                                message=f"[调度异常] {label}: {exc}",
                                status="ERROR",
                                error_code="RUN_EXCEPTION",
                                error_message=str(exc),
                                display_name=label,
                            )

            elapsed = int((time.perf_counter() - t_item) * 1000)
            self._enrich_result(result, alg_id, label, manifest, elapsed)
            result.preprocess_std = std_p
            result.preprocess_test = test_p
            result.hit_layer = f"{hit} | 参数:{param_hit}" if param_hit else hit
            results.append(result)
            if stop_on_error and result.status == "ERROR":
                break

        overall = job_status(gate_blocked=False, items=results)
        overall_ok = overall == "OK"
        if not need_std and len(results) == 1 and results[0].diff_image is not None:
            vis = results[0].diff_image
        else:
            vis = self.draw_result(image_test, results, overall_ok)
        return DetectSummary(
            overall_ok=overall_ok,
            overall=overall,
            results=results,
            vis_image=vis,
            gate_blocked=False,
            gate_note=gate_note,
            elapsed_ms=int((time.perf_counter() - t_job) * 1000),
            ng_count=len([r for r in results if r.status == "NG"]),
            template_id=template_id or template_name,
            template_version=template_version,
            source=source,
            stop_on_error=stop_on_error,
        )

    def _resolve_params(
        self,
        adapter,
        alg_id: str,
        template_name: str | None,
        category: str | None,
        param_overrides: dict[str, dict],
        prefer_frozen_params: bool,
    ) -> tuple[dict, str]:
        catalog = get_catalog()
        shared_id = catalog.shared_param_id(alg_id)
        defaults = adapter.default_params(alg_id)
        own_keys = set(defaults.keys())

        def _migrate(item_id: str, raw: dict | None) -> dict:
            item_adapter = get_adapter_for(item_id) or adapter
            return item_adapter.migrate_params(item_id, raw or {})

        if prefer_frozen_params and alg_id in param_overrides:
            # v2 修复：模板冻结参数应"直接生效"，不做 specs 过滤/迁移。
            # PCB_Ins 原版这里走 _migrate()，会把 param_specs 未声明的深层
            # 参数（如 SMT 的 template_mode / mask_h_low / silk_*）过滤丢弃，
            # 导致模板冻结参数与实际判定不一致（特定图漏检）。
            resolved = dict(param_overrides[alg_id])
            hit = "模板冻结参数"
            if shared_id:
                shared_raw = dict(param_overrides.get(shared_id) or {})
                if shared_raw:
                    resolved = {**shared_raw, **resolved}
                    hit = "模板冻结参数(+共用)"
            return resolved, hit

        if shared_id:
            shared_adapter = get_adapter_for(shared_id) or adapter
            shared_defaults = shared_adapter.default_params(shared_id)
            shared_resolved, shared_hit = self.param_store.resolve(
                template_name, None, shared_id, shared_defaults
            )
            shared_resolved = _migrate(shared_id, shared_resolved)
            own_resolved, own_hit = self.param_store.resolve(template_name, category, alg_id, defaults)
            own_resolved = {k: v for k, v in _migrate(alg_id, own_resolved).items() if k in own_keys}
            if alg_id in param_overrides:
                own_resolved = {**own_resolved, **{k: v for k, v in param_overrides[alg_id].items() if k in own_keys}}
            if shared_id in param_overrides:
                shared_resolved = {**shared_resolved, **param_overrides[shared_id]}
            return {**shared_resolved, **own_resolved}, f"共用参数:{shared_hit}；本项:{own_hit}"

        resolved, hit = self.param_store.resolve(template_name, category, alg_id, defaults)
        resolved = _migrate(alg_id, resolved)
        if alg_id in param_overrides:
            resolved = {**resolved, **param_overrides[alg_id]}
            if prefer_frozen_params:
                hit = "模板冻结参数"
        return resolved, hit

    @staticmethod
    def _enrich_result(result: AlgorithmResult, alg_id: str, label: str, manifest, elapsed_ms: int) -> None:
        result.item_id = result.item_id or alg_id
        result.algorithm = result.algorithm or alg_id
        result.display_name = result.display_name or (manifest.display_name if manifest else label)
        result.elapsed_ms = result.elapsed_ms or elapsed_ms
        if manifest:
            result.algorithm_version = result.algorithm_version or manifest.algorithm_version
            result.param_schema_version = manifest.param_schema_version
        if not result.status:
            if result.error_code:
                result.status = "ERROR"
            elif result.skipped:
                result.status = "SKIP"
            else:
                result.status = "OK" if result.ok else "NG"
        if result.status == "NG" and not result.defect_type:
            result.defect_type = result.display_name or label
        if not result.defect_count:
            result.defect_count = len(result.boxes)
        color = manifest.color if manifest else "#dc2626"
        for box in result.boxes:
            if not box.item_id:
                box.item_id = alg_id
            if not box.color:
                box.color = color
            if not box.label:
                box.label = result.display_name or label

    @staticmethod
    def _detect_stub(req: DetectRequest, label: str) -> AlgorithmResult:
        return AlgorithmResult(
            algorithm=req.algorithm,
            ok=True,
            message=f"[stub] {label} 对照检测未接入，暂返回 OK",
            boxes=[],
            status="OK",
            display_name=label,
        )

    @staticmethod
    def draw_result(
        image_test_bgr: np.ndarray,
        results: list[AlgorithmResult],
        overall_ok: bool,
    ) -> np.ndarray:
        from app.utils.cv_text import put_text_bgr

        vis = image_test_bgr.copy()
        catalog = get_catalog()
        merged = merge_defect_overlays(results)
        for item in merged:
            x1, y1, x2, y2 = item["x"], item["y"], item["x"] + item["w"], item["y"] + item["h"]
            color = _hex_to_bgr(item.get("color") or "#dc2626")
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f'{item.get("index", "")} {item.get("label") or ""}'.strip()
            put_text_bgr(vis, label, (x1 + 2, y1 - 4), color, font_size=16, thickness_box=True)

        tag = "OK" if overall_ok else "NG"
        if any(r.status == "ERROR" or r.error_code for r in results):
            tag = "ERR"
        tag_color = (0, 180, 0) if tag == "OK" else ((0, 102, 255) if tag == "ERR" else (0, 0, 255))
        cv2.rectangle(vis, (4, 4), (62, 28), tag_color, -1)
        cv2.putText(vis, tag, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        _ = catalog
        return vis


def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    text = (hex_color or "#dc2626").lstrip("#")
    if len(text) != 6:
        return (0, 0, 255)
    r, g, b = int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)
    return (b, g, r)


def merge_defect_overlays(results: list[AlgorithmResult], *, iou_thresh: float = 0.45) -> list[dict]:
    """同位置多算法命中合并为一框，label 用「缺件 · 移位」。"""
    raw: list[dict] = []
    for result in results:
        if result.ok or result.skipped or result.status in {"SKIP", "ERROR"}:
            continue
        color = ""
        try:
            color = get_catalog().color(result.item_id or result.algorithm)
        except Exception:
            color = "#dc2626"
        for box in result.boxes:
            raw.append(
                {
                    "x": int(box.x),
                    "y": int(box.y),
                    "w": int(box.w),
                    "h": int(box.h),
                    "label": box.label or result.defect_type or result.display_name or result.algorithm,
                    "color": box.color or color,
                    "item_id": box.item_id or result.item_id,
                    "score": box.score,
                    "roi_id": box.roi_id,
                }
            )
    # 按几何和来源固定排序，并用连通分量合并，避免结果依赖算法/框的输入顺序。
    raw.sort(
        key=lambda box: (
            box["x"],
            box["y"],
            box["w"],
            box["h"],
            box["item_id"] or "",
            box["label"] or "",
            box["roi_id"] or "",
        )
    )
    visited: set[int] = set()
    merged: list[dict] = []
    for start in range(len(raw)):
        if start in visited:
            continue
        stack = [start]
        visited.add(start)
        members: list[dict] = []
        while stack:
            current = stack.pop()
            members.append(raw[current])
            for candidate in range(len(raw)):
                if candidate not in visited and _iou(raw[current], raw[candidate]) >= iou_thresh:
                    visited.add(candidate)
                    stack.append(candidate)

        x1 = min(box["x"] for box in members)
        y1 = min(box["y"] for box in members)
        x2 = max(box["x"] + box["w"] for box in members)
        y2 = max(box["y"] + box["h"] for box in members)
        labels = sorted({box["label"] for box in members if box["label"]})
        first = members[0]
        merged.append(
            {
                **first,
                "x": x1,
                "y": y1,
                "w": x2 - x1,
                "h": y2 - y1,
                "labels": labels,
            }
        )

    merged.sort(key=lambda box: (box["x"], box["y"], box["w"], box["h"], box["label"] or ""))
    for i, item in enumerate(merged, start=1):
        item["index"] = i
        item["label"] = " · ".join(item.get("labels") or [item.get("label") or ""])
    return merged


def _iou(a: dict, b: dict) -> float:
    ax2, ay2 = a["x"] + a["w"], a["y"] + a["h"]
    bx2, by2 = b["x"] + b["w"], b["y"] + b["h"]
    ix1, iy1 = max(a["x"], b["x"]), max(a["y"], b["y"])
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union else 0.0
