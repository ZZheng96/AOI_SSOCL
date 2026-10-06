"""模板驱动的检测服务：从 InspectionTemplate 展开规则并调用 DetectScheduler。"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from app.detect.catalog import get_catalog
from app.detect.scheduler import DetectScheduler
from app.detect.types import DefectBox, DetectSummary
from app.result.store import ResultStore
from app.template.model import InspectionTemplate
from app.template.store import TemplateStore
from app.utils.cv_io import imread_unicode


def algorithm_labels() -> dict[str, str]:
    return get_catalog().labels()


def job_item_ids(algorithm_ids: list[str]) -> list[str]:
    catalog = get_catalog()
    return [a for a in algorithm_ids if a and catalog.is_job_item(a)]


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class InspectRunResult:
    test_path: str
    summary: DetectSummary
    folder: Path | None = None
    error: str = ""


@dataclass
class BatchInspectResult:
    results: list[InspectRunResult] = field(default_factory=list)
    ok_count: int = 0
    ng_count: int = 0
    error_count: int = 0
    summary_path: Path | None = None
    cancelled: bool = False

    @property
    def total(self) -> int:
        return len(self.results)


class InspectService:
    def __init__(
        self,
        scheduler: DetectScheduler | None = None,
        result_store: ResultStore | None = None,
        template_store: TemplateStore | None = None,
    ) -> None:
        self.scheduler = scheduler or DetectScheduler()
        self.result_store = result_store or ResultStore()
        self.template_store = template_store or TemplateStore()

    def prepare_template(self, template: InspectionTemplate) -> tuple[np.ndarray | None, str]:
        """同步标定到 legacy，并加载标准图。返回 (std_bgr, error_message)。"""
        self.template_store.sync_calibration_to_legacy(template)
        enabled = job_item_ids(template.enabled_algorithm_ids())
        if not get_catalog().requires_standard(enabled):
            return None, ""

        ok, msg = self.template_store.validate_standard_image(template)
        if not ok:
            return None, msg
        std = imread_unicode(template.standard_image.path)
        if std is None:
            return None, f"标准图无法读取：{template.standard_image.path}"
        return std, ""

    def can_run(self, template: InspectionTemplate) -> tuple[bool, str]:
        if template.status != "published":
            return False, "请选择已发布的模板（草稿请先在建模页发布）"
        enabled = job_item_ids(template.enabled_algorithm_ids())
        if not enabled:
            return False, "模板未启用任何缺陷算法"
        _std, err = self.prepare_template(template)
        if err:
            return False, err
        issues = self.template_store.publish_checklist(template)
        blockers = [i for i in issues if i.startswith("[阻断]")]
        path_blockers = [i for i in blockers if "标准图" in i or "标定" in i or "算法大类" in i]
        if path_blockers:
            return False, "\n".join(path_blockers)
        return True, ""

    def run_with_template(
        self,
        template: InspectionTemplate,
        test_image: np.ndarray | str | Path,
        *,
        test_path: str = "",
        archive: bool = True,
        allow_stub: bool = False,
        source: str = "inspect",
        stop_on_error: bool | None = None,
        cancel_cb: Callable[[], bool] | None = None,
    ) -> InspectRunResult:
        from app.config import get_gate_settings, get_stop_on_error, set_gate_settings

        path_str = test_path
        if isinstance(test_image, (str, Path)):
            path_str = str(test_image)
            img = imread_unicode(path_str)
            if img is None:
                return InspectRunResult(
                    test_path=path_str,
                    summary=DetectSummary(overall_ok=False, overall="ERROR"),
                    error="无法读取测试图",
                )
            test_bgr = img
        else:
            test_bgr = test_image
            path_str = test_path or "memory_test.png"

        if cancel_cb and cancel_cb():
            return InspectRunResult(
                test_path=path_str,
                summary=DetectSummary(overall_ok=False, overall="ERROR"),
                error="已取消",
            )

        std_bgr, err = self.prepare_template(template)
        if err:
            summary = DetectSummary(
                overall_ok=False,
                overall="ERROR",
                gate_blocked=True,
                gate_message=err,
                template_id=template.id,
                template_version=template.version,
                source=source,
            )
            return InspectRunResult(test_path=path_str, summary=summary, error=err)

        algorithm_ids = job_item_ids(template.enabled_algorithm_ids())
        params_map = template.params_map()
        catalog = get_catalog()
        for alg in template.algorithms:
            shared = catalog.shared_param_id(alg.algorithm_id)
            if shared and alg.algorithm_id == shared and alg.params:
                params_map[alg.algorithm_id] = dict(alg.params)
            if not catalog.is_job_item(alg.algorithm_id) and alg.params:
                params_map[alg.algorithm_id] = dict(alg.params)

        if stop_on_error is None:
            stop_on_error = get_stop_on_error()

        prev_gate = get_gate_settings()
        try:
            set_gate_settings(
                bool(template.gate.size_gate_enabled),
                int(template.gate.size_gate_max_diff_px),
            )
            summary = self.scheduler.run(
                std_bgr,
                test_bgr,
                template.category or None,
                template.calib_key or template.id,
                algorithm_ids,
                algorithm_labels(),
                param_overrides=params_map,
                allow_stub=allow_stub,
                prefer_frozen_params=True,
                region_sets=template.region_sets or {},
                template_id=template.id,
                template_version=template.version,
                trigger="on_job",
                stop_on_error=bool(stop_on_error),
                source=source,
            )
        finally:
            set_gate_settings(
                bool(prev_gate.get("gate_enabled", True)),
                int(prev_gate.get("size_gate_max_diff_px", 10)),
            )

        folder = None
        if archive and not summary.gate_blocked:
            folder = self.result_store.save(
                template.calib_key or template.id,
                template.standard_image.path,
                path_str,
                template.category or None,
                algorithm_ids,
                summary,
                template_id=template.id,
                template_version=template.version,
                rule_snapshot=template.rule_snapshot(),
                source=source,
            )
        return InspectRunResult(test_path=path_str, summary=summary, folder=folder)

    def run_dual(
        self,
        template: InspectionTemplate,
        test_image: np.ndarray | str | Path,
        *,
        test_path: str = "",
        allow_stub: bool = False,
    ) -> DualSummary:
        """双检：传统引擎（模板差分）+ 特征引擎（品类模型）→ 融合判定。

        engine_mode 决定引擎组合（traditional / feature / dual）。
        dual 时传统与特征真正并行执行（传统走 CPU、特征走 GPU，可重叠），
        模板的 model_category（或回退 category）关联 AOI 品类模型。
        """
        from app.inspect.fusion import ENGINE_MODES, DualSummary, feature_to_dict, fuse_dual

        mode = template.engine_mode if template.engine_mode in ENGINE_MODES else "dual"

        if mode != "dual":
            trad = None
            if mode == "traditional":
                run = self.run_with_template(template, test_image, allow_stub=allow_stub,
                                             archive=False, test_path=test_path)
                trad = run.summary
            feat = None
            feat_err = ""
            if mode == "feature":
                feat, feat_err = self._run_feature(template, test_image)
            ds = fuse_dual(trad, feat, feat_err, mode)
            ds.template_id = template.id
            ds.template_version = template.version
            return ds

        # dual：两引擎并行执行
        from concurrent.futures import ThreadPoolExecutor

        def _run_trad():
            run = self.run_with_template(template, test_image, allow_stub=allow_stub,
                                         archive=False, test_path=test_path)
            return run.summary

        def _run_feat():
            return self._run_feature(template, test_image)

        with ThreadPoolExecutor(max_workers=2) as ex:
            trad_fut = ex.submit(_run_trad)
            feat_fut = ex.submit(_run_feat)
            trad = trad_fut.result()
            feat, feat_err = feat_fut.result()

        ds = fuse_dual(trad, feat, feat_err, mode)
        ds.template_id = template.id
        ds.template_version = template.version
        return ds

    def run_template_aware(
        self,
        template: InspectionTemplate,
        test_image: np.ndarray | str | Path,
        *,
        test_path: str = "",
        allow_stub: bool = False,
        source: str = "inspect",
    ) -> InspectRunResult:
        """模板感知检测入口：按模板 engine_mode 分流传统/双检，统一返回
        InspectRunResult（双检时把 AOI 判定并入传统结果视图 + 归档）。"""
        mode = template.engine_mode if template.engine_mode in ("traditional", "feature", "dual") else "traditional"
        if mode == "traditional":
            return self.run_with_template(template, test_image, allow_stub=allow_stub,
                                          archive=True, test_path=test_path, source=source)
        ds = self.run_dual(template, test_image, test_path=test_path, allow_stub=allow_stub)
        run = self.dual_to_run_result(ds, test_path=test_path, source=source)
        # 归档（复判/历史依赖归档目录；与 run_with_template 行为对齐）
        if run.summary is not None and not run.summary.gate_blocked:
            try:
                from app.detect.catalog import get_catalog
                path_str = test_path
                run.folder = self.result_store.save(
                    template.calib_key or template.id,
                    template.standard_image.path,
                    path_str,
                    template.category or None,
                    job_item_ids(template.enabled_algorithm_ids()),
                    run.summary,
                    template_id=template.id,
                    template_version=template.version,
                    rule_snapshot=template.rule_snapshot(),
                    source=source,
                )
            except Exception:  # noqa: BLE001 归档失败不阻断检测结果展示
                pass
        return run

    def run_batch_aware(
        self,
        template: InspectionTemplate,
        test_paths: list[str | Path],
        *,
        progress_cb=None, allow_stub: bool = False, write_summary: bool = True,
        archive: bool = True, cancel_cb=None, item_cb=None,
    ) -> BatchInspectResult:
        """模板感知批处理：逐张按 engine_mode 分流。"""
        batch = BatchInspectResult()
        paths = [Path(p) for p in test_paths]
        total = len(paths)
        for i, path in enumerate(paths, start=1):
            if cancel_cb and cancel_cb():
                batch.cancelled = True
                break
            if progress_cb:
                progress_cb(i, total, str(path))
            run = self.run_template_aware(template, path, allow_stub=allow_stub,
                                          source="inspect", test_path=str(path))
            batch.results.append(run)
            if item_cb:
                item_cb(i - 1, run)
            if run.error or run.summary.gate_blocked or run.summary.overall == "ERROR":
                batch.error_count += 1
            elif run.summary.overall_ok:
                batch.ok_count += 1
            else:
                batch.ng_count += 1
        if write_summary and batch.results:
            batch.summary_path = self._write_batch_summary(template, batch)
        return batch

    def dual_to_run_result(self, ds: "DualSummary", *,
                           test_path: str = "", source: str = "inspect") -> InspectRunResult:
        """把 DualSummary 包装为 InspectRunResult：AOI 判定作为附加算法行并入。"""
        from app.detect.types import AlgorithmResult, DetectSummary

        trad = ds.traditional
        results = list(trad.results) if trad is not None else []
        feat = ds.feature
        if feat is not None:
            # AOI 判定行（并入传统结果视图，便于检测页直接展示）
            aoi_box = [b for b in ds.boxes if b.get("engine") == "feature"]
            aoi = AlgorithmResult(
                algorithm="aoi_feature",
                item_id="aoi_feature",
                display_name="AOI 特征学习",
                ok=feat["decision"] != "anomaly",
                message=f"AOI: {feat['decision']} score={feat['score']} "
                        f"主导槽={feat.get('triggered_slot')} ({feat.get('latency_ms')}ms)",
                boxes=[DefectBox(x=b["x"], y=b["y"], w=b["w"], h=b["h"],
                                 label=b["label"]) for b in aoi_box],
                defect_count=len(aoi_box),
                elapsed_ms=int(feat.get("latency_ms", 0)),
                hit_layer="特征引擎",
            )
            results.append(aoi)
        summary = DetectSummary(
            overall_ok=ds.overall_ok,
            overall=ds.overall,
            results=results,
            vis_image=trad.vis_image if trad is not None else None,
            gate_blocked=bool(trad is not None and trad.gate_blocked),
            gate_message=(trad.gate_message if trad is not None else "") or ds.feature_error,
            elapsed_ms=ds.elapsed_ms,
            ng_count=len([r for r in results if r.status == "NG"]),
            template_id=ds.template_id,
            template_version=ds.template_version,
            source=source,
        )
        return InspectRunResult(test_path=test_path, summary=summary, error=ds.feature_error)

    def _run_feature(self, template: InspectionTemplate, test_image):
        """特征引擎推理。返回 (feature_dict, error)。品类模型未准备 → 空+错误。"""
        category = (template.model_category or "").strip() or (template.category or "").strip()
        if not category:
            return None, "模板未配置品类模型绑定（model_category 或 category）"
        try:
            from app.engines.feature import get_engine
            from app.inspect.fusion import feature_to_dict
            engine = get_engine()
            if isinstance(test_image, (str, Path)):
                r = engine.predict_image_path(category, str(test_image))
            else:
                import cv2
                rgb = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
                r = engine.predict_ndarray(category, rgb)
            return feature_to_dict(r), ""
        except Exception as exc:  # noqa: BLE001
            return None, f"特征引擎失败: {exc}"

    def run_items(
        self,
        algorithm_ids: list[str],
        test_image: np.ndarray | str | Path,
        *,
        image_std: np.ndarray | None = None,
        test_path: str = "",
        template: InspectionTemplate | None = None,
        param_overrides: dict[str, dict] | None = None,
        archive: bool = False,
        allow_stub: bool = False,
        source: str = "debug",
        trigger: str = "live",
    ) -> InspectRunResult:
        """调试 / 试跑：可跑子集，默认不归档。

        默认 ``allow_stub=False``：适配器未加载记 ERROR 而非 stub 占位 OK；
        确需 stub 的调试调用须显式传 True。"""
        path_str = test_path
        if isinstance(test_image, (str, Path)):
            path_str = str(test_image)
            img = imread_unicode(path_str)
            if img is None:
                return InspectRunResult(
                    test_path=path_str,
                    summary=DetectSummary(overall_ok=False, overall="ERROR"),
                    error="无法读取测试图",
                )
            test_bgr = img
        else:
            test_bgr = test_image
            path_str = test_path or "memory_test.png"

        ids = job_item_ids([a for a in algorithm_ids if a])
        if not ids:
            return InspectRunResult(
                test_path=path_str,
                summary=DetectSummary(overall_ok=False, overall="ERROR"),
                error="没有可执行的检测项",
            )

        region_sets = template.region_sets if template is not None else {}
        template_name = (template.calib_key or template.id) if template is not None else None
        category = template.category if template is not None else None
        if template is not None:
            self.template_store.sync_calibration_to_legacy(template)

        summary = self.scheduler.run(
            image_std,
            test_bgr,
            category,
            template_name,
            ids,
            algorithm_labels(),
            param_overrides=param_overrides or {},
            allow_stub=allow_stub,
            prefer_frozen_params=bool(param_overrides),
            region_sets=region_sets or {},
            template_id=template.id if template is not None else None,
            template_version=template.version if template is not None else None,
            trigger=trigger,
            source=source,
        )
        folder = None
        if archive and not summary.gate_blocked:
            folder = self.result_store.save(
                template_name,
                template.standard_image.path if template is not None else "",
                path_str,
                category,
                ids,
                summary,
                template_id=template.id if template is not None else None,
                template_version=template.version if template is not None else None,
                rule_snapshot=template.rule_snapshot() if template is not None else {"mode": source, "algorithms": ids},
                source=source,
            )
        return InspectRunResult(test_path=path_str, summary=summary, folder=folder)

    def run_batch(
        self,
        template: InspectionTemplate,
        test_paths: list[str | Path],
        *,
        progress_cb: Callable[[int, int, str], None] | None = None,
        allow_stub: bool = False,
        write_summary: bool = True,
        archive: bool = True,
        cancel_cb: Callable[[], bool] | None = None,
        item_cb: Callable[[int, InspectRunResult], None] | None = None,
    ) -> BatchInspectResult:
        batch = BatchInspectResult()
        paths = [Path(p) for p in test_paths]
        total = len(paths)
        for i, path in enumerate(paths, start=1):
            if cancel_cb and cancel_cb():
                batch.cancelled = True
                break
            if progress_cb:
                progress_cb(i, total, str(path))
            run = self.run_with_template(
                template,
                path,
                allow_stub=allow_stub,
                archive=archive,
                source="inspect",
                cancel_cb=cancel_cb,
            )
            batch.results.append(run)
            if item_cb:
                item_cb(i - 1, run)
            if run.error or run.summary.gate_blocked or run.summary.overall == "ERROR":
                batch.error_count += 1
            elif run.summary.overall_ok:
                batch.ok_count += 1
            else:
                batch.ng_count += 1

        if write_summary and batch.results:
            batch.summary_path = self._write_batch_summary(template, batch)
        return batch

    def collect_images_in_folder(self, folder: str | Path) -> list[Path]:
        root = Path(folder)
        if not root.is_dir():
            return []
        files = [p for p in sorted(root.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
        return files

    def _write_batch_summary(self, template: InspectionTemplate, batch: BatchInspectResult) -> Path:
        now = datetime.now()
        day = now.strftime("%Y-%m-%d")
        stamp = now.strftime("%H%M%S")
        folder = self.result_store.root / day / f"batch_{stamp}_{template.id}"
        folder.mkdir(parents=True, exist_ok=True)

        rows = []
        for r in batch.results:
            rows.append(
                {
                    "test_image": r.test_path,
                    "overall": r.summary.overall
                    or ("ERROR" if (r.error or r.summary.gate_blocked) else ("OK" if r.summary.overall_ok else "NG")),
                    "elapsed_ms": r.summary.elapsed_ms,
                    "ng_count": r.summary.ng_count,
                    "gate_message": r.summary.gate_message or r.error,
                    "result_folder": str(r.folder) if r.folder else "",
                    "ng_algorithms": ",".join(x.item_id or x.algorithm for x in r.summary.ng_list),
                }
            )

        json_path = folder / "batch_summary.json"
        payload = {
            "time": now.isoformat(timespec="seconds"),
            "template_id": template.id,
            "template_version": template.version,
            "display_name": template.display_name,
            "ok_count": batch.ok_count,
            "ng_count": batch.ng_count,
            "error_count": batch.error_count,
            "total": batch.total,
            "cancelled": batch.cancelled,
            "items": rows,
        }
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        csv_path = folder / "batch_summary.csv"
        with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "test_image",
                    "overall",
                    "elapsed_ms",
                    "ng_count",
                    "gate_message",
                    "result_folder",
                    "ng_algorithms",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        return json_path
