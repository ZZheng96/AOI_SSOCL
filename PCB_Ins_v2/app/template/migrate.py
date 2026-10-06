"""从 template_calib / ParamStore / template_map 生成 InspectionTemplate 草稿。"""
from __future__ import annotations

import json
from pathlib import Path

from app.config import ROOT_DIR, TEMPLATE_CALIB_DIR, TEMPLATE_MAP_FILE
from app.detect.catalog import get_catalog
from app.recipe.param_store import ParamStore
from app.template.model import InspectionTemplate, StandardImageRef, TemplateAlgorithm
from app.template.store import TemplateStore, safe_id


def load_template_map() -> dict[str, str]:
    if not TEMPLATE_MAP_FILE.exists():
        return {}
    try:
        data = json.loads(TEMPLATE_MAP_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}


def _infer_algorithms_from_params(stem: str, param_store: ParamStore) -> list[str]:
    root = param_store.root / "template_algorithm"
    if not root.exists():
        return []
    prefix = f"{stem}__"
    ids: list[str] = []
    for path in sorted(root.glob(f"{stem}__*.json")):
        alg_id = path.stem[len(prefix) :] if path.stem.startswith(prefix) else ""
        if not alg_id or not get_catalog().is_job_item(alg_id):
            continue
        if alg_id and alg_id not in ids:
            ids.append(alg_id)
    return ids


def _guess_category(algorithm_ids: list[str]) -> str:
    catalog = get_catalog()
    for a in algorithm_ids:
        if catalog.group_id(a) == "through_hole":
            return "插件器件"
    return "其他"


def migrate_legacy_to_templates(
    *,
    store: TemplateStore | None = None,
    param_store: ParamStore | None = None,
    overwrite: bool = False,
) -> dict:
    """扫描旧标定与 template_map，生成 draft 模板。

    返回 {"created": [...], "skipped": [...], "updated": [...], "warnings": [...]}
    """
    store = store or TemplateStore()
    param_store = param_store or ParamStore()
    template_map = load_template_map()

    stems: set[str] = set()
    if TEMPLATE_CALIB_DIR.exists():
        for path in TEMPLATE_CALIB_DIR.glob("*.json"):
            stems.add(path.stem)
    stems.update(template_map.keys())

    created: list[str] = []
    skipped: list[str] = []
    updated: list[str] = []
    warnings: list[str] = []

    for stem in sorted(stems):
        tid = safe_id(stem)
        exists = store.exists(tid)
        if exists and not overwrite:
            skipped.append(tid)
            continue

        std_path = template_map.get(stem, "")
        if std_path and not Path(std_path).exists():
            warnings.append(f"{tid}: 标准图路径失效 → {std_path}")

        alg_ids = _infer_algorithms_from_params(stem, param_store)
        needs_confirm = False
        if not alg_ids:
            # 无参数文件时给空算法列表，标需确认
            needs_confirm = True
            warnings.append(f"{tid}: 未能从参数文件推断算法，请在建模页勾选后发布")

        category = _guess_category(alg_ids)
        if not alg_ids and category:
            alg_ids = get_catalog().recommended_ids(category)
            needs_confirm = True

        from app.calibration.store import load_calib
        from app.detect.registry import get_adapter_for

        calib = load_calib(stem)
        algorithms: list[TemplateAlgorithm] = []
        for alg_id in alg_ids:
            adapter = get_adapter_for(alg_id)
            defaults = adapter.default_params(alg_id) if adapter is not None else {}
            params, _hit = param_store.resolve(stem, None, alg_id, defaults)
            algorithms.append(TemplateAlgorithm(algorithm_id=alg_id, enabled=True, params=params))

        # solder 共用参数一并冻结（若存在）
        if any(get_catalog().shared_param_id(a.algorithm_id) for a in algorithms):
            shared_ids = {
                get_catalog().shared_param_id(a.algorithm_id)
                for a in algorithms
                if get_catalog().shared_param_id(a.algorithm_id)
            }
            for shared_id in shared_ids:
                adapter = get_adapter_for(shared_id)
                defaults = adapter.default_params(shared_id) if adapter is not None else {}
                shared_params, _ = param_store.resolve(stem, None, shared_id, defaults)
                if shared_params:
                    algorithms.append(
                        TemplateAlgorithm(algorithm_id=shared_id, enabled=False, params=shared_params)
                    )

        notes = "由旧标定/参数迁移生成"
        if needs_confirm:
            notes += "；算法勾选需人工确认后再发布"

        std_ref = StandardImageRef(path=std_path)
        if std_path and Path(std_path).exists():
            from app.utils.cv_io import imread_unicode

            img = imread_unicode(std_path)
            if img is not None:
                std_ref.height, std_ref.width = img.shape[:2]

        if exists:
            tpl = store.load(tid) or InspectionTemplate(id=tid)
            tpl.display_name = tpl.display_name or stem
            tpl.standard_image = std_ref if std_path else tpl.standard_image
            tpl.category = category or tpl.category
            tpl.detection_items = algorithms or tpl.detection_items
            tpl.calibration = calib or tpl.calibration
            tpl.sync_regions_from_calibration()
            tpl.calib_key = tid
            tpl.status = "draft"
            tpl.notes = notes
            store.save(tpl)
            updated.append(tid)
        else:
            tpl = InspectionTemplate(
                id=tid,
                display_name=stem,
                status="draft",
                standard_image=std_ref,
                category=category,
                detection_items=algorithms,
                calibration=dict(calib or {}),
                notes=notes,
                calib_key=tid,
            )
            tpl.sync_regions_from_calibration()
            store.save(tpl)
            created.append(tid)

    return {
        "created": created,
        "skipped": skipped,
        "updated": updated,
        "warnings": warnings,
        "root": str(ROOT_DIR / "templates"),
    }
