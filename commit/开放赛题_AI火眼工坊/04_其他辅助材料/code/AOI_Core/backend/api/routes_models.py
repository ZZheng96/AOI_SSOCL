"""模型注册表与评估接口（M2 起模型管理走 Demo5 快照）。"""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import time
import zipfile

from typing import Optional

from fastapi import (APIRouter, Depends, File, Form, HTTPException,
                     UploadFile)
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from ..core.config import get_settings
from ..core.security import require_role
from ..core.tasks import task_manager
from ..db.database import log_action, session_scope
from ..db.models import EvalRun, Image as ImageRow, Model
from ..evaluation import service as eval_service
from ..self_learning import service as sl_service
from .schemas import (ABCompareRequest, CalibrationExcludeRequest,
                      EvalAccuracyRequest, EvalBenchmarkRequest,
                      EvalCompareRequest, EvalRobustnessRequest,
                      ModelPrepareRequest, RollbackRequest, to_dict)

router = APIRouter()


# ── M16a 品类上线 readiness 状态机（review R1）────────────────
_STAGE_CN = {
    "empty": "① 数据未导入",
    "insufficient": "① 数据不足",
    "ready": "② 可训练",
    "trained": "③ 已训练未验收",
    "validated": "④ 已验收未激活",
    "active": "⑤ 已投产",
}
_STAGE_NEXT = {   # (下一步文案, 导航页名)
    "empty": ("去数据管理导入正常图/缺陷图", "数据管理"),
    "insufficient": ("补充数据（正常图≥3 张才可准备；缺陷不足可用伪异常）", "数据管理"),
    "ready": ("去模型管理准备模型（冷启动训练）", "模型管理"),
    "trained": ("去评估看板做精度验收（达标再上线）", "评估看板"),
    "validated": ("去模型管理激活版本，正式投产", "模型管理"),
    "active": ("已投产：日常监控 + 反馈学习", "实时监控"),
}


def _category_readiness(s, cat: str) -> dict:
    """单品类上线状态机聚合（全部基于现有 DB，纯只读）。

    判据：数据构成（train 正常/缺陷数）→ 快照版本数 → 精度评估记录 →
    激活版本。与 precheck/prepare/eval/activate 的实际门槛同口径。"""
    imgs = s.query(ImageRow.split, ImageRow.label).filter(
        ImageRow.category == cat).all()
    n_train_normal = sum(1 for sp, lb in imgs
                         if sp == "train" and lb == "normal")
    n_defect = sum(1 for sp, lb in imgs if lb == "anomaly")
    n_images = len(imgs)
    models = s.query(Model).filter(Model.category == cat).all()
    active = next((m for m in models if m.is_active), None)
    evals = (s.query(EvalRun)
             .filter(EvalRun.category == cat, EvalRun.run_type == "accuracy")
             .order_by(EvalRun.id.desc()).all())
    best_auroc = None
    for e in evals:
        a = (e.metrics or {}).get("auroc")
        if a is not None:
            best_auroc = max(best_auroc or 0.0, float(a))

    if n_images == 0:
        stage = "empty"
    elif n_train_normal < 3:
        stage = "insufficient"
    elif not models:
        stage = "ready"
    elif active is not None:
        stage = "active"              # 已激活是既成事实，优先于"未验收"
    elif not evals:
        stage = "trained"
    else:
        stage = "validated"
    nxt, page = _STAGE_NEXT[stage]
    return {
        "category": cat, "stage": stage, "stage_cn": _STAGE_CN[stage],
        "next_action": nxt, "next_page": page,
        "counts": {"images": n_images, "train_normal": n_train_normal,
                   "defect": n_defect},
        "n_versions": len(models),
        "active_version": active.version if active else None,
        "best_auroc": best_auroc, "n_evals": len(evals),
    }


@router.get("/models/readiness")
def api_models_readiness(category: str | None = None):
    """品类上线状态机（M16a，review R1）：每品类处于 数据→训练→验收→投产
    哪一环 + 下一步动作 + 直达页名。工作台首页与模型页共用。"""
    from .routes_system import _all_categories
    with session_scope() as s:
        cats = [category] if category else _all_categories()
        # U-opsflow(2026-08-26)：_all_categories() 返回 dict 列表
        # ({category,...})，原代码直接当品类名查库 → SQLite 绑定 dict 报 500，
        # 工作台"品类上线状态表"整体加载失败。这里提取品类名。
        if not category and cats and isinstance(cats[0], dict):
            cats = [c["category"] for c in cats]
        return {"items": [_category_readiness(s, c) for c in cats]}


def register_demo5_snapshots(protect: Optional[set] = None) -> int:
    """扫描 {demo5_storage}/snapshots/{category}/v*/ 并幂等登记/更新 Model 表。

    按 (category, version) upsert；is_active 与 current.json 指针对齐。
    origin：meta.trigger=="fit" → "demo5_fit"，其余 → "self_learned_demo5"。
    返回新增行数。

    protect：调用方声明"刚完成数据血缘操作"的品类集合（如 prepare 刚按
    该品类数据 fit 完、dataset_ids 尚未落库），本轮扫描豁免无血缘判定，
    防止激活指针在谱系写入前被切断（B2 修复，2026-08-30）。

    2026-08-30（MPDD 评测 P1 修复）：残留快照治理。
    - 孤儿品类（不在任一数据源的图/批次品类中）：整个品类目录归档到
      snapshots/_archive/，删除其 Model 行——不注册、不激活、不出现在
      版本历史（防 bottle/zipper 等历史残留自动激活误导首次用户）。
    - 无血缘品类（品类在数据源中，但当前库无任何该品类检测记录，即
      "清库留 storage"场景）：切断激活指针（移除 current.json，防准备
      模型前用旧快照兜底检测产生误报），已登记版本 metrics 标
      orphan=True（"历史残留"），版本号仍单调递增（快照不可变）。
    """
    import shutil
    import time

    from ..db.models import Dataset, Detection
    from ..engine import get_engine
    engine = get_engine()
    root = engine.snap_root
    registered = 0
    if not os.path.isdir(root):
        return 0
    archived = []                            # 归档留痕（with 块外统一 log_action，
    # 避免 session 内嵌套开库写 → SQLite database is locked 回滚，2026-08-30）
    with session_scope() as s:
        # 数据源品类集合（孤儿校验）：图/批次两层并集
        valid_cats = ({r[0] for r in s.query(Dataset.category).distinct().all()} |
                      {r[0] for r in s.query(ImageRow.category).distinct().all()})
        valid_cats.discard(None)
        valid_cats.discard("")
        # 显式导入的快照品类（origin="uploaded"）：用户主动导入即合法资产，
        # 豁免孤儿归档 / 无血缘切指针（/models/snapshot/import 先落 Model
        # 行再触发本扫描，保证导入目录不被当孤儿归档）
        imported_cats = {r[0] for r in s.query(Model.category)
                         .filter(Model.origin == "uploaded").distinct().all()}
        # 血缘品类集合：当前库有检测记录 → 历史版本视为合法演进
        lineage_cats = {r[0] for r in s.query(Detection.category).distinct().all()}
        # B2 修复（2026-08-30 首用质检）：血缘判定并入"品类数据血缘"——
        # Model.metrics.dataset_ids 非空（prepare 落库谱系）的品类视为有血缘，
        # 保留其激活指针。否则首次 prepare 后、首次检测前触发扫描（如启动时
        # app.py 自动扫描）会把刚准备的模型误判"无血缘残留"并删 current.json，
        # 表现为 UI 状态倒退（已验收→未激活）。
        for m in s.query(Model).all():
            if (m.metrics or {}).get("dataset_ids"):
                lineage_cats.add(m.category)
        if protect:
            lineage_cats |= set(protect)
        for cat in sorted(os.listdir(root)):
            if cat == "_archive":
                continue
            cat_dir = os.path.join(root, cat)
            if not os.path.isdir(cat_dir):
                continue
            if cat not in valid_cats and cat not in imported_cats:
                # 孤儿品类：归档整个目录（可恢复）+ 删 Model 行（互相独立：
                # 目录已不在时行仍要删，防 register 中断后行目录不一致）
                ts = time.strftime("%Y%m%d_%H%M%S")
                dst = os.path.join(root, "_archive", f"{cat}_{ts}")
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                try:
                    shutil.move(cat_dir, dst)
                    archived.append(f"category={cat} -> {dst}")
                except OSError:
                    pass                       # 归档失败不阻断扫描
                s.query(Model).filter(Model.category == cat).delete(
                    synchronize_session=False)
                continue
            has_lineage = cat in lineage_cats
            cur = engine.current_version(cat)
            if not has_lineage and cur is not None and cat not in imported_cats:
                # 无血缘残留：切断激活指针（旧快照不再兜底服务检测）
                # 显式导入品类豁免：用户导入并激活的快照不因此静默失效
                cur_json = os.path.join(cat_dir, "current.json")
                try:
                    os.remove(cur_json)
                except OSError:
                    pass
                cur = None
            for v in engine.list_versions(cat):
                vdir = os.path.join(cat_dir, f"v{v}")
                meta_path = os.path.join(vdir, "meta.json")
                if not os.path.exists(meta_path):
                    continue
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                except Exception:  # noqa: BLE001 meta 损坏不阻断扫描
                    meta = {}
                row = (s.query(Model)
                       .filter(Model.category == cat,
                               Model.version == f"v{v}").first())
                # origin：导入快照（meta 带 imported_from 或行已是 uploaded）
                # 恒为 "uploaded"，其余按 trigger 推断
                if meta.get("imported_from") or (
                        row is not None and row.origin == "uploaded"):
                    origin = "uploaded"
                else:
                    origin = ("demo5_fit" if meta.get("trigger") == "fit"
                              else "self_learned_demo5")
                metrics = {"fit_seconds": meta.get("fit_seconds"),
                           "slot_names": meta.get("slot_names"),
                           "note": meta.get("note")}
                if not has_lineage and origin != "uploaded":
                    metrics["orphan"] = True
                    metrics["note"] = ("历史残留（当前库无该品类数据血缘，"
                                       "已切断激活）。" + str(meta.get("note") or ""))
                is_active = (v == cur)
                if row is None:
                    s.add(Model(name=f"{cat}_v{v}", level="0", category=cat,
                                version=f"v{v}", path=vdir,
                                format="demo5_snapshot", origin=origin,
                                metrics=metrics, is_active=is_active))
                    registered += 1
                else:
                    row.name = f"{cat}_v{v}"
                    row.level = "0"
                    row.path = vdir
                    row.format = "demo5_snapshot"
                    row.origin = origin
                    # M16f：合并而非覆盖——保留 prepare 写入的数据谱系
                    # dataset_ids 等扩展键，重扫描不丢
                    merged = dict(row.metrics or {})
                    merged.update(metrics)
                    row.metrics = merged
                    row.is_active = is_active
        # 清理 DB 中孤儿品类的残留 Model 行（目录可能已被先前归档而扫不到）；
        # uploaded 行豁免——导入品类可能本就无数据源数据，不应被清掉
        s.query(Model).filter(Model.format == "demo5_snapshot",
                              Model.origin != "uploaded",
                              ~Model.category.in_(valid_cats)).delete(
            synchronize_session=False)
        s.flush()
    for rec in archived:
        log_action("snapshot_orphan_archive", rec)
    return registered


# ── 模型注册表 ───────────────────────────────────────────────
@router.post("/models/scan",
             dependencies=[Depends(require_role("engineer"))])
def api_models_scan():
    """重新扫描 Demo5 快照目录并幂等登记进 Model 表。"""
    registered = register_demo5_snapshots()
    log_action("models_scan", f"registered={registered}")
    return {"registered": registered}


@router.get("/models")
def api_list_models():
    with session_scope() as s:
        rows = s.query(Model).order_by(Model.id.desc()).all()
        return {"items": [to_dict(r) for r in rows]}


@router.post("/models/{model_id}/activate",
             dependencies=[Depends(require_role("admin"))])
def api_activate_model(model_id: int, gate: bool = True, force: bool = False):
    """激活指定模型（按 (category, version) 热切换 Demo5 快照）。

    M5a 质量门控（gate=true 默认开）：激活前用品类锚定集对候选版本与当前
    版本平行打分，候选 AUROC < 当前 - 0.02 时返回 409（detail 带
    gate_report 双方指标）；force=true 跳过阻断但仍附 gate_report。
    锚定有标注图 <4 张时门控自动跳过（skipped=True 放行）。
    """
    gate_report = None
    if gate:
        try:
            category, version = sl_service.resolve_model_version(model_id)
            gate_report = eval_service.evaluate_activation_gate(
                category, version)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:  # noqa: BLE001 门控自身故障不阻断激活
            raise HTTPException(status_code=500, detail=f"门控评估失败: {e}")
        if not gate_report.get("passed", True) and not force:
            raise HTTPException(
                status_code=409,
                detail={"message": "激活门控拒绝：候选版本锚定集 AUROC "
                                   "显著低于当前版本（可 force=true 强制激活）",
                        "gate_report": gate_report})
    try:
        out = sl_service.activate_model(model_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"激活失败: {e}")
    if gate_report is not None:
        out["gate_report"] = gate_report
    return out


@router.post("/models/rollback",
             dependencies=[Depends(require_role("admin"))])
def api_rollback_model(req: RollbackRequest):
    """回滚到上一快照版本（engine.rollback → 次新版本）。"""
    try:
        from ..engine import get_engine
        target = get_engine().rollback(req.category)
        register_demo5_snapshots()  # 刷新 is_active 指针
        log_action("models_rollback",
                   f"category={req.category} active=v{target}")
        return {"ok": True, "category": req.category,
                "active_version": f"v{target}"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"回滚失败: {e}")


# ── 模型快照导出 / 导入（跨机器/跨品类复用）──────────────────
# 合法快照的关键文件（persist.save_snapshot 落盘契约，导入校验用）
_SNAP_REQUIRED = ("cfg.yaml", "slot_state.pkl", "weights.json",
                  "decider.json", "calibrators.npz", "meta.json")


def _validate_snapshot_category(category: str) -> str:
    """品类名安全校验：将拼进快照目录路径，必须防路径穿越。"""
    cat = (category or "").strip()
    if (not cat or cat in (".", "..", "_archive")
            or "/" in cat or "\\" in cat or ".." in cat):
        raise HTTPException(status_code=400, detail=f"非法品类名: {category!r}")
    return cat


@router.get("/models/{category}/snapshot/export",
            dependencies=[Depends(require_role("engineer"))])
def api_export_snapshot(category: str, version: str | None = None,
                        archive: bool = True):
    """导出品类快照为 zip（version 缺省=当前激活版本，可传 v3 或 3）。

    zip 内容为快照目录全部文件（相对路径），附加 export_manifest.json
    记录 category/version/导出时间，供导入端溯源。

    archive=True（默认）：同步落盘一份到 storage/models/ 模型档案库
    （2026-08-30 补强：该目录此前是遗留空挂目录、无任何写入方）。
    归档名 {category}_v{v}_snapshot.zip，同版本重导出覆盖（幂等，容量
    受 品类×版本 数约束，生命周期登记见 docs/数据生命周期与清理策略.md）。
    """
    from ..engine import get_engine
    category = _validate_snapshot_category(category)
    engine = get_engine()
    cat_dir = os.path.join(engine.snap_root, category)
    if version in (None, "", "current"):
        v = engine.current_version(category)
        if v is None:
            raise HTTPException(status_code=404,
                                detail=f"品类 '{category}' 没有激活版本可导出")
    else:
        m = re.search(r"(\d+)", str(version))
        if m is None:
            raise HTTPException(status_code=400,
                                detail=f"版本号无法解析: {version}")
        v = int(m.group(1))
    vdir = os.path.join(cat_dir, f"v{v}")
    if not os.path.isdir(vdir):
        raise HTTPException(status_code=404,
                            detail=f"快照不存在: {category}/v{v}")
    missing = [f for f in _SNAP_REQUIRED
               if not os.path.exists(os.path.join(vdir, f))]
    if missing:
        raise HTTPException(status_code=409,
                            detail=f"快照文件缺失，不可导出: {missing}")

    # 打包到临时目录，响应发送完毕后由 background 任务清理
    tmp_dir = tempfile.mkdtemp(prefix="snap_export_")
    zip_name = f"{category}_v{v}_snapshot.zip"
    zip_path = os.path.join(tmp_dir, zip_name)
    files: list = []
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, fns in os.walk(vdir):
            for fn in sorted(fns):
                fp = os.path.join(root, fn)
                rel = os.path.relpath(fp, vdir)
                zf.write(fp, rel)
                files.append(rel)
        manifest = {"category": category, "version": v,
                    "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "files": files}
        zf.writestr("export_manifest.json",
                    json.dumps(manifest, ensure_ascii=False, indent=2))
    # 模型档案库落盘（storage/models 写入方，见 docstring）
    archive_path = ""
    if archive:
        archive_path = str(get_settings().storage("models") / zip_name)
        shutil.copy2(zip_path, archive_path)
    log_action("snapshot_export", f"category={category} version=v{v} "
                                  f"files={len(files)} "
                                  f"archive={archive_path or '-'}",
               extra={"category": category, "version": v,
                      "archive_path": archive_path})
    return FileResponse(
        zip_path, filename=zip_name, media_type="application/zip",
        background=BackgroundTask(shutil.rmtree, tmp_dir, True),
        headers={"X-Archive-Path": archive_path} if archive_path else None)


def _rewrite_snapshot_paths(vdir: str, engine) -> list:
    """跨机器兼容：修正快照 cfg.yaml 内的绝对路径为本机路径。

    快照内 cfg.yaml 由源机 save_snapshot 原样落盘，slots.disc/shead 的
    init_from（预训练头）与 backbone.checkpoint（PDN 蒸馏权重）是源机
    绝对路径；本机不存在时按当前基础配置（engine.base_cfg）同名项替换，
    无对应项则移除该键（disc/shead 的 init_from 缺失时槽位先随机建壳、
    随后被 slot_state.pkl 覆盖，不影响加载，见 slots/disc.py:157）。
    ssocl/regress_gate.json 的 anchor_paths 是源机训练图路径，无法映射，
    保持原样——persist.load_snapshot 对缺失路径有容错（降级警告）。
    返回修正记录列表（日志用）。
    """
    import yaml
    fixed: list = []
    cfg_path = os.path.join(vdir, "cfg.yaml")
    if not os.path.exists(cfg_path):
        return fixed
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except Exception:  # noqa: BLE001 cfg 损坏交加载阶段报错
        return fixed
    base = engine.base_cfg or {}

    def _fix(container, base_container, key: str, label: str) -> bool:
        old = (container or {}).get(key)
        if not old or not os.path.isabs(str(old)) or os.path.exists(str(old)):
            return False                     # 无路径/相对路径/本机存在：不动
        new = (base_container or {}).get(key)
        if new and os.path.exists(str(new)):
            container[key] = new
            fixed.append(f"{label}: {old} -> {new}")
        else:
            container.pop(key, None)
            fixed.append(f"{label}: {old}（源机路径本机不存在，已移除）")
        return True

    changed = False
    slots_cfg = cfg.get("slots") or {}
    base_slots = base.get("slots") or {}
    for slot_name in ("disc", "shead"):
        changed |= _fix(slots_cfg.get(slot_name), base_slots.get(slot_name),
                        "init_from", f"slots.{slot_name}.init_from")
    changed |= _fix(cfg.get("backbone"), base.get("backbone"),
                    "checkpoint", "backbone.checkpoint")
    if changed:
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return fixed


@router.post("/models/snapshot/import",
             dependencies=[Depends(require_role("engineer"))])
def api_import_snapshot(file: UploadFile = File(...),
                        category: str = Form(...)):
    """导入快照 zip 到目标品类（允许与导出品类不同，跨品类复用）。

    流程：校验 zip 为合法快照（含 cfg.yaml/slot_state.pkl 等关键文件）→
    解压到 snapshots/{目标品类}/v{下一版本号}/（不覆盖现有版本）→
    修正源机绝对路径 → 登记 Model 行（origin=uploaded，防孤儿归档）→
    返回新版本号。**不自动激活**（不写 current.json），由用户验收后
    在版本历史中显式激活；目标品类已有激活版本时保持其激活状态。
    """
    from ..engine import get_engine
    cat = _validate_snapshot_category(category)
    engine = get_engine()

    # 1. 上传落临时文件并打开 zip
    tmp_dir = tempfile.mkdtemp(prefix="snap_import_")
    zip_path = os.path.join(tmp_dir, "upload.zip")
    try:
        with open(zip_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        try:
            zf = zipfile.ZipFile(zip_path)
        except zipfile.BadZipFile:
            raise HTTPException(status_code=400,
                                detail="上传文件不是合法的 zip 包")
        with zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
            names_set = set(names)
            # 2. 定位快照根：导出包为平铺（根级），兼容单顶层目录包裹的 zip
            roots = [""]
            tops = {n.split("/", 1)[0] for n in names if "/" in n}
            if len(tops) == 1:
                roots.append(next(iter(tops)) + "/")
            prefix = next(
                (r for r in roots
                 if all(f"{r}{f}" in names_set for f in _SNAP_REQUIRED)),
                None)
            if prefix is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"zip 不是合法模型快照（缺关键文件 "
                           f"{list(_SNAP_REQUIRED)}）")
            manifest: dict = {}
            if f"{prefix}export_manifest.json" in names_set:
                try:
                    manifest = json.loads(
                        zf.read(f"{prefix}export_manifest.json")
                        .decode("utf-8"))
                except Exception:  # noqa: BLE001 manifest 损坏不阻断导入
                    manifest = {}

            # 3. 计算新版本号（不覆盖现有版本）并解压（防 zip-slip）
            cat_dir = os.path.join(engine.snap_root, cat)
            os.makedirs(cat_dir, exist_ok=True)
            vs = engine.list_versions(cat)
            new_v = max(vs) + 1 if vs else 1
            vdir = os.path.join(cat_dir, f"v{new_v}")
            vdir_abs = os.path.abspath(vdir)
            os.makedirs(vdir, exist_ok=True)
            try:
                for n in names:
                    if not n.startswith(prefix):
                        continue
                    rel = n[len(prefix):]
                    if not rel:
                        continue
                    dest = os.path.abspath(
                        os.path.join(vdir, *rel.split("/")))
                    if os.path.commonpath([dest, vdir_abs]) != vdir_abs:
                        raise HTTPException(status_code=400,
                                            detail=f"zip 含非法路径: {n}")
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    with zf.open(n) as src, open(dest, "wb") as dst:
                        shutil.copyfileobj(src, dst)
            except Exception:
                shutil.rmtree(vdir, ignore_errors=True)   # 失败不残留半成品
                raise

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # 4. meta.json 改写：版本号对齐新落地版本，记录导入来源
    meta_path = os.path.join(vdir, "meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta["imported_from"] = {
        "category": manifest.get("category"),
        "version": manifest.get("version") or meta.get("version"),
        "exported_at": manifest.get("exported_at")}
    meta["imported_from"] = {k: v for k, v in meta["imported_from"].items()
                             if v is not None}
    meta["version"] = new_v
    meta["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 5. 跨机器路径修正（cfg.yaml 内 init_from/checkpoint 绝对路径）
    path_fixes = _rewrite_snapshot_paths(vdir, engine)

    # 6. 登记 Model 行（origin=uploaded → register 扫描豁免孤儿归档），
    #    is_active=False：导入不自动激活
    with session_scope() as s:
        row = (s.query(Model)
               .filter(Model.category == cat,
                       Model.version == f"v{new_v}").first())
        if row is None:
            s.add(Model(name=f"{cat}_v{new_v}", level="0", category=cat,
                        version=f"v{new_v}", path=vdir,
                        format="demo5_snapshot", origin="uploaded",
                        metrics={"fit_seconds": meta.get("fit_seconds"),
                                 "slot_names": meta.get("slot_names"),
                                 "note": meta.get("note"),
                                 "imported_from": meta["imported_from"]},
                        is_active=False))
    register_demo5_snapshots()           # 同步 is_active 与现有激活指针
    log_action("snapshot_import",
               f"category={cat} new_version=v{new_v} "
               f"from={meta.get('imported_from')} path_fixes={len(path_fixes)}")
    return {"ok": True, "category": cat, "version": f"v{new_v}",
            "version_num": new_v, "imported_from": meta["imported_from"],
            "path_fixes": path_fixes,
            "active_version": engine.current_version(cat),
            "activated": False,
            "message": f"已导入为 v{new_v}（未激活）；请在版本历史中"
                       f"验收后手动激活"}


def _prepare_demo5(req: ModelPrepareRequest, progress_cb) -> dict:
    """从 DB 组装 bundle 并调 engine.prepare，完成后登记快照。

    前端反馈 v9：请求带 datasource_id 时，样本取自该数据源的
    「预训练组」（正常 N + 异常 M），val 取「检测组」；否则回退旧逻辑
    （DB split=train/test 取）。
    """
    from ..engine import get_engine

    progress_cb(0, 3, "读取数据库样本...")
    from ..core.groups import compute_groups
    from ..db.models import DataSource
    if req.datasource_id is not None:
        # ── 前端反馈 v9：样本取自数据源预训练组/检测组 ──
        with session_scope() as s:
            src = s.get(DataSource, req.datasource_id)
            if src is None:
                raise ValueError(f"数据源不存在: {req.datasource_id}")
            pj = src.plan_json or {}
            groups = compute_groups(
                s, req.datasource_id,
                (src.pretrain_normal if src.pretrain_normal is not None else 100),
                (src.pretrain_anomaly if src.pretrain_anomaly is not None else 30),
                (src.batch_size if src.batch_size is not None else 30),
                pj.get("pretrain_ids"))
        g = groups.get(req.category) or {}
        pretrain_ids = g.get("pretrain") or []
        detect_ids = [i for b in (g.get("detect_batches") or []) for i in b]
        need_ids = pretrain_ids + detect_ids[:90]
        with session_scope() as s:
            rows = (s.query(ImageRow.id, ImageRow.path, ImageRow.label,
                            ImageRow.dataset_id)
                    .filter(ImageRow.id.in_(need_ids)).all())
        info = {i: (p, lb) for i, p, lb, _d in rows}
        pre = [info[i] for i in pretrain_ids if i in info]
        det = [info[i] for i in detect_ids if i in info]
        normals = [p for p, lb in pre if lb == "normal"]
        defects = [p for p, lb in pre if lb == "anomaly"]
        defect_paths = set(defects)
        val = [(p, 1 if lb == "anomaly" else 0)
               for p, lb in det if p not in defect_paths][:30]
        ds_ids = sorted({d for _, _, _, d in rows if d is not None})
        templates, template_rows = [], []
        if req.scenario == "L3":
            with session_scope() as s:
                template_rows = (s.query(ImageRow)
                                 .filter(ImageRow.category == req.category,
                                         ImageRow.split == "template")
                                 .order_by(ImageRow.id).limit(500).all())
            templates = [r.path for r in template_rows]
        # 2026-08-31 走查改进：门槛 3 → 1。引擎层 Pipeline.fit 实测支持
        # 1 张正常图 + 0 缺陷冷启动（shead 用伪异常训练）。此前 ≥3 门槛
        # 拒绝了"仅品类+单金样板"场景（shift2 走查复现：模板图即唯一正常参照），
        # 与 L0/L1a 场景设计矛盾。仍保留 precheck 的少样本提示。
        if len(normals) < 1:
            raise ValueError(
                f"数据源预训练组正常图不足（{len(normals)}<1），"
                "请在数据源编辑中调整预训练组数量或补充数据")
    else:
        with session_scope() as s:
            normal_rows = (s.query(ImageRow)
                           .filter(ImageRow.category == req.category,
                                   ImageRow.split == "train",
                                   ImageRow.label == "normal")
                           .order_by(ImageRow.id).all())
            normals = [r.path for r in normal_rows]
            # 先取缺陷（init_defect 优先），val 从剩余 test 行取——M12c 修复：
            # 旧顺序 val 先取前 30 行 test（含缺陷），隔离过滤后 init_defect 可能
            # 被吃空（bottle 实测 defect=0，判别头退化为仅伪异常训练）
            defect_rows = (s.query(ImageRow)
                           .filter(ImageRow.category == req.category,
                                   ImageRow.split.in_(("train_anomaly", "test")),
                                   ImageRow.label == "anomaly")
                           .order_by(ImageRow.id).limit(30).all())
            defects = [r.path for r in defect_rows]
            defect_paths = set(defects)
            val_rows = (s.query(ImageRow)
                        .filter(ImageRow.category == req.category,
                                ImageRow.split == "test",
                                ImageRow.label.in_(("normal", "anomaly")))
                        .order_by(ImageRow.id).limit(90).all())
            val = [(r.path, 1 if r.label == "anomaly" else 0)
                   for r in val_rows if r.path not in defect_paths][:30]
            # L3 模板图（M14a）：用户显式标记 split=template 的图（红线：显式提供，
            # 禁止隐式配对）。L3 场景无模板时 tpl 槽位空转——precheck 已提示。
            # 2026-08-30（P0）：模板收集上限 8 → 500——8 张覆盖不了正常样式分布，
            # gold_finger 系统链路实测 tpl 反向（0.03 vs 全库 85 张 1.0000）。
            templates = []
            template_rows = []
            if req.scenario == "L3":
                template_rows = (s.query(ImageRow)
                                 .filter(ImageRow.category == req.category,
                                         ImageRow.split == "template")
                                 .order_by(ImageRow.id).limit(500).all())
                templates = [r.path for r in template_rows]
            # M16f 数据谱系（review R6）：本次 prepare 用到的批次 id 集合，
            # 写入 Model.metrics.dataset_ids，批次详情可反查"参与了哪个版本"
            ds_ids = sorted({int(r.dataset_id)
                             for r in (list(normal_rows) + list(defect_rows)
                                       + list(val_rows) + list(template_rows))
                             if r.dataset_id is not None})
        # 2026-08-31 走查改进：门槛 3 → 1。引擎层 Pipeline.fit 实测支持
        # 1 张正常图 + 0 缺陷冷启动（shead 用伪异常训练）。此前 ≥3 门槛
        # 拒绝了"仅品类+单金样板"场景（shift2 走查复现：模板图即唯一正常参照），
        # 与 L0/L1a 场景设计矛盾。仍保留 precheck 的少样本提示。
        if len(normals) < 1:
            raise ValueError(f"品类 '{req.category}' train 正常图不足 "
                             f"（{len(normals)}<1），请先在数据管理导入数据")

    bundle = {"init_normal": normals, "init_defect": defects,
              "val": val, "test": [], "templates": templates}
    progress_cb(1, 3, f"Demo5 准备（normal={len(normals)} "
                      f"defect={len(defects)} val={len(val)} "
                      f"tpl={len(templates)} "
                      f"scenario={req.scenario} profile={req.profile}）...")
    version = get_engine().prepare(
        req.category, bundle, scenario=req.scenario, profile=req.profile,
        trigger="fit", note=req.note, force=req.force)
    progress_cb(2, 3, "登记快照...")
    register_demo5_snapshots(protect={req.category})
    # M16f：谱系落库（register 合并写，重扫描不丢）
    if ds_ids:
        with session_scope() as s:
            row = (s.query(Model)
                   .filter(Model.category == req.category,
                           Model.version == f"v{version}").first())
            if row is not None:
                merged = dict(row.metrics or {})
                merged["dataset_ids"] = ds_ids
                row.metrics = merged
                s.flush()
    progress_cb(3, 3, "完成")
    log_action("model_prepare",
               f"category={req.category} scenario={req.scenario} "
               f"profile={req.profile} version=v{version}")
    return {"category": req.category, "version": f"v{version}",
            "version_num": version, "scenario": req.scenario,
            "profile": req.profile, "n_normal": len(normals),
            "n_defect": len(defects), "n_val": len(val),
            "n_templates": len(templates),
            "message": "模型准备完成"}


@router.get("/models/precheck/{category}")
def api_models_precheck(category: str):
    """prepare 前数据体检（M6a）：split 分布 + 分辨率分布 + 建议 scenario/profile。

    纯 DB 统计（毫秒级，无模型加载），供准备模型对话框在拟合前
    判断"数据够不够、该选哪层场景"。取数口径与 _prepare_demo5 一致：
    init_normal=train/normal，init_defect=train_anomaly|test/anomaly。
    """
    with session_scope() as s:
        rows = s.query(ImageRow).filter(ImageRow.category == category).all()

    counts = {"train_normal": 0, "train_anomaly": 0,
              "test_normal": 0, "test_anomaly": 0, "feedback": 0,
              "template": 0}
    res_counter: dict = {}
    for r in rows:
        if r.split == "train" and r.label == "normal":
            counts["train_normal"] += 1
        elif r.split == "train_anomaly" and r.label == "anomaly":
            counts["train_anomaly"] += 1
        elif r.split == "test" and r.label == "normal":
            counts["test_normal"] += 1
        elif r.split == "test" and r.label == "anomaly":
            counts["test_anomaly"] += 1
        elif r.split == "feedback":
            counts["feedback"] += 1
        elif r.split == "template":
            counts["template"] += 1
        if r.width and r.height:
            key = (int(r.width), int(r.height))
            res_counter[key] = res_counter.get(key, 0) + 1

    resolutions = [{"width": w, "height": h, "count": c}
                   for (w, h), c in sorted(res_counter.items(),
                                           key=lambda kv: -kv[1])[:3]]

    n_normal = counts["train_normal"]
    has_defect = counts["train_anomaly"] + counts["test_anomaly"] > 0
    warnings: list = []
    if n_normal == 0:
        suggested = None
        warnings.append("无训练数据（train 正常图 0 张），请先在数据管理导入数据")
    elif n_normal < 10:
        suggested = "L0"
        warnings.append(f"正常图不足 10 张（当前 {n_normal} 张），建议 L0 零样本或先补图")
    elif n_normal < 50:
        suggested = "L1a"
        warnings.append(
            f"正常图 {n_normal} 张处于少样本区（10-49）：CDF 校准粒度较粗，"
            f"AUROC 可能系统性低估约 0.1（M9 验收实测）；"
            f"建议补到 50 张以上达到协议口径")
        if not has_defect:
            warnings.append("暂无缺陷样本，disc/shead 判别头将仅用伪异常训练；"
                            "可先补少量缺陷图提升判别力")
    elif n_normal < 100:
        suggested = "L1b"
        if not has_defect:
            warnings.append("暂无缺陷样本，判别头将仅用伪异常训练")
    else:
        suggested = "L1b"
        if not has_defect:
            warnings.append("暂无缺陷样本，判别头将仅用伪异常训练")
    if len(res_counter) > 1:
        warnings.append(f"分辨率不统一（{len(res_counter)} 种），"
                        f"建议导入前统一尺寸以保证分块一致性")
    # L3 模板提示（M14a）：有模板图时提示可选 L3；选了 L3 但无模板时 tpl 空转
    if counts["template"] > 0:
        warnings.append(f"已有 {counts['template']} 张模板图（split=template），"
                        f"强配准品类可选 L3 激活 tpl 模板差分槽位")
    return {"category": category, "counts": counts, "resolutions": resolutions,
            "has_template": counts["template"] > 0,
            "suggested_scenario": suggested, "suggested_profile": "fast",
            "warnings": warnings}


@router.post("/models/prepare",
             dependencies=[Depends(require_role("engineer"))])
def api_models_prepare(req: ModelPrepareRequest):
    """后台任务：Demo5 品类准备（fit + 快照 + 激活）。

    样本取自 DB：split=train&label=normal → init_normal；
    split∈(train_anomaly,test) 缺陷图（≤30）→ init_defect；
    split=test 抽样（≤30）→ val。
    """
    task_id = task_manager.submit("model_prepare", req.model_dump(),
                                  lambda cb: _prepare_demo5(req, cb))
    return {"task_id": task_id}


# ── 评估 ─────────────────────────────────────────────────────
@router.post("/eval/accuracy")
def api_eval_accuracy(req: EvalAccuracyRequest):
    """后台任务：精度评估（AUROC/AP/F1/阈值/混淆矩阵）。"""
    def fn(progress_cb):
        return eval_service.run_accuracy_eval(
            req.category, split=req.split, name=req.name, limit=req.limit,
            progress_cb=progress_cb)

    task_id = task_manager.submit("eval", req.model_dump(), fn)
    return {"task_id": task_id}


@router.post("/eval/benchmark")
def api_eval_benchmark(req: EvalBenchmarkRequest):
    """后台任务：延迟基准（对照 200ms 考核预算）。"""
    def fn(progress_cb):
        return eval_service.run_latency_benchmark(
            req.category, n_images=req.n_images, warmup=req.warmup,
            progress_cb=progress_cb)

    task_id = task_manager.submit("benchmark", req.model_dump(), fn)
    return {"task_id": task_id}


@router.get("/eval/runs")
def api_list_eval_runs(page: int = 1, page_size: int = 50):
    with session_scope() as s:
        q = s.query(EvalRun)
        total = q.count()
        rows = (q.order_by(EvalRun.id.desc())
                 .offset((page - 1) * page_size).limit(page_size).all())
        return {"total": total, "items": [to_dict(r) for r in rows]}


@router.get("/eval/runs/{run_id}")
def api_get_eval_run(run_id: int):
    with session_scope() as s:
        row = s.get(EvalRun, run_id)
        if row is None:
            raise HTTPException(status_code=404, detail="评估记录不存在")
        return to_dict(row)


@router.delete("/eval/runs/{run_id}",
               dependencies=[Depends(require_role("admin"))])
def api_delete_eval_run(run_id: int):
    """删除评估记录（admin；仅删记录行，不触碰快照/图片/模型数据）。

    2026-08-29 用户要求补充：此前评估记录只读不可删，误建/噪音记录
    无法清理。审计走 log_action(eval_run_delete)。
    """
    with session_scope() as s:
        row = s.get(EvalRun, run_id)
        if row is None:
            raise HTTPException(status_code=404, detail="评估记录不存在")
        info = {"run_id": row.id, "run_type": row.run_type,
                "category": row.category, "name": row.name}
        s.delete(row)
    log_action("eval_run_delete",
               f"run_id={info['run_id']} type={info['run_type']} "
               f"category={info['category']} name={info['name']}")
    return {"deleted": run_id}


@router.post("/eval/compare")
def api_eval_compare(req: EvalCompareRequest):
    """多次评估结果对比。"""
    try:
        return eval_service.compare_eval_runs(req.run_ids)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"对比失败: {e}")


@router.post("/eval/ab_compare")
def api_eval_ab_compare(req: ABCompareRequest):
    """后台任务：A/B 影子对比（version_a 默认当前激活 vs version_b 候选快照，
    同集平行打分 + 分歧样本）。"""
    def fn(progress_cb):
        return eval_service.run_ab_compare(
            req.category, version_a=req.version_a, version_b=req.version_b,
            split=req.split, limit=req.limit, progress_cb=progress_cb)

    task_id = task_manager.submit("ab_compare", req.model_dump(), fn)
    return {"task_id": task_id}


@router.post("/eval/robustness")
def api_eval_robustness(req: EvalRobustnessRequest):
    """六维评价·维度2：噪声鲁棒性（扰动下 AUC 衰减 + A_rob 曲线下面积）。"""
    def fn(progress_cb):
        return eval_service.run_robustness_eval(
            req.category, split=req.split, limit=req.limit,
            noise_kinds=req.noise_kinds, levels=req.levels,
            progress_cb=progress_cb)

    task_id = task_manager.submit("robustness", req.model_dump(), fn)
    return {"task_id": task_id}


# ── 校准集管理（Demo5 无此概念：CDF 在线校准）─────────────────
_CALIBRATION_GONE = ("Demo5 引擎使用 CDF 在线校准，无需手动校准集管理"
                     "（/calibration/* 已随 demo1 引擎下线）")


@router.get("/calibration/{category}")
def api_calibration_info(category: str):
    raise HTTPException(status_code=410, detail=_CALIBRATION_GONE)


@router.post("/calibration/exclude")
def api_calibration_exclude(req: CalibrationExcludeRequest):
    raise HTTPException(status_code=410, detail=_CALIBRATION_GONE)
