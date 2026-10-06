"""品类模型 API：准备（fit）/ 列表 / 激活 / 回滚。"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter()


@router.get("/models")
def list_models() -> dict:
    """已准备的品类模型列表（含版本/激活态）。"""
    from app.config import get_settings
    from app.engines.feature import get_engine
    settings = get_settings()
    engine = get_engine()
    snap_root = settings.snapshots_dir
    cats = []
    if snap_root.is_dir():
        for cat_dir in sorted(snap_root.iterdir()):
            if not cat_dir.is_dir():
                continue
            versions = engine.list_snapshots(cat_dir.name)
            cats.append({"category": cat_dir.name,
                         "versions": versions,
                         "current": engine.current_version(cat_dir.name)})
    return {"categories": cats}


@router.post("/models/prepare")
def prepare_model(payload: dict) -> dict:
    """准备品类模型：fit + 存快照 + 激活。

    body: {
      category,                 # 品类名
      normal_dir, defect_dir,   # 正常/缺陷图目录（任意位置，API 整理到 train 域）
      scenario?=L1a, profile?=fast, note?, force?
    }
    """
    from app.config import get_settings
    from app.engines.feature import get_engine

    category = payload.get("category")
    normal_dir = payload.get("normal_dir")
    defect_dir = payload.get("defect_dir")
    if not category or not normal_dir:
        raise HTTPException(status_code=400, detail="需要 category 与 normal_dir")
    ndir = Path(normal_dir)
    if not ndir.is_dir():
        raise HTTPException(status_code=400, detail=f"正常图目录不存在: {normal_dir}")
    ddir = Path(defect_dir) if defect_dir else None
    if ddir is not None and not ddir.is_dir():
        raise HTTPException(status_code=400, detail=f"缺陷图目录不存在: {defect_dir}")

    # 红线：init_normal 必须来自 train 域路径 → 复制到 storage/train/{category}/normal|defect/
    import shutil

    settings = get_settings()
    train_cat = settings.storage("train") / category
    normal_dst = train_cat / "normal"
    defect_dst = train_cat / "defect"
    for d in (normal_dst, defect_dst):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    n_files = sorted(p for p in ndir.iterdir() if p.is_file() and p.suffix.lower() in exts)
    d_files = sorted(p for p in (ddir.iterdir() if ddir else []) if p.is_file() and p.suffix.lower() in exts) \
        if ddir else []
    if not n_files:
        raise HTTPException(status_code=400, detail="正常图目录无图片")
    for i, p in enumerate(n_files):
        shutil.copy2(p, normal_dst / f"n{i:04d}{p.suffix.lower()}")
    for i, p in enumerate(d_files):
        shutil.copy2(p, defect_dst / f"d{i:04d}{p.suffix.lower()}")
    train_normal = [str(p) for p in sorted(normal_dst.iterdir())]
    train_defect = [str(p) for p in sorted(defect_dst.iterdir())]

    engine = get_engine()
    try:
        ver = engine.prepare(
            category,
            {"init_normal": train_normal, "init_defect": train_defect, "val": [], "test": []},
            scenario=payload.get("scenario", "L1a"),
            profile=payload.get("profile", "fast"),
            note=payload.get("note", ""),
            force=bool(payload.get("force", False)),
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"品类准备失败: {exc}")
    return {"category": category, "version": ver, "n_normal": len(train_normal),
            "n_defect": len(train_defect)}


@router.post("/models/{category}/activate")
def activate_model(category: str, payload: dict) -> dict:
    from app.engines.feature import get_engine
    version = payload.get("version")
    if version is None:
        raise HTTPException(status_code=400, detail="需要 version")
    try:
        get_engine().activate(category, int(version))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc))
    return {"category": category, "active_version": int(version)}


@router.post("/models/{category}/rollback")
def rollback_model(category: str) -> dict:
    from app.engines.feature import get_engine
    try:
        target = get_engine().rollback(category)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc))
    return {"category": category, "active_version": target}
