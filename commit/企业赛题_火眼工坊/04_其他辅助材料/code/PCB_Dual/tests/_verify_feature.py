"""P2 验证：用 SMT test_images 收集 OK/NG 图 → prepare 品类模型 → predict。

特征引擎品类模型与模板是两个正交概念：
- 模板 = 金样板（传统差分参考）
- 品类模型 = 特征库快照（AOI 特征学习参考）
这里用 SMT 同域图准备一个 mini 品类模型，验证 predict 路径。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.engines.feature import get_engine  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SMT = ROOT / "alg_repo" / "smt"
TEST_IMAGES = SMT / "test_images"


def collect() -> tuple[list[str], list[str]]:
    normals, defects = [], []
    for pair_dir in sorted(TEST_IMAGES.iterdir()):
        if not pair_dir.is_dir():
            continue
        for img in sorted(pair_dir.rglob("*")):
            if img.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                continue
            name = img.name
            if "_OK" in name:
                normals.append(str(img))
            elif "_OK." not in name and "OK" not in name:
                defects.append(str(img))
    return normals, defects


def main() -> None:
    normals, defects = collect()
    print(f"收集 OK 图 {len(normals)} 张，NG 图 {len(defects)} 张", flush=True)
    assert normals, "无正常图"
    assert defects, "无缺陷图"

    # 红线：init_normal 必须来自 train 域路径。API 层负责把用户图整理到
    # storage/train/{category}/normal|defect/（此处为验证直接整理）
    from app.config import get_settings
    settings = get_settings()
    train_cat = settings.storage("train") / "smt"
    normal_dir = train_cat / "normal"
    defect_dir = train_cat / "defect"
    import shutil
    for d in (normal_dir, defect_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(normals):
        shutil.copy2(p, normal_dir / f"n{i:03d}{Path(p).suffix}")
    for i, p in enumerate(defects):
        shutil.copy2(p, defect_dir / f"d{i:03d}{Path(p).suffix}")
    train_normal = [str(p) for p in sorted(normal_dir.iterdir())]
    train_defect = [str(p) for p in sorted(defect_dir.iterdir())]
    print(f"整理 train 域：正常 {len(train_normal)}，缺陷 {len(train_defect)}", flush=True)

    engine = get_engine()
    t0 = time.time()
    ver = engine.prepare(
        "smt_solder",
        {"init_normal": train_normal, "init_defect": train_defect, "val": [], "test": []},
        scenario="L1a", profile="fast",
        note="SMT test_images mini 品类模型",
        force=True,
    )
    print(f"品类模型已准备 v{ver}，耗时 {time.time() - t0:.1f}s", flush=True)

    # predict NG / OK（用原始图测）
    for label, paths in (("NG", defects), ("OK", normals)):
        for p in paths[:2]:
            r = engine.predict_image_path("smt_solder", p)
            print(f"[{label}] {Path(p).name} -> decision={r.decision} score={r.score:.3f} "
                  f"is_anomaly={r.is_anomaly} latency={r.latency_ms:.0f}ms "
                  f"slots={ {k: round(v, 2) for k, v in r.slot_scores.items()} }", flush=True)


if __name__ == "__main__":
    main()
