"""冒烟：数据源级联删除（不足清单 #6 补强，2026-08-30）。

直连 DB 层（无需 server）验证三点：
1. DB 级联：批次/图片/检测/反馈/伪异常行全部清理，不留孤儿；
2. purge_files=True：storage 内副本/热力图/伪异常/批次目录被删；
3. 安全护栏：外部原始目录（用户数据）永不自动删除。

用法：python tests/_smoke_ds_cascade.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image as PILImage

from app.config import get_settings
from app.core.datasets import delete_datasource_cascade
from app.core.ingest import import_folder
from app.db.database import session_scope
from app.db.models import (DataSource, Dataset, Detection, Feedback,
                           Image as ImageRow, PseudoAnomaly)


def _make_png(path: Path) -> None:
    PILImage.new("RGB", (64, 64), color=(120, 80, 60)).save(path)


def _count(model, **cond) -> int:
    with session_scope() as s:
        q = s.query(model)
        for k, v in cond.items():
            q = q.filter(getattr(model, k) == v)
        return q.count()


def _new_datasource(name: str) -> int:
    with session_scope() as s:
        s.add(DataSource(name=name, modality="image"))
        s.flush()
        return s.query(DataSource).filter(DataSource.name == name).one().id


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="ds_cascade_"))
    files = [tmp / f"img_{i}.png" for i in range(3)]
    for f in files:
        _make_png(f)

    # ── 场景 A：copy_to_storage=True（storage 副本 + 检测/反馈/伪异常）──
    ds_a = _new_datasource("smoke_cascade_A")
    imp_a = import_folder(tmp, category="cascade_smoke", source="datasource",
                          copy_to_storage=True, datasource_id=ds_a,
                          name="cascade_A_batch")
    assert imp_a["imported"] == 3, imp_a
    dataset_a = imp_a["dataset_id"]
    with session_scope() as s:
        img = (s.query(ImageRow)
               .filter(ImageRow.dataset_id == dataset_a)
               .order_by(ImageRow.id).first())
        img_id = img.id
        hm = get_settings().storage("heatmaps") / f"smoke_hm_{img_id}.png"
        _make_png(hm)
        d = Detection(target_type="image", image_id=img_id, image_path=img.path,
                      category="cascade_smoke", final_score=0.42,
                      is_anomaly=False, latency_ms=12.5, heatmap_path=str(hm))
        s.add(d)
        s.flush()
        det_id = d.id
        s.add(Feedback(detection_id=det_id, feedback_type="false_positive",
                       operator_label=0))
        ps = get_settings().storage("pseudo") / f"smoke_ps_{img_id}.png"
        _make_png(ps)
        s.add(PseudoAnomaly(base_image_id=img_id, method="cutpaste",
                            image_path=str(ps)))
    assert _count(Detection, image_id=img_id) == 1
    assert _count(Feedback, detection_id=det_id) == 1
    assert _count(PseudoAnomaly, base_image_id=img_id) == 1

    r_a = delete_datasource_cascade(ds_a, purge_files=True)
    print("场景A 级联删除:", {k: r_a[k] for k in
          ("datasets", "images", "detections", "feedback", "pseudo",
           "files_deleted", "dirs_deleted", "bytes_freed")}, flush=True)
    assert _count(DataSource, id=ds_a) == 0, "数据源行应删除"
    assert _count(Dataset, id=dataset_a) == 0, "批次行应删除（旧实现留孤儿）"
    assert _count(ImageRow, dataset_id=dataset_a) == 0, "图片行应删除"
    assert _count(Detection, image_id=img_id) == 0
    assert _count(Feedback, detection_id=det_id) == 0
    assert _count(PseudoAnomaly, base_image_id=img_id) == 0
    assert not hm.exists() and not ps.exists(), "热力图/伪异常文件应被删"
    batch_dir = get_settings().storage("images") / "cascade_smoke" / f"ds{dataset_a}"
    assert not batch_dir.exists(), "批次目录应被删"
    assert all(f.exists() for f in files), "外部原始文件不应被删"

    # ── 场景 B：copy_to_storage=False（外部原始目录，永不自动删）──
    ds_b = _new_datasource("smoke_cascade_B")
    imp_b = import_folder(tmp, category="cascade_smoke", source="datasource",
                          copy_to_storage=False, datasource_id=ds_b,
                          name="cascade_B_batch")
    assert imp_b["imported"] == 3, imp_b
    r_b = delete_datasource_cascade(ds_b, purge_files=True)
    print("场景B 级联删除:", {k: r_b[k] for k in
          ("datasets", "images", "external_skipped")}, flush=True)
    assert r_b["external_skipped"], "外部文件应进 external_skipped 报告"
    assert all(f.exists() for f in files), "外部原始文件不应被删"
    assert _count(DataSource, id=ds_b) == 0

    for f in files:
        f.unlink()
    tmp.rmdir()
    print("=== 冒烟通过：级联删除无孤儿行，storage 内文件已清，外部用户数据完好 ===")


if __name__ == "__main__":
    main()
