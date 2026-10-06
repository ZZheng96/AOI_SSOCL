"""P0 验证：板级 Inspect + 缺陷明细 + 对位落库。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.database import session_scope  # noqa: E402
from app.db.models import DefectRecord, Detection, Inspect  # noqa: E402
from app.inspect.detect_service import get_detection_service  # noqa: E402
from app.template.store import TemplateStore  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
NG = ROOT / "alg_repo" / "smt" / "test_images" / "excess_pair" / "pair4" / "PixPin_2026-07-01_11-21-15.png"
STD = ROOT / "alg_repo" / "smt" / "test_images" / "excess_pair" / "pair4" / "PixPin_2026-07-01_11-21-15_OK.png"


def main() -> None:
    tpl = TemplateStore().find_by_standard_path(STD)
    t0 = time.time()
    out = get_detection_service().detect_dual(tpl, str(NG), target_type="pipeline")
    print(f"检测 {(time.time()-t0)*1000:.0f}ms overall={out['overall']} "
          f"det={out.get('detection_id')} insp={out.get('inspect_id')} "
          f"align={out.get('align')}")
    with session_scope() as s:
        insp = s.get(Inspect, out["inspect_id"])
        print("Inspect:", insp.id, "total", insp.total_count, "ng", insp.ng_count, "pass", insp.pass_)
        defs = (s.query(DefectRecord)
                .filter(DefectRecord.inspect_id == out["inspect_id"]).all())
        print("DefectRecords:", [(d.defect_type, d.engine, d.x, d.y, d.width, d.height)
                                 for d in defs])
        det = s.get(Detection, out["detection_id"])
        print("Detection align:", det.align_offset_x, det.align_offset_y,
              "warn", det.align_warn, "inspect_id", det.inspect_id)
        assert insp.total_count == 1 and insp.ng_count == 1 and not insp.pass_
        assert len(defs) == 2  # 传统多锡 + AOI


if __name__ == "__main__":
    main()
