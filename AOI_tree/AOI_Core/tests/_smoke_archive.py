# -*- coding: utf-8 -*-
"""存档功能冒烟测试（2026-10-09，临时脚本，验证后可删）。
用独立临时 SQLite 库验证：
1. v6 迁移：旧库补 archived 列且 user_version=6
2. archive/unarchive API 行为 + 列表过滤
3. 删除工单级联清除 Detection/Feedback/ConsolidationFeedback
4. 全局统计口径排除存档工单、保留 NULL 孤儿记录
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_tmp = tempfile.mkdtemp(prefix="aoi_smoke_")
_cfg = os.path.join(_tmp, "config.yaml")
with open(_cfg, "w", encoding="utf-8") as f:
    f.write(
        "system:\n"
        f"  database_url: sqlite:///{os.path.join(_tmp, 'smoke.db').replace(os.sep, '/')}\n"
        f"  storage_dir: {os.path.join(_tmp, 'storage').replace(os.sep, '/')}\n"
    )
os.environ["AOI_CONFIG"] = _cfg

from backend.db import database as dbmod  # noqa: E402
from backend.db.models import (  # noqa: E402
    ConsolidationFeedback, Detection, Feedback, WorkOrder,
)
from backend.api import routes_wo_order as rwo  # noqa: E402
from backend.api import routes_stats as rst  # noqa: E402

ok = 0
fail = 0


def check(name, cond):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS {name}")
    else:
        fail += 1
        print(f"  FAIL {name}")


# ---------- 1. 迁移 ----------
print("[1] v6 migration")
eng = dbmod.get_engine()
with eng.connect() as c:
    from sqlalchemy import text
    ver = c.execute(text("PRAGMA user_version")).scalar()
    cols = [r[1] for r in c.execute(text("PRAGMA table_info(work_orders)")).all()]
check("user_version == 6", ver == 6)
check("work_orders.archived exists", "archived" in cols)

Session = dbmod._SessionLocal

# ---------- 2. 造数据 ----------
s = Session()
wo1 = WorkOrder(name="WO-进行中", status="pending")
wo2 = WorkOrder(name="WO-待存档", status="pending")
s.add_all([wo1, wo2])
s.commit()
w1, w2 = wo1.id, wo2.id

# wo2 挂一条检测 + 反馈 + 合并反馈；另造一条孤儿检测(workorder_id=None)
det = Detection(workorder_id=w2, image_path="/x/a.jpg", status="done")
orphan = Detection(workorder_id=None, image_path="/x/b.jpg", status="done")
s.add_all([det, orphan])
s.commit()
fb = Feedback(detection_id=det.id, action="confirm")
s.add(fb)
s.commit()
cf = ConsolidationFeedback(feedback_id=fb.id, final_label="ok")
s.add(cf)
s.commit()
det_id, orphan_id, fb_id = det.id, orphan.id, fb.id
s.close()

# ---------- 3. archive/unarchive + 列表过滤 ----------
print("[2] archive/unarchive + list filter")
import asyncio

r = asyncio.run(rwo.api_archive_workorder(w2))
check("archive returns archived=True", r.get("archived") is True)

lst_default = asyncio.run(rwo.api_list_workorders())
ids_default = [w["id"] for w in lst_default["items"]]
check("default list hides archived", w2 not in ids_default and w1 in ids_default)

lst_arch = asyncio.run(rwo.api_list_workorders(archived=True))
ids_arch = [w["id"] for w in lst_arch["items"]]
check("archived=True list shows archived", w2 in ids_arch and w1 not in ids_arch)
check("item carries archived flag",
      all("archived" in w for w in lst_arch["items"]))

r = asyncio.run(rwo.api_unarchive_workorder(w2))
check("unarchive returns archived=False", r.get("archived") is False)
lst_default = asyncio.run(rwo.api_list_workorders())
check("unarchive restores to default list",
      w2 in [w["id"] for w in lst_default["items"]])

# ---------- 4. 全局统计排除存档 ----------
print("[3] stats excludes archived (global scope)")
asyncio.run(rwo.api_archive_workorder(w2))
ov = asyncio.run(rst.api_stats_overview(workorder_id=None, range_name="all"))
# 只剩孤儿检测 1 条
check("overview total == 1 (orphan only)", ov.get("total") == 1 or ov.get("total_detections") == 1)

asyncio.run(rst.api_stats_quality_overview(workorder_id=None, range_name="all"))
asyncio.run(rst.api_stats_mistake_trend(workorder_id=None, range_name="all"))
asyncio.run(rst.api_stats_defect_types(workorder_id=None, range_name="all"))
asyncio.run(rst.api_stats_batch_summary(workorder_id=None, range_name="all"))
asyncio.run(rst.api_stats_health(workorder_id=None))
print("  PASS stats endpoints no exception with archived exclusion")
ok += 1

# ---------- 5. 级联删除 ----------
print("[4] delete workorder cascades")
asyncio.run(rwo.api_delete_workorder(w2))
s = Session()
check("detection deleted", s.query(Detection).filter(Detection.id == det_id).count() == 0)
check("feedback deleted", s.query(Feedback).filter(Feedback.id == fb_id).count() == 0)
check("consolidation deleted", s.query(ConsolidationFeedback).filter(
    ConsolidationFeedback.feedback_id == fb_id).count() == 0)
check("orphan detection kept", s.query(Detection).filter(Detection.id == orphan_id).count() == 1)
check("workorder deleted", s.query(WorkOrder).filter(WorkOrder.id == w2).count() == 0)
s.close()

print(f"\nRESULT: {ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
