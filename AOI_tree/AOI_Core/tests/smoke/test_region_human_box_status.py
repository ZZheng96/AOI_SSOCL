"""AC-063：人工复核框以 source=human_box 进入区域契约 + 每个 region 派生 train_status。"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from algo.decision import localization as L
from backend.db import database
from backend.db.models import Base, Detection


def _regions():
    return [{"region_id": f"r{i}", "bbox": [i * 10, 0, i * 10 + 8, 8], "area": 64,
             "source": "engine_box", "score_peak": 0.9 - i * 0.1, "postproc": "pp-x",
             "status": "candidate", "confirmed": False} for i in range(3)]


def _op(rv, who, op, rid=None, ev="e", **kw):
    return L.apply_op(rv, {"op": op, "region_id": rid, "operator": who, "event_id": ev,
                           "expected_revision": rv["revision"], **kw}, size=[100, 100])


def _by_id(rv):
    return {r["region_id"]: r for r in rv["regions"]}


class RegionHumanBoxPureTest(unittest.TestCase):
    def test_candidate_vs_human_truth(self):
        print("[1/3] 候选框 vs 人工真值：source/reviewer/revision/train_status", flush=True)
        rv = L.new_review(_regions(), "pp-x")
        view = _by_id(L.annotate_review(rv))
        self.assertEqual({r["source"] for r in view.values()}, {"engine_box"})
        self.assertEqual({r["train_status"] for r in view.values()}, {"excluded_candidate"})
        self.assertTrue(all(r["reviewer"] is None and r["review_revision"] is None
                            and r["confirmed"] is False for r in view.values()))

        rv = _op(rv, "甲", "confirm", "r0", "e1")
        rv = _op(rv, "甲", "revise", "r1", "e2", bbox=[1, 1, 9, 9])
        rv = _op(rv, "乙", "add", ev="e3", bbox=[50, 50, 60, 60])
        rv = _op(rv, "甲", "reject", "r2", "e4")
        view = _by_id(L.annotate_review(rv))
        r0, r1, h0, r2 = view["r0"], view["r1"], view["h0"], view["r2"]
        # 人工确认：对外 human_box，保留原始来源
        self.assertEqual((r0["source"], r0["origin_source"], r0["reviewer"], r0["review_revision"],
                          r0["confirmed"], r0["train_status"]),
                         ("human_box", "engine_box", "甲", 1, True, "eligible"))
        self.assertEqual((r1["source"], r1["origin_source"], r1["reviewer"], r1["review_revision"],
                          r1["bbox"], r1["train_status"]),
                         ("human_box", "engine_box", "甲", 2, [1, 1, 9, 9], "eligible"))
        self.assertEqual((h0["source"], h0["origin_source"], h0["reviewer"], h0["review_revision"],
                          h0["train_status"]), ("human_box", "human_box", "乙", 3, "eligible"))
        # 人工拒绝的候选框不冒充人工真值
        self.assertEqual((r2["source"], r2["reviewer"], r2["review_revision"], r2["confirmed"],
                          r2["train_status"]), ("engine_box", "甲", 4, False, "excluded_rejected"))
        # 视图为纯函数：存储中的 engine_box 来源不被改写
        self.assertEqual(_by_id(rv)["r0"]["source"], "engine_box")
        self.assertNotIn("train_status", _by_id(rv)["r0"])
        # 训练选择只取 eligible；视图与原始 review 结果一致
        self.assertEqual(L.trainable_region_boxes(rv), [[0, 0, 8, 8], [1, 1, 9, 9], [50, 50, 60, 60]])
        self.assertEqual(L.trainable_region_boxes(L.annotate_review(rv)), L.trainable_region_boxes(rv))

    def test_disputed_and_inconsistent_excluded(self):
        print("[2/3] disputed / 不一致状态不进训练", flush=True)
        rv = L.new_review(_regions(), "pp-x")
        rv = _op(rv, "甲", "confirm", "r0", "e1")
        rv = _op(rv, "乙", "reject", "r0", "e2")        # 无理由推翻 → disputed
        rv = _op(rv, "甲", "revise", "r1", "e3", bbox=[2, 2, 9, 9])
        rv = _op(rv, "丙", "dispute", "r1", "e4")
        view = _by_id(L.annotate_review(rv))
        self.assertEqual((view["r0"]["source"], view["r0"]["reviewer"], view["r0"]["review_revision"],
                          view["r0"]["confirmed"], view["r0"]["train_status"]),
                         ("engine_box", "乙", 2, False, "excluded_disputed"))
        # 人工修订几何保留 human_box，但争议隔离不训
        self.assertEqual((view["r1"]["source"], view["r1"]["origin_source"],
                          view["r1"]["train_status"], view["r1"]["review_revision"]),
                         ("human_box", "engine_box", "excluded_disputed", 4))
        self.assertEqual(L.trainable_region_boxes(rv), [])
        # 状态与 confirmed 不一致 / 未知状态 / 仅可视化热图框 → 一律排除
        self.assertEqual(L.train_status({"status": "confirmed", "confirmed": False}), "excluded_candidate")
        self.assertEqual(L.train_status({"status": "weird", "confirmed": True}), "excluded_candidate")
        self.assertEqual(L.train_status({"status": "visual_only", "confirmed": False}), "excluded_visual")
        self.assertEqual(L.trainable_region_boxes(
            {"regions": [{"bbox": [0, 0, 1, 1], "status": "confirmed", "confirmed": False}]}), [])


class RegionHumanBoxApiTest(unittest.TestCase):
    def test_api_returns_human_box_contract(self):
        from backend.api.region_review import register
        from backend.core.security import ApiKeyMiddleware
        print("[3/3] 区域复核 API 透出 human_box/reviewer/revision/train_status", flush=True)
        loc = {"regions": _regions(), "postproc": {"version_id": "pp-x"},
               "coord_space": {"size": [100, 100]}}
        with tempfile.TemporaryDirectory() as tmp:
            eng = create_engine(f"sqlite:///{Path(tmp) / 'h.db'}", connect_args={"check_same_thread": False})
            try:
                Base.metadata.create_all(eng)
                SL = sessionmaker(bind=eng, expire_on_commit=False)
                with SL() as s:
                    s.add(Detection(id=1, image_path="x.png", final_score=0.9, is_anomaly=True,
                                    latency_ms=1.0, n_tiles={"localization": loc}))
                    s.commit()
                sec = Mock()
                sec.section.return_value = {"keys": {"t-eng": "engineer"}}
                with patch.object(database, "_engine", eng), patch.object(database, "_SessionLocal", SL), \
                        patch("backend.core.security.get_settings", return_value=sec), \
                        patch("backend.core.security.get_configured_api_key", return_value=""):
                    app = FastAPI()
                    app.add_middleware(ApiKeyMiddleware)
                    router = APIRouter()
                    register(router)
                    app.include_router(router, prefix="/api")
                    url, h = "/api/detections/1/regions", {"X-API-Key": "t-eng"}
                    with TestClient(app) as c:
                        got = c.get(url, headers=h).json()
                        self.assertEqual(got["detection_id"], 1)   # 关联 Detection
                        self.assertTrue(all(r["source"] == "engine_box" and r["train_status"] == "excluded_candidate"
                                            for r in got["regions"]))
                        body = {"op": "confirm", "region_id": "r0", "operator": "工程师甲",
                                "event_id": "ev1", "expected_revision": 0}
                        r = c.post(url, headers=h, json=body).json()
                        r0 = _by_id(r["review"])["r0"]
                        self.assertEqual((r0["source"], r0["reviewer"], r0["review_revision"],
                                          r0["train_status"]), ("human_box", "工程师甲", 1, "eligible"))
                        # 幂等重放也返回同一契约视图
                        again = c.post(url, headers=h, json=body).json()
                        self.assertTrue(again["idempotent"])
                        self.assertEqual(_by_id(again["review"])["r0"]["source"], "human_box")
                        add = {"op": "add", "bbox": [20, 20, 40, 40], "operator": "工程师乙",
                               "event_id": "ev2", "expected_revision": 1}
                        self.assertEqual(c.post(url, headers=h, json=add).status_code, 200)
                        view = _by_id(c.get(url, headers=h).json())
                        self.assertEqual((view["h0"]["source"], view["h0"]["reviewer"],
                                          view["h0"]["review_revision"], view["h0"]["confirmed"]),
                                         ("human_box", "工程师乙", 2, True))
                        self.assertEqual(view["r1"]["train_status"], "excluded_candidate")
                    with database.session_scope() as s:
                        stored = s.get(Detection, 1).n_tiles["region_review"]
                        # 存储不变：派生字段只在视图中出现；训练选择仍只取已确认框
                        self.assertEqual(_by_id(stored)["r0"]["source"], "engine_box")
                        self.assertNotIn("train_status", _by_id(stored)["r0"])
                        self.assertEqual(L.trainable_region_boxes(stored), [[0, 0, 8, 8], [20, 20, 40, 40]])
            finally:
                eng.dispose()


if __name__ == "__main__":
    unittest.main(verbosity=2)
