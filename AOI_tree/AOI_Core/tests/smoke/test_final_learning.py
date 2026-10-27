"""final_learning：AC-038 引擎级守卫 / AC-072 迟到标签不入训 + 保留期预览审批 / AC-063 热图复算 API。

全部用 TEMP sqlite / TEMP 存储目录 / Mock 引擎模拟，不触碰生产 DB 与快照。
"""
import inspect
import json
import os
import tempfile
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.db import database
from backend.db.models import Base, Detection, Feedback

_DAY = 86400.0
_BUDGET = {"heatmap_overlay_days": 30, "raw_hot_days": 7, "raw_cold_months": 3,
           "label_late_tolerance_days": 7}


class _TempDb(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.eng = create_engine(f"sqlite:///{Path(self.tmp.name) / 'f.db'}",
                                 connect_args={"check_same_thread": False})
        Base.metadata.create_all(self.eng)
        self.SL = sessionmaker(bind=self.eng, expire_on_commit=False)
        self.patches = [patch.object(database, "_engine", self.eng),
                        patch.object(database, "_SessionLocal", self.SL)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.eng.dispose()
        self.tmp.cleanup()


def _app(register_fn=None, router=None):
    from backend.core.security import ApiKeyMiddleware
    app = FastAPI()
    app.add_middleware(ApiKeyMiddleware)
    if register_fn is not None:
        router = APIRouter()
        register_fn(router)
    app.include_router(router, prefix="/api")
    return app


def _sec():
    sec = Mock()
    sec.section.return_value = {"keys": {"t-op": "operator", "t-eng": "engineer",
                                         "t-adm": "admin"}}
    return [patch("backend.core.security.get_settings", return_value=sec),
            patch("backend.core.security.get_configured_api_key", return_value="")]


# ───────────────────────── AC-038 ─────────────────────────
class EngineGuardTest(unittest.TestCase):
    def test_guard_blocks_unauthorized_legacy(self):
        print("[AC-038 1/3] 生产单例守卫：未授权 submit_feedback 抛 LegacyOnlineLearningBlocked", flush=True)
        from backend.engine import DetectionEngine, LegacyOnlineLearningBlocked
        eng = DetectionEngine.__new__(DetectionEngine)
        self.assertFalse(eng.legacy_online_guard)          # 直接构造（测试/bench）默认不拦
        eng.legacy_online_guard = True
        eng._ensure_loaded = Mock(side_effect=RuntimeError("reached"))
        with self.assertRaises(LegacyOnlineLearningBlocked):
            eng.submit_feedback("cat", "x.png", verdict="wrong", label=1)
        eng._ensure_loaded.assert_not_called()             # 守卫在加载/学习之前
        self.assertTrue(issubclass(LegacyOnlineLearningBlocked, PermissionError))
        with self.assertRaisesRegex(RuntimeError, "reached"):   # 显式授权放行
            eng.submit_feedback("cat", "x.png", verdict="wrong", label=1,
                                allow_legacy_online=True)

    def test_get_engine_sets_guard(self):
        print("[AC-038 2/3] get_engine() 单例置 legacy_online_guard=True", flush=True)
        import backend.engine as be
        fake = SimpleNamespace(engine_base_cfg=Path("cfg.yaml"), engine_storage=Path("st"),
                               get=lambda *a, **k: 8)
        with patch.object(be, "_engine", None), \
                patch("backend.core.config.get_settings", return_value=fake), \
                patch.object(be.DetectionEngine, "__init__", return_value=None):
            eng = be.get_engine()
            self.assertTrue(eng.legacy_online_guard)

    def test_service_legacy_branch_authorizes(self):
        print("[AC-038 3/3] service legacy 分支透传入口授权 allow_legacy_online", flush=True)
        from backend.self_learning import service
        src = inspect.getsource(service.submit_feedback_record).replace(" ", "").replace("\n", "")
        call = src[src.index("get_engine().submit_feedback(") + len("get_engine()."):]
        self.assertIn("allow_legacy_online=allow_legacy_online)", call[:call.index(")") + 1])
        engine = Mock()                                  # 未授权在入口即被拒，不触达引擎
        with patch.object(service, "get_engine", return_value=engine):
            with self.assertRaises(service.FeedbackConflict):
                service.submit_feedback_record(1, "false_positive", 0, candidate_only=False)
        engine.submit_feedback.assert_not_called()


# ───────────────────────── AC-072 迟到标签 ─────────────────────────
class LateLabelTest(_TempDb):
    def _seed(self, lag_days, fid=1, category="test"):
        t0 = datetime(2026, 9, 1, 8, 0, 0)
        with self.SL() as s:
            s.add(Detection(id=fid, image_path=f"x{fid}.png", category=category, final_score=0.9,
                            is_anomaly=True, latency_ms=1.0, created_at=t0,
                            n_tiles={}))
            s.add(Feedback(id=fid, detection_id=fid, feedback_type="false_negative",
                           operator_label=1, learning_trace={"candidate_only": True},
                           created_at=t0 + timedelta(days=lag_days)))
            s.commit()

    def _build(self, fid):
        from backend.self_learning import service
        engine = Mock()
        engine.learn_candidate.return_value = {"event_id": f"feedback:{fid}", "version_num": 2,
                                               "parent_version": 1}
        cfg = Mock()
        cfg.business_budget.return_value = dict(_BUDGET)
        with patch.object(service, "get_engine", return_value=engine), \
                patch("backend.core.config.get_settings", return_value=cfg):
            out = service.build_feedback_candidate(fid, 1)
        return out, engine

    def test_label_lateness_pure(self):
        print("[AC-072 1/4] label_lateness 纯函数：边界/缺失时间", flush=True)
        from backend.self_learning.service import label_lateness
        t = datetime(2026, 9, 1)
        self.assertFalse(label_lateness(t + timedelta(days=7), t, 7)["late"])
        r = label_lateness(t + timedelta(days=7, hours=1), t, 7)
        self.assertTrue(r["late"])
        self.assertAlmostEqual(r["lag_days"], 7.042, places=3)
        self.assertEqual(label_lateness(None, t, 7),
                         {"late": False, "lag_days": None, "tolerance_days": 7})

    def test_late_excluded_not_trained_and_counted(self):
        print("[AC-072 2/4] 超容忍标签 → late_excluded，不调 learn_candidate，计入统计", flush=True)
        from backend.self_learning.service import late_label_stats
        self._seed(10, fid=1, category="a")
        self._seed(1, fid=2, category="b")
        out, engine = self._build(1)
        self.assertEqual(out["status"], "late_excluded")
        self.assertFalse(out["trained"])
        self.assertTrue(out["label_lateness"]["late"])
        engine.learn_candidate.assert_not_called()
        with database.session_scope() as s:
            fb = s.get(Feedback, 1)
            self.assertEqual(fb.learning_trace["learning_status"], "late_excluded")
            self.assertTrue(fb.learning_trace["candidate_only"])
            self.assertFalse(fb.consumed)
        self.assertEqual(late_label_stats(), {"late_excluded": 1, "feedback_ids": [1]})
        self.assertEqual(late_label_stats("a")["late_excluded"], 1)
        self.assertEqual(late_label_stats("b")["late_excluded"], 0)

    def test_on_time_label_trains(self):
        print("[AC-072 3/4] 容忍期内标签照常入候选训练", flush=True)
        self._seed(1, fid=2)
        out, engine = self._build(2)
        self.assertNotEqual(out.get("status"), "late_excluded")
        engine.learn_candidate.assert_called_once()

    def test_late_label_api(self):
        print("[AC-072 4/4] GET /system/retention/late_labels", flush=True)
        from backend.api import routes_system
        self._seed(30, fid=3)
        self._build(3)
        ps = _sec()
        for p in ps:
            p.start()
        try:
            with TestClient(_app(router=routes_system.router)) as c:
                r = c.get("/api/system/retention/late_labels", headers={"X-API-Key": "t-op"})
                self.assertEqual(r.status_code, 200, r.text)
                self.assertEqual(r.json()["feedback_ids"], [3])
        finally:
            for p in reversed(ps):
                p.stop()


# ───────────────────────── AC-072 保留期 ─────────────────────────
class RetentionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "storage"
        self.now = time.time()
        self.files = {}
        for sub, name, age in [("heatmaps", "old_overlay.jpg", 40), ("heatmaps", "new_raw.npy", 5),
                               ("raw_images", "old.png", 200), ("raw_images", "warm.png", 60),
                               ("uploads", "u_old.png", 120), ("snapshots", "s.pt", 999),
                               ("feedback_events", "wal.jsonl", 999)]:
            p = self.root / sub / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"x" * 10)
            t = self.now - age * _DAY
            os.utime(p, (t, t))
            self.files[name] = p

    def tearDown(self):
        self.tmp.cleanup()

    def test_preview_and_dry_run_delete_nothing(self):
        print("[AC-072 R1/4] 预览/dry-run 只列计划不删除；快照与 WAL 永不入计划", flush=True)
        from backend.core.retention import execute_retention, preview_retention
        plan = preview_retention(self.root, _BUDGET, now=self.now)
        names = sorted(Path(i["path"]).name for i in plan["items"])
        self.assertEqual(names, ["old.png", "old_overlay.jpg", "u_old.png"])
        self.assertTrue(plan["dry_run"])
        self.assertEqual(plan["total_bytes"], 30)
        self.assertEqual(plan["per_dir"]["raw_images"]["keep_days"], 97.0)
        dry = execute_retention(self.root, _BUDGET, now=self.now)       # 默认 dry_run
        self.assertEqual(dry["plan_id"], plan["plan_id"])
        self.assertTrue(all(p.exists() for p in self.files.values()))
        self.assertFalse((self.root / "retention_audit.jsonl").exists())

    def test_execute_requires_approval_and_matching_plan(self):
        print("[AC-072 R2/4] 无审批 / confirm 缺失 / plan_id 不符 → 拒绝且不删除", flush=True)
        from backend.core.retention import (RetentionApprovalError, execute_retention,
                                            preview_retention)
        pid = preview_retention(self.root, _BUDGET, now=self.now)["plan_id"]
        bad = [None, {"plan_id": pid, "approver": "甲"},
               {"plan_id": pid, "approver": " ", "confirm": True},
               {"plan_id": "deadbeef", "approver": "甲", "confirm": True}]
        for ap in bad:
            with self.assertRaises(RetentionApprovalError):
                execute_retention(self.root, _BUDGET, approval=ap, dry_run=False, now=self.now)
        self.assertTrue(all(p.exists() for p in self.files.values()))

    def test_execute_with_approval_deletes_and_audits(self):
        print("[AC-072 R3/4] 审批后执行：仅删过期项 + 审计；计划变化后旧 plan_id 失效", flush=True)
        from backend.core.retention import (RetentionApprovalError, execute_retention,
                                            preview_retention)
        pid = preview_retention(self.root, _BUDGET, now=self.now)["plan_id"]
        ap = {"plan_id": pid, "approver": "工程师甲", "confirm": True, "reason": "月度清理"}
        out = execute_retention(self.root, _BUDGET, approval=ap, dry_run=False, now=self.now)
        self.assertEqual((out["deleted"], out["skipped"], out["freed_bytes"]), (3, 0, 30))
        for n in ("old.png", "old_overlay.jpg", "u_old.png"):
            self.assertFalse(self.files[n].exists())
        for n in ("new_raw.npy", "warm.png", "s.pt", "wal.jsonl"):
            self.assertTrue(self.files[n].exists())
        audit = [json.loads(l) for l in
                 (self.root / "retention_audit.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual((audit[-1]["approver"], audit[-1]["deleted"]), ("工程师甲", 3))
        # 新增过期文件 → 计划变化，旧审批不能复用
        p = self.root / "heatmaps" / "late.jpg"
        p.write_bytes(b"y")
        os.utime(p, (self.now - 50 * _DAY,) * 2)
        with self.assertRaises(RetentionApprovalError):
            execute_retention(self.root, _BUDGET, approval=ap, dry_run=False, now=self.now)
        self.assertTrue(p.exists())

    def test_api_roles_and_default_dry_run(self):
        print("[AC-072 R4/4] API：preview 需 engineer、execute 需 admin，默认 dry-run，缺审批 403", flush=True)
        from backend.api import routes_system
        fake = SimpleNamespace(storage_dir=self.root, business_budget=lambda: dict(_BUDGET))
        ps = _sec() + [patch.object(routes_system, "get_settings", return_value=fake),
                       patch.object(routes_system, "log_action")]
        for p in ps:
            p.start()
        try:
            with TestClient(_app(router=routes_system.router)) as c:
                url_p, url_e = "/api/system/retention/preview", "/api/system/retention/execute"
                self.assertEqual(c.get(url_p, headers={"X-API-Key": "t-op"}).status_code, 403)
                plan = c.get(url_p, headers={"X-API-Key": "t-eng"}).json()
                self.assertEqual(plan["total_files"], 3)
                self.assertEqual(c.post(url_e, headers={"X-API-Key": "t-eng"}, json={}).status_code, 403)
                adm = {"X-API-Key": "t-adm"}
                r = c.post(url_e, headers=adm, json={})
                self.assertTrue(r.json()["dry_run"])
                self.assertTrue(all(p.exists() for p in self.files.values()))
                r = c.post(url_e, headers=adm, json={"dry_run": False})
                self.assertEqual(r.status_code, 403)
                r = c.post(url_e, headers=adm, json={"dry_run": False, "approval": {
                    "plan_id": plan["plan_id"], "approver": "admin甲", "confirm": True}})
                self.assertEqual(r.status_code, 200, r.text)
                self.assertEqual(r.json()["deleted"], 3)
                routes_system.log_action.assert_called_once()
        finally:
            for p in reversed(ps):
                p.stop()


# ───────────────────────── AC-063 热图复算 API ─────────────────────────
class HeatmapRecomputeApiTest(_TempDb):
    def _make_det(self, hm_dir):
        from tests.smoke.test_raw_heatmap_retention import RawHeatmapRetentionTest
        det = RawHeatmapRetentionTest._run(None, str(hm_dir), {"overlay_max_dim": 1280})
        with self.SL() as s:
            s.add(Detection(id=1, image_path="raw.png", category="test", final_score=0.9,
                            is_anomaly=True, latency_ms=1.0, n_tiles=det["n_tiles"]))
            s.add(Detection(id=2, image_path="n.png", category="test", final_score=0.1,
                            is_anomaly=False, latency_ms=1.0, n_tiles={}))
            s.commit()
        return det

    def test_recompute_paths(self):
        print("[AC-063 1/2] 复算：默认=在线、改阈值、404/409/403/410、只读不落库", flush=True)
        from backend.api.region_review import HeatmapRecompute, heatmap_recompute
        hm_dir = Path(self.tmp.name) / "heatmaps"
        hm_dir.mkdir()
        det = self._make_det(hm_dir)
        online = [h["bbox"] for h in det["n_tiles"]["localization"]["heatmap_boxes"]]
        base = heatmap_recompute(1, HeatmapRecompute(), base_dir=hm_dir)
        self.assertEqual(base["boxes_image"], online)
        self.assertFalse(base["persisted"])
        high = heatmap_recompute(1, HeatmapRecompute(rel_thresh=0.8), base_dir=hm_dir)
        self.assertEqual(high["params"]["rel_thresh"], 0.8)
        self.assertLess(len(high["boxes_image"]), len(base["boxes_image"]))
        with database.session_scope() as s:                 # 只读：库内记录不变
            self.assertEqual(s.get(Detection, 1).n_tiles, det["n_tiles"])
        for did, code in ((99, 404), (2, 409)):
            with self.assertRaises(HTTPException) as e:
                heatmap_recompute(did, HeatmapRecompute(), base_dir=hm_dir)
            self.assertEqual(e.exception.status_code, code)
        other = Path(self.tmp.name) / "other"
        other.mkdir()
        with self.assertRaises(HTTPException) as e:          # 热图不在 base_dir 内 → 403
            heatmap_recompute(1, HeatmapRecompute(), base_dir=other)
        self.assertEqual(e.exception.status_code, 403)
        Path(det["raw_heatmap_path"]).unlink()
        with self.assertRaises(HTTPException) as e:          # 已被保留期清理 → 410
            heatmap_recompute(1, HeatmapRecompute(), base_dir=hm_dir)
        self.assertEqual(e.exception.status_code, 410)

    def test_route_validation(self):
        print("[AC-063 2/2] 路由：参数校验 422、无 raw 409、默认 base_dir 取 storage(heatmaps)", flush=True)
        from backend.api.region_review import register
        hm_dir = Path(self.tmp.name) / "heatmaps"
        hm_dir.mkdir()
        self._make_det(hm_dir)
        cfg = Mock()
        cfg.storage.return_value = hm_dir
        ps = _sec() + [patch("backend.core.config.get_settings", return_value=cfg)]
        for p in ps:
            p.start()
        try:
            with TestClient(_app(register_fn=register)) as c:
                h = {"X-API-Key": "t-eng"}
                url = "/api/detections/{}/heatmap/recompute"
                self.assertEqual(c.post(url.format(1), headers=h,
                                        json={"rel_thresh": 1.5}).status_code, 422)
                self.assertEqual(c.post(url.format(2), headers=h, json={}).status_code, 409)
                r = c.post(url.format(1), headers=h, json={"rel_thresh": 0.3, "min_area": 4})
                self.assertEqual(r.status_code, 200, r.text)
                self.assertEqual((r.json()["params"]["rel_thresh"], r.json()["params"]["min_area"]),
                                 (0.3, 4))
                cfg.storage.assert_called_with("heatmaps")
        finally:
            for p in reversed(ps):
                p.stop()


if __name__ == "__main__":
    unittest.main(verbosity=2)
