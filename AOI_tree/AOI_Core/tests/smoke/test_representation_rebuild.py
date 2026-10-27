"""AC-015 表示变化：stale 拒学习/巩固/激活、allow_stale 放行、表示重建产出未激活新版本
（TEMP SQLite + TEMP 目录 + 假 pipe，不碰生产）。"""
import contextlib
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.db import database
from backend.db.models import Base, Image, Task

STALE = {"status": "stale", "drift": ["registry_sha256"], "action": "rebuild"}
CURRENT = {"status": "current", "drift": [], "action": None}


def _fake_engine(tmp: Path, rep: dict, versions=(1, 2), current=2):
    from backend.engine import DetectionEngine
    eng = DetectionEngine.__new__(DetectionEngine)
    state = {"current": current, "learned": 0, "cached": None}
    pipe = SimpleNamespace(feature_representation=dict(rep), slots={},
                           feedback=lambda *a, **k: state.__setitem__(
                               "learned", state["learned"] + 1) or {})
    eng._ensure_loaded = lambda c: None
    eng._pipe_lock = lambda c: contextlib.nullcontext()
    eng._write_lock = threading.Lock()
    eng._pipe_entry = lambda c: (pipe, state["current"])
    eng._cat_dir = lambda c: str(tmp / c)
    eng._v_dir = lambda c, v: str(tmp / c / f"v{v}")
    eng.current_version = lambda c: state["current"]
    eng.list_versions = lambda c: list(versions)
    eng._cache_put = lambda c, p, v: state.__setitem__("cached", v)
    eng.feedback_timeout_ms = 2000
    eng.device = "cpu"
    eng.storage_dir = str(tmp)  # 发布账本/审计链（AC-074）写 TEMP
    eng._feedback_identity = lambda *a: (None, None)
    eng._append_journal = lambda *a: None
    return eng, pipe, state


class EngineGuardTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_learning_and_consolidate_blocked(self):
        print("[1/5] stale 拒在线学习/巩固（激活+候选路径），current 放行", flush=True)
        eng, pipe, st = _fake_engine(self.root, STALE)
        with self.assertRaisesRegex(ValueError, "禁止在线学习"):
            eng.submit_feedback("c", "a.png", "ok", label=0)
        self.assertEqual(st["learned"], 0)
        (self.root / "c").mkdir()
        with patch("algo.persist.save_snapshot") as ss:
            with self.assertRaisesRegex(ValueError, "禁止巩固"):
                eng.consolidate("c", activate=True)
            ss.assert_not_called()
        with patch("algo.persist.write_snapshot_state") as ws:
            with self.assertRaisesRegex(ValueError, "禁止巩固"):
                eng.consolidate("c", activate=False)
            ws.assert_not_called()
        self.assertEqual([p.name for p in (self.root / "c").iterdir()], [])  # 暂存已清
        pipe.feature_representation = dict(CURRENT)
        eng.submit_feedback("c", "a.png", "ok", label=0)
        self.assertEqual(st["learned"], 1)

    def test_activate_rollback(self):
        print("[2/5] activate/rollback 遇 stale 默认拒绝，allow_stale 放行", flush=True)
        eng, _, st = _fake_engine(self.root, CURRENT)
        loaded = SimpleNamespace(feature_representation=dict(STALE))
        with patch("algo.persist.load_snapshot", return_value=loaded), \
                patch("algo.persist.list_versions", return_value=[1, 2]), \
                patch("algo.persist.set_current_version") as scv:
            with self.assertRaisesRegex(ValueError, "特征表示已过期.*拒绝激活"):
                eng.activate("c", 1)
            with self.assertRaisesRegex(ValueError, "拒绝激活"):
                eng.rollback("c")
            scv.assert_not_called()
            self.assertIsNone(st["cached"])
            # AC-073：allow_stale 无恢复授权一律拒绝
            with self.assertRaisesRegex(ValueError, "恢复授权"):
                eng.rollback("c", allow_stale=True)
            scv.assert_not_called()
            import time as _t
            auth = {"actor": "admin", "reason": "表示重建失败回退", "ticket": "INC-1",
                    "category": "c", "version": 1, "expires_at": _t.time() + 60}
            self.assertEqual(eng.rollback("c", allow_stale=True, recovery_authorization=auth), 1)
            self.assertEqual(st["cached"], 1)
            loaded.feature_representation = dict(CURRENT)
            eng.activate("c", 2)
            self.assertEqual(st["cached"], 2)


class RebuildRouteTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.db = create_engine(f"sqlite:///{self.root / 'rb.db'}",
                                connect_args={"check_same_thread": False})
        Base.metadata.create_all(self.db)
        factory = sessionmaker(bind=self.db, expire_on_commit=False)
        from backend.api import routes_models
        self.rm = routes_models
        self.eng, _, self.st = _fake_engine(self.root, STALE, versions=(1, 2, 3), current=2)
        self.submitted = []

        def _submit(ttype, payload, fn):
            with database.session_scope() as s:
                t = Task(task_type=ttype, status="pending", payload=payload)
                s.add(t)
                s.flush()
                tid = t.id
            self.submitted.append((ttype, payload))
            return tid

        self._patches = [
            patch.object(database, "_engine", self.db),
            patch.object(database, "_SessionLocal", factory),
            patch.object(routes_models.task_manager, "submit", side_effect=_submit),
            patch("backend.engine.get_engine", return_value=self.eng),
        ]
        for p in self._patches:
            p.start()
        app = FastAPI()

        @app.middleware("http")
        async def _role(request: Request, call_next):
            request.state.role = request.headers.get("X-Role", "admin")
            return await call_next(request)

        app.include_router(routes_models.router, prefix="/api")
        self.client = TestClient(app)

    def tearDown(self):
        for p in reversed(self._patches):
            p.stop()
        self.db.dispose()
        self.tmp.cleanup()

    def _meta(self, v, **kw):
        d = self.root / "c" / f"v{v}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "meta.json").write_text(json.dumps(kw), encoding="utf-8")

    def test_rebuild_route(self):
        print("[3/5] rebuild 路由：403 / 409 无版本 / 继承 scenario,profile / pending", flush=True)
        c = self.client
        self.assertEqual(c.post("/api/models/feature-set/c/rebuild", json={},
                                headers={"X-Role": "operator"}).status_code, 403)
        self.st["current"] = None
        self.assertEqual(c.post("/api/models/feature-set/c/rebuild", json={}).status_code, 409)
        self.st["current"] = 2
        self._meta(2, trigger="fit", note="prep (scenario=L3, profile=accurate)")
        d = c.post("/api/models/feature-set/c/rebuild", json={}).json()
        self.assertEqual((d["parent_version"], d["activated"]), (2, False))
        ttype, payload = self.submitted[-1]
        self.assertEqual(ttype, "model_prepare")
        self.assertTrue(payload["rebuild_representation"])
        self.assertTrue(payload["force"])
        self.assertEqual((payload["scenario"], payload["profile"]), ("L3", "accurate"))
        self.assertIsNone(self.rm._pending_rebuild("c"))
        self._meta(3, trigger="representation_rebuild")
        self.assertEqual(self.rm._pending_rebuild("c"), 3)
        self.st["current"] = 3
        self.assertIsNone(self.rm._pending_rebuild("c"))

    def test_prepare_engine_rebuild(self):
        print("[4/5] _prepare_engine 重建：trigger/parent/不激活/跳过自动档案", flush=True)
        from backend.api.schemas import ModelPrepareRequest
        with database.session_scope() as s:
            for i in range(3):
                s.add(Image(path=f"n{i}.png", category="c", split="train", label="normal"))
        calls = {}
        self.eng.prepare = lambda cat, bundle, **kw: calls.update(kw) or 3
        req = ModelPrepareRequest(category="c", rebuild_representation=True)
        with patch.object(self.rm, "register_engine_snapshots"), \
                patch.object(self.rm, "_auto_feature_report") as afr:
            out = self.rm._prepare_engine(req, lambda *a: None)
            afr.assert_not_called()
        self.assertEqual(calls["trigger"], "representation_rebuild")
        self.assertEqual(calls["parent_version"], 2)
        self.assertFalse(calls["activate"])
        self.assertTrue(calls["force"])
        self.assertFalse(out["activated"])
        self.assertTrue(out["representation_rebuild"])
        self.assertEqual(out["auto_report"]["status"], "skipped_rebuild")

    def test_activate_api_409(self):
        print("[5/5] /models/{id}/activate 遇 stale → 409 带重建指引", flush=True)
        sl = self.rm.sl_service
        err = ValueError("v1 特征表示已过期（registry_sha256），拒绝激活")
        with patch.object(sl, "resolve_model_version", return_value=("c", 1)), \
                patch.object(self.rm, "_production_sample_count", return_value=10), \
                patch.object(sl, "activate_model", side_effect=err) as am:
            r = self.client.post("/api/models/7/activate?gate=false")
            self.assertEqual(r.status_code, 409)
            self.assertEqual(r.json()["detail"]["representation"], "stale")
            self.assertIn("rebuild", r.json()["detail"]["action"])
            am.assert_called_with(7, allow_stale=False)


if __name__ == "__main__":
    unittest.main()
