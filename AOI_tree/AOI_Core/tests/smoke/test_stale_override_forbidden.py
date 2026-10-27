"""AC-073/074/076：生产 HTTP 入口禁止 allow_stale=true 绕过特征表示过期检查
（TEMP SQLite + TEMP 目录 + 假引擎，不碰生产 DB/快照）。"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.db import database
from backend.db.models import Base, Model

from tests.smoke.test_representation_rebuild import STALE, _fake_engine


class StaleOverrideForbiddenTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.db = create_engine(f"sqlite:///{self.root / 'so.db'}",
                                connect_args={"check_same_thread": False})
        Base.metadata.create_all(self.db)
        factory = sessionmaker(bind=self.db, expire_on_commit=False)
        from backend.api import routes_models
        self.rm = routes_models
        self.eng, _, self.st = _fake_engine(self.root, STALE, versions=(1, 2), current=2)
        self.eng.activate = MagicMock(name="engine.activate")
        self.eng.rollback = MagicMock(name="engine.rollback", return_value=1)
        self.log = MagicMock(name="log_action")
        self._patches = [
            patch.object(database, "_engine", self.db),
            patch.object(database, "_SessionLocal", factory),
            patch("backend.engine.get_engine", return_value=self.eng),
            patch.object(routes_models, "log_action", self.log),
            patch.object(routes_models, "register_engine_snapshots", MagicMock()),
        ]
        for p in self._patches:
            p.start()
        with database.session_scope() as s:
            s.add_all([
                Model(id=1, name="c-v1", category="c", version="v1",
                      path=str(self.root / "c" / "v1"), is_active=False),
                Model(id=2, name="c-v2", category="c", version="v2",
                      path=str(self.root / "c" / "v2"), is_active=True),
            ])
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

    def _active_flags(self):
        with database.session_scope() as s:
            return {m.id: m.is_active for m in s.query(Model).all()}

    def _assert_forbidden(self, r):
        self.assertEqual(r.status_code, 409)
        self.assertEqual(r.json()["detail"]["reason_code"], "STALE_OVERRIDE_FORBIDDEN")

    def test_activate_allow_stale_forbidden(self):
        print("[1/4] activate allow_stale=true → 409，service/engine 未调用、指针不变", flush=True)
        sl = self.rm.sl_service
        before = self._active_flags()
        with patch.object(sl, "activate_model") as am, \
                patch.object(sl, "resolve_model_version") as rv:
            for q in ("allow_stale=true", "allow_stale=true&gate=false&force=true"):
                self._assert_forbidden(self.client.post(f"/api/models/1/activate?{q}"))
            am.assert_not_called()
            rv.assert_not_called()
        self.eng.activate.assert_not_called()
        self.assertEqual(self.st["current"], 2)
        self.assertEqual(self._active_flags(), before)
        self.log.assert_not_called()

    def test_rollback_allow_stale_forbidden(self):
        print("[2/4] rollback allow_stale=true → 409，engine.rollback 未调用、指针不变", flush=True)
        before = self._active_flags()
        r = self.client.post("/api/models/rollback",
                             json={"category": "c", "allow_stale": True})
        self._assert_forbidden(r)
        self.eng.rollback.assert_not_called()
        self.rm.register_engine_snapshots.assert_not_called()
        self.assertEqual(self.st["current"], 2)
        self.assertEqual(self._active_flags(), before)
        self.log.assert_not_called()

    def test_activate_default_path_unchanged(self):
        print("[3/4] activate 缺省/false 保持原路径，service 收到 allow_stale=False", flush=True)
        sl = self.rm.sl_service
        with patch.object(sl, "resolve_model_version", return_value=("c", 1)), \
                patch.object(self.rm, "_production_sample_count", return_value=10), \
                patch.object(self.rm, "_auto_feature_report", return_value=None), \
                patch.object(sl, "activate_model", return_value={"ok": True}) as am:
            for q in ("gate=false", "gate=false&allow_stale=false"):
                r = self.client.post(f"/api/models/1/activate?{q}")
                self.assertEqual(r.status_code, 200, r.text)
                am.assert_called_with(1, allow_stale=False)
            self.assertEqual(am.call_count, 2)

    def test_rollback_default_path_unchanged(self):
        print("[4/4] rollback 缺省保持原路径，engine 收到 allow_stale=False", flush=True)
        for body in ({"category": "c"}, {"category": "c", "allow_stale": False}):
            r = self.client.post("/api/models/rollback", json=body)
            self.assertEqual(r.status_code, 200, r.text)
            self.assertEqual(r.json()["active_version"], "v1")
            self.eng.rollback.assert_called_with("c", allow_stale=False)
        self.assertEqual(self.eng.rollback.call_count, 2)


if __name__ == "__main__":
    unittest.main()
