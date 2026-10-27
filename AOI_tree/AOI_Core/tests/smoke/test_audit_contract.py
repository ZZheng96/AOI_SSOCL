"""Lightweight alarm audit must preserve synchronous persistence and actor context."""
import asyncio
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.core import audit, security
from backend.core.alarm import AlarmMonitor
from backend.db import database
from backend.db.models import Base, OperationLog
from backend.pipeline import service
from starlette.requests import Request


class AuditContractTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="aoi_audit_contract_")
        self.engine = create_engine("sqlite:///" + (Path(self.tmp.name) / "audit.db").as_posix(),
                                    connect_args={"check_same_thread": False})
        Base.metadata.create_all(self.engine)
        self.patches = [patch.object(database, "_engine", self.engine),
                        patch.object(database, "_SessionLocal", sessionmaker(
                            bind=self.engine, expire_on_commit=False))]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.engine.dispose()
        self.tmp.cleanup()

    def rows(self):
        with database.session_scope() as s:
            return [(r.user, r.action, r.detail, r.extra) for r in s.query(OperationLog).all()]

    def test_exact_fields_and_shared_actor(self):
        self.assertIs(audit.current_actor, security.current_actor)
        for actor in ("operator", "engineer", "admin"):
            token = security.current_actor.set(actor)
            try:
                for writer in (database.log_action, audit.log_action):
                    writer("alarm", "category=test window_hits=5 window=20 k=5",
                           extra={"category": "test", "hits": 5})
                    writer("explicit", "", user="", extra=None)
            finally:
                security.current_actor.reset(token)
        rows = self.rows()
        for i in range(0, len(rows), 4):
            self.assertEqual(rows[i:i + 2], rows[i + 2:i + 4])
        self.assertEqual(audit.current_actor.get(), "operator")

    def test_alarm_commit_before_notify_and_cooldown(self):
        for writer in (database.log_action, audit.log_action):
            mon = AlarmMonitor(window=3, k=2, cooldown_s=60)
            before = len(self.rows())
            def notify(event, payload):
                self.assertEqual(len(self.rows()), before + 1)
                self.assertEqual(self.rows()[-1], ("operator", "alarm",
                    "category=test window_hits=2 window=3 k=2", None))
                self.assertEqual((event, payload), ("alarm", {
                    "category": "test", "window_hits": 2, "window": 3, "k": 2}))
            with patch.object(service, "log_action", writer), \
                 patch.object(service, "get_alarm_monitor", return_value=mon), \
                 patch.object(service, "notify_event", side_effect=notify) as notification:
                self.assertEqual(service.DetectionService._record_alarm("test", True),
                                 {"active": False, "window_hits": 1})
                self.assertEqual(service.DetectionService._record_alarm("test", True),
                                 {"active": True, "window_hits": 2})
                service.DetectionService._record_alarm("test", True)
                self.assertEqual(notification.call_count, 1)

    def test_commit_failure_rolls_back_and_does_not_notify(self):
        from sqlalchemy.orm import Session
        for writer in (database.log_action, audit.log_action):
            mon = AlarmMonitor(window=1, k=1)
            with patch.object(service, "log_action", writer), \
                 patch.object(service, "get_alarm_monitor", return_value=mon), \
                 patch.object(Session, "commit", side_effect=RuntimeError("commit failed")), \
                 patch.object(service, "notify_event") as notify, \
                 self.assertLogs(service.logger, level="ERROR"):
                self.assertEqual(service.DetectionService._record_alarm("fail", True),
                                 {"active": True, "window_hits": 1})
                notify.assert_not_called()
            self.assertEqual(self.rows(), [])

    def test_middleware_permissions_context_and_thread_default(self):
        settings = SimpleNamespace(section=lambda key: {"keys": {
            "test-op": "operator", "test-eng": "engineer", "test-adm": "admin"}})
        middleware = security.ApiKeyMiddleware(lambda scope, receive, send: None)
        async def scenario():
            for key, role in (("test-op", "operator"), ("test-eng", "engineer"),
                              ("test-adm", "admin"), ("bad", None)):
                request = Request({"type": "http", "method": "POST", "path": "/api/test",
                    "headers": [(b"x-api-key", key.encode())]})
                async def endpoint(req):
                    self.assertEqual(audit.current_actor.get(), role)
                    await asyncio.to_thread(audit.log_action, "request")
                    for minimum in security.ROLE_RANK:
                        if security.ROLE_RANK[role] < security.ROLE_RANK[minimum]:
                            with self.assertRaises(security.HTTPException) as caught:
                                security.require_role(minimum)(req)
                            self.assertEqual(caught.exception.status_code, 403)
                        else:
                            security.require_role(minimum)(req)
                    return "ok"
                result = await middleware.dispatch(request, endpoint)
                if role is None:
                    self.assertEqual(result.status_code, 401)
                else:
                    self.assertEqual(result, "ok")
                self.assertEqual(audit.current_actor.get(), "operator")
        with patch.object(security, "get_settings", return_value=settings), \
             patch.object(security, "get_configured_api_key", return_value=""):
            asyncio.run(scenario())
        token = audit.current_actor.set("admin")
        try:
            thread = threading.Thread(target=audit.log_action, args=("worker",))
            thread.start()
            thread.join()
        finally:
            audit.current_actor.reset(token)
        self.assertEqual([r[0] for r in self.rows()],
                         ["operator", "engineer", "admin", "operator"])

    def test_cold_audit_does_not_import_web_stack(self):
        code = '''
import sys
from unittest.mock import patch
from backend.core import audit
from backend.db import database
with patch.object(database, "log_action") as persist:
    audit.log_action("alarm", "detail")
    persist.assert_called_once_with("alarm", "detail", user="operator", extra=None)
assert not any(n == "backend.core.security" or n.split(".")[0] in ("fastapi", "starlette") for n in sys.modules)
'''
        subprocess.run([sys.executable, "-c", code], check=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
