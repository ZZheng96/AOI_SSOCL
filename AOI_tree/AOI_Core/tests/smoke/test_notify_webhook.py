"""notify webhook 最小 smoke：禁用/订阅、payload、异步失败隔离与 events=list。"""
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.core import notify
from backend.db import database
from backend.db.models import Base, OperationLog


class NotifyWebhookSmokeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="aoi_notify_webhook_")
        self.engine = create_engine(
            "sqlite:///" + (Path(self.tmp.name) / "notify.db").as_posix(),
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(self.engine)
        self.patches = [
            patch.object(database, "_engine", self.engine),
            patch.object(database, "_SessionLocal", sessionmaker(
                bind=self.engine, expire_on_commit=False)),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.engine.dispose()
        self.tmp.cleanup()

    def _settings(self, url="http://example.test/hook", events=None):
        return SimpleNamespace(get_notify=lambda: {
            "webhook_url": url,
            "events": events if events is not None else ["anomaly"],
            "timeout_s": 1,
        })

    def _rows(self):
        with database.session_scope() as session:
            return session.query(OperationLog).filter(
                OperationLog.action == "webhook_error").all()

    def test_empty_url_and_unsubscribed_event_do_not_submit(self):
        with patch.object(notify, "get_settings", side_effect=[
            self._settings(url=""),
            self._settings(events=["anomaly"]),
        ]), patch.object(notify._executor, "submit") as submit:
            notify.notify_event("anomaly", {"id": 1})
            notify.notify_event("feedback", {"id": 2})
        submit.assert_not_called()

    def test_subscribed_event_posts_payload_and_events_is_list(self):
        submitted = []

        def capture(fn, *args):
            submitted.append((fn, args))
            return SimpleNamespace()

        payload = {"category": "pcb", "detection_id": 7}
        with patch.object(notify, "get_settings",
                          return_value=self._settings(events=["anomaly", "gray"])), \
                patch.object(notify._executor, "submit", side_effect=capture):
            notify.notify_event("gray", payload)
            notify.notify_event("feedback", {"ignored": True})

        self.assertEqual(len(submitted), 1)
        fn, args = submitted[0]
        self.assertIs(fn, notify._post)
        self.assertEqual(args[0], "http://example.test/hook")
        self.assertEqual(args[1]["event"], "gray")
        self.assertEqual(args[1]["category"], "pcb")
        self.assertEqual(args[1]["detection_id"], 7)
        self.assertIsInstance(self._settings(events=["gray"]).get_notify()["events"], list)

    def test_http_500_is_async_failure_and_logged(self):
        class Response:
            status_code = 500

        with patch.object(notify, "get_settings",
                          return_value=self._settings()), \
                patch("requests.post", return_value=Response()) as post, \
                self.assertLogs(notify.logger, level="WARNING") as logs:
            start = time.perf_counter()
            notify.notify_event("anomaly", {"id": 3})
            self.assertLess(time.perf_counter() - start, 0.5)
            notify._executor.submit(lambda: None).result(timeout=2)

        post.assert_called_once_with(
            "http://example.test/hook", json=ANY, timeout=1)
        self.assertTrue(any("HTTP 500" in message for message in logs.output))
        self.assertEqual(len(self._rows()), 1)
        self.assertIn("event=anomaly", self._rows()[0].detail)


if __name__ == "__main__":
    unittest.main(verbosity=2)
