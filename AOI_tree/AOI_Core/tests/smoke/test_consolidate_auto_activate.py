"""UI-P0.4：consolidate(activate=None) 必须服从 self_learning.auto_activate。"""
import contextlib
import unittest
from unittest.mock import MagicMock, patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.db.models import Base, ConsolidationBatch
from backend.self_learning import service


class ConsolidateAutoActivateTest(unittest.TestCase):
    def setUp(self):
        db = create_engine("sqlite://")
        self.addCleanup(db.dispose)
        Base.metadata.create_all(db)
        self.sessions = sessionmaker(bind=db, expire_on_commit=False)

        @contextlib.contextmanager
        def scope():
            with self.sessions.begin() as s:
                yield s

        self.engine = MagicMock()
        self.engine.current_version.return_value = 1
        self.engine.consolidate.return_value = 2
        for target, value in (("session_scope", scope), ("get_engine", lambda: self.engine),
                              ("log_action", MagicMock()), ("notify_event", MagicMock())):
            p = patch.object(service, target, value)
            p.start()
            self.addCleanup(p.stop)

    def run_with(self, setting, activate=None):
        settings = MagicMock()
        settings.get.side_effect = lambda sec, key, default=None: (
            setting if (sec, key) == ("self_learning", "auto_activate") else default)
        with patch("backend.core.config.get_settings", return_value=settings):
            out = service.consolidate("test", activate=activate)
        kw = self.engine.consolidate.call_args.kwargs
        with self.sessions() as s:
            status = s.query(ConsolidationBatch).order_by(ConsolidationBatch.id.desc()).first().status
        return out["activated"], kw["activate"], status

    def test_none_follows_setting(self):
        self.assertEqual(self.run_with(False), (False, False, "candidate"))
        self.assertEqual(self.run_with(True), (True, True, "activated"))

    def test_explicit_overrides_setting(self):
        self.assertEqual(self.run_with(True, activate=False), (False, False, "candidate"))
        self.assertEqual(self.run_with(False, activate=True), (True, True, "activated"))


if __name__ == "__main__":
    unittest.main()
