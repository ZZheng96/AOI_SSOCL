"""异步模板快照标准图回归测试。"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.task_queue import DetectTaskQueue
from app.db.models import Base, Task
from app.template.model import InspectionTemplate, StandardImageRef
from app.template.store import TemplateStore


class AsyncTemplateSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.store = TemplateStore(self.root / "templates")
        self.image = self.root / "standard.png"
        self.image.write_bytes(b"original")
        tpl = InspectionTemplate(id="board", standard_image=StandardImageRef(
            path=str(self.image), sha1=TemplateStore._file_sha1(self.image)))
        self.store.save(tpl, sync_legacy=False)
        self.engine = create_engine("sqlite:///:memory:")
        self.addCleanup(self.engine.dispose)
        Base.metadata.create_all(self.engine)
        self.sessions = sessionmaker(bind=self.engine)
        self.queue = DetectTaskQueue(workers=0)

    def session_scope(self):
        from contextlib import contextmanager

        @contextmanager
        def scope():
            with self.sessions() as session:
                try:
                    yield session
                    session.commit()
                except Exception:
                    session.rollback()
                    raise
        return scope()

    def submit(self, **extra):
        with patch("app.db.database.session_scope", self.session_scope), patch(
                "app.template.store.TEMPLATES_DIR", self.store.root):
            task_id = self.queue.submit("dual", {
                "task_type": "dual", "template_id": "board", "image_path": "test.png",
                **extra,
            })
        with self.sessions() as session:
            return task_id, session.get(Task, task_id).payload

    def test_copy_survives_original_change_and_is_cleaned_on_cancel(self):
        task_id, payload = self.submit()
        snapshot = payload["template_snapshot"]
        frozen = Path(snapshot["standard_image"]["path"])
        self.image.write_bytes(b"changed")
        self.assertEqual(frozen.read_bytes(), b"original")
        TemplateStore.verify_task_snapshot(snapshot)
        with patch("app.db.database.session_scope", self.session_scope), patch(
                "app.template.store.TEMPLATES_DIR", self.store.root):
            self.assertTrue(self.queue.cancel(task_id))
        self.assertFalse(frozen.exists())

    def test_execution_uses_independent_task_images(self):
        from unittest.mock import Mock

        first_id, first = self.submit(workorder_id=11)
        second_id, second = self.submit(workorder_id=22)
        self.assertNotEqual(first["template_snapshot"]["standard_image"]["path"],
                            second["template_snapshot"]["standard_image"]["path"])
        self.image.write_bytes(b"changed")
        seen = []

        def detect(template, image_path, **kwargs):
            seen.append((Path(template.standard_image.path).read_bytes(), kwargs["workorder_id"]))
            return {"overall": "OK"}

        with patch("app.db.database.session_scope", self.session_scope), patch(
                "app.template.store.TEMPLATES_DIR", self.store.root), patch(
                "app.inspect.detect_service.get_detection_service",
                return_value=Mock(detect_dual=detect)):
            self.queue._execute({"task_id": first_id, **first})
            self.queue._execute({"task_id": second_id, **second})
        self.assertEqual(seen, [(b"original", 11), (b"original", 22)])
        self.store.cleanup_task_snapshot(first_id)
        self.store.cleanup_task_snapshot(second_id)

    def test_tampered_snapshot_fails_before_detection(self):
        task_id, payload = self.submit()
        Path(payload["template_snapshot"]["standard_image"]["path"]).write_bytes(b"bad")
        with patch("app.db.database.session_scope", self.session_scope), patch(
                "app.template.store.TEMPLATES_DIR", self.store.root):
            with self.assertRaisesRegex(RuntimeError, "快照缺失或哈希校验失败"):
                self.queue._execute({"task_id": task_id, **payload})
        self.store.cleanup_task_snapshot(task_id)

    def test_worker_cleans_snapshot_on_failure(self):
        task_id, payload = self.submit()
        frozen = Path(payload["template_snapshot"]["standard_image"]["path"])
        frozen.write_bytes(b"tampered")
        with patch("app.db.database.session_scope", self.session_scope), patch(
                "app.template.store.TEMPLATES_DIR", self.store.root):
            def fail(_payload):
                self.queue._stop = True
                raise RuntimeError("failed")

            with patch.object(self.queue, "_execute", side_effect=fail):
                self.queue._worker()
        self.assertFalse(frozen.exists())

    def test_stale_original_and_version_do_not_enqueue(self):
        self.image.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "哈希不一致"):
            self.submit()
        with self.sessions() as session:
            self.assertEqual(session.query(Task).count(), 0)
        self.assertEqual(self.queue._q.qsize(), 0)
        self.image.write_bytes(b"original")
        with self.assertRaisesRegex(ValueError, "版本已变化"):
            self.submit(template_version=99)


if __name__ == "__main__":
    unittest.main()
