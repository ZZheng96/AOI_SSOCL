"""AC-016 当前特征集：只读快照 / 档案状态 / 候选提交 / 角色（TEMP SQLite + TEMP 注册表，不碰生产）。"""
import contextlib
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from algo.features import registry as freg
from backend.db import database
from backend.db.models import Base, EvalRun, Image, Task

SIG = {"registry_sha256": "x" * 64, "groups": ["glcm"], "scoring": "legacy"}


def _fake_trad(groups=("glcm", "edge")):
    return SimpleNamespace(groups=list(groups), scoring="legacy", group_spec="all",
                           group_weight={"glcm": 0.5}, group_scale={},
                           selection_report=None, signature=lambda: dict(SIG))


def _engine_with(trad, version=3):
    from backend.engine import DetectionEngine
    eng = DetectionEngine.__new__(DetectionEngine)
    pipe = SimpleNamespace(slots={"trad": trad} if trad else {},
                           feature_representation={"state": "current"},
                           production_feature_metadata={"feature_signature": "abc"})
    eng._ensure_loaded = lambda c: None
    eng._pipe_lock = lambda c: contextlib.nullcontext()
    eng._pipe_entry = lambda c: (pipe, version)
    eng._v_dir = lambda c, v: str(Path(tempfile.gettempdir()) / "aoi_fs_none" / f"v{v}")
    eng.current_version = lambda c: version
    eng.list_versions = lambda c: [version]
    return eng


class EngineFeatureSetTest(unittest.TestCase):
    def test_nontraditional_slots(self):
        eng = _engine_with(None)
        pipe, version = eng._pipe_entry("c")
        pipe.slots = {key: SimpleNamespace(backbone=SimpleNamespace(dim=384),
                                          version="1", needs_dino=True)
                      for key in ("sem", "disc", "shead")}
        fs = eng.feature_set("c")
        self.assertEqual(fs["status"], "not_applicable")
        self.assertNotIn("groups", fs)
        self.assertEqual([s["key"] for s in fs["production_slots"]],
                         ["sem", "disc", "shead"])
        self.assertEqual([s["dim"] for s in fs["production_slots"]], [384] * 3)
        self.assertEqual(eng._pipe_entry("c"), (pipe, version))

    def test_snapshot(self):
        print("[1/4] engine.feature_set 只读快照 / not_applicable / 未加载", flush=True)
        fs = _engine_with(_fake_trad()).feature_set("c")
        self.assertEqual(fs["status"], "available")
        self.assertEqual(len(fs["groups"]), len(freg.TRAD_GROUPS))
        self.assertEqual(fs["enabled_groups"], ["glcm", "edge"])
        dims = {g["key"]: g["dim"] for g in freg.TRAD_GROUPS}
        self.assertEqual(fs["enabled_dim"], dims["glcm"] + dims["edge"])
        self.assertEqual(fs["total_dim"], freg.TRAD_DIM)
        self.assertEqual(fs["signature"], SIG)
        self.assertEqual(fs["snapshot_signature"], "abc")
        self.assertEqual(_engine_with(None).feature_set("c")["status"], "not_applicable")
        eng = _engine_with(_fake_trad())
        eng._ensure_loaded = lambda c: (_ for _ in ()).throw(RuntimeError("no snap"))
        self.assertEqual(eng.feature_set("c"), {"category": "c", "enabled": False})


class FeatureSetRouteTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = create_engine(f"sqlite:///{Path(self.tmp.name) / 'fs.db'}",
                                connect_args={"check_same_thread": False})
        Base.metadata.create_all(self.db)
        factory = sessionmaker(bind=self.db, expire_on_commit=False)
        from backend.api import routes_learning, routes_models
        from algo.feature_revision import FeatureSetRevision
        self.rm = routes_models
        self.store_path = Path(self.tmp.name) / "fr" / "registry.json"
        self.submitted = []

        def _submit(ttype, payload, fn):
            with database.session_scope() as s:
                t = Task(task_type=ttype, status="pending", payload=payload)
                s.add(t)
                s.flush()
                tid = t.id
            self.submitted.append((ttype, payload))
            return tid

        self.eng = _engine_with(_fake_trad())
        self._patches = [
            patch.object(database, "_engine", self.db),
            patch.object(database, "_SessionLocal", factory),
            patch.object(routes_models, "_feature_revision_store",
                         lambda: FeatureSetRevision(self.store_path)),
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
        app.include_router(routes_learning.router, prefix="/api")
        self.client = TestClient(app)

    def tearDown(self):
        for p in reversed(self._patches):
            p.stop()
        self.db.dispose()
        self.tmp.cleanup()

    def _images(self, normal=1, anomaly=1):
        with database.session_scope() as s:
            for i in range(normal):
                s.add(Image(path=f"n{i}.png", category="c", split="test", label="normal"))
            for i in range(anomaly):
                s.add(Image(path=f"a{i}.png", category="c", split="test", label="anomaly"))

    def _eval(self, signature, version=3, status="ok"):
        with database.session_scope() as s:
            s.add(EvalRun(name="contrib", run_type="contribution", category="c", metrics={
                "trad_groups": {"status": status,
                                "meta": {"signature": signature, "model_version": version}}}))

    def test_get_roles_and_report_status(self):
        print("[2/4] GET 角色 can_edit + 档案状态 insufficient/not_generated/stale/current",
              flush=True)
        c = self.client
        d = c.get("/api/models/feature-set/c", headers={"X-Role": "operator"}).json()
        self.assertFalse(d["can_edit"])
        self.assertEqual(d["report"]["status"], "insufficient")
        self.assertTrue(c.get("/api/models/feature-set/c",
                              headers={"X-Role": "engineer"}).json()["can_edit"])
        self._images()
        self.assertEqual(c.get("/api/models/feature-set/c").json()["report"]["status"],
                         "not_generated")
        self._eval({"other": 1})
        self.assertEqual(c.get("/api/models/feature-set/c").json()["report"]["status"], "stale")
        self._eval(SIG, version=2)
        self.assertEqual(c.get("/api/models/feature-set/c").json()["report"]["status"], "stale")
        self._eval(SIG, version=3)
        r = c.get("/api/models/feature-set/c").json()["report"]
        self.assertEqual(r["status"], "current")
        self.assertEqual(r["profile_version"], 3)
        self._eval(SIG, status="insufficient")
        self.assertEqual(c.get("/api/models/feature-set/c").json()["report"]["status"],
                         "insufficient")

    def test_draft_and_report(self):
        print("[3/4] draft 只登记候选 / 400 / 403 / 手动档案 / 自动钩子", flush=True)
        c = self.client
        op = {"X-Role": "operator"}
        body = {"groups": ["glcm"], "scoring": "grouped", "note": "t"}
        self.assertEqual(c.post("/api/models/feature-set/c/draft", json=body,
                                headers=op).status_code, 403)
        self.assertEqual(c.post("/api/models/feature-set/c/report", headers=op).status_code, 403)
        self.assertEqual(c.post("/api/learning/contribution", json={"category": "c"},
                                headers=op).status_code, 403)
        self.assertEqual(c.post("/api/models/feature-set/c/draft",
                                json={"groups": ["nope"]}).status_code, 400)
        self.assertEqual(c.post("/api/models/feature-set/c/draft",
                                json={"groups": ["glcm"], "scoring": "max"}).status_code, 400)
        before = self.eng.feature_set("c")
        d = c.post("/api/models/feature-set/c/draft", json=body).json()
        self.assertEqual(d["state"], "draft")
        self.assertFalse(d["changes_production"])
        reg = json.loads(self.store_path.read_text(encoding="utf-8"))
        self.assertIn(d["revision_id"], json.dumps(reg))
        self.assertEqual(self.eng.feature_set("c"), before)   # 生产未变
        # 幂等：同 manifest 不重复登记
        self.assertEqual(c.post("/api/models/feature-set/c/draft", json=body).json()["revision_id"],
                         d["revision_id"])
        # 手动档案：数据不足 409 → 补数据后 queued → 再点 already_running
        self.assertEqual(c.post("/api/models/feature-set/c/report").status_code, 409)
        self._images()
        r1 = c.post("/api/models/feature-set/c/report").json()
        self.assertEqual(r1["status"], "queued")
        r2 = c.post("/api/models/feature-set/c/report").json()
        self.assertEqual((r2["status"], r2["task_id"]), ("already_running", r1["task_id"]))
        self.assertEqual(len(self.submitted), 1)
        self.assertEqual(c.get("/api/models/feature-set/c").json()["report"]["running_task_id"],
                         r1["task_id"])

    def test_hard_pool_and_admission_routes(self):
        print("[3c/4] AC-017 难例池 GET / 准入试算 POST（403/409/queued 不激活）", flush=True)
        c = self.client
        op = {"X-Role": "operator"}
        self.assertEqual(c.get("/api/models/feature-set/c/hard-pool", headers=op).status_code, 403)
        d = c.get("/api/models/feature-set/c/hard-pool").json()
        self.assertEqual((d["n_items"], d["counts"]["pos"]), (0, 0))
        self.assertEqual(len(d["sha256"]), 64)
        self.assertEqual(c.post("/api/models/feature-revisions/nope/admission").status_code, 409)
        rid = c.post("/api/models/feature-set/c/draft",
                     json={"groups": ["glcm"], "scoring": "legacy"}).json()["revision_id"]
        reg = json.loads(self.store_path.read_text(encoding="utf-8"))
        self.assertTrue(reg[rid]["manifest"]["admission_required"])
        self.assertEqual(c.post(f"/api/models/feature-revisions/{rid}/admission",
                                headers=op).status_code, 403)
        r = c.post(f"/api/models/feature-revisions/{rid}/admission", json={"limit": 9999}).json()
        self.assertEqual((r["status"], r["activated"]), ("queued", False))
        self.assertEqual(self.submitted[-1],
                         ("feature_admission", {"category": "c", "revision_id": rid, "limit": 200}))
        reg = json.loads(self.store_path.read_text(encoding="utf-8"))
        self.assertEqual(reg[rid]["state"], "draft")

    def test_auto_hook(self):
        print("[3b/4] 自动档案钩子 insufficient/queued/already_running/disabled/error", flush=True)
        rm = self.rm
        self.assertEqual(rm._auto_feature_report("c", "prepare")["status"], "insufficient")
        self._images()
        self.assertEqual(rm._auto_feature_report("c", "prepare")["status"], "queued")
        self.assertEqual(rm._auto_feature_report("c", "activate")["status"], "already_running")
        with patch.object(rm, "get_settings") as gs:
            gs.return_value.get.return_value = False
            self.assertEqual(rm._auto_feature_report("c", "x")["status"], "disabled")
        with patch.object(rm, "_test_label_counts", side_effect=RuntimeError("db")):
            self.assertEqual(rm._auto_feature_report("c", "x")["status"], "error")


class FeatureSetUiTest(unittest.TestCase):
    def test_feature_page_scope_slots_and_stale_response(self):
        from PySide6.QtCore import QObject, Signal
        from PySide6.QtWidgets import QApplication
        from ui.pages import feature_page
        app = QApplication.instance() or QApplication([])

        class Client(QObject):
            error_occurred = Signal(str)

            def categories_full(self):
                return [{"category": "02"}, {"category": "tubes", "prepared": True}]

            def feature_revisions(self):
                return {}

        pending = []
        with patch.object(feature_page, "run_async",
                          side_effect=lambda owner, fn, cb: pending.append((fn, cb))):
            page = feature_page.FeaturePage(Client(), lambda: "tubes")
            old_callback = pending[0][1]
            page.reload(["02"])
            self.assertEqual(page.combo_category.count(), 1)
            self.assertEqual(page._category(), "02")
            old_callback({"groups": [{"key": "old", "enabled": True}]})
            self.assertEqual(page.table.rowCount(), 0)
            page._on_feature({"enabled": True, "can_edit": True,
                              "status": "not_applicable", "production_slots": [
                                  {"key": key, "dim": 384, "enabled": True}
                                  for key in ("sem", "disc", "shead")]})
            self.assertEqual(page.table.rowCount(), 3)
            self.assertIn("语义", page.table.item(0, 0).text())
            self.assertFalse(page.btn_draft.isEnabled())
            self.assertFalse(page.btn_report.isEnabled())
            page._on_revisions({"items": {
                "a": {"manifest": {"category": "tubes"}, "state": "draft"},
                "b": {"manifest": {"category": "02"}, "state": "draft"}}})
            self.assertEqual(page.revisions.rowCount(), 1)
            self.assertEqual(page.revisions.item(0, 1).text(), "候选草稿")
            self.assertEqual(feature_page._display({"baseline": 1, "ablated": 0,
                                                   "contribution": 1, "revision": "x"}),
                             "基准得分：1；移除后得分：0；贡献值：1；配置版本：x")
            self.assertIn("贡献值=基准得分-移除后得分", feature_page._display(
                "传统特征原始距离；contribution=baseline-ablated，非融合分数或因果贡献"))
            page.reload([])
            self.assertEqual(page._category(), "")
            self.assertEqual(page.table.rowCount(), 0)
            self.assertEqual(page.revisions.rowCount(), 0)
            page.deleteLater()
        app.processEvents()

    def test_card(self):
        print("[4/4] 学习页特征集卡片 offscreen（操作员只读）", flush=True)
        from PySide6.QtWidgets import QApplication
        from ui.pages.learning_page import LearningPage
        QApplication.instance() or QApplication([])
        page = LearningPage.__new__(LearningPage)
        from PySide6.QtWidgets import QWidget
        QWidget.__init__(page)
        card = page._build_feature_set_card()  # noqa: F841 持有引用防 GC 删布局
        fs = _engine_with(_fake_trad()).feature_set("c")
        page._fill_feature_set({**fs, "can_edit": False,
                                "report": {"status": "stale"}})
        self.assertEqual(len(page._fs_checks), len(freg.TRAD_GROUPS))
        self.assertTrue(page._fs_checks["glcm"].isChecked())
        self.assertFalse(page._fs_checks["glcm"].isEnabled())
        self.assertFalse(page.btn_fs_draft.isEnabled())
        self.assertIn("过期", page.lbl_fs_status.text())
        self.assertIn("仅工程师", page.lbl_fs_status.text())
        page._fill_feature_set({**fs, "can_edit": True, "report": {"status": "insufficient"}})
        self.assertTrue(page.btn_fs_draft.isEnabled())
        self.assertFalse(page.btn_fs_report.isEnabled())
        self.assertTrue(page.btn_fs_rebuild.isEnabled())
        page._fill_feature_set({**fs, "can_edit": True, "report": {"status": "current"},
                                "rebuild_required": True, "pending_rebuild_version": 4,
                                "feature_representation": {"drift": ["registry_sha256"]}})
        self.assertIn("特征表示已过期（registry_sha256）", page.lbl_fs_status.text())
        self.assertIn("v4，待激活门控", page.lbl_fs_status.text())
        page._fill_feature_set({"status": "not_applicable", "enabled": True})
        self.assertFalse(page.btn_fs_rebuild.isEnabled())
        self.assertIn("不适用", page.lbl_fs_status.text())
        card.deleteLater()
        page.deleteLater()


if __name__ == "__main__":
    unittest.main()
