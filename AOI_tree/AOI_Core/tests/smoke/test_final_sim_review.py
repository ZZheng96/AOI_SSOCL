"""AC-064 收尾（模拟验收）：拖边 / 框内平移 / 重做（纯函数 + 后端 redo 链 + 离屏 UI）。"""
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent

from algo.decision import localization as L
from tests.smoke.test_region_review_edit import FakeClient, _by_id, _mouse, _op, _png, _regs
from ui.widgets import region_review_dialog as R

N = 6


def _close(a, b, tol=1):
    return len(a) == len(b) and all(abs(x - y) <= tol for x, y in zip(a, b))


class PureGeometryTest(unittest.TestCase):
    def test_edge_and_move(self):
        print(f"[1/{N}] 纯函数：hit_edge / drag_edge / move_box（对边固定、最小尺寸、图内）", flush=True)
        b = [10, 10, 50, 50]
        self.assertEqual(R.hit_edge(b, 50.5, 30, 2), "r")
        self.assertEqual(R.hit_edge(b, 9, 30, 2), "l")
        self.assertEqual(R.hit_edge(b, 30, 11, 2), "t")
        self.assertEqual(R.hit_edge(b, 30, 49, 2), "b")
        self.assertIsNone(R.hit_edge(b, 30, 30, 2))
        self.assertIsNone(R.hit_edge(b, 50, 80, 2))              # 超出边跨度
        self.assertEqual(R.drag_edge(b, "r", 70, 0, (200, 100)), [10, 10, 70, 50])
        self.assertEqual(R.drag_edge(b, "l", -5, 0), [0, 10, 50, 50])
        self.assertEqual(R.drag_edge(b, "t", 0, 99), [10, 50 - R.MIN_BOX, 50, 50])  # 不翻转
        self.assertEqual(R.drag_edge(b, "b", 0, 999, (200, 100)), [10, 10, 50, 100])
        self.assertEqual(R.move_box(b, 5, -3), [15, 7, 55, 47])
        self.assertEqual(R.move_box(b, -99, -99), [0, 0, 40, 40])   # 夹到图内，尺寸不变
        self.assertEqual(R.move_box(b, 999, 999, (200, 100)), [160, 60, 200, 100])
        body, err = R.build_redo_payload("甲", "", 3, event_id="x1")
        self.assertEqual((err, body), (None, {"op": "redo", "operator": "甲", "event_id": "x1",
                                              "expected_revision": 3, "reason": None}))
        self.assertIn("重做", R.build_redo_payload(" ", "", 3)[1])


class RedoBackendTest(unittest.TestCase):
    def test_redo_chain(self):
        print(f"[2/{N}] 后端 redo：undo→redo→undo、栈式多级重做、事件只追加", flush=True)
        rv = L.new_review(_regs(), "pp-x")
        with self.assertRaisesRegex(ValueError, "没有可重做"):
            _op(rv, "甲", "redo", ev="x0")
        rv = _op(rv, "甲", "revise", "r0", "v1", bbox=[2, 12, 38, 48], reason="偏")
        rv = _op(rv, "甲", "confirm", "r1", "c1")
        rv = _op(rv, "甲", "undo", ev="u1")     # 撤 confirm r1
        rv = _op(rv, "甲", "undo", ev="u2")     # 撤 revise r0
        self.assertEqual(_by_id(rv)["r0"]["bbox"], [0, 10, 40, 50])
        rv = _op(rv, "甲", "redo", ev="x1")     # 重做最近撤销 → revise r0 恢复
        r0 = _by_id(rv)["r0"]
        self.assertEqual((r0["bbox"], r0["source"], r0["redone_by"]["redo_of"]),
                         ([2, 12, 38, 48], "human_box", "u2"))
        self.assertNotIn("undone_by", r0)
        rv = _op(rv, "甲", "redo", ev="x2")     # 再重做 → confirm r1 恢复
        self.assertEqual(_by_id(rv)["r1"]["status"], "confirmed")
        with self.assertRaisesRegex(ValueError, "没有可重做"):
            _op(rv, "甲", "redo", ev="x3")
        view = _by_id(L.annotate_review(rv))
        self.assertEqual((view["r0"]["train_status"], view["r1"]["train_status"]), ("eligible", "eligible"))
        self.assertIn([2, 12, 38, 48], L.trainable_region_boxes(rv))
        # 重做后可再次撤销（原操作重新生效）
        rv = _op(rv, "甲", "undo", ev="u3")
        self.assertEqual(rv["events"][-1]["undo_of"], "c1")
        self.assertEqual(_by_id(rv)["r1"]["status"], "candidate")
        self.assertEqual([e["op"] for e in rv["events"]],
                         ["revise", "confirm", "undo", "undo", "redo", "redo", "undo"])
        self.assertEqual(rv["revision"], len(rv["events"]))

    def test_redo_invalidated_and_conflict(self):
        print(f"[3/{N}] 后端 redo：撤销后新操作→链失效；他人改过→冲突；幂等", flush=True)
        rv = L.new_review(_regs(), "pp-x")
        rv = _op(rv, "甲", "confirm", "r0", "c1")
        rv = _op(rv, "甲", "undo", ev="u1")
        rv2 = _op(rv, "甲", "confirm", "r1", "c2")
        with self.assertRaisesRegex(ValueError, "失效"):
            _op(rv2, "甲", "redo", ev="x1")
        rv3 = _op(rv, "乙", "reject", "r0", "j1", reason="噪点")
        with self.assertRaises(L.ReviewConflict):
            _op(rv3, "甲", "redo", ev="x1")
        with self.assertRaisesRegex(ValueError, "没有可重做"):
            _op(rv, "乙", "redo", ev="x1")                       # 不重做他人撤销
        ok = _op(rv, "甲", "redo", ev="x1")
        self.assertEqual(_by_id(ok)["r0"]["status"], "confirmed")
        with self.assertRaisesRegex(ValueError, "没有可重做"):
            _op(ok, "甲", "redo", ev="x2")                       # 同一撤销不可重做两次

    def test_api_redo_idempotent(self):
        print(f"[3b/{N}] API：redo 走 Literal 校验、同 event_id 幂等、链失效 422", flush=True)
        import tempfile
        from pathlib import Path
        from unittest.mock import Mock, patch

        from fastapi import APIRouter, FastAPI
        from fastapi.testclient import TestClient
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        from backend.api.region_review import register
        from backend.core.security import ApiKeyMiddleware
        from backend.db import database
        from backend.db.models import Base, Detection
        loc = {"regions": _regs(), "postproc": {"version_id": "pp-x"}, "coord_space": {"size": [200, 100]}}
        with tempfile.TemporaryDirectory() as tmp:
            eng = create_engine(f"sqlite:///{Path(tmp) / 'e.db'}", connect_args={"check_same_thread": False})
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
                        def post(**body):
                            return c.post(url, headers=h, json=body)
                        self.assertEqual(post(op="confirm", region_id="r0", operator="甲", event_id="c1",
                                              expected_revision=0).status_code, 200)
                        self.assertEqual(post(op="undo", operator="甲", event_id="u1",
                                              expected_revision=1).status_code, 200)
                        rd = dict(op="redo", operator="甲", event_id="x1", expected_revision=2)
                        r = post(**rd)
                        self.assertEqual(r.status_code, 200)
                        self.assertEqual(_by_id(r.json()["review"])["r0"]["train_status"], "eligible")
                        again = post(**rd).json()
                        self.assertTrue(again["idempotent"])
                        self.assertEqual(again["review"]["revision"], 3)
                        self.assertEqual(post(op="redo", operator="甲", event_id="x2",
                                              expected_revision=3).status_code, 422)
            finally:
                eng.dispose()


class DialogDragRedoTest(unittest.TestCase):
    def setUp(self):
        self.c = FakeClient()
        self.d = R.RegionReviewDialog(self.c, 1)
        self.d.resize(1120, 560)
        self.d.canvas.set_image_bytes(_png(100, 50))   # 显示图为原图 1/2
        cv = self.d.canvas
        cv.resize(800, 400)
        cv.resetTransform()
        cv.scale(4.0, 4.0)   # 固定视图放大 4×：手柄容差 = 6/4 屏幕→原图 3px，避免离屏小视口下容差过大
        self.d.edit_operator.setText("甲")
        self.addCleanup(self.d.deleteLater)

    def _drag(self, a, b):
        cv = self.d.canvas
        cv.mousePressEvent(_mouse(cv, QEvent.MouseButtonPress, a))
        drag = cv._drag
        cv.mouseMoveEvent(_mouse(cv, QEvent.MouseMove, b))
        cv.mouseReleaseEvent(_mouse(cv, QEvent.MouseButtonRelease, b))
        return drag

    def test_edge_drag_revise(self):
        print(f"[4/{N}] 离屏 UI：拖 r1 右边 → 仅 x1 变化 → revise 提交", flush=True)
        self.d.select_region("r1")
        drag = self._drag((45, 15), (48, 15))       # 显示右边 x=45 → 原图 x≈96
        self.assertEqual(drag, ("r1", "r"))
        b = self.d._pending_bbox
        self.assertTrue(_close(b, [50, 10, 96, 50]), b)
        self.assertEqual(b[1::2], [10, 50])
        self.d.edit_reason.setText("右边偏")
        self.assertTrue(self.d.submit_box("revise"))
        self.assertEqual(_by_id(self.c.rv)["r1"]["bbox"], b)

    def test_move_and_click_noop(self):
        print(f"[5/{N}] 离屏 UI：框内拖动整框平移（尺寸不变）；原地单击不产生待提交", flush=True)
        self.d.select_region("r1")
        drag = self._drag((35, 15), (35, 15))       # 原地按下松开
        self.assertEqual(drag[:2], ("r1", "move"))
        self.assertIsNone(self.d._pending_bbox)
        self._drag((35, 15), (38, 17))              # 原图位移 ≈ (6, 4)
        b = self.d._pending_bbox
        self.assertTrue(_close(b, [56, 14, 96, 54]), b)
        self.assertEqual((b[2] - b[0], b[3] - b[1]), (40, 40))
        self.d.edit_reason.setText("整体偏移")
        self.assertTrue(self.d.submit_box("revise"))
        self.assertEqual(_by_id(self.c.rv)["r1"]["bbox"], b)

    def test_redo_button(self):
        print(f"[6/{N}] 离屏 UI：撤销 → 重做按钮恢复 → 无可重做时失败；表格同步", flush=True)
        self.d.select_region("r0")
        self.assertTrue(self.d.apply("confirm"))
        self.d.btn_undo.click()
        self.assertEqual(self.d.table.item(0, 6).text(), "排除:候选")
        self.d.btn_redo.click()
        self.assertEqual(_by_id(self.c.rv)["r0"]["status"], "confirmed")
        self.assertEqual(self.d.table.item(0, 6).text(), "可训")
        self.assertEqual(self.c.rv["events"][-1]["op"], "redo")
        self.assertFalse(self.d.redo_last())        # 已无可重做（422→None）
        self.d.edit_operator.setText("")
        self.assertFalse(self.d.redo_last())
        self.assertIn("操作员", self.d.lbl_status.text())


if __name__ == "__main__":
    unittest.main(verbosity=2)
