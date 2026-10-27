"""AC-064 剩余项：复核对话框角点微调 / 删除人工框 / 撤销上一步（纯函数 + 后端语义 + API + 离屏 UI）。"""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QBuffer, QByteArray, QEvent, QIODevice, QPointF, Qt
from PySide6.QtGui import QColor, QImage, QMouseEvent
from PySide6.QtWidgets import QApplication

from algo.decision import localization as L
from ui.widgets import region_review_dialog as R

_app = QApplication.instance() or QApplication([])
N = 9


def _png(w, h):
    img = QImage(w, h, QImage.Format_RGB32)
    img.fill(QColor("#808080"))
    ba = QByteArray()
    buf = QBuffer(ba)
    buf.open(QIODevice.WriteOnly)
    img.save(buf, "PNG")
    return bytes(ba)


def _regs():
    return [{"region_id": f"r{i}", "bbox": [i * 50, 10, i * 50 + 40, 50], "area": 1600,
             "source": "engine_box", "score_peak": 0.9} for i in range(2)]


def _op(rv, who, op, rid=None, ev="e", **kw):
    return L.apply_op(rv, {"op": op, "region_id": rid, "operator": who, "event_id": ev,
                           "expected_revision": rv["revision"], **kw}, size=[200, 100])


def _by_id(rv):
    return {r["region_id"]: r for r in rv["regions"]}


class FakeClient:
    """内存版 /regions：GET=annotate_review；POST 走真实 apply_op（冲突→409，非法→None 同 ApiClient 弹错）。"""

    def __init__(self):
        self.rv = L.new_review(_regs(), "pp-x")
        self.size = [200, 100]

    def get(self, path, **kw):
        return {"detection_id": 1, "coord_space": self.size, **L.annotate_review(self.rv)}

    def post(self, path, json=None, conflict_ok=False, **kw):
        if json["expected_revision"] != self.rv.get("revision", 0):
            return {"_status": 409, "detail": "区域复核已被修改，请刷新后重试"}
        try:
            self.rv = L.apply_op(self.rv, json, size=self.size)
        except L.ReviewConflict as e:
            return {"_status": 409, "detail": str(e)}
        except (ValueError, KeyError):
            return None
        return {"review": L.annotate_review(self.rv)}


class PureFunctionTest(unittest.TestCase):
    def test_corner_geometry(self):
        print(f"[1/{N}] 纯函数：角点/命中容差/拖拽约束（对角固定、最小尺寸、图内、不翻转）", flush=True)
        self.assertEqual(R.bbox_corners([1, 2, 3, 4]),
                         {"tl": (1.0, 2.0), "tr": (3.0, 2.0), "br": (3.0, 4.0), "bl": (1.0, 4.0)})
        # 显示图为原图 1/2、视图放大 2 倍 → 6 屏幕 px = 6 原图 px
        self.assertAlmostEqual(R.handle_tolerance(2.0, (100, 50), (200, 100)), 6.0)
        self.assertAlmostEqual(R.handle_tolerance(1.0, (100, 50), (200, 100), px=4), 8.0)
        b = [10, 10, 50, 50]
        self.assertEqual(R.hit_corner(b, 12, 9, 3), "tl")
        self.assertEqual(R.hit_corner(b, 49, 52, 3), "br")
        self.assertIsNone(R.hit_corner(b, 30, 30, 3))
        self.assertEqual(R.hit_corner([0, 0, 2, 2], 1.6, 1.6, 5), "br")   # 小框取最近角
        self.assertEqual(R.drag_corner(b, "br", 70, 80, (200, 100)), [10, 10, 70, 80])
        self.assertEqual(R.drag_corner(b, "tl", 5, 6), [5, 6, 50, 50])
        self.assertEqual(R.drag_corner(b, "tr", 60, 0), [10, 0, 60, 50])
        self.assertEqual(R.drag_corner(b, "bl", 0, 90, (200, 100)), [0, 10, 50, 90])
        self.assertEqual(R.drag_corner(b, "br", 999, 999, (200, 100)), [10, 10, 200, 100])  # 裁剪
        self.assertEqual(R.drag_corner(b, "tl", -9, -9), [0, 0, 50, 50])
        self.assertEqual(R.drag_corner(b, "br", 0, 0), [10, 10, 10 + R.MIN_BOX, 10 + R.MIN_BOX])  # 不翻转
        self.assertEqual(R.drag_corner(b, "tl", 99, 99), [50 - R.MIN_BOX, 50 - R.MIN_BOX, 50, 50])
        self.assertEqual(R.drag_corner(b, "br", 13, 13, min_size=8), [10, 10, 18, 18])

    def test_delete_undo_payload(self):
        print(f"[2/{N}] 纯函数：can_delete / delete、undo payload 校验", flush=True)
        human = {"region_id": "h0", "source": "human_box", "origin_source": "human_box", "status": "confirmed"}
        engine_rev = {"region_id": "r0", "source": "human_box", "origin_source": "engine_box", "status": "confirmed"}
        self.assertTrue(R.can_delete(human))
        self.assertFalse(R.can_delete(engine_rev))              # 被修订过的模型框仍不可删
        self.assertFalse(R.can_delete({**human, "status": "deleted"}))
        self.assertFalse(R.can_delete(None))
        body, err = R.build_delete_payload(human, " 甲 ", "误画", 4, event_id="d1")
        self.assertIsNone(err)
        self.assertEqual(body, {"op": "delete", "region_id": "h0", "operator": "甲", "event_id": "d1",
                                "expected_revision": 4, "reason": "误画"})
        self.assertIn("拒绝", R.build_delete_payload(engine_rev, "甲", "x", 0)[1])
        self.assertIn("理由", R.build_delete_payload(human, "甲", " ", 0)[1])
        self.assertIn("操作员", R.build_delete_payload(human, "", "x", 0)[1])
        self.assertIn("修订号", R.build_delete_payload(human, "甲", "x", None)[1])
        self.assertIn("选择", R.build_delete_payload(None, "甲", "x", 0)[1])
        body, err = R.build_undo_payload("甲", "", 7, event_id="u1")
        self.assertEqual((err, body), (None, {"op": "undo", "operator": "甲", "event_id": "u1",
                                              "expected_revision": 7, "reason": None}))
        self.assertIn("操作员", R.build_undo_payload(" ", "x", 0)[1])
        self.assertIn("修订号", R.build_undo_payload("甲", "x", None)[1])


class BackendSemanticsTest(unittest.TestCase):
    def test_delete_semantics(self):
        print(f"[3/{N}] 后端 delete：仅人工框、须 reason、软删除留档、不进训练", flush=True)
        rv = L.new_review(_regs(), "pp-x")
        rv = _op(rv, "甲", "add", ev="a1", bbox=[100, 60, 140, 90], reason="漏检")
        rv = _op(rv, "甲", "revise", "r0", "v1", bbox=[2, 12, 38, 48], reason="偏")
        rv = _op(rv, "甲", "revise", "r0", "v2", bbox=[3, 12, 38, 48], reason="再偏")
        self.assertEqual(_by_id(rv)["r0"]["revised_from"]["source"], "engine_box")  # 二次修订保留最初来源
        with self.assertRaisesRegex(ValueError, "拒绝"):
            _op(rv, "甲", "delete", "r0", "d0", reason="x")   # 修订过的模型框
        with self.assertRaisesRegex(ValueError, "拒绝"):
            _op(rv, "甲", "delete", "r1", "d0", reason="x")   # 候选框
        with self.assertRaisesRegex(ValueError, "reason"):
            _op(rv, "甲", "delete", "h0", "d0")
        self.assertIn([100, 60, 140, 90], L.trainable_region_boxes(rv))
        rv2 = _op(rv, "乙", "delete", "h0", "d1", reason="误画")
        h0 = _by_id(rv2)["h0"]
        self.assertEqual((h0["status"], h0["confirmed"], h0["bbox"], h0["operator"]),
                         ("deleted", False, [100, 60, 140, 90], "乙"))
        self.assertEqual(h0["deleted_from"], {"status": "confirmed", "operator": "甲"})
        self.assertNotIn([100, 60, 140, 90], L.trainable_region_boxes(rv2))
        self.assertNotIn([100, 60, 140, 90], L.trainable_region_boxes(L.annotate_review(rv2)))
        view = _by_id(L.annotate_review(rv2))["h0"]
        self.assertEqual((view["train_status"], view["review_revision"]), ("excluded_deleted", 4))
        ev = rv2["events"][-1]
        self.assertEqual((ev["op"], ev["region_id"], ev["before"]["status"], rv2["revision"]),
                         ("delete", "h0", "confirmed", 4))
        self.assertEqual(len(rv2["events"]), 4)                   # 修订链只追加
        with self.assertRaisesRegex(ValueError, "已删除"):
            _op(rv2, "乙", "confirm", "h0", "c9")

    def test_undo_semantics(self):
        print(f"[4/{N}] 后端 undo：反向修订、仅本人、他人改过→冲突、链完整", flush=True)
        rv = L.new_review(_regs(), "pp-x")
        with self.assertRaisesRegex(ValueError, "没有可撤销"):
            _op(rv, "甲", "undo", ev="u0")
        rv = _op(rv, "甲", "revise", "r0", "v1", bbox=[2, 12, 38, 48], reason="偏")
        rv = _op(rv, "乙", "add", ev="a1", bbox=[100, 60, 140, 90], reason="漏检")
        rv = _op(rv, "甲", "confirm", "r1", "c1")
        # 甲撤销最近一次（confirm r1）→ 恢复 candidate
        rv = _op(rv, "甲", "undo", ev="u1")
        r1 = _by_id(rv)["r1"]
        self.assertEqual((r1["status"], r1["confirmed"], r1["undone_by"]["undo_of"]), ("candidate", False, "c1"))
        self.assertEqual((rv["events"][-1]["op"], rv["events"][-1]["undo_of"], rv["revision"]), ("undo", "c1", 4))
        # 再撤销 → 撤 revise r0：坐标/来源回到模型框，不再计为人工
        rv = _op(rv, "甲", "undo", ev="u2")
        r0v = _by_id(L.annotate_review(rv))["r0"]
        self.assertEqual((r0v["bbox"], r0v["source"], r0v["train_status"]),
                         ([0, 10, 40, 50], "engine_box", "excluded_candidate"))
        with self.assertRaisesRegex(ValueError, "没有可撤销"):
            _op(rv, "甲", "undo", ev="u3")                     # 甲已无可撤操作；不撤他人
        # 乙撤销 add → retracted，不训练，region_id 不复用
        rv = _op(rv, "乙", "undo", ev="u4")
        h0 = _by_id(rv)["h0"]
        self.assertEqual((h0["status"], L.train_status(h0)), ("retracted", "excluded_retracted"))
        self.assertEqual(L.trainable_region_boxes(rv), [])
        rv = _op(rv, "乙", "add", ev="a2", bbox=[10, 60, 30, 90], reason="补")
        self.assertEqual(rv["regions"][-1]["region_id"], "h1")
        self.assertEqual([e["op"] for e in rv["events"]],
                         ["revise", "add", "confirm", "undo", "undo", "undo", "add"])
        self.assertEqual(rv["revision"], len(rv["events"]))
        # 冲突：甲确认 r1 后乙改了 r1 → 甲撤销返回冲突
        rv = _op(rv, "甲", "confirm", "r1", "c2")
        rv = _op(rv, "乙", "reject", "r1", "j2", reason="其实是噪点")
        with self.assertRaises(L.ReviewConflict):
            _op(rv, "甲", "undo", ev="u5")
        # 撤销 delete → 恢复人工框可训
        rv = _op(rv, "乙", "delete", "h1", "d1", reason="误画")
        rv = _op(rv, "乙", "undo", ev="u6")
        self.assertEqual(_by_id(rv)["h1"]["status"], "confirmed")
        self.assertIn([10, 60, 30, 90], L.trainable_region_boxes(rv))


class ApiTest(unittest.TestCase):
    def test_api_delete_undo(self):
        from fastapi import APIRouter, FastAPI
        from fastapi.testclient import TestClient
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        from backend.api.region_review import register
        from backend.core.security import ApiKeyMiddleware
        from backend.db import database
        from backend.db.models import Base, Detection
        print(f"[5/{N}] API：delete/undo 幂等、revision 409、撤销冲突 409、模型框删除 422、落库不进训练", flush=True)
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

                    def post(**body):
                        return c.post(url, headers=h, json=body)
                    with TestClient(app) as c:
                        self.assertEqual(post(op="add", bbox=[100, 60, 140, 90], operator="甲", event_id="a1",
                                              expected_revision=0, reason="漏检").status_code, 200)
                        self.assertEqual(post(op="delete", region_id="r0", operator="甲", event_id="d0",
                                              expected_revision=1, reason="x").status_code, 422)
                        self.assertEqual(post(op="delete", region_id="h0", operator="甲", event_id="d0",
                                              expected_revision=1).status_code, 422)   # 缺 reason
                        self.assertEqual(post(op="delete", region_id="h0", operator="甲", event_id="d1",
                                              expected_revision=0, reason="误画").status_code, 409)
                        dl = dict(op="delete", region_id="h0", operator="甲", event_id="d1",
                                  expected_revision=1, reason="误画")
                        r = post(**dl)
                        self.assertEqual(r.status_code, 200)
                        self.assertEqual(_by_id(r.json()["review"])["h0"]["train_status"], "excluded_deleted")
                        again = post(**dl).json()
                        self.assertTrue(again["idempotent"])
                        self.assertEqual(again["review"]["revision"], 2)
                        self.assertEqual(post(**{**dl, "reason": "别的"}).status_code, 409)  # 同 id 异载荷
                        # undo（撤销 delete）幂等
                        ud = dict(op="undo", operator="甲", event_id="u1", expected_revision=2)
                        r = post(**ud)
                        self.assertEqual(r.status_code, 200)
                        self.assertEqual(_by_id(r.json()["review"])["h0"]["status"], "confirmed")
                        self.assertTrue(post(**ud).json()["idempotent"])
                        # 甲确认 r1，乙推翻（带理由）→ 甲撤销 409
                        post(op="confirm", region_id="r1", operator="甲", event_id="c1", expected_revision=3)
                        post(op="reject", region_id="r1", operator="乙", event_id="j1", expected_revision=4,
                             reason="噪点")
                        r = post(op="undo", operator="甲", event_id="u2", expected_revision=5)
                        self.assertEqual(r.status_code, 409)
                        self.assertIn("他人", r.json()["detail"])
                        self.assertEqual(post(op="undo", operator="丙", event_id="u3",
                                              expected_revision=5).status_code, 422)   # 无本人操作
                    with database.session_scope() as s:
                        stored = s.get(Detection, 1).n_tiles["region_review"]
                        self.assertEqual(stored["revision"], 5)
                        self.assertEqual([e["op"] for e in stored["events"]],
                                         ["add", "delete", "undo", "confirm", "reject"])
                        self.assertEqual(L.trainable_region_boxes(stored), [[100, 60, 140, 90]])
            finally:
                eng.dispose()


def _mouse(canvas, kind, scene_xy, buttons=Qt.LeftButton):
    p = canvas.mapFromScene(QPointF(*scene_xy))
    return QMouseEvent(kind, QPointF(p), QPointF(canvas.viewport().mapToGlobal(p)), Qt.LeftButton,
                       Qt.NoButton if kind == QEvent.MouseButtonRelease else buttons, Qt.NoModifier)


class DialogEditTest(unittest.TestCase):
    def setUp(self):
        self.c = FakeClient()
        self.d = R.RegionReviewDialog(self.c, 1)
        self.d.resize(1120, 560)
        self.d.canvas.set_image_bytes(_png(100, 50))   # 显示图为原图 1/2
        self.d.canvas.fit_to_view()
        self.d.edit_operator.setText("甲")
        self.addCleanup(self.d.deleteLater)

    def test_handles_and_corner_drag_revise(self):
        print(f"[6/{N}] 离屏 UI：选中显示 4 角手柄 → 拖 br 角 → revise 提交（须理由）", flush=True)
        cv = self.d.canvas
        self.assertEqual(cv.handle_items(), {})
        self.d.select_region("r1")
        hs = cv.handle_items()
        self.assertEqual(set(hs), {"tl", "tr", "br", "bl"})
        self.assertEqual((hs["br"].pos().x(), hs["br"].pos().y()), (45.0, 25.0))  # 原图 (90,50) → 显示
        # 鼠标事件：按住 br 角拖到显示 (48, 35) → 原图 (96, 70)
        cv.mousePressEvent(_mouse(cv, QEvent.MouseButtonPress, (45, 25)))
        self.assertEqual(cv._drag, ("r1", "br"))
        cv.mouseMoveEvent(_mouse(cv, QEvent.MouseMove, (48, 35)))
        cv.mouseReleaseEvent(_mouse(cv, QEvent.MouseButtonRelease, (48, 35)))
        self.assertIsNone(cv._drag)
        b = self.d._pending_bbox
        self.assertEqual((self.d._pending_rid, b[:2]), ("r1", [50, 10]))
        self.assertTrue(abs(b[2] - 96) <= 1 and abs(b[3] - 70) <= 1, b)
        self.assertEqual(self.d.selected_region_id(), "r1")
        self.assertIn("角点微调", self.d.lbl_pending.text())
        self.assertFalse(self.d.submit_box("revise"))           # 无理由
        self.assertIn("理由", self.d.lbl_status.text())
        self.d.edit_reason.setText("右下角偏")
        self.assertTrue(self.d.submit_box("revise"))
        r1 = _by_id(self.c.rv)["r1"]
        self.assertEqual((r1["bbox"], r1["revised_from"]["bbox"], r1["source"]),
                         (b, [50, 10, 90, 50], "human_box"))
        self.assertEqual(self.c.rv["events"][-1]["payload"]["reason"], "右下角偏")
        self.assertIsNone(self.d._pending_rid)
        # 拖到极小 → 最小尺寸约束；切换选中其他框 → 放弃未提交微调
        self.d.select_region("r0")
        cv.mousePressEvent(_mouse(cv, QEvent.MouseButtonPress, (0, 5)))
        self.assertEqual(cv._drag, ("r0", "tl"))
        cv.mouseMoveEvent(_mouse(cv, QEvent.MouseMove, (40, 40)))
        cv.mouseReleaseEvent(_mouse(cv, QEvent.MouseButtonRelease, (40, 40)))
        self.assertEqual(self.d._pending_bbox, [40 - R.MIN_BOX, 50 - R.MIN_BOX, 40, 50])
        self.d.select_region("r1")
        self.assertIsNone(self.d._pending_bbox)
        self.assertEqual(cv.region_items()["r0"].rect().width(), 20.0)   # 恢复服务端坐标

    def test_delete_button(self):
        print(f"[7/{N}] 离屏 UI：删除人工框（模型框拦截）→ 画布隐藏、表格留档", flush=True)
        self.d.select_region("r0")
        self.d.edit_reason.setText("误画")
        self.assertFalse(self.d.delete_selected())
        self.assertIn("拒绝", self.d.lbl_status.text())
        self.d._on_box_drawn([100, 60, 140, 90])
        self.assertTrue(self.d.submit_box("add"))
        self.d.select_region("h0")
        self.d.edit_reason.setText("")
        self.assertFalse(self.d.delete_selected())
        self.d.edit_reason.setText("误画")
        self.d.btn_delete.click()
        self.assertEqual(_by_id(self.c.rv)["h0"]["status"], "deleted")
        self.assertNotIn("h0", self.d.canvas.region_items())
        self.assertEqual(self.d.table.rowCount(), 3)
        self.assertEqual(self.d.table.item(2, 4).text(), "已删除(留档)")
        self.assertEqual(self.d.table.item(2, 6).text(), "排除:删除")
        self.assertEqual(self.d.canvas.handle_items(), {})          # 已删除框无手柄
        self.assertEqual(R.hit_region([r for r in self.d._review["regions"] if R.is_active(r)], 120, 70), None)

    def test_undo_button_and_conflict(self):
        print(f"[8/{N}] 离屏 UI：撤销上一步（恢复框/状态）与他人修改后 409 提示", flush=True)
        self.d.select_region("r0")
        self.assertTrue(self.d.apply("confirm"))
        self.d.btn_undo.click()
        self.assertEqual(_by_id(self.c.rv)["r0"]["status"], "candidate")
        self.assertEqual(self.d.table.item(0, 6).text(), "排除:候选")
        self.assertFalse(self.d.undo_last())                        # 无可撤（FakeClient 422→None）
        self.assertTrue(self.d.apply("confirm"))
        self.c.rv = _op(self.c.rv, "乙", "reject", "r0", "j1", reason="噪点")
        self.d.refresh()
        self.assertFalse(self.d.undo_last())
        self.assertIn("他人", self.d.lbl_status.text())
        self.assertEqual(self.d._review["revision"], self.c.rv["revision"])
        self.d.edit_operator.setText("")
        self.assertFalse(self.d.undo_last())
        self.assertIn("操作员", self.d.lbl_status.text())

    def test_stale_revision_refresh(self):
        print(f"[9/{N}] 离屏 UI：删除/撤销携带过期 revision → 409 刷新后可重试", flush=True)
        self.d._on_box_drawn([100, 60, 140, 90])
        self.d.edit_reason.setText("漏检")
        self.assertTrue(self.d.submit_box("add"))
        self.c.rv = _op(self.c.rv, "丙", "confirm", "r1", "x1")
        self.d.select_region("h0")
        self.assertFalse(self.d.delete_selected())
        self.assertIn("已被修改", self.d.lbl_status.text())
        self.d.select_region("h0")
        self.assertTrue(self.d.delete_selected())
        self.assertTrue(self.d.undo_last())
        self.assertEqual(_by_id(self.c.rv)["h0"]["status"], "confirmed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
