"""AC-051/056/057 离屏模拟 TEST-04 桌面交互链路：
临时 SQLite 产线落库 → 真实 api_detections_live → 真实 MonitorPage._on_live_batch
（条带三态着色/短脉冲降级/开放集徽标独立/回看取消跟随/增量游标），
真实 api_incomplete_report → 真实 DashboardPage._fill_incomplete（仅认 alert）。
不连真实后端、不起下载线程（viewer.set_image_url 打桩记录）。"""
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.api import routes_detect as rd
from backend.api import routes_review as rr
from backend.db import database
from backend.db.models import Base, Detection
from ui.pages import dashboard_page as D
from ui.pages import monitor_page as M
from ui.theme import DANGER, SUCCESS, TEXT_SUB, WARNING

_app = QApplication.instance() or QApplication([])
N = 4
_T0 = time.perf_counter()


def _log(i, msg):
    print(f"[{i}/{N}] {msg}（+{time.perf_counter() - _T0:.2f}s）", flush=True)


class FakeClient(QObject):
    error_occurred = Signal(str)

    def file_url(self, p):
        return f"fake://{p}"


# (target_type, category, score, decision, open_alert)；decision=None 为旧行
_ROWS = [
    ("pipeline", "A", 0.10, "normal", False),
    ("pipeline", "A", 0.55, "gray", True),
    ("upload", "A", 0.90, "anomaly", False),     # 非产线：不得进 live 流
    ("pipeline", "A", 0.80, "anomaly", False),   # 孤立单帧 → 短脉冲
    ("pipeline", "A", 0.12, "normal", False),
    ("pipeline", "A", 0.85, "anomaly", False),
    ("pipeline", "A", 0.88, "anomaly", True),    # 连续异常 + 开放集
    ("pipeline", "A", 0.30, None, False),        # 旧行无 decision
]


def _add(s, tt, cat, score, decision, open_alert):
    nt = {"threshold": 0.6, "gray_threshold": 0.4, "open_alert": open_alert}
    if decision is not None:
        nt["decision"] = decision
    s.add(Detection(target_type=tt, category=cat, image_path=f"{cat}/{score}.png",
                    final_score=score, is_anomaly=decision == "anomaly",
                    latency_ms=10.0, n_tiles=nt))


class _DbCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="aoi_final_sim_ui_")
        self.addCleanup(tmp.cleanup)
        db = create_engine("sqlite:///" + (Path(tmp.name) / "a.db").as_posix(),
                           connect_args={"check_same_thread": False})
        self.addCleanup(db.dispose)
        Base.metadata.create_all(db)
        for p in (patch.object(database, "_engine", db),
                  patch.object(database, "_SessionLocal",
                               sessionmaker(bind=db, expire_on_commit=False))):
            p.start()
            self.addCleanup(p.stop)
        with database.session_scope() as s:
            for r in _ROWS:
                _add(s, *r)


def _color(cell):
    ss = cell.styleSheet()
    return ss.split("background:")[1].split(";")[0].strip()


class MonitorLiveChainTest(_DbCase):
    def _page(self):
        p = M.MonitorPage(FakeClient(), lambda: "A")
        p._pipe_timer.stop()
        p._live_timer.stop()
        self.urls = []
        p.viewer.set_image_url = lambda url, layer="original": self.urls.append((url, layer))
        self.addCleanup(p.deleteLater)
        return p

    def test_live_strip_colors_and_badges(self):
        _log(1, "真实 live 首批 → 条带三态着色/短脉冲/旧行中性/跟随最新帧")
        p = self._page()
        data = rd.api_detections_live(after_id=0, limit=M.LIVE_WINDOW)
        self.assertTrue(all(it["target_type"] in rd._LIVE_TARGET_TYPES for it in data["items"]))
        p._on_live_batch(data)
        self.assertEqual(len(p._frames), 7)               # upload 行被过滤
        self.assertEqual(len(p._strip_cells), 7)
        colors = [_color(c) for c in p._strip_cells]
        self.assertEqual(colors, [SUCCESS, WARNING, WARNING, SUCCESS, DANGER, DANGER, TEXT_SUB])
        self.assertEqual(p._pulses, [False, False, True, False, False, False, False])
        self.assertIn("短脉冲", p._strip_cells[2].toolTip())
        # 开放集帧（idx5 连续异常+open）颜色仍为 DANGER：open 不改判定色
        self.assertEqual(colors[5], DANGER)
        # 跟随：展示最新帧（旧行）→ 未知/未判定，gauge 原值
        self.assertEqual(p.badge_verdict.text(), "未知 / 未判定")
        self.assertIsNone(p.gauge._decision)
        self.assertAlmostEqual(p.gauge._score, 0.30)
        self.assertAlmostEqual(p.gauge._threshold, 0.6)
        self.assertTrue(p.badge_open.isHidden())
        self.assertIn("异常 3 · 灰区 1", p.lbl_live_kpi.text())
        self.assertIn("（3/7）", p.lbl_strip_stats.text())
        self.assertTrue(self.urls and self.urls[-1][0].startswith("fake://"))

    def test_strip_click_review_and_incremental(self):
        _log(2, "条带回看灰区帧（取消跟随/开放集徽标独立）→ 增量 after_id 推进不跳帧")
        p = self._page()
        p._on_live_batch(rd.api_detections_live(after_id=0, limit=M.LIVE_WINDOW))
        last = p._last_id
        self.assertTrue(p.chk_follow.isChecked())
        p._on_strip_click(1)
        self.assertFalse(p.chk_follow.isChecked())
        self.assertEqual(p._selected_frame, 1)
        self.assertEqual(p.badge_verdict.text(), "待复核（灰区）")
        self.assertEqual(p.gauge._decision, "gray")
        self.assertAlmostEqual(p.gauge._score, 0.55)
        self.assertFalse(p.badge_open.isHidden())
        self.assertFalse(p.lbl_review_hint.isHidden())
        self.assertIn("2px solid", p._strip_cells[1].styleSheet())
        # 回看 idx5：异常 + open，徽标显示但判定仍为“异 常”
        p._on_strip_click(5)
        self.assertEqual(p.badge_verdict.text(), "异 常")
        self.assertFalse(p.badge_open.isHidden())
        self.assertTrue(p.lbl_review_hint.isHidden())
        # 新帧落库 → 增量拉取
        with database.session_scope() as s:
            _add(s, "pipeline", "A", 0.05, "normal", False)
        inc = rd.api_detections_live(after_id=last, limit=M.LIVE_WINDOW)
        self.assertEqual(len(inc["items"]), 1)
        self.assertFalse(inc.get("gap"))
        p._on_live_batch(inc)
        self.assertEqual(p._last_id, last + 1)
        self.assertEqual(len(p._frames), 8)
        self.assertEqual(p._selected_frame, 5)            # 未跟随：不跳到最新帧
        self.assertEqual(p.badge_verdict.text(), "异 常")
        # 重复投递同一批（幂等）不重复入窗
        p._on_live_batch(inc)
        self.assertEqual(len(p._frames), 8)
        # 空增量：不改状态
        p._on_live_batch(rd.api_detections_live(after_id=p._last_id, limit=M.LIVE_WINDOW))
        self.assertEqual(len(p._frames), 8)


class DashboardIncompleteChainTest(_DbCase):
    def test_real_incomplete_report_to_dashboard(self):
        _log(3, "真实 incomplete_report → 工作台待办：仅 alert 计数，旧 warn 忽略")
        with database.session_scope() as s:
            for i in range(4):
                _add(s, "pipeline", "B", 0.7, "anomaly", i < 3)
        data = rr.api_incomplete_report()
        by = {r["category"]: r for r in data["reports"]}
        self.assertFalse(by["A"]["alert"])               # A：2/8 命中 < min_count
        self.assertTrue(by["B"]["alert"])                # B：3/4
        self.assertTrue(data["any_alert"])
        data["reports"].append({"category": "C", "warn": True})
        p = D.DashboardPage(FakeClient())
        self.addCleanup(p.deleteLater)
        p._fill_incomplete(data)
        self.assertEqual(p.todo_incomplete[1].text(), "1")
        self.assertEqual(p.todo_incomplete[1].toolTip(), "预警品类：B")
        self.assertIn(DANGER, p.todo_incomplete[1].styleSheet())


class ResetChainTest(_DbCase):
    def test_reset_required_resubscribe(self):
        _log(4, "游标超出全表最大 ID（库重置）→ 页面回到 0 重新订阅")
        p = M.MonitorPage(FakeClient(), lambda: "A")
        p._pipe_timer.stop()
        p._live_timer.stop()
        p.viewer.set_image_url = lambda *a, **k: None
        self.addCleanup(p.deleteLater)
        data = rd.api_detections_live(after_id=999, limit=M.LIVE_WINDOW)
        self.assertTrue(data["reset_required"])
        p._last_id = 999
        p._on_live_batch(data)
        self.assertEqual(p._last_id, 0)
        self.assertIn("重新订阅", p.lbl_live.text())


if __name__ == "__main__":
    unittest.main()
