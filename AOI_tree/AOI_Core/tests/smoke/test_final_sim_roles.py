"""AC-073 离屏模拟：角色工作区 × 四阶段（检测→复核→评估→发布）权限联调。

- 后端：子进程内 TEMP cfg（security.keys 三角色）+ 临时 SQLite + TestClient，
  release() 打桩为 fake（不加载真实引擎/快照）；低权限发布 403 且 fake 未被调用，
  admin 过鉴权后由 fake 返回 409 并落 *_rejected 审计（user=admin）。
- 前端：真实 role_allows_nav + MainWindow._apply_role 作用于离屏 QListWidget。
子进程隔离：AOI_CONFIG 须在 import backend 前设置，避免污染同进程其他测试。"""
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[2]
N = 3
_T0 = time.perf_counter()

# 四阶段 → (导航页, [(method, path, body, 最低角色)])
STAGES = {
    "检测": ("实时监控", [("get", "/api/detections/live", None, "operator")]),
    "复核": ("标注反馈", [("post", "/api/detections/1/regions",
                         {"op": "confirm", "region_id": "r1", "operator": "x",
                          "event_id": "e1", "expected_revision": 0}, "engineer")]),
    "评估": ("评估看板", [("get", "/api/candidates/A/2", None, "engineer"),
                        ("post", "/api/candidates/A/2/acceptance",
                         {"old_image_ids": [], "new_image_ids": []}, "engineer")]),
    "发布": ("模型管理", [("post", "/api/candidates/A/2/decision",
                         {"approved": True, "report_hash": "h", "reason": "r"}, "admin"),
                        ("post", "/api/candidates/A/2/activate", None, "admin"),
                        ("post", "/api/candidates/A/2/rollback", None, "admin"),
                        ("post", "/api/candidates/A/recover", None, "admin")]),
}
RANK = {"operator": 0, "engineer": 1, "admin": 2}
KEYS = {"operator": "op-key", "engineer": "eng-key", "admin": "adm-key"}


def _log(i, msg):
    print(f"[{i}/{N}] {msg}（+{time.perf_counter() - _T0:.2f}s）", flush=True)


def _child(out_path: str) -> None:
    """子进程：真实 FastAPI 依赖链 + fake release。"""
    from unittest.mock import patch
    from fastapi.testclient import TestClient
    from backend.api import app as app_mod, routes_candidates as rc
    from backend.db.database import session_scope
    from backend.db.models import OperationLog
    from backend.self_learning.candidate_release import CandidateReleaseError

    calls = []

    class FakeRelease:
        def __getattr__(self, name):
            def _f(*a, **k):
                calls.append(name)
                raise CandidateReleaseError("fake: 无候选", reason_code="FAKE_NO_CANDIDATE")
            return _f

    res = {"matrix": {}, "release_calls": {}, "audit": []}
    with patch.object(rc, "release", lambda: FakeRelease()), \
            patch.object(rc.task_manager, "submit", lambda *a, **k: "fake-task"):
        client = TestClient(app_mod.app, raise_server_exceptions=False)
        res["no_key"] = client.post("/api/candidates/A/2/activate").status_code
        for role, key in KEYS.items():
            for stage, (_page, eps) in STAGES.items():
                for method, path, body, _min in eps:
                    before = len(calls)
                    kw = {"headers": {"X-API-Key": key}}
                    if body is not None:
                        kw["json"] = body
                    r = getattr(client, method)(path, **kw)
                    res["matrix"][f"{role}|{method} {path}"] = r.status_code
                    res["release_calls"][f"{role}|{method} {path}"] = len(calls) - before
    with session_scope() as s:
        res["audit"] = [(o.user, o.action) for o in s.query(OperationLog).all()]
    Path(out_path).write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")


class RoleApiMatrixTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import yaml
        cls._tmp = tempfile.TemporaryDirectory(prefix="aoi_final_sim_roles_")
        tmp = Path(cls._tmp.name)
        cfg = tmp / "cfg.yaml"
        cfg.write_text(yaml.safe_dump({
            "system": {"database_url": "sqlite:///" + (tmp / "aoi.db").as_posix(),
                       "storage_dir": (tmp / "storage").as_posix(),
                       "engine_storage": (tmp / "storage" / "engine").as_posix(),
                       "engine_base_cfg": "configs/engine_fast.yaml"},
            "security": {"keys": {v: k for k, v in KEYS.items()}},
            "archive": {"enabled": False}}, allow_unicode=True), encoding="utf-8")
        out = tmp / "out.json"
        env = {**os.environ, "AOI_CONFIG": str(cfg), "AOI_FINAL_SIM_ROLES_CHILD": str(out)}
        p = subprocess.run([sys.executable, "-m", "tests.smoke.test_final_sim_roles"],
                           cwd=str(ROOT), env=env, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=300)
        if p.returncode != 0 or not out.exists():
            raise AssertionError(f"子进程失败 rc={p.returncode}\n{p.stderr[-3000:]}")
        cls.res = json.loads(out.read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_stage_permission_matrix(self):
        _log(1, "四阶段 × 三角色 API 矩阵：低于最低角色 403 且不触达发布逻辑")
        m, calls = self.res["matrix"], self.res["release_calls"]
        self.assertEqual(self.res["no_key"], 401)
        for role in KEYS:
            for stage, (_p, eps) in STAGES.items():
                for method, path, _b, need in eps:
                    k = f"{role}|{method} {path}"
                    if RANK[role] < RANK[need]:
                        self.assertEqual(m[k], 403, f"{stage} {k}")
                        self.assertEqual(calls[k], 0, f"{k} 403 前不得调用 release")
                    else:
                        self.assertNotIn(m[k], (401, 403), f"{stage} {k}")
        # 发布阶段：仅 admin 过鉴权（fake 409），operator/engineer 一律 403
        for path in ("decision", "activate", "rollback"):
            k = f"/api/candidates/A/2/{path}"
            self.assertEqual(m[f"operator|post {k}"], 403)
            self.assertEqual(m[f"engineer|post {k}"], 403)
            self.assertEqual(m[f"admin|post {k}"], 409)
            self.assertEqual(calls[f"admin|post {k}"], 1)

    def test_publish_audit_bound_to_admin(self):
        _log(2, "发布写操作审计：仅 admin 产生 *_rejected 记录，低权限 403 不落审计")
        audit = [tuple(a) for a in self.res["audit"]]
        acts = {a for _u, a in audit}
        for a in ("candidate_decision_rejected", "candidate_activate_rejected",
                  "candidate_rollback_rejected", "candidate_recover_rejected"):
            self.assertIn(a, acts)
        cand = [(u, a) for u, a in audit if a.startswith("candidate_")]
        self.assertTrue(cand)
        self.assertTrue(all(u == "admin" for u, _a in cand), cand)


class RoleWorkspaceNavTest(unittest.TestCase):
    def test_role_workspace_nav(self):
        _log(3, "工作区导航：operator 仅检测/复核入口，engineer/admin 全量；_apply_role 幂等")
        from types import SimpleNamespace
        from PySide6.QtWidgets import QApplication, QLabel, QListWidget
        from ui import main_window as MW
        _app = QApplication.instance() or QApplication([])
        titles = [t for t, _i in MW.NAV_ITEMS]
        for stage, (page, _eps) in STAGES.items():
            self.assertIn(page, titles)
        self.assertIn("特征工程", titles)
        self.assertTrue(MW.role_allows_nav("engineer", "特征工程"))
        self.assertTrue(MW.role_allows_nav("admin", "特征工程"))
        self.assertFalse(MW.role_allows_nav("operator", "特征工程"))
        op_ok = {s for s, (p, _e) in STAGES.items() if MW.role_allows_nav("operator", p)}
        self.assertEqual(op_ok, {"检测", "复核"})
        for role in ("engineer", "admin"):
            self.assertTrue(all(MW.role_allows_nav(role, t) for t in titles))
        nav = QListWidget()
        for t in titles:
            nav.addItem(t)
        stub = SimpleNamespace(nav=nav, badge_role=QLabel(), _repolish=lambda w: None,
                               _title2row={t: i for i, t in enumerate(titles)},
                               _row2page={i: object() for i in range(len(titles))})
        nav.setCurrentRow(titles.index("模型管理"))
        MW.MainWindow._apply_role(stub, "operator")
        hidden = {t for i, t in enumerate(titles) if nav.item(i).isHidden()}
        self.assertEqual(set(titles) - hidden, MW.ROLE_NAV_VISIBLE["operator"])
        self.assertIn("模型管理", hidden)
        self.assertIn("评估看板", hidden)
        self.assertFalse(nav.item(nav.currentRow()).isHidden())   # 被隐藏页自动退回
        self.assertIn("操作员", stub.badge_role.text())
        MW.MainWindow._apply_role(stub, "admin")
        self.assertFalse(any(nav.item(i).isHidden() for i in range(len(titles))))


if __name__ == "__main__":
    if os.environ.get("AOI_FINAL_SIM_ROLES_CHILD"):
        _child(os.environ["AOI_FINAL_SIM_ROLES_CHILD"])
    else:
        unittest.main()
