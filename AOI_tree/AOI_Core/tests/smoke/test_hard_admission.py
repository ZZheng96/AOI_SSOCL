"""AC-017 难例池准入试算 smoke：合成 z 向量 + TEMP SQLite/TEMP 注册表，不读生产存储。"""
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from algo.feature_revision import FeatureSetRevision
from algo.features import registry as freg
from backend.db import database
from backend.db.models import Base, Detection, EvalRun, Feedback, Image
from backend.evaluation import hard_admission as ha

BENCH = {"eval_run_id": 1, "p95": 120.0, "budget_ms": 200}


def _trad(groups=("glcm", "edge"), n_bank=30, seed=0):
    rng = np.random.RandomState(seed)
    return SimpleNamespace(groups=list(groups), scoring="legacy", group_weight={},
                           group_scale={}, group_valid_dims={},
                           bank_w=rng.randn(n_bank, freg.TRAD_DIM))


def _pool(n_pos=8, n_neg=10, shift=("hog",), pos_src="miss", seed=1):
    """正样本在 shift 组维度上偏移；负样本 fp/train_normal 交替。"""
    rng = np.random.RandomState(seed)
    Z = rng.randn(n_pos + n_neg, freg.TRAD_DIM)
    ids = freg.dim_index(list(shift))
    Z[:n_pos, ids] += 4.0
    items = [{"path": f"p{i}", "label": 1, "sources": [pos_src], "drift": False}
             for i in range(n_pos)]
    items += [{"path": f"n{i}", "label": 0,
               "sources": ["fp" if i % 2 else "train_normal"], "drift": False}
              for i in range(n_neg)]
    return Z, items


def _manifest(groups, scoring="legacy", **kw):
    return {"enabled_groups": list(groups), "scoring": scoring, **kw}


class AdmissionMetricsTest(unittest.TestCase):
    def test_pass_and_fail_paths(self):
        print("[1/4] admission_metrics 通过/增益/冗余/G1/G2/open/planned/不足", flush=True)
        trad = _trad()
        Z, items = _pool()
        Za, ya = _pool(seed=7)[0], [1] * 8 + [0] * 10
        cand = _manifest(["glcm", "edge", "hog"])
        out = ha.admission_metrics(trad, Z, items, cand, Z_anchor=Za, anchor_labels=ya, bench=BENCH)
        self.assertEqual(out["acceptance"], "passed", out.get("reason"))
        self.assertEqual(out["added"], ["hog"])
        self.assertGreater(out["gain"]["delta"], ha.GAIN_MIN)
        self.assertLessEqual(out["redundancy"]["max_abs_spearman"]["hog"], ha.REDUNDANCY_MAX)
        self.assertEqual(out["regression"]["by_source"]["miss"]["status"], "passed")
        self.assertEqual(out["g1"]["status"], "passed")
        self.assertEqual(out["g2"]["status"], "passed")
        # 裁剪有效组 → 增益退化 failed
        trad3 = _trad(groups=("glcm", "edge", "hog"))
        out = ha.admission_metrics(trad3, Z, items, _manifest(["glcm", "edge"]),
                                   Z_anchor=Za, anchor_labels=ya, bench=BENCH)
        self.assertEqual(out["acceptance"], "failed")
        self.assertEqual(out["gain"]["status"], "failed")
        # 冗余超阈 → failed
        with patch.object(ha, "REDUNDANCY_MAX", -1.0):
            out = ha.admission_metrics(trad, Z, items, cand, Z_anchor=Za, anchor_labels=ya,
                                       bench=BENCH)
        self.assertEqual((out["acceptance"], out["redundancy"]["status"]), ("failed", "failed"))
        # 无基准 → G1 unknown → 不通过；无锚定集 → G2 skipped
        out = ha.admission_metrics(trad, Z, items, cand)
        self.assertEqual(out["acceptance"], "failed")
        self.assertEqual((out["g1"]["status"], out["g2"]["status"]), ("unknown", "skipped"))
        # G1 预估超预算
        out = ha.admission_metrics(trad, Z, items, cand, Z_anchor=Za, anchor_labels=ya,
                                   bench={"p95": 250.0, "budget_ms": 200})
        self.assertEqual(out["g1"]["status"], "failed")
        # open 正样本占比过高：分数再好也不准入
        Zo, io = _pool(pos_src="open_defect")
        out = ha.admission_metrics(trad, Zo, io, cand, Z_anchor=Za, anchor_labels=ya, bench=BENCH)
        self.assertEqual(out["acceptance"], "insufficient_evidence")
        self.assertEqual(out["open_share"], 1.0)
        # 自动发现/未注册组 → planned
        self.assertEqual(ha.admission_metrics(trad, Z, items, _manifest(["glcm", "zzz"]))
                         ["acceptance"], "planned")
        self.assertEqual(ha.admission_metrics(trad, Z, items, _manifest(
            ["glcm"], origin="auto_discovery"))["acceptance"], "planned")
        # 样本不足 / grouped 无组权重
        Zs, its = _pool(n_pos=3)
        self.assertEqual(ha.admission_metrics(trad, Zs, its, cand)["acceptance"],
                         "insufficient_evidence")
        self.assertEqual(ha.admission_metrics(trad, Z, items, _manifest(
            ["glcm"], scoring="grouped"))["acceptance"], "insufficient_evidence")

    def test_grouped_matches_trad_dist(self):
        print("[2/4] grouped 打分与 TradSlot._dist 同口径（跳最近邻 [1:4]）", flush=True)
        from algo.slots.trad import TradSlot
        from algo.eval.contribution import _knn_skip1
        slot = TradSlot.__new__(TradSlot)
        base = _trad(groups=("glcm", "edge"))
        slot.__dict__.update(vars(base))
        slot.scoring = "grouped"
        slot.group_weight = {"glcm": 0.7, "edge": 0.3}
        slot.group_scale = {"glcm": 2.0, "edge": 5.0}
        slot.group_valid_dims = {k: list(freg.dim_index([k])) for k in ("glcm", "edge")}
        Z, _ = _pool(n_pos=2, n_neg=3)
        sq = ha._group_sq(slot, Z, ["glcm", "edge"])
        mine = ha._combine(slot, sq, ["glcm", "edge"], "grouped", [])
        ref = _knn_skip1(np.stack([slot._dist(slot.bank_w, z) for z in Z]))
        np.testing.assert_allclose(mine, ref, rtol=1e-9, atol=1e-9)


class HardPoolAndGateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.db = create_engine(f"sqlite:///{self.root / 'ha.db'}",
                                connect_args={"check_same_thread": False})
        Base.metadata.create_all(self.db)
        factory = sessionmaker(bind=self.db, expire_on_commit=False)
        self._patches = [patch.object(database, "_engine", self.db),
                         patch.object(database, "_SessionLocal", factory),
                         patch.object(ha, "_store_dir", lambda: self.root)]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in reversed(self._patches):
            p.stop()
        self.db.dispose()
        self.tmp.cleanup()

    def _fb(self, s, path, ftype, label, invalidated=False, trace=None):
        d = Detection(image_path=path, category="c", final_score=0.5, is_anomaly=True,
                      latency_ms=1.0)
        s.add(d)
        s.flush()
        s.add(Feedback(detection_id=d.id, feedback_type=ftype, operator_label=label,
                       invalidated=invalidated, learning_trace=trace))

    def _seed(self, n_miss=2, n_fp=2):
        with database.session_scope() as s:
            s.add(Image(path="t.png", category="c", split="test", label="anomaly"))
            s.add(Image(path="r0.png", category="c", split="train", label="normal"))
            s.add(Image(path="r1.png", category="c", split="train", label="normal"))
            for i in range(n_miss):
                self._fb(s, f"m{i}.png", "false_negative", 1)
            for i in range(n_fp):
                self._fb(s, f"f{i}.png", "false_positive", 0,
                         trace={"engine_update": {"action": "rolled_back"}} if i == 0 else None)
            self._fb(s, "t.png", "false_negative", 1)                  # test 图：排除
            self._fb(s, "x.png", "false_negative", 1, invalidated=True)  # 作废：排除
            self._fb(s, "k.png", "false_negative", 1)                  # 同图矛盾 → conflict
            self._fb(s, "k.png", "false_positive", 0)

    def test_pool_sources_conflict_drift(self):
        print("[3/4] 难例池：来源/冲突/漂移/排除 test 与作废/train 参照", flush=True)
        self._seed()
        pool = ha.build_hard_pool("c")
        paths = {r["path"] for r in pool["items"]}
        self.assertEqual(paths, {"m0.png", "m1.png", "f0.png", "f1.png", "r0.png", "r1.png"})
        self.assertEqual([r["path"] for r in pool["conflicts"]], ["k.png"])
        c = pool["counts"]
        self.assertEqual((c["pos"], c["neg"], c["miss"], c["fp"], c["train_normal"],
                          c["conflict"], c["drift"]), (2, 4, 2, 2, 2, 1, 1))
        self.assertEqual(pool["sha256"], ha.build_hard_pool("c")["sha256"])

    def test_open_cause_counted(self):
        print("[AC-034] open 原因归类：drift 标漂移、各原因单列计数", flush=True)
        with database.session_scope() as s:
            self._fb(s, "o0.png", "open_alert", 0, trace={"open_cause": "drift"})
            self._fb(s, "o1.png", "open_alert", 1, trace={"open_cause": "new_defect"})
            self._fb(s, "o2.png", "open_alert", 0, trace={"open_cause": "noise"})
        pool = ha.build_hard_pool("c")
        c = pool["counts"]
        self.assertEqual(c["open_causes"], {"drift": 1, "new_defect": 1, "noise": 1})
        self.assertEqual((c["open_defect"], c["open_normal"], c["drift"]), (1, 2, 1))
        drift = {r["path"]: r["drift"] for r in pool["items"]}
        self.assertTrue(drift["o0.png"])
        self.assertFalse(drift["o2.png"])

    def test_evaluate_and_revision_gate(self):
        print("[4/4] 试算端到端 → 证据 JSON → revision 门禁（未通过拒批/通过可批）", flush=True)
        self._seed(n_miss=6, n_fp=6)
        trad = _trad()
        trad._z = lambda f, norm: f
        trad.norm_w = None
        hog = freg.dim_index(["hog"])
        rng = np.random.RandomState(3)

        def _vec(img):
            v = rng.randn(freg.TRAD_DIM)
            if img.startswith("m") or img.startswith("a"):
                v[hog] += 4.0
            return v
        trad._vec = _vec
        pipe = SimpleNamespace(slots={"trad": trad}, _preprocess=lambda img: img)
        eng = SimpleNamespace(list_versions=lambda c: [1], build_variant=lambda c, v: pipe)
        anchors = [(f"n{i}", 0) for i in range(6)] + [(f"a{i}", 1) for i in range(6)]
        store = FeatureSetRevision(self.root / "registry.json")
        base = {"schema_version": 1, "groups": [{"key": "a", "dim": 2, "slice": [0, 2],
                                                 "params": {}}], "dimension": 2,
                "preprocess": {"max_side": 256}, "implementation_hashes": {"t": "a" * 64},
                "seed": 0, "dataset_split": {"category": "c"}, "category": "c",
                "model_snapshot": {"category": "c", "version": 1,
                                   "artifact_hashes": {"m": "c" * 64}},
                "admission_required": True, "scoring": "legacy"}
        good = store.register({**deepcopy(base), "enabled_groups": ["glcm", "edge", "hog"]})
        bad = store.register({**deepcopy(base), "enabled_groups": ["glcm", "edge", "hog"],
                              "note": "nobench"})
        with patch("backend.engine.get_engine", return_value=eng), \
                patch("algo.common.io.load_image", side_effect=lambda p: Path(p).stem), \
                patch("backend.evaluation.service._gate_anchor_items", return_value=anchors):
            with patch.object(ha, "_latest_benchmark", return_value=None):
                r_bad = ha.evaluate_feature_admission("c", bad)
            with patch.object(ha, "_latest_benchmark", return_value=BENCH):
                prog = []
                r_ok = ha.evaluate_feature_admission("c", good,
                                                     progress_cb=lambda *a: prog.append(a))
        self.assertEqual(r_bad["acceptance"], "failed")
        self.assertEqual(r_ok["acceptance"], "passed", r_ok["admission"].get("reason"))
        self.assertEqual(prog[-1][0], prog[-1][1])
        with database.session_scope() as s:
            self.assertEqual(s.query(EvalRun).filter_by(run_type="feature_admission").count(), 2)
        ev_bad, ev_ok = self.root / r_bad["evidence_path"], self.root / r_ok["evidence_path"]
        self.assertTrue(json.loads(ev_ok.read_text(encoding="utf-8"))["independent"])
        # 未通过证据：可进入 evaluated，但拒绝 approved
        store.transition(bad, "evaluated", actor="t", evidence_path=ev_bad)
        with self.assertRaises(ValueError):
            store.transition(bad, "approved", actor="t")
        # 通过证据：evaluated → approved；生产未被激活
        store.transition(good, "evaluated", actor="t", evidence_path=ev_ok)
        self.assertEqual(store.transition(good, "approved", actor="t")["state"], "approved")
        # 篡改 admission 子项 → 拒批（总 acceptance=passed 也不行）
        data = json.loads(ev_ok.read_text(encoding="utf-8"))
        data["admission"]["g2"]["status"] = "skipped"
        tam = self.root / "tampered.json"
        tam.write_text(json.dumps(data), encoding="utf-8")
        third = store.register({**deepcopy(base), "enabled_groups": ["glcm", "hog"]})
        data["revision_id"] = third
        tam.write_text(json.dumps(data), encoding="utf-8")
        store.transition(third, "evaluated", actor="t", evidence_path=tam)
        with self.assertRaisesRegex(ValueError, "g2"):
            store.transition(third, "approved", actor="t")


if __name__ == "__main__":
    unittest.main()
