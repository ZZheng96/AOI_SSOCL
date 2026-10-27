"""第四轮：approved_extra_good_batches 在重建流程中生效（防泄漏 fit/eval 划分）。

- split_extra_good：seed=0 确定性、fit/eval 互不相交、与输入顺序无关、只含登记批次、floor 比例
- _prepare_engine（DB 路径）：默认空登记 bundle 不变；登记后 fit 子集并入 init_normal、
  从 val 剔除，eval 子集与缺陷图不进入拟合
- run_accuracy_eval：登记后剔除 fit 子集；默认空登记不剔除
全部用 TEMP 下合成小文件 + TEMP SQLite + 假 engine/pipe，不碰生产 DB/快照/配置。"""
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from algo.common.isolation import guard_bundle
from algo.eval import coverage as COV
from backend.api import routes_models as RM
from backend.api.schemas import ModelPrepareRequest
from backend.core import config as C
from backend.db import database
from backend.db.models import Base, Image
from backend.evaluation import service as SV

N = 6
_T0 = time.perf_counter()


def _log(i, msg):
    print(f"[{i}/{N}] {msg}（+{time.perf_counter() - _T0:.2f}s）", flush=True)


def _tmp_settings(tc, extra_yaml: str = "") -> C.Settings:
    tmp = tempfile.TemporaryDirectory(prefix="aoi_round4_rb_")
    tc.addCleanup(tmp.cleanup)
    p = Path(tmp.name) / "config.yaml"
    sd = (Path(tmp.name) / "storage").as_posix()
    p.write_text(f"system:\n  storage_dir: '{sd}'\n" + extra_yaml, encoding="utf-8")
    return C.Settings(p)


APPROVED = "governance_decisions:\n  approved_extra_good_batches: Image_2026, t_fov\n"
IMG = [f"/d/test/good/Image_20260116_{100 + i}_repaired.png" for i in range(12)]
TFOV = [f"/d/test/good/t_fov_{i}_repaired.png" for i in (3, 6, 13, 14, 26)]
OTHER = ["/d/test/good/0700_repaired.png", "/d/test/good/0701_repaired.png"]


class SplitTest(unittest.TestCase):
    def test_split(self):
        _log(1, "split_extra_good：确定性/不相交/顺序无关/只含登记批次/floor 比例")
        pool = IMG + TFOV + OTHER
        a = COV.split_extra_good(pool, ["Image_2026", "t_fov"], seed=0)
        b = COV.split_extra_good(list(reversed(pool)), ["Image_2026", "t_fov"], seed=0)
        self.assertEqual(a, b)
        self.assertFalse(set(a["fit"]) & set(a["eval"]))
        self.assertEqual(set(a["fit"]) | set(a["eval"]), set(IMG + TFOV))
        fit_img = [p for p in a["fit"] if COV.infer_batch(p) == "Image_2026"]
        fit_tf = [p for p in a["fit"] if COV.infer_batch(p) == "t_fov"]
        self.assertEqual((len(fit_img), len(fit_tf)), (6, 2))
        self.assertEqual(COV.split_extra_good(pool, [], seed=0), {"fit": [], "eval": []})
        self.assertEqual(COV.split_extra_good(pool, ["nope"], seed=0), {"fit": [], "eval": []})
        # 不同 seed 只是不同划分，仍不相交（seed 固定为 0 由调用方保证）
        c = COV.split_extra_good(pool, ["Image_2026"], seed=1)
        self.assertFalse(set(c["fit"]) & set(c["eval"]))


class _DBCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="aoi_round4_db_")
        self.root = Path(self.tmp.name)
        self.db = create_engine(f"sqlite:///{self.root / 'r4.db'}",
                                connect_args={"check_same_thread": False})
        Base.metadata.create_all(self.db)
        factory = sessionmaker(bind=self.db, expire_on_commit=False)
        self._patches = [patch.object(database, "_engine", self.db),
                         patch.object(database, "_SessionLocal", factory)]
        for p in self._patches:
            p.start()
        # 真实小文件（内容各不相同），guard_bundle 需要按内容哈希校验
        def mk(p):
            f = self.root / p.lstrip("/")
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(p.encode("utf-8"))
            return str(f)

        self.train_n = [mk(f"/d/train/good/{600 + i:04d}_repaired.png") for i in range(4)]
        self.train_a = [mk(f"/d/train_anomaly/{600 + i:04d}.png") for i in range(3)]
        self.img, self.tfov = [mk(p) for p in IMG], [mk(p) for p in TFOV]
        self.pool = self.img + self.tfov
        self.test_defects = [mk("/d/test/defect/Image_20260116_100.png"),
                             mk("/d/test/defect/1308.jpg")]
        self.sp = COV.split_extra_good(self.pool, ["Image_2026", "t_fov"], seed=0)
        with database.session_scope() as s:
            for p in self.train_n:
                s.add(Image(path=p, category="c", split="train", label="normal"))
            for p in self.train_a:
                s.add(Image(path=p, category="c", split="train_anomaly", label="anomaly"))
            for p in self.pool:
                s.add(Image(path=p, category="c", split="test", label="normal"))
            for p in self.test_defects:
                s.add(Image(path=p, category="c", split="test", label="anomaly"))

    def tearDown(self):
        for p in reversed(self._patches):
            p.stop()
        self.db.dispose()
        self.tmp.cleanup()


class PrepareTest(_DBCase):
    def _bundle(self, settings):
        calls = {}
        eng = SimpleNamespace(prepare=lambda cat, bundle, **kw: calls.update(bundle) or 1,
                              current_version=lambda c: None)
        with patch.object(RM, "get_settings", lambda: settings), \
                patch("backend.engine.get_engine", return_value=eng), \
                patch.object(RM, "register_engine_snapshots"), \
                patch.object(RM, "_auto_feature_report", return_value={}):
            RM._prepare_engine(ModelPrepareRequest(category="c"), lambda *a: None)
        return calls

    def test_default_unchanged(self):
        _log(2, "_prepare_engine：默认空登记 bundle 与改动前口径一致")
        settings = _tmp_settings(self)
        b = self._bundle(settings)
        self.assertEqual(b["init_normal"], self.train_n)
        self.assertEqual(len(b["init_defect"]), 3)
        val_paths = [p for p, _ in b["val"]]
        self.assertEqual(set(val_paths), set(self.pool + self.test_defects))
        self.assertFalse((settings.storage("images") / "c" / "approved_extra").exists())
        self.assertTrue(guard_bundle(b, {"n_init_defect": 30}))

    def test_approved_merges_fit(self):
        _log(3, "_prepare_engine：登记后 fit 副本并入 init_normal 且过 guard_bundle、"
                "从 val 剔除，缺陷/eval 不进入拟合")
        settings = _tmp_settings(self, APPROVED)
        b = self._bundle(settings)
        sp = self.sp
        self.assertEqual(b["init_normal"][:4], self.train_n)
        staged = b["init_normal"][4:]
        self.assertEqual(len(staged), len(sp["fit"]))
        stage_dir = settings.storage("images") / "c" / "approved_extra" / "train" / "good"
        for src, dst in zip(sp["fit"], staged):
            self.assertEqual(Path(dst).parent, stage_dir)
            self.assertTrue(Path(dst).name.startswith(Path(src).stem))
            self.assertEqual(COV.infer_batch(dst), COV.infer_batch(src))
            self.assertEqual(Path(dst).read_bytes(), Path(src).read_bytes())
        self.assertFalse(set(b["init_normal"]) & set(sp["eval"]))
        self.assertFalse(set(b["init_normal"]) & set(self.test_defects))
        self.assertEqual(len(b["init_defect"]), 3)
        val_paths = {p for p, _ in b["val"]}
        self.assertFalse(val_paths & set(sp["fit"]))
        self.assertTrue(set(sp["eval"]) <= val_paths)
        # 生产 test 正常图路径含 test 段，直接并入会被红线拒绝；staging 后应通过
        self.assertTrue(guard_bundle(b, {"n_init_defect": 30}))
        # 重复准备幂等：不重复复制
        b2 = self._bundle(settings)
        self.assertEqual(b2["init_normal"], b["init_normal"])


class EvalTest(_DBCase):
    def _run(self, settings):
        seen = []

        def _predict(path):
            seen.append(path)
            return {"fused": 1.0 if "defect" in path else 0.1,
                    "decision": "anomaly" if "defect" in path else "normal"}

        pipe = SimpleNamespace(predict=_predict)
        eng = SimpleNamespace(list_versions=lambda c: [1], current_version=lambda c: 1,
                              build_variant=lambda c, v: pipe, snap_root=str(self.root))
        with patch.object(SV, "get_settings", lambda: settings), \
                patch("backend.engine.get_engine", return_value=eng):
            out = SV.run_accuracy_eval("c")
        return out, seen

    def test_default_no_exclusion(self):
        _log(4, "run_accuracy_eval：默认空登记不剔除任何 test 图")
        out, seen = self._run(_tmp_settings(self))
        self.assertEqual(set(seen), set(self.pool + self.test_defects))
        self.assertEqual(out["n_images"], len(self.pool) + len(self.test_defects))

    def test_approved_excludes_fit(self):
        _log(5, "run_accuracy_eval：登记后剔除 fit 子集，仅评 eval 子集 + 缺陷")
        out, seen = self._run(_tmp_settings(self, APPROVED))
        sp = self.sp
        self.assertEqual(set(seen), set(sp["eval"]) | set(self.test_defects))
        self.assertEqual(out["metrics"]["auroc"], 1.0)

    def test_limit_does_not_change_split(self):
        _log(6, "run_accuracy_eval：limit 截断不改变划分池（仍按全部 test 正常图划分）")
        sp = self.sp
        settings = _tmp_settings(self, APPROVED)
        seen2 = []
        pipe = SimpleNamespace(predict=lambda p: seen2.append(p) or {
            "fused": 0.5, "decision": "normal"})
        eng = SimpleNamespace(list_versions=lambda c: [1], current_version=lambda c: 1,
                              build_variant=lambda c, v: pipe, snap_root=str(self.root))
        with patch.object(SV, "get_settings", lambda: settings), \
                patch("backend.engine.get_engine", return_value=eng):
            SV.run_accuracy_eval("c", limit=8)
        self.assertFalse(set(seen2) & set(sp["fit"]))


if __name__ == "__main__":
    unittest.main()
