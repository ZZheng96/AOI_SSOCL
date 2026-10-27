"""AC-078 模拟：合成视频 3 品类 × 3 种子走现有 iter_video_detections。

模拟口径（非实机证据）：
- 每品类一种合成纹理（fabric 噪声 / metal 渐变拉丝 / pcb 网格），逐帧轻微平移+时序噪声；
  已知帧注入强对比(+120)与低对比(+45)亮斑缺陷，MJPG avi 落 TEMP。
- 假引擎按局部残差打分；阈值只在独立 dev 种子(900)的正常帧上冻结（max×1.05），
  测试种子(1,2,3)只做评估，不回调阈值。
- 按域报 recall(强/弱)/FPR/帧级时延/最差域；另走一路 persist=True 到临时 sqlite，
  核 Video/VideoFrame/Detection.video_id/frame_idx/image_id 资产时序。
G2 业务 FPR/召回数字未冻结，本测试的断言只是合成数据健全性，不是业务门槛。
"""
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.db import database
from backend.db.models import Base, Detection, Video, VideoFrame
from backend.engine import DetectionResult
from backend.pipeline import service as module

CATS = ("fabric", "metal", "pcb")
DEV_SEED = 900
TEST_SEEDS = (1, 2, 3)
N_FRAMES = 24
H, W = 96, 128
STRONG = {5, 13, 20}          # 强对比缺陷帧
WEAK = {9, 17}                # 低对比缺陷帧
RAW_DIR = Path(r"d:\CGAIC\AOI_SSOCL\review\task\final_sim\raw")


def _texture(cat, rng):
    yy, xx = np.mgrid[0:H + 16, 0:W + 16].astype(np.float32)
    if cat == "fabric":
        base = 128 + rng.normal(0, 10, yy.shape)
        base = cv2.GaussianBlur(base.astype(np.float32), (0, 0), 1.0)
    elif cat == "metal":
        base = 90 + 0.6 * xx + rng.normal(0, 3, (1, xx.shape[1])) * 4
    else:
        base = np.full(yy.shape, 60.0, np.float32)
        base[(yy.astype(int) % 16) < 2] = 170
        base[(xx.astype(int) % 16) < 2] = 170
    return np.clip(base, 0, 255).astype(np.float32)


def _make_video(path, cat, seed):
    rng = np.random.default_rng(seed * 31 + CATS.index(cat))
    tex = _texture(cat, rng)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (W, H))
    truth = {}
    for i in range(N_FRAMES):
        dy, dx = int(rng.integers(0, 8)), int(rng.integers(0, 8))
        g = tex[dy:dy + H, dx:dx + W] + rng.normal(0, 2.5, (H, W))
        kind = "strong" if i in STRONG else "weak" if i in WEAK else "normal"
        if kind != "normal":
            cy, cx = int(rng.integers(12, H - 12)), int(rng.integers(12, W - 12))
            g[cy - 3:cy + 3, cx - 3:cx + 3] += 120 if kind == "strong" else 45
        truth[i] = kind
        g = np.clip(g, 0, 255).astype(np.uint8)
        vw.write(cv2.merge([g, g, g]))
    vw.release()
    return truth


class _ResidualEngine:
    """局部残差打分：|g - blur(g)| 经 3×3 均值后取最大值。"""

    def __init__(self):
        self.thresholds = {}

    @staticmethod
    def score(rgb):
        g = rgb.mean(axis=2).astype(np.float32)
        r = np.abs(g - cv2.GaussianBlur(g, (0, 0), 3.0))
        return float(cv2.blur(r, (3, 3)).max()) / 255.0

    def predict_ndarray(self, category, rgb, path_hint=None):
        s = self.score(rgb)
        thr = self.thresholds.get(category)
        anomaly = thr is not None and s >= thr
        return DetectionResult(score=s, decision="anomaly" if anomaly else "normal",
                               is_anomaly=bool(anomaly),
                               threshold=float(thr if thr is not None else 1.0),
                               model_version=1, slot_scores={"resid": s})


def _p(q, xs):
    return float(np.percentile(np.asarray(xs, float), q)) if xs else None


class FinalSimVideoTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="aoi_final_sim_video_")
        cls.root = Path(cls._tmp.name)
        for key in ("uploads", "archive", "heatmaps"):
            (cls.root / key).mkdir()
        cls.db = create_engine("sqlite:///" + (cls.root / "t.db").as_posix())
        Base.metadata.create_all(cls.db)
        factory = sessionmaker(bind=cls.db, expire_on_commit=False)
        cls.engine = _ResidualEngine()
        cls.svc = module.DetectionService.__new__(module.DetectionService)
        cls.svc.settings = SimpleNamespace(
            get=lambda sec, key, default=None: {"video_frame_interval": 1}.get(key, default),
            section=lambda key: {"enabled": False},
            storage=lambda key: cls.root / key)
        cls.patches = [patch.object(database, "_engine", cls.db),
                       patch.object(database, "_SessionLocal", factory),
                       patch.object(module, "get_engine", return_value=cls.engine),
                       patch.object(module, "notify_event"),
                       patch.object(cls.svc, "_record_alarm",
                                    return_value={"active": False, "window_hits": 0})]
        for p in cls.patches:
            p.start()
        cls.videos = {}
        for cat in CATS:
            for seed in (DEV_SEED,) + TEST_SEEDS:
                path = cls.root / f"{cat}_s{seed}.avi"
                cls.videos[(cat, seed)] = (path, _make_video(path, cat, seed))

    @classmethod
    def tearDownClass(cls):
        for p in reversed(cls.patches):
            p.stop()
        cls.db.dispose()
        cls._tmp.cleanup()

    def _run(self, cat, seed, **kw):
        path, truth = self.videos[(cat, seed)]
        out, lat = [], []
        it = self.svc.iter_video_detections(path, cat, with_heatmap=False, **kw)
        while True:
            t0 = time.perf_counter()
            try:
                det = next(it)
            except StopIteration:
                break
            lat.append((time.perf_counter() - t0) * 1000)
            out.append(det)
        return out, lat, truth

    def test_1_three_cats_three_seeds_dev_frozen(self):
        # 阶段一：dev 种子只用正常帧冻结阈值
        frozen = {}
        for cat in CATS:
            dets, _, truth = self._run(cat, DEV_SEED, interval=1, persist=False)
            normal = [d["final_score"] for d in dets if truth[d["frame_idx"]] == "normal"]
            frozen[cat] = round(max(normal) * 1.05, 6)
        self.engine.thresholds = dict(frozen)
        print(f"\n[AC-078] dev 冻结阈值 {frozen}", file=sys.stderr, flush=True)

        # 阶段二：测试种子评估（不再改阈值）
        domains, all_lat = {}, []
        for cat in CATS:
            per = {"tp_strong": 0, "n_strong": 0, "tp_weak": 0, "n_weak": 0,
                   "fp": 0, "n_normal": 0, "lat_ms": []}
            for seed in TEST_SEEDS:
                dets, lat, truth = self._run(cat, seed, interval=1, persist=False)
                self.assertEqual([d["frame_idx"] for d in dets], list(range(N_FRAMES)))
                self.assertEqual([d["sampled_idx"] for d in dets], list(range(N_FRAMES)))
                for d in dets:
                    self.assertAlmostEqual(d["threshold"], frozen[cat], places=5)
                    kind, hit = truth[d["frame_idx"]], d["decision"] == "anomaly"
                    if kind == "normal":
                        per["n_normal"] += 1
                        per["fp"] += hit
                    else:
                        per[f"n_{kind}"] += 1
                        per[f"tp_{kind}"] += hit
                per["lat_ms"] += lat
                print(f"[AC-078] {cat} seed={seed} 帧={len(dets)} "
                      f"p95={_p(95, lat):.2f}ms", file=sys.stderr, flush=True)
            all_lat += per["lat_ms"]
            domains[cat] = {
                "recall_strong": per["tp_strong"] / per["n_strong"],
                "recall_weak": per["tp_weak"] / per["n_weak"],
                "fpr": per["fp"] / per["n_normal"],
                "n_normal": per["n_normal"], "n_strong": per["n_strong"],
                "n_weak": per["n_weak"],
                "lat_p50_ms": _p(50, per["lat_ms"]), "lat_p95_ms": _p(95, per["lat_ms"]),
            }
        worst_fpr = max(domains, key=lambda c: domains[c]["fpr"])
        worst_rec = min(domains, key=lambda c: domains[c]["recall_strong"])
        report = {
            "ac": "AC-078", "simulation": "synthetic MJPG video + residual fake engine",
            "not_production_evidence": True, "dev_seed": DEV_SEED, "test_seeds": TEST_SEEDS,
            "frozen_threshold_rule": "max(dev normal score)*1.05, per category",
            "frozen_thresholds": frozen, "domains": domains,
            "worst_domain_fpr": [worst_fpr, domains[worst_fpr]["fpr"]],
            "worst_domain_recall_strong": [worst_rec, domains[worst_rec]["recall_strong"]],
            "frame_latency_ms": {"p50": _p(50, all_lat), "p95": _p(95, all_lat),
                                 "max": max(all_lat), "n": len(all_lat),
                                 "note": "fake engine, 128x96, 非 G1 口径"},
        }
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        (RAW_DIR / "ac078_video_domains.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[AC-078] {json.dumps(domains, ensure_ascii=False)}", file=sys.stderr, flush=True)
        # 合成健全性（非业务门槛，不按测试结果回调阈值）：强对比多数检出、
        # 强≥弱、FPR 有界；pcb 网格上亮斑饱和截断导致的漏检如实保留为最差域
        for cat, d in domains.items():
            self.assertGreaterEqual(d["recall_strong"], 0.8, cat)
            self.assertGreaterEqual(d["recall_strong"], d["recall_weak"], cat)
            self.assertLessEqual(d["fpr"], 0.2, cat)

    def test_2_interval_sampling(self):
        self.engine.thresholds = {}
        dets, _, _ = self._run("pcb", 1, interval=3, persist=False)
        self.assertEqual([d["frame_idx"] for d in dets], list(range(0, N_FRAMES, 3)))
        self.assertEqual([d["sampled_idx"] for d in dets], list(range(len(dets))))

    def test_3_persist_video_assets_and_timeline(self):
        self.engine.thresholds = {"metal": 0.0}       # 全判 anomaly，验证落库链路
        path, _ = self.videos[("metal", 2)]
        with database.session_scope() as s:
            v = Video(path=str(path), category="metal", n_frames=N_FRAMES, width=W, height=H)
            s.add(v)
            s.flush()
            vid = v.id
        dets, _, _ = self._run("metal", 2, interval=4, video_id=vid, persist=True)
        idxs = list(range(0, N_FRAMES, 4))
        self.assertEqual([d["frame_idx"] for d in dets], idxs)
        with database.session_scope() as s:
            frames = (s.query(VideoFrame).filter(VideoFrame.video_id == vid)
                      .order_by(VideoFrame.frame_idx).all())
            self.assertEqual([f.frame_idx for f in frames], idxs)
            ts = [f.timestamp_ms for f in frames]
            self.assertEqual(ts, sorted(ts))
            img_by_idx = {f.frame_idx: f.image_id for f in frames}
            for d in dets:
                row = s.get(Detection, d["detection_id"])
                self.assertEqual((row.video_id, row.frame_idx, row.target_type),
                                 (vid, d["frame_idx"], "video_frame"))
                self.assertIsNotNone(row.image_id)
                self.assertEqual(row.image_id, img_by_idx[d["frame_idx"]])
                self.assertTrue(Path(row.image_path).is_file())
        # 重跑同视频：VideoFrame 幂等不重复
        self._run("metal", 2, interval=4, video_id=vid, persist=True)
        with database.session_scope() as s:
            self.assertEqual(s.query(VideoFrame).filter(VideoFrame.video_id == vid).count(),
                             len(idxs))


if __name__ == "__main__":
    unittest.main()
