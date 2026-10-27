"""REMAIN-02：搭建 TEMP 隔离运行环境（独立 DB/storage/端口 8031）。

- 不修改稳定版源码；仅从稳定版 storage 只读复制 gold_finger 快照到 TEMP。
- 生成合成测试图并登记进 TEMP DB（train normal×8 / train_anomaly×2 /
  test normal×4 / test anomaly×4），供 feature-set report/rebuild 使用。
"""
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

ROOT = Path(r"d:\CGAIC\AOI_SSOCL")
STABLE = ROOT / "stable_AOI_tree" / "AOI_Core"
TASK = ROOT / "review" / "task" / "remain_02"
TEMP = TASK / "temp_core"
CAT = "gold_finger"

# ── 1. 目录与配置 ────────────────────────────────────────────
for d in (TEMP, TEMP / "storage", TEMP / "database", TASK / "raw"):
    Path(d).mkdir(parents=True, exist_ok=True)

cfg = f"""system:
  host: 127.0.0.1
  port: 8031
  storage_dir: {TEMP.as_posix()}/storage
  engine_storage: {TEMP.as_posix()}/storage/engine
  engine_base_cfg: {TEMP.as_posix()}/engine_fast.yaml
  engine_max_cached_categories: 4
  database_url: sqlite:///{(TEMP / 'database' / 'aoi.db').as_posix()}
data:
  default_root: {TEMP.as_posix()}/data
pipeline:
  tile_size: 512
  tile_stride: 512
  input_size: 256
  ahl_input_size: 224
  full_image_max_area: 1048576
  fusion_weights:
    1: 0.15
    2: 0.25
    3: 0.6
  l1_batch_size: 8
  overlay_max_dim: 1280
  video_frame_interval: 5
  latency_budget_ms: 200
pseudo_anomaly:
  default_method: cutpaste
  default_count: 20
self_learning:
  trigger_min_feedback: 10
  lr: 1.0e-05
  epochs: 10
  auto_activate: false
  learning_parallel: false
evaluation:
  default_split: test
  benchmark_n_images: 30
  benchmark_warmup: 5
notify:
  webhook_url: ''
  events: [anomaly]
  timeout_s: 3
  alarm_window: 20
  alarm_k: 5
  alarm_cooldown_s: 60.0
security:
  api_key: ''
  api_key_env: ''
  keys:
    remain02-operator-key: operator
    remain02-engineer-key: engineer
    remain02-admin-key: admin
  max_upload_mb: 200
logging:
  max_mb: 10
  backups: 5
archive:
  enabled: false
  decisions: [anomaly, gray]
  keep_original: true
plc:
  trigger_dir: ''
  exts: [.png, .jpg, .jpeg, .bmp]
governance_decisions:
  candidate_release_mode: manual
  dependency_lock_policy: warn
  gt_threshold_frozen: false
"""
(TEMP / "config.yaml").write_text(cfg, encoding="utf-8")
shutil.copy2(STABLE / "configs" / "engine_fast.yaml", TEMP / "engine_fast.yaml")
print("[1/4] config.yaml / engine_fast.yaml 就绪", flush=True)

# ── 2. 只读复制快照（不动稳定版文件本身）────────────────────
src_snap = STABLE / "storage" / "engine" / "snapshots" / CAT
dst_snap = TEMP / "storage" / "engine" / "snapshots" / CAT
if dst_snap.exists():
    shutil.rmtree(dst_snap)
shutil.copytree(src_snap, dst_snap)
print(f"[2/4] 快照复制完成: {src_snap} -> {dst_snap}", flush=True)

# ── 3. 合成测试图 ───────────────────────────────────────────
img_root = TEMP / "storage" / "seed_images" / CAT
rng = np.random.default_rng(7)


def make_img(anomaly: bool, seed: int) -> np.ndarray:
    r = np.random.default_rng(seed)
    base = 180 + 20 * r.standard_normal((512, 512, 3))
    base = np.clip(base, 0, 255)
    # 加纹理，避免纯色图特征退化
    yy, xx = np.mgrid[0:512, 0:512]
    base += (15 * np.sin(xx / 9.0 + seed))[..., None]
    if anomaly:
        base[200:280, 200:280] = 30  # 明显缺陷块
    return np.clip(base, 0, 255).astype(np.uint8)


seeds = []
spec = [("train", "normal", 8), ("train_anomaly", "anomaly", 2),
        ("test", "normal", 4), ("test", "anomaly", 4)]
k = 0
for split, label, n in spec:
    for i in range(n):
        arr = make_img(label == "anomaly", seed=100 + k)
        p = img_root / split / f"{label}_{i:02d}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        PILImage.fromarray(arr).save(p)
        seeds.append((p, split, label))
        k += 1
print(f"[3/4] 合成图 {len(seeds)} 张 -> {img_root}", flush=True)

# ── 4. 登记 TEMP DB ─────────────────────────────────────────
os.environ["AOI_CONFIG"] = str(TEMP / "config.yaml")
sys.path.insert(0, str(STABLE))
os.chdir(STABLE)
from backend.db.database import session_scope  # noqa: E402
from backend.db.models import Image  # noqa: E402

with session_scope() as s:
    for p, split, label in seeds:
        s.add(Image(path=str(p), category=CAT, split=split, label=label,
                    source="remain02_seed", width=512, height=512,
                    sha1=hashlib.sha1(p.read_bytes()).hexdigest()))
print("[4/4] TEMP DB 登记完成", flush=True)
print("SETUP_OK", flush=True)
