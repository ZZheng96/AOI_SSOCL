"""SYNC-14 复验：稳定版代码加载真实生产快照 gold_finger v2（TEMP 副本），验证降级模式。"""
import json, os, random, shutil, sys, tempfile, time
from pathlib import Path

import yaml

ROOT = Path(r"d:\CGAIC\AOI_SSOCL\stable_AOI_tree\AOI_Core")
TASK = Path(r"d:\CGAIC\AOI_SSOCL\review\task\test01_e2e_latency")
ARCHIVE = ROOT / "storage" / "archive" / "gold_finger" / "20261011"

outdir = Path(tempfile.mkdtemp(prefix="aoi_fixsnap_"))
shutil.copytree(ROOT / "storage" / "engine" / "snapshots" / "gold_finger",
                outdir / "engine" / "snapshots" / "gold_finger")
cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
cfg["system"].update(storage_dir=str(outdir / "storage"),
                     engine_storage=str(outdir / "engine"),
                     engine_base_cfg=str(ROOT / "configs" / "engine_fast.yaml"),
                     database_url="sqlite:///" + (outdir / "bench.db").as_posix(),
                     host="127.0.0.1")
cfg.setdefault("notify", {})["webhook_url"] = ""
cfgp = outdir / "config.yaml"
cfgp.write_text(yaml.safe_dump(cfg), encoding="utf-8")
os.environ["AOI_CONFIG"] = str(cfgp)
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from backend.pipeline.service import get_detection_service
from backend.engine import get_engine

engine = get_engine()
service = get_detection_service()

t0 = time.perf_counter()
engine._ensure_loaded("gold_finger")
load_s = time.perf_counter() - t0
entry = engine._pipes.get("gold_finger")
pipe = entry[0]
degraded = getattr(pipe, "degraded_slots", None)
print(f"[load] real v2 snapshot loaded in {load_s:.2f}s")
print(f"[degraded_slots] {json.dumps(degraded, ensure_ascii=False, default=str)}")

from PIL import Image
pool = []
for p in sorted(ARCHIVE.iterdir()):
    if p.suffix.lower() in (".jpg", ".png"):
        with Image.open(p) as im:
            if im.size == (2500, 2500):
                pool.append(p)
random.seed(99)
random.shuffle(pool)
rows = []
for p in pool[:5]:
    t1 = time.perf_counter()
    r = service.detect_image(str(p), "gold_finger", persist=True, with_heatmap=True)
    rows.append({"image": p.name, "wall_ms": round((time.perf_counter() - t1) * 1000, 1),
                 "decision": r["decision"], "score": round(r["final_score"], 4),
                 "model_version": r["model_version"],
                 "degraded_in_trace": "degraded_slots" in json.dumps(
                     r.get("learning_trace") or {}, ensure_ascii=False)})
    print(f"  {p.name}: {rows[-1]['wall_ms']}ms {r['decision']} v{r['model_version']}",
          flush=True)

out = {"temp": str(outdir), "load_s": round(load_s, 3), "degraded_slots": degraded,
       "inference": rows}
(TASK / "raw" / "fixsnap_real_v2.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
