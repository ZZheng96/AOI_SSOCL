"""SYNC-14 复查：degraded_slots 是否出现在响应 decision_trace 中。"""
import json, os, random, shutil, sys, tempfile
from pathlib import Path

import yaml

ROOT = Path(r"d:\CGAIC\AOI_SSOCL\stable_AOI_tree\AOI_Core")
ARCHIVE = ROOT / "storage" / "archive" / "gold_finger" / "20261011"

outdir = Path(tempfile.mkdtemp(prefix="aoi_fixsnap2_"))
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

svc = get_detection_service()
imgs = sorted(p for p in ARCHIVE.iterdir()
              if p.suffix.lower() == ".png" and "bench" not in p.name)[:2]
for p in imgs:
    r = svc.detect_image(str(p), "gold_finger", persist=True, with_heatmap=True)
    trace = r.get("decision_trace") or []
    hit = [t for t in trace if "degraded" in json.dumps(t, ensure_ascii=False, default=str)]
    print(p.name, r["decision"], round(r["final_score"], 4),
          "trace_len", len(trace), "degraded_hits", len(hit))
    if hit:
        print(json.dumps(hit[0], ensure_ascii=False, default=str)[:400])
