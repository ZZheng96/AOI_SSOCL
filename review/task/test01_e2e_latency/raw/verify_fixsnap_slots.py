"""SYNC-14 复查 2：降级模式下不同图像的槽位分数是否恒定。"""
import json, os, shutil, sys, tempfile
from pathlib import Path

import yaml

ROOT = Path(r"d:\CGAIC\AOI_SSOCL\stable_AOI_tree\AOI_Core")
ARCHIVE = ROOT / "storage" / "archive" / "gold_finger" / "20261011"

outdir = Path(tempfile.mkdtemp(prefix="aoi_fixsnap3_"))
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
picks = [p for p in sorted(ARCHIVE.iterdir()) if p.suffix.lower() == ".png"]
normals = [p for p in picks if "bench" not in p.name][:3]
defects = [p for p in picks if "bench" in p.name][:3]
for p in normals + defects:
    r = svc.detect_image(str(p), "gold_finger", persist=False, with_heatmap=False)
    print(p.name, "| decision", r["decision"], "| score", round(r["final_score"], 4),
          "| slots", json.dumps({k: round(v, 4) if isinstance(v, (int, float)) else v
                                 for k, v in (r.get("slots") or {}).items()}, ensure_ascii=False),
          "| weights", json.dumps(r.get("weights"), ensure_ascii=False))
