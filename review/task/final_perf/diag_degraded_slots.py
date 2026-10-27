"""§65.2 排查：降级快照（gold_finger v2 TEMP 副本）下 blob/trad 分数恒定的原因。
开发版代码 + 稳定版快照只读副本；打印 raw/cal 与校准器训练分布边界。"""
import json, os, random, shutil, sys, tempfile, time
from pathlib import Path

import numpy as np
import yaml

DEV = Path(r"d:\CGAIC\AOI_SSOCL\AOI_tree\AOI_Core")
STABLE = Path(r"d:\CGAIC\AOI_SSOCL\stable_AOI_tree\AOI_Core")
ARCHIVE = STABLE / "storage" / "archive" / "gold_finger" / "20261011"
OUT = Path(__file__).with_name("diag_degraded_slots.json")

tmp = Path(tempfile.mkdtemp(prefix="aoi_finalperf_deg_"))
shutil.copytree(STABLE / "storage" / "engine" / "snapshots" / "gold_finger",
                tmp / "engine" / "snapshots" / "gold_finger")
cfg = yaml.safe_load((DEV / "configs" / "default.yaml").read_text(encoding="utf-8"))
cfg["system"].update(storage_dir=str(tmp / "storage"), engine_storage=str(tmp / "engine"),
                     engine_base_cfg=str(DEV / "configs" / "engine_fast.yaml"),
                     database_url="sqlite:///" + (tmp / "bench.db").as_posix(), host="127.0.0.1")
cfg.setdefault("notify", {})["webhook_url"] = ""
(tmp / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
os.environ["AOI_CONFIG"] = str(tmp / "config.yaml")
os.chdir(DEV)
sys.path.insert(0, str(DEV))

from backend.pipeline.service import get_detection_service  # noqa: E402
from backend.engine import get_engine  # noqa: E402

svc = get_detection_service()
pngs = sorted(p for p in ARCHIVE.iterdir() if p.suffix.lower() == ".png")
random.seed(20261013)
normals = random.sample([p for p in pngs if "bench" not in p.name], 4)
defects = random.sample([p for p in pngs if "bench" in p.name], 2)
rows = []
for i, p in enumerate(normals + defects):
    t = time.perf_counter()
    r = svc.detect_image(str(p), "gold_finger", persist=True, with_heatmap=False)
    tr = r.get("decision_trace") or []
    fus = next((x for x in tr if x.get("stage") == "fusion"), {})
    rows.append({"image": p.name, "decision": r["decision"], "score": r["final_score"],
                 "raw": r.get("raw_scores"), "cal": r.get("slots"),
                 "trace_len": len(tr), "fusion_degraded": fus.get("degraded_slots"),
                 "wall_ms": round((time.perf_counter() - t) * 1000, 1)})
    print(f"[{i + 1}/6] {p.name} raw={r.get('raw_scores')} cal={r.get('slots')} trace_len={len(tr)}",
          flush=True)

pipe = get_engine().get_pipeline("gold_finger")
cals = {}
for n, c in pipe.calibrators.items():
    info = {"type": type(c).__name__}
    for a in ("lo", "hi", "mu", "sd", "q_lo", "q_hi", "ref", "sorted_", "scores", "a", "b",
              "x_min", "x_max", "clip"):
        v = getattr(c, a, None)
        if v is None:
            continue
        arr = np.asarray(v, dtype=float).ravel()
        info[a] = (arr.tolist() if arr.size <= 4 else
                   {"n": int(arr.size), "min": float(arr.min()), "p50": float(np.median(arr)),
                    "max": float(arr.max())})
    info["dict_keys"] = sorted(k for k in vars(c).keys())[:30]
    cals[n] = info
# 持久化记录里 decision_trace 是否非空（§65.2 问题 2 入库侧）
svc.wait_artifacts() if hasattr(svc, "wait_artifacts") else None
from backend.db.database import session_scope  # noqa: E402
from backend.db.models import Detection  # noqa: E402
with session_scope() as s:
    db_trace = [len((d.n_tiles or {}).get("decision_trace") or []) for d in s.query(Detection).all()]
res = {"temp": str(tmp), "rows": rows, "calibrators": cals, "db_trace_lens": db_trace,
       "degraded_slots": getattr(pipe, "degraded_slots", None)}
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
print("calibrators:", json.dumps(cals, ensure_ascii=False, default=str)[:1500])
print("db_trace_lens:", db_trace)
