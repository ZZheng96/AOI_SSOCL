"""G1 复测：/detect/upload 去掉 PNG 重编码后的 HTTP 往返（与 bench_decode 同 seed 同 10 张）。

口径：TestClient 进程内 ASGI 往返（含 multipart 解析 + 落盘 + 检测 + 落库，不含真实网络）。
另做逐图精度门：upload 与 frame 两条路径的 score/decision 必须一致。
"""
import json, os, random, shutil, statistics, sys, tempfile, time
from pathlib import Path

import yaml

DEV = Path(r"d:\CGAIC\AOI_SSOCL\AOI_tree\AOI_Core")
STABLE = Path(r"d:\CGAIC\AOI_SSOCL\stable_AOI_tree\AOI_Core")
ARCHIVE = STABLE / "storage" / "archive" / "gold_finger" / "20261011"
OUT = Path(__file__).with_name("bench_upload.json")
CAT = "gold_finger"

tmp = Path(tempfile.mkdtemp(prefix="aoi_finalperf_up_"))
shutil.copytree(STABLE / "storage" / "engine" / "snapshots" / CAT,
                tmp / "engine" / "snapshots" / CAT)
cfg = yaml.safe_load((DEV / "configs" / "default.yaml").read_text(encoding="utf-8"))
cfg["system"].update(storage_dir=str(tmp / "storage"), engine_storage=str(tmp / "engine"),
                     engine_base_cfg=str(DEV / "configs" / "engine_fast.yaml"),
                     database_url="sqlite:///" + (tmp / "bench.db").as_posix(), host="127.0.0.1")
cfg.setdefault("notify", {})["webhook_url"] = ""
(tmp / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
os.environ["AOI_CONFIG"] = str(tmp / "config.yaml")
os.chdir(DEV)
sys.path.insert(0, str(DEV))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from backend.api.app import create_app  # noqa: E402
from backend.engine import get_engine  # noqa: E402


def pct(xs, q):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))], 1)


random.seed(20261013)
pngs = sorted(p for p in ARCHIVE.iterdir() if p.suffix.lower() == ".png")
sample = random.sample(pngs, 20)[:10]
eng = get_engine()
eng.warmup(CAT, np.array(Image.open(sample[0]).convert("RGB")))
client = TestClient(create_app())
rows, up, fr = [], [], []
for i, p in enumerate(sample):
    data = p.read_bytes()
    out = {}
    for key, route in (("upload", "/api/detect/upload"), ("frame", "/api/detect/frame")):
        t = time.perf_counter()
        rr = client.post(route, files={"file": ("a.png", data)}, data={"category": CAT})
        ms = (time.perf_counter() - t) * 1000
        assert rr.status_code == 200, rr.text[:300]
        j = rr.json()
        sc = j.get("final_score", j.get("anomaly_score"))
        assert sc is not None, sorted(j)
        out[key] = {"ms": round(ms, 1), "score": sc, "is_anomaly": j["is_anomaly"]}
    up.append(out["upload"]["ms"])
    fr.append(out["frame"]["ms"])
    with Image.open(p) as im:
        size = list(im.size)
    rows.append({"image": p.name, "size": size, **out,
                 "same": out["upload"]["score"] == out["frame"]["score"]
                 and out["upload"]["is_anomaly"] == out["frame"]["is_anomaly"]})
    print(f"  [{i + 1}/10] {size} upload={out['upload']['ms']:.1f} frame={out['frame']['ms']:.1f} "
          f"same={rows[-1]['same']}", flush=True)
res = {"temp": str(tmp), "rows": rows,
       "upload_png": {"p50": pct(up, .5), "p95": pct(up, .95), "max": max(up),
                      "mean": round(statistics.mean(up), 1)},
       "frame_png": {"p50": pct(fr, .5), "p95": pct(fr, .95), "max": max(fr),
                     "mean": round(statistics.mean(fr), 1)},
       "all_same": all(r["same"] for r in rows), "over_1s": sum(x > 1000 for x in up),
       "scope": __doc__}
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
print(json.dumps({k: res[k] for k in ("upload_png", "frame_png", "all_same", "over_1s")},
                 ensure_ascii=False), flush=True)
