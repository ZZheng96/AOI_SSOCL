"""G1 / AC-066 / AC-007：解码构成 + e2e P95（空闲/负载）。

开发版代码 + 稳定版快照只读 TEMP 副本；真实图随机 <=20 张，2500² 用合成图。
口径：
  svc_e2e_ms   = detect_image 内部 latency_e2e_ms（解码+推理+热图同步段，不含 HTTP/落库）
  svc_wall_ms  = detect_image 调用墙钟（含落库 commit 与 IO join，不含 HTTP）
  http_*_ms    = TestClient 进程内 ASGI 往返（含 multipart 解析，不含真实网络）
  负载         = 另 2 线程持续同品类推理；含排队（lock_wait_ms 单列）
"""
import io, json, os, random, shutil, statistics, sys, tempfile, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml

DEV = Path(r"d:\CGAIC\AOI_SSOCL\AOI_tree\AOI_Core")
STABLE = Path(r"d:\CGAIC\AOI_SSOCL\stable_AOI_tree\AOI_Core")
ARCHIVE = STABLE / "storage" / "archive" / "gold_finger" / "20261011"
OUT = Path(__file__).with_name("bench_decode.json")
CAT = "gold_finger"
N_IMG = 20

tmp = Path(tempfile.mkdtemp(prefix="aoi_finalperf_dec_"))
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

import cv2  # noqa: E402
from PIL import Image  # noqa: E402

from backend.pipeline import service as svc_mod  # noqa: E402
from backend.pipeline.service import get_detection_service  # noqa: E402
from backend.engine import get_engine  # noqa: E402


def pct(xs, q):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))], 1)


def summ(xs):
    return {"n": len(xs), "p50": pct(xs, 0.5), "p95": pct(xs, 0.95), "max": round(max(xs), 1)}


def timeit(fn, n=5):
    fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000)
    return round(statistics.median(ts), 1)


res = {"temp": str(tmp), "cv2": cv2.__version__, "pil": Image.__version__,
       "has_IMREAD_COLOR_RGB": hasattr(cv2, "IMREAD_COLOR_RGB")}

# ── A. 解码微基准 ────────────────────────────────────────────
print("[A] 解码微基准", flush=True)
rng = np.random.default_rng(20261013)
syn = cv2.GaussianBlur(rng.integers(0, 256, (2500, 2500, 3), dtype=np.uint8), (5, 5), 0)
syn_dir = tmp / "syn"
syn_dir.mkdir()
syn_png, syn_jpg = syn_dir / "syn2500.png", syn_dir / "syn2500.jpg"
Image.fromarray(syn).save(syn_png)
Image.fromarray(syn).save(syn_jpg, quality=95)
random.seed(20261013)
pngs = sorted(p for p in ARCHIVE.iterdir() if p.suffix.lower() == ".png")
sample = random.sample(pngs, N_IMG)
real_png = sample[0]
real_jpg = syn_dir / "real1024.jpg"
Image.open(real_png).convert("RGB").save(real_jpg, quality=95)

dec_rows = []
for name, p in (("real1024_png", real_png), ("real1024_jpg", real_jpg),
                ("syn2500_png", syn_png), ("syn2500_jpg", syn_jpg)):
    ref = np.array(Image.open(p).convert("RGB"))
    data = p.read_bytes()
    buf = np.frombuffer(data, np.uint8)
    row = {"image": name, "shape": list(ref.shape), "bytes": len(data)}
    row["pil_ms"] = timeit(lambda: np.array(Image.open(io.BytesIO(data)).convert("RGB")))
    row["cv2_bgr2rgb_ms"] = timeit(lambda: cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR),
                                                        cv2.COLOR_BGR2RGB))
    if hasattr(cv2, "IMREAD_COLOR_RGB"):
        row["cv2_rgb_ms"] = timeit(lambda: cv2.imdecode(buf, cv2.IMREAD_COLOR_RGB))
    cur = svc_mod._decode_rgb(p)
    row["decode_rgb_ms"] = timeit(lambda: svc_mod._decode_rgb(p))
    row["decode_rgb_maxdiff"] = int(np.abs(cur.astype(int) - ref).max())
    red = cv2.imdecode(buf, cv2.IMREAD_REDUCED_COLOR_2)
    row["reduced2_ms"] = timeit(lambda: cv2.imdecode(buf, cv2.IMREAD_REDUCED_COLOR_2))
    red_up = cv2.resize(cv2.cvtColor(red, cv2.COLOR_BGR2RGB), ref.shape[1::-1])
    row["reduced2_shape"] = list(red.shape)
    row["reduced2_up_maxdiff"] = int(np.abs(red_up.astype(int) - ref).max())
    # detect_pil 落盘的 PNG 重编码成本（/detect/upload 路径独有）
    pil = Image.open(io.BytesIO(data))
    row["pil_png_encode_ms"] = timeit(lambda: pil.convert("RGB").save(io.BytesIO(), "PNG"), n=3)
    dec_rows.append(row)
    print("   ", json.dumps(row, ensure_ascii=False), flush=True)
res["decode"] = dec_rows

# ── B. 引擎：REDUCED 精度门 + 解码/推理流水重叠 ──────────────
print("[B] 引擎装载 + 预热", flush=True)
eng = get_engine()
svc = get_detection_service()
res["warmup"] = eng.warmup(CAT, np.array(Image.open(real_png).convert("RGB")))
print("    warmup:", res["warmup"], flush=True)

print("[B1] IMREAD_REDUCED_2 精度门（前 6 张）", flush=True)
red_rows = []
for i, p in enumerate(sample[:6]):
    full = svc_mod._decode_rgb(p)
    red = cv2.cvtColor(cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_REDUCED_COLOR_2),
                       cv2.COLOR_BGR2RGB)
    red = cv2.resize(red, full.shape[1::-1])
    a = eng.predict_ndarray(CAT, full)
    b = eng.predict_ndarray(CAT, red)
    red_rows.append({"image": p.name, "full": round(a.score, 6), "reduced": round(b.score, 6),
                     "abs_diff": round(abs(a.score - b.score), 6),
                     "decision_same": a.decision == b.decision})
    print(f"    [{i + 1}/6] {red_rows[-1]}", flush=True)
res["reduced_gate"] = {"rows": red_rows,
                       "bit_identical": all(r["abs_diff"] == 0 for r in red_rows),
                       "decision_flips": sum(not r["decision_same"] for r in red_rows)}

print("[B2] 串行 vs 解码/推理重叠（20 张）", flush=True)
ser_ms, ser_scores = [], []
t_ser = time.perf_counter()
for p in sample:
    t = time.perf_counter()
    r = eng.predict_ndarray(CAT, svc_mod._decode_rgb(p))
    ser_ms.append((time.perf_counter() - t) * 1000)
    ser_scores.append(r.score)
t_ser = (time.perf_counter() - t_ser) * 1000
pip_scores = []
with ThreadPoolExecutor(max_workers=1) as ex:
    t_pip = time.perf_counter()
    fut = ex.submit(svc_mod._decode_rgb, sample[0])
    for i in range(len(sample)):
        arr = fut.result()
        if i + 1 < len(sample):
            fut = ex.submit(svc_mod._decode_rgb, sample[i + 1])
        pip_scores.append(eng.predict_ndarray(CAT, arr).score)
    t_pip = (time.perf_counter() - t_pip) * 1000
res["overlap"] = {"serial_total_ms": round(t_ser, 1), "pipelined_total_ms": round(t_pip, 1),
                  "serial_per_img": summ(ser_ms),
                  "throughput_gain": round(t_ser / t_pip, 3),
                  "scores_identical": ser_scores == pip_scores}
print("   ", res["overlap"], flush=True)

# ── C. e2e 空闲 ──────────────────────────────────────────────
print("[C] e2e 空闲（svc + HTTP）", flush=True)
idle = {"svc_e2e": [], "svc_wall": [], "infer": []}
for i, p in enumerate(sample):
    t = time.perf_counter()
    r = svc.detect_image(str(p), CAT, persist=True, with_heatmap=True)
    idle["svc_wall"].append((time.perf_counter() - t) * 1000)
    idle["svc_e2e"].append(r["latency_e2e_ms"])
    idle["infer"].append(r["latency_ms"])
    print(f"    [{i + 1}/{N_IMG}] e2e={r['latency_e2e_ms']:.1f} wall={idle['svc_wall'][-1]:.1f}",
          flush=True)

from fastapi.testclient import TestClient  # noqa: E402
from backend.api.app import create_app  # noqa: E402
client = TestClient(create_app())
http = {"upload_png": [], "frame_png": [], "frame_jpg": [], "frame_jpg_e2e": []}
for i, p in enumerate(sample[:10]):
    png = p.read_bytes()
    bio = io.BytesIO()
    Image.open(p).convert("RGB").save(bio, "JPEG", quality=95)
    jpg = bio.getvalue()
    for key, route, data, fn in (("upload_png", "/api/detect/upload", png, "a.png"),
                                 ("frame_png", "/api/detect/frame", png, "a.png"),
                                 ("frame_jpg", "/api/detect/frame", jpg, "a.jpg")):
        t = time.perf_counter()
        rr = client.post(route, files={"file": (fn, data)}, data={"category": CAT})
        http[key].append((time.perf_counter() - t) * 1000)
        assert rr.status_code == 200, rr.text[:300]
        if key == "frame_jpg":
            http["frame_jpg_e2e"].append(rr.json()["latency_e2e_ms"])
    print(f"    [{i + 1}/10] upload_png={http['upload_png'][-1]:.1f} "
          f"frame_png={http['frame_png'][-1]:.1f} frame_jpg={http['frame_jpg'][-1]:.1f}", flush=True)

syn2500 = []
for i in range(5):
    t = time.perf_counter()
    r = svc.detect_image(str(syn_png), CAT, persist=False, with_heatmap=True)
    syn2500.append({"e2e": r["latency_e2e_ms"], "wall": (time.perf_counter() - t) * 1000,
                    "infer": r["latency_ms"]})
    print(f"    [syn2500 {i + 1}/5] {syn2500[-1]}", flush=True)
res["idle"] = {k: summ(v) for k, v in idle.items()}
res["http_idle"] = {k: summ(v) for k, v in http.items()}
res["syn2500_png_idle"] = {k: summ([x[k] for x in syn2500]) for k in ("e2e", "wall", "infer")}

# ── D. e2e 负载（2 后台线程持续同品类推理） ─────────────────
print("[D] e2e 负载", flush=True)
stop = threading.Event()
bg_n = [0]


def bg(seed):
    rr = random.Random(seed)
    while not stop.is_set():
        svc.detect_image(str(rr.choice(sample)), CAT, persist=False, with_heatmap=False)
        bg_n[0] += 1


ths = [threading.Thread(target=bg, args=(s,), daemon=True) for s in (1, 2)]
for th in ths:
    th.start()
time.sleep(0.5)
load = {"svc_e2e": [], "svc_wall": [], "infer": [], "lock_wait": []}
for i, p in enumerate(sample):
    t = time.perf_counter()
    r = svc.detect_image(str(p), CAT, persist=True, with_heatmap=True)
    load["svc_wall"].append((time.perf_counter() - t) * 1000)
    load["svc_e2e"].append(r["latency_e2e_ms"])
    load["infer"].append(r["latency_ms"])
    load["lock_wait"].append(float(r.get("lock_wait_ms") or 0.0))
    print(f"    [{i + 1}/{N_IMG}] e2e={r['latency_e2e_ms']:.1f} wait={load['lock_wait'][-1]:.1f}",
          flush=True)
stop.set()
for th in ths:
    th.join()
res["load"] = {k: summ(v) for k, v in load.items()}
res["load"]["bg_requests"] = bg_n[0]
res["over_1s"] = sum(x > 1000 for v in (idle["svc_wall"], load["svc_wall"]) for x in v)
res["scope"] = __doc__
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
print("idle:", res["idle"], "\nload:", res["load"], "\nhttp:", res["http_idle"],
      "\nsyn2500:", res["syn2500_png_idle"], flush=True)
