"""AC-070 显存峰值 + AC-071 冷启动分段（真实 GPU，TEMP 快照副本）。

口径：torch.cuda.max_memory_allocated（PyTorch 分配器口径，不含 CUDA context / cudnn workspace
以外的驱动占用）；每个场景前 reset_peak_memory_stats，报告 base=场景前 allocated、peak、delta。
场景：idle（单品类推理 5 张）/ train（learn_candidate 候选隔离 1 次 + 旧在线即学 1 次）/
replay（驱逐 pipe 后重载并重放 WAL）/ multi（gold_finger + smt_solder 双品类常驻交替推理）。
"""
import json, os, random, shutil, sys, tempfile, time
from pathlib import Path

import yaml

DEV = Path(r"d:\CGAIC\AOI_SSOCL\AOI_tree\AOI_Core")
STABLE = Path(r"d:\CGAIC\AOI_SSOCL\stable_AOI_tree\AOI_Core")
ARCHIVE = STABLE / "storage" / "archive" / "gold_finger" / "20261011"
OUT = Path(__file__).with_name("bench_vram.json")
CATS = ("gold_finger", "smt_solder")

tmp = Path(tempfile.mkdtemp(prefix="aoi_finalperf_vram_"))
for c in CATS:
    shutil.copytree(STABLE / "storage" / "engine" / "snapshots" / c, tmp / "engine" / "snapshots" / c)
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
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from backend.engine import get_engine  # noqa: E402
from backend.pipeline import service as svc_mod  # noqa: E402

MiB = 1024 ** 2
res = {"temp": str(tmp), "scope": __doc__}
if not torch.cuda.is_available():
    res["skipped"] = "no_cuda_device：无 CUDA，显存峰值不可测（CandidateBudget 同样降级 enabled=False）"
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(res["skipped"])
    sys.exit(0)
props = torch.cuda.get_device_properties(0)
res["device"] = {"name": props.name, "total_mib": round(props.total_memory / MiB, 1)}

random.seed(20261013)
pngs = sorted(p for p in ARCHIVE.iterdir() if p.suffix.lower() == ".png")
sample = [p for p in random.sample(pngs, 20)
          if Image.open(p).size[0] <= 2500][:6]  # 跳过 4096 大图，单图 <1s
eng = get_engine()


class Scene:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.base = torch.cuda.memory_allocated()
        self.t = time.perf_counter()
        print(f"[{self.name}] start base={self.base / MiB:.1f}MiB", flush=True)
        return self

    def __exit__(self, *exc):
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        res.setdefault("scenes", {})[self.name] = {
            "base_mib": round(self.base / MiB, 1), "peak_mib": round(peak / MiB, 1),
            "delta_mib": round((peak - self.base) / MiB, 1),
            "after_mib": round(torch.cuda.memory_allocated() / MiB, 1),
            "reserved_mib": round(torch.cuda.memory_reserved() / MiB, 1),
            "peak_frac_of_device": round(peak / props.total_memory, 4),
            "wall_ms": round((time.perf_counter() - self.t) * 1000, 1),
            **({"error": repr(exc[1])[:300]} if exc[1] else {})}
        print(f"[{self.name}] {res['scenes'][self.name]}", flush=True)
        return True  # 记录错误后继续下一场景


# AC-071 冷启动分段（首个品类）
print("[cold] gold_finger warmup", flush=True)
rgb0 = svc_mod._decode_rgb(sample[0])
res["cold_start"] = {"gold_finger": eng.warmup("gold_finger", rgb0)}
print("   ", res["cold_start"]["gold_finger"], flush=True)

with Scene("idle"):
    for i, p in enumerate(sample[:5]):
        eng.predict_ndarray("gold_finger", svc_mod._decode_rgb(p))
        print(f"    [{i + 1}/5]", flush=True)

T = "smt_solder"  # gold_finger 为 legacy 非 DINO 快照（GPU 占用 0、不可候选学习），训练/回放用 DINO 品类
with Scene("idle_dino"):
    res["cold_start"][T] = eng.warmup(T, rgb0)
    for i, p in enumerate(sample[:5]):
        eng.predict_ndarray(T, svc_mod._decode_rgb(p))
        print(f"    [{i + 1}/5]", flush=True)

v = eng.current_version(T)
with Scene("train_candidate"):
    r = eng.learn_candidate(T, str(sample[1]), verdict="wrong", label=1,
                            event_id="bench-vram-1", parent_version=v)
    res["train_candidate_budget"] = r.get("candidate_budget")

with Scene("train_online"):
    res["train_online"] = eng.submit_feedback(T, str(sample[2]), verdict="wrong", label=1,
                                              event_id="bench-vram-2", allow_legacy_online=True)

with Scene("replay"):
    for pipe, _ in list(eng._pipes.values()):
        ex = getattr(pipe, "_cpu_exec", None)
        if ex is not None:
            ex.shutdown(wait=True)
    eng._pipes.clear()
    torch.cuda.empty_cache()
    eng._ensure_loaded(T)
    eng.predict_ndarray(T, rgb0)
res["cold_start"][T + "_replay"] = eng.cold_start_stats(T)

with Scene("multi"):
    eng._ensure_loaded("gold_finger")
    for i, p in enumerate(sample[:4]):
        rgb = svc_mod._decode_rgb(p)
        eng.predict_ndarray("gold_finger", rgb)
        eng.predict_ndarray("smt_solder", rgb)
        print(f"    [{i + 1}/4]", flush=True)
res["resident_categories"] = list(eng._pipes.keys())

peaks = [s["peak_mib"] for s in res["scenes"].values()]
res["max_peak_mib"] = max(peaks)
res["default_limit_mib"] = round(props.total_memory * 0.85 / MiB, 1)
res["headroom_ok"] = res["max_peak_mib"] < res["default_limit_mib"]
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
print(json.dumps({k: res[k] for k in ("max_peak_mib", "default_limit_mib", "headroom_ok")}), flush=True)
