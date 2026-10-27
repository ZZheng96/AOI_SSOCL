"""TEST-03 显存场景采样（AC-070）。

场景：idle_after_load / infer_single / infer_dual_process / train_prepare /
replay_reload。nvidia-smi 约 100ms 间隔采样整卡 memory.used，同时记录
torch.cuda.max_memory_allocated（进程口径）。隔离 TEMP 环境，不改生产 DB/快照。
OOM 破坏性复现不在本脚本内（见 result.md 边界）。
"""
import json, os, random, shutil, subprocess, sys, tempfile, threading, time
from pathlib import Path

import yaml

ROOT = Path(r"d:\CGAIC\AOI_SSOCL\stable_AOI_tree\AOI_Core")
TASK = Path(r"d:\CGAIC\AOI_SSOCL\review\task\test03_vram")
ARCHIVE = ROOT / "storage" / "archive" / "gold_finger" / "20261011"

samples = []
scenario = {"name": "boot", "t0": time.perf_counter()}
stop_flag = threading.Event()


def sampler():
    while not stop_flag.is_set():
        t = time.perf_counter()
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                text=True, timeout=5).strip()
            mib = float(out.splitlines()[0])
        except Exception:
            mib = float("nan")
        samples.append({"t": round(t, 3), "scenario": scenario["name"], "used_mib": mib})
        time.sleep(0.1)


def set_scenario(name):
    scenario["name"] = name
    scenario["t0"] = time.perf_counter()
    print(f"[scenario] {name}", flush=True)


def main():
    outdir = Path(tempfile.mkdtemp(prefix="aoi_test03_"))
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

    th = threading.Thread(target=sampler, daemon=True)
    th.start()

    from PIL import Image
    pool = []
    for p in sorted(ARCHIVE.iterdir()):
        if p.suffix.lower() not in (".jpg", ".png"):
            continue
        with Image.open(p) as im:
            if im.size == (2500, 2500):
                pool.append(p)
    random.seed(321)
    random.shuffle(pool)

    import torch
    from backend.pipeline.service import get_detection_service
    from backend.engine import get_engine
    engine = get_engine()
    engine.learning_parallel = False
    service = get_detection_service()

    peaks = {}

    def mark(name):
        peaks[name] = int(torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else None
        torch.cuda.reset_peak_memory_stats()

    def prep(category, imgs):
        train_dir = outdir / "train" / category
        train_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, src in enumerate(imgs):
            dst = train_dir / f"n{i:03d}{src.suffix}"
            shutil.copy2(src, dst)
            paths.append(str(dst))
        engine.prepare(category, {"init_normal": paths, "init_defect": [], "val": [], "test": []},
                       activate=True, progress_cb=lambda f, m: None)

    set_scenario("train_prepare_gold_finger")
    prep("gold_finger", pool[:30])
    mark("train_prepare_gold_finger")

    set_scenario("idle_after_load")
    time.sleep(5)
    mark("idle_after_load")

    set_scenario("infer_single")
    for p in pool[40:58]:
        service.detect_image(str(p), "gold_finger", persist=True, with_heatmap=True)
    mark("infer_single")

    # 多槽：第二品类装载（不同类别并行驻留）
    set_scenario("train_prepare_second_slot")
    prep("gold_finger_b", pool[60:90])
    mark("train_prepare_second_slot")

    set_scenario("idle_two_slots")
    time.sleep(3)
    mark("idle_two_slots")

    # 双实例：子进程共享同一 GPU 同时推理
    set_scenario("infer_dual_process")
    child = subprocess.Popen(
        [sys.executable, str(TASK / "raw" / "vram_child.py"), str(cfgp), str(pool[100]),
         "gold_finger", "12"],
        cwd=str(ROOT))
    for p in pool[110:122]:
        service.detect_image(str(p), "gold_finger", persist=False, with_heatmap=False)
    child.wait(timeout=300)
    mark("infer_dual_process")

    # 回放：卸载后重装载（WAL replay 路径）
    set_scenario("replay_reload")
    with engine._write_lock:
        engine._pipes.pop("gold_finger", None)
    engine._ensure_loaded("gold_finger")
    mark("replay_reload")

    set_scenario("done")
    time.sleep(1)
    stop_flag.set()
    th.join(timeout=5)

    import csv
    with open(TASK / "raw" / "vram_samples.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["t", "scenario", "used_mib"])
        w.writeheader()
        w.writerows(samples)

    agg = {}
    for s in samples:
        if s["used_mib"] != s["used_mib"]:
            continue
        a = agg.setdefault(s["scenario"], [])
        a.append(s["used_mib"])
    summary = {k: {"n": len(v), "peak_mib": max(v), "mean_mib": sum(v) / len(v)}
               for k, v in agg.items() if v}
    out = {"temp": str(outdir), "card_peak_mib_by_scenario": summary,
           "torch_process_peak_mib": peaks, "child_exit": child.returncode}
    (TASK / "raw" / "vram_summary.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
