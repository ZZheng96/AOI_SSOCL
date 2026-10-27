"""内存占用诊断：定位 fit 阶段峰值（阶段边界同步采样 RSS，无外部依赖）

用法：
  python scripts/diag_mem.py --category gold_finger [--mode single]
回答"之前内存占用过高在哪"。
"""
import os
import sys
import time
import argparse
import ctypes
import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PREFIX = "[" + os.path.basename(__file__) + "] "


def _rss_bytes():
    """Windows 本进程工作集（ctypes 失败时 tasklist 后备）"""
    try:
        from ctypes import wintypes
        psapi = ctypes.WinDLL("psapi")
        k32 = ctypes.WinDLL("kernel32")

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD),
                        ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]

        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        h = k32.GetCurrentProcess()
        if psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb):
            return int(pmc.WorkingSetSize)
    except Exception:
        pass
    # tasklist 后备
    import subprocess, io, csv
    pid = os.getpid()
    try:
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=10).stdout
        row = next(csv.reader(io.StringIO(out)))
        mem_str = row[4].strip().replace(",", "").replace(" K", "").replace("K", "")
        return int(float(mem_str) * 1024)
    except Exception:
        return 0


def _vram_bytes():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_reserved()
    except Exception:
        pass
    return 0


_peak = {"rss": 0, "rss_gb": 0.0, "vram": 0, "vram_gb": 0.0, "stage": ""}


def _mark(name):
    rss = _rss_bytes()
    vram = _vram_bytes()
    if rss > _peak["rss"]:
        _peak["rss"] = rss
        _peak["rss_gb"] = rss / 2 ** 30
        _peak["vram"] = vram
        _peak["vram_gb"] = vram / 2 ** 30
        _peak["stage"] = name
    print(PREFIX + f"[{name}] rss={rss/2**30:.2f}GB vram={vram/2**30:.2f}GB", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="gold_finger")
    ap.add_argument("--mode", default="single", choices=["single", "tiles9", "tiles36"])
    ap.add_argument("--max-train", type=int, default=None, help="子采样 train/good 数量（测内存斜率用）")
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "configs", "m0.yaml"))
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["tiling"]["mode"] = args.mode

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from src.data import datalocal
    from scripts.m0_baseline import build
    from src.eval.offline import Pipeline

    _mark("启动")

    root = cfg["datasets"]["data_local"]
    bundle = datalocal.load_category(root, args.category, 100, 30, cfg["seed"])
    if args.max_train:
        rng = np.random.default_rng(cfg["seed"])
        bundle = dict(bundle)
        bundle["init_normal"] = sorted(rng.choice(
            bundle["init_normal"], min(args.max_train, len(bundle["init_normal"])), replace=False).tolist())
        bundle["init_defect"] = bundle["init_defect"][: max(1, args.max_train // 2)]
    _mark("加载数据（bundle）")

    backbone, slots = build(cfg, device)
    _mark("构建 backbone+slots（DINO 实例）")

    pipe = Pipeline(cfg, backbone, slots)
    pipe.fit(bundle)
    _mark("fit 完成")

    test = bundle["test"][:20]
    pipe.evaluate(test)
    _mark("evaluate 完成（n=20）")

    print(PREFIX + "=== 峰值总结 ===")
    print(PREFIX + f"峰值 RSS {_peak['rss_gb']:.2f}GB @ 阶段[{_peak['stage']}]")
    print(PREFIX + f"峰值 VRAM {_peak['vram_gb']:.2f}GB @ 阶段[{_peak['stage']}]")


if __name__ == "__main__":
    main()
