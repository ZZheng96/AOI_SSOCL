"""TEST-05 子进程：在受控环境变量下加载 EXPs DINOv2 主干，报告来源/错误/耗时。"""
import os, sys, time, traceback

EXP_SRC = r"d:\CGAIC\AOI_SSOCL\AOI_tree\EXPs\src"
sys.path.insert(0, EXP_SRC)

t0 = time.perf_counter()
try:
    from backbone.dino import _load_dinov2_from_cache
    model = _load_dinov2_from_cache("vits14")
    dt = time.perf_counter() - t0
    n = sum(p.numel() for p in model.parameters())
    print(f"RESULT OK load_s={dt:.2f} params={n}")
except Exception as exc:
    dt = time.perf_counter() - t0
    print(f"RESULT FAIL load_s={dt:.2f} type={type(exc).__name__} msg={str(exc)[:300]}")
    traceback.print_exc()
