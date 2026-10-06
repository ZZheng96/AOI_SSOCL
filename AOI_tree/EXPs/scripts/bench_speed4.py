"""U87 crop 打分速度优化实测（零精度损失验证，2026-08-21）：分段计时 + 前向变体 + 解码变体。

目标：U86v8 crop 打分（ov0.3，853ms/张，batch8 eager）在不降精度前提下的提速空间。
变体（前向）：base(batch8 eager，对齐 U86v8 口径) / b32 / b32trc(jit.trace) / b32ort(ONNX Runtime CUDA)。
变体（CPU）：+pool = crop resize 线程池（cv2.resize 释放 GIL，零精度损失）。
解码变体：imdecode(现行) / imread / PIL / REDUCED_2(降采样，有损仅计时参考)。
精度验收：每图最终分 vs base 的 max|Δ| + 30 张 AUROC 逐位对照（labels 存在=缺陷）。

用法: python scripts/bench_speed4.py [--n 30]
"""
import os
import sys
import time
import argparse

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import cv2
import torch
import torch.nn.functional as F

from src.backbone.dino import FrozenDINO
from src.slots.shead import SheadSlot, _image_level_feats, _sliding_grid

GYU_ROOT = r"D:\CGAIC\data_origin\GYU-DET"
HEAD = os.path.join(os.path.dirname(__file__), "..", "outputs", "m0", "u86v8_head.pt")
OV = 0.3
SCORE_MAX_CROPS = 200
K33 = None


def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    pos = s[y == 1]; neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    # 平分处理（分数几乎连续，平局罕见）
    return (ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def decode_img(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class FwdWrap(torch.nn.Module):
    """ONNX/trace 导出壳：输入 (B,3,448,448) float[0,1] -> (B,384,32,32)，跳过数据依赖分支。"""

    def __init__(self, bb):
        super().__init__()
        self.bb = bb

    def forward(self, x):
        m = self.bb
        x = (x - m.mean) / m.std
        tok = m._intermediate(x)
        B, N, D = tok.shape
        g = int(N ** 0.5)
        feat = tok.permute(0, 2, 1).reshape(B, D, g, N // g)
        if g != m.grid:
            feat = F.interpolate(feat, size=(m.grid, m.grid), mode="bilinear",
                                 align_corners=False)
        return feat


def make_grid_crops(img, crop_size):
    return _sliding_grid(img, crop_size, OV, SCORE_MAX_CROPS)


def run_variant(name, fwd_fn, batch, imgs, crop_size, shead, device, use_pool, pool):
    """返回 (scores, 阶段ms dict)。fwd_fn: (tensor B,3,448,448 on device) -> feats"""
    import concurrent.futures as cf
    tgt = 448
    st = {"grid": 0.0, "resize": 0.0, "fwd": 0.0, "head": 0.0}
    scores = []
    nw = 0
    for img in imgs:
        t = time.perf_counter()
        crops, ny, nx, trunc = make_grid_crops(img, crop_size)
        st["grid"] += time.perf_counter() - t
        nw += len(crops)
        t = time.perf_counter()
        feats_all = []
        for i in range(0, len(crops), batch):
            chunk = crops[i:i + batch]
            if use_pool:
                resized = list(pool.map(lambda c: cv2.resize(c, (tgt, tgt))
                                        if c.shape[:2] != (tgt, tgt) else c, chunk))
            else:
                resized = [cv2.resize(c, (tgt, tgt)) if c.shape[:2] != (tgt, tgt) else c
                           for c in chunk]
            x = torch.from_numpy(np.stack(resized).astype(np.float32) / 255.0)
            st["resize"] += time.perf_counter() - t
            t = time.perf_counter()
            feats_all.append(fwd_fn(x.permute(0, 3, 1, 2).to(device)))
            st["fwd"] += time.perf_counter() - t
            t = time.perf_counter()
        f = torch.cat(feats_all, dim=0)
        x2 = _image_level_feats(f.float().to(device), shead.chan_stats, shead.patch_stats)
        s = shead.head(x2.to(device)).reshape(-1).float()
        if not trunc:
            smap = s.reshape(ny, nx)[None, None]
            pad = F.pad(smap, (1, 1, 1, 1), mode="replicate")
            scores.append(float(F.conv2d(pad, K33).max().item()))
        else:
            scores.append(float(s.topk(min(3, len(s))).values.mean().item()))
        st["head"] += time.perf_counter() - t
    n = len(imgs)
    tot = {k: v / n * 1000 for k, v in st.items()}
    print(f"  [{name}] {n} 张: grid {tot['grid']:.0f} | resize {tot['resize']:.0f} | "
          f"fwd {tot['fwd']:.0f} | head {tot['head']:.0f} | 合计(不含解码) "
          f"{sum(tot.values()):.0f}ms/张 | 平均 {nw / n:.0f} 窗", flush=True)
    return scores, tot


def main():
    global K33
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--skip-ort", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    K33 = torch.ones(1, 1, 3, 3, device=device) / 9.0

    backbone = FrozenDINO(output_layer=9, grid=32, device=device)
    torch.manual_seed(0)
    ckpt = torch.load(HEAD, map_location=device)
    shead = SheadSlot({"seed": 42}, backbone, device)
    shead.head.load_state_dict(ckpt["head"])
    shead.head.eval()  # 修复：头含 Dropout(0.3)，不 eval 则打分随机（首跑精度对照全部作废的原因）
    shead.chan_stats, shead.patch_stats = ckpt["chan_stats"], ckpt["patch_stats"]
    crop_size = int(ckpt["crop_size"])
    print(f"[U87] device={device} head=u86v8 crop_size={crop_size} ov={OV}", flush=True)

    # ---- 选图（与 bench_speed2 同口径：按文件名排序取前 n）----
    img_dir = os.path.join(GYU_ROOT, "test", "images")
    lbl_dir = os.path.join(GYU_ROOT, "test", "labels")
    names = sorted(f for f in os.listdir(img_dir)
                   if f.lower().endswith((".jpg", ".png", ".jpeg")))[:args.n]
    paths = [os.path.join(img_dir, f) for f in names]
    ys = [int(os.path.exists(os.path.join(lbl_dir, os.path.splitext(f)[0] + ".txt")))
          for f in names]

    # ---- 解码（一次性，含计时；30 张全留内存 ~1.5GB）----
    t0 = time.perf_counter()
    imgs = []
    for p in paths:
        imgs.append(decode_img(p))
    dec_ms = (time.perf_counter() - t0) / len(paths) * 1000
    sizes = [f"{im.shape[1]}x{im.shape[0]}" for im in imgs[:3]]
    print(f"[U87] 解码(imdecode 现行) {dec_ms:.0f}ms/张 | 前3张尺寸 {sizes} | "
          f"缺陷 {sum(ys)}/{len(ys)}", flush=True)

    # ---- 解码变体对比（前 10 张）----
    from PIL import Image
    t0 = time.perf_counter()
    for p in paths[:10]:
        cv2.imread(p)
    r_imread = (time.perf_counter() - t0) / 10 * 1000
    t0 = time.perf_counter()
    for p in paths[:10]:
        np.asarray(Image.open(p).convert("RGB"))
    r_pil = (time.perf_counter() - t0) / 10 * 1000
    t0 = time.perf_counter()
    for p in paths[:10]:
        cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_REDUCED_COLOR_2)
    r_red2 = (time.perf_counter() - t0) / 10 * 1000
    print(f"[U87] 解码变体(10张): imread {r_imread:.0f} | PIL {r_pil:.0f} | "
          f"REDUCED_2(有损) {r_red2:.0f} ms/张", flush=True)

    # ---- warmup ----
    dummy = torch.rand(8, 3, 448, 448, device=device)
    for _ in range(3):
        backbone.forward(dummy)

    # ---- 变体 1: base（batch8 eager，对齐 U86v8）----
    with torch.inference_mode():
        s_base, tot_base = run_variant("base b8 eager", backbone.forward, 8, imgs,
                                       crop_size, shead, device, False, None)

    # ---- 变体 2: b32 eager ----
    with torch.inference_mode():
        s_b32, _ = run_variant("b32 eager", backbone.forward, 32, imgs,
                               crop_size, shead, device, False, None)

    # ---- 变体 3: b32 + jit.trace ----
    wrap = FwdWrap(backbone).to(device).eval()
    with torch.inference_mode():
        traced = torch.jit.trace(wrap, torch.rand(8, 3, 448, 448, device=device))
    traced = traced.eval()
    with torch.inference_mode():
        s_trc, _ = run_variant("b32 trace", traced, 32, imgs, crop_size, shead,
                               device, False, None)

    # ---- 变体 4: b32 + ORT CUDA（--skip-ort 可跳过：本机 CUDA EP 对该图病态慢，疑似回退 CPU）----
    s_ort = None
    if not args.skip_ort:
        import onnxruntime as ort
        onnx_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "m0",
                                 "u87_dino_448.onnx")
        with torch.inference_mode():
            torch.onnx.export(wrap, torch.rand(1, 3, 448, 448, device=device), onnx_path,
                              input_names=["x"], output_names=["feat"],
                              dynamic_axes={"x": {0: "B"}, "feat": {0: "B"}},
                              opset_version=17, do_constant_folding=True)
        sess = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider",
                                                          "CPUExecutionProvider"])
        ort_fn = lambda x: torch.from_numpy(
            sess.run(None, {"x": x.cpu().numpy()})[0]).to(device)
        with torch.inference_mode():
            s_ort, _ = run_variant("b32 ORT-CUDA", ort_fn, 32, imgs, crop_size, shead,
                                   device, False, None)

    # ---- 变体 5: b32 trace + resize 线程池 ----
    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        with torch.inference_mode():
            s_pool, _ = run_variant("b32 trace+pool8", traced, 32, imgs, crop_size,
                                    shead, device, True, pool)

    # ---- 变体 6: b8 + fp16 autocast（速度参考 + 精度风险量化，非零损失）----
    def fwd_fp16(x):
        with torch.autocast("cuda", dtype=torch.float16):
            return backbone.forward(x).float()
    with torch.inference_mode():
        s_fp16, tot_fp16 = run_variant("b8 fp16autocast", fwd_fp16, 8, imgs, crop_size,
                                       shead, device, False, None)

    # ---- 纯前向 microbench（固定 32 窗，5 次平均）----
    xx = torch.rand(32, 3, 448, 448, device=device)
    with torch.inference_mode():
        for fn, tag in [(backbone.forward, "eager"), (traced, "trace")]:
            for _ in range(2):
                fn(xx)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(5):
                fn(xx)
            torch.cuda.synchronize()
            print(f"[U87] 纯前向 32窗 {tag}: {(time.perf_counter() - t0) / 5 * 1000:.0f}ms",
                  flush=True)
        for _ in range(2):
            fwd_fp16(xx)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            fwd_fp16(xx)
        torch.cuda.synchronize()
        print(f"[U87] 纯前向 32窗 fp16autocast: {(time.perf_counter() - t0) / 5 * 1000:.0f}ms",
              flush=True)
        if not args.skip_ort:
            xs = xx.cpu().numpy()
            for _ in range(2):
                sess.run(None, {"x": xs})
            t0 = time.perf_counter()
            for _ in range(5):
                sess.run(None, {"x": xs})
            print(f"[U87] 纯前向 32窗 ORT-CUDA: {(time.perf_counter() - t0) / 5 * 1000:.0f}ms"
                  f"(含 H2D/D2H)", flush=True)

    # ---- 精度一致性 ----
    au_base = auroc(ys, s_base)
    print(f"\n[U87] 精度对照（base AUROC={au_base:.4f}, n={len(ys)} 缺陷{sum(ys)}）",
          flush=True)
    variants = [("b32", s_b32), ("b32 trace", s_trc), ("b32 trace+pool", s_pool),
                ("b8 fp16(有风险)", s_fp16)]
    if s_ort is not None:
        variants.append(("b32 ORT", s_ort))
    for tag, s in variants:
        d = float(np.max(np.abs(np.array(s) - np.array(s_base))))
        au = auroc(ys, s)
        same = "一致" if abs(au - au_base) < 1e-9 else ("近似" if abs(au - au_base) < 0.02 else "变化")
        print(f"  {tag:16s} max|Δ分|={d:.2e}  AUROC={au:.4f}  ({same})", flush=True)

    # ---- 窗数外推（按 base 实测线性外推：fwd+head 与窗数成正比）----
    per_win = (tot_base["fwd"] + tot_base["head"]) / 37.0
    dec = 46.0  # imread 实测
    print(f"\n[U87] 窗数外推（fwd+head {per_win:.1f}ms/窗 + 解码{dec:.0f} + resize{tot_base['resize']:.0f}）：",
          flush=True)
    for nw in (10, 16, 20, 37, 48):
        print(f"  {nw:2d} 窗 -> 约 {per_win * nw + dec + tot_base['resize']:.0f}ms/张", flush=True)


if __name__ == "__main__":
    main()
