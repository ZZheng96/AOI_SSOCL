"""全链路速度口径实测（不足清单 #5 补强，2026-08-30）。

同一批图（默认随机 8 张，符合"调优测试随机少量图"规则）对比两种产线形态：

  A. 磁盘路径口径  POST /api/detect/image (with_heatmap=False)
     -- 引擎内含 load_image 读盘+切块+前向+融合（读盘计时在内）。
     首轮=冷缓存（产线每图首检），第 2 轮起 _item_cache 命中（复检口径）。
  B. 零拷贝内存口径 POST /api/detect/frame
     -- 计时前先把帧 bytes 读进内存（模拟相机帧缓冲直入），
        imdecode+切块+前向+融合，全程无磁盘 IO；predict_frame
     天然不走特征缓存，每轮都是冷口径。

双延迟口径：server_ms = latency_ms（引擎段，赛题检测时间口径）；
e2e_ms = latency_e2e_ms（接口内全链路：解码+推理，不含 persist/alarm）。
1s 红线（用户 2026-08-19 性能红线）按 e2e_ms 判定。
报告写 storage/logs/bench_fullchain_{时间戳}.json。

用法（需后端已启动且品类已 prepare）：
    python test/bench_fullchain.py --category bottle
    python test/bench_fullchain.py --category gold_finger --sample 12 --rounds 3
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

import requests

RED_LINE_MS = 1000.0


def _headers() -> dict:
    key = os.environ.get("AOI_API_KEY", "")
    return {"X-API-Key": key} if key else {}


def _stats(vals: list) -> dict:
    if not vals:
        return {}
    s = sorted(vals)
    return {"n": len(vals),
            "mean": round(statistics.mean(s), 1),
            "p50": round(statistics.median(s), 1),
            "p95": round(s[max(0, int(len(s) * 0.95) - 1)], 1),
            "max": round(s[-1], 1)}


def _pick_images(base: str, category: str, n: int) -> list:
    r = requests.get(f"{base}/api/images",
                     params={"category": category, "page_size": 500},
                     headers=_headers(), timeout=30)
    r.raise_for_status()
    items = (r.json() or {}).get("items", [])
    # 排除掩膜/真值图（与 bench_stability 同规则）
    ok = [it for it in items
          if it.get("id") and "ground_truth" not in str(it.get("path", ""))
          and "_mask" not in Path(str(it.get("path", ""))).stem]
    if not ok:
        raise SystemExit(f"品类 {category} 无图片，请先导入数据")
    # 优先大图（width>=2000，贴近 2500² 考核场景），不足则任意补齐
    big = [it for it in ok if (it.get("width") or 0) >= 2000]
    pool = big if len(big) >= n else ok
    random.seed(42)
    random.shuffle(pool)
    picked = pool[:n]
    print(f"[fullchain] 取样 {len(picked)} 张"
          f"（大图 {len(big)}/{len(ok)} 张可选，"
          f"尺寸 {min(it.get('width') or 0 for it in picked)}x"
          f"{min(it.get('height') or 0 for it in picked)} 起）", flush=True)
    return picked


def _expand_paths(spec: str) -> list:
    """--paths 展开：逗号分隔的文件/目录/glob，返回图片文件列表。"""
    from PIL import Image as PILImage
    out: list = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        p = Path(part)
        if p.is_dir():
            files = sorted(f for f in p.rglob("*")
                           if f.suffix.lower() in {".png", ".jpg", ".jpeg",
                                                   ".bmp", ".webp"}
                           and "_mask" not in f.stem
                           and "ground_truth" not in str(f))
        else:
            files = [p] if p.is_file() else []
        for f in files:
            try:
                with PILImage.open(f) as im:
                    w, h = im.size
            except Exception:  # noqa: BLE001 非图片文件跳过
                continue
            out.append({"id": None, "path": str(f), "width": w, "height": h})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8017")
    ap.add_argument("--category", required=True)
    ap.add_argument("--sample", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--paths", default="",
                    help="直测指定图（文件/目录/glob，逗号分隔）而非 DB 取样；"
                         "用于 2500²+ 大图产线口径实测（品类须已 prepare）")
    args = ap.parse_args()

    if args.paths:
        imgs = _expand_paths(args.paths)
        if not imgs:
            raise SystemExit(f"--paths 未找到图片: {args.paths}")
        print(f"[fullchain] 直测模式 {len(imgs)} 张，"
              f"尺寸 {min(i['width'] for i in imgs)}x"
              f"{min(i['height'] for i in imgs)} 起", flush=True)
    else:
        imgs = _pick_images(args.base, args.category, args.sample)

    a_cold_srv, a_cold_e2e = [], []      # A 口径首轮（冷缓存，产线首检）
    a_hot_srv, a_hot_e2e = [], []        # A 口径第 2 轮起（缓存命中，复检）
    b_srv, b_e2e = [], []                # B 口径零拷贝（每轮皆冷）
    errs: list = []

    # 相机帧缓冲预读（不计时）：模拟相机已把帧写入内存
    buffers = {}
    for it in imgs:
        try:
            buffers[str(it["path"])] = Path(str(it["path"])).read_bytes()
        except OSError as e:
            errs.append(f"读帧失败 {it['path']}: {e}")

    t0 = time.time()
    for rnd in range(1, args.rounds + 1):
        for it in imgs:
            iid = it["id"]
            # A：磁盘路径口径（读盘计时在引擎内）
            try:
                body = {"category": args.category, "with_heatmap": False}
                if iid is not None:
                    body["image_id"] = iid
                else:
                    body["path"] = str(it["path"])
                r = requests.post(
                    f"{args.base}/api/detect/image", json=body,
                    headers=_headers(), timeout=120)
                if r.status_code != 200:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                d = r.json()
                (a_cold_srv if rnd == 1 else a_hot_srv).append(
                    float(d["latency_ms"]))
                (a_cold_e2e if rnd == 1 else a_hot_e2e).append(
                    float(d.get("latency_e2e_ms") or d["latency_ms"]))
            except Exception as e:  # noqa: BLE001
                errs.append(f"[A r{rnd} id{iid}] {e}")
            # B：零拷贝内存口径（帧已在内存）
            buf = buffers.get(str(it["path"]))
            if buf is not None:
                try:
                    r = requests.post(
                        f"{args.base}/api/detect/frame",
                        files={"file": (f"frame_{rnd}.jpg", buf,
                                        "application/octet-stream")},
                        data={"category": args.category,
                              "with_heatmap": "false", "persist": "false"},
                        headers=_headers(), timeout=120)
                    if r.status_code != 200:
                        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                    d = r.json()
                    b_srv.append(float(d["latency_ms"]))
                    b_e2e.append(float(d.get("latency_e2e_ms")
                                       or d["latency_ms"]))
                except Exception as e:  # noqa: BLE001
                    errs.append(f"[B r{rnd} {it['path']}] {e}")
        print(f"[fullchain] 进度 第 {rnd}/{args.rounds} 轮完成 "
              f"({time.time() - t0:.0f}s, 错误 {len(errs)})", flush=True)

    report = {
        "category": args.category, "sample": len(imgs), "rounds": args.rounds,
        "images": [{"id": it["id"], "path": str(it["path"]),
                    "width": it.get("width"), "height": it.get("height")}
                   for it in imgs],
        "n_err": len(errs), "errors": errs[:20],
        "A_disk": {  # 磁盘路径口径（含读盘）
            "cold": {"server_ms": _stats(a_cold_srv),
                     "e2e_ms": _stats(a_cold_e2e)},
            "hot_cache_r2plus": {"server_ms": _stats(a_hot_srv),
                                 "e2e_ms": _stats(a_hot_e2e)},
        },
        "B_zero_copy": {  # 相机帧直入内存（无磁盘 IO，恒冷）
            "server_ms": _stats(b_srv), "e2e_ms": _stats(b_e2e),
        },
        "red_line_ms": RED_LINE_MS,
        "over_red_line": {
            "A_cold_e2e": sum(1 for v in a_cold_e2e if v > RED_LINE_MS),
            "B_zero_copy_e2e": sum(1 for v in b_e2e if v > RED_LINE_MS),
        },
    }
    # 结论摘要（终端一眼可见）
    summary = {
        "A_磁盘_冷_e2e_ms": report["A_disk"]["cold"]["e2e_ms"],
        "A_磁盘_热_e2e_ms": report["A_disk"]["hot_cache_r2plus"]["e2e_ms"],
        "B_零拷贝_e2e_ms": report["B_zero_copy"]["e2e_ms"],
        "B_零拷贝_server_ms": report["B_zero_copy"]["server_ms"],
    }
    log_dir = Path(__file__).resolve().parent.parent / "storage" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    out_path = log_dir / f"bench_fullchain_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"[fullchain] 报告已写 {out_path}", flush=True)
    over = report["over_red_line"]
    ok = not errs and not over["A_cold_e2e"] and not over["B_zero_copy_e2e"]
    print(f"[fullchain] 结论: {'PASS' if ok else 'FAIL'}"
          f"（错误 {len(errs)}，A冷e2e超红线 {over['A_cold_e2e']}，"
          f"B零拷贝e2e超红线 {over['B_zero_copy_e2e']}）", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
