"""全链路速度口径实测（不足清单 #5 补强，2026-08-30）。

同一批图（默认随机 8 张，符合"调优测试随机少量图"规则）对比两种产线形态：

  A. 磁盘路径口径  POST /api/detect/image (with_heatmap=False)
     -- 引擎内含 load_image 读盘+切块+前向+融合（读盘计时在内）。
     首轮=冷缓存（产线每图首检），第 2 轮起 _item_cache 命中（复检口径）。
  B. 零拷贝内存口径 POST /api/detect/frame
     -- 计时前先把帧 bytes 读进内存（模拟相机帧缓冲直入），
        imdecode+切块+前向+融合，全程无磁盘 IO；predict_frame
     天然不走特征缓存，每轮都是冷口径。

三层计时口径：model_path_ms = latency_ms（引擎 Pipeline.predict / predict_frame
调用段，含预处理/模型/融合；A 含读盘，B 不含 HTTP 解码及排队）；
e2e_ms = latency_e2e_ms（接口内检测段；B 当前从解码后开始，
不含 persist/alarm，均不含网络与排队）；
client_ms = 调用方实测（含排队与网络，仅作参照）。

通过标准（三层互不混用）：
  * 工程红线（恒判定）：A冷/B零拷贝 e2e 无超 1000ms 且无请求错误；
  * 赛题口径（--competition 追加）：B 零拷贝 model_path P95 低于设备预算
    （CUDA 可用 200ms；纯 CPU 按挑战目标 2000ms），max 仅报告不作通过线。
赛题模式强制 2500×2500 输入（--paths 或数据集过滤）、默认 5 轮预热，
报告 manifest 含 commit/python/torch/cuda/驱动/config_hash/input_size/
warmup/计时边界，可在 2060 机器上一键复测。未在指定硬件实测前，
任何 <200ms 结论都不得写进交付文档。
报告写 storage/logs/bench_fullchain_{时间戳}.json。

用法（需后端已启动且品类已 prepare）：
    python -m tests.bench.fullchain --category <品类名>
    python -m tests.bench.fullchain --category <品类名> --sample 12 --rounds 3
    python -m tests.bench.fullchain --category <品类名> --competition \
        --paths <2500x2500图目录> --warmup 5 --rounds 3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import requests

COMPETITION_BUDGET_MS = 200.0        # 赛题硬指标：2060 及以下 GPU 模型运行时间
CPU_CHALLENGE_BUDGET_MS = 2000.0     # 赛题挑战目标：无 GPU 时 CPU 模型运行时间
ENGINEERING_RED_LINE_MS = 1000.0     # 项目工程红线：单图端到端（与赛题口径分离）


def _headers() -> dict:
    key = os.environ.get("AOI_API_KEY", "")
    return {"X-API-Key": key} if key else {}


def _torch_manifest() -> dict:
    try:
        import torch
        return {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception as exc:
        return {"error": str(exc)}


def _stats(vals: list) -> dict:
    if not vals:
        return {}
    s = sorted(vals)
    return {"n": len(vals),
            "mean": round(statistics.mean(s), 1),
            "p50": round(statistics.median(s), 1),
            "p95": round(s[max(0, int(len(s) * 0.95) - 1)], 1),
            "p99": round(s[max(0, int(len(s) * 0.99) - 1)], 1),
            "max": round(s[-1], 1)}


def _driver_manifest() -> dict:
    """GPU 驱动/CUDA 驱动侧版本（best effort，无 nvidia-smi 时记 None）。"""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"], text=True, timeout=10).strip()
        return {"nvidia_smi": out.splitlines()[0] if out else None}
    except Exception:
        return {"nvidia_smi": None}


def _pick_images(base: str, category: str, n: int,
                 require_size: tuple | None = None) -> list:
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
    if require_size is not None:
        # 赛题口径：输入尺寸必须严格等于要求值（如 2500×2500）
        req = [it for it in ok
               if (it.get("width"), it.get("height")) == require_size]
        if len(req) < n:
            raise SystemExit(
                f"赛题口径要求 {require_size[0]}x{require_size[1]} 输入，"
                f"品类 {category} 仅有 {len(req)} 张（需要 {n} 张）——"
                f"请导入足量 {require_size[0]}x{require_size[1]} 图片，"
                f"或用 --paths 直测指定图")
        ok = req
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
    ap.add_argument("--sample", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--competition", action="store_true",
                    help="按赛题口径执行：严格 2500×2500 输入、至少100张、"
                         "预热5轮、GPU<200ms / CPU<2s 模型运行时间预算")
    ap.add_argument("--paths", default="",
                    help="直测指定图（文件/目录/glob，逗号分隔）而非 DB 取样；"
                         "用于 2500²+ 大图产线口径实测（品类须已 prepare）")
    ap.add_argument("--warmup", type=int, default=0,
                    help="计时前的预热轮数（不计时；--competition 默认 5）")
    args = ap.parse_args()
    require_size = None
    if args.competition:
        args.sample = max(args.sample, 100)
        require_size = (2500, 2500)
        if args.warmup == 0:
            args.warmup = 5

    if args.paths:
        imgs = _expand_paths(args.paths)
        if not imgs:
            raise SystemExit(f"--paths 未找到图片: {args.paths}")
        if require_size is not None:
            bad = [i for i in imgs
                   if (i["width"], i["height"]) != require_size]
            if bad:
                raise SystemExit(
                    f"赛题口径要求 2500x2500 输入，--paths 含 {len(bad)} 张"
                    f"其它尺寸（首张 {bad[0]['path']} "
                    f"{bad[0]['width']}x{bad[0]['height']}）")
        print(f"[fullchain] 直测模式 {len(imgs)} 张，"
              f"尺寸 {min(i['width'] for i in imgs)}x"
              f"{min(i['height'] for i in imgs)} 起", flush=True)
    else:
        imgs = _pick_images(args.base, args.category, args.sample, require_size)

    a_cold_srv, a_cold_e2e, a_cold_cli = [], [], []   # A 口径首轮（冷缓存，产线首检）
    a_hot_srv, a_hot_e2e, a_hot_cli = [], [], []      # A 口径第 2 轮起（缓存命中，复检）
    b_srv, b_e2e, b_cli = [], [], []                  # B 口径零拷贝（每轮皆冷）
    errs: list = []

    # 相机帧缓冲预读（不计时）：模拟相机已把帧写入内存
    buffers = {}
    for it in imgs:
        try:
            buffers[str(it["path"])] = Path(str(it["path"])).read_bytes()
        except OSError as e:
            errs.append(f"读帧失败 {it['path']}: {e}")

    # 预热（不计时）：覆盖模型装载/cudnn autotune/首图一次性成本
    for w in range(args.warmup):
        it = imgs[w % len(imgs)]
        try:
            body = {"category": args.category, "with_heatmap": False}
            if it["id"] is not None:
                body["image_id"] = it["id"]
            else:
                body["path"] = str(it["path"])
            requests.post(f"{args.base}/api/detect/image", json=body,
                          headers=_headers(), timeout=120)
        except Exception as e:  # noqa: BLE001 预热失败不阻断，正式计时仍会暴露
            print(f"[fullchain] 预热第 {w + 1} 轮失败（继续）: {e}", flush=True)
    if args.warmup:
        print(f"[fullchain] 预热 {args.warmup} 轮完成，开始正式计时", flush=True)

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
                t_req = time.perf_counter()
                r = requests.post(
                    f"{args.base}/api/detect/image", json=body,
                    headers=_headers(), timeout=120)
                cli_ms = (time.perf_counter() - t_req) * 1000  # 含排队+网络
                if r.status_code != 200:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                d = r.json()
                (a_cold_srv if rnd == 1 else a_hot_srv).append(
                    float(d["latency_ms"]))
                (a_cold_e2e if rnd == 1 else a_hot_e2e).append(
                    float(d.get("latency_e2e_ms") or d["latency_ms"]))
                (a_cold_cli if rnd == 1 else a_hot_cli).append(cli_ms)
            except Exception as e:  # noqa: BLE001
                errs.append(f"[A r{rnd} id{iid}] {e}")
            # B：零拷贝内存口径（帧已在内存）
            buf = buffers.get(str(it["path"]))
            if buf is not None:
                try:
                    t_req = time.perf_counter()
                    r = requests.post(
                        f"{args.base}/api/detect/frame",
                        files={"file": (f"frame_{rnd}.jpg", buf,
                                        "application/octet-stream")},
                        data={"category": args.category,
                              "with_heatmap": "false", "persist": "false"},
                        headers=_headers(), timeout=120)
                    cli_ms = (time.perf_counter() - t_req) * 1000
                    if r.status_code != 200:
                        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                    d = r.json()
                    b_srv.append(float(d["latency_ms"]))
                    b_e2e.append(float(d.get("latency_e2e_ms")
                                       or d["latency_ms"]))
                    b_cli.append(cli_ms)
                except Exception as e:  # noqa: BLE001
                    errs.append(f"[B r{rnd} {it['path']}] {e}")
        print(f"[fullchain] 进度 第 {rnd}/{args.rounds} 轮完成 "
              f"({time.time() - t0:.0f}s, 错误 {len(errs)})", flush=True)

    def _git_commit() -> str:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True,
                cwd=Path(__file__).resolve().parents[2]).strip()
        except Exception:
            return "unknown"

    def _file_hash(path: str) -> str:
        h = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            return h.hexdigest()
        except OSError:
            return "unavailable"

    torch_m = _torch_manifest()
    on_gpu = bool(torch_m.get("cuda_available"))
    model_budget = COMPETITION_BUDGET_MS if on_gpu else CPU_CHALLENGE_BUDGET_MS
    report = {
        "manifest": {
            "commit": _git_commit(),
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "torch": torch_m,
            "driver": _driver_manifest(),
            "device": "gpu" if on_gpu else "cpu",
            "config_hash": _file_hash(os.environ.get("AOI_CONFIG", "")),
            "input_size": ([imgs[0]["width"], imgs[0]["height"]]
                           if imgs else None),
            "require_size": list(require_size) if require_size else None,
            "warmup_rounds": args.warmup,
            "seed": 42,
            "timing_boundary": "model_path_ms=引擎predict段(含预处理/模型/融合)；"
                               "e2e_ms=接口检测段；client_ms=调用方实测(含排队网络)",
        },
        "category": args.category, "sample": len(imgs), "rounds": args.rounds,
        "images": [{"id": it["id"], "path": str(it["path"]),
                    "width": it.get("width"), "height": it.get("height")}
                   for it in imgs],
        "n_err": len(errs), "errors": errs[:20],
        "A_disk": {  # 磁盘路径口径（含读盘）
            "cold": {"server_ms": _stats(a_cold_srv),
                     "e2e_ms": _stats(a_cold_e2e),
                     "client_ms": _stats(a_cold_cli)},
            "hot_cache_r2plus": {"server_ms": _stats(a_hot_srv),
                                 "e2e_ms": _stats(a_hot_e2e),
                                 "client_ms": _stats(a_hot_cli)},
        },
        "B_zero_copy": {  # 相机帧直入内存（无磁盘 IO，恒冷）
            "server_ms": _stats(b_srv), "e2e_ms": _stats(b_e2e),
            "client_ms": _stats(b_cli),
        },
        "competition_budget_ms": COMPETITION_BUDGET_MS,
        "cpu_challenge_budget_ms": CPU_CHALLENGE_BUDGET_MS,
        "engineering_red_line_ms": ENGINEERING_RED_LINE_MS,
        "model_budget_ms": model_budget,
        "over_competition_budget": {
            "A_cold_model_path": sum(1 for v in a_cold_srv if v >= COMPETITION_BUDGET_MS),
            "B_zero_copy_model_path": sum(1 for v in b_srv if v >= COMPETITION_BUDGET_MS),
        },
        "over_cpu_challenge": {
            "B_zero_copy_model_path": sum(1 for v in b_srv if v >= CPU_CHALLENGE_BUDGET_MS),
        },
        "over_engineering_red_line": {
            "A_cold_e2e": sum(1 for v in a_cold_e2e if v > ENGINEERING_RED_LINE_MS),
            "B_zero_copy_e2e": sum(1 for v in b_e2e if v > ENGINEERING_RED_LINE_MS),
        },
    }
    # 结论摘要（终端一眼可见）
    summary = {
        "A_磁盘_冷_e2e_ms": report["A_disk"]["cold"]["e2e_ms"],
        "A_磁盘_热_e2e_ms": report["A_disk"]["hot_cache_r2plus"]["e2e_ms"],
        "B_零拷贝_e2e_ms": report["B_zero_copy"]["e2e_ms"],
        "B_零拷贝_server_ms": report["B_zero_copy"]["server_ms"],
    }
    log_dir = Path(__file__).resolve().parents[2] / "storage" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    out_path = log_dir / f"bench_fullchain_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"[fullchain] 报告已写 {out_path}", flush=True)
    over_eng = report["over_engineering_red_line"]
    eng_ok = (not errs and not over_eng["A_cold_e2e"]
              and not over_eng["B_zero_copy_e2e"])
    criterion = (f"工程红线：A冷/B零拷贝 e2e P99 以下无超 "
                 f"{ENGINEERING_RED_LINE_MS:.0f}ms 且无请求错误")
    ok = eng_ok
    if args.competition:
        # 赛题正式通过标准：B 零拷贝 model_path（引擎 predict 段）P95 低于
        # 设备预算（GPU 200ms / CPU 挑战 2000ms）；max 仅报告不作为通过线。
        p95 = report["B_zero_copy"]["server_ms"].get("p95", float("inf"))
        comp_ok = p95 < model_budget
        criterion += (f"；赛题口径：B零拷贝model_path P95 {p95}ms < "
                      f"{model_budget:.0f}ms（{'GPU' if on_gpu else 'CPU挑战'}预算）")
        ok = ok and comp_ok
    print(f"[fullchain] 结论: {'PASS' if ok else 'FAIL'}（{criterion}；"
          f"错误 {len(errs)}，A冷e2e超红线 {over_eng['A_cold_e2e']}，"
          f"B零拷贝e2e超红线 {over_eng['B_zero_copy_e2e']}）", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
