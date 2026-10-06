"""M11e 长稳/并发压测（评审自检 §11.1）：对运行中的后端并发打 /api/detect/image。

用法（需后端已启动并完成模型准备）：
    python -m tests.bench.stability --category <品类名> --rounds 200 --concurrency 4
    python -m tests.bench.stability --category <品类名> --rounds 50 --concurrency 8 --sample 12
    # 时长模式（2026-08-30 新增，24h 长稳实测入口）：
    python -m tests.bench.stability --category <品类名> --duration-min 1440 --concurrency 4

口径：
- 从 /api/images 随机取样 --sample 张（少量图轮询，不整数据集扫，符合调优规则）；
- --concurrency 个线程循环发检测请求，共 --rounds 次（或持续 --duration-min 分钟）；
- 每 10% 打印进度（长测必须进度信息）；
- 双延迟口径（2026-08-25 M12c 实测修正，2026-08-30 补 e2e）：
  * server_ms = 响应里的 latency_ms（引擎单图处理耗时，赛题检测时间口径）；
  * e2e_ms    = latency_e2e_ms（接口内全链路：解码+推理）；
  * 1s 红线（用户 2026-08-19）按 e2e_ms 判定；
  * wall_ms   = 客户端墙钟（含引擎品类锁排队）--仅作并发排队观测，不判红线；
- 内存监控（2026-08-30 新增，长稳核心指标）：psutil 可用时自动定位后端进程
  （按监听端口 / --server-pid），每个进度点采样 RSS_MB，报告首/峰/末值
  与增长量（泄漏判定原料）；
- 报告写 storage/logs/bench_stability_{时间戳}.json 并打印结论。

注意：/api/detect/image 服务端并发由引擎品类级锁串行化（同品类），并发下
墙钟含排队（4 并发均摊 ≈ 2.5×单图处理），并发压测主要验证排队稳定性与
长稳（内存/句柄泄漏、异常恢复），吞吐上限应看 server_ms。
本压测复检同图会命中引擎特征缓存（复检工位口径）；产线首检口径看
tests/bench/fullchain.py（python -m tests.bench.fullchain）。
鉴权开启时通过环境变量 AOI_API_KEY 传密钥。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import threading
import time
from pathlib import Path

import requests

RED_LINE_MS = 1000.0  # 单图耗时红线（用户明确，2026-08-19）


def _headers() -> dict:
    key = os.environ.get("AOI_API_KEY", "")
    return {"X-API-Key": key} if key else {}


def _pick_images(base: str, category: str, n: int) -> list:
    r = requests.get(f"{base}/api/images",
                     params={"category": category, "page_size": 500},
                     headers=_headers(), timeout=30)
    r.raise_for_status()
    items = (r.json() or {}).get("items", [])
    # 排除掩膜/真值图（ground_truth/*_mask.png 不是检测对象，M12c 实机踩坑）
    ids = [it["id"] for it in items
           if it.get("id") and "ground_truth" not in str(it.get("path", ""))
           and "_mask" not in Path(str(it.get("path", ""))).stem]
    if not ids:
        raise SystemExit(f"品类 {category} 无图片，请先导入数据")
    random.seed(42)
    random.shuffle(ids)
    return ids[:n]


def _find_server_proc(base: str, explicit_pid: int):
    """定位后端进程（内存监控用）。优先 --server-pid，其次按监听端口，
    再退化为 cmdline 含 server.py。找不到/无 psutil 返回 None。"""
    try:
        import psutil
    except ImportError:
        print("[bench] psutil 未安装，内存监控禁用（pip install psutil）",
              flush=True)
        return None
    if explicit_pid:
        try:
            return psutil.Process(explicit_pid)
        except Exception:  # noqa: BLE001
            return None
    port = int(base.rsplit(":", 1)[-1])
    try:
        for c in psutil.net_connections(kind="tcp"):
            if c.status == "LISTEN" and c.laddr and c.laddr.port == port:
                if c.pid:
                    try:
                        return psutil.Process(c.pid)
                    except Exception:  # noqa: BLE001
                        pass
    except Exception:  # noqa: BLE001 权限不足等
        pass
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmd = " ".join(p.info["cmdline"] or [])
            if "server.py" in cmd:
                return p
        except Exception:  # noqa: BLE001
            continue
    return None


def _rss_mb(proc) -> float | None:
    try:
        return round(proc.memory_info().rss / (1 << 20), 1)
    except Exception:  # noqa: BLE001 进程退出等
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8017")
    ap.add_argument("--category", required=True)
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--duration-min", type=float, default=0.0,
                    help="时长模式：持续 N 分钟（优先于 --rounds；24h=1440）")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--sample", type=int, default=8)
    ap.add_argument("--server-pid", type=int, default=0,
                    help="后端进程 PID（内存监控；缺省自动按端口定位）")
    args = ap.parse_args()

    ids = _pick_images(args.base, args.category, args.sample)
    duration_mode = args.duration_min > 0
    if duration_mode:
        print(f"[bench] 取样 {len(ids)} 张图，{args.concurrency} 并发 × "
              f"{args.duration_min} 分钟时长模式", flush=True)
    else:
        print(f"[bench] 取样 {len(ids)} 张图，{args.concurrency} 并发 × "
              f"{args.rounds} 轮 = {args.concurrency * args.rounds} 次检测",
              flush=True)

    proc = _find_server_proc(args.base, args.server_pid)
    rss_first = _rss_mb(proc) if proc else None
    rss_peak = rss_first
    if rss_first is not None:
        print(f"[bench] 内存监控：后端 PID {proc.pid}，初始 RSS "
              f"{rss_first} MB", flush=True)

    lat: list = []        # server_ms（引擎处理，赛题检测时间口径）
    e2e: list = []        # latency_e2e_ms（接口内全链路，1s 红线口径）
    wall: list = []       # wall_ms（客户端墙钟，含品类锁排队）
    errs: list = []
    over: list = []
    lock = threading.Lock()
    done = [0]
    stop_at = (time.time() + args.duration_min * 60) if duration_mode else None
    t0 = time.time()

    def worker(wi: int):
        i = 0
        while True:
            if stop_at is not None and time.time() >= stop_at:
                return
            if stop_at is None and i >= args.rounds:
                return
            image_id = ids[(wi * 1000 + i) % len(ids)]
            t1 = time.time()
            try:
                r = requests.post(f"{args.base}/api/detect/image",
                                  json={"image_id": image_id,
                                        "category": args.category,
                                        "with_heatmap": False},
                                  headers=_headers(), timeout=60)
                dt = (time.time() - t1) * 1000
                if r.status_code != 200:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                srv = float(r.json().get("latency_ms") or dt)
                e2e_v = float(r.json().get("latency_e2e_ms") or srv)
                with lock:
                    lat.append(srv)
                    e2e.append(e2e_v)
                    wall.append(dt)
                    if e2e_v > RED_LINE_MS:
                        over.append((image_id, round(e2e_v, 1)))
            except Exception as e:  # noqa: BLE001 压测需记录全部异常
                with lock:
                    errs.append(str(e))
            i += 1
            with lock:
                done[0] += 1
                n_done = done[0]
                # 进度：轮次模式按 10% 步进；时长模式每 20 次一报
                tick = (n_done % max(1, (args.concurrency * args.rounds) // 10) == 0) \
                    if not duration_mode else ((n_done % 20) == 0)
                if tick:
                    rss_now = _rss_mb(proc) if proc else None
                    if rss_now is not None:
                        nonlocal rss_peak
                        if rss_peak is None or rss_now > rss_peak:
                            rss_peak = rss_now
                    print(f"[bench] 进度 {n_done} 次 "
                          f"({time.time() - t0:.0f}s, 错误 {len(errs)}"
                          f"{f', RSS {rss_now} MB' if rss_now is not None else ''})",
                          flush=True)

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(args.concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    rss_last = _rss_mb(proc) if proc else None
    if rss_last is not None and (rss_peak is None or rss_last > rss_peak):
        rss_peak = rss_last

    report = {
        "category": args.category, "rounds": args.rounds,
        "duration_min": args.duration_min,
        "concurrency": args.concurrency, "total": done[0],
        "elapsed_s": round(time.time() - t0, 1),
        "n_ok": len(lat), "n_err": len(errs),
        "errors": errs[:20],
        "over_red_line": {"count": len(over), "samples": over[:20]},
    }
    if lat:
        lat_sorted = sorted(lat)
        report["latency_ms"] = {
            "p50": round(statistics.median(lat_sorted), 1),
            "p95": round(lat_sorted[int(len(lat_sorted) * 0.95) - 1], 1),
            "max": round(lat_sorted[-1], 1),
            "mean": round(statistics.mean(lat_sorted), 1),
        }
    if e2e:
        e2e_sorted = sorted(e2e)
        report["latency_e2e_ms"] = {
            "p50": round(statistics.median(e2e_sorted), 1),
            "p95": round(e2e_sorted[int(len(e2e_sorted) * 0.95) - 1], 1),
            "max": round(e2e_sorted[-1], 1),
            "mean": round(statistics.mean(e2e_sorted), 1),
        }
    if wall:
        wall_sorted = sorted(wall)
        report["wall_ms_queue_included"] = {
            "p50": round(statistics.median(wall_sorted), 1),
            "p95": round(wall_sorted[int(len(wall_sorted) * 0.95) - 1], 1),
            "max": round(wall_sorted[-1], 1),
            "mean": round(statistics.mean(wall_sorted), 1),
        }
    if rss_first is not None:
        report["server_rss_mb"] = {"first": rss_first, "peak": rss_peak,
                                   "last": rss_last,
                                   "growth_mb": (round(rss_last - rss_first, 1)
                                                 if rss_last is not None else None),
                                   "pid": proc.pid}
    log_dir = Path(__file__).resolve().parents[2] / "storage" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    out_path = log_dir / f"bench_stability_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"[bench] 报告已写 {out_path}", flush=True)
    ok = not errs and not over
    print(f"[bench] 结论: {'PASS' if ok else 'FAIL'}"
          f"（错误 {len(errs)}，e2e 超 1s 红线 {len(over)}）", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
