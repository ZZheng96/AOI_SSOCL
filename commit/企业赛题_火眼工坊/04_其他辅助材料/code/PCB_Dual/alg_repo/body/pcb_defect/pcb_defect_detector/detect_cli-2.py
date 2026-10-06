from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pcb_defect_detector import PCBDefectDetector
from pcb_defect_detector.image_io import imread_unicode


def main() -> int:
    ap = argparse.ArgumentParser(description="PCB 元件缺陷检测命令行")
    ap.add_argument("--template", required=True, help="模板图（金板）路径")
    ap.add_argument("--test", required=True, help="待检图（来料）路径")
    ap.add_argument("--alg", default="", help="算法 code，逗号分隔；不传=全部 7 项")
    ap.add_argument("--out", default="", help="结果图保存目录；不传=不保存")
    ap.add_argument("--config", default="", help='覆盖阈值的 JSON，如 \'{"size_tol_ratio":0.3}\'（PowerShell 下建议改用 --config-file 避免引号转义）')
    ap.add_argument("--config-file", default="", help="覆盖阈值的 JSON 文件路径（内容如 {\"pos_tol_mm\":0.5}）")
    ap.add_argument("--warmup", action="store_true",
                    help="正式计时前先跑一次预热（建池 / numpy 预热），不计入耗时")
    args = ap.parse_args()

    tpl_path = Path(args.template)
    test_path = Path(args.test)
    if not tpl_path.is_file():
        print(f"[ERR] 模板图不存在: {tpl_path}")
        return 1
    if not test_path.is_file():
        print(f"[ERR] 待检图不存在: {test_path}")
        return 1

    config = None
    if args.config_file:
        cf = Path(args.config_file)
        if not cf.is_file():
            print(f"[ERR] 配置文件不存在: {cf}")
            return 1
        try:
            with open(cf, "r", encoding="utf-8-sig") as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            print(f"[ERR] --config-file 不是合法 JSON: {e}")
            return 1
    elif args.config:
        try:
            config = json.loads(args.config)
        except json.JSONDecodeError as e:
            print(f"[ERR] --config 不是合法 JSON: {e}")
            return 1

    # 图只从磁盘读一次，之后传内存数组给各算法，避免重复 IO
    tpl = imread_unicode(str(tpl_path))
    test = imread_unicode(str(test_path))
    if tpl is None:
        print(f"[ERR] 模板图读取失败: {tpl_path}")
        return 1
    if test is None:
        print(f"[ERR] 待检图读取失败: {test_path}")
        return 1
    print(f"图片尺寸: 模板 {tpl.shape[1]}x{tpl.shape[0]}, 待检 {test.shape[1]}x{test.shape[0]}")

    det = PCBDefectDetector()
    algs = det.list_algorithms()
    print(f"已加载 {len(algs)} 个算法: {algs}")

    # 预先构建算法池，把这部分一次性开销移出计时
    t_pool = time.perf_counter()
    det.available_algorithms()
    pool_ms = (time.perf_counter() - t_pool) * 1000.0
    print(f"(算法池加载完成，耗时 {pool_ms:.1f}ms，不计入下方单算法统计)")

    if args.alg.strip():
        codes = [c.strip() for c in args.alg.split(",") if c.strip()]
    else:
        codes = list(algs.keys())

    out_dir = Path(args.out) if args.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"模板图: {tpl_path}")
    print(f"待检图: {test_path}")
    print(f"待跑算法: {codes}")
    print("-" * 72)

    # 预热：触发算法池构建，正式计时不含这部分一次性开销
    if args.warmup:
        t0 = time.perf_counter()
        _ = det.detect(codes[0], tpl, test, config=config)
        print(f"(预热完成，耗时 {(time.perf_counter()-t0)*1000:.1f}ms，不计入下方统计)\n")

    t_total = time.perf_counter()
    ng_count = 0
    for code in codes:
        t0 = time.perf_counter()
        r = det.detect(code, tpl, test, config=config)
        wall_ms = (time.perf_counter() - t0) * 1000.0
        # r.cost_time 是算法内部计时；wall_ms 此处≈算法耗时（图已在内存，无磁盘 IO）
        flag = "NG " if r.is_ng else ("OK " if r.is_ok else "ERR")
        if r.is_ng:
            ng_count += 1
        defect_brief = ", ".join(
            f"{d.label}@({d.x},{d.y}) {d.width}x{d.height} conf={d.confidence:.2f}"
            for d in r.defects
        ) or "-"
        print(f"[{flag}] {code:28s} 缺陷数={r.defect_count} "
              f"算法耗时={r.cost_time*1000:7.2f}ms 总耗时={wall_ms:7.2f}ms")
        if r.has_error:
            print(f"      错误: {r.error}")
        if r.defects:
            print(f"      缺陷: {defect_brief}")
        if out_dir and r.output_image is not None:
            out_png = out_dir / f"{code}.png"
            r.save_image(str(out_png))
            print(f"      结果图: {out_png}")

    total_ms = (time.perf_counter() - t_total) * 1000.0
    print("-" * 72)
    print(f"完成: 共 {len(codes)} 个算法, 检出 NG {ng_count} 项, 合计耗时 {total_ms:.2f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
