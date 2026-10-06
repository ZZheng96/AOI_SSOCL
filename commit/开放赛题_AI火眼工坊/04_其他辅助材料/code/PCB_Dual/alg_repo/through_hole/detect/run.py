"""插件焊点缺陷检测 - 命令行入口。

输入目录结构 (默认 input/):
  input/standard_roi/templ_<n>.png   标准图
  input/roi/<n>-<m>.png              测试图

用法:
  python run.py
  python run.py --input input --out outputs
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import cv2

# 本地 CLI：工作目录为 detect/ 时需能 import src/algorithms/...
_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from algorithms.pcb_through_hole import config as C
from algorithms.pcb_through_hole.pipeline import inspect
from algorithms.pcb_through_hole.solder_extract import build_template_assets
from algorithms.pcb_through_hole.template import TemplateModel
from algorithms.pcb_through_hole.tuning import active_void_preset
from algorithms.pcb_through_hole.yaml_config import load_if_exists, apply_yaml, default_config_path
from algorithms.pcb_through_hole.visualize import draw_annotation


IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
TEMPL_RE = re.compile(r"^templ_(\d+)$", re.IGNORECASE)
TEST_RE = re.compile(r"^(\d+)[-_].+$")


def load_image(path):
    """读图（中文路径/RGBA 兜底，2026-08-31 评审修复）：
    np.fromfile+imdecode 兼容非 ASCII 路径；OpenCV 读不出的 RGBA/特殊编码
    用 PIL 解码转 BGR，避免产线相机 RGBA 出图时链路直接挂。"""
    import numpy as np
    img = None
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size:
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except OSError:
        img = None
    if img is None:
        try:
            from PIL import Image
            with Image.open(str(path)) as im:
                img = cv2.cvtColor(np.array(im.convert("RGB")),
                                   cv2.COLOR_RGB2BGR)
        except Exception:
            img = None
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    return img


def index_standards(std_dir):
    table = {}
    for f in sorted(os.listdir(std_dir)):
        name, ext = os.path.splitext(f)
        if ext.lower() not in IMG_EXT:
            continue
        m = TEMPL_RE.match(name)
        if m:
            table[m.group(1)] = os.path.join(std_dir, f)
    return table


def gather_tests(test_dir):
    tests = []
    for f in sorted(os.listdir(test_dir)):
        name, ext = os.path.splitext(f)
        if ext.lower() not in IMG_EXT:
            continue
        m = TEST_RE.match(name)
        if m:
            tests.append((m.group(1), os.path.join(test_dir, f)))
    return tests


def main(argv=None):
    parser = argparse.ArgumentParser(description="插件焊点缺陷检测")
    parser.add_argument("--input", default="input", help="输入根目录")
    parser.add_argument("--std-sub", default="standard_roi", help="标准图子目录")
    parser.add_argument("--test-sub", default="roi", help="测试图子目录")
    parser.add_argument("--out", default="outputs", help="输出目录")
    parser.add_argument(
        "--config", default=None, metavar="PATH",
        help="YAML 配置文件 (默认若存在 config.yaml 则自动加载)",
    )
    args = parser.parse_args(argv)

    cfg_path = args.config
    if cfg_path:
        try:
            yaml_data = apply_yaml(cfg_path)
        except (FileNotFoundError, ImportError) as e:
            print(f"[错误] 配置文件: {e}")
            return 1
    else:
        yaml_data = load_if_exists()

    std_dir = os.path.join(args.input, args.std_sub)
    test_dir = os.path.join(args.input, args.test_sub)
    for d in (std_dir, test_dir):
        if not os.path.isdir(d):
            print(f"[错误] 目录不存在: {d}")
            return 1

    standards = index_standards(std_dir)
    tests = gather_tests(test_dir)
    if not standards or not tests:
        print("[错误] 未找到标准图或测试图")
        return 1

    std_cache: dict[str, TemplateModel] = {}
    needed_nums = {num for num, _ in tests if num in standards}
    for num in sorted(needed_nums):
        std_bgr = load_image(standards[num])
        try:
            mask, bridge_joints = build_template_assets(std_dir, num, std_bgr)
        except FileNotFoundError as e:
            print(f"[错误] {e}")
            return 1
        std_cache[num] = TemplateModel.build(
            std_bgr, roi_mask=mask, templ_id=num, bridge_joints=bridge_joints)

    ok_dir = os.path.join(args.out, "OK")
    ng_dir = os.path.join(args.out, "NG")
    review_dir = os.path.join(args.out, "REVIEW")
    for d in (ok_dir, ng_dir, review_dir):
        os.makedirs(d, exist_ok=True)

    void_preset = active_void_preset()
    insuf_preset = getattr(C, "INSUF_PRESET", "MED")
    cfg_note = cfg_path or (str(default_config_path()) if yaml_data else "内置默认")
    print(f"标准图 {len(standards)} 个, 测试图 {len(tests)} 张")
    print(f"  配置: {cfg_note}")
    print(f"  孔洞挡位: {void_preset}  |  少锡挡位: {insuf_preset}")

    n_ok = n_ng = n_review = n_skip = 0

    for num, path in tests:
        name = os.path.splitext(os.path.basename(path))[0]
        if num not in standards:
            print(f"  - {name:<12} => [跳过] 找不到标准图 templ_{num}")
            n_skip += 1
            continue

        std = std_cache[num]
        try:
            test_img = load_image(path)
            t0 = time.perf_counter()
            res = inspect(test_img, std, name=name)
            elapsed_ms = (time.perf_counter() - t0) * 1000
        except Exception as e:
            print(f"  - {name:<12} (templ_{num}) => [跳过] 检测异常: {e}")
            n_skip += 1
            continue

        annotated = draw_annotation(res)
        cov_note = f" cov={res.coverage:.2f}" if res.coverage is not None else ""
        if res.status == "待复检":
            n_review += 1
            cv2.imwrite(os.path.join(review_dir, f"{name}.png"), annotated)
            status = f"待复检 ({res.review_reason})"
        elif res.status == "NG" or (not res.status and res.is_defect):
            n_ng += 1
            cv2.imwrite(os.path.join(ng_dir, f"{name}.png"), annotated)
            names = "、".join(C.DEFECT_NAMES[i] for i in res.defect_ids)
            status = f"NG [{names}]"
        else:
            n_ok += 1
            cv2.imwrite(os.path.join(ok_dir, f"{name}.png"), annotated)
            status = "OK"

        print(f"  - {name:<12} (templ_{num}) => {status}{cov_note}  "
              f"检测 {elapsed_ms:.1f}ms  [{res.align_method}]")

    print(f"\n输出: {os.path.abspath(args.out)}")
    print(f"  OK={n_ok}  NG={n_ng}  待复检={n_review}  跳过={n_skip}")
    print("  目录: OK/ NG/ REVIEW/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
