# -*- coding: utf-8 -*-
"""
种子点自适应取色：在良品图上点击若干板面上的点，自动统计每个点邻域的
HSV 分布，生成 inRange 用的多段范围（自动处理红色跨 H=0/180 边界的情况），
取代手写固定阈值。生成的范围直接喂给 main_pipeline 的 custom 提取链。

取色窗口操作说明:
  鼠标左键    在板面上点一个种子点（不同颜色的区域各点几个）
  鼠标右键    删除离点击处最近的种子点
  + / -       放宽 / 收紧容差（实时预览绿色覆盖区变化）
  z           撤销上一个种子点
  r           清空全部种子点
  Enter / 空格 确认并返回结果
  Esc         取消（返回 None）
"""
import json
import os

import cv2
import numpy as np


# ==========================================
# 范围生成
# ==========================================
def _seed_stats(hsv_img, x, y, half):
    """统计种子点邻域 patch 的 HSV 分布。色相按圆周量处理：
    先求圆周均值 h0，再统计各像素相对 h0 的偏差，避免红色在 0/180 处断裂。"""
    h_img, w_img = hsv_img.shape[:2]
    x0, x1 = max(0, x - half), min(w_img, x + half + 1)
    y0, y1 = max(0, y - half), min(h_img, y + half + 1)
    patch = hsv_img[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32)
    hch, sch, vch = patch[:, 0], patch[:, 1], patch[:, 2]

    ang = hch * (np.pi / 90.0)  # OpenCV 的 H 取值 0~180，映射到 0~2pi
    h0 = (np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()) * 90.0 / np.pi) % 180.0
    dh = (hch - h0 + 90.0) % 180.0 - 90.0

    return {
        "h0": float(h0),
        "dh_lo": float(np.percentile(dh, 1)), "dh_hi": float(np.percentile(dh, 99)),
        "dh_std": float(dh.std()),
        "s_lo": float(np.percentile(sch, 1)), "s_hi": float(np.percentile(sch, 99)),
        "s_std": float(sch.std()),
        "v_lo": float(np.percentile(vch, 1)), "v_hi": float(np.percentile(vch, 99)),
        "v_std": float(vch.std()),
    }


def _ranges_from_stats(st, tol):
    """由邻域统计生成 HSV 范围。范围 = 1%~99% 分位区间再向外扩一个
    与邻域标准差和容差 tol 挂钩的余量；色相跨 0/180 边界时拆成两段。"""
    mh = 2.0 + tol * max(2.0, 2.0 * st["dh_std"])
    ms = 12.0 + tol * max(8.0, 2.0 * st["s_std"])
    mv = 12.0 + tol * max(8.0, 2.0 * st["v_std"])

    s_lo = int(max(0, st["s_lo"] - ms))
    s_hi = int(min(255, st["s_hi"] + ms))
    v_lo = int(max(0, st["v_lo"] - mv))
    v_hi = int(min(255, st["v_hi"] + mv))

    lo = st["h0"] + st["dh_lo"] - mh
    hi = st["h0"] + st["dh_hi"] + mh
    span = hi - lo
    if span >= 178.0:  # 色相几乎无区分度（灰白/低饱和区），放开全色相靠 S/V 约束
        return [((0, s_lo, v_lo), (180, s_hi, v_hi))]

    lo = lo % 180.0
    hi = lo + span
    if hi <= 180.0:
        return [((int(lo), s_lo, v_lo), (int(round(hi)), s_hi, v_hi))]
    # 跨越 H=0/180 边界，拆成两段取并集
    return [
        ((int(lo), s_lo, v_lo), (180, s_hi, v_hi)),
        ((0, s_lo, v_lo), (int(round(hi - 180.0)), s_hi, v_hi)),
    ]


def ranges_from_seeds(image, seeds, tol=1.0):
    """由种子点列表生成多段 HSV 范围（各种子的范围取并集）。
    image 为 BGR 良品图，seeds 为原图坐标 [(x, y), ...]。

    邻域 patch 尺寸自适应：金手指指条可能比"按图像尺寸取的 patch"更窄，
    固定大 patch 会把指缝背景混进统计，生成覆盖全图的垃圾范围。这里在
    几个候选尺寸中选 HSV 分布最均匀（标准差最小）的，自动收缩到点击
    目标色块内部。"""
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    ls = max(image.shape[:2])
    halves = sorted({4, max(4, int(round(ls * 0.003))), max(4, int(round(ls * 0.006)))})
    ranges = []
    for (x, y) in seeds:
        best = None
        for half in halves:
            st = _seed_stats(hsv, int(x), int(y), half)
            spread = 2.0 * st["dh_std"] + st["s_std"] + st["v_std"]
            if best is None or spread < best[0]:
                best = (spread, st)
        ranges.extend(_ranges_from_stats(best[1], tol))
    return ranges


def mask_from_ranges(hsv, ranges):
    """多段范围取并集生成二值掩膜（与 main_pipeline._mask_custom 等价）。"""
    mask = None
    for lower, upper in ranges:
        m = cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    return mask


def ranges_to_config(ranges):
    """[( (H,S,V), (H,S,V) ), ...] → 算法 config 用的 [[[H,S,V],[H,S,V]], ...]。"""
    return [[list(lo), list(hi)] for lo, hi in ranges]


def preview_from_seeds(image_bgr, seeds, tol=1.0, overlay_alpha=0.45,
                       draw_seeds=True, seed_color=(0, 0, 255)):
    """UI 非交互入口：种子点 → HSV 范围 → 掩膜 → 预览图。

    供自研点选 UI 在每次加点/改容差时调用；不弹窗、不写文件。
    返回 dict: ok, message, seeds, tol, ranges, custom_hsv_ranges,
               mask, preview, cover_ratio, board_pixels。
    """
    h, w = image_bgr.shape[:2]
    empty_mask = np.zeros((h, w), np.uint8)
    base = {
        "ok": False,
        "message": "",
        "seeds": [],
        "tol": float(tol),
        "ranges": [],
        "custom_hsv_ranges": [],
        "mask": empty_mask,
        "preview": image_bgr.copy(),
        "cover_ratio": 0.0,
        "board_pixels": 0,
    }
    if image_bgr is None or image_bgr.size == 0:
        base["message"] = "empty image"
        return base

    pts = [(int(x), int(y)) for x, y in (seeds or [])]
    pts = [(x, y) for x, y in pts if 0 <= x < w and 0 <= y < h]
    base["seeds"] = [list(p) for p in pts]
    if not pts:
        base["message"] = "no seeds"
        return base

    ranges = ranges_from_seeds(image_bgr, pts, float(tol))
    hsv = cv2.cvtColor(cv2.GaussianBlur(image_bgr, (5, 5), 0), cv2.COLOR_BGR2HSV)
    mask = mask_from_ranges(hsv, ranges)
    if mask is None:
        mask = empty_mask

    preview = image_bgr.copy()
    if np.any(mask):
        green = np.zeros_like(preview)
        green[:, :] = (0, 200, 0)
        blend = cv2.addWeighted(preview, 1.0 - overlay_alpha, green, overlay_alpha, 0)
        preview = np.where(mask[:, :, None] > 0, blend, preview)

    if draw_seeds:
        r = max(3, int(round(max(h, w) / 400)))
        for sx, sy in pts:
            cv2.circle(preview, (sx, sy), r, seed_color, 2)
            cv2.circle(preview, (sx, sy), 1, seed_color, -1)

    board_pixels = int(np.count_nonzero(mask))
    return {
        "ok": True,
        "message": "ok",
        "seeds": [list(p) for p in pts],
        "tol": float(tol),
        "ranges": ranges,
        "custom_hsv_ranges": ranges_to_config(ranges),
        "mask": mask,
        "preview": preview,
        "cover_ratio": float(board_pixels) / float(h * w),
        "board_pixels": board_pixels,
    }


# ==========================================
# 交互式取色窗口
# ==========================================
def pick_seed_ranges(image, init_tol=1.0, window_name="Seed Color Picker"):
    """弹窗交互取色。返回 {"seeds": [...], "tol": float, "ranges": [...]}，
    用户按 Esc 取消时返回 None。"""
    h, w = image.shape[:2]
    disp_scale = min(1.0, 1400.0 / max(h, w))
    disp_size = (int(w * disp_scale), int(h * disp_scale)) if disp_scale < 1.0 else (w, h)

    state = {"seeds": [], "tol": init_tol, "dirty": True}

    def on_mouse(event, mx, my, flags, param):
        ox, oy = int(mx / disp_scale), int(my / disp_scale)
        if event == cv2.EVENT_LBUTTONDOWN:
            state["seeds"].append((ox, oy))
            state["dirty"] = True
        elif event == cv2.EVENT_RBUTTONDOWN and state["seeds"]:
            d = [(sx - ox) ** 2 + (sy - oy) ** 2 for sx, sy in state["seeds"]]
            state["seeds"].pop(int(np.argmin(d)))
            state["dirty"] = True

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, on_mouse)
    print("🎯 取色窗口已打开：左键点板面取色，右键删点，+/- 调容差，Enter 确认，Esc 取消。")

    ranges = []
    frame = cv2.resize(image, disp_size, interpolation=cv2.INTER_AREA) if disp_scale < 1.0 \
        else image.copy()
    while True:
        if state["dirty"]:
            out = preview_from_seeds(image, state["seeds"], state["tol"])
            ranges = out["ranges"]
            frame = out["preview"]
            if disp_scale < 1.0:
                frame = cv2.resize(frame, disp_size, interpolation=cv2.INTER_AREA)
            info = (f"seeds={len(state['seeds'])}  tol={state['tol']:.2f}  "
                    f"cover={out['cover_ratio']:.1%}   "
                    f"[L]add [R]del [+/-]tol [z]undo [r]reset [Enter]OK [Esc]cancel")
            cv2.putText(frame, info, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, info, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 1, cv2.LINE_AA)
            state["dirty"] = False

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 32):  # Enter / 空格
            if not state["seeds"]:
                print("⚠️ 尚未选择任何种子点，请先左键取色，或按 Esc 取消。")
                continue
            cv2.destroyWindow(window_name)
            return {"seeds": state["seeds"], "tol": state["tol"], "ranges": ranges}
        if key == 27:  # Esc
            cv2.destroyWindow(window_name)
            return None
        if key in (ord("+"), ord("=")):
            state["tol"] = min(3.0, state["tol"] + 0.25)
            state["dirty"] = True
        elif key == ord("-"):
            state["tol"] = max(0.25, state["tol"] - 0.25)
            state["dirty"] = True
        elif key == ord("z") and state["seeds"]:
            state["seeds"].pop()
            state["dirty"] = True
        elif key == ord("r"):
            state["seeds"] = []
            state["dirty"] = True


# ==========================================
# 配置持久化：取一次色，之后直接复用
# ==========================================
def load_or_pick_seed_ranges(image, config_path, image_path=None, force_repick=False):
    """优先读取已保存的取色配置；没有（或 force_repick=True）时弹窗取色并保存。
    返回 [( (H低,S低,V低), (H高,S高,V高) ), ...]；用户取消返回 None。"""
    if not force_repick and os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            if image_path is None or cfg.get("image") == image_path:
                print(f"   -> [seed] 载入已保存的取色配置 ({len(cfg['seeds'])} 个种子, tol={cfg['tol']}): {config_path}")
                return [tuple(map(tuple, r)) for r in cfg["ranges"]]
            print("   -> [seed] 良品图已更换，重新取色。")
        except Exception as e:
            print(f"   -> [seed] 配置读取失败，将重新取色: {e}")

    result = pick_seed_ranges(image)
    if result is None:
        return None
    cfg = {
        "image": image_path,
        "seeds": [list(s) for s in result["seeds"]],
        "tol": result["tol"],
        "ranges": [[list(lo), list(hi)] for lo, hi in result["ranges"]],
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print(f"   -> [seed] 取色配置已保存: {config_path}（下次运行直接复用，删除该文件可重新取色）")
    return [tuple(map(tuple, r)) for r in result["ranges"]]


# ==========================================
# 方案 A：离线取色 CLI（勿在算法池 run() 内调用）
# ==========================================
# 用法:
#   python seed_extract.py <良品图> [-o seed_ranges.json]
#   或拖拽良品图到「取色工具.bat」
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="离线种子点取色工具：在良品图上点选板面颜色，导出检测用 JSON")
    parser.add_argument("image", help="良品模板图路径")
    parser.add_argument(
        "-o", "--output", default=None,
        help="输出 JSON 路径（默认：与图片同目录的 <图名>_seed_ranges.json）")
    parser.add_argument(
        "--tol", type=float, default=1.0,
        help="初始容差，默认 1.0（窗口内仍可用 +/- 调节）")
    args = parser.parse_args()

    img_path = os.path.abspath(args.image)
    img = cv2.imread(img_path)
    if img is None:
        raise SystemExit(f"无法读取图片: {img_path}")

    if args.output:
        out_path = os.path.abspath(args.output)
    else:
        stem, _ = os.path.splitext(img_path)
        out_path = stem + "_seed_ranges.json"

    result = pick_seed_ranges(img, init_tol=args.tol)
    if result is None:
        raise SystemExit("取色已取消，未写入文件。")

    cfg = {
        "image": img_path,
        "seeds": [list(s) for s in result["seeds"]],
        "tol": result["tol"],
        "ranges": ranges_to_config(result["ranges"]),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    print(f"已保存 {len(result['ranges'])} 段 HSV 范围 -> {out_path}")
    print("检测配置示例:")
    print(f'  board_color: "seed"')
    print(f'  seed_config_path: "{out_path}"')
