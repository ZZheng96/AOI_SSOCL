"""分层输出 L2/L3/L4（§6.2）：检测框 + 像素级掩码 + 类型归因 + 追溯

L1 判定已在 Decider 完成。本模块在其上叠加：
  L2 定位：融合热力图 → 阈值化 + 连通域检测框 + 像素级掩码
  L3 类型归因：槽位签名 × 固定映射矩阵（规则化，非监督分类器）
  L4 追溯：主导槽位 + 组级分数分解 + router_w

诚实边界（§6.2）：掩码粒度为 patch 级（tile 内 14-28px），
类型归因是可解释规则推断，每个标签必须附槽位证据。
"""
import numpy as np
import cv2


# §6.2 L3 槽位→类型映射矩阵（规则化归因，证据附在每个标签上）
SLOT_TYPE_MAP = {
    "layout": ["尺寸偏差", "缺件少件", "逻辑错误"],
    "color": ["色彩变化"],
    "blob": ["常见外观缺陷"],
    "disc": ["常见外观缺陷"],
    "trad": ["常见外观缺陷"],
    "sem": ["常见外观缺陷"],
    "inp": ["常见外观缺陷"],
}


def build_heatmap_mask(heatmaps, image_size, thresh, weight_by_slot=None):
    """槽位热力图 → 融合热力图 + 像素级掩码（L2 定位）。

    heatmaps: dict[slot_name -> list[(G,G) ndarray]]（每 tile 一张，None 表示无热力图）
    融合：每槽位 tile max 拼合到全图坐标，跨槽位按 weight_by_slot 加权平均（默认等权）。
    掩码：heatmap >= thresh，形态学开闭降噪。
    返回 (fused_heatmap HxW float, mask HxW uint8, per_slot_maps dict)
    """
    h, w = image_size
    per_slot = {}
    for name, hms in heatmaps.items():
        if not hms or hms[0] is None:
            continue
        slot_map = np.zeros((h, w), dtype=np.float32)
        for i, hm in enumerate(hms):
            if hm is None:
                continue
            hm = np.asarray(hm, dtype=np.float32)
            # 当前实现：单 tile 整图，直接 resize 到全图
            r = cv2.resize(hm, (w, h), interpolation=cv2.INTER_LINEAR)
            slot_map = np.maximum(slot_map, r)
        per_slot[name] = slot_map
    if not per_slot:
        return None, None, {}
    # U102（2026-08-24）：融合前逐槽位 min-max 归一化——各槽位热力图尺度不一
    # （shead 统计量 3-11 vs sem 距离 0-0.5），直接 mean 时 shead 主导把 fused 整体
    # 拉高 → mask_thresh=0.5 全图覆盖（PRO 假 1.0）。归一化后各槽位 [0,1] 等权。
    for n in per_slot:
        v = per_slot[n]
        vmin, vmax = float(v.min()), float(v.max())
        if vmax > vmin + 1e-6:
            per_slot[n] = (v - vmin) / (vmax - vmin)
        else:
            per_slot[n] = np.zeros_like(v)
    names = sorted(per_slot)
    if weight_by_slot:
        w_vec = np.array([weight_by_slot.get(n, 1.0) for n in names], dtype=np.float32)
        w_vec = w_vec / max(w_vec.sum(), 1e-8)
        fused = np.zeros((h, w), dtype=np.float32)
        for n, wi in zip(names, w_vec):
            fused += per_slot[n] * wi
    else:
        fused = np.stack([per_slot[n] for n in names]).mean(axis=0)
    mask = (fused >= thresh).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return fused, mask, per_slot


def boxes_from_mask(mask, min_area=16):
    """像素级掩码 → 连通域检测框（L2）。"""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area >= min_area:
            boxes.append({"bbox": [int(x), int(y), int(x + w), int(y + h)],
                          "area": int(area)})
    return boxes


def attribute_type(slot_scores, per_slot_maps, mask, image_size, firing_thresh=0.5,
                   layout_violation=None):
    """L3 类型归因：槽位签名 × 映射矩阵（规则化，证据附在每个标签上）。

    layout_violation（U94，2026-08-24）：layout 槽位内部的主导违例类型（缺件少件/
    尺寸偏差/逻辑错误），由 _hier 从 LayoutSlot._last_violation 传入——替代原
    _layout_subtype 掩码启发式（U93 缺件归因 0% 的根因：layout 有缺件信号但归因层
    用了错误的掩码连通域启发式）。传入 None 时回退旧启发式。

    返回 list[dict]: [{"type": 标签, "confidence": float, "evidence": str}]
    低置信（所有槽位分都低）如实报"未分类异常"。
    """
    if not slot_scores:
        return []
    mx = max(slot_scores.values())
    if mx < firing_thresh:
        return [{"type": "未分类异常", "confidence": 0.0,
                 "evidence": "所有槽位分均低，无可归类的槽位签名"}]
    types = []
    for name, score in slot_scores.items():
        if score < firing_thresh:
            continue
        cand = SLOT_TYPE_MAP.get(name, [])
        if not cand:
            continue
        if name == "layout":
            detail = layout_violation or _layout_subtype(mask, image_size)
            types.append({"type": detail, "confidence": round(float(score), 3),
                          "evidence": f"layout firing (score={score:.3f}), "
                                      f"违例={detail}, 掩码: {_mask_spread_desc(mask)}"})
        else:
            types.append({"type": cand[0], "confidence": round(float(score), 3),
                          "evidence": f"{name} firing (score={score:.3f})"})
    # 同类型去重：只保留最高置信的槽位证据（多槽位报同一"外观缺陷"→ 合并）
    seen = {}
    for t in types:
        cur = seen.get(t["type"])
        if cur is None or t["confidence"] > cur["confidence"]:
            seen[t["type"]] = t
    # U94 结论（2026-08-24）：不做"专属槽位优先"tie-break——实测该 tie-break 使
    # 色彩归因 0→0.475 但外观归因 0.971→0.773（layout/color 对部分外观缺陷也
    # fire 饱和 2.0），总归因准确率 0.868→0.721 净负。归因精确性受限于特征非
    # 正交性（layout Otsu 对线缆不敏感、HSV 全局直方图不色彩专属），属图像级
    # 槽位体系边界，如实保留简单 confidence 排序。
    types = sorted(seen.values(), key=lambda x: -x["confidence"])
    return types if types else [{"type": "未分类异常", "confidence": 0.0,
                                  "evidence": "无槽位超过 firing 阈值"}]


def _layout_subtype(mask, image_size):
    """layout 槽位细分：用掩码分布判断是尺寸/缺件/逻辑错误。"""
    if mask is None or mask.sum() == 0:
        return "尺寸偏差"
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n - 1 <= 1:
        return "缺件少件"          # 单一大块缺失
    h, w = image_size
    spread = float(min(mask.sum() / (h * w) * n, 1.0))
    if spread > 0.4:
        return "尺寸偏差"          # 掩码弥散 → 整体尺寸异常
    return "逻辑错误"              # 多个分散小块 → 顺序/错位


def _mask_spread_desc(mask):
    if mask is None or mask.sum() == 0:
        return "空"
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    return f"连通域 {n - 1} 个，覆盖 {int(mask.sum())}px"


def trace_output(path, decision, fused, slot_scores, router_w, boxes, types,
                 heatmap_path=None):
    """L4 追溯落盘记录（§6.2）。"""
    return {"path": path, "decision": decision, "fused": round(fused, 4),
            "slot_scores": {k: round(v, 4) for k, v in slot_scores.items()},
            "router_w": {k: round(v, 4) for k, v in router_w.items()},
            "boxes": boxes, "types": types,
            "heatmap_ref": heatmap_path}
