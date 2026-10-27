"""M3 失效归因决策树（§7.7）：误检/漏检检查序列，驱动更新升级阶梯。

误检（正常→异常）：库覆盖 → 域差 → 路由权重 → 校准失效
漏检（异常→正常）：信号有无 → 特征复评 → 粒度 → 能力图谱/标签仲裁
三类归因判定标准（§7.7 可操作定义）：训练不足 / 特征不完备 / 算法边界。
"""
import numpy as np


def _norm_patch(feats):
    from .banks import _norm_patch as _np
    return _np(feats)


def attribute_failure(record, pipeline, mode="fp", item=None):
    """record: predict 记录（slot_scores/fused/raw/decision）；mode: fp|fn。
    返回结论 dict：{series:[检查步骤结论], verdict: 归因类别, next: 升级阶梯下一级}。"""
    slots = sorted(record["slot_scores"])
    cal = np.array([record["slot_scores"][n] for n in slots])   # 校准分 [0,1]
    raw = np.array([record["raw_scores"][n] for n in slots])
    series = []
    verdict = None
    if mode == "fp":                     # 误检：正常→异常
        # 1. 库覆盖：该样本与正常库距离（sim 低=未覆盖的新正常模式？）
        if item is not None and pipeline.handler is not None:
            nb = pipeline.handler.normal_bank
            sim = nb._sim_to(_norm_patch(item["feats"]))
            if sim is not None:
                max_sim = float(np.max(sim))
                series.append({"step": "bank_coverage", "max_sim": round(max_sim, 4),
                               "ok": max_sim >= nb.sim_thresh})
                if max_sim < nb.sim_thresh:
                    verdict = "new_normal_mode"     # 未覆盖 → 成簇入库（§7.2）
        # 2. 校准失效：正常流上 CDF 分数系统性偏高？（fused 与各槽位校准分是否普遍高）
        if verdict is None:
            series.append({"step": "calibration_drift",
                           "fused": round(record["fused"], 4),
                           "median_cal": round(float(np.median(cal)), 4)})
            if float(np.median(cal)) > 0.5:
                verdict = "calibration_drift"   # 重校准（§7.7 误检序列 4）
        if verdict is None:
            series.append({"step": "slot_stability",
                           "max_slot": slots[int(np.argmax(cal))],
                           "max_cal": round(float(np.max(cal)), 4)})
            verdict = "slot_noise"              # 噪声槽位主导 → 降权（§7.7 序列 3）
    else:                                # 漏检：缺陷→正常
        # 1. 信号有无：有无槽位给出正确方向的高分？
        top_slot = slots[int(np.argmax(cal))]
        top_cal = float(np.max(cal))
        has_signal = top_cal >= 0.5
        series.append({"step": "signal_presence", "top_slot": top_slot,
                       "top_cal": round(top_cal, 4), "ok": has_signal})
        if has_signal:
            # 2. 有信号但 fused 低 → 融合学坏 → 训练不足（路由微调/复训）
            series.append({"step": "fusion_dilution",
                           "fused": round(record["fused"], 4)})
            verdict = "undertrained_fusion"
        else:
            # 3. 无信号 → 粒度检查：该槽位 raw 分是否在更细尺度可见？（M3 无级联，直接边界）
            series.append({"step": "capability_scope",
                           "max_raw": round(float(np.max(raw)), 4)})
            # 4. E_open/能力图谱：无机理覆盖 → 算法边界 / 疑似标签错误（提交仲裁）
            series.append({"step": "label_arbitration",
                           "hint": "无槽位信号且无更细尺度 → 算法边界或标签错误"})
            verdict = "capability_boundary"
    # 升级阶梯（§7.7）：同类反馈累积 K 次自动升级（此处置评，触发在 FeedbackHandler）
    ladder = {"new_normal_mode": "库更新(成簇入库) → 路由微调 → 头复训",
              "calibration_drift": "在线回流重建 + 重估 CDF",
              "slot_noise": "查稳定性档案 → 裁剪或降权",
              "undertrained_fusion": "路由定向微调（难例参与 MIL 排序）",
              "capability_boundary": "评估新增机理槽位 / 人工仲裁"}
    return {"mode": mode, "series": series, "verdict": verdict,
            "next": ladder.get(verdict, "无")}
