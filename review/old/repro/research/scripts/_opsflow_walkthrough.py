# -*- coding: utf-8 -*-
"""操作流走查（对齐 demo5用户操作流.md §3-§7）：用 MPDD 数据集真实跑一遍用户操作。

模拟角色：部署工程师(§3 冷启动) → 操作员(§4 送检/看判定) → 质检管理员(§5 反馈/§6 选样/§7 追溯)
调用方式与文档完全一致：build() -> Pipeline() -> fit() -> predict() / enable_ssocl+feedback /
active_select / attach_trace_api 查询。
"""
import os
import sys
import json
import time
import yaml

sys.stdout.reconfigure(line_buffering=True)          # 长任务进度实时可见
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import cv2

from scripts.m0_baseline import build               # 文档 §4.1 同款入口
from src.eval.offline import Pipeline
from src.data import mvtec_like

CAT = os.environ.get("OPS_CAT", "bracket_black")
SSOCL_CFG = {                                       # 与 m3_ssocl_sim.py 同款
    "banks": {"ext_max": 64, "cluster_k": 3, "sim_thresh": 0.75,
              "intercept_topk": 8, "intercept_boost": 0.25, "intercept_sim": 0.15},
    "gate": {"tol": 0.05, "anchor_size": 20},
    "active": {"top_n": 10},
    "incubate": {"min_samples": 8},
    "router_finetune": {"trigger": 10, "lr": 1e-4, "epochs": 5},
}


def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = yaml.safe_load(open(os.path.join(root_dir, "configs", "m0.yaml"), encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = cfg["datasets"]["mpdd"]
    out_dir = cfg["output_dir"]
    issues = []

    # ================= §3 冷启动训练（每品类一次） =================
    bundle = mvtec_like.load_category(
        root, CAT, n_init_normal=cfg["protocol"]["n_init_normal"],
        n_init_defect=cfg["protocol"]["n_init_defect"],
        seed=cfg["seed"], max_test=40)
    print(f"\n[§3 冷启动] {CAT}: init_normal={len(bundle['init_normal'])} "
          f"init_defect={len(bundle['init_defect'])} test={len(bundle['test'])}")
    backbone, slots = build(cfg, device)
    pipe = Pipeline(cfg, backbone, slots)
    t0 = time.time()
    pipe.fit(bundle)
    print(f"[§3] fit={pipe.fit_seconds:.1f}s | train_auroc={pipe.train_auroc} | "
          f"tau_high={pipe.decider.tau_high:.4f} tau_gray={pipe.decider.tau_gray:.4f}")
    if not (0 < pipe.decider.tau_gray < pipe.decider.tau_high <= 1.0):
        issues.append(f"§3 阈值异常: tau_high={pipe.decider.tau_high} tau_gray={pipe.decider.tau_gray}")

    # ================= §7 追溯挂载（部署时就位，后续送检自动留痕） =================
    trace_dir = os.path.join(out_dir, "trace_api_opsflow")
    if os.path.exists(os.path.join(trace_dir, "inferences.jsonl")):   # 重跑清旧留痕
        os.remove(os.path.join(trace_dir, "inferences.jsonl"))
    api = pipe.attach_trace_api(trace_dir)
    print(f"\n[§7 追溯] 已挂载 -> {trace_dir}")

    # ================= §4 日常检测（图片送检） =================
    demo_path = bundle["test"][0][0]
    r0 = pipe.predict(demo_path)                      # 文档 §4.1 单张 predict
    print(f"\n[§4 送检] 单张示例 {os.path.basename(demo_path)}: "
          f"decision={r0['decision']} fused={r0['fused']:.4f}")
    print(f"[§4] predict 返回字段: {sorted(r0.keys())}")
    print(f"[§4]   单张示例: boxes={len(r0.get('boxes') or [])} 个 "
          f"types={r0.get('types')} open_alert={r0.get('open_alert')}")
    for field in ("decision", "boxes", "types", "slot_scores"):
        if field not in r0:
            issues.append(f"§4 predict 缺字段: {field}")
    if "open_alert" not in r0:
        issues.append("§4 predict 缺 open_alert 字段")

    print(f"\n[§4 送检] {len(bundle['test'])} 张（模拟操作员逐张送检）...")
    stats = {"normal": 0, "gray": 0, "anomaly": 0}
    per_img = []
    t0 = time.time()
    for p, y in bundle["test"]:
        rr = pipe.predict(p)
        stats[rr["decision"]] += 1
        ms = None
        per_img.append({"path": p, "label": y, "decision": rr["decision"],
                        "fused": round(rr["fused"], 4),
                        "n_boxes": len(rr.get("boxes") or []),
                        "types": [t.get("type") for t in (rr.get("types") or [])]})
    dt = time.time() - t0
    avg_ms = dt / len(bundle["test"]) * 1000
    print(f"[§4] 判定分布: {stats} | 平均 {avg_ms:.0f}ms/图")
    if avg_ms > 1000:
        issues.append(f"§4 单图耗时超红线: {avg_ms:.0f}ms (>1s)")

    # 按真值交叉检查（供反馈环节取材）
    correct_normal = [x for x in per_img if x["decision"] == "normal" and x["label"] == 0]
    correct_defect = [x for x in per_img if x["decision"] == "anomaly" and x["label"] == 1]
    missed = [x for x in per_img if x["decision"] in ("normal", "gray") and x["label"] == 1]
    false_alarm = [x for x in per_img if x["decision"] in ("anomaly", "gray") and x["label"] == 0]
    grays = [x for x in per_img if x["decision"] == "gray"]
    print(f"[§4] 判对normal={len(correct_normal)} 判对defect={len(correct_defect)} "
          f"漏检={len(missed)} 误检={len(false_alarm)} 灰区={len(grays)}")
    print("[§4] 漏检样本 fused（decision=normal/gray 但实为缺陷）:")
    for x in missed[:6]:
        print(f"    {os.path.basename(x['path'])} fused={x['fused']:.3f} "
              f"decision={x['decision']}")
    if false_alarm:
        print("[§4] 误检样本:")
        for x in false_alarm[:3]:
            print(f"    {os.path.basename(x['path'])} fused={x['fused']:.3f} "
                  f"decision={x['decision']}")

    # ================= §4.2 视频检测（合成小视频冒烟） =================
    try:
        from src.video.video_pipeline import VideoInspector
        video_path = os.path.join(out_dir, "_opsflow_syn.mp4")
        good_imgs = [x["path"] for x in per_img if x["label"] == 0][:3]
        def_imgs = [x["path"] for x in per_img if x["label"] == 1][:2]
        seq = [good_imgs[0], good_imgs[1], good_imgs[2], def_imgs[0], good_imgs[0],
               def_imgs[1], good_imgs[1], good_imgs[2]] * 3        # 24 帧
        h, w = cv2.imread(seq[0]).shape[:2]
        vw = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
        for s in seq:
            vw.write(cv2.imread(s))
        vw.release()
        out = VideoInspector(pipe, step=2, alpha=0.5).inspect(video_path)
        s = out["summary"]
        print(f"\n[§4.2 视频] frames={s['total_frames']} anomaly={s['anomaly_frames']} "
              f"gray={s['gray_frames']} final_decision={s['final_decision']} "
              f"dominant_type={s['dominant_type']}")
        for fr in out["frames"][:6]:
            print(f"   frame#{fr['index']}: fused={fr['fused']:.3f} "
                  f"smooth={fr['fused_smooth']:.3f} decision={fr['decision']} "
                  f"smooth_dec={fr['decision_smooth']}"
                  f"{' [pulse_downgraded]' if fr.get('pulse_downgraded') else ''}")
        if not out.get("frames"):
            issues.append("§4.2 视频无帧结果")
    except Exception as e:
        issues.append(f"§4.2 视频检测异常: {type(e).__name__}: {e}")

    # ================= §5 反馈操作（质检管理员复看后打反馈） =================
    print(f"\n[§5 反馈] 启用 SSOCL 在线学习 ...")
    pipe.enable_ssocl(SSOCL_CFG, train_normal_paths=bundle["init_normal"],
                      rng=np.random.default_rng(cfg["seed"]))
    fb_pool = {"correct_normal": correct_normal, "correct_defect": correct_defect,
               "missed": missed, "false_alarm": false_alarm}
    plan = [
        ("correct", 0, fb_pool["correct_normal"], "判对-正常回流"),
        ("correct", 1, fb_pool["correct_defect"], "判对-缺陷入库"),
        ("wrong", 1, fb_pool["missed"], "漏检-难例入库"),
        ("wrong", 0, fb_pool["false_alarm"], "误检-回流"),
        ("none", None, grays, "灰区-主动选样"),
    ]
    fb_results = []
    for verdict, label, pool, desc in plan:
        if not pool:
            print(f"  [skip] {desc}: 无此类样本")
            continue
        p = pool[0]["path"]
        try:
            t0 = time.time()
            entry = pipe.feedback(p, verdict, label=label)
            dt = (time.time() - t0) * 1000
            fb_results.append(entry)
            act = entry.get("action", entry.get("reflow", {}).get("status", "?"))
            print(f"  [fb] {desc:12s} {os.path.basename(p)} verdict={verdict} "
                  f"label={label} -> action={act} ({dt:.0f}ms)")
        except Exception as e:
            issues.append(f"§5 反馈 {verdict}/{label} 异常: {type(e).__name__}: {e}")
            print(f"  [fb] {desc} 异常: {e}")

    # 回访：反馈后同批再预测一张缺陷（验证"拦截秒级生效"）
    if fb_results and missed:
        probe = missed[0]["path"]
        rp = pipe.predict(probe)
        print(f"[§5 回访] 原漏检样本再检: decision={rp['decision']} "
              f"fused={rp['fused']:.4f} boost={rp.get('boost', 0):.4f}")

    # ================= §6 主动选样 =================
    print(f"\n[§6 主动选样] ...")
    ask_list = pipe.active_select(top_n=10)
    print(f"[§6] 询问清单 {len(ask_list)} 条:")
    for a in ask_list[:5]:
        print(f"    - {a.get('path')} fused={a.get('fused', 0):.3f} "
              f"score={a.get('score', a.get('uncertainty', 0)):.3f}")
    if not ask_list and not grays:
        print("    (无灰区样本入队，清单为空——需确认行为是否符合预期)")

    # ================= §7 追溯查询 =================
    print(f"\n[§7 追溯] recent_inferences(5) / system_state / update_log / export ...")
    recents = api.recent_inferences(5)
    print(f"[§7] recent_inferences: {len(recents)} 条，最后一条 decision={recents[-1]['decision']}")
    st = api.system_state()
    print(f"[§7] system_state: weights={ {k: round(v,3) for k,v in st['weights'].items()} }")
    if st.get("ssocl"):
        print(f"[§7]   ssocl: defect_bank={st['ssocl']['defect_bank']} "
              f"normal_bank_ext={st['ssocl']['normal_bank']['ext_size']} "
              f"difficult={st['ssocl']['difficult_queue']}")
    ul = api.update_log()
    print(f"[§7] update_log: feedback={len(ul['feedback'])} rollback={len(ul['rollback'])} "
          f"gate={len(ul['gate'])}")
    audit = api.export(os.path.join(out_dir, "audit_opsflow.json"))
    print(f"[§7] export -> {audit}")
    api.close()

    # ================= 汇总 =================
    print("\n" + "=" * 60)
    if issues:
        print(f"[走查发现 {len(issues)} 个问题]:")
        for it in issues:
            print(f"  - {it}")
    else:
        print("[走查未发现问题]")
    summary = {"category": CAT, "n_issues": len(issues), "issues": issues,
               "detect_stats": stats, "avg_ms": round(avg_ms, 1),
               "fit_seconds": pipe.fit_seconds, "train_auroc": pipe.train_auroc,
               "fb_actions": [e.get("action", "?") for e in fb_results],
               "video_summary": s if "s" in dir() else None,
               "active_topn": len(ask_list)}
    with open(os.path.join(out_dir, "_opsflow_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[保存] _opsflow_summary.json")


if __name__ == "__main__":
    main()
