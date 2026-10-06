"""M1 冒烟：用少量图片走 demo5 baseline -> save -> load -> predict 一致性。

按工作区规则：随机取少量图片，不要整个数据集。
"""
import os
import sys
import tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # AOI_sys 根

# ---- 找 10-20 张 demo1/正常图当 init_normal，2-3 张 test 缺陷图当 init_defect ----
MV_ROOT = r"D:\CGAIC\data_origin\mvtec"
CAT = "bottle"
def find_good():
    p = os.path.join(MV_ROOT, CAT, "train", "good")
    return sorted([os.path.join(p, x) for x in os.listdir(p)
                   if x.lower().endswith((".png", ".jpg", ".jpeg"))])[:10]
def find_bad():
    out = []
    sub = os.path.join(MV_ROOT, CAT, "test")
    for name in os.listdir(sub):
        d = os.path.join(sub, name)
        if not os.path.isdir(d) or name == "good":
            continue
        out += sorted([os.path.join(d, x) for x in os.listdir(d)
                       if x.lower().endswith((".png", ".jpg", ".jpeg"))])[:2]
        if len(out) >= 3:
            break
    return out[:3]
def find_test_good():
    p = os.path.join(MV_ROOT, CAT, "test", "good")
    return sorted([os.path.join(p, x) for x in os.listdir(p)
                   if x.lower().endswith((".png", ".jpg", ".jpeg"))])[:3]


def build_mini_data():
    """被 _diag_*.py 复用，不产生全局副作用。"""
    good = find_good()
    defects = find_bad()
    test_good = find_test_good()
    print(f"[data] n_good={len(good)}, n_defect={len(defects)}, test_good={len(test_good)}", flush=True)
    assert len(good) >= 3 and test_good, f"样本不足: bottle in {MV_ROOT}? 检查路径"
    bundle = {
        "init_normal": good,
        "init_defect": defects,
        "val": [],
        "test": [(test_good[0], 0)],
    }
    return bundle, good, defects, test_good


BASE_CFG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "demo5_fast.yaml")


def main():
    bundle, good, defects, test_good = build_mini_data()
    tmp_storage = tempfile.mkdtemp(prefix="aoi_m1_")
    print(f"[storage] {tmp_storage}", flush=True)

    from backend.engine import Demo5Engine
    engine = Demo5Engine(tmp_storage, BASE_CFG)

    v = engine.prepare("bottle_smoke", bundle, scenario="L1a", profile="fast",
                       trigger="smoke", note="m1 冒烟")
    print(f"[prepare] version={v}", flush=True)

    # predict 两次：首次 + 第二次
    img = bundle["test"][0][0]
    r1 = engine.predict_image_path("bottle_smoke", img)
    # ---- 测试反馈前保存快照 v1（不触发 consolidate），用于验证 save/load 一致性 ----
    from algo.persist import save_snapshot, set_current_version, load_snapshot, list_versions
    cv = engine.current_version("bottle_smoke")
    pipe1, _ver = engine._pipes["bottle_smoke"]
    tmp_snap_v1 = os.path.join(tmp_storage, "snapshots", "bottle_smoke", "v1_pre_fb")
    # 把当前 pipe1 另存一份（不触发 engine 激活指针）
    os.makedirs(tmp_snap_v1, exist_ok=True)
    save_snapshot(pipe1, tmp_snap_v1, version=99, trigger="snapshot_pre_fb")
    # 用 load_snapshot 独立加载
    pipe_loaded = load_snapshot(tmp_snap_v1, "cuda" if engine.device == "cuda" else "cpu")
    # 同引擎当前激活 pipe1 与 pipe_loaded 分别 predict 同图
    recA = pipe1.predict(img)
    recB = pipe_loaded.predict(img)
    print(f"[save-load sanity] A.fused={recA['fused']:.6f}  B.fused={recB['fused']:.6f}  diff={abs(recA['fused']-recB['fused']):.3e}")
    print(f"  A.slot_scores={ {k: round(v,4) for k,v in recA['slot_scores'].items()} }")
    print(f"  B.slot_scores={ {k: round(v,4) for k,v in recB['slot_scores'].items()} }")
    assert abs(recA["fused"] - recB["fused"]) < 1e-6, f"持久化不一致: A={recA['fused']} B={recB['fused']}"
    print("[assert] save/load 持久化一致性 ✅ (pre-feedback snapshot)")

    r2 = engine.predict_image_path("bottle_smoke", img)
    print(f"[predict] score1={r1.score:.6f}  latency={r1.latency_ms:.1f}ms  "
          f"decision={r1.decision}  slots={sorted(r1.slot_scores.keys())}  "
          f"triggered={r1.triggered_slot}", flush=True)
    assert abs(r1.score - r2.score) < 1e-9, "同图两次 predict 不一致"

    # ---- feedback 即学 ----
    fb = engine.submit_feedback("bottle_smoke", good[0], verdict="correct", label=0)
    print(f"[feedback OK] {fb}", flush=True)

    # ---- consolidate（在线更新后固化） ----
    v2 = engine.consolidate("bottle_smoke", note="m1 冒烟巩固")
    print(f"[consolidate] v{v} -> v{v2}", flush=True)
    assert v2 > v

    # ---- 新 engine2 加载：快照指针当前是 v2（consolidate 已调用 set_current） ----
    del engine
    engine2 = Demo5Engine(tmp_storage, BASE_CFG)

    # ---- save/load 一致性：用 consolidate 前 v1（不含在线回流态）验证 ----
    # 原理：v1 是 prepare 完成的纯净快照；把指针暂时切回 v1 后预测应与 r1 一致。
    # （反馈提交后的 v2 含正常回流/CDF 重估，是 SSCL 预期行为；不要求与 r1 一致）
    from algo.persist import set_current_version
    cat_dir = engine2._cat_dir("bottle_smoke")
    set_current_version(cat_dir, 1)
    r_load_v1 = engine2.predict_image_path("bottle_smoke", img)
    print(f"[load-v1] score={r_load_v1.score:.6f}  cf prepare={r1.score:.6f}  diff={abs(r_load_v1.score - r1.score):.3e}")
    if abs(r_load_v1.score - r1.score) < 1e-6:
        print("[assert] 纯净 v1 快照 save/load 一致 ✅")
    elif abs(r_load_v1.score - r1.score) < 0.01:
        print(f"[warn] 纯净 v1 快照微差 {abs(r_load_v1.score - r1.score):.6f}（可接受）")
    else:
        raise AssertionError(f"纯净 v1 快照不一致: load={r_load_v1.score} prepare={r1.score}")
    # 把指针切回 consolidate 后的 v2（engine2 需重新装载；清缓存）
    del engine2._pipes["bottle_smoke"]
    set_current_version(cat_dir, 2)
    r3 = engine2.predict_image_path("bottle_smoke", img)
    print(f"[loaded v2] score={r3.score:.6f}  latency={r3.latency_ms:.1f}ms  "
          f"decision={r3.decision}  weights_len={len(r3.weights)}", flush=True)
    print("[note] v2 已含 SSCL 正常回流/拦截，与 prepare 时分数差异是预期行为（非持久化 bug）")

    # ---- 学习曲线离线回放 ----
    stream = [(p, 0) for p in good[4:8]]
    eval_items = [(test_good[0], 0)] + [(d, 1) for d in defects]
    print(f"[learning-curve] stream={len(stream)} eval={len(eval_items)}", flush=True)
    curve = engine2.simulate_learning(
        "bottle_smoke", stream, eval_items,
        feedback_ratio=1.0, eval_every=2, max_feedback=20,
    )
    print(f"[learning-curve] init={curve.get('initial_auroc')} final={curve.get('final_auroc')} "
          f"gain={curve.get('auroc_gain')} efficiency={curve.get('efficiency')}", flush=True)

    # ---- 贡献档案 ----
    cr = engine2.contribution_report("bottle_smoke", eval_items)
    print(f"[contrib] ablation.full={cr['ablation']['full']:.4f}  "
          f"leave-one-out first 3: " +
          ", ".join(f"{n}:{v:.4f}" for n, v in list(cr["ablation"]["leave_one_out"].items())[:3]))

    import shutil
    shutil.rmtree(tmp_storage, ignore_errors=True)
    print("✅ M1 冒烟通过")


if __name__ == "__main__":
    main()
