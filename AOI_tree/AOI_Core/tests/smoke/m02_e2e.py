"""M2 冒烟：后端核心链路（检测引擎）。

链路：DB 登记样本 → POST /models/prepare → POST /detect/image
→ POST /feedback（即学）→ GET /self_learning/status
→ POST /self_learning/update（巩固落盘）→ GET /models / /categories。

按工作区规则：随机取少量图片（10 good + 3 defect + 4 val），不跑整个数据集；
单图推理红线 1s（m0_fast + fast profile，预估 100~300ms，此处实测校验）。
使用独立临时 sqlite + 临时 storage，不污染生产 database/aoi.db 与 storage/。
"""
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # AOI_sys 根

# ---- 隔离环境（必须在 import backend 之前设置 AOI_CONFIG）----
TMP = tempfile.mkdtemp(prefix="aoi_m2_")
CFG_PATH = os.path.join(TMP, "cfg.yaml")
import yaml  # noqa: E402

with open(CFG_PATH, "w", encoding="utf-8") as f:
    yaml.safe_dump({"system": {
        "database_url": "sqlite:///" + os.path.join(TMP, "aoi.db").replace("\\", "/"),
        "storage_dir": os.path.join(TMP, "storage").replace("\\", "/"),
        "engine_storage": os.path.join(TMP, "storage", "engine").replace("\\", "/"),
        "engine_base_cfg": "configs/engine_fast.yaml",
    }}, f, allow_unicode=True)
os.environ["AOI_CONFIG"] = CFG_PATH

CAT = "smoke_m2"
# ---- 数据供给：自动探测可用数据集（MVTec-like 结构），无则合成兜底 ----
from tests.testdata import pick_category

DATA_DIR, _DATA_CAT, _SYNTHETIC = pick_category()


def _ls(d, n):
    return sorted(os.path.join(d, x) for x in os.listdir(d)
                  if x.lower().endswith((".png", ".jpg", ".jpeg")))[:n]


def collect_data():
    good = _ls(os.path.join(DATA_DIR, "train", "good"), 10)
    defects = []
    for sub in sorted(os.listdir(os.path.join(DATA_DIR, "test"))):
        dd = os.path.join(DATA_DIR, "test", sub)
        if os.path.isdir(dd) and sub != "good":
            defects += _ls(dd, 3)  # 单缺陷类型数据集也能凑满 3 张
        if len(defects) >= 3:
            break
    test_good = _ls(os.path.join(DATA_DIR, "test", "good"), 2)
    print(f"[data] good={len(good)} defect={len(defects)} test_good={len(test_good)}",
          flush=True)
    assert len(good) >= 3 and len(defects) >= 3 and test_good, "样本不足"
    return good, defects[:3], test_good


def wait_task(task_manager, task_id, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        t = task_manager.get(task_id)
        if t["status"] in ("done", "failed"):
            return t
        print(f"  [task {task_id}] {t['progress']*100:5.1f}% {t['message']}",
              flush=True)
        time.sleep(3)
    raise TimeoutError(f"task {task_id} 超时")


def main():
    good, defects, test_good = collect_data()
    print(f"[env] TMP={TMP}", flush=True)

    # 1) 样本登记进 DB
    from backend.db.database import session_scope
    from backend.db.models import Image as ImageRow
    with session_scope() as s:
        s.query(ImageRow).filter(ImageRow.category == CAT).delete(
            synchronize_session=False)
        for p in good:
            s.add(ImageRow(path=p, category=CAT, split="train",
                           label="normal", source="smoke"))
        for p in defects[1:]:
            s.add(ImageRow(path=p, category=CAT, split="train_anomaly",
                           label="anomaly", source="smoke"))
        for p in test_good:
            s.add(ImageRow(path=p, category=CAT, split="test",
                           label="normal", source="smoke"))
        s.add(ImageRow(path=defects[0], category=CAT, split="test",
                       label="anomaly", source="smoke"))
    print("[db] 样本登记完成（10 normal train / 3 defect / 3 test）", flush=True)

    from fastapi.testclient import TestClient

    from backend.api.app import app
    from backend.core.tasks import task_manager

    with TestClient(app) as client:
        # 2) prepare
        r = client.post("/api/models/prepare",
                        json={"category": CAT, "scenario": "L1a",
                              "profile": "fast", "force": True})
        assert r.status_code == 200, r.text
        tid = r.json()["task_id"]
        print(f"[prepare] task_id={tid}，等待完成...", flush=True)
        t = wait_task(task_manager, tid)
        assert t["status"] == "done", f"prepare 失败: {t['message']}"
        res = t["result"]
        print(f"[prepare] ✅ version={res['version']} "
              f"n_normal={res['n_normal']} n_defect={res['n_defect']}", flush=True)
        assert res["version"] == "v1"

        # 3) detect（缺陷图 + 正常图各一张）
        det_bad = client.post("/api/detect/image",
                              json={"path": defects[1], "category": CAT}).json()
        print(f"[detect bad] score={det_bad['final_score']:.4f} "
              f"decision={det_bad['decision']} is_anomaly={det_bad['is_anomaly']} "
              f"latency={det_bad['latency_ms']:.1f}ms "
              f"slots={ {k: round(v,3) for k,v in det_bad['slots'].items()} } "
              f"triggered={det_bad['triggered_slot']}", flush=True)
        for k in ("detection_id", "final_score", "decision", "threshold",
                  "gray_threshold", "slots", "weights", "defect_boxes",
                  "level1_score", "n_tiles", "heatmap_path"):
            assert k in det_bad, f"检测响应缺字段 {k}"
        assert det_bad["n_tiles"]["decision"] == det_bad["decision"]
        assert det_bad["latency_ms"] < 1000, \
            f"单图超红线: {det_bad['latency_ms']:.1f}ms >= 1000ms"

        det_ok = client.post("/api/detect/image",
                             json={"path": test_good[0], "category": CAT}).json()
        print(f"[detect good] score={det_ok['final_score']:.4f} "
              f"decision={det_ok['decision']} "
              f"latency={det_ok['latency_ms']:.1f}ms", flush=True)
        assert det_ok["latency_ms"] < 1000, "单图超红线"

        # 未准备品类应给出清晰错误
        r = client.post("/api/detect/image",
                        json={"path": test_good[0], "category": "not_prepared"})
        assert r.status_code == 400 and "/models/prepare" in r.json()["detail"], \
            f"未准备品类错误提示不符合预期: {r.status_code} {r.text}"
        print("[detect] 未准备品类 400 + 提示 ✅", flush=True)

        # 4) feedback 即学（漏检 + 框选区域）
        fb = client.post("/api/feedback", json={
            "detection_id": det_bad["detection_id"],
            "feedback_type": "false_negative", "operator_label": 1,
            "region": [100, 100, 300, 200], "comment": "m2 冒烟"}).json()
        print(f"[feedback] engine_update={fb['engine_update']} note={fb['note']}",
              flush=True)
        assert fb["engine_update"] is not None, f"即学未生效: {fb['note']}"
        fb_list = client.get("/api/feedback").json()
        assert fb_list["items"][0]["consumed"] is True, "反馈未标记消费"
        print("[feedback] 即学即消费 consumed=True ✅", flush=True)

        # 5) self_learning/status
        st = client.get("/api/self_learning/status").json()
        cat_info = [c for c in st["categories"] if c["category"] == CAT]
        assert cat_info, "status 缺少目标品类"
        print(f"[status] {cat_info[0]}", flush=True)
        assert cat_info[0]["feedback_log_len"] is not None \
            and cat_info[0]["feedback_log_len"] >= 1

        # 6) self_learning/update（巩固落盘 → v2）
        r = client.post("/api/self_learning/update",
                        json={"category": CAT, "note": "m2 冒烟巩固"})
        tid2 = r.json()["task_id"]
        t2 = wait_task(task_manager, tid2)
        assert t2["status"] == "done", f"consolidate 失败: {t2['message']}"
        print(f"[consolidate] ✅ {t2['result']}", flush=True)
        assert t2["result"]["version"] == "v2"

        # 7) models / categories / calibration 410
        models = client.get("/api/models").json()["items"]
        mine = {m["version"]: m for m in models if m["category"] == CAT}
        assert set(mine) >= {"v1", "v2"}, f"Model 登记不完整: {list(mine)}"
        assert mine["v2"]["is_active"] and not mine["v1"]["is_active"]
        assert mine["v1"]["origin"] == "engine_fit"
        assert mine["v2"]["origin"] == "self_learned_engine"
        print(f"[models] v1/v2 登记与激活指针 ✅ "
              f"(v1.origin={mine['v1']['origin']} v2.origin={mine['v2']['origin']})",
              flush=True)

        cats = client.get("/api/categories").json()
        c = [x for x in cats if x["category"] == CAT][0]
        assert c["prepared"] and c["current_version"] == 2, c
        print(f"[categories] prepared={c['prepared']} cur=v{c['current_version']} ✅",
              flush=True)

        r = client.get(f"/api/calibration/{CAT}")
        assert r.status_code == 410, r.status_code
        print("[calibration] 410 Gone ✅", flush=True)

        # 8) activate 回 v1 再回 v2（版本切换链路）
        r = client.post(f"/api/models/{mine['v1']['id']}/activate")
        assert r.status_code == 200, r.text
        r = client.post(f"/api/models/{mine['v2']['id']}/activate")
        assert r.status_code == 200, r.text
        print("[activate] v1→v2 切换 ✅", flush=True)

        # 9) 巩固后再检测仍工作（含反馈回流后的内存态）
        det2 = client.post("/api/detect/image",
                           json={"path": defects[2], "category": CAT,
                                 "with_heatmap": False}).json()
        assert det2["latency_ms"] < 1000
        print(f"[detect after consolidate] score={det2['final_score']:.4f} "
              f"decision={det2['decision']} latency={det2['latency_ms']:.1f}ms ✅",
              flush=True)

        # ══════════ M2b：评估服务改造 + 学习曲线/贡献报告 ══════════
        # 10) accuracy 评估（后台任务）
        r = client.post("/api/eval/accuracy", json={"category": CAT})
        assert r.status_code == 200, r.text
        t = wait_task(task_manager, r.json()["task_id"], timeout=120)
        assert t["status"] == "done", f"accuracy 失败: {t['message']}"
        acc = t["result"]
        print(f"[M2b accuracy] ✅ auroc={acc['metrics']['auroc']:.4f} "
              f"f1={acc['metrics']['f1']:.4f} n={acc['n_images']}", flush=True)
        assert acc["metrics"]["auroc"] is not None

        # 11) A/B 对比：version_a 缺省=当前激活 v2，候选 version_b=v1
        r = client.post("/api/eval/ab_compare",
                        json={"category": CAT, "version_b": 1})
        assert r.status_code == 200, r.text
        t = wait_task(task_manager, r.json()["task_id"], timeout=120)
        assert t["status"] == "done", f"ab_compare 失败: {t['message']}"
        ab = t["result"]
        print(f"[M2b ab_compare] ✅ v{ab['version_a']} vs v{ab['version_b']} "
              f"verdict={ab['verdict']} "
              f"auroc {ab['active_metrics']['auroc']} -> "
              f"{ab['candidate_metrics']['auroc']} "
              f"分歧={len(ab['disagreements'])}", flush=True)
        assert ab["version_a"] == 2 and ab["version_b"] == 1
        assert "auroc" in ab["active_metrics"] and "auroc" in ab["candidate_metrics"]
        assert ab["verdict"] in ("better", "worse", "tie")

        # 12) 学习曲线离线回放（后台） + latest
        # 2026-08-30：对齐 rounds 模式契约（旧 n_stream/max_feedback 参数
        # 仅 stream 模式有效；rounds 模式按 round_size×n_rounds 取轮池，
        # 本 smoke 仅 17 张登记图，用小档位）
        r = client.post("/api/learning/curve",
                        json={"category": CAT, "mode": "rounds",
                              "round_size": 5, "n_rounds": 2, "n_eval": 3})
        assert r.status_code == 200, r.text
        assert "无需暂停产线" in r.json()["description"]
        t = wait_task(task_manager, r.json()["task_id"], timeout=120)
        assert t["status"] == "done", f"learning_curve 失败: {t['message']}"
        lc = client.get(f"/api/learning/curve/{CAT}/latest").json()
        print(f"[M2b learning_curve] ✅ steps={lc['steps']} "
              f"auroc={lc['auroc']} initial={lc['initial_auroc']} "
              f"final={lc['final_auroc']} gain={lc['auroc_gain']} "
              f"efficiency={lc['efficiency']}", flush=True)
        for k in ("steps", "auroc", "f1", "initial_auroc", "final_auroc",
                  "auroc_gain", "efficiency"):
            assert k in lc and lc[k] is not None, f"learning_curve 缺字段 {k}"
        assert lc["steps"], "学习曲线 steps 为空"

        # 13) 槽位贡献档案（后台） + latest
        r = client.post("/api/learning/contribution", json={"category": CAT})
        assert r.status_code == 200, r.text
        t = wait_task(task_manager, r.json()["task_id"], timeout=120)
        assert t["status"] == "done", f"contribution 失败: {t['message']}"
        cb = client.get(f"/api/learning/contribution/{CAT}/latest").json()
        loo = (cb["ablation"] or {}).get("leave_one_out", {})
        print(f"[M2b contribution] ✅ ablation.full={cb['ablation'].get('full')} "
              f"leave_one_out={ {k: round(v, 4) for k, v in loo.items()} }",
              flush=True)
        assert cb["ablation"] and loo, "贡献档案 ablation.leave_one_out 为空"

        # 14) /eval/shapley 已下线
        r = client.post("/api/eval/shapley", json={"category": CAT})
        assert r.status_code in (404, 405), r.status_code
        print("[M2b shapley] 路由已下线（404/405）✅", flush=True)

        # ══════════ M5a：工业化完善（复核队列/激活门控/webhook/在线曲线）══════════
        from backend.db.models import Detection, Feedback, OperationLog

        # 15) 灰区复核队列：直接插入一条 decision=gray 的检测记录
        #     （2026-08-29 口径：复核队列只收产线路径 pipeline/stream/plc）
        with session_scope() as s:
            gray = Detection(target_type="pipeline", image_path=test_good[1],
                             category=CAT, final_score=0.5, is_anomaly=False,
                             latency_ms=10.0,
                             n_tiles={"slots": {"sem": 0.5}, "decision": "gray"})
            s.add(gray)
            s.flush()
            gray_id = gray.id
        q = client.get(f"/api/review/queue?category={CAT}").json()
        assert any(it["id"] == gray_id for it in q["items"]), f"gray 帧未入队: {q}"
        print(f"[M5a review/queue] ✅ 待复核 {q['total']} 条（含插入的 gray 帧）",
              flush=True)

        r = client.post(f"/api/review/{gray_id}",
                        json={"label": 1, "comment": "m5 冒烟复核"})
        assert r.status_code == 200, r.text
        rv = r.json()
        assert rv["engine_update"] is not None, f"复核未触发即学: {rv}"
        with session_scope() as s:
            fb = s.query(Feedback).filter(Feedback.detection_id == gray_id).one()
            assert fb.feedback_type == "review" and fb.consumed is True
        q2 = client.get(f"/api/review/queue?category={CAT}").json()
        assert all(it["id"] != gray_id for it in q2["items"]), "复核后仍在队列"
        st = client.get("/api/review/stats").json()
        assert st["reviewed_today"] >= 1 and st["reviewed_total"] >= 1, st
        assert CAT not in (st["pending_by_category"] or {}), st
        print(f"[M5a review submit] ✅ feedback_type=review consumed=True "
              f"engine_update={rv['engine_update']} stats={st}", flush=True)

        r = client.post(f"/api/review/{gray_id}", json={"label": 0})
        assert r.status_code == 409, f"重复复核应 409: {r.status_code}"
        r = client.post("/api/review/999999", json={"label": 0})
        assert r.status_code == 404, f"不存在记录应 404: {r.status_code}"
        print("[M5a review] 重复复核 409 / 不存在 404 ✅", flush=True)

        # 16) 激活质量门控：锚定集正负样本不足（pos=2<4）→ skipped 放行；
        #     若实际打分且被拒（409），force=true 强制激活仍应成功且带 gate_report
        models = client.get("/api/models").json()["items"]
        mine = {m["version"]: m for m in models if m["category"] == CAT}
        t0 = time.time()
        r = client.post(f"/api/models/{mine['v1']['id']}/activate?gate=true")
        if r.status_code == 409:
            gr = r.json()["detail"]["gate_report"]
            print(f"[M5a gate] v1 被门控拒绝（AUROC 退化超 margin）→ force 激活: "
                  f"auroc {gr['current']['auroc']} -> {gr['candidate']['auroc']}",
                  flush=True)
            r = client.post(
                f"/api/models/{mine['v1']['id']}/activate?gate=true&force=true")
        assert r.status_code == 200, r.text
        gr = r.json()["gate_report"]
        assert "skipped" in gr and "passed" in gr and "n_images" in gr, gr
        print(f"[M5a gate] v2→v1 ✅ {time.time()-t0:.1f}s skipped={gr['skipped']} "
              f"n={gr['n_images']} auroc {gr['current']['auroc']} -> "
              f"{gr['candidate']['auroc']} divergent={gr['divergent']} "
              f"reason={gr.get('skip_reason')}", flush=True)
        # 回切 v2 保持与既有口径一致（候选即当前时门控短路跳过）
        r = client.post(f"/api/models/{mine['v2']['id']}/activate?gate=true")
        assert r.status_code == 200, r.text
        print("[M5a gate] 回切 v2 ✅", flush=True)

        # 17) webhook：不可达地址静默记录 + 不阻塞检测；禁用路径不报错
        from backend.core.config import get_settings
        from backend.core.notify import notify_event
        cfg = get_settings()
        cfg._cfg["notify"] = {"webhook_url": "http://127.0.0.1:1/hook",
                              "events": ["anomaly", "gray", "feedback"],
                              "timeout_s": 1}
        t0 = time.time()
        notify_event("anomaly", {"category": CAT, "detection_id": 0})
        assert time.time() - t0 < 0.5, "notify_event 阻塞调用方"
        det = client.post("/api/detect/image",
                          json={"path": defects[1], "category": CAT,
                                "with_heatmap": False})
        assert det.status_code == 200, det.text
        time.sleep(2.5)   # 等后台线程把连接失败落进 OperationLog
        with session_scope() as s:
            n_err = s.query(OperationLog).filter(
                OperationLog.action == "webhook_error").count()
        assert n_err >= 1, "webhook 失败未静默记录 OperationLog"
        print(f"[M5a webhook] ✅ 不可达地址静默（webhook_error×{n_err}）"
              f"且检测不阻塞", flush=True)

        cfg._cfg["notify"] = {"webhook_url": "", "events": ["anomaly"],
                              "timeout_s": 1}
        det = client.post("/api/detect/image",
                          json={"path": test_good[0], "category": CAT,
                                "with_heatmap": False})
        assert det.status_code == 200, det.text
        print("[M5a webhook] 禁用路径 ✅", flush=True)

        # 18) 在线学习曲线：补一条 confirmed 反馈后校验曲线与判对映射
        fb2 = client.post("/api/feedback", json={
            "detection_id": det_ok["detection_id"],
            "feedback_type": "confirmed", "operator_label": 0}).json()
        assert fb2["engine_update"] is not None, fb2
        oc = client.get(f"/api/learning/online_curve/{CAT}").json()
        pts = oc["points"]
        assert [p["k"] for p in pts] == list(range(1, len(pts) + 1))
        assert oc["summary"]["total_feedback"] == len(pts) >= 3
        assert all(0.0 <= p["rolling_acc"] <= 1.0 for p in pts)
        # 判对映射：false_negative→错；review(label=1, 系统 is_anomaly=False)→错；
        # confirmed→对
        corr = [p["correct"] for p in pts]
        assert corr == [0, 0, 1], f"判对映射不符合预期: {corr}"
        assert oc["weights"], "pipe 已加载，权重快照不应为空"
        print(f"[M5a online_curve] ✅ points={len(pts)} correctness={corr} "
              f"correct_rate={oc['summary']['correct_rate']:.2f} "
              f"last7d={oc['summary']['last7d_feedback']} "
              f"weights={ {k: round(v, 3) for k, v in oc['weights'].items()} }",
              flush=True)

        # ══════════ M6a：复核框选 / 连续异常告警 / open_set / precheck ══════════

        # 19) review + region 框选：label=1 带 region -> 归一化 box 喂缺陷拦截通道
        with session_scope() as s:
            gray_m6 = Detection(target_type="image", image_path=defects[1],
                                category=CAT, final_score=0.5, is_anomaly=False,
                                latency_ms=10.0,
                                n_tiles={"slots": {"sem": 0.5}, "decision": "gray"})
            s.add(gray_m6)
            s.flush()
            gray_m6_id = gray_m6.id
        r = client.post(f"/api/review/{gray_m6_id}", json={
            "label": 1, "comment": "m6 框选复核", "region": [10, 10, 50, 50]})
        assert r.status_code == 200, r.text
        rv6 = r.json()
        assert rv6["engine_update"] is not None, f"框选复核未触发即学: {rv6}"
        with session_scope() as s:
            fb6 = s.query(Feedback).filter(
                Feedback.detection_id == gray_m6_id).one()
            assert fb6.consumed is True
            # M7a 起 region 落库为 {"box":.., "pre":..}
            assert fb6.region["box"] == [10, 10, 50, 50], fb6.region
        print(f"[M6a review region] ✅ consumed=True region={fb6.region} "
              f"engine_update={rv6['engine_update']}", flush=True)

        # label=0 时 region 仅落库、不喂拦截通道（正常回流），不报错
        with session_scope() as s:
            gray_m6b = Detection(target_type="image", image_path=test_good[0],
                                 category=CAT, final_score=0.5, is_anomaly=False,
                                 latency_ms=10.0,
                                 n_tiles={"slots": {"sem": 0.5}, "decision": "gray"})
            s.add(gray_m6b)
            s.flush()
            gray_m6b_id = gray_m6b.id
        r = client.post(f"/api/review/{gray_m6b_id}",
                        json={"label": 0, "region": [10, 10, 50, 50]})
        assert r.status_code == 200, r.text
        print(f"[M6a review region] label=0 忽略框选（正常回流）✅ "
              f"engine_update={r.json()['engine_update']}", flush=True)

        # 20) 连续异常告警
        from backend.core.alarm import get_alarm_monitor
        mon = get_alarm_monitor()
        mon.reset("alarm_unit")
        fired = hits = None
        for _ in range(4):
            fired, hits = mon.record("alarm_unit", True)
        assert not fired and hits == 4, (fired, hits)
        fired, hits = mon.record("alarm_unit", True)        # 第 5 帧 -> 触发
        assert fired and hits == 5, (fired, hits)
        fired2, hits2 = mon.record("alarm_unit", True)      # 冷却期内不重复触发
        assert (not fired2) and hits2 == 6, (fired2, hits2)
        for _ in range(16):                                 # 正常帧把异常帧滑出窗口
            fired, hits = mon.record("alarm_unit", False)
        assert not fired and hits == 4, (fired, hits)
        mon.reset("alarm_unit")
        print("[M6a alarm] 单元级：第5帧触发 / 冷却期去重 / 滑窗滑出 ✅", flush=True)

        # 接线级：fired -> OperationLog("alarm")；REST detect 响应带 alarm/open_alert
        from backend.pipeline.service import get_detection_service
        mon.reset(CAT)
        st_alarm = None
        for _ in range(5):
            st_alarm = get_detection_service()._record_alarm(CAT, True)
        assert st_alarm == {"active": True, "window_hits": 5}, st_alarm
        with session_scope() as s:
            n_alarm_log = s.query(OperationLog).filter(
                OperationLog.action == "alarm").count()
        assert n_alarm_log >= 1, "alarm fired 未记 OperationLog"
        print(f"[M6a alarm] 接线级 fired -> OperationLog×{n_alarm_log} ✅", flush=True)

        # 21) detect 响应含 alarm 与 open_alert 字段（20 步窗口已灌 5 帧 True）
        det_m6 = client.post("/api/detect/image",
                             json={"path": defects[1], "category": CAT,
                                   "with_heatmap": False}).json()
        assert "alarm" in det_m6 and "open_alert" in det_m6
        assert isinstance(det_m6["alarm"]["window_hits"], int)
        assert isinstance(det_m6["alarm"]["active"], bool)
        assert det_m6["alarm"]["window_hits"] >= 5
        print(f"[M6a detect] alarm={det_m6['alarm']} "
              f"open_alert={det_m6['open_alert']} ✅", flush=True)

        # 22) precheck：split 分布 + 建议 scenario + 警告
        pc = client.get(f"/api/models/precheck/{CAT}").json()
        for k in ("counts", "resolutions", "has_template",
                  "suggested_scenario", "suggested_profile", "warnings"):
            assert k in pc, f"precheck 缺字段 {k}"
        assert pc["has_template"] is False
        assert pc["counts"] == {"train_normal": 10, "train_anomaly": 2,
                                "test_normal": 2, "test_anomaly": 1,
                                "feedback": 0, "template": 0}, pc["counts"]
        assert pc["suggested_scenario"] == "L1a" and pc["suggested_profile"] == "fast"
        pc_empty = client.get("/api/models/precheck/never_imported").json()
        assert pc_empty["counts"]["train_normal"] == 0
        assert pc_empty["suggested_scenario"] is None and pc_empty["warnings"]
        print(f"[M6a precheck] ✅ counts={pc['counts']} "
              f"scenario={pc['suggested_scenario']} "
              f"resolutions={pc['resolutions']} warnings={pc['warnings']}; "
              f"空品类 -> {pc_empty['suggested_scenario']} + 警告", flush=True)

        # ══════════ M7a：追溯→学习→提升闭环 ══════════

        # 23) 反馈带 pre：false_positive 反馈 → 响应含 pre；DB region 为 dict
        det7 = client.post("/api/detect/image",
                           json={"path": defects[0], "category": CAT,
                                 "with_heatmap": False}).json()
        det7_id = det7["detection_id"]
        r = client.post("/api/feedback", json={
            "detection_id": det7_id,
            "feedback_type": "false_positive", "operator_label": 0,
            "comment": "m7 冒烟"})
        assert r.status_code == 200, r.text
        fb7 = r.json()
        assert fb7["pre"] is not None, f"响应缺 pre: {fb7}"
        assert "score" in fb7["pre"] and "decision" in fb7["pre"], fb7["pre"]
        print(f"[M7a feedback pre] ✅ pre={fb7['pre']} "
              f"engine_update={fb7['engine_update']}", flush=True)
        with session_scope() as s:
            fb7_row = s.get(Feedback, client.get("/api/feedback").json()
                          ["items"][0]["id"])
            assert isinstance(fb7_row.region, dict), fb7_row.region
            assert fb7_row.region["pre"] is not None \
                and fb7_row.region["box"] is None
        print("[M7a feedback pre] DB region=dict 且 pre 非空 ✅", flush=True)

        # 存量兼容：手工插一条 region=list 的旧格式反馈
        with session_scope() as s:
            legacy_det = Detection(target_type="image", image_path=test_good[0],
                                   category=CAT, final_score=0.3,
                                   is_anomaly=False, latency_ms=10.0,
                                   n_tiles={"decision": "normal"})
            s.add(legacy_det)
            s.flush()
            s.add(Feedback(detection_id=legacy_det.id,
                           feedback_type="false_positive", operator_label=0,
                           region=[1, 2, 3, 4]))
        fb_list = client.get("/api/feedback").json()
        assert fb_list["total"] >= 1, "旧格式反馈导致反馈列表接口异常"
        print("[M7a legacy] region=list 存量反馈兼容（列表接口不炸）✅",
              flush=True)

        # 24) 翻案曲线：同步返回（N<=50），字段齐全 + legacy_count>=1 + 409
        r = client.get(f"/api/learning/flip_curve/{CAT}")
        assert r.status_code == 200, r.text
        fc = r.json()
        assert fc["summary"]["total"] >= 1, fc["summary"]
        assert fc["summary"]["legacy_count"] >= 1, fc["summary"]
        for k in ("flipped_count", "flip_rate"):
            assert k in fc["summary"], fc["summary"]
        p0 = fc["points"][0]
        for k in ("k", "image_path", "feedback_type", "label", "pre_score",
                  "post_score", "pre_decision", "post_decision", "flipped"):
            assert k in p0, f"flip_curve point 缺字段 {k}"
        n_flipped = sum(1 for p in fc["points"] if p["flipped"])
        print(f"[M7a flip_curve] ✅ total={fc['summary']['total']} "
              f"flipped={n_flipped} flip_rate={fc['summary']['flip_rate']} "
              f"legacy={fc['summary']['legacy_count']} "
              f"首点 pre={p0['pre_score']}->{p0['post_score']}", flush=True)

        r = client.get("/api/learning/flip_curve/never_prepared_cat")
        assert r.status_code == 409, f"未准备品类应 409: {r.status_code}"
        print("[M7a flip_curve] 未准备品类 409 ✅", flush=True)

        # 25) 单帧追溯链
        r = client.get(f"/api/detections/{det7_id}/trace")
        assert r.status_code == 200, r.text
        tr = r.json()
        assert tr["detection"]["id"] == det7_id
        assert "model" in tr and "post" in tr
        assert tr["feedbacks"], "trace 缺反馈时间线"
        assert tr["feedbacks"][0]["pre"] is not None, tr["feedbacks"][0]
        assert tr["post"] is not None and "score" in tr["post"], tr["post"]
        print(f"[M7a trace] ✅ feedbacks={len(tr['feedbacks'])} "
              f"model={tr['model']} post={tr['post']}", flush=True)
        r = client.get("/api/detections/999999/trace")
        assert r.status_code == 404, r.status_code
        print("[M7a trace] 不存在记录 404 ✅", flush=True)

        # 26) 错检集：JSON + CSV
        mj = client.get("/api/detections/misjudged",
                        params={"category": CAT}).json()
        assert mj["items"], "错检集为空"
        it0 = mj["items"][0]
        for k in ("detection_id", "image_path", "category", "final_score",
                  "decision", "feedback_type", "operator_label", "pre_score"):
            assert k in it0, f"misjudged 行缺字段 {k}"
        assert any(it["pre_score"] is not None for it in mj["items"]), \
            "错检集所有行 pre_score 均为空"
        print(f"[M7a misjudged] ✅ total={mj['total']} "
              f"首行={ {k: it0[k] for k in ('detection_id', 'feedback_type', 'pre_score')} }",
              flush=True)
        r = client.get("/api/detections/misjudged",
                       params={"category": CAT, "format": "csv"})
        assert r.status_code == 200 and "text/csv" in r.headers["content-type"]
        assert r.text.splitlines()[0].startswith("detection_id,image_path"), \
            r.text[:200]
        print("[M7a misjudged] CSV 导出（表头正确）✅", flush=True)

        # ══════════ M8a：数据资产工业化治理 ══════════
        import shutil as _shutil
        from pathlib import Path

        from backend.db.models import Dataset

        # 27) 启动归组：步骤1直接入库的 15 张样本应已归入"历史未分组"批次
        ds_all = client.get(f"/api/datasets?category={CAT}").json()["items"]
        legacy = [d for d in ds_all if d["source_type"] == "legacy"]
        assert legacy, f"缺少历史未分组批次: {ds_all}"
        assert legacy[0]["n_images"] == 15, legacy[0]
        with session_scope() as s:
            n_null = (s.query(ImageRow)
                      .filter(ImageRow.category == CAT,
                              ImageRow.dataset_id.is_(None)).count())
            n_legacy = (s.query(ImageRow)
                        .filter(ImageRow.category == CAT,
                                ImageRow.dataset_id == legacy[0]["id"]).count())
        assert n_null == 0 and n_legacy == 15, (n_null, n_legacy)
        print(f"[M8a legacy] ✅ 历史未分组批次 id={legacy[0]['id']} "
              f"n_images={legacy[0]['n_images']}，存量图已全部归组", flush=True)

        # 28) import_folder 挂批次 + 新目录结构 storage/images/{cat}/ds{id}/
        M8CAT = "smoke_m8"
        imp_dir = Path(TMP) / "m8_import"
        imp_dir.mkdir(exist_ok=True)
        for i, src in enumerate(test_good + [defects[0]]):
            _shutil.copy2(src, imp_dir / f"m8_{i}.png")
        r = client.post("/api/images/import", json={
            "folder": str(imp_dir), "category": M8CAT, "split": "test",
            "label": "normal", "name": "m8批次", "note": "M8 冒烟"})
        assert r.status_code == 200, r.text
        ds_id = r.json()["dataset_id"]
        t = wait_task(task_manager, r.json()["task_id"], timeout=120)
        assert t["status"] == "done", t["message"]
        assert t["result"]["dataset_id"] == ds_id
        with session_scope() as s:
            imgs = s.query(ImageRow).filter(ImageRow.dataset_id == ds_id).all()
            assert len(imgs) == 3, f"批次图片数不对: {len(imgs)}"
            for im in imgs:
                fp = Path(im.path)
                assert fp.is_file(), fp
                assert fp.parent.parent.parent.name == f"ds{ds_id}", fp
                assert fp.parent.parent.parent.parent.name == M8CAT, fp
                assert fp.name.startswith(f"ds{ds_id}_"), fp
            ds = s.get(Dataset, ds_id)
            assert ds.n_images == 3 and ds.name == "m8批次", (ds.n_images, ds.name)
        print(f"[M8a import_folder] ✅ dataset_id={ds_id} 3 张图挂批次，"
              f"文件在 storage/images/{M8CAT}/ds{ds_id}/ 下", flush=True)

        # 29) GET /datasets 列表 + 详情分页
        lst = client.get(f"/api/datasets?category={M8CAT}").json()["items"]
        assert any(d["id"] == ds_id for d in lst), lst
        pg = client.get(f"/api/datasets/{ds_id}?page=1&page_size=2").json()
        assert pg["total"] == 3 and len(pg["items"]) == 2, pg["total"]
        assert pg["dataset"]["name"] == "m8批次"
        img_id = pg["items"][0]["id"]
        print(f"[M8a datasets] ✅ 列表含新批次；详情分页 total=3 "
              f"page_size=2 -> {len(pg['items'])} 条", flush=True)

        # 30) annotate 标注流转 + OperationLog extra
        r = client.post(f"/api/images/{img_id}/annotate",
                        json={"label": "anomaly", "defect_type": "scratch"})
        assert r.status_code == 200, r.text
        assert r.json()["label"] == "anomaly"
        with session_scope() as s:
            row = s.get(ImageRow, img_id)
            assert row.label == "anomaly" and row.defect_type == "scratch"
        logs = client.get("/api/logs?action=annotate_image").json()["items"]
        assert logs and all(l["action"] == "annotate_image" for l in logs)
        assert logs[0]["extra"]["image_id"] == img_id, logs[0].get("extra")
        assert logs[0]["extra"]["old"]["label"] == "normal"
        print(f"[M8a annotate] ✅ image_id={img_id} label normal->anomaly，"
              f"OperationLog extra 含 image_id/old/new", flush=True)

        # 31) DELETE dataset 级联（删行 + 删文件 + 计数）
        r = client.delete(f"/api/datasets/{ds_id}?delete_files=true")
        assert r.status_code == 200, r.text
        out = r.json()
        assert out["deleted_images"] == 3 and out["deleted_files"] == 3, out
        assert out["affected_detections"] == 0, out
        with session_scope() as s:
            assert s.query(ImageRow).filter(
                ImageRow.dataset_id == ds_id).count() == 0
            assert s.get(Dataset, ds_id) is None
        ds_dir = get_settings().storage("images") / M8CAT / f"ds{ds_id}"
        assert not ds_dir.is_dir() or not any(ds_dir.iterdir())
        print(f"[M8a delete_dataset] ✅ {out}", flush=True)

        # 32) upload 默认挂 uploads 批次 + 单图删文件
        with open(test_good[0], "rb") as f:
            up = client.post("/api/images/upload",
                             files={"file": ("m8up.png", f.read(), "image/png")},
                             data={"category": M8CAT})
        assert up.status_code == 200, up.text
        up_j = up.json()
        assert up_j["dataset_id"] is not None, up_j
        ds_up = client.get(f"/api/datasets/{up_j['dataset_id']}").json()["dataset"]
        assert ds_up["name"] == "uploads" and ds_up["source_type"] == "upload"
        up_path = up_j["path"]
        r = client.delete(f"/api/images/{up_j['id']}?delete_file=true").json()
        assert r["file_deleted"] is True and not Path(up_path).exists(), r
        print(f"[M8a upload] ✅ 挂 uploads 批次 ds={up_j['dataset_id']}；"
              f"delete_file=true 连文件删除", flush=True)

        # 33) 孤儿文件扫描/清理
        orphan = Path(str(get_settings().storage("uploads"))) / "orphan_m8.png"
        orphan.write_bytes(b"not an image")
        r = client.post("/api/maintenance/orphan_scan").json()
        t = wait_task(task_manager, r["task_id"], timeout=120)
        assert t["status"] == "done", t["message"]
        assert str(orphan) in t["result"]["orphans"], t["result"]
        r = client.post("/api/maintenance/orphan_clean",
                        json={"paths": [str(orphan)]}).json()
        assert r["deleted"] == 1 and not orphan.exists(), r
        print(f"[M8a orphan] ✅ scan 扫到孤儿文件，clean 删除 "
              f"(n_scanned={t['result']['n_scanned']})", flush=True)

        # 34) 统计按品类 + logs 过滤
        with session_scope() as s:
            s.add(Detection(target_type="image", image_path="x_other_m8.png",
                            category="other_m8", final_score=0.9,
                            is_anomaly=True, latency_ms=5.0, n_tiles={}))
        g = client.get("/api/stats/overview").json()
        c = client.get(f"/api/stats/overview?category={CAT}").json()
        assert c["totals"]["n_detections"] == g["totals"]["n_detections"] - 1, \
            (c["totals"], g["totals"])
        assert c["totals"]["n_anomalies"] == g["totals"]["n_anomalies"] - 1
        ts = client.get(f"/api/stats/timeseries?category={CAT}").json()["items"]
        assert ts, "按品类时序为空"
        assert sum(i["n_inspected"] for i in ts) == c["totals"]["n_detections"]
        for k in ("date", "n_inspected", "n_anomaly", "avg_latency_ms"):
            assert k in ts[0], ts[0]
        ts_g = client.get("/api/stats/timeseries").json()["items"]
        assert ts_g, "全局时序为空"
        logs_c = client.get(
            f"/api/logs?action=annotate_image&category={M8CAT}").json()["items"]
        assert logs_c and all(
            l["extra"]["category"] == M8CAT for l in logs_c)
        print(f"[M8a stats/logs] ✅ overview?category 与全局差 1（other_m8）；"
              f"timeseries?category 合计={c['totals']['n_detections']}；"
              f"logs action+category 过滤生效", flush=True)

        # 35) 反馈图不在 Image 表 -> 自动登记 split=feedback + 反馈回流批次
        fb_img = Path(TMP) / "m8_fb.png"
        _shutil.copy2(defects[0], fb_img)   # 新路径，Image 表中不存在
        det_fb = client.post("/api/detect/image",
                             json={"path": str(fb_img), "category": CAT,
                                   "with_heatmap": False}).json()
        r = client.post("/api/feedback", json={
            "detection_id": det_fb["detection_id"],
            "feedback_type": "confirmed", "operator_label": 1}).json()
        assert r["engine_update"] is not None, r
        with session_scope() as s:
            row = (s.query(ImageRow)
                   .filter(ImageRow.path == str(fb_img)).one())
            assert row.split == "feedback" and row.source == "feedback", (
                row.split, row.source)
            assert row.label == "anomaly"
            ds_fb = s.get(Dataset, row.dataset_id)
            assert ds_fb.name == "反馈回流" \
                and ds_fb.source_type == "feedback", ds_fb.name
            assert ds_fb.n_images >= 1
        print(f"[M8a feedback] ✅ 反馈图自动登记 split=feedback，"
              f"挂反馈回流批次 ds={ds_fb.id}", flush=True)

        # 36) /images/tree 第四维 dataset（向后兼容保留 tree）
        tree = client.get("/api/images/tree").json()
        assert "tree" in tree and "datasets" in tree, list(tree)
        assert tree["tree"], "原三维 tree 为空"
        cat_ds = tree["datasets"].get(CAT, {})
        assert str(legacy[0]["id"]) in cat_ds, cat_ds
        assert cat_ds[str(legacy[0]["id"])]["name"] == "历史未分组"
        assert str(ds_fb.id) in cat_ds
        print(f"[M8a tree] ✅ 第四维 datasets 生效（{CAT} 含 "
              f"{len(cat_ds)} 个批次节点），原 tree 保留", flush=True)

    print("✅ M2 + M2b + M5a + M6a + M7a + M8a 冒烟通过", flush=True)


if __name__ == "__main__":
    main()
