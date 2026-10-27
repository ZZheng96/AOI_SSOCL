"""REMAIN-02：Core 特征工程页 API 真实 HTTP 联调（requests，非 TestClient）。

阶段：
  A 鉴权/角色机制   B 只读端点×角色   C draft（含隔离快照）
  D report（后台任务） E admission/transition/register 门禁
  F 阈值冻结 409     G 候选路由角色矩阵 + 依赖锁
  H rebuild（后台任务，验证不自动激活）
每个请求落 raw/NN_*.json：方法/路径/身份/请求体/状态码/响应。
"""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import requests

BASE = "http://127.0.0.1:8031"
TASK = Path(r"d:\CGAIC\AOI_SSOCL\review\task\remain_02")
RAW = TASK / "raw"
TEMP = TASK / "temp_core"
CAT = "gold_finger"
KEYS = {"operator": "remain02-operator-key",
        "engineer": "remain02-engineer-key",
        "admin": "remain02-admin-key"}
SUMMARY = []
_n = [0]


def snap(label):
    """TEMP DB 表级快照 + 生产指针/注册表指针哈希。"""
    out = RAW / f"snap_{label}.json"
    subprocess.run([sys.executable, str(TASK / "db_dump.py"),
                    str(TEMP / "database" / "aoi.db"), str(out)],
                   check=True, capture_output=True, text=True)
    data = json.loads(out.read_text(encoding="utf-8"))
    for name, p in {
        "engine_current_pointer":
            TEMP / "storage" / "engine" / "snapshots" / CAT / "current.json",
        "feature_revision_registry":
            TEMP / "storage" / "feature_revisions" / "registry.json",
        "feature_revision_production_pointer":
            TEMP / "storage" / "feature_revisions" / "registry.production.json",
    }.items():
        if p.is_file():
            data[name] = {"path": str(p),
                          "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        else:
            data[name] = {"path": str(p), "status": "missing"}
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    return data


def req(name, method, path, role="admin", body="__none__", note="",
        expect=None):
    _n[0] += 1
    headers = {"X-API-Key": KEYS[role]} if role else {}
    kwargs = {"headers": headers, "timeout": 60}
    if body != "__none__":
        kwargs["json"] = body
    r = requests.request(method, BASE + path, **kwargs)
    try:
        resp = r.json()
    except Exception:
        resp = {"_text": r.text[:2000]}
    rec = {"name": name, "method": method, "path": path, "role": role or "NONE",
           "request_body": None if body == "__none__" else body,
           "status_code": r.status_code, "response": resp, "note": note}
    (RAW / f"{_n[0]:02d}_{name}.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    ok = "?" if expect is None else ("PASS" if r.status_code in expect else "FAIL")
    SUMMARY.append({"name": name, "method": method, "path": path,
                    "role": role or "NONE", "status": r.status_code,
                    "expect": sorted(expect) if expect else None, "verdict": ok,
                    "note": note})
    print(f"[{_n[0]:02d}] {ok} {method} {path} role={role or 'NONE'} "
          f"-> {r.status_code} (expect {sorted(expect) if expect else '-'})",
          flush=True)
    return r.status_code, resp


def wait_task(tid, label, timeout=900):
    t0 = time.time()
    while True:
        r = requests.get(f"{BASE}/api/tasks/{tid}",
                         headers={"X-API-Key": KEYS["admin"]}, timeout=30)
        t = r.json()
        st = t.get("status")
        el = time.time() - t0
        print(f"    [{label}] task={tid} status={st} "
              f"progress={t.get('progress')} msg={str(t.get('message'))[:60]} "
              f"({el:.0f}s)", flush=True)
        if st in ("done", "failed") or el > timeout:
            rec = {"task_id": tid, "final": t, "elapsed_s": round(el, 1)}
            (RAW / f"task_{label}_{tid}.json").write_text(
                json.dumps(rec, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8")
            return t
        time.sleep(8)


print("=== A. 鉴权与角色机制 ===", flush=True)
r = requests.get(BASE + "/api/health", timeout=30)
print(f"[A0] GET /api/health 无密钥 -> {r.status_code} {r.text[:120]}", flush=True)
SUMMARY.append({"name": "health_no_key", "method": "GET", "path": "/api/health",
                "role": "NONE", "status": r.status_code, "expect": [200],
                "verdict": "PASS" if r.status_code == 200 else "FAIL",
                "note": "探活豁免"})
req("sysinfo_no_key", "GET", "/api/system/info", role=None, expect=[401],
    note="鉴权开启后无密钥一律 401")
r = requests.get(BASE + "/api/system/info",
                 headers={"X-API-Key": "wrong-key"}, timeout=30)
print(f"[A2] GET /api/system/info 错误密钥 -> {r.status_code}", flush=True)
SUMMARY.append({"name": "sysinfo_wrong_key", "method": "GET",
                "path": "/api/system/info", "role": "WRONG",
                "status": r.status_code, "expect": [401],
                "verdict": "PASS" if r.status_code == 401 else "FAIL",
                "note": "错误密钥 401"})
for role in ("operator", "engineer", "admin"):
    sc, resp = req(f"sysinfo_{role}", "GET", "/api/system/info", role=role,
                   expect=[200], note="回显角色与依赖锁/治理决策")
    assert resp.get("role") == role, f"角色回显不符: {resp.get('role')}"
print("    角色回显 operator/engineer/admin 均正确", flush=True)

print("=== B. 只读端点 × 角色 ===", flush=True)
for role in ("operator", "engineer", "admin"):
    req(f"revisions_list_{role}", "GET", "/api/models/feature-revisions",
        role=role, expect=[200],
        note="含 production_binding/changes_production_extractor 字段")
for role in ("operator", "engineer", "admin"):
    sc, resp = req(f"feature_set_get_{role}", "GET",
                   f"/api/models/feature-set/{CAT}", role=role, expect=[200],
                   note="操作员只读 can_edit=false")
    print(f"    can_edit={resp.get('can_edit')} role={resp.get('role')} "
          f"status={resp.get('status')} version={resp.get('version')}",
          flush=True)
req("feature_set_unknown_cat", "GET", "/api/models/feature-set/__no_such_cat__",
    role="operator", expect=[200], note="未准备品类 enabled=false")
req("hard_pool_operator", "GET", f"/api/models/feature-set/{CAT}/hard-pool",
    role="operator", expect=[403], note="难例池 engineer 起")
req("hard_pool_engineer", "GET", f"/api/models/feature-set/{CAT}/hard-pool",
    role="engineer", expect=[200])
req("slot_health_operator", "GET", f"/api/models/{CAT}/slot-health",
    role="operator", expect=[403], note="槽位健康 engineer 起")
req("slot_health_engineer", "GET", f"/api/models/{CAT}/slot-health",
    role="engineer", expect=[200])

print("=== C. draft 隔离验证 ===", flush=True)
req("draft_operator", "POST", f"/api/models/feature-set/{CAT}/draft",
    role="operator", body={"groups": ["glcm"], "scoring": "grouped"},
    expect=[403])
req("draft_bad_groups", "POST", f"/api/models/feature-set/{CAT}/draft",
    role="engineer", body={"groups": ["nope"]}, expect=[400])
req("draft_bad_scoring", "POST", f"/api/models/feature-set/{CAT}/draft",
    role="engineer", body={"groups": ["glcm"], "scoring": "max"}, expect=[400])
snap_before = snap("before_draft")
sc, resp = req("draft_engineer", "POST", f"/api/models/feature-set/{CAT}/draft",
               role="engineer",
               body={"groups": ["glcm"], "scoring": "grouped",
                     "note": "REMAIN-02 联调"},
               expect=[200], note="changes_production 必须为 false")
RID = resp.get("revision_id")
print(f"    revision_id={RID}", flush=True)
sc, resp2 = req("draft_admin_idempotent", "POST",
                f"/api/models/feature-set/{CAT}/draft", role="admin",
                body={"groups": ["glcm"], "scoring": "grouped",
                      "note": "REMAIN-02 联调"},
                expect=[200], note="幂等：同 manifest 同 revision_id")
print(f"    幂等={resp2.get('revision_id') == RID}", flush=True)
snap_after = snap("after_draft")
sc, fs_after = req("feature_set_after_draft", "GET",
                   f"/api/models/feature-set/{CAT}", role="operator",
                   expect=[200], note="draft 后生产特征集不变")
isolation = {
    "db_business_tables_unchanged": True, "diffs": {},
}
for t in ("images", "models", "eval_runs"):
    b = json.dumps(snap_before["tables"][t], sort_keys=True, default=str)
    a = json.dumps(snap_after["tables"][t], sort_keys=True, default=str)
    if b != a:
        isolation["db_business_tables_unchanged"] = False
        isolation["diffs"][t] = "CHANGED"
isolation["engine_current_pointer_unchanged"] = (
    snap_before["engine_current_pointer"].get("sha256")
    == snap_after["engine_current_pointer"].get("sha256"))
isolation["production_pointer_missing_before"] = (
    snap_before["feature_revision_production_pointer"].get("status") == "missing")
isolation["production_pointer_missing_after"] = (
    snap_after["feature_revision_production_pointer"].get("status") == "missing")
(RAW / "isolation_draft.json").write_text(
    json.dumps(isolation, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"    draft 隔离: {json.dumps(isolation, ensure_ascii=False)}", flush=True)
sc, rev = req("revisions_list_after_draft", "GET",
              "/api/models/feature-revisions", role="operator", expect=[200],
              note="draft 已登记；production_binding 保持 unknown")
print(f"    production_binding={rev.get('production_binding')} "
      f"changes_production_extractor={rev.get('changes_production_extractor')}",
      flush=True)

print("=== D. report（后台任务）===", flush=True)
req("report_operator", "POST", f"/api/models/feature-set/{CAT}/report",
    role="operator", expect=[403])
sc, resp = req("report_engineer", "POST", f"/api/models/feature-set/{CAT}/report",
               role="engineer", expect=[200], note="queued -> 后台贡献档案")
tid = resp.get("task_id")
if sc == 200 and tid:
    t = wait_task(tid, "report", timeout=600)
    print(f"    report 任务终态: {t.get('status')}", flush=True)
    req("report_again_running_or_done", "POST",
        f"/api/models/feature-set/{CAT}/report", role="engineer",
        expect=[200], note="完成后再次请求（应重新排队或提示）")
    sc, fsn = req("feature_set_after_report", "GET",
                  f"/api/models/feature-set/{CAT}", role="engineer",
                  expect=[200], note="档案状态应转 current")
    print(f"    report.status={ (fsn.get('report') or {}).get('status') }",
          flush=True)
snap("after_report")

print("=== E. admission / transition / register 门禁 ===", flush=True)
req("admission_operator", "POST", f"/api/models/feature-revisions/{RID}/admission",
    role="operator", body={"limit": 8}, expect=[403])
req("admission_no_such", "POST", "/api/models/feature-revisions/nope/admission",
    role="engineer", body={}, expect=[409])
sc, resp = req("admission_engineer", "POST",
               f"/api/models/feature-revisions/{RID}/admission",
               role="engineer", body={"limit": 8}, expect=[200],
               note="只产证据 activated=false")
tid = resp.get("task_id")
if sc == 200 and tid:
    t = wait_task(tid, "admission", timeout=600)
    print(f"    admission 任务终态: {t.get('status')}", flush=True)
req("transition_engineer", "POST",
    f"/api/models/feature-revisions/{RID}/transition", role="engineer",
    body={"target": "evaluated", "actor": "eng"}, expect=[403],
    note="门禁 transition 需 admin")
sc, resp = req("transition_admin_no_evidence", "POST",
               f"/api/models/feature-revisions/{RID}/transition", role="admin",
               body={"target": "evaluated", "actor": "remain02-admin"},
               expect=[409], note="无独立评估证据 → 拒绝")
print(f"    detail={json.dumps(resp.get('detail'), ensure_ascii=False)[:200]}",
      flush=True)
req("transition_admin_skip_level", "POST",
    f"/api/models/feature-revisions/{RID}/transition", role="admin",
    body={"target": "active", "actor": "remain02-admin"}, expect=[409],
    note="禁止越级激活")
req("register_operator", "POST", "/api/models/feature-revisions",
    role="operator", body={}, expect=[403])
req("register_engineer_invalid", "POST", "/api/models/feature-revisions",
    role="engineer", body={"schema_version": 1}, expect=[409],
    note="manifest 血缘不全 → 409")

print("=== F. 阈值冻结 409 THRESHOLD_FROZEN ===", flush=True)
req("config_freeze_operator", "POST", "/api/system/config", role="operator",
    body={"governance_decisions.gt_threshold_frozen": True}, expect=[403])
req("config_freeze_engineer", "POST", "/api/system/config", role="engineer",
    body={"governance_decisions.gt_threshold_frozen": True}, expect=[403])
req("config_freeze_admin", "POST", "/api/system/config", role="admin",
    body={"governance_decisions.gt_threshold_frozen": True}, expect=[200])
sc, resp = req("config_threshold_frozen_409", "POST", "/api/system/config",
               role="admin", body={"business_budget.g2_max_fpr_pct": 10.0},
               expect=[409], note="冻结期改阈值 → 409 THRESHOLD_FROZEN")
print(f"    reason_code={ (resp.get('detail') or {}).get('reason_code') if isinstance(resp.get('detail'), dict) else resp.get('detail') }",
      flush=True)
req("config_unfreeze_admin", "POST", "/api/system/config", role="admin",
    body={"governance_decisions.gt_threshold_frozen": False}, expect=[200])
req("config_threshold_after_unfreeze", "POST", "/api/system/config", role="admin",
    body={"business_budget.g2_max_fpr_pct": 5.0}, expect=[200],
    note="解冻后可改")

print("=== G. 候选路由角色矩阵 + 依赖锁 ===", flush=True)
req("candidates_prepared_operator", "GET", "/api/candidates/recovery/prepared",
    role="operator", expect=[403])
req("candidates_prepared_engineer", "GET", "/api/candidates/recovery/prepared",
    role="engineer", expect=[200])
req("candidates_decision_operator", "POST", f"/api/candidates/{CAT}/1/decision",
    role="operator", body={"approved": True, "report_hash": "x", "reason": "r"},
    expect=[403])
req("candidates_decision_engineer", "POST", f"/api/candidates/{CAT}/1/decision",
    role="engineer", body={"approved": True, "report_hash": "x", "reason": "r"},
    expect=[403], note="decision 需 admin")
sc, resp = req("candidates_decision_admin_bogus", "POST",
               f"/api/candidates/{CAT}/999/decision", role="admin",
               body={"approved": True, "report_hash": "0" * 64,
                     "reason": "REMAIN-02  bogus"},
               expect=[409], note="不存在的候选 → 409")
sc, info = req("sysinfo_deplock", "GET", "/api/system/info", role="admin",
               expect=[200], note="读取 dependency_lock 状态")
dl = info.get("dependency_lock") or {}
print(f"    dependency_lock.ok={dl.get('ok')} mismatches={len(dl.get('mismatches') or [])} "
      f"missing={len(dl.get('missing') or [])} policy={dl.get('policy')}", flush=True)
if dl.get("ok") is False:
    req("config_deplock_block", "POST", "/api/system/config", role="admin",
        body={"governance_decisions.dependency_lock_policy": "block"},
        expect=[200])
    sc, resp = req("candidates_activate_deplock_409", "POST",
                   f"/api/candidates/{CAT}/1/activate", role="admin",
                   expect=[409],
                   note="锁不一致+block → DEPENDENCY_LOCK_MISMATCH；指针不得变")
    d = resp.get("detail")
    print(f"    reason_code={d.get('reason_code') if isinstance(d, dict) else d}",
          flush=True)
    req("config_deplock_warn", "POST", "/api/system/config", role="admin",
        body={"governance_decisions.dependency_lock_policy": "warn"},
        expect=[200])
else:
    print("    依赖锁一致，DEPENDENCY_LOCK_MISMATCH 不可触发（记为缺口）",
          flush=True)
    SUMMARY.append({"name": "candidates_activate_deplock_409", "method": "POST",
                    "path": f"/api/candidates/{CAT}/1/activate", "role": "admin",
                    "status": None, "expect": [409], "verdict": "GAP",
                    "note": "环境依赖锁一致，block 路径无法实测"})

print("=== H. rebuild（后台任务，验证不自动激活）===", flush=True)
snap_before_rebuild = snap("before_rebuild")
req("rebuild_operator", "POST", f"/api/models/feature-set/{CAT}/rebuild",
    role="operator", body={}, expect=[403])
req("rebuild_no_active_cat", "POST",
    "/api/models/feature-set/__no_such_cat__/rebuild", role="engineer",
    body={}, expect=[409], note="无激活版本 → 409")
sc, resp = req("rebuild_engineer", "POST",
               f"/api/models/feature-set/{CAT}/rebuild", role="engineer",
               body={"note": "REMAIN-02 联调"}, expect=[200],
               note="activated=false；parent=当前版本")
tid = resp.get("task_id")
if sc == 200 and tid:
    t = wait_task(tid, "rebuild", timeout=1800)
    res = t.get("result") or {}
    print(f"    rebuild 终态={t.get('status')} activated={res.get('activated')} "
          f"auto_report={res.get('auto_report')} version={res.get('version')}",
          flush=True)
    snap_after_rebuild = snap("after_rebuild")
    unchanged = (snap_before_rebuild["engine_current_pointer"].get("sha256")
                 == snap_after_rebuild["engine_current_pointer"].get("sha256"))
    print(f"    rebuild 后 current.json 指针不变={unchanged}", flush=True)
    sc, fsn = req("feature_set_after_rebuild", "GET",
                  f"/api/models/feature-set/{CAT}", role="engineer",
                  expect=[200],
                  note="version 仍为旧版；pending_rebuild_version 出现")
    print(f"    version={fsn.get('version')} "
          f"pending_rebuild_version={fsn.get('pending_rebuild_version')}",
          flush=True)
    iso = {"current_pointer_unchanged": unchanged,
           "task_activated": res.get("activated"),
           "task_auto_report": res.get("auto_report"),
           "feature_set_version": fsn.get("version"),
           "pending_rebuild_version": fsn.get("pending_rebuild_version"),
           "production_pointer_status":
               snap_after_rebuild["feature_revision_production_pointer"].get(
                   "status")}
    (RAW / "isolation_rebuild.json").write_text(
        json.dumps(iso, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")

(RAW / "summary.json").write_text(
    json.dumps(SUMMARY, ensure_ascii=False, indent=2), encoding="utf-8")
n_fail = sum(1 for s in SUMMARY if s["verdict"] == "FAIL")
print(f"=== 完成：{len(SUMMARY)} 项，FAIL={n_fail} ===", flush=True)
print("HTTP_ACCEPT_DONE", flush=True)
