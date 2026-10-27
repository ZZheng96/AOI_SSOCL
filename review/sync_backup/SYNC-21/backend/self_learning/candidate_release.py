"""显式候选验收与发布账本；不替代生产G1/G2验收或多进程缓存协议。"""
from contextlib import contextmanager
import hashlib
import json
import math
import os
import sqlite3
import time
import uuid
from pathlib import Path


def encode(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(encode(value).encode()).hexdigest()


class CandidateReleaseError(ValueError):
    """可安全映射到 API 的候选发布拒绝原因。"""

    def __init__(self, message, reason_code, violations=None, retryable=False):
        super().__init__(message)
        self.reason_code = reason_code
        self.violations = list(violations or [])
        self.retryable = bool(retryable)


def _violation(code, message, **details):
    item = {"code": code, "message": message}
    if details:
        item["details"] = details
    return item


def failure_details(exc, phase):
    """将候选失败统一为稳定机器字段，同时保留原始文本。"""
    if isinstance(exc, CandidateReleaseError):
        return {"reason_code": exc.reason_code, "violations": list(exc.violations),
                "retryable": exc.retryable}
    code = "CANDIDATE_BUILD_FAILED" if phase == "build" else "CANDIDATE_ACCEPTANCE_FAILED"
    retryable = getattr(exc, "state", None) in ("deferred", "cancelled")
    return {"reason_code": code,
            "violations": [_violation(code, str(exc))], "retryable": retryable}


class CandidateRelease:
    def __init__(self, engine):
        self.engine = engine

    def _mark_blocked(self, candidate_id, reason_code, connection=None):
        if connection is not None:
            connection.execute("UPDATE candidate_release SET state=? WHERE candidate_id=? AND state IN ('validated','approved')",
                              ("blocked", candidate_id))
            return
        with self.connection() as c:
            c.execute("UPDATE candidate_release SET state=? WHERE candidate_id=? AND state IN ('validated','approved')",
                      ("blocked", candidate_id))

    def _guard_error(self, message, reason_code, violations=None, retryable=False):
        violations = violations if violations is not None else [_violation(reason_code, message)]
        raise CandidateReleaseError(message, reason_code, violations, retryable)

    @staticmethod
    def _failure(message, reason_code="CANDIDATE_ACCEPTANCE_FAILED", *, violations=None,
                 retryable=False):
        return CandidateReleaseError(message, reason_code, violations, retryable)

    @contextmanager
    def connection(self):
        c = self.engine._feedback_receipts()
        c.execute("CREATE TABLE IF NOT EXISTS candidate_release (candidate_id TEXT PRIMARY KEY, report TEXT NOT NULL, state TEXT NOT NULL, decision TEXT)")
        c.execute("CREATE TABLE IF NOT EXISTS release_events (id INTEGER PRIMARY KEY, category TEXT NOT NULL, candidate_id TEXT, action TEXT NOT NULL, state TEXT NOT NULL, payload TEXT NOT NULL)")
        try:
            with c:
                yield c
        finally:
            c.close()

    @contextmanager
    def readonly_connection(self):
        """打开已有收据库的只读连接；历史未建库/未建表时返回空结果，不创建任何对象。"""
        path = os.path.join(self.engine.storage_dir, "feedback_receipts.sqlite3")
        if not os.path.isfile(path):
            yield None
            return
        uri = "file:" + path.replace("\\", "/") + "?mode=ro"
        c = sqlite3.connect(uri, uri=True, timeout=30)
        try:
            yield c
        finally:
            c.close()

    def check_category(self, category):
        if not category or category in (".", "..") or any(c in category for c in ("/", "\\", ":")):
            raise ValueError("非法品类路径")

    def discover_prepared(self, category=None):
        """只读发现prepared发布事件；不建库、不建表、不改指针、DB或快照。"""
        if category is not None:
            self.check_category(category)
        with self.readonly_connection() as c:
            if c is None:
                return []
            tables = {row[0] for row in c.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='release_events'" )}
            if "release_events" not in tables:
                return []
            query = ("SELECT id,category,candidate_id,action,state,payload "
                     "FROM release_events WHERE state='prepared'")
            args = ()
            if category is not None:
                query += " AND category=?"
                args = (category,)
            rows = c.execute(query + " ORDER BY id", args).fetchall()
        result = []
        for event_id, item_category, candidate_id, action, state, raw in rows:
            item = {"event_id": event_id, "category": item_category,
                    "candidate_id": candidate_id, "action": action,
                    "state": state, "pointer_consistency": "manual_required",
                    "wal_consistent": False, "recommendation": "manual_required"}
            try:
                payload = json.loads(raw)
                item["payload"] = payload
                current = self.engine.current_version(item_category)
                item["current_version"] = current
                item["wal"] = self.wal(item_category)
                item["wal_consistent"] = item["wal"] == payload.get("event_watermark")
                if current == payload.get("target_version"):
                    item["pointer_consistency"] = "target"
                    item["recommendation"] = "recover"
                elif current == payload.get("from_version"):
                    item["pointer_consistency"] = "parent"
                    item["recommendation"] = "manual_abort_or_recover"
                else:
                    item["pointer_consistency"] = "conflict"
                    item["recommendation"] = "manual_required"
                if not item["wal_consistent"]:
                    item["recommendation"] = "manual_required"
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                item["error"] = str(exc)
            result.append(item)
        return result

    def replay(self, category, pipe, version):
        """独立重放，不写生产事件收据；资产缺失或失败必须关闭发布。"""
        path = Path(self.engine._journal_path(category))
        seen = {}
        for line in path.read_text(encoding="utf-8").splitlines() if path.exists() else []:
            rec = json.loads(line)
            if rec.get("base") != version:
                continue
            key, payload = self.engine._feedback_identity(category, rec.get("path"), rec.get("verdict"), rec.get("label"), rec.get("box"), rec.get("feedback_id"), rec.get("event_id"))
            if key is not None and key in seen:
                if seen[key] != payload:
                    raise ValueError("WAL事件冲突")
                continue
            if not Path(rec["path"]).is_file():
                raise ValueError("WAL资产缺失")
            pipe.feedback(rec["path"], verdict=rec.get("verdict"), label=rec.get("label"), box=rec.get("box"))
            if key is not None:
                seen[key] = payload

    def intent(self, category, version):
        self.check_category(category)
        with self.connection() as c:
            rows = c.execute("SELECT event_key,result FROM candidate_events WHERE state='completed'").fetchall()
        for key, raw in rows:
            value = json.loads(raw)
            result = value.get("result", {})
            if result.get("category") == category and result.get("version_num") == version:
                return key, value
        raise ValueError("候选尚未完成或缺少持久意图")

    def wal(self, category):
        p = Path(self.engine._journal_path(category))
        raw = p.read_bytes() if p.exists() else b""
        return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                "records": len(raw.splitlines())}

    def _validate_feedback_guard(self, intent):
        event_id = intent.get("result", {}).get("event_id", "")
        if not isinstance(event_id, str) or not event_id.startswith("feedback:"):
            return
        try:
            feedback_id = int(event_id.split(":", 1)[1])
        except (TypeError, ValueError):
            raise ValueError("候选反馈身份无效")
        from ..db.database import session_scope
        from ..db.models import Detection, Feedback
        with session_scope() as s:
            fb = s.get(Feedback, feedback_id)
            if fb is None:
                raise ValueError("候选关联反馈不存在")
            guard = (fb.learning_trace or {}).get("candidate_guard")
            if not isinstance(guard, dict):
                return
            if fb.invalidated:
                self._guard_error("候选关联反馈已作废，需从可信parent重建", "PARENT_REBUILD_REQUIRED_INVALIDATED_FEEDBACK")
            det = s.get(Detection, guard.get("detection_id"))
            if det is None:
                raise ValueError("候选关联检测不存在")
            review = (det.n_tiles or {}).get("region_review") or {}
            current_revision = int(review.get("revision", 0))
            if current_revision != int(guard.get("region_revision", 0)):
                self._guard_error("候选关联检测已发生人工修订，需从可信parent重建", "PARENT_REBUILD_REQUIRED_REVIEW_REVISION")

    def validate(self, category, version, intent):
        try:
            self._validate_feedback_guard(intent)
        except CandidateReleaseError as exc:
            event_id = intent.get("result", {}).get("event_id", "")
            self._mark_blocked("candidate:" + event_id, exc.reason_code)
            raise
        e = self.engine
        if e.current_version(category) != intent["result"]["parent_version"]:
            raise ValueError("候选父版本已过期")
        if self.wal(category)["sha256"] != intent["wal_sha256"]:
            raise ValueError("候选父WAL已变化")
        for v, files in ((version, intent["files"]), (intent["result"]["parent_version"], intent["parent_files"])):
            if e._candidate_files(e._v_dir(category, v)) != files:
                raise ValueError("候选或父快照完整性校验失败")

    def evaluate(self, category, version, domains, policy, data_cutoff,
                 progress_cb=lambda *a: None, replay_evidence=None):
        """domains由服务从登记图像读取，不能接受客户端预计算指标。"""
        from algo.persist import load_snapshot
        key, intent = self.intent(category, version)
        e = self.engine
        report = {"candidate_id": key, "category": category, "version": version,
                  "parent_version": intent["result"]["parent_version"],
                  "wal": self.wal(category), "data_cutoff": data_cutoff,
                  "config_hash": digest({"files": intent["files"], "policy": policy}),
                  "policy": policy, "domains": {}, "created_at": time.time(),
                  "replay_evidence": replay_evidence,
                  "acceptance": "unavailable", "reason": "缺少新旧域数据或冻结政策",
                  "reason_code": "CANDIDATE_ACCEPTANCE_UNAVAILABLE",
                  "violations": [], "retryable": False}
        try:
            self.validate(category, version, intent)
        except CandidateReleaseError:
            raise
        except Exception as exc:
            details = failure_details(exc, "acceptance")
            raise CandidateReleaseError(str(exc), details["reason_code"],
                                        details["violations"], details["retryable"])
        if replay_evidence is not None:
            self._validate_replay_evidence(category, intent["result"]["parent_version"], replay_evidence)
        budget = intent["result"].get("candidate_budget", {})
        report["budget"] = budget
        required = {"max_fpr", "min_recall", "max_gray", "old_recall_drop", "max_predict_ms", "valid_seconds"}
        ready = (set(policy) == required and all(isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0 for v in policy.values()) and policy.get("valid_seconds", 0) > 0
                 and all(0 <= policy[k] <= 1 for k in ("max_fpr", "min_recall", "max_gray", "old_recall_drop")))
        available = ready and set(domains) == {"old", "new"} and all({r["label"] for r in rows} == {0, 1} for rows in domains.values())
        budget_ok = bool(budget.get("limits")) and all(0 <= budget["used"].get(k, -1) <= v for k, v in budget.get("limits", {}).items())
        report["budget_passed"] = budget_ok
        # 显式数据身份与字节hash绑定；拒绝候选事件/父WAL训练资产复用。
        training = {json.loads(self._payload(key))[1]}
        wp = Path(e._journal_path(category))
        if wp.exists():
            training.update(json.loads(line).get("path") for line in wp.read_text(encoding="utf-8").splitlines())
        paths = [r["path"] for rows in domains.values() for r in rows]
        if len(paths) != len(set(paths)) or any(p in training for p in paths):
            raise ValueError("审计域重复或与候选/WAL训练数据重叠")
        report["data"] = {d: [{**r, "sha256": hashlib.sha256(Path(r["path"]).read_bytes()).hexdigest()} for r in rows] for d, rows in domains.items()}
        if available:
            pipes = []
            try:
                for v in (report["parent_version"], version):
                    pipes.append(load_snapshot(e._v_dir(category, v), e.device))
                self.replay(category, pipes[0], report["parent_version"])
                for domain, rows in domains.items():
                    metrics = []
                    for pipe in pipes:
                        predictions = []
                        for r in rows:
                            t = time.perf_counter()
                            rec = pipe.predict(r["path"])
                            elapsed = (time.perf_counter() - t) * 1000
                            if rec.get("decision") not in ("normal", "anomaly", "gray") or not math.isfinite(float(rec["fused"])):
                                raise ValueError("独立验收输出非法")
                            predictions.append({"id": r["id"], "label": r["label"], "decision": rec["decision"], "score": float(rec["fused"]), "predict_ms": elapsed})
                            progress_cb(len(predictions), len(rows), "独立域验收: " + domain)
                        normal = [p for p in predictions if p["label"] == 0]
                        defect = [p for p in predictions if p["label"] == 1]
                        metrics.append({"fpr": sum(p["decision"] == "anomaly" for p in normal) / len(normal),
                                        "recall": sum(p["decision"] == "anomaly" for p in defect) / len(defect),
                                        "gray": sum(p["decision"] == "gray" for p in predictions) / len(predictions),
                                        "max_predict_ms": max(p["predict_ms"] for p in predictions), "predictions": predictions})
                    report["domains"][domain] = {"parent": metrics[0], "candidate": metrics[1]}
                passed = budget_ok
                for domain, m in report["domains"].items():
                    a, b = m["parent"], m["candidate"]
                    passed &= b["fpr"] <= policy["max_fpr"] and b["recall"] >= policy["min_recall"] and b["gray"] <= policy["max_gray"] and b["max_predict_ms"] <= policy["max_predict_ms"]
                    if domain == "old":
                        passed &= b["recall"] >= a["recall"] - policy["old_recall_drop"]
                violations = []
                for domain, m in report["domains"].items():
                    candidate = m["candidate"]
                    checks = (("max_fpr", candidate["fpr"] <= policy["max_fpr"]),
                              ("min_recall", candidate["recall"] >= policy["min_recall"]),
                              ("max_gray", candidate["gray"] <= policy["max_gray"]),
                              ("max_predict_ms", candidate["max_predict_ms"] <= policy["max_predict_ms"]))
                    violations.extend(_violation("POLICY_" + key.upper(),
                                                  "独立域指标未满足政策", domain=domain,
                                                  value=candidate[key], limit=policy[key])
                                      for key, ok in checks if not ok)
                if not budget_ok:
                    violations.append(_violation("CANDIDATE_BUDGET_EXCEEDED", "候选计数预算未通过"))
                report.update(acceptance="passed" if passed else "rejected",
                              reason="独立域指标及计数预算",
                              reason_code="CANDIDATE_ACCEPTED" if passed else "CANDIDATE_ACCEPTANCE_REJECTED",
                              violations=violations, retryable=False)
            finally:
                for pipe in pipes:
                    pipe._cpu_exec.shutdown()
        self.validate(category, version, intent)
        if replay_evidence is not None:
            self._validate_replay_evidence(category, intent["result"]["parent_version"], replay_evidence)
        report["report_hash"] = digest(report)
        with e._pipe_lock(category), e._write_lock, self.connection() as c:
            c.execute("BEGIN IMMEDIATE")
            self.validate(category, version, intent)
            old = c.execute("SELECT state FROM candidate_release WHERE candidate_id=?", (key,)).fetchone()
            if old and old[0] in ("approved", "activated", "rolled_back", "rejected"):
                raise ValueError("候选已有决定，禁止覆盖报告")
            c.execute("INSERT OR REPLACE INTO candidate_release VALUES (?,?,?,NULL)", (key, encode(report), "validated" if report["acceptance"] == "passed" else report["acceptance"]))
        return report

    def _validate_replay_evidence(self, category, parent_version, evidence):
        required = ("replay_run_id", "parent_version", "wal_watermark",
                    "input_summary", "config_hash")
        if any(key not in evidence for key in required):
            raise ValueError("回放证据不完整")
        if evidence["parent_version"] != parent_version:
            raise ValueError("回放证据父版本不匹配")
        if self.engine.current_version(category) != parent_version:
            raise ValueError("回放证据父版本已过期")
        if evidence["wal_watermark"] != self.engine._journal_mark(category):
            raise ValueError("回放证据WAL水位不匹配")
        if not evidence["replay_run_id"]:
            raise ValueError("回放证据缺少replay_run_id")

    def _payload(self, key):
        with self.connection() as c:
            return c.execute("SELECT payload FROM candidate_events WHERE event_key=?", (key,)).fetchone()[0]

    def status(self, category, version):
        key, _ = self.intent(category, version)
        with self.connection() as c:
            row = c.execute("SELECT report,state,decision FROM candidate_release WHERE candidate_id=?", (key,)).fetchone()
            events = c.execute("SELECT id,candidate_id,action,state,payload FROM release_events WHERE category=? ORDER BY id", (category,)).fetchall()
        result = []
        for event_id, event_candidate_id, action, state, raw in events:
            payload = json.loads(raw)
            if event_candidate_id == key:
                resolved_candidate_id = key
            elif event_candidate_id is None and payload.get("candidate_event") == key:
                resolved_candidate_id = key
            elif event_candidate_id is not None:
                continue
            else:
                result.append({"id": event_id, "candidate_id": None, "action": action,
                               "state": "orphan", "availability": "unavailable", "payload": payload})
                continue
            result.append({"id": event_id, "candidate_id": resolved_candidate_id,
                           "action": action, "state": state, "payload": payload})
        return {"candidate_id": key, "state": row[1] if row else "unavailable", "report": json.loads(row[0]) if row else None,
                "decision": json.loads(row[2]) if row and row[2] else None,
                "events": result}

    def check_report(self, report):
        original = dict(report)
        expected = original.pop("report_hash")
        if digest(original) != expected:
            raise ValueError("验收报告完整性失败")
        for rows in report["data"].values():
            for r in rows:
                if not Path(r["path"]).is_file() or hashlib.sha256(Path(r["path"]).read_bytes()).hexdigest() != r["sha256"]:
                    raise ValueError("验收数据已变化")

    def decide(self, category, version, approved, report_hash, actor, reason):
        key, intent = self.intent(category, version)
        decision = {"approved": approved, "report_hash": report_hash, "actor": actor, "reason": reason}
        with self.engine._pipe_lock(category), self.engine._write_lock, self.connection() as c:
            c.execute("BEGIN IMMEDIATE")
            row = c.execute("SELECT report,state,decision FROM candidate_release WHERE candidate_id=?", (key,)).fetchone()
            if not row:
                raise ValueError("候选未验收")
            report = json.loads(row[0])
            if approved:
                try:
                    self._validate_feedback_guard(intent)
                except CandidateReleaseError as exc:
                    self._mark_blocked(key, exc.reason_code, connection=c)
                    c.commit()
                    raise
            if row[2]:
                if json.loads(row[2]) == decision:
                    return {"state": row[1], "idempotent": True}
                raise ValueError("候选批准重复冲突")
            if report["report_hash"] != report_hash:
                raise ValueError("验收报告hash冲突")
            if approved:
                self.check_report(report)
                self.validate(category, version, intent)
                if row[1] != "validated" or time.time() > report["created_at"] + report["policy"]["valid_seconds"]:
                    raise ValueError("独立验收缺失、未通过或过期")
            state = "approved" if approved else "rejected"
            c.execute("UPDATE candidate_release SET state=?,decision=? WHERE candidate_id=?", (state, encode(decision), key))
            c.execute("INSERT INTO release_events(category,candidate_id,action,state,payload) VALUES (?,?,?,?,?)", (category, key, "approve" if approved else "reject", state, encode(decision)))
        return {"state": state}

    def sync_model(self, category):
        """仅镜像最近已提交发布的实际指针；不扫描或清理其他快照。"""
        from ..db.database import session_scope
        from ..db.models import Model
        e = self.engine
        # 管理写锁串行指针变化；不持品类推理锁或发布SQLite事务访问业务DB。
        with e._write_lock:
            with self.connection() as c:
                row = c.execute("SELECT id,state,payload FROM release_events WHERE category=? AND action IN ('activate','rollback') AND state!='aborted' ORDER BY id DESC LIMIT 1", (category,)).fetchone()
            if not row:
                return None
            event, state, raw = row
            version = json.loads(raw)["target_version"]
            if state != "committed" or e.current_version(category) != version:
                raise ValueError("发布镜像需已提交事件且实际指针一致，请显式恢复或人工处置")
            try:
                with session_scope() as s:
                    target = (s.query(Model).filter(Model.category == category,
                              Model.version == f"v{version}", Model.format == "engine_snapshot")
                              .order_by(Model.id).first())
                    if target is None:
                        # 历史缺行只补实际快照身份，不伪造精度、数据血缘或批准信息。
                        target = Model(name=f"{category}_v{version}", category=category,
                                       version=f"v{version}", path=e._v_dir(category, version),
                                       format="engine_snapshot", origin="self_learned_engine",
                                       metrics={"release_event_id": event, "mirror_recovered": True})
                        s.add(target)
                        s.flush()
                    s.query(Model).filter(Model.category == category, Model.id != target.id).update(
                        {"is_active": False}, synchronize_session=False)
                    target.is_active = True
                    model_id = target.id
            except Exception as exc:
                raise ValueError(f"发布已提交，实际版本v{version}，业务Model镜像同步失败；请显式recover重试") from exc
            return {"model_id": model_id, "version": version, "state": "synced"}

    def switch(self, category, version, rollback=False):
        from algo.persist import load_snapshot, set_current_version
        e = self.engine
        self.check_category(category)
        before = e.current_version(category)
        if before == version:
            with self.connection() as c:
                row = c.execute("SELECT action,state,payload FROM release_events WHERE category=? AND action IN ('activate','rollback') AND state!='aborted' ORDER BY id DESC LIMIT 1", (category,)).fetchone()
            if row and row[:2] == ("rollback" if rollback else "activate", "committed") and json.loads(row[2])["target_version"] == version:
                return {**json.loads(row[2]), "idempotent": True, "model_mirror": self.sync_model(category)}
        source = before if rollback else version
        key, intent = self.intent(category, source)
        status = self.status(category, source)
        if not rollback and status["state"] != "approved":
            raise ValueError("候选独立验收且明确approved后才允许激活")
        if rollback and (status["state"] != "activated" or version != intent["result"]["parent_version"]):
            raise ValueError("回滚仅允许已激活候选的可信父版本")
        if not rollback:
            self.check_report(status["report"])
            try:
                self._validate_feedback_guard(intent)
            except CandidateReleaseError as exc:
                self._mark_blocked(key, exc.reason_code)
                raise
        watermark = self.wal(category)
        pipe = load_snapshot(e._v_dir(category, version), e.device)
        installed = False
        try:
            if rollback:
                self.replay(category, pipe, version)
            with e._pipe_lock(category), e._write_lock, self.connection() as c:
                c.execute("BEGIN IMMEDIATE")
                if c.execute("SELECT 1 FROM release_events WHERE category=? AND state='prepared'", (category,)).fetchone():
                    raise ValueError("存在未完成发布，请先显式恢复")
                if self.wal(category) != watermark:
                    raise ValueError("发布期间WAL已变化")
                if e.current_version(category) != before:
                    raise ValueError("发布期间父版本已变化")
                row = c.execute("SELECT report,state FROM candidate_release WHERE candidate_id=?", (key,)).fetchone()
                if row[1] != ("activated" if rollback else "approved"):
                    raise ValueError("候选发布状态冲突")
                if not rollback:
                    self.validate(category, version, intent)
                    r = json.loads(row[0])
                    if r.get("replay_evidence") is not None:
                        self._validate_replay_evidence(category, intent["result"]["parent_version"], r["replay_evidence"])
                    if time.time() > r["created_at"] + r["policy"]["valid_seconds"]:
                        raise ValueError("候选验收已过期")
                elif e._candidate_files(e._v_dir(category, version)) != intent["parent_files"]:
                    raise ValueError("回滚目标完整性失败")
                payload = {"from_version": before, "target_version": version, "event_watermark": self.wal(category), "candidate_event": key, "candidate_wal": intent["wal_sha256"]}
                action = "rollback" if rollback else "activate"
                cur = c.execute("INSERT INTO release_events(category,candidate_id,action,state,payload) VALUES (?,?,?,'prepared',?)", (category, key, action, encode(payload)))
                event = cur.lastrowid
                c.commit()
                e._candidate_checkpoint("release_prepared")
                set_current_version(e._cat_dir(category), version)
                e._candidate_checkpoint("release_pointer")
                e._cache_put(category, pipe, version)
                installed = True
                c.execute("UPDATE release_events SET state='committed' WHERE id=?", (event,))
                c.execute("UPDATE candidate_release SET state=? WHERE candidate_id=?", ("rolled_back" if rollback else "activated", key))
                c.commit()
            e._candidate_checkpoint("release_committed")
            return {**payload, "model_mirror": self.sync_model(category)}
        finally:
            if not installed:
                pipe._cpu_exec.shutdown()

    def recover(self, category):
        """显式补记prepared；不猜测其他生产指针，不重训、不清WAL。"""
        from algo.persist import load_snapshot
        e = self.engine
        self.check_category(category)
        with self.connection() as c:
            rows = c.execute("SELECT id,candidate_id,action,payload FROM release_events WHERE category=? AND state='prepared' ORDER BY id", (category,)).fetchall()
        recovered = 0
        for event, key, action, raw in rows:
            p = json.loads(raw)
            current = e.current_version(category)
            pipe = None
            installed = False
            try:
                if current == p["target_version"]:
                    _, intent = self.intent(category, p["from_version"] if action == "rollback" else p["target_version"])
                    expected = intent["parent_files"] if action == "rollback" else intent["files"]
                    pipe = load_snapshot(e._v_dir(category, current), e.device)
                    if action == "rollback":
                        self.replay(category, pipe, current)
                with e._pipe_lock(category), e._write_lock, self.connection() as c:
                    c.execute("BEGIN IMMEDIATE")
                    state = c.execute("SELECT state FROM release_events WHERE id=?", (event,)).fetchone()[0]
                    if state != "prepared":
                        continue
                    if e.current_version(category) != current or self.wal(category) != p["event_watermark"]:
                        raise ValueError("发布恢复水位/指针冲突，需人工处置")
                    if current == p["target_version"]:
                        if e._candidate_files(e._v_dir(category, current)) != expected:
                            raise ValueError("恢复目标完整性失败")
                        e._cache_put(category, pipe, current)
                        installed = True
                        c.execute("UPDATE release_events SET state='committed' WHERE id=?", (event,))
                        c.execute("UPDATE candidate_release SET state=? WHERE candidate_id=?", ("rolled_back" if action == "rollback" else "activated", key))
                    elif current == p["from_version"]:
                        c.execute("UPDATE release_events SET state='aborted' WHERE id=?", (event,))
                    else:
                        raise ValueError("发布恢复水位/指针冲突，需人工处置")
                    recovered += 1
            finally:
                if pipe is not None and not installed:
                    pipe._cpu_exec.shutdown()
        return {"recovered": recovered, "model_mirror": self.sync_model(category)}
